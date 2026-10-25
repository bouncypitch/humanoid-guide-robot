import Foundation
import CoreVideo
import CoreImage
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import UIKit

enum ImageEncoding {
    static let ciContext = CIContext(options: [.cacheIntermediates: false])

    // MARK: - Pixel buffer copies (so we never retain ARFrame-owned buffers)

    /// Deep-copies a CVPixelBuffer (planar or not) into a freshly allocated buffer.
    static func copy(_ src: CVPixelBuffer) -> CVPixelBuffer? {
        let width = CVPixelBufferGetWidth(src)
        let height = CVPixelBufferGetHeight(src)
        let format = CVPixelBufferGetPixelFormatType(src)
        var dstOut: CVPixelBuffer?
        let attrs: [String: Any] = [kCVPixelBufferIOSurfacePropertiesKey as String: [:]]
        guard CVPixelBufferCreate(nil, width, height, format, attrs as CFDictionary, &dstOut) == kCVReturnSuccess,
              let dst = dstOut else { return nil }

        CVPixelBufferLockBaseAddress(src, .readOnly)
        CVPixelBufferLockBaseAddress(dst, [])
        defer {
            CVPixelBufferUnlockBaseAddress(dst, [])
            CVPixelBufferUnlockBaseAddress(src, .readOnly)
        }

        if CVPixelBufferIsPlanar(src) {
            let planes = CVPixelBufferGetPlaneCount(src)
            for p in 0..<planes {
                guard let s = CVPixelBufferGetBaseAddressOfPlane(src, p),
                      let d = CVPixelBufferGetBaseAddressOfPlane(dst, p) else { return nil }
                let rows = CVPixelBufferGetHeightOfPlane(src, p)
                let sbpr = CVPixelBufferGetBytesPerRowOfPlane(src, p)
                let dbpr = CVPixelBufferGetBytesPerRowOfPlane(dst, p)
                copyRows(s, sbpr, d, dbpr, rows)
            }
        } else {
            guard let s = CVPixelBufferGetBaseAddress(src),
                  let d = CVPixelBufferGetBaseAddress(dst) else { return nil }
            copyRows(s, CVPixelBufferGetBytesPerRow(src), d, CVPixelBufferGetBytesPerRow(dst), height)
        }
        return dst
    }

    private static func copyRows(_ s: UnsafeMutableRawPointer, _ sbpr: Int,
                                 _ d: UnsafeMutableRawPointer, _ dbpr: Int, _ rows: Int) {
        if sbpr == dbpr {
            memcpy(d, s, sbpr * rows)
        } else {
            let n = min(sbpr, dbpr)
            for r in 0..<rows { memcpy(d + r * dbpr, s + r * sbpr, n) }
        }
    }

    /// Reads a single-channel buffer of element type T into a tightly packed array.
    static func readPlane<T>(_ buf: CVPixelBuffer, as: T.Type) -> (data: [T], width: Int, height: Int)? {
        let w = CVPixelBufferGetWidth(buf)
        let h = CVPixelBufferGetHeight(buf)
        CVPixelBufferLockBaseAddress(buf, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buf, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(buf) else { return nil }
        let bpr = CVPixelBufferGetBytesPerRow(buf)
        guard bpr >= w * MemoryLayout<T>.stride else { return nil }
        let out = [T](unsafeUninitializedCapacity: w * h) { ptr, count in
            let dst = UnsafeMutableRawPointer(ptr.baseAddress!)
            let rowBytes = w * MemoryLayout<T>.stride
            for r in 0..<h { memcpy(dst + r * rowBytes, base + r * bpr, rowBytes) }
            count = w * h
        }
        return (out, w, h)
    }

    // MARK: - RGB -> JPEG

    /// Downscales to <= maxWidth (keeping aspect) and JPEG-encodes. Returns data + output size.
    static func jpeg(from buffer: CVPixelBuffer, maxWidth: Int, quality: CGFloat) -> (Data, Int, Int)? {
        let srcW = CVPixelBufferGetWidth(buffer)
        let srcH = CVPixelBufferGetHeight(buffer)
        var image = CIImage(cvPixelBuffer: buffer)
        var outW = srcW, outH = srcH
        if srcW > maxWidth {
            outW = maxWidth
            outH = Int((Double(srcH) * Double(maxWidth) / Double(srcW)).rounded())
            let sx = CGFloat(outW) / CGFloat(srcW)
            let sy = CGFloat(outH) / CGFloat(srcH)
            image = image.transformed(by: CGAffineTransform(scaleX: sx, y: sy))
            image = image.cropped(to: CGRect(x: 0, y: 0, width: outW, height: outH))
        }
        let cs = CGColorSpace(name: CGColorSpace.sRGB)!
        let opts: [CIImageRepresentationOption: Any] = [
            CIImageRepresentationOption(rawValue: kCGImageDestinationLossyCompressionQuality as String): quality
        ]
        guard let data = ciContext.jpegRepresentation(of: image, colorSpace: cs, options: opts) else { return nil }
        return (data, outW, outH)
    }

    // MARK: - PNG

    static func png16(_ values: [UInt16], width: Int, height: Int) -> Data? {
        // Little-endian 16-bit grayscale; ImageIO writes it as a 16-bit PNG (big-endian on disk).
        let data = values.withUnsafeBufferPointer { Data(buffer: $0) }
        let info = CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue | CGBitmapInfo.byteOrder16Little.rawValue)
        return pngGray(data, width: width, height: height, bits: 16, bitmapInfo: info)
    }

    static func png8(_ values: [UInt8], width: Int, height: Int) -> Data? {
        let data = Data(values)
        return pngGray(data, width: width, height: height, bits: 8,
                       bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue))
    }

    private static func pngGray(_ data: Data, width: Int, height: Int, bits: Int, bitmapInfo: CGBitmapInfo) -> Data? {
        guard let provider = CGDataProvider(data: data as CFData),
              let image = CGImage(width: width, height: height,
                                  bitsPerComponent: bits, bitsPerPixel: bits,
                                  bytesPerRow: width * bits / 8,
                                  space: CGColorSpaceCreateDeviceGray(),
                                  bitmapInfo: bitmapInfo, provider: provider,
                                  decode: nil, shouldInterpolate: false,
                                  intent: .defaultIntent) else { return nil }
        let out = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(out, UTType.png.identifier as CFString, 1, nil) else { return nil }
        CGImageDestinationAddImage(dest, image, nil)
        guard CGImageDestinationFinalize(dest) else { return nil }
        return out as Data
    }

    // MARK: - Depth preview (colormapped)

    /// Turbo-ish colormap of depth (0..maxMeters). Invalid -> black. Rotated for portrait display.
    static func depthPreview(_ depth: [Float], width: Int, height: Int, maxMeters: Float = 5) -> UIImage? {
        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        for i in 0..<(width * height) {
            let d = depth[i]
            let o = i * 4
            rgba[o + 3] = 255
            guard d.isFinite, d > 0 else { continue }
            let t = min(1, d / maxMeters)
            let (r, g, b) = turbo(t)
            rgba[o] = r; rgba[o + 1] = g; rgba[o + 2] = b
        }
        let data = Data(rgba)
        guard let provider = CGDataProvider(data: data as CFData),
              let cg = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
                               bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
                               bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                               provider: provider, decode: nil, shouldInterpolate: false,
                               intent: .defaultIntent) else { return nil }
        // ARKit buffers are landscape-right; rotate for a portrait UI.
        return UIImage(cgImage: cg, scale: 1, orientation: .right)
    }

    private static func turbo(_ x: Float) -> (UInt8, UInt8, UInt8) {
        // Polynomial approximation of Google's Turbo colormap.
        let x2 = x * x, x3 = x2 * x, x4 = x3 * x, x5 = x4 * x
        let r = 0.13572138 + 4.61539260 * x - 42.66032258 * x2 + 132.13108234 * x3 - 152.94239396 * x4 + 59.28637943 * x5
        let g = 0.09140261 + 2.19418839 * x + 4.84296658 * x2 - 14.18503333 * x3 + 4.27729857 * x4 + 2.82956604 * x5
        let b = 0.10667330 + 12.64194608 * x - 60.58204836 * x2 + 110.36276771 * x3 - 89.90310912 * x4 + 27.34824973 * x5
        func c(_ v: Float) -> UInt8 { UInt8(max(0, min(255, v * 255))) }
        return (c(r), c(g), c(b))
    }
}
