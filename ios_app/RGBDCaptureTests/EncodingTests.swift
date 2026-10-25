import XCTest
import ImageIO
import CoreVideo

final class EncodingTests: XCTestCase {
    func testPNG16RoundTrip() throws {
        let w = 4, h = 2
        let values: [UInt16] = [0, 1, 255, 256, 1234, 5000, 65535, 300]
        let data = try XCTUnwrap(ImageEncoding.png16(values, width: w, height: h))
        let src = try XCTUnwrap(CGImageSourceCreateWithData(data as CFData, nil))
        let img = try XCTUnwrap(CGImageSourceCreateImageAtIndex(src, 0, nil))
        XCTAssertEqual(img.bitsPerComponent, 16)
        XCTAssertEqual(img.bitsPerPixel, 16)
        // Raw PNG decode via ImageIO gives big-endian samples unless byteOrder16Little.
        let raw = try XCTUnwrap(img.dataProvider?.data) as Data
        let little = img.bitmapInfo.contains(.byteOrder16Little)
        var decoded: [UInt16] = []
        for y in 0..<h {
            for x in 0..<w {
                let o = y * img.bytesPerRow + x * 2
                let a = UInt16(raw[o]), b = UInt16(raw[o + 1])
                decoded.append(little ? (a | b << 8) : (a << 8 | b))
            }
        }
        XCTAssertEqual(decoded, values)
        let out = FileManager.default.temporaryDirectory.appendingPathComponent("test_depth.png")
        try data.write(to: out)
    }

    func testJPEGDownscale() throws {
        var pb: CVPixelBuffer?
        CVPixelBufferCreate(nil, 1920, 1440, kCVPixelFormatType_32BGRA, nil, &pb)
        let copy = try XCTUnwrap(ImageEncoding.copy(try XCTUnwrap(pb)))
        let (data, w, h) = try XCTUnwrap(ImageEncoding.jpeg(from: copy, maxWidth: 960, quality: 0.85))
        XCTAssertEqual(w, 960); XCTAssertEqual(h, 720)
        let src = try XCTUnwrap(CGImageSourceCreateWithData(data as CFData, nil))
        let img = try XCTUnwrap(CGImageSourceCreateImageAtIndex(src, 0, nil))
        XCTAssertEqual(img.width, 960); XCTAssertEqual(img.height, 720)
    }

    func testZip() throws {
        let fm = FileManager.default
        let folder = fm.temporaryDirectory.appendingPathComponent("session_test")
        try? fm.removeItem(at: folder)
        try fm.createDirectory(at: folder.appendingPathComponent("frames"), withIntermediateDirectories: true)
        try Data("{\"device\":\"iphone\"}".utf8).write(to: folder.appendingPathComponent("session.json"))
        try Data(repeating: 7, count: 1000).write(to: folder.appendingPathComponent("frames/000000.jpg"))
        let zip = try SessionZipper.zip(folder: folder)
        let store = fm.temporaryDirectory.appendingPathComponent("store_test.zip")
        try? fm.removeItem(at: store)
        try StoreZipWriter.write(folder: folder, to: store)
    }
}
