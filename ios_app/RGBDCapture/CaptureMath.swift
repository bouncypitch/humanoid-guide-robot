import Foundation
import simd

// Pure-Swift helpers (no ARKit / UIKit) so they can be unit tested.

/// Keyframe policy from docs/rgbd_format.md:
/// save when moved >= 0.15 m or rotated >= 10 deg since last keyframe,
/// at most 3 Hz, and only when tracking is normal.
struct KeyframePolicy {
    var minTranslation: Float = 0.15          // metres
    var minRotationDegrees: Float = 10        // degrees
    var minInterval: Double = 1.0 / 3.0       // seconds (3 Hz cap)

    private(set) var lastPose: simd_float4x4?
    private(set) var lastTime: Double?

    func shouldSave(pose: simd_float4x4, time: Double, trackingNormal: Bool) -> Bool {
        guard trackingNormal else { return false }
        guard let lastPose, let lastTime else { return true }   // first keyframe
        if time - lastTime < minInterval - 1e-6 { return false }
        let moved = simd_distance(KeyframePolicy.translation(pose), KeyframePolicy.translation(lastPose))
        if moved >= minTranslation { return true }
        return KeyframePolicy.rotationAngleDegrees(lastPose, pose) >= minRotationDegrees
    }

    mutating func commit(pose: simd_float4x4, time: Double) {
        lastPose = pose
        lastTime = time
    }

    mutating func reset() {
        lastPose = nil
        lastTime = nil
    }

    static func translation(_ m: simd_float4x4) -> SIMD3<Float> {
        SIMD3(m.columns.3.x, m.columns.3.y, m.columns.3.z)
    }

    static func rotation(_ m: simd_float4x4) -> simd_float3x3 {
        simd_float3x3(SIMD3(m.columns.0.x, m.columns.0.y, m.columns.0.z),
                      SIMD3(m.columns.1.x, m.columns.1.y, m.columns.1.z),
                      SIMD3(m.columns.2.x, m.columns.2.y, m.columns.2.z))
    }

    /// Angle of the relative rotation R_a^T * R_b, in degrees.
    static func rotationAngleDegrees(_ a: simd_float4x4, _ b: simd_float4x4) -> Float {
        let rel = simd_mul(rotation(a).transpose, rotation(b))
        let trace = rel.columns.0.x + rel.columns.1.y + rel.columns.2.z
        let c = max(-1, min(1, (trace - 1) / 2))
        return acos(c) * 180 / .pi
    }
}

enum DepthConversion {
    /// Float32 metres -> UInt16 millimetres. NaN / inf / <= 0 -> 0, clamps at 65535.
    static func metersToMillimeters(_ value: Float) -> UInt16 {
        guard value.isFinite, value > 0 else { return 0 }
        let mm = (value * 1000).rounded()
        if mm >= 65535 { return 65535 }
        return UInt16(mm)
    }

    static func metersToMillimeters(_ values: [Float]) -> [UInt16] {
        values.map { metersToMillimeters($0) }
    }

    /// Nearest-neighbour resize of an 8-bit single-channel image.
    static func resizeNearest(_ src: [UInt8], srcWidth: Int, srcHeight: Int,
                              dstWidth: Int, dstHeight: Int) -> [UInt8] {
        if srcWidth == dstWidth && srcHeight == dstHeight { return src }
        var dst = [UInt8](repeating: 0, count: dstWidth * dstHeight)
        guard srcWidth > 0, srcHeight > 0 else { return dst }
        for y in 0..<dstHeight {
            let sy = min(srcHeight - 1, (y * srcHeight) / dstHeight)
            let srow = sy * srcWidth
            let drow = y * dstWidth
            for x in 0..<dstWidth {
                let sx = min(srcWidth - 1, (x * srcWidth) / dstWidth)
                dst[drow + x] = src[srow + sx]
            }
        }
        return dst
    }
}

struct Intrinsics: Equatable {
    var fx: Double, fy: Double, cx: Double, cy: Double
    var width: Int, height: Int

    /// Scale intrinsics to a new image resolution (same field of view).
    func scaled(toWidth w: Int, height h: Int) -> Intrinsics {
        let sx = Double(w) / Double(width)
        let sy = Double(h) / Double(height)
        return Intrinsics(fx: fx * sx, fy: fy * sy, cx: cx * sx, cy: cy * sy, width: w, height: h)
    }

    var json: [String: Any] {
        ["fx": fx, "fy": fy, "cx": cx, "cy": cy, "width": width, "height": height]
    }
}

enum PoseFormat {
    /// simd matrices are column-major; the spec wants a row-major 4x4 nested array.
    static func rowMajor(_ m: simd_float4x4) -> [[Double]] {
        let c = [m.columns.0, m.columns.1, m.columns.2, m.columns.3]
        return (0..<4).map { r in (0..<4).map { col in Double(c[col][r]) } }
    }
}
