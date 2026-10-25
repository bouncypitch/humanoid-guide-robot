import XCTest
import simd

final class CaptureMathTests: XCTestCase {
    private func pose(x: Float = 0, yawDeg: Float = 0) -> simd_float4x4 {
        let q = simd_quatf(angle: yawDeg * .pi / 180, axis: SIMD3(0, 1, 0))
        var m = simd_float4x4(q)
        m.columns.3 = SIMD4(x, 0, 0, 1)
        return m
    }

    func testKeyframePolicy() {
        var p = KeyframePolicy()
        XCTAssertFalse(p.shouldSave(pose: pose(), time: 0, trackingNormal: false))
        XCTAssertTrue(p.shouldSave(pose: pose(), time: 0, trackingNormal: true))
        p.commit(pose: pose(), time: 0)
        XCTAssertFalse(p.shouldSave(pose: pose(x: 0.1), time: 1, trackingNormal: true))   // too small
        XCTAssertFalse(p.shouldSave(pose: pose(x: 0.5), time: 0.2, trackingNormal: true)) // > 3 Hz
        XCTAssertTrue(p.shouldSave(pose: pose(x: 0.16), time: 0.34, trackingNormal: true))
        XCTAssertTrue(p.shouldSave(pose: pose(yawDeg: 11), time: 1, trackingNormal: true))
        XCTAssertFalse(p.shouldSave(pose: pose(yawDeg: 9), time: 1, trackingNormal: true))
        XCTAssertFalse(p.shouldSave(pose: pose(x: 1), time: 1, trackingNormal: false))
    }

    func testDepthConversion() {
        XCTAssertEqual(DepthConversion.metersToMillimeters(1.2345), 1235)
        XCTAssertEqual(DepthConversion.metersToMillimeters(.nan), 0)
        XCTAssertEqual(DepthConversion.metersToMillimeters(.infinity), 0)
        XCTAssertEqual(DepthConversion.metersToMillimeters(-1), 0)
        XCTAssertEqual(DepthConversion.metersToMillimeters(100), 65535)
    }

    func testResizeAndIntrinsics() {
        let src: [UInt8] = [1, 2, 3, 4]
        XCTAssertEqual(DepthConversion.resizeNearest(src, srcWidth: 2, srcHeight: 2, dstWidth: 4, dstHeight: 2),
                       [1, 1, 2, 2, 3, 3, 4, 4])
        let k = Intrinsics(fx: 1400, fy: 1400, cx: 960, cy: 720, width: 1920, height: 1440)
        let d = k.scaled(toWidth: 256, height: 192)
        XCTAssertEqual(d.fx, 1400 * 256 / 1920, accuracy: 1e-9)
        XCTAssertEqual(d.cy, 96, accuracy: 1e-9)
    }

    func testPoseRowMajor() {
        let rows = PoseFormat.rowMajor(pose(x: 2))
        XCTAssertEqual(rows[0][3], 2)
        XCTAssertEqual(rows[3], [0, 0, 0, 1])
    }
}
