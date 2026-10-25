package com.kdotmahesh.rgbdcapture

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.cos
import kotlin.math.sin

class CaptureMathTest {

    private fun poseRm(tx: Double = 0.0, ty: Double = 0.0, tz: Double = 0.0, yawDeg: Double = 0.0): DoubleArray {
        val a = Math.toRadians(yawDeg)
        val c = cos(a)
        val s = sin(a)
        // rotation about +Y (gravity-up), row-major
        return doubleArrayOf(
            c, 0.0, s, tx,
            0.0, 1.0, 0.0, ty,
            -s, 0.0, c, tz,
            0.0, 0.0, 0.0, 1.0,
        )
    }

    // ---- pose -> row-major ----

    @Test
    fun columnMajorToRows_matchesOpenGlLayout() {
        // OpenGL column-major: translation lives in elements 12..14.
        val m = FloatArray(16) { it.toFloat() }
        val rows = CaptureMath.columnMajorToRows(m)
        assertEquals(listOf(0.0, 4.0, 8.0, 12.0), rows[0])
        assertEquals(listOf(1.0, 5.0, 9.0, 13.0), rows[1])
        assertEquals(listOf(2.0, 6.0, 10.0, 14.0), rows[2])
        assertEquals(listOf(3.0, 7.0, 11.0, 15.0), rows[3])
    }

    @Test
    fun columnMajorToRowMajor_translationInLastColumn() {
        // identity rotation, translation (1, 2, 3), column-major
        val m = floatArrayOf(1f, 0f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 0f, 1f, 0f, 1f, 2f, 3f, 1f)
        val rm = CaptureMath.columnMajorToRowMajor(m)
        assertEquals(1.0, rm[3], 0.0)
        assertEquals(2.0, rm[7], 0.0)
        assertEquals(3.0, rm[11], 0.0)
        assertEquals(1.0, rm[15], 0.0)
        val rows = CaptureMath.columnMajorToRows(m)
        assertEquals(rows.flatten(), rm.toList())
    }

    @Test
    fun rotationAngle_and_distance() {
        assertEquals(0.0, CaptureMath.rotationAngleDeg(poseRm(), poseRm()), 1e-6)
        assertEquals(25.0, CaptureMath.rotationAngleDeg(poseRm(yawDeg = 5.0), poseRm(yawDeg = 30.0)), 1e-6)
        assertEquals(5.0, CaptureMath.translationDistance(poseRm(), poseRm(tx = 3.0, tz = 4.0)), 1e-9)
    }

    // ---- keyframe policy ----

    @Test
    fun keyframePolicy_firstNormalFrameIsKeyframe() {
        val p = KeyframePolicy()
        assertFalse(p.shouldSave(poseRm(), 0.0, Tracking.LIMITED))
        assertFalse(p.shouldSave(poseRm(), 0.0, Tracking.NOT_AVAILABLE))
        assertTrue(p.shouldSave(poseRm(), 0.0, Tracking.NORMAL))
    }

    @Test
    fun keyframePolicy_motionThresholds() {
        val p = KeyframePolicy()
        p.commit(poseRm(), 0.0)
        assertFalse("14 cm is not enough", p.shouldSave(poseRm(tx = 0.14), 1.0, Tracking.NORMAL))
        assertTrue("15 cm is", p.shouldSave(poseRm(tx = 0.15), 1.0, Tracking.NORMAL))
        assertFalse("9 deg is not enough", p.shouldSave(poseRm(yawDeg = 9.0), 1.0, Tracking.NORMAL))
        assertTrue("10 deg is", p.shouldSave(poseRm(yawDeg = 10.0), 1.0, Tracking.NORMAL))
        assertFalse("only when tracking is normal", p.shouldSave(poseRm(tx = 1.0), 1.0, Tracking.LIMITED))
    }

    @Test
    fun keyframePolicy_atMost3Hz() {
        val p = KeyframePolicy()
        p.commit(poseRm(), 0.0)
        assertFalse(p.shouldSave(poseRm(tx = 1.0), 0.30, Tracking.NORMAL))
        assertTrue(p.shouldSave(poseRm(tx = 1.0), 1.0 / 3.0, Tracking.NORMAL))
        // Simulate a fast walk: at most 3 keyframes per second.
        val q = KeyframePolicy()
        var n = 0
        var t = 0.0
        while (t < 10.0) {
            val pose = poseRm(tx = t * 2.0) // 2 m/s
            if (q.shouldSave(pose, t, Tracking.NORMAL)) { q.commit(pose, t); n++ }
            t += 1.0 / 30
        }
        assertTrue("got $n", n in 28..31)
    }

    @Test
    fun keyframePolicy_uncommittedFrameDoesNotMoveReference() {
        val p = KeyframePolicy()
        p.commit(poseRm(), 0.0)
        assertTrue(p.shouldSave(poseRm(tx = 0.2), 1.0, Tracking.NORMAL))
        // dropped (not committed): still a keyframe candidate next frame
        assertTrue(p.shouldSave(poseRm(tx = 0.2), 1.05, Tracking.NORMAL))
        p.commit(poseRm(tx = 0.2), 1.05)
        assertFalse(p.shouldSave(poseRm(tx = 0.25), 2.0, Tracking.NORMAL))
    }

    // ---- intrinsics ----

    @Test
    fun intrinsics_scaledToDepthResolution() {
        val rgb = Intrinsics(fx = 1500.0, fy = 1500.0, cx = 960.0, cy = 540.0, width = 1920, height = 1080)
        val d = rgb.scaledTo(160, 90)
        assertEquals(125.0, d.fx, 1e-9)
        assertEquals(125.0, d.fy, 1e-9)
        assertEquals(80.0, d.cx, 1e-9)
        assertEquals(45.0, d.cy, 1e-9)
        assertEquals(160, d.width)
        assertEquals(90, d.height)
    }

    @Test
    fun intrinsics_cropThenScale() {
        val k = Intrinsics(500.0, 500.0, 320.0, 240.0, 640, 480)
        val crop = CaptureMath.centerCropToAspect(640, 480, 16, 9)
        assertEquals(CropRect(0, 60, 640, 360), crop)
        val c = k.cropped(crop.left, crop.top, crop.width, crop.height)
        assertEquals(320.0, c.cx, 1e-9)
        assertEquals(180.0, c.cy, 1e-9)
        assertEquals(360, c.height)
        val s = c.scaledTo(320, 180)
        assertEquals(250.0, s.fx, 1e-9)
        assertEquals(90.0, s.cy, 1e-9)
    }

    @Test
    fun centerCrop_noopWhenAspectMatches() {
        assertEquals(CropRect(0, 0, 640, 480), CaptureMath.centerCropToAspect(640, 480, 160, 120))
        assertEquals(CropRect(0, 0, 1920, 1080), CaptureMath.centerCropToAspect(1920, 1080, 160, 90))
        // 16:9 source onto a 4:3 depth map: crop the sides
        assertEquals(CropRect(240, 0, 1440, 1080), CaptureMath.centerCropToAspect(1920, 1080, 160, 120))
    }

    @Test
    fun downscaledSize_capsWidthAt960() {
        assertEquals(960 to 540, CaptureMath.downscaledSize(1920, 1080))
        assertEquals(640 to 480, CaptureMath.downscaledSize(640, 480))
        assertEquals(960 to 720, CaptureMath.downscaledSize(1440, 1080))
    }

    // ---- depth / confidence conversion ----

    @Test
    fun depthPlane_littleEndianWithRowStride() {
        // 2x2 image, row stride 6 bytes (2 bytes padding per row)
        val mm = intArrayOf(1000, 65535, 0, 2345)
        val bytes = ByteArray(12)
        fun put(off: Int, v: Int) { bytes[off] = (v and 0xFF).toByte(); bytes[off + 1] = (v ushr 8).toByte() }
        put(0, mm[0]); put(2, mm[1]); put(6, mm[2]); put(8, mm[3])
        bytes[4] = 0x7F; bytes[10] = 0x7F // padding garbage must be ignored
        assertArrayEquals(mm, CaptureMath.depthPlaneToMillimetres(bytes, 2, 2, 6))
    }

    @Test
    fun confidence_levels() {
        assertEquals(0, CaptureMath.confidenceLevel(0))
        assertEquals(0, CaptureMath.confidenceLevel(84))
        assertEquals(1, CaptureMath.confidenceLevel(85))
        assertEquals(1, CaptureMath.confidenceLevel(169))
        assertEquals(2, CaptureMath.confidenceLevel(170))
        assertEquals(2, CaptureMath.confidenceLevel(255))
        assertArrayEquals(byteArrayOf(0, 1, 2), CaptureMath.confidenceToLevels(byteArrayOf(10, 100, 0xFF.toByte())))
    }

    @Test
    fun plane8_compactsStride() {
        val src = byteArrayOf(1, 2, 9, 3, 4, 9)
        assertArrayEquals(byteArrayOf(1, 2, 3, 4), CaptureMath.plane8ToCompact(src, 2, 2, 3))
    }

    @Test
    fun resizeNearest8_upsamples() {
        val out = CaptureMath.resizeNearest8(byteArrayOf(0, 1, 2, 3), 2, 2, 4, 4)
        assertEquals(0, out[0].toInt())
        assertEquals(1, out[3].toInt())
        assertEquals(3, out[15].toInt())
    }

    @Test
    fun formatElapsed() {
        assertEquals("0:00", CaptureMath.formatElapsed(0))
        assertEquals("1:05", CaptureMath.formatElapsed(65))
        assertEquals("12:00", CaptureMath.formatElapsed(720))
    }

    @Test
    fun depthColor_invalidIsBlack() {
        assertEquals(0xFF000000.toInt(), CaptureMath.depthColor(0))
        assertTrue(CaptureMath.depthColor(500) != CaptureMath.depthColor(4500))
    }
}
