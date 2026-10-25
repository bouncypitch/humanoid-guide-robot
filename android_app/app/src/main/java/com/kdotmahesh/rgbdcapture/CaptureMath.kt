package com.kdotmahesh.rgbdcapture

import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.roundToInt
import kotlin.math.sqrt

/**
 * Pure-Kotlin helpers (no Android/ARCore types) so they can be unit tested on the JVM.
 */

/** Pinhole intrinsics in pixels for an image of size width x height. */
data class Intrinsics(
    val fx: Double,
    val fy: Double,
    val cx: Double,
    val cy: Double,
    val width: Int,
    val height: Int,
) {
    /** Same camera, image resampled to newWidth x newHeight (no crop). */
    fun scaledTo(newWidth: Int, newHeight: Int): Intrinsics {
        val sx = newWidth.toDouble() / width
        val sy = newHeight.toDouble() / height
        return Intrinsics(fx * sx, fy * sy, cx * sx, cy * sy, newWidth, newHeight)
    }

    /** Same camera, image cropped to the rectangle [left, left+w) x [top, top+h). */
    fun cropped(left: Int, top: Int, w: Int, h: Int): Intrinsics =
        Intrinsics(fx, fy, cx - left, cy - top, w, h)

    fun toMap(): Map<String, Any> = linkedMapOf(
        "fx" to fx, "fy" to fy, "cx" to cx, "cy" to cy, "width" to width, "height" to height,
    )

    companion object {
        /** From ARCore CameraIntrinsics arrays: focal [fx, fy], principal [cx, cy], dims [w, h]. */
        fun of(focal: FloatArray, principal: FloatArray, dims: IntArray) = Intrinsics(
            focal[0].toDouble(), focal[1].toDouble(), principal[0].toDouble(), principal[1].toDouble(),
            dims[0], dims[1],
        )
    }
}

/** Integer crop rectangle. */
data class CropRect(val left: Int, val top: Int, val width: Int, val height: Int)

object CaptureMath {

    /**
     * ARCore `Pose.toMatrix(m, 0)` fills a column-major (OpenGL) 4x4. The data contract wants a
     * row-major nested list: rows[r][c] = m[c * 4 + r].
     */
    fun columnMajorToRows(m: FloatArray): List<List<Double>> {
        require(m.size == 16) { "need 16 floats" }
        return List(4) { r -> List(4) { c -> m[c * 4 + r].toDouble() } }
    }

    /** Flat row-major array (index r*4+c) from a column-major one. */
    fun columnMajorToRowMajor(m: FloatArray): DoubleArray {
        require(m.size == 16)
        return DoubleArray(16) { i -> m[(i % 4) * 4 + i / 4].toDouble() }
    }

    /** Euclidean distance between the translations of two row-major 4x4 poses. */
    fun translationDistance(a: DoubleArray, b: DoubleArray): Double {
        val dx = a[3] - b[3]
        val dy = a[7] - b[7]
        val dz = a[11] - b[11]
        return sqrt(dx * dx + dy * dy + dz * dz)
    }

    /** Angle in degrees of the relative rotation R_a^T R_b (row-major 4x4 inputs). */
    fun rotationAngleDeg(a: DoubleArray, b: DoubleArray): Double {
        // trace(A^T B) = sum_ij A_ij * B_ij over the 3x3 block
        var tr = 0.0
        for (r in 0 until 3) for (c in 0 until 3) tr += a[r * 4 + c] * b[r * 4 + c]
        val cos = ((tr - 1.0) / 2.0).coerceIn(-1.0, 1.0)
        return Math.toDegrees(acos(cos))
    }

    /**
     * Largest centred rectangle inside srcW x srcH with the aspect ratio targetW:targetH.
     * Returns the full image if the aspect ratios already agree within [tolerance] (relative).
     */
    fun centerCropToAspect(srcW: Int, srcH: Int, targetW: Int, targetH: Int, tolerance: Double = 0.01): CropRect {
        val src = srcW.toDouble() / srcH
        val dst = targetW.toDouble() / targetH
        if (abs(src - dst) / dst <= tolerance) return CropRect(0, 0, srcW, srcH)
        return if (src > dst) {
            val w = (srcH * dst).roundToInt().coerceIn(1, srcW)
            CropRect((srcW - w) / 2, 0, w, srcH)
        } else {
            val h = (srcW / dst).roundToInt().coerceIn(1, srcH)
            CropRect(0, (srcH - h) / 2, srcW, h)
        }
    }

    /** Output size for an image downscaled (never upscaled) to at most [maxWidth] px wide. */
    fun downscaledSize(w: Int, h: Int, maxWidth: Int = 960): Pair<Int, Int> {
        if (w <= maxWidth) return w to h
        val nh = (h.toDouble() * maxWidth / w).roundToInt().coerceAtLeast(1)
        return maxWidth to nh
    }

    /**
     * Copies a strided little-endian 16-bit plane (ARCore DEPTH16 / raw depth: unsigned
     * millimetres, 0 = invalid) into a compact row-major IntArray of millimetres.
     *
     * @param bytes plane bytes, starting at the first pixel
     * @param rowStrideBytes bytes between rows (>= width * 2)
     */
    fun depthPlaneToMillimetres(bytes: ByteArray, width: Int, height: Int, rowStrideBytes: Int): IntArray {
        val out = IntArray(width * height)
        for (y in 0 until height) {
            var p = y * rowStrideBytes
            val row = y * width
            for (x in 0 until width) {
                val lo = bytes[p].toInt() and 0xFF
                val hi = bytes[p + 1].toInt() and 0xFF
                out[row + x] = (hi shl 8) or lo
                p += 2
            }
        }
        return out
    }

    /** Copies a strided 8-bit plane into a compact array. */
    fun plane8ToCompact(bytes: ByteArray, width: Int, height: Int, rowStride: Int): ByteArray {
        val out = ByteArray(width * height)
        for (y in 0 until height) System.arraycopy(bytes, y * rowStride, out, y * width, width)
        return out
    }

    /** ARCore raw-depth confidence (0..255) -> contract's 0/1/2 (low/medium/high), by thirds. */
    fun confidenceLevel(c: Int): Int = when {
        c < 85 -> 0
        c < 170 -> 1
        else -> 2
    }

    fun confidenceToLevels(conf: ByteArray): ByteArray =
        ByteArray(conf.size) { i -> confidenceLevel(conf[i].toInt() and 0xFF).toByte() }

    /** Nearest-neighbour resample of an 8-bit mask. */
    fun resizeNearest8(src: ByteArray, sw: Int, sh: Int, dw: Int, dh: Int): ByteArray {
        val out = ByteArray(dw * dh)
        for (y in 0 until dh) {
            val sy = ((y + 0.5) * sh / dh).toInt().coerceIn(0, sh - 1)
            for (x in 0 until dw) {
                val sx = ((x + 0.5) * sw / dw).toInt().coerceIn(0, sw - 1)
                out[y * dw + x] = src[sy * sw + sx]
            }
        }
        return out
    }

    /** "m:ss" */
    fun formatElapsed(seconds: Long): String {
        val s = seconds.coerceAtLeast(0)
        return "%d:%02d".format(s / 60, s % 60)
    }

    /**
     * Turbo-like colormap for metric depth; returns ARGB. 0 (invalid) -> black.
     * Near = red/warm, far = blue.
     */
    fun depthColor(mm: Int, maxMm: Int = 5000): Int {
        if (mm <= 0) return 0xFF000000.toInt()
        val t = 1.0 - (mm.toDouble() / maxMm).coerceIn(0.0, 1.0)
        // Polynomial approximation of Google's Turbo colormap (Mikhailov 2019).
        val r = 0.13572138 + t * (4.61539260 + t * (-42.66032258 + t * (132.13108234 + t * (-152.94239396 + t * 59.28637943))))
        val g = 0.09140261 + t * (2.19418839 + t * (4.84296658 + t * (-14.18503333 + t * (4.27729857 + t * 2.82956604))))
        val b = 0.10667330 + t * (12.64194608 + t * (-60.58204836 + t * (110.36276771 + t * (-89.90310912 + t * 27.34824973))))
        fun ch(v: Double) = (v.coerceIn(0.0, 1.0) * 255).roundToInt()
        return (0xFF shl 24) or (ch(r) shl 16) or (ch(g) shl 8) or ch(b)
    }
}

/** Tracking states in the contract's vocabulary. */
enum class Tracking(val json: String) {
    NORMAL("normal"), LIMITED("limited"), NOT_AVAILABLE("not_available");
}

/**
 * Keyframe policy from docs/rgbd_format.md: save when the camera moved >= 0.15 m or rotated
 * >= 10 deg since the last keyframe, at most 3 Hz, and only when tracking == normal.
 * The first keyframe is taken at the first normal-tracking frame.
 *
 * [shouldSave] is a pure query; call [commit] once the keyframe was actually accepted, so a frame
 * dropped because the writer is behind does not reset the reference pose.
 */
class KeyframePolicy(
    val minMoveM: Double = 0.15,
    val minRotDeg: Double = 10.0,
    val maxHz: Double = 3.0,
) {
    private var lastPose: DoubleArray? = null
    private var lastTime = Double.NEGATIVE_INFINITY

    fun reset() {
        lastPose = null
        lastTime = Double.NEGATIVE_INFINITY
    }

    /** @param pose row-major 4x4 camera-to-world, @param t seconds */
    fun shouldSave(pose: DoubleArray, t: Double, tracking: Tracking): Boolean {
        if (tracking != Tracking.NORMAL) return false
        if (t - lastTime < 1.0 / maxHz - 1e-9) return false
        val prev = lastPose ?: return true
        // small epsilons so exactly-at-threshold motion counts despite float round-off (acos near 1)
        return CaptureMath.translationDistance(prev, pose) >= minMoveM - 1e-9 ||
            CaptureMath.rotationAngleDeg(prev, pose) >= minRotDeg - 1e-4
    }

    fun commit(pose: DoubleArray, t: Double) {
        lastPose = pose.copyOf()
        lastTime = t
    }
}
