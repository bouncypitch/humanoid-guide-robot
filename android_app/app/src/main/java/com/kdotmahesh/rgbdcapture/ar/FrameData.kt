package com.kdotmahesh.rgbdcapture.ar

import android.media.Image
import com.kdotmahesh.rgbdcapture.CaptureMath

/** CPU copy of a YUV_420_888 camera image, so the ARCore Image can be released immediately. */
class YuvCopy(
    val width: Int,
    val height: Int,
    val y: ByteArray,
    val yRowStride: Int,
    val u: ByteArray,
    val v: ByteArray,
    val uvRowStride: Int,
    val uvPixelStride: Int,
) {
    /** ARGB pixels of the crop rectangle at full resolution. */
    fun toArgb(left: Int, top: Int, w: Int, h: Int): IntArray {
        val out = IntArray(w * h)
        for (row in 0 until h) {
            val sy = top + row
            val yBase = sy * yRowStride
            val uvBase = (sy shr 1) * uvRowStride
            for (col in 0 until w) {
                val sx = left + col
                val yy = (y[yBase + sx].toInt() and 0xFF) - 16
                val uvi = uvBase + (sx shr 1) * uvPixelStride
                val uu = (u[uvi].toInt() and 0xFF) - 128
                val vv = (v[uvi].toInt() and 0xFF) - 128
                // BT.601 limited range
                val y1192 = 1192 * yy.coerceAtLeast(0)
                val r = ((y1192 + 1634 * vv) shr 10).coerceIn(0, 255)
                val g = ((y1192 - 833 * vv - 400 * uu) shr 10).coerceIn(0, 255)
                val b = ((y1192 + 2066 * uu) shr 10).coerceIn(0, 255)
                out[row * w + col] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
            }
        }
        return out
    }

    companion object {
        fun of(img: Image): YuvCopy {
            val p = img.planes
            fun bytes(i: Int): ByteArray {
                val b = p[i].buffer.duplicate()
                b.rewind()
                return ByteArray(b.remaining()).also { b.get(it) }
            }
            return YuvCopy(
                img.width, img.height, bytes(0), p[0].rowStride,
                bytes(1), bytes(2), p[1].rowStride, p[1].pixelStride,
            )
        }
    }
}

/** Depth (mm) and optional confidence (0..255), compact row-major. */
class DepthCopy(val width: Int, val height: Int, val mm: IntArray, val conf: ByteArray?) {
    companion object {
        fun of(depth: Image, confidence: Image?): DepthCopy {
            val dp = depth.planes[0]
            val db = dp.buffer.duplicate().also { it.rewind() }
            val dBytes = ByteArray(db.remaining()).also { db.get(it) }
            val mm = CaptureMath.depthPlaneToMillimetres(dBytes, depth.width, depth.height, dp.rowStride)
            val conf = confidence?.let { c ->
                val cp = c.planes[0]
                val cb = cp.buffer.duplicate().also { it.rewind() }
                val cBytes = ByteArray(cb.remaining()).also { cb.get(it) }
                CaptureMath.plane8ToCompact(cBytes, c.width, c.height, cp.rowStride)
            }
            return DepthCopy(depth.width, depth.height, mm, conf)
        }
    }
}
