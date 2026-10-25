package com.kdotmahesh.rgbdcapture.ml

import android.graphics.Bitmap
import android.graphics.Matrix
import com.google.android.gms.tasks.Tasks
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.segmentation.Segmentation
import com.google.mlkit.vision.segmentation.Segmenter
import com.google.mlkit.vision.segmentation.selfie.SelfieSegmenterOptions
import java.util.concurrent.TimeUnit

/**
 * Person mask with ML Kit Selfie Segmentation. The keyframe is landscape (sensor orientation);
 * people are upright in portrait, so the image is rotated 90 deg clockwise for the model and the
 * mask is rotated back.
 */
class PersonSegmenter : AutoCloseable {
    private val segmenter: Segmenter = Segmentation.getClient(
        SelfieSegmenterOptions.Builder()
            .setDetectorMode(SelfieSegmenterOptions.SINGLE_IMAGE_MODE)
            .build(),
    )

    /**
     * Returns an 8-bit mask (255 = person, 0 = not) at [outW] x [outH], in the orientation of
     * [landscape]. Blocking; call off the main thread.
     */
    fun mask(landscape: Bitmap, outW: Int, outH: Int): ByteArray {
        val upright = Bitmap.createBitmap(
            landscape, 0, 0, landscape.width, landscape.height, Matrix().apply { postRotate(90f) }, true,
        )
        val result = Tasks.await(segmenter.process(InputImage.fromBitmap(upright, 0)), 5, TimeUnit.SECONDS)
        val mw = result.width // upright mask: mw ~ landscape.height, mh ~ landscape.width
        val mh = result.height
        val buf = result.buffer.asFloatBuffer()
        if (upright !== landscape) upright.recycle()
        // Sample: landscape pixel (x, y) maps to upright pixel (H - 1 - y, x) for a 90 deg CW rotation.
        val out = ByteArray(outW * outH)
        for (y in 0 until outH) {
            val ly = (y + 0.5) / outH // 0..1 in landscape
            val ux = ((1.0 - ly) * mw).toInt().coerceIn(0, mw - 1)
            for (x in 0 until outW) {
                val lx = (x + 0.5) / outW
                val uy = (lx * mh).toInt().coerceIn(0, mh - 1)
                if (buf.get(uy * mw + ux) > 0.5f) out[y * outW + x] = 0xFF.toByte()
            }
        }
        return out
    }

    override fun close() = segmenter.close()
}
