package com.kdotmahesh.rgbdcapture.ar

import android.graphics.Bitmap
import android.os.Build
import android.util.Log
import androidx.core.graphics.scale
import com.google.gson.JsonObject
import com.kdotmahesh.rgbdcapture.AppState
import com.kdotmahesh.rgbdcapture.BuildConfig
import com.kdotmahesh.rgbdcapture.CaptureMath
import com.kdotmahesh.rgbdcapture.Intrinsics
import com.kdotmahesh.rgbdcapture.KeyframePolicy
import com.kdotmahesh.rgbdcapture.Tracking
import com.kdotmahesh.rgbdcapture.io.Json
import com.kdotmahesh.rgbdcapture.io.ObjectParsing
import com.kdotmahesh.rgbdcapture.io.PngWriter
import com.kdotmahesh.rgbdcapture.ml.ObjectLabeler
import com.kdotmahesh.rgbdcapture.ml.PersonSegmenter
import com.kdotmahesh.rgbdcapture.stream.LiveStreamer
import java.io.ByteArrayOutputStream
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.TimeZone
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger

/** Everything the GL thread hands over for one keyframe (ARCore images already copied). */
class KeyframeInput(
    val timestamp: Double,
    val poseRows: List<List<Double>>,
    val tracking: Tracking,
    val yuv: YuvCopy,
    val depth: DepthCopy,
    val imageIntrinsics: Intrinsics,
    val textureIntrinsics: Intrinsics,
)

/**
 * One recording: writes `session_<ts>/` (docs/rgbd_format.md) on a background executor,
 * optionally streams it live, adds person masks and on-device Gemma labels.
 */
class SessionRecorder(
    val folder: File,
    private val state: AppState,
    streamHost: String?,
    private val segmenter: PersonSegmenter?,
    private val labeler: ObjectLabeler?,
    private val streamObjects: Boolean,
) {
    val policy = KeyframePolicy()
    private val writer = Executors.newSingleThreadExecutor { r -> Thread(r, "keyframe-writer") }
    private val labelExec = Executors.newSingleThreadExecutor { r -> Thread(r, "gemma-labeler") }
    private val pending = AtomicInteger(0)
    private val labelBusy = AtomicBoolean(false)
    @Volatile private var lastLabelStart = 0L
    private var nextIndex = 0
    private val framesDir = File(folder, "frames")
    private val streamer: LiveStreamer? = streamHost?.takeIf { it.isNotBlank() }?.let { host ->
        LiveStreamer(host) { status, warning ->
            state.post {
                streamStatus = status
                if (warning != null) streamWarning = warning
            }
        }
    }
    private var maskWarned = false

    val hasMask get() = segmenter != null

    init {
        framesDir.mkdirs()
        val session = Json.sessionJson(
            model = "${Build.MANUFACTURER} ${Build.MODEL}".trim(),
            appVersion = BuildConfig.VERSION_NAME,
            startedIso = iso8601(Date()),
            hasPersonMask = segmenter != null,
            hasObjects = labeler != null,
        )
        File(folder, "session.json").writeText(Json.toPrettyJson(session))
        streamer?.connect(Json.toJson(session))
    }

    /** Called on the GL thread. Returns false (and counts a drop) if the writer is too far behind. */
    fun canAccept(): Boolean = pending.get() < MAX_PENDING

    /** Called on the GL thread after [canAccept]; assigns the index and queues the work. */
    fun submit(input: KeyframeInput): Int {
        val index = nextIndex++
        pending.incrementAndGet()
        writer.execute {
            try {
                writeKeyframe(index, input)
                state.post { keyframes = index + 1 }
            } catch (t: Throwable) {
                Log.e(TAG, "keyframe $index failed", t)
                state.post { errorMessage = "Keyframe $index failed: ${t.message}" }
            } finally {
                pending.decrementAndGet()
            }
        }
        return index
    }

    private fun writeKeyframe(index: Int, k: KeyframeInput) {
        val stem = "%06d".format(index)
        val dw = k.depth.width
        val dh = k.depth.height

        // RGB: crop to the depth aspect ratio (so the ingester can resize RGB onto depth), downscale to <= 960 px.
        val crop = CaptureMath.centerCropToAspect(k.yuv.width, k.yuv.height, dw, dh)
        val argb = k.yuv.toArgb(crop.left, crop.top, crop.width, crop.height)
        val full = Bitmap.createBitmap(argb, crop.width, crop.height, Bitmap.Config.ARGB_8888)
        val (ow, oh) = CaptureMath.downscaledSize(crop.width, crop.height, 960)
        val rgb = if (ow == crop.width) full else full.scale(ow, oh, filter = true).also { full.recycle() }
        val jpg = ByteArrayOutputStream().use { bos ->
            rgb.compress(Bitmap.CompressFormat.JPEG, 85, bos)
            bos.toByteArray()
        }

        // Intrinsics: CPU image intrinsics follow the crop+scale; the depth image covers the GPU texture's
        // field of view, so depth intrinsics come from textureIntrinsics cropped/scaled to the depth size.
        val kRgb = k.imageIntrinsics.scaledTo(k.yuv.width, k.yuv.height)
            .cropped(crop.left, crop.top, crop.width, crop.height)
            .scaledTo(ow, oh)
        val tex = k.textureIntrinsics
        val tcrop = CaptureMath.centerCropToAspect(tex.width, tex.height, dw, dh)
        val kDepth = tex.cropped(tcrop.left, tcrop.top, tcrop.width, tcrop.height).scaledTo(dw, dh)

        val depthPng = PngWriter.gray16(dw, dh, k.depth.mm)
        val confPng = k.depth.conf?.let { PngWriter.gray8(dw, dh, CaptureMath.confidenceToLevels(it)) }
        val maskPng = segmenter?.let { seg ->
            try {
                PngWriter.gray8(dw, dh, seg.mask(rgb, dw, dh))
            } catch (t: Throwable) {
                if (!maskWarned) {
                    maskWarned = true
                    Log.w(TAG, "person mask failed", t)
                }
                null
            }
        }
        rgb.recycle()

        val files = mutableListOf("$stem.jpg" to jpg, "${stem}_depth.png" to depthPng)
        confPng?.let { files += "${stem}_conf.png" to it }
        maskPng?.let { files += "${stem}_mask.png" to it }
        for ((name, data) in files) File(framesDir, name).writeBytes(data)

        val meta: JsonObject = Json.frameJson(index, k.timestamp, k.poseRows, kRgb, kDepth, k.tracking)
        File(framesDir, "$stem.json").writeText(Json.toJson(meta))

        streamer?.sendKeyframe(Json.toJson(Json.withFiles(meta, files.map { it.first })), files.map { it.second })

        maybeLabel(index, File(framesDir, "$stem.jpg"))
    }

    /** At most one Gemma request in flight, started at most every ~1 s, on the newest keyframe. */
    private fun maybeLabel(index: Int, jpg: File) {
        val l = labeler ?: return
        if (!l.isLoaded) return
        val now = System.currentTimeMillis()
        if (now - lastLabelStart < LABEL_INTERVAL_MS) return
        if (!labelBusy.compareAndSet(false, true)) return
        lastLabelStart = now
        state.post { gemmaBusy = true }
        labelExec.execute {
            try {
                val r = l.label(jpg)
                val out = JsonObject().apply {
                    addProperty("index", index)
                    add("objects", ObjectParsing.toJsonArray(r.objects))
                    addProperty("backend", r.backend)
                    addProperty("latency_ms", r.latencyMs)
                    if (r.objects.isEmpty()) addProperty("raw", r.rawReply.take(4000))
                }
                File(framesDir, "%06d_objects.json".format(index)).writeText(Json.toPrettyJson(out))
                if (streamObjects) {
                    val msg = JsonObject().apply {
                        add("objects", ObjectParsing.toJsonArray(r.objects))
                        addProperty("index", index)
                    }
                    streamer?.sendText(Json.toJson(msg))
                }
                state.post {
                    gemmaStatus = "Gemma (${r.backend}): ${r.objects.size} objects in ${r.latencyMs} ms (#$index)"
                    lastObjects = r.objects.map { "${it.label} · ${it.category}" }
                }
            } catch (t: Throwable) {
                Log.w(TAG, "labelling failed", t)
                state.post { gemmaStatus = "Gemma error: ${t.message}" }
            } finally {
                labelBusy.set(false)
                state.post { gemmaBusy = false }
            }
        }
    }

    /** Drains the writer, ends the stream. Blocking (call off the main thread). */
    fun finish() {
        writer.shutdown()
        writer.awaitTermination(30, TimeUnit.SECONDS)
        labelExec.shutdown() // only the writer submits label jobs, so nothing new arrives now
        labelExec.awaitTermination(30, TimeUnit.SECONDS)
        streamer?.finish(Json.endMessage())
    }

    companion object {
        private const val TAG = "SessionRecorder"
        const val MAX_PENDING = 5
        const val LABEL_INTERVAL_MS = 1000L

        fun iso8601(d: Date): String =
            SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ssXXX", Locale.US).apply { timeZone = TimeZone.getDefault() }.format(d)

        /** `session_YYYYMMDD-HHMM`, with `_2`, `_3`... on collision. */
        fun newSessionFolder(root: File, now: Date = Date()): File {
            root.mkdirs()
            val base = "session_" + SimpleDateFormat("yyyyMMdd-HHmm", Locale.US).format(now)
            var f = File(root, base)
            var n = 2
            while (f.exists()) f = File(root, "${base}_${n++}")
            f.mkdirs()
            return f
        }
    }
}
