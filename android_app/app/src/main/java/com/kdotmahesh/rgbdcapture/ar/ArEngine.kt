package com.kdotmahesh.rgbdcapture.ar

import android.content.Context
import android.graphics.Bitmap
import android.opengl.EGL14
import android.opengl.EGLConfig
import android.opengl.EGLContext
import android.opengl.EGLDisplay
import android.opengl.EGLSurface
import android.opengl.GLES11Ext
import android.opengl.GLES20
import android.os.Handler
import android.os.HandlerThread
import android.os.SystemClock
import android.util.Log
import androidx.compose.ui.graphics.asImageBitmap
import com.google.ar.core.CameraConfig
import com.google.ar.core.CameraConfigFilter
import com.google.ar.core.Config
import com.google.ar.core.Frame
import com.google.ar.core.Session
import com.google.ar.core.TrackingFailureReason
import com.google.ar.core.TrackingState
import com.google.ar.core.exceptions.NotYetAvailableException
import com.google.ar.core.exceptions.SessionPausedException
import com.kdotmahesh.rgbdcapture.AppState
import com.kdotmahesh.rgbdcapture.CaptureMath
import com.kdotmahesh.rgbdcapture.Intrinsics
import com.kdotmahesh.rgbdcapture.Tracking
import java.util.concurrent.CountDownLatch
import kotlin.math.abs

/**
 * Headless ARCore loop. ARCore needs a GL context and an external texture to stream the camera
 * into, but nothing is drawn: an offscreen EGL pbuffer context lives on a dedicated thread that
 * calls Session.update() in a loop. CPU images (RGB, depth, confidence) are copied out per keyframe.
 */
class ArEngine(private val context: Context, private val state: AppState) {

    private val thread = HandlerThread("arcore-gl").apply { start() }
    private val handler = Handler(thread.looper)

    private var eglDisplay: EGLDisplay = EGL14.EGL_NO_DISPLAY
    private var eglContext: EGLContext = EGL14.EGL_NO_CONTEXT
    private var eglSurface: EGLSurface = EGL14.EGL_NO_SURFACE
    private var cameraTexture = 0

    private var session: Session? = null
    @Volatile private var running = false
    private var depthSupported = false

    @Volatile var recorder: SessionRecorder? = null
    private var recordStartNs = 0L
    private var lastPreviewMs = 0L

    /** Creates and configures the ARCore session (caller has done the install/permission checks). */
    fun createSession(): Result<Unit> = runOnGl {
        if (session != null) return@runOnGl
        initEgl()
        val s = Session(context)
        chooseCameraConfig(s)
        depthSupported = s.isDepthModeSupported(Config.DepthMode.AUTOMATIC)
        val cfg = Config(s).apply {
            depthMode = if (depthSupported) Config.DepthMode.AUTOMATIC else Config.DepthMode.DISABLED
            focusMode = Config.FocusMode.AUTO
            updateMode = Config.UpdateMode.BLOCKING
            lightEstimationMode = Config.LightEstimationMode.DISABLED
            planeFindingMode = Config.PlaneFindingMode.DISABLED
            instantPlacementMode = Config.InstantPlacementMode.DISABLED
        }
        s.configure(cfg)
        s.setCameraTextureName(cameraTexture)
        s.setDisplayGeometry(android.view.Surface.ROTATION_0, 1080, 1920)
        session = s
        val cc = s.cameraConfig
        Log.i(TAG, "camera config: cpu ${cc.imageSize} texture ${cc.textureSize} depthSensor ${cc.depthSensorUsage} depthSupported=$depthSupported")
        state.post {
            canRecord = depthSupported
            supportMessage = if (depthSupported) null else
                "This device doesn't support the ARCore Depth API, so recording is disabled."
        }
    }

    /**
     * Prefer: back camera, a hardware depth sensor if there is one, CPU image with the same aspect
     * ratio as the GPU texture (same field of view as the depth image), then the largest CPU image
     * up to 1920 px wide (it is downscaled to <= 960 for the JPEG anyway).
     */
    private fun chooseCameraConfig(s: Session) {
        val filter = CameraConfigFilter(s).setFacingDirection(CameraConfig.FacingDirection.BACK)
        val configs = s.getSupportedCameraConfigs(filter)
        if (configs.isEmpty()) return
        fun aspectMatches(c: CameraConfig): Boolean {
            val a = c.imageSize.width.toDouble() / c.imageSize.height
            val b = c.textureSize.width.toDouble() / c.textureSize.height
            return abs(a - b) / b < 0.01
        }
        val best = configs.sortedWith(
            compareByDescending<CameraConfig> { it.depthSensorUsage == CameraConfig.DepthSensorUsage.REQUIRE_AND_USE }
                .thenByDescending { aspectMatches(it) }
                .thenByDescending { it.imageSize.width <= 1920 }
                .thenByDescending { if (it.imageSize.width <= 1920) it.imageSize.width else -it.imageSize.width }
                .thenByDescending { it.fpsRange.upper },
        ).first()
        s.cameraConfig = best
    }

    fun resume() = handler.post {
        val s = session ?: return@post
        try {
            s.resume()
            running = true
            handler.post(::tick)
        } catch (t: Throwable) {
            Log.e(TAG, "resume failed", t)
            state.post { supportMessage = "Camera unavailable: ${t.message}"; canRecord = false }
        }
    }

    fun pause() {
        running = false
        handler.post { runCatching { session?.pause() } }
    }

    fun destroy() {
        running = false
        handler.post {
            runCatching { session?.close() }
            session = null
            releaseEgl()
            thread.quitSafely()
        }
    }

    fun startRecording(r: SessionRecorder) = handler.post {
        recordStartNs = 0L
        r.policy.reset()
        recorder = r
    }

    /** Detaches the recorder; returns it so the caller can finish it off the main thread. */
    fun stopRecording(onDetached: (SessionRecorder?) -> Unit) = handler.post {
        val r = recorder
        recorder = null
        onDetached(r)
    }

    private fun tick() {
        if (!running) return
        val s = session ?: return
        try {
            val frame = s.update()
            onFrame(frame)
        } catch (_: SessionPausedException) {
            return
        } catch (t: Throwable) {
            Log.w(TAG, "frame failed: ${t.message}")
        }
        if (running) handler.post(::tick)
    }

    private fun onFrame(frame: Frame) {
        val camera = frame.camera
        val tracking = when (camera.trackingState) {
            TrackingState.TRACKING -> Tracking.NORMAL
            TrackingState.PAUSED -> Tracking.LIMITED
            else -> Tracking.NOT_AVAILABLE
        }
        val label = when (tracking) {
            Tracking.NORMAL -> "Tracking: Normal"
            Tracking.LIMITED -> "Tracking: Limited" + reasonSuffix(camera.trackingFailureReason)
            Tracking.NOT_AVAILABLE -> "Tracking: Not available"
        }
        if (label != state.trackingLabel) state.post { trackingLabel = label; trackingNormal = tracking == Tracking.NORMAL }

        val now = SystemClock.elapsedRealtime()
        if (depthSupported && now - lastPreviewMs >= PREVIEW_INTERVAL_MS) {
            lastPreviewMs = now
            updatePreview(frame)
        }

        val r = recorder ?: return
        if (recordStartNs == 0L) recordStartNs = frame.timestamp
        val t = (frame.timestamp - recordStartNs) / 1e9
        val m = FloatArray(16)
        camera.pose.toMatrix(m, 0)
        val poseRm = CaptureMath.columnMajorToRowMajor(m)
        if (!r.policy.shouldSave(poseRm, t, tracking)) return
        if (!r.canAccept()) {
            state.post { dropped++ }
            return
        }
        val input = grab(frame, t, CaptureMath.columnMajorToRows(m), tracking) ?: return
        r.policy.commit(poseRm, t)
        r.submit(input)
    }

    /** Copies the CPU image + depth for one keyframe; null if not available this frame. */
    private fun grab(frame: Frame, t: Double, rows: List<List<Double>>, tracking: Tracking): KeyframeInput? {
        val depth = acquireDepth(frame, withConfidence = true) ?: return null
        val yuv = try {
            frame.acquireCameraImage().use { YuvCopy.of(it) }
        } catch (_: NotYetAvailableException) {
            return null
        }
        val cam = frame.camera
        val ii = cam.imageIntrinsics
        val ti = cam.textureIntrinsics
        return KeyframeInput(
            timestamp = t,
            poseRows = rows,
            tracking = tracking,
            yuv = yuv,
            depth = depth,
            imageIntrinsics = Intrinsics.of(ii.focalLength, ii.principalPoint, ii.imageDimensions),
            textureIntrinsics = Intrinsics.of(ti.focalLength, ti.principalPoint, ti.imageDimensions),
        )
    }

    /** Raw depth (+ confidence) first, falling back to the smoothed full depth image. */
    private fun acquireDepth(frame: Frame, withConfidence: Boolean): DepthCopy? {
        try {
            frame.acquireRawDepthImage16Bits().use { d ->
                val conf = if (withConfidence) {
                    try {
                        frame.acquireRawDepthConfidenceImage()
                    } catch (_: NotYetAvailableException) {
                        null
                    }
                } else null
                conf.use { return DepthCopy.of(d, it) }
            }
        } catch (_: NotYetAvailableException) {
        } catch (t: Throwable) {
            Log.w(TAG, "raw depth failed: ${t.message}")
        }
        return try {
            frame.acquireDepthImage16Bits().use { DepthCopy.of(it, null) }
        } catch (_: NotYetAvailableException) {
            null
        }
    }

    /** Small colour-mapped depth preview, rotated to portrait. Uses the smoothed depth for looks. */
    private fun updatePreview(frame: Frame) {
        val d = try {
            frame.acquireDepthImage16Bits().use { DepthCopy.of(it, null) }
        } catch (_: NotYetAvailableException) {
            return
        } catch (_: Throwable) {
            return
        }
        val w = d.width
        val h = d.height
        // rotate 90 deg clockwise: out (x', y') = (h - 1 - y, x), out size h x w
        val px = IntArray(w * h)
        for (y in 0 until h) for (x in 0 until w) {
            px[x * h + (h - 1 - y)] = CaptureMath.depthColor(d.mm[y * w + x])
        }
        val bmp = Bitmap.createBitmap(px, h, w, Bitmap.Config.ARGB_8888).asImageBitmap()
        state.post { depthPreview = bmp }
    }

    private fun reasonSuffix(r: TrackingFailureReason): String = when (r) {
        TrackingFailureReason.NONE -> ""
        TrackingFailureReason.BAD_STATE -> " (bad state)"
        TrackingFailureReason.INSUFFICIENT_LIGHT -> " (too dark)"
        TrackingFailureReason.EXCESSIVE_MOTION -> " (moving too fast)"
        TrackingFailureReason.INSUFFICIENT_FEATURES -> " (point at more texture)"
        TrackingFailureReason.CAMERA_UNAVAILABLE -> " (camera unavailable)"
    }

    // ---- EGL ----

    private fun initEgl() {
        if (eglContext != EGL14.EGL_NO_CONTEXT) return
        eglDisplay = EGL14.eglGetDisplay(EGL14.EGL_DEFAULT_DISPLAY)
        val ver = IntArray(2)
        check(EGL14.eglInitialize(eglDisplay, ver, 0, ver, 1)) { "eglInitialize failed" }
        val attribs = intArrayOf(
            EGL14.EGL_RENDERABLE_TYPE, EGL14.EGL_OPENGL_ES2_BIT,
            EGL14.EGL_SURFACE_TYPE, EGL14.EGL_PBUFFER_BIT,
            EGL14.EGL_RED_SIZE, 8, EGL14.EGL_GREEN_SIZE, 8, EGL14.EGL_BLUE_SIZE, 8,
            EGL14.EGL_NONE,
        )
        val configs = arrayOfNulls<EGLConfig>(1)
        val n = IntArray(1)
        check(EGL14.eglChooseConfig(eglDisplay, attribs, 0, configs, 0, 1, n, 0) && n[0] > 0) { "no EGL config" }
        eglContext = EGL14.eglCreateContext(
            eglDisplay, configs[0], EGL14.EGL_NO_CONTEXT,
            intArrayOf(EGL14.EGL_CONTEXT_CLIENT_VERSION, 2, EGL14.EGL_NONE), 0,
        )
        eglSurface = EGL14.eglCreatePbufferSurface(
            eglDisplay, configs[0], intArrayOf(EGL14.EGL_WIDTH, 1, EGL14.EGL_HEIGHT, 1, EGL14.EGL_NONE), 0,
        )
        check(EGL14.eglMakeCurrent(eglDisplay, eglSurface, eglSurface, eglContext)) { "eglMakeCurrent failed" }
        val tex = IntArray(1)
        GLES20.glGenTextures(1, tex, 0)
        cameraTexture = tex[0]
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, cameraTexture)
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)
    }

    private fun releaseEgl() {
        if (eglDisplay == EGL14.EGL_NO_DISPLAY) return
        if (cameraTexture != 0) GLES20.glDeleteTextures(1, intArrayOf(cameraTexture), 0)
        EGL14.eglMakeCurrent(eglDisplay, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_CONTEXT)
        EGL14.eglDestroySurface(eglDisplay, eglSurface)
        EGL14.eglDestroyContext(eglDisplay, eglContext)
        EGL14.eglTerminate(eglDisplay)
        eglDisplay = EGL14.EGL_NO_DISPLAY
        eglContext = EGL14.EGL_NO_CONTEXT
        eglSurface = EGL14.EGL_NO_SURFACE
    }

    /** Runs [block] on the GL thread and waits for it. */
    private fun runOnGl(block: () -> Unit): Result<Unit> {
        var result: Result<Unit> = Result.failure(IllegalStateException("not run"))
        val done = CountDownLatch(1)
        handler.post {
            result = runCatching(block)
            done.countDown()
        }
        done.await()
        return result
    }

    companion object {
        private const val TAG = "ArEngine"
        private const val PREVIEW_INTERVAL_MS = 100L
    }
}
