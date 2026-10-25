package com.kdotmahesh.rgbdcapture

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import android.provider.Settings as AndroidSettings
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import androidx.core.net.toUri
import com.google.ar.core.ArCoreApk
import com.google.ar.core.exceptions.UnavailableApkTooOldException
import com.google.ar.core.exceptions.UnavailableArcoreNotInstalledException
import com.google.ar.core.exceptions.UnavailableDeviceNotCompatibleException
import com.google.ar.core.exceptions.UnavailableSdkTooOldException
import com.google.ar.core.exceptions.UnavailableUserDeclinedInstallationException
import com.kdotmahesh.rgbdcapture.ar.ArEngine
import com.kdotmahesh.rgbdcapture.ar.SessionRecorder
import com.kdotmahesh.rgbdcapture.io.SessionZipper
import com.kdotmahesh.rgbdcapture.ml.LiteRtLmLabeler
import com.kdotmahesh.rgbdcapture.ml.ModelStore
import com.kdotmahesh.rgbdcapture.ml.ObjectLabeler
import com.kdotmahesh.rgbdcapture.ml.PersonSegmenter
import com.kdotmahesh.rgbdcapture.ui.CaptureScreen
import com.kdotmahesh.rgbdcapture.ui.ScreenActions
import java.io.File
import java.util.concurrent.Executors

class MainActivity : ComponentActivity() {

    private val state = AppState()
    private lateinit var settings: Settings
    private var engine: ArEngine? = null
    private var installRequested = false
    private val bg = Executors.newSingleThreadExecutor { r -> Thread(r, "app-bg") }
    private val modelExec = Executors.newSingleThreadExecutor { r -> Thread(r, "model-load") }
    private var labeler: ObjectLabeler? = null
    private var segmenter: PersonSegmenter? = null

    private val cameraPermission = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) setUpAr() else state.supportMessage = "Camera permission is needed to record. Grant it in Settings > Apps > RGBD Capture."
    }

    private val pickModel = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri: Uri? ->
        if (uri != null) importModel(uri)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        settings = Settings(this)
        settings.lastSessionPath?.let { File(it) }?.takeIf { it.isDirectory }?.let { state.lastSession = it }
        findModelOnDisk()

        setContent {
            CaptureScreen(
                state = state,
                settings = settings,
                actions = ScreenActions(
                    onStartStop = ::toggleRecording,
                    onShare = ::shareLastSession,
                    onLoadModel = { pickModel.launch(arrayOf("*/*")) },
                    onRescanModels = ::findModelOnDisk,
                    onGemmaToggled = { on -> if (on) ensureModelLoaded() },
                    onGrantAllFiles = ::requestAllFilesAccess,
                    needsAllFilesAccess = { !hasAllFilesAccess() },
                ),
            )
        }
    }

    override fun onResume() {
        super.onResume()
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            cameraPermission.launch(Manifest.permission.CAMERA)
            return
        }
        setUpAr()
    }

    override fun onPause() {
        super.onPause()
        if (state.isRecording) toggleRecording() // finalise the session rather than leave it half-written
        engine?.pause()
    }

    override fun onDestroy() {
        super.onDestroy()
        engine?.destroy()
        engine = null
        modelExec.execute { labeler?.close() }
        segmenter?.close()
    }

    /** ARCore availability / install prompt, then session creation and resume. */
    private fun setUpAr() {
        if (engine == null) {
            try {
                when (ArCoreApk.getInstance().requestInstall(this, !installRequested)) {
                    ArCoreApk.InstallStatus.INSTALL_REQUESTED -> {
                        installRequested = true
                        state.supportMessage = "Installing Google Play Services for AR…"
                        return
                    }
                    ArCoreApk.InstallStatus.INSTALLED -> Unit
                }
            } catch (e: UnavailableUserDeclinedInstallationException) {
                state.supportMessage = "Google Play Services for AR is required. Reopen the app to install it."
                return
            } catch (e: UnavailableDeviceNotCompatibleException) {
                state.supportMessage = "This device doesn't support ARCore, so it can't record RGB-D."
                return
            } catch (e: Exception) {
                state.supportMessage = "ARCore unavailable: ${e.message}"
                return
            }
            val eng = ArEngine(applicationContext, state)
            val r = eng.createSession()
            r.exceptionOrNull()?.let { e ->
                state.supportMessage = when (e) {
                    is UnavailableArcoreNotInstalledException -> "Google Play Services for AR is not installed."
                    is UnavailableApkTooOldException -> "Please update Google Play Services for AR."
                    is UnavailableSdkTooOldException -> "Please update this app (ARCore SDK too old)."
                    is UnavailableDeviceNotCompatibleException -> "This device doesn't support ARCore."
                    else -> "Could not start ARCore (${e.javaClass.simpleName}${e.message?.let { ": $it" } ?: ""}). " +
                        "The arm64 Android emulator can't run ARCore; use a real ARCore phone."
                }
                Log.e(TAG, "session create failed", e)
                eng.destroy()
                return
            }
            engine = eng
        }
        engine?.resume()
    }

    private fun toggleRecording() {
        val eng = engine ?: return
        if (!state.isRecording) {
            val root = File(getExternalFilesDir(null) ?: filesDir, "sessions")
            val folder = SessionRecorder.newSessionFolder(root)
            val live = settings.liveStream && settings.macIp.isNotBlank()
            val seg = if (settings.personMask) (segmenter ?: PersonSegmenter().also { segmenter = it }) else null
            val lab = if (settings.gemma) labeler?.takeIf { it.isLoaded } else null
            state.keyframes = 0
            state.dropped = 0
            state.streamWarning = null
            state.streamStatus = if (live) "Connecting…" else "Off"
            state.startMillis = System.currentTimeMillis()
            state.isRecording = true
            val rec = SessionRecorder(folder, state, if (live) settings.macIp else null, seg, lab, settings.streamObjects)
            eng.startRecording(rec)
            Log.i(TAG, "recording to $folder")
        } else {
            state.isRecording = false
            eng.stopRecording { rec ->
                bg.execute {
                    rec?.finish()
                    state.post {
                        startMillis = 0L
                        rec?.folder?.let {
                            lastSession = it
                            settings.lastSessionPath = it.path
                        }
                    }
                }
            }
        }
    }

    private fun shareLastSession() {
        val folder = state.lastSession ?: return
        bg.execute {
            try {
                val zip = SessionZipper.zip(folder, File(cacheDir, "share"))
                val uri = FileProvider.getUriForFile(this, "$packageName.fileprovider", zip)
                val send = Intent(Intent.ACTION_SEND).apply {
                    type = "application/zip"
                    putExtra(Intent.EXTRA_STREAM, uri)
                    addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                }
                runOnUiThread { startActivity(Intent.createChooser(send, "Share ${zip.name}")) }
            } catch (t: Throwable) {
                state.post { errorMessage = "Zip failed: ${t.message}" }
            }
        }
    }

    // ---- Gemma model ----

    private fun findModelOnDisk() {
        val saved = settings.modelPath?.let { File(it) }?.takeIf { it.canRead() }
        val found = saved ?: ModelStore.pickBest(ModelStore.findModels(this))
        state.modelFile = found
        if (found == null) {
            state.gemmaStatus = "No .litertlm model found. Use \"Load model\"."
        } else {
            state.gemmaStatus = "Model: ${found.name} (not loaded)"
            if (settings.gemma) ensureModelLoaded()
        }
    }

    private fun importModel(uri: Uri) {
        state.gemmaStatus = "Copying model…"
        modelExec.execute {
            try {
                val f = ModelStore.import(this, uri) { p -> state.post { gemmaStatus = "Copying model… ${(p * 100).toInt()}%" } }
                settings.modelPath = f.path
                state.post { modelFile = f }
                loadModel(f)
            } catch (t: Throwable) {
                state.post { gemmaStatus = "Model import failed: ${t.message}" }
            }
        }
    }

    private fun ensureModelLoaded() {
        val f = state.modelFile ?: run {
            state.gemmaStatus = "No model. Tap \"Load model\" or adb push one (see README)."
            return
        }
        if (labeler?.isLoaded == true) return
        modelExec.execute { loadModel(f) }
    }

    /** On modelExec. */
    private fun loadModel(f: File) {
        state.post { gemmaStatus = "Loading ${f.name}…" }
        try {
            val l = labeler ?: LiteRtLmLabeler(applicationContext).also { labeler = it }
            val backend = l.load(f)
            settings.modelPath = f.path
            state.post { gemmaStatus = "Gemma ready on ${backend.uppercase()} (${f.name})" }
        } catch (t: Throwable) {
            Log.e(TAG, "model load failed", t)
            state.post { gemmaStatus = "Model load failed: ${t.message}" }
        }
    }

    private fun hasAllFilesAccess(): Boolean = Environment.isExternalStorageManager()

    private fun requestAllFilesAccess() {
        val i = Intent(AndroidSettings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION, "package:$packageName".toUri())
        runCatching { startActivity(i) }.onFailure {
            startActivity(Intent(AndroidSettings.ACTION_MANAGE_ALL_FILES_ACCESS_PERMISSION))
        }
    }

    companion object {
        private const val TAG = "RGBDCapture"
    }
}
