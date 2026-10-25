package com.kdotmahesh.rgbdcapture

import android.content.Context
import android.os.Handler
import android.os.Looper
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableLongStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.graphics.ImageBitmap
import androidx.core.content.edit
import java.io.File

/** UI-observable state. Background threads update it through [post]. */
class AppState {
    private val main = Handler(Looper.getMainLooper())

    var trackingLabel by mutableStateOf("Tracking: Not available")
    var trackingNormal by mutableStateOf(false)
    var depthPreview by mutableStateOf<ImageBitmap?>(null)
    var supportMessage by mutableStateOf<String?>(null)
    var canRecord by mutableStateOf(false)

    var isRecording by mutableStateOf(false)
    var keyframes by mutableIntStateOf(0)
    var dropped by mutableIntStateOf(0)
    var startMillis by mutableLongStateOf(0L)
    var streamStatus by mutableStateOf("Off")
    var streamWarning by mutableStateOf<String?>(null)
    var lastSession by mutableStateOf<File?>(null)
    var errorMessage by mutableStateOf<String?>(null)

    var modelFile by mutableStateOf<File?>(null)
    var gemmaStatus by mutableStateOf("No model loaded")
    var gemmaBusy by mutableStateOf(false)
    var lastObjects by mutableStateOf<List<String>>(emptyList())

    fun post(block: AppState.() -> Unit) {
        if (Looper.myLooper() == Looper.getMainLooper()) block() else main.post { block() }
    }
}

/** Persisted toggles (SharedPreferences). */
class Settings(context: Context) {
    private val prefs = context.getSharedPreferences("settings", Context.MODE_PRIVATE)

    var macIp: String
        get() = prefs.getString("macIP", "") ?: ""
        set(v) = prefs.edit { putString("macIP", v) }
    var liveStream: Boolean
        get() = prefs.getBoolean("liveStream", false)
        set(v) = prefs.edit { putBoolean("liveStream", v) }
    var personMask: Boolean
        get() = prefs.getBoolean("personMask", true)
        set(v) = prefs.edit { putBoolean("personMask", v) }
    var gemma: Boolean
        get() = prefs.getBoolean("gemma", false)
        set(v) = prefs.edit { putBoolean("gemma", v) }
    var streamObjects: Boolean
        get() = prefs.getBoolean("streamObjects", false)
        set(v) = prefs.edit { putBoolean("streamObjects", v) }
    var modelPath: String?
        get() = prefs.getString("modelPath", null)
        set(v) = prefs.edit { putString("modelPath", v) }
    var lastSessionPath: String?
        get() = prefs.getString("lastSession", null)
        set(v) = prefs.edit { putString("lastSession", v) }
}
