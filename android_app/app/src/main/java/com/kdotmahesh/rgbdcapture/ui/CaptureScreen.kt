package com.kdotmahesh.rgbdcapture.ui

import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableLongStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.FilterQuality
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalFocusManager
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.kdotmahesh.rgbdcapture.AppState
import com.kdotmahesh.rgbdcapture.CaptureMath
import com.kdotmahesh.rgbdcapture.Settings
import kotlinx.coroutines.delay

class ScreenActions(
    val onStartStop: () -> Unit,
    val onShare: () -> Unit,
    val onLoadModel: () -> Unit,
    val onRescanModels: () -> Unit,
    val onGemmaToggled: (Boolean) -> Unit,
    val onGrantAllFiles: () -> Unit,
    val needsAllFilesAccess: () -> Boolean,
)

private val Green = Color(0xFF2E9E44)
private val Orange = Color(0xFFE08A1E)
private val Red = Color(0xFFD93A2B)

@Composable
fun CaptureScreen(state: AppState, settings: Settings, actions: ScreenActions) {
    val dark = isSystemInDarkTheme()
    MaterialTheme(colorScheme = if (dark) darkColorScheme() else lightColorScheme()) {
        Surface(Modifier.fillMaxSize()) {
            Content(state, settings, actions)
        }
    }
}

@Composable
private fun Content(state: AppState, settings: Settings, actions: ScreenActions) {
    var macIp by remember { mutableStateOf(settings.macIp) }
    var live by remember { mutableStateOf(settings.liveStream) }
    var mask by remember { mutableStateOf(settings.personMask) }
    var gemma by remember { mutableStateOf(settings.gemma) }
    var streamObjects by remember { mutableStateOf(settings.streamObjects) }
    val focus = LocalFocusManager.current
    val recording = state.isRecording

    Column(
        Modifier
            .safeDrawingPadding()
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(14.dp),
    ) {
        // Header + tracking badge
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text("RGBD Capture", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold)
            Spacer(Modifier.weight(1f))
            Text(
                state.trackingLabel,
                color = Color.White,
                fontWeight = FontWeight.Bold,
                fontSize = 12.sp,
                modifier = Modifier
                    .clip(RoundedCornerShape(50))
                    .background(if (state.trackingNormal) Green else Orange)
                    .padding(horizontal = 10.dp, vertical = 5.dp),
            )
        }

        state.supportMessage?.let { Banner(it, Orange) }
        state.streamWarning?.let { Banner(it, Color(0xFFE6C229)) }

        // Depth preview
        Box(
            Modifier
                .fillMaxWidth()
                .height(260.dp)
                .clip(RoundedCornerShape(12.dp))
                .background(Color.Black),
            contentAlignment = Alignment.Center,
        ) {
            val img = state.depthPreview
            if (img != null) {
                Image(
                    img, contentDescription = "Depth preview", contentScale = ContentScale.Fit,
                    filterQuality = FilterQuality.None, modifier = Modifier.fillMaxSize(),
                )
            } else {
                Text(if (state.canRecord) "Waiting for depth…" else "No depth preview", color = Color.Gray)
            }
        }

        // Stats
        var nowMs by remember { mutableLongStateOf(System.currentTimeMillis()) }
        LaunchedEffect(recording) {
            while (recording) {
                nowMs = System.currentTimeMillis()
                delay(500)
            }
        }
        val elapsed = if (state.startMillis > 0) (nowMs - state.startMillis) / 1000 else 0
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
            Stat("Keyframes", "${state.keyframes}")
            Stat("Elapsed", CaptureMath.formatElapsed(elapsed))
            Stat("Dropped", "${state.dropped}")
            Stat("Stream", if (live) state.streamStatus else "Off")
        }

        // Stream controls
        OutlinedTextField(
            value = macIp,
            onValueChange = { macIp = it.trim(); settings.macIp = macIp },
            label = { Text("Mac IP (e.g. 192.168.1.20)") },
            singleLine = true,
            enabled = !recording,
            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Uri),
            modifier = Modifier.fillMaxWidth(),
        )
        ToggleRow("Live stream to Mac (port 8770)", live, !recording) { live = it; settings.liveStream = it }
        ToggleRow("Person masks (ML Kit)", mask, !recording) { mask = it; settings.personMask = it }

        // START / STOP
        Button(
            onClick = { focus.clearFocus(); actions.onStartStop() },
            enabled = recording || state.canRecord,
            colors = ButtonDefaults.buttonColors(containerColor = if (recording) Red else Green, contentColor = Color.White),
            shape = RoundedCornerShape(22.dp),
            modifier = Modifier
                .fillMaxWidth()
                .heightIn(min = 110.dp),
        ) {
            Text(if (recording) "STOP" else "START", fontSize = 34.sp, fontWeight = FontWeight.Black)
        }

        // On-device Gemma
        Column(
            Modifier
                .fillMaxWidth()
                .clip(RoundedCornerShape(12.dp))
                .background(MaterialTheme.colorScheme.surfaceVariant)
                .padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text("On-device Gemma labels", fontWeight = FontWeight.Bold, modifier = Modifier.weight(1f))
                if (state.gemmaBusy) CircularProgressIndicator(Modifier.size(18.dp), strokeWidth = 2.dp)
                Spacer(Modifier.width(8.dp))
                Switch(checked = gemma, enabled = !recording, onCheckedChange = {
                    gemma = it
                    settings.gemma = it
                    actions.onGemmaToggled(it)
                })
            }
            Text(state.gemmaStatus, style = MaterialTheme.typography.bodySmall)
            if (state.lastObjects.isNotEmpty()) {
                Text(state.lastObjects.take(8).joinToString("\n"), style = MaterialTheme.typography.bodySmall, fontFamily = FontFamily.Monospace)
            }
            ToggleRow("Send object messages on the live stream", streamObjects, !recording) {
                streamObjects = it
                settings.streamObjects = it
            }
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                OutlinedButton(onClick = actions.onLoadModel, enabled = !recording) { Text("Load model") }
                OutlinedButton(onClick = actions.onRescanModels, enabled = !recording) { Text("Rescan") }
                if (actions.needsAllFilesAccess()) {
                    TextButton(onClick = actions.onGrantAllFiles) { Text("Read Downloads") }
                }
            }
        }

        // Share
        val last = state.lastSession
        OutlinedButton(
            onClick = actions.onShare,
            enabled = !recording && last != null,
            modifier = Modifier
                .fillMaxWidth()
                .heightIn(min = 50.dp),
        ) {
            Text(if (last != null) "Share last session (${last.name})" else "Share last session")
        }
    }

    state.errorMessage?.let { msg ->
        AlertDialog(
            onDismissRequest = { state.errorMessage = null },
            confirmButton = { TextButton(onClick = { state.errorMessage = null }) { Text("OK") } },
            title = { Text("RGBD Capture") },
            text = { Text(msg) },
        )
    }
}

@Composable
private fun Stat(title: String, value: String) {
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        Text(value, fontWeight = FontWeight.Bold, fontSize = 20.sp, fontFamily = FontFamily.Monospace)
        Text(title, style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}

@Composable
private fun ToggleRow(label: String, checked: Boolean, enabled: Boolean, onChange: (Boolean) -> Unit) {
    Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
        Text(label, modifier = Modifier.weight(1f))
        Switch(checked = checked, enabled = enabled, onCheckedChange = onChange)
    }
}

@Composable
private fun Banner(text: String, color: Color) {
    Row(
        Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(10.dp))
            .background(color.copy(alpha = 0.25f))
            .padding(10.dp),
    ) {
        Text("⚠ ", fontWeight = FontWeight.Bold)
        Text(text, style = MaterialTheme.typography.bodySmall)
    }
}
