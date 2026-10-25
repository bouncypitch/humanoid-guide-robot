package com.kdotmahesh.rgbdcapture.stream

import android.util.Log
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import okio.ByteString.Companion.toByteString
import java.util.concurrent.TimeUnit

/**
 * Live stream to the Mac ingester (docs/rgbd_format.md, "Live streaming"):
 *   1. text: session.json
 *   2. per keyframe: text frame JSON (with "files"), then one binary message per file, in order
 *   3. text: {"end": true}
 * The socket never blocks capture: if it fails, [onStatus] reports it and the caller keeps
 * recording locally. If OkHttp's outgoing queue backs up, whole keyframes are skipped (the
 * ingester tolerates gaps in indices).
 */
class LiveStreamer(
    host: String,
    private val onStatus: (status: String, warning: String?) -> Unit,
) {
    enum class State { CONNECTING, OPEN, FAILED, CLOSED }

    @Volatile var state = State.CONNECTING
        private set
    @Volatile var sentKeyframes = 0
        private set
    @Volatile var skippedKeyframes = 0
        private set

    private val url = buildUrl(host)
    private val client = OkHttpClient.Builder()
        .connectTimeout(5, TimeUnit.SECONDS)
        .pingInterval(10, TimeUnit.SECONDS)
        .build()
    private val lock = Any()
    private var ws: WebSocket? = null

    fun connect(sessionJson: String) {
        onStatus("Connecting…", null)
        val req = Request.Builder().url(url).build()
        ws = client.newWebSocket(req, object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                state = State.OPEN
                onStatus("Streaming", null)
            }

            override fun onMessage(webSocket: WebSocket, text: String) {
                Log.i(TAG, "ingester: $text")
            }

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                if (state == State.CLOSED) return
                state = State.FAILED
                Log.w(TAG, "stream failed", t)
                onStatus("Failed", "Live stream to $url failed (${t.message ?: t.javaClass.simpleName}). Still recording locally.")
            }

            override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                if (state != State.FAILED) state = State.CLOSED
            }
        })
        // OkHttp queues messages sent before the handshake completes, so the session record goes first.
        ws?.send(sessionJson)
    }

    /** Sends one keyframe: metadata (already containing "files") then the binaries in order. */
    fun sendKeyframe(metaJson: String, binaries: List<ByteArray>) {
        val socket = ws ?: return
        if (state == State.FAILED || state == State.CLOSED) return
        synchronized(lock) {
            val size = binaries.sumOf { it.size.toLong() } + metaJson.length
            if (socket.queueSize() + size > MAX_QUEUE_BYTES) {
                skippedKeyframes++
                onStatus("Behind", null)
                return
            }
            var ok = socket.send(metaJson)
            for (b in binaries) ok = ok && socket.send(b.toByteString())
            if (ok) {
                sentKeyframes++
                if (state == State.OPEN) onStatus("Streaming", null)
            }
        }
    }

    /** Extra text message (e.g. on-device object labels). */
    fun sendText(text: String) {
        val socket = ws ?: return
        if (state == State.FAILED || state == State.CLOSED) return
        synchronized(lock) { socket.send(text) }
    }

    /** Sends {"end": true} and closes gracefully once the queue drains (OkHttp handles that). */
    fun finish(endJson: String) {
        val socket = ws ?: return
        synchronized(lock) {
            if (state != State.FAILED) {
                socket.send(endJson)
                socket.close(1000, "done")
                state = State.CLOSED
                onStatus("Sent $sentKeyframes", null)
            }
        }
        client.dispatcher.executorService.shutdown()
    }

    fun cancel() {
        state = State.CLOSED
        ws?.cancel()
        client.dispatcher.executorService.shutdown()
    }

    companion object {
        private const val TAG = "LiveStreamer"
        const val PORT = 8770
        private const val MAX_QUEUE_BYTES = 8L * 1024 * 1024 // OkHttp hard limit is 16 MiB

        fun buildUrl(host: String): String {
            val h = host.trim().removePrefix("ws://").removePrefix("http://").trimEnd('/')
            val hostPort = if (h.contains(':') && !h.startsWith("[")) h else "$h:$PORT"
            return "ws://$hostPort/ingest"
        }
    }
}
