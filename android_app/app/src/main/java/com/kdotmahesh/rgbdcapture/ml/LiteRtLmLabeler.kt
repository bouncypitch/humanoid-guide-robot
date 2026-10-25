package com.kdotmahesh.rgbdcapture.ml

import android.content.Context
import android.util.Log
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.SamplerConfig
import com.kdotmahesh.rgbdcapture.io.ObjectParsing
import java.io.File

/**
 * Gemma (e.g. gemma-4-E2B-it .litertlm) via LiteRT-LM. Tries NPU (Google Tensor), then GPU,
 * then CPU. An NPU-compiled model file (…_Google_Tensor_G6.litertlm) normally only runs on the
 * NPU; the generic file runs on GPU/CPU.
 *
 * If a backend initialises but fails on the first image, the next backend is tried.
 */
class LiteRtLmLabeler(private val context: Context) : ObjectLabeler {

    private data class Attempt(val name: String, val make: () -> Pair<Backend, Backend>)

    private val attempts = listOf(
        Attempt("npu") { npu() to npu() },
        Attempt("npu+gpu-vision") { npu() to Backend.GPU() },
        Attempt("gpu") { Backend.GPU() to Backend.GPU() },
        Attempt("cpu") { Backend.CPU() to Backend.CPU() },
    )

    private fun npu() = Backend.NPU(nativeLibraryDir = context.applicationInfo.nativeLibraryDir)

    private var engine: Engine? = null
    private var model: File? = null
    private var attemptIndex = 0
    private var verified = false

    override var backendName: String? = null
        private set

    override val isLoaded: Boolean get() = engine != null

    @Synchronized
    override fun load(model: File): String {
        close()
        requireLiteRtLmFile(model)
        this.model = model
        return initFrom(0)
    }

    private fun initFrom(start: Int): String {
        val m = model ?: error("no model")
        var lastError: Throwable? = null
        for (i in start until attempts.size) {
            val a = attempts[i]
            try {
                val (backend, vision) = a.make()
                val cfg = EngineConfig(
                    modelPath = m.absolutePath,
                    backend = backend,
                    visionBackend = vision,
                    maxNumImages = 1,
                    cacheDir = context.cacheDir.path,
                )
                val e = Engine(cfg)
                val t0 = System.currentTimeMillis()
                e.initialize()
                Log.i(TAG, "LiteRT-LM ready on ${a.name} in ${System.currentTimeMillis() - t0} ms")
                engine = e
                attemptIndex = i
                backendName = a.name
                verified = false
                return a.name
            } catch (t: Throwable) {
                Log.w(TAG, "backend ${a.name} failed: ${t.message}")
                lastError = t
            }
        }
        throw IllegalStateException("No LiteRT-LM backend could load ${m.name}: ${lastError?.message}", lastError)
    }

    @Synchronized
    override fun label(jpeg: File): LabelResult {
        while (true) {
            val e = engine ?: error("model not loaded")
            try {
                val t0 = System.currentTimeMillis()
                val conv = e.createConversation(
                    ConversationConfig(
                        samplerConfig = SamplerConfig(topK = 1, topP = 1.0, temperature = 0.0, seed = 0),
                    ),
                )
                val reply = conv.use {
                    val msg = it.sendMessage(Contents.of(Content.ImageFile(jpeg.absolutePath), Content.Text(LABEL_PROMPT)))
                    msg.contents.contents.filterIsInstance<Content.Text>().joinToString("") { c -> c.text }
                }
                verified = true
                return LabelResult(ObjectParsing.parse(reply), reply, System.currentTimeMillis() - t0, backendName ?: "?")
            } catch (t: Throwable) {
                // A backend can initialise but fail at inference (e.g. vision not supported on it): move on.
                if (verified || attemptIndex + 1 >= attempts.size) throw t
                Log.w(TAG, "inference failed on $backendName, trying next backend", t)
                runCatching { engine?.close() }
                engine = null
                initFrom(attemptIndex + 1)
            }
        }
    }

    @Synchronized
    override fun close() {
        runCatching { engine?.close() }
        engine = null
        backendName = null
    }

    companion object {
        private const val TAG = "LiteRtLmLabeler"
    }
}
