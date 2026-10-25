package com.kdotmahesh.rgbdcapture.ml

import com.kdotmahesh.rgbdcapture.io.DetectedObject
import java.io.File

/**
 * On-device object labelling for keyframes. The capture pipeline only talks to this interface,
 * so the LiteRT-LM implementation can be swapped out (or removed) without touching capture code.
 */
interface ObjectLabeler : AutoCloseable {
    /** Loads the model, trying accelerators in order. Returns the backend that worked ("npu"/"gpu"/"cpu"). Blocking. */
    fun load(model: File): String

    val isLoaded: Boolean
    val backendName: String?

    /** Labels one JPEG. Blocking; call from a background thread. */
    fun label(jpeg: File): LabelResult
}

data class LabelResult(
    val objects: List<DetectedObject>,
    val rawReply: String,
    val latencyMs: Long,
    val backend: String,
)

const val LABEL_PROMPT =
    "List notable objects as JSON [{label, box_2d:[ymin,xmin,ymax,xmax] 0-1000, " +
        "category in structure|movable_furniture|clutter|person}]"

/** Every .litertlm file starts with the ASCII magic "LITERTLM". */
fun isLiteRtLmHeader(head: ByteArray): Boolean =
    head.size >= 8 && String(head, 0, 8, Charsets.US_ASCII) == "LITERTLM"

/**
 * Checked before handing the file to native code: LiteRT-LM 0.18.1 aborts the whole process
 * (invalid UTF-8 in its JNI error message) when given a non-.litertlm file, e.g. the HTML/text page
 * a gated Hugging Face download returns without a login.
 */
fun requireLiteRtLmFile(f: File) {
    val head = ByteArray(8)
    val n = f.inputStream().use { it.read(head) }
    require(n == 8 && isLiteRtLmHeader(head)) {
        "${f.name} is not a .litertlm model (bad header). Was the download complete and logged in to Hugging Face?"
    }
}
