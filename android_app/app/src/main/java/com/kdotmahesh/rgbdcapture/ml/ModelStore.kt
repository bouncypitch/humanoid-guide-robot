package com.kdotmahesh.rgbdcapture.ml

import android.content.Context
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.OpenableColumns
import java.io.File

/** Finds or imports `.litertlm` model files. Weights are never bundled with the app. */
object ModelStore {

    fun modelsDir(context: Context): File =
        File(context.getExternalFilesDir(null) ?: context.filesDir, "models").apply { mkdirs() }

    /** Places we look, in order. /sdcard/Download is readable only with all-files access. */
    fun searchDirs(context: Context): List<File> = listOfNotNull(
        modelsDir(context),
        context.getExternalFilesDir(null),
        File(context.filesDir, "models"),
        context.filesDir,
        Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS),
    )

    fun findModels(context: Context): List<File> = searchDirs(context)
        .flatMap { d -> runCatching { d.listFiles()?.toList() }.getOrNull().orEmpty() }
        .filter { it.isFile && it.name.endsWith(".litertlm") && it.canRead() }
        .distinctBy { it.canonicalPath }

    /** Prefers the NPU build for this phone's SoC (e.g. "…_Google_Tensor_G6.litertlm"), then any Gemma 4 E2B. */
    fun pickBest(models: List<File>): File? {
        val soc = if (Build.VERSION.SDK_INT >= 31) Build.SOC_MODEL.orEmpty() else ""
        val socKey = soc.replace(' ', '_').lowercase()
        return models.sortedWith(
            compareByDescending<File> { socKey.isNotEmpty() && it.name.lowercase().contains(socKey) }
                .thenByDescending { it.name.contains("E2B", ignoreCase = true) }
                .thenBy { it.name },
        ).firstOrNull { f ->
            // An NPU build for a different SoC won't run; skip it.
            val n = f.name.lowercase()
            !(n.contains("google_tensor") && socKey.isNotEmpty() && !n.contains(socKey))
        } ?: models.firstOrNull()
    }

    /** Copies a picked document into [modelsDir]. Blocking; reports progress 0..1. */
    fun import(context: Context, uri: Uri, onProgress: (Float) -> Unit): File {
        val cr = context.contentResolver
        var name = "model.litertlm"
        var size = -1L
        cr.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME, OpenableColumns.SIZE), null, null, null)?.use { c ->
            if (c.moveToFirst()) {
                c.getString(0)?.let { name = it }
                if (!c.isNull(1)) size = c.getLong(1)
            }
        }
        if (!name.endsWith(".litertlm")) name += ".litertlm"
        val dest = File(modelsDir(context), File(name).name)
        val tmp = File(dest.path + ".part")
        cr.openInputStream(uri).use { input ->
            requireNotNull(input) { "cannot open $uri" }
            tmp.outputStream().use { out ->
                val buf = ByteArray(1 shl 20)
                var done = 0L
                var lastReport = 0L
                while (true) {
                    val n = input.read(buf)
                    if (n < 0) break
                    out.write(buf, 0, n)
                    done += n
                    if (size > 0 && done - lastReport > 32L shl 20) {
                        lastReport = done
                        onProgress(done.toFloat() / size)
                    }
                }
            }
        }
        if (dest.exists()) dest.delete()
        check(tmp.renameTo(dest)) { "rename failed" }
        onProgress(1f)
        return dest
    }
}
