package com.kdotmahesh.rgbdcapture.io

import java.io.File
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

/** Zips a session folder so the zip holds `session_.../` at its top level (as the iOS app does). */
object SessionZipper {
    fun zip(folder: File, destDir: File): File {
        require(folder.isDirectory) { "not a folder: $folder" }
        destDir.mkdirs()
        val zipFile = File(destDir, folder.name + ".zip")
        if (zipFile.exists()) zipFile.delete()
        ZipOutputStream(zipFile.outputStream().buffered()).use { zos ->
            // JPEG/PNG are already compressed; STORED would need CRCs up front, so just deflate fast.
            zos.setLevel(1)
            folder.walkTopDown().sortedBy { it.path }.forEach { f ->
                val rel = folder.name + "/" + f.relativeTo(folder).invariantSeparatorsPath
                if (f.isDirectory) {
                    val name = if (rel.endsWith("/")) rel else "$rel/"
                    zos.putNextEntry(ZipEntry(name.replace("//", "/")))
                    zos.closeEntry()
                } else {
                    zos.putNextEntry(ZipEntry(rel))
                    f.inputStream().use { it.copyTo(zos) }
                    zos.closeEntry()
                }
            }
        }
        return zipFile
    }
}
