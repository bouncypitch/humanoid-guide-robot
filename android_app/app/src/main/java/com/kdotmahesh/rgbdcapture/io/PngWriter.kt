package com.kdotmahesh.rgbdcapture.io

import java.io.ByteArrayOutputStream
import java.io.DataOutputStream
import java.util.zip.CRC32
import java.util.zip.Deflater

/**
 * Minimal grayscale PNG encoder. Android's Bitmap can't write 16-bit PNGs, which the contract
 * needs for depth in millimetres, so this writes the chunks directly (IHDR, IDAT, IEND).
 */
object PngWriter {
    private val SIGNATURE = byteArrayOf(0x89.toByte(), 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A)

    /** 16-bit grayscale; values are clamped to 0..65535 and written big-endian. */
    fun gray16(width: Int, height: Int, values: IntArray): ByteArray {
        require(values.size == width * height)
        val raw = ByteArray(height * (1 + width * 2))
        var p = 0
        for (y in 0 until height) {
            raw[p++] = 0 // filter: none
            val row = y * width
            for (x in 0 until width) {
                val v = values[row + x].coerceIn(0, 0xFFFF)
                raw[p++] = (v ushr 8).toByte()
                raw[p++] = v.toByte()
            }
        }
        return encode(width, height, 16, raw)
    }

    /** 8-bit grayscale. */
    fun gray8(width: Int, height: Int, values: ByteArray): ByteArray {
        require(values.size == width * height)
        val raw = ByteArray(height * (1 + width))
        var p = 0
        for (y in 0 until height) {
            raw[p++] = 0
            System.arraycopy(values, y * width, raw, p, width)
            p += width
        }
        return encode(width, height, 8, raw)
    }

    private fun encode(width: Int, height: Int, bitDepth: Int, raw: ByteArray): ByteArray {
        val out = ByteArrayOutputStream(raw.size / 2 + 64)
        out.write(SIGNATURE)
        val ihdr = ByteArrayOutputStream()
        DataOutputStream(ihdr).apply {
            writeInt(width)
            writeInt(height)
            writeByte(bitDepth)
            writeByte(0) // colour type: grayscale
            writeByte(0) // compression
            writeByte(0) // filter
            writeByte(0) // interlace
        }
        chunk(out, "IHDR", ihdr.toByteArray())
        chunk(out, "IDAT", deflate(raw))
        chunk(out, "IEND", ByteArray(0))
        return out.toByteArray()
    }

    private fun deflate(data: ByteArray): ByteArray {
        val d = Deflater(6)
        d.setInput(data)
        d.finish()
        val bos = ByteArrayOutputStream(data.size / 2 + 64)
        val buf = ByteArray(64 * 1024)
        while (!d.finished()) {
            val n = d.deflate(buf)
            bos.write(buf, 0, n)
        }
        d.end()
        return bos.toByteArray()
    }

    private fun chunk(out: ByteArrayOutputStream, type: String, data: ByteArray) {
        val typeBytes = type.toByteArray(Charsets.US_ASCII)
        val dos = DataOutputStream(out)
        dos.writeInt(data.size)
        dos.write(typeBytes)
        dos.write(data)
        val crc = CRC32()
        crc.update(typeBytes)
        crc.update(data)
        dos.writeInt(crc.value.toInt())
    }
}
