package com.kdotmahesh.rgbdcapture

import com.kdotmahesh.rgbdcapture.io.Json
import com.kdotmahesh.rgbdcapture.io.ObjectParsing
import com.kdotmahesh.rgbdcapture.io.PngWriter
import com.kdotmahesh.rgbdcapture.io.SessionZipper
import com.kdotmahesh.rgbdcapture.stream.LiveStreamer
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.awt.image.DataBufferUShort
import java.io.ByteArrayInputStream
import java.nio.file.Files
import java.util.zip.ZipFile
import javax.imageio.ImageIO

class IoTest {

    @Test
    fun png16_roundTripsMillimetres() {
        val w = 5
        val h = 3
        val mm = IntArray(w * h) { it * 4000 } // up to 56000, exercises the high byte
        val png = PngWriter.gray16(w, h, mm)
        val img = ImageIO.read(ByteArrayInputStream(png))
        assertEquals(w, img.width)
        assertEquals(h, img.height)
        assertEquals(16, img.colorModel.pixelSize)
        val data = (img.raster.dataBuffer as DataBufferUShort).data
        for (i in mm.indices) assertEquals(mm[i], data[i].toInt() and 0xFFFF)
    }

    @Test
    fun png8_roundTrips() {
        val v = byteArrayOf(0, 1, 2, 0xFF.toByte(), 0, 2)
        val img = ImageIO.read(ByteArrayInputStream(PngWriter.gray8(3, 2, v)))
        assertEquals(8, img.colorModel.pixelSize)
        val r = img.raster
        assertEquals(2, r.getSample(2, 0, 0))
        assertEquals(255, r.getSample(0, 1, 0))
        assertEquals(2, r.getSample(2, 1, 0))
    }

    @Test
    fun frameJson_matchesContract() {
        val k = Intrinsics(500.0, 501.0, 320.0, 240.0, 640, 480)
        val rows = listOf(
            listOf(1.0, 0.0, 0.0, 0.5), listOf(0.0, 1.0, 0.0, 1.5),
            listOf(0.0, 0.0, 1.0, -2.0), listOf(0.0, 0.0, 0.0, 1.0),
        )
        val j = Json.frameJson(7, 12.345, rows, k, k.scaledTo(160, 120), Tracking.NORMAL)
        assertEquals(setOf("index", "timestamp", "pose", "intrinsics_rgb", "intrinsics_depth", "tracking"), j.keySet())
        assertEquals(7, j["index"].asInt)
        assertEquals("normal", j["tracking"].asString)
        assertEquals(0.5, j.getAsJsonArray("pose")[0].asJsonArray[3].asDouble, 0.0)
        assertEquals(-2.0, j.getAsJsonArray("pose")[2].asJsonArray[3].asDouble, 0.0)
        assertEquals(160, j.getAsJsonObject("intrinsics_depth")["width"].asInt)
        assertEquals(125.0, j.getAsJsonObject("intrinsics_depth")["fx"].asDouble, 1e-9)

        val streamed = Json.withFiles(j, listOf("000007.jpg", "000007_depth.png"))
        assertEquals(2, streamed.getAsJsonArray("files").size())
        assertFalse("original untouched", j.has("files"))
    }

    @Test
    fun sessionJson_android() {
        val s = Json.sessionJson("Google Pixel 11", "1.0", "2026-10-24T13:10:00-07:00", hasPersonMask = true, hasObjects = false)
        assertEquals("android", s["device"].asString)
        assertEquals("arcore_y_up", s["coord"].asString)
        assertEquals("meters", s["depth_units"].asString)
        assertTrue(s["has_person_mask"].asBoolean)
        assertFalse(s.has("kind"))
        assertFalse(s.has("has_objects"))
        assertEquals("""{"end":true}""", Json.endMessage())
    }

    @Test
    fun objectParsing_toleratesFencesAndJunk() {
        val reply = """
            Here you go:
            ```json
            [{"label": "sofa", "box_2d": [500, 100, 900, 600], "category": "movable_furniture"},
             {"label": "wall", "box_2d": [0, 0, 1000, 1000], "category": "structure"},
             {"label": "thing", "box_2d": [1, 2, 3], "category": "clutter"},
             {"label": "cat toy", "box_2d": [10, 20, 1200, 40.6], "category": "toys"}]
            ```
        """.trimIndent()
        val objs = ObjectParsing.parse(reply)
        assertEquals(3, objs.size)
        assertEquals("sofa", objs[0].label)
        assertEquals(listOf(500, 100, 900, 600), objs[0].box)
        assertEquals("clutter", objs[2].category) // unknown category -> clutter
        assertEquals(listOf(10, 20, 1000, 40), objs[2].box) // clamped
        assertTrue(ObjectParsing.parse("no json here").isEmpty())
        assertEquals(1, ObjectParsing.parse("""{"objects":[{"label":"person","box_2d":[0,0,10,10],"category":"person"}]}""").size)
    }

    @Test
    fun zip_hasSessionFolderAtTopLevel() {
        val tmp = Files.createTempDirectory("rgbd").toFile()
        val s = tmp.resolve("session_20261024-1310").apply { resolve("frames").mkdirs() }
        s.resolve("session.json").writeText("{}")
        s.resolve("frames/000000.json").writeText("{}")
        val zip = SessionZipper.zip(s, tmp.resolve("out"))
        val names = ZipFile(zip).use { z -> z.entries().toList().map { it.name } }
        assertTrue(names.toString(), "session_20261024-1310/session.json" in names)
        assertTrue(names.toString(), "session_20261024-1310/frames/000000.json" in names)
        assertTrue(names.all { it.startsWith("session_20261024-1310/") })
        tmp.deleteRecursively()
    }

    @Test
    fun litertlmHeader() {
        assertTrue(com.kdotmahesh.rgbdcapture.ml.isLiteRtLmHeader("LITERTLM\u0001\u0000".toByteArray()))
        assertFalse(com.kdotmahesh.rgbdcapture.ml.isLiteRtLmHeader("Access to model litert".toByteArray()))
        assertFalse(com.kdotmahesh.rgbdcapture.ml.isLiteRtLmHeader(ByteArray(4)))
    }

    @Test
    fun liveUrl() {
        assertEquals("ws://192.168.1.20:8770/ingest", LiveStreamer.buildUrl("192.168.1.20"))
        assertEquals("ws://192.168.1.20:9000/ingest", LiveStreamer.buildUrl(" 192.168.1.20:9000 "))
        assertEquals("ws://mac.local:8770/ingest", LiveStreamer.buildUrl("ws://mac.local/"))
    }
}
