package com.kdotmahesh.rgbdcapture.io

import com.google.gson.GsonBuilder
import com.google.gson.JsonArray
import com.google.gson.JsonElement
import com.google.gson.JsonObject
import com.google.gson.JsonParser
import com.kdotmahesh.rgbdcapture.Intrinsics
import com.kdotmahesh.rgbdcapture.Tracking

/** JSON records of docs/rgbd_format.md (pure JVM, Gson only). */
object Json {
    private val gson = GsonBuilder().disableHtmlEscaping().create()
    private val pretty = GsonBuilder().disableHtmlEscaping().setPrettyPrinting().create()

    fun toJson(e: JsonElement): String = gson.toJson(e)
    fun toPrettyJson(e: JsonElement): String = pretty.toJson(e)

    fun sessionJson(
        model: String,
        appVersion: String,
        startedIso: String,
        hasPersonMask: Boolean,
        hasObjects: Boolean,
    ): JsonObject = JsonObject().apply {
        addProperty("device", "android")
        addProperty("model", model)
        addProperty("app_version", appVersion)
        addProperty("started", startedIso)
        addProperty("coord", "arcore_y_up")
        addProperty("depth_units", "meters") // as the contract says; the PNGs hold millimetres
        addProperty("has_person_mask", hasPersonMask)
        // Android extension (see "Android extensions" in docs/rgbd_format.md)
        if (hasObjects) addProperty("has_objects", true)
    }

    fun intrinsics(k: Intrinsics): JsonObject = JsonObject().apply {
        addProperty("fx", k.fx)
        addProperty("fy", k.fy)
        addProperty("cx", k.cx)
        addProperty("cy", k.cy)
        addProperty("width", k.width)
        addProperty("height", k.height)
    }

    /** Per-frame record; [poseRows] is the row-major camera-to-world 4x4. */
    fun frameJson(
        index: Int,
        timestamp: Double,
        poseRows: List<List<Double>>,
        rgb: Intrinsics,
        depth: Intrinsics,
        tracking: Tracking,
    ): JsonObject = JsonObject().apply {
        addProperty("index", index)
        addProperty("timestamp", timestamp)
        add("pose", JsonArray().also { rows ->
            poseRows.forEach { row -> rows.add(JsonArray().also { a -> row.forEach { a.add(it) } }) }
        })
        add("intrinsics_rgb", intrinsics(rgb))
        add("intrinsics_depth", intrinsics(depth))
        addProperty("tracking", tracking.json)
    }

    /** Copy of [frame] with a "files" list (only for the live stream; never written to disk). */
    fun withFiles(frame: JsonObject, files: List<String>): JsonObject = frame.deepCopy().apply {
        add("files", JsonArray().also { a -> files.forEach { a.add(it) } })
    }

    fun endMessage(): String = """{"end":true}"""

    fun parse(text: String): JsonElement = JsonParser.parseString(text)
}

/** One object from the on-device Gemma labeler. box = [ymin, xmin, ymax, xmax], 0..1000. */
data class DetectedObject(val label: String, val box: List<Int>, val category: String)

object ObjectParsing {
    val CATEGORIES = listOf("structure", "movable_furniture", "clutter", "person")

    /**
     * Extracts the JSON array from a model reply (tolerates ```json fences, prose around it,
     * or an object wrapping the list under "objects"). Invalid items are skipped.
     */
    fun parse(reply: String): List<DetectedObject> {
        val text = reply.trim()
        val el: JsonElement = run {
            val start = text.indexOf('[')
            val end = text.lastIndexOf(']')
            if (start >= 0 && end > start) {
                runCatching { JsonParser.parseString(text.substring(start, end + 1)) }.getOrNull()
            } else null
        } ?: run {
            val s = text.indexOf('{')
            val e = text.lastIndexOf('}')
            if (s >= 0 && e > s) runCatching { JsonParser.parseString(text.substring(s, e + 1)) }.getOrNull() else null
        } ?: return emptyList()

        val arr: JsonArray = when {
            el.isJsonArray -> el.asJsonArray
            el.isJsonObject && el.asJsonObject.get("objects")?.isJsonArray == true -> el.asJsonObject.getAsJsonArray("objects")
            else -> return emptyList()
        }
        return arr.mapNotNull { item ->
            runCatching {
                val o = item.asJsonObject
                val label = o.get("label")?.asString?.trim().orEmpty()
                if (label.isEmpty()) return@runCatching null
                val boxEl = o.get("box_2d") ?: o.get("box")
                val box = boxEl?.asJsonArray?.map { it.asDouble.toInt().coerceIn(0, 1000) }
                if (box == null || box.size != 4) return@runCatching null
                val catRaw = o.get("category")?.asString?.trim()?.lowercase().orEmpty()
                val cat = if (catRaw in CATEGORIES) catRaw else "clutter"
                DetectedObject(label, box, cat)
            }.getOrNull()
        }
    }

    fun toJsonArray(objects: List<DetectedObject>): JsonArray = JsonArray().also { a ->
        objects.forEach { d ->
            a.add(JsonObject().apply {
                addProperty("label", d.label)
                add("box_2d", JsonArray().also { b -> d.box.forEach { b.add(it) } })
                addProperty("category", d.category)
            })
        }
    }
}
