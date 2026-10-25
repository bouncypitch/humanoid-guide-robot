// Face boxes with Apple Vision (on-device). Usage: swift faces.swift <dir-or-images...>
// Prints one JSON line per image: {"path": ..., "faces": [[x0, y0, x1, y1], ...]} normalised, origin top-left.
import Foundation
import Vision

var paths: [String] = []
for a in CommandLine.arguments.dropFirst() {
    var isDir: ObjCBool = false
    if FileManager.default.fileExists(atPath: a, isDirectory: &isDir), isDir.boolValue {
        let items = (try? FileManager.default.contentsOfDirectory(atPath: a)) ?? []
        paths += items.filter { $0.hasSuffix(".jpg") }.sorted().map { a + "/" + $0 }
    } else { paths.append(a) }
}
for p in paths {
    let req = VNDetectFaceRectanglesRequest()
    let h = VNImageRequestHandler(url: URL(fileURLWithPath: p))
    try? h.perform([req])
    let boxes = (req.results ?? []).map { r -> [Double] in
        let b = r.boundingBox
        return [b.minX, 1 - b.maxY, b.maxX, 1 - b.minY].map { Double($0) }
    }
    let d = try! JSONSerialization.data(withJSONObject: ["path": p, "faces": boxes])
    print(String(data: d, encoding: .utf8)!)
}
