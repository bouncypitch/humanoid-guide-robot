import Foundation
import ARKit
import UIKit
import simd

/// Everything needed to write one keyframe, copied out of the ARFrame on the delegate
/// thread so the ARFrame itself is never retained.
private struct KeyframeSnapshot {
    let index: Int
    let timestamp: Double
    let pose: simd_float4x4
    let tracking: String
    let rgb: CVPixelBuffer              // our own deep copy
    let rgbIntrinsics: Intrinsics       // at capturedImage resolution
    let depth: [Float]
    let depthWidth: Int
    let depthHeight: Int
    let confidence: [UInt8]?            // depth resolution
    let mask: (data: [UInt8], width: Int, height: Int)?  // segmentation resolution
}

final class CaptureManager: NSObject, ObservableObject, ARSessionDelegate {
    // MARK: Published UI state (main thread)
    @Published var isRecording = false
    @Published var keyframeCount = 0
    @Published var droppedCount = 0
    @Published var trackingLabel = "Starting…"
    @Published var trackingIsNormal = false
    @Published var depthPreview: UIImage?
    @Published var supportMessage: String?
    @Published var hasLiDAR = false
    @Published var hasPersonMask = false
    @Published var streamStatus = "Off"
    @Published var streamWarning: String?
    @Published var startDate: Date?
    @Published var lastSessionURL: URL?
    @Published var errorMessage: String?

    let session = ARSession()

    private let arQueue = DispatchQueue(label: "rgbd.ar.delegate", qos: .userInteractive)
    private let writeQueue = DispatchQueue(label: "rgbd.writer", qos: .userInitiated)
    private let previewQueue = DispatchQueue(label: "rgbd.preview", qos: .utility)

    private let pendingLock = NSLock()
    private var pendingWrites = 0
    private let maxPendingWrites = 5

    // arQueue-owned state
    private var recording: ActiveRecording?
    private var lastTrackingLabel = ""
    private var lastPreviewTime: TimeInterval = 0
    private var previewBusy = false
    private var usePersonSegmentation = true

    private final class ActiveRecording {
        let folder: URL
        let framesDir: URL
        let streamer: LiveStreamer?
        var policy = KeyframePolicy()
        var nextIndex = 0
        var startTimestamp: TimeInterval?
        init(folder: URL, framesDir: URL, streamer: LiveStreamer?) {
            self.folder = folder; self.framesDir = framesDir; self.streamer = streamer
        }
    }

    override init() {
        super.init()
        session.delegate = self
        session.delegateQueue = arQueue
        lastSessionURL = Self.newestSession()
    }

    static var documentsURL: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    }

    static func newestSession() -> URL? {
        let fm = FileManager.default
        guard let items = try? fm.contentsOfDirectory(at: documentsURL, includingPropertiesForKeys: [.creationDateKey]) else { return nil }
        return items
            .filter { $0.lastPathComponent.hasPrefix("session_") && $0.hasDirectoryPath }
            .max { a, b in
                let da = (try? a.resourceValues(forKeys: [.creationDateKey]).creationDate) ?? .distantPast
                let db = (try? b.resourceValues(forKeys: [.creationDateKey]).creationDate) ?? .distantPast
                return da < db
            }
    }

    // MARK: - AR session lifecycle

    func startSession() {
        guard ARWorldTrackingConfiguration.isSupported else {
            supportMessage = "ARKit world tracking is not supported on this device (the Simulator never supports it). Run on an iPhone Pro with LiDAR."
            trackingLabel = "Unsupported"
            return
        }
        let lidar = ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth)
        hasLiDAR = lidar
        if !lidar {
            supportMessage = "This device has no LiDAR (sceneDepth unsupported). Recording is disabled — use an iPhone/iPad Pro."
        }
        // Don't reset the world frame if we come back from background mid-recording.
        session.run(makeConfiguration(), options: isRecording ? [] : [.resetTracking, .removeExistingAnchors])
    }

    func pauseSession() {
        session.pause()
    }

    private func makeConfiguration() -> ARWorldTrackingConfiguration {
        let config = ARWorldTrackingConfiguration()
        var semantics: ARConfiguration.FrameSemantics = []
        for s in [ARConfiguration.FrameSemantics.sceneDepth, .smoothedSceneDepth] where ARWorldTrackingConfiguration.supportsFrameSemantics(s) {
            semantics.insert(s)
        }
        var personMask = false
        if usePersonSegmentation,
           ARWorldTrackingConfiguration.supportsFrameSemantics(.personSegmentationWithDepth) {
            semantics.insert(.personSegmentationWithDepth)
            personMask = true
        }
        // Some devices reject certain semantic combinations; validate before running.
        if personMask, !ARWorldTrackingConfiguration.supportsFrameSemantics(semantics) {
            semantics.remove(.personSegmentationWithDepth)
            personMask = false
        }
        config.frameSemantics = semantics
        config.worldAlignment = .gravity
        DispatchQueue.main.async { self.hasPersonMask = personMask }
        return config
    }

    // MARK: - Recording control (main thread)

    func startRecording(streamHost: String?) {
        guard !isRecording else { return }
        guard hasLiDAR else {
            errorMessage = "Can't record: LiDAR depth is not available on this device."
            return
        }
        let fm = FileManager.default
        let fmt = DateFormatter()
        fmt.locale = Locale(identifier: "en_US_POSIX")
        fmt.dateFormat = "yyyyMMdd-HHmm"
        let now = Date()
        var name = "session_" + fmt.string(from: now)
        var folder = Self.documentsURL.appendingPathComponent(name, isDirectory: true)
        var n = 2
        while fm.fileExists(atPath: folder.path) {
            name = "session_" + fmt.string(from: now) + "_\(n)"
            folder = Self.documentsURL.appendingPathComponent(name, isDirectory: true)
            n += 1
        }
        let frames = folder.appendingPathComponent("frames", isDirectory: true)
        do {
            try fm.createDirectory(at: frames, withIntermediateDirectories: true)
        } catch {
            errorMessage = "Could not create session folder: \(error.localizedDescription)"
            return
        }

        let iso = ISO8601DateFormatter()
        iso.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        let sessionInfo: [String: Any] = [
            "device": "iphone",
            "model": Self.modelIdentifier(),
            "app_version": (Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String) ?? "1.0",
            "started": iso.string(from: now),
            "coord": "arkit_y_up",
            "depth_units": "meters",
            "has_person_mask": hasPersonMask,
        ]
        let sessionData = (try? JSONSerialization.data(withJSONObject: sessionInfo, options: [.prettyPrinted, .sortedKeys])) ?? Data("{}".utf8)
        do {
            try sessionData.write(to: folder.appendingPathComponent("session.json"), options: .atomic)
        } catch {
            errorMessage = "Could not write session.json: \(error.localizedDescription)"
            return
        }

        var streamer: LiveStreamer?
        streamWarning = nil
        if let host = streamHost {
            if let s = LiveStreamer(host: host) {
                streamer = s
                streamStatus = "Connecting…"
                s.onStateChange = { [weak self] state in
                    guard let self else { return }
                    switch state {
                    case .connecting: self.streamStatus = "Connecting…"
                    case .connected: self.streamStatus = "Streaming"
                    case .finished: self.streamStatus = "Sent"
                    case .failed(let msg):
                        self.streamStatus = "Failed"
                        self.streamWarning = "Live stream failed — still saving locally. \(msg)"
                    }
                }
                s.onDrop = { [weak self] in self?.droppedCount += 1 }
                let compact = (try? JSONSerialization.data(withJSONObject: sessionInfo, options: [.sortedKeys])) ?? sessionData
                s.start(sessionJSON: compact)
            } else {
                streamStatus = "Failed"
                streamWarning = "Invalid Mac IP address — saving locally only."
            }
        } else {
            streamStatus = "Off"
        }

        keyframeCount = 0
        droppedCount = 0
        startDate = now
        isRecording = true
        UIApplication.shared.isIdleTimerDisabled = true

        let rec = ActiveRecording(folder: folder, framesDir: frames, streamer: streamer)
        arQueue.async { self.recording = rec }
    }

    func stopRecording() {
        guard isRecording else { return }
        isRecording = false
        UIApplication.shared.isIdleTimerDisabled = false
        arQueue.async {
            guard let rec = self.recording else { return }
            self.recording = nil
            // writeQueue is serial: this runs after all pending keyframes are written.
            self.writeQueue.async {
                rec.streamer?.finish()
                DispatchQueue.main.async {
                    self.lastSessionURL = rec.folder
                    self.startDate = nil
                }
            }
        }
    }

    // MARK: - ARSessionDelegate (arQueue)

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        let tracking = frame.camera.trackingState
        let (label, normal) = Self.describe(tracking)
        if label != lastTrackingLabel {
            lastTrackingLabel = label
            DispatchQueue.main.async {
                self.trackingLabel = label
                self.trackingIsNormal = normal
            }
        }

        let depthData = frame.smoothedSceneDepth ?? frame.sceneDepth

        // Depth preview at ~5 Hz.
        if let depthData, frame.timestamp - lastPreviewTime > 0.2, !previewBusy,
           let plane = ImageEncoding.readPlane(depthData.depthMap, as: Float32.self) {
            lastPreviewTime = frame.timestamp
            previewBusy = true
            previewQueue.async {
                let img = ImageEncoding.depthPreview(plane.data, width: plane.width, height: plane.height)
                DispatchQueue.main.async { self.depthPreview = img }
                self.arQueue.async { self.previewBusy = false }
            }
        }

        guard let rec = recording else { return }
        if rec.startTimestamp == nil { rec.startTimestamp = frame.timestamp }
        let t = frame.timestamp - (rec.startTimestamp ?? frame.timestamp)
        let pose = frame.camera.transform

        guard rec.policy.shouldSave(pose: pose, time: t, trackingNormal: normal) else { return }
        guard let depthData else { return }

        pendingLock.lock()
        let backedUp = pendingWrites >= maxPendingWrites
        if !backedUp { pendingWrites += 1 }
        pendingLock.unlock()
        if backedUp {
            DispatchQueue.main.async { self.droppedCount += 1 }
            return   // don't commit policy; next frame will try again
        }

        guard let snapshot = makeSnapshot(frame: frame, depthData: depthData, index: rec.nextIndex,
                                          timestamp: t, pose: pose, tracking: Self.trackingString(tracking)) else {
            decrementPending()
            return
        }
        rec.policy.commit(pose: pose, time: t)
        rec.nextIndex += 1

        writeQueue.async {
            let ok = self.write(snapshot, to: rec)
            self.decrementPending()
            DispatchQueue.main.async {
                if ok { self.keyframeCount += 1 } else { self.droppedCount += 1 }
            }
        }
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        // If the semantic combination is rejected, retry without person segmentation.
        if usePersonSegmentation {
            usePersonSegmentation = false
            DispatchQueue.main.async {
                self.session.run(self.makeConfiguration(), options: [.resetTracking, .removeExistingAnchors])
            }
            return
        }
        DispatchQueue.main.async {
            self.errorMessage = "AR session error: \(error.localizedDescription)"
        }
    }

    func sessionWasInterrupted(_ session: ARSession) {
        DispatchQueue.main.async { self.trackingLabel = "Interrupted"; self.trackingIsNormal = false }
        lastTrackingLabel = "Interrupted"
    }

    private func decrementPending() {
        pendingLock.lock()
        pendingWrites -= 1
        pendingLock.unlock()
    }

    // MARK: - Snapshot + write

    private func makeSnapshot(frame: ARFrame, depthData: ARDepthData, index: Int, timestamp: Double,
                              pose: simd_float4x4, tracking: String) -> KeyframeSnapshot? {
        guard let rgb = ImageEncoding.copy(frame.capturedImage),
              let depth = ImageEncoding.readPlane(depthData.depthMap, as: Float32.self) else { return nil }

        // Confidence from sceneDepth (per spec), falling back to the smoothed map's confidence.
        var conf: [UInt8]?
        if let cmap = frame.sceneDepth?.confidenceMap ?? depthData.confidenceMap,
           let c = ImageEncoding.readPlane(cmap, as: UInt8.self) {
            conf = DepthConversion.resizeNearest(c.data, srcWidth: c.width, srcHeight: c.height,
                                                 dstWidth: depth.width, dstHeight: depth.height)
        }

        var mask: (data: [UInt8], width: Int, height: Int)?
        if let seg = frame.segmentationBuffer, let m = ImageEncoding.readPlane(seg, as: UInt8.self) {
            mask = (m.data, m.width, m.height)
        }

        let k = frame.camera.intrinsics   // column-major: [0][0]=fx, [1][1]=fy, [2][0]=cx, [2][1]=cy
        let res = frame.camera.imageResolution
        let intr = Intrinsics(fx: Double(k[0][0]), fy: Double(k[1][1]), cx: Double(k[2][0]), cy: Double(k[2][1]),
                              width: Int(res.width), height: Int(res.height))

        return KeyframeSnapshot(index: index, timestamp: timestamp, pose: pose, tracking: tracking,
                                rgb: rgb, rgbIntrinsics: intr, depth: depth.data,
                                depthWidth: depth.width, depthHeight: depth.height,
                                confidence: conf, mask: mask)
    }

    /// Runs on writeQueue.
    private func write(_ s: KeyframeSnapshot, to rec: ActiveRecording) -> Bool {
        let stem = String(format: "%06d", s.index)
        guard let (jpg, outW, outH) = ImageEncoding.jpeg(from: s.rgb, maxWidth: 960, quality: 0.85) else { return false }
        let mm = DepthConversion.metersToMillimeters(s.depth)
        guard let depthPNG = ImageEncoding.png16(mm, width: s.depthWidth, height: s.depthHeight) else { return false }

        var files: [(String, Data)] = [("\(stem).jpg", jpg), ("\(stem)_depth.png", depthPNG)]
        if let c = s.confidence, let png = ImageEncoding.png8(c, width: s.depthWidth, height: s.depthHeight) {
            files.append(("\(stem)_conf.png", png))
        }
        if let m = s.mask {
            let resized = DepthConversion.resizeNearest(m.data, srcWidth: m.width, srcHeight: m.height,
                                                        dstWidth: s.depthWidth, dstHeight: s.depthHeight)
            // ARKit segmentation: 255 = person, 0 = none. Normalise anything non-zero to 255.
            let binary = resized.map { $0 > 0 ? UInt8(255) : UInt8(0) }
            if let png = ImageEncoding.png8(binary, width: s.depthWidth, height: s.depthHeight) {
                files.append(("\(stem)_mask.png", png))
            }
        }

        let rgbIntr = s.rgbIntrinsics.scaled(toWidth: outW, height: outH)
        let depthIntr = s.rgbIntrinsics.scaled(toWidth: s.depthWidth, height: s.depthHeight)
        var meta: [String: Any] = [
            "index": s.index,
            "timestamp": s.timestamp,
            "pose": PoseFormat.rowMajor(s.pose),
            "intrinsics_rgb": rgbIntr.json,
            "intrinsics_depth": depthIntr.json,
            "tracking": s.tracking,
        ]

        do {
            for (name, data) in files {
                try data.write(to: rec.framesDir.appendingPathComponent(name))
            }
            let json = try JSONSerialization.data(withJSONObject: meta, options: [.sortedKeys])
            try json.write(to: rec.framesDir.appendingPathComponent("\(stem).json"))
        } catch {
            DispatchQueue.main.async { self.errorMessage = "Write failed: \(error.localizedDescription)" }
            return false
        }

        if let streamer = rec.streamer {
            meta["files"] = files.map { $0.0 }
            if let json = try? JSONSerialization.data(withJSONObject: meta, options: [.sortedKeys]) {
                streamer.sendKeyframe(json: json, binaries: files.map { $0.1 })
            }
        }
        return true
    }

    // MARK: - Helpers

    static func describe(_ state: ARCamera.TrackingState) -> (String, Bool) {
        switch state {
        case .normal: return ("Tracking: Normal", true)
        case .notAvailable: return ("Tracking: Not available", false)
        case .limited(let reason):
            let r: String
            switch reason {
            case .initializing: r = "initializing"
            case .excessiveMotion: r = "moving too fast"
            case .insufficientFeatures: r = "low detail"
            case .relocalizing: r = "relocalizing"
            @unknown default: r = "limited"
            }
            return ("Tracking: Limited (\(r))", false)
        }
    }

    static func trackingString(_ state: ARCamera.TrackingState) -> String {
        switch state {
        case .normal: return "normal"
        case .limited: return "limited"
        case .notAvailable: return "not_available"
        }
    }

    static func modelIdentifier() -> String {
        var info = utsname()
        uname(&info)
        let mirror = Mirror(reflecting: info.machine)
        let id = mirror.children.reduce(into: "") { acc, el in
            if let v = el.value as? Int8, v != 0 { acc.append(Character(UnicodeScalar(UInt8(v)))) }
        }
        return id.isEmpty ? UIDevice.current.model : id
    }
}
