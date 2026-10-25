import Foundation
import SwiftUI
import UIKit
import RoomPlan

/// Drives a RoomPlan scan through RoomCaptureView (standard on-screen coaching), builds
/// CapturedRooms with RoomBuilder(.beautifyObjects), optionally merges several rooms with
/// StructureBuilder, and exports a `session_<ts>_roomplan/` folder.
@MainActor
final class RoomScanController: NSObject, ObservableObject {
    enum Phase: Equatable {
        case idle
        case scanning
        case processing(String)
        case done(URL)
        case failed(String)
    }

    @Published var phase: Phase = .idle
    @Published var roomCount = 0

    static var isSupported: Bool { RoomCaptureSession.isSupported }

    private(set) var captureView: RoomCaptureView?
    private var rooms: [CapturedRoom] = []
    private var continueAfterStop = false
    private var startedAt = Date()

    /// Called on main with the exported folder.
    var onExported: ((URL) -> Void)?

    private lazy var delegateProxy = RoomCaptureDelegateProxy(owner: self)

    // MARK: - Control

    func makeView() -> RoomCaptureView {
        if let v = captureView { return v }
        let v = RoomCaptureView(frame: .zero)
        v.delegate = delegateProxy
        captureView = v
        return v
    }

    func start() {
        guard Self.isSupported else {
            phase = .failed("RoomPlan is not supported on this device (needs LiDAR; never available in the Simulator).")
            return
        }
        rooms = []
        roomCount = 0
        startedAt = Date()
        phase = .scanning
        makeView().captureSession.run(configuration: RoomCaptureSession.Configuration())
    }

    /// Finish the current room. `addAnother` keeps the ARSession alive so the next room
    /// shares the same world frame (required for StructureBuilder).
    func finishRoom(addAnother: Bool) {
        guard phase == .scanning, let v = captureView else { return }
        continueAfterStop = addAnother
        phase = .processing("Processing room \(rooms.count + 1)…")
        v.captureSession.stop(pauseARSession: !addAnother)
    }

    func cancel() {
        captureView?.captureSession.stop()
        rooms = []
        roomCount = 0
        phase = .idle
    }

    func reset() {
        captureView = nil
        phase = .idle
    }

    // MARK: - Build + export

    func build(data: CapturedRoomData, captureError: Error?) async {
        do {
            let room = try await RoomBuilder(options: [.beautifyObjects]).capturedRoom(from: data)
            rooms.append(room)
            roomCount = rooms.count
        } catch {
            let msg = (captureError ?? error).localizedDescription
            if rooms.isEmpty {
                phase = .failed("Room processing failed: \(msg)")
                return
            }
            // Keep earlier rooms; just report.
            phase = .failed("Room \(rooms.count + 1) failed (\(msg)). Earlier rooms kept — tap Export.")
            return
        }

        if continueAfterStop {
            phase = .scanning
            captureView?.captureSession.run(configuration: RoomCaptureSession.Configuration())
        } else {
            await export()
        }
    }

    /// Export whatever rooms have been captured so far.
    func export() async {
        guard !rooms.isEmpty else { phase = .failed("Nothing captured."); return }
        captureView?.captureSession.stop()
        phase = .processing("Exporting…")

        var structure: CapturedStructure?
        var structureError: String?
        if rooms.count > 1 {
            do {
                structure = try await StructureBuilder(options: [.beautifyObjects]).capturedStructure(from: rooms)
            } catch {
                structureError = error.localizedDescription
            }
        }

        let rooms = self.rooms
        let started = startedAt
        do {
            let folder = try await Task.detached(priority: .userInitiated) {
                try RoomPlanExporter.write(rooms: rooms, structure: structure,
                                           structureError: structureError, started: started)
            }.value
            phase = .done(folder)
            onExported?(folder)
        } catch {
            phase = .failed("Export failed: \(error.localizedDescription)")
        }
    }

    var hasRooms: Bool { !rooms.isEmpty }
}

/// RoomCaptureViewDelegate inherits NSCoding, which can't be main-actor isolated,
/// so a tiny non-isolated proxy forwards to the controller.
final class RoomCaptureDelegateProxy: NSObject, RoomCaptureViewDelegate {
    private weak var owner: RoomScanController?

    init(owner: RoomScanController) {
        self.owner = owner
        super.init()
    }

    required init?(coder: NSCoder) { nil }
    func encode(with coder: NSCoder) {}

    func captureView(shouldPresent roomDataForProcessing: CapturedRoomData, error: Error?) -> Bool {
        let data = roomDataForProcessing
        Task { @MainActor [weak owner] in
            await owner?.build(data: data, captureError: error)
        }
        return false   // we export ourselves; skip RoomPlan's built-in result UI
    }

    func captureView(didPresent processedResult: CapturedRoom, error: Error?) {}
}

enum RoomPlanExporter {
    /// Layout (see docs/rgbd_format.md, "RoomPlan session"):
    ///   session_<ts>_roomplan/
    ///     session.json
    ///     roomplan.json / roomplan.usdz        first (or only) room
    ///     rooms/room_NN.json / .usdz           every room, only when > 1 room
    ///     structure.json / structure.usdz      merged CapturedStructure, only when > 1 room and merge ok
    static func write(rooms: [CapturedRoom], structure: CapturedStructure?,
                      structureError: String?, started: Date) throws -> URL {
        let fm = FileManager.default
        let fmt = DateFormatter()
        fmt.locale = Locale(identifier: "en_US_POSIX")
        fmt.dateFormat = "yyyyMMdd-HHmm"
        let ts = fmt.string(from: started)
        let docs = CaptureManager.documentsURL
        var folder = docs.appendingPathComponent("session_\(ts)_roomplan", isDirectory: true)
        var n = 2
        while fm.fileExists(atPath: folder.path) {
            folder = docs.appendingPathComponent("session_\(ts)_roomplan_\(n)", isDirectory: true)
            n += 1
        }
        try fm.createDirectory(at: folder, withIntermediateDirectories: true)

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        var files: [String] = []
        var warnings: [String] = []

        func save(_ room: CapturedRoom, stem: String) throws {
            try encoder.encode(room).write(to: folder.appendingPathComponent("\(stem).json"))
            files.append("\(stem).json")
            do {
                try room.export(to: folder.appendingPathComponent("\(stem).usdz"), exportOptions: .parametric)
                files.append("\(stem).usdz")
            } catch {
                warnings.append("\(stem).usdz export failed: \(error.localizedDescription)")
            }
        }

        try save(rooms[0], stem: "roomplan")
        if rooms.count > 1 {
            try fm.createDirectory(at: folder.appendingPathComponent("rooms"), withIntermediateDirectories: true)
            for (i, room) in rooms.enumerated() {
                try save(room, stem: String(format: "rooms/room_%02d", i))
            }
        }
        if let structure {
            try encoder.encode(structure).write(to: folder.appendingPathComponent("structure.json"))
            files.append("structure.json")
            do {
                try structure.export(to: folder.appendingPathComponent("structure.usdz"), exportOptions: .parametric)
                files.append("structure.usdz")
            } catch {
                warnings.append("structure.usdz export failed: \(error.localizedDescription)")
            }
        } else if let structureError {
            warnings.append("StructureBuilder failed: \(structureError)")
        }

        let iso = ISO8601DateFormatter()
        iso.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        var info: [String: Any] = [
            "device": "iphone",
            "kind": "roomplan",
            "model": CaptureManager.modelIdentifier(),
            "app_version": (Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String) ?? "1.0",
            "started": iso.string(from: started),
            "coord": "arkit_y_up",
            "units": "meters",
            "room_count": rooms.count,
            "has_structure": structure != nil,
            "files": files,
        ]
        if !warnings.isEmpty { info["warnings"] = warnings }
        let data = try JSONSerialization.data(withJSONObject: info, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: folder.appendingPathComponent("session.json"), options: .atomic)
        return folder
    }
}

/// Hosts the controller's RoomCaptureView in SwiftUI.
struct RoomCaptureContainer: UIViewRepresentable {
    let controller: RoomScanController
    func makeUIView(context: Context) -> RoomCaptureView { controller.makeView() }
    func updateUIView(_ uiView: RoomCaptureView, context: Context) {}
}

/// Full-screen scanning UI.
struct RoomScanScreen: View {
    @ObservedObject var controller: RoomScanController
    let onClose: () -> Void

    var body: some View {
        ZStack {
            RoomCaptureContainer(controller: controller).ignoresSafeArea()
            VStack {
                HStack {
                    Button("Cancel") { controller.cancel(); onClose() }
                        .buttonStyle(.borderedProminent).tint(.gray)
                    Spacer()
                    if controller.roomCount > 0 {
                        Text("Rooms: \(controller.roomCount)")
                            .font(.caption.bold()).padding(8)
                            .background(.ultraThinMaterial).clipShape(Capsule())
                    }
                }
                .padding()
                Spacer()
                bottom.padding()
            }
        }
    }

    @ViewBuilder private var bottom: some View {
        switch controller.phase {
        case .scanning:
            HStack(spacing: 12) {
                Button { controller.finishRoom(addAnother: true) } label: {
                    Text("Next room").frame(maxWidth: .infinity, minHeight: 50)
                }
                .buttonStyle(.borderedProminent).tint(.blue)
                Button { controller.finishRoom(addAnother: false) } label: {
                    Text("Done").font(.title3.bold()).frame(maxWidth: .infinity, minHeight: 50)
                }
                .buttonStyle(.borderedProminent).tint(.green)
            }
        case .processing(let msg):
            HStack { ProgressView(); Text(msg) }
                .padding().background(.ultraThinMaterial).clipShape(RoundedRectangle(cornerRadius: 12))
        case .done(let url):
            VStack(spacing: 8) {
                Text("Saved \(url.lastPathComponent)").font(.headline)
                Button("Close") { onClose() }.buttonStyle(.borderedProminent)
            }
            .padding().background(.ultraThinMaterial).clipShape(RoundedRectangle(cornerRadius: 12))
        case .failed(let msg):
            VStack(spacing: 8) {
                Text(msg).font(.footnote)
                HStack {
                    if controller.hasRooms {
                        Button("Export") { Task { await controller.export() } }.buttonStyle(.borderedProminent)
                    }
                    Button("Close") { controller.cancel(); onClose() }.buttonStyle(.bordered)
                }
            }
            .padding().background(.ultraThinMaterial).clipShape(RoundedRectangle(cornerRadius: 12))
        case .idle:
            EmptyView()
        }
    }
}
