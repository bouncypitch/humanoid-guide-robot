import SwiftUI
import UIKit

struct ContentView: View {
    @StateObject private var capture = CaptureManager()
    @AppStorage("macIP") private var macIP: String = ""
    @AppStorage("liveStream") private var liveStream: Bool = false
    @AppStorage("captureMode") private var mode: CaptureMode = .rgbd
    @Environment(\.scenePhase) private var scenePhase
    @StateObject private var roomScan = RoomScanController()
    @State private var showRoomScan = false

    @State private var zipping = false
    @State private var shareItem: ShareItem?
    @State private var alertText: String?

    var body: some View {
        ScrollView {
            VStack(spacing: 14) {
                header
                Picker("Mode", selection: $mode) {
                    ForEach(CaptureMode.allCases) { Text($0.title).tag($0) }
                }
                .pickerStyle(.segmented)
                .disabled(capture.isRecording)
                if mode == .rgbd {
                    if let msg = capture.supportMessage {
                        banner(msg, color: .orange)
                    }
                    if let warn = capture.streamWarning {
                        banner(warn, color: .yellow)
                    }
                    preview
                    stats
                    streamControls
                    recordButton
                } else {
                    roomPanel
                }
                shareButton
            }
            .padding(16)
        }
        .scrollDismissesKeyboard(.interactively)
        .onAppear {
            roomScan.onExported = { url in capture.lastSessionURL = url }
            if mode == .rgbd { capture.startSession() }
        }
        .onChange(of: mode) { _, newMode in
            if newMode == .rgbd { capture.startSession() } else { capture.pauseSession() }
        }
        .fullScreenCover(isPresented: $showRoomScan) {
            RoomScanScreen(controller: roomScan) {
                showRoomScan = false
                roomScan.reset()
            }
        }
        .onChange(of: scenePhase) { _, phase in
            switch phase {
            case .active: if mode == .rgbd { capture.startSession() }
            case .background: capture.pauseSession()
            default: break
            }
        }
        .onChange(of: capture.errorMessage) { _, msg in
            if let msg { alertText = msg; capture.errorMessage = nil }
        }
        .alert("RGBD Capture", isPresented: Binding(get: { alertText != nil }, set: { if !$0 { alertText = nil } })) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(alertText ?? "")
        }
        .sheet(item: $shareItem) { item in
            ActivityView(items: [item.url])
        }
    }

    // MARK: - Pieces

    private var roomPanel: some View {
        VStack(alignment: .leading, spacing: 12) {
            if !RoomScanController.isSupported {
                banner("RoomPlan is not supported on this device. It needs a LiDAR iPhone/iPad Pro and never runs in the Simulator.", color: .orange)
            }
            Text("How to scan").font(.headline)
            Text("1. Walk the walls first, keeping the phone pointed at them.\n2. Then sweep the middle of the room (furniture, floor).\n3. End where you started, then tap Done.\nUse \"Next room\" to walk into the next room without stopping: rooms are merged into one floor plan.")
                .font(.callout)
            Text("Room scans are saved locally only (no live stream). Use Share to AirDrop.")
                .font(.footnote).foregroundStyle(.secondary)
            Button {
                roomScan.start()
                showRoomScan = true
            } label: {
                Text("START ROOM SCAN")
                    .font(.system(size: 28, weight: .heavy))
                    .frame(maxWidth: .infinity, minHeight: 110)
                    .background(RoomScanController.isSupported ? Color.green : Color.gray)
                    .foregroundStyle(.white)
                    .clipShape(RoundedRectangle(cornerRadius: 22))
            }
            .disabled(!RoomScanController.isSupported)
        }
    }

    private var header: some View {
        HStack {
            Text("RGBD Capture").font(.title2.bold())
            Spacer()
            Text(capture.trackingLabel)
                .font(.caption.bold())
                .padding(.horizontal, 10).padding(.vertical, 5)
                .background(capture.trackingIsNormal ? Color.green.opacity(0.85) : Color.orange.opacity(0.85))
                .foregroundStyle(.white)
                .clipShape(Capsule())
        }
    }

    private func banner(_ text: String, color: Color) -> some View {
        HStack(alignment: .top) {
            Image(systemName: "exclamationmark.triangle.fill")
            Text(text).font(.footnote)
            Spacer(minLength: 0)
        }
        .padding(10)
        .background(color.opacity(0.25))
        .clipShape(RoundedRectangle(cornerRadius: 10))
    }

    private var preview: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 12).fill(Color.black)
            if let img = capture.depthPreview {
                Image(uiImage: img)
                    .resizable()
                    .interpolation(.none)
                    .aspectRatio(contentMode: .fit)
                    .clipShape(RoundedRectangle(cornerRadius: 12))
            } else {
                Text(capture.hasLiDAR ? "Waiting for depth…" : "No depth preview")
                    .foregroundStyle(.gray)
            }
        }
        .frame(height: 260)
    }

    private var stats: some View {
        HStack {
            stat("Keyframes", "\(capture.keyframeCount)")
            Spacer()
            TimelineView(.periodic(from: .now, by: 1)) { ctx in
                stat("Elapsed", elapsedString(now: ctx.date))
            }
            Spacer()
            stat("Dropped", "\(capture.droppedCount)")
            Spacer()
            stat("Stream", liveStream ? capture.streamStatus : "Off")
        }
    }

    private func stat(_ title: String, _ value: String) -> some View {
        VStack(spacing: 2) {
            Text(value).font(.title3.monospacedDigit().bold())
            Text(title).font(.caption).foregroundStyle(.secondary)
        }
    }

    private func elapsedString(now: Date) -> String {
        guard let start = capture.startDate else { return "0:00" }
        let s = max(0, Int(now.timeIntervalSince(start)))
        return String(format: "%d:%02d", s / 60, s % 60)
    }

    private var streamControls: some View {
        VStack(spacing: 8) {
            TextField("Mac IP (e.g. 192.168.1.20)", text: $macIP)
                .keyboardType(.numbersAndPunctuation)
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled()
                .textFieldStyle(.roundedBorder)
                .disabled(capture.isRecording)
            Toggle("Live stream to Mac (port 8770)", isOn: $liveStream)
                .disabled(capture.isRecording)
        }
    }

    private var recordButton: some View {
        Button {
            if capture.isRecording {
                capture.stopRecording()
            } else {
                hideKeyboard()
                capture.startRecording(streamHost: liveStream ? macIP : nil)
            }
        } label: {
            Text(capture.isRecording ? "STOP" : "START")
                .font(.system(size: 34, weight: .heavy))
                .frame(maxWidth: .infinity, minHeight: 110)
                .background(capture.isRecording ? Color.red : (capture.hasLiDAR ? Color.green : Color.gray))
                .foregroundStyle(.white)
                .clipShape(RoundedRectangle(cornerRadius: 22))
        }
        .disabled(!capture.hasLiDAR && !capture.isRecording)
    }

    private var shareButton: some View {
        Button {
            guard let folder = capture.lastSessionURL else { return }
            zipping = true
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    let zip = try SessionZipper.zip(folder: folder)
                    DispatchQueue.main.async { zipping = false; shareItem = ShareItem(url: zip) }
                } catch {
                    DispatchQueue.main.async { zipping = false; alertText = "Zip failed: \(error.localizedDescription)" }
                }
            }
        } label: {
            HStack {
                if zipping { ProgressView() } else { Image(systemName: "square.and.arrow.up") }
                Text(zipping ? "Zipping…" : shareTitle)
            }
            .frame(maxWidth: .infinity, minHeight: 50)
        }
        .buttonStyle(.bordered)
        .disabled(capture.isRecording || zipping || capture.lastSessionURL == nil)
    }

    private var shareTitle: String {
        if let url = capture.lastSessionURL { return "Share last session (\(url.lastPathComponent))" }
        return "Share last session"
    }

    private func hideKeyboard() {
        UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
    }
}

struct ShareItem: Identifiable {
    let url: URL
    var id: String { url.path }
}

struct ActivityView: UIViewControllerRepresentable {
    let items: [Any]
    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: items, applicationActivities: nil)
    }
    func updateUIViewController(_ vc: UIActivityViewController, context: Context) {}
}

enum CaptureMode: String, CaseIterable, Identifiable {
    case rgbd, roomplan
    var id: String { rawValue }
    var title: String { self == .rgbd ? "RGB-D walkthrough" : "Room scan (RoomPlan)" }
}
