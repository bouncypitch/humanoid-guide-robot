import Foundation

/// Streams a session over WebSocket per docs/rgbd_format.md:
///   1. text: session.json
///   2. per keyframe: text frame JSON (with "files"), then one binary message per file, in order
///   3. text: {"end": true}
/// All sends are serialized on a private queue. Any failure flips the streamer into a
/// failed state; it never throws or blocks the recorder (local saving continues).
final class LiveStreamer: NSObject, URLSessionWebSocketDelegate {
    enum State: Equatable {
        case connecting
        case connected
        case failed(String)
        case finished
    }

    /// Called on the main queue.
    var onStateChange: ((State) -> Void)?
    /// Called on the main queue when a keyframe is dropped because the network is behind.
    var onDrop: (() -> Void)?

    private let queue = DispatchQueue(label: "rgbd.livestream")
    private var urlSession: URLSession!
    private var task: URLSessionWebSocketTask!
    private var state: State = .connecting
    private var outbox: [[URLSessionWebSocketTask.Message]] = []   // groups = keyframes
    private var current: [URLSessionWebSocketTask.Message] = []
    private var sending = false
    private var finishRequested = false
    private let maxQueuedGroups = 8

    init?(host: String) {
        let trimmed = host.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, let url = URL(string: "ws://\(trimmed):8770/ingest") else { return nil }
        super.init()
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 8
        config.waitsForConnectivity = false
        urlSession = URLSession(configuration: config, delegate: self, delegateQueue: nil)
        var request = URLRequest(url: url)
        request.timeoutInterval = 8
        task = urlSession.webSocketTask(with: request)
        task.maximumMessageSize = 64 * 1024 * 1024
    }

    func start(sessionJSON: Data) {
        queue.async {
            self.task.resume()
            self.receiveLoop()
            let text = String(decoding: sessionJSON, as: UTF8.self)
            self.outbox.append([.string(text)])
            self.pump()
        }
    }

    /// Enqueue one keyframe: metadata JSON (must already contain "files") + binaries in order.
    func sendKeyframe(json: Data, binaries: [Data]) {
        queue.async {
            guard !self.isDead, !self.finishRequested else { return }
            if self.outbox.count >= self.maxQueuedGroups {
                DispatchQueue.main.async { self.onDrop?() }
                return
            }
            var group: [URLSessionWebSocketTask.Message] = [.string(String(decoding: json, as: UTF8.self))]
            group += binaries.map { .data($0) }
            self.outbox.append(group)
            self.pump()
        }
    }

    /// Sends {"end": true} after everything queued, then closes.
    func finish() {
        queue.async {
            guard !self.isDead else { self.teardown(); return }
            self.finishRequested = true
            self.outbox.append([.string("{\"end\": true}")])
            self.pump()
        }
    }

    // MARK: - Internals (all on `queue`)

    private var isDead: Bool {
        if case .failed = state { return true }
        return state == .finished
    }

    private func pump() {
        guard !sending, !isDead else { return }
        if current.isEmpty {
            if outbox.isEmpty {
                if finishRequested { closeNormally() }
                return
            }
            current = outbox.removeFirst()
        }
        let msg = current.removeFirst()
        sending = true
        task.send(msg) { [weak self] error in
            guard let self else { return }
            self.queue.async {
                self.sending = false
                if let error {
                    self.fail("Send failed: \(error.localizedDescription)")
                    return
                }
                self.pump()
            }
        }
    }

    private func receiveLoop() {
        task.receive { [weak self] result in
            guard let self else { return }
            self.queue.async {
                switch result {
                case .success:
                    self.receiveLoop()   // ignore server messages (acks etc.)
                case .failure(let error):
                    if !self.isDead && !self.finishRequested {
                        self.fail("Connection lost: \(error.localizedDescription)")
                    }
                }
            }
        }
    }

    private func setState(_ s: State) {
        state = s
        DispatchQueue.main.async { self.onStateChange?(s) }
    }

    private func fail(_ message: String) {
        guard !isDead else { return }
        setState(.failed(message))
        outbox.removeAll()
        current.removeAll()
        teardown()
    }

    private func closeNormally() {
        guard !isDead else { return }
        setState(.finished)
        task.cancel(with: .normalClosure, reason: nil)
        urlSession.finishTasksAndInvalidate()
    }

    private func teardown() {
        task.cancel(with: .goingAway, reason: nil)
        urlSession.invalidateAndCancel()
    }

    // MARK: - URLSessionWebSocketDelegate

    func urlSession(_ session: URLSession, webSocketTask: URLSessionWebSocketTask,
                    didOpenWithProtocol protocol: String?) {
        queue.async {
            if self.state == .connecting { self.setState(.connected) }
        }
    }

    func urlSession(_ session: URLSession, webSocketTask: URLSessionWebSocketTask,
                    didCloseWith closeCode: URLSessionWebSocketTask.CloseCode, reason: Data?) {
        queue.async {
            if !self.finishRequested { self.fail("Server closed the connection (\(closeCode.rawValue))") }
        }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        guard let error else { return }
        queue.async {
            if !self.finishRequested { self.fail("Connection error: \(error.localizedDescription)") }
        }
    }
}
