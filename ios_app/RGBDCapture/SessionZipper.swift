import Foundation

enum SessionZipper {
    enum ZipError: LocalizedError {
        case failed(String)
        var errorDescription: String? {
            if case .failed(let m) = self { return m }
            return nil
        }
    }

    /// Zips `folder` to <tmp>/<folder name>.zip. The archive contains the folder itself
    /// as its top-level directory (session_xxx/session.json, session_xxx/frames/...).
    /// First tries NSFileCoordinator(.forUploading) (system zip), then falls back to a
    /// minimal store-only ZIP writer.
    static func zip(folder: URL) throws -> URL {
        let dest = FileManager.default.temporaryDirectory
            .appendingPathComponent(folder.lastPathComponent + ".zip")
        try? FileManager.default.removeItem(at: dest)

        var coordError: NSError?
        var copyError: Error?
        NSFileCoordinator().coordinate(readingItemAt: folder, options: [.forUploading], error: &coordError) { zipURL in
            do { try FileManager.default.copyItem(at: zipURL, to: dest) } catch { copyError = error }
        }
        if coordError == nil, copyError == nil, FileManager.default.fileExists(atPath: dest.path) {
            return dest
        }

        try? FileManager.default.removeItem(at: dest)
        try StoreZipWriter.write(folder: folder, to: dest)
        return dest
    }
}

/// Minimal store-only (no compression) ZIP writer. Streams file contents to disk.
/// Limits: < 4 GB total, < 65535 entries (plenty for a walkthrough).
enum StoreZipWriter {
    private struct Entry {
        let name: Data
        let crc: UInt32
        let size: UInt32
        let offset: UInt32
    }

    static func write(folder: URL, to dest: URL) throws {
        let fm = FileManager.default
        let root = folder.lastPathComponent
        guard let enumerator = fm.enumerator(at: folder, includingPropertiesForKeys: [.isDirectoryKey]) else {
            throw SessionZipper.ZipError.failed("Cannot read \(folder.path)")
        }
        var files: [(URL, String)] = []
        let basePath = folder.standardizedFileURL.resolvingSymlinksInPath().path
        for case let url as URL in enumerator {
            let isDir = (try? url.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) ?? false
            if isDir { continue }
            let full = url.standardizedFileURL.resolvingSymlinksInPath().path
            var rel = String(full.dropFirst(basePath.count))
            if rel.hasPrefix("/") { rel.removeFirst() }
            files.append((url, root + "/" + rel))
        }
        files.sort { $0.1 < $1.1 }

        fm.createFile(atPath: dest.path, contents: nil)
        let out = try FileHandle(forWritingTo: dest)
        defer { try? out.close() }

        var entries: [Entry] = []
        var offset: UInt64 = 0
        for (url, name) in files {
            let data = try Data(contentsOf: url, options: .mappedIfSafe)
            let crc = CRC32.checksum(data)
            let nameData = Data(name.utf8)
            guard offset < UInt64(UInt32.max), data.count < Int(UInt32.max) else {
                throw SessionZipper.ZipError.failed("Session too large for zip writer")
            }
            var h = Data()
            h.le32(0x04034b50); h.le16(20); h.le16(0x0800); h.le16(0)   // sig, version, flags(UTF8), store
            h.le16(0); h.le16(0x21)                                       // mod time/date (1980-01-01)
            h.le32(crc); h.le32(UInt32(data.count)); h.le32(UInt32(data.count))
            h.le16(UInt16(nameData.count)); h.le16(0)
            h.append(nameData)
            try out.write(contentsOf: h)
            try out.write(contentsOf: data)
            entries.append(Entry(name: nameData, crc: crc, size: UInt32(data.count), offset: UInt32(offset)))
            offset += UInt64(h.count + data.count)
        }

        let cdStart = offset
        var cd = Data()
        for e in entries {
            cd.le32(0x02014b50); cd.le16(20); cd.le16(20); cd.le16(0x0800); cd.le16(0)
            cd.le16(0); cd.le16(0x21)
            cd.le32(e.crc); cd.le32(e.size); cd.le32(e.size)
            cd.le16(UInt16(e.name.count)); cd.le16(0); cd.le16(0); cd.le16(0); cd.le16(0)
            cd.le32(0); cd.le32(e.offset)
            cd.append(e.name)
        }
        var end = Data()
        end.le32(0x06054b50); end.le16(0); end.le16(0)
        end.le16(UInt16(entries.count)); end.le16(UInt16(entries.count))
        end.le32(UInt32(cd.count)); end.le32(UInt32(cdStart)); end.le16(0)
        try out.write(contentsOf: cd)
        try out.write(contentsOf: end)
    }
}

enum CRC32 {
    private static let table: [UInt32] = (0..<256).map { i -> UInt32 in
        var c = UInt32(i)
        for _ in 0..<8 { c = (c & 1) != 0 ? (0xEDB88320 ^ (c >> 1)) : (c >> 1) }
        return c
    }

    static func checksum(_ data: Data) -> UInt32 {
        var crc: UInt32 = 0xFFFFFFFF
        data.withUnsafeBytes { (buf: UnsafeRawBufferPointer) in
            for b in buf { crc = table[Int((crc ^ UInt32(b)) & 0xFF)] ^ (crc >> 8) }
        }
        return crc ^ 0xFFFFFFFF
    }
}

private extension Data {
    mutating func le16(_ v: UInt16) { append(UInt8(v & 0xFF)); append(UInt8(v >> 8)) }
    mutating func le32(_ v: UInt32) { for s in stride(from: 0, to: 32, by: 8) { append(UInt8((v >> UInt32(s)) & 0xFF)) } }
}
