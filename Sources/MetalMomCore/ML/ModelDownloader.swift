import Foundation
#if canImport(CoreML)
import CoreML
#endif

/// Families of pre-trained CoreML models available for download.
public enum ModelFamily: String, CaseIterable, Sendable {
    case rnnBeatProcessor = "rnn_beat_processor"
    case rnnOnsetProcessor = "rnn_onset_processor"
    case rnnDownbeatProcessor = "rnn_downbeat_processor"
    case cnnOnsetDetector = "cnn_onset_detector"
    case combFilterTempoEstimator = "comb_filter_tempo"
    case cnnChordRecognizer = "cnn_chord_recognizer"
    case keyDetector = "key_detector"
    case pianoTranscriber = "piano_transcriber"
    case spectralOnsetProcessor = "spectral_onset_processor"
}

/// Downloads and caches CoreML models from Hugging Face Hub.
///
/// Core models (beat tracking, onset, downbeat, tempo, onset detector) are
/// bundled with the package. Extended models are downloaded on demand.
///
/// Thread-safe: all public API is protected by an internal lock.
public final class ModelDownloader: @unchecked Sendable {
    public static let shared = ModelDownloader()

    /// Base URL for the Hugging Face model repository.
    public var repositoryURL: URL

    private let lock = NSLock()

    /// Local cache directory for downloaded models.
    public var cacheDirectory: URL {
        let appSupport = FileManager.default.urls(
            for: .applicationSupportDirectory, in: .userDomainMask
        ).first!
        return appSupport.appendingPathComponent("MetalMom/Models")
    }

    private init() {
        self.repositoryURL = URL(string: "https://huggingface.co/zkeown/metalmom-coreml-models/resolve/main/")!
    }

    /// Check if a model family is already cached locally.
    public func isCached(_ family: ModelFamily) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        let modelDir = cacheDirectory.appendingPathComponent("\(family.rawValue).mlmodelc")
        var isDir: ObjCBool = false
        return FileManager.default.fileExists(atPath: modelDir.path, isDirectory: &isDir)
            && isDir.boolValue
    }

    /// Download a model family from HF Hub, compile if needed, and cache locally.
    ///
    /// - Parameters:
    ///   - family: The model family to download.
    ///   - progress: Optional progress callback (0.0 to 1.0).
    /// - Returns: URL to the compiled `.mlmodelc` bundle.
    @available(macOS 12.0, iOS 15.0, *)
    public func download(
        _ family: ModelFamily,
        progress: ((Double) -> Void)? = nil
    ) async throws -> URL {
        let cachedURL = cacheDirectory.appendingPathComponent("\(family.rawValue).mlmodelc")

        // Return cached if available
        if isCached(family) {
            return cachedURL
        }

        // Ensure cache directory exists
        try FileManager.default.createDirectory(
            at: cacheDirectory, withIntermediateDirectories: true
        )

        // Download from HF
        let remoteURL = repositoryURL
            .appendingPathComponent("extended")
            .appendingPathComponent("\(family.rawValue).mlmodel")

        let (tempURL, _) = try await URLSession.shared.download(from: remoteURL)

        // Compile the model on-device
        let compiledURL = try await MLModel.compileModel(at: tempURL)

        // Move to cache
        if FileManager.default.fileExists(atPath: cachedURL.path) {
            try FileManager.default.removeItem(at: cachedURL)
        }
        try FileManager.default.moveItem(at: compiledURL, to: cachedURL)

        // Clean up temp file
        try? FileManager.default.removeItem(at: tempURL)

        return cachedURL
    }

    /// Clear all cached models.
    public func clearCache() throws {
        lock.lock()
        defer { lock.unlock() }
        if FileManager.default.fileExists(atPath: cacheDirectory.path) {
            try FileManager.default.removeItem(at: cacheDirectory)
        }
    }
}
