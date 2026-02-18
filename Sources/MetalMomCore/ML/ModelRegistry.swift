import Foundation

// MARK: - Errors

/// Errors that can occur when using the model registry.
public enum ModelRegistryError: Error, Equatable, Sendable {
    /// The registry has not been configured with a models directory.
    case notConfigured
    /// No model with the given name was found in the models directory.
    case modelNotFound(String)
}

// MARK: - ModelRegistry

/// Discovers, loads, and caches pre-trained CoreML models from a directory.
///
/// Models are expected to be compiled `.mlmodelc` bundles stored in a single
/// directory.  The registry lazily scans that directory on first access to
/// ``availableModels`` and loads each model on first request via
/// ``model(named:)``.
///
/// Thread-safe: all public API is protected by an internal lock.
///
/// Usage:
/// ```swift
/// let registry = ModelRegistry.shared
/// registry.configure(modelsDirectory: someURL)
/// let names = registry.availableModels   // ["beat_rnn_0", "onset_cnn", ...]
/// let engine = try registry.model(named: "beat_rnn_0")
/// ```
public final class ModelRegistry: @unchecked Sendable {
    /// Shared singleton instance.
    public static let shared = ModelRegistry()

    /// The directory where compiled `.mlmodelc` bundles are stored.
    private var modelsDirectory: URL?

    /// Cache of loaded ``InferenceEngine`` instances keyed by model name.
    private var cache: [String: InferenceEngine] = [:]

    /// Lock for thread safety.
    private let lock = NSLock()

    /// Lazily-discovered model names (reset when directory changes).
    private var _availableModels: [String]?

    // MARK: - Initializer

    public init() {}

    // MARK: - Configuration

    /// Set the directory containing compiled `.mlmodelc` bundles.
    ///
    /// Resets the list of discovered models so they are re-scanned on next
    /// access to ``availableModels``.  Already-loaded models remain cached.
    public func configure(modelsDirectory: URL) {
        lock.lock()
        defer { lock.unlock() }
        self.modelsDirectory = modelsDirectory
        self._availableModels = nil  // force re-discovery
    }

    // MARK: - Discovery

    /// Names of models available in the configured directory (sorted).
    ///
    /// The directory is scanned lazily on first access; the result is cached
    /// until ``configure(modelsDirectory:)`` is called again.  Returns an
    /// empty array if the registry has not been configured.
    public var availableModels: [String] {
        lock.lock()
        defer { lock.unlock() }

        if let cached = _availableModels {
            return cached
        }

        guard let dir = modelsDirectory else { return [] }
        let fm = FileManager.default
        guard let contents = try? fm.contentsOfDirectory(
            at: dir,
            includingPropertiesForKeys: nil
        ) else {
            return []
        }

        let models = contents
            .filter { $0.pathExtension == "mlmodelc" }
            .map { $0.deletingPathExtension().lastPathComponent }
            .sorted()
        _availableModels = models
        return models
    }

    // MARK: - Loading

    /// Load and return an ``InferenceEngine`` for the given model name.
    ///
    /// On the first call the model is loaded from disk and cached; subsequent
    /// calls return the cached engine.
    ///
    /// - Parameter name: Base name of the model (without `.mlmodelc` extension).
    /// - Returns: A ready-to-use ``InferenceEngine``.
    /// - Throws: ``ModelRegistryError/notConfigured`` if no directory has been
    ///   set, ``ModelRegistryError/modelNotFound(_:)`` if the bundle does not
    ///   exist, or an ``InferenceError`` if loading fails.
    public func model(named name: String) throws -> InferenceEngine {
        lock.lock()
        defer { lock.unlock() }

        if let cached = cache[name] {
            return cached
        }

        guard let dir = modelsDirectory else {
            throw ModelRegistryError.notConfigured
        }

        let modelURL = dir.appendingPathComponent("\(name).mlmodelc")
        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            throw ModelRegistryError.modelNotFound(name)
        }

        let engine = try InferenceEngine(compiledModelURL: modelURL)
        cache[name] = engine
        return engine
    }

    // MARK: - Cache inspection

    /// Whether a model with the given name is currently loaded in cache.
    public func isLoaded(name: String) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return cache[name] != nil
    }

    // MARK: - Memory management

    /// Remove a specific model from the cache, freeing its memory.
    public func unload(name: String) {
        lock.lock()
        defer { lock.unlock() }
        cache.removeValue(forKey: name)
    }

    /// Remove all models from the cache, freeing their memory.
    public func unloadAll() {
        lock.lock()
        defer { lock.unlock() }
        cache.removeAll()
    }
}
