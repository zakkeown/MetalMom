import XCTest
@testable import MetalMomCore

final class ModelRegistryTests: XCTestCase {

    // MARK: - Helpers

    /// Resolve path to the Tests/fixtures directory via #filePath.
    private static var fixturesURL: URL {
        let thisFile = URL(fileURLWithPath: #filePath)
        let testsDir = thisFile
            .deletingLastPathComponent()   // MetalMomTests/
            .deletingLastPathComponent()   // Tests/
        return testsDir.appendingPathComponent("fixtures")
    }

    /// Fresh registry for each test — avoids shared state between tests.
    private func makeRegistry() -> ModelRegistry {
        ModelRegistry()
    }

    // MARK: - Not-configured tests

    func testAvailableModelsReturnsEmptyWhenNotConfigured() {
        let registry = makeRegistry()
        XCTAssertEqual(registry.availableModels, [])
    }

    func testModelNamedThrowsNotConfigured() {
        let registry = makeRegistry()
        XCTAssertThrowsError(try registry.model(named: "anything")) { error in
            guard let registryError = error as? ModelRegistryError else {
                XCTFail("Expected ModelRegistryError, got \(error)")
                return
            }
            XCTAssertEqual(registryError, .notConfigured)
        }
    }

    // MARK: - Discovery tests

    func testDiscoverAvailableModels() {
        let registry = makeRegistry()
        registry.configure(modelsDirectory: Self.fixturesURL)

        let models = registry.availableModels
        // Fixtures contain test_double.mlmodelc and test_identity.mlmodelc
        XCTAssertTrue(models.contains("test_identity"),
            "Should discover test_identity model")
        XCTAssertTrue(models.contains("test_double"),
            "Should discover test_double model")
        // Should be sorted
        XCTAssertEqual(models, models.sorted(),
            "Available models should be sorted")
    }

    func testAvailableModelsCached() {
        let registry = makeRegistry()
        registry.configure(modelsDirectory: Self.fixturesURL)

        let first = registry.availableModels
        let second = registry.availableModels
        // Same result each time (cached).
        XCTAssertEqual(first, second)
    }

    func testConfigureResetsDiscovery() {
        let registry = makeRegistry()
        registry.configure(modelsDirectory: Self.fixturesURL)
        let before = registry.availableModels
        XCTAssertFalse(before.isEmpty)

        // Reconfigure with a bogus directory.
        let bogus = FileManager.default.temporaryDirectory.appendingPathComponent("nonexistent_models_dir")
        registry.configure(modelsDirectory: bogus)
        XCTAssertEqual(registry.availableModels, [],
            "Should re-scan after reconfigure")
    }

    func testAvailableModelsWithBogusDirectory() {
        let registry = makeRegistry()
        let bogus = FileManager.default.temporaryDirectory.appendingPathComponent("nonexistent_models_dir_\(UUID().uuidString)")
        registry.configure(modelsDirectory: bogus)
        XCTAssertEqual(registry.availableModels, [])
    }

    // MARK: - Loading and caching

    func testModelNamedLoadsAndCaches() throws {
        guard fixtureExists("test_identity.mlmodelc") else {
            throw XCTSkip("test_identity.mlmodelc not found — run scripts/create_test_model.py first")
        }

        let registry = makeRegistry()
        registry.configure(modelsDirectory: Self.fixturesURL)

        XCTAssertFalse(registry.isLoaded(name: "test_identity"))

        let engine = try registry.model(named: "test_identity")
        XCTAssertTrue(registry.isLoaded(name: "test_identity"),
            "Model should be cached after loading")

        // Verify the engine works.
        let input = Signal(data: [1, 2, 3, 4], shape: [1, 4], sampleRate: 22050)
        let output = try engine.predict(input: input)
        output.withUnsafeBufferPointer { buf in
            for i in 0..<4 {
                XCTAssertEqual(buf[i], Float(i + 1), accuracy: 1e-4)
            }
        }
    }

    func testModelNamedReturnsCachedInstance() throws {
        guard fixtureExists("test_identity.mlmodelc") else {
            throw XCTSkip("test_identity.mlmodelc not found")
        }

        let registry = makeRegistry()
        registry.configure(modelsDirectory: Self.fixturesURL)

        let first = try registry.model(named: "test_identity")
        let second = try registry.model(named: "test_identity")
        // Should be the exact same object (identity).
        XCTAssertTrue(first === second,
            "Subsequent calls should return the cached instance")
    }

    func testModelNotFoundThrows() {
        let registry = makeRegistry()
        registry.configure(modelsDirectory: Self.fixturesURL)

        XCTAssertThrowsError(try registry.model(named: "nonexistent_model")) { error in
            guard let registryError = error as? ModelRegistryError else {
                XCTFail("Expected ModelRegistryError, got \(error)")
                return
            }
            XCTAssertEqual(registryError, .modelNotFound("nonexistent_model"))
        }
    }

    // MARK: - Unloading

    func testUnloadRemovesFromCache() throws {
        guard fixtureExists("test_identity.mlmodelc") else {
            throw XCTSkip("test_identity.mlmodelc not found")
        }

        let registry = makeRegistry()
        registry.configure(modelsDirectory: Self.fixturesURL)

        _ = try registry.model(named: "test_identity")
        XCTAssertTrue(registry.isLoaded(name: "test_identity"))

        registry.unload(name: "test_identity")
        XCTAssertFalse(registry.isLoaded(name: "test_identity"),
            "Model should be removed after unload")
    }

    func testUnloadAllClearsCache() throws {
        guard fixtureExists("test_identity.mlmodelc"),
              fixtureExists("test_double.mlmodelc") else {
            throw XCTSkip("Test fixtures not found")
        }

        let registry = makeRegistry()
        registry.configure(modelsDirectory: Self.fixturesURL)

        _ = try registry.model(named: "test_identity")
        _ = try registry.model(named: "test_double")
        XCTAssertTrue(registry.isLoaded(name: "test_identity"))
        XCTAssertTrue(registry.isLoaded(name: "test_double"))

        registry.unloadAll()
        XCTAssertFalse(registry.isLoaded(name: "test_identity"))
        XCTAssertFalse(registry.isLoaded(name: "test_double"))
    }

    func testUnloadNonexistentNameIsNoOp() {
        let registry = makeRegistry()
        registry.configure(modelsDirectory: Self.fixturesURL)
        // Should not crash or throw.
        registry.unload(name: "does_not_exist")
    }

    // MARK: - Re-load after unload

    func testReloadAfterUnload() throws {
        guard fixtureExists("test_identity.mlmodelc") else {
            throw XCTSkip("test_identity.mlmodelc not found")
        }

        let registry = makeRegistry()
        registry.configure(modelsDirectory: Self.fixturesURL)

        let first = try registry.model(named: "test_identity")
        registry.unload(name: "test_identity")
        let second = try registry.model(named: "test_identity")

        // Should be a different instance (reloaded from disk).
        XCTAssertFalse(first === second,
            "After unload, a fresh instance should be created")

        // But should still work.
        let output = try second.predict(data: [5, 6, 7, 8], shape: [1, 4])
        output.withUnsafeBufferPointer { buf in
            XCTAssertEqual(buf[0], 5.0, accuracy: 1e-4)
        }
    }

    // MARK: - Error equality

    func testModelRegistryErrorEquality() {
        XCTAssertEqual(ModelRegistryError.notConfigured, .notConfigured)
        XCTAssertEqual(ModelRegistryError.modelNotFound("a"), .modelNotFound("a"))
        XCTAssertNotEqual(ModelRegistryError.modelNotFound("a"), .modelNotFound("b"))
        XCTAssertNotEqual(ModelRegistryError.notConfigured, .modelNotFound("x"))
    }

    // MARK: - isLoaded without loading

    func testIsLoadedReturnsFalseForUnknownModel() {
        let registry = makeRegistry()
        registry.configure(modelsDirectory: Self.fixturesURL)
        XCTAssertFalse(registry.isLoaded(name: "never_loaded"))
    }

    // MARK: - Private helpers

    private func fixtureExists(_ name: String) -> Bool {
        var isDir: ObjCBool = false
        return FileManager.default.fileExists(
            atPath: Self.fixturesURL.appendingPathComponent(name).path,
            isDirectory: &isDir
        )
    }
}
