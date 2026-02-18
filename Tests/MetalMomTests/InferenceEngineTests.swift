import XCTest
@testable import MetalMomCore

final class InferenceEngineTests: XCTestCase {

    // MARK: - Helpers

    /// Resolve path to the Tests/fixtures directory via #filePath.
    private static var fixturesURL: URL {
        // #filePath → .../Tests/MetalMomTests/InferenceEngineTests.swift
        let thisFile = URL(fileURLWithPath: #filePath)
        let testsDir = thisFile
            .deletingLastPathComponent()  // MetalMomTests/
            .deletingLastPathComponent()  // Tests/
        return testsDir.appendingPathComponent("fixtures")
    }

    /// URL to a compiled test model, or nil if fixtures haven't been generated.
    private static func compiledModelURL(named name: String) -> URL? {
        let url = fixturesURL.appendingPathComponent("\(name).mlmodelc")
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir),
              isDir.boolValue else {
            return nil
        }
        return url
    }

    /// URL to an uncompiled .mlmodel fixture, or nil if it doesn't exist.
    private static func sourceModelURL(named name: String) -> URL? {
        let url = fixturesURL.appendingPathComponent("\(name).mlmodel")
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        return url
    }

    // MARK: - Error handling tests

    func testInitWithInvalidCompiledURLThrows() {
        let bogus = URL(fileURLWithPath: "/tmp/nonexistent_model.mlmodelc")
        XCTAssertThrowsError(try InferenceEngine(compiledModelURL: bogus)) { error in
            // Any error is expected — just make sure it throws.
            XCTAssertNotNil(error)
        }
    }

    func testInitWithInvalidSourceURLThrows() {
        let bogus = URL(fileURLWithPath: "/tmp/nonexistent_model.mlmodel")
        XCTAssertThrowsError(try InferenceEngine(sourceModelURL: bogus)) { error in
            XCTAssertNotNil(error)
        }
    }

    func testInferenceErrorEquality() {
        XCTAssertEqual(InferenceError.missingOutput("x"), InferenceError.missingOutput("x"))
        XCTAssertNotEqual(InferenceError.missingOutput("x"), InferenceError.missingOutput("y"))
        XCTAssertNotEqual(
            InferenceError.missingOutput("x"),
            InferenceError.invalidModel("x")
        )
    }

    // MARK: - Identity model tests

    func testIdentityModelCompiledURL() throws {
        guard let url = Self.compiledModelURL(named: "test_identity") else {
            throw XCTSkip("test_identity.mlmodelc not found — run scripts/create_test_model.py first")
        }

        let engine = try InferenceEngine(compiledModelURL: url)

        // Verify introspection.
        XCTAssertTrue(engine.inputFeatureNames.contains("input"))
        XCTAssertTrue(engine.outputFeatureNames.contains("output"))

        // Run prediction.
        let inputData: [Float] = [1.0, 2.0, 3.0, 4.0]
        let input = Signal(data: inputData, shape: [1, 4], sampleRate: 22050)
        let output = try engine.predict(input: input)

        // Identity model: output should equal input.
        XCTAssertEqual(output.shape, [1, 4])
        XCTAssertEqual(output.count, 4)
        XCTAssertEqual(output.sampleRate, 22050)
        output.withUnsafeBufferPointer { buf in
            for i in 0..<4 {
                XCTAssertEqual(buf[i], inputData[i], accuracy: 1e-4,
                    "Identity output[\(i)] should match input")
            }
        }
    }

    func testIdentityModelFromSourceURL() throws {
        guard let url = Self.sourceModelURL(named: "test_identity") else {
            throw XCTSkip("test_identity.mlmodel not found — run scripts/create_test_model.py first")
        }

        let engine = try InferenceEngine(sourceModelURL: url)
        let inputData: [Float] = [5.0, 6.0, 7.0, 8.0]
        let output = try engine.predict(
            data: inputData,
            shape: [1, 4],
            sampleRate: 44100
        )

        XCTAssertEqual(output.sampleRate, 44100)
        output.withUnsafeBufferPointer { buf in
            for i in 0..<4 {
                XCTAssertEqual(buf[i], inputData[i], accuracy: 1e-4)
            }
        }
    }

    // MARK: - Double model tests

    func testDoubleModel() throws {
        guard let url = Self.compiledModelURL(named: "test_double") else {
            throw XCTSkip("test_double.mlmodelc not found — run scripts/create_test_model.py first")
        }

        let engine = try InferenceEngine(compiledModelURL: url)
        let inputData: [Float] = [1.0, 2.0, 3.0, 4.0]
        let output = try engine.predict(
            data: inputData,
            shape: [1, 4]
        )

        let expected: [Float] = [2.0, 4.0, 6.0, 8.0]
        output.withUnsafeBufferPointer { buf in
            for i in 0..<4 {
                XCTAssertEqual(buf[i], expected[i], accuracy: 1e-4,
                    "Double output[\(i)] should be 2 * input")
            }
        }
    }

    // MARK: - Missing output name

    func testPredictWithWrongOutputNameThrows() throws {
        guard let url = Self.compiledModelURL(named: "test_identity") else {
            throw XCTSkip("test_identity.mlmodelc not found")
        }

        let engine = try InferenceEngine(compiledModelURL: url)
        let input = Signal(data: [1, 2, 3, 4], shape: [1, 4], sampleRate: 22050)

        XCTAssertThrowsError(
            try engine.predict(input: input, outputName: "nonexistent")
        ) { error in
            guard let ie = error as? InferenceError,
                  case .missingOutput(let name) = ie else {
                XCTFail("Expected InferenceError.missingOutput, got \(error)")
                return
            }
            XCTAssertEqual(name, "nonexistent")
        }
    }

    // MARK: - predictAll (multi-output)

    func testPredictAllReturnsAllOutputs() throws {
        guard let url = Self.compiledModelURL(named: "test_identity") else {
            throw XCTSkip("test_identity.mlmodelc not found")
        }

        let engine = try InferenceEngine(compiledModelURL: url)
        let input = Signal(data: [1, 2, 3, 4], shape: [1, 4], sampleRate: 22050)
        let outputs = try engine.predictAll(input: input)

        // Identity model has one output named "output".
        XCTAssertTrue(outputs.keys.contains("output"))
        let out = outputs["output"]!
        out.withUnsafeBufferPointer { buf in
            for i in 0..<4 {
                XCTAssertEqual(buf[i], Float(i + 1), accuracy: 1e-4)
            }
        }
    }

    // MARK: - Compute units

    func testCPUOnlyComputeUnits() throws {
        guard let url = Self.compiledModelURL(named: "test_identity") else {
            throw XCTSkip("test_identity.mlmodelc not found")
        }

        // Should load fine restricted to CPU only.
        let engine = try InferenceEngine(compiledModelURL: url, computeUnits: .cpuOnly)
        let output = try engine.predict(
            data: [10, 20, 30, 40],
            shape: [1, 4]
        )
        output.withUnsafeBufferPointer { buf in
            XCTAssertEqual(buf[0], 10.0, accuracy: 1e-4)
        }
    }

    // MARK: - Sample rate passthrough

    func testSampleRatePassthrough() throws {
        guard let url = Self.compiledModelURL(named: "test_identity") else {
            throw XCTSkip("test_identity.mlmodelc not found")
        }

        let engine = try InferenceEngine(compiledModelURL: url)
        let output = try engine.predict(
            data: [1, 2, 3, 4],
            shape: [1, 4],
            sampleRate: 48000
        )
        XCTAssertEqual(output.sampleRate, 48000,
            "Output sample rate should match input sample rate")
    }
}
