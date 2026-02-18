import XCTest
@testable import MetalMomCore

final class EnsembleRunnerTests: XCTestCase {

    // MARK: - Helpers

    /// Resolve path to the Tests/fixtures directory via #filePath.
    private static var fixturesURL: URL {
        let thisFile = URL(fileURLWithPath: #filePath)
        let testsDir = thisFile
            .deletingLastPathComponent()   // MetalMomTests/
            .deletingLastPathComponent()   // Tests/
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

    /// Helper to extract float data from a Signal.
    private func signalData(_ signal: Signal) -> [Float] {
        var data = [Float](repeating: 0, count: signal.count)
        signal.withUnsafeBufferPointer { src in
            data.withUnsafeMutableBufferPointer { dst in
                dst.baseAddress!.update(from: src.baseAddress!, count: signal.count)
            }
        }
        return data
    }

    // MARK: - Combine logic tests (no CoreML models needed)

    func testCombineMean() throws {
        let a = Signal(data: [1.0, 2.0, 3.0, 4.0], shape: [1, 4], sampleRate: 22050)
        let b = Signal(data: [3.0, 4.0, 5.0, 6.0], shape: [1, 4], sampleRate: 22050)
        let c = Signal(data: [2.0, 6.0, 1.0, 8.0], shape: [1, 4], sampleRate: 22050)

        let result = try EnsembleRunner.combine(outputs: [a, b, c], strategy: .mean)

        let expected: [Float] = [2.0, 4.0, 3.0, 6.0]
        let data = signalData(result)
        XCTAssertEqual(result.shape, [1, 4])
        XCTAssertEqual(result.sampleRate, 22050)
        for i in 0..<4 {
            XCTAssertEqual(data[i], expected[i], accuracy: 1e-5,
                "Mean at index \(i): expected \(expected[i]), got \(data[i])")
        }
    }

    func testCombineMedianOddCount() throws {
        // 3 models: median is the middle value for each element.
        let a = Signal(data: [1.0, 5.0, 2.0, 9.0], shape: [4], sampleRate: 44100)
        let b = Signal(data: [3.0, 3.0, 8.0, 1.0], shape: [4], sampleRate: 44100)
        let c = Signal(data: [5.0, 1.0, 5.0, 5.0], shape: [4], sampleRate: 44100)

        let result = try EnsembleRunner.combine(outputs: [a, b, c], strategy: .median)

        let expected: [Float] = [3.0, 3.0, 5.0, 5.0]
        let data = signalData(result)
        XCTAssertEqual(result.shape, [4])
        XCTAssertEqual(result.sampleRate, 44100)
        for i in 0..<4 {
            XCTAssertEqual(data[i], expected[i], accuracy: 1e-5,
                "Median at index \(i): expected \(expected[i]), got \(data[i])")
        }
    }

    func testCombineMedianEvenCount() throws {
        // 2 models: median is average of the two values.
        let a = Signal(data: [1.0, 4.0], shape: [2], sampleRate: 22050)
        let b = Signal(data: [3.0, 8.0], shape: [2], sampleRate: 22050)

        let result = try EnsembleRunner.combine(outputs: [a, b], strategy: .median)

        let expected: [Float] = [2.0, 6.0]
        let data = signalData(result)
        for i in 0..<2 {
            XCTAssertEqual(data[i], expected[i], accuracy: 1e-5,
                "Even median at index \(i): expected \(expected[i]), got \(data[i])")
        }
    }

    func testCombineMax() throws {
        let a = Signal(data: [1.0, 9.0, 3.0, 4.0], shape: [1, 4], sampleRate: 22050)
        let b = Signal(data: [5.0, 2.0, 7.0, 1.0], shape: [1, 4], sampleRate: 22050)
        let c = Signal(data: [3.0, 4.0, 5.0, 8.0], shape: [1, 4], sampleRate: 22050)

        let result = try EnsembleRunner.combine(outputs: [a, b, c], strategy: .max)

        let expected: [Float] = [5.0, 9.0, 7.0, 8.0]
        let data = signalData(result)
        XCTAssertEqual(result.shape, [1, 4])
        for i in 0..<4 {
            XCTAssertEqual(data[i], expected[i], accuracy: 1e-5,
                "Max at index \(i): expected \(expected[i]), got \(data[i])")
        }
    }

    func testCombineSingleOutput() throws {
        // With 1 model, the combined output should be identical.
        let a = Signal(data: [1.0, 2.0, 3.0], shape: [3], sampleRate: 16000)

        let result = try EnsembleRunner.combine(outputs: [a], strategy: .mean)

        let data = signalData(result)
        XCTAssertEqual(result.shape, [3])
        XCTAssertEqual(result.sampleRate, 16000)
        XCTAssertEqual(data, [1.0, 2.0, 3.0])
    }

    func testCombineShapeMismatchThrows() {
        let a = Signal(data: [1.0, 2.0], shape: [2], sampleRate: 22050)
        let b = Signal(data: [1.0, 2.0, 3.0], shape: [3], sampleRate: 22050)

        XCTAssertThrowsError(try EnsembleRunner.combine(outputs: [a, b], strategy: .mean)) { error in
            guard let ee = error as? EnsembleError,
                  case .shapeMismatch = ee else {
                XCTFail("Expected EnsembleError.shapeMismatch, got \(error)")
                return
            }
        }
    }

    func testCombineEmptyOutputsThrows() {
        XCTAssertThrowsError(try EnsembleRunner.combine(outputs: [], strategy: .mean)) { error in
            guard let ee = error as? EnsembleError,
                  case .emptyEnsemble = ee else {
                XCTFail("Expected EnsembleError.emptyEnsemble, got \(error)")
                return
            }
        }
    }

    func testShapePreservation2D() throws {
        let a = Signal(data: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape: [2, 3], sampleRate: 22050)
        let b = Signal(data: [6.0, 5.0, 4.0, 3.0, 2.0, 1.0], shape: [2, 3], sampleRate: 22050)

        let result = try EnsembleRunner.combine(outputs: [a, b], strategy: .mean)

        XCTAssertEqual(result.shape, [2, 3])
        XCTAssertEqual(result.count, 6)
        XCTAssertEqual(result.sampleRate, 22050)

        // Mean of [1,6]=3.5, [2,5]=3.5, [3,4]=3.5, [4,3]=3.5, [5,2]=3.5, [6,1]=3.5
        let data = signalData(result)
        for i in 0..<6 {
            XCTAssertEqual(data[i], 3.5, accuracy: 1e-5)
        }
    }

    // MARK: - Init validation tests

    func testInitWithEmptyEnginesThrows() {
        XCTAssertThrowsError(try EnsembleRunner(engines: [], strategy: .mean)) { error in
            guard let ee = error as? EnsembleError,
                  case .emptyEnsemble = ee else {
                XCTFail("Expected EnsembleError.emptyEnsemble, got \(error)")
                return
            }
        }
    }

    func testInitWithEmptyURLsThrows() {
        XCTAssertThrowsError(try EnsembleRunner(modelURLs: [], strategy: .mean)) { error in
            guard let ee = error as? EnsembleError,
                  case .emptyEnsemble = ee else {
                XCTFail("Expected EnsembleError.emptyEnsemble, got \(error)")
                return
            }
        }
    }

    // MARK: - CoreML integration tests (require test fixtures)

    func testEnsembleWithIdentityModels() throws {
        guard let url = Self.compiledModelURL(named: "test_identity") else {
            throw XCTSkip("test_identity.mlmodelc not found - run scripts/create_test_model.py first")
        }

        // Create ensemble of 3 copies of the identity model.
        let engine1 = try InferenceEngine(compiledModelURL: url)
        let engine2 = try InferenceEngine(compiledModelURL: url)
        let engine3 = try InferenceEngine(compiledModelURL: url)
        let ensemble = try EnsembleRunner(engines: [engine1, engine2, engine3], strategy: .mean)

        XCTAssertEqual(ensemble.count, 3)

        let inputData: [Float] = [1.0, 2.0, 3.0, 4.0]
        let input = Signal(data: inputData, shape: [1, 4], sampleRate: 22050)
        let output = try ensemble.predict(input: input)

        // Mean of 3 identical identity outputs should equal the input.
        XCTAssertEqual(output.shape, [1, 4])
        XCTAssertEqual(output.sampleRate, 22050)
        let data = signalData(output)
        for i in 0..<4 {
            XCTAssertEqual(data[i], inputData[i], accuracy: 1e-4,
                "Identity ensemble mean[\(i)] should match input")
        }
    }

    func testEnsembleWithIdentityAndDoubleModels() throws {
        guard let identityURL = Self.compiledModelURL(named: "test_identity"),
              let doubleURL = Self.compiledModelURL(named: "test_double") else {
            throw XCTSkip("test model fixtures not found - run scripts/create_test_model.py first")
        }

        // identity output = input, double output = 2*input
        // mean = (input + 2*input) / 2 = 1.5 * input
        let identityEngine = try InferenceEngine(compiledModelURL: identityURL)
        let doubleEngine = try InferenceEngine(compiledModelURL: doubleURL)
        let ensemble = try EnsembleRunner(engines: [identityEngine, doubleEngine], strategy: .mean)

        let inputData: [Float] = [2.0, 4.0, 6.0, 8.0]
        let input = Signal(data: inputData, shape: [1, 4], sampleRate: 22050)
        let output = try ensemble.predict(input: input)

        let expected: [Float] = [3.0, 6.0, 9.0, 12.0]  // 1.5 * input
        let data = signalData(output)
        for i in 0..<4 {
            XCTAssertEqual(data[i], expected[i], accuracy: 1e-4,
                "Identity+Double mean[\(i)]: expected \(expected[i]), got \(data[i])")
        }
    }

    func testEnsembleMaxWithIdentityAndDouble() throws {
        guard let identityURL = Self.compiledModelURL(named: "test_identity"),
              let doubleURL = Self.compiledModelURL(named: "test_double") else {
            throw XCTSkip("test model fixtures not found")
        }

        // max(input, 2*input) = 2*input for all positive inputs
        let identityEngine = try InferenceEngine(compiledModelURL: identityURL)
        let doubleEngine = try InferenceEngine(compiledModelURL: doubleURL)
        let ensemble = try EnsembleRunner(engines: [identityEngine, doubleEngine], strategy: .max)

        let inputData: [Float] = [1.0, 2.0, 3.0, 4.0]
        let input = Signal(data: inputData, shape: [1, 4], sampleRate: 22050)
        let output = try ensemble.predict(input: input)

        let expected: [Float] = [2.0, 4.0, 6.0, 8.0]  // max = double output
        let data = signalData(output)
        for i in 0..<4 {
            XCTAssertEqual(data[i], expected[i], accuracy: 1e-4,
                "Max[\(i)]: expected \(expected[i]), got \(data[i])")
        }
    }

    func testEnsembleMedianWithThreeIdentityModels() throws {
        guard let url = Self.compiledModelURL(named: "test_identity") else {
            throw XCTSkip("test_identity.mlmodelc not found")
        }

        // Median of 3 identical identity outputs = input.
        let engines = try (0..<3).map { _ in try InferenceEngine(compiledModelURL: url) }
        let ensemble = try EnsembleRunner(engines: engines, strategy: .median)

        let inputData: [Float] = [10.0, 20.0, 30.0, 40.0]
        let input = Signal(data: inputData, shape: [1, 4], sampleRate: 44100)
        let output = try ensemble.predict(input: input)

        XCTAssertEqual(output.shape, [1, 4])
        XCTAssertEqual(output.sampleRate, 44100)
        let data = signalData(output)
        for i in 0..<4 {
            XCTAssertEqual(data[i], inputData[i], accuracy: 1e-4)
        }
    }

    func testPredictWithDetails() throws {
        guard let identityURL = Self.compiledModelURL(named: "test_identity"),
              let doubleURL = Self.compiledModelURL(named: "test_double") else {
            throw XCTSkip("test model fixtures not found")
        }

        let identityEngine = try InferenceEngine(compiledModelURL: identityURL)
        let doubleEngine = try InferenceEngine(compiledModelURL: doubleURL)
        let ensemble = try EnsembleRunner(engines: [identityEngine, doubleEngine], strategy: .mean)

        let inputData: [Float] = [1.0, 2.0, 3.0, 4.0]
        let input = Signal(data: inputData, shape: [1, 4], sampleRate: 22050)
        let (combined, individual) = try ensemble.predictWithDetails(input: input)

        // Check individual outputs.
        XCTAssertEqual(individual.count, 2)

        // First is identity: output = input.
        let identityData = signalData(individual[0])
        for i in 0..<4 {
            XCTAssertEqual(identityData[i], inputData[i], accuracy: 1e-4)
        }

        // Second is double: output = 2 * input.
        let doubleData = signalData(individual[1])
        for i in 0..<4 {
            XCTAssertEqual(doubleData[i], inputData[i] * 2.0, accuracy: 1e-4)
        }

        // Combined is mean: (input + 2*input) / 2 = 1.5 * input.
        let combinedData = signalData(combined)
        for i in 0..<4 {
            XCTAssertEqual(combinedData[i], inputData[i] * 1.5, accuracy: 1e-4)
        }
    }

    func testCountProperty() throws {
        guard let url = Self.compiledModelURL(named: "test_identity") else {
            throw XCTSkip("test_identity.mlmodelc not found")
        }

        let engines = try (0..<5).map { _ in try InferenceEngine(compiledModelURL: url) }
        let ensemble = try EnsembleRunner(engines: engines)
        XCTAssertEqual(ensemble.count, 5)
    }

    func testInitFromURLs() throws {
        guard let url = Self.compiledModelURL(named: "test_identity") else {
            throw XCTSkip("test_identity.mlmodelc not found")
        }

        // Load 3 copies via URL-based initializer.
        let ensemble = try EnsembleRunner(
            modelURLs: [url, url, url],
            strategy: .mean,
            computeUnits: .cpuOnly
        )

        XCTAssertEqual(ensemble.count, 3)

        let input = Signal(data: [1.0, 2.0, 3.0, 4.0], shape: [1, 4], sampleRate: 22050)
        let output = try ensemble.predict(input: input)

        let data = signalData(output)
        for i in 0..<4 {
            XCTAssertEqual(data[i], Float(i + 1), accuracy: 1e-4)
        }
    }

    // MARK: - Mean correctness with many models

    func testMeanWithLargerEnsemble() throws {
        // Purely testing combine logic: 6 models with known outputs.
        let signals = (0..<6).map { k in
            Signal(data: [Float(k), Float(k * 2), Float(k * 3)], shape: [3], sampleRate: 22050)
        }
        // mean[i] = sum of k*i for k in 0..<6, divided by 6
        // i=0: (0+0+0+0+0+0)/6 = 0
        // i=1: (0+1+2+3+4+5)/6 = 15/6 = 2.5    (k*1)
        // i=2: (0+2+4+6+8+10)/6 = 30/6 = 5.0   (k*2)
        // i=3: (0+3+6+9+12+15)/6 = 45/6 = 7.5  (k*3)
        // Wait, shape is [3] so i goes 0..2, and data[j] = k * (j+1) for signal k
        // Actually data is [Float(k), Float(k*2), Float(k*3)]
        // So for element 0: (0+1+2+3+4+5)/6 = 2.5
        // For element 1: (0+2+4+6+8+10)/6 = 5.0
        // For element 2: (0+3+6+9+12+15)/6 = 7.5
        let result = try EnsembleRunner.combine(outputs: signals, strategy: .mean)
        let data = signalData(result)

        XCTAssertEqual(data[0], 2.5, accuracy: 1e-5)
        XCTAssertEqual(data[1], 5.0, accuracy: 1e-5)
        XCTAssertEqual(data[2], 7.5, accuracy: 1e-5)
    }
}
