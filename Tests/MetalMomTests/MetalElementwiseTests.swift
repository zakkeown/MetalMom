import XCTest
import Foundation
@testable import MetalMomCore

final class MetalElementwiseTests: XCTestCase {

    // MARK: - Helper

    /// Get a MetalShaders instance or skip the test.
    private func getShaders() throws -> MetalShaders {
        guard let backend = MetalBackend.shared else {
            throw XCTSkip("No Metal device available")
        }
        guard let shaders = backend.shaders else {
            throw XCTSkip("Metal shader compilation failed")
        }
        return shaders
    }

    // MARK: - Elementwise Log

    func testElementwiseLog() throws {
        let shaders = try getShaders()
        let input: [Float] = [1.0, 2.718281828, 10.0, 100.0, 0.5]
        guard let result = shaders.log(input) else {
            XCTFail("Metal log failed")
            return
        }
        let expected: [Float] = [0.0, 1.0, 2.302585, 4.60517, -0.693147]
        XCTAssertEqual(result.count, expected.count)
        for (r, e) in zip(result, expected) {
            XCTAssertEqual(r, e, accuracy: 1e-4, "log(\(input[result.firstIndex(of: r) ?? 0]))")
        }
    }

    // MARK: - Elementwise Exp

    func testElementwiseExp() throws {
        let shaders = try getShaders()
        let input: [Float] = [0.0, 1.0, 2.0, -1.0, 0.5]
        guard let result = shaders.exp(input) else {
            XCTFail("Metal exp failed")
            return
        }
        let expected: [Float] = [1.0, 2.718281828, 7.389056, 0.367879, 1.648721]
        XCTAssertEqual(result.count, expected.count)
        for (r, e) in zip(result, expected) {
            XCTAssertEqual(r, e, accuracy: 1e-4)
        }
    }

    // MARK: - Elementwise Pow

    func testElementwisePow() throws {
        let shaders = try getShaders()
        let input: [Float] = [1.0, 2.0, 3.0, 4.0, 9.0]
        guard let result = shaders.pow(input, exponent: 2.0) else {
            XCTFail("Metal pow failed")
            return
        }
        let expected: [Float] = [1.0, 4.0, 9.0, 16.0, 81.0]
        XCTAssertEqual(result.count, expected.count)
        for (r, e) in zip(result, expected) {
            XCTAssertEqual(r, e, accuracy: 1e-4)
        }

        // Also test fractional exponent (square root)
        guard let sqrtResult = shaders.pow(input, exponent: 0.5) else {
            XCTFail("Metal pow(0.5) failed")
            return
        }
        let expectedSqrt: [Float] = [1.0, 1.414214, 1.732051, 2.0, 3.0]
        for (r, e) in zip(sqrtResult, expectedSqrt) {
            XCTAssertEqual(r, e, accuracy: 1e-4)
        }
    }

    // MARK: - Elementwise Abs

    func testElementwiseAbs() throws {
        let shaders = try getShaders()
        let input: [Float] = [-3.0, -1.5, 0.0, 1.5, 3.0]
        guard let result = shaders.abs(input) else {
            XCTFail("Metal abs failed")
            return
        }
        let expected: [Float] = [3.0, 1.5, 0.0, 1.5, 3.0]
        XCTAssertEqual(result.count, expected.count)
        for (r, e) in zip(result, expected) {
            XCTAssertEqual(r, e, accuracy: 1e-6)
        }
    }

    // MARK: - Amplitude to dB

    func testAmplitudeToDb() throws {
        let shaders = try getShaders()
        // amplitude_to_db: 20 * log10(max(x, amin))
        let input: [Float] = [1.0, 10.0, 100.0, 0.1, 0.01]
        guard let result = shaders.amplitudeToDb(input) else {
            XCTFail("Metal amplitude_to_db failed")
            return
        }
        // Expected: 20*log10(1)=0, 20*log10(10)=20, 20*log10(100)=40,
        //           20*log10(0.1)=-20, 20*log10(0.01)=-40
        let expected: [Float] = [0.0, 20.0, 40.0, -20.0, -40.0]
        XCTAssertEqual(result.count, expected.count)
        for (r, e) in zip(result, expected) {
            XCTAssertEqual(r, e, accuracy: 1e-3)
        }
    }

    func testAmplitudeToDbWithAmin() throws {
        let shaders = try getShaders()
        // Values below amin should be clamped
        let input: [Float] = [1e-12, 1e-15, 1.0]
        let amin: Float = 1e-10
        guard let result = shaders.amplitudeToDb(input, amin: amin) else {
            XCTFail("Metal amplitude_to_db with amin failed")
            return
        }
        // First two should both return 20*log10(1e-10) = -200
        let aminDb: Float = 20.0 * Foundation.log10(amin)
        XCTAssertEqual(result[0], aminDb, accuracy: 1e-2)
        XCTAssertEqual(result[1], aminDb, accuracy: 1e-2)
        XCTAssertEqual(result[2], 0.0, accuracy: 1e-3)
    }

    // MARK: - Power to dB

    func testPowerToDb() throws {
        let shaders = try getShaders()
        // power_to_db: 10 * log10(max(x, amin))
        let input: [Float] = [1.0, 10.0, 100.0, 0.1, 0.01]
        guard let result = shaders.powerToDb(input) else {
            XCTFail("Metal power_to_db failed")
            return
        }
        // Expected: 10*log10(1)=0, 10*log10(10)=10, 10*log10(100)=20,
        //           10*log10(0.1)=-10, 10*log10(0.01)=-20
        let expected: [Float] = [0.0, 10.0, 20.0, -10.0, -20.0]
        XCTAssertEqual(result.count, expected.count)
        for (r, e) in zip(result, expected) {
            XCTAssertEqual(r, e, accuracy: 1e-3)
        }
    }

    // MARK: - Log1p

    func testLog1p() throws {
        let shaders = try getShaders()
        // log1p: log(1 + x)
        let input: [Float] = [0.0, 1.0, 9.0, 99.0, 0.001]
        guard let result = shaders.log1p(input) else {
            XCTFail("Metal log1p failed")
            return
        }
        // log(1+0)=0, log(2)=0.693147, log(10)=2.302585, log(100)=4.60517, log(1.001)=0.000999
        let expected: [Float] = [0.0, 0.693147, 2.302585, 4.60517, 0.0009995]
        XCTAssertEqual(result.count, expected.count)
        for (r, e) in zip(result, expected) {
            XCTAssertEqual(r, e, accuracy: 1e-4)
        }
    }

    // MARK: - Scale + Bias

    func testScaleBias() throws {
        let shaders = try getShaders()
        let input: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0]
        guard let result = shaders.scaleBias(input, scale: 2.0, bias: -1.0) else {
            XCTFail("Metal scale_bias failed")
            return
        }
        // 1*2-1=1, 2*2-1=3, 3*2-1=5, 4*2-1=7, 5*2-1=9
        let expected: [Float] = [1.0, 3.0, 5.0, 7.0, 9.0]
        XCTAssertEqual(result.count, expected.count)
        for (r, e) in zip(result, expected) {
            XCTAssertEqual(r, e, accuracy: 1e-6)
        }
    }

    func testScaleBiasIdentity() throws {
        let shaders = try getShaders()
        let input: [Float] = [1.0, -2.0, 3.5, 0.0, 100.0]
        guard let result = shaders.scaleBias(input, scale: 1.0, bias: 0.0) else {
            XCTFail("Metal scale_bias identity failed")
            return
        }
        for (r, e) in zip(result, input) {
            XCTAssertEqual(r, e, accuracy: 1e-6)
        }
    }

    // MARK: - Large Array

    func testLargeArray() throws {
        let shaders = try getShaders()

        // 1M elements — verify GPU handles large data correctly
        let count = 1_000_000
        var input = [Float](repeating: 0, count: count)
        for i in 0..<count {
            input[i] = Float(i + 1)  // 1, 2, 3, ..., 1M
        }

        // Test with log
        guard let result = shaders.log(input) else {
            XCTFail("Metal log failed for large array")
            return
        }
        XCTAssertEqual(result.count, count)

        // Check first, middle, and last elements
        XCTAssertEqual(result[0], Foundation.log(1.0), accuracy: 1e-4)
        XCTAssertEqual(result[count / 2], Foundation.log(Float(count / 2 + 1)), accuracy: 1e-2)
        XCTAssertEqual(result[count - 1], Foundation.log(Float(count)), accuracy: 1e-1)
    }

    // MARK: - Pipeline Caching

    func testPipelineCaching() throws {
        let shaders = try getShaders()
        // Call the same kernel twice — second call should use cached pipeline
        let input: [Float] = [1.0, 2.0, 3.0]
        _ = shaders.log(input)
        let result = shaders.log(input)
        XCTAssertNotNil(result)
        XCTAssertEqual(result?.count, 3)
    }

    // MARK: - Shader Initialization via MetalBackend

    func testMetalBackendShadersProperty() throws {
        guard let backend = MetalBackend.shared else {
            throw XCTSkip("No Metal device available")
        }
        // The shaders property should be non-nil when Metal is available
        XCTAssertNotNil(backend.shaders)
    }

    // MARK: - Edge Cases

    func testEmptyArray() throws {
        let shaders = try getShaders()
        // Empty array should return empty result (count=0 dispatch is a no-op)
        let input: [Float] = []
        let result = shaders.log(input)
        // With count=0, the dispatch issues zero threads; the output buffer
        // has zero length, so we get an empty array back.
        XCTAssertNotNil(result)
        XCTAssertEqual(result?.count, 0)
    }

    func testSingleElement() throws {
        let shaders = try getShaders()
        let input: [Float] = [2.718281828]
        guard let result = shaders.log(input) else {
            XCTFail("Metal log failed for single element")
            return
        }
        XCTAssertEqual(result.count, 1)
        XCTAssertEqual(result[0], 1.0, accuracy: 1e-4)
    }
}
