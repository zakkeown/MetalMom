import XCTest
import Foundation
@testable import MetalMomCore

final class MetalReductionTests: XCTestCase {

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

    // MARK: - Sum

    func testSum() throws {
        let shaders = try getShaders()
        let input: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0]
        guard let result = shaders.sum(input) else {
            XCTFail("Metal sum failed")
            return
        }
        // Expected: 1+2+3+4+5 = 15
        XCTAssertEqual(result, 15.0, accuracy: 1e-4)
    }

    func testSumLargeArray() throws {
        let shaders = try getShaders()
        // 1M elements: values 1.0, 2.0, ..., 1000000.0
        let count = 1_000_000
        var input = [Float](repeating: 0, count: count)
        for i in 0..<count {
            input[i] = Float(i + 1)
        }

        guard let gpuResult = shaders.sum(input) else {
            XCTFail("Metal sum failed for large array")
            return
        }

        // CPU reference: n*(n+1)/2
        let cpuResult: Float = Float(count) * Float(count + 1) / 2.0

        // Float32 accumulation loses precision for large sums, so use relative tolerance.
        let relError = Swift.abs(gpuResult - cpuResult) / cpuResult
        XCTAssertLessThan(relError, 0.01, "GPU sum \(gpuResult) vs CPU sum \(cpuResult), relative error \(relError)")
    }

    // MARK: - Max

    func testMax() throws {
        let shaders = try getShaders()
        let input: [Float] = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]
        guard let result = shaders.max(input) else {
            XCTFail("Metal max failed")
            return
        }
        XCTAssertEqual(result, 9.0, accuracy: 1e-6)
    }

    // MARK: - Min

    func testMin() throws {
        let shaders = try getShaders()
        let input: [Float] = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]
        guard let result = shaders.min(input) else {
            XCTFail("Metal min failed")
            return
        }
        XCTAssertEqual(result, 1.0, accuracy: 1e-6)
    }

    // MARK: - Mean

    func testMean() throws {
        let shaders = try getShaders()
        let input: [Float] = [2.0, 4.0, 6.0, 8.0, 10.0]
        guard let result = shaders.mean(input) else {
            XCTFail("Metal mean failed")
            return
        }
        // Expected: (2+4+6+8+10) / 5 = 6.0
        XCTAssertEqual(result, 6.0, accuracy: 1e-4)
    }

    // MARK: - Argmax

    func testArgmax() throws {
        let shaders = try getShaders()
        let input: [Float] = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]
        guard let result = shaders.argmax(input) else {
            XCTFail("Metal argmax failed")
            return
        }
        // Maximum value 9.0 is at index 5
        XCTAssertEqual(result, 5)
    }

    // MARK: - Edge Cases

    func testSingleElement() throws {
        let shaders = try getShaders()
        let input: [Float] = [42.0]

        guard let sumResult = shaders.sum(input) else {
            XCTFail("Metal sum failed for single element")
            return
        }
        XCTAssertEqual(sumResult, 42.0, accuracy: 1e-6)

        guard let maxResult = shaders.max(input) else {
            XCTFail("Metal max failed for single element")
            return
        }
        XCTAssertEqual(maxResult, 42.0, accuracy: 1e-6)

        guard let minResult = shaders.min(input) else {
            XCTFail("Metal min failed for single element")
            return
        }
        XCTAssertEqual(minResult, 42.0, accuracy: 1e-6)

        guard let meanResult = shaders.mean(input) else {
            XCTFail("Metal mean failed for single element")
            return
        }
        XCTAssertEqual(meanResult, 42.0, accuracy: 1e-6)

        guard let argmaxResult = shaders.argmax(input) else {
            XCTFail("Metal argmax failed for single element")
            return
        }
        XCTAssertEqual(argmaxResult, 0)
    }

    func testAllSameValues() throws {
        let shaders = try getShaders()
        let input: [Float] = [7.0, 7.0, 7.0, 7.0, 7.0]

        guard let sumResult = shaders.sum(input) else {
            XCTFail("Metal sum failed")
            return
        }
        XCTAssertEqual(sumResult, 35.0, accuracy: 1e-4)

        guard let maxResult = shaders.max(input) else {
            XCTFail("Metal max failed")
            return
        }
        XCTAssertEqual(maxResult, 7.0, accuracy: 1e-6)

        guard let minResult = shaders.min(input) else {
            XCTFail("Metal min failed")
            return
        }
        XCTAssertEqual(minResult, 7.0, accuracy: 1e-6)

        guard let meanResult = shaders.mean(input) else {
            XCTFail("Metal mean failed")
            return
        }
        XCTAssertEqual(meanResult, 7.0, accuracy: 1e-4)

        // Argmax: all same, so the first occurrence (index 0) is acceptable.
        guard let argmaxResult = shaders.argmax(input) else {
            XCTFail("Metal argmax failed")
            return
        }
        // When all values are equal, any valid index [0, count) is correct.
        XCTAssertTrue(argmaxResult >= 0 && argmaxResult < input.count,
                       "argmax index \(argmaxResult) out of range")
    }

    func testNegativeValues() throws {
        let shaders = try getShaders()
        let input: [Float] = [-5.0, -3.0, -8.0, -1.0, -4.0]

        guard let sumResult = shaders.sum(input) else {
            XCTFail("Metal sum failed")
            return
        }
        // -5 + -3 + -8 + -1 + -4 = -21
        XCTAssertEqual(sumResult, -21.0, accuracy: 1e-4)

        guard let maxResult = shaders.max(input) else {
            XCTFail("Metal max failed")
            return
        }
        XCTAssertEqual(maxResult, -1.0, accuracy: 1e-6)

        guard let minResult = shaders.min(input) else {
            XCTFail("Metal min failed")
            return
        }
        XCTAssertEqual(minResult, -8.0, accuracy: 1e-6)

        guard let meanResult = shaders.mean(input) else {
            XCTFail("Metal mean failed")
            return
        }
        XCTAssertEqual(meanResult, -4.2, accuracy: 1e-4)

        guard let argmaxResult = shaders.argmax(input) else {
            XCTFail("Metal argmax failed")
            return
        }
        // Maximum value -1.0 is at index 3
        XCTAssertEqual(argmaxResult, 3)
    }

    func testEmptyArray() throws {
        let shaders = try getShaders()
        let input: [Float] = []

        // sum of empty array returns 0
        let sumResult = shaders.sum(input)
        XCTAssertNotNil(sumResult)
        XCTAssertEqual(sumResult, 0)

        // max, min, mean, argmax of empty array return nil
        XCTAssertNil(shaders.max(input))
        XCTAssertNil(shaders.min(input))
        XCTAssertNil(shaders.mean(input))
        XCTAssertNil(shaders.argmax(input))
    }
}
