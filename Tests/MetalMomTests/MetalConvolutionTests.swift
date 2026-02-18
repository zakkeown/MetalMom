import XCTest
import Foundation
@testable import MetalMomCore

final class MetalConvolutionTests: XCTestCase {

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

    /// CPU reference: 1D cross-correlation (valid mode).
    private func cpuConv1d(_ input: [Float], kernel: [Float]) -> [Float] {
        let outputLen = input.count - kernel.count + 1
        guard outputLen > 0 else { return [] }
        var output = [Float](repeating: 0, count: outputLen)
        for i in 0..<outputLen {
            var sum: Float = 0
            for k in 0..<kernel.count {
                sum += input[i + k] * kernel[k]
            }
            output[i] = sum
        }
        return output
    }

    /// CPU reference: 1D cross-correlation (same mode).
    private func cpuConv1dSame(_ input: [Float], kernel: [Float]) -> [Float] {
        let padLeft = kernel.count / 2
        var output = [Float](repeating: 0, count: input.count)
        for i in 0..<input.count {
            var sum: Float = 0
            for k in 0..<kernel.count {
                let idx = i - padLeft + k
                if idx >= 0 && idx < input.count {
                    sum += input[idx] * kernel[k]
                }
            }
            output[i] = sum
        }
        return output
    }

    /// CPU reference: 2D cross-correlation (valid mode).
    private func cpuConv2d(
        _ input: [Float], inputH: Int, inputW: Int,
        kernel: [Float], kernelH: Int, kernelW: Int
    ) -> [Float] {
        let outH = inputH - kernelH + 1
        let outW = inputW - kernelW + 1
        guard outH > 0, outW > 0 else { return [] }
        var output = [Float](repeating: 0, count: outH * outW)
        for y in 0..<outH {
            for x in 0..<outW {
                var sum: Float = 0
                for ky in 0..<kernelH {
                    for kx in 0..<kernelW {
                        sum += input[(y + ky) * inputW + (x + kx)] * kernel[ky * kernelW + kx]
                    }
                }
                output[y * outW + x] = sum
            }
        }
        return output
    }

    // MARK: - Conv1D Basic

    /// [1,2,3,4,5] * [1,1] = [3,5,7,9]
    func testConv1dBasic() throws {
        let shaders = try getShaders()
        let input: [Float] = [1, 2, 3, 4, 5]
        let kernel: [Float] = [1, 1]
        guard let result = shaders.conv1d(input, kernel: kernel) else {
            XCTFail("Metal conv1d failed")
            return
        }
        let expected: [Float] = [3, 5, 7, 9]
        XCTAssertEqual(result.count, expected.count)
        for i in 0..<expected.count {
            XCTAssertEqual(result[i], expected[i], accuracy: 1e-6,
                           "Mismatch at index \(i): \(result[i]) vs \(expected[i])")
        }
    }

    /// Convolve with [1] = identity.
    func testConv1dIdentity() throws {
        let shaders = try getShaders()
        let input: [Float] = [10, 20, 30, 40, 50]
        let kernel: [Float] = [1]
        guard let result = shaders.conv1d(input, kernel: kernel) else {
            XCTFail("Metal conv1d identity failed")
            return
        }
        XCTAssertEqual(result.count, input.count)
        for i in 0..<input.count {
            XCTAssertEqual(result[i], input[i], accuracy: 1e-6)
        }
    }

    // MARK: - Conv1D Same

    /// Same padding preserves length.
    func testConv1dSame() throws {
        let shaders = try getShaders()
        let input: [Float] = [1, 2, 3, 4, 5, 6, 7, 8]
        let kernel: [Float] = [0.25, 0.5, 0.25]
        guard let result = shaders.conv1dSame(input, kernel: kernel) else {
            XCTFail("Metal conv1d_same failed")
            return
        }
        // Output length must equal input length.
        XCTAssertEqual(result.count, input.count)

        // Compare against CPU reference.
        let expected = cpuConv1dSame(input, kernel: kernel)
        for i in 0..<expected.count {
            XCTAssertEqual(result[i], expected[i], accuracy: 1e-5,
                           "Mismatch at index \(i): \(result[i]) vs \(expected[i])")
        }
    }

    /// Center portion of "same" output matches the "valid" output.
    func testConv1dSameMatchesValid() throws {
        let shaders = try getShaders()
        let input: [Float] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        let kernel: [Float] = [1, 2, 1]

        guard let sameResult = shaders.conv1dSame(input, kernel: kernel) else {
            XCTFail("Metal conv1d_same failed")
            return
        }
        guard let validResult = shaders.conv1d(input, kernel: kernel) else {
            XCTFail("Metal conv1d failed")
            return
        }

        // Valid output length = 10 - 3 + 1 = 8
        XCTAssertEqual(validResult.count, 8)
        XCTAssertEqual(sameResult.count, 10)

        // The "same" output center (indices 1..8) should match "valid" output.
        let padLeft = kernel.count / 2  // 1
        for i in 0..<validResult.count {
            XCTAssertEqual(sameResult[i + padLeft], validResult[i], accuracy: 1e-5,
                           "Mismatch at valid index \(i): same[\(i + padLeft)]=\(sameResult[i + padLeft]) vs valid[\(i)]=\(validResult[i])")
        }
    }

    // MARK: - Conv2D Basic

    /// Small 3x3 image with 2x2 kernel.
    func testConv2dBasic() throws {
        let shaders = try getShaders()
        // 3x3 input:
        // 1 2 3
        // 4 5 6
        // 7 8 9
        let input: [Float] = [1, 2, 3, 4, 5, 6, 7, 8, 9]

        // 2x2 kernel:
        // 1 0
        // 0 1
        let kernel: [Float] = [1, 0, 0, 1]

        guard let result = shaders.conv2d(input, inputH: 3, inputW: 3,
                                          kernel: kernel, kernelH: 2, kernelW: 2) else {
            XCTFail("Metal conv2d failed")
            return
        }

        // Expected output (2x2):
        // (1*1+2*0+4*0+5*1)=6   (2*1+3*0+5*0+6*1)=8
        // (4*1+5*0+7*0+8*1)=12  (5*1+6*0+8*0+9*1)=14
        let expected: [Float] = [6, 8, 12, 14]
        XCTAssertEqual(result.count, expected.count)
        for i in 0..<expected.count {
            XCTAssertEqual(result[i], expected[i], accuracy: 1e-6,
                           "Mismatch at index \(i): \(result[i]) vs \(expected[i])")
        }
    }

    /// 2D convolve with [[1]] = identity.
    func testConv2dIdentity() throws {
        let shaders = try getShaders()
        let input: [Float] = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        let kernel: [Float] = [1]

        guard let result = shaders.conv2d(input, inputH: 3, inputW: 3,
                                          kernel: kernel, kernelH: 1, kernelW: 1) else {
            XCTFail("Metal conv2d identity failed")
            return
        }
        XCTAssertEqual(result.count, input.count)
        for i in 0..<input.count {
            XCTAssertEqual(result[i], input[i], accuracy: 1e-6)
        }
    }

    // MARK: - Large Arrays

    /// 100K elements, compare GPU vs CPU reference.
    func testConv1dLargeArray() throws {
        let shaders = try getShaders()
        let count = 100_000
        var input = [Float](repeating: 0, count: count)
        for i in 0..<count {
            input[i] = Float(i % 100) * 0.01
        }
        let kernel: [Float] = [0.2, 0.3, 0.3, 0.2]

        guard let gpuResult = shaders.conv1d(input, kernel: kernel) else {
            XCTFail("Metal conv1d failed for large array")
            return
        }

        let cpuResult = cpuConv1d(input, kernel: kernel)
        XCTAssertEqual(gpuResult.count, cpuResult.count)

        // Spot-check first, middle, and last elements.
        let checkIndices = [0, cpuResult.count / 4, cpuResult.count / 2, cpuResult.count * 3 / 4, cpuResult.count - 1]
        for idx in checkIndices {
            XCTAssertEqual(gpuResult[idx], cpuResult[idx], accuracy: 1e-4,
                           "Mismatch at index \(idx): GPU=\(gpuResult[idx]) vs CPU=\(cpuResult[idx])")
        }
    }

    /// 256x256 image with 5x5 kernel.
    func testConv2dLargeImage() throws {
        let shaders = try getShaders()
        let h = 256
        let w = 256
        var input = [Float](repeating: 0, count: h * w)
        for i in 0..<input.count {
            input[i] = Float(i % 256) / 255.0
        }

        let kH = 5
        let kW = 5
        var kernel = [Float](repeating: 0, count: kH * kW)
        // Simple box filter
        let val: Float = 1.0 / Float(kH * kW)
        for i in 0..<kernel.count { kernel[i] = val }

        guard let gpuResult = shaders.conv2d(input, inputH: h, inputW: w,
                                             kernel: kernel, kernelH: kH, kernelW: kW) else {
            XCTFail("Metal conv2d failed for large image")
            return
        }

        let outH = h - kH + 1
        let outW = w - kW + 1
        XCTAssertEqual(gpuResult.count, outH * outW)

        // Compare a few elements against CPU reference.
        let cpuResult = cpuConv2d(input, inputH: h, inputW: w,
                                  kernel: kernel, kernelH: kH, kernelW: kW)
        let checkPositions = [(0, 0), (outH / 2, outW / 2), (outH - 1, outW - 1)]
        for (cy, cx) in checkPositions {
            let idx = cy * outW + cx
            XCTAssertEqual(gpuResult[idx], cpuResult[idx], accuracy: 1e-4,
                           "Mismatch at (\(cy),\(cx)): GPU=\(gpuResult[idx]) vs CPU=\(cpuResult[idx])")
        }
    }

    // MARK: - GPU vs CPU Parity

    /// Compare conv1d GPU result against CPU reference for parity.
    func testConv1dGPUvsCPU() throws {
        let shaders = try getShaders()
        let input: [Float] = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
        let kernel: [Float] = [0.5, -1.0, 0.5]

        guard let gpuResult = shaders.conv1d(input, kernel: kernel) else {
            XCTFail("Metal conv1d failed")
            return
        }

        let cpuResult = cpuConv1d(input, kernel: kernel)
        XCTAssertEqual(gpuResult.count, cpuResult.count)

        for i in 0..<cpuResult.count {
            XCTAssertEqual(gpuResult[i], cpuResult[i], accuracy: 1e-5,
                           "Mismatch at index \(i): GPU=\(gpuResult[i]) vs CPU=\(cpuResult[i])")
        }
    }

    // MARK: - Edge Cases

    /// Empty input returns nil.
    func testEmptyInput() throws {
        let shaders = try getShaders()

        // Conv1D with empty input
        XCTAssertNil(shaders.conv1d([], kernel: [1, 2, 3]))
        // Conv1D with empty kernel
        XCTAssertNil(shaders.conv1d([1, 2, 3], kernel: []))
        // Conv1D with kernel longer than input
        XCTAssertNil(shaders.conv1d([1, 2], kernel: [1, 2, 3]))

        // Conv1D same with empty input
        XCTAssertNil(shaders.conv1dSame([], kernel: [1]))
        // Conv1D same with empty kernel
        XCTAssertNil(shaders.conv1dSame([1], kernel: []))

        // Conv2D with kernel larger than input
        XCTAssertNil(shaders.conv2d([1, 2, 3, 4], inputH: 2, inputW: 2,
                                    kernel: [1, 2, 3, 4, 5, 6, 7, 8, 9], kernelH: 3, kernelW: 3))
    }
}
