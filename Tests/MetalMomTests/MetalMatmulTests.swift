import XCTest
import Accelerate
@testable import MetalMomCore

final class MetalMatmulTests: XCTestCase {

    // MARK: - Basic Matrix Multiplication

    func testBasicMatmul() throws {
        guard MetalBackend.shared != nil else { throw XCTSkip("Metal not available") }

        // 2x3 @ 3x2 = 2x2
        let a: [Float] = [1, 2, 3, 4, 5, 6]
        let b: [Float] = [7, 8, 9, 10, 11, 12]

        guard let c = MetalMatmul.multiply(a: a, aRows: 2, aCols: 3,
                                           b: b, bRows: 3, bCols: 2) else {
            XCTFail("Metal matmul returned nil")
            return
        }

        // Expected: [[1*7+2*9+3*11, 1*8+2*10+3*12],
        //            [4*7+5*9+6*11, 4*8+5*10+6*12]]
        //         = [[58, 64], [139, 154]]
        XCTAssertEqual(c.count, 4)
        XCTAssertEqual(c[0], 58,  accuracy: 1e-4)
        XCTAssertEqual(c[1], 64,  accuracy: 1e-4)
        XCTAssertEqual(c[2], 139, accuracy: 1e-4)
        XCTAssertEqual(c[3], 154, accuracy: 1e-4)
    }

    // MARK: - Identity Matrix

    func testIdentityMatrix() throws {
        guard MetalBackend.shared != nil else { throw XCTSkip("Metal not available") }

        // A @ I = A
        let n = 4
        let a: [Float] = [1, 2, 3, 4,
                          5, 6, 7, 8,
                          9, 10, 11, 12,
                          13, 14, 15, 16]
        // Build identity
        var identity = [Float](repeating: 0, count: n * n)
        for i in 0..<n { identity[i * n + i] = 1.0 }

        guard let c = MetalMatmul.multiply(a: a, aRows: n, aCols: n,
                                           b: identity, bRows: n, bCols: n) else {
            XCTFail("Metal matmul returned nil")
            return
        }

        XCTAssertEqual(c.count, n * n)
        for i in 0..<c.count {
            XCTAssertEqual(c[i], a[i], accuracy: 1e-5, "Mismatch at index \(i)")
        }
    }

    // MARK: - Single Row/Column

    func testSingleRowColumn() throws {
        guard MetalBackend.shared != nil else { throw XCTSkip("Metal not available") }

        // Row vector [1, 2, 3] @ column vector [[4], [5], [6]] = [[32]]
        let a: [Float] = [1, 2, 3]
        let b: [Float] = [4, 5, 6]

        guard let c = MetalMatmul.multiply(a: a, aRows: 1, aCols: 3,
                                           b: b, bRows: 3, bCols: 1) else {
            XCTFail("Metal matmul returned nil")
            return
        }

        // 1*4 + 2*5 + 3*6 = 32
        XCTAssertEqual(c.count, 1)
        XCTAssertEqual(c[0], 32, accuracy: 1e-4)
    }

    // MARK: - Large Matmul (GPU vs CPU)

    func testLargeMatmul() throws {
        guard MetalBackend.shared != nil else { throw XCTSkip("Metal not available") }

        // 256x1024 @ 1024x512 = 256x512
        let M = 256, K = 1024, N = 512

        // Deterministic pseudo-random data
        var a = [Float](repeating: 0, count: M * K)
        var b = [Float](repeating: 0, count: K * N)
        for i in 0..<a.count { a[i] = Float((i * 7 + 3) % 1000) / 1000.0 }
        for i in 0..<b.count { b[i] = Float((i * 13 + 5) % 1000) / 1000.0 }

        // GPU
        guard let gpuResult = MetalMatmul.multiply(a: a, aRows: M, aCols: K,
                                                   b: b, bRows: K, bCols: N) else {
            XCTFail("GPU matmul failed")
            return
        }

        // CPU (vDSP)
        var cpuResult = [Float](repeating: 0, count: M * N)
        vDSP_mmul(a, 1, b, 1, &cpuResult, 1,
                  vDSP_Length(M), vDSP_Length(N), vDSP_Length(K))

        XCTAssertEqual(gpuResult.count, cpuResult.count)
        for i in 0..<cpuResult.count {
            XCTAssertEqual(cpuResult[i], gpuResult[i], accuracy: 1e-2,
                           "Mismatch at index \(i): CPU=\(cpuResult[i]) GPU=\(gpuResult[i])")
        }
    }

    // MARK: - Mel Filterbank GPU vs CPU

    func testMelFilterbankGPUvsCPU() throws {
        guard MetalBackend.shared != nil else { throw XCTSkip("Metal not available") }

        // Generate mel filterbank weights
        let nMels = 128
        let nFFT = 2048
        let nFreqs = nFFT / 2 + 1
        let melFB = FilterBank.mel(sr: 22050, nFFT: nFFT, nMels: nMels)
        let melWeights: [Float] = melFB.withUnsafeBufferPointer { Array($0) }

        // Create a deterministic "power spectrogram" for testing
        let nFrames = 10
        var powerSpec = [Float](repeating: 0, count: nFreqs * nFrames)
        for i in 0..<powerSpec.count {
            powerSpec[i] = Float((i * 17 + 11) % 997) / 997.0
        }

        // CPU matmul (vDSP)
        var cpuResult = [Float](repeating: 0, count: nMels * nFrames)
        vDSP_mmul(melWeights, 1, powerSpec, 1, &cpuResult, 1,
                  vDSP_Length(nMels), vDSP_Length(nFrames), vDSP_Length(nFreqs))

        // GPU matmul (MPS)
        guard let gpuResult = MetalMatmul.multiply(
            a: melWeights, aRows: nMels, aCols: nFreqs,
            b: powerSpec, bRows: nFreqs, bCols: nFrames
        ) else {
            XCTFail("GPU matmul failed")
            return
        }

        // Compare
        XCTAssertEqual(cpuResult.count, gpuResult.count)
        for i in 0..<cpuResult.count {
            XCTAssertEqual(cpuResult[i], gpuResult[i], accuracy: 1e-3,
                           "Mel filterbank mismatch at index \(i): CPU=\(cpuResult[i]) GPU=\(gpuResult[i])")
        }
    }

    // MARK: - Rectangular Matrices

    func testRectangularMatrices() throws {
        guard MetalBackend.shared != nil else { throw XCTSkip("Metal not available") }

        // 3x2 @ 2x4 = 3x4
        let a: [Float] = [1, 2,
                          3, 4,
                          5, 6]
        let b: [Float] = [1, 2, 3, 4,
                          5, 6, 7, 8]

        guard let c = MetalMatmul.multiply(a: a, aRows: 3, aCols: 2,
                                           b: b, bRows: 2, bCols: 4) else {
            XCTFail("Metal matmul returned nil")
            return
        }

        // Expected:
        // [1*1+2*5, 1*2+2*6, 1*3+2*7, 1*4+2*8] = [11, 14, 17, 20]
        // [3*1+4*5, 3*2+4*6, 3*3+4*7, 3*4+4*8] = [23, 30, 37, 44]
        // [5*1+6*5, 5*2+6*6, 5*3+6*7, 5*4+6*8] = [35, 46, 57, 68]
        let expected: [Float] = [11, 14, 17, 20,
                                 23, 30, 37, 44,
                                 35, 46, 57, 68]

        XCTAssertEqual(c.count, 12)
        for i in 0..<expected.count {
            XCTAssertEqual(c[i], expected[i], accuracy: 1e-4, "Mismatch at index \(i)")
        }
    }

    // MARK: - Zero Matrix

    func testZeroMatrix() throws {
        guard MetalBackend.shared != nil else { throw XCTSkip("Metal not available") }

        let a: [Float] = [1, 2, 3, 4]
        let b = [Float](repeating: 0, count: 4)

        guard let c = MetalMatmul.multiply(a: a, aRows: 2, aCols: 2,
                                           b: b, bRows: 2, bCols: 2) else {
            XCTFail("Metal matmul returned nil")
            return
        }

        XCTAssertEqual(c.count, 4)
        for i in 0..<c.count {
            XCTAssertEqual(c[i], 0.0, accuracy: 1e-6, "Expected zero at index \(i)")
        }
    }

    // MARK: - 1x1 Matrix

    func testScalarMultiply() throws {
        guard MetalBackend.shared != nil else { throw XCTSkip("Metal not available") }

        let a: [Float] = [3.0]
        let b: [Float] = [7.0]

        guard let c = MetalMatmul.multiply(a: a, aRows: 1, aCols: 1,
                                           b: b, bRows: 1, bCols: 1) else {
            XCTFail("Metal matmul returned nil")
            return
        }

        XCTAssertEqual(c.count, 1)
        XCTAssertEqual(c[0], 21.0, accuracy: 1e-5)
    }
}
