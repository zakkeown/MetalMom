import XCTest
import Accelerate
import Foundation
@testable import MetalMomCore

/// Cross-validates GPU (Metal shader) results against CPU (vDSP/Accelerate) reference
/// implementations. Every test guards on Metal availability via XCTSkip so the suite
/// runs cleanly on CI machines without a GPU.
final class GPUParityTests: XCTestCase {

    // MARK: - Helpers

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

    /// Generate a deterministic pseudo-random Float array in [lo, hi).
    /// Uses a simple LCG seeded with 42 for reproducibility.
    private func deterministicRandom(count: Int, lo: Float = 0.0, hi: Float = 1.0) -> [Float] {
        var state: UInt64 = 42
        return (0..<count).map { _ in
            // LCG: state = state * 6364136223846793005 + 1442695040888963407
            state = state &* 6364136223846793005 &+ 1442695040888963407
            let normalized = Float(state >> 33) / Float(UInt32.max)
            return lo + normalized * (hi - lo)
        }
    }

    /// Generate a sine wave signal.
    private func makeSine(frequency: Float, sr: Int = 22050, duration: Float = 1.0) -> Signal {
        let count = Int(Float(sr) * duration)
        let data = (0..<count).map { i in
            sinf(2.0 * .pi * frequency * Float(i) / Float(sr))
        }
        return Signal(data: data, sampleRate: sr)
    }

    /// Generate a linear chirp from f0 to f1.
    private func makeChirp(f0: Float, f1: Float, sr: Int = 22050, duration: Float = 1.0) -> Signal {
        let count = Int(Float(sr) * duration)
        let data = (0..<count).map { i -> Float in
            let t = Float(i) / Float(sr)
            let freq = f0 + (f1 - f0) * t / duration
            return sinf(2.0 * .pi * freq * t)
        }
        return Signal(data: data, sampleRate: sr)
    }

    // =========================================================================
    // MARK: - Group 1: Elementwise Parity (10,000-element arrays)
    // =========================================================================

    func testLogGPUvsCPU() throws {
        let shaders = try getShaders()

        // Positive random values in [0.01, 100] to avoid log(0)
        let input = deterministicRandom(count: 10_000, lo: 0.01, hi: 100.0)

        // GPU
        guard let gpuResult = shaders.log(input) else {
            XCTFail("Metal log returned nil")
            return
        }

        // CPU: vvlogf
        var cpuResult = [Float](repeating: 0, count: input.count)
        var cpuInput = input
        var count = Int32(input.count)
        vvlogf(&cpuResult, &cpuInput, &count)

        XCTAssertEqual(gpuResult.count, cpuResult.count)
        var maxDiff: Float = 0
        for i in 0..<cpuResult.count {
            let diff = Swift.abs(gpuResult[i] - cpuResult[i])
            if diff > maxDiff { maxDiff = diff }
        }
        XCTAssertLessThan(maxDiff, 1e-4,
            "log GPU vs CPU max diff: \(maxDiff)")
    }

    func testExpGPUvsCPU() throws {
        let shaders = try getShaders()

        // Values in [-5, 5] to avoid overflow
        let input = deterministicRandom(count: 10_000, lo: -5.0, hi: 5.0)

        // GPU
        guard let gpuResult = shaders.exp(input) else {
            XCTFail("Metal exp returned nil")
            return
        }

        // CPU: vvexpf
        var cpuResult = [Float](repeating: 0, count: input.count)
        var cpuInput = input
        var count = Int32(input.count)
        vvexpf(&cpuResult, &cpuInput, &count)

        XCTAssertEqual(gpuResult.count, cpuResult.count)
        var maxDiff: Float = 0
        for i in 0..<cpuResult.count {
            let diff = Swift.abs(gpuResult[i] - cpuResult[i])
            if diff > maxDiff { maxDiff = diff }
        }
        XCTAssertLessThan(maxDiff, 1e-3,
            "exp GPU vs CPU max diff: \(maxDiff)")
    }

    func testPowGPUvsCPU() throws {
        let shaders = try getShaders()

        // Positive values in [0.1, 10] with exponent 2.0
        let input = deterministicRandom(count: 10_000, lo: 0.1, hi: 10.0)
        let exponent: Float = 2.0

        // GPU
        guard let gpuResult = shaders.pow(input, exponent: exponent) else {
            XCTFail("Metal pow returned nil")
            return
        }

        // CPU: manual squaring via vDSP_vsq
        var cpuResult = [Float](repeating: 0, count: input.count)
        input.withUnsafeBufferPointer { src in
            cpuResult.withUnsafeMutableBufferPointer { dst in
                vDSP_vsq(src.baseAddress!, 1, dst.baseAddress!, 1, vDSP_Length(input.count))
            }
        }

        XCTAssertEqual(gpuResult.count, cpuResult.count)
        var maxDiff: Float = 0
        for i in 0..<cpuResult.count {
            let diff = Swift.abs(gpuResult[i] - cpuResult[i])
            if diff > maxDiff { maxDiff = diff }
        }
        XCTAssertLessThan(maxDiff, 1e-3,
            "pow(x, 2) GPU vs CPU max diff: \(maxDiff)")
    }

    func testAmplitudeToDbGPUvsCPU() throws {
        let shaders = try getShaders()

        // Positive random values in [0.001, 100]
        let input = deterministicRandom(count: 10_000, lo: 0.001, hi: 100.0)

        // GPU: shaders.amplitudeToDb
        guard let gpuResult = shaders.amplitudeToDb(input) else {
            XCTFail("Metal amplitudeToDb returned nil")
            return
        }

        // CPU: Scaling.amplitudeToDb on a Signal
        // Use amin=1e-10 to match Metal shader default, and topDb=nil to disable
        // clipping so we compare the raw dB conversion.
        let cpuSignal = Signal(data: input, shape: [1, input.count], sampleRate: 22050)
        let cpuDbSignal = Scaling.amplitudeToDb(cpuSignal, amin: 1e-10, topDb: nil)
        let cpuResult = cpuDbSignal.withUnsafeBufferPointer { Array($0) }

        XCTAssertEqual(gpuResult.count, cpuResult.count)
        var maxDiff: Float = 0
        for i in 0..<cpuResult.count {
            let diff = Swift.abs(gpuResult[i] - cpuResult[i])
            if diff > maxDiff { maxDiff = diff }
        }
        XCTAssertLessThan(maxDiff, 1e-2,
            "amplitudeToDb GPU vs CPU max diff: \(maxDiff)")
    }

    func testPowerToDbGPUvsCPU() throws {
        let shaders = try getShaders()

        // Positive random values in [0.001, 100]
        let input = deterministicRandom(count: 10_000, lo: 0.001, hi: 100.0)

        // GPU: shaders.powerToDb
        guard let gpuResult = shaders.powerToDb(input) else {
            XCTFail("Metal powerToDb returned nil")
            return
        }

        // CPU: Scaling.powerToDb on a Signal
        // Use amin=1e-10 to match Metal shader default, and topDb=nil to disable clipping.
        let cpuSignal = Signal(data: input, shape: [1, input.count], sampleRate: 22050)
        let cpuDbSignal = Scaling.powerToDb(cpuSignal, amin: 1e-10, topDb: nil)
        let cpuResult = cpuDbSignal.withUnsafeBufferPointer { Array($0) }

        XCTAssertEqual(gpuResult.count, cpuResult.count)
        var maxDiff: Float = 0
        for i in 0..<cpuResult.count {
            let diff = Swift.abs(gpuResult[i] - cpuResult[i])
            if diff > maxDiff { maxDiff = diff }
        }
        XCTAssertLessThan(maxDiff, 1e-2,
            "powerToDb GPU vs CPU max diff: \(maxDiff)")
    }

    // =========================================================================
    // MARK: - Group 2: Reductions (100,000-element arrays)
    // =========================================================================

    func testSumGPUvsCPU() throws {
        let shaders = try getShaders()

        let input = deterministicRandom(count: 100_000, lo: -1.0, hi: 1.0)

        // GPU
        guard let gpuResult = shaders.sum(input) else {
            XCTFail("Metal sum returned nil")
            return
        }

        // CPU: vDSP_sve
        var cpuResult: Float = 0
        input.withUnsafeBufferPointer { buf in
            vDSP_sve(buf.baseAddress!, 1, &cpuResult, vDSP_Length(input.count))
        }

        // Relative tolerance 1% (float32 accumulation precision)
        let absDiff = Swift.abs(gpuResult - cpuResult)
        let denom = Swift.max(Swift.abs(cpuResult), 1e-10)
        let relError = absDiff / denom
        XCTAssertLessThan(relError, 0.01,
            "sum GPU=\(gpuResult) CPU=\(cpuResult) relError=\(relError)")
    }

    func testMaxGPUvsCPU() throws {
        let shaders = try getShaders()

        let input = deterministicRandom(count: 100_000, lo: -1000.0, hi: 1000.0)

        // GPU
        guard let gpuResult = shaders.max(input) else {
            XCTFail("Metal max returned nil")
            return
        }

        // CPU: vDSP_maxv
        var cpuResult: Float = 0
        input.withUnsafeBufferPointer { buf in
            vDSP_maxv(buf.baseAddress!, 1, &cpuResult, vDSP_Length(input.count))
        }

        // Exact match expected for max
        XCTAssertEqual(gpuResult, cpuResult, accuracy: 1e-6,
            "max GPU=\(gpuResult) CPU=\(cpuResult)")
    }

    func testMinGPUvsCPU() throws {
        let shaders = try getShaders()

        let input = deterministicRandom(count: 100_000, lo: -1000.0, hi: 1000.0)

        // GPU
        guard let gpuResult = shaders.min(input) else {
            XCTFail("Metal min returned nil")
            return
        }

        // CPU: vDSP_minv
        var cpuResult: Float = 0
        input.withUnsafeBufferPointer { buf in
            vDSP_minv(buf.baseAddress!, 1, &cpuResult, vDSP_Length(input.count))
        }

        // Exact match expected for min
        XCTAssertEqual(gpuResult, cpuResult, accuracy: 1e-6,
            "min GPU=\(gpuResult) CPU=\(cpuResult)")
    }

    func testMeanGPUvsCPU() throws {
        let shaders = try getShaders()

        let input = deterministicRandom(count: 100_000, lo: -100.0, hi: 100.0)

        // GPU
        guard let gpuResult = shaders.mean(input) else {
            XCTFail("Metal mean returned nil")
            return
        }

        // CPU: vDSP_meanv
        var cpuResult: Float = 0
        input.withUnsafeBufferPointer { buf in
            vDSP_meanv(buf.baseAddress!, 1, &cpuResult, vDSP_Length(input.count))
        }

        // Relative tolerance 0.1%
        let absDiff = Swift.abs(gpuResult - cpuResult)
        let denom = Swift.max(Swift.abs(cpuResult), 1e-10)
        let relError = absDiff / denom
        XCTAssertLessThan(relError, 0.001,
            "mean GPU=\(gpuResult) CPU=\(cpuResult) relError=\(relError)")
    }

    // =========================================================================
    // MARK: - Group 3: Full Pipeline Parity
    // =========================================================================

    func testMelSpectrogramGPUvsCPU() throws {
        guard MetalBackend.shared != nil else {
            throw XCTSkip("No Metal device available")
        }

        // 1-second 440 Hz sine at 22050 Hz
        let sig = makeSine(frequency: 440.0, sr: 22050, duration: 1.0)

        // Compute mel spectrogram via the default path (SmartDispatcher decides GPU/CPU).
        // To cross-validate, we also compute the mel filterbank matmul on CPU directly.
        let nFFT = 2048
        let hopLength = 512
        let nMels = 128

        // CPU-only mel spectrogram: use power spectrogram + vDSP matmul
        let powSpec = STFT.computePowerSpectrogram(
            signal: sig, nFFT: nFFT, hopLength: hopLength, winLength: nFFT, center: true
        )
        let nFreqs = powSpec.shape[0]
        let nFrames = powSpec.shape[1]

        let melFB = FilterBank.mel(sr: 22050, nFFT: nFFT, nMels: nMels)
        let melWeights: [Float] = melFB.withUnsafeBufferPointer { Array($0) }
        let poweredArray: [Float] = powSpec.withUnsafeBufferPointer { Array($0) }

        // CPU matmul
        var cpuMel = [Float](repeating: 0, count: nMels * nFrames)
        vDSP_mmul(melWeights, 1, poweredArray, 1, &cpuMel, 1,
                  vDSP_Length(nMels), vDSP_Length(nFrames), vDSP_Length(nFreqs))

        // GPU matmul
        guard let gpuMel = MetalMatmul.multiply(
            a: melWeights, aRows: nMels, aCols: nFreqs,
            b: poweredArray, bRows: nFreqs, bCols: nFrames
        ) else {
            XCTFail("GPU mel matmul returned nil")
            return
        }

        XCTAssertEqual(cpuMel.count, gpuMel.count)

        // Compare: check max absolute diff for bins with significant energy
        var maxDiff: Float = 0
        let energyThreshold: Float = 0.001
        var checkedCount = 0
        for i in 0..<cpuMel.count {
            if cpuMel[i] > energyThreshold || gpuMel[i] > energyThreshold {
                let diff = Swift.abs(cpuMel[i] - gpuMel[i])
                if diff > maxDiff { maxDiff = diff }
                checkedCount += 1
            }
        }

        XCTAssertGreaterThan(checkedCount, 0, "Should have some significant mel bins")
        XCTAssertLessThan(maxDiff, 0.05,
            "Mel spectrogram GPU vs CPU max diff: \(maxDiff) over \(checkedCount) significant bins")
    }

    func testMFCCEndToEnd() throws {
        guard MetalBackend.shared != nil else {
            throw XCTSkip("No Metal device available")
        }

        let sig = makeSine(frequency: 440.0, sr: 22050, duration: 1.0)

        // Standard MFCC (CPU path)
        let standardResult = MFCC.compute(signal: sig)

        // Fused MFCC (GPU path when available)
        guard let fusedResult = FusedMFCC.computeGPU(
            signal: sig, sr: 22050, nFFT: 2048, hopLength: 512,
            winLength: 2048, center: true, nMels: 128, nMFCC: 20,
            fMin: 0.0, fMax: 11025.0
        ) else {
            throw XCTSkip("GPU FusedMFCC returned nil — GPU path unavailable")
        }

        // Shapes should match
        XCTAssertEqual(standardResult.shape[0], fusedResult.shape[0],
            "nMFCC dimension mismatch: standard=\(standardResult.shape[0]) fused=\(fusedResult.shape[0])")
        XCTAssertEqual(standardResult.shape[1], fusedResult.shape[1],
            "nFrames dimension mismatch: standard=\(standardResult.shape[1]) fused=\(fusedResult.shape[1])")

        let cpuData = standardResult.withUnsafeBufferPointer { Array($0) }
        let gpuData = fusedResult.withUnsafeBufferPointer { Array($0) }

        // Compare values. The fused GPU pipeline uses MPSGraph FFT which has
        // different numerical properties than vDSP FFT, and the dB scale amplifies
        // small differences, so we use a generous tolerance.
        var maxDiff: Float = 0
        for i in 0..<Swift.min(cpuData.count, gpuData.count) {
            let diff = Swift.abs(cpuData[i] - gpuData[i])
            if diff > maxDiff { maxDiff = diff }
        }
        XCTAssertLessThan(maxDiff, 0.5,
            "MFCC end-to-end max diff: \(maxDiff)")
    }

    // =========================================================================
    // MARK: - Group 4: Multi-Signal STFT Robustness
    // =========================================================================

    func testSTFTParitySine() throws {
        guard MetalBackend.shared != nil else {
            throw XCTSkip("No Metal device available")
        }

        let sig = makeSine(frequency: 440.0, sr: 22050, duration: 1.0)
        let input = STFTInput(signal: sig, nFFT: 2048, hopLength: 512, winLength: 2048, center: true)
        let stft = STFT()

        let cpuResult = stft.executeCPU(input)
        let gpuResult = stft.executeGPU(input)

        XCTAssertEqual(cpuResult.magnitude.shape, gpuResult.magnitude.shape,
            "Shape mismatch: cpu=\(cpuResult.magnitude.shape) gpu=\(gpuResult.magnitude.shape)")

        let cpuData = cpuResult.magnitude.withUnsafeBufferPointer { Array($0) }
        let gpuData = gpuResult.magnitude.withUnsafeBufferPointer { Array($0) }

        var maxDiff: Float = 0
        for i in 0..<cpuData.count {
            let diff = Swift.abs(cpuData[i] - gpuData[i])
            if diff > maxDiff { maxDiff = diff }
        }
        XCTAssertLessThan(maxDiff, 1e-2,
            "STFT sine parity max diff: \(maxDiff)")
    }

    func testSTFTParityChirp() throws {
        guard MetalBackend.shared != nil else {
            throw XCTSkip("No Metal device available")
        }

        let sig = makeChirp(f0: 200.0, f1: 4000.0, sr: 22050, duration: 1.0)
        let input = STFTInput(signal: sig, nFFT: 2048, hopLength: 512, winLength: 2048, center: true)
        let stft = STFT()

        let cpuResult = stft.executeCPU(input)
        let gpuResult = stft.executeGPU(input)

        XCTAssertEqual(cpuResult.magnitude.shape, gpuResult.magnitude.shape,
            "Shape mismatch: cpu=\(cpuResult.magnitude.shape) gpu=\(gpuResult.magnitude.shape)")

        let cpuData = cpuResult.magnitude.withUnsafeBufferPointer { Array($0) }
        let gpuData = gpuResult.magnitude.withUnsafeBufferPointer { Array($0) }

        var maxDiff: Float = 0
        for i in 0..<cpuData.count {
            let diff = Swift.abs(cpuData[i] - gpuData[i])
            if diff > maxDiff { maxDiff = diff }
        }
        XCTAssertLessThan(maxDiff, 1e-2,
            "STFT chirp parity max diff: \(maxDiff)")
    }

    func testSTFTParityNoise() throws {
        guard MetalBackend.shared != nil else {
            throw XCTSkip("No Metal device available")
        }

        // Deterministic noise: 1 second at 22050 Hz
        let noiseData = deterministicRandom(count: 22050, lo: -1.0, hi: 1.0)
        let sig = Signal(data: noiseData, sampleRate: 22050)
        let input = STFTInput(signal: sig, nFFT: 2048, hopLength: 512, winLength: 2048, center: true)
        let stft = STFT()

        let cpuResult = stft.executeCPU(input)
        let gpuResult = stft.executeGPU(input)

        XCTAssertEqual(cpuResult.magnitude.shape, gpuResult.magnitude.shape,
            "Shape mismatch: cpu=\(cpuResult.magnitude.shape) gpu=\(gpuResult.magnitude.shape)")

        let cpuData = cpuResult.magnitude.withUnsafeBufferPointer { Array($0) }
        let gpuData = gpuResult.magnitude.withUnsafeBufferPointer { Array($0) }

        var maxDiff: Float = 0
        for i in 0..<cpuData.count {
            let diff = Swift.abs(cpuData[i] - gpuData[i])
            if diff > maxDiff { maxDiff = diff }
        }
        XCTAssertLessThan(maxDiff, 1e-2,
            "STFT noise parity max diff: \(maxDiff)")
    }

    func testSTFTParityMultiFFTSize() throws {
        guard MetalBackend.shared != nil else {
            throw XCTSkip("No Metal device available")
        }

        // Use a chirp signal that exercises many frequency bins
        let sig = makeChirp(f0: 100.0, f1: 8000.0, sr: 22050, duration: 1.0)
        let stft = STFT()

        let fftSizes = [256, 512, 1024, 2048, 4096]

        for nFFT in fftSizes {
            let hopLength = nFFT / 4
            let input = STFTInput(signal: sig, nFFT: nFFT, hopLength: hopLength,
                                  winLength: nFFT, center: true)

            let cpuResult = stft.executeCPU(input)
            let gpuResult = stft.executeGPU(input)

            // Verify shapes match
            XCTAssertEqual(cpuResult.magnitude.shape, gpuResult.magnitude.shape,
                "nFFT=\(nFFT): Shape mismatch cpu=\(cpuResult.magnitude.shape) gpu=\(gpuResult.magnitude.shape)")

            let cpuData = cpuResult.magnitude.withUnsafeBufferPointer { Array($0) }
            let gpuData = gpuResult.magnitude.withUnsafeBufferPointer { Array($0) }

            XCTAssertEqual(cpuData.count, gpuData.count,
                "nFFT=\(nFFT): Element count mismatch")

            var maxDiff: Float = 0
            for i in 0..<cpuData.count {
                let diff = Swift.abs(cpuData[i] - gpuData[i])
                if diff > maxDiff { maxDiff = diff }
            }
            XCTAssertLessThan(maxDiff, 1e-2,
                "nFFT=\(nFFT): STFT parity max diff \(maxDiff)")
        }
    }
}
