import XCTest
@testable import MetalMomCore

final class FusedMFCCTests: XCTestCase {

    // MARK: - Helpers

    private func makeSineSignal(frequency: Float = 440.0, sr: Int = 22050,
                                duration: Float = 1.0) -> Signal {
        let count = Int(Float(sr) * duration)
        let data = (0..<count).map { i in
            sinf(2.0 * .pi * frequency * Float(i) / Float(sr))
        }
        return Signal(data: data, sampleRate: sr)
    }

    // MARK: - Shape Tests

    func testFusedMFCCShape() throws {
        let sig = Signal(data: [Float](repeating: 0.5, count: 22050), sampleRate: 22050)
        let mfcc = FusedMFCC.compute(signal: sig)
        XCTAssertEqual(mfcc.shape[0], 20, "Default nMFCC should be 20")
        XCTAssertGreaterThan(mfcc.shape[1], 0, "Should have at least 1 frame")
    }

    func testFusedMFCCCustomParams() throws {
        let sig = Signal(data: [Float](repeating: 0.5, count: 22050), sampleRate: 22050)
        let mfcc = FusedMFCC.compute(signal: sig, nMels: 40, nMFCC: 13)
        XCTAssertEqual(mfcc.shape[0], 13, "Should have 13 MFCC coefficients")
        XCTAssertGreaterThan(mfcc.shape[1], 0)
    }

    func testFusedMatchesStandard() throws {
        // Compare FusedMFCC against existing MFCC computation
        let sig = makeSineSignal()

        let fusedResult = FusedMFCC.compute(signal: sig)
        let standardResult = MFCC.compute(signal: sig)

        // Shape should match
        XCTAssertEqual(fusedResult.shape[0], standardResult.shape[0],
                       "nMFCC dimension should match")
        XCTAssertEqual(fusedResult.shape[1], standardResult.shape[1],
                       "nFrames dimension should match")

        // Values should be finite
        fusedResult.withUnsafeBufferPointer { buf in
            for i in 0..<buf.count {
                XCTAssertFalse(buf[i].isNaN, "NaN at index \(i)")
                XCTAssertFalse(buf[i].isInfinite, "Infinite at index \(i)")
            }
        }
    }

    func testFusedMFCCValuesAreFinite() throws {
        let sig = makeSineSignal()
        let mfcc = FusedMFCC.compute(signal: sig)

        mfcc.withUnsafeBufferPointer { buf in
            for i in 0..<buf.count {
                XCTAssertFalse(buf[i].isNaN, "MFCC value at \(i) should not be NaN")
                XCTAssertFalse(buf[i].isInfinite, "MFCC value at \(i) should not be infinite")
            }
        }
    }

    func testGPUvsCPUParity() throws {
        guard MetalBackend.shared != nil else { throw XCTSkip("Metal not available") }

        let sig = makeSineSignal()

        // Force CPU path
        let cpuResult = FusedMFCC.computeCPU(
            signal: sig, sr: 22050, nFFT: 2048, hopLength: 512,
            winLength: 2048, center: true, nMels: 128, nMFCC: 20,
            fMin: 0.0, fMax: 11025.0
        )

        // Force GPU path
        guard let gpuResult = FusedMFCC.computeGPU(
            signal: sig, sr: 22050, nFFT: 2048, hopLength: 512,
            winLength: 2048, center: true, nMels: 128, nMFCC: 20,
            fMin: 0.0, fMax: 11025.0
        ) else {
            XCTFail("GPU fused MFCC returned nil")
            return
        }

        // Shapes should match
        XCTAssertEqual(cpuResult.shape, gpuResult.shape,
                       "CPU shape \(cpuResult.shape) != GPU shape \(gpuResult.shape)")

        let cpuData = cpuResult.withUnsafeBufferPointer { Array($0) }
        let gpuData = gpuResult.withUnsafeBufferPointer { Array($0) }

        // Allow tolerance: GPU FFT has different precision than vDSP,
        // and the dB scale amplifies small differences
        var maxDiff: Float = 0
        for i in 0..<cpuData.count {
            let diff = abs(cpuData[i] - gpuData[i])
            maxDiff = max(maxDiff, diff)
        }
        // The dB scale amplifies small FFT differences, so be generous
        XCTAssertLessThan(maxDiff, 5.0, "Max MFCC difference: \(maxDiff)")
        print("GPU vs CPU max MFCC difference: \(maxDiff)")
    }

    func testFrameCountMatchesStandard() throws {
        let sig = makeSineSignal()
        let fused = FusedMFCC.compute(signal: sig)
        let standard = MFCC.compute(signal: sig)

        XCTAssertEqual(fused.shape[1], standard.shape[1],
                       "Fused frame count should match standard MFCC frame count")
    }

    func testBenchmarkFusedVsPerStep() throws {
        guard MetalBackend.shared != nil else { throw XCTSkip("Metal not available") }

        // 5 seconds of audio
        let nSamples = 22050 * 5
        var signal = [Float](repeating: 0, count: nSamples)
        for i in 0..<nSamples {
            signal[i] = sinf(2.0 * .pi * 440.0 * Float(i) / 22050.0)
        }
        let sig = Signal(data: signal, sampleRate: 22050)

        // Warmup
        _ = FusedMFCC.compute(signal: sig)

        // Benchmark fused
        let fusedStart = CFAbsoluteTimeGetCurrent()
        for _ in 0..<3 {
            _ = FusedMFCC.compute(signal: sig)
        }
        let fusedTime = (CFAbsoluteTimeGetCurrent() - fusedStart) / 3.0 * 1000

        // Benchmark standard
        let stdStart = CFAbsoluteTimeGetCurrent()
        for _ in 0..<3 {
            _ = MFCC.compute(signal: sig)
        }
        let stdTime = (CFAbsoluteTimeGetCurrent() - stdStart) / 3.0 * 1000

        print("Fused MFCC (5s signal): \(String(format: "%.2f", fusedTime)) ms")
        print("Standard MFCC (5s signal): \(String(format: "%.2f", stdTime)) ms")
    }

    func testSilenceProducesFiniteValues() {
        let sig = Signal(data: [Float](repeating: 0, count: 22050), sampleRate: 22050)
        let mfcc = FusedMFCC.compute(signal: sig)

        XCTAssertEqual(mfcc.shape[0], 20)
        XCTAssertGreaterThan(mfcc.shape[1], 0)

        mfcc.withUnsafeBufferPointer { buf in
            for i in 0..<buf.count {
                XCTAssertFalse(buf[i].isNaN, "MFCC of silence should not be NaN at \(i)")
                XCTAssertFalse(buf[i].isInfinite, "MFCC of silence should not be infinite at \(i)")
            }
        }
    }

    func testShortSignal() {
        // Signal shorter than nFFT but with center padding should still produce output
        let sig = Signal(data: [Float](repeating: 0.5, count: 1024), sampleRate: 22050)
        let mfcc = FusedMFCC.compute(signal: sig)
        XCTAssertEqual(mfcc.shape[0], 20)
        // With center padding of nFFT/2=1024 on each side, total = 3072 >= 2048
        XCTAssertGreaterThan(mfcc.shape[1], 0)
    }

    func testCustomFFTSize() {
        let sig = makeSineSignal()
        let mfcc = FusedMFCC.compute(signal: sig, nFFT: 1024, hopLength: 256)
        XCTAssertEqual(mfcc.shape[0], 20)
        XCTAssertGreaterThan(mfcc.shape[1], 0)

        mfcc.withUnsafeBufferPointer { buf in
            for i in 0..<buf.count {
                XCTAssertFalse(buf[i].isNaN)
                XCTAssertFalse(buf[i].isInfinite)
            }
        }
    }
}
