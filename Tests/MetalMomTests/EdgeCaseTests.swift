import XCTest
@testable import MetalMomCore

final class EdgeCaseTests: XCTestCase {

    // MARK: - Helpers

    /// Generate a 1-second signal at the given sample rate filled with the given value.
    private func makeConstantSignal(value: Float, sr: Int = 22050) -> Signal {
        let count = sr
        return Signal(data: [Float](repeating: value, count: count), sampleRate: sr)
    }

    /// Generate a 1-second silence signal.
    private func makeSilence(sr: Int = 22050) -> Signal {
        return makeConstantSignal(value: 0.0, sr: sr)
    }

    // MARK: - NaN / Inf Propagation

    func testSTFTNaNInput() {
        // Signal containing NaN values -- should not crash
        var data = [Float](repeating: 0, count: 4096)
        data[100] = Float.nan
        data[500] = Float.nan
        let signal = Signal(data: data, sampleRate: 22050)

        let result = STFT.compute(signal: signal)
        XCTAssertEqual(result.shape.count, 2, "Result should be 2D")
        XCTAssertGreaterThan(result.shape[0], 0, "Should have frequency bins")
        // Reaching this point without a crash is the primary assertion
    }

    func testSTFTInfInput() {
        // Signal containing Inf values -- should not crash
        var data = [Float](repeating: 0, count: 4096)
        data[200] = Float.infinity
        data[300] = -Float.infinity
        let signal = Signal(data: data, sampleRate: 22050)

        let result = STFT.compute(signal: signal)
        XCTAssertEqual(result.shape.count, 2, "Result should be 2D")
        XCTAssertGreaterThan(result.shape[0], 0, "Should have frequency bins")
    }

    func testMelSpectrogramNaNInput() {
        // Mel spectrogram with NaN signal -- should not crash
        var data = [Float](repeating: 0, count: 4096)
        data[0] = Float.nan
        data[1000] = Float.nan
        let signal = Signal(data: data, sampleRate: 22050)

        let result = MelSpectrogram.compute(signal: signal)
        XCTAssertEqual(result.shape.count, 2, "Result should be 2D")
    }

    func testMFCCNaNInput() {
        // MFCC with NaN signal -- should not crash
        var data = [Float](repeating: 0, count: 4096)
        data[50] = Float.nan
        let signal = Signal(data: data, sampleRate: 22050)

        let result = MFCC.compute(signal: signal)
        XCTAssertEqual(result.shape.count, 2, "Result should be 2D")
    }

    // MARK: - Zero-Length and Single-Sample Signals

    func testSTFTEmptySignal() {
        // With center=true (default), empty signal gets nFFT padding -> 1 frame.
        // With center=false, empty signal < nFFT -> 0 frames.
        let signal = Signal(data: [], sampleRate: 22050)

        let centered = STFT.compute(signal: signal)
        XCTAssertEqual(centered.shape.count, 2, "Result should be 2D")
        XCTAssertEqual(centered.shape[1], 1, "Centered empty signal produces 1 frame from padding")

        let uncentered = STFT.compute(signal: signal, center: false)
        XCTAssertEqual(uncentered.shape.count, 2, "Result should be 2D")
        XCTAssertEqual(uncentered.shape[1], 0, "Uncentered empty signal should produce 0 frames")
    }

    func testSTFTSingleSample() {
        let signal = Signal(data: [0.5], sampleRate: 22050)
        let result = STFT.compute(signal: signal)

        XCTAssertEqual(result.shape.count, 2, "Result should be 2D")
        // With center padding, 1 sample + 2*1024 padding = 2049 which is >= nFFT=2048
        // so we should get exactly 1 frame. Without center: too short -> 0 frames.
        // Default is center=true.
        XCTAssertGreaterThanOrEqual(result.shape[1], 0, "Shape should be valid")
        XCTAssertLessThanOrEqual(result.shape[1], 1, "At most 1 frame from a single sample")
    }

    func testMelSpectrogramEmptySignal() {
        // center=true (default) pads empty signal -> 1 frame
        let signal = Signal(data: [], sampleRate: 22050)
        let result = MelSpectrogram.compute(signal: signal)

        XCTAssertEqual(result.shape.count, 2, "Result should be 2D")
        XCTAssertEqual(result.shape[1], 1, "Centered empty signal produces 1 frame from padding")
    }

    func testMFCCEmptySignal() {
        // center=true (default) pads empty signal -> 1 frame
        let signal = Signal(data: [], sampleRate: 22050)
        let result = MFCC.compute(signal: signal)

        XCTAssertEqual(result.shape.count, 2, "Result should be 2D")
        XCTAssertEqual(result.shape[1], 1, "Centered empty signal produces 1 frame from padding")
    }

    // MARK: - Boundary Params (Bug Fix Verification)

    func testSTFTNonPowerOfTwoNFFT() {
        // Bug fix: non-power-of-2 nFFT used to call fatalError(). Now returns 0 frames.
        let signal = Signal(data: [Float](repeating: 0.1, count: 8000), sampleRate: 22050)
        let result = STFT.compute(signal: signal, nFFT: 1000)

        XCTAssertEqual(result.shape.count, 2, "Result should be 2D")
        XCTAssertEqual(result.shape[1], 0, "Non-power-of-2 nFFT should return 0 frames")
    }

    func testSTFTNonPowerOfTwoNFFT_500() {
        // Another non-power-of-2 case
        let signal = Signal(data: [Float](repeating: 0.1, count: 8000), sampleRate: 22050)
        let result = STFT.compute(signal: signal, nFFT: 500)

        XCTAssertEqual(result.shape.count, 2, "Result should be 2D")
        XCTAssertEqual(result.shape[1], 0, "nFFT=500 should return 0 frames")
    }

    func testSTFTPowerSpecNonPowerOfTwo() {
        // computePowerSpectrogram should also handle non-power-of-2 gracefully
        let signal = Signal(data: [Float](repeating: 0.1, count: 8000), sampleRate: 22050)
        let result = STFT.computePowerSpectrogram(signal: signal, nFFT: 1000)

        XCTAssertEqual(result.shape.count, 2, "Result should be 2D")
        XCTAssertEqual(result.shape[1], 0, "Non-power-of-2 nFFT should return 0 frames")
    }

    func testSTFTComplexNonPowerOfTwo() {
        // computeComplex should also handle non-power-of-2 gracefully
        let signal = Signal(data: [Float](repeating: 0.1, count: 8000), sampleRate: 22050)
        let result = STFT.computeComplex(signal: signal, nFFT: 1000)

        XCTAssertEqual(result.shape.count, 2, "Result should be 2D")
        XCTAssertEqual(result.shape[1], 0, "Non-power-of-2 nFFT should return 0 frames")
        XCTAssertEqual(result.dtype, .complex64, "Should still have complex dtype")
    }

    func testSTFTWinLengthGreaterThanNFFT() {
        // Bug fix: winLength > nFFT used to read past window buffer (UB).
        // Now truncates window to center nFFT samples.
        let signal = Signal(data: [Float](repeating: 0.1, count: 22050), sampleRate: 22050)
        let result = STFT.compute(signal: signal, nFFT: 2048, winLength: 4096)

        XCTAssertEqual(result.shape.count, 2, "Result should be 2D")
        XCTAssertEqual(result.shape[0], 1025, "nFreqs should be nFFT/2 + 1 = 1025")
        XCTAssertGreaterThan(result.shape[1], 0, "Should produce frames")
    }

    func testSTFTHopGreaterThanNFFT() {
        // hopLength > nFFT: some frames will skip input data, but should still work
        let signal = Signal(data: [Float](repeating: 0.1, count: 22050), sampleRate: 22050)
        let result = STFT.compute(signal: signal, nFFT: 2048, hopLength: 4096)

        XCTAssertEqual(result.shape.count, 2, "Result should be 2D")
        XCTAssertEqual(result.shape[0], 1025, "nFreqs should be nFFT/2 + 1")
        XCTAssertGreaterThan(result.shape[1], 0, "Should produce at least some frames")
    }

    func testMelZeroMels() {
        // nMels=0 should not crash
        let signal = Signal(data: [Float](repeating: 0.1, count: 4096), sampleRate: 22050)
        let result = MelSpectrogram.compute(signal: signal, nMels: 0)

        XCTAssertEqual(result.shape.count, 2, "Result should be 2D")
        // With 0 mel bands, first dimension should be 0
        XCTAssertEqual(result.shape[0], 0, "0 mel bands should produce 0 rows")
    }

    // MARK: - nMFCC Edge Cases

    func testMFCCGreaterThanMels() {
        // nMFCC > nMels: should be capped to nMels (via min(nMFCC, melRows))
        let signal = Signal(data: [Float](repeating: 0.1, count: 22050), sampleRate: 22050)
        let result = MFCC.compute(signal: signal, nMFCC: 200, nMels: 40)

        XCTAssertEqual(result.shape.count, 2, "Result should be 2D")
        XCTAssertEqual(result.shape[0], 40, "nMFCC should be capped at nMels=40")
    }

    func testMFCCSingleCoefficient() {
        // nMFCC=1: only the energy-like coefficient
        let signal = Signal(data: [Float](repeating: 0.1, count: 22050), sampleRate: 22050)
        let result = MFCC.compute(signal: signal, nMFCC: 1)

        XCTAssertEqual(result.shape.count, 2, "Result should be 2D")
        XCTAssertEqual(result.shape[0], 1, "Should have exactly 1 MFCC coefficient")
        XCTAssertGreaterThan(result.shape[1], 0, "Should have at least 1 frame")
    }

    func testMFCCEqualToMels() {
        // nMFCC == nMels: should return exactly nMels coefficients
        let signal = Signal(data: [Float](repeating: 0.1, count: 22050), sampleRate: 22050)
        let result = MFCC.compute(signal: signal, nMFCC: 40, nMels: 40)

        XCTAssertEqual(result.shape.count, 2, "Result should be 2D")
        XCTAssertEqual(result.shape[0], 40, "Should have 40 MFCC coefficients")
        XCTAssertGreaterThan(result.shape[1], 0, "Should have at least 1 frame")
    }

    // MARK: - Silence and DC Signals

    func testSTFTAllSilence() {
        // 1 second of zeros: STFT output should be near-zero everywhere
        let signal = makeSilence()
        let result = STFT.compute(signal: signal)

        XCTAssertGreaterThan(result.count, 0, "Should have output elements")
        for i in 0..<result.count {
            XCTAssertEqual(result[i], 0.0, accuracy: 1e-7,
                           "Silence STFT should be near-zero at index \(i)")
        }
    }

    func testSTFTDCConstant() {
        // 1 second of constant 1.0: DC bin (index 0) should dominate
        let signal = makeConstantSignal(value: 1.0)
        let result = STFT.compute(signal: signal)

        let nFrames = result.shape[1]
        XCTAssertGreaterThan(nFrames, 0, "Should have frames")

        // Check the middle frame: DC bin should be much larger than others
        let midFrame = nFrames / 2
        let dcValue = result[0 * nFrames + midFrame]  // freq bin 0, middle frame

        // DC value for constant signal windowed by Hann should be positive and significant
        XCTAssertGreaterThan(dcValue, 0.0, "DC bin should be positive for constant signal")

        // Sum of non-DC bins should be much smaller than DC
        var nonDCSum: Float = 0
        let nFreqs = result.shape[0]
        for bin in 1..<nFreqs {
            nonDCSum += result[bin * nFrames + midFrame]
        }

        // Non-DC sum may not be strictly zero (windowing effects), but DC should dominate
        XCTAssertGreaterThan(dcValue, nonDCSum / Float(nFreqs),
                             "DC bin should dominate over average non-DC bin")
    }

    func testMelSpectrogramSilence() {
        // Silence through mel spectrogram should be near-zero
        let signal = makeSilence()
        let result = MelSpectrogram.compute(signal: signal)

        XCTAssertGreaterThan(result.count, 0, "Should have output elements")
        for i in 0..<result.count {
            XCTAssertEqual(result[i], 0.0, accuracy: 1e-6,
                           "Silence mel spectrogram should be near-zero at index \(i)")
        }
    }

    func testMFCCSilence() {
        // Silence through MFCC should produce finite values (no crash, no NaN)
        let signal = makeSilence()
        let result = MFCC.compute(signal: signal)

        XCTAssertEqual(result.shape.count, 2, "Result should be 2D")
        XCTAssertGreaterThan(result.count, 0, "Should have output elements")
        for i in 0..<result.count {
            XCTAssertTrue(result[i].isFinite,
                          "MFCC of silence should produce finite values, got \(result[i]) at index \(i)")
        }
    }

    // MARK: - Large Float Values

    func testSTFTLargeValues() {
        // Very large float values should not crash
        let largeVal = Float.greatestFiniteMagnitude / 1000.0
        let data = [Float](repeating: largeVal, count: 4096)
        let signal = Signal(data: data, sampleRate: 22050)

        let result = STFT.compute(signal: signal)
        XCTAssertEqual(result.shape.count, 2, "Result should be 2D")
        XCTAssertGreaterThan(result.shape[1], 0, "Should produce frames")
        // Reaching this point without a crash is the primary assertion
    }

    func testAmplitudeToDbLargeValues() {
        // Very large signal values through amplitudeToDb should not crash
        let largeVal = Float.greatestFiniteMagnitude / 1000.0
        let data = [Float](repeating: largeVal, count: 1024)
        let signal = Signal(data: data, shape: [32, 32], sampleRate: 22050)

        let result = Scaling.amplitudeToDb(signal)
        XCTAssertEqual(result.count, 1024, "Should have same number of elements")
        for i in 0..<result.count {
            XCTAssertTrue(result[i].isFinite,
                          "amplitudeToDb of large values should be finite, got \(result[i]) at index \(i)")
        }
    }

    func testAmplitudeToDbZeroValues() {
        // All-zero signal through amplitudeToDb: should clamp at amin and produce finite dB values
        let data = [Float](repeating: 0.0, count: 1024)
        let signal = Signal(data: data, shape: [32, 32], sampleRate: 22050)

        let result = Scaling.amplitudeToDb(signal)
        XCTAssertEqual(result.count, 1024, "Should have same number of elements")
        for i in 0..<result.count {
            XCTAssertTrue(result[i].isFinite,
                          "amplitudeToDb of zeros should be finite (clamped by amin), got \(result[i]) at index \(i)")
        }
    }
}
