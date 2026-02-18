import XCTest
@testable import MetalMomCore

final class CQTTests: XCTestCase {

    // MARK: - Helpers

    /// Generate a sine wave signal.
    private func makeSineSignal(frequency: Float = 440.0, sr: Int = 22050,
                                 duration: Float = 1.0) -> Signal {
        let count = Int(Float(sr) * duration)
        let data = (0..<count).map { i in
            sinf(2.0 * .pi * frequency * Float(i) / Float(sr))
        }
        return Signal(data: data, sampleRate: sr)
    }

    // MARK: - CQT Filterbank Tests

    func testCQTFilterbankShape() {
        let fb = CQT.cqtFilterbank(sr: 22050, nFFT: 2048, fMin: 32.70, nBins: 84, binsPerOctave: 12)
        XCTAssertEqual(fb.shape.count, 2, "CQT filterbank should be 2D")
        XCTAssertEqual(fb.shape[0], 84, "Should have 84 CQT bins")
        XCTAssertEqual(fb.shape[1], 1025, "Should have nFFT/2+1 frequency bins")
    }

    func testCQTFilterbankNonNegative() {
        let fb = CQT.cqtFilterbank(sr: 22050, nFFT: 2048, fMin: 32.70, nBins: 84, binsPerOctave: 12)
        for i in 0..<fb.count {
            XCTAssertGreaterThanOrEqual(fb[i], 0.0, "CQT filterbank values should be non-negative")
        }
    }

    func testCQTFilterbankRowsNonZero() {
        // Every CQT bin should have at least some non-zero weights (at reasonable nFFT)
        let fb = CQT.cqtFilterbank(sr: 22050, nFFT: 4096, fMin: 32.70, nBins: 84, binsPerOctave: 12)
        let nBins = fb.shape[0]
        let nFreqs = fb.shape[1]

        for k in 0..<nBins {
            var rowSum: Float = 0
            for j in 0..<nFreqs {
                rowSum += fb[k * nFreqs + j]
            }
            XCTAssertGreaterThan(rowSum, 0, "CQT bin \(k) should have non-zero weights")
        }
    }

    func testCQTFilterbankCustomParams() {
        let fb = CQT.cqtFilterbank(sr: 44100, nFFT: 4096, fMin: 65.41, nBins: 48, binsPerOctave: 24)
        XCTAssertEqual(fb.shape[0], 48, "Should have 48 bins")
        XCTAssertEqual(fb.shape[1], 2049, "Should have 4096/2+1 frequency bins")
    }

    // MARK: - CQT Compute Tests

    func testCQTOutputShape() {
        let signal = makeSineSignal(frequency: 440.0, sr: 22050, duration: 1.0)
        let result = CQT.compute(signal: signal, fMin: 32.70, fMax: 8000.0, binsPerOctave: 12)

        XCTAssertEqual(result.shape.count, 2, "CQT should be 2D")
        XCTAssertGreaterThan(result.shape[0], 0, "Should have CQT bins")
        XCTAssertGreaterThan(result.shape[1], 0, "Should have frames")
    }

    func testCQTValuesNonNegative() {
        let signal = makeSineSignal(frequency: 440.0, sr: 22050, duration: 1.0)
        let result = CQT.compute(signal: signal, fMin: 32.70, fMax: 8000.0, binsPerOctave: 12)

        for i in 0..<result.count {
            XCTAssertGreaterThanOrEqual(result[i], 0.0, "CQT magnitudes should be non-negative")
            XCTAssertFalse(result[i].isNaN, "CQT values should not be NaN")
            XCTAssertFalse(result[i].isInfinite, "CQT values should not be infinite")
        }
    }

    func testCQTSineWavePeakAtCorrectBin() {
        // 440 Hz sine wave should produce a peak at the CQT bin closest to 440 Hz
        let sr = 22050
        let signal = makeSineSignal(frequency: 440.0, sr: sr, duration: 2.0)
        let fMin: Float = 32.70
        let binsPerOctave = 12
        let result = CQT.compute(signal: signal, fMin: fMin, fMax: 8000.0,
                                  binsPerOctave: binsPerOctave)

        let nBins = result.shape[0]
        let nFrames = result.shape[1]

        // Average energy per bin
        var avgEnergy = [Float](repeating: 0, count: nBins)
        for k in 0..<nBins {
            for f in 0..<nFrames {
                avgEnergy[k] += result[k * nFrames + f]
            }
            avgEnergy[k] /= Float(nFrames)
        }

        // Find peak bin
        let peakBin = avgEnergy.enumerated().max(by: { $0.element < $1.element })!.offset
        let peakFreq = fMin * powf(2.0, Float(peakBin) / Float(binsPerOctave))

        // The peak frequency should be within 1 semitone of 440 Hz
        let ratio = peakFreq / 440.0
        XCTAssertGreaterThan(ratio, 0.9, "Peak frequency \(peakFreq) Hz should be near 440 Hz")
        XCTAssertLessThan(ratio, 1.1, "Peak frequency \(peakFreq) Hz should be near 440 Hz")
    }

    func testCQTSilenceProducesZeros() {
        let signal = Signal(data: [Float](repeating: 0, count: 22050), sampleRate: 22050)
        let result = CQT.compute(signal: signal, fMin: 32.70, fMax: 8000.0)

        for i in 0..<result.count {
            XCTAssertEqual(result[i], 0.0, accuracy: 1e-10, "CQT of silence should be zero")
        }
    }

    func testCQTShortSignal() {
        // Very short signal -- should not crash
        let signal = Signal(data: [Float](repeating: 0.5, count: 512), sampleRate: 22050)
        let result = CQT.compute(signal: signal, fMin: 32.70, fMax: 4000.0, binsPerOctave: 12)
        // May produce zero frames but should not crash
        XCTAssertEqual(result.shape.count, 2)
    }

    func testCQTSingleBin() {
        // Only a narrow frequency range -- should produce a small number of bins
        let signal = makeSineSignal(frequency: 440.0, sr: 22050, duration: 1.0)
        let result = CQT.compute(signal: signal, fMin: 400.0, fMax: 500.0, binsPerOctave: 12)

        XCTAssertEqual(result.shape.count, 2)
        XCTAssertGreaterThan(result.shape[0], 0, "Should have at least 1 bin")
    }

    func testCQTDefaultParams() {
        // Test that calling with no optional parameters works
        let signal = makeSineSignal(frequency: 440.0, sr: 22050, duration: 1.0)
        let result = CQT.compute(signal: signal)

        XCTAssertEqual(result.shape.count, 2)
        XCTAssertGreaterThan(result.shape[0], 0)
        XCTAssertGreaterThan(result.shape[1], 0)
    }

    // MARK: - VQT Tests

    func testVQTOutputShape() {
        let signal = makeSineSignal(frequency: 440.0, sr: 22050, duration: 1.0)
        let result = CQT.vqt(signal: signal, fMin: 32.70, fMax: 8000.0,
                              binsPerOctave: 12, gamma: 20.0)

        XCTAssertEqual(result.shape.count, 2, "VQT should be 2D")
        XCTAssertGreaterThan(result.shape[0], 0, "Should have VQT bins")
        XCTAssertGreaterThan(result.shape[1], 0, "Should have frames")
    }

    func testVQTWithZeroGammaMatchesCQT() {
        // VQT with gamma=0 should be identical to CQT
        let signal = makeSineSignal(frequency: 440.0, sr: 22050, duration: 1.0)
        let cqt = CQT.compute(signal: signal, fMin: 32.70, fMax: 8000.0, binsPerOctave: 12)
        let vqt = CQT.vqt(signal: signal, fMin: 32.70, fMax: 8000.0,
                           binsPerOctave: 12, gamma: 0.0)

        XCTAssertEqual(cqt.shape, vqt.shape, "VQT with gamma=0 should have same shape as CQT")

        // Values should be very close
        for i in 0..<min(cqt.count, vqt.count) {
            XCTAssertEqual(cqt[i], vqt[i], accuracy: 1e-5,
                          "VQT with gamma=0 should match CQT")
        }
    }

    func testVQTValuesNonNegative() {
        let signal = makeSineSignal(frequency: 440.0, sr: 22050, duration: 1.0)
        let result = CQT.vqt(signal: signal, fMin: 32.70, fMax: 8000.0,
                              binsPerOctave: 12, gamma: 20.0)

        for i in 0..<result.count {
            XCTAssertGreaterThanOrEqual(result[i], 0.0, "VQT magnitudes should be non-negative")
            XCTAssertFalse(result[i].isNaN, "VQT values should not be NaN")
        }
    }

    func testVQTDifferentGammaValues() {
        // Different gamma values should produce different results
        let signal = makeSineSignal(frequency: 440.0, sr: 22050, duration: 1.0)
        let vqt0 = CQT.vqt(signal: signal, fMin: 32.70, fMax: 8000.0,
                            binsPerOctave: 12, gamma: 0.0)
        let vqt20 = CQT.vqt(signal: signal, fMin: 32.70, fMax: 8000.0,
                             binsPerOctave: 12, gamma: 20.0)

        // Same shape
        XCTAssertEqual(vqt0.shape, vqt20.shape, "Different gammas should produce same shape")

        // But different values (at least for some elements)
        var anyDiff = false
        for i in 0..<min(vqt0.count, vqt20.count) {
            if abs(vqt0[i] - vqt20[i]) > 1e-5 {
                anyDiff = true
                break
            }
        }
        XCTAssertTrue(anyDiff, "gamma=0 and gamma=20 should produce different values")
    }

    // MARK: - Hybrid CQT Tests

    func testHybridCQTOutputShape() {
        let signal = makeSineSignal(frequency: 440.0, sr: 22050, duration: 1.0)
        let result = CQT.hybridCQT(signal: signal, fMin: 32.70, fMax: 8000.0,
                                     binsPerOctave: 12)

        XCTAssertEqual(result.shape.count, 2, "Hybrid CQT should be 2D")
        XCTAssertGreaterThan(result.shape[0], 0, "Should have frequency bins")
        XCTAssertGreaterThan(result.shape[1], 0, "Should have frames")
    }

    func testHybridCQTValuesNonNegative() {
        let signal = makeSineSignal(frequency: 440.0, sr: 22050, duration: 1.0)
        let result = CQT.hybridCQT(signal: signal, fMin: 32.70, fMax: 8000.0,
                                     binsPerOctave: 12)

        for i in 0..<result.count {
            XCTAssertGreaterThanOrEqual(result[i], 0.0, "Hybrid CQT magnitudes should be non-negative")
            XCTAssertFalse(result[i].isNaN, "Hybrid CQT values should not be NaN")
        }
    }

    func testHybridCQTValuesFinite() {
        let signal = makeSineSignal(frequency: 440.0, sr: 22050, duration: 1.0)
        let result = CQT.hybridCQT(signal: signal, fMin: 32.70, fMax: 8000.0,
                                     binsPerOctave: 12)

        for i in 0..<result.count {
            XCTAssertFalse(result[i].isInfinite, "Hybrid CQT should not contain infinities")
        }
    }

    func testHybridCQTHasMoreBinsThanCQTAlone() {
        // Hybrid CQT should generally have more total bins than a pure CQT
        // because it adds linear STFT bins above the transition frequency
        let signal = makeSineSignal(frequency: 440.0, sr: 22050, duration: 1.0)
        let hybrid = CQT.hybridCQT(signal: signal, fMin: 32.70, fMax: 8000.0, binsPerOctave: 12)
        let cqt = CQT.compute(signal: signal, fMin: 32.70, fMax: 8000.0, binsPerOctave: 12)

        // Hybrid should have at least as many bins (CQT bins + linear bins)
        XCTAssertGreaterThanOrEqual(hybrid.shape[0], cqt.shape[0],
                                    "Hybrid CQT should have >= CQT bins total")
    }

    // MARK: - Edge Cases

    func testCQTEmptySignal() {
        let signal = Signal(data: [Float](), shape: [0], sampleRate: 22050)
        let result = CQT.compute(signal: signal, fMin: 32.70, fMax: 8000.0)
        // Should handle gracefully
        XCTAssertEqual(result.shape.count, 2)
    }

    func testCQTHighBinsPerOctave() {
        // 36 bins per octave (quarter-tone resolution)
        let signal = makeSineSignal(frequency: 440.0, sr: 22050, duration: 1.0)
        let result = CQT.compute(signal: signal, fMin: 100.0, fMax: 4000.0, binsPerOctave: 36)

        XCTAssertEqual(result.shape.count, 2)
        XCTAssertGreaterThan(result.shape[0], 0)
    }
}
