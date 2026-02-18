import XCTest
@testable import MetalMomCore

final class ChromaVariantsTests: XCTestCase {

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

    // MARK: - CQT Chroma Shape Tests

    func testCQTChromaOutputShape() {
        let signal = makeSineSignal()
        let result = Chroma.cqt(signal: signal)
        XCTAssertEqual(result.shape.count, 2, "CQT chroma should be 2D")
        XCTAssertEqual(result.shape[0], 12, "Default should have 12 chroma bins")
        XCTAssertGreaterThan(result.shape[1], 0, "Should have at least 1 frame")
    }

    func testCQTChromaCustomNChroma() {
        let signal = makeSineSignal()
        let result = Chroma.cqt(signal: signal, nChroma: 24)
        XCTAssertEqual(result.shape[0], 24, "Should have 24 chroma bins")
    }

    // MARK: - CQT Chroma Value Tests

    func testCQTChromaValuesNonNegative() {
        let signal = makeSineSignal()
        let result = Chroma.cqt(signal: signal)
        for i in 0..<result.count {
            XCTAssertGreaterThanOrEqual(result[i], 0.0,
                                        "CQT chroma values should be non-negative")
        }
    }

    func testCQTChromaValuesFinite() {
        let signal = makeSineSignal()
        let result = Chroma.cqt(signal: signal)
        for i in 0..<result.count {
            XCTAssertFalse(result[i].isNaN, "CQT chroma value should not be NaN")
            XCTAssertFalse(result[i].isInfinite, "CQT chroma value should not be infinite")
        }
    }

    func testCQTChromaWithL2Norm() {
        let signal = makeSineSignal()
        let result = Chroma.cqt(signal: signal, norm: 2.0)
        let nChroma = result.shape[0]
        let nFrames = result.shape[1]

        for f in 0..<nFrames {
            var l2Sq: Float = 0
            for c in 0..<nChroma {
                let val = result[c * nFrames + f]
                l2Sq += val * val
            }
            let l2 = sqrtf(l2Sq)
            if l2 > 1e-8 {
                XCTAssertEqual(l2, 1.0, accuracy: 1e-4,
                              "L2 norm of frame \(f) should be 1.0, got \(l2)")
            }
        }
    }

    // MARK: - CQT Chroma Pitch Detection

    func testCQTChroma440HzPeaksAtA() {
        // 440 Hz = A4, which is chroma index 9 (C=0, C#=1, ..., A=9)
        let signal = makeSineSignal(frequency: 440.0)
        let result = Chroma.cqt(signal: signal)

        let nChroma = result.shape[0]
        let nFrames = result.shape[1]

        var avgEnergy = [Float](repeating: 0, count: nChroma)
        for c in 0..<nChroma {
            for f in 0..<nFrames {
                avgEnergy[c] += result[c * nFrames + f]
            }
            avgEnergy[c] /= Float(nFrames)
        }

        let peakBin = avgEnergy.enumerated().max(by: { $0.element < $1.element })!.offset
        XCTAssertEqual(peakBin, 9,
                       "440 Hz should peak at chroma bin A (index 9), got \(peakBin)")
    }

    // MARK: - CENS Shape Tests

    func testCENSOutputShape() {
        let signal = makeSineSignal()
        let result = Chroma.cens(signal: signal)
        XCTAssertEqual(result.shape.count, 2, "CENS should be 2D")
        XCTAssertEqual(result.shape[0], 12, "Default should have 12 chroma bins")
        XCTAssertGreaterThan(result.shape[1], 0, "Should have at least 1 frame")
    }

    // MARK: - CENS Value Tests

    func testCENSValuesNonNegative() {
        let signal = makeSineSignal()
        let result = Chroma.cens(signal: signal)
        for i in 0..<result.count {
            XCTAssertGreaterThanOrEqual(result[i], 0.0,
                                        "CENS values should be non-negative")
        }
    }

    func testCENSValuesBounded() {
        // After L2 normalization, individual values should be <= 1.0
        let signal = makeSineSignal()
        let result = Chroma.cens(signal: signal)
        for i in 0..<result.count {
            XCTAssertLessThanOrEqual(result[i], 1.0 + 1e-6,
                                     "CENS values should be bounded by 1.0")
        }
    }

    func testCENSSmootherThanRawChroma() {
        // CENS should be smoother (less frame-to-frame variation) than raw CQT chroma
        let signal = makeSineSignal(duration: 2.0)

        let raw = Chroma.cqt(signal: signal)
        let smoothed = Chroma.cens(signal: signal)

        let rawFrames = raw.shape[1]
        let smoothedFrames = smoothed.shape[1]

        // Compute average absolute frame-to-frame difference for first chroma bin
        var rawVariation: Float = 0
        for f in 1..<rawFrames {
            rawVariation += abs(raw[0 * rawFrames + f] - raw[0 * rawFrames + (f - 1)])
        }
        rawVariation /= Float(max(1, rawFrames - 1))

        var smoothedVariation: Float = 0
        for f in 1..<smoothedFrames {
            smoothedVariation += abs(smoothed[0 * smoothedFrames + f] - smoothed[0 * smoothedFrames + (f - 1)])
        }
        smoothedVariation /= Float(max(1, smoothedFrames - 1))

        // CENS should have less or equal variation (quantization + smoothing)
        XCTAssertLessThanOrEqual(smoothedVariation, rawVariation + 1e-3,
                                 "CENS should be smoother than raw CQT chroma")
    }

    // MARK: - VQT Chroma Tests

    func testVQTChromaOutputShape() {
        let signal = makeSineSignal()
        let result = Chroma.vqt(signal: signal)
        XCTAssertEqual(result.shape.count, 2, "VQT chroma should be 2D")
        XCTAssertEqual(result.shape[0], 12, "Default should have 12 chroma bins")
        XCTAssertGreaterThan(result.shape[1], 0, "Should have at least 1 frame")
    }

    func testVQTChromaValuesNonNegative() {
        let signal = makeSineSignal()
        let result = Chroma.vqt(signal: signal)
        for i in 0..<result.count {
            XCTAssertGreaterThanOrEqual(result[i], 0.0,
                                        "VQT chroma values should be non-negative")
        }
    }

    func testVQTChromaWithGamma() {
        // VQT with gamma>0 should still produce valid output
        let signal = makeSineSignal()
        let result = Chroma.vqt(signal: signal, gamma: 24.0)
        XCTAssertEqual(result.shape[0], 12)
        XCTAssertGreaterThan(result.shape[1], 0)
        for i in 0..<result.count {
            XCTAssertFalse(result[i].isNaN, "VQT chroma value should not be NaN")
        }
    }

    // MARK: - Deep Chroma Tests

    func testDeepChromaOutputShape() {
        let signal = makeSineSignal()
        let result = Chroma.deep(signal: signal)
        XCTAssertEqual(result.shape.count, 2, "Deep chroma should be 2D")
        XCTAssertEqual(result.shape[0], 12, "Default should have 12 chroma bins")
        XCTAssertGreaterThan(result.shape[1], 0, "Should have at least 1 frame")
    }

    func testDeepChromaIsL2Normalized() {
        // Deep chroma uses L2 normalization
        let signal = makeSineSignal()
        let result = Chroma.deep(signal: signal)
        let nChroma = result.shape[0]
        let nFrames = result.shape[1]

        for f in 0..<nFrames {
            var l2Sq: Float = 0
            for c in 0..<nChroma {
                let val = result[c * nFrames + f]
                l2Sq += val * val
            }
            let l2 = sqrtf(l2Sq)
            if l2 > 1e-8 {
                XCTAssertEqual(l2, 1.0, accuracy: 1e-4,
                              "Deep chroma frame \(f) should have unit L2 norm, got \(l2)")
            }
        }
    }

    // MARK: - Silence Tests

    func testCQTChromaSilence() {
        let signal = Signal(data: [Float](repeating: 0, count: 22050), sampleRate: 22050)
        let result = Chroma.cqt(signal: signal)
        for i in 0..<result.count {
            XCTAssertEqual(result[i], 0.0, accuracy: 1e-8,
                          "CQT chroma of silence should be zero")
        }
    }

    func testCENSSilence() {
        let signal = Signal(data: [Float](repeating: 0, count: 22050), sampleRate: 22050)
        let result = Chroma.cens(signal: signal)
        for i in 0..<result.count {
            XCTAssertEqual(result[i], 0.0, accuracy: 1e-8,
                          "CENS of silence should be zero")
        }
    }

    // MARK: - Frame Count Consistency

    func testCQTChromaAndCENSSameFrameCount() {
        let signal = makeSineSignal()
        let cqtResult = Chroma.cqt(signal: signal)
        let censResult = Chroma.cens(signal: signal)

        XCTAssertEqual(cqtResult.shape[1], censResult.shape[1],
                       "CQT chroma and CENS should have same frame count")
    }
}
