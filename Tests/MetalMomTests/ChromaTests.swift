import XCTest
@testable import MetalMomCore

final class ChromaTests: XCTestCase {

    // MARK: - Helpers

    /// 1 second of 440 Hz sine at 22050 Hz.
    private func makeSineSignal(frequency: Float = 440.0, sr: Int = 22050,
                                 duration: Float = 1.0) -> Signal {
        let count = Int(Float(sr) * duration)
        let data = (0..<count).map { i in
            sinf(2.0 * .pi * frequency * Float(i) / Float(sr))
        }
        return Signal(data: data, sampleRate: sr)
    }

    // MARK: - Filterbank Shape Tests

    func testChromaFilterbankShape() {
        // Default: nChroma=12, nFFT=2048 -> shape [12, 1025]
        let fb = Chroma.chromaFilterbank(sr: 22050, nFFT: 2048)
        XCTAssertEqual(fb.shape.count, 2, "Chroma filterbank should be 2D")
        XCTAssertEqual(fb.shape[0], 12, "Should have 12 chroma bins")
        XCTAssertEqual(fb.shape[1], 1025, "Should have nFFT/2+1 frequency bins")
    }

    func testChromaFilterbankCustomNChroma() {
        let fb = Chroma.chromaFilterbank(sr: 22050, nFFT: 2048, nChroma: 24)
        XCTAssertEqual(fb.shape[0], 24, "Should have 24 chroma bins")
        XCTAssertEqual(fb.shape[1], 1025)
    }

    func testChromaFilterbankSmallFFT() {
        let fb = Chroma.chromaFilterbank(sr: 22050, nFFT: 1024)
        XCTAssertEqual(fb.shape[0], 12)
        XCTAssertEqual(fb.shape[1], 513, "Should have 1024/2+1=513 frequency bins")
    }

    // MARK: - Filterbank Value Tests

    func testChromaFilterbankValuesNonNegative() {
        let fb = Chroma.chromaFilterbank(sr: 22050, nFFT: 2048)
        for i in 0..<fb.count {
            XCTAssertGreaterThanOrEqual(fb[i], 0.0, "Chroma filterbank values should be non-negative")
        }
    }

    func testChromaFilterbankRowsNonZero() {
        // Each chroma bin should have at least some non-zero weights
        let fb = Chroma.chromaFilterbank(sr: 22050, nFFT: 2048)
        let nChroma = fb.shape[0]
        let nFreqs = fb.shape[1]

        for c in 0..<nChroma {
            var rowSum: Float = 0
            for k in 0..<nFreqs {
                rowSum += fb[c * nFreqs + k]
            }
            XCTAssertGreaterThan(rowSum, 0, "Chroma bin \(c) should have non-zero weights")
        }
    }

    func testChromaFilterbankDCBinNearZero() {
        // DC bin (k=0, frequency=0) should have near-zero weight
        // (Gaussian tails produce tiny non-zero values, matching librosa)
        let fb = Chroma.chromaFilterbank(sr: 22050, nFFT: 2048)
        let nFreqs = fb.shape[1]

        for c in 0..<12 {
            XCTAssertLessThan(fb[c * nFreqs + 0], 1e-3,
                             "DC bin should have near-zero weight")
        }
    }

    // MARK: - Chroma STFT Shape Tests

    func testDefaultShape() {
        // Default: nChroma=12, nFFT=2048, hopLength=512
        let signal = makeSineSignal()
        let result = Chroma.stft(signal: signal)

        XCTAssertEqual(result.shape.count, 2, "Chroma STFT should be 2D")
        XCTAssertEqual(result.shape[0], 12, "Default nChroma should be 12")
        XCTAssertGreaterThan(result.shape[1], 0, "Should have at least 1 frame")
    }

    func testCustomNChroma() {
        let signal = makeSineSignal()
        let result = Chroma.stft(signal: signal, nChroma: 24)
        XCTAssertEqual(result.shape[0], 24, "Should have 24 chroma bins")
    }

    func testCustomFFTSize() {
        let signal = makeSineSignal()
        let result = Chroma.stft(signal: signal, nFFT: 1024)
        XCTAssertEqual(result.shape[0], 12)
        XCTAssertGreaterThan(result.shape[1], 0)
    }

    func testFrameCountMatchesSTFT() {
        // Chroma frame count should match the STFT frame count
        let signal = makeSineSignal()
        let stftMag = STFT.compute(signal: signal)
        let chroma = Chroma.stft(signal: signal)

        XCTAssertEqual(chroma.shape[1], stftMag.shape[1],
                       "Chroma frame count should match STFT frame count")
    }

    // MARK: - Chroma STFT Value Tests

    func testValuesAreFinite() {
        let signal = makeSineSignal()
        let result = Chroma.stft(signal: signal)

        for i in 0..<result.count {
            XCTAssertFalse(result[i].isNaN, "Chroma value at \(i) should not be NaN")
            XCTAssertFalse(result[i].isInfinite, "Chroma value at \(i) should not be infinite")
        }
    }

    func testValuesNonNegative() {
        // Chroma from power spectrogram should be non-negative
        let signal = makeSineSignal()
        let result = Chroma.stft(signal: signal)

        for i in 0..<result.count {
            XCTAssertGreaterThanOrEqual(result[i], 0.0,
                                        "Chroma values should be non-negative")
        }
    }

    func test440HzPeaksAtA() {
        // 440 Hz = A4, which is chroma index 9 (C=0, C#=1, ..., A=9, A#=10, B=11)
        let signal = makeSineSignal(frequency: 440.0)
        let result = Chroma.stft(signal: signal)

        let nChroma = result.shape[0]
        let nFrames = result.shape[1]

        // Average energy per chroma bin across all frames
        var avgEnergy = [Float](repeating: 0, count: nChroma)
        for c in 0..<nChroma {
            for f in 0..<nFrames {
                avgEnergy[c] += result[c * nFrames + f]
            }
            avgEnergy[c] /= Float(nFrames)
        }

        // Find the peak chroma bin
        let peakBin = avgEnergy.enumerated().max(by: { $0.element < $1.element })!.offset
        XCTAssertEqual(peakBin, 9, "440 Hz should peak at chroma bin A (index 9), got \(peakBin)")
    }

    func test261HzPeaksAtC() {
        // 261.63 Hz = C4, which is chroma index 0
        let signal = makeSineSignal(frequency: 261.63)
        let result = Chroma.stft(signal: signal)

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
        XCTAssertEqual(peakBin, 0, "261.63 Hz (C4) should peak at chroma bin C (index 0), got \(peakBin)")
    }

    // MARK: - Silence

    func testSilenceProducesZeros() {
        let signal = Signal(data: [Float](repeating: 0, count: 22050), sampleRate: 22050)
        let result = Chroma.stft(signal: signal)

        for i in 0..<result.count {
            XCTAssertEqual(result[i], 0.0, accuracy: 1e-10,
                          "Chroma of silence should be zero")
        }
    }

    // MARK: - Normalization

    func testL2Normalization() {
        // With norm=2.0, each frame should have unit L2 norm (or zero for silent frames)
        let signal = makeSineSignal()
        let result = Chroma.stft(signal: signal, norm: 2.0)

        let nChroma = result.shape[0]
        let nFrames = result.shape[1]

        for f in 0..<nFrames {
            var l2: Float = 0
            for c in 0..<nChroma {
                let val = result[c * nFrames + f]
                l2 += val * val
            }
            l2 = sqrtf(l2)
            if l2 > 1e-10 {
                XCTAssertEqual(l2, 1.0, accuracy: 1e-5,
                              "L2 norm of frame \(f) should be 1.0, got \(l2)")
            }
        }
    }

    // MARK: - Sample rate override

    func testSampleRateOverride() {
        let signal = Signal(data: [Float](repeating: 0.1, count: 16000), sampleRate: 16000)
        let result = Chroma.stft(signal: signal, sr: 16000, nFFT: 1024)

        XCTAssertEqual(result.shape[0], 12)
        XCTAssertGreaterThan(result.shape[1], 0)
    }
}
