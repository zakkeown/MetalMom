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

    // MARK: - Logarithmic Filterbank Tests

    func testLogarithmicFilterbankShape() {
        // With nFFT=8192, sr=44100, 24 bands/octave, fmin=65, fmax=2100:
        // madmom's algorithm produces ~105 unique bins after deduplication.
        let fb = FilterBank.logarithmic(nFFT: 8192, sampleRate: 44100)
        XCTAssertEqual(fb.shape.count, 2, "Log filterbank should be 2D")
        XCTAssertEqual(fb.shape[1], 8192 / 2 + 1, "Should have nFFT/2+1 frequency bins")
        // The exact count depends on FFT resolution and deduplication; should be ~105
        XCTAssertGreaterThanOrEqual(fb.shape[0], 90, "Should have at least 90 bins")
        XCTAssertLessThanOrEqual(fb.shape[0], 120, "Should have at most 120 bins")
    }

    func testLogarithmicFilterbankValuesNonNegative() {
        let fb = FilterBank.logarithmic(nFFT: 8192, sampleRate: 44100)
        for i in 0..<fb.count {
            XCTAssertGreaterThanOrEqual(fb[i], 0.0,
                                        "Log filterbank values should be non-negative")
        }
    }

    func testLogarithmicFilterbankRowsNonZero() {
        let fb = FilterBank.logarithmic(nFFT: 8192, sampleRate: 44100)
        let nBins = fb.shape[0]
        let nFreqs = fb.shape[1]

        for b in 0..<nBins {
            var rowSum: Float = 0
            for k in 0..<nFreqs {
                rowSum += fb[b * nFreqs + k]
            }
            XCTAssertGreaterThan(rowSum, 0, "Filter band \(b) should have non-zero weights")
        }
    }

    func testLogarithmicFilterbankProduces105BinsForDeepChroma() {
        // The deep chroma DNN expects exactly 105 * 15 = 1575 input features.
        // This means the filterbank must produce exactly 105 bins with the
        // standard deep chroma parameters (nFFT=8192, sr=44100, 24 bands/oct, 65-2100 Hz).
        let fb = FilterBank.logarithmic(nFFT: 8192, sampleRate: 44100)
        XCTAssertEqual(fb.shape[0], 105,
                       "Deep chroma filterbank should produce exactly 105 bins, got \(fb.shape[0])")
    }

    // MARK: - Deep Chroma Tests

    /// Resolve the models/converted/chroma directory via #filePath.
    private static var chromaModelDir: URL? {
        // #filePath → .../Tests/MetalMomTests/ChromaTests.swift
        let thisFile = URL(fileURLWithPath: #filePath)
        let projectRoot = thisFile
            .deletingLastPathComponent()   // MetalMomTests/
            .deletingLastPathComponent()   // Tests/
            .deletingLastPathComponent()   // project root
        let modelDir = projectRoot.appendingPathComponent("models/converted/chroma")
        let modelFile = modelDir.appendingPathComponent("chroma_dnn.mlmodel")
        guard FileManager.default.fileExists(atPath: modelFile.path) else {
            return nil
        }
        return modelDir
    }

    func testDeepChromaFallbackShape() {
        // Without a model, deep() should fall back to CQT and still return correct shape.
        let signal = makeSineSignal(frequency: 440.0, sr: 44100, duration: 1.0)
        let result = Chroma.deep(signal: signal, sr: 44100)

        XCTAssertEqual(result.shape.count, 2, "Deep chroma should be 2D")
        XCTAssertEqual(result.shape[0], 12, "Should have 12 chroma bins")
        XCTAssertGreaterThan(result.shape[1], 0, "Should have at least 1 frame")
    }

    func testDeepChromaFallbackValuesFinite() {
        let signal = makeSineSignal(frequency: 440.0, sr: 44100, duration: 1.0)
        let result = Chroma.deep(signal: signal, sr: 44100)

        for i in 0..<result.count {
            XCTAssertFalse(result[i].isNaN, "Deep chroma value at \(i) should not be NaN")
            XCTAssertFalse(result[i].isInfinite, "Deep chroma value at \(i) should not be infinite")
        }
    }

    func testDeepChromaDNNOutputShape() throws {
        guard let modelDir = Self.chromaModelDir else {
            throw XCTSkip("chroma_dnn.mlmodel not found — skipping DNN test")
        }

        let signal = makeSineSignal(frequency: 440.0, sr: 44100, duration: 1.0)
        let result = Chroma.deep(
            signal: signal,
            sr: 44100,
            modelsDirectory: modelDir
        )

        XCTAssertEqual(result.shape.count, 2, "Deep chroma should be 2D")
        XCTAssertEqual(result.shape[0], 12, "Should have 12 chroma bins")
        XCTAssertGreaterThan(result.shape[1], 0, "Should have at least 1 frame")

        // ~10 fps for 1 second => expect roughly 10 frames (with center padding)
        let expectedFrames = result.shape[1]
        XCTAssertGreaterThanOrEqual(expectedFrames, 8, "Should have roughly 10 frames for 1s at 10fps")
        XCTAssertLessThanOrEqual(expectedFrames, 15, "Should not have far more than 10 frames for 1s")
    }

    func testDeepChromaDNNValuesInRange() throws {
        guard let modelDir = Self.chromaModelDir else {
            throw XCTSkip("chroma_dnn.mlmodel not found — skipping DNN test")
        }

        // First, verify the raw model works with a known input
        let modelURL = modelDir.appendingPathComponent("chroma_dnn.mlmodel")
        let engine = try InferenceEngine(sourceModelURL: modelURL)
        let testInput = Signal(data: [Float](repeating: 0.0, count: 1575),
                               shape: [1575], sampleRate: 44100)
        let testOutput = try engine.predict(input: testInput)
        XCTAssertEqual(testOutput.count, 12, "Model should output 12 values")
        testOutput.withUnsafeBufferPointer { buf in
            for i in 0..<12 {
                XCTAssertGreaterThanOrEqual(buf[i], 0.0,
                    "Sigmoid output should be >= 0, got \(buf[i]) at index \(i)")
                XCTAssertLessThanOrEqual(buf[i], 1.0,
                    "Sigmoid output should be <= 1, got \(buf[i]) at index \(i)")
            }
        }

        // Now test the full pipeline
        let signal = makeSineSignal(frequency: 440.0, sr: 44100, duration: 1.0)
        let result = Chroma.deep(
            signal: signal,
            sr: 44100,
            modelsDirectory: modelDir
        )

        for i in 0..<result.count {
            XCTAssertFalse(result[i].isNaN, "DNN chroma value at \(i) should not be NaN")
            XCTAssertFalse(result[i].isInfinite, "DNN chroma value at \(i) should not be infinite")
            // DNN output (sigmoid final layer) should be in [0, 1]
            XCTAssertGreaterThanOrEqual(result[i], -0.01,
                "DNN chroma value at \(i) should be >= -0.01, got \(result[i])")
            XCTAssertLessThanOrEqual(result[i], 1.01,
                "DNN chroma value at \(i) should be <= 1.01, got \(result[i])")
        }
    }

    func testDeepChromaDNNDiffersFromCQT() throws {
        guard let modelDir = Self.chromaModelDir else {
            throw XCTSkip("chroma_dnn.mlmodel not found — skipping DNN test")
        }

        let signal = makeSineSignal(frequency: 440.0, sr: 44100, duration: 1.0)

        // DNN deep chroma
        let dnnResult = Chroma.deep(
            signal: signal,
            sr: 44100,
            modelsDirectory: modelDir
        )

        // CQT fallback (no model dir → falls back)
        let cqtResult = Chroma.deep(
            signal: signal,
            sr: 44100
        )

        // They should differ — either in shape or values.
        // Shape may differ because the DNN pipeline uses different STFT params.
        let shapesMatch = dnnResult.shape == cqtResult.shape

        if shapesMatch {
            // If shapes happen to match, values must differ
            var sumDiff: Float = 0
            let count = min(dnnResult.count, cqtResult.count)
            for i in 0..<count {
                sumDiff += abs(dnnResult[i] - cqtResult[i])
            }
            XCTAssertGreaterThan(sumDiff, 0.01,
                "DNN deep chroma should produce different values than CQT fallback")
        }
        // If shapes don't match, that already proves the DNN is running a different pipeline.
    }
}
