import XCTest
@testable import MetalMomCore

final class SpectralDescriptorsTests: XCTestCase {

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

    /// Compute magnitude spectrogram for a signal (convenience).
    private func makeSpectrogram(signal: Signal, nFFT: Int = 2048,
                                  hopLength: Int? = nil) -> Signal {
        return STFT.compute(signal: signal, nFFT: nFFT, hopLength: hopLength)
    }

    // MARK: - Centroid Tests

    func testCentroidShape() {
        let signal = makeSineSignal()
        let spec = makeSpectrogram(signal: signal)
        let result = SpectralDescriptors.centroid(spectrogram: spec, sr: 22050, nFFT: 2048)

        XCTAssertEqual(result.shape.count, 1, "Centroid should be 1D")
        XCTAssertEqual(result.shape[0], spec.shape[1],
                       "Centroid length should equal number of frames")
    }

    func testCentroidValuesFinite() {
        let signal = makeSineSignal()
        let spec = makeSpectrogram(signal: signal)
        let result = SpectralDescriptors.centroid(spectrogram: spec, sr: 22050, nFFT: 2048)

        for i in 0..<result.count {
            XCTAssertFalse(result[i].isNaN, "Centroid should not be NaN at index \(i)")
            XCTAssertFalse(result[i].isInfinite, "Centroid should not be infinite at index \(i)")
        }
    }

    func testCentroidNonNegative() {
        let signal = makeSineSignal()
        let spec = makeSpectrogram(signal: signal)
        let result = SpectralDescriptors.centroid(spectrogram: spec, sr: 22050, nFFT: 2048)

        for i in 0..<result.count {
            XCTAssertGreaterThanOrEqual(result[i], 0.0,
                                        "Centroid should be non-negative")
        }
    }

    func testCentroid440HzNearExpected() {
        // For a pure 440 Hz sine, centroid should be near 440 Hz
        let signal = makeSineSignal(frequency: 440.0)
        let spec = makeSpectrogram(signal: signal)
        let result = SpectralDescriptors.centroid(spectrogram: spec, sr: 22050, nFFT: 2048)

        let nFrames = result.shape[0]
        // Average centroid across frames (skip edge frames)
        var sum: Float = 0
        let start = 2
        let end = nFrames - 2
        for i in start..<end {
            sum += result[i]
        }
        let avgCentroid = sum / Float(end - start)

        // Should be within ~50 Hz of 440 Hz (windowing spreads energy)
        XCTAssertEqual(avgCentroid, 440.0, accuracy: 50.0,
                       "Centroid of 440 Hz sine should be near 440 Hz, got \(avgCentroid)")
    }

    func testCentroidSilenceIsZero() {
        let silence = Signal(data: [Float](repeating: 0, count: 22050), sampleRate: 22050)
        let spec = makeSpectrogram(signal: silence)
        let result = SpectralDescriptors.centroid(spectrogram: spec, sr: 22050, nFFT: 2048)

        for i in 0..<result.count {
            XCTAssertEqual(result[i], 0.0, accuracy: 1e-10,
                          "Centroid of silence should be 0")
        }
    }

    // MARK: - Bandwidth Tests

    func testBandwidthShape() {
        let signal = makeSineSignal()
        let spec = makeSpectrogram(signal: signal)
        let result = SpectralDescriptors.bandwidth(spectrogram: spec, sr: 22050, nFFT: 2048)

        XCTAssertEqual(result.shape.count, 1, "Bandwidth should be 1D")
        XCTAssertEqual(result.shape[0], spec.shape[1],
                       "Bandwidth length should equal number of frames")
    }

    func testBandwidthNonNegative() {
        let signal = makeSineSignal()
        let spec = makeSpectrogram(signal: signal)
        let result = SpectralDescriptors.bandwidth(spectrogram: spec, sr: 22050, nFFT: 2048)

        for i in 0..<result.count {
            XCTAssertGreaterThanOrEqual(result[i], 0.0,
                                        "Bandwidth should be non-negative")
        }
    }

    func testBandwidthPureToneIsNarrow() {
        // A pure sine wave should have narrow bandwidth (most energy at one frequency)
        let signal = makeSineSignal(frequency: 440.0)
        let spec = makeSpectrogram(signal: signal)
        let bw = SpectralDescriptors.bandwidth(spectrogram: spec, sr: 22050, nFFT: 2048)

        let nFrames = bw.shape[0]
        var sum: Float = 0
        let start = 2
        let end = nFrames - 2
        for i in start..<end {
            sum += bw[i]
        }
        let avgBW = sum / Float(end - start)

        // Pure tone bandwidth should be relatively small (< 500 Hz)
        XCTAssertLessThan(avgBW, 500.0,
                          "Pure tone bandwidth should be narrow, got \(avgBW)")
    }

    func testBandwidthSilenceIsZero() {
        let silence = Signal(data: [Float](repeating: 0, count: 22050), sampleRate: 22050)
        let spec = makeSpectrogram(signal: silence)
        let result = SpectralDescriptors.bandwidth(spectrogram: spec, sr: 22050, nFFT: 2048)

        for i in 0..<result.count {
            XCTAssertEqual(result[i], 0.0, accuracy: 1e-10,
                          "Bandwidth of silence should be 0")
        }
    }

    // MARK: - Contrast Tests

    func testContrastShape() {
        let signal = makeSineSignal()
        let spec = makeSpectrogram(signal: signal)
        let result = SpectralDescriptors.contrast(spectrogram: spec, sr: 22050, nFFT: 2048)

        // Default nBands=6 -> output shape [nBands+1, nFrames] = [7, nFrames]
        XCTAssertEqual(result.shape.count, 2, "Contrast should be 2D")
        XCTAssertEqual(result.shape[0], 7, "Default contrast should have 7 bands (nBands+1)")
        XCTAssertEqual(result.shape[1], spec.shape[1],
                       "Contrast frames should match spectrogram frames")
    }

    func testContrastCustomBands() {
        let signal = makeSineSignal()
        let spec = makeSpectrogram(signal: signal)
        let result = SpectralDescriptors.contrast(spectrogram: spec, sr: 22050, nFFT: 2048,
                                                   nBands: 4)
        XCTAssertEqual(result.shape[0], 5, "nBands=4 should produce 5 rows")
    }

    func testContrastValuesFinite() {
        let signal = makeSineSignal()
        let spec = makeSpectrogram(signal: signal)
        let result = SpectralDescriptors.contrast(spectrogram: spec, sr: 22050, nFFT: 2048)

        for i in 0..<result.count {
            XCTAssertFalse(result[i].isNaN, "Contrast should not be NaN")
            XCTAssertFalse(result[i].isInfinite, "Contrast should not be infinite")
        }
    }

    // MARK: - Rolloff Tests

    func testRolloffShape() {
        let signal = makeSineSignal()
        let spec = makeSpectrogram(signal: signal)
        let result = SpectralDescriptors.rolloff(spectrogram: spec, sr: 22050, nFFT: 2048)

        XCTAssertEqual(result.shape.count, 1, "Rolloff should be 1D")
        XCTAssertEqual(result.shape[0], spec.shape[1],
                       "Rolloff length should equal number of frames")
    }

    func testRolloffNonNegative() {
        let signal = makeSineSignal()
        let spec = makeSpectrogram(signal: signal)
        let result = SpectralDescriptors.rolloff(spectrogram: spec, sr: 22050, nFFT: 2048)

        for i in 0..<result.count {
            XCTAssertGreaterThanOrEqual(result[i], 0.0,
                                        "Rolloff should be non-negative")
        }
    }

    func testRolloffBelowNyquist() {
        let signal = makeSineSignal()
        let spec = makeSpectrogram(signal: signal)
        let result = SpectralDescriptors.rolloff(spectrogram: spec, sr: 22050, nFFT: 2048)

        let nyquist = Float(22050) / 2.0
        for i in 0..<result.count {
            XCTAssertLessThanOrEqual(result[i], nyquist + 1.0,
                                     "Rolloff should not exceed Nyquist")
        }
    }

    func testRolloff440HzNear440() {
        // For a pure 440 Hz sine, rolloff at 85% should be near 440 Hz
        let signal = makeSineSignal(frequency: 440.0)
        let spec = makeSpectrogram(signal: signal)
        let result = SpectralDescriptors.rolloff(spectrogram: spec, sr: 22050, nFFT: 2048)

        let nFrames = result.shape[0]
        var sum: Float = 0
        let start = 2
        let end = nFrames - 2
        for i in start..<end {
            sum += result[i]
        }
        let avgRolloff = sum / Float(end - start)

        // Rolloff of a pure tone should be near the tone's frequency (within a few bins)
        XCTAssertEqual(avgRolloff, 440.0, accuracy: 200.0,
                       "Rolloff of 440 Hz sine should be near 440 Hz, got \(avgRolloff)")
    }

    func testRolloffSilenceIsZero() {
        let silence = Signal(data: [Float](repeating: 0, count: 22050), sampleRate: 22050)
        let spec = makeSpectrogram(signal: silence)
        let result = SpectralDescriptors.rolloff(spectrogram: spec, sr: 22050, nFFT: 2048)

        for i in 0..<result.count {
            XCTAssertEqual(result[i], 0.0, accuracy: 1e-10,
                          "Rolloff of silence should be 0")
        }
    }

    // MARK: - Flatness Tests

    func testFlatnessShape() {
        let signal = makeSineSignal()
        let spec = makeSpectrogram(signal: signal)
        let result = SpectralDescriptors.flatness(spectrogram: spec)

        XCTAssertEqual(result.shape.count, 1, "Flatness should be 1D")
        XCTAssertEqual(result.shape[0], spec.shape[1],
                       "Flatness length should equal number of frames")
    }

    func testFlatnessRange() {
        // Flatness should be in [0, 1]
        let signal = makeSineSignal()
        let spec = makeSpectrogram(signal: signal)
        let result = SpectralDescriptors.flatness(spectrogram: spec)

        for i in 0..<result.count {
            XCTAssertGreaterThanOrEqual(result[i], 0.0,
                                        "Flatness should be >= 0")
            XCTAssertLessThanOrEqual(result[i], 1.0 + 1e-6,
                                     "Flatness should be <= 1")
        }
    }

    func testFlatnessPureToneIsLow() {
        // A pure sine wave should have low flatness (tonal, not noise-like)
        let signal = makeSineSignal(frequency: 440.0)
        let spec = makeSpectrogram(signal: signal)
        let result = SpectralDescriptors.flatness(spectrogram: spec)

        let nFrames = result.shape[0]
        var sum: Float = 0
        let start = 2
        let end = nFrames - 2
        for i in start..<end {
            sum += result[i]
        }
        let avgFlatness = sum / Float(end - start)

        // Pure tone: flatness should be close to 0 (very tonal)
        XCTAssertLessThan(avgFlatness, 0.1,
                          "Pure tone flatness should be low, got \(avgFlatness)")
    }

    func testFlatnessWhiteNoiseIsHigh() {
        // White noise should have high flatness (near 1.0)
        // Use a fixed seed via a simple LCG for reproducibility
        var rng: UInt64 = 42
        let count = 22050
        var noiseData = [Float](repeating: 0, count: count)
        for i in 0..<count {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            let val = Float(Int64(bitPattern: rng)) / Float(Int64.max)
            noiseData[i] = val
        }
        let noise = Signal(data: noiseData, sampleRate: 22050)
        let spec = makeSpectrogram(signal: noise)
        let result = SpectralDescriptors.flatness(spectrogram: spec)

        let nFrames = result.shape[0]
        var sum: Float = 0
        for i in 0..<nFrames {
            sum += result[i]
        }
        let avgFlatness = sum / Float(nFrames)

        // White noise flatness should be high (> 0.3, ideally close to 1)
        XCTAssertGreaterThan(avgFlatness, 0.3,
                             "White noise flatness should be high, got \(avgFlatness)")
    }

    func testFlatnessSilenceIsZero() {
        let silence = Signal(data: [Float](repeating: 0, count: 22050), sampleRate: 22050)
        let spec = makeSpectrogram(signal: silence)
        let result = SpectralDescriptors.flatness(spectrogram: spec)

        for i in 0..<result.count {
            XCTAssertEqual(result[i], 0.0, accuracy: 1e-10,
                          "Flatness of silence should be 0")
        }
    }

    // MARK: - Cross-Feature Consistency

    func testHigherFreqHasHigherCentroid() {
        // 1000 Hz signal should have higher centroid than 440 Hz
        let sig440 = makeSineSignal(frequency: 440.0)
        let sig1k = makeSineSignal(frequency: 1000.0)
        let spec440 = makeSpectrogram(signal: sig440)
        let spec1k = makeSpectrogram(signal: sig1k)

        let cent440 = SpectralDescriptors.centroid(spectrogram: spec440, sr: 22050, nFFT: 2048)
        let cent1k = SpectralDescriptors.centroid(spectrogram: spec1k, sr: 22050, nFFT: 2048)

        // Compare middle frames
        let mid = cent440.shape[0] / 2
        XCTAssertGreaterThan(cent1k[mid], cent440[mid],
                             "1000 Hz centroid should be higher than 440 Hz centroid")
    }
}
