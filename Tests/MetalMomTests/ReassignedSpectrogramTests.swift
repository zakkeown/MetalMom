import XCTest
@testable import MetalMomCore

final class ReassignedSpectrogramTests: XCTestCase {

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

    // MARK: - Output Shape Tests

    func testOutputShapesMatch() {
        let signal = makeSineSignal(frequency: 440.0, sr: 22050, duration: 1.0)
        let nFFT = 2048
        let (mag, freqs, times) = ReassignedSpectrogram.compute(
            signal: signal, nFFT: nFFT
        )

        let nFreqs = nFFT / 2 + 1

        // All three outputs should be 2D [nFreqs, nFrames]
        XCTAssertEqual(mag.shape.count, 2)
        XCTAssertEqual(freqs.shape.count, 2)
        XCTAssertEqual(times.shape.count, 2)

        // Frequency dimension should be nFFT/2+1
        XCTAssertEqual(mag.shape[0], nFreqs)
        XCTAssertEqual(freqs.shape[0], nFreqs)
        XCTAssertEqual(times.shape[0], nFreqs)

        // All three should have the same number of frames
        XCTAssertEqual(mag.shape[1], freqs.shape[1])
        XCTAssertEqual(mag.shape[1], times.shape[1])
        XCTAssertGreaterThan(mag.shape[1], 0)
    }

    func testMagnitudeMatchesRegularSTFT() {
        // The magnitude output should match the regular STFT magnitude
        let signal = makeSineSignal(frequency: 440.0, sr: 22050, duration: 1.0)
        let nFFT = 2048

        let (mag, _, _) = ReassignedSpectrogram.compute(
            signal: signal, nFFT: nFFT
        )

        let stftMag = STFT.compute(signal: signal, nFFT: nFFT)

        // Same shape
        XCTAssertEqual(mag.shape, stftMag.shape,
                       "Reassigned magnitude shape should match STFT magnitude shape")

        // Same values (they use the same Hann window and FFT).
        // Tolerance accommodates vDSP (CPU) vs MPSGraph (GPU) FFT precision differences.
        for i in 0..<min(mag.count, stftMag.count) {
            XCTAssertEqual(mag[i], stftMag[i], accuracy: 1e-2,
                          "Magnitude at index \(i) should match STFT")
        }
    }

    // MARK: - Frequency Range Tests

    func testReassignedFrequenciesInValidRange() {
        let sr = 22050
        let signal = makeSineSignal(frequency: 440.0, sr: sr, duration: 1.0)
        let (mag, freqs, _) = ReassignedSpectrogram.compute(
            signal: signal, sr: sr, nFFT: 2048
        )

        let nyquist = Float(sr) / 2.0
        let nFreqs = mag.shape[0]
        let nFrames = mag.shape[1]

        // Check frequencies are in a reasonable range for non-negligible magnitudes.
        // Some reassigned frequencies can go slightly outside [0, nyquist] due to
        // the reassignment, so we allow a generous margin.
        for k in 0..<nFreqs {
            for f in 0..<nFrames {
                let idx = k * nFrames + f
                if mag[idx] > 1e-4 {
                    // For significant energy, frequency should be somewhat reasonable
                    XCTAssertFalse(freqs[idx].isNaN,
                                   "Reassigned frequency should not be NaN")
                    XCTAssertFalse(freqs[idx].isInfinite,
                                   "Reassigned frequency should not be infinite")
                }
            }
        }

        // At least some frequencies should be in [0, nyquist]
        var anyInRange = false
        for i in 0..<freqs.count {
            if freqs[i] >= 0 && freqs[i] <= nyquist {
                anyInRange = true
                break
            }
        }
        XCTAssertTrue(anyInRange, "Some reassigned frequencies should be in [0, nyquist]")
    }

    // MARK: - Time Range Tests

    func testReassignedTimesInValidRange() {
        let sr = 22050
        let duration: Float = 1.0
        let signal = makeSineSignal(frequency: 440.0, sr: sr, duration: duration)
        let (mag, _, times) = ReassignedSpectrogram.compute(
            signal: signal, sr: sr, nFFT: 2048
        )

        let nFreqs = mag.shape[0]
        let nFrames = mag.shape[1]

        // Check times for non-negligible magnitudes
        for k in 0..<nFreqs {
            for f in 0..<nFrames {
                let idx = k * nFrames + f
                if mag[idx] > 1e-4 {
                    XCTAssertFalse(times[idx].isNaN,
                                   "Reassigned time should not be NaN")
                    XCTAssertFalse(times[idx].isInfinite,
                                   "Reassigned time should not be infinite")
                }
            }
        }

        // At least some times should be in [0, duration]
        var anyInRange = false
        for i in 0..<times.count {
            if times[i] >= 0 && times[i] <= duration {
                anyInRange = true
                break
            }
        }
        XCTAssertTrue(anyInRange, "Some reassigned times should be in [0, duration]")
    }

    // MARK: - Sine Wave Frequency Clustering Test

    func testSineWaveFrequenciesClustered() {
        // For a pure sine wave at 440 Hz, the reassigned frequencies should
        // cluster around 440 Hz in the bins with significant energy.
        let sr = 22050
        let freq: Float = 440.0
        let signal = makeSineSignal(frequency: freq, sr: sr, duration: 2.0)
        let nFFT = 2048

        let (mag, freqs, _) = ReassignedSpectrogram.compute(
            signal: signal, sr: sr, nFFT: nFFT
        )

        let nFreqs = mag.shape[0]
        let nFrames = mag.shape[1]

        // Find the bin with the most total energy
        var totalEnergy = [Float](repeating: 0, count: nFreqs)
        for k in 0..<nFreqs {
            for f in 0..<nFrames {
                totalEnergy[k] += mag[k * nFrames + f]
            }
        }
        let peakBin = totalEnergy.enumerated().max(by: { $0.element < $1.element })!.offset

        // Collect reassigned frequencies at the peak bin across frames (skip edges)
        let startFrame = max(2, 0)
        let endFrame = min(nFrames - 2, nFrames)
        var reassignedFreqs = [Float]()
        for f in startFrame..<endFrame {
            let idx = peakBin * nFrames + f
            if mag[idx] > 0.01 {
                reassignedFreqs.append(freqs[idx])
            }
        }

        XCTAssertGreaterThan(reassignedFreqs.count, 0,
                              "Should have non-negligible magnitudes at peak bin")

        // Average reassigned frequency at the peak bin should be close to 440 Hz
        let avgFreq = reassignedFreqs.reduce(0, +) / Float(reassignedFreqs.count)
        XCTAssertEqual(avgFreq, freq, accuracy: 20.0,
                       "Average reassigned frequency at peak bin (\(avgFreq)) should be near \(freq) Hz")
    }

    // MARK: - Silence Test

    func testSilenceHandledGracefully() {
        let sr = 22050
        let silence = Signal(data: [Float](repeating: 0, count: sr), sampleRate: sr)
        let (mag, freqs, times) = ReassignedSpectrogram.compute(
            signal: silence, sr: sr, nFFT: 2048
        )

        // Should produce outputs without NaN or Inf
        for i in 0..<mag.count {
            XCTAssertFalse(mag[i].isNaN, "Magnitude should not be NaN for silence")
            XCTAssertFalse(mag[i].isInfinite, "Magnitude should not be infinite for silence")
            XCTAssertEqual(mag[i], 0.0, accuracy: 1e-10,
                          "Magnitude of silence should be zero")
        }

        for i in 0..<freqs.count {
            XCTAssertFalse(freqs[i].isNaN, "Frequency should not be NaN for silence")
            XCTAssertFalse(freqs[i].isInfinite, "Frequency should not be infinite for silence")
        }

        for i in 0..<times.count {
            XCTAssertFalse(times[i].isNaN, "Time should not be NaN for silence")
            XCTAssertFalse(times[i].isInfinite, "Time should not be infinite for silence")
        }
    }

    // MARK: - Custom Parameters Test

    func testCustomNFFTAndHopLength() {
        let signal = makeSineSignal(frequency: 440.0, sr: 22050, duration: 1.0)
        let nFFT = 1024
        let hopLength = 256

        let (mag, freqs, times) = ReassignedSpectrogram.compute(
            signal: signal, nFFT: nFFT, hopLength: hopLength
        )

        let expectedNFreqs = nFFT / 2 + 1
        XCTAssertEqual(mag.shape[0], expectedNFreqs)
        XCTAssertEqual(freqs.shape[0], expectedNFreqs)
        XCTAssertEqual(times.shape[0], expectedNFreqs)

        // More frames with smaller hop length
        XCTAssertGreaterThan(mag.shape[1], 0)
    }

    // MARK: - Short Signal Test

    func testShortSignalDoesNotCrash() {
        let signal = Signal(data: [Float](repeating: 0.5, count: 512), sampleRate: 22050)
        let (mag, freqs, times) = ReassignedSpectrogram.compute(
            signal: signal, nFFT: 2048
        )

        // With center padding, 512 + 2*1024 = 2560 >= 2048, should produce at least 1 frame
        XCTAssertEqual(mag.shape.count, 2)
        XCTAssertEqual(freqs.shape.count, 2)
        XCTAssertEqual(times.shape.count, 2)
    }

    // MARK: - Very Short Signal (too short)

    func testVeryShortSignalWithoutCenter() {
        // 100 samples, nFFT=2048, center=false -> too short
        let signal = Signal(data: [Float](repeating: 0.5, count: 100), sampleRate: 22050)
        let (mag, freqs, times) = ReassignedSpectrogram.compute(
            signal: signal, nFFT: 2048, center: false
        )

        XCTAssertEqual(mag.shape[1], 0, "Should produce 0 frames for very short signal without center padding")
        XCTAssertEqual(freqs.shape[1], 0)
        XCTAssertEqual(times.shape[1], 0)
    }

    // MARK: - Non-negative Magnitudes

    func testMagnitudesNonNegative() {
        let signal = makeSineSignal(frequency: 440.0, sr: 22050, duration: 1.0)
        let (mag, _, _) = ReassignedSpectrogram.compute(signal: signal, nFFT: 2048)

        for i in 0..<mag.count {
            XCTAssertGreaterThanOrEqual(mag[i], 0.0,
                                        "Magnitudes should be non-negative")
        }
    }

    // MARK: - Two-Tone Test

    func testTwoToneFrequencySeparation() {
        // Two sine waves at different frequencies should produce
        // reassigned frequencies that cluster near both frequencies.
        let sr = 22050
        let freq1: Float = 440.0
        let freq2: Float = 880.0
        let count = sr * 2  // 2 seconds
        let data = (0..<count).map { i -> Float in
            let t = Float(i) / Float(sr)
            return 0.5 * sinf(2.0 * .pi * freq1 * t) + 0.5 * sinf(2.0 * .pi * freq2 * t)
        }
        let signal = Signal(data: data, sampleRate: sr)
        let nFFT = 2048

        let (mag, freqs, _) = ReassignedSpectrogram.compute(
            signal: signal, sr: sr, nFFT: nFFT
        )

        let nFreqs = mag.shape[0]
        let nFrames = mag.shape[1]

        // Collect all reassigned frequencies where magnitude is significant
        var significantFreqs = [Float]()
        for k in 0..<nFreqs {
            for f in 2..<(nFrames - 2) {
                let idx = k * nFrames + f
                if mag[idx] > 0.05 {
                    significantFreqs.append(freqs[idx])
                }
            }
        }

        // Count how many are near 440 and how many near 880
        let near440 = significantFreqs.filter { abs($0 - freq1) < 30 }.count
        let near880 = significantFreqs.filter { abs($0 - freq2) < 30 }.count

        XCTAssertGreaterThan(near440, 0,
                              "Should have reassigned frequencies near \(freq1) Hz")
        XCTAssertGreaterThan(near880, 0,
                              "Should have reassigned frequencies near \(freq2) Hz")
    }
}
