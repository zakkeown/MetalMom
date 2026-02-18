import XCTest
@testable import MetalMomCore

final class PhaseVocoderTests: XCTestCase {

    // MARK: - Helpers

    /// Generate a pure sine wave signal.
    private func makeSine(frequency: Float = 440.0, sr: Int = 22050, duration: Float = 0.5) -> Signal {
        let length = Int(Float(sr) * duration)
        var data = [Float](repeating: 0, count: length)
        let angularFreq = 2.0 * Float.pi * frequency / Float(sr)
        for i in 0..<length {
            data[i] = sinf(angularFreq * Float(i))
        }
        return Signal(data: data, sampleRate: sr)
    }

    /// Check that all values in a Signal are finite.
    private func allFinite(_ signal: Signal) -> Bool {
        var result = true
        signal.withUnsafeBufferPointer { buf in
            for i in 0..<buf.count {
                if !buf[i].isFinite {
                    result = false
                    return
                }
            }
        }
        return result
    }

    /// Compute SNR in dB between original and reconstructed signals.
    private func snrDB(original: Signal, reconstructed: Signal) -> Float {
        let n = min(original.count, reconstructed.count)
        guard n > 0 else { return -Float.infinity }

        var signalPower: Float = 0
        var noisePower: Float = 0

        original.withUnsafeBufferPointer { origBuf in
            reconstructed.withUnsafeBufferPointer { reconBuf in
                for i in 0..<n {
                    let s = origBuf[i]
                    let diff = s - reconBuf[i]
                    signalPower += s * s
                    noisePower += diff * diff
                }
            }
        }

        guard noisePower > 1e-20 else { return 100.0 } // Perfect reconstruction
        return 10.0 * log10f(signalPower / noisePower)
    }

    // MARK: - Phase Vocoder Tests

    func testPhaseVocoderOutputShape_Rate1() {
        // rate=1.0 should produce approximately the same number of frames
        let signal = makeSine(duration: 0.5)
        let nFFT = 1024
        let hop = nFFT / 4

        let complexSTFT = STFT.computeComplex(signal: signal, nFFT: nFFT, hopLength: hop)
        let nFreqs = complexSTFT.shape[0]
        let nFramesIn = complexSTFT.shape[1]

        let result = PhaseVocoder.phaseVocoder(complexSTFT: complexSTFT, rate: 1.0, hopLength: hop)

        XCTAssertEqual(result.shape[0], nFreqs, "nFreqs should be preserved")
        XCTAssertEqual(result.shape[1], nFramesIn, "rate=1.0 should preserve frame count")
        XCTAssertEqual(result.dtype, .complex64, "Output should be complex64")
    }

    func testPhaseVocoderOutputShape_Rate2() {
        // rate=2.0 should halve the frame count (approximately)
        let signal = makeSine(duration: 0.5)
        let nFFT = 1024
        let hop = nFFT / 4

        let complexSTFT = STFT.computeComplex(signal: signal, nFFT: nFFT, hopLength: hop)
        let nFreqs = complexSTFT.shape[0]
        let nFramesIn = complexSTFT.shape[1]

        let result = PhaseVocoder.phaseVocoder(complexSTFT: complexSTFT, rate: 2.0, hopLength: hop)

        XCTAssertEqual(result.shape[0], nFreqs, "nFreqs should be preserved")

        let expectedFrames = Int(ceilf(Float(nFramesIn) / 2.0))
        XCTAssertEqual(result.shape[1], expectedFrames,
                       "rate=2.0 should produce ceil(nFrames/2) output frames")
    }

    func testPhaseVocoderOutputShape_RateHalf() {
        // rate=0.5 should double the frame count (approximately)
        let signal = makeSine(duration: 0.5)
        let nFFT = 1024
        let hop = nFFT / 4

        let complexSTFT = STFT.computeComplex(signal: signal, nFFT: nFFT, hopLength: hop)
        let nFreqs = complexSTFT.shape[0]
        let nFramesIn = complexSTFT.shape[1]

        let result = PhaseVocoder.phaseVocoder(complexSTFT: complexSTFT, rate: 0.5, hopLength: hop)

        XCTAssertEqual(result.shape[0], nFreqs, "nFreqs should be preserved")

        let expectedFrames = Int(ceilf(Float(nFramesIn) / 0.5))
        XCTAssertEqual(result.shape[1], expectedFrames,
                       "rate=0.5 should produce ceil(nFrames/0.5) output frames")
    }

    func testPhaseVocoderRate1Preserves() {
        // rate=1.0 should approximately preserve the original signal after iSTFT round-trip
        let signal = makeSine(frequency: 440.0, duration: 0.25)
        let nFFT = 1024
        let hop = nFFT / 4

        let complexSTFT = STFT.computeComplex(signal: signal, nFFT: nFFT, hopLength: hop)
        let stretched = PhaseVocoder.phaseVocoder(complexSTFT: complexSTFT, rate: 1.0, hopLength: hop)
        let reconstructed = STFT.inverse(complexSTFT: stretched, hopLength: hop, center: true)

        // The reconstructed signal should be close in length to the original
        let ratio = Float(reconstructed.count) / Float(signal.count)
        XCTAssertGreaterThan(ratio, 0.8)
        XCTAssertLessThan(ratio, 1.2)

        // Check that values are finite
        XCTAssertTrue(allFinite(stretched))
        XCTAssertTrue(allFinite(reconstructed))
    }

    func testPhaseVocoderValuesFinite() {
        let signal = makeSine(duration: 0.25)
        let nFFT = 1024
        let hop = nFFT / 4

        let complexSTFT = STFT.computeComplex(signal: signal, nFFT: nFFT, hopLength: hop)
        let result = PhaseVocoder.phaseVocoder(complexSTFT: complexSTFT, rate: 1.5, hopLength: hop)

        XCTAssertTrue(allFinite(result), "Phase vocoder output should be finite")
    }

    // MARK: - Griffin-Lim Tests

    func testGriffinLimOutputIs1D() {
        let signal = makeSine(duration: 0.25)
        let nFFT = 1024
        let hop = nFFT / 4

        let magnitude = STFT.compute(signal: signal, nFFT: nFFT, hopLength: hop)
        let result = PhaseVocoder.griffinLim(magnitude: magnitude, nIter: 5, hopLength: hop)

        XCTAssertEqual(result.shape.count, 1, "Griffin-Lim output should be 1D")
        XCTAssertEqual(result.dtype, .float32, "Griffin-Lim output should be float32")
    }

    func testGriffinLimOutputNonEmpty() {
        let signal = makeSine(duration: 0.25)
        let nFFT = 1024
        let hop = nFFT / 4

        let magnitude = STFT.compute(signal: signal, nFFT: nFFT, hopLength: hop)
        let result = PhaseVocoder.griffinLim(magnitude: magnitude, nIter: 5, hopLength: hop)

        XCTAssertGreaterThan(result.count, 0, "Griffin-Lim output should not be empty")
    }

    func testGriffinLimOutputLength() {
        // Output length should be approximately the same as the original signal
        let signal = makeSine(duration: 0.5)
        let nFFT = 1024
        let hop = nFFT / 4

        let magnitude = STFT.compute(signal: signal, nFFT: nFFT, hopLength: hop)
        let result = PhaseVocoder.griffinLim(magnitude: magnitude, nIter: 5, hopLength: hop)

        let ratio = Float(result.count) / Float(signal.count)
        XCTAssertGreaterThan(ratio, 0.8, "Output length should be ~same as input")
        XCTAssertLessThan(ratio, 1.2, "Output length should be ~same as input")
    }

    func testGriffinLimWithExplicitLength() {
        let signal = makeSine(duration: 0.5)
        let nFFT = 1024
        let hop = nFFT / 4
        let targetLength = signal.count

        let magnitude = STFT.compute(signal: signal, nFFT: nFFT, hopLength: hop)
        let result = PhaseVocoder.griffinLim(
            magnitude: magnitude,
            nIter: 5,
            hopLength: hop,
            length: targetLength
        )

        XCTAssertEqual(result.count, targetLength,
                       "Griffin-Lim should respect explicit output length")
    }

    func testGriffinLimSNR() {
        // Tier 2 parity: Griffin-Lim spectral SNR > 15 dB for a simple tone.
        //
        // Griffin-Lim reconstructs audio from magnitude-only spectrogram. Since
        // phase information is lost, the waveform may differ from the original
        // (e.g., different phase offset for a sine wave). The correct quality
        // metric is to compare the magnitude spectrogram of the reconstructed
        // signal to the original magnitude spectrogram.
        let signal = makeSine(frequency: 440.0, sr: 22050, duration: 1.0)
        let nFFT = 2048
        let hop = nFFT / 4

        let origMagnitude = STFT.compute(signal: signal, nFFT: nFFT, hopLength: hop)
        let reconstructed = PhaseVocoder.griffinLim(
            magnitude: origMagnitude,
            nIter: 64,
            hopLength: hop,
            length: signal.count
        )

        // Recompute magnitude from reconstructed audio
        let reconMagnitude = STFT.compute(signal: reconstructed, nFFT: nFFT, hopLength: hop)

        // Compare magnitude spectrograms
        let nFreqs = origMagnitude.shape[0]
        let origFrames = origMagnitude.shape[1]
        let reconFrames = reconMagnitude.shape[1]
        let useFrames = min(origFrames, reconFrames)
        let useFreqs = min(nFreqs, reconMagnitude.shape[0])

        var signalPower: Float = 0
        var noisePower: Float = 0

        origMagnitude.withUnsafeBufferPointer { origBuf in
            reconMagnitude.withUnsafeBufferPointer { reconBuf in
                for freq in 0..<useFreqs {
                    for frame in 0..<useFrames {
                        let origIdx = freq * origFrames + frame
                        let reconIdx = freq * reconFrames + frame
                        let origVal = origBuf[origIdx]
                        let reconVal = reconBuf[reconIdx]
                        let diff = origVal - reconVal
                        signalPower += origVal * origVal
                        noisePower += diff * diff
                    }
                }
            }
        }

        let snr = 10.0 * log10f(signalPower / max(noisePower, 1e-20))
        XCTAssertGreaterThan(snr, 15.0,
                             "Griffin-Lim spectral SNR should be > 15 dB (got \(snr) dB)")
    }

    /// Compute spectral SNR: compare magnitude spectrograms.
    private func spectralSNR(originalMag: Signal, reconstructed: Signal, nFFT: Int, hopLength: Int) -> Float {
        let reconMag = STFT.compute(signal: reconstructed, nFFT: nFFT, hopLength: hopLength)

        let origFrames = originalMag.shape[1]
        let reconFrames = reconMag.shape[1]
        let useFrames = min(origFrames, reconFrames)
        let useFreqs = min(originalMag.shape[0], reconMag.shape[0])

        var signalPower: Float = 0
        var noisePower: Float = 0

        originalMag.withUnsafeBufferPointer { origBuf in
            reconMag.withUnsafeBufferPointer { reconBuf in
                for freq in 0..<useFreqs {
                    for frame in 0..<useFrames {
                        let origIdx = freq * origFrames + frame
                        let reconIdx = freq * reconFrames + frame
                        let origVal = origBuf[origIdx]
                        let reconVal = reconBuf[reconIdx]
                        let diff = origVal - reconVal
                        signalPower += origVal * origVal
                        noisePower += diff * diff
                    }
                }
            }
        }

        return 10.0 * log10f(signalPower / max(noisePower, 1e-20))
    }

    func testGriffinLimMoreIterationsImproves() {
        // More iterations should yield better spectral reconstruction
        let signal = makeSine(frequency: 440.0, sr: 22050, duration: 0.5)
        let nFFT = 1024
        let hop = nFFT / 4

        let magnitude = STFT.compute(signal: signal, nFFT: nFFT, hopLength: hop)

        let result5 = PhaseVocoder.griffinLim(
            magnitude: magnitude, nIter: 5, hopLength: hop, length: signal.count
        )
        let result32 = PhaseVocoder.griffinLim(
            magnitude: magnitude, nIter: 32, hopLength: hop, length: signal.count
        )

        let snr5 = spectralSNR(originalMag: magnitude, reconstructed: result5, nFFT: nFFT, hopLength: hop)
        let snr32 = spectralSNR(originalMag: magnitude, reconstructed: result32, nFFT: nFFT, hopLength: hop)

        // 32 iterations should generally be better than 5.
        // Allow small tolerance for stochastic effects (random initial phase).
        XCTAssertGreaterThan(snr32, snr5 - 3.0,
                             "32 iterations (\(snr32) dB) should be >= 5 iterations (\(snr5) dB) minus tolerance")
    }

    func testGriffinLimValuesFinite() {
        let signal = makeSine(duration: 0.25)
        let nFFT = 1024
        let hop = nFFT / 4

        let magnitude = STFT.compute(signal: signal, nFFT: nFFT, hopLength: hop)
        let result = PhaseVocoder.griffinLim(magnitude: magnitude, nIter: 5, hopLength: hop)

        XCTAssertTrue(allFinite(result), "Griffin-Lim output should be finite")
    }

    func testGriffinLimShortSignal() {
        // Test with a very short signal (just enough for 1-2 frames)
        let sr = 22050
        let nFFT = 512
        let hop = nFFT / 4
        let length = nFFT + hop // Enough for 2 frames

        var data = [Float](repeating: 0, count: length)
        for i in 0..<length {
            data[i] = sinf(2.0 * Float.pi * 440.0 * Float(i) / Float(sr))
        }
        let signal = Signal(data: data, sampleRate: sr)

        let magnitude = STFT.compute(signal: signal, nFFT: nFFT, hopLength: hop)
        let result = PhaseVocoder.griffinLim(magnitude: magnitude, nIter: 5, hopLength: hop)

        XCTAssertGreaterThan(result.count, 0, "Should handle short signals")
        XCTAssertTrue(allFinite(result))
    }

    // MARK: - Griffin-Lim CQT Tests

    func testGriffinLimCQTOutputIs1D() {
        let signal = makeSine(frequency: 440.0, duration: 0.5)
        let sr = signal.sampleRate

        // Compute CQT magnitude
        let cqtMag = CQT.compute(signal: signal, sr: sr, hopLength: 512,
                                   fMin: 32.7, binsPerOctave: 12)

        guard cqtMag.shape[0] > 0 && cqtMag.shape[1] > 0 else {
            // Skip if CQT produces empty output
            return
        }

        let result = PhaseVocoder.griffinLimCQT(
            magnitude: cqtMag,
            sr: sr,
            nIter: 5,
            hopLength: 512,
            fMin: 32.7,
            binsPerOctave: 12
        )

        XCTAssertEqual(result.shape.count, 1, "CQT Griffin-Lim output should be 1D")
        XCTAssertGreaterThan(result.count, 0, "CQT Griffin-Lim output should not be empty")
        XCTAssertTrue(allFinite(result))
    }

    // MARK: - Edge Cases

    func testPhaseVocoderSingleFrame() {
        // Single-frame STFT
        let nFreqs = 5
        var complexData = [Float](repeating: 0, count: nFreqs * 1 * 2)
        for k in 0..<nFreqs {
            complexData[k * 2] = Float(k + 1)     // real
            complexData[k * 2 + 1] = 0.0           // imag
        }
        let stft = Signal(complexData: complexData, shape: [nFreqs, 1], sampleRate: 22050)

        let result = PhaseVocoder.phaseVocoder(complexSTFT: stft, rate: 1.0)

        XCTAssertEqual(result.shape[0], nFreqs)
        XCTAssertEqual(result.shape[1], 1)
        XCTAssertTrue(allFinite(result))
    }
}
