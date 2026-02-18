import XCTest
@testable import MetalMomCore

final class InversionTests: XCTestCase {

    // MARK: - Helpers

    /// Generate a pure sine wave signal.
    private func makeSine(frequency: Float = 440.0, sr: Int = 22050, duration: Float = 1.0) -> Signal {
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

    /// Check that all values in a Signal are non-negative.
    private func allNonNegative(_ signal: Signal) -> Bool {
        var result = true
        signal.withUnsafeBufferPointer { buf in
            for i in 0..<buf.count {
                if buf[i] < 0 {
                    result = false
                    return
                }
            }
        }
        return result
    }

    // MARK: - melToSTFT Tests

    func testMelToSTFTOutputShape() {
        let signal = makeSine(duration: 0.5)
        let sr = signal.sampleRate
        let nFFT = 2048
        let nMels = 128

        let melSpec = MelSpectrogram.compute(signal: signal, sr: sr, nFFT: nFFT, nMels: nMels)
        let nFrames = melSpec.shape[1]

        let stftMag = Inversion.melToSTFT(melSpectrogram: melSpec, sr: sr, nFFT: nFFT)

        let expectedNFreqs = nFFT / 2 + 1
        XCTAssertEqual(stftMag.shape.count, 2, "Output should be 2D")
        XCTAssertEqual(stftMag.shape[0], expectedNFreqs,
                       "First dimension should be nFFT/2+1 = \(expectedNFreqs)")
        XCTAssertEqual(stftMag.shape[1], nFrames,
                       "Second dimension should match input nFrames")
    }

    func testMelToSTFTNonNegative() {
        let signal = makeSine(duration: 0.5)
        let sr = signal.sampleRate
        let nFFT = 2048

        let melSpec = MelSpectrogram.compute(signal: signal, sr: sr, nFFT: nFFT)
        let stftMag = Inversion.melToSTFT(melSpectrogram: melSpec, sr: sr, nFFT: nFFT)

        XCTAssertTrue(allNonNegative(stftMag),
                       "melToSTFT output should be non-negative")
    }

    func testMelToSTFTFinite() {
        let signal = makeSine(duration: 0.5)
        let sr = signal.sampleRate
        let nFFT = 2048

        let melSpec = MelSpectrogram.compute(signal: signal, sr: sr, nFFT: nFFT)
        let stftMag = Inversion.melToSTFT(melSpectrogram: melSpec, sr: sr, nFFT: nFFT)

        XCTAssertTrue(allFinite(stftMag), "melToSTFT output should be finite")
    }

    // MARK: - melToAudio Tests

    func testMelToAudioOutputIs1D() {
        let signal = makeSine(duration: 0.5)
        let sr = signal.sampleRate
        let nFFT = 1024

        let melSpec = MelSpectrogram.compute(
            signal: signal, sr: sr, nFFT: nFFT,
            hopLength: nFFT / 4, nMels: 64
        )
        let audio = Inversion.melToAudio(
            melSpectrogram: melSpec, sr: sr, nFFT: nFFT,
            hopLength: nFFT / 4, nIter: 5
        )

        XCTAssertEqual(audio.shape.count, 1, "melToAudio output should be 1D")
        XCTAssertGreaterThan(audio.count, 0, "melToAudio output should not be empty")
        XCTAssertTrue(allFinite(audio), "melToAudio output should be finite")
    }

    func testMelToAudioReconstructionSNR() {
        // Tier 2 parity: mel spectral SNR > 8 dB for a clean tone.
        //
        // Mel-to-audio inversion is inherently lossy: the mel filterbank
        // compresses ~1025 frequency bins into 128 mel bands, losing fine
        // spectral detail. The quality metric compares mel spectrograms
        // (not raw audio waveforms) since phase is estimated by Griffin-Lim.
        //
        // For a 440 Hz sine wave, energy concentrates in a few mel bands.
        // The pseudo-inverse spreads energy to nearby linear-frequency bins,
        // and Griffin-Lim estimates phase from this approximate magnitude.
        // An SNR > 8 dB indicates reasonable spectral reconstruction.
        let signal = makeSine(frequency: 440.0, sr: 22050, duration: 1.0)
        let sr = signal.sampleRate
        let nFFT = 2048
        let hop = nFFT / 4
        let nMels = 128

        // Forward: audio -> mel spectrogram
        let origMel = MelSpectrogram.compute(
            signal: signal, sr: sr, nFFT: nFFT,
            hopLength: hop, nMels: nMels
        )

        // Invert: mel spectrogram -> audio
        let reconstructed = Inversion.melToAudio(
            melSpectrogram: origMel, sr: sr, nFFT: nFFT,
            hopLength: hop, nIter: 64
        )

        // Recompute mel from reconstructed audio
        let reconMel = MelSpectrogram.compute(
            signal: reconstructed, sr: sr, nFFT: nFFT,
            hopLength: hop, nMels: nMels
        )

        // Compare mel spectrograms (spectral SNR)
        let origMels = origMel.shape[0]
        let origFrames = origMel.shape[1]
        let reconFrames = reconMel.shape[1]
        let useFrames = min(origFrames, reconFrames)
        let useMels = min(origMels, reconMel.shape[0])

        var signalPower: Float = 0
        var noisePower: Float = 0

        origMel.withUnsafeBufferPointer { origBuf in
            reconMel.withUnsafeBufferPointer { reconBuf in
                for m in 0..<useMels {
                    for f in 0..<useFrames {
                        let origIdx = m * origFrames + f
                        let reconIdx = m * reconFrames + f
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
        XCTAssertGreaterThan(snr, 8.0,
                             "Mel-to-audio spectral SNR should be > 8 dB (got \(snr) dB)")
    }

    // MARK: - mfccToMel Tests

    func testMFCCToMelOutputShape() {
        let signal = makeSine(duration: 0.5)
        let sr = signal.sampleRate
        let nMFCC = 20
        let nMels = 128

        let mfcc = MFCC.compute(signal: signal, sr: sr, nMFCC: nMFCC, nMels: nMels)
        let nFrames = mfcc.shape[1]

        let mel = Inversion.mfccToMel(mfcc: mfcc, nMels: nMels)

        XCTAssertEqual(mel.shape.count, 2, "Output should be 2D")
        XCTAssertEqual(mel.shape[0], nMels,
                       "First dimension should be nMels = \(nMels)")
        XCTAssertEqual(mel.shape[1], nFrames,
                       "Second dimension should match input nFrames")
    }

    func testMFCCToMelFinite() {
        let signal = makeSine(duration: 0.5)
        let sr = signal.sampleRate

        let mfcc = MFCC.compute(signal: signal, sr: sr, nMFCC: 20, nMels: 128)
        let mel = Inversion.mfccToMel(mfcc: mfcc, nMels: 128)

        XCTAssertTrue(allFinite(mel), "mfccToMel output should be finite")
    }

    func testMFCCToMelNonNegative() {
        // Since mel spectrogram should be non-negative (it's power),
        // and we exponentiate the log mel, output should be non-negative.
        let signal = makeSine(duration: 0.5)
        let sr = signal.sampleRate

        let mfcc = MFCC.compute(signal: signal, sr: sr, nMFCC: 20, nMels: 128)
        let mel = Inversion.mfccToMel(mfcc: mfcc, nMels: 128)

        XCTAssertTrue(allNonNegative(mel), "mfccToMel output should be non-negative")
    }

    func testMFCCToMelRoundTrip() {
        // Round-trip: compute MFCCs from audio, reconstruct mel, compare shape.
        // Since MFCCs discard info (only first 20 of 128 coefficients),
        // we can't expect exact reconstruction, but the shape should match
        // and the reconstructed mel should capture broad spectral shape.
        let signal = makeSine(frequency: 440.0, duration: 0.5)
        let sr = signal.sampleRate
        let nMFCC = 20
        let nMels = 128

        let mfcc = MFCC.compute(signal: signal, sr: sr, nMFCC: nMFCC, nMels: nMels)
        let reconMel = Inversion.mfccToMel(mfcc: mfcc, nMels: nMels)

        XCTAssertEqual(reconMel.shape[0], nMels)
        XCTAssertEqual(reconMel.shape[1], mfcc.shape[1])
        XCTAssertGreaterThan(reconMel.count, 0)
    }

    // MARK: - mfccToAudio Tests

    func testMFCCToAudioOutputIs1D() {
        let signal = makeSine(duration: 0.5)
        let sr = signal.sampleRate
        let nFFT = 1024

        let mfcc = MFCC.compute(
            signal: signal, sr: sr, nMFCC: 20,
            nFFT: nFFT, hopLength: nFFT / 4, nMels: 64
        )
        let audio = Inversion.mfccToAudio(
            mfcc: mfcc, sr: sr, nMels: 64, nFFT: nFFT,
            hopLength: nFFT / 4, nIter: 5
        )

        XCTAssertEqual(audio.shape.count, 1, "mfccToAudio output should be 1D")
        XCTAssertGreaterThan(audio.count, 0, "mfccToAudio output should not be empty")
        XCTAssertTrue(allFinite(audio), "mfccToAudio output should be finite")
    }

    // MARK: - Edge Cases

    func testMelToSTFTSingleFrame() {
        // Single-frame mel spectrogram
        let nMels = 40
        let data = [Float](repeating: 0.5, count: nMels)
        let melSpec = Signal(data: data, shape: [nMels, 1], sampleRate: 22050)

        let stftMag = Inversion.melToSTFT(melSpectrogram: melSpec, sr: 22050, nFFT: 1024)

        XCTAssertEqual(stftMag.shape[0], 513)  // 1024/2 + 1
        XCTAssertEqual(stftMag.shape[1], 1)
        XCTAssertTrue(allFinite(stftMag))
        XCTAssertTrue(allNonNegative(stftMag))
    }

    func testMFCCToMelSingleFrame() {
        // Single-frame MFCC
        let nMFCC = 13
        let nMels = 64
        var data = [Float](repeating: 0, count: nMFCC)
        data[0] = 10.0  // Only DC component
        let mfcc = Signal(data: data, shape: [nMFCC, 1], sampleRate: 22050)

        let mel = Inversion.mfccToMel(mfcc: mfcc, nMels: nMels)

        XCTAssertEqual(mel.shape[0], nMels)
        XCTAssertEqual(mel.shape[1], 1)
        XCTAssertTrue(allFinite(mel))
    }

    func testMelToAudioShortSignal() {
        // Very short signal (a few frames only)
        let sr = 22050
        let nFFT = 512
        let hop = nFFT / 4
        let nMels = 40

        let shortSignal = makeSine(frequency: 440.0, sr: sr, duration: 0.05)
        let melSpec = MelSpectrogram.compute(
            signal: shortSignal, sr: sr, nFFT: nFFT,
            hopLength: hop, nMels: nMels
        )

        let audio = Inversion.melToAudio(
            melSpectrogram: melSpec, sr: sr, nFFT: nFFT,
            hopLength: hop, nIter: 5
        )

        XCTAssertGreaterThan(audio.count, 0, "Should handle short signals")
        XCTAssertTrue(allFinite(audio))
    }

    func testMelToSTFTDifferentNMels() {
        // Test with various nMels values
        let sr = 22050
        let nFFT = 1024

        for nMels in [32, 64, 128] {
            let signal = makeSine(sr: sr, duration: 0.25)
            let melSpec = MelSpectrogram.compute(
                signal: signal, sr: sr, nFFT: nFFT, nMels: nMels
            )
            let stftMag = Inversion.melToSTFT(melSpectrogram: melSpec, sr: sr, nFFT: nFFT)

            XCTAssertEqual(stftMag.shape[0], nFFT / 2 + 1,
                           "nFreqs should be nFFT/2+1 for nMels=\(nMels)")
            XCTAssertTrue(allFinite(stftMag))
            XCTAssertTrue(allNonNegative(stftMag))
        }
    }
}
