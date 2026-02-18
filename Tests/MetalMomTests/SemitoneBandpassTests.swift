import XCTest
@testable import MetalMomCore

final class SemitoneBandpassTests: XCTestCase {

    // MARK: - Semitone Frequencies

    func testSemitoneFrequenciesCount() {
        // C1 (MIDI 24) through B8 (MIDI 119) = 96 semitones
        let freqs = SemitoneBandpass.semitoneFrequencies()
        XCTAssertEqual(freqs.count, 96, "Default range should produce 96 frequencies")
    }

    func testSemitoneFrequenciesA4() {
        // MIDI 69 = A4 = 440 Hz
        let freqs = SemitoneBandpass.semitoneFrequencies(midiLow: 69, midiHigh: 69)
        XCTAssertEqual(freqs.count, 1)
        XCTAssertEqual(freqs[0], 440.0, accuracy: 0.01, "MIDI 69 should be 440 Hz")
    }

    func testSemitoneFrequenciesMonotonic() {
        let freqs = SemitoneBandpass.semitoneFrequencies()
        for i in 1..<freqs.count {
            XCTAssertGreaterThan(freqs[i], freqs[i - 1],
                                 "Frequencies should be strictly increasing")
        }
    }

    func testSemitoneFrequenciesC1() {
        // MIDI 24 = C1 ~ 32.70 Hz
        let freqs = SemitoneBandpass.semitoneFrequencies(midiLow: 24, midiHigh: 24)
        XCTAssertEqual(freqs[0], 32.7032, accuracy: 0.01)
    }

    func testMidiToHz() {
        XCTAssertEqual(SemitoneBandpass.midiToHz(69), 440.0, accuracy: 0.001)
        XCTAssertEqual(SemitoneBandpass.midiToHz(60), 261.6256, accuracy: 0.01)   // C4
        XCTAssertEqual(SemitoneBandpass.midiToHz(81), 880.0, accuracy: 0.01)      // A5
    }

    // MARK: - Filter Design

    func testDesignBandpassReturnsCoeffs() {
        let coeffs = SemitoneBandpass.designBandpass(centerFreq: 440.0, sr: 22050)
        // Order 4 = 2 biquad sections
        XCTAssertEqual(coeffs.count, 2, "Order 4 should produce 2 biquad sections")
    }

    func testDesignBandpassCoeffsFinite() {
        let freqs: [Float] = [32.7, 261.6, 440.0, 880.0, 4186.0]
        let sampleRates = [22050, 44100, 48000]
        for sr in sampleRates {
            for freq in freqs {
                let coeffs = SemitoneBandpass.designBandpass(centerFreq: freq, sr: sr)
                for section in coeffs {
                    XCTAssert(section.b0.isFinite, "b0 should be finite for freq=\(freq), sr=\(sr)")
                    XCTAssert(section.b1.isFinite, "b1 should be finite for freq=\(freq), sr=\(sr)")
                    XCTAssert(section.b2.isFinite, "b2 should be finite for freq=\(freq), sr=\(sr)")
                    XCTAssert(section.a1.isFinite, "a1 should be finite for freq=\(freq), sr=\(sr)")
                    XCTAssert(section.a2.isFinite, "a2 should be finite for freq=\(freq), sr=\(sr)")
                }
            }
        }
    }

    func testDesignBandpassHighFreqNearNyquist() {
        // 10000 Hz at sr=22050 is close to Nyquist (11025). Should still work.
        let coeffs = SemitoneBandpass.designBandpass(centerFreq: 10000.0, sr: 22050)
        XCTAssertEqual(coeffs.count, 2)
        for section in coeffs {
            XCTAssert(section.b0.isFinite)
        }
    }

    func testDesignBandpassOrder2() {
        let coeffs = SemitoneBandpass.designBandpass(centerFreq: 440.0, sr: 22050, order: 2)
        XCTAssertEqual(coeffs.count, 1, "Order 2 should produce 1 biquad section")
    }

    // MARK: - Biquad Cascade

    func testApplyBiquadCascadePreservesLength() {
        let data = [Float](repeating: 0.5, count: 1024)
        let signal = Signal(data: data, sampleRate: 22050)
        let coeffs = SemitoneBandpass.designBandpass(centerFreq: 440.0, sr: 22050)
        let filtered = SemitoneBandpass.applyBiquadCascade(signal: signal, coefficients: coeffs)
        XCTAssertEqual(filtered.count, 1024, "Output length should match input length")
    }

    func testApplyBiquadCascadeSilenceProducesNearZero() {
        let data = [Float](repeating: 0.0, count: 512)
        let signal = Signal(data: data, sampleRate: 22050)
        let coeffs = SemitoneBandpass.designBandpass(centerFreq: 440.0, sr: 22050)
        let filtered = SemitoneBandpass.applyBiquadCascade(signal: signal, coefficients: coeffs)

        var maxVal: Float = 0
        for i in 0..<filtered.count {
            maxVal = max(maxVal, abs(filtered[i]))
        }
        XCTAssertLessThan(maxVal, 1e-10, "Filtering silence should produce near-zero output")
    }

    func testApplyBiquadCascadeFiniteOutput() {
        // Random-ish signal
        var data = [Float](repeating: 0, count: 2048)
        for i in 0..<data.count {
            data[i] = sinf(Float(i) * 0.1) * 0.8
        }
        let signal = Signal(data: data, sampleRate: 22050)
        let coeffs = SemitoneBandpass.designBandpass(centerFreq: 440.0, sr: 22050)
        let filtered = SemitoneBandpass.applyBiquadCascade(signal: signal, coefficients: coeffs)

        for i in 0..<filtered.count {
            XCTAssert(filtered[i].isFinite, "Output at \(i) should be finite")
        }
    }

    // MARK: - Filterbank Output Shape

    func testFilterbankShape() {
        let data = [Float](repeating: 0.1, count: 512)
        let signal = Signal(data: data, sampleRate: 22050)
        let result = SemitoneBandpass.filterbank(signal: signal)

        // Default MIDI range: 24..119 = 96 semitones
        XCTAssertEqual(result.shape, [96, 512],
                       "Filterbank should produce [nSemitones, nSamples]")
    }

    func testFilterbankCustomRange() {
        let data = [Float](repeating: 0.1, count: 256)
        let signal = Signal(data: data, sampleRate: 44100)
        let result = SemitoneBandpass.filterbank(
            signal: signal,
            midiLow: 60,
            midiHigh: 72
        )
        // MIDI 60..72 = 13 semitones (C4 to C5)
        XCTAssertEqual(result.shape, [13, 256])
    }

    // MARK: - Sine Wave Response

    func testSineWaveStrongestInCorrectBand() {
        // Generate a pure 440 Hz sine wave (A4, MIDI 69)
        let sr = 22050
        let duration: Float = 0.5  // 0.5 seconds
        let nSamples = Int(duration * Float(sr))
        var data = [Float](repeating: 0, count: nSamples)
        for i in 0..<nSamples {
            data[i] = sinf(2.0 * Float.pi * 440.0 * Float(i) / Float(sr))
        }
        let signal = Signal(data: data, sampleRate: sr)

        // Filter a small range around A4
        let midiLow = 66   // F#4
        let midiHigh = 72  // C5
        let result = SemitoneBandpass.filterbank(
            signal: signal,
            midiLow: midiLow,
            midiHigh: midiHigh,
            order: 4
        )

        let nSemitones = midiHigh - midiLow + 1

        // Compute energy in each band
        var energies = [Float](repeating: 0, count: nSemitones)
        for band in 0..<nSemitones {
            var energy: Float = 0
            for j in 0..<nSamples {
                let val = result[band * nSamples + j]
                energy += val * val
            }
            energies[band] = energy
        }

        // A4 = MIDI 69, which is band index 69 - 66 = 3
        let a4BandIdx = 69 - midiLow

        // The A4 band should have the highest energy
        let maxEnergy = energies.max()!
        XCTAssertEqual(energies[a4BandIdx], maxEnergy, accuracy: maxEnergy * 0.01,
                       "A4 band should have the highest energy for a 440 Hz sine")

        // Adjacent bands should have less energy
        for band in 0..<nSemitones where band != a4BandIdx {
            XCTAssertLessThan(energies[band], energies[a4BandIdx],
                              "Band \(midiLow + band) should have less energy than A4")
        }
    }

    // MARK: - Sample Rate Variants

    func testFilterbank44100() {
        let data = [Float](repeating: 0.1, count: 256)
        let signal = Signal(data: data, sampleRate: 44100)
        let result = SemitoneBandpass.filterbank(signal: signal, sr: 44100)
        XCTAssertEqual(result.shape, [96, 256])
    }

    func testFilterbank48000() {
        let data = [Float](repeating: 0.1, count: 256)
        let signal = Signal(data: data, sampleRate: 48000)
        let result = SemitoneBandpass.filterbank(signal: signal, sr: 48000)
        XCTAssertEqual(result.shape, [96, 256])
    }

    // MARK: - Edge Cases

    func testFilterbankEmptySignal() {
        let signal = Signal(data: [Float](), shape: [0], sampleRate: 22050)
        let result = SemitoneBandpass.filterbank(signal: signal)
        XCTAssertEqual(result.shape, [0, 0])
    }

    func testFilterbankSingleSample() {
        let signal = Signal(data: [1.0], sampleRate: 22050)
        let result = SemitoneBandpass.filterbank(
            signal: signal,
            midiLow: 69,
            midiHigh: 69
        )
        XCTAssertEqual(result.shape, [1, 1])
    }

    func testDesignForMIDI() {
        let coeffs = SemitoneBandpass.designForMIDI(midi: 69, sr: 22050)
        XCTAssertEqual(coeffs.count, 2)
        // Should be same as designing for 440 Hz
        let coeffsDirect = SemitoneBandpass.designBandpass(centerFreq: 440.0, sr: 22050)
        XCTAssertEqual(coeffs[0].b0, coeffsDirect[0].b0, accuracy: 1e-6)
        XCTAssertEqual(coeffs[0].a1, coeffsDirect[0].a1, accuracy: 1e-6)
    }
}
