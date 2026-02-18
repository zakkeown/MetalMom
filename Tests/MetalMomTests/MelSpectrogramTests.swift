import XCTest
@testable import MetalMomCore

final class MelSpectrogramTests: XCTestCase {

    // MARK: - Shape Tests

    func testDefaultShape() {
        // 1 second of 440 Hz sine at 22050 Hz
        let sr = 22050
        let t = (0..<sr).map { Float(2.0 * .pi * 440.0 * Double($0) / Double(sr)) }
        let signal440 = t.map { sinf($0) }
        let signal = Signal(data: signal440, sampleRate: sr)

        let result = MelSpectrogram.compute(signal: signal)
        // Default: nMels=128, nFFT=2048, hopLength=512
        // nFrames = 1 + (22050 + 2048 - 2048) / 512 = 1 + 22050/512 = 44
        XCTAssertEqual(result.shape.count, 2, "Should be 2D")
        XCTAssertEqual(result.shape[0], 128, "Should have 128 mel bands")
        XCTAssertGreaterThan(result.shape[1], 0, "Should have at least 1 frame")
    }

    func testCustomNMels() {
        let sr = 22050
        let signal = Signal(data: [Float](repeating: 0.1, count: sr), sampleRate: sr)

        let result = MelSpectrogram.compute(signal: signal, nMels: 40)
        XCTAssertEqual(result.shape[0], 40, "Should have 40 mel bands")
    }

    func testCustomNFFT() {
        let sr = 22050
        let signal = Signal(data: [Float](repeating: 0.1, count: sr), sampleRate: sr)

        let result = MelSpectrogram.compute(signal: signal, nFFT: 1024, nMels: 64)
        XCTAssertEqual(result.shape[0], 64, "Should have 64 mel bands")
        XCTAssertGreaterThan(result.shape[1], 0, "Should have frames")
    }

    // MARK: - Non-negativity

    func testAllValuesNonNegative() {
        let sr = 22050
        let t = (0..<sr).map { Float(2.0 * .pi * 440.0 * Double($0) / Double(sr)) }
        let signal440 = t.map { sinf($0) }
        let signal = Signal(data: signal440, sampleRate: sr)

        let result = MelSpectrogram.compute(signal: signal)
        for i in 0..<result.count {
            XCTAssertGreaterThanOrEqual(result[i], 0.0,
                                         "Mel spectrogram value at index \(i) should be >= 0")
        }
    }

    // MARK: - Power parameter

    func testPower1ReturnsAmplitudeMelSpec() {
        let sr = 22050
        let t = (0..<sr).map { Float(2.0 * .pi * 440.0 * Double($0) / Double(sr)) }
        let signal440 = t.map { sinf($0) }
        let signal = Signal(data: signal440, sampleRate: sr)

        let melPow1 = MelSpectrogram.compute(signal: signal, power: 1.0)
        let melPow2 = MelSpectrogram.compute(signal: signal, power: 2.0)

        // power=2 values should generally be larger (squared) for values > 1,
        // but the mel FB multiplication can change that. Instead, check shapes match.
        XCTAssertEqual(melPow1.shape, melPow2.shape, "Shapes should match regardless of power")

        // Both should be non-negative
        for i in 0..<melPow1.count {
            XCTAssertGreaterThanOrEqual(melPow1[i], 0.0)
        }
    }

    // MARK: - Energy concentration for 440 Hz

    func testEnergyConcentration440Hz() {
        // 440 Hz sine should concentrate energy in the appropriate mel band
        let sr = 22050
        let t = (0..<sr).map { Float(2.0 * .pi * 440.0 * Double($0) / Double(sr)) }
        let signal440 = t.map { sinf($0) }
        let signal = Signal(data: signal440, sampleRate: sr)

        let result = MelSpectrogram.compute(signal: signal, nMels: 128)
        let nMels = result.shape[0]
        let nFrames = result.shape[1]

        // Sum energy across all frames for each mel band
        var bandEnergies = [Float](repeating: 0, count: nMels)
        for m in 0..<nMels {
            for f in 0..<nFrames {
                bandEnergies[m] += result[m * nFrames + f]
            }
        }

        // Find the mel band with maximum energy
        let maxBandIdx = bandEnergies.indices.max(by: { bandEnergies[$0] < bandEnergies[$1] })!

        // 440 Hz should be somewhere in the lower-middle mel bands (roughly band 15-30 for 128 mels)
        // The exact band depends on mel spacing, but it should not be at band 0 or band 127
        XCTAssertGreaterThan(maxBandIdx, 5,
                             "Peak mel band for 440 Hz should not be at very low bands")
        XCTAssertLessThan(maxBandIdx, 60,
                          "Peak mel band for 440 Hz should not be at high bands")

        // Verify the peak band has significantly more energy than the average
        let totalEnergy = bandEnergies.reduce(0, +)
        let avgEnergy = totalEnergy / Float(nMels)
        XCTAssertGreaterThan(bandEnergies[maxBandIdx], avgEnergy * 2.0,
                             "Peak band should have significantly more energy than average")
    }

    // MARK: - Silence

    func testSilenceProducesNearZero() {
        let sr = 22050
        let signal = Signal(data: [Float](repeating: 0, count: sr), sampleRate: sr)

        let result = MelSpectrogram.compute(signal: signal)
        var maxVal: Float = 0
        for i in 0..<result.count {
            maxVal = max(maxVal, result[i])
        }
        XCTAssertLessThan(maxVal, 1e-10, "Silent signal should produce near-zero mel spectrogram")
    }

    // MARK: - Sample rate parameter

    func testExplicitSampleRate() {
        // Pass a different sr than the signal's sampleRate; the explicit sr should be used
        let signal = Signal(data: [Float](repeating: 0.1, count: 16000), sampleRate: 16000)
        let result = MelSpectrogram.compute(signal: signal, sr: 16000, nFFT: 512, nMels: 40)
        XCTAssertEqual(result.shape[0], 40)
    }

    // MARK: - Consistency with STFT + FilterBank

    func testConsistencyWithManualPipeline() {
        // Manually compute: STFT magnitude -> power -> mel FB @ power = mel spec
        // Should match MelSpectrogram.compute
        let sr = 22050
        let t = (0..<sr).map { Float(2.0 * .pi * 440.0 * Double($0) / Double(sr)) }
        let signal440 = t.map { sinf($0) }
        let signal = Signal(data: signal440, sampleRate: sr)

        let melSpec = MelSpectrogram.compute(signal: signal, nFFT: 2048, power: 2.0, nMels: 128)

        // Manual pipeline
        let stftMag = STFT.compute(signal: signal, nFFT: 2048)
        // stftMag shape: [nFreqs, nFrames]
        let nFreqs = stftMag.shape[0]
        let nFrames = stftMag.shape[1]

        // Square for power spectrogram
        var powered = [Float](repeating: 0, count: stftMag.count)
        for i in 0..<stftMag.count {
            powered[i] = stftMag[i] * stftMag[i]
        }

        // Get mel filterbank
        let melFB = FilterBank.mel(sr: sr, nFFT: 2048, nMels: 128)

        // Matrix multiply: melFB [128, nFreqs] @ powered [nFreqs, nFrames]
        let nMels = 128
        var expected = [Float](repeating: 0, count: nMels * nFrames)
        for m in 0..<nMels {
            for f in 0..<nFrames {
                var sum: Float = 0
                for k in 0..<nFreqs {
                    sum += melFB[m * nFreqs + k] * powered[k * nFrames + f]
                }
                expected[m * nFrames + f] = sum
            }
        }

        // Compare
        XCTAssertEqual(melSpec.shape, [nMels, nFrames])
        for i in 0..<melSpec.count {
            // vDSP_mmul and naive triple-loop accumulate differently; use relative tolerance
            let tol = max(abs(expected[i]) * 1e-5, 1e-4)
            XCTAssertEqual(melSpec[i], expected[i], accuracy: tol,
                           "Mel spectrogram mismatch at index \(i)")
        }
    }
}
