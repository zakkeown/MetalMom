import XCTest
@testable import MetalMomCore

final class CombFilterTests: XCTestCase {

    // MARK: - Forward Comb Filter

    func testForwardCombImpulseResponse() {
        // An impulse [1, 0, 0, 0, ...] through a forward comb filter with delay=3
        // should produce echoes at positions 0, 3, 6, 9, ...
        // y[0] = 1, y[3] = alpha * y[0] = alpha, y[6] = alpha^2, etc.
        var impulse = [Float](repeating: 0, count: 20)
        impulse[0] = 1.0
        let alpha: Float = 0.8

        let output = CombFilter.forward(signal: impulse, delay: 3, alpha: alpha)

        XCTAssertEqual(output.count, 20)
        XCTAssertEqual(output[0], 1.0, accuracy: 1e-6, "Impulse at position 0")
        XCTAssertEqual(output[1], 0.0, accuracy: 1e-6, "Zero between echoes")
        XCTAssertEqual(output[2], 0.0, accuracy: 1e-6, "Zero between echoes")
        XCTAssertEqual(output[3], alpha, accuracy: 1e-6, "First echo at delay=3")
        XCTAssertEqual(output[6], alpha * alpha, accuracy: 1e-6, "Second echo at delay=6")
        XCTAssertEqual(output[9], alpha * alpha * alpha, accuracy: 1e-5, "Third echo at delay=9")
    }

    func testForwardCombEmptySignal() {
        let output = CombFilter.forward(signal: [], delay: 3)
        XCTAssertTrue(output.isEmpty)
    }

    func testForwardCombZeroDelay() {
        let signal: [Float] = [1, 2, 3, 4, 5]
        let output = CombFilter.forward(signal: signal, delay: 0)
        XCTAssertEqual(output, signal, "Zero delay should return input unchanged")
    }

    // MARK: - Backward Comb Filter

    func testBackwardCombFilter() {
        // FIR comb: y[n] = x[n] + alpha * x[n - delay]
        let signal: [Float] = [1, 0, 0, 1, 0, 0, 1, 0, 0]
        let alpha: Float = 0.5
        let delay = 3

        let output = CombFilter.backward(signal: signal, delay: delay, alpha: alpha)

        XCTAssertEqual(output.count, signal.count)
        // y[0] = x[0] = 1.0  (no delayed sample)
        XCTAssertEqual(output[0], 1.0, accuracy: 1e-6)
        // y[1] = x[1] = 0.0
        XCTAssertEqual(output[1], 0.0, accuracy: 1e-6)
        // y[3] = x[3] + alpha * x[0] = 1.0 + 0.5 * 1.0 = 1.5
        XCTAssertEqual(output[3], 1.5, accuracy: 1e-6)
        // y[4] = x[4] + alpha * x[1] = 0.0 + 0.5 * 0.0 = 0.0
        XCTAssertEqual(output[4], 0.0, accuracy: 1e-6)
        // y[6] = x[6] + alpha * x[3] = 1.0 + 0.5 * 1.0 = 1.5
        XCTAssertEqual(output[6], 1.5, accuracy: 1e-6)
    }

    func testBackwardCombImpulse() {
        // Impulse through backward comb should produce exactly two non-zero samples
        var impulse = [Float](repeating: 0, count: 10)
        impulse[0] = 1.0
        let alpha: Float = 0.7

        let output = CombFilter.backward(signal: impulse, delay: 4, alpha: alpha)

        XCTAssertEqual(output[0], 1.0, accuracy: 1e-6, "Original impulse")
        XCTAssertEqual(output[4], alpha, accuracy: 1e-6, "Delayed copy at position 4")
        // All other samples should be zero
        for i in [1, 2, 3, 5, 6, 7, 8, 9] {
            XCTAssertEqual(output[i], 0.0, accuracy: 1e-6, "Should be zero at position \(i)")
        }
    }

    // MARK: - Comb Filter at Known Tempo

    func testCombFilterAtKnownTempo() {
        // Create a click track at 120 BPM with fps=43.07 (22050/512)
        let fps: Float = 22050.0 / 512.0  // ~43.07 fps
        let periodFrames = Int(round(fps * 60.0 / 120.0))  // ~21.5 -> 22 frames

        // Create periodic signal with peaks every `periodFrames` frames
        let nFrames = 400
        var signal = [Float](repeating: 0, count: nFrames)
        var idx = 0
        while idx < nFrames {
            signal[idx] = 1.0
            idx += periodFrames
        }

        // Apply forward comb at matching period -> high energy
        let matched = CombFilter.forward(signal: signal, delay: periodFrames, alpha: 0.99)
        var matchedEnergy: Float = 0
        for v in matched { matchedEnergy += v * v }

        // Apply forward comb at non-matching period -> lower energy
        let mismatched = CombFilter.forward(signal: signal, delay: periodFrames + 7, alpha: 0.99)
        var mismatchedEnergy: Float = 0
        for v in mismatched { mismatchedEnergy += v * v }

        XCTAssertGreaterThan(matchedEnergy, mismatchedEnergy,
                             "Comb filter at matching period should produce higher energy")
    }

    // MARK: - Filter Bank Peak at Correct BPM

    func testFilterBankPeakAt120BPM() {
        let fps: Float = 22050.0 / 512.0
        let periodFrames = Int(round(fps * 60.0 / 120.0))

        let nFrames = 500
        var signal = [Float](repeating: 0, count: nFrames)
        var idx = 0
        while idx < nFrames {
            signal[idx] = 1.0
            idx += periodFrames
        }

        let (tempos, energies) = CombFilter.tempoFilterBank(
            signal: signal,
            fps: fps,
            minBPM: 60,
            maxBPM: 200,
            bpmStep: 1.0,
            alpha: 0.99
        )

        // Find peak
        var maxEnergy: Float = -1
        var peakBPM: Float = 0
        for i in 0..<tempos.count {
            if energies[i] > maxEnergy {
                maxEnergy = energies[i]
                peakBPM = tempos[i]
            }
        }

        // Peak should be near 120 BPM (within 10 BPM; quantization from
        // rounding BPM -> integer delay can shift the peak slightly)
        XCTAssertGreaterThanOrEqual(peakBPM, 110, "Peak BPM should be >= 110")
        XCTAssertLessThanOrEqual(peakBPM, 130, "Peak BPM should be <= 130")
    }

    func testFilterBankPeakAt90BPM() {
        let fps: Float = 22050.0 / 512.0
        let periodFrames = Int(round(fps * 60.0 / 90.0))

        let nFrames = 600
        var signal = [Float](repeating: 0, count: nFrames)
        var idx = 0
        while idx < nFrames {
            signal[idx] = 1.0
            idx += periodFrames
        }

        let (tempos, energies) = CombFilter.tempoFilterBank(
            signal: signal,
            fps: fps,
            minBPM: 60,
            maxBPM: 200,
            bpmStep: 1.0,
            alpha: 0.99
        )

        var maxEnergy: Float = -1
        var peakBPM: Float = 0
        for i in 0..<tempos.count {
            if energies[i] > maxEnergy {
                maxEnergy = energies[i]
                peakBPM = tempos[i]
            }
        }

        // Peak should be near 90 BPM (within 5 BPM)
        XCTAssertGreaterThan(peakBPM, 85, "Peak BPM should be > 85")
        XCTAssertLessThan(peakBPM, 95, "Peak BPM should be < 95")
    }

    // MARK: - combFilterTempo with Click Track

    func testCombFilterTempoWithClickTrack() {
        // Create a click track at 120 BPM: short sine bursts at regular intervals
        let sr = 22050
        let duration: Double = 5.0
        let n = Int(Double(sr) * duration)
        var samples = [Float](repeating: 0, count: n)

        let interval = 60.0 / 120.0  // 0.5 seconds between clicks
        var t = 0.0
        while t < duration {
            let idx = Int(t * Double(sr))
            // Short burst of 1 kHz tone
            for j in 0..<256 where idx + j < n {
                samples[idx + j] = sin(Float(j) * 1000.0 * 2.0 * .pi / Float(sr)) * 0.9
            }
            t += interval
        }

        let signal = Signal(data: samples, sampleRate: sr)

        let tempo = TempoEstimation.combFilterTempo(
            signal: signal,
            sr: sr,
            minBPM: 60,
            maxBPM: 200
        )

        // Should detect tempo near 120 BPM (within 25% tolerance for comb filter method)
        XCTAssertGreaterThan(tempo, 90, "Tempo should be > 90 BPM for 120 BPM click track")
        XCTAssertLessThan(tempo, 160, "Tempo should be < 160 BPM for 120 BPM click track")
    }

    // MARK: - Alpha Parameter Effect

    func testAlphaParameterEffect() {
        let fps: Float = 22050.0 / 512.0
        let periodFrames = Int(round(fps * 60.0 / 120.0))

        let nFrames = 400
        var signal = [Float](repeating: 0, count: nFrames)
        var idx = 0
        while idx < nFrames {
            signal[idx] = 1.0
            idx += periodFrames
        }

        // High alpha should produce sharper peak (higher ratio of peak to mean)
        let (_, energiesHigh) = CombFilter.tempoFilterBank(
            signal: signal, fps: fps, minBPM: 60, maxBPM: 200, alpha: 0.99
        )
        let (_, energiesLow) = CombFilter.tempoFilterBank(
            signal: signal, fps: fps, minBPM: 60, maxBPM: 200, alpha: 0.5
        )

        // Compute peak-to-mean ratio for each
        let meanHigh = energiesHigh.reduce(0, +) / Float(energiesHigh.count)
        let peakHigh = energiesHigh.max() ?? 0
        let ratioHigh = peakHigh / max(meanHigh, 1e-10)

        let meanLow = energiesLow.reduce(0, +) / Float(energiesLow.count)
        let peakLow = energiesLow.max() ?? 0
        let ratioLow = peakLow / max(meanLow, 1e-10)

        XCTAssertGreaterThan(ratioHigh, ratioLow,
                             "Higher alpha should produce sharper tempo peak")
    }

    // MARK: - BPM Range

    func testBPMRange() {
        let sr = 22050
        let signal = Signal(data: [Float](repeating: 0.1, count: sr * 3), sampleRate: sr)

        let minBPM: Float = 60
        let maxBPM: Float = 200

        let tempo = TempoEstimation.combFilterTempo(
            signal: signal,
            sr: sr,
            minBPM: minBPM,
            maxBPM: maxBPM
        )

        XCTAssertGreaterThanOrEqual(tempo, minBPM, "Result should be >= minBPM")
        XCTAssertLessThanOrEqual(tempo, maxBPM, "Result should be <= maxBPM")
    }

    // MARK: - Short Signal

    func testShortSignalDoesNotCrash() {
        // Very short signal — should not crash, just return some value
        let signal = Signal(data: [0.5, 0.3, 0.1], sampleRate: 22050)

        let tempo = TempoEstimation.combFilterTempo(
            signal: signal,
            sr: 22050,
            minBPM: 60,
            maxBPM: 200
        )

        // Just check it returns a finite number in range
        XCTAssertGreaterThanOrEqual(tempo, 60)
        XCTAssertLessThanOrEqual(tempo, 200)
    }

    // MARK: - Energy Values Non-Negative

    func testEnergyValuesNonNegative() {
        // Random-ish signal
        let nFrames = 200
        var signal = [Float](repeating: 0, count: nFrames)
        for i in 0..<nFrames {
            signal[i] = sin(Float(i) * 0.1) * 0.5
        }

        let fps: Float = 22050.0 / 512.0
        let (_, energies) = CombFilter.tempoFilterBank(
            signal: signal, fps: fps, minBPM: 60, maxBPM: 200
        )

        for (i, energy) in energies.enumerated() {
            XCTAssertGreaterThanOrEqual(energy, 0,
                                        "Energy at index \(i) should be non-negative")
        }
    }
}
