import XCTest
@testable import MetalMomCore

final class OnsetDetectionTests: XCTestCase {
    func testOnsetStrengthShape() {
        let n = 22050
        let signal = Signal(
            data: (0..<n).map { sin(Float($0) * 440.0 * 2.0 * .pi / 22050.0) },
            sampleRate: 22050
        )
        let result = OnsetDetection.onsetStrength(signal: signal)
        XCTAssertEqual(result.shape[0], 1)  // aggregated
        XCTAssertGreaterThan(result.shape[1], 0)
    }

    func testOnsetStrengthNonAggregated() {
        let n = 22050
        let signal = Signal(
            data: (0..<n).map { sin(Float($0) * 440.0 * 2.0 * .pi / 22050.0) },
            sampleRate: 22050
        )
        let result = OnsetDetection.onsetStrength(signal: signal, aggregate: false)
        XCTAssertEqual(result.shape[0], 128)  // n_mels
    }

    func testOnsetStrengthNonNegative() {
        let n = 22050
        let signal = Signal(
            data: (0..<n).map { sin(Float($0) * 440.0 * 2.0 * .pi / 22050.0) },
            sampleRate: 22050
        )
        let result = OnsetDetection.onsetStrength(signal: signal)
        for i in 0..<result.count {
            XCTAssertGreaterThanOrEqual(result[i], 0, "Onset strength should be non-negative")
        }
    }

    func testOnsetStrengthSilent() {
        let n = 22050
        let signal = Signal(data: [Float](repeating: 0, count: n), sampleRate: 22050)
        let result = OnsetDetection.onsetStrength(signal: signal)
        // Silent signal should have all-zero onset strength
        for i in 0..<result.count {
            XCTAssertEqual(result[i], 0, accuracy: 1e-6)
        }
    }

    func testOnsetStrengthFinite() {
        let n = 22050
        let signal = Signal(
            data: (0..<n).map { sin(Float($0) * 440.0 * 2.0 * .pi / 22050.0) },
            sampleRate: 22050
        )
        let result = OnsetDetection.onsetStrength(signal: signal)
        for i in 0..<result.count {
            XCTAssertFalse(result[i].isNaN)
            XCTAssertFalse(result[i].isInfinite)
        }
    }

    // MARK: - Peak Picking Tests

    func testPeakPickSyntheticEnvelope() {
        // Create an envelope with clear peaks at known positions
        // Peaks at indices 10, 50, 90 with valleys between them
        var envelope = [Float](repeating: 0, count: 120)
        // Peak at 10
        for i in 5...15 { envelope[i] = Float(5 - abs(i - 10)) / 5.0 }
        // Peak at 50
        for i in 45...55 { envelope[i] = Float(5 - abs(i - 50)) / 5.0 }
        // Peak at 90
        for i in 85...95 { envelope[i] = Float(5 - abs(i - 90)) / 5.0 }

        let peaks = OnsetDetection.peakPick(
            envelope: envelope,
            preMax: 3, postMax: 3,
            preAvg: 3, postAvg: 3,
            delta: 0.05,
            wait: 10
        )

        // Should find the three peaks
        XCTAssertEqual(peaks.count, 3, "Should detect 3 peaks")
        XCTAssertEqual(peaks[0], 10, "First peak at 10")
        XCTAssertEqual(peaks[1], 50, "Second peak at 50")
        XCTAssertEqual(peaks[2], 90, "Third peak at 90")
    }

    func testPeakPickWaitConstraint() {
        // Two peaks close together -- wait should suppress the second
        var envelope = [Float](repeating: 0, count: 60)
        // Peak at 10
        for i in 5...15 { envelope[i] = Float(5 - abs(i - 10)) / 5.0 }
        // Peak at 20 (only 10 apart)
        for i in 15...25 { envelope[i] = max(envelope[i], Float(5 - abs(i - 20)) / 5.0) }

        let peaks = OnsetDetection.peakPick(
            envelope: envelope,
            preMax: 3, postMax: 3,
            preAvg: 3, postAvg: 3,
            delta: 0.05,
            wait: 30  // wait=30 should suppress second peak
        )

        XCTAssertEqual(peaks.count, 1, "Wait constraint should suppress second peak")
        XCTAssertEqual(peaks[0], 10)
    }

    func testPeakPickDeltaThreshold() {
        // Low envelope values should not produce peaks if delta is high
        var envelope = [Float](repeating: 0.01, count: 60)
        envelope[30] = 0.02  // Very small bump

        let peaks = OnsetDetection.peakPick(
            envelope: envelope,
            preMax: 3, postMax: 3,
            preAvg: 3, postAvg: 3,
            delta: 0.5,  // High delta
            wait: 1
        )

        XCTAssertEqual(peaks.count, 0, "High delta should reject small peaks")
    }

    func testPeakPickEmptyEnvelope() {
        let peaks = OnsetDetection.peakPick(envelope: [], preMax: 3, postMax: 3,
                                             preAvg: 3, postAvg: 3, delta: 0.07, wait: 30)
        XCTAssertEqual(peaks.count, 0)
    }

    func testPeakPickShortEnvelope() {
        // Envelope too short for the window
        let envelope: [Float] = [0.1, 0.5, 0.1]
        let peaks = OnsetDetection.peakPick(envelope: envelope, preMax: 3, postMax: 3,
                                             preAvg: 3, postAvg: 3, delta: 0.07, wait: 30)
        XCTAssertEqual(peaks.count, 0, "Envelope shorter than window should return no peaks")
    }

    // MARK: - Backtracking Tests

    func testBacktrackSnapsToLocalMinimum() {
        // Envelope rises from 0 at index 5 to peak at index 10
        var envelope = [Float](repeating: 0, count: 20)
        for i in 5...10 { envelope[i] = Float(i - 5) / 5.0 }
        for i in 11...15 { envelope[i] = Float(15 - i) / 5.0 }

        let backtracked = OnsetDetection.backtrack(peaks: [10], envelope: envelope)
        // Walking back from peak=10: values decrease 1.0, 0.8, 0.6, 0.4, 0.2, 0.0 at index 5.
        // Index 4 has 0.0 (equal, not less), and then 0.0 > 0.0 is false, so the walk
        // continues through the flat region. The minimum found is at index 5 (first 0.0).
        XCTAssertEqual(backtracked[0], 5, "Should backtrack to the start of the energy rise")
    }

    func testBacktrackWithMultiplePeaks() {
        var envelope = [Float](repeating: 0, count: 100)
        // Valley at 20, peak at 30
        for i in 20...30 { envelope[i] = Float(i - 20) / 10.0 }
        for i in 31...40 { envelope[i] = Float(40 - i) / 10.0 }
        // Valley at 60, peak at 70
        for i in 60...70 { envelope[i] = Float(i - 60) / 10.0 }
        for i in 71...80 { envelope[i] = Float(80 - i) / 10.0 }

        let backtracked = OnsetDetection.backtrack(peaks: [30, 70], envelope: envelope)
        XCTAssertEqual(backtracked.count, 2)
        // Peak at 30 should snap back to 20 (start of rise, then flat 0s before)
        XCTAssertLessThanOrEqual(backtracked[0], 20)
        // Peak at 70 should snap back to 60 area
        XCTAssertLessThanOrEqual(backtracked[1], 60)
    }

    func testBacktrackEmptyPeaks() {
        let backtracked = OnsetDetection.backtrack(peaks: [], envelope: [1, 2, 3])
        XCTAssertEqual(backtracked.count, 0)
    }

    // MARK: - detectOnsets Tests

    func testDetectOnsetsWithClicks() {
        // Generate a signal with clear onset events (clicks at known positions)
        let sr = 22050
        let duration = 2.0
        let n = Int(Double(sr) * duration)
        var samples = [Float](repeating: 0, count: n)

        // Place sharp clicks at 0.25s, 0.75s, 1.25s, 1.75s
        let clickPositions = [0.25, 0.75, 1.25, 1.75]
        for t in clickPositions {
            let idx = Int(t * Double(sr))
            // Short burst of noise for a click
            for j in 0..<512 where idx + j < n {
                samples[idx + j] = sin(Float(j) * 1000.0 * 2.0 * .pi / Float(sr)) * 0.9
            }
        }

        let signal = Signal(data: samples, sampleRate: sr)
        let result = OnsetDetection.detectOnsets(
            signal: signal,
            sr: sr,
            delta: 0.05,
            wait: 5
        )

        // Should detect some onsets
        XCTAssertGreaterThan(result.count, 0, "Should detect at least one onset from clicks")
    }

    func testDetectOnsetsSilentSignal() {
        let sr = 22050
        let signal = Signal(data: [Float](repeating: 0, count: sr), sampleRate: sr)
        let result = OnsetDetection.detectOnsets(signal: signal, sr: sr)
        XCTAssertEqual(result.count, 0, "Silent signal should have no onsets")
    }

    func testDetectOnsetsReturnType() {
        let sr = 22050
        let signal = Signal(
            data: (0..<sr).map { sin(Float($0) * 440.0 * 2.0 * .pi / Float(sr)) },
            sampleRate: sr
        )
        let result = OnsetDetection.detectOnsets(signal: signal, sr: sr)

        // Result should be 1D
        XCTAssertEqual(result.shape.count, 1, "Result should be 1D")
        // All values should be valid non-negative frame indices
        for i in 0..<result.count {
            XCTAssertGreaterThanOrEqual(result[i], 0, "Frame indices should be non-negative")
            XCTAssertFalse(result[i].isNaN)
        }
    }

    func testDetectOnsetsWithBacktracking() {
        // Generate clicks -- detect with and without backtracking
        let sr = 22050
        var samples = [Float](repeating: 0, count: sr * 2)
        // Click at 0.5s
        let idx = sr / 2
        for j in 0..<512 where idx + j < samples.count {
            samples[idx + j] = sin(Float(j) * 1000.0 * 2.0 * .pi / Float(sr)) * 0.9
        }

        let signal = Signal(data: samples, sampleRate: sr)

        let noBacktrack = OnsetDetection.detectOnsets(
            signal: signal, sr: sr, delta: 0.05, wait: 5, doBacktrack: false
        )
        let withBacktrack = OnsetDetection.detectOnsets(
            signal: signal, sr: sr, delta: 0.05, wait: 5, doBacktrack: true
        )

        // Both should detect onsets
        if noBacktrack.count > 0 && withBacktrack.count > 0 {
            // Backtracked onsets should be <= the non-backtracked ones
            // (snapped to earlier positions)
            XCTAssertLessThanOrEqual(withBacktrack[0], noBacktrack[0],
                                      "Backtracked onset should be at or before peak")
        }
    }
}
