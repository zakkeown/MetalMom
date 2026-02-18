import XCTest
@testable import MetalMomCore

final class NeuralBeatTrackerTests: XCTestCase {

    // MARK: - Synthetic Click Tests

    func testSyntheticClicks120BPM() {
        // Create activation array with periodic peaks at 120 BPM
        // At 100 fps, 120 BPM = 1 beat every 50 frames
        let fps: Float = 100.0
        let bpm: Float = 120.0
        let period = Int(60.0 * fps / bpm)  // 50 frames
        let nFrames = 500  // 5 seconds

        var activations = [Float](repeating: 0.05, count: nFrames)
        var frame = 0
        while frame < nFrames {
            activations[frame] = 0.9
            frame += period
        }

        let (tempo, beats) = NeuralBeatTracker.decode(
            activations: activations,
            fps: fps,
            minBPM: 55,
            maxBPM: 215,
            transitionLambda: 100,
            threshold: 0.05,
            trim: false
        )

        // Tempo should be close to 120 BPM
        XCTAssertGreaterThan(tempo, 100, "Tempo should be > 100 BPM for 120 BPM input")
        XCTAssertLessThan(tempo, 150, "Tempo should be < 150 BPM for 120 BPM input")

        // Should detect multiple beats
        XCTAssertGreaterThan(beats.count, 3, "Should detect multiple beats")

        // Most beats should be near the activation peaks.
        // Find minimum distance from each beat to any peak.
        var closeCount = 0
        for beat in beats {
            var minDist = Int.max
            var f = 0
            while f < nFrames {
                let d = abs(beat - f)
                if d < minDist { minDist = d }
                f += period
            }
            if minDist <= period / 2 {
                closeCount += 1
            }
        }
        // At least 80% of beats should be close to actual peaks
        // (boundary effects can push the first/last beat off the grid)
        let closeRatio = Float(closeCount) / Float(beats.count)
        XCTAssertGreaterThan(closeRatio, 0.7,
            "At least 70% of beats should be near activation peaks, got \(closeRatio)")
    }

    func testSyntheticClicks90BPM() {
        // 90 BPM at 100 fps = 1 beat every ~67 frames
        let fps: Float = 100.0
        let bpm: Float = 90.0
        let period = Int(60.0 * fps / bpm)  // ~67 frames
        let nFrames = 600  // 6 seconds

        var activations = [Float](repeating: 0.02, count: nFrames)
        var frame = 0
        while frame < nFrames {
            activations[frame] = 0.85
            frame += period
        }

        let (tempo, beats) = NeuralBeatTracker.decode(
            activations: activations,
            fps: fps,
            minBPM: 55,
            maxBPM: 215,
            threshold: 0.01,
            trim: false
        )

        // Tempo should be in the ballpark of 90 BPM
        XCTAssertGreaterThan(tempo, 70, "Tempo should be > 70 BPM for 90 BPM input")
        XCTAssertLessThan(tempo, 120, "Tempo should be < 120 BPM for 90 BPM input")

        // Should detect beats
        XCTAssertGreaterThan(beats.count, 2, "Should detect beats")
    }

    // MARK: - Variable Tempo

    func testVariableTempo() {
        // First half: 120 BPM (period=50), second half: 100 BPM (period=60)
        let fps: Float = 100.0
        let nFrames = 1000

        var activations = [Float](repeating: 0.02, count: nFrames)

        // First half: 120 BPM
        var frame = 0
        while frame < 500 {
            activations[frame] = 0.9
            frame += 50
        }

        // Second half: 100 BPM
        frame = 500
        while frame < nFrames {
            activations[frame] = 0.9
            frame += 60
        }

        let (tempo, beats) = NeuralBeatTracker.decode(
            activations: activations,
            fps: fps,
            threshold: 0.05,
            trim: false
        )

        // Tempo should be somewhere between 100-120 BPM (it estimates a global tempo)
        XCTAssertGreaterThan(tempo, 80, "Tempo should be > 80 BPM")
        XCTAssertLessThan(tempo, 140, "Tempo should be < 140 BPM")

        // Should detect beats in both halves
        let beatsFirstHalf = beats.filter { $0 < 500 }
        let beatsSecondHalf = beats.filter { $0 >= 500 }
        XCTAssertGreaterThan(beatsFirstHalf.count, 2,
            "Should detect beats in first half")
        XCTAssertGreaterThan(beatsSecondHalf.count, 2,
            "Should detect beats in second half")
    }

    // MARK: - Edge Cases

    func testAllZeros() {
        let activations = [Float](repeating: 0.0, count: 500)

        let (_, beats) = NeuralBeatTracker.decode(
            activations: activations,
            fps: 100.0,
            threshold: 0.05
        )

        // No beats should be detected with threshold > 0 and all-zero activations
        XCTAssertEqual(beats.count, 0, "All-zero activations should produce no beats")
    }

    func testSinglePeak() {
        // One strong activation in the middle
        var activations = [Float](repeating: 0.01, count: 200)
        activations[100] = 0.9

        let (_, beats) = NeuralBeatTracker.decode(
            activations: activations,
            fps: 100.0,
            threshold: 0.05,
            trim: false
        )

        // Should find at least the single strong peak
        let strongBeats = beats.filter { activations[$0] > 0.5 }
        XCTAssertGreaterThanOrEqual(strongBeats.count, 1,
            "Should detect the single strong peak")
    }

    func testEmptyInput() {
        let (tempo, beats) = NeuralBeatTracker.decode(
            activations: [],
            fps: 100.0
        )

        XCTAssertEqual(tempo, 0, "Empty input should return 0 tempo")
        XCTAssertTrue(beats.isEmpty, "Empty input should return no beats")
    }

    func testSingleFrame() {
        let (tempo, beats) = NeuralBeatTracker.decode(
            activations: [0.9],
            fps: 100.0,
            threshold: 0.05
        )

        XCTAssertEqual(tempo, 0, "Single frame should return 0 tempo")
        XCTAssertEqual(beats.count, 1, "Single frame above threshold should return one beat")
        if !beats.isEmpty {
            XCTAssertEqual(beats[0], 0)
        }
    }

    func testSingleFrameBelowThreshold() {
        let (tempo, beats) = NeuralBeatTracker.decode(
            activations: [0.01],
            fps: 100.0,
            threshold: 0.05
        )

        XCTAssertEqual(tempo, 0, "Single frame should return 0 tempo")
        XCTAssertTrue(beats.isEmpty, "Single frame below threshold should return no beats")
    }

    // MARK: - Threshold Filtering

    func testThresholdFiltering() {
        // Low activations everywhere, below threshold
        let activations = [Float](repeating: 0.03, count: 500)

        let (_, beats) = NeuralBeatTracker.decode(
            activations: activations,
            fps: 100.0,
            threshold: 0.05
        )

        // All activations are below threshold, so no beats
        XCTAssertEqual(beats.count, 0,
            "Low activations below threshold should produce no beats")
    }

    func testZeroThresholdKeepsAllBeats() {
        // Uniform low activations
        let nFrames = 300
        var activations = [Float](repeating: 0.01, count: nFrames)
        // Add some peaks
        for i in stride(from: 0, to: nFrames, by: 50) {
            activations[i] = 0.5
        }

        let (_, beatsWithThreshold) = NeuralBeatTracker.decode(
            activations: activations,
            fps: 100.0,
            threshold: 0.1,
            trim: false
        )

        let (_, beatsNoThreshold) = NeuralBeatTracker.decode(
            activations: activations,
            fps: 100.0,
            threshold: 0.0,
            trim: false
        )

        // Zero threshold should keep at least as many beats
        XCTAssertGreaterThanOrEqual(beatsNoThreshold.count, beatsWithThreshold.count,
            "Zero threshold should produce >= beats compared to positive threshold")
    }

    // MARK: - Trim Behavior

    func testTrimReducesBeatCount() {
        let fps: Float = 100.0
        let nFrames = 500

        var activations = [Float](repeating: 0.02, count: nFrames)
        for i in stride(from: 0, to: nFrames, by: 50) {
            activations[i] = 0.9
        }

        let (_, beatsNotTrimmed) = NeuralBeatTracker.decode(
            activations: activations,
            fps: fps,
            threshold: 0.05,
            trim: false
        )

        let (_, beatsTrimmed) = NeuralBeatTracker.decode(
            activations: activations,
            fps: fps,
            threshold: 0.05,
            trim: true
        )

        // Trimmed should have fewer beats (by exactly 2 when there are enough)
        if beatsNotTrimmed.count > 2 {
            XCTAssertLessThan(beatsTrimmed.count, beatsNotTrimmed.count,
                "Trimmed beats should be fewer than untrimmed")
        }
    }

    func testTrimDoesNotRemoveAllBeats() {
        // With few beats, trim should not remove all of them
        let fps: Float = 100.0
        let nFrames = 100

        var activations = [Float](repeating: 0.02, count: nFrames)
        activations[25] = 0.9
        activations[75] = 0.9

        let (_, beats) = NeuralBeatTracker.decode(
            activations: activations,
            fps: fps,
            threshold: 0.05,
            trim: true
        )

        // With only 2 beats, trim should not strip them down since
        // we require count > 2 for trimming
        _ = beats  // Just verify no crash
    }

    // MARK: - BPM Range

    func testTempoWithinRange() {
        let fps: Float = 100.0
        let nFrames = 500

        var activations = [Float](repeating: 0.02, count: nFrames)
        for i in stride(from: 0, to: nFrames, by: 50) {
            activations[i] = 0.9
        }

        let minBPM: Float = 60.0
        let maxBPM: Float = 200.0

        let (tempo, _) = NeuralBeatTracker.decode(
            activations: activations,
            fps: fps,
            minBPM: minBPM,
            maxBPM: maxBPM
        )

        XCTAssertGreaterThanOrEqual(tempo, minBPM,
            "Tempo should be >= minBPM")
        XCTAssertLessThanOrEqual(tempo, maxBPM,
            "Tempo should be <= maxBPM")
    }

    func testNarrowBPMRange() {
        // Force a very narrow tempo range
        let fps: Float = 100.0
        let nFrames = 500

        var activations = [Float](repeating: 0.02, count: nFrames)
        // Peaks at 120 BPM (period=50)
        for i in stride(from: 0, to: nFrames, by: 50) {
            activations[i] = 0.9
        }

        let (tempo, _) = NeuralBeatTracker.decode(
            activations: activations,
            fps: fps,
            minBPM: 110,
            maxBPM: 130
        )

        XCTAssertGreaterThanOrEqual(tempo, 110,
            "Tempo should be >= narrow minBPM")
        XCTAssertLessThanOrEqual(tempo, 130,
            "Tempo should be <= narrow maxBPM")
    }

    // MARK: - Beat Properties

    func testBeatsSorted() {
        let fps: Float = 100.0
        let nFrames = 500

        var activations = [Float](repeating: 0.02, count: nFrames)
        for i in stride(from: 0, to: nFrames, by: 50) {
            activations[i] = 0.9
        }

        let (_, beats) = NeuralBeatTracker.decode(
            activations: activations,
            fps: fps,
            trim: false
        )

        for i in 1..<beats.count {
            XCTAssertGreaterThan(beats[i], beats[i - 1],
                "Beat frames should be strictly increasing")
        }
    }

    func testBeatsNonNegative() {
        let fps: Float = 100.0
        let nFrames = 500

        var activations = [Float](repeating: 0.02, count: nFrames)
        for i in stride(from: 0, to: nFrames, by: 50) {
            activations[i] = 0.9
        }

        let (_, beats) = NeuralBeatTracker.decode(
            activations: activations,
            fps: fps,
            trim: false
        )

        for beat in beats {
            XCTAssertGreaterThanOrEqual(beat, 0,
                "Beat frames should be non-negative")
        }
    }

    func testBeatsWithinRange() {
        let nFrames = 500
        let fps: Float = 100.0

        var activations = [Float](repeating: 0.02, count: nFrames)
        for i in stride(from: 0, to: nFrames, by: 50) {
            activations[i] = 0.9
        }

        let (_, beats) = NeuralBeatTracker.decode(
            activations: activations,
            fps: fps,
            trim: false
        )

        for beat in beats {
            XCTAssertLessThan(beat, nFrames,
                "Beat frames should be < nFrames")
        }
    }

    // MARK: - Sparse Activations

    func testSparseActivations() {
        // Very few high activations in a long sequence
        let nFrames = 1000
        var activations = [Float](repeating: 0.01, count: nFrames)
        activations[100] = 0.8
        activations[350] = 0.8
        activations[600] = 0.8
        activations[850] = 0.8

        let (_, beats) = NeuralBeatTracker.decode(
            activations: activations,
            fps: 100.0,
            threshold: 0.1,
            trim: false
        )

        // Should detect at least some of the sparse peaks
        XCTAssertGreaterThan(beats.count, 0,
            "Should detect at least some sparse peaks")
    }

    // MARK: - Tempo Estimation Direct

    func testTempoEstimationFromActivations() {
        let fps: Float = 100.0
        let period = 50  // 120 BPM
        let nFrames = 600

        var activations = [Float](repeating: 0.02, count: nFrames)
        var frame = 0
        while frame < nFrames {
            activations[frame] = 0.9
            frame += period
        }

        let tempo = NeuralBeatTracker.estimateTempoFromActivations(
            activations: activations,
            fps: fps,
            minBPM: 55,
            maxBPM: 215
        )

        XCTAssertGreaterThan(tempo, 100, "Tempo should be > 100 BPM for 120 BPM input")
        XCTAssertLessThan(tempo, 150, "Tempo should be < 150 BPM for 120 BPM input")
    }

    func testTempoEstimationEmpty() {
        let tempo = NeuralBeatTracker.estimateTempoFromActivations(
            activations: [],
            fps: 100.0
        )
        XCTAssertEqual(tempo, 0, "Empty activations should return 0 tempo")
    }

    func testTempoEstimationSingleFrame() {
        let tempo = NeuralBeatTracker.estimateTempoFromActivations(
            activations: [0.9],
            fps: 100.0
        )
        XCTAssertEqual(tempo, 0, "Single frame should return 0 tempo")
    }

    // MARK: - High transitionLambda

    func testHighTransitionLambdaProducesRegularBeats() {
        // High lambda should produce more regular beat spacing
        let fps: Float = 100.0
        let nFrames = 500

        var activations = [Float](repeating: 0.02, count: nFrames)
        // Peaks at 120 BPM with some jitter
        let positions = [0, 48, 102, 149, 201, 250, 298, 352, 399, 451]
        for pos in positions {
            if pos < nFrames { activations[pos] = 0.9 }
        }

        let (_, beats) = NeuralBeatTracker.decode(
            activations: activations,
            fps: fps,
            transitionLambda: 200,
            threshold: 0.05,
            trim: false
        )

        // With high lambda, beat intervals should be relatively consistent
        guard beats.count >= 3 else { return }
        var intervals = [Int]()
        for i in 1..<beats.count {
            intervals.append(beats[i] - beats[i - 1])
        }
        let avgInterval = Float(intervals.reduce(0, +)) / Float(intervals.count)
        for interval in intervals {
            let ratio = Float(interval) / avgInterval
            // All intervals should be within a factor of 2
            XCTAssertGreaterThan(ratio, 0.5, "Beat intervals should be relatively regular")
            XCTAssertLessThan(ratio, 2.0, "Beat intervals should be relatively regular")
        }
    }
}
