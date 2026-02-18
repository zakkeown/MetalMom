import XCTest
@testable import MetalMomCore

final class NeuralOnsetDetectorTests: XCTestCase {

    // MARK: - Synthetic Pulses

    func testSyntheticPulses() {
        // Clear peaks at known positions
        var activations = [Float](repeating: 0.05, count: 100)
        activations[10] = 0.9
        activations[30] = 0.8
        activations[50] = 0.85
        activations[70] = 0.95
        activations[90] = 0.7

        let onsets = NeuralOnsetDetector.detect(
            activations: activations,
            threshold: 0.2,
            preMax: 2,
            postMax: 2,
            preAvg: 3,
            postAvg: 3,
            combineMethod: .adaptive,
            wait: 5
        )

        // Should find all 5 peaks
        XCTAssertEqual(onsets.count, 5, "Should detect all 5 synthetic peaks")
        XCTAssertTrue(onsets.contains(10), "Should detect peak at 10")
        XCTAssertTrue(onsets.contains(30), "Should detect peak at 30")
        XCTAssertTrue(onsets.contains(50), "Should detect peak at 50")
        XCTAssertTrue(onsets.contains(70), "Should detect peak at 70")
        XCTAssertTrue(onsets.contains(90), "Should detect peak at 90")
    }

    // MARK: - Threshold Filtering

    func testThresholdFiltersBelowThreshold() {
        // Some peaks above and some below threshold
        var activations = [Float](repeating: 0.0, count: 50)
        activations[10] = 0.8   // above
        activations[20] = 0.1   // below (for fixed at 0.5)
        activations[30] = 0.9   // above
        activations[40] = 0.15  // below

        let onsets = NeuralOnsetDetector.detect(
            activations: activations,
            threshold: 0.5,
            preMax: 1,
            postMax: 1,
            combineMethod: .fixed,
            wait: 1
        )

        // Only positions 10 and 30 should be detected (above 0.5 threshold)
        XCTAssertEqual(onsets.count, 2, "Should only detect peaks above threshold")
        XCTAssertTrue(onsets.contains(10))
        XCTAssertTrue(onsets.contains(30))
    }

    // MARK: - Wait Constraint

    func testWaitConstraint() {
        // Two peaks very close together
        var activations = [Float](repeating: 0.0, count: 30)
        activations[5] = 0.9
        activations[7] = 0.85  // Only 2 frames away

        // With wait=5, second peak should be suppressed
        let onsets = NeuralOnsetDetector.detect(
            activations: activations,
            threshold: 0.3,
            preMax: 1,
            postMax: 1,
            combineMethod: .fixed,
            wait: 5
        )

        XCTAssertEqual(onsets.count, 1, "Wait constraint should suppress second peak")
        XCTAssertEqual(onsets[0], 5, "Should keep the first peak")
    }

    func testWaitConstraintAllowsDistantPeaks() {
        var activations = [Float](repeating: 0.0, count: 30)
        activations[5] = 0.9
        activations[15] = 0.85  // 10 frames away

        let onsets = NeuralOnsetDetector.detect(
            activations: activations,
            threshold: 0.3,
            preMax: 1,
            postMax: 1,
            combineMethod: .fixed,
            wait: 5
        )

        XCTAssertEqual(onsets.count, 2, "Should detect both peaks when far enough apart")
    }

    // MARK: - Smooth Activations

    func testSmooth() {
        let activations: [Float] = [0, 0, 0, 1, 0, 0, 0]
        let smoothed = NeuralOnsetDetector.smooth(activations: activations, width: 3)

        XCTAssertEqual(smoothed.count, 7, "Smoothed should have same length")
        // Center element (index 3) should be the average of [0, 1, 0] = 0.333...
        XCTAssertEqual(smoothed[3], 1.0 / 3.0, accuracy: 1e-6,
            "Center of impulse should be smoothed to ~0.333")
        // Index 2: average of [0, 0, 1] = 0.333...
        XCTAssertEqual(smoothed[2], 1.0 / 3.0, accuracy: 1e-6)
        // Index 4: average of [1, 0, 0] = 0.333...
        XCTAssertEqual(smoothed[4], 1.0 / 3.0, accuracy: 1e-6)
        // Index 0: average of [0, 0] = 0.0
        XCTAssertEqual(smoothed[0], 0.0, accuracy: 1e-6)
        // Index 6: average of [0, 0] = 0.0
        XCTAssertEqual(smoothed[6], 0.0, accuracy: 1e-6)
    }

    func testSmoothWidth1ReturnsOriginal() {
        let activations: [Float] = [0.1, 0.5, 0.3, 0.8, 0.2]
        let smoothed = NeuralOnsetDetector.smooth(activations: activations, width: 1)

        for i in 0..<activations.count {
            XCTAssertEqual(smoothed[i], activations[i], accuracy: 1e-6,
                "Width 1 should return original values")
        }
    }

    func testSmoothEmptyInput() {
        let smoothed = NeuralOnsetDetector.smooth(activations: [], width: 3)
        XCTAssertTrue(smoothed.isEmpty, "Smoothing empty input should return empty")
    }

    // MARK: - All Zeros

    func testAllZeros() {
        let activations = [Float](repeating: 0.0, count: 100)

        let onsetsFixed = NeuralOnsetDetector.detect(
            activations: activations,
            threshold: 0.3,
            combineMethod: .fixed
        )
        let onsetsAdaptive = NeuralOnsetDetector.detect(
            activations: activations,
            threshold: 0.3,
            combineMethod: .adaptive
        )
        let onsetsCombined = NeuralOnsetDetector.detect(
            activations: activations,
            threshold: 0.3,
            combineMethod: .combined
        )

        XCTAssertEqual(onsetsFixed.count, 0, "All zeros: fixed should detect nothing")
        XCTAssertEqual(onsetsAdaptive.count, 0, "All zeros: adaptive should detect nothing")
        XCTAssertEqual(onsetsCombined.count, 0, "All zeros: combined should detect nothing")
    }

    // MARK: - All Ones (Flat Activation)

    func testAllOnes() {
        // Flat activation -- nothing is a local max when everything is equal.
        // isLocalMax returns true for the first candidate if all values are equal,
        // but adaptive threshold mean+delta will filter it out when delta > 0.
        let activations = [Float](repeating: 1.0, count: 100)

        let onsetsAdaptive = NeuralOnsetDetector.detect(
            activations: activations,
            threshold: 0.3,
            preMax: 1,
            postMax: 1,
            preAvg: 3,
            postAvg: 3,
            combineMethod: .adaptive,
            wait: 1
        )

        // With adaptive threshold, mean + delta > 1.0 (since max=1.0, delta=0.3*1.0=0.3,
        // mean=1.0, so 1.0 < 1.0+0.3), so nothing should pass.
        XCTAssertEqual(onsetsAdaptive.count, 0,
            "Flat activations should produce no onsets with adaptive thresholding")
    }

    // MARK: - Single Peak

    func testSinglePeak() {
        var activations = [Float](repeating: 0.0, count: 50)
        activations[25] = 0.9

        let onsets = NeuralOnsetDetector.detect(
            activations: activations,
            threshold: 0.2,
            preMax: 1,
            postMax: 1,
            combineMethod: .fixed,
            wait: 1
        )

        XCTAssertEqual(onsets.count, 1, "Should detect exactly one onset")
        XCTAssertEqual(onsets[0], 25, "Onset should be at position 25")
    }

    // MARK: - Adaptive vs Fixed

    func testAdaptiveVsFixed() {
        // Create activations where adaptive should differ from fixed.
        // Background noise at 0.4, with a peak at 0.5.
        // Fixed threshold at 0.3 will detect the peak (0.5 > 0.3).
        // Adaptive threshold: mean~0.4, delta=0.3*0.5=0.15, so threshold=0.55.
        // The peak at 0.5 < 0.55, so adaptive should NOT detect it.
        var activations = [Float](repeating: 0.4, count: 20)
        activations[10] = 0.5

        let onsetsFixed = NeuralOnsetDetector.detect(
            activations: activations,
            threshold: 0.3,
            preMax: 1,
            postMax: 1,
            combineMethod: .fixed,
            wait: 1
        )

        let onsetsAdaptive = NeuralOnsetDetector.detect(
            activations: activations,
            threshold: 0.3,
            preMax: 1,
            postMax: 1,
            preAvg: 3,
            postAvg: 3,
            combineMethod: .adaptive,
            wait: 1
        )

        // Fixed should detect the peak at 10
        XCTAssertGreaterThanOrEqual(onsetsFixed.count, 1,
            "Fixed threshold should detect the peak")

        // Adaptive should not detect it (peak is only slightly above background)
        XCTAssertEqual(onsetsAdaptive.count, 0,
            "Adaptive threshold should not detect peak barely above background")
    }

    func testCombinedRequiresBothThresholds() {
        // Create a peak that passes fixed but fails adaptive
        var activations = [Float](repeating: 0.4, count: 20)
        activations[10] = 0.5  // Above 0.3 fixed, but not above adaptive

        let onsetsCombined = NeuralOnsetDetector.detect(
            activations: activations,
            threshold: 0.3,
            preMax: 1,
            postMax: 1,
            preAvg: 3,
            postAvg: 3,
            combineMethod: .combined,
            wait: 1
        )

        XCTAssertEqual(onsetsCombined.count, 0,
            "Combined method should reject peaks that fail adaptive threshold")
    }

    // MARK: - Frame to Time Conversion

    func testFramesToTimes() {
        let frames = [0, 10, 50, 100]
        let fps: Float = 100.0

        let times = NeuralOnsetDetector.framesToTimes(frames: frames, fps: fps)

        XCTAssertEqual(times.count, 4)
        XCTAssertEqual(times[0], 0.0, accuracy: 1e-6)
        XCTAssertEqual(times[1], 0.1, accuracy: 1e-6)
        XCTAssertEqual(times[2], 0.5, accuracy: 1e-6)
        XCTAssertEqual(times[3], 1.0, accuracy: 1e-6)
    }

    func testFramesToTimesEmpty() {
        let times = NeuralOnsetDetector.framesToTimes(frames: [], fps: 100.0)
        XCTAssertTrue(times.isEmpty)
    }

    // MARK: - Frame to Sample Conversion

    func testFramesToSamples() {
        let frames = [0, 5, 10, 20]
        let hopLength = 512

        let samples = NeuralOnsetDetector.framesToSamples(frames: frames, hopLength: hopLength)

        XCTAssertEqual(samples.count, 4)
        XCTAssertEqual(samples[0], 0)
        XCTAssertEqual(samples[1], 2560)
        XCTAssertEqual(samples[2], 5120)
        XCTAssertEqual(samples[3], 10240)
    }

    func testFramesToSamplesEmpty() {
        let samples = NeuralOnsetDetector.framesToSamples(frames: [], hopLength: 512)
        XCTAssertTrue(samples.isEmpty)
    }

    // MARK: - Local Max Check

    func testLocalMaxCheck() {
        // Peak must be strictly >= neighbors within the window
        var activations = [Float](repeating: 0.0, count: 20)
        activations[5] = 0.5
        activations[6] = 0.9  // This is the local max
        activations[7] = 0.5

        let onsets = NeuralOnsetDetector.detect(
            activations: activations,
            threshold: 0.3,
            preMax: 2,
            postMax: 2,
            combineMethod: .fixed,
            wait: 1
        )

        // Only position 6 should be detected (the local maximum)
        XCTAssertEqual(onsets.count, 1, "Should detect only the local maximum")
        XCTAssertEqual(onsets[0], 6, "Local max should be at position 6")
    }

    func testNoLocalMaxInPlateau() {
        // A small plateau (3 equal values). The first one where isLocalMax
        // returns true should be detected if it passes threshold.
        var activations = [Float](repeating: 0.0, count: 20)
        activations[8] = 0.8
        activations[9] = 0.8
        activations[10] = 0.8

        let onsets = NeuralOnsetDetector.detect(
            activations: activations,
            threshold: 0.3,
            preMax: 1,
            postMax: 1,
            combineMethod: .fixed,
            wait: 1
        )

        // With preMax=1, postMax=1: position 8 has window [7,9], all <= 0.8: local max.
        // Position 9 has window [8,10], all equal: local max too.
        // Position 10 has window [9,11], values [0.8, 0.0]: local max too.
        // With wait=1, all three could be detected... but the first is at 8.
        // Actually, 8 is local max (0.8 >= 0.0 and 0.8), 9 is (0.8 >= 0.8 and 0.8),
        // 10 is (0.8 >= 0.8 and 0.0). wait=1 allows all.
        // But that's fine -- the test just verifies at least one peak is found.
        XCTAssertGreaterThanOrEqual(onsets.count, 1,
            "Should detect at least one onset in plateau")
    }

    // MARK: - Empty Input

    func testEmptyInput() {
        let onsets = NeuralOnsetDetector.detect(
            activations: [],
            threshold: 0.3,
            combineMethod: .adaptive
        )
        XCTAssertTrue(onsets.isEmpty, "Empty input should return no onsets")
    }

    // MARK: - Onsets are Sorted

    func testOnsetsSorted() {
        var activations = [Float](repeating: 0.0, count: 100)
        activations[15] = 0.9
        activations[35] = 0.8
        activations[55] = 0.85
        activations[75] = 0.7

        let onsets = NeuralOnsetDetector.detect(
            activations: activations,
            threshold: 0.2,
            preMax: 1,
            postMax: 1,
            combineMethod: .fixed,
            wait: 1
        )

        for i in 1..<onsets.count {
            XCTAssertGreaterThan(onsets[i], onsets[i - 1],
                "Onset frames should be strictly increasing")
        }
    }

    // MARK: - Onsets Within Range

    func testOnsetsWithinRange() {
        let nFrames = 80
        var activations = [Float](repeating: 0.0, count: nFrames)
        activations[10] = 0.9
        activations[40] = 0.8
        activations[70] = 0.85

        let onsets = NeuralOnsetDetector.detect(
            activations: activations,
            threshold: 0.2,
            preMax: 1,
            postMax: 1,
            combineMethod: .fixed,
            wait: 1
        )

        for onset in onsets {
            XCTAssertGreaterThanOrEqual(onset, 0, "Onset should be >= 0")
            XCTAssertLessThan(onset, nFrames, "Onset should be < nFrames")
        }
    }

    // MARK: - Large Wait Suppresses All But First

    func testLargeWaitSuppressesAll() {
        var activations = [Float](repeating: 0.0, count: 50)
        activations[5] = 0.9
        activations[10] = 0.8
        activations[15] = 0.85
        activations[20] = 0.7

        let onsets = NeuralOnsetDetector.detect(
            activations: activations,
            threshold: 0.3,
            preMax: 1,
            postMax: 1,
            combineMethod: .fixed,
            wait: 100  // Larger than entire array
        )

        XCTAssertEqual(onsets.count, 1,
            "Large wait should suppress all peaks except the first")
        XCTAssertEqual(onsets[0], 5, "Should keep first peak")
    }

    // MARK: - Smooth Then Detect

    func testSmoothThenDetect() {
        // Noisy activations with a clear peak region
        var activations = [Float](repeating: 0.1, count: 50)
        // Create a broad peak region
        activations[20] = 0.3
        activations[21] = 0.5
        activations[22] = 0.9
        activations[23] = 0.5
        activations[24] = 0.3

        let smoothed = NeuralOnsetDetector.smooth(activations: activations, width: 3)

        // The peak should still be near position 22 after smoothing
        let maxIdx = smoothed.enumerated().max(by: { $0.element < $1.element })!.offset
        XCTAssertEqual(maxIdx, 22, "Peak of smoothed signal should be at original peak position")

        // Detect from smoothed
        let onsets = NeuralOnsetDetector.detect(
            activations: smoothed,
            threshold: 0.2,
            preMax: 2,
            postMax: 2,
            combineMethod: .fixed,
            wait: 1
        )

        XCTAssertGreaterThanOrEqual(onsets.count, 1, "Should detect at least one onset")
        // The onset should be near position 22
        let closest = onsets.min(by: { abs($0 - 22) < abs($1 - 22) })!
        assertApproxEqual(closest, 22, accuracy: 2,
            "Detected onset should be near the original peak")
    }

    // Helper for approximate Int equality
    func assertApproxEqual(_ a: Int, _ b: Int, accuracy: Int, _ msg: String) {
        XCTAssertTrue(abs(a - b) <= accuracy, "\(msg): \(a) vs \(b)")
    }
}
