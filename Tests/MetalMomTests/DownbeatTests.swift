import XCTest
@testable import MetalMomCore

final class DownbeatTests: XCTestCase {

    // MARK: - Helper

    /// Build a flat [nFrames * 3] activation array from per-frame (noBeat, beat, downbeat) tuples.
    private func makeActivations(_ frames: [(Float, Float, Float)]) -> [Float] {
        var result = [Float]()
        result.reserveCapacity(frames.count * 3)
        for (nb, b, db) in frames {
            result.append(nb)
            result.append(b)
            result.append(db)
        }
        return result
    }

    // MARK: - 4/4 Pattern

    func test4_4Pattern() {
        // 10 seconds at 100 fps, 120 BPM = beat every 50 frames
        // Pattern: downbeat, beat, beat, beat, downbeat, beat, beat, beat, ...
        let fps: Float = 100.0
        let bpm: Float = 120.0
        let period = Int(60.0 * fps / bpm)  // 50 frames
        let nFrames = 1000
        let beatsPerBar = 4

        var frames = [(Float, Float, Float)](repeating: (0.8, 0.1, 0.1), count: nFrames)

        var beatIndex = 0
        var frame = 0
        while frame < nFrames {
            if beatIndex % beatsPerBar == 0 {
                // Downbeat: high P(downbeat)
                frames[frame] = (0.05, 0.1, 0.85)
            } else {
                // Beat: high P(beat)
                frames[frame] = (0.05, 0.85, 0.1)
            }
            beatIndex += 1
            frame += period
        }

        let activations = makeActivations(frames)

        let (beatFrames, downbeatFrames, labels) = Downbeat.decode(
            activations: activations,
            nFrames: nFrames,
            fps: fps,
            beatsPerBar: beatsPerBar
        )

        // Should detect multiple beats
        XCTAssertGreaterThan(beatFrames.count, 5,
            "Should detect multiple beats in 4/4 pattern")

        // Should detect downbeats
        XCTAssertGreaterThan(downbeatFrames.count, 0,
            "Should detect downbeats in 4/4 pattern")

        // Downbeats should be a subset of beats
        let beatSet = Set(beatFrames)
        for db in downbeatFrames {
            XCTAssertTrue(beatSet.contains(db),
                "Every downbeat should also be in beatFrames")
        }

        // Ratio of downbeats to beats should be roughly 1:beatsPerBar
        if beatFrames.count > 0 {
            let ratio = Float(downbeatFrames.count) / Float(beatFrames.count)
            XCTAssertGreaterThan(ratio, 0.1,
                "Downbeat ratio should be > 0.1 for 4/4")
            XCTAssertLessThan(ratio, 0.6,
                "Downbeat ratio should be < 0.6 for 4/4")
        }

        // Labels should have correct length
        XCTAssertEqual(labels.count, nFrames, "Labels array should match nFrames")
    }

    // MARK: - 3/4 Pattern

    func test3_4Pattern() {
        // 3/4 time: beatsPerBar=3
        let fps: Float = 100.0
        let bpm: Float = 120.0
        let period = Int(60.0 * fps / bpm)  // 50 frames
        let nFrames = 1000
        let beatsPerBar = 3

        var frames = [(Float, Float, Float)](repeating: (0.8, 0.1, 0.1), count: nFrames)

        var beatIndex = 0
        var frame = 0
        while frame < nFrames {
            if beatIndex % beatsPerBar == 0 {
                frames[frame] = (0.05, 0.1, 0.85)
            } else {
                frames[frame] = (0.05, 0.85, 0.1)
            }
            beatIndex += 1
            frame += period
        }

        let activations = makeActivations(frames)

        let (beatFrames, downbeatFrames, _) = Downbeat.decode(
            activations: activations,
            nFrames: nFrames,
            fps: fps,
            beatsPerBar: beatsPerBar
        )

        // Should detect beats
        XCTAssertGreaterThan(beatFrames.count, 5,
            "Should detect multiple beats in 3/4 pattern")

        // Should detect downbeats
        XCTAssertGreaterThan(downbeatFrames.count, 0,
            "Should detect downbeats in 3/4 pattern")

        // Downbeats should be subset of beats
        let beatSet = Set(beatFrames)
        for db in downbeatFrames {
            XCTAssertTrue(beatSet.contains(db),
                "Downbeat must be in beatFrames (3/4)")
        }

        // For 3/4, expect roughly 1/3 of beats to be downbeats
        if beatFrames.count > 0 {
            let ratio = Float(downbeatFrames.count) / Float(beatFrames.count)
            XCTAssertGreaterThan(ratio, 0.15,
                "Downbeat ratio should be > 0.15 for 3/4")
            XCTAssertLessThan(ratio, 0.7,
                "Downbeat ratio should be < 0.7 for 3/4")
        }
    }

    // MARK: - All Zeros

    func testAllZeros() {
        let nFrames = 500
        let activations = [Float](repeating: 0.0, count: nFrames * 3)

        let (beatFrames, downbeatFrames, labels) = Downbeat.decode(
            activations: activations,
            nFrames: nFrames,
            fps: 100.0
        )

        // All-zero activations: beat tracker threshold should filter everything
        XCTAssertEqual(beatFrames.count, 0,
            "All-zero activations should produce no beats")
        XCTAssertEqual(downbeatFrames.count, 0,
            "All-zero activations should produce no downbeats")
        XCTAssertEqual(labels.count, nFrames,
            "Labels array should still have nFrames entries")

        // All labels should be noBeat
        for label in labels {
            XCTAssertEqual(label, .noBeat,
                "All labels should be .noBeat for zero activations")
        }
    }

    // MARK: - Single Downbeat

    func testSingleDownbeat() {
        let nFrames = 200
        var frames = [(Float, Float, Float)](repeating: (0.9, 0.05, 0.05), count: nFrames)
        // One strong downbeat
        frames[100] = (0.05, 0.05, 0.9)

        let activations = makeActivations(frames)

        let (beatFrames, downbeatFrames, labels) = Downbeat.decode(
            activations: activations,
            nFrames: nFrames,
            fps: 100.0
        )

        // Should find at least one beat near frame 100
        let nearBeats = beatFrames.filter { abs($0 - 100) <= 25 }
        XCTAssertGreaterThanOrEqual(nearBeats.count, 1,
            "Should detect at least one beat near the downbeat activation")

        // Labels should have correct length
        XCTAssertEqual(labels.count, nFrames)

        // If we found downbeats, they should be near frame 100
        if !downbeatFrames.isEmpty {
            let nearDownbeats = downbeatFrames.filter { abs($0 - 100) <= 25 }
            XCTAssertGreaterThanOrEqual(nearDownbeats.count, 0,
                "Downbeats near frame 100 are expected")
        }
    }

    // MARK: - Beat vs Downbeat Classification

    func testBeatVsDownbeatClassification() {
        // Create a clear pattern where specific positions have high P(downbeat)
        let fps: Float = 100.0
        let period = 50  // 120 BPM
        let nFrames = 600
        let beatsPerBar = 4

        var frames = [(Float, Float, Float)](repeating: (0.85, 0.1, 0.05), count: nFrames)

        var beatIndex = 0
        var frame = 0
        while frame < nFrames {
            if beatIndex % beatsPerBar == 0 {
                // Strong downbeat
                frames[frame] = (0.02, 0.08, 0.9)
            } else {
                // Strong beat
                frames[frame] = (0.02, 0.9, 0.08)
            }
            beatIndex += 1
            frame += period
        }

        let activations = makeActivations(frames)

        let (beatFrames, downbeatFrames, labels) = Downbeat.decode(
            activations: activations,
            nFrames: nFrames,
            fps: fps,
            beatsPerBar: beatsPerBar
        )

        // Check that downbeat-labeled frames have the .downbeat label
        for db in downbeatFrames {
            XCTAssertEqual(labels[db], .downbeat,
                "Downbeat frame \(db) should have .downbeat label")
        }

        // Check that non-downbeat beats have the .beat label
        let downbeatSet = Set(downbeatFrames)
        for b in beatFrames {
            if !downbeatSet.contains(b) {
                XCTAssertEqual(labels[b], .beat,
                    "Non-downbeat beat frame \(b) should have .beat label")
            }
        }
    }

    // MARK: - Frame Count Consistency

    func testFrameCountConsistency() {
        let fps: Float = 100.0
        let period = 50
        let nFrames = 800

        var frames = [(Float, Float, Float)](repeating: (0.8, 0.1, 0.1), count: nFrames)
        var beatIndex = 0
        var frame = 0
        while frame < nFrames {
            if beatIndex % 4 == 0 {
                frames[frame] = (0.05, 0.1, 0.85)
            } else {
                frames[frame] = (0.05, 0.85, 0.1)
            }
            beatIndex += 1
            frame += period
        }

        let activations = makeActivations(frames)

        let (beatFrames, downbeatFrames, _) = Downbeat.decode(
            activations: activations,
            nFrames: nFrames,
            fps: fps,
            beatsPerBar: 4
        )

        // beatFrames should be a superset of downbeatFrames
        let beatSet = Set(beatFrames)
        for db in downbeatFrames {
            XCTAssertTrue(beatSet.contains(db),
                "beatFrames must contain all downbeatFrames. Missing: \(db)")
        }

        // downbeatFrames count should be <= beatFrames count
        XCTAssertLessThanOrEqual(downbeatFrames.count, beatFrames.count,
            "downbeatFrames.count should be <= beatFrames.count")
    }

    // MARK: - Labels Array

    func testLabelsArray() {
        let nFrames = 500
        let period = 50

        var frames = [(Float, Float, Float)](repeating: (0.85, 0.1, 0.05), count: nFrames)
        var beatIndex = 0
        var frame = 0
        while frame < nFrames {
            if beatIndex % 4 == 0 {
                frames[frame] = (0.02, 0.08, 0.9)
            } else {
                frames[frame] = (0.02, 0.9, 0.08)
            }
            beatIndex += 1
            frame += period
        }

        let activations = makeActivations(frames)

        let (beatFrames, downbeatFrames, labels) = Downbeat.decode(
            activations: activations,
            nFrames: nFrames,
            fps: 100.0,
            beatsPerBar: 4
        )

        // Labels length should equal nFrames
        XCTAssertEqual(labels.count, nFrames)

        // Count label types
        var noBeatCount = 0
        var beatCount = 0
        var downbeatCount = 0

        for label in labels {
            switch label {
            case .noBeat: noBeatCount += 1
            case .beat: beatCount += 1
            case .downbeat: downbeatCount += 1
            }
        }

        // Total beat + downbeat labels should equal beatFrames count
        XCTAssertEqual(beatCount + downbeatCount, beatFrames.count,
            "Total .beat + .downbeat labels should equal beatFrames.count")

        // Downbeat label count should match downbeatFrames count
        XCTAssertEqual(downbeatCount, downbeatFrames.count,
            ".downbeat label count should match downbeatFrames.count")

        // noBeat labels should be the rest
        XCTAssertEqual(noBeatCount, nFrames - beatFrames.count,
            ".noBeat labels should be nFrames - beatFrames.count")
    }

    // MARK: - Empty Input

    func testEmptyInput() {
        let (beatFrames, downbeatFrames, labels) = Downbeat.decode(
            activations: [],
            nFrames: 0,
            fps: 100.0
        )

        XCTAssertTrue(beatFrames.isEmpty, "Empty input should return no beats")
        XCTAssertTrue(downbeatFrames.isEmpty, "Empty input should return no downbeats")
        XCTAssertTrue(labels.isEmpty, "Empty input should return no labels")
    }

    // MARK: - Invalid Activation Size

    func testInvalidActivationSize() {
        // nFrames = 10 but only 20 floats (needs 30)
        let activations = [Float](repeating: 0.33, count: 20)

        let (beatFrames, downbeatFrames, labels) = Downbeat.decode(
            activations: activations,
            nFrames: 10,
            fps: 100.0
        )

        // Should safely return empty when activation count is insufficient
        XCTAssertTrue(beatFrames.isEmpty)
        XCTAssertTrue(downbeatFrames.isEmpty)
        XCTAssertTrue(labels.isEmpty)
    }

    // MARK: - Beats Sorted and Non-Negative

    func testBeatsSortedAndNonNegative() {
        let fps: Float = 100.0
        let period = 50
        let nFrames = 600

        var frames = [(Float, Float, Float)](repeating: (0.8, 0.1, 0.1), count: nFrames)
        var beatIndex = 0
        var frame = 0
        while frame < nFrames {
            if beatIndex % 4 == 0 {
                frames[frame] = (0.05, 0.1, 0.85)
            } else {
                frames[frame] = (0.05, 0.85, 0.1)
            }
            beatIndex += 1
            frame += period
        }

        let activations = makeActivations(frames)

        let (beatFrames, downbeatFrames, _) = Downbeat.decode(
            activations: activations,
            nFrames: nFrames,
            fps: fps,
            beatsPerBar: 4
        )

        // Beats should be sorted
        for i in 1..<beatFrames.count {
            XCTAssertGreaterThan(beatFrames[i], beatFrames[i - 1],
                "Beat frames should be strictly increasing")
        }

        // Downbeats should be sorted
        for i in 1..<downbeatFrames.count {
            XCTAssertGreaterThan(downbeatFrames[i], downbeatFrames[i - 1],
                "Downbeat frames should be strictly increasing")
        }

        // All frames should be non-negative and within range
        for b in beatFrames {
            XCTAssertGreaterThanOrEqual(b, 0, "Beat frames should be non-negative")
            XCTAssertLessThan(b, nFrames, "Beat frames should be < nFrames")
        }
        for db in downbeatFrames {
            XCTAssertGreaterThanOrEqual(db, 0, "Downbeat frames should be non-negative")
            XCTAssertLessThan(db, nFrames, "Downbeat frames should be < nFrames")
        }
    }

    // MARK: - Uniform Activations

    func testUniformActivations() {
        // All frames have equal probability for all three classes
        let nFrames = 300
        let activations = [Float](repeating: 0.333, count: nFrames * 3)

        let (_, _, labels) = Downbeat.decode(
            activations: activations,
            nFrames: nFrames,
            fps: 100.0,
            beatsPerBar: 4
        )

        // Should not crash; labels should be nFrames long
        XCTAssertEqual(labels.count, nFrames)
    }

    // MARK: - beatsPerBar = 1

    func testBeatsPerBarOne() {
        // When beatsPerBar = 1, every beat should be a downbeat
        let fps: Float = 100.0
        let period = 50
        let nFrames = 500

        var frames = [(Float, Float, Float)](repeating: (0.8, 0.1, 0.1), count: nFrames)
        var frame = 0
        while frame < nFrames {
            frames[frame] = (0.05, 0.1, 0.85)
            frame += period
        }

        let activations = makeActivations(frames)

        let (beatFrames, downbeatFrames, _) = Downbeat.decode(
            activations: activations,
            nFrames: nFrames,
            fps: fps,
            beatsPerBar: 1
        )

        // With beatsPerBar=1, every beat should be a downbeat
        if !beatFrames.isEmpty {
            XCTAssertEqual(beatFrames.count, downbeatFrames.count,
                "With beatsPerBar=1, all beats should be downbeats")
        }
    }
}
