import XCTest
@testable import MetalMomCore

final class ChordRecognitionTests: XCTestCase {

    // MARK: - Chord Labels

    func testChordLabelsCount() {
        XCTAssertEqual(ChordRecognition.chordLabels.count, 25)
        XCTAssertEqual(ChordRecognition.nClasses, 25)
    }

    func testChordLabelsFormat() {
        // First label is "N" (no chord)
        XCTAssertEqual(ChordRecognition.chordLabels[0], "N")

        // Labels 1-12: major chords
        XCTAssertEqual(ChordRecognition.chordLabels[1], "C:maj")
        XCTAssertEqual(ChordRecognition.chordLabels[2], "C#:maj")
        XCTAssertEqual(ChordRecognition.chordLabels[12], "B:maj")

        // Labels 13-24: minor chords
        XCTAssertEqual(ChordRecognition.chordLabels[13], "C:min")
        XCTAssertEqual(ChordRecognition.chordLabels[14], "C#:min")
        XCTAssertEqual(ChordRecognition.chordLabels[24], "B:min")
    }

    // MARK: - Label Lookup

    func testLabelForIndex() {
        // All indices return valid labels
        for i in 0..<25 {
            let label = ChordRecognition.label(forIndex: i)
            XCTAssertEqual(label, ChordRecognition.chordLabels[i])
        }
    }

    func testLabelForIndexOutOfRange() {
        // Out-of-range returns "N"
        XCTAssertEqual(ChordRecognition.label(forIndex: -1), "N")
        XCTAssertEqual(ChordRecognition.label(forIndex: 25), "N")
        XCTAssertEqual(ChordRecognition.label(forIndex: 100), "N")
    }

    // MARK: - Single Chord (All Same)

    func testSingleChord() {
        let nFrames = 20
        let nClasses = 25
        let cMajIndex = 1  // C:maj

        // All frames have highest activation for C:maj
        var activations = [Float](repeating: 0, count: nFrames * nClasses)
        for t in 0..<nFrames {
            activations[t * nClasses + cMajIndex] = 10.0
        }

        let events = ChordRecognition.decode(
            activations: activations,
            nFrames: nFrames
        )

        XCTAssertEqual(events.count, 1)
        XCTAssertEqual(events[0].startFrame, 0)
        XCTAssertEqual(events[0].endFrame, nFrames)
        XCTAssertEqual(events[0].chordIndex, cMajIndex)
        XCTAssertEqual(events[0].chordLabel, "C:maj")
    }

    // MARK: - Two Chords

    func testTwoChords() {
        let nFrames = 20
        let nClasses = 25
        let half = nFrames / 2
        let cMajIdx = 1   // C:maj
        let gMajIdx = 8   // G:maj

        var activations = [Float](repeating: 0, count: nFrames * nClasses)
        // First half: C:maj
        for t in 0..<half {
            activations[t * nClasses + cMajIdx] = 10.0
        }
        // Second half: G:maj
        for t in half..<nFrames {
            activations[t * nClasses + gMajIdx] = 10.0
        }

        let events = ChordRecognition.decode(
            activations: activations,
            nFrames: nFrames
        )

        XCTAssertEqual(events.count, 2)
        XCTAssertEqual(events[0].chordLabel, "C:maj")
        XCTAssertEqual(events[0].startFrame, 0)
        XCTAssertEqual(events[0].endFrame, half)
        XCTAssertEqual(events[1].chordLabel, "G:maj")
        XCTAssertEqual(events[1].startFrame, half)
        XCTAssertEqual(events[1].endFrame, nFrames)
    }

    // MARK: - CRF Smoothing

    func testCRFSmoothing() {
        // With CRF smoothing and high self-transition bias, a brief noise blip
        // in the middle should be smoothed out.
        let nFrames = 30
        let nClasses = 25
        let cMajIdx = 1   // C:maj
        let noiseIdx = 5  // some other chord

        var activations = [Float](repeating: 0, count: nFrames * nClasses)
        // All frames: strong C:maj
        for t in 0..<nFrames {
            activations[t * nClasses + cMajIdx] = 5.0
        }
        // Brief noise: 2 frames in the middle with slightly higher activation for another chord
        for t in 14..<16 {
            activations[t * nClasses + cMajIdx] = 4.0
            activations[t * nClasses + noiseIdx] = 5.5
        }

        // Simple (argmax) should show the noise
        let simpleEvents = ChordRecognition.detectSimple(
            activations: activations,
            nFrames: nFrames
        )
        // argmax will pick up the noise chord in the middle
        XCTAssertGreaterThan(simpleEvents.count, 1, "Simple detection should show noise")

        // CRF with high self-transition bias should smooth it out
        let crfEvents = ChordRecognition.decode(
            activations: activations,
            nFrames: nFrames,
            selfTransitionBias: 5.0
        )
        // CRF should smooth to a single chord (or at least fewer events)
        XCTAssertLessThan(crfEvents.count, simpleEvents.count,
                          "CRF should produce fewer chord changes than argmax")
    }

    // MARK: - Simple Detection

    func testDetectSimple() {
        let nFrames = 5
        let nClasses = 25

        // Each frame has a different strongest chord
        var activations = [Float](repeating: 0, count: nFrames * nClasses)
        let chordSeq = [0, 1, 1, 3, 3]  // N, C:maj, C:maj, D:maj, D:maj
        for t in 0..<nFrames {
            activations[t * nClasses + chordSeq[t]] = 10.0
        }

        let events = ChordRecognition.detectSimple(
            activations: activations,
            nFrames: nFrames
        )

        XCTAssertEqual(events.count, 3)
        XCTAssertEqual(events[0].chordIndex, 0)
        XCTAssertEqual(events[0].chordLabel, "N")
        XCTAssertEqual(events[0].startFrame, 0)
        XCTAssertEqual(events[0].endFrame, 1)

        XCTAssertEqual(events[1].chordIndex, 1)
        XCTAssertEqual(events[1].chordLabel, "C:maj")
        XCTAssertEqual(events[1].startFrame, 1)
        XCTAssertEqual(events[1].endFrame, 3)

        XCTAssertEqual(events[2].chordIndex, 3)
        XCTAssertEqual(events[2].chordLabel, "D:maj")
        XCTAssertEqual(events[2].startFrame, 3)
        XCTAssertEqual(events[2].endFrame, 5)
    }

    // MARK: - No Chord Class

    func testNoChordClass() {
        let nFrames = 10
        let nClasses = 25

        // All frames have highest activation at index 0 ("N")
        var activations = [Float](repeating: 0, count: nFrames * nClasses)
        for t in 0..<nFrames {
            activations[t * nClasses + 0] = 10.0
        }

        let events = ChordRecognition.decode(
            activations: activations,
            nFrames: nFrames
        )

        XCTAssertEqual(events.count, 1)
        XCTAssertEqual(events[0].chordLabel, "N")
        XCTAssertEqual(events[0].chordIndex, 0)
    }

    // MARK: - Run-Length Encoding

    func testRunLengthEncoding() {
        let nFrames = 6
        let nClasses = 25

        // Alternating frames: C:maj, G:maj, C:maj, G:maj, C:maj, G:maj
        var activations = [Float](repeating: 0, count: nFrames * nClasses)
        let cMajIdx = 1
        let gMajIdx = 8
        for t in 0..<nFrames {
            let idx = (t % 2 == 0) ? cMajIdx : gMajIdx
            activations[t * nClasses + idx] = 10.0
        }

        let events = ChordRecognition.detectSimple(
            activations: activations,
            nFrames: nFrames
        )

        // Alternating single frames -> 6 events
        XCTAssertEqual(events.count, 6)
        for (i, event) in events.enumerated() {
            XCTAssertEqual(event.startFrame, i)
            XCTAssertEqual(event.endFrame, i + 1)
            if i % 2 == 0 {
                XCTAssertEqual(event.chordLabel, "C:maj")
            } else {
                XCTAssertEqual(event.chordLabel, "G:maj")
            }
        }
    }

    // MARK: - Events to Times

    func testEventsToTimes() {
        let events = [
            ChordRecognition.ChordEvent(startFrame: 0, endFrame: 10, chordIndex: 1, chordLabel: "C:maj"),
            ChordRecognition.ChordEvent(startFrame: 10, endFrame: 20, chordIndex: 8, chordLabel: "G:maj"),
        ]

        let times = ChordRecognition.eventsToTimes(events: events, fps: 100.0)

        XCTAssertEqual(times.count, 2)
        XCTAssertEqual(times[0].start, 0.0, accuracy: 1e-6)
        XCTAssertEqual(times[0].end, 0.1, accuracy: 1e-6)
        XCTAssertEqual(times[0].chord, "C:maj")
        XCTAssertEqual(times[1].start, 0.1, accuracy: 1e-6)
        XCTAssertEqual(times[1].end, 0.2, accuracy: 1e-6)
        XCTAssertEqual(times[1].chord, "G:maj")
    }

    func testEventsToTimesZeroFPS() {
        let events = [
            ChordRecognition.ChordEvent(startFrame: 0, endFrame: 10, chordIndex: 1, chordLabel: "C:maj"),
        ]
        let times = ChordRecognition.eventsToTimes(events: events, fps: 0.0)
        XCTAssertTrue(times.isEmpty)
    }

    // MARK: - Self-Transition Bias Effect

    func testSelfTransitionBiasEffect() {
        // With rapidly alternating activations, higher self-transition bias
        // should produce fewer chord changes.
        let nFrames = 20
        let nClasses = 25
        let cMajIdx = 1
        let gMajIdx = 8

        // Alternating frames with small differences
        var activations = [Float](repeating: 0, count: nFrames * nClasses)
        for t in 0..<nFrames {
            if t % 2 == 0 {
                activations[t * nClasses + cMajIdx] = 2.0
                activations[t * nClasses + gMajIdx] = 1.5
            } else {
                activations[t * nClasses + cMajIdx] = 1.5
                activations[t * nClasses + gMajIdx] = 2.0
            }
        }

        // Low bias: more changes
        let eventsLow = ChordRecognition.decode(
            activations: activations,
            nFrames: nFrames,
            selfTransitionBias: 0.0
        )

        // High bias: fewer changes
        let eventsHigh = ChordRecognition.decode(
            activations: activations,
            nFrames: nFrames,
            selfTransitionBias: 10.0
        )

        XCTAssertGreaterThanOrEqual(eventsLow.count, eventsHigh.count,
                                    "Higher self-transition bias should produce fewer or equal chord changes")
    }

    // MARK: - Empty Input

    func testEmptyInput() {
        let events = ChordRecognition.decode(activations: [], nFrames: 0)
        XCTAssertTrue(events.isEmpty)

        let simple = ChordRecognition.detectSimple(activations: [], nFrames: 0)
        XCTAssertTrue(simple.isEmpty)
    }

    // MARK: - Custom Transition Scores

    func testCustomTransitionScores() {
        let nFrames = 10
        let nClasses = 25
        let cMajIdx = 1

        // All frames: C:maj
        var activations = [Float](repeating: 0, count: nFrames * nClasses)
        for t in 0..<nFrames {
            activations[t * nClasses + cMajIdx] = 10.0
        }

        // Custom transition scores: identity * 2.0
        var transitionScores = [Float](repeating: 0, count: nClasses * nClasses)
        for i in 0..<nClasses {
            transitionScores[i * nClasses + i] = 2.0
        }

        let events = ChordRecognition.decode(
            activations: activations,
            nFrames: nFrames,
            transitionScores: transitionScores
        )

        XCTAssertEqual(events.count, 1)
        XCTAssertEqual(events[0].chordLabel, "C:maj")
    }

    // MARK: - Minor Chord Detection

    func testMinorChordDetection() {
        let nFrames = 10
        let nClasses = 25
        let aMinIdx = 22  // A:min is index 22 (13 + 9)

        var activations = [Float](repeating: 0, count: nFrames * nClasses)
        for t in 0..<nFrames {
            activations[t * nClasses + aMinIdx] = 10.0
        }

        let events = ChordRecognition.decode(
            activations: activations,
            nFrames: nFrames
        )

        XCTAssertEqual(events.count, 1)
        XCTAssertEqual(events[0].chordLabel, "A:min")
        XCTAssertEqual(events[0].chordIndex, aMinIdx)
    }
}
