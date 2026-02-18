import XCTest
@testable import MetalMomCore

final class PianoTranscriptionTests: XCTestCase {

    // MARK: - Threshold-Based Detection

    /// Single note: one pitch active for 10 frames -> one NoteEvent.
    func testSingleNote() {
        let nFrames = 20
        let nPitches = 88
        var activations = [Float](repeating: 0, count: nFrames * nPitches)

        // Pitch 39 (MIDI 60 = C4) active frames 5..<15
        let pitch = 39  // 60 - 21
        for t in 5..<15 {
            activations[t * nPitches + pitch] = 0.8
        }

        let events = PianoTranscription.detect(
            activations: activations, nFrames: nFrames,
            threshold: 0.5, minDuration: 3
        )

        XCTAssertEqual(events.count, 1)
        XCTAssertEqual(events[0].midiNote, 60)
        XCTAssertEqual(events[0].onsetFrame, 5)
        XCTAssertEqual(events[0].offsetFrame, 15)
        XCTAssertEqual(events[0].velocity, 0.8, accuracy: 1e-5)
    }

    /// Two simultaneous notes: two pitches active at same time -> two events.
    func testTwoSimultaneousNotes() {
        let nFrames = 20
        let nPitches = 88
        var activations = [Float](repeating: 0, count: nFrames * nPitches)

        // MIDI 60 (C4) and MIDI 64 (E4) both active frames 3..<13
        let pitch1 = 60 - 21  // C4
        let pitch2 = 64 - 21  // E4
        for t in 3..<13 {
            activations[t * nPitches + pitch1] = 0.9
            activations[t * nPitches + pitch2] = 0.7
        }

        let events = PianoTranscription.detect(
            activations: activations, nFrames: nFrames,
            threshold: 0.5, minDuration: 3
        )

        XCTAssertEqual(events.count, 2)
        // Should be sorted by onset (same onset) then by MIDI note
        XCTAssertEqual(events[0].midiNote, 60)
        XCTAssertEqual(events[1].midiNote, 64)
        XCTAssertEqual(events[0].onsetFrame, 3)
        XCTAssertEqual(events[1].onsetFrame, 3)
    }

    /// Sequential notes: same pitch, gap, same pitch -> two events.
    func testSequentialNotes() {
        let nFrames = 30
        let nPitches = 88
        var activations = [Float](repeating: 0, count: nFrames * nPitches)

        let pitch = 60 - 21  // C4
        // First note: frames 2..<8
        for t in 2..<8 {
            activations[t * nPitches + pitch] = 0.8
        }
        // Gap: frames 8..<12 (below threshold)
        // Second note: frames 12..<20
        for t in 12..<20 {
            activations[t * nPitches + pitch] = 0.6
        }

        let events = PianoTranscription.detect(
            activations: activations, nFrames: nFrames,
            threshold: 0.5, minDuration: 3
        )

        XCTAssertEqual(events.count, 2)
        XCTAssertEqual(events[0].midiNote, 60)
        XCTAssertEqual(events[0].onsetFrame, 2)
        XCTAssertEqual(events[0].offsetFrame, 8)
        XCTAssertEqual(events[1].midiNote, 60)
        XCTAssertEqual(events[1].onsetFrame, 12)
        XCTAssertEqual(events[1].offsetFrame, 20)
    }

    /// Below threshold: low activations -> no notes detected.
    func testBelowThreshold() {
        let nFrames = 10
        let nPitches = 88
        let activations = [Float](repeating: 0.1, count: nFrames * nPitches)

        // All activations are 0.1, well below threshold=0.5
        let events = PianoTranscription.detect(
            activations: activations, nFrames: nFrames,
            threshold: 0.5, minDuration: 3
        )

        XCTAssertEqual(events.count, 0)
    }

    /// Min duration filter: 2-frame note with minDuration=3 -> filtered out.
    func testMinDurationFilter() {
        let nFrames = 10
        let nPitches = 88
        var activations = [Float](repeating: 0, count: nFrames * nPitches)

        let pitch = 60 - 21
        // Only 2 frames active (below minDuration=3)
        activations[3 * nPitches + pitch] = 0.9
        activations[4 * nPitches + pitch] = 0.9

        let events = PianoTranscription.detect(
            activations: activations, nFrames: nFrames,
            threshold: 0.5, minDuration: 3
        )

        XCTAssertEqual(events.count, 0)
    }

    // MARK: - HMM-Based Detection

    /// HMM detection: same test as threshold but with detectHMM.
    func testHMMDetection() {
        let nFrames = 20
        let nPitches = 88
        var activations = [Float](repeating: 0.01, count: nFrames * nPitches)

        // Pitch 39 (MIDI 60 = C4) active frames 5..<15 with high probability
        let pitch = 39
        for t in 5..<15 {
            activations[t * nPitches + pitch] = 0.95
        }

        let events = PianoTranscription.detectHMM(
            activations: activations, nFrames: nFrames,
            onsetProb: 0.01, offsetProb: 0.05, minDuration: 3
        )

        XCTAssertGreaterThanOrEqual(events.count, 1)
        // The HMM should find the note around frames 5-15
        let noteC4 = events.first { $0.midiNote == 60 }
        XCTAssertNotNil(noteC4)
        if let note = noteC4 {
            XCTAssertLessThanOrEqual(note.onsetFrame, 6)
            XCTAssertGreaterThanOrEqual(note.offsetFrame, 14)
        }
    }

    /// HMM smoothing: brief gap in activation -> HMM bridges it.
    func testHMMSmoothingBridgesGap() {
        let nFrames = 20
        let nPitches = 88
        var activations = [Float](repeating: 0.01, count: nFrames * nPitches)

        let pitch = 39  // C4
        // High activation with a brief 1-frame gap
        for t in 3..<8 {
            activations[t * nPitches + pitch] = 0.95
        }
        // Frame 8: gap (low activation)
        activations[8 * nPitches + pitch] = 0.15
        for t in 9..<14 {
            activations[t * nPitches + pitch] = 0.95
        }

        // Threshold-based: should produce 2 notes
        let thresholdEvents = PianoTranscription.detect(
            activations: activations, nFrames: nFrames,
            threshold: 0.5, minDuration: 3
        )
        let thresholdC4 = thresholdEvents.filter { $0.midiNote == 60 }
        XCTAssertEqual(thresholdC4.count, 2, "Threshold should see 2 separate notes")

        // HMM-based: should bridge the gap -> 1 note
        let hmmEvents = PianoTranscription.detectHMM(
            activations: activations, nFrames: nFrames,
            onsetProb: 0.01, offsetProb: 0.05, minDuration: 3
        )
        let hmmC4 = hmmEvents.filter { $0.midiNote == 60 }
        XCTAssertEqual(hmmC4.count, 1, "HMM should bridge the 1-frame gap into 1 note")
    }

    // MARK: - Utility Tests

    /// Note name conversion: MIDI 60 -> "C4", MIDI 69 -> "A4", MIDI 21 -> "A0".
    func testNoteName() {
        XCTAssertEqual(PianoTranscription.noteName(midiNote: 60), "C4")
        XCTAssertEqual(PianoTranscription.noteName(midiNote: 69), "A4")
        XCTAssertEqual(PianoTranscription.noteName(midiNote: 21), "A0")
        XCTAssertEqual(PianoTranscription.noteName(midiNote: 108), "C8")
        XCTAssertEqual(PianoTranscription.noteName(midiNote: 61), "C#4")
        XCTAssertEqual(PianoTranscription.noteName(midiNote: 48), "C3")
    }

    /// Events to notes: verify time conversion.
    func testEventsToNotes() {
        let events = [
            PianoTranscription.NoteEvent(midiNote: 60, onsetFrame: 100, offsetFrame: 200, velocity: 0.8),
            PianoTranscription.NoteEvent(midiNote: 64, onsetFrame: 150, offsetFrame: 250, velocity: 0.6),
        ]

        let notes = PianoTranscription.eventsToNotes(events: events, fps: 100.0)

        XCTAssertEqual(notes.count, 2)
        XCTAssertEqual(notes[0].onset, 1.0, accuracy: 1e-5)
        XCTAssertEqual(notes[0].offset, 2.0, accuracy: 1e-5)
        XCTAssertEqual(notes[0].midi, 60)
        XCTAssertEqual(notes[0].velocity, 0.8, accuracy: 1e-5)

        XCTAssertEqual(notes[1].onset, 1.5, accuracy: 1e-5)
        XCTAssertEqual(notes[1].offset, 2.5, accuracy: 1e-5)
        XCTAssertEqual(notes[1].midi, 64)
        XCTAssertEqual(notes[1].velocity, 0.6, accuracy: 1e-5)
    }

    /// All 88 pitches: each pitch individually -> 88 events.
    func testAll88Pitches() {
        let nFrames = 10
        let nPitches = 88
        var activations = [Float](repeating: 0, count: nFrames * nPitches)

        // Activate each pitch for frames 2..<7 (5 frames, above minDuration=3)
        for p in 0..<nPitches {
            for t in 2..<7 {
                activations[t * nPitches + p] = 0.8
            }
        }

        let events = PianoTranscription.detect(
            activations: activations, nFrames: nFrames,
            threshold: 0.5, minDuration: 3
        )

        XCTAssertEqual(events.count, 88)
        // Check MIDI range
        let midiNotes = Set(events.map { $0.midiNote })
        for midi in 21...108 {
            XCTAssertTrue(midiNotes.contains(midi), "Missing MIDI note \(midi)")
        }
    }

    /// Empty input: no activations -> no notes.
    func testEmptyInput() {
        let events = PianoTranscription.detect(
            activations: [], nFrames: 0,
            threshold: 0.5, minDuration: 3
        )
        XCTAssertEqual(events.count, 0)
    }

    /// Velocity computation: average activation matches expected value.
    func testVelocityComputation() {
        let nFrames = 10
        let nPitches = 88
        var activations = [Float](repeating: 0, count: nFrames * nPitches)

        let pitch = 60 - 21  // C4
        // Varying activations: 0.6, 0.7, 0.8, 0.9, 1.0
        activations[2 * nPitches + pitch] = 0.6
        activations[3 * nPitches + pitch] = 0.7
        activations[4 * nPitches + pitch] = 0.8
        activations[5 * nPitches + pitch] = 0.9
        activations[6 * nPitches + pitch] = 1.0

        let events = PianoTranscription.detect(
            activations: activations, nFrames: nFrames,
            threshold: 0.5, minDuration: 3
        )

        XCTAssertEqual(events.count, 1)
        // Average = (0.6 + 0.7 + 0.8 + 0.9 + 1.0) / 5 = 0.8
        XCTAssertEqual(events[0].velocity, 0.8, accuracy: 1e-5)
    }

    // MARK: - Constants

    func testConstants() {
        XCTAssertEqual(PianoTranscription.midiMin, 21)
        XCTAssertEqual(PianoTranscription.midiMax, 108)
        XCTAssertEqual(PianoTranscription.nPitches, 88)
    }
}
