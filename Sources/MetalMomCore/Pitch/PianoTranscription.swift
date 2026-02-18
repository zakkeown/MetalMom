import Foundation

/// Polyphonic piano transcription from per-frame, per-pitch activation probabilities.
///
/// Decodes note events from an ``[nFrames, 88]`` activation matrix (e.g., from
/// madmom's ``RNNPianoNoteProcessor``). Two strategies are available:
///
/// - **Threshold-based**: contiguous above-threshold regions become note events.
/// - **HMM-based**: per-pitch 2-state Viterbi decoding for smoother onset/offset detection.
public enum PianoTranscription {

    /// A detected note event.
    public struct NoteEvent {
        /// MIDI note number (21-108 for standard 88-key piano).
        public let midiNote: Int
        /// Onset frame index.
        public let onsetFrame: Int
        /// Offset frame index (exclusive).
        public let offsetFrame: Int
        /// Average activation probability during the note.
        public let velocity: Float
    }

    /// MIDI note range for standard 88-key piano.
    public static let midiMin = 21
    public static let midiMax = 108
    public static let nPitches = 88

    // MARK: - Threshold-Based Detection

    /// Detect piano notes from per-frame, per-pitch activation probabilities.
    ///
    /// Simple threshold-based approach: for each pitch, find contiguous
    /// regions where activation exceeds the threshold, emit a ``NoteEvent``
    /// for each region.
    ///
    /// - Parameters:
    ///   - activations: ``[nFrames * 88]`` flat array of per-frame, per-pitch activations.
    ///     Layout is row-major: ``activations[frame * 88 + pitch]``.
    ///   - nFrames: Number of frames.
    ///   - threshold: Minimum activation to trigger a note. Default 0.5.
    ///   - minDuration: Minimum note duration in frames. Default 3.
    ///   - fps: Frames per second (unused in threshold mode but kept for API symmetry). Default 100.
    /// - Returns: Array of detected ``NoteEvent``s, sorted by onset time then MIDI note.
    public static func detect(
        activations: [Float],
        nFrames: Int,
        threshold: Float = 0.5,
        minDuration: Int = 3,
        fps: Float = 100.0
    ) -> [NoteEvent] {
        guard nFrames > 0, activations.count >= nFrames * nPitches else { return [] }

        var events = [NoteEvent]()

        for p in 0..<nPitches {
            var inNote = false
            var onsetFrame = 0
            var activationSum: Float = 0
            var frameCount = 0

            for t in 0..<nFrames {
                let act = activations[t * nPitches + p]
                if act >= threshold {
                    if !inNote {
                        inNote = true
                        onsetFrame = t
                        activationSum = 0
                        frameCount = 0
                    }
                    activationSum += act
                    frameCount += 1
                } else {
                    if inNote {
                        if frameCount >= minDuration {
                            let velocity = activationSum / Float(frameCount)
                            events.append(NoteEvent(
                                midiNote: p + midiMin,
                                onsetFrame: onsetFrame,
                                offsetFrame: t,
                                velocity: velocity
                            ))
                        }
                        inNote = false
                    }
                }
            }

            // Close any note still open at end of signal
            if inNote && frameCount >= minDuration {
                let velocity = activationSum / Float(frameCount)
                events.append(NoteEvent(
                    midiNote: p + midiMin,
                    onsetFrame: onsetFrame,
                    offsetFrame: nFrames,
                    velocity: velocity
                ))
            }
        }

        // Sort by onset frame, then by MIDI note
        events.sort { a, b in
            if a.onsetFrame != b.onsetFrame { return a.onsetFrame < b.onsetFrame }
            return a.midiNote < b.midiNote
        }

        return events
    }

    // MARK: - HMM-Based Detection

    /// Detect piano notes using HMM smoothing per pitch.
    ///
    /// For each of the 88 pitches, runs a 2-state HMM (OFF=0, ON=1) via
    /// Viterbi decoding to determine note boundaries more smoothly than
    /// simple thresholding.
    ///
    /// - Parameters:
    ///   - activations: ``[nFrames * 88]`` flat activation probabilities.
    ///   - nFrames: Number of frames.
    ///   - onsetProb: Prior probability of note onset (OFF->ON transition). Default 0.01.
    ///   - offsetProb: Prior probability of note offset (ON->OFF transition). Default 0.05.
    ///   - minDuration: Minimum note duration in frames. Default 3.
    ///   - fps: Frames per second. Default 100.
    /// - Returns: Array of detected ``NoteEvent``s, sorted by onset time then MIDI note.
    public static func detectHMM(
        activations: [Float],
        nFrames: Int,
        onsetProb: Float = 0.01,
        offsetProb: Float = 0.05,
        minDuration: Int = 3,
        fps: Float = 100.0
    ) -> [NoteEvent] {
        guard nFrames > 0, activations.count >= nFrames * nPitches else { return [] }

        let eps: Float = 1e-7

        // Build 2-state HMM parameters (shared across all pitches)
        let logInitial: [Float] = [log(Float(0.99)), log(Float(0.01))]
        let logTransition: [[Float]] = [
            [log(1.0 - onsetProb), log(onsetProb)],         // OFF -> {OFF, ON}
            [log(offsetProb), log(1.0 - offsetProb)]         // ON -> {OFF, ON}
        ]

        var events = [NoteEvent]()

        for p in 0..<nPitches {
            // Build per-frame observations for this pitch
            var logObs = [[Float]](repeating: [Float](repeating: 0, count: 2), count: nFrames)
            for t in 0..<nFrames {
                let act = activations[t * nPitches + p]
                let clampedAct = min(max(act, eps), 1.0 - eps)
                logObs[t][0] = log(1.0 - clampedAct)  // OFF observation
                logObs[t][1] = log(clampedAct)          // ON observation
            }

            // Run Viterbi
            let (path, _) = HMM.viterbi(
                logObservations: logObs,
                logInitial: logInitial,
                logTransition: logTransition
            )

            // Extract contiguous ON regions from the state path
            var inNote = false
            var onsetFrame = 0
            var activationSum: Float = 0
            var frameCount = 0

            for t in 0..<nFrames {
                if path[t] == 1 {  // ON state
                    if !inNote {
                        inNote = true
                        onsetFrame = t
                        activationSum = 0
                        frameCount = 0
                    }
                    activationSum += activations[t * nPitches + p]
                    frameCount += 1
                } else {  // OFF state
                    if inNote {
                        if frameCount >= minDuration {
                            let velocity = activationSum / Float(frameCount)
                            events.append(NoteEvent(
                                midiNote: p + midiMin,
                                onsetFrame: onsetFrame,
                                offsetFrame: t,
                                velocity: velocity
                            ))
                        }
                        inNote = false
                    }
                }
            }

            // Close any note still open at end
            if inNote && frameCount >= minDuration {
                let velocity = activationSum / Float(frameCount)
                events.append(NoteEvent(
                    midiNote: p + midiMin,
                    onsetFrame: onsetFrame,
                    offsetFrame: nFrames,
                    velocity: velocity
                ))
            }
        }

        // Sort by onset frame, then by MIDI note
        events.sort { a, b in
            if a.onsetFrame != b.onsetFrame { return a.onsetFrame < b.onsetFrame }
            return a.midiNote < b.midiNote
        }

        return events
    }

    // MARK: - Utilities

    /// Convert note events to MIDI-like tuples ``(onset_time, offset_time, midi_note, velocity)``.
    ///
    /// - Parameters:
    ///   - events: Array of ``NoteEvent``s.
    ///   - fps: Frames per second for time conversion.
    /// - Returns: Array of tuples with onset/offset in seconds, MIDI note, and velocity.
    public static func eventsToNotes(
        events: [NoteEvent],
        fps: Float
    ) -> [(onset: Float, offset: Float, midi: Int, velocity: Float)] {
        return events.map { e in
            (
                onset: Float(e.onsetFrame) / fps,
                offset: Float(e.offsetFrame) / fps,
                midi: e.midiNote,
                velocity: e.velocity
            )
        }
    }

    /// Convert MIDI note number to note name (e.g., 60 -> "C4").
    ///
    /// - Parameter midiNote: MIDI note number.
    /// - Returns: Note name string.
    public static func noteName(midiNote: Int) -> String {
        let noteNames = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        let noteName = noteNames[midiNote % 12]
        let octave = (midiNote / 12) - 1
        return "\(noteName)\(octave)"
    }
}
