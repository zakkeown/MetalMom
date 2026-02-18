import Foundation

/// Chord recognition that decodes chord labels from CNN feature activations using CRF.
///
/// The pipeline follows madmom's approach:
/// 1. CNN produces per-frame chord activations (nFrames x nClasses)
/// 2. CRF Viterbi decoding finds the most likely chord sequence
/// 3. Run-length encoding produces chord events with start/end boundaries
///
/// Supports 25 chord classes: N (no chord), 12 major triads, 12 minor triads.
public enum ChordRecognition {

    // MARK: - Constants

    /// Standard chord labels: N (no chord), then major and minor triads.
    /// madmom uses 25 classes: N, C:maj, C#:maj, D:maj, ..., B:maj, C:min, ..., B:min
    public static let chordLabels: [String] = {
        let notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        var labels = ["N"]  // no chord
        labels += notes.map { "\($0):maj" }
        labels += notes.map { "\($0):min" }
        return labels
    }()

    /// Number of chord classes.
    public static let nClasses = 25

    // MARK: - Types

    /// Result of chord recognition for a single segment.
    public struct ChordEvent {
        /// Start frame index.
        public let startFrame: Int
        /// End frame index (exclusive).
        public let endFrame: Int
        /// Chord label index (0-24).
        public let chordIndex: Int
        /// Chord label string.
        public let chordLabel: String
    }

    // MARK: - CRF-based Decoding

    /// Decode chord sequence from per-frame chord activations using CRF.
    ///
    /// Uses Viterbi decoding on a CRF with unary potentials from the activations
    /// and pairwise potentials that encourage temporal smoothness (self-transition bias).
    ///
    /// - Parameters:
    ///   - activations: `[nFrames * nClasses]` flat array of per-frame chord scores.
    ///   - nFrames: Number of frames.
    ///   - nClasses: Number of chord classes (default 25).
    ///   - transitionScores: `[nClasses * nClasses]` flat CRF transition scores.
    ///     If nil, uses a simple self-transition bias.
    ///   - selfTransitionBias: Bias for staying in the same chord (used when
    ///     transitionScores is nil). Default 1.0.
    /// - Returns: Array of ChordEvents (run-length encoded chord sequence).
    public static func decode(
        activations: [Float],
        nFrames: Int,
        nClasses: Int = 25,
        transitionScores: [Float]? = nil,
        selfTransitionBias: Float = 1.0
    ) -> [ChordEvent] {
        guard nFrames > 0 else { return [] }
        guard activations.count == nFrames * nClasses else { return [] }

        // Reshape flat activations into [nFrames][nClasses]
        var unary = [[Float]](repeating: [Float](repeating: 0, count: nClasses), count: nFrames)
        for t in 0..<nFrames {
            for c in 0..<nClasses {
                unary[t][c] = activations[t * nClasses + c]
            }
        }

        // Build or use provided transition scores
        let pairwise: [[Float]]
        if let provided = transitionScores {
            guard provided.count == nClasses * nClasses else { return [] }
            var pw = [[Float]](repeating: [Float](repeating: 0, count: nClasses), count: nClasses)
            for i in 0..<nClasses {
                for j in 0..<nClasses {
                    pw[i][j] = provided[i * nClasses + j]
                }
            }
            pairwise = pw
        } else {
            // Default: identity matrix * selfTransitionBias (encourages staying in same chord)
            var pw = [[Float]](repeating: [Float](repeating: 0, count: nClasses), count: nClasses)
            for i in 0..<nClasses {
                pw[i][i] = selfTransitionBias
            }
            pairwise = pw
        }

        // Viterbi decode via existing CRF module
        let (path, _) = CRF.viterbiDecode(unaryScores: unary, pairwiseScores: pairwise)

        // Run-length encode into ChordEvents
        return runLengthEncode(path: path, nClasses: nClasses)
    }

    // MARK: - Simple (Argmax) Detection

    /// Simple argmax-based chord detection (no CRF smoothing).
    ///
    /// Takes argmax at each frame and run-length encodes the result.
    ///
    /// - Parameters:
    ///   - activations: `[nFrames * nClasses]` flat chord scores.
    ///   - nFrames: Number of frames.
    ///   - nClasses: Number of classes. Default 25.
    /// - Returns: Array of ChordEvents.
    public static func detectSimple(
        activations: [Float],
        nFrames: Int,
        nClasses: Int = 25
    ) -> [ChordEvent] {
        guard nFrames > 0 else { return [] }
        guard activations.count == nFrames * nClasses else { return [] }

        var path = [Int](repeating: 0, count: nFrames)
        for t in 0..<nFrames {
            var bestIdx = 0
            var bestVal = activations[t * nClasses]
            for c in 1..<nClasses {
                let val = activations[t * nClasses + c]
                if val > bestVal {
                    bestVal = val
                    bestIdx = c
                }
            }
            path[t] = bestIdx
        }

        return runLengthEncode(path: path, nClasses: nClasses)
    }

    // MARK: - Utilities

    /// Convert frame-based events to time-based events.
    ///
    /// - Parameters:
    ///   - events: Array of frame-based ChordEvents.
    ///   - fps: Frames per second.
    /// - Returns: Array of (start time, end time, chord label) tuples.
    public static func eventsToTimes(
        events: [ChordEvent],
        fps: Float
    ) -> [(start: Float, end: Float, chord: String)] {
        guard fps > 0 else { return [] }
        return events.map { event in
            (
                start: Float(event.startFrame) / fps,
                end: Float(event.endFrame) / fps,
                chord: event.chordLabel
            )
        }
    }

    /// Get chord label for a given index.
    ///
    /// - Parameter index: Chord class index (0-24).
    /// - Returns: Chord label string, or "N" if index is out of range.
    public static func label(forIndex index: Int) -> String {
        guard index >= 0 && index < chordLabels.count else { return "N" }
        return chordLabels[index]
    }

    // MARK: - Private Helpers

    /// Run-length encode a path of chord indices into ChordEvents.
    private static func runLengthEncode(path: [Int], nClasses: Int) -> [ChordEvent] {
        guard !path.isEmpty else { return [] }

        var events = [ChordEvent]()
        var currentChord = path[0]
        var startFrame = 0

        for t in 1..<path.count {
            if path[t] != currentChord {
                events.append(ChordEvent(
                    startFrame: startFrame,
                    endFrame: t,
                    chordIndex: currentChord,
                    chordLabel: label(forIndex: currentChord)
                ))
                currentChord = path[t]
                startFrame = t
            }
        }

        // Final event
        events.append(ChordEvent(
            startFrame: startFrame,
            endFrame: path.count,
            chordIndex: currentChord,
            chordLabel: label(forIndex: currentChord)
        ))

        return events
    }
}
