import Foundation

/// CNN-based key detection from activation probabilities.
///
/// Compatible with madmom's ``CNNKeyRecognitionProcessor`` output.
/// The 24-dimensional activation vector follows the standard ordering:
///   - Index 0-11: A major, A# major, B major, C major, C# major, D major,
///                 D# major, E major, F major, F# major, G major, G# major
///   - Index 12-23: A minor, A# minor, B minor, C minor, C# minor, D minor,
///                  D# minor, E minor, F minor, F# minor, G minor, G# minor
public enum KeyDetection {

    /// Musical key labels in the standard order used by madmom.
    /// Index 0-11: A major through G# major
    /// Index 12-23: A minor through G# minor
    public static let keyLabels: [String] = [
        "A major", "A# major", "B major", "C major", "C# major", "D major",
        "D# major", "E major", "F major", "F# major", "G major", "G# major",
        "A minor", "A# minor", "B minor", "C minor", "C# minor", "D minor",
        "D# minor", "E minor", "F minor", "F# minor", "G minor", "G# minor"
    ]

    /// Result of key detection.
    public struct KeyResult: Sendable {
        /// Detected key index (0-23).
        public let keyIndex: Int
        /// Key label string (e.g., "C major").
        public let keyLabel: String
        /// Whether the key is major (true) or minor (false).
        public let isMajor: Bool
        /// The pitch class (0=A, 1=A#, ..., 11=G#).
        public let pitchClass: Int
        /// Confidence (probability of the detected key).
        public let confidence: Float
        /// All 24 key probabilities.
        public let probabilities: [Float]
    }

    /// Detect the musical key from CNN activation probabilities.
    ///
    /// - Parameters:
    ///   - activations: [24] probabilities for each key class.
    ///     Index 0-11: major keys (A, A#, B, C, ..., G#)
    ///     Index 12-23: minor keys (A, A#, B, C, ..., G#)
    /// - Returns: KeyResult with the detected key.
    public static func detect(activations: [Float]) -> KeyResult {
        precondition(activations.count == 24,
                     "KeyDetection.detect requires exactly 24 activations, got \(activations.count)")

        // Find argmax
        var bestIndex = 0
        var bestValue = activations[0]
        for i in 1..<24 {
            if activations[i] > bestValue {
                bestValue = activations[i]
                bestIndex = i
            }
        }

        let isMajor = bestIndex < 12
        let pitchClass = bestIndex % 12

        return KeyResult(
            keyIndex: bestIndex,
            keyLabel: keyLabels[bestIndex],
            isMajor: isMajor,
            pitchClass: pitchClass,
            confidence: bestValue,
            probabilities: activations
        )
    }

    /// Detect key from a sequence of frame-level activations.
    ///
    /// Averages the per-frame activations, then picks the key with
    /// highest average probability. This is the typical usage when
    /// processing a full song.
    ///
    /// - Parameters:
    ///   - activations: [nFrames * 24] flat array of per-frame key probabilities.
    ///   - nFrames: Number of frames.
    /// - Returns: KeyResult for the whole sequence.
    public static func detectFromSequence(activations: [Float], nFrames: Int) -> KeyResult {
        precondition(activations.count == nFrames * 24,
                     "activations.count (\(activations.count)) must equal nFrames * 24 (\(nFrames * 24))")
        guard nFrames > 0 else {
            // Return a default result for empty input
            let zeros = [Float](repeating: 0, count: 24)
            return KeyResult(
                keyIndex: 0,
                keyLabel: keyLabels[0],
                isMajor: true,
                pitchClass: 0,
                confidence: 0,
                probabilities: zeros
            )
        }

        // Average activations across frames
        var averaged = [Float](repeating: 0, count: 24)
        let invN = 1.0 / Float(nFrames)
        for frame in 0..<nFrames {
            let offset = frame * 24
            for i in 0..<24 {
                averaged[i] += activations[offset + i]
            }
        }
        for i in 0..<24 {
            averaged[i] *= invN
        }

        return detect(activations: averaged)
    }

    /// Get the key label for a given index.
    public static func label(forIndex index: Int) -> String {
        precondition(index >= 0 && index < 24, "Key index must be 0-23, got \(index)")
        return keyLabels[index]
    }

    /// Get the relative major/minor key index.
    /// If given a minor key, returns the relative major. Vice versa.
    ///
    /// For example:
    /// - C major (index 3) -> A minor (index 12)
    /// - A minor (index 12) -> C major (index 3)
    /// - G major (index 10) -> E minor (index 19)
    /// - E minor (index 19) -> G major (index 10)
    public static func relativeKey(index: Int) -> Int {
        precondition(index >= 0 && index < 24, "Key index must be 0-23, got \(index)")

        if index < 12 {
            // Major -> relative minor: go up 9 semitones (or down 3)
            // Major key at pitch class p -> relative minor at pitch class (p + 9) % 12
            let minorPitchClass = (index + 9) % 12
            return minorPitchClass + 12
        } else {
            // Minor -> relative major: go up 3 semitones
            // Minor key at pitch class p -> relative major at pitch class (p + 3) % 12
            let majorPitchClass = ((index - 12) + 3) % 12
            return majorPitchClass
        }
    }
}
