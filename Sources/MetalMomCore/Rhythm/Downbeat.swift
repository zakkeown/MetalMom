import Accelerate
import Foundation

/// Downbeat detection from neural network activations.
///
/// Replicates the approach of madmom's `DBNDownBeatTrackingProcessor`:
/// 1. RNN ensemble produces activations with shape [nFrames, 3]:
///    P(no beat), P(beat), P(downbeat)
/// 2. Beat positions are found via combined beat evidence
/// 3. Downbeats are identified using bar-position decoding
///
/// The `decode` method is the core testable component that works with
/// pre-computed activations (no CoreML required).
public enum Downbeat {

    // MARK: - Types

    /// Output labels for downbeat detection.
    public enum Label: Int {
        case noBeat = 0
        case beat = 1
        case downbeat = 2
    }

    // MARK: - Core Decoder

    /// Decode downbeat positions from activation probabilities.
    ///
    /// Takes neural network output with per-frame probabilities for
    /// (no-beat, beat, downbeat) and finds the most likely sequence
    /// of bar positions using beat tracking + downbeat classification.
    ///
    /// Algorithm:
    /// 1. Combine beat evidence: P(beat) + P(downbeat) forms beat activation
    /// 2. Use DP beat tracking (via NeuralBeatTracker) to find beat positions
    /// 3. For each beat, compute downbeat score = P(downbeat) / (P(beat) + P(downbeat))
    /// 4. Find the anchor beat with highest downbeat score
    /// 5. Mark every `beatsPerBar`-th beat as downbeat relative to anchor
    ///
    /// - Parameters:
    ///   - activations: [nFrames * 3] activation probabilities (row-major).
    ///     For frame t: activations[t*3+0] = P(no beat),
    ///                  activations[t*3+1] = P(beat),
    ///                  activations[t*3+2] = P(downbeat).
    ///   - nFrames: Number of frames.
    ///   - fps: Frames per second. Default 100.
    ///   - beatsPerBar: Expected beats per bar. Default 4 (4/4 time).
    ///   - minBPM: Minimum tempo. Default 55.
    ///   - maxBPM: Maximum tempo. Default 215.
    ///   - transitionLambda: Tempo smoothness parameter. Default 100.
    /// - Returns: Named tuple with:
    ///   - beatFrames: all beat positions (including downbeats)
    ///   - downbeatFrames: only downbeat positions
    ///   - labels: per-frame label assignments
    public static func decode(
        activations: [Float],
        nFrames: Int,
        fps: Float = 100.0,
        beatsPerBar: Int = 4,
        minBPM: Float = 55.0,
        maxBPM: Float = 215.0,
        transitionLambda: Float = 100.0
    ) -> (beatFrames: [Int], downbeatFrames: [Int], labels: [Label]) {
        guard nFrames > 0 else {
            return ([], [], [])
        }
        guard activations.count >= nFrames * 3 else {
            return ([], [], [])
        }

        // 1. Extract per-class activations and combine beat evidence
        var beatActivation = [Float](repeating: 0, count: nFrames)
        var downbeatScore = [Float](repeating: 0, count: nFrames)

        for t in 0..<nFrames {
            let pBeat = activations[t * 3 + 1]
            let pDownbeat = activations[t * 3 + 2]

            // Combined beat activation (both beats and downbeats are beats)
            beatActivation[t] = pBeat + pDownbeat

            // Downbeat score relative to total beat evidence
            let total = pBeat + pDownbeat
            if total > 1e-7 {
                downbeatScore[t] = pDownbeat / total
            } else {
                downbeatScore[t] = 0
            }
        }

        // 2. Use NeuralBeatTracker to find beat positions from combined activation
        let (_, beatFrames) = NeuralBeatTracker.decode(
            activations: beatActivation,
            fps: fps,
            minBPM: minBPM,
            maxBPM: maxBPM,
            transitionLambda: transitionLambda,
            threshold: 0.05,
            trim: false
        )

        guard !beatFrames.isEmpty else {
            let labels = [Label](repeating: .noBeat, count: nFrames)
            return ([], [], labels)
        }

        // 3. Find anchor beat (the beat with the highest downbeat score)
        var anchorIndex = 0
        var bestDownbeatScore: Float = -1
        for (i, frame) in beatFrames.enumerated() {
            if downbeatScore[frame] > bestDownbeatScore {
                bestDownbeatScore = downbeatScore[frame]
                anchorIndex = i
            }
        }

        // 4. Mark downbeats using modular counting from the anchor
        let bpb = max(1, beatsPerBar)
        var downbeatFrames: [Int] = []

        for (i, frame) in beatFrames.enumerated() {
            // Compute bar position relative to anchor
            // anchor is position 0 (downbeat), then 1, 2, ... bpb-1
            let offset = ((i - anchorIndex) % bpb + bpb) % bpb
            if offset == 0 {
                downbeatFrames.append(frame)
            }
        }

        // 5. Build per-frame labels
        let beatSet = Set(beatFrames)
        let downbeatSet = Set(downbeatFrames)
        var labels = [Label](repeating: .noBeat, count: nFrames)

        for frame in beatFrames {
            if downbeatSet.contains(frame) {
                labels[frame] = .downbeat
            } else if beatSet.contains(frame) {
                labels[frame] = .beat
            }
        }

        return (beatFrames, downbeatFrames, labels)
    }
}
