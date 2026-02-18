import Foundation

public enum TempoEval {
    /// Evaluate tempo estimation.
    ///
    /// Returns true if estimated tempo matches reference within tolerance.
    /// Allows double/half tempo matches (standard in MIR).
    ///
    /// - Parameters:
    ///   - referenceTempo: Reference tempo in BPM.
    ///   - estimatedTempo: Estimated tempo in BPM.
    ///   - tolerance: Relative tolerance. Default 0.08 (8%, standard in mir_eval).
    /// - Returns: Whether the estimated tempo matches.
    public static func evaluate(
        referenceTempo: Float,
        estimatedTempo: Float,
        tolerance: Float = 0.08
    ) -> Bool {
        guard referenceTempo > 0 && estimatedTempo > 0 else { return false }

        // Check 1x, 2x, 0.5x, 3x, 1/3x tempo matches
        let ratios: [Float] = [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0]
        for ratio in ratios {
            let target = referenceTempo * ratio
            let error = abs(estimatedTempo - target) / target
            if error <= tolerance {
                return true
            }
        }
        return false
    }

    /// P-score for tempo: 1 if match, 0 if not.
    ///
    /// - Parameters:
    ///   - referenceTempo: Reference tempo in BPM.
    ///   - estimatedTempo: Estimated tempo in BPM.
    ///   - tolerance: Relative tolerance. Default 0.08 (8%).
    /// - Returns: 1.0 if match, 0.0 if not.
    public static func pScore(
        referenceTempo: Float,
        estimatedTempo: Float,
        tolerance: Float = 0.08
    ) -> Float {
        return evaluate(referenceTempo: referenceTempo, estimatedTempo: estimatedTempo, tolerance: tolerance) ? 1.0 : 0.0
    }
}
