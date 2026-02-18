import Foundation

public enum OnsetEval {
    /// Result of onset evaluation.
    public struct Result {
        public let precision: Float
        public let recall: Float
        public let fMeasure: Float
    }

    /// Evaluate predicted onsets against reference onsets.
    ///
    /// Uses a greedy matching algorithm: each reference onset is matched to
    /// the closest unmatched predicted onset within the tolerance window.
    ///
    /// - Parameters:
    ///   - reference: Reference onset times in seconds (sorted).
    ///   - estimated: Estimated onset times in seconds (sorted).
    ///   - window: Tolerance window in seconds. Default 0.05 (50ms).
    /// - Returns: Evaluation result with precision, recall, and F-measure.
    public static func evaluate(
        reference: [Float],
        estimated: [Float],
        window: Float = 0.05
    ) -> Result {
        guard !reference.isEmpty || !estimated.isEmpty else {
            return Result(precision: 0, recall: 0, fMeasure: 0)
        }

        if reference.isEmpty {
            return Result(precision: 0, recall: 0, fMeasure: 0)
        }

        if estimated.isEmpty {
            return Result(precision: 0, recall: 0, fMeasure: 0)
        }

        // Sort both arrays
        let sortedRef = reference.sorted()
        let sortedEst = estimated.sorted()

        // Greedy matching: for each reference onset, find the closest
        // unmatched estimated onset within the tolerance window
        var matchedEst = Set<Int>()
        var truePositives = 0

        for refTime in sortedRef {
            var bestIdx = -1
            var bestDist: Float = Float.infinity

            for (j, estTime) in sortedEst.enumerated() {
                if matchedEst.contains(j) { continue }
                let dist = abs(refTime - estTime)
                if dist <= window && dist < bestDist {
                    bestDist = dist
                    bestIdx = j
                }
            }

            if bestIdx >= 0 {
                matchedEst.insert(bestIdx)
                truePositives += 1
            }
        }

        let precision = Float(truePositives) / Float(sortedEst.count)
        let recall = Float(truePositives) / Float(sortedRef.count)
        let fMeasure: Float
        if precision + recall > 0 {
            fMeasure = 2.0 * precision * recall / (precision + recall)
        } else {
            fMeasure = 0
        }

        return Result(precision: precision, recall: recall, fMeasure: fMeasure)
    }
}
