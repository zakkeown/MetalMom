import Foundation

public enum ChordEval {
    /// Evaluate chord recognition by computing frame-level accuracy.
    ///
    /// Simplified: compares chord labels at each time point.
    /// Each chord is represented as an integer label.
    ///
    /// - Parameters:
    ///   - reference: Reference chord labels (one per frame).
    ///   - estimated: Estimated chord labels (one per frame).
    /// - Returns: Accuracy (fraction of frames with matching labels).
    public static func accuracy(
        reference: [Int32],
        estimated: [Int32]
    ) -> Float {
        guard !reference.isEmpty else { return 0 }
        let n = min(reference.count, estimated.count)
        guard n > 0 else { return 0 }

        var correct = 0
        for i in 0..<n {
            if reference[i] == estimated[i] {
                correct += 1
            }
        }
        return Float(correct) / Float(n)
    }

    /// Evaluate chord recognition with weighted segment overlap.
    ///
    /// Mirrors `mir_eval.chord.weighted_overlap`. Computes accuracy weighted
    /// by the duration of each reference chord segment.
    ///
    /// - Parameters:
    ///   - refIntervals: Reference interval boundaries [[start, end], ...].
    ///   - refLabels: Reference chord labels per segment.
    ///   - estIntervals: Estimated interval boundaries.
    ///   - estLabels: Estimated chord labels per segment.
    /// - Returns: Weighted overlap accuracy.
    public static func weightedOverlap(
        refIntervals: [[Float]],
        refLabels: [Int32],
        estIntervals: [[Float]],
        estLabels: [Int32]
    ) -> Float {
        guard !refIntervals.isEmpty && !estIntervals.isEmpty else { return 0 }

        var totalDuration: Float = 0
        var correctDuration: Float = 0

        for (ri, refInterval) in refIntervals.enumerated() {
            guard ri < refLabels.count else { break }
            let refStart = refInterval[0]
            let refEnd = refInterval[1]
            let refLabel = refLabels[ri]

            for (ei, estInterval) in estIntervals.enumerated() {
                guard ei < estLabels.count else { break }
                let estStart = estInterval[0]
                let estEnd = estInterval[1]

                // Compute overlap
                let overlapStart = max(refStart, estStart)
                let overlapEnd = min(refEnd, estEnd)

                if overlapEnd > overlapStart {
                    let overlapDur = overlapEnd - overlapStart
                    if refLabel == estLabels[ei] {
                        correctDuration += overlapDur
                    }
                }
            }
            totalDuration += refEnd - refStart
        }

        return totalDuration > 0 ? correctDuration / totalDuration : 0
    }
}
