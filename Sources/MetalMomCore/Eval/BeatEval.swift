import Foundation

public enum BeatEval {
    public struct Result {
        public let fMeasure: Float
        public let cemgil: Float
        public let pScore: Float
        public let cmlC: Float
        public let cmlT: Float
        public let amlC: Float
        public let amlT: Float
    }

    /// Evaluate beat tracking performance.
    ///
    /// Mirrors `mir_eval.beat.evaluate`. Computes F-measure, Cemgil accuracy,
    /// P-score, CML (Correct Metrical Level), and AML (Any Metrical Level).
    ///
    /// - Parameters:
    ///   - reference: Reference beat times in seconds.
    ///   - estimated: Estimated beat times in seconds.
    ///   - fMeasureWindow: Tolerance window for F-measure in seconds. Default 0.07 (70ms).
    /// - Returns: Evaluation result with all beat metrics.
    public static func evaluate(
        reference: [Float],
        estimated: [Float],
        fMeasureWindow: Float = 0.07
    ) -> Result {
        let fMeasure = computeFMeasure(reference: reference, estimated: estimated, window: fMeasureWindow)
        let cemgil = computeCemgil(reference: reference, estimated: estimated)
        let pScore = computePScore(reference: reference, estimated: estimated)
        let (cmlC, cmlT) = computeCML(reference: reference, estimated: estimated)
        let (amlC, amlT) = computeAML(reference: reference, estimated: estimated)

        return Result(fMeasure: fMeasure, cemgil: cemgil, pScore: pScore,
                      cmlC: cmlC, cmlT: cmlT, amlC: amlC, amlT: amlT)
    }

    // MARK: - F-Measure (reuse OnsetEval)

    private static func computeFMeasure(reference: [Float], estimated: [Float], window: Float) -> Float {
        let result = OnsetEval.evaluate(reference: reference, estimated: estimated, window: window)
        return result.fMeasure
    }

    // MARK: - Cemgil accuracy

    /// Gaussian error-based beat accuracy (Cemgil et al., 2001).
    private static func computeCemgil(reference: [Float], estimated: [Float], sigma: Float = 0.04) -> Float {
        guard !reference.isEmpty && !estimated.isEmpty else { return 0 }

        var totalScore: Float = 0
        for refBeat in reference {
            var bestScore: Float = 0
            for estBeat in estimated {
                let error = refBeat - estBeat
                let score = expf(-0.5 * (error * error) / (sigma * sigma))
                bestScore = max(bestScore, score)
            }
            totalScore += bestScore
        }
        return totalScore / Float(reference.count)
    }

    // MARK: - P-score (cross-correlation)

    /// Beat cross-correlation score using Gaussian-windowed indicator functions.
    private static func computePScore(reference: [Float], estimated: [Float], window: Float = 0.02) -> Float {
        guard !reference.isEmpty && !estimated.isEmpty else { return 0 }

        // Create beat indicator functions at high resolution and cross-correlate
        let maxTime = max(reference.max() ?? 0, estimated.max() ?? 0) + 1.0
        let fs: Float = 100.0  // 100 Hz resolution
        let nSamples = Int(maxTime * fs) + 1

        var refIndicator = [Float](repeating: 0, count: nSamples)
        var estIndicator = [Float](repeating: 0, count: nSamples)

        // Gaussian windows around each beat
        let halfWidth = Int(window * fs * 3)  // 3 sigma
        for beat in reference {
            let center = Int(beat * fs)
            for i in max(0, center - halfWidth)..<min(nSamples, center + halfWidth + 1) {
                let t = (Float(i) - beat * fs) / (window * fs)
                refIndicator[i] += expf(-0.5 * t * t)
            }
        }
        for beat in estimated {
            let center = Int(beat * fs)
            for i in max(0, center - halfWidth)..<min(nSamples, center + halfWidth + 1) {
                let t = (Float(i) - beat * fs) / (window * fs)
                estIndicator[i] += expf(-0.5 * t * t)
            }
        }

        // Normalized cross-correlation at zero lag
        var dotProduct: Float = 0
        var refNorm: Float = 0
        var estNorm: Float = 0
        for i in 0..<nSamples {
            dotProduct += refIndicator[i] * estIndicator[i]
            refNorm += refIndicator[i] * refIndicator[i]
            estNorm += estIndicator[i] * estIndicator[i]
        }

        let denom = sqrtf(refNorm * estNorm)
        return denom > 0 ? dotProduct / denom : 0
    }

    // MARK: - CML (Correct Metrical Level)

    /// Correct Metrical Level continuity: longest continuous segment of correct beats.
    private static func computeCML(reference: [Float], estimated: [Float], window: Float = 0.175) -> (Float, Float) {
        guard reference.count >= 2 && estimated.count >= 2 else { return (0, 0) }

        let sortedRef = reference.sorted()
        let sortedEst = estimated.sorted()

        // Compute inter-beat intervals
        var refIBI = [Float](repeating: 0, count: sortedRef.count - 1)
        for i in 0..<refIBI.count { refIBI[i] = sortedRef[i + 1] - sortedRef[i] }

        // Find longest continuous segment where est matches ref
        var maxContinuous = 0
        var current = 0

        var j = 0
        for i in 0..<sortedRef.count {
            // Find closest estimated beat
            while j < sortedEst.count - 1 && sortedEst[j] < sortedRef[i] - window * (refIBI.isEmpty ? 0.5 : refIBI[min(i, refIBI.count - 1)]) {
                j += 1
            }

            if j < sortedEst.count {
                let tolerance = window * (i < refIBI.count ? refIBI[i] : (refIBI.last ?? 0.5))
                if abs(sortedRef[i] - sortedEst[j]) <= tolerance {
                    current += 1
                    maxContinuous = max(maxContinuous, current)
                } else {
                    current = 0
                }
            } else {
                current = 0
            }
        }

        let cmlC = Float(maxContinuous) / Float(sortedRef.count)
        let cmlT = cmlC  // Total = continuity for now
        return (cmlC, cmlT)
    }

    // MARK: - AML (Any Metrical Level)

    /// Any Metrical Level: allows half/double tempo matches.
    private static func computeAML(reference: [Float], estimated: [Float], window: Float = 0.175) -> (Float, Float) {
        guard reference.count >= 2 && estimated.count >= 2 else { return (0, 0) }

        // AML: try original tempo, half tempo, double tempo, off-beat
        var bestCmlC: Float = 0
        var bestCmlT: Float = 0

        // Original
        let (c1, t1) = computeCML(reference: reference, estimated: estimated, window: window)
        if c1 > bestCmlC { bestCmlC = c1; bestCmlT = t1 }

        // Double tempo: insert beats between each estimated beat
        if estimated.count >= 2 {
            var doubled = [Float]()
            let sortedEst = estimated.sorted()
            for i in 0..<sortedEst.count {
                doubled.append(sortedEst[i])
                if i < sortedEst.count - 1 {
                    doubled.append((sortedEst[i] + sortedEst[i + 1]) / 2)
                }
            }
            let (c2, t2) = computeCML(reference: reference, estimated: doubled, window: window)
            if c2 > bestCmlC { bestCmlC = c2; bestCmlT = t2 }
        }

        // Half tempo: take every other beat
        if estimated.count >= 4 {
            let sortedEst = estimated.sorted()
            let half1 = stride(from: 0, to: sortedEst.count, by: 2).map { sortedEst[$0] }
            let half2 = stride(from: 1, to: sortedEst.count, by: 2).map { sortedEst[$0] }

            let (c3, t3) = computeCML(reference: reference, estimated: half1, window: window)
            if c3 > bestCmlC { bestCmlC = c3; bestCmlT = t3 }

            let (c4, t4) = computeCML(reference: reference, estimated: half2, window: window)
            if c4 > bestCmlC { bestCmlC = c4; bestCmlT = t4 }
        }

        return (bestCmlC, bestCmlT)
    }
}
