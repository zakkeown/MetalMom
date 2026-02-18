import Foundation

/// Linear-chain Conditional Random Field (CRF) decoding algorithms.
///
/// Unlike HMM (which uses log probabilities that must form valid distributions),
/// CRF operates on raw scores (arbitrary real numbers). The algorithms are
/// structurally identical to HMM but with different semantics:
///
/// - **Unary potentials** `[nFrames][nLabels]`: Score for each label at each position
///   (typically neural network output). Can be any real number.
/// - **Pairwise potentials** `[nLabels][nLabels]`: Score for transitioning from one
///   label to another. `pairwise[i][j]` is the score for transition from label `i`
///   to label `j`. Can be any real number.
///
/// The total score of a label sequence `path` is:
/// ```
/// score = sum(unary[t][path[t]]) + sum(pairwise[path[t]][path[t+1]])
/// ```
///
/// This enum is used internally by chord recognition and other sequence labeling tasks.
public enum CRF {

    // MARK: - Viterbi Decoding

    /// Find the label sequence that maximizes the total CRF score.
    ///
    /// This is the Viterbi algorithm applied to CRF potentials.
    ///
    /// - Parameters:
    ///   - unaryScores: `[nFrames][nLabels]` unary potentials (scores for each label at each position).
    ///   - pairwiseScores: `[nLabels][nLabels]` pairwise potentials (transition scores).
    /// - Returns: The best label sequence and its total score.
    public static func viterbiDecode(
        unaryScores: [[Float]],
        pairwiseScores: [[Float]]
    ) -> (path: [Int], score: Float) {
        let nFrames = unaryScores.count
        guard nFrames > 0 else { return (path: [], score: -.infinity) }

        let nLabels = unaryScores[0].count

        // delta[s] = best total score ending in label s at the current frame
        var deltaPrev = [Float](repeating: -.infinity, count: nLabels)
        // psi[t][s] = best predecessor label for label s at frame t
        var psi = [[Int]](repeating: [Int](repeating: 0, count: nLabels), count: nFrames)

        // Initialize: frame 0 — only unary scores, no pairwise yet
        for s in 0..<nLabels {
            deltaPrev[s] = unaryScores[0][s]
        }

        // Recursion: frames 1..<nFrames
        for t in 1..<nFrames {
            var deltaCurr = [Float](repeating: -.infinity, count: nLabels)
            for s in 0..<nLabels {
                var bestVal: Float = -.infinity
                var bestSrc = 0
                for sPrime in 0..<nLabels {
                    let val = deltaPrev[sPrime] + pairwiseScores[sPrime][s]
                    if val > bestVal {
                        bestVal = val
                        bestSrc = sPrime
                    }
                }
                deltaCurr[s] = bestVal + unaryScores[t][s]
                psi[t][s] = bestSrc
            }
            deltaPrev = deltaCurr
        }

        // Termination: find the best final label
        var bestFinalLabel = 0
        var bestFinalScore: Float = -.infinity
        for s in 0..<nLabels {
            if deltaPrev[s] > bestFinalScore {
                bestFinalScore = deltaPrev[s]
                bestFinalLabel = s
            }
        }

        // Backtrace
        var path = [Int](repeating: 0, count: nFrames)
        path[nFrames - 1] = bestFinalLabel
        for t in stride(from: nFrames - 2, through: 0, by: -1) {
            path[t] = psi[t + 1][path[t + 1]]
        }

        return (path: path, score: bestFinalScore)
    }

    // MARK: - Log-Partition Function

    /// Compute the log-partition function (normalizing constant in the log domain).
    ///
    /// Uses the forward algorithm with log-sum-exp. The partition function Z is the
    /// sum of scores over all possible label sequences (in the exp domain):
    /// `Z = sum over all paths of exp(score(path))`
    ///
    /// - Parameters:
    ///   - unaryScores: `[nFrames][nLabels]` unary potentials.
    ///   - pairwiseScores: `[nLabels][nLabels]` pairwise potentials.
    /// - Returns: `log(Z)`, the log-partition function value.
    public static func logPartition(
        unaryScores: [[Float]],
        pairwiseScores: [[Float]]
    ) -> Float {
        let alpha = forwardPass(unaryScores: unaryScores, pairwiseScores: pairwiseScores)
        guard let lastFrame = alpha.last else { return -.infinity }

        // logZ = logSumExp over labels of alpha at last frame
        var logZ: Float = -.infinity
        for s in 0..<lastFrame.count {
            logZ = logSumExp(logZ, lastFrame[s])
        }
        return logZ
    }

    // MARK: - Marginals

    /// Compute log marginal probabilities for each position and label.
    ///
    /// Uses the forward-backward algorithm. The marginal probability of label `s`
    /// at position `t` is: `P(y_t = s | x) = exp(alpha[t][s] + beta[t][s] - logZ)`
    ///
    /// - Parameters:
    ///   - unaryScores: `[nFrames][nLabels]` unary potentials.
    ///   - pairwiseScores: `[nLabels][nLabels]` pairwise potentials.
    /// - Returns: `[nFrames][nLabels]` log marginal probabilities.
    public static func marginals(
        unaryScores: [[Float]],
        pairwiseScores: [[Float]]
    ) -> [[Float]] {
        let nFrames = unaryScores.count
        guard nFrames > 0 else { return [] }

        let nLabels = unaryScores[0].count

        let alpha = forwardPass(unaryScores: unaryScores, pairwiseScores: pairwiseScores)
        let beta = backwardPass(unaryScores: unaryScores, pairwiseScores: pairwiseScores)

        // logZ from the forward pass at the last frame
        var logZ: Float = -.infinity
        for s in 0..<nLabels {
            logZ = logSumExp(logZ, alpha[nFrames - 1][s])
        }

        // marginals[t][s] = alpha[t][s] + beta[t][s] - logZ
        var result = [[Float]](repeating: [Float](repeating: -.infinity, count: nLabels),
                               count: nFrames)
        for t in 0..<nFrames {
            for s in 0..<nLabels {
                result[t][s] = alpha[t][s] + beta[t][s] - logZ
            }
        }

        return result
    }

    // MARK: - Internal: Forward Pass

    /// Compute forward scores (alpha) for the CRF.
    ///
    /// `alpha[t][s]` is the log of the sum of exp(score) over all partial label
    /// sequences ending in label `s` at position `t`.
    ///
    /// - Parameters:
    ///   - unaryScores: `[nFrames][nLabels]` unary potentials.
    ///   - pairwiseScores: `[nLabels][nLabels]` pairwise potentials.
    /// - Returns: `[nFrames][nLabels]` forward scores.
    static func forwardPass(
        unaryScores: [[Float]],
        pairwiseScores: [[Float]]
    ) -> [[Float]] {
        let nFrames = unaryScores.count
        guard nFrames > 0 else { return [] }

        let nLabels = unaryScores[0].count
        var alpha = [[Float]](repeating: [Float](repeating: -.infinity, count: nLabels),
                              count: nFrames)

        // Initialize: frame 0
        for s in 0..<nLabels {
            alpha[0][s] = unaryScores[0][s]
        }

        // Recursion
        for t in 1..<nFrames {
            for s in 0..<nLabels {
                var acc: Float = -.infinity
                for sPrime in 0..<nLabels {
                    acc = logSumExp(acc, alpha[t - 1][sPrime] + pairwiseScores[sPrime][s])
                }
                alpha[t][s] = acc + unaryScores[t][s]
            }
        }

        return alpha
    }

    // MARK: - Internal: Backward Pass

    /// Compute backward scores (beta) for the CRF.
    ///
    /// `beta[t][s]` is the log of the sum of exp(score) over all partial label
    /// sequences from position `t+1` to the end, given label `s` at position `t`.
    ///
    /// - Parameters:
    ///   - unaryScores: `[nFrames][nLabels]` unary potentials.
    ///   - pairwiseScores: `[nLabels][nLabels]` pairwise potentials.
    /// - Returns: `[nFrames][nLabels]` backward scores.
    static func backwardPass(
        unaryScores: [[Float]],
        pairwiseScores: [[Float]]
    ) -> [[Float]] {
        let nFrames = unaryScores.count
        guard nFrames > 0 else { return [] }

        let nLabels = unaryScores[0].count
        var beta = [[Float]](repeating: [Float](repeating: -.infinity, count: nLabels),
                             count: nFrames)

        // Initialize: last frame, beta[T-1][s] = 0 (log(1))
        for s in 0..<nLabels {
            beta[nFrames - 1][s] = 0
        }

        // Recursion (backwards)
        for t in stride(from: nFrames - 2, through: 0, by: -1) {
            for s in 0..<nLabels {
                var acc: Float = -.infinity
                for sPrime in 0..<nLabels {
                    acc = logSumExp(
                        acc,
                        pairwiseScores[s][sPrime] + unaryScores[t + 1][sPrime] + beta[t + 1][sPrime]
                    )
                }
                beta[t][s] = acc
            }
        }

        return beta
    }

    // MARK: - Private Helpers

    /// Numerically stable log-sum-exp of two values.
    ///
    /// Computes `log(exp(a) + exp(b))` without overflow.
    private static func logSumExp(_ a: Float, _ b: Float) -> Float {
        let maxVal = max(a, b)
        if maxVal == -.infinity { return -.infinity }
        return maxVal + log(exp(a - maxVal) + exp(b - maxVal))
    }
}
