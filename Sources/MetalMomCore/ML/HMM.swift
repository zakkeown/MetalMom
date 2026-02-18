import Foundation

/// General-purpose Hidden Markov Model algorithms.
///
/// All methods operate in the log domain for numerical stability.
/// Inputs are log probabilities; outputs are log probabilities (unless noted).
public enum HMM {

    // MARK: - Viterbi Decoding

    /// Find the most likely state sequence (Viterbi decoding).
    ///
    /// All inputs must be in the log domain.
    ///
    /// - Parameters:
    ///   - logObservations: `[nFrames][nStates]` log observation likelihoods.
    ///   - logInitial: `[nStates]` log initial state probabilities.
    ///   - logTransition: `[nStates][nStates]` log transition probabilities
    ///     where `logTransition[i][j]` is the log probability of transitioning
    ///     from state `i` to state `j`.
    /// - Returns: The most likely state sequence and its log probability.
    public static func viterbi(
        logObservations: [[Float]],
        logInitial: [Float],
        logTransition: [[Float]]
    ) -> (path: [Int], logProbability: Float) {
        let nFrames = logObservations.count
        guard nFrames > 0 else { return (path: [], logProbability: -.infinity) }

        let nStates = logInitial.count

        // delta[s] = best log probability ending in state s at current frame
        var delta = [Float](repeating: -.infinity, count: nStates)
        // psi[t][s] = best predecessor state for state s at frame t
        var psi = [[Int]](repeating: [Int](repeating: 0, count: nStates), count: nFrames)

        // Initialize: frame 0
        for s in 0..<nStates {
            delta[s] = logInitial[s] + logObservations[0][s]
        }

        // Recursion: frames 1..<nFrames
        var deltaPrev = delta
        for t in 1..<nFrames {
            var deltaCurr = [Float](repeating: -.infinity, count: nStates)
            for s in 0..<nStates {
                var bestVal: Float = -.infinity
                var bestSrc = 0
                for sPrime in 0..<nStates {
                    let val = deltaPrev[sPrime] + logTransition[sPrime][s]
                    if val > bestVal {
                        bestVal = val
                        bestSrc = sPrime
                    }
                }
                deltaCurr[s] = bestVal + logObservations[t][s]
                psi[t][s] = bestSrc
            }
            deltaPrev = deltaCurr
        }

        // Termination: find best final state
        var bestFinalState = 0
        var bestFinalVal: Float = -.infinity
        for s in 0..<nStates {
            if deltaPrev[s] > bestFinalVal {
                bestFinalVal = deltaPrev[s]
                bestFinalState = s
            }
        }

        // Backtrace
        var path = [Int](repeating: 0, count: nFrames)
        path[nFrames - 1] = bestFinalState
        for t in stride(from: nFrames - 2, through: 0, by: -1) {
            path[t] = psi[t + 1][path[t + 1]]
        }

        return (path: path, logProbability: bestFinalVal)
    }

    // MARK: - Forward Algorithm

    /// Compute log forward probabilities (alpha).
    ///
    /// `alpha[t][s]` is the log probability of observing frames `0...t`
    /// and being in state `s` at frame `t`.
    ///
    /// - Parameters:
    ///   - logObservations: `[nFrames][nStates]` log observation likelihoods.
    ///   - logInitial: `[nStates]` log initial state probabilities.
    ///   - logTransition: `[nStates][nStates]` log transition probabilities.
    /// - Returns: `[nFrames][nStates]` log forward probabilities.
    public static func forward(
        logObservations: [[Float]],
        logInitial: [Float],
        logTransition: [[Float]]
    ) -> [[Float]] {
        let nFrames = logObservations.count
        guard nFrames > 0 else { return [] }

        let nStates = logInitial.count
        var alpha = [[Float]](repeating: [Float](repeating: -.infinity, count: nStates),
                              count: nFrames)

        // Initialize: frame 0
        for s in 0..<nStates {
            alpha[0][s] = logInitial[s] + logObservations[0][s]
        }

        // Recursion
        for t in 1..<nFrames {
            for s in 0..<nStates {
                var acc: Float = -.infinity
                for sPrime in 0..<nStates {
                    acc = logSumExp(acc, alpha[t - 1][sPrime] + logTransition[sPrime][s])
                }
                alpha[t][s] = acc + logObservations[t][s]
            }
        }

        return alpha
    }

    // MARK: - Backward Algorithm

    /// Compute log backward probabilities (beta).
    ///
    /// `beta[t][s]` is the log probability of observing frames `(t+1)...(T-1)`
    /// given state `s` at frame `t`.
    ///
    /// - Parameters:
    ///   - logObservations: `[nFrames][nStates]` log observation likelihoods.
    ///   - logInitial: `[nStates]` log initial state probabilities (unused but
    ///     included for API consistency).
    ///   - logTransition: `[nStates][nStates]` log transition probabilities.
    /// - Returns: `[nFrames][nStates]` log backward probabilities.
    public static func backward(
        logObservations: [[Float]],
        logInitial: [Float],
        logTransition: [[Float]]
    ) -> [[Float]] {
        let nFrames = logObservations.count
        guard nFrames > 0 else { return [] }

        let nStates = logInitial.count
        var beta = [[Float]](repeating: [Float](repeating: -.infinity, count: nStates),
                             count: nFrames)

        // Initialize: last frame, beta[T-1][s] = log(1) = 0
        for s in 0..<nStates {
            beta[nFrames - 1][s] = 0
        }

        // Recursion (backwards)
        for t in stride(from: nFrames - 2, through: 0, by: -1) {
            for s in 0..<nStates {
                var acc: Float = -.infinity
                for sPrime in 0..<nStates {
                    acc = logSumExp(
                        acc,
                        logTransition[s][sPrime] + logObservations[t + 1][sPrime] + beta[t + 1][sPrime]
                    )
                }
                beta[t][s] = acc
            }
        }

        return beta
    }

    // MARK: - Forward-Backward (Posterior)

    /// Compute log posterior state probabilities via the forward-backward algorithm.
    ///
    /// `gamma[t][s]` is the log probability of being in state `s` at frame `t`
    /// given the entire observation sequence.
    ///
    /// - Parameters:
    ///   - logObservations: `[nFrames][nStates]` log observation likelihoods.
    ///   - logInitial: `[nStates]` log initial state probabilities.
    ///   - logTransition: `[nStates][nStates]` log transition probabilities.
    /// - Returns: `[nFrames][nStates]` log posterior probabilities.
    public static func forwardBackward(
        logObservations: [[Float]],
        logInitial: [Float],
        logTransition: [[Float]]
    ) -> [[Float]] {
        let nFrames = logObservations.count
        guard nFrames > 0 else { return [] }

        let nStates = logInitial.count

        let alpha = forward(logObservations: logObservations,
                            logInitial: logInitial,
                            logTransition: logTransition)
        let beta = backward(logObservations: logObservations,
                            logInitial: logInitial,
                            logTransition: logTransition)

        // Log evidence: logSumExp over states of alpha at the last frame
        var logEvidence: Float = -.infinity
        for s in 0..<nStates {
            logEvidence = logSumExp(logEvidence, alpha[nFrames - 1][s])
        }

        // Posterior: gamma[t][s] = alpha[t][s] + beta[t][s] - logEvidence
        var gamma = [[Float]](repeating: [Float](repeating: -.infinity, count: nStates),
                              count: nFrames)
        for t in 0..<nFrames {
            for s in 0..<nStates {
                gamma[t][s] = alpha[t][s] + beta[t][s] - logEvidence
            }
        }

        return gamma
    }

    // MARK: - Transition Matrix Helpers

    /// Create a uniform transition matrix in the log domain.
    ///
    /// Every transition has equal probability `1/nStates`.
    ///
    /// - Parameter nStates: Number of states.
    /// - Returns: `[nStates][nStates]` log transition matrix.
    public static func uniformTransition(nStates: Int) -> [[Float]] {
        let logP = -log(Float(nStates))
        return [[Float]](repeating: [Float](repeating: logP, count: nStates),
                         count: nStates)
    }

    /// Create a left-to-right transition matrix in the log domain.
    ///
    /// Each state can either stay in the same state (with `selfLoopProb`) or
    /// advance to the next state (with `1 - selfLoopProb`). The last state
    /// can only self-loop.
    ///
    /// - Parameters:
    ///   - nStates: Number of states.
    ///   - selfLoopProb: Probability of staying in the same state. Default 0.9.
    /// - Returns: `[nStates][nStates]` log transition matrix.
    public static func leftToRightTransition(nStates: Int, selfLoopProb: Float = 0.9) -> [[Float]] {
        var trans = [[Float]](repeating: [Float](repeating: -.infinity, count: nStates),
                              count: nStates)
        let logSelf = log(selfLoopProb)
        let logNext = log(1.0 - selfLoopProb)

        for s in 0..<nStates {
            trans[s][s] = logSelf
            if s + 1 < nStates {
                trans[s][s + 1] = logNext
            }
            // Last state: selfLoopProb + (1 - selfLoopProb) = 1.0 → just self-loop
            // But we already set trans[last][last] = logSelf. For the last state,
            // all probability mass should go to self-loop (prob = 1.0).
        }
        // Fix last state: it can only self-loop with probability 1.0
        trans[nStates - 1][nStates - 1] = 0 // log(1.0)

        return trans
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
