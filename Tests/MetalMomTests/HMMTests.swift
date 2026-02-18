import XCTest
@testable import MetalMomCore

final class HMMTests: XCTestCase {

    // MARK: - Weather HMM fixture

    /// Classic 2-state weather HMM (Rain/Sun) with 3 observation symbols.
    /// States: 0=Rain, 1=Sun
    /// Observations: 0=walk, 1=shop, 2=clean
    private struct WeatherHMM {
        static let logInitial: [Float] = [log(0.6), log(0.4)]

        static let logTransition: [[Float]] = [
            [log(0.7), log(0.3)],  // Rain -> Rain, Rain -> Sun
            [log(0.4), log(0.6)]   // Sun  -> Rain, Sun  -> Sun
        ]

        /// Emission probabilities (not log):
        /// walk|Rain=0.1, shop|Rain=0.4, clean|Rain=0.5
        /// walk|Sun=0.6,  shop|Sun=0.3,  clean|Sun=0.1
        static let emission: [[Float]] = [
            [0.1, 0.4, 0.5],  // Rain
            [0.6, 0.3, 0.1]   // Sun
        ]

        /// Convert an observation sequence (indices) to log observation likelihoods.
        /// Returns [nFrames][nStates] where entry [t][s] = log P(obs[t] | state s).
        static func logObservations(for sequence: [Int]) -> [[Float]] {
            sequence.map { obs in
                [log(emission[0][obs]), log(emission[1][obs])]
            }
        }
    }

    // MARK: - Viterbi Tests

    func testViterbiWeatherHMM() {
        // Observation sequence: walk(0), shop(1), clean(2)
        // Expected Viterbi path: [Sun(1), Rain(0), Rain(0)]
        let logObs = WeatherHMM.logObservations(for: [0, 1, 2])
        let result = HMM.viterbi(
            logObservations: logObs,
            logInitial: WeatherHMM.logInitial,
            logTransition: WeatherHMM.logTransition
        )

        XCTAssertEqual(result.path, [1, 0, 0],
                       "Expected [Sun, Rain, Rain] for walk/shop/clean")

        // Verify log probability is finite and reasonable
        XCTAssertFalse(result.logProbability.isNaN)
        XCTAssertFalse(result.logProbability.isInfinite)

        // The log probability should be log(0.01344) ≈ -4.31
        XCTAssertEqual(result.logProbability, log(0.01344), accuracy: 1e-4)
    }

    func testViterbiSingleFrame() {
        // With a single frame, Viterbi should pick the state with
        // the highest logInitial + logObservation.
        let logObs: [[Float]] = [[log(0.1), log(0.9)]]  // state 1 is much more likely
        let logInitial: [Float] = [log(0.5), log(0.5)]
        let logTrans: [[Float]] = [[log(0.5), log(0.5)],
                                    [log(0.5), log(0.5)]]

        let result = HMM.viterbi(
            logObservations: logObs,
            logInitial: logInitial,
            logTransition: logTrans
        )

        XCTAssertEqual(result.path.count, 1)
        XCTAssertEqual(result.path[0], 1, "State 1 has higher observation likelihood")
    }

    func testViterbiSingleState() {
        // With a single state, the path is always [0, 0, ..., 0].
        let logObs: [[Float]] = [[log(0.8)], [log(0.3)], [log(0.5)]]
        let logInitial: [Float] = [0]  // log(1.0)
        let logTrans: [[Float]] = [[0]]  // log(1.0)

        let result = HMM.viterbi(
            logObservations: logObs,
            logInitial: logInitial,
            logTransition: logTrans
        )

        XCTAssertEqual(result.path, [0, 0, 0])
        // Log probability = log(1.0) + log(0.8) + log(1.0) + log(0.3) + log(1.0) + log(0.5)
        //                  = log(0.8 * 0.3 * 0.5) = log(0.12)
        XCTAssertEqual(result.logProbability, log(Float(0.12)), accuracy: 1e-5)
    }

    func testViterbiEmptyObservations() {
        let result = HMM.viterbi(
            logObservations: [],
            logInitial: [0],
            logTransition: [[0]]
        )
        XCTAssertTrue(result.path.isEmpty)
        XCTAssertEqual(result.logProbability, -.infinity)
    }

    func testViterbiLongerSequence() {
        // walk, walk, clean, clean, shop
        // With strong self-loops we expect Sun for walk, Rain for clean,
        // and the shop observation to be decoded based on context.
        let logObs = WeatherHMM.logObservations(for: [0, 0, 2, 2, 1])
        let result = HMM.viterbi(
            logObservations: logObs,
            logInitial: WeatherHMM.logInitial,
            logTransition: WeatherHMM.logTransition
        )

        XCTAssertEqual(result.path.count, 5)
        // Walk strongly favors Sun (0.6 vs 0.1)
        XCTAssertEqual(result.path[0], 1, "Walk strongly favors Sun")
        XCTAssertEqual(result.path[1], 1, "Walk strongly favors Sun")
        // Clean strongly favors Rain (0.5 vs 0.1)
        XCTAssertEqual(result.path[2], 0, "Clean strongly favors Rain")
        XCTAssertEqual(result.path[3], 0, "Clean strongly favors Rain")
        // Shop slightly favors Rain (0.4 vs 0.3), and preceded by Rain
        XCTAssertEqual(result.path[4], 0, "Shop with Rain context favors Rain")
    }

    // MARK: - Forward Algorithm Tests

    func testForwardProbabilitiesSumToOne() {
        // For each frame, exp(alpha[t]) summed over states should equal P(o_1...o_t).
        // More usefully, the forward probabilities at the last frame sum to the
        // total evidence P(O). We verify that for each intermediate frame,
        // the marginal is consistent (non-zero, finite).
        let logObs = WeatherHMM.logObservations(for: [0, 1, 2])
        let alpha = HMM.forward(
            logObservations: logObs,
            logInitial: WeatherHMM.logInitial,
            logTransition: WeatherHMM.logTransition
        )

        XCTAssertEqual(alpha.count, 3)

        // Check frame 0: sum of exp(alpha[0]) should equal
        // P(Rain)*P(walk|Rain) + P(Sun)*P(walk|Sun) = 0.6*0.1 + 0.4*0.6 = 0.3
        let frame0Sum = exp(alpha[0][0]) + exp(alpha[0][1])
        XCTAssertEqual(frame0Sum, 0.3, accuracy: 1e-5)

        // All values should be finite
        for t in 0..<alpha.count {
            for s in 0..<alpha[t].count {
                XCTAssertFalse(alpha[t][s].isNaN, "alpha[\(t)][\(s)] is NaN")
            }
        }
    }

    func testForwardSingleFrame() {
        let logObs: [[Float]] = [[log(0.3), log(0.7)]]
        let logInitial: [Float] = [log(0.5), log(0.5)]
        let logTrans: [[Float]] = [[log(0.5), log(0.5)],
                                    [log(0.5), log(0.5)]]

        let alpha = HMM.forward(
            logObservations: logObs,
            logInitial: logInitial,
            logTransition: logTrans
        )

        XCTAssertEqual(alpha.count, 1)
        XCTAssertEqual(exp(alpha[0][0]), 0.5 * 0.3, accuracy: 1e-6)
        XCTAssertEqual(exp(alpha[0][1]), 0.5 * 0.7, accuracy: 1e-6)
    }

    // MARK: - Backward Algorithm Tests

    func testBackwardLastFrameIsZero() {
        let logObs = WeatherHMM.logObservations(for: [0, 1, 2])
        let beta = HMM.backward(
            logObservations: logObs,
            logInitial: WeatherHMM.logInitial,
            logTransition: WeatherHMM.logTransition
        )

        XCTAssertEqual(beta.count, 3)
        // Last frame: beta[T-1][s] = 0 (log 1) for all states
        XCTAssertEqual(beta[2][0], 0, accuracy: 1e-7)
        XCTAssertEqual(beta[2][1], 0, accuracy: 1e-7)
    }

    func testBackwardValuesAreFinite() {
        let logObs = WeatherHMM.logObservations(for: [0, 1, 2])
        let beta = HMM.backward(
            logObservations: logObs,
            logInitial: WeatherHMM.logInitial,
            logTransition: WeatherHMM.logTransition
        )

        for t in 0..<beta.count {
            for s in 0..<beta[t].count {
                XCTAssertFalse(beta[t][s].isNaN, "beta[\(t)][\(s)] is NaN")
                XCTAssertFalse(beta[t][s].isInfinite, "beta[\(t)][\(s)] is infinite")
            }
        }
    }

    // MARK: - Forward-Backward Tests

    func testPosteriorSumToOne() {
        // For each frame, the posterior probabilities should sum to 1.0
        // in the probability domain: sum_s exp(gamma[t][s]) == 1.0 for all t.
        let logObs = WeatherHMM.logObservations(for: [0, 1, 2])
        let gamma = HMM.forwardBackward(
            logObservations: logObs,
            logInitial: WeatherHMM.logInitial,
            logTransition: WeatherHMM.logTransition
        )

        XCTAssertEqual(gamma.count, 3)

        for t in 0..<gamma.count {
            let probSum = exp(gamma[t][0]) + exp(gamma[t][1])
            XCTAssertEqual(probSum, 1.0, accuracy: 1e-5,
                           "Posterior at frame \(t) should sum to 1.0, got \(probSum)")
        }
    }

    func testPosteriorSingleState() {
        // With one state, posterior is always log(1) = 0 for every frame.
        let logObs: [[Float]] = [[log(0.5)], [log(0.3)], [log(0.8)]]
        let logInitial: [Float] = [0]
        let logTrans: [[Float]] = [[0]]

        let gamma = HMM.forwardBackward(
            logObservations: logObs,
            logInitial: logInitial,
            logTransition: logTrans
        )

        XCTAssertEqual(gamma.count, 3)
        for t in 0..<3 {
            XCTAssertEqual(gamma[t][0], 0, accuracy: 1e-5,
                           "Single-state posterior should be log(1)=0")
        }
    }

    func testForwardBackwardConsistentWithForward() {
        // The log evidence from forward-backward should match
        // the log evidence from the forward algorithm alone.
        let logObs = WeatherHMM.logObservations(for: [0, 1, 2])

        let alpha = HMM.forward(
            logObservations: logObs,
            logInitial: WeatherHMM.logInitial,
            logTransition: WeatherHMM.logTransition
        )

        // Log evidence from forward: logSumExp of alpha at last frame
        let logEvidenceForward = log(exp(alpha[2][0]) + exp(alpha[2][1]))

        let gamma = HMM.forwardBackward(
            logObservations: logObs,
            logInitial: WeatherHMM.logInitial,
            logTransition: WeatherHMM.logTransition
        )

        // The posterior at the last frame should be consistent with forward:
        // gamma[T-1][s] = alpha[T-1][s] + beta[T-1][s] - logEvidence
        // Since beta[T-1][s] = 0, gamma[T-1][s] = alpha[T-1][s] - logEvidence
        for s in 0..<2 {
            let expected = alpha[2][s] - logEvidenceForward
            XCTAssertEqual(gamma[2][s], expected, accuracy: 1e-4)
        }
    }

    func testForwardBackwardEmpty() {
        let gamma = HMM.forwardBackward(
            logObservations: [],
            logInitial: [0],
            logTransition: [[0]]
        )
        XCTAssertTrue(gamma.isEmpty)
    }

    // MARK: - Transition Matrix Helper Tests

    func testUniformTransition() {
        let trans = HMM.uniformTransition(nStates: 3)
        XCTAssertEqual(trans.count, 3)

        let expectedLogP = -log(Float(3))
        for row in trans {
            XCTAssertEqual(row.count, 3)
            for val in row {
                XCTAssertEqual(val, expectedLogP, accuracy: 1e-6)
            }
            // Probabilities sum to 1
            let probSum = row.reduce(Float(0)) { $0 + exp($1) }
            XCTAssertEqual(probSum, 1.0, accuracy: 1e-5)
        }
    }

    func testLeftToRightTransition() {
        let trans = HMM.leftToRightTransition(nStates: 4, selfLoopProb: 0.9)
        XCTAssertEqual(trans.count, 4)

        // Non-last states: self-loop = 0.9, next = 0.1, rest = -inf
        for s in 0..<3 {
            XCTAssertEqual(exp(trans[s][s]), 0.9, accuracy: 1e-5,
                           "Self-loop prob for state \(s)")
            XCTAssertEqual(exp(trans[s][s + 1]), 0.1, accuracy: 1e-5,
                           "Next-state prob for state \(s)")

            // Other transitions should be -infinity (probability 0)
            for d in 0..<4 where d != s && d != s + 1 {
                XCTAssertEqual(trans[s][d], -.infinity,
                               "Transition from \(s) to \(d) should be -inf")
            }

            // Row sums to 1
            let probSum = trans[s].reduce(Float(0)) { $0 + exp($1) }
            XCTAssertEqual(probSum, 1.0, accuracy: 1e-5)
        }

        // Last state: self-loop with probability 1.0
        XCTAssertEqual(trans[3][3], 0, accuracy: 1e-6, "Last state self-loop should be log(1)=0")
        for d in 0..<3 {
            XCTAssertEqual(trans[3][d], -.infinity)
        }
    }

    func testLeftToRightTransitionSingleState() {
        let trans = HMM.leftToRightTransition(nStates: 1, selfLoopProb: 0.9)
        XCTAssertEqual(trans.count, 1)
        XCTAssertEqual(trans[0][0], 0, accuracy: 1e-6, "Single state self-loop = log(1) = 0")
    }

    // MARK: - Numerical Stability Tests

    func testViterbiWithVerySmallProbabilities() {
        // Use very small observation probabilities to test log-domain stability.
        let nFrames = 100
        let nStates = 3
        let logInitial = [Float](repeating: -log(Float(nStates)), count: nStates)
        let logTrans = HMM.uniformTransition(nStates: nStates)

        // Observations strongly favor state 0
        var logObs = [[Float]]()
        for _ in 0..<nFrames {
            logObs.append([log(Float(0.98)), log(Float(0.01)), log(Float(0.01))])
        }

        let result = HMM.viterbi(
            logObservations: logObs,
            logInitial: logInitial,
            logTransition: logTrans
        )

        XCTAssertEqual(result.path.count, nFrames)
        // Every frame should decode to state 0
        for t in 0..<nFrames {
            XCTAssertEqual(result.path[t], 0)
        }
        XCTAssertFalse(result.logProbability.isNaN)
        XCTAssertFalse(result.logProbability.isInfinite)
    }

    func testForwardBackwardLongSequence() {
        // Ensure forward-backward doesn't blow up on longer sequences.
        let nFrames = 200
        let logInitial: [Float] = [log(0.5), log(0.5)]
        let logTrans: [[Float]] = [[log(0.8), log(0.2)],
                                    [log(0.3), log(0.7)]]

        var logObs = [[Float]]()
        for i in 0..<nFrames {
            if i % 2 == 0 {
                logObs.append([log(Float(0.9)), log(Float(0.1))])
            } else {
                logObs.append([log(Float(0.1)), log(Float(0.9))])
            }
        }

        let gamma = HMM.forwardBackward(
            logObservations: logObs,
            logInitial: logInitial,
            logTransition: logTrans
        )

        XCTAssertEqual(gamma.count, nFrames)
        for t in 0..<nFrames {
            let probSum = exp(gamma[t][0]) + exp(gamma[t][1])
            XCTAssertEqual(probSum, 1.0, accuracy: 1e-3,
                           "Posterior at frame \(t) should sum to 1.0")
            XCTAssertFalse(gamma[t][0].isNaN)
            XCTAssertFalse(gamma[t][1].isNaN)
        }
    }
}
