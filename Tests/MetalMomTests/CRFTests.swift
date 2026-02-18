import XCTest
@testable import MetalMomCore

final class CRFTests: XCTestCase {

    // MARK: - Fixtures

    /// 3-label (A=0, B=1, C=2), 4-frame CRF with known best path.
    ///
    /// Unary scores strongly favor a specific label at each position:
    ///   t=0: A=5, B=1, C=0   → A is best
    ///   t=1: A=0, B=6, C=1   → B is best
    ///   t=2: A=1, B=0, C=7   → C is best
    ///   t=3: A=0, B=8, C=1   → B is best
    ///
    /// Pairwise scores: small positive values for all transitions,
    /// slightly favoring staying in same label (+0.5 self, +0.1 others).
    /// This makes A→B→C→B the optimal path since unary scores dominate.
    private static let unary3x4: [[Float]] = [
        [5.0, 1.0, 0.0],
        [0.0, 6.0, 1.0],
        [1.0, 0.0, 7.0],
        [0.0, 8.0, 1.0]
    ]

    private static let pairwise3x3: [[Float]] = [
        [0.5, 0.1, 0.1],   // from A
        [0.1, 0.5, 0.1],   // from B
        [0.1, 0.1, 0.5]    // from C
    ]

    /// Expected best path: A(0) → B(1) → C(2) → B(1)
    /// Score: unary[0][0] + pairwise[0][1] + unary[1][1] + pairwise[1][2] + unary[2][2] + pairwise[2][1] + unary[3][1]
    ///      = 5.0 + 0.1 + 6.0 + 0.1 + 7.0 + 0.1 + 8.0 = 26.3
    private static let expectedPath3x4 = [0, 1, 2, 1]
    private static let expectedScore3x4: Float = 26.3

    // MARK: - Viterbi Decoding Tests

    func testViterbiKnownPath() {
        let result = CRF.viterbiDecode(
            unaryScores: CRFTests.unary3x4,
            pairwiseScores: CRFTests.pairwise3x3
        )

        XCTAssertEqual(result.path, CRFTests.expectedPath3x4,
                       "Expected path [A, B, C, B] = [0, 1, 2, 1]")
        XCTAssertEqual(result.score, CRFTests.expectedScore3x4, accuracy: 1e-5,
                       "Expected total score 26.3")
    }

    func testViterbiSingleFrame() {
        // Single frame: picks the label with the highest unary score.
        let unary: [[Float]] = [[-1.0, 3.0, 2.0]]
        let pairwise: [[Float]] = [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ]

        let result = CRF.viterbiDecode(unaryScores: unary, pairwiseScores: pairwise)

        XCTAssertEqual(result.path, [1], "Label 1 has the highest unary score (3.0)")
        XCTAssertEqual(result.score, 3.0, accuracy: 1e-6)
    }

    func testViterbiSingleLabel() {
        // Single label: only one possible path.
        let unary: [[Float]] = [[2.0], [3.0], [1.0]]
        let pairwise: [[Float]] = [[0.5]]

        let result = CRF.viterbiDecode(unaryScores: unary, pairwiseScores: pairwise)

        XCTAssertEqual(result.path, [0, 0, 0])
        // Score: 2.0 + 0.5 + 3.0 + 0.5 + 1.0 = 7.0
        XCTAssertEqual(result.score, 7.0, accuracy: 1e-6)
    }

    func testViterbiEmptyInput() {
        let result = CRF.viterbiDecode(unaryScores: [], pairwiseScores: [[0.0]])
        XCTAssertTrue(result.path.isEmpty)
        XCTAssertEqual(result.score, -.infinity)
    }

    func testViterbiNegativeScores() {
        // CRF allows negative scores (unlike HMM log-probabilities which are <= 0
        // but have distribution constraints). Here all scores are negative.
        let unary: [[Float]] = [
            [-1.0, -5.0],
            [-3.0, -0.5],
            [-2.0, -4.0]
        ]
        let pairwise: [[Float]] = [
            [-0.1, -0.2],
            [-0.3, -0.1]
        ]

        let result = CRF.viterbiDecode(unaryScores: unary, pairwiseScores: pairwise)

        // t=0: label 0 (-1.0) vs label 1 (-5.0) → label 0
        // t=1: from 0→0: -1.0 + (-0.1) + (-3.0) = -4.1
        //       from 0→1: -1.0 + (-0.2) + (-0.5) = -1.7 → label 1 from 0
        // t=2: from 1→0: -1.7 + (-0.3) + (-2.0) = -4.0
        //       from 1→1: -1.7 + (-0.1) + (-4.0) = -5.8 → label 0 from 1
        XCTAssertEqual(result.path, [0, 1, 0])
        XCTAssertEqual(result.score, -4.0, accuracy: 1e-5)
    }

    func testViterbiPairwiseDominates() {
        // Pairwise scores dominate unary scores, forcing the path
        // to stay in label 0 despite label 1 having better unary scores.
        let unary: [[Float]] = [
            [1.0, 2.0],   // label 1 better by +1
            [1.0, 2.0],   // label 1 better by +1
            [1.0, 2.0]    // label 1 better by +1
        ]
        let pairwise: [[Float]] = [
            [10.0, -100.0],   // strongly penalize leaving label 0
            [-100.0, 10.0]    // strongly penalize leaving label 1
        ]

        let result = CRF.viterbiDecode(unaryScores: unary, pairwiseScores: pairwise)

        // Starting from label 1 (unary=2) and staying: 2 + 10 + 2 + 10 + 2 = 26
        // Starting from label 0 (unary=1) and staying: 1 + 10 + 1 + 10 + 1 = 23
        // Switching is heavily penalized (-100), so path stays in label 1.
        XCTAssertEqual(result.path, [1, 1, 1])
        XCTAssertEqual(result.score, 26.0, accuracy: 1e-5)
    }

    // MARK: - Log-Partition Function Tests

    func testLogPartitionSingleFrame() {
        // Single frame, 3 labels: logZ = logSumExp of unary scores
        let unary: [[Float]] = [[1.0, 2.0, 3.0]]
        let pairwise: [[Float]] = [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ]

        let logZ = CRF.logPartition(unaryScores: unary, pairwiseScores: pairwise)

        // logSumExp(1, 2, 3) = log(exp(1) + exp(2) + exp(3))
        let expected = log(exp(Float(1.0)) + exp(Float(2.0)) + exp(Float(3.0)))
        XCTAssertEqual(logZ, expected, accuracy: 1e-5)
    }

    func testLogPartitionSingleLabel() {
        // Single label: logZ = sum of all unary scores + sum of all pairwise self-transitions
        let unary: [[Float]] = [[2.0], [3.0], [1.0]]
        let pairwise: [[Float]] = [[0.5]]

        let logZ = CRF.logPartition(unaryScores: unary, pairwiseScores: pairwise)

        // Only one possible path: 2.0 + 0.5 + 3.0 + 0.5 + 1.0 = 7.0
        // logZ = log(exp(7.0)) = 7.0
        XCTAssertEqual(logZ, 7.0, accuracy: 1e-5)
    }

    func testLogPartitionTwoLabelsTwoFrames() {
        // 2 labels, 2 frames: enumerate all 4 paths.
        let unary: [[Float]] = [[1.0, 2.0], [3.0, 4.0]]
        let pairwise: [[Float]] = [
            [0.1, 0.2],
            [0.3, 0.4]
        ]

        let logZ = CRF.logPartition(unaryScores: unary, pairwiseScores: pairwise)

        // Path (0,0): 1.0 + 0.1 + 3.0 = 4.1
        // Path (0,1): 1.0 + 0.2 + 4.0 = 5.2
        // Path (1,0): 2.0 + 0.3 + 3.0 = 5.3
        // Path (1,1): 2.0 + 0.4 + 4.0 = 6.4
        let expected = log(exp(Float(4.1)) + exp(Float(5.2)) + exp(Float(5.3)) + exp(Float(6.4)))
        XCTAssertEqual(logZ, expected, accuracy: 1e-4)
    }

    func testLogPartitionEmpty() {
        let logZ = CRF.logPartition(unaryScores: [], pairwiseScores: [[0.0]])
        XCTAssertEqual(logZ, -.infinity)
    }

    func testLogPartitionGeqViterbiScore() {
        // The log-partition function must be >= the Viterbi score,
        // because Z sums over all paths (including the best one).
        let viterbiResult = CRF.viterbiDecode(
            unaryScores: CRFTests.unary3x4,
            pairwiseScores: CRFTests.pairwise3x3
        )
        let logZ = CRF.logPartition(
            unaryScores: CRFTests.unary3x4,
            pairwiseScores: CRFTests.pairwise3x3
        )

        XCTAssertGreaterThanOrEqual(logZ, viterbiResult.score,
                                     "logZ must be >= best path score")
    }

    // MARK: - Marginals Tests

    func testMarginalsSumToOnePerFrame() {
        // For each frame, exp(marginals) should sum to 1.0.
        let marg = CRF.marginals(
            unaryScores: CRFTests.unary3x4,
            pairwiseScores: CRFTests.pairwise3x3
        )

        XCTAssertEqual(marg.count, 4)
        for t in 0..<4 {
            let probSum = marg[t].reduce(Float(0.0)) { $0 + exp($1) }
            XCTAssertEqual(probSum, 1.0, accuracy: 1e-4,
                           "Marginals at frame \(t) should sum to 1.0, got \(probSum)")
        }
    }

    func testMarginalsSingleLabel() {
        // Single label: marginal is always log(1) = 0 for every frame.
        let unary: [[Float]] = [[5.0], [3.0], [7.0]]
        let pairwise: [[Float]] = [[1.0]]

        let marg = CRF.marginals(unaryScores: unary, pairwiseScores: pairwise)

        XCTAssertEqual(marg.count, 3)
        for t in 0..<3 {
            XCTAssertEqual(marg[t].count, 1)
            XCTAssertEqual(marg[t][0], 0.0, accuracy: 1e-5,
                           "Single-label marginal should be log(1)=0")
        }
    }

    func testMarginalsSingleFrame() {
        // Single frame: marginals are just softmax of unary scores.
        let unary: [[Float]] = [[1.0, 2.0, 3.0]]
        let pairwise: [[Float]] = [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ]

        let marg = CRF.marginals(unaryScores: unary, pairwiseScores: pairwise)

        XCTAssertEqual(marg.count, 1)
        // softmax([1,2,3])
        let denom = exp(Float(1.0)) + exp(Float(2.0)) + exp(Float(3.0))
        let expected = [log(exp(Float(1.0)) / denom),
                        log(exp(Float(2.0)) / denom),
                        log(exp(Float(3.0)) / denom)]

        for s in 0..<3 {
            XCTAssertEqual(marg[0][s], expected[s], accuracy: 1e-5)
        }
    }

    func testMarginalsEmpty() {
        let marg = CRF.marginals(unaryScores: [], pairwiseScores: [[0.0]])
        XCTAssertTrue(marg.isEmpty)
    }

    func testMarginalsTwoLabelsTwoFramesExact() {
        // 2 labels, 2 frames: verify marginals by enumerating all 4 paths.
        let unary: [[Float]] = [[1.0, 2.0], [3.0, 4.0]]
        let pairwise: [[Float]] = [
            [0.1, 0.2],
            [0.3, 0.4]
        ]

        let marg = CRF.marginals(unaryScores: unary, pairwiseScores: pairwise)

        // Path scores:
        // (0,0): 1.0 + 0.1 + 3.0 = 4.1
        // (0,1): 1.0 + 0.2 + 4.0 = 5.2
        // (1,0): 2.0 + 0.3 + 3.0 = 5.3
        // (1,1): 2.0 + 0.4 + 4.0 = 6.4
        let s00 = exp(Float(4.1))
        let s01 = exp(Float(5.2))
        let s10 = exp(Float(5.3))
        let s11 = exp(Float(6.4))
        let Z = s00 + s01 + s10 + s11

        // Marginals at t=0:
        // P(y0=0) = (s00 + s01) / Z
        // P(y0=1) = (s10 + s11) / Z
        let p0_0 = (s00 + s01) / Z
        let p0_1 = (s10 + s11) / Z
        XCTAssertEqual(exp(marg[0][0]), p0_0, accuracy: 1e-4)
        XCTAssertEqual(exp(marg[0][1]), p0_1, accuracy: 1e-4)

        // Marginals at t=1:
        // P(y1=0) = (s00 + s10) / Z
        // P(y1=1) = (s01 + s11) / Z
        let p1_0 = (s00 + s10) / Z
        let p1_1 = (s01 + s11) / Z
        XCTAssertEqual(exp(marg[1][0]), p1_0, accuracy: 1e-4)
        XCTAssertEqual(exp(marg[1][1]), p1_1, accuracy: 1e-4)
    }

    // MARK: - Consistency Tests

    func testLogPartitionConsistentWithMarginals() {
        // The marginals should be consistent with the log-partition function:
        // sum of exp(marginals) per frame should be 1.0, and the log-partition
        // should equal what we get from the forward pass.
        let logZ = CRF.logPartition(
            unaryScores: CRFTests.unary3x4,
            pairwiseScores: CRFTests.pairwise3x3
        )

        // Also compute logZ from the forward pass used by marginals
        let alpha = CRF.forwardPass(
            unaryScores: CRFTests.unary3x4,
            pairwiseScores: CRFTests.pairwise3x3
        )
        var logZFromAlpha: Float = -.infinity
        for s in 0..<3 {
            logZFromAlpha = logZFromAlpha < alpha[3][s]
                ? alpha[3][s] + log(1 + exp(logZFromAlpha - alpha[3][s]))
                : logZFromAlpha + log(1 + exp(alpha[3][s] - logZFromAlpha))
        }

        XCTAssertEqual(logZ, logZFromAlpha, accuracy: 1e-5,
                       "logPartition should be consistent with forward pass")
        XCTAssertFalse(logZ.isNaN)
        XCTAssertFalse(logZ.isInfinite)
    }

    func testViterbiPathScoreMatchesManualComputation() {
        // Verify that the Viterbi score matches a manual computation of the path score.
        let result = CRF.viterbiDecode(
            unaryScores: CRFTests.unary3x4,
            pairwiseScores: CRFTests.pairwise3x3
        )

        // Manually compute score for the returned path
        var manualScore: Float = CRFTests.unary3x4[0][result.path[0]]
        for t in 1..<4 {
            manualScore += CRFTests.pairwise3x3[result.path[t - 1]][result.path[t]]
            manualScore += CRFTests.unary3x4[t][result.path[t]]
        }

        XCTAssertEqual(result.score, manualScore, accuracy: 1e-5,
                       "Viterbi score should match manual path score computation")
    }

    // MARK: - Numerical Stability Tests

    func testViterbiLargeScores() {
        // Large positive scores should not cause overflow.
        let nFrames = 50
        let nLabels = 5
        var unary = [[Float]]()
        for t in 0..<nFrames {
            var row = [Float](repeating: 0.0, count: nLabels)
            row[t % nLabels] = 100.0  // strongly favor one label per frame
            unary.append(row)
        }
        let pairwise = [[Float]](repeating: [Float](repeating: 1.0, count: nLabels),
                                  count: nLabels)

        let result = CRF.viterbiDecode(unaryScores: unary, pairwiseScores: pairwise)

        XCTAssertEqual(result.path.count, nFrames)
        XCTAssertFalse(result.score.isNaN)
        XCTAssertFalse(result.score.isInfinite)

        // Each frame should pick the strongly-favored label
        for t in 0..<nFrames {
            XCTAssertEqual(result.path[t], t % nLabels)
        }
    }

    func testLogPartitionLargeScores() {
        // logZ should remain finite even with large input scores.
        let unary: [[Float]] = [[100.0, 200.0], [150.0, 250.0]]
        let pairwise: [[Float]] = [[10.0, 20.0], [15.0, 25.0]]

        let logZ = CRF.logPartition(unaryScores: unary, pairwiseScores: pairwise)

        XCTAssertFalse(logZ.isNaN, "logZ should not be NaN with large scores")
        XCTAssertFalse(logZ.isInfinite, "logZ should not be infinite with large scores")
    }

    func testMarginalsLongSequence() {
        // Marginals should remain valid on longer sequences.
        let nFrames = 100
        let nLabels = 4
        var unary = [[Float]]()
        for _ in 0..<nFrames {
            var row = [Float]()
            for s in 0..<nLabels {
                row.append(Float(s) * 0.5 - 1.0)
            }
            unary.append(row)
        }
        let pairwise = [[Float]](repeating: [Float](repeating: 0.1, count: nLabels),
                                  count: nLabels)

        let marg = CRF.marginals(unaryScores: unary, pairwiseScores: pairwise)

        XCTAssertEqual(marg.count, nFrames)
        for t in 0..<nFrames {
            let probSum = marg[t].reduce(Float(0.0)) { $0 + exp($1) }
            XCTAssertEqual(probSum, 1.0, accuracy: 1e-3,
                           "Marginals at frame \(t) should sum to 1.0")
            for s in 0..<nLabels {
                XCTAssertFalse(marg[t][s].isNaN, "marginal[\(t)][\(s)] is NaN")
            }
        }
    }

    // MARK: - Edge Case: Uniform Scores

    func testViterbiUniformScores() {
        // All scores are equal: any path is optimal, so just check score is correct.
        let unary: [[Float]] = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]
        let pairwise: [[Float]] = [[0.5, 0.5], [0.5, 0.5]]

        let result = CRF.viterbiDecode(unaryScores: unary, pairwiseScores: pairwise)

        XCTAssertEqual(result.path.count, 3)
        // Score = 1.0 + 0.5 + 1.0 + 0.5 + 1.0 = 4.0 regardless of path
        XCTAssertEqual(result.score, 4.0, accuracy: 1e-5)
    }

    func testMarginalsUniformScores() {
        // All scores equal: marginals should be uniform (1/nLabels per label).
        let unary: [[Float]] = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        let pairwise: [[Float]] = [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ]

        let marg = CRF.marginals(unaryScores: unary, pairwiseScores: pairwise)

        XCTAssertEqual(marg.count, 2)
        let expectedLogProb = log(Float(1.0) / Float(3.0))
        for t in 0..<2 {
            for s in 0..<3 {
                XCTAssertEqual(marg[t][s], expectedLogProb, accuracy: 1e-4,
                               "Uniform scores should give uniform marginals")
            }
        }
    }
}
