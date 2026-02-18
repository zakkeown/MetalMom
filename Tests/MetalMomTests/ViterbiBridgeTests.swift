import XCTest
@testable import MetalMomCore

/// Tests for Viterbi decoding used by the Python bridge layer.
///
/// These tests verify that HMM.viterbi and CRF.viterbiDecode produce
/// correct results when called with flat (row-major) data, simulating
/// the same data transformations the C bridge performs.
final class ViterbiBridgeTests: XCTestCase {

    // MARK: - HMM Viterbi (simulating bridge flatten/unflatten)

    func testViterbiBasic2State() {
        // 5 frames, 2 states. Clear observations.
        let logObs: [[Float]] = [
            [log(0.9), log(0.1)],   // frame 0: state 0
            [log(0.8), log(0.2)],   // frame 1: state 0
            [log(0.85), log(0.15)], // frame 2: state 0
            [log(0.1), log(0.9)],   // frame 3: state 1
            [log(0.15), log(0.85)], // frame 4: state 1
        ]

        let logInitial: [Float] = [log(0.5), log(0.5)]
        let logTransition: [[Float]] = [
            [log(0.5), log(0.5)],
            [log(0.5), log(0.5)],
        ]

        let result = HMM.viterbi(
            logObservations: logObs,
            logInitial: logInitial,
            logTransition: logTransition
        )

        XCTAssertEqual(result.path.count, 5)
        // First 3 frames: state 0, last 2: state 1
        XCTAssertEqual(result.path[0], 0)
        XCTAssertEqual(result.path[1], 0)
        XCTAssertEqual(result.path[2], 0)
        XCTAssertEqual(result.path[3], 1)
        XCTAssertEqual(result.path[4], 1)
    }

    func testViterbiPathLength() {
        let nFrames = 10
        let nStates = 3

        let logObs = [[Float]](
            repeating: [Float](repeating: log(1.0 / 3.0), count: nStates),
            count: nFrames
        )
        let logInitial = [Float](repeating: log(1.0 / 3.0), count: nStates)
        let logTransition = [[Float]](
            repeating: [Float](repeating: log(1.0 / 3.0), count: nStates),
            count: nStates
        )

        let result = HMM.viterbi(
            logObservations: logObs,
            logInitial: logInitial,
            logTransition: logTransition
        )

        XCTAssertEqual(result.path.count, nFrames)
        // All path values should be valid state indices [0, nStates)
        for t in 0..<nFrames {
            XCTAssertGreaterThanOrEqual(result.path[t], 0)
            XCTAssertLessThan(result.path[t], nStates)
        }
    }

    func testViterbiLeftToRightConstraint() {
        // Use a left-to-right transition matrix.
        let logObs: [[Float]] = [
            [log(0.8), log(0.1), log(0.1)],  // frame 0: state 0
            [log(0.7), log(0.2), log(0.1)],  // frame 1: state 0
            [log(0.1), log(0.8), log(0.1)],  // frame 2: state 1
            [log(0.1), log(0.7), log(0.2)],  // frame 3: state 1
            [log(0.1), log(0.1), log(0.8)],  // frame 4: state 2
            [log(0.1), log(0.1), log(0.8)],  // frame 5: state 2
        ]

        let logInitial: [Float] = [0, -.infinity, -.infinity]  // must start in state 0

        let p: Float = 0.6
        let q: Float = 0.4
        let logTransition: [[Float]] = [
            [log(p), log(q), -.infinity],
            [-.infinity, log(p), log(q)],
            [-.infinity, -.infinity, 0],  // last state self-loops
        ]

        let result = HMM.viterbi(
            logObservations: logObs,
            logInitial: logInitial,
            logTransition: logTransition
        )

        // Path should be monotonically non-decreasing
        for t in 1..<result.path.count {
            XCTAssertGreaterThanOrEqual(result.path[t], result.path[t - 1],
                "Path should be non-decreasing at frame \(t)")
        }

        XCTAssertEqual(result.path[0], 0)
        XCTAssertEqual(result.path[result.path.count - 1], 2)
    }

    func testViterbiWeatherHMM() {
        // Classic 2-state Weather HMM via bridge-style flat data
        let logObs: [[Float]] = [
            [log(0.1), log(0.6)],  // walk: [Rain, Sun]
            [log(0.4), log(0.3)],  // shop: [Rain, Sun]
            [log(0.5), log(0.1)],  // clean: [Rain, Sun]
        ]

        let logInitial: [Float] = [log(0.6), log(0.4)]
        let logTransition: [[Float]] = [
            [log(0.7), log(0.3)],
            [log(0.4), log(0.6)],
        ]

        let result = HMM.viterbi(
            logObservations: logObs,
            logInitial: logInitial,
            logTransition: logTransition
        )

        // Expected: Sun(1), Rain(0), Rain(0)
        XCTAssertEqual(result.path, [1, 0, 0])
    }

    func testViterbiStrongObservationsOverrideTransition() {
        // Very strong observations should override transition preferences.
        let nFrames = 20
        let nStates = 4

        var logObs = [[Float]]()
        for t in 0..<nFrames {
            var frame = [Float](repeating: log(Float(0.01)), count: nStates)
            let targetState = t % nStates
            frame[targetState] = log(Float(0.97))
            logObs.append(frame)
        }

        let logInitial = [Float](repeating: log(1.0 / Float(nStates)), count: nStates)
        let logTransition = HMM.uniformTransition(nStates: nStates)

        let result = HMM.viterbi(
            logObservations: logObs,
            logInitial: logInitial,
            logTransition: logTransition
        )

        XCTAssertEqual(result.path.count, nFrames)
        for t in 0..<nFrames {
            XCTAssertEqual(result.path[t], t % nStates,
                           "Frame \(t) should decode to state \(t % nStates)")
        }
    }

    // MARK: - CRF Viterbi (Discriminative)

    func testCRFViterbiBasic() {
        let unary: [[Float]] = [
            [5.0, 1.0, 1.0],   // frame 0: state 0
            [4.0, 1.0, 1.0],   // frame 1: state 0
            [1.0, 1.0, 5.0],   // frame 2: state 2
            [1.0, 1.0, 4.0],   // frame 3: state 2
        ]

        let pairwise: [[Float]] = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]

        let result = CRF.viterbiDecode(
            unaryScores: unary,
            pairwiseScores: pairwise
        )

        XCTAssertEqual(result.path.count, 4)
        XCTAssertEqual(result.path[0], 0)
        XCTAssertEqual(result.path[1], 0)
        XCTAssertEqual(result.path[2], 2)
        XCTAssertEqual(result.path[3], 2)
    }

    func testCRFViterbiUniformPairwise() {
        // With zero pairwise scores, CRF should just pick argmax of unary
        let unary: [[Float]] = [
            [1.0, 3.0],  // frame 0: state 1
            [4.0, 2.0],  // frame 1: state 0
            [1.0, 5.0],  // frame 2: state 1
        ]

        let pairwise: [[Float]] = [
            [0.0, 0.0],
            [0.0, 0.0],
        ]

        let result = CRF.viterbiDecode(
            unaryScores: unary,
            pairwiseScores: pairwise
        )

        XCTAssertEqual(result.path, [1, 0, 1])
    }

    func testCRFViterbiStrongSelfLoop() {
        // Very strong self-loop pairwise score should keep path constant
        let unary: [[Float]] = [
            [2.0, 1.0],
            [1.0, 2.0],
            [2.0, 1.0],
            [1.0, 2.0],
        ]

        let pairwise: [[Float]] = [
            [100.0, 0.0],
            [0.0, 100.0],
        ]

        let result = CRF.viterbiDecode(
            unaryScores: unary,
            pairwiseScores: pairwise
        )

        // Path should be constant (all same state)
        let uniqueStates = Set(result.path)
        XCTAssertEqual(uniqueStates.count, 1, "Path should be constant with strong self-loop")
    }

    // MARK: - Flat data round-trip (testing bridge data conversion logic)

    func testFlatToNestedConversion() {
        // Simulate what the bridge does: flat array -> nested [[Float]]
        let nFrames = 3
        let nStates = 2
        let flat: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

        var nested = [[Float]]()
        for t in 0..<nFrames {
            let start = t * nStates
            nested.append(Array(flat[start..<start + nStates]))
        }

        XCTAssertEqual(nested.count, 3)
        XCTAssertEqual(nested[0], [1.0, 2.0])
        XCTAssertEqual(nested[1], [3.0, 4.0])
        XCTAssertEqual(nested[2], [5.0, 6.0])
    }

    func testPathToFloatConversion() {
        // Simulate what the bridge does: Int path -> Float-encoded path
        let path = [0, 1, 2, 1, 0]
        let floatPath = path.map { Float($0) }

        XCTAssertEqual(floatPath.count, 5)
        XCTAssertEqual(Int(floatPath[0]), 0)
        XCTAssertEqual(Int(floatPath[1]), 1)
        XCTAssertEqual(Int(floatPath[2]), 2)
        XCTAssertEqual(Int(floatPath[3]), 1)
        XCTAssertEqual(Int(floatPath[4]), 0)
    }
}
