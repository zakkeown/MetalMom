import XCTest
@testable import MetalMomCore

final class ChordEvalTests: XCTestCase {
    // MARK: - Frame-level accuracy

    func testPerfectAccuracy() {
        let ref: [Int32] = [0, 1, 2, 3, 0, 1]
        let est: [Int32] = [0, 1, 2, 3, 0, 1]
        XCTAssertEqual(ChordEval.accuracy(reference: ref, estimated: est), 1.0, accuracy: 1e-5)
    }

    func testHalfCorrect() {
        let ref: [Int32] = [0, 1, 2, 3]
        let est: [Int32] = [0, 1, 5, 6]  // 2 out of 4 correct
        XCTAssertEqual(ChordEval.accuracy(reference: ref, estimated: est), 0.5, accuracy: 1e-5)
    }

    func testNoCorrect() {
        let ref: [Int32] = [0, 1, 2]
        let est: [Int32] = [3, 4, 5]
        XCTAssertEqual(ChordEval.accuracy(reference: ref, estimated: est), 0, accuracy: 1e-5)
    }

    func testEmptyReference() {
        XCTAssertEqual(ChordEval.accuracy(reference: [], estimated: [1, 2, 3]), 0, accuracy: 1e-5)
    }

    func testEmptyEstimated() {
        XCTAssertEqual(ChordEval.accuracy(reference: [1, 2, 3], estimated: []), 0, accuracy: 1e-5)
    }

    func testMismatchedLengths() {
        // Should use min of both lengths
        let ref: [Int32] = [0, 1, 2, 3, 4]
        let est: [Int32] = [0, 1, 2]  // 3 matching out of 3 compared
        XCTAssertEqual(ChordEval.accuracy(reference: ref, estimated: est), 1.0, accuracy: 1e-5)
    }

    func testSingleFrame() {
        XCTAssertEqual(ChordEval.accuracy(reference: [5], estimated: [5]), 1.0, accuracy: 1e-5)
        XCTAssertEqual(ChordEval.accuracy(reference: [5], estimated: [6]), 0, accuracy: 1e-5)
    }

    // MARK: - Weighted overlap

    func testWeightedOverlapPerfect() {
        let refIntervals: [[Float]] = [[0, 1], [1, 2], [2, 3]]
        let refLabels: [Int32] = [0, 1, 2]
        let estIntervals: [[Float]] = [[0, 1], [1, 2], [2, 3]]
        let estLabels: [Int32] = [0, 1, 2]

        let result = ChordEval.weightedOverlap(
            refIntervals: refIntervals, refLabels: refLabels,
            estIntervals: estIntervals, estLabels: estLabels
        )
        XCTAssertEqual(result, 1.0, accuracy: 1e-5)
    }

    func testWeightedOverlapNoMatch() {
        let refIntervals: [[Float]] = [[0, 1], [1, 2]]
        let refLabels: [Int32] = [0, 1]
        let estIntervals: [[Float]] = [[0, 1], [1, 2]]
        let estLabels: [Int32] = [2, 3]  // All wrong

        let result = ChordEval.weightedOverlap(
            refIntervals: refIntervals, refLabels: refLabels,
            estIntervals: estIntervals, estLabels: estLabels
        )
        XCTAssertEqual(result, 0, accuracy: 1e-5)
    }

    func testWeightedOverlapPartial() {
        // First 2 seconds correct, last 2 seconds wrong -> 50%
        let refIntervals: [[Float]] = [[0, 2], [2, 4]]
        let refLabels: [Int32] = [0, 1]
        let estIntervals: [[Float]] = [[0, 2], [2, 4]]
        let estLabels: [Int32] = [0, 2]

        let result = ChordEval.weightedOverlap(
            refIntervals: refIntervals, refLabels: refLabels,
            estIntervals: estIntervals, estLabels: estLabels
        )
        XCTAssertEqual(result, 0.5, accuracy: 1e-5)
    }

    func testWeightedOverlapDifferentBoundaries() {
        // Ref: [0-2]=A, [2-4]=B
        // Est: [0-3]=A, [3-4]=B
        // Overlap matching: [0-2] A matches [0-3] A -> 2s; [2-4] B matches [3-4] B -> 1s
        // Total ref duration: 4s. Correct: 3s. Accuracy: 0.75
        let refIntervals: [[Float]] = [[0, 2], [2, 4]]
        let refLabels: [Int32] = [0, 1]
        let estIntervals: [[Float]] = [[0, 3], [3, 4]]
        let estLabels: [Int32] = [0, 1]

        let result = ChordEval.weightedOverlap(
            refIntervals: refIntervals, refLabels: refLabels,
            estIntervals: estIntervals, estLabels: estLabels
        )
        XCTAssertEqual(result, 0.75, accuracy: 1e-5)
    }

    func testWeightedOverlapEmpty() {
        XCTAssertEqual(
            ChordEval.weightedOverlap(refIntervals: [], refLabels: [], estIntervals: [[0, 1]], estLabels: [0]),
            0, accuracy: 1e-5
        )
        XCTAssertEqual(
            ChordEval.weightedOverlap(refIntervals: [[0, 1]], refLabels: [0], estIntervals: [], estLabels: []),
            0, accuracy: 1e-5
        )
    }

    func testWeightedOverlapUnequalDurations() {
        // Ref segment is longer so only overlap counts
        let refIntervals: [[Float]] = [[0, 10]]
        let refLabels: [Int32] = [0]
        let estIntervals: [[Float]] = [[0, 5]]
        let estLabels: [Int32] = [0]

        let result = ChordEval.weightedOverlap(
            refIntervals: refIntervals, refLabels: refLabels,
            estIntervals: estIntervals, estLabels: estLabels
        )
        XCTAssertEqual(result, 0.5, accuracy: 1e-5)
    }
}
