import XCTest
@testable import MetalMomCore

final class OnsetEvalTests: XCTestCase {
    func testPerfectMatch() {
        let ref: [Float] = [1.0, 2.0, 3.0]
        let est: [Float] = [1.0, 2.0, 3.0]
        let result = OnsetEval.evaluate(reference: ref, estimated: est)
        XCTAssertEqual(result.precision, 1.0, accuracy: 1e-6)
        XCTAssertEqual(result.recall, 1.0, accuracy: 1e-6)
        XCTAssertEqual(result.fMeasure, 1.0, accuracy: 1e-6)
    }

    func testWithinTolerance() {
        let ref: [Float] = [1.0, 2.0, 3.0]
        let est: [Float] = [1.02, 2.03, 2.98]  // Within 50ms
        let result = OnsetEval.evaluate(reference: ref, estimated: est)
        XCTAssertEqual(result.precision, 1.0, accuracy: 1e-6)
        XCTAssertEqual(result.recall, 1.0, accuracy: 1e-6)
        XCTAssertEqual(result.fMeasure, 1.0, accuracy: 1e-6)
    }

    func testMissedOnsets() {
        let ref: [Float] = [1.0, 2.0, 3.0]
        let est: [Float] = [1.0]  // Only detected first onset
        let result = OnsetEval.evaluate(reference: ref, estimated: est)
        XCTAssertEqual(result.precision, 1.0, accuracy: 1e-6)  // 1/1 correct
        XCTAssertEqual(result.recall, 1.0/3.0, accuracy: 1e-6)  // 1/3 found
    }

    func testFalsePositives() {
        let ref: [Float] = [1.0]
        let est: [Float] = [1.0, 2.0, 3.0]  // Extra detections
        let result = OnsetEval.evaluate(reference: ref, estimated: est)
        XCTAssertEqual(result.precision, 1.0/3.0, accuracy: 1e-6)  // 1/3 correct
        XCTAssertEqual(result.recall, 1.0, accuracy: 1e-6)  // 1/1 found
    }

    func testNoMatch() {
        let ref: [Float] = [1.0, 2.0, 3.0]
        let est: [Float] = [5.0, 6.0, 7.0]  // All too far
        let result = OnsetEval.evaluate(reference: ref, estimated: est)
        XCTAssertEqual(result.precision, 0, accuracy: 1e-6)
        XCTAssertEqual(result.recall, 0, accuracy: 1e-6)
        XCTAssertEqual(result.fMeasure, 0, accuracy: 1e-6)
    }

    func testEmptyReference() {
        let ref: [Float] = []
        let est: [Float] = [1.0, 2.0]
        let result = OnsetEval.evaluate(reference: ref, estimated: est)
        XCTAssertEqual(result.fMeasure, 0, accuracy: 1e-6)
    }

    func testEmptyEstimated() {
        let ref: [Float] = [1.0, 2.0]
        let est: [Float] = []
        let result = OnsetEval.evaluate(reference: ref, estimated: est)
        XCTAssertEqual(result.precision, 0, accuracy: 1e-6)
        XCTAssertEqual(result.recall, 0, accuracy: 1e-6)
    }

    func testCustomWindow() {
        let ref: [Float] = [1.0]
        let est: [Float] = [1.08]  // 80ms away
        // Default 50ms window: miss
        let result1 = OnsetEval.evaluate(reference: ref, estimated: est, window: 0.05)
        XCTAssertEqual(result1.fMeasure, 0, accuracy: 1e-6)
        // 100ms window: match
        let result2 = OnsetEval.evaluate(reference: ref, estimated: est, window: 0.1)
        XCTAssertEqual(result2.fMeasure, 1.0, accuracy: 1e-6)
    }
}
