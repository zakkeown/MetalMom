import XCTest
@testable import MetalMomCore

final class BeatEvalTests: XCTestCase {
    func testPerfectMatch() {
        let ref: [Float] = [1.0, 2.0, 3.0, 4.0]
        let est: [Float] = [1.0, 2.0, 3.0, 4.0]
        let result = BeatEval.evaluate(reference: ref, estimated: est)
        XCTAssertEqual(result.fMeasure, 1.0, accuracy: 1e-5)
        XCTAssertEqual(result.cemgil, 1.0, accuracy: 1e-5)
        XCTAssertGreaterThan(result.pScore, 0.99)
        XCTAssertEqual(result.cmlC, 1.0, accuracy: 1e-5)
        XCTAssertEqual(result.amlC, 1.0, accuracy: 1e-5)
    }

    func testWithinTolerance() {
        let ref: [Float] = [1.0, 2.0, 3.0, 4.0]
        let est: [Float] = [1.01, 2.02, 3.01, 4.02]  // within 70ms
        let result = BeatEval.evaluate(reference: ref, estimated: est)
        XCTAssertEqual(result.fMeasure, 1.0, accuracy: 1e-5)
        XCTAssertGreaterThan(result.cemgil, 0.9)
    }

    func testNoMatch() {
        let ref: [Float] = [1.0, 2.0, 3.0]
        let est: [Float] = [10.0, 11.0, 12.0]
        let result = BeatEval.evaluate(reference: ref, estimated: est)
        XCTAssertEqual(result.fMeasure, 0, accuracy: 1e-5)
        XCTAssertLessThan(result.cemgil, 0.01)
        XCTAssertLessThan(result.pScore, 0.01)
    }

    func testEmptyReference() {
        let result = BeatEval.evaluate(reference: [], estimated: [1.0, 2.0])
        XCTAssertEqual(result.fMeasure, 0, accuracy: 1e-5)
        XCTAssertEqual(result.cemgil, 0, accuracy: 1e-5)
        XCTAssertEqual(result.pScore, 0, accuracy: 1e-5)
    }

    func testEmptyEstimated() {
        let result = BeatEval.evaluate(reference: [1.0, 2.0], estimated: [])
        XCTAssertEqual(result.fMeasure, 0, accuracy: 1e-5)
        XCTAssertEqual(result.cemgil, 0, accuracy: 1e-5)
        XCTAssertEqual(result.pScore, 0, accuracy: 1e-5)
    }

    func testBothEmpty() {
        let result = BeatEval.evaluate(reference: [], estimated: [])
        XCTAssertEqual(result.fMeasure, 0, accuracy: 1e-5)
        XCTAssertEqual(result.cemgil, 0, accuracy: 1e-5)
    }

    func testPartialMatch() {
        let ref: [Float] = [1.0, 2.0, 3.0, 4.0]
        let est: [Float] = [1.0, 2.0]  // missed half
        let result = BeatEval.evaluate(reference: ref, estimated: est)
        XCTAssertGreaterThan(result.fMeasure, 0)
        XCTAssertLessThan(result.fMeasure, 1.0)
    }

    func testCemgilPerfect() {
        // With perfect match, Cemgil should be 1.0
        let ref: [Float] = [1.0, 2.0, 3.0]
        let est: [Float] = [1.0, 2.0, 3.0]
        let result = BeatEval.evaluate(reference: ref, estimated: est)
        XCTAssertEqual(result.cemgil, 1.0, accuracy: 1e-5)
    }

    func testCemgilSlightOffset() {
        // Slight offset reduces Cemgil score
        let ref: [Float] = [1.0, 2.0, 3.0]
        let est: [Float] = [1.02, 2.02, 3.02]
        let result = BeatEval.evaluate(reference: ref, estimated: est)
        XCTAssertGreaterThan(result.cemgil, 0.8)
        XCTAssertLessThan(result.cemgil, 1.0)
    }

    func testCMLSingleBeat() {
        // Fewer than 2 beats: CML should be 0
        let ref: [Float] = [1.0]
        let est: [Float] = [1.0]
        let result = BeatEval.evaluate(reference: ref, estimated: est)
        XCTAssertEqual(result.cmlC, 0, accuracy: 1e-5)
    }

    func testCustomFMeasureWindow() {
        let ref: [Float] = [1.0, 2.0, 3.0]
        let est: [Float] = [1.1, 2.1, 3.1]  // 100ms offset

        // Default 70ms window should miss
        let result1 = BeatEval.evaluate(reference: ref, estimated: est, fMeasureWindow: 0.07)
        XCTAssertEqual(result1.fMeasure, 0, accuracy: 1e-5)

        // Wider 150ms window should match
        let result2 = BeatEval.evaluate(reference: ref, estimated: est, fMeasureWindow: 0.15)
        XCTAssertEqual(result2.fMeasure, 1.0, accuracy: 1e-5)
    }

    func testAMLBetterThanCML() {
        // AML should always be >= CML since it tries additional metrical levels
        let ref: [Float] = [1.0, 2.0, 3.0, 4.0]
        let est: [Float] = [1.0, 3.0]  // Half tempo
        let result = BeatEval.evaluate(reference: ref, estimated: est)
        XCTAssertGreaterThanOrEqual(result.amlC, result.cmlC)
    }
}
