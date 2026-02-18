import XCTest
@testable import MetalMomCore

final class TempoEvalTests: XCTestCase {
    func testExactMatch() {
        XCTAssertTrue(TempoEval.evaluate(referenceTempo: 120.0, estimatedTempo: 120.0))
        XCTAssertEqual(TempoEval.pScore(referenceTempo: 120.0, estimatedTempo: 120.0), 1.0)
    }

    func testWithinTolerance() {
        // 8% of 120 = 9.6, so 128 should match
        XCTAssertTrue(TempoEval.evaluate(referenceTempo: 120.0, estimatedTempo: 128.0))
        // But 132 should not (10% off)
        XCTAssertFalse(TempoEval.evaluate(referenceTempo: 120.0, estimatedTempo: 132.0))
    }

    func testDoubleTempo() {
        // 240 = 2x 120 -> should match
        XCTAssertTrue(TempoEval.evaluate(referenceTempo: 120.0, estimatedTempo: 240.0))
    }

    func testHalfTempo() {
        // 60 = 0.5x 120 -> should match
        XCTAssertTrue(TempoEval.evaluate(referenceTempo: 120.0, estimatedTempo: 60.0))
    }

    func testTripleTempo() {
        // 360 = 3x 120 -> should match
        XCTAssertTrue(TempoEval.evaluate(referenceTempo: 120.0, estimatedTempo: 360.0))
    }

    func testThirdTempo() {
        // 40 = 1/3 x 120 -> should match
        XCTAssertTrue(TempoEval.evaluate(referenceTempo: 120.0, estimatedTempo: 40.0))
    }

    func testNoMatch() {
        // 90 BPM vs 120 BPM = 25% off at 1x, not close to 2x/0.5x/3x/1/3x
        XCTAssertFalse(TempoEval.evaluate(referenceTempo: 120.0, estimatedTempo: 90.0))
        XCTAssertEqual(TempoEval.pScore(referenceTempo: 120.0, estimatedTempo: 90.0), 0.0)
    }

    func testZeroReference() {
        XCTAssertFalse(TempoEval.evaluate(referenceTempo: 0, estimatedTempo: 120.0))
    }

    func testZeroEstimated() {
        XCTAssertFalse(TempoEval.evaluate(referenceTempo: 120.0, estimatedTempo: 0))
    }

    func testNegativeValues() {
        XCTAssertFalse(TempoEval.evaluate(referenceTempo: -120.0, estimatedTempo: 120.0))
        XCTAssertFalse(TempoEval.evaluate(referenceTempo: 120.0, estimatedTempo: -120.0))
    }

    func testCustomTolerance() {
        // 130 BPM vs 120 BPM = ~8.3% off
        // Default 8% tolerance: should miss
        XCTAssertFalse(TempoEval.evaluate(referenceTempo: 120.0, estimatedTempo: 130.0, tolerance: 0.08))
        // 10% tolerance: should match
        XCTAssertTrue(TempoEval.evaluate(referenceTempo: 120.0, estimatedTempo: 130.0, tolerance: 0.10))
    }

    func testPScoreValues() {
        XCTAssertEqual(TempoEval.pScore(referenceTempo: 100.0, estimatedTempo: 100.0), 1.0)
        XCTAssertEqual(TempoEval.pScore(referenceTempo: 100.0, estimatedTempo: 200.0), 1.0)
        XCTAssertEqual(TempoEval.pScore(referenceTempo: 100.0, estimatedTempo: 50.0), 1.0)
        XCTAssertEqual(TempoEval.pScore(referenceTempo: 100.0, estimatedTempo: 75.0), 0.0)
    }
}
