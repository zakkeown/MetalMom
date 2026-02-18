import XCTest
@testable import MetalMomCore

final class OnsetDetectionTests: XCTestCase {
    func testOnsetStrengthShape() {
        let n = 22050
        let signal = Signal(
            data: (0..<n).map { sin(Float($0) * 440.0 * 2.0 * .pi / 22050.0) },
            sampleRate: 22050
        )
        let result = OnsetDetection.onsetStrength(signal: signal)
        XCTAssertEqual(result.shape[0], 1)  // aggregated
        XCTAssertGreaterThan(result.shape[1], 0)
    }

    func testOnsetStrengthNonAggregated() {
        let n = 22050
        let signal = Signal(
            data: (0..<n).map { sin(Float($0) * 440.0 * 2.0 * .pi / 22050.0) },
            sampleRate: 22050
        )
        let result = OnsetDetection.onsetStrength(signal: signal, aggregate: false)
        XCTAssertEqual(result.shape[0], 128)  // n_mels
    }

    func testOnsetStrengthNonNegative() {
        let n = 22050
        let signal = Signal(
            data: (0..<n).map { sin(Float($0) * 440.0 * 2.0 * .pi / 22050.0) },
            sampleRate: 22050
        )
        let result = OnsetDetection.onsetStrength(signal: signal)
        for i in 0..<result.count {
            XCTAssertGreaterThanOrEqual(result[i], 0, "Onset strength should be non-negative")
        }
    }

    func testOnsetStrengthSilent() {
        let n = 22050
        let signal = Signal(data: [Float](repeating: 0, count: n), sampleRate: 22050)
        let result = OnsetDetection.onsetStrength(signal: signal)
        // Silent signal should have all-zero onset strength
        for i in 0..<result.count {
            XCTAssertEqual(result[i], 0, accuracy: 1e-6)
        }
    }

    func testOnsetStrengthFinite() {
        let n = 22050
        let signal = Signal(
            data: (0..<n).map { sin(Float($0) * 440.0 * 2.0 * .pi / 22050.0) },
            sampleRate: 22050
        )
        let result = OnsetDetection.onsetStrength(signal: signal)
        for i in 0..<result.count {
            XCTAssertFalse(result[i].isNaN)
            XCTAssertFalse(result[i].isInfinite)
        }
    }
}
