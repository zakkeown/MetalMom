import XCTest
@testable import MetalMomCore

final class ZeroCrossingTests: XCTestCase {

    // MARK: - Helpers

    private func makeSineSignal(frequency: Float = 440.0, sr: Int = 22050,
                                 duration: Float = 1.0) -> Signal {
        let count = Int(Float(sr) * duration)
        let data = (0..<count).map { i in
            sinf(2.0 * .pi * frequency * Float(i) / Float(sr))
        }
        return Signal(data: data, sampleRate: sr)
    }

    // MARK: - Shape Tests

    func testZCRShape() {
        let signal = makeSineSignal()
        let result = ZeroCrossing.rate(signal: signal, frameLength: 2048, hopLength: 512, center: true)

        // librosa.feature.zero_crossing_rate returns shape [1, nFrames]
        XCTAssertEqual(result.shape.count, 2, "ZCR should be 2D [1, nFrames]")
        XCTAssertEqual(result.shape[0], 1, "ZCR first dimension should be 1")

        // Same frame count as RMS: 44 frames
        XCTAssertEqual(result.shape[1], 44, "Expected 44 frames for 1s signal at default params")
    }

    func testZCRShapeNoCentering() {
        let signal = makeSineSignal()
        let result = ZeroCrossing.rate(signal: signal, frameLength: 2048, hopLength: 512, center: false)

        XCTAssertEqual(result.shape.count, 2, "ZCR should be 2D")
        XCTAssertEqual(result.shape[0], 1)
        XCTAssertEqual(result.shape[1], 40, "Expected 40 frames without center padding")
    }

    // MARK: - Constant Signal Tests

    func testZCROfConstantIsZero() {
        let constant = Signal(data: [Float](repeating: 1.0, count: 22050), sampleRate: 22050)
        let result = ZeroCrossing.rate(signal: constant, frameLength: 2048, hopLength: 512, center: false)

        for i in 0..<result.count {
            XCTAssertEqual(result[i], 0.0, accuracy: 1e-10,
                          "ZCR of constant signal should be 0")
        }
    }

    // MARK: - Sine Wave Tests

    func testZCRProportionalToFrequency() {
        // Higher frequency sine should have higher ZCR
        let sig440 = makeSineSignal(frequency: 440.0)
        let sig1k = makeSineSignal(frequency: 1000.0)

        let zcr440 = ZeroCrossing.rate(signal: sig440, frameLength: 2048, hopLength: 512, center: false)
        let zcr1k = ZeroCrossing.rate(signal: sig1k, frameLength: 2048, hopLength: 512, center: false)

        // Average over interior frames
        let nFrames = zcr440.shape[1]
        let start = 2
        let end = nFrames - 2
        var sum440: Float = 0
        var sum1k: Float = 0
        for f in start..<end {
            sum440 += zcr440[f]
            sum1k += zcr1k[f]
        }
        let avg440 = sum440 / Float(end - start)
        let avg1k = sum1k / Float(end - start)

        XCTAssertGreaterThan(avg1k, avg440,
                             "1000 Hz ZCR should be higher than 440 Hz ZCR")

        // ZCR should be roughly 2*freq/sr (two crossings per cycle)
        // 440 Hz at 22050 sr: 2 * 440 / 22050 ~ 0.0399
        let expectedZCR440 = 2.0 * 440.0 / 22050.0
        XCTAssertEqual(avg440, Float(expectedZCR440), accuracy: 0.005,
                       "ZCR of 440 Hz should be ~\(expectedZCR440), got \(avg440)")
    }

    // MARK: - Range Tests

    func testZCRRange() {
        // ZCR should be in [0, 1]
        let signal = makeSineSignal()
        let result = ZeroCrossing.rate(signal: signal, frameLength: 2048, hopLength: 512)

        for i in 0..<result.count {
            XCTAssertGreaterThanOrEqual(result[i], 0.0, "ZCR should be >= 0")
            XCTAssertLessThanOrEqual(result[i], 1.0 + 1e-6, "ZCR should be <= 1")
        }
    }

    // MARK: - Values Finite

    func testZCRValuesFinite() {
        let signal = makeSineSignal()
        let result = ZeroCrossing.rate(signal: signal, frameLength: 2048, hopLength: 512)

        for i in 0..<result.count {
            XCTAssertFalse(result[i].isNaN, "ZCR should not be NaN at index \(i)")
            XCTAssertFalse(result[i].isInfinite, "ZCR should not be infinite at index \(i)")
        }
    }

    // MARK: - Silence Tests

    func testZCROfSilenceIsZero() {
        let silence = Signal(data: [Float](repeating: 0, count: 22050), sampleRate: 22050)
        let result = ZeroCrossing.rate(signal: silence, frameLength: 2048, hopLength: 512)

        for i in 0..<result.count {
            XCTAssertEqual(result[i], 0.0, accuracy: 1e-10,
                          "ZCR of silence should be 0")
        }
    }
}
