import XCTest
@testable import MetalMomCore

final class RMSTests: XCTestCase {

    // MARK: - Helpers

    private func makeSineSignal(frequency: Float = 440.0, amplitude: Float = 1.0,
                                 sr: Int = 22050, duration: Float = 1.0) -> Signal {
        let count = Int(Float(sr) * duration)
        let data = (0..<count).map { i in
            amplitude * sinf(2.0 * .pi * frequency * Float(i) / Float(sr))
        }
        return Signal(data: data, sampleRate: sr)
    }

    // MARK: - Shape Tests

    func testRMSShape() {
        let signal = makeSineSignal()
        let result = RMS.compute(signal: signal, frameLength: 2048, hopLength: 512, center: true)

        // librosa.feature.rms returns shape [1, nFrames]
        XCTAssertEqual(result.shape.count, 2, "RMS should be 2D [1, nFrames]")
        XCTAssertEqual(result.shape[0], 1, "RMS first dimension should be 1")

        // Expected nFrames: with center=true, pad = frameLength/2 on each side
        // paddedLength = 22050 + 2048 = 24098
        // nFrames = 1 + (24098 - 2048) / 512 = 1 + 22050 / 512 = 1 + 43 = 44
        XCTAssertEqual(result.shape[1], 44, "Expected 44 frames for 1s signal at default params")
    }

    func testRMSShapeNoCentering() {
        let signal = makeSineSignal()
        let result = RMS.compute(signal: signal, frameLength: 2048, hopLength: 512, center: false)

        XCTAssertEqual(result.shape.count, 2, "RMS should be 2D")
        XCTAssertEqual(result.shape[0], 1)

        // Without centering: nFrames = 1 + (22050 - 2048) / 512 = 1 + 39 = 40
        XCTAssertEqual(result.shape[1], 40, "Expected 40 frames without center padding")
    }

    // MARK: - Silence Tests

    func testRMSOfSilenceIsZero() {
        let silence = Signal(data: [Float](repeating: 0, count: 22050), sampleRate: 22050)
        let result = RMS.compute(signal: silence, frameLength: 2048, hopLength: 512)

        for i in 0..<result.count {
            XCTAssertEqual(result[i], 0.0, accuracy: 1e-10,
                          "RMS of silence should be 0")
        }
    }

    // MARK: - Constant Signal Tests

    func testRMSOfConstantSignal() {
        // RMS of a constant signal c = |c|
        let c: Float = 0.5
        let constant = Signal(data: [Float](repeating: c, count: 22050), sampleRate: 22050)
        let result = RMS.compute(signal: constant, frameLength: 2048, hopLength: 512, center: false)

        // Interior frames should be exactly c
        let nFrames = result.shape[1]
        for f in 0..<nFrames {
            XCTAssertEqual(result[f], c, accuracy: 1e-5,
                          "RMS of constant \(c) should be \(c), got \(result[f])")
        }
    }

    // MARK: - Sine Wave Tests

    func testRMSSineWave() {
        // RMS of a sine wave with amplitude A = A / sqrt(2)
        let amplitude: Float = 1.0
        let signal = makeSineSignal(amplitude: amplitude)
        let expected = amplitude / sqrtf(2.0)
        let result = RMS.compute(signal: signal, frameLength: 2048, hopLength: 512, center: false)

        // Average over interior frames (skip edge effects)
        let nFrames = result.shape[1]
        let start = 2
        let end = nFrames - 2
        var sum: Float = 0
        for f in start..<end {
            sum += result[f]
        }
        let avgRMS = sum / Float(end - start)

        XCTAssertEqual(avgRMS, expected, accuracy: 0.02,
                       "RMS of sine wave should be A/sqrt(2) = \(expected), got \(avgRMS)")
    }

    // MARK: - Non-negativity

    func testRMSNonNegative() {
        let signal = makeSineSignal()
        let result = RMS.compute(signal: signal, frameLength: 2048, hopLength: 512)

        for i in 0..<result.count {
            XCTAssertGreaterThanOrEqual(result[i], 0.0,
                                        "RMS should be non-negative")
        }
    }

    // MARK: - Values Finite

    func testRMSValuesFinite() {
        let signal = makeSineSignal()
        let result = RMS.compute(signal: signal, frameLength: 2048, hopLength: 512)

        for i in 0..<result.count {
            XCTAssertFalse(result[i].isNaN, "RMS should not be NaN at index \(i)")
            XCTAssertFalse(result[i].isInfinite, "RMS should not be infinite at index \(i)")
        }
    }
}
