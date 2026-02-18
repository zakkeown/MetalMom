import XCTest
@testable import MetalMomCore

final class YINTests: XCTestCase {

    // MARK: - Helpers

    private func makeSineSignal(frequency: Float = 440.0, amplitude: Float = 1.0,
                                 sr: Int = 22050, duration: Float = 1.0) -> Signal {
        return SignalGen.tone(frequency: frequency, sr: sr,
                              duration: Double(duration), phi: 0)
    }

    // MARK: - Pure Sine Wave (440 Hz)

    func testYINSineWave440() {
        let sr = 22050
        let signal = makeSineSignal(frequency: 440.0, sr: sr, duration: 1.0)

        let result = YIN.yin(
            signal: signal,
            fMin: 65.0,
            fMax: 2093.0,
            sr: sr,
            frameLength: 2048,
            troughThreshold: 0.1,
            center: true
        )

        // Result should be 1-D
        XCTAssertEqual(result.shape.count, 1, "YIN output should be 1-D")

        // Check that most frames are close to 440 Hz
        // Skip edge frames which may be unreliable due to padding
        let nFrames = result.shape[0]
        XCTAssertGreaterThan(nFrames, 0, "Should have at least one frame")

        var voicedCount = 0
        var totalError: Float = 0
        let startFrame = 2
        let endFrame = max(startFrame, nFrames - 2)

        for f in startFrame..<endFrame {
            let f0 = result[f]
            if f0 > 0 {
                voicedCount += 1
                totalError += abs(f0 - 440.0)
            }
        }

        let interiorCount = endFrame - startFrame
        guard interiorCount > 0 else { return }

        // Most interior frames should be voiced
        let voicedRatio = Float(voicedCount) / Float(interiorCount)
        XCTAssertGreaterThan(voicedRatio, 0.8,
                             "At least 80% of interior frames should be voiced for a pure sine")

        if voicedCount > 0 {
            let avgError = totalError / Float(voicedCount)
            XCTAssertLessThan(avgError, 5.0,
                              "Average F0 error should be < 5 Hz for a 440 Hz sine, got \(avgError)")
        }
    }

    func testYINSineWave220() {
        let sr = 22050
        let signal = makeSineSignal(frequency: 220.0, sr: sr, duration: 1.0)

        let result = YIN.yin(
            signal: signal,
            fMin: 65.0,
            fMax: 2093.0,
            sr: sr,
            frameLength: 2048,
            troughThreshold: 0.1,
            center: true
        )

        let nFrames = result.shape[0]
        let startFrame = 2
        let endFrame = max(startFrame, nFrames - 2)
        var voicedCount = 0
        var totalError: Float = 0

        for f in startFrame..<endFrame {
            let f0 = result[f]
            if f0 > 0 {
                voicedCount += 1
                totalError += abs(f0 - 220.0)
            }
        }

        if voicedCount > 0 {
            let avgError = totalError / Float(voicedCount)
            XCTAssertLessThan(avgError, 5.0,
                              "Average F0 error should be < 5 Hz for a 220 Hz sine, got \(avgError)")
        }
    }

    // MARK: - Silence Test

    func testYINSilence() {
        let sr = 22050
        let silence = Signal(data: [Float](repeating: 0, count: sr), sampleRate: sr)

        let result = YIN.yin(
            signal: silence,
            fMin: 65.0,
            fMax: 2093.0,
            sr: sr,
            frameLength: 2048,
            troughThreshold: 0.1,
            center: true
        )

        // All frames should be 0 (unvoiced) for silence
        for i in 0..<result.count {
            XCTAssertEqual(result[i], 0.0,
                           "Silence should yield unvoiced (0) at frame \(i), got \(result[i])")
        }
    }

    // MARK: - Output Shape

    func testYINOutputShape() {
        let sr = 22050
        let signal = makeSineSignal(frequency: 440.0, sr: sr, duration: 1.0)
        let frameLength = 2048
        let hopLength = 512

        let result = YIN.yin(
            signal: signal,
            fMin: 65.0,
            fMax: 2093.0,
            sr: sr,
            frameLength: frameLength,
            hopLength: hopLength,
            center: true
        )

        // With center=true, padded length = 22050 + 2048 = 24098
        // nFrames = 1 + (24098 - 2048) / 512 = 1 + 22050/512 = 1 + 43 = 44
        let paddedLength = sr + frameLength
        let expectedFrames = 1 + (paddedLength - frameLength) / hopLength
        XCTAssertEqual(result.shape[0], expectedFrames,
                       "Expected \(expectedFrames) frames, got \(result.shape[0])")
    }

    func testYINOutputShapeNoCentering() {
        let sr = 22050
        let signal = makeSineSignal(frequency: 440.0, sr: sr, duration: 1.0)
        let frameLength = 2048
        let hopLength = 512

        let result = YIN.yin(
            signal: signal,
            fMin: 65.0,
            fMax: 2093.0,
            sr: sr,
            frameLength: frameLength,
            hopLength: hopLength,
            center: false
        )

        // Without centering: nFrames = 1 + (22050 - 2048) / 512 = 1 + 39 = 40
        let expectedFrames = 1 + (sr - frameLength) / hopLength
        XCTAssertEqual(result.shape[0], expectedFrames,
                       "Expected \(expectedFrames) frames without centering, got \(result.shape[0])")
    }

    // MARK: - Frequency Clipping

    func testYINClippingBelowFMin() {
        // Use a very low frequency signal and set fMin above it
        let sr = 22050
        let signal = makeSineSignal(frequency: 50.0, sr: sr, duration: 1.0)

        let result = YIN.yin(
            signal: signal,
            fMin: 100.0,
            fMax: 2093.0,
            sr: sr,
            frameLength: 2048,
            center: true
        )

        // All detected frequencies should be >= fMin or 0 (unvoiced)
        for i in 0..<result.count {
            let f0 = result[i]
            XCTAssertTrue(f0 == 0 || f0 >= 100.0,
                          "F0 should be >= fMin or 0, got \(f0) at frame \(i)")
        }
    }

    func testYINClippingAboveFMax() {
        // Use a high frequency signal and set fMax below it
        let sr = 22050
        let signal = makeSineSignal(frequency: 4000.0, sr: sr, duration: 1.0)

        let result = YIN.yin(
            signal: signal,
            fMin: 65.0,
            fMax: 2000.0,
            sr: sr,
            frameLength: 2048,
            center: true
        )

        // All detected frequencies should be <= fMax or 0 (unvoiced)
        for i in 0..<result.count {
            let f0 = result[i]
            XCTAssertTrue(f0 == 0 || f0 <= 2000.0,
                          "F0 should be <= fMax or 0, got \(f0) at frame \(i)")
        }
    }

    // MARK: - Values Finite

    func testYINValuesFinite() {
        let sr = 22050
        let signal = makeSineSignal(frequency: 440.0, sr: sr, duration: 1.0)

        let result = YIN.yin(
            signal: signal,
            fMin: 65.0,
            fMax: 2093.0,
            sr: sr,
            center: true
        )

        for i in 0..<result.count {
            XCTAssertFalse(result[i].isNaN, "YIN should not produce NaN at frame \(i)")
            XCTAssertFalse(result[i].isInfinite, "YIN should not produce Inf at frame \(i)")
        }
    }

    // MARK: - Non-negative

    func testYINNonNegative() {
        let sr = 22050
        let signal = makeSineSignal(frequency: 440.0, sr: sr, duration: 1.0)

        let result = YIN.yin(
            signal: signal,
            fMin: 65.0,
            fMax: 2093.0,
            sr: sr,
            center: true
        )

        for i in 0..<result.count {
            XCTAssertGreaterThanOrEqual(result[i], 0.0,
                                        "YIN F0 should be non-negative at frame \(i)")
        }
    }

    // MARK: - Default Hop Length

    func testYINDefaultHopLength() {
        let sr = 22050
        let signal = makeSineSignal(frequency: 440.0, sr: sr, duration: 1.0)
        let frameLength = 2048

        // Default hopLength should be frameLength / 4 = 512
        let result = YIN.yin(
            signal: signal,
            fMin: 65.0,
            fMax: 2093.0,
            sr: sr,
            frameLength: frameLength,
            center: true
        )

        let paddedLength = sr + frameLength
        let expectedFrames = 1 + (paddedLength - frameLength) / (frameLength / 4)
        XCTAssertEqual(result.shape[0], expectedFrames,
                       "Default hop should be frameLength/4")
    }

    // MARK: - pYIN Tests

    func testPYINOutputShape() {
        let sr = 22050
        let signal = makeSineSignal(frequency: 440.0, sr: sr, duration: 1.0)
        let frameLength = 2048
        let hopLength = 512

        let result = YIN.pyin(
            signal: signal,
            fMin: 65.0,
            fMax: 2093.0,
            sr: sr,
            frameLength: frameLength,
            hopLength: hopLength,
            center: true
        )

        // Should be [3, nFrames]
        XCTAssertEqual(result.shape.count, 2, "pYIN output should be 2-D")
        XCTAssertEqual(result.shape[0], 3, "pYIN output row count should be 3")

        let paddedLength = sr + frameLength
        let expectedFrames = 1 + (paddedLength - frameLength) / hopLength
        XCTAssertEqual(result.shape[1], expectedFrames,
                       "Expected \(expectedFrames) frames, got \(result.shape[1])")
    }

    func testPYINSineWave440() {
        let sr = 22050
        let signal = makeSineSignal(frequency: 440.0, sr: sr, duration: 1.0)

        let result = YIN.pyin(
            signal: signal,
            fMin: 65.0,
            fMax: 2093.0,
            sr: sr,
            frameLength: 2048,
            center: true
        )

        let nFrames = result.shape[1]
        XCTAssertGreaterThan(nFrames, 0)

        // Check f0 values (row 0) — skip edge frames
        let startFrame = 2
        let endFrame = max(startFrame, nFrames - 2)
        var voicedCount = 0
        var totalError: Float = 0

        for f in startFrame..<endFrame {
            let f0 = result[f]  // row 0, col f
            let voicedFlag = result[nFrames + f]  // row 1, col f

            if voicedFlag > 0.5 && f0 > 0 {
                voicedCount += 1
                totalError += abs(f0 - 440.0)
            }
        }

        let interiorCount = endFrame - startFrame
        guard interiorCount > 0 else { return }

        let voicedRatio = Float(voicedCount) / Float(interiorCount)
        XCTAssertGreaterThan(voicedRatio, 0.7,
                             "At least 70% of interior frames should be voiced for a pure sine, got \(voicedRatio)")

        if voicedCount > 0 {
            let avgError = totalError / Float(voicedCount)
            XCTAssertLessThan(avgError, 10.0,
                              "Average F0 error should be < 10 Hz for a 440 Hz sine, got \(avgError)")
        }
    }

    func testPYINSilence() {
        let sr = 22050
        let silence = Signal(data: [Float](repeating: 0, count: sr), sampleRate: sr)

        let result = YIN.pyin(
            signal: silence,
            fMin: 65.0,
            fMax: 2093.0,
            sr: sr,
            frameLength: 2048,
            center: true
        )

        let nFrames = result.shape[1]
        XCTAssertGreaterThan(nFrames, 0)

        // Most frames should be unvoiced for silence
        var unvoicedCount = 0
        for f in 0..<nFrames {
            let voicedFlag = result[nFrames + f]
            if voicedFlag < 0.5 {
                unvoicedCount += 1
            }
        }

        let unvoicedRatio = Float(unvoicedCount) / Float(nFrames)
        XCTAssertGreaterThan(unvoicedRatio, 0.7,
                             "At least 70% of frames should be unvoiced for silence, got \(unvoicedRatio)")
    }

    func testPYINVoicedProbabilitiesInRange() {
        let sr = 22050
        let signal = makeSineSignal(frequency: 440.0, sr: sr, duration: 1.0)

        let result = YIN.pyin(
            signal: signal,
            fMin: 65.0,
            fMax: 2093.0,
            sr: sr,
            center: true
        )

        let nFrames = result.shape[1]
        for f in 0..<nFrames {
            let prob = result[2 * nFrames + f]  // row 2
            XCTAssertGreaterThanOrEqual(prob, 0.0,
                                         "Voiced prob should be >= 0 at frame \(f), got \(prob)")
            XCTAssertLessThanOrEqual(prob, 1.0,
                                      "Voiced prob should be <= 1 at frame \(f), got \(prob)")
        }
    }

    func testPYINF0InRange() {
        let sr = 22050
        let signal = makeSineSignal(frequency: 440.0, sr: sr, duration: 1.0)

        let result = YIN.pyin(
            signal: signal,
            fMin: 65.0,
            fMax: 2093.0,
            sr: sr,
            center: true
        )

        let nFrames = result.shape[1]
        for f in 0..<nFrames {
            let f0 = result[f]
            let voicedFlag = result[nFrames + f]
            if voicedFlag > 0.5 {
                XCTAssertGreaterThanOrEqual(f0, 65.0,
                                             "Voiced f0 should be >= fMin at frame \(f)")
                XCTAssertLessThanOrEqual(f0, 2093.0,
                                          "Voiced f0 should be <= fMax at frame \(f)")
            } else {
                XCTAssertEqual(f0, 0.0,
                               "Unvoiced f0 should be 0 at frame \(f), got \(f0)")
            }
        }
    }

    func testPYINVoicedFlagBinary() {
        let sr = 22050
        let signal = makeSineSignal(frequency: 440.0, sr: sr, duration: 1.0)

        let result = YIN.pyin(
            signal: signal,
            fMin: 65.0,
            fMax: 2093.0,
            sr: sr,
            center: true
        )

        let nFrames = result.shape[1]
        for f in 0..<nFrames {
            let flag = result[nFrames + f]
            XCTAssertTrue(flag == 0.0 || flag == 1.0,
                          "Voiced flag should be 0 or 1, got \(flag) at frame \(f)")
        }
    }
}
