import XCTest
@testable import MetalMomCore

final class SignalGenTests: XCTestCase {
    func testToneShape() {
        let result = SignalGen.tone(frequency: 440, sr: 22050, length: 22050)
        XCTAssertEqual(result.count, 22050)
        XCTAssertEqual(result.sampleRate, 22050)
    }

    func testToneFrequency() {
        // A 440 Hz tone at 22050 Hz should cross zero approximately 2*440 times in 1 second
        let result = SignalGen.tone(frequency: 440, sr: 22050, length: 22050)
        var crossings = 0
        for i in 1..<result.count {
            if (result[i-1] >= 0 && result[i] < 0) || (result[i-1] < 0 && result[i] >= 0) {
                crossings += 1
            }
        }
        // 2 zero crossings per cycle
        XCTAssertEqual(crossings, 880, accuracy: 2)
    }

    func testTonePhaseOffset() {
        let result = SignalGen.tone(frequency: 440, sr: 22050, length: 100, phi: Float.pi / 2)
        // sin(pi/2) = 1
        XCTAssertEqual(result[0], 1.0, accuracy: 0.01)
    }

    func testChirpShape() {
        let result = SignalGen.chirp(fmin: 100, fmax: 1000, sr: 22050, length: 22050)
        XCTAssertEqual(result.count, 22050)
    }

    func testChirpLinear() {
        let result = SignalGen.chirp(fmin: 100, fmax: 1000, sr: 22050, length: 22050, linear: true)
        // All values should be in [-1, 1]
        for i in 0..<result.count {
            XCTAssertLessThanOrEqual(abs(result[i]), 1.001)
        }
    }

    func testChirpLog() {
        let result = SignalGen.chirp(fmin: 100, fmax: 1000, sr: 22050, length: 22050, linear: false)
        for i in 0..<result.count {
            XCTAssertLessThanOrEqual(abs(result[i]), 1.001)
        }
    }

    func testClicksShape() {
        let result = SignalGen.clicks(times: [0.0, 0.5, 1.0], sr: 22050, length: 33075)
        XCTAssertEqual(result.count, 33075)
    }

    func testClicksNonZero() {
        let result = SignalGen.clicks(times: [0.0], sr: 22050, length: 22050)
        // Should have some non-zero values near sample 0
        var hasNonZero = false
        for i in 0..<min(2205, result.count) {
            if abs(result[i]) > 0.01 {
                hasNonZero = true
                break
            }
        }
        XCTAssertTrue(hasNonZero, "Clicks should produce non-zero samples")
    }

    func testToneDuration() {
        let result = SignalGen.tone(frequency: 440, sr: 22050, duration: 0.5)
        XCTAssertEqual(result.count, 11025)
    }
}
