import XCTest
@testable import MetalMomCore

final class TuningTests: XCTestCase {

    // MARK: - Helpers

    private func makeSineSignal(frequency: Float = 440.0, sr: Int = 22050,
                                 duration: Float = 1.0) -> Signal {
        return SignalGen.tone(frequency: frequency, sr: sr,
                              duration: Double(duration), phi: 0)
    }

    // MARK: - 440 Hz → tuning near 0.0

    func testTuning440HzNearZero() {
        let sr = 22050
        let signal = makeSineSignal(frequency: 440.0, sr: sr, duration: 1.0)

        let tuning = Tuning.estimateTuning(
            signal: signal,
            sr: sr,
            nFFT: 2048,
            resolution: 0.01,
            binsPerOctave: 12,
            center: true
        )

        XCTAssertEqual(tuning, 0.0, accuracy: 0.15,
                        "440 Hz should produce tuning near 0.0")
    }

    // MARK: - 442 Hz → slightly positive tuning

    func testTuning442HzSlightlyPositive() {
        let sr = 22050
        let signal = makeSineSignal(frequency: 442.0, sr: sr, duration: 1.0)

        let tuning = Tuning.estimateTuning(
            signal: signal,
            sr: sr,
            nFFT: 2048,
            resolution: 0.01,
            binsPerOctave: 12,
            center: true
        )

        // 442 Hz is about +0.079 semitones from 440
        // 12 * log2(442/440) ≈ 0.0787
        XCTAssertGreaterThan(tuning, 0.0,
                              "442 Hz should produce positive tuning")
        XCTAssertLessThan(tuning, 0.5,
                           "442 Hz tuning should be less than 0.5")
    }

    // MARK: - 438 Hz → slightly negative tuning

    func testTuning438HzSlightlyNegative() {
        let sr = 22050
        let signal = makeSineSignal(frequency: 438.0, sr: sr, duration: 1.0)

        let tuning = Tuning.estimateTuning(
            signal: signal,
            sr: sr,
            nFFT: 2048,
            resolution: 0.01,
            binsPerOctave: 12,
            center: true
        )

        // 438 Hz is about -0.079 semitones from 440
        // 12 * log2(438/440) ≈ -0.0789
        XCTAssertLessThan(tuning, 0.0,
                           "438 Hz should produce negative tuning")
        XCTAssertGreaterThan(tuning, -0.5,
                              "438 Hz tuning should be greater than -0.5")
    }

    // MARK: - Result in range [-0.5, 0.5]

    func testTuningResultInRange() {
        let sr = 22050
        let signal = makeSineSignal(frequency: 440.0, sr: sr, duration: 1.0)

        let tuning = Tuning.estimateTuning(
            signal: signal,
            sr: sr,
            nFFT: 2048,
            resolution: 0.01,
            binsPerOctave: 12,
            center: true
        )

        XCTAssertGreaterThanOrEqual(tuning, -0.5,
            "Tuning should be >= -0.5")
        XCTAssertLessThanOrEqual(tuning, 0.5,
            "Tuning should be <= 0.5")
    }

    // MARK: - Silence returns 0.0

    func testSilenceReturnsZero() {
        let sr = 22050
        let silence = Signal(data: [Float](repeating: 0, count: sr), sampleRate: sr)

        let tuning = Tuning.estimateTuning(
            signal: silence,
            sr: sr,
            nFFT: 2048,
            resolution: 0.01,
            binsPerOctave: 12,
            center: true
        )

        XCTAssertEqual(tuning, 0.0,
                        "Silence should produce tuning of 0.0")
    }

    // MARK: - Different FFT sizes

    func testTuningWithSmallFFT() {
        let sr = 22050
        let signal = makeSineSignal(frequency: 440.0, sr: sr, duration: 1.0)

        let tuning = Tuning.estimateTuning(
            signal: signal,
            sr: sr,
            nFFT: 1024,
            resolution: 0.01,
            binsPerOctave: 12,
            center: true
        )

        XCTAssertGreaterThanOrEqual(tuning, -0.5)
        XCTAssertLessThanOrEqual(tuning, 0.5)
    }
}
