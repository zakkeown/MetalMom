import XCTest
@testable import MetalMomCore

final class TrimTests: XCTestCase {

    // MARK: - Helpers

    /// Generate a pure sine wave signal.
    private func makeSine(frequency: Float = 440.0, sr: Int = 22050, duration: Float = 0.5) -> Signal {
        let length = Int(Float(sr) * duration)
        var data = [Float](repeating: 0, count: length)
        let angularFreq = 2.0 * Float.pi * frequency / Float(sr)
        for i in 0..<length {
            data[i] = sinf(angularFreq * Float(i))
        }
        return Signal(data: data, sampleRate: sr)
    }

    /// Generate a signal with silence at start and end surrounding a sine burst.
    private func makeSilencePadded(
        silenceStart: Float = 0.2,
        toneDuration: Float = 0.3,
        silenceEnd: Float = 0.2,
        frequency: Float = 440.0,
        sr: Int = 22050
    ) -> Signal {
        let startSamples = Int(silenceStart * Float(sr))
        let toneSamples = Int(toneDuration * Float(sr))
        let endSamples = Int(silenceEnd * Float(sr))
        let totalLength = startSamples + toneSamples + endSamples

        var data = [Float](repeating: 0, count: totalLength)
        let angularFreq = 2.0 * Float.pi * frequency / Float(sr)
        for i in 0..<toneSamples {
            data[startSamples + i] = sinf(angularFreq * Float(i))
        }
        return Signal(data: data, sampleRate: sr)
    }

    // MARK: - Tests

    func testTrimRemovesSilence() {
        // Signal with silence padding at start and end
        let signal = makeSilencePadded(silenceStart: 0.3, toneDuration: 0.3, silenceEnd: 0.3)
        let originalLength = signal.count

        let (trimmed, startIdx, endIdx) = Trim.trim(signal: signal, topDb: 60.0)

        // Trimmed should be shorter than original
        XCTAssertLessThan(trimmed.count, originalLength,
            "Trimmed (\(trimmed.count)) should be shorter than original (\(originalLength))")
        XCTAssertGreaterThan(trimmed.count, 0, "Trimmed should not be empty")

        // Indices should make sense
        XCTAssertGreaterThan(startIdx, 0, "Start index should be past the leading silence")
        XCTAssertLessThan(endIdx, originalLength, "End index should be before trailing silence ends")
        XCTAssertEqual(trimmed.count, endIdx - startIdx, "Trimmed length should match index range")
    }

    func testTrimAllSilenceReturnsEmpty() {
        // All-zero signal
        let data = [Float](repeating: 0, count: 22050)
        let signal = Signal(data: data, sampleRate: 22050)

        let (trimmed, startIdx, endIdx) = Trim.trim(signal: signal, topDb: 60.0)

        XCTAssertEqual(trimmed.count, 0, "All-silence should produce empty output")
        XCTAssertEqual(startIdx, 0)
        XCTAssertEqual(endIdx, 0)
    }

    func testTrimNoSilenceReturnsSame() {
        // Pure sine with no silence padding
        let signal = makeSine(duration: 0.5)
        let originalLength = signal.count

        let (trimmed, startIdx, endIdx) = Trim.trim(signal: signal, topDb: 60.0)

        // The trimmed signal should be approximately the same length
        // (may differ slightly due to frame alignment)
        XCTAssertGreaterThan(trimmed.count, 0)

        // Start should be at or near the beginning
        XCTAssertLessThanOrEqual(startIdx, 512, "Start index should be near the beginning")

        // End should be at or near the end of the signal
        XCTAssertGreaterThan(endIdx, originalLength - 2048,
            "End index should be near the end of the signal")
    }

    func testTrimIndicesAreCorrect() {
        let signal = makeSilencePadded(silenceStart: 0.2, toneDuration: 0.3, silenceEnd: 0.2)
        let sr = signal.sampleRate

        let (trimmed, startIdx, endIdx) = Trim.trim(signal: signal, topDb: 60.0)

        // The tone starts at 0.2 seconds = 4410 samples
        // Start index should be close to where the tone begins
        let expectedToneStart = Int(0.2 * Float(sr))
        XCTAssertLessThan(abs(startIdx - expectedToneStart), 2048,
            "Start index \(startIdx) should be within 2048 of tone start \(expectedToneStart)")

        // The tone ends at 0.5 seconds = 11025 samples
        let expectedToneEnd = Int(0.5 * Float(sr))
        XCTAssertLessThan(abs(endIdx - expectedToneEnd), 2048,
            "End index \(endIdx) should be within 2048 of tone end \(expectedToneEnd)")

        // Trimmed length should match
        XCTAssertEqual(trimmed.count, endIdx - startIdx)
    }

    func testTrimCustomTopDb() {
        // With a very low threshold (high topDb), more signal should be kept
        let signal = makeSilencePadded(silenceStart: 0.2, toneDuration: 0.3, silenceEnd: 0.2)

        let (trimmedStrict, _, _) = Trim.trim(signal: signal, topDb: 20.0)
        let (trimmedLoose, _, _) = Trim.trim(signal: signal, topDb: 80.0)

        // With a looser threshold, we should keep at least as much signal
        XCTAssertGreaterThanOrEqual(trimmedLoose.count, trimmedStrict.count,
            "Looser threshold (80dB) should keep at least as much signal as strict (20dB)")
    }

    func testTrimCustomFrameAndHop() {
        let signal = makeSilencePadded(silenceStart: 0.2, toneDuration: 0.3, silenceEnd: 0.2)

        // Should not crash with different frame/hop sizes
        let (trimmed1, s1, e1) = Trim.trim(signal: signal, frameLength: 1024, hopLength: 256)
        XCTAssertGreaterThan(trimmed1.count, 0)
        XCTAssertEqual(trimmed1.count, e1 - s1)

        let (trimmed2, s2, e2) = Trim.trim(signal: signal, frameLength: 4096, hopLength: 1024)
        XCTAssertGreaterThan(trimmed2.count, 0)
        XCTAssertEqual(trimmed2.count, e2 - s2)
    }

    func testTrimEmptySignal() {
        let signal = Signal(data: [], sampleRate: 22050)
        let (trimmed, startIdx, endIdx) = Trim.trim(signal: signal)

        XCTAssertEqual(trimmed.count, 0)
        XCTAssertEqual(startIdx, 0)
        XCTAssertEqual(endIdx, 0)
    }

    func testTrimVeryShortSignal() {
        // Signal shorter than one frame
        let data: [Float] = [0.0, 0.0, 0.5, 0.5, 0.0, 0.0]
        let signal = Signal(data: data, sampleRate: 22050)

        // With default frame_length=2048, this is too short for any frames
        let (trimmed, _, _) = Trim.trim(signal: signal)
        // Should handle gracefully (empty or the original)
        // Since no frames can be computed, we expect empty
        XCTAssertEqual(trimmed.count, 0)
    }
}
