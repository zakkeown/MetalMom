import XCTest
@testable import MetalMomCore

final class SplitTests: XCTestCase {

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

    /// Generate a signal with silence gaps: [tone][silence][tone][silence][tone]
    private func makeGappedSignal(
        toneDuration: Float = 0.2,
        silenceDuration: Float = 0.2,
        toneCount: Int = 3,
        frequency: Float = 440.0,
        sr: Int = 22050
    ) -> Signal {
        let toneSamples = Int(toneDuration * Float(sr))
        let silenceSamples = Int(silenceDuration * Float(sr))
        let totalLength = toneCount * toneSamples + (toneCount - 1) * silenceSamples

        var data = [Float](repeating: 0, count: totalLength)
        let angularFreq = 2.0 * Float.pi * frequency / Float(sr)

        for t in 0..<toneCount {
            let offset = t * (toneSamples + silenceSamples)
            for i in 0..<toneSamples {
                data[offset + i] = sinf(angularFreq * Float(i))
            }
        }
        return Signal(data: data, sampleRate: sr)
    }

    // MARK: - Tests

    func testSplitWithSilenceGaps() {
        // Signal with 3 tone bursts separated by silence
        let signal = makeGappedSignal(toneDuration: 0.2, silenceDuration: 0.3, toneCount: 3)

        let intervals = Split.split(signal: signal, topDb: 40.0)

        // Should detect multiple non-silent intervals
        XCTAssertGreaterThanOrEqual(intervals.count, 2,
            "Should detect at least 2 intervals, got \(intervals.count)")

        // Each interval should have valid bounds
        for (i, interval) in intervals.enumerated() {
            XCTAssertGreaterThanOrEqual(interval.start, 0,
                "Interval \(i) start should be >= 0")
            XCTAssertLessThanOrEqual(interval.end, signal.count,
                "Interval \(i) end should be <= signal length")
            XCTAssertLessThan(interval.start, interval.end,
                "Interval \(i) start should be < end")
        }

        // Intervals should be non-overlapping and in order
        for i in 1..<intervals.count {
            XCTAssertGreaterThanOrEqual(intervals[i].start, intervals[i-1].end,
                "Interval \(i) should start after interval \(i-1) ends")
        }
    }

    func testSplitContinuousSignal() {
        // Pure sine with no silence → should return 1 interval
        let signal = makeSine(duration: 0.5)

        let intervals = Split.split(signal: signal, topDb: 60.0)

        XCTAssertEqual(intervals.count, 1,
            "Continuous signal should produce 1 interval, got \(intervals.count)")

        if let interval = intervals.first {
            // Should cover most of the signal
            XCTAssertLessThanOrEqual(interval.start, 512,
                "Start should be near the beginning")
            XCTAssertGreaterThan(interval.end, signal.count - 2048,
                "End should be near the signal end")
        }
    }

    func testSplitAllSilence() {
        // All-zero signal → no intervals
        let data = [Float](repeating: 0, count: 22050)
        let signal = Signal(data: data, sampleRate: 22050)

        let intervals = Split.split(signal: signal, topDb: 60.0)

        XCTAssertEqual(intervals.count, 0,
            "All-silence should produce 0 intervals, got \(intervals.count)")
    }

    func testSplitEmptySignal() {
        let signal = Signal(data: [], sampleRate: 22050)
        let intervals = Split.split(signal: signal)

        XCTAssertEqual(intervals.count, 0,
            "Empty signal should produce 0 intervals")
    }

    func testSplitBoundariesAreReasonable() {
        // Create a signal: 0.3s silence, 0.3s tone, 0.3s silence
        let sr = 22050
        let silenceSamples = Int(0.3 * Float(sr))
        let toneSamples = Int(0.3 * Float(sr))
        let totalLength = silenceSamples + toneSamples + silenceSamples
        var data = [Float](repeating: 0, count: totalLength)
        let angularFreq = 2.0 * Float.pi * 440.0 / Float(sr)
        for i in 0..<toneSamples {
            data[silenceSamples + i] = sinf(angularFreq * Float(i))
        }
        let signal = Signal(data: data, sampleRate: sr)

        let intervals = Split.split(signal: signal, topDb: 60.0)

        XCTAssertEqual(intervals.count, 1,
            "Should detect exactly 1 interval, got \(intervals.count)")

        if let interval = intervals.first {
            // Start should be close to where the tone begins (0.3s = 6615 samples)
            let expectedStart = silenceSamples
            XCTAssertLessThan(abs(interval.start - expectedStart), 2048,
                "Start \(interval.start) should be near tone start \(expectedStart)")

            // End should be close to where the tone ends
            let expectedEnd = silenceSamples + toneSamples
            XCTAssertLessThan(abs(interval.end - expectedEnd), 2048,
                "End \(interval.end) should be near tone end \(expectedEnd)")
        }
    }

    func testSplitCustomFrameAndHop() {
        // Should not crash with different frame/hop sizes
        let signal = makeSine(duration: 0.5)

        let intervals1 = Split.split(signal: signal, frameLength: 1024, hopLength: 256)
        XCTAssertGreaterThan(intervals1.count, 0)

        let intervals2 = Split.split(signal: signal, frameLength: 4096, hopLength: 1024)
        XCTAssertGreaterThan(intervals2.count, 0)
    }

    func testSplitVeryShortSignal() {
        // Signal shorter than one frame
        let data: [Float] = [0.0, 0.0, 0.5, 0.5, 0.0, 0.0]
        let signal = Signal(data: data, sampleRate: 22050)

        // With default frame_length=2048, this is too short for any frames
        let intervals = Split.split(signal: signal)
        // Should handle gracefully (empty since no frames can be computed)
        XCTAssertEqual(intervals.count, 0)
    }
}
