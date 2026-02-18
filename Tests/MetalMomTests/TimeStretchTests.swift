import XCTest
@testable import MetalMomCore

final class TimeStretchTests: XCTestCase {

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

    /// Check that all values in a Signal are finite (not NaN or Inf).
    private func allFinite(_ signal: Signal) -> Bool {
        var result = true
        signal.withUnsafeBufferPointer { buf in
            for i in 0..<buf.count {
                if !buf[i].isFinite {
                    result = false
                    return
                }
            }
        }
        return result
    }

    // MARK: - Tests

    func testIdentityRate() {
        // rate=1.0 should produce output approximately the same length as input
        let signal = makeSine(duration: 0.5)
        let inputLength = signal.count

        let result = TimeStretch.timeStretch(signal: signal, rate: 1.0, nFFT: 1024)

        // Allow some tolerance due to STFT windowing effects
        let ratio = Float(result.count) / Float(inputLength)
        XCTAssertGreaterThan(ratio, 0.8, "rate=1.0 output should be ~same length as input")
        XCTAssertLessThan(ratio, 1.2, "rate=1.0 output should be ~same length as input")
    }

    func testSpeedUp() {
        // rate=2.0 should produce output approximately half the length of input
        let signal = makeSine(duration: 0.5)
        let inputLength = signal.count

        let result = TimeStretch.timeStretch(signal: signal, rate: 2.0, nFFT: 1024)

        let expectedLength = Float(inputLength) / 2.0
        let ratio = Float(result.count) / expectedLength
        XCTAssertGreaterThan(ratio, 0.6, "rate=2.0 output should be ~half input length")
        XCTAssertLessThan(ratio, 1.4, "rate=2.0 output should be ~half input length")
    }

    func testSlowDown() {
        // rate=0.5 should produce output approximately double the length of input
        let signal = makeSine(duration: 0.5)
        let inputLength = signal.count

        let result = TimeStretch.timeStretch(signal: signal, rate: 0.5, nFFT: 1024)

        let expectedLength = Float(inputLength) * 2.0
        let ratio = Float(result.count) / expectedLength
        XCTAssertGreaterThan(ratio, 0.6, "rate=0.5 output should be ~double input length")
        XCTAssertLessThan(ratio, 1.4, "rate=0.5 output should be ~double input length")
    }

    func testOutputValuesAreFinite() {
        let signal = makeSine(duration: 0.25)
        let result = TimeStretch.timeStretch(signal: signal, rate: 1.5, nFFT: 1024)

        XCTAssertTrue(allFinite(result), "Output should contain only finite values")
    }

    func testOutputIs1D() {
        let signal = makeSine(duration: 0.25)
        let result = TimeStretch.timeStretch(signal: signal, rate: 1.0, nFFT: 1024)

        XCTAssertEqual(result.shape.count, 1, "Output should be 1D")
    }

    func testNonEmptyOutput() {
        let signal = makeSine(duration: 0.25)
        let result = TimeStretch.timeStretch(signal: signal, rate: 1.5, nFFT: 1024)

        XCTAssertGreaterThan(result.count, 0, "Output should not be empty")
    }

    func testDifferentRatesProduceDifferentLengths() {
        let signal = makeSine(duration: 0.5)

        let fast = TimeStretch.timeStretch(signal: signal, rate: 2.0, nFFT: 1024)
        let slow = TimeStretch.timeStretch(signal: signal, rate: 0.5, nFFT: 1024)

        XCTAssertGreaterThan(slow.count, fast.count,
            "Slow (rate=0.5) should be longer than fast (rate=2.0)")
    }

    func testCustomHopLength() {
        let signal = makeSine(duration: 0.25)
        // Should not crash with custom hop length
        let result = TimeStretch.timeStretch(signal: signal, rate: 1.5, nFFT: 1024, hopLength: 128)

        XCTAssertGreaterThan(result.count, 0)
        XCTAssertTrue(allFinite(result))
    }
}
