import XCTest
@testable import MetalMomCore

final class PitchShiftTests: XCTestCase {

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

    func testZeroSteps() {
        // n_steps=0 should return output of same length as input
        let signal = makeSine(duration: 0.5)
        let result = PitchShift.pitchShift(signal: signal, nSteps: 0, nFFT: 1024)

        XCTAssertEqual(result.count, signal.count,
            "nSteps=0 should return same length as input")
    }

    func testOutputLengthMatchesInput() {
        // Output should always match input length (trim/pad)
        let signal = makeSine(duration: 0.5)

        let upResult = PitchShift.pitchShift(signal: signal, nSteps: 12, nFFT: 1024)
        XCTAssertEqual(upResult.count, signal.count,
            "Pitch shift output should match input length")

        let downResult = PitchShift.pitchShift(signal: signal, nSteps: -12, nFFT: 1024)
        XCTAssertEqual(downResult.count, signal.count,
            "Pitch shift output should match input length")
    }

    func testOutputIs1D() {
        let signal = makeSine(duration: 0.25)
        let result = PitchShift.pitchShift(signal: signal, nSteps: 5, nFFT: 1024)

        XCTAssertEqual(result.shape.count, 1, "Output should be 1D")
    }

    func testOutputValuesAreFinite() {
        let signal = makeSine(duration: 0.25)

        let upResult = PitchShift.pitchShift(signal: signal, nSteps: 7, nFFT: 1024)
        XCTAssertTrue(allFinite(upResult), "Output should contain only finite values (shift up)")

        let downResult = PitchShift.pitchShift(signal: signal, nSteps: -3, nFFT: 1024)
        XCTAssertTrue(allFinite(downResult), "Output should contain only finite values (shift down)")
    }

    func testPositiveAndNegativeShiftsDiffer() {
        // Shifting up vs down should produce different results
        let signal = makeSine(duration: 0.25)

        let up = PitchShift.pitchShift(signal: signal, nSteps: 5, nFFT: 1024)
        let down = PitchShift.pitchShift(signal: signal, nSteps: -5, nFFT: 1024)

        // Compare a few samples to verify they differ
        var differ = false
        up.withUnsafeBufferPointer { upBuf in
            down.withUnsafeBufferPointer { downBuf in
                for i in stride(from: 100, to: min(up.count, down.count), by: 100) {
                    if abs(upBuf[i] - downBuf[i]) > 1e-6 {
                        differ = true
                        break
                    }
                }
            }
        }
        XCTAssertTrue(differ, "Positive and negative pitch shifts should produce different results")
    }

    func testNonEmptyOutput() {
        let signal = makeSine(duration: 0.25)
        let result = PitchShift.pitchShift(signal: signal, nSteps: 3, nFFT: 1024)

        XCTAssertGreaterThan(result.count, 0, "Output should not be empty")
    }

    func testCustomBinsPerOctave() {
        // Should work with non-default bins_per_octave (e.g., 24 for quarter-tones)
        let signal = makeSine(duration: 0.25)
        let result = PitchShift.pitchShift(signal: signal, nSteps: 6, binsPerOctave: 24, nFFT: 1024)

        XCTAssertEqual(result.count, signal.count)
        XCTAssertTrue(allFinite(result))
    }

    func testCustomHopLength() {
        let signal = makeSine(duration: 0.25)
        let result = PitchShift.pitchShift(signal: signal, nSteps: 2, nFFT: 1024, hopLength: 128)

        XCTAssertGreaterThan(result.count, 0)
        XCTAssertTrue(allFinite(result))
    }

    func testEmptySignal() {
        let signal = Signal(data: [], sampleRate: 22050)
        let result = PitchShift.pitchShift(signal: signal, nSteps: 5)

        XCTAssertEqual(result.count, 0, "Empty input should produce empty output")
    }
}
