import XCTest
@testable import MetalMomCore

final class PreemphasisTests: XCTestCase {

    // MARK: - Helpers

    /// Generate a pure sine wave signal.
    private func makeSine(frequency: Float = 440.0, sr: Int = 22050, duration: Float = 0.1) -> Signal {
        let length = Int(Float(sr) * duration)
        var data = [Float](repeating: 0, count: length)
        let angularFreq = 2.0 * Float.pi * frequency / Float(sr)
        for i in 0..<length {
            data[i] = sinf(angularFreq * Float(i))
        }
        return Signal(data: data, sampleRate: sr)
    }

    // MARK: - Preemphasis Tests

    func testPreemphasisOutputLength() {
        let signal = makeSine()
        let result = Preemphasis.preemphasis(signal: signal)
        XCTAssertEqual(result.count, signal.count,
            "Output length should match input length")
    }

    func testPreemphasisFirstSample() {
        // y[0] = x[0] since x[-1] is assumed to be 0
        let data: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0]
        let signal = Signal(data: data, sampleRate: 22050)

        let result = Preemphasis.preemphasis(signal: signal, coef: 0.97)

        result.withUnsafeBufferPointer { buf in
            XCTAssertEqual(buf[0], 1.0, accuracy: 1e-6,
                "First sample should equal input x[0]")
        }
    }

    func testPreemphasisManualValues() {
        // Manual computation: y[n] = x[n] - 0.97 * x[n-1]
        let data: [Float] = [1.0, 2.0, 3.0, 4.0]
        let signal = Signal(data: data, sampleRate: 22050)
        let coef: Float = 0.97

        let result = Preemphasis.preemphasis(signal: signal, coef: coef)

        let expected: [Float] = [
            1.0,                      // 1.0 - 0.97 * 0
            2.0 - coef * 1.0,        // 1.03
            3.0 - coef * 2.0,        // 1.06
            4.0 - coef * 3.0,        // 1.09
        ]

        result.withUnsafeBufferPointer { buf in
            for i in 0..<4 {
                XCTAssertEqual(buf[i], expected[i], accuracy: 1e-5,
                    "Sample \(i): expected \(expected[i]), got \(buf[i])")
            }
        }
    }

    func testPreemphasisCoefZeroIsIdentity() {
        // With coef=0, preemphasis should be identity: y[n] = x[n] - 0 * x[n-1] = x[n]
        let signal = makeSine()
        let result = Preemphasis.preemphasis(signal: signal, coef: 0.0)

        signal.withUnsafeBufferPointer { inBuf in
            result.withUnsafeBufferPointer { outBuf in
                for i in 0..<signal.count {
                    XCTAssertEqual(outBuf[i], inBuf[i], accuracy: 1e-6,
                        "With coef=0, output should equal input at sample \(i)")
                }
            }
        }
    }

    func testPreemphasisEmpty() {
        let signal = Signal(data: [], sampleRate: 22050)
        let result = Preemphasis.preemphasis(signal: signal)
        XCTAssertEqual(result.count, 0, "Empty signal should produce empty output")
    }

    func testPreemphasisSingleSample() {
        let signal = Signal(data: [0.5], sampleRate: 22050)
        let result = Preemphasis.preemphasis(signal: signal)
        XCTAssertEqual(result.count, 1)
        result.withUnsafeBufferPointer { buf in
            XCTAssertEqual(buf[0], 0.5, accuracy: 1e-6)
        }
    }

    // MARK: - Deemphasis Tests

    func testDeemphasisOutputLength() {
        let signal = makeSine()
        let result = Preemphasis.deemphasis(signal: signal)
        XCTAssertEqual(result.count, signal.count,
            "Output length should match input length")
    }

    func testDeemphasisFirstSample() {
        // y[0] = x[0] since y[-1] is assumed to be 0
        let data: [Float] = [1.0, 2.0, 3.0]
        let signal = Signal(data: data, sampleRate: 22050)

        let result = Preemphasis.deemphasis(signal: signal, coef: 0.97)

        result.withUnsafeBufferPointer { buf in
            XCTAssertEqual(buf[0], 1.0, accuracy: 1e-6,
                "First sample should equal input x[0]")
        }
    }

    func testDeemphasisManualValues() {
        // Manual computation: y[n] = x[n] + 0.5 * y[n-1]
        let data: [Float] = [1.0, 1.0, 1.0, 1.0]
        let signal = Signal(data: data, sampleRate: 22050)
        let coef: Float = 0.5

        let result = Preemphasis.deemphasis(signal: signal, coef: coef)

        // y[0] = 1.0
        // y[1] = 1.0 + 0.5 * 1.0 = 1.5
        // y[2] = 1.0 + 0.5 * 1.5 = 1.75
        // y[3] = 1.0 + 0.5 * 1.75 = 1.875
        let expected: [Float] = [1.0, 1.5, 1.75, 1.875]

        result.withUnsafeBufferPointer { buf in
            for i in 0..<4 {
                XCTAssertEqual(buf[i], expected[i], accuracy: 1e-5,
                    "Sample \(i): expected \(expected[i]), got \(buf[i])")
            }
        }
    }

    func testDeemphasisCoefZeroIsIdentity() {
        // With coef=0, deemphasis should be identity: y[n] = x[n] + 0 * y[n-1] = x[n]
        let signal = makeSine()
        let result = Preemphasis.deemphasis(signal: signal, coef: 0.0)

        signal.withUnsafeBufferPointer { inBuf in
            result.withUnsafeBufferPointer { outBuf in
                for i in 0..<signal.count {
                    XCTAssertEqual(outBuf[i], inBuf[i], accuracy: 1e-6,
                        "With coef=0, output should equal input at sample \(i)")
                }
            }
        }
    }

    func testDeemphasisEmpty() {
        let signal = Signal(data: [], sampleRate: 22050)
        let result = Preemphasis.deemphasis(signal: signal)
        XCTAssertEqual(result.count, 0, "Empty signal should produce empty output")
    }

    // MARK: - Round-Trip Tests

    func testRoundTripPreemphasisThenDeemphasis() {
        // deemphasis(preemphasis(x)) should approximately equal x
        let signal = makeSine(duration: 0.05)
        let preemphasized = Preemphasis.preemphasis(signal: signal, coef: 0.97)
        let roundTrip = Preemphasis.deemphasis(signal: preemphasized, coef: 0.97)

        XCTAssertEqual(roundTrip.count, signal.count)

        signal.withUnsafeBufferPointer { inBuf in
            roundTrip.withUnsafeBufferPointer { outBuf in
                for i in 0..<signal.count {
                    XCTAssertEqual(outBuf[i], inBuf[i], accuracy: 1e-4,
                        "Round-trip should recover original at sample \(i)")
                }
            }
        }
    }

    func testRoundTripDeemphasisThenPreemphasis() {
        // preemphasis(deemphasis(x)) should approximately equal x
        let signal = makeSine(duration: 0.05)
        let deemphasized = Preemphasis.deemphasis(signal: signal, coef: 0.97)
        let roundTrip = Preemphasis.preemphasis(signal: deemphasized, coef: 0.97)

        XCTAssertEqual(roundTrip.count, signal.count)

        signal.withUnsafeBufferPointer { inBuf in
            roundTrip.withUnsafeBufferPointer { outBuf in
                for i in 0..<signal.count {
                    XCTAssertEqual(outBuf[i], inBuf[i], accuracy: 1e-4,
                        "Round-trip should recover original at sample \(i)")
                }
            }
        }
    }

    func testPreemphasisPreservesSampleRate() {
        let signal = Signal(data: [1.0, 2.0, 3.0], sampleRate: 44100)
        let result = Preemphasis.preemphasis(signal: signal)
        XCTAssertEqual(result.sampleRate, 44100)
    }

    func testDeemphasisPreservesSampleRate() {
        let signal = Signal(data: [1.0, 2.0, 3.0], sampleRate: 44100)
        let result = Preemphasis.deemphasis(signal: signal)
        XCTAssertEqual(result.sampleRate, 44100)
    }
}
