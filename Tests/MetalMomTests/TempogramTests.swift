import XCTest
@testable import MetalMomCore

final class TempogramTests: XCTestCase {

    // MARK: - Click Track Helper

    /// Generate a click track at the given BPM.
    private func makeClickTrack(bpm: Float, sr: Int = 22050, duration: Double = 5.0) -> Signal {
        let n = Int(Double(sr) * duration)
        var samples = [Float](repeating: 0, count: n)

        let interval = 60.0 / Double(bpm)
        var t = 0.0
        while t < duration {
            let idx = Int(t * Double(sr))
            for j in 0..<256 where idx + j < n {
                samples[idx + j] = sin(Float(j) * 1000.0 * 2.0 * .pi / Float(sr)) * 0.9
            }
            t += interval
        }

        return Signal(data: samples, sampleRate: sr)
    }

    // MARK: - Autocorrelation Tempogram Tests

    func testAutocorrelationShape() {
        let signal = makeClickTrack(bpm: 120, duration: 3.0)
        let winLength = 384

        let result = Tempogram.autocorrelation(
            signal: signal,
            sr: 22050,
            winLength: winLength
        )

        XCTAssertEqual(result.shape.count, 2, "Tempogram should be 2D")
        XCTAssertEqual(result.shape[0], winLength, "First dimension should be winLength")
        XCTAssertGreaterThan(result.shape[1], 0, "Should have at least one frame")
    }

    func testAutocorrelationNonNegative() {
        let signal = makeClickTrack(bpm: 120, duration: 3.0)

        let result = Tempogram.autocorrelation(
            signal: signal,
            sr: 22050,
            winLength: 384
        )

        // ACF at lag 0 is always >= 0 (it's the energy).
        // Other lags can be negative, but lag 0 must be >= 0.
        let nFrames = result.shape[1]
        result.withUnsafeBufferPointer { buf in
            for frame in 0..<nFrames {
                // Lag 0 is at index [0 * nFrames + frame]
                XCTAssertGreaterThanOrEqual(buf[0 * nFrames + frame], 0,
                    "ACF at lag 0 should be non-negative")
            }
        }
    }

    func testAutocorrelationValuesFinite() {
        let signal = makeClickTrack(bpm: 120, duration: 2.0)

        let result = Tempogram.autocorrelation(
            signal: signal,
            sr: 22050,
            winLength: 128
        )

        result.withUnsafeBufferPointer { buf in
            for i in 0..<result.count {
                XCTAssert(buf[i].isFinite, "All tempogram values should be finite, got \(buf[i]) at index \(i)")
            }
        }
    }

    func testAutocorrelationSmallWinLength() {
        let signal = makeClickTrack(bpm: 120, duration: 2.0)
        let winLength = 64

        let result = Tempogram.autocorrelation(
            signal: signal,
            sr: 22050,
            winLength: winLength
        )

        XCTAssertEqual(result.shape[0], winLength)
        XCTAssertGreaterThan(result.shape[1], 0)
    }

    func testAutocorrelationPeriodicSignalHasStructure() {
        // A click track at 120 BPM should produce an autocorrelation
        // tempogram with non-trivial structure (not all zeros).
        let signal = makeClickTrack(bpm: 120, duration: 5.0)

        let result = Tempogram.autocorrelation(
            signal: signal,
            sr: 22050,
            winLength: 384
        )

        let nFrames = result.shape[1]
        guard nFrames > 0 else {
            XCTFail("Should have frames")
            return
        }

        // Check that the tempogram has non-zero energy for the click track.
        // At lag 0 the ACF equals the windowed energy, so for non-silent
        // frames this should be positive.
        var maxVal: Float = 0
        result.withUnsafeBufferPointer { buf in
            for i in 0..<result.count {
                let absVal = abs(buf[i])
                if absVal > maxVal { maxVal = absVal }
            }
        }

        XCTAssertGreaterThan(maxVal, 0,
            "Autocorrelation tempogram of click track should have non-zero values")

        // Additionally: across multiple frames, lag-0 should generally
        // dominate (since it represents total energy).
        var lag0Sum: Float = 0
        var lag10Sum: Float = 0
        result.withUnsafeBufferPointer { buf in
            for frame in 0..<nFrames {
                lag0Sum += abs(buf[0 * nFrames + frame])
                if 10 < result.shape[0] {
                    lag10Sum += abs(buf[10 * nFrames + frame])
                }
            }
        }
        XCTAssertGreaterThan(lag0Sum, lag10Sum,
            "Lag-0 ACF (energy) should typically be larger than higher lags")
    }

    // MARK: - Fourier Tempogram Tests

    func testFourierShape() {
        let signal = makeClickTrack(bpm: 120, duration: 3.0)
        let winLength = 384
        let expectedFreqs = winLength / 2 + 1

        let result = Tempogram.fourier(
            signal: signal,
            sr: 22050,
            winLength: winLength
        )

        XCTAssertEqual(result.shape.count, 2, "Fourier tempogram should be 2D")
        XCTAssertEqual(result.shape[0], expectedFreqs,
            "First dimension should be winLength/2+1")
        XCTAssertGreaterThan(result.shape[1], 0, "Should have at least one frame")
    }

    func testFourierNonNegative() {
        let signal = makeClickTrack(bpm: 120, duration: 3.0)

        let result = Tempogram.fourier(
            signal: signal,
            sr: 22050,
            winLength: 384
        )

        // Magnitude spectrum should be non-negative
        result.withUnsafeBufferPointer { buf in
            for i in 0..<result.count {
                XCTAssertGreaterThanOrEqual(buf[i], 0,
                    "Fourier tempogram magnitude should be non-negative, got \(buf[i]) at index \(i)")
            }
        }
    }

    func testFourierValuesFinite() {
        let signal = makeClickTrack(bpm: 120, duration: 2.0)

        let result = Tempogram.fourier(
            signal: signal,
            sr: 22050,
            winLength: 128
        )

        result.withUnsafeBufferPointer { buf in
            for i in 0..<result.count {
                XCTAssert(buf[i].isFinite, "All Fourier tempogram values should be finite")
            }
        }
    }

    func testFourierSmallWinLength() {
        let signal = makeClickTrack(bpm: 120, duration: 2.0)
        let winLength = 64
        let expectedFreqs = winLength / 2 + 1

        let result = Tempogram.fourier(
            signal: signal,
            sr: 22050,
            winLength: winLength
        )

        XCTAssertEqual(result.shape[0], expectedFreqs)
        XCTAssertGreaterThan(result.shape[1], 0)
    }

    // MARK: - Edge Cases

    func testSilenceAutocorrelation() {
        let sr = 22050
        let signal = Signal(data: [Float](repeating: 0, count: sr * 2), sampleRate: sr)

        let result = Tempogram.autocorrelation(signal: signal, sr: sr, winLength: 64)

        XCTAssertEqual(result.shape[0], 64)
        XCTAssertGreaterThan(result.shape[1], 0)

        // All values should be zero (or very near zero) for silence
        result.withUnsafeBufferPointer { buf in
            for i in 0..<result.count {
                XCTAssertEqual(buf[i], 0, accuracy: 1e-6,
                    "Tempogram of silence should be all zeros")
            }
        }
    }

    func testSilenceFourier() {
        let sr = 22050
        let signal = Signal(data: [Float](repeating: 0, count: sr * 2), sampleRate: sr)

        let result = Tempogram.fourier(signal: signal, sr: sr, winLength: 64)

        let expectedFreqs = 64 / 2 + 1
        XCTAssertEqual(result.shape[0], expectedFreqs)
        XCTAssertGreaterThan(result.shape[1], 0)

        // All values should be zero for silence
        result.withUnsafeBufferPointer { buf in
            for i in 0..<result.count {
                XCTAssertEqual(buf[i], 0, accuracy: 1e-6,
                    "Fourier tempogram of silence should be all zeros")
            }
        }
    }

    func testFrameCountMatchesBetweenVariants() {
        let signal = makeClickTrack(bpm: 120, duration: 3.0)
        let winLength = 128

        let acf = Tempogram.autocorrelation(signal: signal, sr: 22050, winLength: winLength)
        let ft = Tempogram.fourier(signal: signal, sr: 22050, winLength: winLength)

        // Both should have the same number of frames
        XCTAssertEqual(acf.shape[1], ft.shape[1],
            "ACF and Fourier tempograms should have same number of frames")
    }
}
