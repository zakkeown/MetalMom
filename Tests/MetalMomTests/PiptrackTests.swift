import XCTest
@testable import MetalMomCore

final class PiptrackTests: XCTestCase {

    // MARK: - Helpers

    private func makeSineSignal(frequency: Float = 440.0, amplitude: Float = 1.0,
                                 sr: Int = 22050, duration: Float = 1.0) -> Signal {
        return SignalGen.tone(frequency: frequency, sr: sr,
                              duration: Double(duration), phi: 0)
    }

    // MARK: - Output Shape

    func testOutputShape() {
        let sr = 22050
        let nFFT = 2048
        let signal = makeSineSignal(frequency: 440.0, sr: sr, duration: 1.0)

        let result = Piptrack.piptrack(
            signal: signal,
            sr: sr,
            nFFT: nFFT,
            fMin: 150.0,
            fMax: 4000.0,
            threshold: 0.1,
            center: true
        )

        let nFreqs = nFFT / 2 + 1
        XCTAssertEqual(result.shape.count, 2, "Output should be 2-D")
        XCTAssertEqual(result.shape[0], 2 * nFreqs, "First dim should be 2 * nFreqs")
        XCTAssertGreaterThan(result.shape[1], 0, "Should have at least one frame")
    }

    // MARK: - 440 Hz Sine Peak Detection

    func testSineWave440HasPeakNear440() {
        let sr = 22050
        let nFFT = 2048
        let signal = makeSineSignal(frequency: 440.0, sr: sr, duration: 1.0)

        let result = Piptrack.piptrack(
            signal: signal,
            sr: sr,
            nFFT: nFFT,
            fMin: 100.0,
            fMax: 8000.0,
            threshold: 0.1,
            center: true
        )

        let nFreqs = nFFT / 2 + 1
        let nFrames = result.shape[1]

        // For each interior frame, find the strongest pitch
        var foundNear440 = 0
        let startFrame = 2
        let endFrame = max(startFrame, nFrames - 2)

        for frame in startFrame..<endFrame {
            var maxMag: Float = 0
            var bestPitch: Float = 0

            for bin in 0..<nFreqs {
                let pitch = result[bin * nFrames + frame]         // pitches row
                let mag = result[(nFreqs + bin) * nFrames + frame] // magnitudes row

                if mag > maxMag {
                    maxMag = mag
                    bestPitch = pitch
                }
            }

            if bestPitch > 0 && abs(bestPitch - 440.0) < 20.0 {
                foundNear440 += 1
            }
        }

        let interiorCount = endFrame - startFrame
        guard interiorCount > 0 else { return }

        let ratio = Float(foundNear440) / Float(interiorCount)
        XCTAssertGreaterThan(ratio, 0.7,
                             "At least 70% of interior frames should have peak near 440 Hz")
    }

    // MARK: - Pitch Values in Range

    func testPitchValuesInRange() {
        let sr = 22050
        let fMin: Float = 150.0
        let fMax: Float = 4000.0
        let nFFT = 2048
        let signal = makeSineSignal(frequency: 440.0, sr: sr, duration: 0.5)

        let result = Piptrack.piptrack(
            signal: signal,
            sr: sr,
            nFFT: nFFT,
            fMin: fMin,
            fMax: fMax,
            threshold: 0.1,
            center: true
        )

        let nFreqs = nFFT / 2 + 1
        let nFrames = result.shape[1]

        // Check that non-zero pitches are within [fMin, fMax]
        for bin in 0..<nFreqs {
            for frame in 0..<nFrames {
                let pitch = result[bin * nFrames + frame]
                if pitch != 0 {
                    XCTAssertGreaterThanOrEqual(pitch, fMin,
                        "Non-zero pitch should be >= fMin")
                    XCTAssertLessThanOrEqual(pitch, fMax,
                        "Non-zero pitch should be <= fMax")
                }
            }
        }
    }

    // MARK: - Magnitudes Non-Negative

    func testMagnitudesNonNegative() {
        let sr = 22050
        let nFFT = 2048
        let signal = makeSineSignal(frequency: 440.0, sr: sr, duration: 0.5)

        let result = Piptrack.piptrack(
            signal: signal,
            sr: sr,
            nFFT: nFFT,
            fMin: 150.0,
            fMax: 4000.0,
            threshold: 0.1,
            center: true
        )

        let nFreqs = nFFT / 2 + 1
        let nFrames = result.shape[1]

        for bin in 0..<nFreqs {
            for frame in 0..<nFrames {
                let mag = result[(nFreqs + bin) * nFrames + frame]
                XCTAssertGreaterThanOrEqual(mag, 0,
                    "Magnitudes should be non-negative")
            }
        }
    }

    // MARK: - Silence Gives All Zeros

    func testSilenceAllZeros() {
        let sr = 22050
        let nFFT = 2048
        let silence = Signal(data: [Float](repeating: 0, count: sr), sampleRate: sr)

        let result = Piptrack.piptrack(
            signal: silence,
            sr: sr,
            nFFT: nFFT,
            fMin: 150.0,
            fMax: 4000.0,
            threshold: 0.1,
            center: true
        )

        let totalElements = result.count
        for i in 0..<totalElements {
            XCTAssertEqual(result[i], 0, "Silence should produce all zeros")
        }
    }

    // MARK: - Custom Parameters

    func testCustomHopAndWinLength() {
        let sr = 22050
        let nFFT = 1024
        let hop = 256
        let win = 1024
        let signal = makeSineSignal(frequency: 440.0, sr: sr, duration: 0.5)

        let result = Piptrack.piptrack(
            signal: signal,
            sr: sr,
            nFFT: nFFT,
            hopLength: hop,
            winLength: win,
            fMin: 100.0,
            fMax: 8000.0,
            threshold: 0.05,
            center: true
        )

        let nFreqs = nFFT / 2 + 1
        XCTAssertEqual(result.shape[0], 2 * nFreqs)
        XCTAssertGreaterThan(result.shape[1], 0)
    }
}
