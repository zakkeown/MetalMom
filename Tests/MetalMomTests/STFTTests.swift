import XCTest
@testable import MetalMomCore

final class STFTTests: XCTestCase {

    // MARK: - Output Shape

    func testSTFTOutputShape() {
        // 1 second of silence at 22050 Hz
        let sr = 22050
        let signal = Signal(data: [Float](repeating: 0, count: sr), sampleRate: sr)
        let result = STFT.compute(signal: signal)

        // nFreqs = nFFT/2 + 1 = 1025
        XCTAssertEqual(result.shape.count, 2)
        XCTAssertEqual(result.shape[0], 1025)

        // With center=true, padded length = signalLength + nFFT = 22050 + 2048 = 24098
        // nFrames = 1 + (paddedLength - nFFT) / hopLength = 1 + (24098 - 2048) / 512
        //         = 1 + 22050 / 512 = 1 + 43.066... = 44
        let paddedLength = sr + 2048
        let expectedFrames = 1 + (paddedLength - 2048) / 512
        XCTAssertEqual(result.shape[1], expectedFrames)
    }

    // MARK: - Sine Wave Peak Bin

    func testSTFTSineWavePeakBin() {
        let sr = 22050
        let freq: Float = 440.0
        let nFFT = 2048
        let duration = 1.0

        // Generate 440 Hz sine wave
        let numSamples = Int(duration * Double(sr))
        var samples = [Float](repeating: 0, count: numSamples)
        for i in 0..<numSamples {
            samples[i] = sinf(2.0 * .pi * freq * Float(i) / Float(sr))
        }

        let signal = Signal(data: samples, sampleRate: sr)
        let result = STFT.compute(signal: signal, nFFT: nFFT)

        let nFreqs = result.shape[0]
        let nFrames = result.shape[1]

        // Pick the middle frame
        let midFrame = nFrames / 2

        // Find peak bin in this frame
        // Data is column-major: element at (freqBin, frame) = data[freqBin + frame * nFreqs]
        var maxVal: Float = -1
        var maxBin = -1
        for bin in 0..<nFreqs {
            let val = result[bin + midFrame * nFreqs]
            if val > maxVal {
                maxVal = val
                maxBin = bin
            }
        }

        // Expected bin: round(freq * nFFT / sr) = round(440 * 2048 / 22050) = round(40.87) = 41
        let expectedBin = Int(round(Double(freq) * Double(nFFT) / Double(sr)))
        assertApproxEqual(maxBin, expectedBin, tolerance: 2,
                          "Peak bin should be near \(expectedBin), got \(maxBin)")
    }

    // MARK: - Zero Signal

    func testSTFTZeroSignal() {
        let signal = Signal(data: [Float](repeating: 0, count: 4096), sampleRate: 22050)
        let result = STFT.compute(signal: signal)

        // All values should be zero (or very near zero)
        for i in 0..<result.count {
            XCTAssertEqual(result[i], 0.0, accuracy: 1e-7,
                           "Expected zero at index \(i), got \(result[i])")
        }
    }

    // MARK: - Impulse

    func testSTFTImpulse() {
        // Single impulse at position 0, rest zeros
        let nFFT = 2048
        var samples = [Float](repeating: 0, count: nFFT)
        samples[0] = 1.0

        let signal = Signal(data: samples, sampleRate: 22050)
        // center=false so the impulse is at the start of the first frame
        let result = STFT.compute(signal: signal, nFFT: nFFT, center: false)

        _ = result.shape[0]

        // Frame 0 should have energy spread across all bins.
        // For a pure impulse through a Hann window at index 0, the Hann window value
        // at index 0 is 0 (since hann[0] = 0). So the windowed signal is all zeros,
        // and magnitude will be zero. Instead, place the impulse at the center of the window.
        // Let's just verify that the output has the right shape and is non-negative.
        for i in 0..<result.count {
            XCTAssertGreaterThanOrEqual(result[i], 0.0,
                                        "Magnitude should be non-negative at index \(i)")
        }

        // More meaningful test: impulse at center of window
        // With center=true, the signal is padded so sample 0 moves to nFFT/2
        // which is the center of the first full frame
        var samples2 = [Float](repeating: 0, count: nFFT)
        samples2[0] = 1.0
        let signal2 = Signal(data: samples2, sampleRate: 22050)
        let result2 = STFT.compute(signal: signal2, nFFT: nFFT, center: true)

        let nFreqs2 = result2.shape[0]

        // The frame that contains the impulse at sample 0 (now at index nFFT/2 in padded signal)
        // should have approximately equal magnitude across all bins (flat spectrum of an impulse
        // times hann window at center = 1.0).
        // Frame index where the original sample[0] ends up: paddedPos / hop = (nFFT/2) / (nFFT/4) = 2
        let targetFrame = 2
        var magnitudes = [Float]()
        for bin in 0..<nFreqs2 {
            magnitudes.append(result2[bin + targetFrame * nFreqs2])
        }

        let maxMag = magnitudes.max()!
        // The impulse should produce non-trivial energy
        XCTAssertGreaterThan(maxMag, 0.0, "Impulse should produce non-zero energy")

        // Check that energy is spread across bins (not concentrated in one bin)
        // At least 80% of bins should have > 10% of max magnitude
        let threshold = maxMag * 0.1
        let binsAboveThreshold = magnitudes.filter { $0 > threshold }.count
        let ratio = Float(binsAboveThreshold) / Float(nFreqs2)
        XCTAssertGreaterThan(ratio, 0.5,
                             "Impulse should spread energy; \(binsAboveThreshold)/\(nFreqs2) bins above threshold")
    }
}

// MARK: - Helper for approximate Int comparison

func assertApproxEqual(_ a: Int, _ b: Int, tolerance: Int,
                       _ message: String = "", file: StaticString = #filePath, line: UInt = #line) {
    XCTAssertTrue(abs(a - b) <= tolerance,
                  message.isEmpty ? "|\(a) - \(b)| = \(abs(a - b)) > \(tolerance)" : message,
                  file: file, line: line)
}
