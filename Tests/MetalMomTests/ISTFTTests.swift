import XCTest
@testable import MetalMomCore

final class ISTFTTests: XCTestCase {

    // MARK: - Sine Wave Round-Trip

    func testSineWaveRoundTrip() {
        // Generate a pure 440 Hz sine wave.
        // Use a length that aligns with STFT framing so all samples are reconstructable.
        let sr = 22050
        let nFFT = 2048
        let hopLength = nFFT / 4
        // Choose numSamples so that the padded length (numSamples + nFFT) minus nFFT
        // is evenly divisible by hopLength: numSamples = k * hopLength for some k
        let numSamples = 44 * hopLength  // 22528
        let freq: Float = 440.0
        let original = (0..<numSamples).map { sinf(2.0 * .pi * freq * Float($0) / Float(sr)) }
        let signal = Signal(data: original, sampleRate: sr)

        // Forward complex STFT
        let complexSTFT = STFT.computeComplex(signal: signal, nFFT: nFFT, hopLength: hopLength)

        // Inverse STFT
        let reconstructed = STFT.inverse(complexSTFT: complexSTFT, hopLength: hopLength, length: numSamples)

        XCTAssertEqual(reconstructed.dtype, .float32)
        XCTAssertEqual(reconstructed.count, numSamples)

        // Compare: reconstructed should match original within tolerance
        for i in 0..<numSamples {
            XCTAssertEqual(reconstructed[i], original[i], accuracy: 1e-3,
                "Sine round-trip mismatch at sample \(i): got \(reconstructed[i]), expected \(original[i])")
        }
    }

    // MARK: - Random Signal Round-Trip

    func testRandomSignalRoundTrip() {
        let sr = 22050
        let numSamples = 8192
        let nFFT = 1024
        let hopLength = nFFT / 4

        // Generate random signal
        var rng = SystemRandomNumberGenerator()
        let original = (0..<numSamples).map { _ -> Float in
            Float.random(in: -1.0...1.0, using: &rng)
        }
        let signal = Signal(data: original, sampleRate: sr)

        // Forward complex STFT -> Inverse STFT
        let complexSTFT = STFT.computeComplex(signal: signal, nFFT: nFFT, hopLength: hopLength)
        let reconstructed = STFT.inverse(complexSTFT: complexSTFT, hopLength: hopLength, length: numSamples)

        XCTAssertEqual(reconstructed.count, numSamples)

        for i in 0..<numSamples {
            XCTAssertEqual(reconstructed[i], original[i], accuracy: 1e-3,
                "Random signal round-trip mismatch at sample \(i): got \(reconstructed[i]), expected \(original[i])")
        }
    }

    // MARK: - Output Length Parameter

    func testOutputLengthParameter() {
        let sr = 22050
        let numSamples = 4096
        let nFFT = 1024
        let hopLength = nFFT / 4

        let original = (0..<numSamples).map { sinf(2.0 * .pi * 440.0 * Float($0) / Float(sr)) }
        let signal = Signal(data: original, sampleRate: sr)

        let complexSTFT = STFT.computeComplex(signal: signal, nFFT: nFFT, hopLength: hopLength)

        // Request exact length
        let reconstructed = STFT.inverse(complexSTFT: complexSTFT, hopLength: hopLength, length: numSamples)
        XCTAssertEqual(reconstructed.count, numSamples,
            "Output length should match requested length")

        // Request shorter length
        let shortLen = 2000
        let short = STFT.inverse(complexSTFT: complexSTFT, hopLength: hopLength, length: shortLen)
        XCTAssertEqual(short.count, shortLen,
            "Output length should match requested shorter length")
    }

    // MARK: - No Center Mode

    func testNoCenterRoundTrip() {
        let sr = 22050
        let nFFT = 1024
        let hopLength = nFFT / 4
        // With center=false, the signal must be at least nFFT long
        let numSamples = 4096

        let original = (0..<numSamples).map { sinf(2.0 * .pi * 440.0 * Float($0) / Float(sr)) }
        let signal = Signal(data: original, sampleRate: sr)

        let complexSTFT = STFT.computeComplex(signal: signal, nFFT: nFFT, hopLength: hopLength, center: false)
        let reconstructed = STFT.inverse(complexSTFT: complexSTFT, hopLength: hopLength, center: false, length: numSamples)

        XCTAssertEqual(reconstructed.count, numSamples)

        // With center=false, the edges are affected by windowing, so check interior samples
        let margin = nFFT
        for i in margin..<(numSamples - margin) {
            XCTAssertEqual(reconstructed[i], original[i], accuracy: 1e-3,
                "No-center round-trip mismatch at sample \(i)")
        }
    }

    // MARK: - Small nFFT sizes

    func testSmallNFFTRoundTrip() {
        let sr = 22050
        let numSamples = 2048
        let nFFT = 256
        let hopLength = nFFT / 4

        let original = (0..<numSamples).map { sinf(2.0 * .pi * 440.0 * Float($0) / Float(sr)) }
        let signal = Signal(data: original, sampleRate: sr)

        let complexSTFT = STFT.computeComplex(signal: signal, nFFT: nFFT, hopLength: hopLength)
        let reconstructed = STFT.inverse(complexSTFT: complexSTFT, hopLength: hopLength, length: numSamples)

        XCTAssertEqual(reconstructed.count, numSamples)

        for i in 0..<numSamples {
            XCTAssertEqual(reconstructed[i], original[i], accuracy: 1e-3,
                "Small nFFT round-trip mismatch at sample \(i)")
        }
    }
}
