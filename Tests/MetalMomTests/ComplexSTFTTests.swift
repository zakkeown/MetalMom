import XCTest
@testable import MetalMomCore

final class ComplexSTFTTests: XCTestCase {

    // MARK: - Output Shape

    func testComplexSTFTOutputShape() {
        let sr = 22050
        let signal = Signal(data: [Float](repeating: 0, count: sr), sampleRate: sr)
        let result = STFT.computeComplex(signal: signal)

        XCTAssertEqual(result.dtype, .complex64)
        XCTAssertEqual(result.shape.count, 2)
        XCTAssertEqual(result.shape[0], 1025)  // nFreqs = 2048/2 + 1

        // Same frame count as the real STFT
        let paddedLength = sr + 2048
        let expectedFrames = 1 + (paddedLength - 2048) / 512
        XCTAssertEqual(result.shape[1], expectedFrames)
    }

    // MARK: - Phase Preservation (cosine has dominant real component)

    func testComplexSTFTPhasePreservation() {
        // A pure cosine at an exact bin frequency should have dominant real component.
        // Use sr and nFFT such that the frequency lands exactly on a DFT bin center.
        let nFFT = 2048
        let sr = nFFT   // 1 Hz per bin, so integer frequencies = exact bins
        let bin = 100
        let freq = Float(bin)  // Exactly bin 100
        let numSamples = sr * 2  // 2 seconds for plenty of frames

        let cosine = (0..<numSamples).map { cosf(2.0 * .pi * freq * Float($0) / Float(sr)) }
        let signal = Signal(data: cosine, sampleRate: sr)

        let result = STFT.computeComplex(signal: signal, nFFT: nFFT)

        let nFrames = result.shape[1]
        let midFrame = nFrames / 2

        let real = result.realPart(at: bin * nFrames + midFrame)
        let imag = result.imagPart(at: bin * nFrames + midFrame)

        // For an exact-bin cosine, the real component should dominate over imaginary.
        // The Hann window still causes some spectral leakage but at the exact bin center
        // the imaginary part should be much smaller than the real part.
        XCTAssertGreaterThan(abs(real), abs(imag) * 2,
            "Cosine at exact bin frequency should have dominant real component at peak bin. " +
            "real=\(real), imag=\(imag)")
    }

    // MARK: - Magnitude parity with real STFT

    func testComplexSTFTMagnitudeMatchesRealSTFT() {
        // The magnitude of the complex STFT should match the real STFT output
        let sr = 22050
        let numSamples = sr
        let t = (0..<numSamples).map { Float($0) / Float(sr) }
        let sine = t.map { sinf(2.0 * .pi * 440.0 * $0) }
        let signal = Signal(data: sine, sampleRate: sr)

        let magResult = STFT.compute(signal: signal)
        let complexResult = STFT.computeComplex(signal: signal)

        // Shapes should match (complex shape describes complex elements, not raw floats)
        XCTAssertEqual(magResult.shape, complexResult.shape)

        let nFreqs = magResult.shape[0]
        let nFrames = magResult.shape[1]

        for freq in stride(from: 0, to: nFreqs, by: 100) {
            for frame in stride(from: 0, to: nFrames, by: 10) {
                let idx = freq * nFrames + frame
                let real = complexResult.realPart(at: idx)
                let imag = complexResult.imagPart(at: idx)
                let mag = sqrtf(real * real + imag * imag)
                let expected = magResult[idx]
                // Tolerance allows for vDSP (CPU) vs MPSGraph (GPU) FFT precision differences
                XCTAssertEqual(mag, expected, accuracy: 1e-3,
                    "Magnitude mismatch at freq=\(freq) frame=\(frame): computed=\(mag) expected=\(expected)")
            }
        }
    }

    // MARK: - Zero signal produces zero complex output

    func testComplexSTFTZeroSignal() {
        let signal = Signal(data: [Float](repeating: 0, count: 4096), sampleRate: 22050)
        let result = STFT.computeComplex(signal: signal)

        XCTAssertEqual(result.dtype, .complex64)
        for i in 0..<result.elementCount {
            XCTAssertEqual(result.realPart(at: i), 0.0, accuracy: 1e-7,
                           "Real part should be zero at element \(i)")
            XCTAssertEqual(result.imagPart(at: i), 0.0, accuracy: 1e-7,
                           "Imaginary part should be zero at element \(i)")
        }
    }

    // MARK: - DC and Nyquist bins have zero imaginary part

    func testComplexSTFTDCAndNyquistBinsAreReal() {
        // DC and Nyquist bins should have zero imaginary component
        let sr = 22050
        let t = (0..<sr).map { Float($0) / Float(sr) }
        let sine = t.map { sinf(2.0 * .pi * 440.0 * $0) }
        let signal = Signal(data: sine, sampleRate: sr)

        let result = STFT.computeComplex(signal: signal)
        let nFreqs = result.shape[0]
        let nFrames = result.shape[1]

        for frame in 0..<nFrames {
            // DC bin (freq index 0)
            let dcImag = result.imagPart(at: 0 * nFrames + frame)
            XCTAssertEqual(dcImag, 0.0, accuracy: 1e-7,
                           "DC bin imaginary should be zero at frame \(frame)")

            // Nyquist bin (freq index nFreqs-1)
            let nyquistImag = result.imagPart(at: (nFreqs - 1) * nFrames + frame)
            XCTAssertEqual(nyquistImag, 0.0, accuracy: 1e-7,
                           "Nyquist bin imaginary should be zero at frame \(frame)")
        }
    }
}
