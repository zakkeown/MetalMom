import Foundation
import Accelerate

/// Time stretching via phase vocoder.
///
/// Changes the duration of an audio signal without changing its pitch.
/// Uses the phase vocoder algorithm: compute complex STFT, interpolate
/// magnitude and phase across frames, then reconstruct with iSTFT.
public enum TimeStretch {

    /// Time-stretch audio via phase vocoder.
    ///
    /// - Parameters:
    ///   - signal: Input audio signal (1-D).
    ///   - rate: Stretch rate. rate > 1 speeds up (shorter output),
    ///           rate < 1 slows down (longer output).
    ///   - sr: Sample rate (optional, uses signal.sampleRate if nil).
    ///   - nFFT: FFT size. Default 2048.
    ///   - hopLength: Hop length. Default nFFT/4.
    /// - Returns: Time-stretched 1-D Signal.
    public static func timeStretch(
        signal: Signal,
        rate: Float,
        sr: Int? = nil,
        nFFT: Int = 2048,
        hopLength: Int? = nil
    ) -> Signal {
        precondition(rate > 0, "rate must be positive")

        let hop = hopLength ?? nFFT / 4
        let sampleRate = sr ?? signal.sampleRate

        // Step 1: Compute complex STFT
        let complexSTFT = STFT.computeComplex(
            signal: signal,
            nFFT: nFFT,
            hopLength: hop,
            center: true
        )

        let nFreqs = complexSTFT.shape[0]
        let nFramesIn = complexSTFT.shape[1]

        guard nFreqs > 0 && nFramesIn > 0 else {
            return Signal(data: [], sampleRate: sampleRate)
        }

        // Step 2: Phase vocoder
        let nFramesOut = Int(ceilf(Float(nFramesIn) / rate))

        guard nFramesOut > 0 else {
            return Signal(data: [], sampleRate: sampleRate)
        }

        // Allocate output complex STFT: interleaved [nFreqs, nFramesOut]
        let totalComplex = nFreqs * nFramesOut
        let totalFloats = totalComplex * 2
        let outPtr = UnsafeMutablePointer<Float>.allocate(capacity: totalFloats)
        outPtr.initialize(repeating: 0, count: totalFloats)

        // Phase accumulator for each frequency bin
        var phaseAccum = [Float](repeating: 0, count: nFreqs)

        // Expected phase advance per hop for each frequency bin
        // dphi_expected[k] = 2 * pi * k * hop / nFFT
        let dphiExpected = (0..<nFreqs).map { k in
            2.0 * Float.pi * Float(k) * Float(hop) / Float(nFFT)
        }

        complexSTFT.withUnsafeBufferPointer { rawBuf in
            // rawBuf is row-major [nFreqs, nFramesIn], interleaved real/imag
            // Element (freq, frame): real at rawBuf[2 * (freq * nFramesIn + frame)]
            //                        imag at rawBuf[2 * (freq * nFramesIn + frame) + 1]

            for tOut in 0..<nFramesOut {
                let tIn = Float(tOut) * rate

                // Clamp frame indices
                let frameLo = min(Int(floorf(tIn)), nFramesIn - 1)
                let frameHi = min(frameLo + 1, nFramesIn - 1)
                let alpha = tIn - Float(frameLo)

                for k in 0..<nFreqs {
                    // Get complex values for lo and hi frames
                    let idxLo = 2 * (k * nFramesIn + frameLo)
                    let idxHi = 2 * (k * nFramesIn + frameHi)

                    let realLo = rawBuf[idxLo]
                    let imagLo = rawBuf[idxLo + 1]
                    let realHi = rawBuf[idxHi]
                    let imagHi = rawBuf[idxHi + 1]

                    // Compute magnitudes
                    let magLo = sqrtf(realLo * realLo + imagLo * imagLo)
                    let magHi = sqrtf(realHi * realHi + imagHi * imagHi)

                    // Interpolate magnitude
                    let magOut = (1.0 - alpha) * magLo + alpha * magHi

                    // Phase advancement
                    if tOut == 0 {
                        // First frame: use the phase of the first input frame
                        phaseAccum[k] = atan2f(imagLo, realLo)
                    } else {
                        // Phase difference between consecutive input frames
                        let phaseLo = atan2f(imagLo, realLo)
                        let phaseHi = atan2f(imagHi, realHi)

                        // Instantaneous frequency deviation
                        var dphi = phaseHi - phaseLo - dphiExpected[k]

                        // Wrap to [-pi, pi]
                        dphi = dphi - 2.0 * Float.pi * roundf(dphi / (2.0 * Float.pi))

                        // Accumulate phase
                        phaseAccum[k] += dphiExpected[k] + dphi
                    }

                    // Reconstruct complex value
                    let outIdx = 2 * (k * nFramesOut + tOut)
                    outPtr[outIdx] = magOut * cosf(phaseAccum[k])
                    outPtr[outIdx + 1] = magOut * sinf(phaseAccum[k])
                }
            }
        }

        let outBuffer = UnsafeMutableBufferPointer(start: outPtr, count: totalFloats)
        let stretchedSTFT = Signal(taking: outBuffer, shape: [nFreqs, nFramesOut],
                                    sampleRate: sampleRate, dtype: .complex64)

        // Step 3: Inverse STFT to reconstruct audio
        let result = STFT.inverse(
            complexSTFT: stretchedSTFT,
            hopLength: hop,
            center: true
        )

        return result
    }
}
