import Foundation
import Accelerate

/// Phase vocoder time-stretching and Griffin-Lim magnitude-to-audio reconstruction.
public enum PhaseVocoder {

    // MARK: - Phase Vocoder

    /// Time-stretch a complex STFT by a rate factor.
    ///
    /// Interpolates magnitude and advances phase to produce a new complex STFT
    /// with a different number of frames. The result can be converted back to
    /// audio via `STFT.inverse()`.
    ///
    /// - Parameters:
    ///   - complexSTFT: Complex spectrogram with shape [nFreqs, nFrames],
    ///     dtype `.complex64`. Row-major interleaved `[r,i,r,i,...]`.
    ///   - rate: Stretch rate. `rate > 1` speeds up (fewer output frames),
    ///     `rate < 1` slows down (more output frames).
    ///   - hopLength: Hop length used to produce the STFT.
    ///     If `nil`, defaults to `(nFreqs - 1) * 2 / 4`.
    /// - Returns: Complex spectrogram with shape [nFreqs, newNFrames], dtype `.complex64`.
    public static func phaseVocoder(
        complexSTFT: Signal,
        rate: Float,
        hopLength: Int? = nil
    ) -> Signal {
        precondition(complexSTFT.dtype == .complex64,
                     "phaseVocoder requires complex64 input")
        precondition(complexSTFT.shape.count == 2,
                     "phaseVocoder requires 2D input [nFreqs, nFrames]")
        precondition(rate > 0, "rate must be positive")

        let nFreqs = complexSTFT.shape[0]
        let nFramesIn = complexSTFT.shape[1]
        let nFFT = (nFreqs - 1) * 2
        let hop = hopLength ?? nFFT / 4

        guard nFreqs > 0 && nFramesIn > 0 else {
            return Signal(complexData: [], shape: [nFreqs, 0],
                          sampleRate: complexSTFT.sampleRate)
        }

        let nFramesOut = Int(ceilf(Float(nFramesIn) / rate))
        guard nFramesOut > 0 else {
            return Signal(complexData: [], shape: [nFreqs, 0],
                          sampleRate: complexSTFT.sampleRate)
        }

        // Allocate output complex STFT
        let totalComplex = nFreqs * nFramesOut
        let totalFloats = totalComplex * 2
        let outPtr = UnsafeMutablePointer<Float>.allocate(capacity: totalFloats)
        outPtr.initialize(repeating: 0, count: totalFloats)

        // Phase accumulator for each frequency bin
        var phaseAccum = [Float](repeating: 0, count: nFreqs)

        // Expected phase advance per hop for each bin k: 2*pi*k*hop/nFFT
        let dphiExpected = (0..<nFreqs).map { k in
            2.0 * Float.pi * Float(k) * Float(hop) / Float(nFFT)
        }

        complexSTFT.withUnsafeBufferPointer { rawBuf in
            // Row-major [nFreqs, nFramesIn], interleaved:
            // (freq, frame) -> real at rawBuf[2*(freq*nFramesIn + frame)]
            //                  imag at rawBuf[2*(freq*nFramesIn + frame) + 1]

            for tOut in 0..<nFramesOut {
                let tIn = Float(tOut) * rate

                let frameLo = min(Int(floorf(tIn)), nFramesIn - 1)
                let frameHi = min(frameLo + 1, nFramesIn - 1)
                let alpha = tIn - Float(frameLo)

                for k in 0..<nFreqs {
                    let idxLo = 2 * (k * nFramesIn + frameLo)
                    let idxHi = 2 * (k * nFramesIn + frameHi)

                    let realLo = rawBuf[idxLo]
                    let imagLo = rawBuf[idxLo + 1]
                    let realHi = rawBuf[idxHi]
                    let imagHi = rawBuf[idxHi + 1]

                    // Interpolate magnitude
                    let magLo = sqrtf(realLo * realLo + imagLo * imagLo)
                    let magHi = sqrtf(realHi * realHi + imagHi * imagHi)
                    let magOut = (1.0 - alpha) * magLo + alpha * magHi

                    // Phase advancement
                    if tOut == 0 {
                        phaseAccum[k] = atan2f(imagLo, realLo)
                    } else {
                        let phaseLo = atan2f(imagLo, realLo)
                        let phaseHi = atan2f(imagHi, realHi)
                        var dphi = phaseHi - phaseLo - dphiExpected[k]
                        dphi = dphi - 2.0 * Float.pi * roundf(dphi / (2.0 * Float.pi))
                        phaseAccum[k] += dphiExpected[k] + dphi
                    }

                    let outIdx = 2 * (k * nFramesOut + tOut)
                    outPtr[outIdx] = magOut * cosf(phaseAccum[k])
                    outPtr[outIdx + 1] = magOut * sinf(phaseAccum[k])
                }
            }
        }

        let outBuffer = UnsafeMutableBufferPointer(start: outPtr, count: totalFloats)
        return Signal(taking: outBuffer, shape: [nFreqs, nFramesOut],
                      sampleRate: complexSTFT.sampleRate, dtype: .complex64)
    }

    // MARK: - Griffin-Lim

    /// Reconstruct audio from a magnitude spectrogram using the Griffin-Lim iterative algorithm.
    ///
    /// Given a magnitude-only STFT, iteratively estimates phase by alternating
    /// between iSTFT and STFT, replacing the magnitude at each step to converge
    /// on a consistent phase estimate.
    ///
    /// - Parameters:
    ///   - magnitude: Magnitude spectrogram, shape [nFreqs, nFrames], dtype `.float32`.
    ///   - nIter: Number of Griffin-Lim iterations. Default 32.
    ///   - hopLength: Hop length. If `nil`, defaults to `nFFT / 4`.
    ///   - winLength: Window length. If `nil`, defaults to `nFFT`.
    ///   - center: If `true`, assumes centered STFT. Default `true`.
    ///   - length: If specified, truncates or pads output to this length.
    /// - Returns: Reconstructed 1-D real-valued audio Signal.
    public static func griffinLim(
        magnitude: Signal,
        nIter: Int = 32,
        hopLength: Int? = nil,
        winLength: Int? = nil,
        center: Bool = true,
        length: Int? = nil
    ) -> Signal {
        precondition(magnitude.dtype == .float32,
                     "griffinLim requires float32 magnitude input")
        precondition(magnitude.shape.count == 2,
                     "griffinLim requires 2D input [nFreqs, nFrames]")

        let nFreqs = magnitude.shape[0]
        let nFrames = magnitude.shape[1]
        let nFFT = (nFreqs - 1) * 2
        let hop = hopLength ?? nFFT / 4
        let win = winLength ?? nFFT
        let totalComplex = nFreqs * nFrames

        guard nFreqs > 0 && nFrames > 0 else {
            return Signal(data: [], sampleRate: magnitude.sampleRate)
        }

        // Compute the expected audio length from the STFT shape.
        // This ensures the round-trip iSTFT -> STFT produces consistent frame counts.
        let expectedPaddedLength = nFFT + (nFrames - 1) * hop
        let expectedAudioLength: Int
        if center {
            expectedAudioLength = expectedPaddedLength - nFFT  // nFFT/2 trimmed from each end
        } else {
            expectedAudioLength = expectedPaddedLength
        }

        // Step 1: Initialize with random phase
        let totalFloats = totalComplex * 2
        let complexPtr = UnsafeMutablePointer<Float>.allocate(capacity: totalFloats)

        magnitude.withUnsafeBufferPointer { magBuf in
            for i in 0..<totalComplex {
                let mag = magBuf[i]
                let phase = Float.random(in: -Float.pi...Float.pi)
                complexPtr[i * 2] = mag * cosf(phase)
                complexPtr[i * 2 + 1] = mag * sinf(phase)
            }
        }

        var complexBuffer = UnsafeMutableBufferPointer(start: complexPtr, count: totalFloats)
        var sComplex = Signal(taking: complexBuffer, shape: [nFreqs, nFrames],
                              sampleRate: magnitude.sampleRate, dtype: .complex64)

        // Step 2: Iterate
        for _ in 0..<nIter {
            // a. Reconstruct audio from current complex STFT.
            //    Use the expected audio length to ensure consistent round-trip frame counts.
            let audio = STFT.inverse(
                complexSTFT: sComplex,
                hopLength: hop,
                winLength: win,
                center: center,
                length: expectedAudioLength
            )

            // b. Recompute complex STFT from reconstructed audio
            let newComplex = STFT.computeComplex(
                signal: audio,
                nFFT: nFFT,
                hopLength: hop,
                winLength: win,
                center: center
            )

            // c. Replace magnitude, keep new phase
            let newNFreqs = newComplex.shape[0]
            let newNFrames = newComplex.shape[1]
            let useNFreqs = min(nFreqs, newNFreqs)
            let useNFrames = min(nFrames, newNFrames)

            let nextPtr = UnsafeMutablePointer<Float>.allocate(capacity: nFreqs * nFrames * 2)
            nextPtr.initialize(repeating: 0, count: nFreqs * nFrames * 2)

            magnitude.withUnsafeBufferPointer { magBuf in
                newComplex.withUnsafeBufferPointer { newBuf in
                    for freq in 0..<useNFreqs {
                        for frame in 0..<useNFrames {
                            let magIdx = freq * nFrames + frame
                            let newIdx = 2 * (freq * newNFrames + frame)
                            let outIdx = 2 * (freq * nFrames + frame)

                            let newReal = newBuf[newIdx]
                            let newImag = newBuf[newIdx + 1]
                            let newMag = sqrtf(newReal * newReal + newImag * newImag)
                            let targetMag = magBuf[magIdx]

                            if newMag > 1e-10 {
                                let scale = targetMag / newMag
                                nextPtr[outIdx] = newReal * scale
                                nextPtr[outIdx + 1] = newImag * scale
                            } else {
                                nextPtr[outIdx] = 0
                                nextPtr[outIdx + 1] = 0
                            }
                        }
                    }
                }
            }

            complexBuffer = UnsafeMutableBufferPointer(start: nextPtr,
                                                        count: nFreqs * nFrames * 2)
            sComplex = Signal(taking: complexBuffer, shape: [nFreqs, nFrames],
                              sampleRate: magnitude.sampleRate, dtype: .complex64)
        }

        // Step 3: Final iSTFT with the user-requested output length
        return STFT.inverse(
            complexSTFT: sComplex,
            hopLength: hop,
            winLength: win,
            center: center,
            length: length
        )
    }

    // MARK: - Griffin-Lim CQT

    /// Griffin-Lim for CQT magnitude spectrograms.
    ///
    /// Reconstructs audio from a CQT magnitude spectrogram by iterating between
    /// iSTFT and CQT, replacing CQT magnitudes at each step.
    ///
    /// This is an approximate reconstruction since CQT is not a perfect transform
    /// with a trivial inverse. The approach converts CQT magnitudes back to
    /// approximate STFT magnitudes using a pseudo-inverse of the CQT filterbank,
    /// then applies standard Griffin-Lim on the STFT domain.
    ///
    /// - Parameters:
    ///   - magnitude: CQT magnitude spectrogram, shape [nBins, nFrames].
    ///   - sr: Sample rate.
    ///   - nIter: Number of iterations. Default 32.
    ///   - hopLength: Hop length. If `nil`, auto-selected.
    ///   - fMin: Lowest CQT frequency. Default 32.7 Hz.
    ///   - binsPerOctave: CQT bins per octave. Default 12.
    /// - Returns: Reconstructed 1-D audio Signal.
    public static func griffinLimCQT(
        magnitude: Signal,
        sr: Int,
        nIter: Int = 32,
        hopLength: Int? = nil,
        fMin: Float = 32.7,
        binsPerOctave: Int = 12
    ) -> Signal {
        precondition(magnitude.dtype == .float32,
                     "griffinLimCQT requires float32 input")
        precondition(magnitude.shape.count == 2,
                     "griffinLimCQT requires 2D input [nBins, nFrames]")

        let nBins = magnitude.shape[0]
        let nFrames = magnitude.shape[1]

        guard nBins > 0 && nFrames > 0 else {
            return Signal(data: [], sampleRate: sr)
        }

        // Compute an appropriate FFT size for the CQT parameters
        let Q = 1.0 / (powf(2.0, 1.0 / Float(binsPerOctave)) - 1.0)
        let minWindow = Int(ceilf(Q * Float(sr) / fMin))
        var fftSize = 1
        while fftSize < minWindow {
            fftSize *= 2
        }
        fftSize = max(2048, min(fftSize, 65536))

        let hop = hopLength ?? fftSize / 4
        let nFreqs = fftSize / 2 + 1

        // Build CQT filterbank: [nBins, nFreqs]
        let cqtFB = CQT.cqtFilterbank(
            sr: sr,
            nFFT: fftSize,
            fMin: fMin,
            nBins: nBins,
            binsPerOctave: binsPerOctave
        )

        // Pseudo-inverse: transpose of filterbank, row-normalized
        // cqtFB shape [nBins, nFreqs], pseudoInv shape [nFreqs, nBins]
        let pseudoInvPtr = UnsafeMutablePointer<Float>.allocate(capacity: nFreqs * nBins)
        defer { pseudoInvPtr.deallocate() }

        cqtFB.withUnsafeBufferPointer { fbBuf in
            // Transpose: pseudoInv[f, b] = fb[b, f] = fbBuf[b * nFreqs + f]
            for f in 0..<nFreqs {
                for b in 0..<nBins {
                    pseudoInvPtr[f * nBins + b] = fbBuf[b * nFreqs + f]
                }
            }
        }

        // Normalize each row of pseudoInv so rows sum to 1
        for f in 0..<nFreqs {
            var rowSum: Float = 0
            for b in 0..<nBins {
                rowSum += pseudoInvPtr[f * nBins + b]
            }
            if rowSum > 1e-10 {
                for b in 0..<nBins {
                    pseudoInvPtr[f * nBins + b] /= rowSum
                }
            }
        }

        // Reconstruct approximate STFT magnitude: pseudoInv [nFreqs, nBins] @ cqtMag [nBins, nFrames]
        let stftMagCount = nFreqs * nFrames
        let stftMagPtr = UnsafeMutablePointer<Float>.allocate(capacity: stftMagCount)
        stftMagPtr.initialize(repeating: 0, count: stftMagCount)

        magnitude.withUnsafeBufferPointer { magBuf in
            vDSP_mmul(
                pseudoInvPtr, 1,
                magBuf.baseAddress!, 1,
                stftMagPtr, 1,
                vDSP_Length(nFreqs),
                vDSP_Length(nFrames),
                vDSP_Length(nBins)
            )
        }

        // Clamp negative values
        for i in 0..<stftMagCount {
            if stftMagPtr[i] < 0 { stftMagPtr[i] = 0 }
        }

        let stftMagBuffer = UnsafeMutableBufferPointer(start: stftMagPtr, count: stftMagCount)
        let stftMag = Signal(taking: stftMagBuffer, shape: [nFreqs, nFrames], sampleRate: sr)

        // Use standard Griffin-Lim on the STFT magnitude
        return griffinLim(
            magnitude: stftMag,
            nIter: nIter,
            hopLength: hop,
            winLength: fftSize,
            center: true,
            length: nil
        )
    }
}
