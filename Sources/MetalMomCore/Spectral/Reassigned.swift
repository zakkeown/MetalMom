import Foundation
import Accelerate

/// Reassigned spectrogram computation.
///
/// Corrects the smearing of the standard STFT by reassigning each energy point
/// to a more accurate time-frequency location.  Matches librosa's
/// `reassigned_spectrogram()`.
public enum ReassignedSpectrogram {

    /// Compute the reassigned spectrogram.
    ///
    /// Returns three Signals, each with shape `[nFreqs, nFrames]`:
    /// - `magnitude`: magnitude of the standard windowed STFT
    /// - `frequencies`: reassigned frequency coordinates (Hz)
    /// - `times`: reassigned time coordinates (seconds)
    ///
    /// - Parameters:
    ///   - signal: Input audio signal (1-D).
    ///   - sr: Sample rate.  If `nil`, taken from `signal.sampleRate`.
    ///   - nFFT: FFT size. Must be a power of two.  Default 2048.
    ///   - hopLength: Hop length in samples. Default `nFFT / 4`.
    ///   - winLength: Window length. Default `nFFT`.
    ///   - center: Whether to center-pad the signal. Default `true`.
    public static func compute(
        signal: Signal,
        sr: Int? = nil,
        nFFT: Int = 2048,
        hopLength: Int? = nil,
        winLength: Int? = nil,
        center: Bool = true
    ) -> (magnitude: Signal, frequencies: Signal, times: Signal) {
        let sampleRate = sr ?? signal.sampleRate
        let hop = hopLength ?? nFFT / 4
        let win = winLength ?? nFFT
        let nFreqs = nFFT / 2 + 1

        // --- 1. Build the three windows ---
        // Standard Hann window (h)
        let hWin = Windows.hann(length: win, periodic: true)

        // Time-derivative window via central finite difference: dh[n] = (h[n+1] - h[n-1]) / 2
        var dhWin = [Float](repeating: 0, count: win)
        for n in 0..<win {
            let prev = n > 0 ? hWin[n - 1] : 0.0 as Float
            let next = n < win - 1 ? hWin[n + 1] : 0.0 as Float
            dhWin[n] = (next - prev) / 2.0
        }

        // Time-ramped window: th[n] = (n - win/2) * h[n]
        // The centering offset ensures the ramp is centered on the window.
        var thWin = [Float](repeating: 0, count: win)
        let halfWin = Float(win) / 2.0
        for n in 0..<win {
            thWin[n] = (Float(n) - halfWin) * hWin[n]
        }

        // --- 2. Pad windows to nFFT if needed ---
        func padWindow(_ w: [Float]) -> [Float] {
            if w.count == nFFT { return w }
            let padBefore = (nFFT - w.count) / 2
            let padAfter = nFFT - w.count - padBefore
            return [Float](repeating: 0, count: padBefore) + w + [Float](repeating: 0, count: padAfter)
        }
        let fullH  = padWindow(hWin)
        let fullDH = padWindow(dhWin)
        let fullTH = padWindow(thWin)

        // --- 3. Pad the signal (center padding) ---
        let padded: [Float]
        if center {
            let padAmount = nFFT / 2
            padded = [Float](repeating: 0, count: padAmount)
                   + signal.withUnsafeBufferPointer { Array($0) }
                   + [Float](repeating: 0, count: padAmount)
        } else {
            padded = signal.withUnsafeBufferPointer { Array($0) }
        }

        guard padded.count >= nFFT else {
            // Signal too short -- return empty results
            let empty = Signal(data: [], shape: [nFreqs, 0], sampleRate: sampleRate)
            return (empty,
                    Signal(data: [], shape: [nFreqs, 0], sampleRate: sampleRate),
                    Signal(data: [], shape: [nFreqs, 0], sampleRate: sampleRate))
        }
        let nFrames = 1 + (padded.count - nFFT) / hop

        // --- 4. Set up vDSP FFT ---
        let log2n = vDSP_Length(log2(Double(nFFT)))
        guard let fftSetup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else {
            fatalError("Failed to create FFT setup for nFFT=\(nFFT)")
        }
        defer { vDSP_destroy_fftsetup(fftSetup) }

        let halfN = nFFT / 2

        // --- 5. Allocate column-major buffers for complex STFTs ---
        //  Each stores nFreqs complex pairs per frame in column-major order.
        //  Layout: real[frame * nFreqs + freq], imag[frame * nFreqs + freq]
        let totalComplex = nFreqs * nFrames
        let hReal  = UnsafeMutablePointer<Float>.allocate(capacity: totalComplex)
        let hImag  = UnsafeMutablePointer<Float>.allocate(capacity: totalComplex)
        let dhReal = UnsafeMutablePointer<Float>.allocate(capacity: totalComplex)
        let dhImag = UnsafeMutablePointer<Float>.allocate(capacity: totalComplex)
        let thReal = UnsafeMutablePointer<Float>.allocate(capacity: totalComplex)
        let thImag = UnsafeMutablePointer<Float>.allocate(capacity: totalComplex)

        hReal.initialize(repeating: 0, count: totalComplex)
        hImag.initialize(repeating: 0, count: totalComplex)
        dhReal.initialize(repeating: 0, count: totalComplex)
        dhImag.initialize(repeating: 0, count: totalComplex)
        thReal.initialize(repeating: 0, count: totalComplex)
        thImag.initialize(repeating: 0, count: totalComplex)

        defer {
            hReal.deallocate(); hImag.deallocate()
            dhReal.deallocate(); dhImag.deallocate()
            thReal.deallocate(); thImag.deallocate()
        }

        // Temp buffers for per-frame FFT
        let fftRealBuf = UnsafeMutablePointer<Float>.allocate(capacity: halfN)
        let fftImagBuf = UnsafeMutablePointer<Float>.allocate(capacity: halfN)
        let windowedFrame = UnsafeMutablePointer<Float>.allocate(capacity: nFFT)
        defer {
            fftRealBuf.deallocate()
            fftImagBuf.deallocate()
            windowedFrame.deallocate()
        }

        // Helper: compute one complex STFT column (one frame) using the given window,
        // storing results into (outReal, outImag) at column offset.
        func computeFrame(
            paddedBuf: UnsafeBufferPointer<Float>,
            window: UnsafeBufferPointer<Float>,
            frame: Int,
            outReal: UnsafeMutablePointer<Float>,
            outImag: UnsafeMutablePointer<Float>
        ) {
            let start = frame * hop

            // Apply window
            vDSP_vmul(paddedBuf.baseAddress! + start, 1,
                      window.baseAddress!, 1,
                      windowedFrame, 1,
                      vDSP_Length(nFFT))

            // Pack into split complex
            var splitComplex = DSPSplitComplex(realp: fftRealBuf, imagp: fftImagBuf)
            windowedFrame.withMemoryRebound(to: DSPComplex.self, capacity: halfN) { complexPtr in
                vDSP_ctoz(complexPtr, 2, &splitComplex, 1, vDSP_Length(halfN))
            }

            // Forward FFT
            vDSP_fft_zrip(fftSetup, &splitComplex, 1, log2n, FFTDirection(kFFTDirection_Forward))

            // Scale by 0.5 (vDSP factor-of-2 normalization)
            var scale: Float = 0.5
            vDSP_vsmul(fftRealBuf, 1, &scale, fftRealBuf, 1, vDSP_Length(halfN))
            vDSP_vsmul(fftImagBuf, 1, &scale, fftImagBuf, 1, vDSP_Length(halfN))

            let colBase = frame * nFreqs

            // DC bin: real = realp[0], imag = 0
            outReal[colBase + 0] = fftRealBuf[0]
            outImag[colBase + 0] = 0.0

            // Bins 1..<halfN
            for k in 1..<halfN {
                outReal[colBase + k] = fftRealBuf[k]
                outImag[colBase + k] = fftImagBuf[k]
            }

            // Nyquist bin: real = imagp[0] (packed format), imag = 0
            outReal[colBase + nFreqs - 1] = fftImagBuf[0]
            outImag[colBase + nFreqs - 1] = 0.0
        }

        // --- 6. Compute three STFTs frame by frame ---
        padded.withUnsafeBufferPointer { paddedBuf in
            fullH.withUnsafeBufferPointer { hBuf in
                fullDH.withUnsafeBufferPointer { dhBuf in
                    fullTH.withUnsafeBufferPointer { thBuf in
                        for frame in 0..<nFrames {
                            computeFrame(paddedBuf: paddedBuf, window: hBuf,
                                         frame: frame, outReal: hReal, outImag: hImag)
                            computeFrame(paddedBuf: paddedBuf, window: dhBuf,
                                         frame: frame, outReal: dhReal, outImag: dhImag)
                            computeFrame(paddedBuf: paddedBuf, window: thBuf,
                                         frame: frame, outReal: thReal, outImag: thImag)
                        }
                    }
                }
            }
        }

        // --- 7. Compute reassigned coordinates ---
        //
        // For each (freq_k, frame_t):
        //   magnitude     = |S_h|
        //   S_h * conj(S_h) = |S_h|^2   (real)
        //   S_dh * conj(S_h) = Re + Im*j
        //   S_th * conj(S_h) = Re + Im*j
        //
        // Reassigned frequency:
        //   f_reassigned = f_k - Im(S_dh * conj(S_h)) / |S_h|^2 * sr / (2*pi)
        //
        // Reassigned time:
        //   t_reassigned = t_frame + Re(S_th * conj(S_h)) / |S_h|^2 / sr

        let magOut  = UnsafeMutablePointer<Float>.allocate(capacity: totalComplex)
        let freqOut = UnsafeMutablePointer<Float>.allocate(capacity: totalComplex)
        let timeOut = UnsafeMutablePointer<Float>.allocate(capacity: totalComplex)

        let threshold: Float = 1e-10  // avoid division by near-zero
        let twoPi: Float = 2.0 * .pi
        let srF = Float(sampleRate)

        for frame in 0..<nFrames {
            let colBase = frame * nFreqs
            let tFrame = Float(frame) * Float(hop) / srF  // nominal frame time

            for k in 0..<nFreqs {
                let idx = colBase + k
                let hR = hReal[idx]
                let hI = hImag[idx]
                let magSq = hR * hR + hI * hI
                let mag = sqrtf(magSq)

                magOut[idx] = mag

                if magSq < threshold {
                    // Below threshold: keep nominal coordinates
                    freqOut[idx] = Float(k) * srF / Float(nFFT)
                    timeOut[idx] = tFrame
                } else {
                    let dhR = dhReal[idx]
                    let dhI = dhImag[idx]
                    let thR = thReal[idx]
                    let thI = thImag[idx]

                    // S_dh * conj(S_h) = (dhR + dhI*j) * (hR - hI*j)
                    //   real = dhR*hR + dhI*hI
                    //   imag = dhI*hR - dhR*hI
                    let crossDH_imag = dhI * hR - dhR * hI

                    // S_th * conj(S_h) = (thR + thI*j) * (hR - hI*j)
                    //   real = thR*hR + thI*hI
                    let crossTH_real = thR * hR + thI * hI

                    // Reassigned frequency
                    let fNominal = Float(k) * srF / Float(nFFT)
                    let fReassigned = fNominal - (crossDH_imag / magSq) * srF / twoPi
                    freqOut[idx] = fReassigned

                    // Reassigned time
                    let tReassigned = tFrame + (crossTH_real / magSq) / srF
                    timeOut[idx] = tReassigned
                }
            }
        }

        // --- 8. Transpose from column-major [nFreqs, nFrames] to row-major ---
        func transposeToRowMajor(_ src: UnsafeMutablePointer<Float>) -> Signal {
            let dst = UnsafeMutablePointer<Float>.allocate(capacity: totalComplex)
            // src is column-major: src[frame * nFreqs + freq]
            // dst is row-major:    dst[freq * nFrames + frame]
            vDSP_mtrans(src, 1, dst, 1, vDSP_Length(nFreqs), vDSP_Length(nFrames))
            let buf = UnsafeMutableBufferPointer(start: dst, count: totalComplex)
            return Signal(taking: buf, shape: [nFreqs, nFrames], sampleRate: sampleRate)
        }

        let magSignal  = transposeToRowMajor(magOut)
        let freqSignal = transposeToRowMajor(freqOut)
        let timeSignal = transposeToRowMajor(timeOut)

        magOut.deallocate()
        freqOut.deallocate()
        timeOut.deallocate()

        return (magnitude: magSignal, frequencies: freqSignal, times: timeSignal)
    }
}
