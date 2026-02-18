import Foundation
import Accelerate

/// Harmonic-Percussive Source Separation (Fitzgerald 2010).
///
/// Separates an audio signal into harmonic and percussive components using
/// median filtering on the magnitude spectrogram followed by soft masking.
public enum HPSS {

    // MARK: - Public API

    /// Harmonic-percussive source separation.
    ///
    /// - Parameters:
    ///   - signal: Input audio signal (1-D).
    ///   - sr: Sample rate (optional, uses signal.sampleRate if nil).
    ///   - nFFT: FFT size. Default 2048.
    ///   - hopLength: Hop length. Default nFFT/4.
    ///   - winLength: Window length. Default nFFT.
    ///   - center: Whether to center-pad the signal. Default true.
    ///   - kernelSize: Median filter kernel size. Default 31.
    ///   - power: Exponent for the Wiener-like soft masks. Default 2.0.
    ///   - margin: Margin for mask separation. Default 1.0.
    /// - Returns: Tuple of (harmonic, percussive) 1-D Signals.
    public static func hpss(
        signal: Signal,
        sr: Int? = nil,
        nFFT: Int = 2048,
        hopLength: Int? = nil,
        winLength: Int? = nil,
        center: Bool = true,
        kernelSize: Int = 31,
        power: Float = 2.0,
        margin: Float = 1.0
    ) -> (harmonic: Signal, percussive: Signal) {
        let hop = hopLength ?? nFFT / 4
        let win = winLength ?? nFFT
        let sampleRate = sr ?? signal.sampleRate
        let signalLength = signal.shape[0]

        // Step 1: Compute complex STFT
        let complexSTFT = STFT.computeComplex(
            signal: signal,
            nFFT: nFFT,
            hopLength: hop,
            winLength: win,
            center: center
        )

        let nFreqs = complexSTFT.shape[0]
        let nFrames = complexSTFT.shape[1]

        guard nFreqs > 0 && nFrames > 0 else {
            let empty = Signal(data: [], sampleRate: sampleRate)
            return (harmonic: empty, percussive: empty)
        }

        // Step 2: Compute magnitude spectrogram from complex STFT
        // Complex STFT has interleaved real/imag, row-major [nFreqs, nFrames]
        let magnitude = computeMagnitude(complexSTFT: complexSTFT, nFreqs: nFreqs, nFrames: nFrames)

        // Step 3: Median filtering
        // Harmonic: median filter along time axis (horizontal, for each frequency row)
        let hMag = medianFilterHorizontal(magnitude, nFreqs: nFreqs, nFrames: nFrames, kernelSize: kernelSize)
        // Percussive: median filter along frequency axis (vertical, for each time column)
        let pMag = medianFilterVertical(magnitude, nFreqs: nFreqs, nFrames: nFrames, kernelSize: kernelSize)

        // Step 4: Compute soft masks
        let (hMask, pMask) = computeSoftMasks(
            hMag: hMag, pMag: pMag,
            count: nFreqs * nFrames,
            power: power, margin: margin
        )

        // Step 5: Apply masks to complex STFT
        let hComplex = applyMask(complexSTFT: complexSTFT, mask: hMask, nFreqs: nFreqs, nFrames: nFrames)
        let pComplex = applyMask(complexSTFT: complexSTFT, mask: pMask, nFreqs: nFreqs, nFrames: nFrames)

        // Step 6: Inverse STFT
        let harmonicSignal = STFT.inverse(
            complexSTFT: hComplex,
            hopLength: hop,
            winLength: win,
            center: center,
            length: signalLength
        )

        let percussiveSignal = STFT.inverse(
            complexSTFT: pComplex,
            hopLength: hop,
            winLength: win,
            center: center,
            length: signalLength
        )

        return (harmonic: harmonicSignal, percussive: percussiveSignal)
    }

    /// Return only the harmonic component.
    public static func harmonic(
        signal: Signal,
        sr: Int? = nil,
        nFFT: Int = 2048,
        hopLength: Int? = nil,
        winLength: Int? = nil,
        center: Bool = true,
        kernelSize: Int = 31,
        power: Float = 2.0,
        margin: Float = 1.0
    ) -> Signal {
        let result = hpss(
            signal: signal, sr: sr, nFFT: nFFT, hopLength: hopLength,
            winLength: winLength, center: center, kernelSize: kernelSize,
            power: power, margin: margin
        )
        return result.harmonic
    }

    /// Return only the percussive component.
    public static func percussive(
        signal: Signal,
        sr: Int? = nil,
        nFFT: Int = 2048,
        hopLength: Int? = nil,
        winLength: Int? = nil,
        center: Bool = true,
        kernelSize: Int = 31,
        power: Float = 2.0,
        margin: Float = 1.0
    ) -> Signal {
        let result = hpss(
            signal: signal, sr: sr, nFFT: nFFT, hopLength: hopLength,
            winLength: winLength, center: center, kernelSize: kernelSize,
            power: power, margin: margin
        )
        return result.percussive
    }

    // MARK: - Internal helpers

    /// Compute magnitude spectrogram from complex STFT.
    /// Complex STFT raw layout: row-major [nFreqs, nFrames], interleaved real/imag.
    /// Returns flat array of magnitudes in row-major [nFreqs, nFrames].
    private static func computeMagnitude(complexSTFT: Signal, nFreqs: Int, nFrames: Int) -> [Float] {
        let totalElements = nFreqs * nFrames
        var magnitude = [Float](repeating: 0, count: totalElements)

        complexSTFT.withUnsafeBufferPointer { rawBuf in
            for i in 0..<totalElements {
                let real = rawBuf[i * 2]
                let imag = rawBuf[i * 2 + 1]
                magnitude[i] = sqrtf(real * real + imag * imag)
            }
        }

        return magnitude
    }

    /// 1D median filter applied to each row (along time axis) for harmonic enhancement.
    /// Input/output are row-major [nFreqs, nFrames].
    private static func medianFilterHorizontal(_ data: [Float], nFreqs: Int, nFrames: Int, kernelSize: Int) -> [Float] {
        var result = [Float](repeating: 0, count: nFreqs * nFrames)
        let half = kernelSize / 2

        for freq in 0..<nFreqs {
            let rowOffset = freq * nFrames
            for frame in 0..<nFrames {
                let lo = max(0, frame - half)
                let hi = min(nFrames - 1, frame + half)
                let windowSize = hi - lo + 1

                // Extract window
                var window = [Float](repeating: 0, count: windowSize)
                for j in 0..<windowSize {
                    window[j] = data[rowOffset + lo + j]
                }
                window.sort()
                result[rowOffset + frame] = window[windowSize / 2]
            }
        }

        return result
    }

    /// 1D median filter applied to each column (along frequency axis) for percussive enhancement.
    /// Input/output are row-major [nFreqs, nFrames].
    private static func medianFilterVertical(_ data: [Float], nFreqs: Int, nFrames: Int, kernelSize: Int) -> [Float] {
        var result = [Float](repeating: 0, count: nFreqs * nFrames)
        let half = kernelSize / 2

        for frame in 0..<nFrames {
            for freq in 0..<nFreqs {
                let lo = max(0, freq - half)
                let hi = min(nFreqs - 1, freq + half)
                let windowSize = hi - lo + 1

                // Extract window (along frequency axis)
                var window = [Float](repeating: 0, count: windowSize)
                for j in 0..<windowSize {
                    window[j] = data[(lo + j) * nFrames + frame]
                }
                window.sort()
                result[freq * nFrames + frame] = window[windowSize / 2]
            }
        }

        return result
    }

    /// Compute Wiener-like soft masks from enhanced harmonic and percussive magnitudes.
    ///
    /// H_mask = (H_mag * margin)^power / ((H_mag * margin)^power + P_mag^power + eps)
    /// P_mask = (P_mag * margin)^power / ((H_mag)^power + (P_mag * margin)^power + eps)
    ///
    /// When margin == 1.0, this simplifies to:
    /// H_mask = H_mag^power / (H_mag^power + P_mag^power + eps)
    private static func computeSoftMasks(
        hMag: [Float], pMag: [Float],
        count: Int,
        power: Float, margin: Float
    ) -> (hMask: [Float], pMask: [Float]) {
        let eps: Float = 1e-10
        var hMask = [Float](repeating: 0, count: count)
        var pMask = [Float](repeating: 0, count: count)

        for i in 0..<count {
            let hVal = powf(hMag[i] * margin, power)
            let pVal = powf(pMag[i] * margin, power)
            let hRaw = powf(hMag[i], power)
            let pRaw = powf(pMag[i], power)

            // H_mask: uses margin-scaled H, but raw P
            let hDenom = hVal + pRaw + eps
            hMask[i] = hVal / hDenom

            // P_mask: uses raw H, but margin-scaled P
            let pDenom = hRaw + pVal + eps
            pMask[i] = pVal / pDenom
        }

        return (hMask, pMask)
    }

    /// Apply a real-valued mask to a complex STFT.
    /// Mask is row-major [nFreqs, nFrames]. Complex STFT has interleaved real/imag.
    /// Returns a new complex Signal with the mask applied.
    private static func applyMask(complexSTFT: Signal, mask: [Float], nFreqs: Int, nFrames: Int) -> Signal {
        let totalElements = nFreqs * nFrames
        let totalFloats = totalElements * 2

        let outPtr = UnsafeMutablePointer<Float>.allocate(capacity: totalFloats)

        complexSTFT.withUnsafeBufferPointer { rawBuf in
            for i in 0..<totalElements {
                let m = mask[i]
                outPtr[i * 2] = rawBuf[i * 2] * m       // real
                outPtr[i * 2 + 1] = rawBuf[i * 2 + 1] * m // imag
            }
        }

        let outBuffer = UnsafeMutableBufferPointer(start: outPtr, count: totalFloats)
        return Signal(taking: outBuffer, shape: [nFreqs, nFrames],
                      sampleRate: complexSTFT.sampleRate, dtype: .complex64)
    }
}
