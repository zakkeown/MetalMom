import Accelerate
import Foundation

/// STFT-based chroma features.
///
/// Chroma features represent pitch content independent of octave by mapping
/// the power spectrogram onto 12 (or more) pitch-class bins (C, C#, D, ..., B).
///
/// Pipeline: audio -> STFT -> power spectrogram -> chroma filterbank -> [normalize]
///
/// This implementation matches librosa's `chroma_stft` behaviour, including the
/// Gaussian-windowed chroma filterbank with octave weighting.
public enum Chroma {

    /// Compute STFT-based chroma features.
    ///
    /// Returns a `Signal` with shape `[nChroma, nFrames]`, row-major.
    ///
    /// - Parameters:
    ///   - signal: Input audio signal (1D).
    ///   - sr: Sample rate override. If `nil`, uses `signal.sampleRate`.
    ///   - nFFT: FFT window size. Default 2048.
    ///   - hopLength: Hop length in samples. Default `nFFT / 4`.
    ///   - winLength: Window length. Default `nFFT`.
    ///   - nChroma: Number of chroma bins. Default 12.
    ///   - center: If `true`, pad signal so frames are centred. Default `true`.
    ///   - norm: Normalization order per frame. `nil` = none, `2.0` = L2. Default `nil`.
    ///   - tuning: Tuning deviation from A440 in fractional chroma bins. Default 0.0.
    /// - Returns: Chroma `Signal` with shape `[nChroma, nFrames]`.
    public static func stft(
        signal: Signal,
        sr: Int? = nil,
        nFFT: Int = 2048,
        hopLength: Int? = nil,
        winLength: Int? = nil,
        nChroma: Int = 12,
        center: Bool = true,
        norm: Float? = nil,
        tuning: Float = 0.0
    ) -> Signal {
        let sampleRate = sr ?? signal.sampleRate

        // 1. Compute STFT magnitude spectrogram: shape [nFreqs, nFrames]
        let stftMag = STFT.compute(
            signal: signal,
            nFFT: nFFT,
            hopLength: hopLength,
            winLength: winLength,
            center: center
        )

        let nFreqs = stftMag.shape[0]
        let nFrames = stftMag.shape[1]
        let stftCount = stftMag.count

        // 2. Square for power spectrogram
        let poweredPtr = UnsafeMutablePointer<Float>.allocate(capacity: stftCount)
        defer { poweredPtr.deallocate() }

        stftMag.withUnsafeBufferPointer { src in
            vDSP_vsq(src.baseAddress!, 1, poweredPtr, 1, vDSP_Length(stftCount))
        }

        // 3. Get chroma filterbank: shape [nChroma, nFreqs]
        let chromaFB = chromaFilterbank(
            sr: sampleRate,
            nFFT: nFFT,
            nChroma: nChroma,
            tuning: tuning
        )

        // 4. Matrix multiply: chromaFB [nChroma, nFreqs] @ powered [nFreqs, nFrames] = [nChroma, nFrames]
        let outCount = nChroma * nFrames
        let outPtr = UnsafeMutablePointer<Float>.allocate(capacity: outCount)
        outPtr.initialize(repeating: 0, count: outCount)

        chromaFB.withUnsafeBufferPointer { fbBuf in
            vDSP_mmul(
                fbBuf.baseAddress!, 1,       // A: chromaFB [nChroma x nFreqs], row-major
                poweredPtr, 1,               // B: powered [nFreqs x nFrames], row-major
                outPtr, 1,                   // C: output [nChroma x nFrames], row-major
                vDSP_Length(nChroma),
                vDSP_Length(nFrames),
                vDSP_Length(nFreqs)
            )
        }

        // 5. Normalize if requested
        if let normOrder = norm {
            if normOrder == 2.0 {
                // L2 normalize each frame (column)
                for f in 0..<nFrames {
                    var l2Sq: Float = 0
                    for c in 0..<nChroma {
                        let val = outPtr[c * nFrames + f]
                        l2Sq += val * val
                    }
                    let l2 = sqrtf(l2Sq)
                    if l2 > 1e-10 {
                        for c in 0..<nChroma {
                            outPtr[c * nFrames + f] /= l2
                        }
                    }
                }
            } else if normOrder == 1.0 {
                // L1 normalize each frame (column)
                for f in 0..<nFrames {
                    var l1: Float = 0
                    for c in 0..<nChroma {
                        l1 += abs(outPtr[c * nFrames + f])
                    }
                    if l1 > 1e-10 {
                        for c in 0..<nChroma {
                            outPtr[c * nFrames + f] /= l1
                        }
                    }
                }
            } else if normOrder == Float.infinity {
                // Linf normalize each frame (column)
                for f in 0..<nFrames {
                    var maxVal: Float = 0
                    for c in 0..<nChroma {
                        maxVal = max(maxVal, abs(outPtr[c * nFrames + f]))
                    }
                    if maxVal > 1e-10 {
                        for c in 0..<nChroma {
                            outPtr[c * nFrames + f] /= maxVal
                        }
                    }
                }
            }
        }

        let outBuffer = UnsafeMutableBufferPointer(start: outPtr, count: outCount)
        return Signal(taking: outBuffer, shape: [nChroma, nFrames], sampleRate: sampleRate)
    }

    /// Build a chroma filterbank matrix matching librosa's `librosa.filters.chroma()`.
    ///
    /// Uses Gaussian-windowed pitch class profiles with octave weighting, matching
    /// librosa's default parameters (ctroct=5.0, octwidth=2, norm=2, base_c=true).
    ///
    /// Returns a `Signal` with shape `[nChroma, nFFT/2+1]` (row-major).
    ///
    /// - Parameters:
    ///   - sr: Audio sample rate in Hz.
    ///   - nFFT: FFT size (determines frequency resolution).
    ///   - nChroma: Number of chroma bins. Default 12.
    ///   - tuning: Tuning deviation from A440 in fractional chroma bins. Default 0.0.
    ///   - ctroct: Centre octave for the dominance window. Default 5.0.
    ///   - octwidth: Gaussian half-width for octave weighting. `nil` = flat. Default 2.0.
    ///   - fbNorm: Normalization order for filterbank columns. Default 2.0 (L2).
    ///   - baseC: If `true`, start at C. If `false`, start at A. Default `true`.
    /// - Returns: Chroma filterbank `Signal` of shape `[nChroma, nFFT/2+1]`.
    public static func chromaFilterbank(
        sr: Int,
        nFFT: Int,
        nChroma: Int = 12,
        tuning: Float = 0.0,
        ctroct: Float = 5.0,
        octwidth: Float? = 2.0,
        fbNorm: Float = 2.0,
        baseC: Bool = true
    ) -> Signal {
        let nFreqs = nFFT / 2 + 1

        // Allocate full nFFT-wide matrix (we'll truncate to nFreqs at the end)
        // wts[chroma, fftbin] stored row-major in nFFT columns
        var wts = [Float](repeating: 0, count: nChroma * nFFT)

        // Compute FFT bin frequencies (excluding DC): freq[k] = k * sr / nFFT for k=1..nFFT-1
        // Then convert to fractional chroma bins: frqbins = nChroma * hz_to_octs(freq, tuning, nChroma)
        let a440 = 440.0 * powf(2.0, tuning / Float(nChroma))

        var frqbins = [Float](repeating: 0, count: nFFT)
        // frqbins[0] will be set later (DC placeholder)
        for k in 1..<nFFT {
            let freq = Float(k) * Float(sr) / Float(nFFT)
            // hz_to_octs: log2(freq / (A440 / 16))
            let octs = log2f(freq / (a440 / 16.0))
            frqbins[k] = Float(nChroma) * octs
        }

        // Make up a value for DC bin: 1.5 octaves below bin 1
        frqbins[0] = frqbins[1] - 1.5 * Float(nChroma)

        // Compute bin widths
        var binwidthbins = [Float](repeating: 1.0, count: nFFT)
        for k in 0..<(nFFT - 1) {
            binwidthbins[k] = max(frqbins[k + 1] - frqbins[k], 1.0)
        }
        // Last bin width = 1.0 (already initialized)

        // Build Gaussian bumps
        let nChroma2 = roundf(Float(nChroma) / 2.0)

        for c in 0..<nChroma {
            for k in 0..<nFFT {
                // D = frqbins[k] - c, wrapped to [-nChroma/2, nChroma/2)
                var d = frqbins[k] - Float(c)
                d = fmodf(d + nChroma2 + 10.0 * Float(nChroma), Float(nChroma)) - nChroma2

                // Gaussian: exp(-0.5 * (2*D / binwidth)^2)
                let scaled = 2.0 * d / binwidthbins[k]
                wts[c * nFFT + k] = expf(-0.5 * scaled * scaled)
            }
        }

        // Normalize each column (FFT bin) with L2 norm
        if fbNorm == 2.0 {
            for k in 0..<nFFT {
                var colNorm: Float = 0
                for c in 0..<nChroma {
                    let val = wts[c * nFFT + k]
                    colNorm += val * val
                }
                colNorm = sqrtf(colNorm)
                if colNorm > 0 {
                    for c in 0..<nChroma {
                        wts[c * nFFT + k] /= colNorm
                    }
                }
            }
        } else if fbNorm == 1.0 {
            for k in 0..<nFFT {
                var colNorm: Float = 0
                for c in 0..<nChroma {
                    colNorm += abs(wts[c * nFFT + k])
                }
                if colNorm > 0 {
                    for c in 0..<nChroma {
                        wts[c * nFFT + k] /= colNorm
                    }
                }
            }
        }

        // Apply octave weighting (Gaussian centered on ctroct)
        if let ow = octwidth {
            for k in 0..<nFFT {
                let octave = frqbins[k] / Float(nChroma)
                let weight = expf(-0.5 * powf((octave - ctroct) / ow, 2.0))
                for c in 0..<nChroma {
                    wts[c * nFFT + k] *= weight
                }
            }
        }

        // Roll to start at C if base_c (rotate by -3 * (nChroma / 12) rows)
        if baseC {
            let shift = 3 * (nChroma / 12)
            if shift > 0 {
                // Roll rows: move row i to row (i - shift) % nChroma
                var rolled = [Float](repeating: 0, count: nChroma * nFFT)
                for c in 0..<nChroma {
                    let newC = (c + nChroma - shift) % nChroma
                    for k in 0..<nFFT {
                        rolled[newC * nFFT + k] = wts[c * nFFT + k]
                    }
                }
                wts = rolled
            }
        }

        // Truncate to nFreqs columns and copy to output
        var fb = [Float](repeating: 0, count: nChroma * nFreqs)
        for c in 0..<nChroma {
            for k in 0..<nFreqs {
                fb[c * nFreqs + k] = wts[c * nFFT + k]
            }
        }

        return Signal(data: fb, shape: [nChroma, nFreqs], sampleRate: sr)
    }

    // MARK: - CQT-based Chroma

    /// Compute CQT-based chroma features.
    ///
    /// Uses the Constant-Q Transform instead of the STFT as the underlying spectral
    /// representation, then folds CQT bins across octaves into chroma pitch classes.
    /// CQT-based chroma has better frequency resolution at low frequencies compared
    /// to STFT-based chroma.
    ///
    /// Returns a `Signal` with shape `[nChroma, nFrames]`, row-major.
    ///
    /// - Parameters:
    ///   - signal: Input audio signal (1D).
    ///   - sr: Sample rate override. If `nil`, uses `signal.sampleRate`.
    ///   - hopLength: Hop length in samples. Default auto-selected.
    ///   - fMin: Lowest CQT frequency in Hz. Default 32.7 (C1).
    ///   - binsPerOctave: CQT bins per octave. Default 36 (3x oversampling).
    ///   - nOctaves: Number of octaves to span. Default 7.
    ///   - nChroma: Number of chroma bins. Default 12.
    ///   - norm: Normalization order per frame. `nil` = none, `2.0` = L2. Default `nil`.
    /// - Returns: CQT chroma `Signal` with shape `[nChroma, nFrames]`.
    public static func cqt(
        signal: Signal,
        sr: Int? = nil,
        hopLength: Int? = nil,
        fMin: Float = 32.7,
        binsPerOctave: Int = 36,
        nOctaves: Int = 7,
        nChroma: Int = 12,
        norm: Float? = nil
    ) -> Signal {
        let sampleRate = sr ?? signal.sampleRate

        // Compute CQT magnitude: shape [nOctaves * binsPerOctave, nFrames]
        let fMax = fMin * powf(2.0, Float(nOctaves))
        let cqtMag = CQT.compute(
            signal: signal,
            sr: sampleRate,
            hopLength: hopLength,
            fMin: fMin,
            fMax: fMax,
            binsPerOctave: binsPerOctave
        )

        let actualBins = cqtMag.shape[0]
        let nFrames = cqtMag.shape[1]

        guard nFrames > 0 && actualBins > 0 else {
            return Signal(data: [], shape: [nChroma, 0], sampleRate: sampleRate)
        }

        return foldCQTToChroma(
            cqtMag: cqtMag,
            nBins: actualBins,
            nFrames: nFrames,
            binsPerOctave: binsPerOctave,
            nChroma: nChroma,
            norm: norm,
            sampleRate: sampleRate
        )
    }

    // MARK: - CENS Chroma

    /// Compute CENS (Chroma Energy Normalized Statistics) features.
    ///
    /// CENS applies a series of post-processing steps to chroma features:
    /// 1. L1 normalize each frame
    /// 2. Quantize values using logarithmic thresholds
    /// 3. Smooth with a Hann window
    /// 4. L2 normalize each frame
    ///
    /// CENS features are robust to variations in dynamics and timbre, making
    /// them suitable for audio matching and cover song identification.
    ///
    /// Returns a `Signal` with shape `[nChroma, nFrames]`, row-major.
    ///
    /// - Parameters:
    ///   - signal: Input audio signal (1D).
    ///   - sr: Sample rate override. If `nil`, uses `signal.sampleRate`.
    ///   - hopLength: Hop length in samples. Default auto-selected.
    ///   - fMin: Lowest CQT frequency in Hz. Default 32.7 (C1).
    ///   - binsPerOctave: CQT bins per octave. Default 36.
    ///   - nOctaves: Number of octaves to span. Default 7.
    ///   - nChroma: Number of chroma bins. Default 12.
    ///   - winLenSmooth: Smoothing window length. Default 41.
    /// - Returns: CENS chroma `Signal` with shape `[nChroma, nFrames]`.
    public static func cens(
        signal: Signal,
        sr: Int? = nil,
        hopLength: Int? = nil,
        fMin: Float = 32.7,
        binsPerOctave: Int = 36,
        nOctaves: Int = 7,
        nChroma: Int = 12,
        winLenSmooth: Int = 41
    ) -> Signal {
        let sampleRate = sr ?? signal.sampleRate

        // 1. Compute CQT chroma (unnormalized)
        let rawChroma = cqt(
            signal: signal,
            sr: sampleRate,
            hopLength: hopLength,
            fMin: fMin,
            binsPerOctave: binsPerOctave,
            nOctaves: nOctaves,
            nChroma: nChroma,
            norm: nil
        )

        let nFrames = rawChroma.shape[1]
        guard nFrames > 0 else {
            return Signal(data: [], shape: [nChroma, 0], sampleRate: sampleRate)
        }

        let totalCount = nChroma * nFrames
        let outPtr = UnsafeMutablePointer<Float>.allocate(capacity: totalCount)

        rawChroma.withUnsafeBufferPointer { src in
            outPtr.initialize(from: src.baseAddress!, count: totalCount)
        }

        // 2. L1 normalize each frame
        for f in 0..<nFrames {
            var l1: Float = 0
            for c in 0..<nChroma {
                l1 += abs(outPtr[c * nFrames + f])
            }
            if l1 > 1e-10 {
                for c in 0..<nChroma {
                    outPtr[c * nFrames + f] /= l1
                }
            }
        }

        // 3. Quantize using logarithmic thresholds: [0, 0.05, 0.1, 0.2, 0.4, 1.0] -> [0, 1, 2, 3, 4]
        let thresholds: [Float] = [0.05, 0.1, 0.2, 0.4]
        for i in 0..<totalCount {
            let val = outPtr[i]
            if val < thresholds[0] {
                outPtr[i] = 0
            } else if val < thresholds[1] {
                outPtr[i] = 1
            } else if val < thresholds[2] {
                outPtr[i] = 2
            } else if val < thresholds[3] {
                outPtr[i] = 3
            } else {
                outPtr[i] = 4
            }
        }

        // 4. Smooth with Hann window along the time axis
        if winLenSmooth > 1 && nFrames > 1 {
            // Build Hann window
            var hannWindow = [Float](repeating: 0, count: winLenSmooth)
            for i in 0..<winLenSmooth {
                hannWindow[i] = 0.5 * (1.0 - cosf(2.0 * .pi * Float(i) / Float(winLenSmooth - 1)))
            }
            // Normalize window
            var windowSum: Float = 0
            for i in 0..<winLenSmooth { windowSum += hannWindow[i] }
            if windowSum > 0 {
                for i in 0..<winLenSmooth { hannWindow[i] /= windowSum }
            }

            let halfWin = winLenSmooth / 2
            let tempPtr = UnsafeMutablePointer<Float>.allocate(capacity: totalCount)
            tempPtr.initialize(repeating: 0, count: totalCount)

            for c in 0..<nChroma {
                for f in 0..<nFrames {
                    var sum: Float = 0
                    for w in 0..<winLenSmooth {
                        let srcF = f + w - halfWin
                        // Clamp to valid range (reflect at boundaries)
                        let clampedF = max(0, min(nFrames - 1, srcF))
                        sum += outPtr[c * nFrames + clampedF] * hannWindow[w]
                    }
                    tempPtr[c * nFrames + f] = sum
                }
            }

            // Copy back
            for i in 0..<totalCount {
                outPtr[i] = tempPtr[i]
            }
            tempPtr.deallocate()
        }

        // 5. L2 normalize each frame
        for f in 0..<nFrames {
            var l2Sq: Float = 0
            for c in 0..<nChroma {
                let val = outPtr[c * nFrames + f]
                l2Sq += val * val
            }
            let l2 = sqrtf(l2Sq)
            if l2 > 1e-10 {
                for c in 0..<nChroma {
                    outPtr[c * nFrames + f] /= l2
                }
            }
        }

        let outBuffer = UnsafeMutableBufferPointer(start: outPtr, count: totalCount)
        return Signal(taking: outBuffer, shape: [nChroma, nFrames], sampleRate: sampleRate)
    }

    // MARK: - VQT-based Chroma

    /// Compute VQT-based chroma features.
    ///
    /// Uses the Variable-Q Transform instead of the standard CQT, then folds bins
    /// across octaves into chroma pitch classes. The VQT provides better time
    /// resolution at low frequencies through the gamma parameter.
    ///
    /// Returns a `Signal` with shape `[nChroma, nFrames]`, row-major.
    ///
    /// - Parameters:
    ///   - signal: Input audio signal (1D).
    ///   - sr: Sample rate override. If `nil`, uses `signal.sampleRate`.
    ///   - hopLength: Hop length in samples. Default auto-selected.
    ///   - fMin: Lowest VQT frequency in Hz. Default 32.7 (C1).
    ///   - binsPerOctave: VQT bins per octave. Default 36.
    ///   - nOctaves: Number of octaves to span. Default 7.
    ///   - gamma: VQT gamma parameter controlling bandwidth variation. Default 0 (= standard CQT).
    ///   - nChroma: Number of chroma bins. Default 12.
    ///   - norm: Normalization order per frame. `nil` = none, `2.0` = L2. Default `nil`.
    /// - Returns: VQT chroma `Signal` with shape `[nChroma, nFrames]`.
    public static func vqt(
        signal: Signal,
        sr: Int? = nil,
        hopLength: Int? = nil,
        fMin: Float = 32.7,
        binsPerOctave: Int = 36,
        nOctaves: Int = 7,
        gamma: Float = 0.0,
        nChroma: Int = 12,
        norm: Float? = nil
    ) -> Signal {
        let sampleRate = sr ?? signal.sampleRate

        // Compute VQT magnitude: shape [nOctaves * binsPerOctave, nFrames]
        let fMax = fMin * powf(2.0, Float(nOctaves))
        let vqtMag = CQT.vqt(
            signal: signal,
            sr: sampleRate,
            hopLength: hopLength,
            fMin: fMin,
            fMax: fMax,
            binsPerOctave: binsPerOctave,
            gamma: gamma
        )

        let actualBins = vqtMag.shape[0]
        let nFrames = vqtMag.shape[1]

        guard nFrames > 0 && actualBins > 0 else {
            return Signal(data: [], shape: [nChroma, 0], sampleRate: sampleRate)
        }

        return foldCQTToChroma(
            cqtMag: vqtMag,
            nBins: actualBins,
            nFrames: nFrames,
            binsPerOctave: binsPerOctave,
            nChroma: nChroma,
            norm: norm,
            sampleRate: sampleRate
        )
    }

    // MARK: - Deep Chroma

    /// Compute deep chroma features.
    ///
    /// This is a placeholder that uses CQT-based chroma as the foundation.
    /// A full deep chroma implementation would apply a learned neural network
    /// transformation to the CQT representation. In this version, the CQT chroma
    /// is computed with L2 normalization as a reasonable approximation.
    ///
    /// Returns a `Signal` with shape `[nChroma, nFrames]`, row-major.
    ///
    /// - Parameters:
    ///   - signal: Input audio signal (1D).
    ///   - sr: Sample rate override. If `nil`, uses `signal.sampleRate`.
    ///   - hopLength: Hop length in samples. Default auto-selected.
    ///   - nChroma: Number of chroma bins. Default 12.
    /// - Returns: Deep chroma `Signal` with shape `[nChroma, nFrames]`.
    public static func deep(
        signal: Signal,
        sr: Int? = nil,
        hopLength: Int? = nil,
        nChroma: Int = 12
    ) -> Signal {
        // Placeholder: use CQT chroma with L2 normalization
        return cqt(
            signal: signal,
            sr: sr,
            hopLength: hopLength,
            nChroma: nChroma,
            norm: 2.0
        )
    }

    // MARK: - Private Helpers

    /// Fold CQT/VQT bins across octaves into chroma pitch classes.
    ///
    /// Given a CQT magnitude spectrogram of shape [nBins, nFrames], maps each bin
    /// to a chroma pitch class and sums across octaves. The mapping accounts for
    /// the ratio between binsPerOctave and nChroma.
    private static func foldCQTToChroma(
        cqtMag: Signal,
        nBins: Int,
        nFrames: Int,
        binsPerOctave: Int,
        nChroma: Int,
        norm: Float?,
        sampleRate: Int
    ) -> Signal {
        let outCount = nChroma * nFrames
        let outPtr = UnsafeMutablePointer<Float>.allocate(capacity: outCount)
        outPtr.initialize(repeating: 0, count: outCount)

        // Ratio of CQT bins per chroma bin
        let binsPerChroma = binsPerOctave / nChroma

        cqtMag.withUnsafeBufferPointer { src in
            for bin in 0..<nBins {
                // Map CQT bin to chroma class
                // bin % binsPerOctave gives position within octave
                // Then divide by binsPerChroma to get chroma index
                let posInOctave = bin % binsPerOctave
                let chromaIdx: Int
                if binsPerChroma > 0 {
                    chromaIdx = (posInOctave / binsPerChroma) % nChroma
                } else {
                    chromaIdx = posInOctave % nChroma
                }

                // Sum this CQT bin's energy into the chroma bin
                for f in 0..<nFrames {
                    outPtr[chromaIdx * nFrames + f] += src[bin * nFrames + f]
                }
            }
        }

        // Normalize if requested
        if let normOrder = norm {
            if normOrder == 2.0 {
                for f in 0..<nFrames {
                    var l2Sq: Float = 0
                    for c in 0..<nChroma {
                        let val = outPtr[c * nFrames + f]
                        l2Sq += val * val
                    }
                    let l2 = sqrtf(l2Sq)
                    if l2 > 1e-10 {
                        for c in 0..<nChroma {
                            outPtr[c * nFrames + f] /= l2
                        }
                    }
                }
            } else if normOrder == 1.0 {
                for f in 0..<nFrames {
                    var l1: Float = 0
                    for c in 0..<nChroma {
                        l1 += abs(outPtr[c * nFrames + f])
                    }
                    if l1 > 1e-10 {
                        for c in 0..<nChroma {
                            outPtr[c * nFrames + f] /= l1
                        }
                    }
                }
            }
        }

        let outBuffer = UnsafeMutableBufferPointer(start: outPtr, count: outCount)
        return Signal(taking: outBuffer, shape: [nChroma, nFrames], sampleRate: sampleRate)
    }
}
