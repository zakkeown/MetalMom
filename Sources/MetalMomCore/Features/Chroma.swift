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
}
