import Accelerate
import Foundation

/// Spectral descriptor features: centroid, bandwidth, contrast, rolloff, flatness.
///
/// All functions operate on a magnitude spectrogram (not raw audio).
/// They compute per-frame features, returning a 1D Signal of shape [nFrames]
/// (or [nBands+1, nFrames] for contrast).
///
/// Input spectrograms should have shape [nFreqs, nFrames] (row-major),
/// as produced by `STFT.compute`.
public enum SpectralDescriptors {

    // MARK: - Centroid

    /// Spectral centroid: weighted mean of frequencies.
    ///
    /// `centroid[t] = sum(freq[k] * S[k,t]) / sum(S[k,t])`
    ///
    /// - Parameters:
    ///   - spectrogram: Magnitude spectrogram with shape [nFreqs, nFrames].
    ///   - sr: Sample rate in Hz.
    ///   - nFFT: FFT size used to compute the spectrogram.
    /// - Returns: Signal with shape [nFrames], centroid in Hz per frame.
    public static func centroid(spectrogram: Signal, sr: Int, nFFT: Int) -> Signal {
        let nFreqs = spectrogram.shape[0]
        let nFrames = spectrogram.shape[1]

        // Build frequency vector: freq[k] = k * sr / nFFT
        var freqs = [Float](repeating: 0, count: nFreqs)
        let srFloat = Float(sr)
        let nFFTFloat = Float(nFFT)
        for k in 0..<nFreqs {
            freqs[k] = Float(k) * srFloat / nFFTFloat
        }

        let outPtr = UnsafeMutablePointer<Float>.allocate(capacity: nFrames)
        outPtr.initialize(repeating: 0, count: nFrames)

        spectrogram.withUnsafeBufferPointer { specBuf in
            for f in 0..<nFrames {
                // Column f: specBuf[k * nFrames + f] for k in 0..<nFreqs
                var weightedSum: Float = 0
                var totalSum: Float = 0
                for k in 0..<nFreqs {
                    let val = specBuf[k * nFrames + f]
                    weightedSum += freqs[k] * val
                    totalSum += val
                }
                outPtr[f] = totalSum > 0 ? weightedSum / totalSum : 0
            }
        }

        let outBuffer = UnsafeMutableBufferPointer(start: outPtr, count: nFrames)
        return Signal(taking: outBuffer, shape: [nFrames], sampleRate: sr)
    }

    // MARK: - Bandwidth

    /// Spectral bandwidth: weighted standard deviation of frequencies around centroid.
    ///
    /// `bw[t] = (sum(S[k,t] * |freq[k] - centroid[t]|^p) / sum(S[k,t]))^(1/p)`
    ///
    /// - Parameters:
    ///   - spectrogram: Magnitude spectrogram with shape [nFreqs, nFrames].
    ///   - sr: Sample rate in Hz.
    ///   - nFFT: FFT size used to compute the spectrogram.
    ///   - p: Power for the p-th moment. Default 2.0 (standard deviation).
    /// - Returns: Signal with shape [nFrames], bandwidth in Hz per frame.
    public static func bandwidth(spectrogram: Signal, sr: Int, nFFT: Int,
                                  p: Float = 2.0) -> Signal {
        let nFreqs = spectrogram.shape[0]
        let nFrames = spectrogram.shape[1]

        // Build frequency vector
        var freqs = [Float](repeating: 0, count: nFreqs)
        let srFloat = Float(sr)
        let nFFTFloat = Float(nFFT)
        for k in 0..<nFreqs {
            freqs[k] = Float(k) * srFloat / nFFTFloat
        }

        // First compute centroid
        let cent = centroid(spectrogram: spectrogram, sr: sr, nFFT: nFFT)

        let outPtr = UnsafeMutablePointer<Float>.allocate(capacity: nFrames)
        outPtr.initialize(repeating: 0, count: nFrames)

        spectrogram.withUnsafeBufferPointer { specBuf in
            cent.withUnsafeBufferPointer { centBuf in
                for f in 0..<nFrames {
                    let c = centBuf[f]
                    var weightedSum: Float = 0
                    var totalSum: Float = 0
                    for k in 0..<nFreqs {
                        let val = specBuf[k * nFrames + f]
                        let deviation = abs(freqs[k] - c)
                        weightedSum += val * powf(deviation, p)
                        totalSum += val
                    }
                    if totalSum > 0 {
                        outPtr[f] = powf(weightedSum / totalSum, 1.0 / p)
                    } else {
                        outPtr[f] = 0
                    }
                }
            }
        }

        let outBuffer = UnsafeMutableBufferPointer(start: outPtr, count: nFrames)
        return Signal(taking: outBuffer, shape: [nFrames], sampleRate: sr)
    }

    // MARK: - Contrast

    /// Spectral contrast: difference between peaks and valleys in sub-bands.
    ///
    /// Splits the spectrum into `nBands + 1` octave-spaced sub-bands starting from `fMin`,
    /// and computes the difference between the mean of the top and bottom percentiles
    /// (in dB scale) for each sub-band.
    ///
    /// Matches librosa's `spectral_contrast` algorithm:
    /// - Frequency bands are spaced by octaves from fMin
    /// - Within each band, sort magnitudes
    /// - Peak = mean of top `quantile` fraction of magnitudes
    /// - Valley = mean of bottom `quantile` fraction of magnitudes
    /// - Contrast = peak - valley (in linear domain, converted to dB-like values)
    ///
    /// - Parameters:
    ///   - spectrogram: Magnitude spectrogram with shape [nFreqs, nFrames].
    ///   - sr: Sample rate in Hz.
    ///   - nFFT: FFT size used to compute the spectrogram.
    ///   - nBands: Number of octave sub-bands (excluding the top remainder band). Default 6.
    ///   - fMin: Low frequency bound for the lowest octave band (Hz). Default 200.0.
    ///   - quantile: Fraction of bins to use for peak/valley estimation. Default 0.02.
    /// - Returns: Signal with shape [nBands+1, nFrames], contrast per band per frame.
    public static func contrast(spectrogram: Signal, sr: Int, nFFT: Int,
                                 nBands: Int = 6, fMin: Float = 200.0,
                                 quantile: Float = 0.02) -> Signal {
        let nFreqs = spectrogram.shape[0]
        let nFrames = spectrogram.shape[1]
        let totalBands = nBands + 1

        // Build frequency band edges (octave-spaced from fMin)
        let nyquist = Float(sr) / 2.0
        let freqResolution = Float(sr) / Float(nFFT)

        // Band edges in Hz: fMin, fMin*2, fMin*4, ..., nyquist
        var bandEdges = [Float](repeating: 0, count: nBands + 2)
        bandEdges[0] = 0  // DC
        for b in 1...nBands {
            bandEdges[b] = fMin * powf(2.0, Float(b - 1))
        }
        bandEdges[nBands + 1] = nyquist

        // Convert to bin indices
        var binEdges = [Int](repeating: 0, count: nBands + 2)
        for b in 0..<(nBands + 2) {
            binEdges[b] = min(Int(roundf(bandEdges[b] / freqResolution)), nFreqs)
        }

        let outCount = totalBands * nFrames
        let outPtr = UnsafeMutablePointer<Float>.allocate(capacity: outCount)
        outPtr.initialize(repeating: 0, count: outCount)

        spectrogram.withUnsafeBufferPointer { specBuf in
            for f in 0..<nFrames {
                for b in 0..<totalBands {
                    let startBin = binEdges[b]
                    let endBin = binEdges[b + 1]
                    let bandSize = endBin - startBin

                    if bandSize <= 0 {
                        outPtr[b * nFrames + f] = 0
                        continue
                    }

                    // Extract and sort magnitudes in this band
                    var bandMags = [Float](repeating: 0, count: bandSize)
                    for k in 0..<bandSize {
                        bandMags[k] = specBuf[(startBin + k) * nFrames + f]
                    }
                    bandMags.sort()

                    // Number of bins for quantile (at least 1)
                    let qBins = max(1, Int(roundf(quantile * Float(bandSize))))

                    // Valley: mean of bottom qBins
                    var valley: Float = 0
                    for i in 0..<qBins {
                        valley += bandMags[i]
                    }
                    valley /= Float(qBins)

                    // Peak: mean of top qBins
                    var peak: Float = 0
                    for i in (bandSize - qBins)..<bandSize {
                        peak += bandMags[i]
                    }
                    peak /= Float(qBins)

                    // Convert to log scale (dB-like): 10 * log10(x + amin)
                    let amin: Float = 1e-10
                    let peakDb = 10.0 * log10f(max(peak, amin))
                    let valleyDb = 10.0 * log10f(max(valley, amin))
                    outPtr[b * nFrames + f] = peakDb - valleyDb
                }
            }
        }

        let outBuffer = UnsafeMutableBufferPointer(start: outPtr, count: outCount)
        return Signal(taking: outBuffer, shape: [totalBands, nFrames], sampleRate: sr)
    }

    // MARK: - Rolloff

    /// Spectral rolloff: frequency below which `rollPercent` of the spectral energy is contained.
    ///
    /// For each frame, finds the frequency bin k such that:
    /// `sum(S[0..k, t]) >= rollPercent * sum(S[:, t])`
    ///
    /// - Parameters:
    ///   - spectrogram: Magnitude spectrogram with shape [nFreqs, nFrames].
    ///   - sr: Sample rate in Hz.
    ///   - nFFT: FFT size used to compute the spectrogram.
    ///   - rollPercent: Energy fraction threshold. Default 0.85.
    /// - Returns: Signal with shape [nFrames], rolloff frequency in Hz per frame.
    public static func rolloff(spectrogram: Signal, sr: Int, nFFT: Int,
                                rollPercent: Float = 0.85) -> Signal {
        let nFreqs = spectrogram.shape[0]
        let nFrames = spectrogram.shape[1]

        // Build frequency vector
        let srFloat = Float(sr)
        let nFFTFloat = Float(nFFT)

        let outPtr = UnsafeMutablePointer<Float>.allocate(capacity: nFrames)
        outPtr.initialize(repeating: 0, count: nFrames)

        spectrogram.withUnsafeBufferPointer { specBuf in
            for f in 0..<nFrames {
                // Compute total energy for this frame
                var totalEnergy: Float = 0
                for k in 0..<nFreqs {
                    totalEnergy += specBuf[k * nFrames + f]
                }

                let threshold = rollPercent * totalEnergy

                if totalEnergy <= 0 {
                    outPtr[f] = 0
                    continue
                }

                // Find the bin where cumulative energy reaches threshold
                var cumEnergy: Float = 0
                var rollBin = 0
                for k in 0..<nFreqs {
                    cumEnergy += specBuf[k * nFrames + f]
                    if cumEnergy >= threshold {
                        rollBin = k
                        break
                    }
                }

                // Convert bin index to frequency
                outPtr[f] = Float(rollBin) * srFloat / nFFTFloat
            }
        }

        let outBuffer = UnsafeMutableBufferPointer(start: outPtr, count: nFrames)
        return Signal(taking: outBuffer, shape: [nFrames], sampleRate: sr)
    }

    // MARK: - Flatness

    /// Spectral flatness: geometric mean / arithmetic mean of the spectrum (Wiener entropy).
    ///
    /// Values near 1.0 indicate noise-like (flat) spectrum.
    /// Values near 0.0 indicate tonal (peaked) spectrum.
    ///
    /// Matches librosa's `spectral_flatness` behavior:
    /// 1. Take magnitude spectrogram
    /// 2. Apply power exponent: `S_thresh = max(amin, S^power)`
    /// 3. Compute `gmean / amean` where gmean = exp(mean(log(S_thresh)))
    ///
    /// Uses log-domain computation for numerical stability.
    ///
    /// - Parameters:
    ///   - spectrogram: Magnitude spectrogram with shape [nFreqs, nFrames].
    ///   - power: Power exponent to apply to the magnitude spectrogram. Default 2.0
    ///            (matches librosa default: computes flatness on power spectrogram).
    ///   - amin: Minimum threshold to avoid log(0). Default 1e-10.
    /// - Returns: Signal with shape [nFrames], flatness in [0, 1] per frame.
    public static func flatness(spectrogram: Signal, power: Float = 2.0,
                                 amin: Float = 1e-10) -> Signal {
        let nFreqs = spectrogram.shape[0]
        let nFrames = spectrogram.shape[1]

        let outPtr = UnsafeMutablePointer<Float>.allocate(capacity: nFrames)
        outPtr.initialize(repeating: 0, count: nFrames)

        spectrogram.withUnsafeBufferPointer { specBuf in
            for f in 0..<nFrames {
                // First check if the frame is essentially silent (all zeros)
                var rawSum: Float = 0
                for k in 0..<nFreqs {
                    rawSum += specBuf[k * nFrames + f]
                }

                if rawSum < amin {
                    outPtr[f] = 0
                    continue
                }

                // Apply power exponent and floor, then compute gmean/amean
                var logSum: Float = 0
                var arithSum: Float = 0
                for k in 0..<nFreqs {
                    let mag = specBuf[k * nFrames + f]
                    let val = max(powf(mag, power), amin)
                    logSum += logf(val)
                    arithSum += val
                }

                let arithMean = arithSum / Float(nFreqs)
                let geoMean = expf(logSum / Float(nFreqs))

                if arithMean > amin {
                    outPtr[f] = geoMean / arithMean
                } else {
                    outPtr[f] = 0
                }
            }
        }

        let outBuffer = UnsafeMutableBufferPointer(start: outPtr, count: nFrames)
        return Signal(taking: outBuffer, shape: [nFrames], sampleRate: spectrogram.sampleRate)
    }
}
