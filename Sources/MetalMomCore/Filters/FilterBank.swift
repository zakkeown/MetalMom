import Foundation

/// Mel filterbank generation, matching librosa's `librosa.filters.mel()`.
///
/// Produces a matrix of triangular mel-frequency filters that can be applied
/// to a linear-frequency spectrogram to obtain a mel spectrogram.
public enum FilterBank {

    /// Generate a mel filterbank matrix.
    ///
    /// Returns a `Signal` with shape `[nMels, nFFT/2+1]` (row-major),
    /// where each row is a triangular filter in the frequency domain.
    /// Uses Slaney-style area normalisation, matching librosa's default
    /// `norm="slaney"`.
    ///
    /// - Parameters:
    ///   - sr: Audio sample rate in Hz.
    ///   - nFFT: FFT size (determines frequency resolution).
    ///   - nMels: Number of mel bands. Default 128.
    ///   - fMin: Lowest filter frequency in Hz. Default 0.
    ///   - fMax: Highest filter frequency in Hz. If `nil`, uses `sr / 2`.
    /// - Returns: A `Signal` of shape `[nMels, nFFT/2+1]`.
    public static func mel(
        sr: Int,
        nFFT: Int,
        nMels: Int = 128,
        fMin: Float = 0.0,
        fMax: Float? = nil
    ) -> Signal {
        let fMaxActual = fMax ?? Float(sr) / 2.0
        let nFreqs = nFFT / 2 + 1

        // Compute nMels + 2 mel-spaced centre frequencies, then convert back to Hz.
        let melMin = Units.hzToMel(fMin)
        let melMax = Units.hzToMel(fMaxActual)
        let melPoints: [Float] = (0..<(nMels + 2)).map { i in
            Units.melToHz(melMin + Float(i) * (melMax - melMin) / Float(nMels + 1))
        }

        // Linear FFT bin frequencies: k * sr / nFFT  for k in 0..<nFreqs
        let fftFreqs: [Float] = (0..<nFreqs).map { k in
            Float(k) * Float(sr) / Float(nFFT)
        }

        // Build triangular filters (row-major: filter m occupies row m).
        var weights = [Float](repeating: 0, count: nMels * nFreqs)

        for m in 0..<nMels {
            let fLeft = melPoints[m]
            let fCenter = melPoints[m + 1]
            let fRight = melPoints[m + 2]

            for k in 0..<nFreqs {
                let freq = fftFreqs[k]
                if freq >= fLeft && freq <= fCenter && fCenter != fLeft {
                    weights[m * nFreqs + k] = (freq - fLeft) / (fCenter - fLeft)
                } else if freq > fCenter && freq <= fRight && fRight != fCenter {
                    weights[m * nFreqs + k] = (fRight - freq) / (fRight - fCenter)
                }
            }

            // Slaney-style area normalisation: 2 / (f_right - f_left)
            let enorm = 2.0 / (melPoints[m + 2] - melPoints[m])
            for k in 0..<nFreqs {
                weights[m * nFreqs + k] *= enorm
            }
        }

        return Signal(data: weights, shape: [nMels, nFreqs], sampleRate: sr)
    }
}
