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

        // Compute nMels + 2 mel-spaced centre frequencies in Double precision,
        // then convert back to Hz.  Using Double here matches librosa, which
        // performs all mel-scale arithmetic in float64.
        let melMin = Units.hzToMelD(Double(fMin))
        let melMax = Units.hzToMelD(Double(fMaxActual))
        let melPointsD: [Double] = (0..<(nMels + 2)).map { i in
            Units.melToHzD(melMin + Double(i) * (melMax - melMin) / Double(nMels + 1))
        }
        let melPoints: [Float] = melPointsD.map { Float($0) }

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

    // MARK: - Logarithmic Filterbank

    /// Generate a logarithmic-frequency triangular filterbank matrix, matching
    /// madmom's `LogarithmicFilterbank`.
    ///
    /// Centre frequencies are computed as `fref * 2^(i / bandsPerOctave)` for
    /// integer `i`, clipped to `[fMin, fMax]`.  Each centre frequency is then
    /// snapped to the nearest FFT bin.  When `uniqueFilters` is `true`
    /// (the default, matching madmom's `DeepChromaProcessor`), duplicate bin
    /// indices that arise from insufficient FFT resolution at low frequencies
    /// are collapsed, producing fewer output bands.
    ///
    /// Triangular filters are built in the FFT-bin domain: for three
    /// consecutive unique bin indices (left, centre, right), the filter ramps
    /// linearly from 0 at the left bin to 1 at the centre bin, and back to 0
    /// at the right bin.
    ///
    /// Returns a `Signal` with shape `[nBins, nFFT/2+1]` (row-major).
    ///
    /// - Parameters:
    ///   - nFFT: FFT size (determines frequency resolution).
    ///   - sampleRate: Audio sample rate in Hz.
    ///   - numBandsPerOctave: Number of filter bands per octave. Default 24.
    ///   - fMin: Lowest filter centre frequency in Hz. Default 65.0.
    ///   - fMax: Highest filter centre frequency in Hz. Default 2100.0.
    ///   - fRef: Reference frequency in Hz. Default 440.0 (A4).
    ///   - uniqueFilters: Remove duplicate filters at low frequencies. Default `true`.
    /// - Returns: A `Signal` of shape `[nBins, nFFT/2+1]`.
    public static func logarithmic(
        nFFT: Int,
        sampleRate: Int,
        numBandsPerOctave: Int = 24,
        fMin: Float = 65.0,
        fMax: Float = 2100.0,
        fRef: Float = 440.0,
        uniqueFilters: Bool = true
    ) -> Signal {
        let nFreqs = nFFT / 2 + 1
        let bandsF = Float(numBandsPerOctave)

        // 1. Generate log-spaced centre frequencies (madmom's log_frequencies)
        //    left  = floor(log2(fMin / fRef) * bands)
        //    right = ceil(log2(fMax / fRef) * bands)
        //    freqs = fRef * 2^(arange(left, right) / bands), clipped to [fMin, fMax]
        let leftExp = Int(floor(log2(fMin / fRef) * bandsF))
        let rightExp = Int(ceil(log2(fMax / fRef) * bandsF))

        var centreFreqs: [Float] = []
        for i in leftExp..<rightExp {
            let freq = fRef * powf(2.0, Float(i) / bandsF)
            if freq >= fMin && freq <= fMax {
                centreFreqs.append(freq)
            }
        }

        guard centreFreqs.count >= 3 else {
            let nBins = max(0, centreFreqs.count - 2)
            return Signal(data: [Float](repeating: 0, count: nBins * nFreqs),
                         shape: [nBins, nFreqs], sampleRate: sampleRate)
        }

        // 2. Map each centre frequency to the nearest FFT bin index
        //    (madmom's frequencies2bins with rounding to nearest)
        let binFreqStep = Float(sampleRate) / Float(nFFT)
        var binIndices: [Int] = centreFreqs.map { freq in
            let idx = Int((freq / binFreqStep).rounded())
            return min(max(idx, 0), nFreqs - 1)
        }

        // 3. Remove duplicate bin indices if uniqueFilters is true
        if uniqueFilters {
            var seen = Set<Int>()
            var unique: [Int] = []
            for idx in binIndices {
                if !seen.contains(idx) {
                    seen.insert(idx)
                    unique.append(idx)
                }
            }
            binIndices = unique
        }

        // We need at least 3 bin positions to form one triangular filter
        guard binIndices.count >= 3 else {
            return Signal(data: [], shape: [0, nFreqs], sampleRate: sampleRate)
        }

        // Number of output filters = number of unique bins - 2
        // (each filter uses left/centre/right from consecutive triplet)
        let nBins = binIndices.count - 2

        // 4. Build triangular filters in the bin domain
        var weights = [Float](repeating: 0, count: nBins * nFreqs)

        for m in 0..<nBins {
            let leftBin = binIndices[m]
            let centreBin = binIndices[m + 1]
            let rightBin = binIndices[m + 2]

            // Rising slope: leftBin to centreBin
            if centreBin > leftBin {
                let span = Float(centreBin - leftBin)
                for k in leftBin...centreBin {
                    weights[m * nFreqs + k] = Float(k - leftBin) / span
                }
            } else {
                // Degenerate: leftBin == centreBin
                weights[m * nFreqs + centreBin] = 1.0
            }

            // Falling slope: centreBin to rightBin
            if rightBin > centreBin {
                let span = Float(rightBin - centreBin)
                for k in centreBin...rightBin {
                    weights[m * nFreqs + k] = Float(rightBin - k) / span
                }
            }
            // Note: centreBin is set twice (rising=1.0, falling=1.0), which is correct.
        }

        return Signal(data: weights, shape: [nBins, nFreqs], sampleRate: sampleRate)
    }
}
