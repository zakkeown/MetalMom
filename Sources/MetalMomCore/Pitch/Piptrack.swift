import Foundation
import Accelerate

/// Pitch tracking via parabolic interpolation of STFT spectral peaks.
public enum Piptrack {

    /// Pitch tracking via parabolic interpolation of STFT peaks.
    ///
    /// Returns Signal with shape `[2 * nFreqs, nFrames]`:
    /// - Rows `0..<nFreqs`: pitch values in Hz (0 for non-peak bins)
    /// - Rows `nFreqs..<2*nFreqs`: magnitude values (0 for non-peak bins)
    ///
    /// - Parameters:
    ///   - signal: Input audio signal (1-D).
    ///   - sr: Sample rate. If nil, uses signal.sampleRate.
    ///   - nFFT: FFT size. Default 2048.
    ///   - hopLength: Hop length. Default nFFT/4.
    ///   - winLength: Window length. Default nFFT.
    ///   - fMin: Minimum frequency in Hz. Default 150.
    ///   - fMax: Maximum frequency in Hz. Default 4000.
    ///   - threshold: Magnitude threshold relative to max. Default 0.1.
    ///   - center: Whether to center-pad the signal. Default true.
    public static func piptrack(
        signal: Signal,
        sr: Int? = nil,
        nFFT: Int = 2048,
        hopLength: Int? = nil,
        winLength: Int? = nil,
        fMin: Float = 150.0,
        fMax: Float = 4000.0,
        threshold: Float = 0.1,
        center: Bool = true
    ) -> Signal {
        let sampleRate = sr ?? signal.sampleRate
        let hop = hopLength ?? nFFT / 4
        let win = winLength ?? nFFT
        let nFreqs = nFFT / 2 + 1

        // Compute magnitude spectrogram using STFT.compute
        // Returns shape [nFreqs, nFrames], row-major
        let mag = STFT.compute(
            signal: signal,
            nFFT: nFFT,
            hopLength: hop,
            winLength: win,
            center: center
        )

        let nFrames = mag.shape.count >= 2 ? mag.shape[1] : 0
        guard nFrames > 0 && nFreqs > 2 else {
            return Signal(data: [], shape: [2 * nFreqs, 0], sampleRate: sampleRate)
        }

        // Frequency resolution per bin
        let freqPerBin = Float(sampleRate) / Float(nFFT)

        // Bin range for fMin/fMax
        let binMin = max(1, Int(fMin / freqPerBin))
        let binMax = min(nFreqs - 2, Int(fMax / freqPerBin))

        guard binMin <= binMax else {
            let zeros = [Float](repeating: 0, count: 2 * nFreqs * nFrames)
            return Signal(data: zeros, shape: [2 * nFreqs, nFrames], sampleRate: sampleRate)
        }

        // Output arrays: pitches and magnitudes, both [nFreqs, nFrames], row-major
        var pitches = [Float](repeating: 0, count: nFreqs * nFrames)
        var magnitudes = [Float](repeating: 0, count: nFreqs * nFrames)

        // Process each frame
        for frame in 0..<nFrames {
            // Find max magnitude in this frame for thresholding
            var frameMax: Float = 0
            for bin in 0..<nFreqs {
                let val = mag[bin * nFrames + frame]
                if val > frameMax { frameMax = val }
            }

            let threshVal = threshold * frameMax

            // Scan for local peaks within [binMin, binMax]
            for bin in binMin...binMax {
                let curr = mag[bin * nFrames + frame]
                let prev = mag[(bin - 1) * nFrames + frame]
                let next = mag[(bin + 1) * nFrames + frame]

                // Local peak: curr > both neighbors and above threshold
                if curr > prev && curr > next && curr >= threshVal {
                    // Parabolic interpolation
                    let denom = prev - 2.0 * curr + next
                    var refinedBin = Float(bin)
                    var refinedMag = curr

                    if abs(denom) > 1e-10 {
                        let p = 0.5 * (prev - next) / denom
                        refinedBin = Float(bin) + p
                        // Interpolated magnitude peak
                        refinedMag = curr - 0.25 * (prev - next) * p
                    }

                    let freq = refinedBin * freqPerBin

                    // Check frequency bounds after interpolation
                    if freq >= fMin && freq <= fMax {
                        pitches[bin * nFrames + frame] = freq
                        magnitudes[bin * nFrames + frame] = refinedMag
                    }
                }
            }
        }

        // Pack into single [2*nFreqs, nFrames] array
        var output = [Float](repeating: 0, count: 2 * nFreqs * nFrames)
        // First nFreqs rows: pitches
        for i in 0..<(nFreqs * nFrames) {
            output[i] = pitches[i]
        }
        // Next nFreqs rows: magnitudes
        for i in 0..<(nFreqs * nFrames) {
            output[nFreqs * nFrames + i] = magnitudes[i]
        }

        return Signal(data: output, shape: [2 * nFreqs, nFrames], sampleRate: sampleRate)
    }
}
