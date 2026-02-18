import Foundation
import Accelerate

/// Tuning estimation: deviation from A440 in fractions of a semitone.
public enum Tuning {

    /// Estimate tuning deviation from A440.
    ///
    /// Uses STFT spectral peaks (via piptrack) to estimate how far
    /// the overall tuning deviates from A440 reference.
    ///
    /// - Parameters:
    ///   - signal: Input audio signal (1-D).
    ///   - sr: Sample rate. If nil, uses signal.sampleRate.
    ///   - nFFT: FFT size. Default 2048.
    ///   - hopLength: Hop length. Default nFFT/4.
    ///   - winLength: Window length. Default nFFT.
    ///   - resolution: Histogram bin width in fractions of a semitone. Default 0.01.
    ///   - binsPerOctave: Number of bins per octave (12 for standard). Default 12.
    ///   - center: Whether to center-pad the signal. Default true.
    /// - Returns: Estimated tuning deviation in [-0.5, 0.5] semitone fractions.
    public static func estimateTuning(
        signal: Signal,
        sr: Int? = nil,
        nFFT: Int = 2048,
        hopLength: Int? = nil,
        winLength: Int? = nil,
        resolution: Float = 0.01,
        binsPerOctave: Int = 12,
        center: Bool = true
    ) -> Float {
        let sampleRate = sr ?? signal.sampleRate

        // Step 1: Use piptrack to find spectral peaks
        let pipResult = Piptrack.piptrack(
            signal: signal,
            sr: sampleRate,
            nFFT: nFFT,
            hopLength: hopLength,
            winLength: winLength,
            fMin: 150.0,
            fMax: 4000.0,
            threshold: 0.1,
            center: center
        )

        let nFreqs = nFFT / 2 + 1
        let nFrames = pipResult.shape.count >= 2 ? pipResult.shape[1] : 0

        guard nFrames > 0 else { return 0.0 }

        // Step 2: Extract pitch deviations and magnitudes
        let bpo = Float(binsPerOctave)
        var deviations: [Float] = []
        var weights: [Float] = []

        for frame in 0..<nFrames {
            for bin in 0..<nFreqs {
                let pitch = pipResult[bin * nFrames + frame]
                let mag = pipResult[(nFreqs + bin) * nFrames + frame]

                if pitch > 0 && mag > 0 {
                    // Compute deviation from nearest equal-tempered pitch
                    // dev = bpo * log2(freq / 440.0), then take fractional part
                    let semitoneDeviation = bpo * log2f(pitch / 440.0)
                    var frac = semitoneDeviation - roundf(semitoneDeviation)
                    // Clamp to [-0.5, 0.5]
                    frac = max(-0.5, min(0.5, frac))

                    deviations.append(frac)
                    weights.append(mag)
                }
            }
        }

        guard !deviations.isEmpty else { return 0.0 }

        // Step 3: Build weighted histogram
        let nBins = Int(1.0 / resolution)
        var histogram = [Float](repeating: 0, count: nBins)

        for i in 0..<deviations.count {
            let dev = deviations[i]
            let w = weights[i]
            // Map [-0.5, 0.5] to bin index [0, nBins)
            var binIdx = Int((dev + 0.5) / resolution)
            binIdx = max(0, min(nBins - 1, binIdx))
            histogram[binIdx] += w
        }

        // Step 4: Find peak of histogram
        var maxWeight: Float = 0
        var maxBin = nBins / 2  // default to center (0.0)

        for i in 0..<nBins {
            if histogram[i] > maxWeight {
                maxWeight = histogram[i]
                maxBin = i
            }
        }

        // Convert bin index back to deviation value
        let tuning = (Float(maxBin) + 0.5) * resolution - 0.5

        return tuning
    }
}
