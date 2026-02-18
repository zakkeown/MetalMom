import Accelerate
import Foundation

/// Standalone tempo estimation using autocorrelation with log-normal prior.
///
/// This provides tempo estimation as an independent feature, separate from
/// beat tracking.  The algorithm follows librosa's `tempo()`:
///
/// 1. Compute onset strength envelope
/// 2. Compute full autocorrelation of the envelope
/// 3. Weight by log-normal tempo prior centered on `startBPM`
/// 4. Restrict to reasonable BPM range (30-300)
/// 5. Return tempo at the peak of the weighted ACF
public enum TempoEstimation {

    /// Estimate tempo from an audio signal.
    ///
    /// - Parameters:
    ///   - signal: Input audio signal (1D).
    ///   - sr: Sample rate override. If nil, uses `signal.sampleRate`.
    ///   - hopLength: Hop length in samples. Default 512.
    ///   - nFFT: FFT window size. Default 2048.
    ///   - nMels: Number of mel bands. Default 128.
    ///   - fmin: Minimum frequency for mel filterbank. Default 0.
    ///   - fmax: Maximum frequency. If nil, uses sr/2.
    ///   - startBPM: Prior tempo center in BPM. Default 120.
    ///   - center: Centre-pad signal before STFT. Default true.
    /// - Returns: Estimated tempo in BPM.
    public static func tempo(
        signal: Signal,
        sr: Int? = nil,
        hopLength: Int? = nil,
        nFFT: Int = 2048,
        nMels: Int = 128,
        fmin: Float = 0,
        fmax: Float? = nil,
        startBPM: Float = 120.0,
        center: Bool = true
    ) -> Float {
        let sampleRate = sr ?? signal.sampleRate
        let hop = hopLength ?? 512

        // 1. Compute onset strength envelope
        let oenv = OnsetDetection.onsetStrength(
            signal: signal,
            sr: sampleRate,
            nFFT: nFFT,
            hopLength: hop,
            nMels: nMels,
            fmin: fmin,
            fmax: fmax,
            center: center,
            aggregate: true
        )

        let nFrames = oenv.shape.count > 1 ? oenv.shape[1] : oenv.shape[0]
        guard nFrames > 1 else {
            return startBPM
        }

        // Extract envelope data
        var envData = [Float](repeating: 0, count: nFrames)
        oenv.withUnsafeBufferPointer { buf in
            for i in 0..<nFrames {
                envData[i] = buf[i]
            }
        }

        // Normalize envelope to [0, 1]
        var envMax: Float = 0
        vDSP_maxv(envData, 1, &envMax, vDSP_Length(nFrames))
        if envMax > 0 {
            var scale = 1.0 / envMax
            vDSP_vsmul(envData, 1, &scale, &envData, 1, vDSP_Length(nFrames))
        } else {
            // All-zero envelope (silence): return startBPM as default
            return startBPM
        }

        // 2. Estimate tempo via ACF with log-normal prior
        return estimateFromEnvelope(
            envelope: envData,
            sr: sampleRate,
            hopLength: hop,
            startBPM: startBPM
        )
    }

    // MARK: - Internal

    /// Estimate tempo from a pre-computed onset envelope using ACF with log-normal prior.
    ///
    /// - Parameters:
    ///   - envelope: Normalized onset strength envelope.
    ///   - sr: Sample rate.
    ///   - hopLength: Hop length in samples.
    ///   - startBPM: Prior center tempo in BPM.
    ///   - bpmStd: Standard deviation in octaves for the log-normal prior. Default 1.0.
    /// - Returns: Estimated tempo in BPM.
    static func estimateFromEnvelope(
        envelope: [Float],
        sr: Int,
        hopLength: Int,
        startBPM: Float = 120.0,
        bpmStd: Float = 1.0
    ) -> Float {
        let n = envelope.count
        guard n > 1 else { return startBPM }

        // Compute autocorrelation
        let acf = autocorrelation(envelope)

        // Define BPM search range
        let minBPM: Float = 30.0
        let maxBPM: Float = 300.0

        // Convert BPM range to lag range (in frames)
        let framesPerSec = Float(sr) / Float(hopLength)
        let minLag = max(1, Int(60.0 * framesPerSec / maxBPM))
        let maxLag = min(n - 1, Int(60.0 * framesPerSec / minBPM))

        guard minLag < maxLag else { return startBPM }

        // Apply log-normal tempo prior and find peak
        var bestLag = minLag
        var bestScore: Float = -Float.infinity
        let logStartBPM = log2(startBPM)

        for lag in minLag...maxLag {
            let bpm = 60.0 * framesPerSec / Float(lag)
            let logBPM = log2(bpm)
            let z = (logBPM - logStartBPM) / bpmStd
            let prior = exp(-0.5 * z * z)
            let score = acf[lag] * prior

            if score > bestScore {
                bestScore = score
                bestLag = lag
            }
        }

        let estimatedBPM = 60.0 * framesPerSec / Float(bestLag)
        return estimatedBPM
    }

    // MARK: - Autocorrelation

    /// Compute unnormalized autocorrelation of a signal using vDSP.
    private static func autocorrelation(_ x: [Float]) -> [Float] {
        let n = x.count
        guard n > 0 else { return [] }

        var result = [Float](repeating: 0, count: n)

        for lag in 0..<n {
            var sum: Float = 0
            let count = n - lag
            x.withUnsafeBufferPointer { xBuf in
                vDSP_dotpr(xBuf.baseAddress!, 1,
                           xBuf.baseAddress!.advanced(by: lag), 1,
                           &sum,
                           vDSP_Length(count))
            }
            result[lag] = sum
        }

        return result
    }
}
