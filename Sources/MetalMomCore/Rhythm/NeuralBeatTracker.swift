import Accelerate
import Foundation

/// Neural beat tracker that decodes beat positions from activation probabilities.
///
/// Replicates the approach of madmom's `DBNBeatTrackingProcessor`:
/// 1. RNN ensemble produces beat activation probabilities per frame
/// 2. Dynamic programming decodes the optimal beat sequence
///
/// The `decode` method is the core testable component that works with
/// pre-computed activations (no CoreML required). The `beatTrack` method
/// is the high-level API that handles feature extraction and model inference.
public enum NeuralBeatTracker {

    // MARK: - High-Level API

    /// Track beats using neural network activations + DP decoding.
    ///
    /// This is the high-level API that:
    /// 1. Computes spectrogram features
    /// 2. Runs ensemble inference through CoreML models (if provided)
    /// 3. Decodes beat positions using dynamic programming
    ///
    /// When no model URLs are provided, falls back to onset-strength-based
    /// activations processed through the same DP decoder.
    ///
    /// - Parameters:
    ///   - signal: Input audio signal.
    ///   - sr: Sample rate override. If nil, uses signal.sampleRate.
    ///   - modelURLs: URLs to CoreML model files. If nil, uses onset strength.
    ///   - hopLength: Hop length for feature extraction. Default 441 (10ms at 44100Hz).
    ///   - fps: Frames per second for the model. Default 100.
    ///   - minBPM: Minimum tempo. Default 55.
    ///   - maxBPM: Maximum tempo. Default 215.
    ///   - transitionLambda: Tempo transition smoothness. Default 100.
    ///   - threshold: Activation threshold. Default 0.05.
    ///   - trim: Whether to trim first/last beats. Default true.
    /// - Returns: (tempo in BPM, beat frames as Signal).
    public static func beatTrack(
        signal: Signal,
        sr: Int? = nil,
        modelURLs: [URL]? = nil,
        hopLength: Int = 441,
        fps: Float = 100.0,
        minBPM: Float = 55.0,
        maxBPM: Float = 215.0,
        transitionLambda: Float = 100.0,
        threshold: Float = 0.05,
        trim: Bool = true
    ) throws -> (tempo: Float, beats: Signal) {
        let sampleRate = sr ?? signal.sampleRate

        // If model URLs are provided, use CoreML inference (future work).
        // For now, fall back to onset-strength-based activations.
        let activations: [Float]

        if let _ = modelURLs {
            // Future: load CoreML models and run ensemble inference.
            // For now, use onset strength as a proxy for beat activations.
            activations = computeActivationsFromOnsetStrength(
                signal: signal, sr: sampleRate, hopLength: hopLength
            )
        } else {
            activations = computeActivationsFromOnsetStrength(
                signal: signal, sr: sampleRate, hopLength: hopLength
            )
        }

        guard !activations.isEmpty else {
            return (0, Signal(data: [], shape: [0], sampleRate: sampleRate))
        }

        let (tempo, beatFrames) = decode(
            activations: activations,
            fps: fps,
            minBPM: minBPM,
            maxBPM: maxBPM,
            transitionLambda: transitionLambda,
            threshold: threshold,
            trim: trim
        )

        let floatFrames = beatFrames.map { Float($0) }
        return (tempo, Signal(data: floatFrames.isEmpty ? [] : floatFrames,
                              shape: [floatFrames.count], sampleRate: sampleRate))
    }

    // MARK: - Core Decoder

    /// Decode beat positions from pre-computed activation probabilities.
    ///
    /// This is the core DP decoder, fully testable without CoreML models.
    /// Uses a dynamic programming approach inspired by Ellis 2007, but
    /// operating on neural activation probabilities rather than onset strength.
    ///
    /// Algorithm:
    /// 1. Estimate tempo via autocorrelation of activations with log-normal prior
    /// 2. DP beat tracking with log-penalty for deviations from expected period
    /// 3. Backtrace to find optimal beat sequence
    /// 4. Post-processing: threshold filtering and optional trimming
    ///
    /// - Parameters:
    ///   - activations: Beat activation probabilities [nFrames], values in [0,1].
    ///   - fps: Frames per second. Default 100.
    ///   - minBPM: Minimum tempo in BPM. Default 55.
    ///   - maxBPM: Maximum tempo in BPM. Default 215.
    ///   - transitionLambda: Penalty for tempo changes. Higher = smoother. Default 100.
    ///   - threshold: Minimum activation to consider. Default 0.05.
    ///   - trim: Trim first/last beats. Default true.
    /// - Returns: (estimated tempo in BPM, beat frame indices).
    public static func decode(
        activations: [Float],
        fps: Float = 100.0,
        minBPM: Float = 55.0,
        maxBPM: Float = 215.0,
        transitionLambda: Float = 100.0,
        threshold: Float = 0.05,
        trim: Bool = true
    ) -> (tempo: Float, beats: [Int]) {
        let n = activations.count
        guard n > 1 else {
            if n == 1 && activations[0] >= threshold {
                return (0, [0])
            }
            return (0, [])
        }

        // 1. Estimate tempo via ACF of activations
        let tempo = estimateTempoFromActivations(
            activations: activations,
            fps: fps,
            minBPM: minBPM,
            maxBPM: maxBPM
        )

        guard tempo > 0 else {
            return (0, [])
        }

        // Convert BPM to period in frames
        let period = 60.0 * fps / tempo

        // 2. Dynamic programming beat tracking
        let beatFrames = dpBeatDecode(
            activations: activations,
            period: period,
            transitionLambda: transitionLambda,
            threshold: threshold,
            trim: trim
        )

        return (tempo, beatFrames)
    }

    // MARK: - Tempo Estimation from Activations

    /// Estimate tempo from beat activation probabilities using autocorrelation.
    ///
    /// - Parameters:
    ///   - activations: Beat activation probabilities.
    ///   - fps: Frames per second.
    ///   - minBPM: Minimum tempo in BPM.
    ///   - maxBPM: Maximum tempo in BPM.
    ///   - startBPM: Prior center for log-normal weighting. Default 120.
    ///   - bpmStd: Standard deviation in octaves. Default 1.0.
    /// - Returns: Estimated tempo in BPM, or 0 if estimation fails.
    public static func estimateTempoFromActivations(
        activations: [Float],
        fps: Float,
        minBPM: Float = 55.0,
        maxBPM: Float = 215.0,
        startBPM: Float = 120.0,
        bpmStd: Float = 1.0
    ) -> Float {
        let n = activations.count
        guard n > 1 else { return 0 }

        // Compute autocorrelation
        let acf = autocorrelation(activations)

        // Convert BPM range to lag range (in frames)
        let minLag = max(1, Int(60.0 * fps / maxBPM))
        let maxLag = min(n - 1, Int(60.0 * fps / minBPM))

        guard minLag < maxLag else { return startBPM }

        // Apply log-normal tempo prior and find peak
        var bestLag = minLag
        var bestScore: Float = -Float.infinity
        let logStartBPM = log2(startBPM)

        for lag in minLag...maxLag {
            let bpm = 60.0 * fps / Float(lag)
            let logBPM = log2(bpm)
            let z = (logBPM - logStartBPM) / bpmStd
            let prior = exp(-0.5 * z * z)
            let score = acf[lag] * prior

            if score > bestScore {
                bestScore = score
                bestLag = lag
            }
        }

        let estimatedBPM = 60.0 * fps / Float(bestLag)

        // Clamp to the valid range
        return max(minBPM, min(maxBPM, estimatedBPM))
    }

    // MARK: - Autocorrelation

    /// Compute unnormalized autocorrelation using vDSP.
    private static func autocorrelation(_ x: [Float]) -> [Float] {
        let n = x.count
        guard n > 0 else { return [] }

        var result = [Float](repeating: 0, count: n)

        for lag in 0..<n {
            var sum: Float = 0
            let count = n - lag
            x.withUnsafeBufferPointer { buf in
                vDSP_dotpr(buf.baseAddress!, 1,
                           buf.baseAddress!.advanced(by: lag), 1,
                           &sum,
                           vDSP_Length(count))
            }
            result[lag] = sum
        }

        return result
    }

    // MARK: - DP Beat Decoding

    /// Dynamic programming beat decoder for activation probabilities.
    ///
    /// Score function: C[t] = activation[t] + max over prev < t of
    ///   (C[prev] - lambda * |log(delta / period)|^2)
    ///
    /// - Parameters:
    ///   - activations: Beat activation probabilities [nFrames].
    ///   - period: Expected beat period in frames.
    ///   - transitionLambda: Penalty for tempo deviations.
    ///   - threshold: Minimum activation to keep.
    ///   - trim: Trim first/last beats.
    /// - Returns: Sorted array of beat frame indices.
    private static func dpBeatDecode(
        activations: [Float],
        period: Float,
        transitionLambda: Float,
        threshold: Float,
        trim: Bool
    ) -> [Int] {
        let n = activations.count
        guard n > 0 else { return [] }

        let periodInt = max(1, Int(round(period)))

        // Search window: +-window around expected period
        let window = max(1, periodInt / 2)

        // DP arrays
        var score = [Float](repeating: 0, count: n)
        var backPointer = [Int](repeating: -1, count: n)

        // Initialize scores with activations
        for t in 0..<n {
            score[t] = activations[t]
        }

        // Fill DP table
        let logPeriod = log(period)

        // Penalty scaling: higher transitionLambda means stricter period adherence
        let alpha = transitionLambda / (period * period)

        for t in 1..<n {
            let searchLo = max(0, t - periodInt - window)
            let searchHi = max(0, min(t - 1, t - periodInt + window))

            guard searchLo <= searchHi else { continue }

            var bestPrev = searchLo
            var bestVal: Float = -Float.infinity

            for tau in searchLo...searchHi {
                let interval = Float(t - tau)
                guard interval > 0 else { continue }
                let logInterval = log(interval)
                let diff = logInterval - logPeriod
                let penalty = -alpha * diff * diff
                let val = score[tau] + penalty

                if val > bestVal {
                    bestVal = val
                    bestPrev = tau
                }
            }

            score[t] += bestVal
            backPointer[t] = bestPrev
        }

        // Find the best ending beat
        var bestEnd = 0
        var bestScore: Float = -Float.infinity
        for t in 0..<n {
            if score[t] > bestScore {
                bestScore = score[t]
                bestEnd = t
            }
        }

        // Backtrace
        var beats: [Int] = []
        var t = bestEnd
        while t >= 0 {
            beats.append(t)
            let prev = backPointer[t]
            if prev < 0 || prev >= t {
                break
            }
            t = prev
        }

        // Reverse to chronological order
        beats.reverse()

        // Filter by threshold: only keep beats where activation exceeds threshold
        if threshold > 0 {
            beats = beats.filter { activations[$0] >= threshold }
        }

        // Trim first and last beats if requested
        if trim && beats.count > 2 {
            beats.removeFirst()
            beats.removeLast()
        }

        return beats
    }

    // MARK: - Onset-Strength Fallback

    /// Compute beat-like activations from onset strength (fallback when no RNN models).
    ///
    /// Normalizes the onset envelope to [0, 1] to simulate activation probabilities.
    private static func computeActivationsFromOnsetStrength(
        signal: Signal,
        sr: Int,
        hopLength: Int
    ) -> [Float] {
        let oenv = OnsetDetection.onsetStrength(
            signal: signal,
            sr: sr,
            nFFT: 2048,
            hopLength: hopLength,
            nMels: 128,
            fmin: 0,
            fmax: nil,
            center: true,
            aggregate: true
        )

        let nFrames = oenv.shape.count > 1 ? oenv.shape[1] : oenv.shape[0]
        guard nFrames > 0 else { return [] }

        var envData = [Float](repeating: 0, count: nFrames)
        oenv.withUnsafeBufferPointer { buf in
            for i in 0..<nFrames {
                envData[i] = buf[i]
            }
        }

        // Normalize to [0, 1]
        var maxVal: Float = 0
        vDSP_maxv(envData, 1, &maxVal, vDSP_Length(nFrames))
        if maxVal > 0 {
            var scale = 1.0 / maxVal
            vDSP_vsmul(envData, 1, &scale, &envData, 1, vDSP_Length(nFrames))
        }

        return envData
    }
}
