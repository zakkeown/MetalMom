import Accelerate
import Foundation

public enum BeatTracker {
    /// Track beats using the Ellis 2007 dynamic programming algorithm.
    ///
    /// 1. Compute onset strength envelope
    /// 2. Estimate tempo via ACF with log-normal prior
    /// 3. Dynamic programming to find optimal beat sequence
    /// 4. Backtrace to recover beat frames
    ///
    /// - Parameters:
    ///   - signal: Input audio signal (1D).
    ///   - sr: Sample rate override. If nil, uses signal.sampleRate.
    ///   - hopLength: Hop length. Default 512.
    ///   - nFFT: FFT window size. Default 2048.
    ///   - nMels: Number of mel bands. Default 128.
    ///   - fmin: Minimum frequency for mel filterbank. Default 0.
    ///   - fmax: Maximum frequency. If nil, uses sr/2.
    ///   - startBPM: Prior tempo center in BPM. Default 120.
    ///   - trimFirst: Trim the first beat. Default true.
    ///   - trimLast: Trim the last beat. Default true.
    /// - Returns: (tempo in BPM, beat frame indices as Signal).
    public static func beatTrack(
        signal: Signal,
        sr: Int? = nil,
        hopLength: Int? = nil,
        nFFT: Int = 2048,
        nMels: Int = 128,
        fmin: Float = 0,
        fmax: Float? = nil,
        startBPM: Float = 120.0,
        trimFirst: Bool = true,
        trimLast: Bool = true
    ) -> (tempo: Float, beats: Signal) {
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
            center: true,
            aggregate: true
        )

        let nFrames = oenv.shape.count > 1 ? oenv.shape[1] : oenv.shape[0]
        guard nFrames > 1 else {
            return (0, Signal(data: [], shape: [0], sampleRate: sampleRate))
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
        }

        // 2. Estimate tempo via ACF
        let tempo = estimateTempo(
            envelope: envData,
            sr: sampleRate,
            hopLength: hop,
            startBPM: startBPM
        )

        guard tempo > 0 else {
            return (0, Signal(data: [], shape: [0], sampleRate: sampleRate))
        }

        // Convert tempo BPM to period in frames
        let period = 60.0 * Float(sampleRate) / (tempo * Float(hop))

        // 3. Dynamic programming
        let beatFrames = dpBeatTrack(
            envelope: envData,
            period: period,
            trimFirst: trimFirst,
            trimLast: trimLast
        )

        let floatFrames = beatFrames.map { Float($0) }
        return (tempo, Signal(data: floatFrames.isEmpty ? [] : floatFrames,
                              shape: [floatFrames.count], sampleRate: sampleRate))
    }

    // MARK: - Tempo Estimation via ACF

    /// Estimate tempo from onset envelope using autocorrelation with log-normal prior.
    ///
    /// - Parameters:
    ///   - envelope: Normalized onset strength envelope.
    ///   - sr: Sample rate.
    ///   - hopLength: Hop length in samples.
    ///   - startBPM: Prior center tempo in BPM.
    ///   - bpmStd: Standard deviation in octaves for the log-normal prior. Default 1.0.
    /// - Returns: Estimated tempo in BPM.
    public static func estimateTempo(
        envelope: [Float],
        sr: Int,
        hopLength: Int,
        startBPM: Float = 120.0,
        bpmStd: Float = 1.0
    ) -> Float {
        let n = envelope.count
        guard n > 1 else { return 0 }

        // Compute autocorrelation using vDSP
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

        // Use vDSP_conv for autocorrelation
        // For autocorrelation: convolve x with reversed x
        // Result length = 2*n - 1, but we only need lags 0..<n
        var result = [Float](repeating: 0, count: n)

        // Manual autocorrelation for each lag
        // This is O(n^2) but n is typically small (hundreds of frames)
        for lag in 0..<n {
            var sum: Float = 0
            let count = n - lag
            // Use vDSP_dotpr for the inner product
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

    // MARK: - Dynamic Programming Beat Tracking

    /// Find optimal beat sequence using dynamic programming (Ellis 2007).
    ///
    /// - Parameters:
    ///   - envelope: Normalized onset strength envelope.
    ///   - period: Estimated beat period in frames.
    ///   - trimFirst: Trim the first beat.
    ///   - trimLast: Trim the last beat.
    /// - Returns: Array of beat frame indices, sorted.
    private static func dpBeatTrack(
        envelope: [Float],
        period: Float,
        trimFirst: Bool,
        trimLast: Bool
    ) -> [Int] {
        let n = envelope.count
        guard n > 0 else { return [] }

        let periodInt = max(1, Int(round(period)))
        let alpha: Float = 100.0 / (period * period)

        // Window size: search +-window around expected period
        let window = max(1, periodInt / 2)

        // DP arrays
        var score = [Float](repeating: 0, count: n)
        var backPointer = [Int](repeating: -1, count: n)

        // Initialize scores with onset envelope
        for t in 0..<n {
            score[t] = envelope[t]
        }

        // Fill DP table
        let logPeriod = log(period)

        for t in 1..<n {
            // Search window for predecessors
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

        // Trim first and last beats if requested
        if trimFirst && beats.count > 1 {
            beats.removeFirst()
        }
        if trimLast && beats.count > 1 {
            beats.removeLast()
        }

        return beats
    }
}
