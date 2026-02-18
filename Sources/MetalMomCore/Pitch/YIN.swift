import Foundation
import Accelerate

/// YIN fundamental frequency estimation (de Chevigne & Kawahara 2002).
public enum YIN {

    /// Estimate fundamental frequency using the YIN algorithm.
    ///
    /// Returns a Signal with shape `[nFrames]` containing f0 estimates in Hz.
    /// Unvoiced frames are set to 0.
    ///
    /// - Parameters:
    ///   - signal: Input audio signal (1-D).
    ///   - fMin: Minimum frequency in Hz.
    ///   - fMax: Maximum frequency in Hz.
    ///   - sr: Sample rate. If nil, uses signal.sampleRate.
    ///   - frameLength: Analysis frame length. Default 2048.
    ///   - hopLength: Hop length. Default frameLength/4.
    ///   - troughThreshold: CMNDF threshold. Default 0.1.
    ///   - center: Whether to center-pad the signal. Default true.
    public static func yin(
        signal: Signal,
        fMin: Float,
        fMax: Float,
        sr: Int? = nil,
        frameLength: Int = 2048,
        hopLength: Int? = nil,
        troughThreshold: Float = 0.1,
        center: Bool = true
    ) -> Signal {
        let sampleRate = sr ?? signal.sampleRate
        let hop = hopLength ?? (frameLength / 4)

        // Prepare padded signal if center=true
        let samples: [Float]
        if center {
            let pad = frameLength / 2
            var padded = [Float](repeating: 0, count: pad)
            for i in 0..<signal.count {
                padded.append(signal[i])
            }
            padded.append(contentsOf: [Float](repeating: 0, count: pad))
            samples = padded
        } else {
            var arr = [Float]()
            arr.reserveCapacity(signal.count)
            for i in 0..<signal.count {
                arr.append(signal[i])
            }
            samples = arr
        }

        // Calculate number of frames
        guard samples.count >= frameLength else {
            return Signal(data: [], sampleRate: sampleRate)
        }
        let nFrames = 1 + (samples.count - frameLength) / hop

        // Lag bounds
        let minLag = max(1, Int(Float(sampleRate) / fMax))
        let maxLag = min(frameLength / 2, Int(Float(sampleRate) / fMin))

        guard maxLag > minLag else {
            // Invalid frequency range; return zeros
            return Signal(data: [Float](repeating: 0, count: nFrames), sampleRate: sampleRate)
        }

        var f0 = [Float](repeating: 0, count: nFrames)

        for frame in 0..<nFrames {
            let frameStart = frame * hop

            // Extract frame
            let frameSlice = Array(samples[frameStart..<(frameStart + frameLength)])

            // Compute difference function for lags 0..maxLag
            let diff = differenceFunctionVDSP(frame: frameSlice, maxLag: maxLag)

            // Compute cumulative mean normalized difference function (CMNDF)
            let cmndf = cumulativeMeanNormalized(diff: diff)

            // Find the best lag using threshold + local minimum search
            let bestLag = findBestLag(cmndf: cmndf, minLag: minLag, maxLag: maxLag,
                                      threshold: troughThreshold)

            if bestLag > 0 {
                // Parabolic interpolation to refine the lag
                let refinedLag = parabolicInterpolation(cmndf: cmndf, lag: bestLag, maxLag: maxLag)

                // Convert lag to frequency
                let freq = Float(sampleRate) / refinedLag

                // Clip to [fMin, fMax]; if outside, mark as unvoiced
                if freq >= fMin && freq <= fMax {
                    f0[frame] = freq
                }
                // else remains 0 (unvoiced)
            }
            // else remains 0 (unvoiced)
        }

        return Signal(data: f0, sampleRate: sampleRate)
    }

    // MARK: - Private Helpers

    /// Compute the difference function using vDSP for efficiency.
    /// d(tau) = sum_j (x[j] - x[j+tau])^2  for j = 0..<W-tau
    /// We use the identity: d(tau) = r(0) + r_shifted(0) - 2*r(tau)
    /// where r(tau) = sum_j x[j]*x[j+tau] (autocorrelation)
    private static func differenceFunctionVDSP(frame: [Float], maxLag: Int) -> [Float] {
        let W = frame.count
        var diff = [Float](repeating: 0, count: maxLag + 1)

        // diff[0] = 0 by definition
        diff[0] = 0

        // Compute using direct squared differences for correctness and clarity
        // For each lag tau, compute sum_j (x[j] - x[j+tau])^2
        for tau in 1...maxLag {
            let n = W - tau
            guard n > 0 else {
                diff[tau] = 0
                continue
            }

            // Use vDSP: subtract x[tau..] from x[0..n-1], square, sum
            var result: Float = 0
            frame.withUnsafeBufferPointer { xBuf in
                // x[j] - x[j+tau] for j = 0..<n
                var sub = [Float](repeating: 0, count: n)
                vDSP_vsub(xBuf.baseAddress! + tau, 1,
                          xBuf.baseAddress!, 1,
                          &sub, 1,
                          vDSP_Length(n))
                // Square and sum
                vDSP_dotpr(sub, 1, sub, 1, &result, vDSP_Length(n))
            }
            diff[tau] = result
        }

        return diff
    }

    /// Compute cumulative mean normalized difference function.
    /// d'(0) = 1
    /// d'(tau) = d(tau) / ((1/tau) * sum_{j=1}^{tau} d(j))
    private static func cumulativeMeanNormalized(diff: [Float]) -> [Float] {
        let n = diff.count
        var cmndf = [Float](repeating: 1.0, count: n)

        if n <= 1 { return cmndf }

        var runningSum: Float = 0
        for tau in 1..<n {
            runningSum += diff[tau]
            if runningSum > 0 {
                cmndf[tau] = diff[tau] * Float(tau) / runningSum
            } else {
                cmndf[tau] = 1.0
            }
        }

        return cmndf
    }

    /// Find the best lag using the threshold method.
    /// 1. Find the first lag >= minLag where CMNDF dips below threshold
    /// 2. Then find the local minimum from that point
    /// 3. If no dip below threshold, use the global minimum in [minLag, maxLag]
    private static func findBestLag(cmndf: [Float], minLag: Int, maxLag: Int,
                                     threshold: Float) -> Int {
        let searchMax = min(maxLag, cmndf.count - 1)
        guard minLag <= searchMax else { return 0 }

        // Step 1: Find the first lag where CMNDF < threshold
        var thresholdLag = -1
        for tau in minLag...searchMax {
            if cmndf[tau] < threshold {
                thresholdLag = tau
                break
            }
        }

        if thresholdLag >= 0 {
            // Step 2: Find local minimum starting from thresholdLag
            var bestLag = thresholdLag
            var bestVal = cmndf[thresholdLag]
            for tau in (thresholdLag + 1)...searchMax {
                if cmndf[tau] < bestVal {
                    bestVal = cmndf[tau]
                    bestLag = tau
                } else {
                    // Rising again -- we found the local min
                    break
                }
            }
            return bestLag
        }

        // No dip below threshold: use global minimum in search range
        var globalMinLag = minLag
        var globalMinVal = cmndf[minLag]
        for tau in (minLag + 1)...searchMax {
            if cmndf[tau] < globalMinVal {
                globalMinVal = cmndf[tau]
                globalMinLag = tau
            }
        }
        return globalMinLag
    }

    /// Parabolic interpolation around the lag to refine the estimate.
    private static func parabolicInterpolation(cmndf: [Float], lag: Int, maxLag: Int) -> Float {
        guard lag > 0 && lag < min(maxLag, cmndf.count - 1) else {
            return Float(lag)
        }

        let alpha = cmndf[lag - 1]
        let beta = cmndf[lag]
        let gamma = cmndf[lag + 1]

        let denominator = 2.0 * (2.0 * beta - alpha - gamma)
        if abs(denominator) < 1e-10 {
            return Float(lag)
        }

        let shift = (alpha - gamma) / denominator
        return Float(lag) + shift
    }
}
