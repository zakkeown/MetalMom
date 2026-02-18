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

    // MARK: - pYIN (Probabilistic YIN)

    /// Probabilistic YIN pitch estimation with Viterbi decoding (Mauch & Dixon 2014).
    ///
    /// Returns a Signal with shape `[3, nFrames]`:
    /// - Row 0: f0 in Hz (0 for unvoiced frames)
    /// - Row 1: voiced flag (1.0 = voiced, 0.0 = unvoiced)
    /// - Row 2: voiced probability in [0, 1]
    ///
    /// - Parameters:
    ///   - signal: Input audio signal (1-D).
    ///   - fMin: Minimum frequency in Hz.
    ///   - fMax: Maximum frequency in Hz.
    ///   - sr: Sample rate. If nil, uses signal.sampleRate.
    ///   - frameLength: Analysis frame length. Default 2048.
    ///   - hopLength: Hop length. Default frameLength/4.
    ///   - nThresholds: Number of CMNDF thresholds to test. Default 100.
    ///   - betaAlpha: Alpha parameter for beta distribution. Default 2.
    ///   - betaBeta: Beta parameter for beta distribution. Default 18.
    ///   - resolution: Pitch resolution in semitones. Default 0.1.
    ///   - switchProb: Probability of switching between voiced/unvoiced. Default 0.01.
    ///   - center: Whether to center-pad the signal. Default true.
    public static func pyin(
        signal: Signal,
        fMin: Float,
        fMax: Float,
        sr: Int? = nil,
        frameLength: Int = 2048,
        hopLength: Int? = nil,
        nThresholds: Int = 100,
        betaAlpha: Float = 2.0,
        betaBeta: Float = 18.0,
        resolution: Float = 0.1,
        switchProb: Float = 0.01,
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

        guard samples.count >= frameLength else {
            return Signal(data: [], shape: [3, 0], sampleRate: sampleRate)
        }
        let nFrames = 1 + (samples.count - frameLength) / hop

        // Lag bounds
        let minLag = max(1, Int(Float(sampleRate) / fMax))
        let maxLag = min(frameLength / 2, Int(Float(sampleRate) / fMin))

        guard maxLag > minLag else {
            let zeros = [Float](repeating: 0, count: 3 * nFrames)
            return Signal(data: zeros, shape: [3, nFrames], sampleRate: sampleRate)
        }

        // Build pitch bins: from fMin to fMax in steps of `resolution` semitones
        let nBins = Int(ceil(12.0 * log2(fMax / fMin) / resolution))
        guard nBins > 0 else {
            let zeros = [Float](repeating: 0, count: 3 * nFrames)
            return Signal(data: zeros, shape: [3, nFrames], sampleRate: sampleRate)
        }

        // Precompute bin center frequencies
        var binFreqs = [Float](repeating: 0, count: nBins)
        for b in 0..<nBins {
            binFreqs[b] = fMin * pow(2.0, Float(b) * resolution / 12.0)
        }

        // Total HMM states: nBins voiced + 1 unvoiced
        let nStates = nBins + 1
        let unvoicedState = nBins  // last state is unvoiced

        // Precompute log transition matrix
        // Transition model: voiced↔voiced uses Gaussian in semitone distance
        // voiced→unvoiced: switchProb, unvoiced→voiced: switchProb
        let logSwitchProb = log(max(switchProb, 1e-30))
        let logStayUnvoiced = log(max(1.0 - switchProb, 1e-30))

        // For voiced→voiced: Gaussian with sigma in semitones
        // max_transition_rate of ~35.92 semitones/sec, one hop = hop/sr seconds
        let hopTime = Float(hop) / Float(sampleRate)
        let maxSemitonesPerHop = 35.92 * hopTime
        let transSigma = max(maxSemitonesPerHop, 0.5)

        // Pre-build voiced→voiced transition (normalized per source state)
        // logTransVoicedToVoiced[b] = log probability of going from bin b_src to bin b
        // We only store unnormalized, then normalize per source
        var logTransVV = [Float](repeating: -Float.infinity, count: nBins * nBins)
        for src in 0..<nBins {
            var maxVal: Float = -Float.infinity
            for dst in 0..<nBins {
                let dist = Float(abs(dst - src)) * resolution  // distance in semitones
                let logP = -0.5 * (dist * dist) / (transSigma * transSigma)
                logTransVV[src * nBins + dst] = logP
                if logP > maxVal { maxVal = logP }
            }
            // Log-sum-exp to normalize
            var sumExp: Float = 0
            for dst in 0..<nBins {
                sumExp += exp(logTransVV[src * nBins + dst] - maxVal)
            }
            let logNorm = maxVal + log(sumExp)
            // Scale by (1 - switchProb) so total from voiced state sums to 1
            let logScale = log(max(1.0 - switchProb, 1e-30))
            for dst in 0..<nBins {
                logTransVV[src * nBins + dst] = logTransVV[src * nBins + dst] - logNorm + logScale
            }
        }

        // Thresholds for multi-threshold YIN
        var thresholds = [Float](repeating: 0, count: nThresholds)
        for i in 0..<nThresholds {
            thresholds[i] = Float(i + 1) / Float(nThresholds)
        }

        // For each frame, compute observation log-likelihoods for each state
        // Use Viterbi in log domain
        var viterbiPrev = [Float](repeating: -Float.infinity, count: nStates)
        var backpointer = [[Int]](repeating: [Int](repeating: 0, count: nStates), count: nFrames)

        // Observation log-likelihoods for current frame
        var obsLogLik = [Float](repeating: -Float.infinity, count: nStates)

        for frame in 0..<nFrames {
            let frameStart = frame * hop
            let frameSlice = Array(samples[frameStart..<(frameStart + frameLength)])

            // Compute CMNDF
            let diff = differenceFunctionVDSP(frame: frameSlice, maxLag: maxLag)
            let cmndf = cumulativeMeanNormalized(diff: diff)

            // Extract candidates: for each threshold, find the first lag below it
            // Accumulate probability mass in pitch bins
            var binProbs = [Float](repeating: 0, count: nBins)
            var totalVoicedProb: Float = 0
            var totalUnvoicedProb: Float = 0

            for t in 0..<nThresholds {
                let thresh = thresholds[t]
                let lag = findBestLag(cmndf: cmndf, minLag: minLag, maxLag: maxLag,
                                       threshold: thresh)
                if lag > 0 {
                    let refinedLag = parabolicInterpolation(cmndf: cmndf, lag: lag, maxLag: maxLag)
                    let freq = Float(sampleRate) / refinedLag

                    if freq >= fMin && freq <= fMax {
                        // Compute beta-distribution based probability for this candidate
                        let d = min(max(cmndf[lag], 0), 1)
                        let pVoiced = betaSurvival(x: d, alpha: betaAlpha, beta: betaBeta)

                        // Find the nearest pitch bin
                        let semitones = 12.0 * log2(freq / fMin)
                        let bin = Int(round(semitones / resolution))
                        if bin >= 0 && bin < nBins {
                            binProbs[bin] += pVoiced
                            totalVoicedProb += pVoiced
                            totalUnvoicedProb += (1.0 - pVoiced)
                        } else {
                            totalUnvoicedProb += 1.0
                        }
                    } else {
                        totalUnvoicedProb += 1.0
                    }
                } else {
                    totalUnvoicedProb += 1.0
                }
            }

            // Normalize observation probabilities
            let totalProb = totalVoicedProb + totalUnvoicedProb
            if totalProb > 0 {
                for b in 0..<nBins {
                    let p = binProbs[b] / totalProb
                    obsLogLik[b] = p > 0 ? log(p) : -Float.infinity
                }
                obsLogLik[unvoicedState] = log(max(totalUnvoicedProb / totalProb, 1e-30))
            } else {
                for b in 0..<nBins {
                    obsLogLik[b] = -Float.infinity
                }
                obsLogLik[unvoicedState] = 0  // log(1)
            }

            if frame == 0 {
                // Initialize: uniform prior over all states
                let logPrior = -log(Float(nStates))
                for s in 0..<nStates {
                    viterbiPrev[s] = logPrior + obsLogLik[s]
                }
            } else {
                var viterbiCurr = [Float](repeating: -Float.infinity, count: nStates)

                // For each destination state, find best source
                for dst in 0..<nBins {
                    var bestVal: Float = -Float.infinity
                    var bestSrc = 0

                    // From voiced states
                    for src in 0..<nBins {
                        let val = viterbiPrev[src] + logTransVV[src * nBins + dst]
                        if val > bestVal {
                            bestVal = val
                            bestSrc = src
                        }
                    }
                    // From unvoiced state
                    let fromUnvoiced = viterbiPrev[unvoicedState] + logSwitchProb - log(Float(nBins))
                    if fromUnvoiced > bestVal {
                        bestVal = fromUnvoiced
                        bestSrc = unvoicedState
                    }

                    viterbiCurr[dst] = bestVal + obsLogLik[dst]
                    backpointer[frame][dst] = bestSrc
                }

                // Unvoiced destination
                do {
                    var bestVal: Float = -Float.infinity
                    var bestSrc = 0

                    // From any voiced state
                    for src in 0..<nBins {
                        let val = viterbiPrev[src] + logSwitchProb
                        if val > bestVal {
                            bestVal = val
                            bestSrc = src
                        }
                    }
                    // From unvoiced
                    let fromUnvoiced = viterbiPrev[unvoicedState] + logStayUnvoiced
                    if fromUnvoiced > bestVal {
                        bestVal = fromUnvoiced
                        bestSrc = unvoicedState
                    }

                    viterbiCurr[unvoicedState] = bestVal + obsLogLik[unvoicedState]
                    backpointer[frame][unvoicedState] = bestSrc
                }

                viterbiPrev = viterbiCurr
            }
        }

        // Backtrack
        var stateSeq = [Int](repeating: 0, count: nFrames)
        if nFrames > 0 {
            // Find best final state
            var bestState = 0
            var bestVal: Float = -Float.infinity
            for s in 0..<nStates {
                if viterbiPrev[s] > bestVal {
                    bestVal = viterbiPrev[s]
                    bestState = s
                }
            }
            stateSeq[nFrames - 1] = bestState
            var t = nFrames - 1
            while t > 0 {
                stateSeq[t - 1] = backpointer[t][stateSeq[t]]
                t -= 1
            }
        }

        // Convert state sequence to f0, voiced flag, voiced probability
        // For voiced probability, we use the per-frame observation probability
        // Recompute per-frame voiced probability from the observation likelihoods
        var outputData = [Float](repeating: 0, count: 3 * nFrames)

        for frame in 0..<nFrames {
            let state = stateSeq[frame]

            if state < nBins {
                // Voiced: convert bin to frequency
                outputData[frame] = binFreqs[state]
                outputData[nFrames + frame] = 1.0  // voiced flag
            } else {
                // Unvoiced
                outputData[frame] = 0.0
                outputData[nFrames + frame] = 0.0
            }
        }

        // Compute voiced probabilities: re-run per-frame analysis
        for frame in 0..<nFrames {
            let frameStart = frame * hop
            let frameSlice = Array(samples[frameStart..<(frameStart + frameLength)])
            let diff = differenceFunctionVDSP(frame: frameSlice, maxLag: maxLag)
            let cmndf = cumulativeMeanNormalized(diff: diff)

            var totalVoicedP: Float = 0
            var totalUnvoicedP: Float = 0

            for t in 0..<nThresholds {
                let thresh = thresholds[t]
                let lag = findBestLag(cmndf: cmndf, minLag: minLag, maxLag: maxLag,
                                       threshold: thresh)
                if lag > 0 {
                    let refinedLag = parabolicInterpolation(cmndf: cmndf, lag: lag, maxLag: maxLag)
                    let freq = Float(sampleRate) / refinedLag
                    if freq >= fMin && freq <= fMax {
                        let d = min(max(cmndf[lag], 0), 1)
                        let pV = betaSurvival(x: d, alpha: betaAlpha, beta: betaBeta)
                        totalVoicedP += pV
                        totalUnvoicedP += (1.0 - pV)
                    } else {
                        totalUnvoicedP += 1.0
                    }
                } else {
                    totalUnvoicedP += 1.0
                }
            }

            let total = totalVoicedP + totalUnvoicedP
            let voicedProb: Float = total > 0 ? totalVoicedP / total : 0
            outputData[2 * nFrames + frame] = voicedProb
        }

        return Signal(data: outputData, shape: [3, nFrames], sampleRate: sampleRate)
    }

    // MARK: - Beta Distribution Helpers

    /// Compute the survival function (1 - CDF) of the beta distribution.
    /// Uses a simple approximation via the regularized incomplete beta function.
    private static func betaSurvival(x: Float, alpha: Float, beta: Float) -> Float {
        // 1 - BetaCDF(x, alpha, beta)
        // = 1 - I_x(alpha, beta)
        let cdf = regularizedBeta(x: x, a: alpha, b: beta)
        return max(0, min(1, 1.0 - cdf))
    }

    /// Regularized incomplete beta function I_x(a, b) using a continued fraction
    /// approximation (Lentz's method). Good for a, b > 0 and 0 <= x <= 1.
    private static func regularizedBeta(x: Float, a: Float, b: Float) -> Float {
        if x <= 0 { return 0 }
        if x >= 1 { return 1 }

        // Use symmetry: I_x(a,b) = 1 - I_{1-x}(b,a) when x > (a+1)/(a+b+2)
        let threshold = (a + 1) / (a + b + 2)
        if x > threshold {
            return 1.0 - regularizedBeta(x: 1.0 - x, a: b, b: a)
        }

        // Compute log(x^a * (1-x)^b / (a * Beta(a,b)))
        let logPrefix = a * log(x) + b * log(1.0 - x) - logBeta(a: a, b: b) - log(a)

        // Continued fraction using Lentz's algorithm
        let cf = betaContinuedFraction(x: x, a: a, b: b)
        return min(1, max(0, exp(logPrefix) * cf))
    }

    /// Log of the beta function: log(Beta(a, b)) = lgamma(a) + lgamma(b) - lgamma(a+b)
    private static func logBeta(a: Float, b: Float) -> Float {
        return lgammaf(a) + lgammaf(b) - lgammaf(a + b)
    }

    /// Continued fraction for the regularized incomplete beta function.
    private static func betaContinuedFraction(x: Float, a: Float, b: Float) -> Float {
        let maxIter = 200
        let eps: Float = 1e-7
        let tiny: Float = 1e-30

        var f: Float = 1.0
        var c: Float = 1.0
        var d: Float = 1.0 - (a + b) * x / (a + 1)
        if abs(d) < tiny { d = tiny }
        d = 1.0 / d
        f = d

        for m in 1...maxIter {
            let mf = Float(m)

            // Even step: d_{2m}
            var num = mf * (b - mf) * x / ((a + 2 * mf - 1) * (a + 2 * mf))
            d = 1.0 + num / d
            if abs(d) < tiny { d = tiny }
            c = 1.0 + num / c
            if abs(c) < tiny { c = tiny }
            d = 1.0 / d
            f *= c * d

            // Odd step: d_{2m+1}
            num = -(a + mf) * (a + b + mf) * x / ((a + 2 * mf) * (a + 2 * mf + 1))
            d = 1.0 + num / d
            if abs(d) < tiny { d = tiny }
            c = 1.0 + num / c
            if abs(c) < tiny { c = tiny }
            d = 1.0 / d
            let delta = c * d
            f *= delta

            if abs(delta - 1.0) < eps {
                break
            }
        }

        return f
    }
}
