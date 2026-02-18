import Accelerate
import Foundation

public enum OnsetDetection {
    /// Compute onset strength envelope.
    ///
    /// Measures spectral flux (positive first-order difference of the mel spectrogram in dB).
    /// Matches librosa's ``onset_strength()`` behavior including lag-based differencing
    /// and center-based frame shift compensation.
    ///
    /// - Parameters:
    ///   - signal: Input audio signal (1D).
    ///   - sr: Sample rate override. If nil, uses signal.sampleRate.
    ///   - nFFT: FFT window size. Default 2048.
    ///   - hopLength: Hop length. Default 512.
    ///   - nMels: Number of mel bands. Default 128.
    ///   - fmin: Minimum frequency for mel filterbank. Default 0.
    ///   - fmax: Maximum frequency. If nil, uses sr/2.
    ///   - center: Center-pad the signal and shift onset function. Default true.
    ///   - aggregate: If true, average across mel bands. Default true.
    ///   - lag: Time lag for computing differences. Default 1.
    /// - Returns: Signal with shape [1, nFrames] (if aggregate) or [nMels, nFrames].
    public static func onsetStrength(
        signal: Signal,
        sr: Int? = nil,
        nFFT: Int = 2048,
        hopLength: Int? = nil,
        nMels: Int = 128,
        fmin: Float = 0,
        fmax: Float? = nil,
        center: Bool = true,
        aggregate: Bool = true,
        lag: Int = 1
    ) -> Signal {
        let sampleRate = sr ?? signal.sampleRate
        let hop = hopLength ?? 512

        // 1. Compute mel spectrogram (power)
        let melSpec = MelSpectrogram.compute(
            signal: signal,
            sr: sampleRate,
            nFFT: nFFT,
            hopLength: hop,
            winLength: nFFT,
            center: center,
            power: 2.0,
            nMels: nMels,
            fMin: fmin,
            fMax: fmax
        )

        let nBands = melSpec.shape[0]  // nMels
        let nFrames = melSpec.shape[1]

        // 2. Convert to dB
        let melDb = Scaling.powerToDb(melSpec, ref: 1.0, amin: 1e-10, topDb: 80.0)

        // 3. Compute lag-based difference (spectral flux) along time axis
        // onset_env = S[..., lag:] - S[..., :-lag]
        // This produces nFrames - lag diff values
        guard nFrames > lag else {
            if aggregate {
                return Signal(data: [0], shape: [1, 1], sampleRate: sampleRate)
            } else {
                return Signal(data: [Float](repeating: 0, count: nBands),
                              shape: [nBands, 1], sampleRate: sampleRate)
            }
        }

        let diffCount = nFrames - lag

        // 4. Compute padding width to prepend
        // librosa: pad_width = lag; if center: pad_width += n_fft // (2 * hop_length)
        var padWidth = lag
        if center {
            padWidth += nFFT / (2 * hop)
        }

        // 5. Output length: diffCount + padWidth, then trim to nFrames if center
        let rawLen = diffCount + padWidth
        let outFrames: Int
        if center {
            outFrames = min(rawLen, nFrames)
        } else {
            outFrames = rawLen
        }

        if aggregate {
            // Average across bands for each frame
            var onset = [Float](repeating: 0, count: outFrames)

            melDb.withUnsafeBufferPointer { src in
                for i in 0..<diffCount {
                    let t = i + lag  // index into the S array for S[..., lag:]
                    let tRef = i     // index into the S array for S[..., :-lag]
                    var sum: Float = 0
                    for b in 0..<nBands {
                        let diff = src[b * nFrames + t] - src[b * nFrames + tRef]
                        sum += max(0, diff)  // Half-wave rectification
                    }
                    let outIdx = i + padWidth
                    if outIdx < outFrames {
                        onset[outIdx] = sum / Float(nBands)
                    }
                }
            }

            return Signal(data: onset, shape: [1, outFrames], sampleRate: sampleRate)
        } else {
            // Return per-band onset strength
            var onset = [Float](repeating: 0, count: nBands * outFrames)

            melDb.withUnsafeBufferPointer { src in
                for b in 0..<nBands {
                    for i in 0..<diffCount {
                        let t = i + lag
                        let tRef = i
                        let diff = src[b * nFrames + t] - src[b * nFrames + tRef]
                        let outIdx = i + padWidth
                        if outIdx < outFrames {
                            onset[b * outFrames + outIdx] = max(0, diff)
                        }
                    }
                }
            }

            return Signal(data: onset, shape: [nBands, outFrames], sampleRate: sampleRate)
        }
    }

    // MARK: - SuperFlux

    /// Compute SuperFlux onset detection function (Böck & Widmer, 2013).
    ///
    /// Enhanced spectral flux with maximum filtering along the frequency axis
    /// to suppress vibrato-induced false positives.
    ///
    /// - Parameters:
    ///   - signal: Input audio signal (1D).
    ///   - sr: Sample rate override. If nil, uses signal.sampleRate.
    ///   - nFFT: FFT window size. Default 2048.
    ///   - hopLength: Hop length. Default nFFT/4.
    ///   - maxFilterSize: Width of max filter along frequency axis. Default 3.
    ///   - lag: Time lag for computing differences. Default 1.
    /// - Returns: Signal with shape [1, nFrames].
    public static func superflux(
        signal: Signal,
        sr: Int? = nil,
        nFFT: Int = 2048,
        hopLength: Int? = nil,
        maxFilterSize: Int = 3,
        lag: Int = 1
    ) -> Signal {
        let sampleRate = sr ?? signal.sampleRate
        let hop = hopLength ?? nFFT / 4

        // 1. Compute magnitude spectrogram
        let mag = STFT.compute(signal: signal, nFFT: nFFT, hopLength: hop, winLength: nFFT, center: true)
        let nFreqs = mag.shape[0]
        let nFrames = mag.shape[1]

        guard nFrames > lag else {
            return Signal(data: [0], shape: [1, 1], sampleRate: sampleRate)
        }

        // 2. Apply maximum filter along frequency axis
        // S_filtered[f,t] = max(S[max(0,f-w):min(nFreqs,f+w+1), t])
        let halfW = maxFilterSize / 2
        var filtered = [Float](repeating: 0, count: nFreqs * nFrames)

        mag.withUnsafeBufferPointer { src in
            // src layout: row-major [nFreqs, nFrames] => src[f * nFrames + t]
            for f in 0..<nFreqs {
                let fLo = max(0, f - halfW)
                let fHi = min(nFreqs - 1, f + halfW)
                for t in 0..<nFrames {
                    var maxVal: Float = -Float.infinity
                    for ff in fLo...fHi {
                        let val = src[ff * nFrames + t]
                        if val > maxVal { maxVal = val }
                    }
                    filtered[f * nFrames + t] = maxVal
                }
            }
        }

        // 3. Compute positive flux: diff[f,t] = max(0, S[f,t] - S_filtered[f,t-lag])
        // Then sum across frequency bands
        let outFrames = nFrames
        var result = [Float](repeating: 0, count: outFrames)

        mag.withUnsafeBufferPointer { src in
            for t in lag..<nFrames {
                var sum: Float = 0
                for f in 0..<nFreqs {
                    let current = src[f * nFrames + t]
                    let prevFiltered = filtered[f * nFrames + (t - lag)]
                    let diff = current - prevFiltered
                    if diff > 0 { sum += diff }
                }
                result[t] = sum
            }
        }

        return Signal(data: result, shape: [1, outFrames], sampleRate: sampleRate)
    }

    // MARK: - Complex Flux

    /// Compute complex domain onset detection function (Bello et al., 2004).
    ///
    /// Uses both magnitude and phase information from the complex STFT.
    /// For each frame, predicts phase from two previous frames and computes
    /// the complex deviation between predicted and actual spectra.
    ///
    /// - Parameters:
    ///   - signal: Input audio signal (1D).
    ///   - sr: Sample rate override. If nil, uses signal.sampleRate.
    ///   - nFFT: FFT window size. Default 2048.
    ///   - hopLength: Hop length. Default nFFT/4.
    /// - Returns: Signal with shape [1, nFrames].
    public static func complexFlux(
        signal: Signal,
        sr: Int? = nil,
        nFFT: Int = 2048,
        hopLength: Int? = nil
    ) -> Signal {
        let sampleRate = sr ?? signal.sampleRate
        let hop = hopLength ?? nFFT / 4

        // 1. Compute complex STFT
        let complexSTFT = STFT.computeComplex(
            signal: signal, nFFT: nFFT, hopLength: hop, winLength: nFFT, center: true
        )
        let nFreqs = complexSTFT.shape[0]
        let nFrames = complexSTFT.shape[1]

        guard nFrames >= 3 else {
            let n = max(nFrames, 1)
            return Signal(data: [Float](repeating: 0, count: n), shape: [1, n], sampleRate: sampleRate)
        }

        // 2. Extract magnitude and phase for all frames
        // Complex data layout: row-major [nFreqs, nFrames], interleaved
        // For freq k, frame f: real at raw[2*(k*nFrames+f)], imag at raw[2*(k*nFrames+f)+1]
        var magnitudes = [Float](repeating: 0, count: nFreqs * nFrames)
        var phases = [Float](repeating: 0, count: nFreqs * nFrames)

        complexSTFT.withUnsafeBufferPointer { raw in
            for k in 0..<nFreqs {
                for f in 0..<nFrames {
                    let idx = 2 * (k * nFrames + f)
                    let re = raw[idx]
                    let im = raw[idx + 1]
                    magnitudes[k * nFrames + f] = sqrtf(re * re + im * im)
                    phases[k * nFrames + f] = atan2f(im, re)
                }
            }
        }

        // 3. Compute complex flux for each frame
        var result = [Float](repeating: 0, count: nFrames)

        complexSTFT.withUnsafeBufferPointer { raw in
            for t in 2..<nFrames {
                var sum: Float = 0
                for k in 0..<nFreqs {
                    // Predicted phase from two previous frames
                    let phase_t1 = phases[k * nFrames + (t - 1)]
                    let phase_t2 = phases[k * nFrames + (t - 2)]
                    let predictedPhase = 2.0 * phase_t1 - phase_t2

                    // Target: mag[t] * exp(j * predictedPhase)
                    let mag_t = magnitudes[k * nFrames + t]
                    let targetRe = mag_t * cosf(predictedPhase)
                    let targetIm = mag_t * sinf(predictedPhase)

                    // Actual complex value
                    let actualIdx = 2 * (k * nFrames + t)
                    let actualRe = raw[actualIdx]
                    let actualIm = raw[actualIdx + 1]

                    // Complex deviation: |target - actual|
                    let diffRe = targetRe - actualRe
                    let diffIm = targetIm - actualIm
                    sum += sqrtf(diffRe * diffRe + diffIm * diffIm)
                }
                result[t] = sum
            }
        }

        return Signal(data: result, shape: [1, nFrames], sampleRate: sampleRate)
    }

    // MARK: - High Frequency Content

    /// Compute High Frequency Content onset detection function (Masri, 1996).
    ///
    /// Weights each frequency bin by its index, emphasizing transients
    /// which tend to have more high-frequency energy.
    ///
    /// - Parameters:
    ///   - signal: Input audio signal (1D).
    ///   - sr: Sample rate override. If nil, uses signal.sampleRate.
    ///   - nFFT: FFT window size. Default 2048.
    ///   - hopLength: Hop length. Default nFFT/4.
    /// - Returns: Signal with shape [1, nFrames].
    public static func highFrequencyContent(
        signal: Signal,
        sr: Int? = nil,
        nFFT: Int = 2048,
        hopLength: Int? = nil
    ) -> Signal {
        let sampleRate = sr ?? signal.sampleRate
        let hop = hopLength ?? nFFT / 4

        // 1. Compute magnitude spectrogram
        let mag = STFT.compute(signal: signal, nFFT: nFFT, hopLength: hop, winLength: nFFT, center: true)
        let nFreqs = mag.shape[0]
        let nFrames = mag.shape[1]

        guard nFrames > 0 else {
            return Signal(data: [0], shape: [1, 1], sampleRate: sampleRate)
        }

        // 2. HFC[t] = sum(k * |S[k,t]|^2) for k in 0..<nFreqs
        var result = [Float](repeating: 0, count: nFrames)

        mag.withUnsafeBufferPointer { src in
            // src layout: row-major [nFreqs, nFrames] => src[k * nFrames + t]
            for t in 0..<nFrames {
                var sum: Float = 0
                for k in 0..<nFreqs {
                    let val = src[k * nFrames + t]
                    sum += Float(k) * val * val
                }
                result[t] = sum
            }
        }

        return Signal(data: result, shape: [1, nFrames], sampleRate: sampleRate)
    }

    // MARK: - KL Divergence

    /// Compute KL divergence onset detection function.
    ///
    /// Kullback-Leibler divergence between consecutive normalized spectra.
    /// Each frame is normalized to a probability distribution, then KL
    /// divergence is computed. Result is half-wave rectified.
    ///
    /// - Parameters:
    ///   - signal: Input audio signal (1D).
    ///   - sr: Sample rate override. If nil, uses signal.sampleRate.
    ///   - nFFT: FFT window size. Default 2048.
    ///   - hopLength: Hop length. Default nFFT/4.
    /// - Returns: Signal with shape [1, nFrames].
    public static func klDivergence(
        signal: Signal,
        sr: Int? = nil,
        nFFT: Int = 2048,
        hopLength: Int? = nil
    ) -> Signal {
        let sampleRate = sr ?? signal.sampleRate
        let hop = hopLength ?? nFFT / 4

        // 1. Compute magnitude spectrogram
        let mag = STFT.compute(signal: signal, nFFT: nFFT, hopLength: hop, winLength: nFFT, center: true)
        let nFreqs = mag.shape[0]
        let nFrames = mag.shape[1]

        guard nFrames > 1 else {
            let n = max(nFrames, 1)
            return Signal(data: [Float](repeating: 0, count: n), shape: [1, n], sampleRate: sampleRate)
        }

        // 2. Normalize each frame to sum to 1 (probability distribution)
        // Add small epsilon to avoid division by zero and log(0)
        let epsilon: Float = 1e-10
        var normalized = [Float](repeating: 0, count: nFreqs * nFrames)

        mag.withUnsafeBufferPointer { src in
            for t in 0..<nFrames {
                // Compute sum for this frame
                var frameSum: Float = 0
                for k in 0..<nFreqs {
                    frameSum += src[k * nFrames + t]
                }
                frameSum = max(frameSum, epsilon)

                // Normalize
                for k in 0..<nFreqs {
                    normalized[k * nFrames + t] = max(src[k * nFrames + t] / frameSum, epsilon)
                }
            }
        }

        // 3. KL[t] = sum(S_norm[k,t] * log(S_norm[k,t] / S_norm[k,t-1]))
        // Half-wave rectify
        var result = [Float](repeating: 0, count: nFrames)

        for t in 1..<nFrames {
            var kl: Float = 0
            for k in 0..<nFreqs {
                let p = normalized[k * nFrames + t]
                let q = normalized[k * nFrames + (t - 1)]
                kl += p * logf(p / q)
            }
            result[t] = max(0, kl)  // Half-wave rectification
        }

        return Signal(data: result, shape: [1, nFrames], sampleRate: sampleRate)
    }

    // MARK: - Peak Picking

    /// Pick peaks from a 1-D onset strength envelope.
    ///
    /// A sample at index n is a peak if:
    /// 1. It equals the max in [n - preMax, n + postMax]
    /// 2. It >= mean of [n - preAvg, n + postAvg] + delta
    /// 3. At least `wait` samples have passed since the last detected peak
    ///
    /// This matches librosa's ``peak_pick()`` behavior.
    ///
    /// - Parameters:
    ///   - envelope: 1-D onset strength envelope (flat array).
    ///   - preMax: Number of samples before n to check for local max. Default 3.
    ///   - postMax: Number of samples after n to check for local max. Default 3.
    ///   - preAvg: Number of samples before n for threshold mean. Default 3.
    ///   - postAvg: Number of samples after n for threshold mean. Default 3.
    ///   - delta: Offset added to mean for threshold. Default 0.07.
    ///   - wait: Minimum number of samples between peaks. Default 30.
    /// - Returns: Array of peak indices.
    public static func peakPick(
        envelope: [Float],
        preMax: Int = 3,
        postMax: Int = 3,
        preAvg: Int = 3,
        postAvg: Int = 3,
        delta: Float = 0.07,
        wait: Int = 30
    ) -> [Int] {
        let n = envelope.count
        // Need enough samples for the window lookback/lookahead
        let startIdx = max(preMax, preAvg)
        let endIdx = n - max(postMax, postAvg)
        guard startIdx < endIdx else { return [] }

        var peaks: [Int] = []
        var lastPeak = -wait  // Allow first peak immediately

        for i in startIdx..<endIdx {
            // 1. Check local maximum in [i - preMax, i + postMax]
            let lo = i - preMax
            let hi = min(i + postMax, n - 1)
            var isMax = true
            for j in lo...hi {
                if envelope[j] > envelope[i] {
                    isMax = false
                    break
                }
            }
            guard isMax else { continue }

            // 2. Check threshold: envelope[i] >= mean([i-preAvg..i+postAvg]) + delta
            let avgLo = i - preAvg
            let avgHi = min(i + postAvg, n - 1)
            var sum: Float = 0
            for j in avgLo...avgHi {
                sum += envelope[j]
            }
            let mean = sum / Float(avgHi - avgLo + 1)
            guard envelope[i] >= mean + delta else { continue }

            // 3. Check wait constraint
            guard (i - lastPeak) >= wait else { continue }

            peaks.append(i)
            lastPeak = i
        }

        return peaks
    }

    // MARK: - Backtracking

    /// Snap each onset peak to the nearest preceding local minimum of energy.
    ///
    /// For each peak index, search backwards to find the closest local minimum
    /// in the onset strength envelope. This effectively backtracks each onset
    /// to the beginning of the energy rise.
    ///
    /// - Parameters:
    ///   - peaks: Onset peak indices.
    ///   - envelope: Onset strength envelope.
    /// - Returns: Adjusted onset indices snapped to preceding local minima.
    public static func backtrack(peaks: [Int], envelope: [Float]) -> [Int] {
        guard !peaks.isEmpty, !envelope.isEmpty else { return peaks }

        return peaks.map { peak in
            // Search backwards from peak to find local minimum
            var minIdx = peak
            var minVal = envelope[peak]
            var i = peak - 1
            while i >= 0 {
                if envelope[i] < minVal {
                    minVal = envelope[i]
                    minIdx = i
                } else if envelope[i] > minVal {
                    // Energy is rising; we've passed the local minimum
                    break
                }
                i -= 1
            }
            return minIdx
        }
    }

    // MARK: - Onset Detection

    /// Detect onset events from an audio signal.
    ///
    /// Computes the onset strength envelope, picks peaks, and optionally backtracks
    /// each onset to the nearest preceding local minimum of energy.
    /// Matches librosa's ``onset_detect()`` behavior.
    ///
    /// - Parameters:
    ///   - signal: Input audio signal (1D).
    ///   - sr: Sample rate. If nil, uses signal.sampleRate.
    ///   - nFFT: FFT window size. Default 2048.
    ///   - hopLength: Hop length. Default 512.
    ///   - nMels: Number of mel bands. Default 128.
    ///   - fmin: Minimum frequency. Default 0.
    ///   - fmax: Maximum frequency. If nil, uses sr/2.
    ///   - center: Center-pad signal. Default true.
    ///   - preMax: Samples before n for local max check. Default 3.
    ///   - postMax: Samples after n for local max check. Default 3.
    ///   - preAvg: Samples before n for mean threshold. Default 3.
    ///   - postAvg: Samples after n for mean threshold. Default 3.
    ///   - delta: Threshold offset above mean. Default 0.07.
    ///   - wait: Minimum samples between peaks. Default 30.
    ///   - doBacktrack: If true, snap peaks to preceding local minima. Default false.
    /// - Returns: Signal containing onset frame indices as Float values (1D).
    public static func detectOnsets(
        signal: Signal,
        sr: Int? = nil,
        nFFT: Int = 2048,
        hopLength: Int? = nil,
        nMels: Int = 128,
        fmin: Float = 0,
        fmax: Float? = nil,
        center: Bool = true,
        preMax: Int = 3,
        postMax: Int = 3,
        preAvg: Int = 3,
        postAvg: Int = 3,
        delta: Float = 0.07,
        wait: Int = 30,
        doBacktrack: Bool = false
    ) -> Signal {
        let sampleRate = sr ?? signal.sampleRate
        let hop = hopLength ?? 512

        // 1. Compute onset strength envelope
        let oenv = onsetStrength(
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

        // 2. Extract raw envelope data (shape [1, nFrames])
        let nFrames = oenv.shape.count > 1 ? oenv.shape[1] : oenv.shape[0]
        var envData = [Float](repeating: 0, count: nFrames)
        oenv.withUnsafeBufferPointer { buf in
            for i in 0..<nFrames {
                envData[i] = buf[i]
            }
        }

        // 3. Normalize envelope to [0, 1]
        var envMax: Float = 0
        for v in envData {
            if v > envMax { envMax = v }
        }
        if envMax > 0 {
            for i in 0..<envData.count {
                envData[i] /= envMax
            }
        }

        // 4. Peak pick
        var peaks = peakPick(
            envelope: envData,
            preMax: preMax,
            postMax: postMax,
            preAvg: preAvg,
            postAvg: postAvg,
            delta: delta,
            wait: wait
        )

        // 5. Optional backtracking
        if doBacktrack {
            peaks = backtrack(peaks: peaks, envelope: envData)
        }

        // 6. Return onset frame indices as a Signal of Floats
        let frameIndices = peaks.map { Float($0) }
        return Signal(
            data: frameIndices.isEmpty ? [Float]() : frameIndices,
            shape: [frameIndices.count],
            sampleRate: sampleRate
        )
    }
}
