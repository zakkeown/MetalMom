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
