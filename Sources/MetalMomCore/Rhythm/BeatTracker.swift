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
        let state = Profiler.shared.begin("BeatTrack")
        defer { Profiler.shared.end("BeatTrack", state) }
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

    /// Compute unnormalized autocorrelation of a signal using FFT (O(n log n)).
    ///
    /// Zero-pads to next power-of-2 >= 2n, computes power spectrum via
    /// forward FFT, then inverse FFT to obtain all lags in one pass.
    private static func autocorrelation(_ x: [Float]) -> [Float] {
        let n = x.count
        guard n > 0 else { return [] }

        // Pad to next power of 2 >= 2*n for linear (non-circular) autocorrelation
        var fftSize = 1
        while fftSize < 2 * n { fftSize <<= 1 }
        let halfFFT = fftSize / 2
        let log2n = vDSP_Length(log2(Double(fftSize)))

        guard let fftSetup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else {
            return [Float](repeating: 0, count: n)
        }
        defer { vDSP_destroy_fftsetup(fftSetup) }

        // Zero-pad input
        var padded = [Float](repeating: 0, count: fftSize)
        for i in 0..<n { padded[i] = x[i] }

        // Pack into split complex using stable pointers
        let realPtr = UnsafeMutablePointer<Float>.allocate(capacity: halfFFT)
        let imagPtr = UnsafeMutablePointer<Float>.allocate(capacity: halfFFT)
        realPtr.initialize(repeating: 0, count: halfFFT)
        imagPtr.initialize(repeating: 0, count: halfFFT)
        defer {
            realPtr.deallocate()
            imagPtr.deallocate()
        }
        var splitComplex = DSPSplitComplex(realp: realPtr, imagp: imagPtr)

        padded.withUnsafeBufferPointer { paddedBuf in
            paddedBuf.baseAddress!.withMemoryRebound(to: DSPComplex.self, capacity: halfFFT) { complexPtr in
                vDSP_ctoz(complexPtr, 2, &splitComplex, 1, vDSP_Length(halfFFT))
            }
        }

        // Forward FFT
        vDSP_fft_zrip(fftSetup, &splitComplex, 1, log2n, FFTDirection(kFFTDirection_Forward))

        // Compute power spectrum: real = real^2 + imag^2, imag = 0
        // In the packed format, realp[0] = DC component, imagp[0] = Nyquist component
        // (both are purely real). Save them before vDSP_zvmags overwrites bin 0.
        let dcVal = splitComplex.realp[0]
        let nyquistVal = splitComplex.imagp[0]

        // Compute squared magnitudes for ALL bins (overwrites realp in-place)
        vDSP_zvmags(&splitComplex, 1, splitComplex.realp, 1, vDSP_Length(halfFFT))

        // Fix bin 0: DC and Nyquist are packed as (DC, Nyquist) not (real, imag),
        // so their power is just the square of each, not real^2 + imag^2
        splitComplex.realp[0] = dcVal * dcVal
        splitComplex.imagp[0] = nyquistVal * nyquistVal

        // Clear imaginary part for bins 1..<halfFFT
        vDSP_vclr(splitComplex.imagp + 1, 1, vDSP_Length(halfFFT - 1))

        // Inverse FFT
        vDSP_fft_zrip(fftSetup, &splitComplex, 1, log2n, FFTDirection(kFFTDirection_Inverse))

        // Unpack result
        var result = [Float](repeating: 0, count: fftSize)
        result.withUnsafeMutableBufferPointer { resBuf in
            resBuf.baseAddress!.withMemoryRebound(to: DSPComplex.self, capacity: halfFFT) { complexPtr in
                vDSP_ztoc(&splitComplex, 1, complexPtr, 2, vDSP_Length(halfFFT))
            }
        }

        // Normalize: vDSP forward*inverse scales by fftSize/2, and we want
        // relative values only (estimateTempo uses peak-finding, not absolute magnitudes)
        var scale = 1.0 / Float(fftSize * 2)
        vDSP_vsmul(result, 1, &scale, &result, 1, vDSP_Length(fftSize))

        // Return only the first n lags (0..<n)
        return Array(result.prefix(n))
    }

    // MARK: - Predominant Local Pulse (PLP)

    /// Compute Predominant Local Pulse (Grosche & Müller 2011).
    ///
    /// Estimates a local pulse curve from the tempogram by finding the
    /// dominant tempo frequency in each frame of a windowed FFT of the
    /// onset envelope, generating phased cosines, overlap-adding, and
    /// half-wave rectifying.
    ///
    /// - Parameters:
    ///   - signal: Input audio signal (1D).
    ///   - sr: Sample rate override. If nil, uses signal.sampleRate.
    ///   - hopLength: Hop length for onset envelope. Default 512.
    ///   - nFFT: FFT window size for onset envelope. Default 2048.
    ///   - nMels: Number of mel bands for onset envelope. Default 128.
    ///   - fmin: Minimum frequency for mel filterbank. Default 0.
    ///   - fmax: Maximum frequency. If nil, uses sr/2.
    ///   - center: Center-pad onset envelope windows. Default true.
    ///   - winLength: Window length for local tempogram analysis. Default 384.
    ///   - tempoMin: Minimum tempo in BPM. Default 30.
    ///   - tempoMax: Maximum tempo in BPM. Default 300.
    /// - Returns: 1D Signal of local pulse strength (same length as onset envelope).
    public static func plp(
        signal: Signal,
        sr: Int? = nil,
        hopLength: Int? = nil,
        nFFT: Int = 2048,
        nMels: Int = 128,
        fmin: Float = 0,
        fmax: Float? = nil,
        center: Bool = true,
        winLength: Int = 384,
        tempoMin: Float = 30,
        tempoMax: Float = 300
    ) -> Signal {
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

        let envLen = oenv.shape.count > 1 ? oenv.shape[1] : oenv.shape[0]
        guard envLen > 0 else {
            return Signal(data: [], shape: [0], sampleRate: sampleRate)
        }

        var envData = [Float](repeating: 0, count: envLen)
        oenv.withUnsafeBufferPointer { buf in
            for i in 0..<envLen { envData[i] = buf[i] }
        }

        // Check if envelope has any energy; if not, return zeros
        var envMax: Float = 0
        vDSP_maxv(envData, 1, &envMax, vDSP_Length(envLen))
        if envMax <= 1e-10 {
            return Signal(data: [Float](repeating: 0, count: envLen),
                          shape: [envLen], sampleRate: sampleRate)
        }

        // 2. Pad with winLength/2 zeros on each side for center alignment
        let halfWin = winLength / 2
        let padded = [Float](repeating: 0, count: halfWin)
                     + envData
                     + [Float](repeating: 0, count: halfWin)

        // 3. Build Hann window
        var hann = [Float](repeating: 0, count: winLength)
        let hannScale = 2.0 * Float.pi / Float(winLength)
        for i in 0..<winLength {
            hann[i] = 0.5 * (1.0 - cos(hannScale * Float(i)))
        }

        // 4. Determine FFT size (next power of 2 >= winLength)
        var fftSize = 1
        while fftSize < winLength { fftSize <<= 1 }
        let log2n = vDSP_Length(log2(Double(fftSize)))
        guard let fftSetup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else {
            return Signal(data: [Float](repeating: 0, count: envLen),
                          shape: [envLen], sampleRate: sampleRate)
        }
        defer { vDSP_destroy_fftsetup(fftSetup) }

        let halfFFT = fftSize / 2

        // 5. Convert tempo range to frequency bin range.
        // The FFT of the onset envelope has a "sample rate" of (sr/hop) Hz.
        // Frequency resolution: df = (sr/hop) / fftSize
        // BPM -> Hz: f = bpm / 60
        // bin = f / df = bpm / 60 * fftSize / (sr/hop) = bpm * fftSize * hop / (60 * sr)
        let bpmToFreqBin = Float(fftSize) * Float(hop) / (60.0 * Float(sampleRate))
        let minBin = max(1, Int(floor(tempoMin * bpmToFreqBin)))
        let maxBin = min(halfFFT - 1, Int(ceil(tempoMax * bpmToFreqBin)))

        guard minBin <= maxBin else {
            return Signal(data: [Float](repeating: 0, count: envLen),
                          shape: [envLen], sampleRate: sampleRate)
        }

        // 6. Overlap-add accumulator and normalization
        var output = [Float](repeating: 0, count: envLen + winLength)

        let nFrames = envLen

        // Reusable FFT buffers
        var realBuf = [Float](repeating: 0, count: halfFFT)
        var imagBuf = [Float](repeating: 0, count: halfFFT)

        for frame in 0..<nFrames {
            let centerIdx = frame + halfWin
            let startIdx = centerIdx - halfWin

            // Extract and window the segment
            var segment = [Float](repeating: 0, count: fftSize)
            for i in 0..<winLength {
                let pIdx = startIdx + i
                if pIdx >= 0 && pIdx < padded.count {
                    segment[i] = padded[pIdx] * hann[i]
                }
            }

            // FFT to get complex spectrum
            var peakBin = minBin
            var peakMag: Float = 0
            var peakPhase: Float = 0

            segment.withUnsafeMutableBufferPointer { segBuf in
                realBuf.withUnsafeMutableBufferPointer { rBuf in
                    imagBuf.withUnsafeMutableBufferPointer { iBuf in
                        var splitComplex = DSPSplitComplex(
                            realp: rBuf.baseAddress!,
                            imagp: iBuf.baseAddress!
                        )

                        segBuf.baseAddress!.withMemoryRebound(to: DSPComplex.self, capacity: halfFFT) { complexPtr in
                            vDSP_ctoz(complexPtr, 2, &splitComplex, 1, vDSP_Length(halfFFT))
                        }

                        vDSP_fft_zrip(fftSetup, &splitComplex, 1, log2n, FFTDirection(kFFTDirection_Forward))

                        // Find peak bin in valid BPM range
                        for k in minBin...maxBin {
                            if k < halfFFT {
                                let re = splitComplex.realp[k]
                                let im = splitComplex.imagp[k]
                                let mag = re * re + im * im  // squared magnitude is fine for comparison
                                if mag > peakMag {
                                    peakMag = mag
                                    peakBin = k
                                    peakPhase = atan2(im, re)
                                }
                            }
                        }
                    }
                }
            }

            // Generate a windowed cosine at the dominant frequency with the correct phase
            // The frequency in cycles per sample (of the onset envelope) is:
            //   freq = peakBin / fftSize
            // So the angular frequency is: omega = 2*pi * peakBin / fftSize
            let omega = 2.0 * Float.pi * Float(peakBin) / Float(fftSize)

            // The phase from the FFT corresponds to the phase at the center of the window.
            // We construct a cosine centered at the window center.
            for i in 0..<winLength {
                let t = Float(i - halfWin)  // time relative to center
                let cosVal = cos(omega * t + peakPhase)
                let windowed = cosVal * hann[i]

                // Overlap-add into output at position (frame)
                let outIdx = frame + i
                if outIdx >= 0 && outIdx < output.count {
                    output[outIdx] += windowed
                }
            }
        }

        // 7. Extract the relevant portion (aligned with onset envelope frames)
        // The overlap-add starts at frame 0 and the center of each window is at frame + halfWin.
        // We want output aligned so that index i corresponds to onset envelope frame i.
        // The OLA for frame 0 writes to indices 0..<winLength, with center at halfWin.
        // So the pulse for frame f is at output[f + halfWin].
        // Actually, since we wrote to output[frame + i] where i goes 0..<winLength,
        // and the center of the window is at i=halfWin, the pulse center for frame f
        // is at output[f + halfWin]. But we want output[f] to correspond to frame f.
        // So we shift by halfWin.
        var pulse = [Float](repeating: 0, count: envLen)
        for i in 0..<envLen {
            let idx = i + halfWin
            if idx < output.count {
                pulse[i] = output[idx]
            }
        }

        // 8. Half-wave rectify: clip negative values to 0
        var zero: Float = 0
        vDSP_vthres(pulse, 1, &zero, &pulse, 1, vDSP_Length(envLen))

        // 9. Normalize to [0, 1]
        var maxVal: Float = 0
        vDSP_maxv(pulse, 1, &maxVal, vDSP_Length(envLen))
        if maxVal > 0 {
            var invMax = 1.0 / maxVal
            vDSP_vsmul(pulse, 1, &invMax, &pulse, 1, vDSP_Length(envLen))
        }

        return Signal(data: pulse, shape: [envLen], sampleRate: sampleRate)
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

        // Precompute log table for intervals 1..<maxInterval to avoid per-iteration log() calls
        let maxInterval = min(n, periodInt + window + 1)
        var logTable = [Float](repeating: 0, count: maxInterval + 1)
        for i in 1...maxInterval {
            logTable[i] = log(Float(i))
        }

        for t in 1..<n {
            // Search window for predecessors
            let searchLo = max(0, t - periodInt - window)
            let searchHi = max(0, min(t - 1, t - periodInt + window))

            guard searchLo <= searchHi else { continue }

            var bestPrev = searchLo
            var bestVal: Float = -Float.infinity

            for tau in searchLo...searchHi {
                let interval = t - tau
                guard interval > 0 else { continue }
                let logInterval = interval <= maxInterval ? logTable[interval] : log(Float(interval))
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
