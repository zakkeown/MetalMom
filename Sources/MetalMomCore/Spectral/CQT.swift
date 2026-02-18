import Foundation
import Accelerate

/// Constant-Q Transform (CQT), Variable-Q Transform (VQT), and Hybrid CQT.
///
/// The CQT produces a frequency-domain representation with logarithmically-spaced
/// frequency bins, which better matches human pitch perception than the linearly-spaced
/// STFT. Each bin has a constant quality factor Q = frequency / bandwidth.
///
/// Implementation uses the filterbank approach: compute STFT, then multiply by a CQT
/// filterbank that maps linear STFT bins to logarithmic CQT bins. This is analogous
/// to how mel spectrograms work.
public enum CQT {

    // MARK: - CQT Filterbank

    /// Generate a CQT filterbank matrix.
    ///
    /// Returns a `Signal` with shape `[nBins, nFFT/2+1]` (row-major),
    /// where each row maps linear STFT frequency bins to a single CQT bin
    /// via triangular interpolation.
    ///
    /// - Parameters:
    ///   - sr: Audio sample rate in Hz.
    ///   - nFFT: FFT size (determines frequency resolution of the underlying STFT).
    ///   - fMin: Lowest CQT frequency in Hz. Default 32.70 (C1).
    ///   - nBins: Number of CQT bins. Default 84 (7 octaves * 12 bins/octave).
    ///   - binsPerOctave: Number of bins per octave. Default 12.
    /// - Returns: A `Signal` of shape `[nBins, nFFT/2+1]`.
    public static func cqtFilterbank(
        sr: Int,
        nFFT: Int,
        fMin: Float = 32.70,
        nBins: Int = 84,
        binsPerOctave: Int = 12
    ) -> Signal {
        let nFreqs = nFFT / 2 + 1
        let freqPerBin = Float(sr) / Float(nFFT)

        // Compute CQT center frequencies
        let cqtFreqs = (0..<nBins).map { k in
            fMin * powf(2.0, Float(k) / Float(binsPerOctave))
        }

        // Quality factor: Q = 1 / (2^(1/B) - 1)
        let Q = 1.0 / (powf(2.0, 1.0 / Float(binsPerOctave)) - 1.0)

        // Build filterbank using triangular interpolation
        // Each CQT bin k has center frequency cqtFreqs[k] and bandwidth cqtFreqs[k] / Q.
        // We create triangular filters in the linear frequency domain.
        var weights = [Float](repeating: 0, count: nBins * nFreqs)

        for k in 0..<nBins {
            let centerFreq = cqtFreqs[k]
            let naturalBW = centerFreq / Q

            // Ensure bandwidth spans at least 2 FFT bins so that every CQT bin
            // captures energy from the nearest STFT bin(s), even at low frequencies
            // where the CQT bandwidth is narrower than the STFT resolution.
            let bandwidth = max(naturalBW, 2.0 * freqPerBin)

            // Lower and upper edges of the triangular filter
            let fLow = centerFreq - bandwidth / 2.0
            let fHigh = centerFreq + bandwidth / 2.0

            for j in 0..<nFreqs {
                let freq = Float(j) * freqPerBin

                if freq >= fLow && freq <= centerFreq && centerFreq > fLow {
                    // Rising edge
                    weights[k * nFreqs + j] = (freq - fLow) / (centerFreq - fLow)
                } else if freq > centerFreq && freq <= fHigh && fHigh > centerFreq {
                    // Falling edge
                    weights[k * nFreqs + j] = (fHigh - freq) / (fHigh - centerFreq)
                }
            }

            // Normalize: sum of weights should be 1 for energy preservation
            var rowSum: Float = 0
            for j in 0..<nFreqs {
                rowSum += weights[k * nFreqs + j]
            }
            if rowSum > 0 {
                for j in 0..<nFreqs {
                    weights[k * nFreqs + j] /= rowSum
                }
            }
        }

        return Signal(data: weights, shape: [nBins, nFreqs], sampleRate: sr)
    }

    // MARK: - VQT Filterbank

    /// Generate a Variable-Q filterbank matrix.
    ///
    /// The VQT filterbank is similar to CQT but the quality factor varies with frequency:
    /// `Q_k = Q_base / (1 + gamma / freq_k)`. The gamma parameter controls how much
    /// bandwidth increases at low frequencies, providing better time resolution there.
    ///
    /// - Parameters:
    ///   - sr: Audio sample rate in Hz.
    ///   - nFFT: FFT size.
    ///   - fMin: Lowest frequency in Hz. Default 32.70 (C1).
    ///   - nBins: Number of bins. Default 84.
    ///   - binsPerOctave: Bins per octave. Default 12.
    ///   - gamma: VQT gamma parameter. Controls bandwidth variation. Default 0 (= standard CQT).
    /// - Returns: A `Signal` of shape `[nBins, nFFT/2+1]`.
    public static func vqtFilterbank(
        sr: Int,
        nFFT: Int,
        fMin: Float = 32.70,
        nBins: Int = 84,
        binsPerOctave: Int = 12,
        gamma: Float = 0.0
    ) -> Signal {
        let nFreqs = nFFT / 2 + 1
        let freqPerBin = Float(sr) / Float(nFFT)

        // CQT center frequencies
        let cqtFreqs = (0..<nBins).map { k in
            fMin * powf(2.0, Float(k) / Float(binsPerOctave))
        }

        // Base quality factor
        let Qbase = 1.0 / (powf(2.0, 1.0 / Float(binsPerOctave)) - 1.0)

        var weights = [Float](repeating: 0, count: nBins * nFreqs)

        for k in 0..<nBins {
            let centerFreq = cqtFreqs[k]

            // Variable Q: Q_k = Q_base / (1 + gamma / freq_k)
            let Qk = Qbase / (1.0 + gamma / centerFreq)
            let naturalBW = centerFreq / Qk

            // Ensure bandwidth spans at least 2 FFT bins
            let bandwidth = max(naturalBW, 2.0 * freqPerBin)

            let fLow = centerFreq - bandwidth / 2.0
            let fHigh = centerFreq + bandwidth / 2.0

            for j in 0..<nFreqs {
                let freq = Float(j) * freqPerBin

                if freq >= fLow && freq <= centerFreq && centerFreq > fLow {
                    weights[k * nFreqs + j] = (freq - fLow) / (centerFreq - fLow)
                } else if freq > centerFreq && freq <= fHigh && fHigh > centerFreq {
                    weights[k * nFreqs + j] = (fHigh - freq) / (fHigh - centerFreq)
                }
            }

            // Normalize row
            var rowSum: Float = 0
            for j in 0..<nFreqs {
                rowSum += weights[k * nFreqs + j]
            }
            if rowSum > 0 {
                for j in 0..<nFreqs {
                    weights[k * nFreqs + j] /= rowSum
                }
            }
        }

        return Signal(data: weights, shape: [nBins, nFreqs], sampleRate: sr)
    }

    // MARK: - CQT Compute

    /// Compute the Constant-Q Transform magnitude spectrogram.
    ///
    /// Returns a `Signal` with shape `[nBins, nFrames]`, row-major.
    ///
    /// - Parameters:
    ///   - signal: Input audio signal (1D).
    ///   - sr: Sample rate override. If `nil`, uses `signal.sampleRate`.
    ///   - hopLength: Hop length in samples. Default `nFFT / 4`.
    ///   - fMin: Lowest CQT frequency in Hz. Default 32.70 (C1).
    ///   - fMax: Highest CQT frequency in Hz. If `nil`, uses `sr / 2`.
    ///   - binsPerOctave: Number of bins per octave. Default 12.
    ///   - nFFT: FFT size for the underlying STFT. If `nil`, auto-selected.
    ///   - center: If `true`, pad signal so frames are centred. Default `true`.
    /// - Returns: CQT magnitude spectrogram with shape `[nBins, nFrames]`.
    public static func compute(
        signal: Signal,
        sr: Int? = nil,
        hopLength: Int? = nil,
        fMin: Float = 32.70,
        fMax: Float? = nil,
        binsPerOctave: Int = 12,
        nFFT: Int? = nil,
        center: Bool = true
    ) -> Signal {
        let sampleRate = sr ?? signal.sampleRate
        let fMaxActual = fMax ?? Float(sampleRate) / 2.0

        // Compute number of bins needed to cover fMin to fMax
        let nBins = computeNBins(fMin: fMin, fMax: fMaxActual, binsPerOctave: binsPerOctave)
        guard nBins > 0 else {
            return Signal(data: [], shape: [0, 0], sampleRate: sampleRate)
        }

        // Auto-select FFT size: must resolve the lowest CQT frequency
        let fftSize = selectFFTSize(nFFT: nFFT, fMin: fMin, binsPerOctave: binsPerOctave, sr: sampleRate)

        let hop = hopLength ?? fftSize / 4

        // 1. Compute STFT magnitude spectrogram: shape [nFreqs, nFrames]
        let stftMag = STFT.compute(
            signal: signal,
            nFFT: fftSize,
            hopLength: hop,
            winLength: fftSize,
            center: center
        )

        let nFreqs = stftMag.shape[0]
        let nFrames = stftMag.shape[1]

        guard nFrames > 0 else {
            return Signal(data: [], shape: [nBins, 0], sampleRate: sampleRate)
        }

        // 2. Square for power spectrogram
        let stftCount = stftMag.count
        let poweredPtr = UnsafeMutablePointer<Float>.allocate(capacity: stftCount)
        defer { poweredPtr.deallocate() }

        stftMag.withUnsafeBufferPointer { src in
            vDSP_vsq(src.baseAddress!, 1, poweredPtr, 1, vDSP_Length(stftCount))
        }

        // 3. Get CQT filterbank: shape [nBins, nFreqs]
        let cqtFB = cqtFilterbank(
            sr: sampleRate,
            nFFT: fftSize,
            fMin: fMin,
            nBins: nBins,
            binsPerOctave: binsPerOctave
        )

        // 4. Matrix multiply: cqtFB [nBins, nFreqs] @ powered [nFreqs, nFrames] = [nBins, nFrames]
        let outCount = nBins * nFrames
        let outPtr = UnsafeMutablePointer<Float>.allocate(capacity: outCount)
        outPtr.initialize(repeating: 0, count: outCount)

        cqtFB.withUnsafeBufferPointer { fbBuf in
            vDSP_mmul(
                fbBuf.baseAddress!, 1,
                poweredPtr, 1,
                outPtr, 1,
                vDSP_Length(nBins),
                vDSP_Length(nFrames),
                vDSP_Length(nFreqs)
            )
        }

        // 5. Take square root to get magnitude (we applied power=2 above)
        var count = Int32(outCount)
        vvsqrtf(outPtr, outPtr, &count)

        // Clamp any NaN from sqrt of tiny negative values
        for i in 0..<outCount {
            if outPtr[i].isNaN { outPtr[i] = 0 }
        }

        let outBuffer = UnsafeMutableBufferPointer(start: outPtr, count: outCount)
        return Signal(taking: outBuffer, shape: [nBins, nFrames], sampleRate: sampleRate)
    }

    // MARK: - VQT Compute

    /// Compute the Variable-Q Transform magnitude spectrogram.
    ///
    /// Like CQT but with a variable quality factor controlled by gamma.
    /// Higher gamma values give wider bandwidth (better time resolution) at low frequencies.
    ///
    /// - Parameters:
    ///   - signal: Input audio signal (1D).
    ///   - sr: Sample rate override. If `nil`, uses `signal.sampleRate`.
    ///   - hopLength: Hop length in samples. Default `nFFT / 4`.
    ///   - fMin: Lowest frequency in Hz. Default 32.70 (C1).
    ///   - fMax: Highest frequency in Hz. If `nil`, uses `sr / 2`.
    ///   - binsPerOctave: Number of bins per octave. Default 12.
    ///   - gamma: VQT gamma parameter controlling bandwidth variation. Default 0 (= standard CQT).
    ///   - nFFT: FFT size. If `nil`, auto-selected.
    ///   - center: If `true`, pad signal. Default `true`.
    /// - Returns: VQT magnitude spectrogram with shape `[nBins, nFrames]`.
    public static func vqt(
        signal: Signal,
        sr: Int? = nil,
        hopLength: Int? = nil,
        fMin: Float = 32.70,
        fMax: Float? = nil,
        binsPerOctave: Int = 12,
        gamma: Float = 0.0,
        nFFT: Int? = nil,
        center: Bool = true
    ) -> Signal {
        let sampleRate = sr ?? signal.sampleRate
        let fMaxActual = fMax ?? Float(sampleRate) / 2.0

        let nBins = computeNBins(fMin: fMin, fMax: fMaxActual, binsPerOctave: binsPerOctave)
        guard nBins > 0 else {
            return Signal(data: [], shape: [0, 0], sampleRate: sampleRate)
        }

        let fftSize = selectFFTSize(nFFT: nFFT, fMin: fMin, binsPerOctave: binsPerOctave, sr: sampleRate)
        let hop = hopLength ?? fftSize / 4

        // 1. Compute STFT magnitude spectrogram
        let stftMag = STFT.compute(
            signal: signal,
            nFFT: fftSize,
            hopLength: hop,
            winLength: fftSize,
            center: center
        )

        let nFreqs = stftMag.shape[0]
        let nFrames = stftMag.shape[1]

        guard nFrames > 0 else {
            return Signal(data: [], shape: [nBins, 0], sampleRate: sampleRate)
        }

        // 2. Square for power spectrogram
        let stftCount = stftMag.count
        let poweredPtr = UnsafeMutablePointer<Float>.allocate(capacity: stftCount)
        defer { poweredPtr.deallocate() }

        stftMag.withUnsafeBufferPointer { src in
            vDSP_vsq(src.baseAddress!, 1, poweredPtr, 1, vDSP_Length(stftCount))
        }

        // 3. Get VQT filterbank: shape [nBins, nFreqs]
        let vqtFB = vqtFilterbank(
            sr: sampleRate,
            nFFT: fftSize,
            fMin: fMin,
            nBins: nBins,
            binsPerOctave: binsPerOctave,
            gamma: gamma
        )

        // 4. Matrix multiply
        let outCount = nBins * nFrames
        let outPtr = UnsafeMutablePointer<Float>.allocate(capacity: outCount)
        outPtr.initialize(repeating: 0, count: outCount)

        vqtFB.withUnsafeBufferPointer { fbBuf in
            vDSP_mmul(
                fbBuf.baseAddress!, 1,
                poweredPtr, 1,
                outPtr, 1,
                vDSP_Length(nBins),
                vDSP_Length(nFrames),
                vDSP_Length(nFreqs)
            )
        }

        // 5. Take square root for magnitude
        var sqrtCount = Int32(outCount)
        vvsqrtf(outPtr, outPtr, &sqrtCount)

        for i in 0..<outCount {
            if outPtr[i].isNaN { outPtr[i] = 0 }
        }

        let outBuffer = UnsafeMutableBufferPointer(start: outPtr, count: outCount)
        return Signal(taking: outBuffer, shape: [nBins, nFrames], sampleRate: sampleRate)
    }

    // MARK: - Hybrid CQT

    /// Compute the Hybrid CQT magnitude spectrogram.
    ///
    /// Uses CQT filterbank for bins below a transition frequency and passes through
    /// linear STFT bins above it. The transition frequency is determined by where the
    /// CQT bins become dense enough to match STFT resolution.
    ///
    /// - Parameters:
    ///   - signal: Input audio signal (1D).
    ///   - sr: Sample rate override. If `nil`, uses `signal.sampleRate`.
    ///   - hopLength: Hop length in samples. Default `nFFT / 4`.
    ///   - fMin: Lowest CQT frequency in Hz. Default 32.70 (C1).
    ///   - fMax: Highest frequency in Hz. If `nil`, uses `sr / 2`.
    ///   - binsPerOctave: Number of bins per octave. Default 12.
    ///   - nFFT: FFT size. If `nil`, auto-selected.
    ///   - center: If `true`, pad signal. Default `true`.
    /// - Returns: Hybrid CQT magnitude spectrogram with shape `[nBins, nFrames]`,
    ///   where `nBins = cqtBins + linearBins`.
    public static func hybridCQT(
        signal: Signal,
        sr: Int? = nil,
        hopLength: Int? = nil,
        fMin: Float = 32.70,
        fMax: Float? = nil,
        binsPerOctave: Int = 12,
        nFFT: Int? = nil,
        center: Bool = true
    ) -> Signal {
        let sampleRate = sr ?? signal.sampleRate
        let fMaxActual = fMax ?? Float(sampleRate) / 2.0

        let fftSize = selectFFTSize(nFFT: nFFT, fMin: fMin, binsPerOctave: binsPerOctave, sr: sampleRate)
        let hop = hopLength ?? fftSize / 4
        let freqPerBin = Float(sampleRate) / Float(fftSize)

        // Determine transition frequency: where CQT bandwidth >= STFT bin spacing
        // CQT bandwidth at freq f = f / Q, STFT bin spacing = sr / nFFT
        let Q = 1.0 / (powf(2.0, 1.0 / Float(binsPerOctave)) - 1.0)
        let transitionFreq = min(Q * freqPerBin, fMaxActual)

        // CQT bins: from fMin up to transitionFreq
        let nCQTBins = computeNBins(fMin: fMin, fMax: transitionFreq, binsPerOctave: binsPerOctave)

        // Linear bins: STFT bins from transitionFreq to fMax
        let startLinearBin = max(1, Int(ceilf(transitionFreq / freqPerBin)))
        let endLinearBin = min(fftSize / 2, Int(floorf(fMaxActual / freqPerBin)))
        let nLinearBins = max(0, endLinearBin - startLinearBin + 1)

        let totalBins = nCQTBins + nLinearBins

        guard totalBins > 0 else {
            return Signal(data: [], shape: [0, 0], sampleRate: sampleRate)
        }

        // 1. Compute STFT magnitude spectrogram
        let stftMag = STFT.compute(
            signal: signal,
            nFFT: fftSize,
            hopLength: hop,
            winLength: fftSize,
            center: center
        )

        let nFreqs = stftMag.shape[0]
        let nFrames = stftMag.shape[1]

        guard nFrames > 0 else {
            return Signal(data: [], shape: [totalBins, 0], sampleRate: sampleRate)
        }

        // Output buffer: [totalBins, nFrames]
        let outCount = totalBins * nFrames
        let outPtr = UnsafeMutablePointer<Float>.allocate(capacity: outCount)
        outPtr.initialize(repeating: 0, count: outCount)

        // 2. CQT part: filterbank @ power spectrogram, then sqrt
        if nCQTBins > 0 {
            let stftCount = stftMag.count
            let poweredPtr = UnsafeMutablePointer<Float>.allocate(capacity: stftCount)
            defer { poweredPtr.deallocate() }

            stftMag.withUnsafeBufferPointer { src in
                vDSP_vsq(src.baseAddress!, 1, poweredPtr, 1, vDSP_Length(stftCount))
            }

            let cqtFB = cqtFilterbank(
                sr: sampleRate,
                nFFT: fftSize,
                fMin: fMin,
                nBins: nCQTBins,
                binsPerOctave: binsPerOctave
            )

            let cqtCount = nCQTBins * nFrames
            let cqtPtr = UnsafeMutablePointer<Float>.allocate(capacity: cqtCount)
            cqtPtr.initialize(repeating: 0, count: cqtCount)
            defer { cqtPtr.deallocate() }

            cqtFB.withUnsafeBufferPointer { fbBuf in
                vDSP_mmul(
                    fbBuf.baseAddress!, 1,
                    poweredPtr, 1,
                    cqtPtr, 1,
                    vDSP_Length(nCQTBins),
                    vDSP_Length(nFrames),
                    vDSP_Length(nFreqs)
                )
            }

            // sqrt for magnitude
            var sqrtCount = Int32(cqtCount)
            vvsqrtf(cqtPtr, cqtPtr, &sqrtCount)

            // Copy CQT rows into output
            for i in 0..<cqtCount {
                outPtr[i] = cqtPtr[i].isNaN ? 0 : cqtPtr[i]
            }
        }

        // 3. Linear part: copy STFT magnitude rows directly
        if nLinearBins > 0 {
            stftMag.withUnsafeBufferPointer { src in
                for binIdx in 0..<nLinearBins {
                    let srcRow = startLinearBin + binIdx
                    let dstRow = nCQTBins + binIdx
                    // Row srcRow of stftMag: src[srcRow * nFrames ..< srcRow * nFrames + nFrames]
                    // Row dstRow of output:  outPtr[dstRow * nFrames ..< dstRow * nFrames + nFrames]
                    let srcOffset = srcRow * nFrames
                    let dstOffset = dstRow * nFrames
                    for f in 0..<nFrames {
                        outPtr[dstOffset + f] = src[srcOffset + f]
                    }
                }
            }
        }

        let outBuffer = UnsafeMutableBufferPointer(start: outPtr, count: outCount)
        return Signal(taking: outBuffer, shape: [totalBins, nFrames], sampleRate: sampleRate)
    }

    // MARK: - Private Helpers

    /// Compute number of CQT bins to span from fMin to fMax.
    private static func computeNBins(fMin: Float, fMax: Float, binsPerOctave: Int) -> Int {
        guard fMax > fMin && fMin > 0 else { return 0 }
        let nOctaves = log2f(fMax / fMin)
        return max(1, Int(ceilf(nOctaves * Float(binsPerOctave))))
    }

    /// Select an appropriate FFT size that can resolve the lowest CQT frequency.
    ///
    /// The FFT size must be large enough that the frequency bin spacing (sr / nFFT)
    /// is smaller than the bandwidth of the lowest CQT bin.
    private static func selectFFTSize(nFFT: Int?, fMin: Float, binsPerOctave: Int, sr: Int) -> Int {
        if let nFFT = nFFT {
            return nFFT
        }

        // Q factor
        let Q = 1.0 / (powf(2.0, 1.0 / Float(binsPerOctave)) - 1.0)

        // Minimum window length to resolve fMin with quality Q
        let minWindow = Int(ceilf(Q * Float(sr) / fMin))

        // Round up to next power of 2
        var fftSize = 1
        while fftSize < minWindow {
            fftSize *= 2
        }

        // Clamp to reasonable range
        return max(2048, min(fftSize, 65536))
    }
}
