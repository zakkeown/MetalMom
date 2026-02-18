import Accelerate
import Foundation

/// Semitone bandpass filterbank using cascaded biquad filters.
///
/// Isolates individual musical notes by applying bandpass filters centered
/// at each semitone frequency. Uses the Audio EQ Cookbook biquad formulas
/// with constant-Q bandwidth of one semitone.
///
/// Supports MIDI notes 24 (C1, ~32.7 Hz) through 119 (B8, ~7902 Hz)
/// and common audio sample rates.
public enum SemitoneBandpass {

    // MARK: - Types

    /// Coefficients for a single second-order (biquad) IIR filter section.
    ///
    /// Transfer function: `H(z) = (b0 + b1*z^-1 + b2*z^-2) / (1 + a1*z^-1 + a2*z^-2)`
    public struct BiquadCoeffs {
        public let b0, b1, b2, a1, a2: Float

        public init(b0: Float, b1: Float, b2: Float, a1: Float, a2: Float) {
            self.b0 = b0
            self.b1 = b1
            self.b2 = b2
            self.a1 = a1
            self.a2 = a2
        }
    }

    // MARK: - Coefficient Cache

    /// Thread-safe cache for computed filter coefficients.
    private static let cacheLock = NSLock()
    private static var coeffCache: [String: [BiquadCoeffs]] = [:]

    private static func cacheKey(centerFreq: Float, sr: Int, order: Int) -> String {
        "\(centerFreq)_\(sr)_\(order)"
    }

    // MARK: - Filter Design

    /// Design a bandpass biquad filter for the given center frequency.
    ///
    /// Uses the Audio EQ Cookbook formula for a constant-skirt-gain bandpass:
    /// ```
    /// b0 =  alpha
    /// b1 =  0
    /// b2 = -alpha
    /// a0 =  1 + alpha
    /// a1 = -2 * cos(w0)
    /// a2 =  1 - alpha
    /// ```
    /// where `w0 = 2*pi*f0/sr` and `alpha = sin(w0) / (2*Q)`.
    ///
    /// For semitone bandwidth, Q = 1 / (2^(1/12) - 2^(-1/12)) ~ 8.65.
    ///
    /// - Parameters:
    ///   - centerFreq: Center frequency in Hz.
    ///   - sr: Sample rate in Hz.
    ///   - order: Filter order (must be even; each 2nd-order section is one biquad).
    ///            Default 4 gives two cascaded biquads.
    /// - Returns: Array of `BiquadCoeffs` (one per second-order section).
    public static func designBandpass(
        centerFreq: Float,
        sr: Int,
        order: Int = 4
    ) -> [BiquadCoeffs] {
        let key = cacheKey(centerFreq: centerFreq, sr: sr, order: order)
        cacheLock.lock()
        if let cached = coeffCache[key] {
            cacheLock.unlock()
            return cached
        }
        cacheLock.unlock()

        let nSections = max(1, order / 2)
        let nyquist = Float(sr) / 2.0

        // Clamp center frequency to valid range
        guard centerFreq > 0 && centerFreq < nyquist * 0.95 else {
            // Return unity pass-through for out-of-range frequencies
            let passthrough = BiquadCoeffs(b0: 1, b1: 0, b2: 0, a1: 0, a2: 0)
            return [BiquadCoeffs](repeating: passthrough, count: nSections)
        }

        // Q for one semitone bandwidth
        let bwRatio = powf(2.0, 1.0 / 12.0) - powf(2.0, -1.0 / 12.0)
        let Q = 1.0 / bwRatio  // ~ 8.65

        let w0 = 2.0 * Float.pi * centerFreq / Float(sr)
        let sinW0 = sinf(w0)
        let cosW0 = cosf(w0)
        let alpha = sinW0 / (2.0 * Q)

        let a0 = 1.0 + alpha
        let b0 = alpha / a0
        let b1: Float = 0.0
        let b2 = -alpha / a0
        let a1 = (-2.0 * cosW0) / a0
        let a2 = (1.0 - alpha) / a0

        let section = BiquadCoeffs(b0: b0, b1: b1, b2: b2, a1: a1, a2: a2)
        let sections = [BiquadCoeffs](repeating: section, count: nSections)

        cacheLock.lock()
        coeffCache[key] = sections
        cacheLock.unlock()

        return sections
    }

    /// Design a bandpass biquad for a specific MIDI note.
    ///
    /// - Parameters:
    ///   - midi: MIDI note number (24=C1, 69=A4, 119=B8).
    ///   - sr: Sample rate in Hz.
    ///   - order: Filter order (default 4).
    /// - Returns: Array of `BiquadCoeffs`.
    public static func designForMIDI(
        midi: Int,
        sr: Int,
        order: Int = 4
    ) -> [BiquadCoeffs] {
        let freq = midiToHz(midi)
        return designBandpass(centerFreq: freq, sr: sr, order: order)
    }

    // MARK: - Biquad Application

    /// Apply a cascade of biquad filters to a signal.
    ///
    /// Each biquad section implements the difference equation:
    /// `y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]`
    ///
    /// Uses `vDSP_deq22` for efficient single-precision biquad filtering.
    ///
    /// - Parameters:
    ///   - signal: Input signal.
    ///   - coefficients: Array of biquad sections to cascade.
    /// - Returns: Filtered signal (same length as input).
    public static func applyBiquadCascade(
        signal: Signal,
        coefficients: [BiquadCoeffs]
    ) -> Signal {
        let n = signal.count
        guard n > 2, !coefficients.isEmpty else {
            // Too short to filter or no coefficients — return copy
            var data = [Float](repeating: 0, count: n)
            for i in 0..<n { data[i] = signal[i] }
            return Signal(data: data, sampleRate: signal.sampleRate)
        }

        // Start with a copy of the input
        var current = [Float](repeating: 0, count: n)
        signal.withUnsafeBufferPointer { buf in
            current.withUnsafeMutableBufferPointer { dst in
                dst.baseAddress!.initialize(from: buf.baseAddress!, count: n)
            }
        }

        var output = [Float](repeating: 0, count: n)

        for section in coefficients {
            // vDSP_deq22 coefficients array: [b0, b1, b2, a1, a2]
            // vDSP_deq22 convention:
            //   D[n] = A0*C[n] + A1*C[n-1] + A2*C[n-2] - A3*D[n-1] - A4*D[n-2]
            var coeffs: [Float] = [
                section.b0, section.b1, section.b2,
                section.a1, section.a2
            ]

            // vDSP_deq22 processes elements 2..n+1 of the input array and writes
            // to elements 2..n+1 of the output array. We pad with zeros for
            // initial filter state.
            var padded = [Float](repeating: 0, count: n + 2)
            for i in 0..<n {
                padded[i + 2] = current[i]
            }

            var paddedOutput = [Float](repeating: 0, count: n + 2)

            padded.withUnsafeBufferPointer { srcBuf in
                paddedOutput.withUnsafeMutableBufferPointer { dstBuf in
                    vDSP_deq22(
                        srcBuf.baseAddress!, 1,
                        &coeffs,
                        dstBuf.baseAddress!, 1,
                        vDSP_Length(n)
                    )
                }
            }

            // Extract the filtered output (skip the first 2 padding samples)
            for i in 0..<n {
                output[i] = paddedOutput[i + 2]
            }

            // Feed output into next section
            for i in 0..<n {
                current[i] = output[i]
            }
        }

        return Signal(data: output, sampleRate: signal.sampleRate)
    }

    // MARK: - Filterbank

    /// Apply a semitone bandpass filterbank to a signal.
    ///
    /// Filters the input signal through bandpass filters centered at each
    /// semitone in the specified MIDI range. Returns a 2D signal where each
    /// row is the input filtered through the corresponding semitone band.
    ///
    /// - Parameters:
    ///   - signal: Input audio signal (1D).
    ///   - sr: Sample rate override. If nil, uses `signal.sampleRate`.
    ///   - midiLow: Lowest MIDI note (default 24 = C1).
    ///   - midiHigh: Highest MIDI note (default 119 = B8).
    ///   - order: Filter order (default 4).
    /// - Returns: Signal with shape `[nSemitones, nSamples]`.
    public static func filterbank(
        signal: Signal,
        sr: Int? = nil,
        midiLow: Int = 24,
        midiHigh: Int = 119,
        order: Int = 4
    ) -> Signal {
        let sampleRate = sr ?? signal.sampleRate
        let nSamples = signal.count
        let nSemitones = midiHigh - midiLow + 1

        guard nSemitones > 0, nSamples > 0 else {
            return Signal(data: [], shape: [0, 0], sampleRate: sampleRate)
        }

        // Allocate output: [nSemitones, nSamples] row-major
        let totalCount = nSemitones * nSamples
        let outPtr = UnsafeMutablePointer<Float>.allocate(capacity: totalCount)
        outPtr.initialize(repeating: 0, count: totalCount)

        for i in 0..<nSemitones {
            let midi = midiLow + i
            let coeffs = designBandpass(
                centerFreq: midiToHz(midi),
                sr: sampleRate,
                order: order
            )

            let filtered = applyBiquadCascade(signal: signal, coefficients: coeffs)

            // Copy filtered data into the appropriate row
            let rowOffset = i * nSamples
            filtered.withUnsafeBufferPointer { buf in
                for j in 0..<nSamples {
                    outPtr[rowOffset + j] = buf[j]
                }
            }
        }

        let outBuf = UnsafeMutableBufferPointer(start: outPtr, count: totalCount)
        return Signal(
            taking: outBuf,
            shape: [nSemitones, nSamples],
            sampleRate: sampleRate
        )
    }

    // MARK: - Utilities

    /// Compute semitone center frequencies for a MIDI note range.
    ///
    /// - Parameters:
    ///   - midiLow: Lowest MIDI note (default 24 = C1).
    ///   - midiHigh: Highest MIDI note (default 119 = B8).
    /// - Returns: Array of center frequencies in Hz.
    public static func semitoneFrequencies(
        midiLow: Int = 24,
        midiHigh: Int = 119
    ) -> [Float] {
        guard midiHigh >= midiLow else { return [] }
        return (midiLow...midiHigh).map { midiToHz($0) }
    }

    /// Convert MIDI note number to frequency in Hz.
    ///
    /// Uses the standard formula: `f = 440 * 2^((midi - 69) / 12)`.
    public static func midiToHz(_ midi: Int) -> Float {
        440.0 * powf(2.0, Float(midi - 69) / 12.0)
    }
}
