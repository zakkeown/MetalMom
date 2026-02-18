import Accelerate
import Foundation

/// dB scaling conversions for spectral data.
///
/// Provides amplitude/power ↔ dB conversions with librosa-compatible defaults.
/// Uses vDSP for vectorized operations on the CPU.
public enum Scaling {

    // MARK: - Forward conversions (linear → dB)

    /// Convert an amplitude (magnitude) spectrogram to dB-scaled spectrogram.
    ///
    /// Formula: `20 * log10(max(signal, amin) / ref)`
    /// Then optionally clip to `topDb` below the maximum value.
    ///
    /// - Parameters:
    ///   - signal: Input amplitude signal (non-negative values).
    ///   - ref: Reference value. Amplitudes are scaled relative to `ref`.
    ///   - amin: Minimum amplitude threshold to avoid log(0). Default 1e-5 matches librosa.
    ///   - topDb: If non-nil, clip output to be no more than `topDb` below the peak. Default 80.0.
    /// - Returns: A new Signal with dB values, same shape as input.
    public static func amplitudeToDb(_ signal: Signal, ref: Float = 1.0,
                                     amin: Float = 1e-5, topDb: Float? = 80.0) -> Signal {
        return linearToDb(signal, multiplier: 20.0, ref: ref, amin: amin, topDb: topDb)
    }

    /// Convert a power spectrogram to dB-scaled spectrogram.
    ///
    /// Formula: `10 * log10(max(signal, amin) / ref)`
    /// Then optionally clip to `topDb` below the maximum value.
    ///
    /// - Parameters:
    ///   - signal: Input power signal (non-negative values).
    ///   - ref: Reference value. Powers are scaled relative to `ref`.
    ///   - amin: Minimum power threshold to avoid log(0). Default 1e-10 matches librosa.
    ///   - topDb: If non-nil, clip output to be no more than `topDb` below the peak. Default 80.0.
    /// - Returns: A new Signal with dB values, same shape as input.
    public static func powerToDb(_ signal: Signal, ref: Float = 1.0,
                                 amin: Float = 1e-10, topDb: Float? = 80.0) -> Signal {
        return linearToDb(signal, multiplier: 10.0, ref: ref, amin: amin, topDb: topDb)
    }

    // MARK: - Inverse conversions (dB → linear)

    /// Convert dB values back to amplitude (magnitude).
    ///
    /// Formula: `ref * 10^(signal / 20)`
    ///
    /// - Parameters:
    ///   - signal: Input dB-scaled signal.
    ///   - ref: Reference amplitude (default 1.0).
    /// - Returns: A new Signal with amplitude values.
    public static func dbToAmplitude(_ signal: Signal, ref: Float = 1.0) -> Signal {
        return dbToLinear(signal, divisor: 20.0, ref: ref)
    }

    /// Convert dB values back to power.
    ///
    /// Formula: `ref * 10^(signal / 10)`
    ///
    /// - Parameters:
    ///   - signal: Input dB-scaled signal.
    ///   - ref: Reference power (default 1.0).
    /// - Returns: A new Signal with power values.
    public static func dbToPower(_ signal: Signal, ref: Float = 1.0) -> Signal {
        return dbToLinear(signal, divisor: 10.0, ref: ref)
    }

    // MARK: - Internal

    /// Core forward conversion: `multiplier * log10(max(signal, amin) / ref)`, then topDb clip.
    private static func linearToDb(_ signal: Signal, multiplier: Float,
                                   ref: Float, amin: Float, topDb: Float?) -> Signal {
        let count = signal.count
        guard count > 0 else {
            return Signal(data: [], shape: signal.shape, sampleRate: signal.sampleRate)
        }

        // Allocate output buffer
        let outPtr = UnsafeMutablePointer<Float>.allocate(capacity: count)
        let n = vDSP_Length(count)

        signal.withUnsafeBufferPointer { srcBuf in
            let src = srcBuf.baseAddress!

            // Step 1: Clamp to amin using vDSP_vthr (threshold to lower limit)
            // vDSP_vthr clamps values below the threshold to the threshold value.
            var aminVal = amin
            vDSP_vthr(src, 1, &aminVal, outPtr, 1, n)

            // Step 2: Divide by ref — outPtr = outPtr / ref
            var refVal = ref
            vDSP_vsdiv(outPtr, 1, &refVal, outPtr, 1, n)

            // Step 3: log10 — outPtr = log10(outPtr)
            var countInt = Int32(count)
            vvlog10f(outPtr, outPtr, &countInt)

            // Step 4: Multiply by 10 or 20 — outPtr = multiplier * outPtr
            var mult = multiplier
            vDSP_vsmul(outPtr, 1, &mult, outPtr, 1, n)
        }

        // Step 5: topDb clipping — clip values to be within topDb of the max
        if let topDb = topDb {
            // Find max value
            var maxVal: Float = 0
            vDSP_maxv(outPtr, 1, &maxVal, n)

            // Clamp lower bound to (maxVal - topDb)
            var lowerBound = maxVal - topDb
            vDSP_vthr(outPtr, 1, &lowerBound, outPtr, 1, n)
        }

        let outBuf = UnsafeMutableBufferPointer(start: outPtr, count: count)
        return Signal(taking: outBuf, shape: signal.shape, sampleRate: signal.sampleRate)
    }

    // MARK: - PCEN (Per-Channel Energy Normalization)

    /// Apply Per-Channel Energy Normalization to a spectrogram.
    ///
    /// PCEN is an adaptive gain control that normalizes each frequency channel
    /// independently using a smoothed version of the input as a reference.
    /// It is particularly useful for keyword spotting and other tasks where
    /// robustness to loudness variation is important.
    ///
    /// Formula: `PCEN(S) = (S / (eps + M)^gain + bias)^power - bias^power`
    /// where `M` is the IIR-filtered (smoothed) spectrogram.
    ///
    /// - Parameters:
    ///   - spectrogram: Input spectrogram, shape `[nBands, nFrames]`. Should be non-negative.
    ///   - sr: Sample rate. If nil, uses the spectrogram's sample rate.
    ///   - hopLength: Hop length used to compute the spectrogram. Default: 512.
    ///   - gain: Gain exponent for the AGC denominator. Default: 0.98.
    ///   - bias: Bias added before the power compression. Default: 2.0.
    ///   - power: Compression exponent. Default: 0.5.
    ///   - timeConstant: Time constant for the IIR filter in seconds. Default: 0.06.
    ///   - eps: Small constant for numerical stability. Default: 1e-6.
    /// - Returns: PCEN-normalized spectrogram with the same shape as input.
    public static func pcen(
        _ spectrogram: Signal,
        sr: Int? = nil,
        hopLength: Int = 512,
        gain: Float = 0.98,
        bias: Float = 2.0,
        power: Float = 0.5,
        timeConstant: Float = 0.06,
        eps: Float = 1e-6
    ) -> Signal {
        let count = spectrogram.count
        guard count > 0, spectrogram.shape.count == 2 else {
            return Signal(data: [], shape: spectrogram.shape, sampleRate: spectrogram.sampleRate)
        }

        let nBands = spectrogram.shape[0]
        let nFrames = spectrogram.shape[1]
        let sampleRate = sr ?? spectrogram.sampleRate

        // Compute the IIR smoothing coefficient b
        // b = (sqrt(1 + 4 * t^2) - 1) / (2 * t^2)
        // where t = sr / (hop_length * 2 * pi * time_constant)
        let t = Float(sampleRate) / (Float(hopLength) * 2.0 * Float.pi * timeConstant)
        let t2 = t * t
        let b: Float = (sqrtf(1.0 + 4.0 * t2) - 1.0) / (2.0 * t2)

        let outPtr = UnsafeMutablePointer<Float>.allocate(capacity: count)

        spectrogram.withUnsafeBufferPointer { srcBuf in
            let src = srcBuf.baseAddress!

            // Process each band independently
            for band in 0..<nBands {
                let bandOffset = band * nFrames

                // IIR filter: M[0] = S[0], M[t] = (1-b)*M[t-1] + b*S[t]
                var m = src[bandOffset]  // M[0] = S[0]

                for frame in 0..<nFrames {
                    let idx = bandOffset + frame
                    let s = src[idx]

                    if frame > 0 {
                        m = (1.0 - b) * m + b * s
                    }

                    // PCEN: (S / (eps + M)^gain + bias)^power - bias^power
                    let denominator = powf(eps + m, gain)
                    let normalized = s / denominator + bias
                    outPtr[idx] = powf(normalized, power) - powf(bias, power)
                }
            }
        }

        let outBuf = UnsafeMutableBufferPointer(start: outPtr, count: count)
        return Signal(taking: outBuf, shape: spectrogram.shape, sampleRate: spectrogram.sampleRate)
    }

    // MARK: - Frequency Weighting Curves

    /// Compute A-weighting curve values for given frequencies.
    ///
    /// A-weighting (IEC 61672:2003) is the most commonly used frequency weighting
    /// that approximates the frequency response of human hearing at low to moderate
    /// sound pressure levels.
    ///
    /// - Parameter frequencies: Array of frequencies in Hz.
    /// - Returns: Array of dB adjustments (A-weighting values).
    public static func aWeighting(frequencies: [Float]) -> [Float] {
        // Use Double for intermediate calculations to avoid Float precision issues
        return frequencies.map { f -> Float in
            let fd = Double(f)
            let f2 = fd * fd
            guard f2 > 0 else { return -Float.infinity }

            let f4 = f2 * f2
            let c1: Double = 12194.0 * 12194.0  // 12194^2
            let c2: Double = 20.6 * 20.6         // 20.6^2
            let c3: Double = 107.7 * 107.7       // 107.7^2
            let c4: Double = 737.9 * 737.9       // 737.9^2

            // R_A(f) = 12194^2 * f^4 / ((f^2 + 20.6^2) * sqrt((f^2 + 107.7^2)(f^2 + 737.9^2)) * (f^2 + 12194^2))
            let numerator = c1 * f4
            let denominator = (f2 + c2) * sqrt((f2 + c3) * (f2 + c4)) * (f2 + c1)

            guard denominator > 0 else { return -Float.infinity }

            let rA = numerator / denominator
            guard rA > 0 else { return -Float.infinity }
            return Float(20.0 * log10(rA) + 2.0)
        }
    }

    /// Compute B-weighting curve values for given frequencies.
    ///
    /// B-weighting is similar to A-weighting but with less attenuation at
    /// low frequencies, originally designed for moderate sound levels.
    ///
    /// - Parameter frequencies: Array of frequencies in Hz.
    /// - Returns: Array of dB adjustments (B-weighting values).
    public static func bWeighting(frequencies: [Float]) -> [Float] {
        return frequencies.map { f -> Float in
            let fd = Double(f)
            let f2 = fd * fd
            guard f2 > 0 else { return -Float.infinity }

            let f3 = f2 * abs(fd)
            let c1: Double = 12194.0 * 12194.0  // 12194^2
            let c2: Double = 20.6 * 20.6         // 20.6^2
            let c5: Double = 158.5 * 158.5       // 158.5^2

            // R_B(f) = 12194^2 * f^3 / ((f^2 + 20.6^2) * sqrt(f^2 + 158.5^2) * (f^2 + 12194^2))
            let numerator = c1 * f3
            let denominator = (f2 + c2) * sqrt(f2 + c5) * (f2 + c1)

            guard denominator > 0 else { return -Float.infinity }

            let rB = numerator / denominator
            guard rB > 0 else { return -Float.infinity }
            return Float(20.0 * log10(rB) + 0.17)
        }
    }

    /// Compute C-weighting curve values for given frequencies.
    ///
    /// C-weighting is nearly flat across the audible range, with roll-off only
    /// at the extremes. Used for peak sound level measurements.
    ///
    /// - Parameter frequencies: Array of frequencies in Hz.
    /// - Returns: Array of dB adjustments (C-weighting values).
    public static func cWeighting(frequencies: [Float]) -> [Float] {
        return frequencies.map { f -> Float in
            let fd = Double(f)
            let f2 = fd * fd
            guard f2 > 0 else { return -Float.infinity }

            let c1: Double = 12194.0 * 12194.0  // 12194^2
            let c2: Double = 20.6 * 20.6         // 20.6^2

            // R_C(f) = 12194^2 * f^2 / ((f^2 + 20.6^2) * (f^2 + 12194^2))
            let numerator = c1 * f2
            let denominator = (f2 + c2) * (f2 + c1)

            guard denominator > 0 else { return -Float.infinity }

            let rC = numerator / denominator
            guard rC > 0 else { return -Float.infinity }
            return Float(20.0 * log10(rC) + 0.06)
        }
    }

    /// Compute D-weighting curve values for given frequencies.
    ///
    /// D-weighting is designed for measuring aircraft noise, with a peak
    /// sensitivity around 6 kHz matching the ear's sensitivity to jet noise.
    ///
    /// - Parameter frequencies: Array of frequencies in Hz.
    /// - Returns: Array of dB adjustments (D-weighting values).
    public static func dWeighting(frequencies: [Float]) -> [Float] {
        return frequencies.map { f -> Float in
            let fd = Double(f)
            let f2 = fd * fd
            guard f2 > 0 else { return -Float.infinity }

            // h(f) = ((1037918.48 - f^2)^2 + 1080768.16 * f^2) /
            //        ((9837328 - f^2)^2 + 11723776 * f^2)
            let hNum = (1037918.48 - f2) * (1037918.48 - f2) + 1080768.16 * f2
            let hDen = (9837328.0 - f2) * (9837328.0 - f2) + 11723776.0 * f2

            guard hDen > 0 else { return -Float.infinity }
            let h = hNum / hDen

            // R_D(f) = f / 6.8966888496476e-5 * sqrt(h / ((f^2 + 79919.29) * (f^2 + 1345600)))
            let dDen = (f2 + 79919.29) * (f2 + 1345600.0)
            guard dDen > 0 else { return -Float.infinity }

            let rD = abs(fd) / 6.8966888496476e-5 * sqrt(h / dDen)
            guard rD > 0 else { return -Float.infinity }

            return Float(20.0 * log10(rD))
        }
    }

    /// Apply A-weighting to a dB-scaled spectrogram.
    ///
    /// Convenience method that computes A-weighting values for the FFT frequency bins
    /// and adds them to each frame of the spectrogram.
    ///
    /// - Parameters:
    ///   - spectrogram: Input spectrogram in dB scale, shape `[nFreqs, nFrames]`.
    ///   - sr: Sample rate in Hz.
    ///   - nFFT: FFT size used to compute the spectrogram.
    /// - Returns: A-weighted spectrogram with the same shape as input.
    public static func applyAWeighting(_ spectrogram: Signal, sr: Int, nFFT: Int) -> Signal {
        let count = spectrogram.count
        guard count > 0, spectrogram.shape.count == 2 else {
            return Signal(data: [], shape: spectrogram.shape, sampleRate: spectrogram.sampleRate)
        }

        let nFreqs = spectrogram.shape[0]
        let nFrames = spectrogram.shape[1]

        // Compute frequencies for each FFT bin
        var frequencies = [Float](repeating: 0, count: nFreqs)
        for i in 0..<nFreqs {
            frequencies[i] = Float(i) * Float(sr) / Float(nFFT)
        }

        // Get A-weighting values
        let weights = aWeighting(frequencies: frequencies)

        // Apply weighting: add dB weights to each frame
        let outPtr = UnsafeMutablePointer<Float>.allocate(capacity: count)
        spectrogram.withUnsafeBufferPointer { srcBuf in
            let src = srcBuf.baseAddress!
            for freq in 0..<nFreqs {
                let w = weights[freq]
                for frame in 0..<nFrames {
                    let idx = freq * nFrames + frame
                    outPtr[idx] = src[idx] + w
                }
            }
        }

        let outBuf = UnsafeMutableBufferPointer(start: outPtr, count: count)
        return Signal(taking: outBuf, shape: spectrogram.shape, sampleRate: spectrogram.sampleRate)
    }

    // MARK: - Internal

    /// Core forward conversion: `multiplier * log10(max(signal, amin) / ref)`, then topDb clip.
    private static func dbToLinear(_ signal: Signal, divisor: Float, ref: Float) -> Signal {
        let count = signal.count
        guard count > 0 else {
            return Signal(data: [], shape: signal.shape, sampleRate: signal.sampleRate)
        }

        let outPtr = UnsafeMutablePointer<Float>.allocate(capacity: count)
        let n = vDSP_Length(count)

        signal.withUnsafeBufferPointer { srcBuf in
            let src = srcBuf.baseAddress!

            // Step 1: Divide by divisor (10 or 20) — outPtr = signal / divisor
            var div = divisor
            vDSP_vsdiv(src, 1, &div, outPtr, 1, n)

            // Step 2: 10^x — outPtr = 10^outPtr
            // Compute as exp(x * ln(10)) since vvexp10f may not be available
            var ln10 = Float(log(10.0))
            vDSP_vsmul(outPtr, 1, &ln10, outPtr, 1, n)
            var countInt = Int32(count)
            vvexpf(outPtr, outPtr, &countInt)

            // Step 3: Multiply by ref — outPtr = ref * outPtr
            var refVal = ref
            vDSP_vsmul(outPtr, 1, &refVal, outPtr, 1, n)
        }

        let outBuf = UnsafeMutableBufferPointer(start: outPtr, count: count)
        return Signal(taking: outBuf, shape: signal.shape, sampleRate: signal.sampleRate)
    }
}
