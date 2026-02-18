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

    /// Core inverse conversion: `ref * 10^(signal / divisor)`.
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
