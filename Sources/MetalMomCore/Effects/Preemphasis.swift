import Accelerate
import Foundation

/// First-order IIR filters for preemphasis and deemphasis.
///
/// These filters are commonly used in speech processing to boost or
/// attenuate high frequencies relative to low frequencies.
///
/// - **Preemphasis**: `y[n] = x[n] - coef * x[n-1]` (high-pass)
/// - **Deemphasis**: `y[n] = x[n] + coef * y[n-1]` (low-pass, inverse of preemphasis)
public enum Preemphasis {

    /// Apply a preemphasis filter to an audio signal.
    ///
    /// Implements the first-order FIR filter `y[n] = x[n] - coef * x[n-1]`,
    /// matching the behavior of `librosa.effects.preemphasis`.
    /// The first sample is passed through unchanged (i.e., `x[-1]` is assumed to be 0).
    ///
    /// - Parameters:
    ///   - signal: Input audio signal (1-D).
    ///   - coef: Filter coefficient. Default: 0.97.
    /// - Returns: Filtered signal with the same length and sample rate.
    public static func preemphasis(signal: Signal, coef: Float = 0.97) -> Signal {
        let n = signal.count
        guard n > 0 else {
            return Signal(data: [], sampleRate: signal.sampleRate)
        }

        var output = [Float](repeating: 0, count: n)

        signal.withUnsafeBufferPointer { buf in
            // y[0] = x[0] - coef * 0 = x[0]
            output[0] = buf[0]

            // y[i] = x[i] - coef * x[i-1]
            for i in 1..<n {
                output[i] = buf[i] - coef * buf[i - 1]
            }
        }

        return Signal(data: output, sampleRate: signal.sampleRate)
    }

    /// Apply a deemphasis filter to an audio signal.
    ///
    /// Implements the first-order IIR filter `y[n] = x[n] + coef * y[n-1]`,
    /// matching the behavior of `librosa.effects.deemphasis`.
    /// This is the inverse of preemphasis: `deemphasis(preemphasis(x)) == x`.
    ///
    /// - Parameters:
    ///   - signal: Input audio signal (1-D).
    ///   - coef: Filter coefficient. Default: 0.97.
    /// - Returns: Filtered signal with the same length and sample rate.
    public static func deemphasis(signal: Signal, coef: Float = 0.97) -> Signal {
        let n = signal.count
        guard n > 0 else {
            return Signal(data: [], sampleRate: signal.sampleRate)
        }

        var output = [Float](repeating: 0, count: n)

        signal.withUnsafeBufferPointer { buf in
            // y[0] = x[0] + coef * 0 = x[0]
            output[0] = buf[0]

            // y[n] = x[n] + coef * y[n-1]
            // This is a recursive (IIR) filter, must compute sequentially
            for i in 1..<n {
                output[i] = buf[i] + coef * output[i - 1]
            }
        }

        return Signal(data: output, sampleRate: signal.sampleRate)
    }
}
