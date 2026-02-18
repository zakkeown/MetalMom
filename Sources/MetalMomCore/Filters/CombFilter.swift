import Accelerate
import Foundation

/// Comb filter bank for tempo estimation.
///
/// A comb filter creates resonance at a fundamental frequency and its harmonics.
/// By sweeping the delay parameter across a range of musically relevant periods
/// (corresponding to 30–300 BPM), the filter that produces the highest output
/// energy indicates the dominant tempo of the input signal.
public enum CombFilter {

    /// Apply a forward (feedback / IIR) comb filter to a signal.
    ///
    /// y[n] = x[n] + alpha * y[n - delay]
    ///
    /// This creates resonance at the fundamental frequency 1/delay
    /// and its harmonics.
    ///
    /// - Parameters:
    ///   - signal: Input signal array.
    ///   - delay: Delay in samples (the comb filter period).
    ///   - alpha: Feedback gain (0 < alpha < 1). Default 0.99.
    /// - Returns: Filtered signal (same length as input).
    public static func forward(signal: [Float], delay: Int, alpha: Float = 0.99) -> [Float] {
        let n = signal.count
        guard n > 0, delay > 0 else { return signal }

        var y = [Float](repeating: 0, count: n)

        // For samples before delay, output = input (no feedback available)
        for i in 0..<min(delay, n) {
            y[i] = signal[i]
        }

        // IIR: y[n] = x[n] + alpha * y[n - delay]
        for i in delay..<n {
            y[i] = signal[i] + alpha * y[i - delay]
        }

        return y
    }

    /// Apply a backward (feedforward / FIR) comb filter.
    ///
    /// y[n] = x[n] + alpha * x[n - delay]
    ///
    /// - Parameters:
    ///   - signal: Input signal array.
    ///   - delay: Delay in samples.
    ///   - alpha: Feedforward gain. Default 1.0.
    /// - Returns: Filtered signal (same length as input).
    public static func backward(signal: [Float], delay: Int, alpha: Float = 1.0) -> [Float] {
        let n = signal.count
        guard n > 0, delay > 0 else { return signal }

        var y = [Float](repeating: 0, count: n)

        // For samples before delay, output = input (no delayed sample available)
        for i in 0..<min(delay, n) {
            y[i] = signal[i]
        }

        // FIR: y[n] = x[n] + alpha * x[n - delay]
        for i in delay..<n {
            y[i] = signal[i] + alpha * signal[i - delay]
        }

        return y
    }

    /// Compute comb filter bank response for tempo estimation.
    ///
    /// Applies a bank of comb filters at different periods corresponding to
    /// tempos from minBPM to maxBPM, and returns the energy (sum of squares)
    /// for each tempo.
    ///
    /// - Parameters:
    ///   - signal: Input signal (typically onset strength envelope).
    ///   - fps: Frames per second of the input signal.
    ///   - minBPM: Minimum tempo to test. Default 30.
    ///   - maxBPM: Maximum tempo to test. Default 300.
    ///   - bpmStep: BPM resolution. Default 1.0.
    ///   - alpha: Comb filter feedback gain. Default 0.99.
    /// - Returns: (tempos: [Float], energies: [Float]) — BPM values and corresponding energies.
    public static func tempoFilterBank(
        signal: [Float],
        fps: Float,
        minBPM: Float = 30.0,
        maxBPM: Float = 300.0,
        bpmStep: Float = 1.0,
        alpha: Float = 0.99
    ) -> (tempos: [Float], energies: [Float]) {
        guard !signal.isEmpty, fps > 0, minBPM < maxBPM else {
            return (tempos: [], energies: [])
        }

        var tempos: [Float] = []
        var energies: [Float] = []

        var bpm = minBPM
        while bpm <= maxBPM {
            // Convert BPM to period in frames: period = fps * 60 / bpm
            let period = fps * 60.0 / bpm
            let delay = max(1, Int(round(period)))

            // Apply forward comb filter at this delay
            let filtered = forward(signal: signal, delay: delay, alpha: alpha)

            // Compute energy: sum of squares
            var energy: Float = 0
            vDSP_dotpr(filtered, 1, filtered, 1, &energy, vDSP_Length(filtered.count))

            tempos.append(bpm)
            energies.append(energy)

            bpm += bpmStep
        }

        return (tempos: tempos, energies: energies)
    }
}
