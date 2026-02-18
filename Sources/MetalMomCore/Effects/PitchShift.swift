import Foundation
import Accelerate

/// Pitch shifting via time stretching and resampling.
///
/// Changes the pitch of an audio signal without changing its duration.
/// Uses the phase vocoder for time stretching followed by resampling
/// to restore the original duration at the new pitch.
public enum PitchShift {

    /// Shift pitch by a given number of steps (semitones by default).
    ///
    /// - Parameters:
    ///   - signal: Input audio signal (1-D).
    ///   - sr: Sample rate (optional, uses signal.sampleRate if nil).
    ///   - nSteps: Number of steps to shift. Positive shifts pitch up,
    ///             negative shifts pitch down.
    ///   - binsPerOctave: Number of steps per octave. Default 12 (semitones).
    ///   - nFFT: FFT size for the phase vocoder. Default 2048.
    ///   - hopLength: Hop length for the phase vocoder. Default nFFT/4.
    /// - Returns: Pitch-shifted 1-D Signal of approximately the same length.
    public static func pitchShift(
        signal: Signal,
        sr: Int? = nil,
        nSteps: Float,
        binsPerOctave: Int = 12,
        nFFT: Int = 2048,
        hopLength: Int? = nil
    ) -> Signal {
        let sampleRate = sr ?? signal.sampleRate
        let hop = hopLength ?? nFFT / 4
        let inputLength = signal.count

        guard inputLength > 0 else {
            return Signal(data: [], sampleRate: sampleRate)
        }

        // nSteps == 0 means no pitch change
        if nSteps == 0 {
            // Return a copy of the input
            var copy = [Float](repeating: 0, count: inputLength)
            signal.withUnsafeBufferPointer { buf in
                for i in 0..<inputLength {
                    copy[i] = buf[i]
                }
            }
            return Signal(data: copy, sampleRate: sampleRate)
        }

        // rate = 2^(nSteps / binsPerOctave)
        let rate = powf(2.0, nSteps / Float(binsPerOctave))

        // Step 1: Time-stretch by `rate`.
        // This makes the audio `rate` times faster (shorter by 1/rate).
        let stretched = TimeStretch.timeStretch(
            signal: signal,
            rate: rate,
            sr: sampleRate,
            nFFT: nFFT,
            hopLength: hop
        )

        // Step 2: Resample to restore original duration.
        // The stretched signal has effective sample rate `sampleRate`.
        // We resample from `sampleRate` to `round(sampleRate / rate)`
        // which lengthens the signal back to approximately the original length,
        // thereby shifting the pitch.
        let targetRate = max(1, Int(roundf(Float(sampleRate) / rate)))
        let resampled = Resample.resample(signal: stretched, targetRate: targetRate)

        // Step 3: Trim or pad to match original length
        var output = [Float](repeating: 0, count: inputLength)
        let copyCount = min(inputLength, resampled.count)
        resampled.withUnsafeBufferPointer { buf in
            for i in 0..<copyCount {
                output[i] = buf[i]
            }
        }

        return Signal(data: output, sampleRate: sampleRate)
    }
}
