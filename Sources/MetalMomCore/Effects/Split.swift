import Accelerate
import Foundation

/// Non-silent interval detection for audio signals.
///
/// Detects contiguous non-silent regions in an audio signal based on
/// a dB threshold relative to the peak RMS energy, matching the
/// behavior of `librosa.effects.split`.
public enum Split {

    /// Detect non-silent intervals in a signal.
    ///
    /// Computes RMS energy per frame, converts to a threshold relative
    /// to the peak frame, and finds contiguous runs of frames above
    /// the threshold. Returns each run as a (startSample, endSample) pair.
    ///
    /// - Parameters:
    ///   - signal: Input audio signal (1-D).
    ///   - topDb: Threshold in dB below the peak RMS. Frames with
    ///            energy below `peak - topDb` are considered silence.
    ///            Default: 60.0.
    ///   - frameLength: Length of each analysis frame. Default: 2048.
    ///   - hopLength: Number of samples between successive frames. Default: 512.
    /// - Returns: Array of (startSample, endSample) tuples for each non-silent interval.
    public static func split(
        signal: Signal,
        topDb: Float = 60.0,
        frameLength: Int = 2048,
        hopLength: Int = 512
    ) -> [(start: Int, end: Int)] {
        let signalLength = signal.count

        // Edge case: empty or very short signal
        guard signalLength > 0 else {
            return []
        }

        // Step 1: Compute RMS energy per frame (non-centered, so frame indices
        // map directly to sample positions without padding offset).
        let rms = RMS.compute(
            signal: signal,
            frameLength: frameLength,
            hopLength: hopLength,
            center: false
        )

        let nFrames: Int
        if rms.shape.count == 2 {
            nFrames = rms.shape[1]
        } else {
            nFrames = rms.count
        }

        // Edge case: no frames computed
        guard nFrames > 0 else {
            return []
        }

        // Step 2: Find reference (max RMS) and threshold
        var refValue: Float = 0
        rms.withUnsafeBufferPointer { buf in
            vDSP_maxv(buf.baseAddress!, 1, &refValue, vDSP_Length(nFrames))
        }

        // If max RMS is zero or negligible, signal is all silence
        guard refValue > 1e-10 else {
            return []
        }

        // Threshold: ref * 10^(-topDb/20)
        let threshold = refValue * powf(10.0, -topDb / 20.0)

        // Step 3: Find contiguous runs of non-silent frames
        var intervals: [(start: Int, end: Int)] = []
        var inNonSilent = false
        var runStart = 0

        rms.withUnsafeBufferPointer { buf in
            for f in 0..<nFrames {
                let isAbove = buf[f] >= threshold
                if isAbove && !inNonSilent {
                    // Start of a new non-silent run
                    runStart = f
                    inNonSilent = true
                } else if !isAbove && inNonSilent {
                    // End of a non-silent run
                    let startSample = runStart * hopLength
                    let endSample = min(f * hopLength, signalLength)
                    intervals.append((start: startSample, end: endSample))
                    inNonSilent = false
                }
            }

            // If we ended while still in a non-silent run, close it
            if inNonSilent {
                let startSample = runStart * hopLength
                let endSample = min((nFrames - 1 + 1) * hopLength, signalLength)
                intervals.append((start: startSample, end: endSample))
            }
        }

        return intervals
    }
}
