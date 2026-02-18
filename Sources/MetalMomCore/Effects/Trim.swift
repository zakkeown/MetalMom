import Accelerate
import Foundation

/// Silence trimming for audio signals.
///
/// Trims leading and trailing silence from an audio signal based on
/// a dB threshold relative to the peak RMS energy, matching the
/// behavior of `librosa.effects.trim`.
public enum Trim {

    /// Trim leading and trailing silence from a signal.
    ///
    /// Computes RMS energy per frame, converts to dB relative to the
    /// peak frame, and finds the first and last frames above `-topDb`.
    /// Returns the trimmed signal along with the start and end sample
    /// indices of the non-silent region.
    ///
    /// - Parameters:
    ///   - signal: Input audio signal (1-D).
    ///   - topDb: Threshold in dB below the peak RMS. Frames with
    ///            energy below `peak - topDb` are considered silence.
    ///            Default: 60.0.
    ///   - frameLength: Length of each analysis frame. Default: 2048.
    ///   - hopLength: Number of samples between successive frames. Default: 512.
    /// - Returns: A tuple of (trimmed signal, start sample index, end sample index).
    public static func trim(
        signal: Signal,
        topDb: Float = 60.0,
        frameLength: Int = 2048,
        hopLength: Int = 512
    ) -> (signal: Signal, startIndex: Int, endIndex: Int) {
        let signalLength = signal.count

        // Edge case: empty or very short signal
        guard signalLength > 0 else {
            return (Signal(data: [], sampleRate: signal.sampleRate), 0, 0)
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
            return (Signal(data: [], sampleRate: signal.sampleRate), 0, 0)
        }

        // Step 2: Find reference (max RMS) and threshold
        var refValue: Float = 0
        rms.withUnsafeBufferPointer { buf in
            vDSP_maxv(buf.baseAddress!, 1, &refValue, vDSP_Length(nFrames))
        }

        // If max RMS is zero or negligible, signal is all silence
        guard refValue > 1e-10 else {
            return (Signal(data: [], sampleRate: signal.sampleRate), 0, 0)
        }

        // Threshold: ref * 10^(-topDb/20)
        let threshold = refValue * powf(10.0, -topDb / 20.0)

        // Step 3: Scan from start to find first frame above threshold
        var firstFrame = nFrames  // sentinel: no frame found
        rms.withUnsafeBufferPointer { buf in
            for f in 0..<nFrames {
                if buf[f] >= threshold {
                    firstFrame = f
                    break
                }
            }
        }

        // All frames below threshold → all silence
        guard firstFrame < nFrames else {
            return (Signal(data: [], sampleRate: signal.sampleRate), 0, 0)
        }

        // Step 4: Scan from end to find last frame above threshold
        var lastFrame = firstFrame
        rms.withUnsafeBufferPointer { buf in
            for f in stride(from: nFrames - 1, through: firstFrame, by: -1) {
                if buf[f] >= threshold {
                    lastFrame = f
                    break
                }
            }
        }

        // Step 5: Convert frame indices to sample indices
        let startSample = firstFrame * hopLength
        // End sample: one frame past the last active frame, clipped to signal length
        let endSample = min((lastFrame + 1) * hopLength, signalLength)

        guard endSample > startSample else {
            return (Signal(data: [], sampleRate: signal.sampleRate), 0, 0)
        }

        // Step 6: Extract the trimmed portion
        let trimmedData: [Float] = signal.withUnsafeBufferPointer { buf in
            Array(buf[startSample..<endSample])
        }

        let trimmedSignal = Signal(data: trimmedData, sampleRate: signal.sampleRate)
        return (trimmedSignal, startSample, endSample)
    }
}
