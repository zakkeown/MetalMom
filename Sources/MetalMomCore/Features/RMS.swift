import Accelerate
import Foundation

/// Root-mean-square (RMS) energy computation.
///
/// Computes RMS energy per frame from a raw audio signal.
/// The signal is optionally center-padded and then split into overlapping frames.
///
/// Matches the behavior of `librosa.feature.rms(y=...)`.
public enum RMS {

    /// Compute RMS energy per frame.
    ///
    /// `RMS[t] = sqrt(mean(frame[t]^2))`
    ///
    /// - Parameters:
    ///   - signal: Input audio signal (1D).
    ///   - frameLength: Length of each analysis frame. Default 2048.
    ///   - hopLength: Number of samples between successive frames. Default 512.
    ///   - center: If true, center-pad the signal by frameLength/2 on each side. Default true.
    /// - Returns: Signal with shape [1, nFrames], RMS energy per frame.
    public static func compute(
        signal: Signal,
        frameLength: Int = 2048,
        hopLength: Int = 512,
        center: Bool = true
    ) -> Signal {
        // --- 1. Optionally pad the signal ---
        let padded: [Float]
        if center {
            let padAmount = frameLength / 2
            padded = [Float](repeating: 0, count: padAmount)
                   + signal.withUnsafeBufferPointer { Array($0) }
                   + [Float](repeating: 0, count: padAmount)
        } else {
            padded = signal.withUnsafeBufferPointer { Array($0) }
        }

        let paddedLength = padded.count

        // --- 2. Compute number of frames ---
        guard paddedLength >= frameLength else {
            return Signal(data: [], shape: [1, 0], sampleRate: signal.sampleRate)
        }
        let nFrames = 1 + (paddedLength - frameLength) / hopLength

        // --- 3. Compute RMS per frame using vDSP ---
        let outPtr = UnsafeMutablePointer<Float>.allocate(capacity: nFrames)
        outPtr.initialize(repeating: 0, count: nFrames)

        padded.withUnsafeBufferPointer { paddedBuf in
            for f in 0..<nFrames {
                let start = f * hopLength

                // vDSP_measqv computes mean of squares
                var meanSq: Float = 0
                vDSP_measqv(paddedBuf.baseAddress! + start, 1, &meanSq,
                           vDSP_Length(frameLength))

                // RMS = sqrt(mean_sq)
                outPtr[f] = sqrtf(meanSq)
            }
        }

        let outBuffer = UnsafeMutableBufferPointer(start: outPtr, count: nFrames)
        return Signal(taking: outBuffer, shape: [1, nFrames], sampleRate: signal.sampleRate)
    }
}
