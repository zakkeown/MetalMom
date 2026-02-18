import Foundation

/// Zero-crossing rate (ZCR) computation.
///
/// Computes the rate of sign changes in the signal within each frame.
///
/// Matches the behavior of `librosa.feature.zero_crossing_rate(y=...)`.
public enum ZeroCrossing {

    /// Compute zero-crossing rate per frame.
    ///
    /// `ZCR[t] = (number of sign changes in frame[t]) / frameLength`
    ///
    /// A zero-crossing occurs when consecutive samples have different signs.
    /// Zero values are treated as positive (matching librosa's default behavior
    /// where `np.signbit(0.0)` returns `False`).
    ///
    /// - Parameters:
    ///   - signal: Input audio signal (1D).
    ///   - frameLength: Length of each analysis frame. Default 2048.
    ///   - hopLength: Number of samples between successive frames. Default 512.
    ///   - center: If true, center-pad the signal by frameLength/2 on each side. Default true.
    /// - Returns: Signal with shape [1, nFrames], zero-crossing rate per frame.
    public static func rate(
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

        // --- 3. Compute ZCR per frame ---
        let outPtr = UnsafeMutablePointer<Float>.allocate(capacity: nFrames)
        outPtr.initialize(repeating: 0, count: nFrames)

        padded.withUnsafeBufferPointer { paddedBuf in
            for f in 0..<nFrames {
                let start = f * hopLength

                // Count sign changes within the frame.
                // librosa uses np.signbit which treats 0.0 as non-negative (False),
                // so sign(0) = +1 equivalent, sign(negative) = True.
                var crossings: Int = 0
                for i in 1..<frameLength {
                    let prev = paddedBuf[start + i - 1]
                    let curr = paddedBuf[start + i]
                    // signbit: negative -> true, zero or positive -> false
                    let prevNeg = prev.sign == .minus  // true for negative, false for 0 or positive
                    let currNeg = curr.sign == .minus
                    if prevNeg != currNeg {
                        crossings += 1
                    }
                }

                outPtr[f] = Float(crossings) / Float(frameLength)
            }
        }

        let outBuffer = UnsafeMutableBufferPointer(start: outPtr, count: nFrames)
        return Signal(taking: outBuffer, shape: [1, nFrames], sampleRate: signal.sampleRate)
    }
}
