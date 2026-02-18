import Accelerate
import Foundation

/// Neural onset detector that picks peaks from pre-computed onset activation probabilities.
///
/// Replicates the approach of madmom's `RNNOnsetProcessor` / `CNNOnsetProcessor`:
/// 1. CNN/RNN ensemble produces onset activation probabilities per frame
/// 2. Peak picking with adaptive/fixed/combined thresholding detects onsets
///
/// The `detect` method is the core testable component that works with
/// pre-computed activations (no CoreML required).
public enum NeuralOnsetDetector {

    // MARK: - Threshold Methods

    /// Threshold combination methods for peak picking.
    public enum ThresholdMethod {
        /// Use only the fixed threshold.
        case fixed
        /// Use moving average + delta as adaptive threshold.
        case adaptive
        /// Use max(fixed, adaptive) as threshold.
        case combined
    }

    // MARK: - Core Detection

    /// Detect onsets from pre-computed neural network activation probabilities.
    ///
    /// This is the core method -- takes onset activation probabilities and
    /// picks peaks using the standard peak-picking algorithm with optional
    /// adaptive thresholding.
    ///
    /// - Parameters:
    ///   - activations: Onset activation probabilities [nFrames], values in [0,1].
    ///   - fps: Frames per second (e.g., 100 for 10ms hop). Default 100.
    ///   - threshold: Fixed threshold for peak detection. Default 0.3.
    ///   - preMax: Frames before n for local max check. Default 1.
    ///   - postMax: Frames after n for local max check. Default 1.
    ///   - preAvg: Frames before n for moving average threshold. Default 3.
    ///   - postAvg: Frames after n for moving average threshold. Default 3.
    ///   - combineMethod: How to combine fixed and adaptive thresholds.
    ///     Default: .adaptive.
    ///   - wait: Minimum frames between detected onsets. Default 1.
    /// - Returns: Array of onset frame indices.
    public static func detect(
        activations: [Float],
        fps: Float = 100.0,
        threshold: Float = 0.3,
        preMax: Int = 1,
        postMax: Int = 1,
        preAvg: Int = 3,
        postAvg: Int = 3,
        combineMethod: ThresholdMethod = .adaptive,
        wait: Int = 1
    ) -> [Int] {
        let n = activations.count
        guard n > 0 else { return [] }

        switch combineMethod {
        case .fixed:
            return pickPeaksFixed(
                activations: activations,
                threshold: threshold,
                preMax: preMax,
                postMax: postMax,
                wait: wait
            )

        case .adaptive:
            return pickPeaksAdaptive(
                activations: activations,
                threshold: threshold,
                preMax: preMax,
                postMax: postMax,
                preAvg: preAvg,
                postAvg: postAvg,
                wait: wait
            )

        case .combined:
            return pickPeaksCombined(
                activations: activations,
                threshold: threshold,
                preMax: preMax,
                postMax: postMax,
                preAvg: preAvg,
                postAvg: postAvg,
                wait: wait
            )
        }
    }

    // MARK: - Conversion Utilities

    /// Convert onset frame indices to times in seconds.
    public static func framesToTimes(frames: [Int], fps: Float) -> [Float] {
        return frames.map { Float($0) / fps }
    }

    /// Convert onset frame indices to sample indices.
    public static func framesToSamples(frames: [Int], hopLength: Int) -> [Int] {
        return frames.map { $0 * hopLength }
    }

    // MARK: - Smoothing

    /// Smooth activations with a moving average filter.
    /// Useful as preprocessing before peak picking.
    ///
    /// - Parameters:
    ///   - activations: Input activation array.
    ///   - width: Smoothing window width (must be odd; if even, width+1 is used).
    /// - Returns: Smoothed activations of the same length.
    public static func smooth(activations: [Float], width: Int) -> [Float] {
        let n = activations.count
        guard n > 0 else { return [] }

        let w = max(1, width % 2 == 0 ? width + 1 : width)
        guard w > 1 else { return activations }

        let halfW = w / 2
        var result = [Float](repeating: 0, count: n)

        for i in 0..<n {
            let lo = max(0, i - halfW)
            let hi = min(n - 1, i + halfW)
            var sum: Float = 0
            for j in lo...hi {
                sum += activations[j]
            }
            result[i] = sum / Float(hi - lo + 1)
        }

        return result
    }

    // MARK: - Private Peak Picking Methods

    /// Check if position i is a strict local maximum in [i-preMax, i+postMax].
    /// Returns true if activations[i] >= all values in the window and
    /// activations[i] > 0 (to avoid flat zero regions being "peaks").
    private static func isLocalMax(
        _ activations: [Float],
        _ i: Int,
        _ preMax: Int,
        _ postMax: Int
    ) -> Bool {
        let n = activations.count
        let lo = max(0, i - preMax)
        let hi = min(n - 1, i + postMax)

        for j in lo...hi {
            if activations[j] > activations[i] {
                return false
            }
        }
        return true
    }

    /// Fixed threshold: peak must be a local max and exceed the fixed threshold.
    private static func pickPeaksFixed(
        activations: [Float],
        threshold: Float,
        preMax: Int,
        postMax: Int,
        wait: Int
    ) -> [Int] {
        let n = activations.count
        guard n > 0 else { return [] }

        var peaks: [Int] = []
        var lastPeak = -wait

        for i in 0..<n {
            guard activations[i] >= threshold else { continue }
            guard isLocalMax(activations, i, preMax, postMax) else { continue }
            guard (i - lastPeak) >= wait else { continue }

            peaks.append(i)
            lastPeak = i
        }

        return peaks
    }

    /// Adaptive threshold: peak must be a local max and exceed the local
    /// moving average + (threshold * scale). The scale is the maximum activation
    /// value, so threshold acts as a fraction of the max.
    private static func pickPeaksAdaptive(
        activations: [Float],
        threshold: Float,
        preMax: Int,
        postMax: Int,
        preAvg: Int,
        postAvg: Int,
        wait: Int
    ) -> [Int] {
        let n = activations.count
        guard n > 0 else { return [] }

        // Find max activation for scaling
        var maxAct: Float = 0
        vDSP_maxv(activations, 1, &maxAct, vDSP_Length(n))
        guard maxAct > 0 else { return [] }

        let delta = threshold * maxAct

        var peaks: [Int] = []
        var lastPeak = -wait

        for i in 0..<n {
            // Check local maximum
            guard isLocalMax(activations, i, preMax, postMax) else { continue }

            // Compute moving average
            let avgLo = max(0, i - preAvg)
            let avgHi = min(n - 1, i + postAvg)
            var sum: Float = 0
            for j in avgLo...avgHi {
                sum += activations[j]
            }
            let mean = sum / Float(avgHi - avgLo + 1)

            // Must exceed adaptive threshold
            guard activations[i] >= mean + delta else { continue }

            // Wait constraint
            guard (i - lastPeak) >= wait else { continue }

            peaks.append(i)
            lastPeak = i
        }

        return peaks
    }

    /// Combined threshold: peak must exceed both the fixed threshold AND the
    /// adaptive (moving average + delta) threshold.
    private static func pickPeaksCombined(
        activations: [Float],
        threshold: Float,
        preMax: Int,
        postMax: Int,
        preAvg: Int,
        postAvg: Int,
        wait: Int
    ) -> [Int] {
        let n = activations.count
        guard n > 0 else { return [] }

        // Find max activation for adaptive scaling
        var maxAct: Float = 0
        vDSP_maxv(activations, 1, &maxAct, vDSP_Length(n))
        guard maxAct > 0 else { return [] }

        let delta = threshold * maxAct

        var peaks: [Int] = []
        var lastPeak = -wait

        for i in 0..<n {
            // Must exceed fixed threshold
            guard activations[i] >= threshold else { continue }

            // Check local maximum
            guard isLocalMax(activations, i, preMax, postMax) else { continue }

            // Compute moving average for adaptive threshold
            let avgLo = max(0, i - preAvg)
            let avgHi = min(n - 1, i + postAvg)
            var sum: Float = 0
            for j in avgLo...avgHi {
                sum += activations[j]
            }
            let mean = sum / Float(avgHi - avgLo + 1)

            // Must also exceed adaptive threshold
            guard activations[i] >= mean + delta else { continue }

            // Wait constraint
            guard (i - lastPeak) >= wait else { continue }

            peaks.append(i)
            lastPeak = i
        }

        return peaks
    }
}
