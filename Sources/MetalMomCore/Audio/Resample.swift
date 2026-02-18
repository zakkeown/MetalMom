import Accelerate
import Foundation

public enum Resample {
    /// Resample audio data from one sample rate to another.
    ///
    /// Uses sinc interpolation with a Kaiser-windowed filter for high quality.
    ///
    /// - Parameters:
    ///   - signal: Input audio signal.
    ///   - targetRate: Target sample rate in Hz.
    /// - Returns: Resampled signal at the target rate.
    public static func resample(signal: Signal, targetRate: Int) -> Signal {
        guard signal.sampleRate != targetRate else { return signal }
        guard signal.count > 0 else {
            return Signal(data: [], sampleRate: targetRate)
        }

        let sourceRate = signal.sampleRate
        let ratio = Double(targetRate) / Double(sourceRate)
        let outputLength = Int(ceil(Double(signal.count) * ratio))
        guard outputLength > 0 else {
            return Signal(data: [], sampleRate: targetRate)
        }

        // Use windowed sinc interpolation
        // Filter half-length (number of zero crossings on each side)
        let numZeros = 64  // Quality parameter
        let precision = 9  // log2 of table oversampling factor
        let tableOversample = 1 << precision  // 512

        // Build the sinc filter table (oversampled by tableOversample)
        // Length = numZeros * tableOversample + 1
        let filterLen = numZeros * tableOversample + 1
        var filterTable = [Float](repeating: 0, count: filterLen)

        let beta = 6.0
        let i0Beta = besselI0(beta)
        for i in 0..<filterLen {
            let x = Double(i) / Double(tableOversample)
            // Sinc value
            let sincVal: Double
            if x == 0 {
                sincVal = 1.0
            } else {
                sincVal = sin(.pi * x) / (.pi * x)
            }
            // Kaiser window
            let t = x / Double(numZeros)
            let kaiserVal: Double
            if t >= 1.0 {
                kaiserVal = 0.0
            } else {
                kaiserVal = besselI0(beta * sqrt(1.0 - t * t)) / i0Beta
            }
            filterTable[i] = Float(sincVal * kaiserVal)
        }

        // Perform the resampling
        var output = [Float](repeating: 0, count: outputLength)
        let scale = min(1.0, ratio) // Anti-aliasing: use the lower of the two rates
        let filterScale = scale * Double(tableOversample)

        signal.withUnsafeBufferPointer { src in
            let srcCount = src.count

            for i in 0..<outputLength {
                // Map output sample to input position
                let srcPos = Double(i) / ratio
                let srcIdx = Int(srcPos)
                let _ = srcPos - Double(srcIdx)

                var sum: Float = 0
                var weightSum: Float = 0

                // Iterate over filter support
                let support = Int(ceil(Double(numZeros) / scale))
                let jStart = max(0, srcIdx - support + 1)
                let jEnd = min(srcCount - 1, srcIdx + support)

                for j in jStart...jEnd {
                    let delta = Double(j) - srcPos
                    let filterPos = abs(delta) * filterScale
                    let filterIdx = Int(filterPos)

                    if filterIdx < filterLen - 1 {
                        // Linear interpolation in filter table
                        let filterFrac = Float(filterPos - Double(filterIdx))
                        let w = filterTable[filterIdx] * (1 - filterFrac) + filterTable[filterIdx + 1] * filterFrac
                        sum += src[j] * w
                        weightSum += w
                    }
                }

                // Apply scale factor for energy preservation
                if weightSum > 0 {
                    output[i] = sum * Float(scale)
                }
            }
        }

        return Signal(data: output, sampleRate: targetRate)
    }

    /// Modified Bessel function of the first kind, order 0.
    private static func besselI0(_ x: Double) -> Double {
        // Series expansion: I0(x) = sum_{k=0}^inf [(x/2)^k / k!]^2
        var sum = 1.0
        var term = 1.0
        let halfX = x / 2.0
        for k in 1..<50 {
            term *= halfX / Double(k)
            let t2 = term * term
            sum += t2
            if t2 < sum * 1e-20 { break }
        }
        return sum
    }
}
