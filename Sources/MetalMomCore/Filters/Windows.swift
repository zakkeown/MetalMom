import Foundation
import Accelerate

/// Window functions for spectral analysis.
public enum Windows {

    /// Generate a Hann window.
    ///
    /// - Parameters:
    ///   - length: Window length in samples.
    ///   - periodic: If `true` (default), generates a periodic window suitable for
    ///     spectral analysis (DFT-symmetric). If `false`, generates a symmetric window.
    /// - Returns: Array of `Float` window values.
    public static func hann(length: Int, periodic: Bool = true) -> [Float] {
        guard length > 0 else { return [] }
        guard length > 1 else { return [1.0] }

        // For a periodic window we compute a symmetric window of length N+1 then
        // drop the last sample.  The symmetric formula is:
        //   w[n] = 0.5 * (1 - cos(2 * pi * n / (M - 1)))   for n in 0..<M
        // where M = length (symmetric) or length + 1 (periodic).
        let M = periodic ? length + 1 : length
        var window = [Float](repeating: 0, count: M)

        window.withUnsafeMutableBufferPointer { buf in
            let ptr = buf.baseAddress!

            // Build a ramp [0, step, 2*step, ... , (M-1)*step]
            // where step = 2*pi / (M-1)
            var start: Float = 0
            var step: Float = 2.0 * .pi / Float(M - 1)
            vDSP_vramp(&start, &step, ptr, 1, vDSP_Length(M))

            // cos(ramp)  — in-place via raw pointer
            var count = Int32(M)
            vvcosf(ptr, ptr, &count)

            // Hann = 0.5 * (1 - cos(...))  <==>  -0.5 * cos(...) + 0.5
            var negHalf: Float = -0.5
            var half: Float = 0.5
            vDSP_vsmsa(ptr, 1, &negHalf, &half, ptr, 1, vDSP_Length(M))
        }

        if periodic {
            return Array(window.prefix(length))
        }
        return window
    }
}
