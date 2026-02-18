import Accelerate
import Foundation

/// Tonnetz (tonal centroid) features.
///
/// Tonnetz features compute 6-dimensional tonal centroid features from chroma,
/// representing harmonic relationships on the Tonnetz lattice:
///
/// - dim 0, 1: Fifths — sin/cos of (7/6 * pi * c)
/// - dim 2, 3: Minor thirds — sin/cos of (3/2 * pi * c)
/// - dim 4, 5: Major thirds — sin/cos of (2/3 * pi * c)
///
/// Pipeline: audio -> chroma_stft -> L1 normalize -> angular projection
///
/// This implementation matches librosa's `tonnetz` behaviour when using
/// chroma_stft as the chroma input, with radii r1=1, r2=1, r3=0.5.
public enum Tonnetz {

    /// Compute tonnetz (tonal centroid) features from audio.
    ///
    /// Returns a `Signal` with shape `[6, nFrames]`, row-major.
    ///
    /// - Parameters:
    ///   - signal: Input audio signal (1D).
    ///   - sr: Sample rate override. If `nil`, uses `signal.sampleRate`.
    ///   - nFFT: FFT window size. Default 2048.
    ///   - hopLength: Hop length in samples. Default `nFFT / 4`.
    ///   - winLength: Window length. Default `nFFT`.
    ///   - nChroma: Number of chroma bins. Default 12.
    ///   - center: If `true`, pad signal so frames are centred. Default `true`.
    /// - Returns: Tonnetz `Signal` with shape `[6, nFrames]`.
    public static func compute(
        signal: Signal,
        sr: Int? = nil,
        nFFT: Int = 2048,
        hopLength: Int? = nil,
        winLength: Int? = nil,
        nChroma: Int = 12,
        center: Bool = true
    ) -> Signal {
        let sampleRate = sr ?? signal.sampleRate

        // 1. Compute chroma: shape [nChroma, nFrames]
        let chroma = Chroma.stft(
            signal: signal,
            sr: sampleRate,
            nFFT: nFFT,
            hopLength: hopLength,
            winLength: winLength,
            nChroma: nChroma,
            center: center
        )

        let nFrames = chroma.shape[1]

        // 2. L1-normalize chroma per frame
        // chroma is [nChroma, nFrames] row-major
        // For each frame f, normalize sum of chroma[c, f] for c in 0..<nChroma
        var chromaNorm = [Float](repeating: 0, count: nChroma * nFrames)
        chroma.withUnsafeBufferPointer { src in
            for f in 0..<nFrames {
                var sum: Float = 0
                for c in 0..<nChroma {
                    sum += abs(src[c * nFrames + f])
                }
                let scale: Float = sum > 1e-10 ? 1.0 / sum : 0.0
                for c in 0..<nChroma {
                    chromaNorm[c * nFrames + f] = src[c * nFrames + f] * scale
                }
            }
        }

        // 3. Pre-compute angular basis vectors (6 x nChroma)
        //
        // Matches librosa exactly:
        //   scale = [7/6, 7/6, 3/2, 3/2, 2/3, 2/3]
        //   V = outer(scale, dim_map)  where dim_map = linspace(0, 12, nChroma)
        //   V[even_rows] -= 0.5  (converts cos to sin via cos(pi*(x-0.5)) = sin(pi*x))
        //   R = [1, 1, 1, 1, 0.5, 0.5]
        //   phi = R * cos(pi * V)
        //
        // Result per chroma bin c:
        //   dim 0: r1 * sin(7*pi*c/6)   — fifth sin
        //   dim 1: r1 * cos(7*pi*c/6)   — fifth cos
        //   dim 2: r2 * sin(3*pi*c/2)   — minor third sin
        //   dim 3: r2 * cos(3*pi*c/2)   — minor third cos
        //   dim 4: r3 * sin(2*pi*c/3)   — major third sin
        //   dim 5: r3 * cos(2*pi*c/3)   — major third cos
        let r1: Float = 1.0
        let r2: Float = 1.0
        let r3: Float = 0.5

        var basis = [[Float]](repeating: [Float](repeating: 0, count: nChroma), count: 6)
        for c in 0..<nChroma {
            let fc = Float(c)
            let phi1 = fc * 7.0 * .pi / 6.0   // fifths
            let phi2 = fc * 3.0 * .pi / 2.0   // minor thirds
            let phi3 = fc * 2.0 * .pi / 3.0   // major thirds
            basis[0][c] = r1 * sin(phi1)
            basis[1][c] = r1 * cos(phi1)
            basis[2][c] = r2 * sin(phi2)
            basis[3][c] = r2 * cos(phi2)
            basis[4][c] = r3 * sin(phi3)
            basis[5][c] = r3 * cos(phi3)
        }

        // 4. Compute tonnetz: for each frame, dot each basis vector with the chroma column
        // Output shape: [6, nFrames]
        let outCount = 6 * nFrames
        let outPtr = UnsafeMutablePointer<Float>.allocate(capacity: outCount)
        outPtr.initialize(repeating: 0, count: outCount)

        for d in 0..<6 {
            for f in 0..<nFrames {
                var val: Float = 0
                for c in 0..<nChroma {
                    val += basis[d][c] * chromaNorm[c * nFrames + f]
                }
                outPtr[d * nFrames + f] = val
            }
        }

        let outBuffer = UnsafeMutableBufferPointer(start: outPtr, count: outCount)
        return Signal(taking: outBuffer, shape: [6, nFrames], sampleRate: sampleRate)
    }
}
