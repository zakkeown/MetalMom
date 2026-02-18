import Accelerate
import Foundation

/// Polynomial feature extraction from spectrograms.
///
/// For each frame (column) of the input spectrogram, fits a polynomial
/// of given order to the frequency axis and returns the coefficients.
/// Matches librosa.feature.poly_features semantics.
public enum PolyFeatures {

    /// Compute polynomial features from a spectrogram.
    ///
    /// For each frame, fits a polynomial of given order to the frequency axis
    /// and returns the coefficients in descending power order (highest power first),
    /// matching `np.polyfit` convention.
    ///
    /// The x-axis uses FFT bin center frequencies: `freq[i] = i * sr / nFFT`,
    /// matching librosa's default behavior.
    ///
    /// - Parameters:
    ///   - data: Input feature matrix, shape [nFeatures, nFrames], row-major.
    ///   - order: Polynomial order. Default 1 (linear fit).
    ///   - sr: Sample rate. Default 22050.
    ///   - nFFT: FFT size used to compute the spectrogram. Default 2048.
    /// - Returns: Polynomial coefficients, shape [order+1, nFrames], row-major.
    ///   Coefficients are in descending power order (highest power first).
    public static func compute(
        data: Signal,
        order: Int = 1,
        sr: Int = 22050,
        nFFT: Int = 2048
    ) -> Signal {
        precondition(data.shape.count == 2, "poly_features requires 2D input")

        let nFeatures = data.shape[0]
        let nFrames = data.shape[1]
        let nCoeffs = order + 1

        // Build frequency axis: freq[i] = i * sr / nFFT
        let freqScale = Double(sr) / Double(nFFT)

        // Compute pseudoinverse of Vandermonde matrix using Double precision
        // pinv has shape [nCoeffs, nFeatures]
        let pinv = computePseudoinverse(nFeatures: nFeatures, order: order, freqScale: freqScale)

        // For each frame, coeffs = pinv @ column
        // pinv is [nCoeffs, nFeatures], column is [nFeatures]
        // Result: [nCoeffs, nFrames]
        let outCount = nCoeffs * nFrames
        let outPtr = UnsafeMutablePointer<Float>.allocate(capacity: outCount)

        data.withUnsafeBufferPointer { src in
            for t in 0..<nFrames {
                for c in 0..<nCoeffs {
                    var val: Double = 0
                    for f in 0..<nFeatures {
                        val += pinv[c * nFeatures + f] * Double(src[f * nFrames + t])
                    }
                    outPtr[c * nFrames + t] = Float(val)
                }
            }
        }

        let outBuffer = UnsafeMutableBufferPointer(start: outPtr, count: outCount)
        return Signal(taking: outBuffer, shape: [nCoeffs, nFrames], sampleRate: data.sampleRate)
    }

    // MARK: - Private

    /// Compute the pseudoinverse of the Vandermonde matrix for polynomial fitting.
    ///
    /// V[i, j] = freq[i]^(order-j) for descending power order (matching np.polyfit).
    /// freq[i] = i * freqScale where freqScale = sr / nFFT.
    ///
    /// Returns pinv as flat array [nCoeffs, nFeatures] in row-major order,
    /// computed in Double precision for numerical stability.
    private static func computePseudoinverse(nFeatures: Int, order: Int, freqScale: Double) -> [Double] {
        let nCoeffs = order + 1

        // Build Vandermonde matrix V of shape [nFeatures, nCoeffs]
        // V[i, j] = freq[i]^(order-j) for descending power order
        // freq[i] = i * freqScale (FFT bin center frequencies)
        var V = [Double](repeating: 0, count: nFeatures * nCoeffs)
        for i in 0..<nFeatures {
            let x = Double(i) * freqScale
            for j in 0..<nCoeffs {
                let power = order - j
                if power == 0 {
                    V[i * nCoeffs + j] = 1.0
                } else {
                    V[i * nCoeffs + j] = pow(x, Double(power))
                }
            }
        }

        // V^T V -> [nCoeffs, nCoeffs]
        var VtV = [Double](repeating: 0, count: nCoeffs * nCoeffs)
        for i in 0..<nCoeffs {
            for j in 0..<nCoeffs {
                var sum = 0.0
                for k in 0..<nFeatures {
                    sum += V[k * nCoeffs + i] * V[k * nCoeffs + j]
                }
                VtV[i * nCoeffs + j] = sum
            }
        }

        // Invert VtV using Gauss-Jordan elimination with partial pivoting
        var inv = [Double](repeating: 0, count: nCoeffs * nCoeffs)
        for i in 0..<nCoeffs {
            inv[i * nCoeffs + i] = 1.0
        }
        var aug = VtV

        for col in 0..<nCoeffs {
            // Partial pivoting
            var maxVal = abs(aug[col * nCoeffs + col])
            var maxRow = col
            for row in (col + 1)..<nCoeffs {
                let val = abs(aug[row * nCoeffs + col])
                if val > maxVal {
                    maxVal = val
                    maxRow = row
                }
            }

            // Swap rows
            if maxRow != col {
                for k in 0..<nCoeffs {
                    let tmp = aug[col * nCoeffs + k]
                    aug[col * nCoeffs + k] = aug[maxRow * nCoeffs + k]
                    aug[maxRow * nCoeffs + k] = tmp

                    let tmp2 = inv[col * nCoeffs + k]
                    inv[col * nCoeffs + k] = inv[maxRow * nCoeffs + k]
                    inv[maxRow * nCoeffs + k] = tmp2
                }
            }

            // Scale pivot row
            let pivot = aug[col * nCoeffs + col]
            guard abs(pivot) > 1e-30 else { continue }
            let scale = 1.0 / pivot
            for k in 0..<nCoeffs {
                aug[col * nCoeffs + k] *= scale
                inv[col * nCoeffs + k] *= scale
            }

            // Eliminate other rows
            for row in 0..<nCoeffs {
                if row == col { continue }
                let factor = aug[row * nCoeffs + col]
                for k in 0..<nCoeffs {
                    aug[row * nCoeffs + k] -= factor * aug[col * nCoeffs + k]
                    inv[row * nCoeffs + k] -= factor * inv[col * nCoeffs + k]
                }
            }
        }

        // pinv = inv(V^T V) * V^T -> [nCoeffs, nFeatures]
        var pinv = [Double](repeating: 0, count: nCoeffs * nFeatures)
        for i in 0..<nCoeffs {
            for j in 0..<nFeatures {
                var sum = 0.0
                for k in 0..<nCoeffs {
                    sum += inv[i * nCoeffs + k] * V[j * nCoeffs + k]
                }
                pinv[i * nFeatures + j] = sum
            }
        }

        return pinv
    }
}
