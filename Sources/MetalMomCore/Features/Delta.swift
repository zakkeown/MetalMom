import Accelerate
import Foundation

/// Delta (derivative) features and feature memory stacking.
///
/// Delta features compute the time derivative of a feature matrix using
/// Savitzky-Golay filtering.  Given a feature matrix of shape
/// `[nFeatures, nFrames]`, the delta computes the first-order derivative
/// along the time axis (axis=1, row-major).
///
/// `stackMemory` creates time-lagged copies of features for use in
/// sequential models (e.g. HMMs, LSTMs).
public enum Delta {

    // MARK: - SG Coefficient Computation

    /// Compute Savitzky-Golay filter coefficients for a given derivative order.
    ///
    /// Uses the Vandermonde pseudoinverse approach:
    /// 1. Build Vandermonde matrix V[i, j] = (i - halfW)^j  for i in 0..<width, j in 0..polyorder
    /// 2. Compute pseudoinverse via (V^T V)^{-1} V^T
    /// 3. Extract row `deriv`, multiply by deriv!
    ///
    /// Returns coefficients in scipy's correlation order (coeffs[0] corresponds
    /// to the rightmost sample in the window).
    private static func savgolCoeffs(width: Int, polyorder: Int, deriv: Int) -> [Float] {
        let halfW = width / 2
        let cols = polyorder + 1

        // Build Vandermonde matrix V (width x cols), row-major: V[i, j] = (i - halfW)^j
        var V = [[Double]](repeating: [Double](repeating: 0, count: cols), count: width)
        for i in 0..<width {
            let x = Double(i - halfW)
            var xPow = 1.0
            for j in 0..<cols {
                V[i][j] = xPow
                xPow *= x
            }
        }

        // Compute V^T V (cols x cols), row-major
        var VtV = [[Double]](repeating: [Double](repeating: 0, count: cols), count: cols)
        for i in 0..<cols {
            for j in 0..<cols {
                var sum = 0.0
                for k in 0..<width {
                    sum += V[k][i] * V[k][j]
                }
                VtV[i][j] = sum
            }
        }

        // Invert V^T V using Gauss-Jordan elimination (small matrix: 2x2 or 3x3)
        var augmented = [[Double]](repeating: [Double](repeating: 0, count: 2 * cols), count: cols)
        for i in 0..<cols {
            for j in 0..<cols {
                augmented[i][j] = VtV[i][j]
                augmented[i][cols + j] = (i == j) ? 1.0 : 0.0
            }
        }

        for i in 0..<cols {
            // Find pivot
            var maxVal = abs(augmented[i][i])
            var maxRow = i
            for k in (i + 1)..<cols {
                if abs(augmented[k][i]) > maxVal {
                    maxVal = abs(augmented[k][i])
                    maxRow = k
                }
            }
            if maxRow != i {
                augmented.swapAt(i, maxRow)
            }

            let pivot = augmented[i][i]
            guard abs(pivot) > 1e-15 else {
                return [Float](repeating: 0, count: width)
            }

            // Scale row
            for j in 0..<(2 * cols) {
                augmented[i][j] /= pivot
            }

            // Eliminate column
            for k in 0..<cols where k != i {
                let factor = augmented[k][i]
                for j in 0..<(2 * cols) {
                    augmented[k][j] -= factor * augmented[i][j]
                }
            }
        }

        // Extract inverse from augmented matrix
        var VtVinv = [[Double]](repeating: [Double](repeating: 0, count: cols), count: cols)
        for i in 0..<cols {
            for j in 0..<cols {
                VtVinv[i][j] = augmented[i][cols + j]
            }
        }

        // Compute pinv = (V^T V)^{-1} V^T  -> row `deriv` of pinv
        // pinv[deriv, k] = sum_j (VtVinv[deriv, j] * V^T[j, k])
        //                = sum_j (VtVinv[deriv, j] * V[k][j])
        var coeffs = [Double](repeating: 0, count: width)
        for k in 0..<width {
            var sum = 0.0
            for j in 0..<cols {
                sum += VtVinv[deriv][j] * V[k][j]
            }
            coeffs[k] = sum
        }

        // Multiply by deriv!
        var factorial = 1.0
        for i in 1...deriv {
            factorial *= Double(i)
        }

        return coeffs.map { Float($0 * factorial) }
    }

    /// Compute delta (derivative) features using Savitzky-Golay filtering.
    ///
    /// Matches librosa.feature.delta semantics with `mode='interp'`.
    ///
    /// - Parameters:
    ///   - data: Input feature matrix, shape `[nFeatures, nFrames]`, row-major.
    ///   - width: Full window width (odd integer >= 3). Default 9.
    ///     This matches librosa's `width` parameter.
    ///   - order: Derivative order (1 = delta, 2 = delta-delta). Default 1.
    ///   - axis: Axis along which to compute delta. Default 1 (time axis).
    /// - Returns: Delta features, same shape as input.
    public static func compute(
        data: Signal,
        width: Int = 9,
        order: Int = 1,
        axis: Int = 1
    ) -> Signal {
        precondition(data.shape.count == 2, "Delta requires 2D input")
        precondition(order >= 1, "Order must be >= 1")
        precondition(width >= 3 && width % 2 == 1, "Width must be odd integer >= 3")

        let nFeatures = data.shape[0]
        let nFrames = data.shape[1]
        let halfW = width / 2

        // Compute SG coefficients for the requested derivative order directly
        // (not recursive -- matches librosa which calls savgol_filter with deriv=order)
        let polyorder = order  // polyorder = order, matching librosa default
        let sgCoeffs = savgolCoeffs(width: width, polyorder: polyorder, deriv: order)

        // sgCoeffs are in natural order (Vandermonde): sgCoeffs[k] corresponds to
        // sample at position (k - halfW) relative to center.
        // Our convolution loop: result[t] = sum(kernel[k] * data[t - halfW + k])
        // where k=0 is leftmost, which matches natural order directly.
        let kernel = sgCoeffs

        let outCount = nFeatures * nFrames
        let outPtr = UnsafeMutablePointer<Float>.allocate(capacity: outCount)

        data.withUnsafeBufferPointer { src in
            for f in 0..<nFeatures {
                let rowOffset = f * nFrames

                // Interior frames: apply SG filter via convolution
                for t in halfW..<(nFrames - halfW) {
                    var val: Float = 0
                    for k in 0..<width {
                        val += kernel[k] * src[rowOffset + t - halfW + k]
                    }
                    outPtr[rowOffset + t] = val
                }

                // Edge frames: interp mode
                // Fit a polynomial of degree `polyorder` to the first/last `width` points,
                // evaluate its `order`-th derivative at each edge position.
                //
                // For the left edge (t < halfW):
                //   Fit poly to data[0..width-1], evaluate deriv at position t
                //   deriv[t] = sum over j>=order of (j! / (j-order)!) * a[j] * (t - halfW)^(j-order)
                //   where a = pinv(V) . y  (polynomial coefficients)
                //
                // For first derivative (order=1, polyorder=1):
                //   deriv = a[1] (constant), which is the slope
                //
                // For second derivative (order=2, polyorder=2):
                //   deriv = 2 * a[2] (constant), which is 2x the quadratic coefficient

                // Compute polynomial coefficients for left edge
                do {
                    // Vandermonde pinv row(s) for positions in left window
                    // We need the polynomial coefficients a[j] for y = a[0] + a[1]*x + ... + a[p]*x^p
                    // where x[i] = i - halfW for i in 0..<width
                    // a = pinv(V) . y
                    let cols = polyorder + 1
                    var V = [Double](repeating: 0, count: width * cols)
                    for i in 0..<width {
                        let x = Double(i - halfW)
                        var xPow = 1.0
                        for j in 0..<cols {
                            V[j * width + i] = xPow
                            xPow *= x
                        }
                    }

                    // Compute pinv(V) using least-squares (dgels)
                    // Or reuse the same approach: compute (V^T V)^{-1} V^T
                    // Then a = pinv . y
                    // The derivative at position x is:
                    //   d/dx: a[1] + 2*a[2]*x + 3*a[3]*x^2 + ...
                    //   d2/dx2: 2*a[2] + 6*a[3]*x + ...
                    // For polyorder=order, the highest term is order, so:
                    //   order-th derivative = order! * a[order] (constant)

                    // Since polyorder == order, the order-th derivative is constant = order! * a[order]
                    // a[order] = pinv[order, :] . y
                    // And the coefficients for pinv[order, :] are exactly sgCoeffs / order!
                    // (since sgCoeffs = order! * pinv[order, :])
                    // So the derivative = sgCoeffs . y (dot product)

                    // Left edge: y = first `width` samples
                    var leftDeriv: Float = 0
                    for k in 0..<width {
                        leftDeriv += sgCoeffs[k] * src[rowOffset + k]
                    }
                    for t in 0..<halfW {
                        outPtr[rowOffset + t] = leftDeriv
                    }
                }

                // Right edge: fit to last `width` points
                do {
                    let startIdx = nFrames - width
                    var rightDeriv: Float = 0
                    for k in 0..<width {
                        rightDeriv += sgCoeffs[k] * src[rowOffset + startIdx + k]
                    }
                    for t in (nFrames - halfW)..<nFrames {
                        outPtr[rowOffset + t] = rightDeriv
                    }
                }
            }
        }

        let outBuffer = UnsafeMutableBufferPointer(start: outPtr, count: outCount)
        return Signal(taking: outBuffer, shape: [nFeatures, nFrames], sampleRate: data.sampleRate)
    }

    /// Stack consecutive frames of a feature matrix.
    ///
    /// Creates a vertically stacked matrix of time-delayed copies.
    /// Frames before the start are zero-padded (matching librosa default).
    ///
    /// - Parameters:
    ///   - data: Input feature matrix, shape `[nFeatures, nFrames]`, row-major.
    ///   - nSteps: Number of time steps to stack. Default 2.
    ///   - delay: Number of frames to delay per step. Default 1.
    /// - Returns: Stacked features, shape `[nFeatures * nSteps, nFrames]`.
    public static func stackMemory(
        data: Signal,
        nSteps: Int = 2,
        delay: Int = 1
    ) -> Signal {
        precondition(data.shape.count == 2, "stackMemory requires 2D input")

        let nFeatures = data.shape[0]
        let nFrames = data.shape[1]
        let outRows = nFeatures * nSteps
        let outCount = outRows * nFrames
        let outPtr = UnsafeMutablePointer<Float>.allocate(capacity: outCount)
        outPtr.initialize(repeating: 0, count: outCount)  // Zero-initialize for padding

        data.withUnsafeBufferPointer { src in
            for step in 0..<nSteps {
                let shift = step * delay
                for f in 0..<nFeatures {
                    let outRow = step * nFeatures + f
                    for t in 0..<nFrames {
                        let srcT = t - shift
                        // Zero-pad: only copy if source frame is in valid range
                        if srcT >= 0 && srcT < nFrames {
                            outPtr[outRow * nFrames + t] = src[f * nFrames + srcT]
                        }
                        // else: already zero from initialization
                    }
                }
            }
        }

        let outBuffer = UnsafeMutableBufferPointer(start: outPtr, count: outCount)
        return Signal(taking: outBuffer, shape: [outRows, nFrames], sampleRate: data.sampleRate)
    }
}
