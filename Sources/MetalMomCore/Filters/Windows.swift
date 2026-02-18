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

    // MARK: - General Cosine Window

    /// Generate a general cosine window.
    ///
    /// The general cosine window is defined as:
    /// ```
    /// w[n] = a_0 - a_1 * cos(2*pi*n/(M-1)) + a_2 * cos(4*pi*n/(M-1)) - ...
    /// ```
    /// with alternating signs. Hann, Hamming, and Blackman are special cases.
    ///
    /// - Parameters:
    ///   - length: Window length in samples.
    ///   - coefficients: Cosine coefficients `[a_0, a_1, a_2, ...]`.
    ///   - periodic: If `true` (default), generates a periodic window.
    /// - Returns: Array of `Float` window values.
    public static func generalCosine(length: Int, coefficients: [Float],
                                     periodic: Bool = true) -> [Float] {
        guard length > 0 else { return [] }
        guard length > 1 else { return [1.0] }
        guard !coefficients.isEmpty else { return [Float](repeating: 0, count: length) }

        let M = periodic ? length + 1 : length
        var window = [Float](repeating: coefficients[0], count: M)

        // Temporary buffer for cosine terms
        var ramp = [Float](repeating: 0, count: M)

        for (k, coeff) in coefficients.enumerated() where k > 0 {
            ramp.withUnsafeMutableBufferPointer { rampBuf in
                window.withUnsafeMutableBufferPointer { winBuf in
                    let rampPtr = rampBuf.baseAddress!
                    let winPtr = winBuf.baseAddress!

                    // Build ramp: 2*pi*k*n / (M-1)
                    var start: Float = 0
                    var step: Float = 2.0 * .pi * Float(k) / Float(M - 1)
                    vDSP_vramp(&start, &step, rampPtr, 1, vDSP_Length(M))

                    // cos(ramp)
                    var count = Int32(M)
                    vvcosf(rampPtr, rampPtr, &count)

                    // Alternating sign: subtract for odd k, add for even k
                    // w += sign * coeff * cos(...)
                    let sign: Float = (k % 2 == 1) ? -1.0 : 1.0
                    var scaledCoeff = sign * coeff
                    vDSP_vsma(rampPtr, 1, &scaledCoeff, winPtr, 1, winPtr, 1, vDSP_Length(M))
                }
            }
        }

        if periodic {
            return Array(window.prefix(length))
        }
        return window
    }

    // MARK: - Hamming Window

    /// Generate a Hamming window.
    ///
    /// `w[n] = 0.54 - 0.46 * cos(2*pi*n / (M-1))`
    ///
    /// - Parameters:
    ///   - length: Window length in samples.
    ///   - periodic: If `true` (default), generates a periodic window.
    /// - Returns: Array of `Float` window values.
    public static func hamming(length: Int, periodic: Bool = true) -> [Float] {
        return generalCosine(length: length, coefficients: [0.54, 0.46], periodic: periodic)
    }

    // MARK: - Blackman Window

    /// Generate a Blackman window.
    ///
    /// `w[n] = 0.42 - 0.5 * cos(2*pi*n/(M-1)) + 0.08 * cos(4*pi*n/(M-1))`
    ///
    /// - Parameters:
    ///   - length: Window length in samples.
    ///   - periodic: If `true` (default), generates a periodic window.
    /// - Returns: Array of `Float` window values.
    public static func blackman(length: Int, periodic: Bool = true) -> [Float] {
        return generalCosine(length: length, coefficients: [0.42, 0.5, 0.08], periodic: periodic)
    }

    // MARK: - Bartlett Window

    /// Generate a Bartlett (triangular, zero-endpoint) window.
    ///
    /// `w[n] = 1 - |2*n/(M-1) - 1|`
    ///
    /// The Bartlett window always has zero endpoints. For a triangle that does
    /// not go to zero, use ``triangular(length:periodic:)``.
    ///
    /// - Parameters:
    ///   - length: Window length in samples.
    ///   - periodic: If `true` (default), generates a periodic window.
    /// - Returns: Array of `Float` window values.
    public static func bartlett(length: Int, periodic: Bool = true) -> [Float] {
        guard length > 0 else { return [] }
        guard length > 1 else { return [1.0] }

        let M = periodic ? length + 1 : length
        var window = [Float](repeating: 0, count: M)

        window.withUnsafeMutableBufferPointer { buf in
            let ptr = buf.baseAddress!

            // Build ramp: 2*n / (M-1)
            var start: Float = 0
            var step: Float = 2.0 / Float(M - 1)
            vDSP_vramp(&start, &step, ptr, 1, vDSP_Length(M))

            // Subtract 1: ramp - 1
            var negOne: Float = -1.0
            vDSP_vsadd(ptr, 1, &negOne, ptr, 1, vDSP_Length(M))

            // Absolute value
            vDSP_vabs(ptr, 1, ptr, 1, vDSP_Length(M))

            // 1 - |...|  =>  negate then add 1
            var negativeOne: Float = -1.0
            var one: Float = 1.0
            vDSP_vsmsa(ptr, 1, &negativeOne, &one, ptr, 1, vDSP_Length(M))
        }

        if periodic {
            return Array(window.prefix(length))
        }
        return window
    }

    // MARK: - Kaiser Window

    /// Modified Bessel function of the first kind, order 0.
    ///
    /// Uses the series expansion: `I0(x) = sum_{k=0}^{N} ((x/2)^k / k!)^2`
    /// with 25 terms, sufficient for float32 precision.
    private static func besselI0(_ x: Float) -> Float {
        // I0(x) = sum_{k=0}^{N} ( (x/2)^k / k! )^2
        var sum: Float = 1.0
        let halfX = x / 2.0
        var term: Float = 1.0 // (x/2)^k / k!
        for k in 1...25 {
            term *= halfX / Float(k)
            sum += term * term
        }
        return sum
    }

    /// Generate a Kaiser window.
    ///
    /// `w[n] = I0(beta * sqrt(1 - (2*n/(M-1) - 1)^2)) / I0(beta)`
    ///
    /// - Parameters:
    ///   - length: Window length in samples.
    ///   - beta: Shape parameter. Default 12.0.
    ///     - `beta = 0` yields a rectangular window.
    ///     - Larger beta values give narrower main lobes.
    ///   - periodic: If `true` (default), generates a periodic window.
    /// - Returns: Array of `Float` window values.
    public static func kaiser(length: Int, beta: Float = 12.0,
                              periodic: Bool = true) -> [Float] {
        guard length > 0 else { return [] }
        guard length > 1 else { return [1.0] }

        let M = periodic ? length + 1 : length
        let denominator = besselI0(beta)
        var window = [Float](repeating: 0, count: M)

        // Build the window using vectorized operations where possible.
        window.withUnsafeMutableBufferPointer { buf in
            let ptr = buf.baseAddress!

            // Build ramp: 2*n / (M-1) - 1  for n in 0..<M
            var start: Float = -1.0
            var step: Float = 2.0 / Float(M - 1)
            vDSP_vramp(&start, &step, ptr, 1, vDSP_Length(M))

            // Square: (2*n/(M-1) - 1)^2
            vDSP_vsq(ptr, 1, ptr, 1, vDSP_Length(M))

            // 1 - x^2
            var negOne: Float = -1.0
            var one: Float = 1.0
            vDSP_vsmsa(ptr, 1, &negOne, &one, ptr, 1, vDSP_Length(M))

            // Clamp to >= 0 (numerical safety near endpoints)
            var zero: Float = 0.0
            vDSP_vthres(ptr, 1, &zero, ptr, 1, vDSP_Length(M))

            // sqrt(1 - x^2)
            var count = Int32(M)
            vvsqrtf(ptr, ptr, &count)

            // beta * sqrt(...)
            var b = beta
            vDSP_vsmul(ptr, 1, &b, ptr, 1, vDSP_Length(M))
        }

        // Apply I0 element-wise (no vectorized I0 in Accelerate for float)
        for i in 0..<M {
            window[i] = besselI0(window[i]) / denominator
        }

        if periodic {
            return Array(window.prefix(length))
        }
        return window
    }

    // MARK: - Rectangular Window

    /// Generate a rectangular (boxcar) window — all ones.
    ///
    /// - Parameter length: Window length in samples.
    /// - Returns: Array of `Float` ones.
    public static func rectangular(length: Int) -> [Float] {
        guard length > 0 else { return [] }
        return [Float](repeating: 1.0, count: length)
    }

    // MARK: - Triangular Window

    /// Generate a triangular window.
    ///
    /// Unlike Bartlett, the triangular window does not necessarily go to zero at
    /// the endpoints:
    /// `w[n] = 1 - |2*(n - (M-1)/2) / (M+1)|`
    ///
    /// - Parameters:
    ///   - length: Window length in samples.
    ///   - periodic: If `true` (default), generates a periodic window.
    /// - Returns: Array of `Float` window values.
    public static func triangular(length: Int, periodic: Bool = true) -> [Float] {
        guard length > 0 else { return [] }
        guard length > 1 else { return [1.0] }

        let M = periodic ? length + 1 : length
        var window = [Float](repeating: 0, count: M)

        window.withUnsafeMutableBufferPointer { buf in
            let ptr = buf.baseAddress!

            // Build ramp: n for n in 0..<M
            var start: Float = 0
            var step: Float = 1.0
            vDSP_vramp(&start, &step, ptr, 1, vDSP_Length(M))

            // n - (M-1)/2
            var center: Float = -Float(M - 1) / 2.0
            vDSP_vsadd(ptr, 1, &center, ptr, 1, vDSP_Length(M))

            // 2 * (n - (M-1)/2) / (M+1)
            var scale: Float = 2.0 / Float(M + 1)
            vDSP_vsmul(ptr, 1, &scale, ptr, 1, vDSP_Length(M))

            // |...|
            vDSP_vabs(ptr, 1, ptr, 1, vDSP_Length(M))

            // 1 - |...|
            var negativeOne: Float = -1.0
            var one: Float = 1.0
            vDSP_vsmsa(ptr, 1, &negativeOne, &one, ptr, 1, vDSP_Length(M))
        }

        if periodic {
            return Array(window.prefix(length))
        }
        return window
    }
}
