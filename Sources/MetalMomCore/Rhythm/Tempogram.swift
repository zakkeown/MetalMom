import Accelerate
import Foundation

/// Tempogram computation: time-frequency representations of local tempo.
///
/// Two variants are provided:
///
/// 1. **Autocorrelation tempogram**: For each frame, compute windowed
///    autocorrelation of the onset strength envelope. The result is a 2D
///    array [winLength, nFrames] showing local periodicity at each time
///    position.
///
/// 2. **Fourier tempogram**: For each frame, compute the short-time Fourier
///    transform of the onset strength envelope. The result is a 2D magnitude
///    array [winLength/2+1, nFrames] showing tempo frequencies at each time
///    position.
public enum Tempogram {

    /// Compute local autocorrelation tempogram.
    ///
    /// For each frame, a Hann-windowed segment of the onset strength envelope
    /// is autocorrelated. The stacked autocorrelation columns form the output.
    ///
    /// - Parameters:
    ///   - signal: Input audio signal (1D).
    ///   - sr: Sample rate override. If nil, uses `signal.sampleRate`.
    ///   - hopLength: Hop length for onset envelope. Default 512.
    ///   - nFFT: FFT window size for onset envelope. Default 2048.
    ///   - nMels: Number of mel bands for onset envelope. Default 128.
    ///   - fmin: Minimum frequency for mel filterbank. Default 0.
    ///   - fmax: Maximum frequency. If nil, uses sr/2.
    ///   - center: Center-pad signal. Default true.
    ///   - winLength: Window length for local analysis. Default 384.
    /// - Returns: Signal with shape [winLength, nFrames].
    public static func autocorrelation(
        signal: Signal,
        sr: Int? = nil,
        hopLength: Int? = nil,
        nFFT: Int = 2048,
        nMels: Int = 128,
        fmin: Float = 0,
        fmax: Float? = nil,
        center: Bool = true,
        winLength: Int = 384
    ) -> Signal {
        let sampleRate = sr ?? signal.sampleRate
        let hop = hopLength ?? 512

        // 1. Compute onset strength envelope (aggregated, 1D)
        let oenv = OnsetDetection.onsetStrength(
            signal: signal,
            sr: sampleRate,
            nFFT: nFFT,
            hopLength: hop,
            nMels: nMels,
            fmin: fmin,
            fmax: fmax,
            center: center,
            aggregate: true
        )

        // Extract flat envelope data
        let envLen = oenv.shape.count > 1 ? oenv.shape[1] : oenv.shape[0]
        guard envLen > 0 else {
            return Signal(data: [Float](repeating: 0, count: winLength),
                          shape: [winLength, 1], sampleRate: sampleRate)
        }

        var envData = [Float](repeating: 0, count: envLen)
        oenv.withUnsafeBufferPointer { buf in
            for i in 0..<envLen { envData[i] = buf[i] }
        }

        // 2. Pad with winLength/2 zeros on each side for center alignment
        let halfWin = winLength / 2
        let padded = [Float](repeating: 0, count: halfWin)
                     + envData
                     + [Float](repeating: 0, count: halfWin)

        // 3. Build Hann window
        let hann = makeHannWindow(length: winLength)

        // 4. Number of output frames = envLen (one per onset frame)
        let nFrames = envLen

        // 5. For each frame, extract windowed segment and compute ACF
        var output = [Float](repeating: 0, count: winLength * nFrames)

        for frame in 0..<nFrames {
            // Center of the window in padded array
            let centerIdx = frame + halfWin
            let startIdx = centerIdx - halfWin
            // startIdx should always be >= 0 due to padding

            // Extract and window the segment
            var segment = [Float](repeating: 0, count: winLength)
            for i in 0..<winLength {
                let pIdx = startIdx + i
                if pIdx >= 0 && pIdx < padded.count {
                    segment[i] = padded[pIdx] * hann[i]
                }
            }

            // Compute autocorrelation
            let acf = computeACF(segment)

            // Store column: output is [winLength, nFrames], column-major
            for lag in 0..<winLength {
                output[lag * nFrames + frame] = acf[lag]
            }
        }

        return Signal(data: output, shape: [winLength, nFrames], sampleRate: sampleRate)
    }

    /// Compute Fourier tempogram.
    ///
    /// For each frame, a Hann-windowed segment of the onset strength envelope
    /// is FFT'd and the magnitude of positive frequencies is taken.
    ///
    /// - Parameters:
    ///   - signal: Input audio signal (1D).
    ///   - sr: Sample rate override. If nil, uses `signal.sampleRate`.
    ///   - hopLength: Hop length for onset envelope. Default 512.
    ///   - nFFT: FFT window size for onset envelope. Default 2048.
    ///   - nMels: Number of mel bands for onset envelope. Default 128.
    ///   - fmin: Minimum frequency for mel filterbank. Default 0.
    ///   - fmax: Maximum frequency. If nil, uses sr/2.
    ///   - center: Center-pad signal. Default true.
    ///   - winLength: Window length for local analysis. Default 384.
    /// - Returns: Signal with shape [winLength/2+1, nFrames].
    public static func fourier(
        signal: Signal,
        sr: Int? = nil,
        hopLength: Int? = nil,
        nFFT: Int = 2048,
        nMels: Int = 128,
        fmin: Float = 0,
        fmax: Float? = nil,
        center: Bool = true,
        winLength: Int = 384
    ) -> Signal {
        let sampleRate = sr ?? signal.sampleRate
        let hop = hopLength ?? 512

        // 1. Compute onset strength envelope (aggregated, 1D)
        let oenv = OnsetDetection.onsetStrength(
            signal: signal,
            sr: sampleRate,
            nFFT: nFFT,
            hopLength: hop,
            nMels: nMels,
            fmin: fmin,
            fmax: fmax,
            center: center,
            aggregate: true
        )

        let envLen = oenv.shape.count > 1 ? oenv.shape[1] : oenv.shape[0]
        let nFreqs = winLength / 2 + 1
        guard envLen > 0 else {
            return Signal(data: [Float](repeating: 0, count: nFreqs),
                          shape: [nFreqs, 1], sampleRate: sampleRate)
        }

        var envData = [Float](repeating: 0, count: envLen)
        oenv.withUnsafeBufferPointer { buf in
            for i in 0..<envLen { envData[i] = buf[i] }
        }

        // 2. Pad with winLength/2 zeros on each side
        let halfWin = winLength / 2
        let padded = [Float](repeating: 0, count: halfWin)
                     + envData
                     + [Float](repeating: 0, count: halfWin)

        // 3. Build Hann window
        let hann = makeHannWindow(length: winLength)

        // 4. Determine FFT size (next power of 2 >= winLength)
        let fftSize = nextPowerOf2(winLength)
        let log2n = vDSP_Length(log2(Double(fftSize)))
        guard let fftSetup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else {
            return Signal(data: [Float](repeating: 0, count: nFreqs),
                          shape: [nFreqs, 1], sampleRate: sampleRate)
        }
        defer { vDSP_destroy_fftsetup(fftSetup) }

        let nFrames = envLen

        var output = [Float](repeating: 0, count: nFreqs * nFrames)

        // Reusable buffers for FFT
        let halfFFT = fftSize / 2
        var realBuf = [Float](repeating: 0, count: halfFFT)
        var imagBuf = [Float](repeating: 0, count: halfFFT)

        for frame in 0..<nFrames {
            let centerIdx = frame + halfWin
            let startIdx = centerIdx - halfWin

            // Extract and window
            var segment = [Float](repeating: 0, count: fftSize)
            for i in 0..<winLength {
                let pIdx = startIdx + i
                if pIdx >= 0 && pIdx < padded.count {
                    segment[i] = padded[pIdx] * hann[i]
                }
            }
            // Remaining positions are already zero-padded

            // Pack into split complex for vDSP FFT
            segment.withUnsafeMutableBufferPointer { segBuf in
                realBuf.withUnsafeMutableBufferPointer { rBuf in
                    imagBuf.withUnsafeMutableBufferPointer { iBuf in
                        var splitComplex = DSPSplitComplex(
                            realp: rBuf.baseAddress!,
                            imagp: iBuf.baseAddress!
                        )

                        // Convert real input to split complex (even/odd interleave)
                        segBuf.baseAddress!.withMemoryRebound(to: DSPComplex.self, capacity: halfFFT) { complexPtr in
                            vDSP_ctoz(complexPtr, 2, &splitComplex, 1, vDSP_Length(halfFFT))
                        }

                        // Forward FFT
                        vDSP_fft_zrip(fftSetup, &splitComplex, 1, log2n, FFTDirection(kFFTDirection_Forward))

                        // Compute magnitudes for first nFreqs bins
                        // DC component (bin 0): real part only, imag is packed Nyquist
                        let dcMag = abs(splitComplex.realp[0]) / Float(fftSize)
                        output[0 * nFrames + frame] = dcMag

                        // Bins 1..<halfFFT
                        let limit = min(nFreqs, halfFFT + 1)
                        for k in 1..<limit {
                            if k < halfFFT {
                                let re = splitComplex.realp[k]
                                let im = splitComplex.imagp[k]
                                output[k * nFrames + frame] = sqrt(re * re + im * im) / Float(fftSize) * 2.0
                            } else if k == halfFFT {
                                // Nyquist bin: packed in imagp[0]
                                let nyqMag = abs(splitComplex.imagp[0]) / Float(fftSize)
                                output[k * nFrames + frame] = nyqMag
                            }
                        }
                    }
                }
            }
        }

        return Signal(data: output, shape: [nFreqs, nFrames], sampleRate: sampleRate)
    }

    // MARK: - Private Helpers

    /// Compute unnormalized autocorrelation of a signal.
    private static func computeACF(_ x: [Float]) -> [Float] {
        let n = x.count
        guard n > 0 else { return [] }

        var result = [Float](repeating: 0, count: n)
        for lag in 0..<n {
            var sum: Float = 0
            let count = n - lag
            x.withUnsafeBufferPointer { xBuf in
                vDSP_dotpr(xBuf.baseAddress!, 1,
                           xBuf.baseAddress!.advanced(by: lag), 1,
                           &sum,
                           vDSP_Length(count))
            }
            result[lag] = sum
        }
        return result
    }

    /// Create a Hann (raised cosine) window.
    private static func makeHannWindow(length: Int) -> [Float] {
        guard length > 0 else { return [] }
        var window = [Float](repeating: 0, count: length)
        // Periodic Hann window: w[n] = 0.5 * (1 - cos(2*pi*n / N))
        let scale = 2.0 * Float.pi / Float(length)
        for i in 0..<length {
            window[i] = 0.5 * (1.0 - cos(scale * Float(i)))
        }
        return window
    }

    /// Next power of 2 >= n.
    private static func nextPowerOf2(_ n: Int) -> Int {
        var v = 1
        while v < n { v <<= 1 }
        return v
    }
}
