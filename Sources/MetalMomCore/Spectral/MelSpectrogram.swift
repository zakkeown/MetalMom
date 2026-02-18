import Foundation
import Accelerate

/// Mel spectrogram computation: STFT magnitude -> mel filterbank multiplication -> mel spectrogram.
///
/// The mel spectrogram is: `melFilterbank @ |STFT(signal)|^power`
///
/// This is one of the most commonly used audio features and serves as
/// the foundation for MFCC and many other downstream features.
public enum MelSpectrogram {

    /// Compute mel spectrogram.
    ///
    /// Returns a `Signal` with shape `[nMels, nFrames]`, row-major.
    ///
    /// - Parameters:
    ///   - signal: Input audio signal (1D).
    ///   - sr: Sample rate override. If `nil`, uses `signal.sampleRate`.
    ///   - nFFT: FFT window size. Default 2048.
    ///   - hopLength: Hop length in samples. Default `nFFT / 4`.
    ///   - winLength: Window length. Default `nFFT`.
    ///   - center: If `true`, pad signal so frames are centred. Default `true`.
    ///   - power: Exponent for the magnitude spectrogram. 1.0 = amplitude, 2.0 = power. Default 2.0.
    ///   - nMels: Number of mel bands. Default 128.
    ///   - fMin: Lowest frequency (Hz) for mel filterbank. Default 0.0.
    ///   - fMax: Highest frequency (Hz). If `nil`, uses `sr / 2`.
    /// - Returns: Mel spectrogram `Signal` with shape `[nMels, nFrames]`.
    public static func compute(
        signal: Signal,
        sr: Int? = nil,
        nFFT: Int = 2048,
        hopLength: Int? = nil,
        winLength: Int? = nil,
        center: Bool = true,
        power: Float = 2.0,
        nMels: Int = 128,
        fMin: Float = 0.0,
        fMax: Float? = nil
    ) -> Signal {
        let sampleRate = sr ?? signal.sampleRate

        // 1. Compute STFT magnitude spectrogram: shape [nFreqs, nFrames]
        let stftMag = STFT.compute(
            signal: signal,
            nFFT: nFFT,
            hopLength: hopLength,
            winLength: winLength,
            center: center
        )

        let nFreqs = stftMag.shape[0]
        let nFrames = stftMag.shape[1]
        let stftCount = stftMag.count

        // 2. Apply power (element-wise)
        // power=1.0 keeps amplitude, power=2.0 gives power spectrogram
        let poweredPtr = UnsafeMutablePointer<Float>.allocate(capacity: stftCount)
        defer { poweredPtr.deallocate() }

        if power == 1.0 {
            // No transformation needed, just copy
            stftMag.withUnsafeBufferPointer { src in
                poweredPtr.initialize(from: src.baseAddress!, count: stftCount)
            }
        } else if power == 2.0 {
            // Square each element using vDSP_vsq
            stftMag.withUnsafeBufferPointer { src in
                vDSP_vsq(src.baseAddress!, 1, poweredPtr, 1, vDSP_Length(stftCount))
            }
        } else {
            // General power using vForce
            stftMag.withUnsafeBufferPointer { src in
                // Copy source to mutable buffer for in-place operation
                poweredPtr.initialize(from: src.baseAddress!, count: stftCount)
            }
            // Use vvpowsf: poweredPtr[i] = poweredPtr[i]^power
            var count = Int32(stftCount)
            var powerArr = [Float](repeating: power, count: stftCount)
            vvpowf(poweredPtr, &powerArr, poweredPtr, &count)
        }

        // 3. Get mel filterbank: shape [nMels, nFreqs]
        let melFB = FilterBank.mel(
            sr: sampleRate,
            nFFT: nFFT,
            nMels: nMels,
            fMin: fMin,
            fMax: fMax
        )

        // 4. Matrix multiply: melFB [nMels, nFreqs] @ powered [nFreqs, nFrames] = [nMels, nFrames]
        let outCount = nMels * nFrames
        let outPtr = UnsafeMutablePointer<Float>.allocate(capacity: outCount)
        outPtr.initialize(repeating: 0, count: outCount)

        melFB.withUnsafeBufferPointer { fbBuf in
            vDSP_mmul(
                fbBuf.baseAddress!, 1,       // A: melFB [nMels x nFreqs], row-major
                poweredPtr, 1,               // B: powered [nFreqs x nFrames], row-major
                outPtr, 1,                   // C: output [nMels x nFrames], row-major
                vDSP_Length(nMels),
                vDSP_Length(nFrames),
                vDSP_Length(nFreqs)
            )
        }

        let outBuffer = UnsafeMutableBufferPointer(start: outPtr, count: outCount)
        return Signal(taking: outBuffer, shape: [nMels, nFrames], sampleRate: sampleRate)
    }
}
