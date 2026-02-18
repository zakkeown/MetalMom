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
        let state = Profiler.shared.begin("MelSpectrogram")
        defer { Profiler.shared.end("MelSpectrogram", state) }
        let sampleRate = sr ?? signal.sampleRate

        // 1. Compute spectrogram with the requested power.
        //    For power=2.0, compute power spectrogram directly (r^2 + i^2)
        //    to avoid the precision loss of sqrt(r^2+i^2) followed by squaring.
        let powered: Signal
        if power == 2.0 {
            // Direct power spectrogram: avoids sqrt roundtrip
            powered = STFT.computePowerSpectrogram(
                signal: signal,
                nFFT: nFFT,
                hopLength: hopLength,
                winLength: winLength,
                center: center
            )
        } else {
            // Compute STFT magnitude, then apply power
            let stftMag = STFT.compute(
                signal: signal,
                nFFT: nFFT,
                hopLength: hopLength,
                winLength: winLength,
                center: center
            )

            if power == 1.0 {
                powered = stftMag
            } else {
                let stftCount = stftMag.count
                let poweredPtr = UnsafeMutablePointer<Float>.allocate(capacity: stftCount)
                stftMag.withUnsafeBufferPointer { src in
                    poweredPtr.initialize(from: src.baseAddress!, count: stftCount)
                }
                var count = Int32(stftCount)
                var powerArr = [Float](repeating: power, count: stftCount)
                vvpowf(poweredPtr, &powerArr, poweredPtr, &count)
                let outBuf = UnsafeMutableBufferPointer(start: poweredPtr, count: stftCount)
                powered = Signal(taking: outBuf, shape: stftMag.shape, sampleRate: stftMag.sampleRate)
            }
        }

        let nFreqs = powered.shape[0]
        let nFrames = powered.shape[1]

        // 2. Get mel filterbank: shape [nMels, nFreqs]
        //    (nFreqs = nFFT/2 + 1)
        let melFB = FilterBank.mel(
            sr: sampleRate,
            nFFT: nFFT,
            nMels: nMels,
            fMin: fMin,
            fMax: fMax
        )

        // 3. Matrix multiply: melFB [nMels, nFreqs] @ powered [nFreqs, nFrames] = [nMels, nFrames]
        //
        // Use GPU (MPS) when Metal is available and the workload is large enough
        // to amortise data-transfer overhead.  Falls through to CPU on failure.
        let matmulSize = nMels * nFreqs * nFrames
        let useGPU = MetalBackend.shared != nil
            && matmulSize > (MetalBackend.shared?.chipProfile.threshold(for: .matmul) ?? Int.max)

        if useGPU {
            let melWeights: [Float] = melFB.withUnsafeBufferPointer { Array($0) }
            let poweredArray: [Float] = powered.withUnsafeBufferPointer { Array($0) }

            if let gpuResult = MetalMatmul.multiply(
                a: melWeights, aRows: nMels, aCols: nFreqs,
                b: poweredArray, bRows: nFreqs, bCols: nFrames
            ) {
                let outPtr = UnsafeMutablePointer<Float>.allocate(capacity: gpuResult.count)
                gpuResult.withUnsafeBufferPointer { src in
                    outPtr.initialize(from: src.baseAddress!, count: gpuResult.count)
                }
                let outBuffer = UnsafeMutableBufferPointer(start: outPtr, count: gpuResult.count)
                return Signal(taking: outBuffer, shape: [nMels, nFrames], sampleRate: sampleRate)
            }
            // GPU path failed — fall through to CPU path below.
        }

        // CPU path: vDSP matrix multiply (Accelerate).
        let outCount = nMels * nFrames
        let outPtr = UnsafeMutablePointer<Float>.allocate(capacity: outCount)
        outPtr.initialize(repeating: 0, count: outCount)

        powered.withUnsafeBufferPointer { poweredBuf in
            melFB.withUnsafeBufferPointer { fbBuf in
                vDSP_mmul(
                    fbBuf.baseAddress!, 1,       // A: melFB [nMels x nFreqs], row-major
                    poweredBuf.baseAddress!, 1,  // B: powered [nFreqs x nFrames], row-major
                    outPtr, 1,                   // C: output [nMels x nFrames], row-major
                    vDSP_Length(nMels),
                    vDSP_Length(nFrames),
                    vDSP_Length(nFreqs)
                )
            }
        }

        let outBuffer = UnsafeMutableBufferPointer(start: outPtr, count: outCount)
        return Signal(taking: outBuffer, shape: [nMels, nFrames], sampleRate: sampleRate)
    }
}
