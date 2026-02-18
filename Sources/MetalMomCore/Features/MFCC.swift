import Accelerate
import Foundation

/// Mel-frequency cepstral coefficients (MFCCs).
///
/// MFCC = DCT-II of log-scaled mel spectrogram, keeping the first `nMFCC` coefficients.
/// This is one of the most widely used audio features in speech and music analysis.
///
/// Pipeline: audio -> STFT -> mel filterbank -> power_to_dB -> DCT-II -> truncate to nMFCC
public enum MFCC {

    /// Compute MFCCs from an audio signal.
    ///
    /// Returns a `Signal` with shape `[nMFCC, nFrames]`, row-major.
    ///
    /// - Parameters:
    ///   - signal: Input audio signal (1D).
    ///   - sr: Sample rate override. If `nil`, uses `signal.sampleRate`.
    ///   - nMFCC: Number of MFCC coefficients to return. Default 20.
    ///   - nFFT: FFT window size. Default 2048.
    ///   - hopLength: Hop length in samples. Default `nFFT / 4`.
    ///   - winLength: Window length. Default `nFFT`.
    ///   - nMels: Number of mel bands. Default 128.
    ///   - fMin: Lowest frequency (Hz) for mel filterbank. Default 0.0.
    ///   - fMax: Highest frequency (Hz). If `nil`, uses `sr / 2`.
    ///   - center: If `true`, pad signal so frames are centred. Default `true`.
    /// - Returns: MFCC `Signal` with shape `[nMFCC, nFrames]`.
    public static func compute(
        signal: Signal,
        sr: Int? = nil,
        nMFCC: Int = 20,
        nFFT: Int = 2048,
        hopLength: Int? = nil,
        winLength: Int? = nil,
        nMels: Int = 128,
        fMin: Float = 0.0,
        fMax: Float? = nil,
        center: Bool = true
    ) -> Signal {
        let sampleRate = sr ?? signal.sampleRate

        // 1. Compute mel spectrogram (power=2.0) -> shape [nMels, nFrames]
        let melSpec = MelSpectrogram.compute(
            signal: signal,
            sr: sampleRate,
            nFFT: nFFT,
            hopLength: hopLength,
            winLength: winLength,
            center: center,
            power: 2.0,
            nMels: nMels,
            fMin: fMin,
            fMax: fMax
        )

        // 2. Convert to dB scale (power_to_db equivalent)
        let logMel = Scaling.powerToDb(melSpec)

        let melRows = logMel.shape[0]  // nMels
        let nFrames = logMel.shape[1]

        // 3. Apply DCT-II along the mel axis for each frame, keep first nMFCC coefficients
        //    For each frame column, extract the nMels-length vector, apply DCT-II, take first nMFCC values.
        //    Use ortho normalization to match librosa/scipy behavior.
        let actualNMFCC = min(nMFCC, melRows)
        let outCount = actualNMFCC * nFrames
        let outPtr = UnsafeMutablePointer<Float>.allocate(capacity: outCount)
        outPtr.initialize(repeating: 0, count: outCount)

        // Pre-compute DCT-II basis matrix with ortho normalization:
        //   dctBasis[k, n] = cos(pi * k * (2n + 1) / (2N)) * norm[k]
        //   where norm[0] = sqrt(1/N), norm[k>0] = sqrt(2/N)
        // Matrix shape: [actualNMFCC, melRows]
        let N = melRows
        let basisCount = actualNMFCC * N
        let basis = UnsafeMutablePointer<Float>.allocate(capacity: basisCount)
        defer { basis.deallocate() }

        let norm0 = sqrtf(1.0 / Float(N))
        let normK = sqrtf(2.0 / Float(N))

        for k in 0..<actualNMFCC {
            let norm = (k == 0) ? norm0 : normK
            for n in 0..<N {
                let angle = Float.pi * Float(k) * (2.0 * Float(n) + 1.0) / (2.0 * Float(N))
                basis[k * N + n] = cosf(angle) * norm
            }
        }

        // logMel is row-major [nMels, nFrames]: row m, frame f -> logMel[m * nFrames + f]
        // We need: for each frame f, extract column [logMel[0*nFrames+f], logMel[1*nFrames+f], ..., logMel[(nMels-1)*nFrames+f]]
        // Then multiply: dctBasis [actualNMFCC, nMels] @ column [nMels, 1] = [actualNMFCC, 1]
        //
        // But we can do this as a matrix multiply:
        //   dctBasis [actualNMFCC, nMels] @ logMel [nMels, nFrames] = out [actualNMFCC, nFrames]

        logMel.withUnsafeBufferPointer { logMelBuf in
            vDSP_mmul(
                basis, 1,                         // A: dctBasis [actualNMFCC x nMels]
                logMelBuf.baseAddress!, 1,        // B: logMel [nMels x nFrames]
                outPtr, 1,                        // C: output [actualNMFCC x nFrames]
                vDSP_Length(actualNMFCC),
                vDSP_Length(nFrames),
                vDSP_Length(N)
            )
        }

        let outBuffer = UnsafeMutableBufferPointer(start: outPtr, count: outCount)
        return Signal(taking: outBuffer, shape: [actualNMFCC, nFrames], sampleRate: sampleRate)
    }
}
