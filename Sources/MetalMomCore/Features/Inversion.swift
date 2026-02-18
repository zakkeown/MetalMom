import Foundation
import Accelerate

/// Feature inversion: reconstruct audio from spectral features.
///
/// Provides functions to invert mel spectrograms and MFCCs back to audio,
/// using pseudo-inverse filterbanks and the Griffin-Lim algorithm.
public enum Inversion {

    // MARK: - Mel to STFT

    /// Approximate STFT magnitude from a mel spectrogram.
    ///
    /// Uses the pseudo-inverse of the mel filterbank to map mel-frequency
    /// bins back to linear-frequency bins.
    ///
    /// - Parameters:
    ///   - melSpectrogram: Mel spectrogram with shape `[nMels, nFrames]`.
    ///   - sr: Sample rate in Hz.
    ///   - nFFT: FFT window size. Default 2048.
    ///   - power: The exponent used when the mel spectrogram was computed.
    ///     If `power > 1`, the result is raised to `1/power` to obtain amplitude.
    ///     Default 1.0 (no root applied; input is already in amplitude domain).
    ///   - fMin: Lowest mel filter frequency in Hz. Default 0.0.
    ///   - fMax: Highest mel filter frequency in Hz. If `nil`, uses `sr / 2`.
    /// - Returns: Approximate STFT magnitude with shape `[nFreqs, nFrames]`.
    public static func melToSTFT(
        melSpectrogram: Signal,
        sr: Int,
        nFFT: Int = 2048,
        power: Float = 1.0,
        fMin: Float = 0.0,
        fMax: Float? = nil
    ) -> Signal {
        precondition(melSpectrogram.shape.count == 2,
                     "melToSTFT requires 2D input [nMels, nFrames]")

        let nMels = melSpectrogram.shape[0]
        let nFrames = melSpectrogram.shape[1]
        let nFreqs = nFFT / 2 + 1

        guard nMels > 0 && nFrames > 0 else {
            return Signal(data: [], shape: [nFreqs, 0], sampleRate: sr)
        }

        // 1. Get the mel filterbank: shape [nMels, nFreqs]
        let melFB = FilterBank.mel(sr: sr, nFFT: nFFT, nMels: nMels, fMin: fMin, fMax: fMax)

        // 2. Compute pseudo-inverse: pinv(A) = A^T @ (A @ A^T + lambda*I)^{-1}
        //    A = melFB [nMels, nFreqs]
        let invPtr = computePseudoInverse(melFB: melFB, nMels: nMels, nFreqs: nFreqs)
        defer { invPtr.deallocate() }

        // 3. Matrix multiply: invFB [nFreqs, nMels] @ melSpec [nMels, nFrames] = [nFreqs, nFrames]
        let outCount = nFreqs * nFrames
        let outPtr = UnsafeMutablePointer<Float>.allocate(capacity: outCount)
        outPtr.initialize(repeating: 0, count: outCount)

        melSpectrogram.withUnsafeBufferPointer { melBuf in
            vDSP_mmul(
                invPtr, 1,                    // A: [nFreqs, nMels]
                melBuf.baseAddress!, 1,       // B: [nMels, nFrames]
                outPtr, 1,                    // C: [nFreqs, nFrames]
                vDSP_Length(nFreqs),
                vDSP_Length(nFrames),
                vDSP_Length(nMels)
            )
        }

        // 4. Clamp negative values to zero
        for i in 0..<outCount {
            if outPtr[i] < 0 { outPtr[i] = 0 }
        }

        // 5. Apply power root: if the mel spectrogram was computed with power > 1,
        //    the inverse maps back to the same domain, so we take the 1/power root
        //    to get amplitude (magnitude) suitable for Griffin-Lim.
        if power > 1.0 {
            let invPower = 1.0 / power
            for i in 0..<outCount {
                outPtr[i] = powf(outPtr[i], invPower)
            }
        }

        let outBuffer = UnsafeMutableBufferPointer(start: outPtr, count: outCount)
        return Signal(taking: outBuffer, shape: [nFreqs, nFrames], sampleRate: sr)
    }

    // MARK: - Mel to Audio

    /// Reconstruct audio from a mel spectrogram.
    ///
    /// Converts the mel spectrogram to an approximate STFT magnitude via
    /// pseudo-inverse of the mel filterbank, then uses Griffin-Lim to
    /// estimate phase and reconstruct the time-domain signal.
    ///
    /// - Parameters:
    ///   - melSpectrogram: Mel spectrogram with shape `[nMels, nFrames]`.
    ///   - sr: Sample rate in Hz.
    ///   - nFFT: FFT window size. Default 2048.
    ///   - hopLength: Hop length. If `nil`, defaults to `nFFT / 4`.
    ///   - winLength: Window length. If `nil`, defaults to `nFFT`.
    ///   - center: If `true`, assumes centered STFT. Default `true`.
    ///   - power: The exponent used when the mel spectrogram was computed.
    ///     Default 2.0 (power spectrogram), matching `MelSpectrogram.compute()`.
    ///   - nIter: Number of Griffin-Lim iterations. Default 32.
    ///   - fMin: Lowest mel frequency in Hz. Default 0.0.
    ///   - fMax: Highest mel frequency in Hz. If `nil`, uses `sr / 2`.
    ///   - length: If specified, truncates or pads output to this length.
    /// - Returns: Reconstructed 1-D audio Signal.
    public static func melToAudio(
        melSpectrogram: Signal,
        sr: Int,
        nFFT: Int = 2048,
        hopLength: Int? = nil,
        winLength: Int? = nil,
        center: Bool = true,
        power: Float = 2.0,
        nIter: Int = 32,
        fMin: Float = 0.0,
        fMax: Float? = nil,
        length: Int? = nil
    ) -> Signal {
        // 1. Mel spectrogram -> approximate STFT magnitude
        //    Pass the power parameter so melToSTFT takes the appropriate root.
        let stftMag = melToSTFT(
            melSpectrogram: melSpectrogram,
            sr: sr,
            nFFT: nFFT,
            power: power,
            fMin: fMin,
            fMax: fMax
        )

        // 2. STFT magnitude -> audio via Griffin-Lim
        return PhaseVocoder.griffinLim(
            magnitude: stftMag,
            nIter: nIter,
            hopLength: hopLength,
            winLength: winLength,
            center: center,
            length: length
        )
    }

    // MARK: - MFCC to Mel

    /// Approximate mel spectrogram from MFCCs (inverse DCT).
    ///
    /// Applies DCT-III (the inverse of DCT-II) to reconstruct the log-mel
    /// spectrogram from MFCC coefficients, then exponentiates to get the
    /// power mel spectrogram.
    ///
    /// Note: since MFCCs typically keep only the first few coefficients
    /// (e.g., 20 out of 128 mel bands), the reconstruction is approximate
    /// and represents a smoothed version of the original mel spectrogram.
    ///
    /// - Parameters:
    ///   - mfcc: MFCC coefficients with shape `[nMFCC, nFrames]`.
    ///   - nMels: Number of mel bands in the target mel spectrogram. Default 128.
    /// - Returns: Approximate mel spectrogram with shape `[nMels, nFrames]`.
    public static func mfccToMel(
        mfcc: Signal,
        nMels: Int = 128
    ) -> Signal {
        precondition(mfcc.shape.count == 2,
                     "mfccToMel requires 2D input [nMFCC, nFrames]")

        let nMFCC = mfcc.shape[0]
        let nFrames = mfcc.shape[1]

        guard nMFCC > 0 && nFrames > 0 else {
            return Signal(data: [], shape: [nMels, 0], sampleRate: mfcc.sampleRate)
        }

        // Build DCT-III (inverse of DCT-II, ortho-normalized) basis matrix.
        // The forward DCT-II used for MFCC: Y[k] = sum_n X[n] * cos(pi*k*(2n+1)/(2N)) * norm[k]
        // where norm[0] = sqrt(1/N), norm[k>0] = sqrt(2/N)
        //
        // Inverse (DCT-III, ortho): X[n] = sum_k Y[k] * cos(pi*k*(2n+1)/(2N)) * norm[k]
        // This is the transpose of the forward DCT-II basis.
        //
        // basis_inv[n, k] = basis_fwd[k, n]  (basis_fwd is [nMFCC, N])
        // For reconstruction: log_mel [nMels, nFrames] = basis_inv [nMels, nMFCC] @ mfcc [nMFCC, nFrames]

        let N = nMels  // target mel dimension
        let basisCount = N * nMFCC
        let basis = UnsafeMutablePointer<Float>.allocate(capacity: basisCount)
        defer { basis.deallocate() }

        let norm0 = sqrtf(1.0 / Float(N))
        let normK = sqrtf(2.0 / Float(N))

        // basis_inv[n, k] = cos(pi*k*(2n+1)/(2N)) * norm[k]
        for n in 0..<N {
            for k in 0..<nMFCC {
                let norm = (k == 0) ? norm0 : normK
                let angle = Float.pi * Float(k) * (2.0 * Float(n) + 1.0) / (2.0 * Float(N))
                basis[n * nMFCC + k] = cosf(angle) * norm
            }
        }

        // Matrix multiply: basis_inv [nMels, nMFCC] @ mfcc [nMFCC, nFrames] = log_mel [nMels, nFrames]
        let outCount = N * nFrames
        let outPtr = UnsafeMutablePointer<Float>.allocate(capacity: outCount)
        outPtr.initialize(repeating: 0, count: outCount)

        mfcc.withUnsafeBufferPointer { mfccBuf in
            vDSP_mmul(
                basis, 1,                        // A: [nMels, nMFCC]
                mfccBuf.baseAddress!, 1,         // B: [nMFCC, nFrames]
                outPtr, 1,                        // C: [nMels, nFrames]
                vDSP_Length(N),
                vDSP_Length(nFrames),
                vDSP_Length(nMFCC)
            )
        }

        // Exponentiate: mel = 10^(log_mel / 10)
        // The MFCC pipeline uses power_to_dB (10*log10), so the inverse is 10^(x/10).
        var countInt = Int32(outCount)
        var divisor: Float = 10.0
        let n = vDSP_Length(outCount)
        vDSP_vsdiv(outPtr, 1, &divisor, outPtr, 1, n)
        var ln10 = Float(log(10.0))
        vDSP_vsmul(outPtr, 1, &ln10, outPtr, 1, n)
        vvexpf(outPtr, outPtr, &countInt)

        let outBuffer = UnsafeMutableBufferPointer(start: outPtr, count: outCount)
        return Signal(taking: outBuffer, shape: [N, nFrames], sampleRate: mfcc.sampleRate)
    }

    // MARK: - MFCC to Audio

    /// Reconstruct audio from MFCCs.
    ///
    /// Converts MFCC coefficients to a mel spectrogram via inverse DCT,
    /// then reconstructs audio via mel-to-audio inversion.
    ///
    /// - Parameters:
    ///   - mfcc: MFCC coefficients with shape `[nMFCC, nFrames]`.
    ///   - sr: Sample rate in Hz.
    ///   - nMels: Number of mel bands. Default 128.
    ///   - nFFT: FFT window size. Default 2048.
    ///   - hopLength: Hop length. If `nil`, defaults to `nFFT / 4`.
    ///   - winLength: Window length. If `nil`, defaults to `nFFT`.
    ///   - center: If `true`, assumes centered STFT. Default `true`.
    ///   - nIter: Number of Griffin-Lim iterations. Default 32.
    ///   - fMin: Lowest mel frequency in Hz. Default 0.0.
    ///   - fMax: Highest mel frequency in Hz. If `nil`, uses `sr / 2`.
    ///   - length: If specified, truncates or pads output to this length.
    /// - Returns: Reconstructed 1-D audio Signal.
    public static func mfccToAudio(
        mfcc: Signal,
        sr: Int,
        nMels: Int = 128,
        nFFT: Int = 2048,
        hopLength: Int? = nil,
        winLength: Int? = nil,
        center: Bool = true,
        nIter: Int = 32,
        fMin: Float = 0.0,
        fMax: Float? = nil,
        length: Int? = nil
    ) -> Signal {
        // 1. MFCC -> mel spectrogram (power domain)
        let melSpec = mfccToMel(mfcc: mfcc, nMels: nMels)

        // 2. Mel spectrogram -> audio (power=2.0 since mfccToMel returns power spectrogram)
        return melToAudio(
            melSpectrogram: melSpec,
            sr: sr,
            nFFT: nFFT,
            hopLength: hopLength,
            winLength: winLength,
            center: center,
            power: 2.0,
            nIter: nIter,
            fMin: fMin,
            fMax: fMax,
            length: length
        )
    }

    // MARK: - Internal: Pseudo-Inverse

    /// Compute the pseudo-inverse of the mel filterbank using Gauss-Jordan elimination.
    ///
    /// pinv(A) = A^T @ (A @ A^T + lambda*I)^{-1}
    /// Returns a pointer to nFreqs * nMels floats. Caller must deallocate.
    private static func computePseudoInverse(
        melFB: Signal,
        nMels: Int,
        nFreqs: Int
    ) -> UnsafeMutablePointer<Float> {
        let invCount = nFreqs * nMels

        // Compute A^T [nFreqs, nMels]
        let atPtr = UnsafeMutablePointer<Float>.allocate(capacity: invCount)
        melFB.withUnsafeBufferPointer { fbBuf in
            for f in 0..<nFreqs {
                for m in 0..<nMels {
                    atPtr[f * nMels + m] = fbBuf[m * nFreqs + f]
                }
            }
        }

        // Compute G = A @ A^T  [nMels, nMels]
        let gPtr = UnsafeMutablePointer<Float>.allocate(capacity: nMels * nMels)
        gPtr.initialize(repeating: 0, count: nMels * nMels)

        melFB.withUnsafeBufferPointer { fbBuf in
            vDSP_mmul(
                fbBuf.baseAddress!, 1,     // A: [nMels, nFreqs]
                atPtr, 1,                  // A^T: [nFreqs, nMels]
                gPtr, 1,                   // G: [nMels, nMels]
                vDSP_Length(nMels),
                vDSP_Length(nMels),
                vDSP_Length(nFreqs)
            )
        }

        // Add regularization: G += lambda * I
        let regLambda: Float = 1e-6
        for i in 0..<nMels {
            gPtr[i * nMels + i] += regLambda
        }

        // Invert G via Gauss-Jordan elimination with partial pivoting
        let augCols = 2 * nMels
        let augPtr = UnsafeMutablePointer<Float>.allocate(capacity: nMels * augCols)
        augPtr.initialize(repeating: 0, count: nMels * augCols)
        for i in 0..<nMels {
            for j in 0..<nMels {
                augPtr[i * augCols + j] = gPtr[i * nMels + j]
            }
            augPtr[i * augCols + nMels + i] = 1.0
        }

        var invertSuccess = true
        for col in 0..<nMels {
            // Find pivot
            var maxVal: Float = 0
            var pivotRow = col
            for row in col..<nMels {
                let val = abs(augPtr[row * augCols + col])
                if val > maxVal {
                    maxVal = val
                    pivotRow = row
                }
            }
            if maxVal < 1e-12 {
                invertSuccess = false
                break
            }
            // Swap rows
            if pivotRow != col {
                for j in 0..<augCols {
                    let tmp = augPtr[col * augCols + j]
                    augPtr[col * augCols + j] = augPtr[pivotRow * augCols + j]
                    augPtr[pivotRow * augCols + j] = tmp
                }
            }
            // Scale pivot row
            let pivotVal = augPtr[col * augCols + col]
            for j in 0..<augCols {
                augPtr[col * augCols + j] /= pivotVal
            }
            // Eliminate column in all other rows
            for row in 0..<nMels where row != col {
                let factor = augPtr[row * augCols + col]
                if factor != 0 {
                    for j in 0..<augCols {
                        augPtr[row * augCols + j] -= factor * augPtr[col * augCols + j]
                    }
                }
            }
        }

        let invPtr: UnsafeMutablePointer<Float>
        if invertSuccess {
            // Extract G^{-1} from right half
            let gInvPtr = UnsafeMutablePointer<Float>.allocate(capacity: nMels * nMels)
            for i in 0..<nMels {
                for j in 0..<nMels {
                    gInvPtr[i * nMels + j] = augPtr[i * augCols + nMels + j]
                }
            }

            // pinv(A) = A^T [nFreqs, nMels] @ G^{-1} [nMels, nMels] = [nFreqs, nMels]
            invPtr = UnsafeMutablePointer<Float>.allocate(capacity: invCount)
            invPtr.initialize(repeating: 0, count: invCount)
            vDSP_mmul(
                atPtr, 1,
                gInvPtr, 1,
                invPtr, 1,
                vDSP_Length(nFreqs),
                vDSP_Length(nMels),
                vDSP_Length(nMels)
            )
            gInvPtr.deallocate()
        } else {
            // Fallback to normalized transpose
            invPtr = UnsafeMutablePointer<Float>.allocate(capacity: invCount)
            melFB.withUnsafeBufferPointer { fbBuf in
                for f in 0..<nFreqs {
                    var colSum: Float = 0
                    for m in 0..<nMels {
                        colSum += fbBuf[m * nFreqs + f]
                    }
                    let norm = colSum > 1e-10 ? colSum : 1.0
                    for m in 0..<nMels {
                        invPtr[f * nMels + m] = fbBuf[m * nFreqs + f] / norm
                    }
                }
            }
        }

        atPtr.deallocate()
        gPtr.deallocate()
        augPtr.deallocate()

        return invPtr
    }
}
