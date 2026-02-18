import Accelerate
import Foundation

public enum OnsetDetection {
    /// Compute onset strength envelope.
    ///
    /// Measures spectral flux (positive first-order difference of the mel spectrogram in dB).
    /// Matches librosa's ``onset_strength()`` behavior including lag-based differencing
    /// and center-based frame shift compensation.
    ///
    /// - Parameters:
    ///   - signal: Input audio signal (1D).
    ///   - sr: Sample rate override. If nil, uses signal.sampleRate.
    ///   - nFFT: FFT window size. Default 2048.
    ///   - hopLength: Hop length. Default 512.
    ///   - nMels: Number of mel bands. Default 128.
    ///   - fmin: Minimum frequency for mel filterbank. Default 0.
    ///   - fmax: Maximum frequency. If nil, uses sr/2.
    ///   - center: Center-pad the signal and shift onset function. Default true.
    ///   - aggregate: If true, average across mel bands. Default true.
    ///   - lag: Time lag for computing differences. Default 1.
    /// - Returns: Signal with shape [1, nFrames] (if aggregate) or [nMels, nFrames].
    public static func onsetStrength(
        signal: Signal,
        sr: Int? = nil,
        nFFT: Int = 2048,
        hopLength: Int? = nil,
        nMels: Int = 128,
        fmin: Float = 0,
        fmax: Float? = nil,
        center: Bool = true,
        aggregate: Bool = true,
        lag: Int = 1
    ) -> Signal {
        let sampleRate = sr ?? signal.sampleRate
        let hop = hopLength ?? 512

        // 1. Compute mel spectrogram (power)
        let melSpec = MelSpectrogram.compute(
            signal: signal,
            sr: sampleRate,
            nFFT: nFFT,
            hopLength: hop,
            winLength: nFFT,
            center: center,
            power: 2.0,
            nMels: nMels,
            fMin: fmin,
            fMax: fmax
        )

        let nBands = melSpec.shape[0]  // nMels
        let nFrames = melSpec.shape[1]

        // 2. Convert to dB
        let melDb = Scaling.powerToDb(melSpec, ref: 1.0, amin: 1e-10, topDb: 80.0)

        // 3. Compute lag-based difference (spectral flux) along time axis
        // onset_env = S[..., lag:] - S[..., :-lag]
        // This produces nFrames - lag diff values
        guard nFrames > lag else {
            if aggregate {
                return Signal(data: [0], shape: [1, 1], sampleRate: sampleRate)
            } else {
                return Signal(data: [Float](repeating: 0, count: nBands),
                              shape: [nBands, 1], sampleRate: sampleRate)
            }
        }

        let diffCount = nFrames - lag

        // 4. Compute padding width to prepend
        // librosa: pad_width = lag; if center: pad_width += n_fft // (2 * hop_length)
        var padWidth = lag
        if center {
            padWidth += nFFT / (2 * hop)
        }

        // 5. Output length: diffCount + padWidth, then trim to nFrames if center
        let rawLen = diffCount + padWidth
        let outFrames: Int
        if center {
            outFrames = min(rawLen, nFrames)
        } else {
            outFrames = rawLen
        }

        if aggregate {
            // Average across bands for each frame
            var onset = [Float](repeating: 0, count: outFrames)

            melDb.withUnsafeBufferPointer { src in
                for i in 0..<diffCount {
                    let t = i + lag  // index into the S array for S[..., lag:]
                    let tRef = i     // index into the S array for S[..., :-lag]
                    var sum: Float = 0
                    for b in 0..<nBands {
                        let diff = src[b * nFrames + t] - src[b * nFrames + tRef]
                        sum += max(0, diff)  // Half-wave rectification
                    }
                    let outIdx = i + padWidth
                    if outIdx < outFrames {
                        onset[outIdx] = sum / Float(nBands)
                    }
                }
            }

            return Signal(data: onset, shape: [1, outFrames], sampleRate: sampleRate)
        } else {
            // Return per-band onset strength
            var onset = [Float](repeating: 0, count: nBands * outFrames)

            melDb.withUnsafeBufferPointer { src in
                for b in 0..<nBands {
                    for i in 0..<diffCount {
                        let t = i + lag
                        let tRef = i
                        let diff = src[b * nFrames + t] - src[b * nFrames + tRef]
                        let outIdx = i + padWidth
                        if outIdx < outFrames {
                            onset[b * outFrames + outIdx] = max(0, diff)
                        }
                    }
                }
            }

            return Signal(data: onset, shape: [nBands, outFrames], sampleRate: sampleRate)
        }
    }
}
