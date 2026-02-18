import Accelerate
import Foundation

public enum SignalGen {
    /// Generate a pure sine tone.
    ///
    /// - Parameters:
    ///   - frequency: Frequency in Hz.
    ///   - sr: Sample rate. Default 22050.
    ///   - length: Number of samples. If nil, generates 1 second.
    ///   - duration: Duration in seconds. Overrides `length` if both given.
    ///   - phi: Phase offset in radians. Default 0.
    /// - Returns: Signal containing the sine tone.
    public static func tone(
        frequency: Float,
        sr: Int = 22050,
        length: Int? = nil,
        duration: Double? = nil,
        phi: Float = 0
    ) -> Signal {
        let n: Int
        if let dur = duration {
            n = Int(dur * Double(sr))
        } else if let len = length {
            n = len
        } else {
            n = sr  // 1 second default
        }
        guard n > 0 else { return Signal(data: [], sampleRate: sr) }

        var samples = [Float](repeating: 0, count: n)
        let angularFreq = 2.0 * Float.pi * frequency / Float(sr)
        for i in 0..<n {
            samples[i] = sin(angularFreq * Float(i) + phi)
        }
        return Signal(data: samples, sampleRate: sr)
    }

    /// Generate a frequency sweep (chirp).
    ///
    /// - Parameters:
    ///   - fmin: Start frequency in Hz.
    ///   - fmax: End frequency in Hz.
    ///   - sr: Sample rate. Default 22050.
    ///   - length: Number of samples. If nil, generates 1 second.
    ///   - duration: Duration in seconds. Overrides `length`.
    ///   - linear: If true, linear sweep. If false, logarithmic sweep. Default true.
    /// - Returns: Signal containing the chirp.
    public static func chirp(
        fmin: Float,
        fmax: Float,
        sr: Int = 22050,
        length: Int? = nil,
        duration: Double? = nil,
        linear: Bool = true
    ) -> Signal {
        let n: Int
        if let dur = duration {
            n = Int(dur * Double(sr))
        } else if let len = length {
            n = len
        } else {
            n = sr
        }
        guard n > 0 else { return Signal(data: [], sampleRate: sr) }

        let T = Float(n) / Float(sr)  // Total duration in seconds
        var samples = [Float](repeating: 0, count: n)

        if linear {
            // Linear chirp: f(t) = fmin + (fmax - fmin) * t / T
            // phase(t) = 2*pi * (fmin*t + (fmax-fmin)*t^2/(2T))
            let rate = (fmax - fmin) / T
            for i in 0..<n {
                let t = Float(i) / Float(sr)
                let phase = 2.0 * Float.pi * (fmin * t + 0.5 * rate * t * t)
                samples[i] = sin(phase)
            }
        } else {
            // Logarithmic chirp: f(t) = fmin * (fmax/fmin)^(t/T)
            // phase(t) = 2*pi * fmin * T / ln(fmax/fmin) * ((fmax/fmin)^(t/T) - 1)
            guard fmin > 0 && fmax > 0 else {
                return Signal(data: samples, sampleRate: sr)
            }
            let logRatio = log(fmax / fmin)
            let coeff = 2.0 * Float.pi * fmin * T / logRatio
            for i in 0..<n {
                let t = Float(i) / Float(sr)
                let phase = coeff * (powf(fmax / fmin, t / T) - 1.0)
                samples[i] = sin(phase)
            }
        }

        return Signal(data: samples, sampleRate: sr)
    }

    /// Generate a click track.
    ///
    /// - Parameters:
    ///   - times: Click times in seconds. If nil, uses `frames`.
    ///   - frames: Click positions in sample frames. Used if `times` is nil.
    ///   - sr: Sample rate. Default 22050.
    ///   - length: Total length in samples. If nil, auto-sized.
    ///   - clickFreq: Click frequency in Hz. Default 1000.
    ///   - clickDuration: Click duration in seconds. Default 0.1.
    /// - Returns: Signal containing the click track.
    public static func clicks(
        times: [Float]? = nil,
        frames: [Int]? = nil,
        sr: Int = 22050,
        length: Int? = nil,
        clickFreq: Float = 1000,
        clickDuration: Float = 0.1
    ) -> Signal {
        // Convert times to frame positions
        let clickFrames: [Int]
        if let t = times {
            clickFrames = t.map { Int($0 * Float(sr)) }
        } else if let f = frames {
            clickFrames = f
        } else {
            // Default: clicks every 0.5 seconds
            let dur: Float = length != nil ? Float(length!) / Float(sr) : 5.0
            let numClicks = Int(dur / 0.5)
            clickFrames = (0..<numClicks).map { Int(Float($0) * 0.5 * Float(sr)) }
        }

        // Determine total length
        let clickSamples = Int(clickDuration * Float(sr))
        let maxFrame = (clickFrames.max() ?? 0) + clickSamples
        let totalLength = length ?? maxFrame
        guard totalLength > 0 else { return Signal(data: [], sampleRate: sr) }

        var samples = [Float](repeating: 0, count: totalLength)

        // Generate click template (exponentially decaying sine)
        let angularFreq = 2.0 * Float.pi * clickFreq
        let decayRate = 5.0 / clickDuration  // Decay to ~e^-5 by end

        for frame in clickFrames {
            guard frame >= 0 && frame < totalLength else { continue }
            let endSample = min(frame + clickSamples, totalLength)
            for i in frame..<endSample {
                let t = Float(i - frame) / Float(sr)
                samples[i] += sin(angularFreq * t) * exp(-decayRate * t)
            }
        }

        return Signal(data: samples, sampleRate: sr)
    }
}
