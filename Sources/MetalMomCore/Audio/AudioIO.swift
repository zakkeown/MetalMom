import AVFoundation
import Accelerate
import Foundation

public enum AudioIO {
    /// Load an audio file, decode to float32, optionally convert to mono and resample.
    ///
    /// - Parameters:
    ///   - path: File path to the audio file.
    ///   - sr: Target sample rate. If nil, uses the file's native sample rate.
    ///   - mono: If true (default), mix down to mono.
    ///   - offset: Start reading at this time (seconds). Default: 0.0.
    ///   - duration: Read only this many seconds. If nil, read the entire file.
    /// - Returns: A Signal containing the audio data.
    /// - Throws: If the file cannot be read or decoded.
    public static func load(
        path: String,
        sr: Int? = nil,
        mono: Bool = true,
        offset: Double = 0.0,
        duration: Double? = nil
    ) throws -> Signal {
        let url = URL(fileURLWithPath: path)
        let file = try AVAudioFile(forReading: url)

        let fileFormat = file.processingFormat
        let fileSampleRate = Int(fileFormat.sampleRate)
        let channelCount = Int(fileFormat.channelCount)

        // Calculate frame range
        let startFrame = AVAudioFramePosition(offset * fileFormat.sampleRate)
        let totalFrames = file.length
        let availableFrames = totalFrames - startFrame
        guard availableFrames > 0 else {
            return Signal(data: [], sampleRate: sr ?? fileSampleRate)
        }

        let framesToRead: AVAudioFrameCount
        if let dur = duration {
            framesToRead = AVAudioFrameCount(min(Double(availableFrames), dur * fileFormat.sampleRate))
        } else {
            framesToRead = AVAudioFrameCount(availableFrames)
        }

        // Seek to start position
        file.framePosition = startFrame

        // Read into buffer
        guard let buffer = AVAudioPCMBuffer(pcmFormat: fileFormat, frameCapacity: framesToRead) else {
            throw AudioIOError.allocationFailed
        }
        try file.read(into: buffer, frameCount: framesToRead)

        let frameCount = Int(buffer.frameLength)
        guard frameCount > 0 else {
            return Signal(data: [], sampleRate: sr ?? fileSampleRate)
        }

        // Extract float data
        guard let floatChannelData = buffer.floatChannelData else {
            throw AudioIOError.unsupportedFormat
        }

        var samples: [Float]
        if mono && channelCount > 1 {
            // Mix to mono: average all channels
            samples = [Float](repeating: 0, count: frameCount)
            let scale = 1.0 / Float(channelCount)
            for ch in 0..<channelCount {
                let channelPtr = floatChannelData[ch]
                for i in 0..<frameCount {
                    samples[i] += channelPtr[i] * scale
                }
            }
        } else if channelCount == 1 {
            // Already mono
            samples = Array(UnsafeBufferPointer(start: floatChannelData[0], count: frameCount))
        } else {
            // Multi-channel: return first channel when mono=false
            samples = Array(UnsafeBufferPointer(start: floatChannelData[0], count: frameCount))
        }

        // Resample if needed
        let outputSR = sr ?? fileSampleRate
        if outputSR != fileSampleRate {
            samples = resample(samples, fromRate: fileSampleRate, toRate: outputSR)
        }

        return Signal(data: samples, sampleRate: outputSR)
    }

    /// Simple resampling using linear interpolation.
    private static func resample(_ input: [Float], fromRate: Int, toRate: Int) -> [Float] {
        guard fromRate != toRate, !input.isEmpty else { return input }

        let ratio = Double(toRate) / Double(fromRate)
        let outputLength = Int(Double(input.count) * ratio)
        guard outputLength > 0 else { return [] }

        var output = [Float](repeating: 0, count: outputLength)

        // Linear interpolation resampling
        for i in 0..<outputLength {
            let srcPos = Double(i) / ratio
            let srcIdx = Int(srcPos)
            let frac = Float(srcPos - Double(srcIdx))

            if srcIdx + 1 < input.count {
                output[i] = input[srcIdx] * (1 - frac) + input[srcIdx + 1] * frac
            } else if srcIdx < input.count {
                output[i] = input[srcIdx]
            }
        }

        return output
    }
}

public enum AudioIOError: Error, LocalizedError {
    case allocationFailed
    case unsupportedFormat
    case fileNotFound(String)

    public var errorDescription: String? {
        switch self {
        case .allocationFailed: return "Failed to allocate audio buffer"
        case .unsupportedFormat: return "Unsupported audio format"
        case .fileNotFound(let path): return "Audio file not found: \(path)"
        }
    }
}
