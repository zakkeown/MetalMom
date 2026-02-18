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
        let state = Profiler.shared.begin("AudioIO.load")
        defer { Profiler.shared.end("AudioIO.load", state) }
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

        let outputSR = sr ?? fileSampleRate
        let outputChannels: AVAudioChannelCount = (mono && channelCount > 1) ? 1 : AVAudioChannelCount(channelCount)
        let needsConversion = outputSR != fileSampleRate || outputChannels != channelCount

        if needsConversion {
            // Use AVAudioConverter for hardware-accelerated format conversion
            // (handles both sample rate conversion and channel downmix in one pass)
            guard let outputFormat = AVAudioFormat(
                commonFormat: .pcmFormatFloat32,
                sampleRate: Double(outputSR),
                channels: outputChannels,
                interleaved: false
            ) else {
                throw AudioIOError.unsupportedFormat
            }

            // Read source audio at native format
            guard let inputBuffer = AVAudioPCMBuffer(pcmFormat: fileFormat, frameCapacity: framesToRead) else {
                throw AudioIOError.allocationFailed
            }
            try file.read(into: inputBuffer, frameCount: framesToRead)

            guard inputBuffer.frameLength > 0 else {
                return Signal(data: [], sampleRate: outputSR)
            }

            // Create converter
            guard let converter = AVAudioConverter(from: fileFormat, to: outputFormat) else {
                throw AudioIOError.unsupportedFormat
            }

            // Estimate output frame count
            let ratio = Double(outputSR) / fileFormat.sampleRate
            let estimatedOutputFrames = AVAudioFrameCount(Double(inputBuffer.frameLength) * ratio) + 1
            guard let outputBuffer = AVAudioPCMBuffer(pcmFormat: outputFormat, frameCapacity: estimatedOutputFrames) else {
                throw AudioIOError.allocationFailed
            }

            // Convert in a single hardware-accelerated pass
            var error: NSError?
            var isDone = false
            let status = converter.convert(to: outputBuffer, error: &error) { _, outStatus in
                if isDone {
                    outStatus.pointee = .noDataNow
                    return nil
                }
                isDone = true
                outStatus.pointee = .haveData
                return inputBuffer
            }

            if status == .error, let error = error {
                throw error
            }

            let frameCount = Int(outputBuffer.frameLength)
            guard frameCount > 0, let floatData = outputBuffer.floatChannelData else {
                return Signal(data: [], sampleRate: outputSR)
            }

            // For mono output or single channel, just copy channel 0
            let samples = Array(UnsafeBufferPointer(start: floatData[0], count: frameCount))
            return Signal(data: samples, sampleRate: outputSR)
        } else {
            // No conversion needed — read directly
            guard let buffer = AVAudioPCMBuffer(pcmFormat: fileFormat, frameCapacity: framesToRead) else {
                throw AudioIOError.allocationFailed
            }
            try file.read(into: buffer, frameCount: framesToRead)

            let frameCount = Int(buffer.frameLength)
            guard frameCount > 0 else {
                return Signal(data: [], sampleRate: fileSampleRate)
            }

            guard let floatChannelData = buffer.floatChannelData else {
                throw AudioIOError.unsupportedFormat
            }

            let samples = Array(UnsafeBufferPointer(start: floatChannelData[0], count: frameCount))
            return Signal(data: samples, sampleRate: fileSampleRate)
        }
    }

    /// Get the duration of an audio file in seconds.
    public static func getDuration(path: String) throws -> Double {
        let url = URL(fileURLWithPath: path)
        let file = try AVAudioFile(forReading: url)
        return Double(file.length) / file.processingFormat.sampleRate
    }

    /// Get the sample rate of an audio file in Hz.
    public static func getSampleRate(path: String) throws -> Int {
        let url = URL(fileURLWithPath: path)
        let file = try AVAudioFile(forReading: url)
        return Int(file.processingFormat.sampleRate)
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
