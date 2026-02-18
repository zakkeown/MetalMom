import XCTest
import AVFoundation
@testable import MetalMomCore

final class AudioIOTests: XCTestCase {

    /// Helper: write a mono WAV file with the given samples and sample rate.
    /// Uses autoreleasepool to ensure the write handle is flushed/closed before returning.
    private func writeWAV(url: URL, samples: [Float], sampleRate: Int) throws {
        try autoreleasepool {
            let format = AVAudioFormat(standardFormatWithSampleRate: Double(sampleRate), channels: 1)!
            let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count))!
            buffer.frameLength = AVAudioFrameCount(samples.count)
            memcpy(buffer.floatChannelData![0], samples, samples.count * MemoryLayout<Float>.size)

            let file = try AVAudioFile(forWriting: url, settings: format.settings)
            try file.write(from: buffer)
            // file is released here at end of autoreleasepool
        }
    }

    func testLoadNonExistentFile() {
        XCTAssertThrowsError(try AudioIO.load(path: "/nonexistent/file.wav"))
    }

    func testLoadGeneratedWAV() throws {
        let tempDir = FileManager.default.temporaryDirectory
        let tempFile = tempDir.appendingPathComponent("test_audio_\(UUID().uuidString).wav")
        defer { try? FileManager.default.removeItem(at: tempFile) }

        // Generate 1 second of 440 Hz sine wave at 44100 Hz
        let sampleRate = 44100
        let frequency = 440.0
        let numSamples = sampleRate  // 1 second

        var samples = [Float](repeating: 0, count: numSamples)
        for i in 0..<numSamples {
            samples[i] = sin(Float(i) * Float(frequency) * 2.0 * .pi / Float(sampleRate))
        }

        try writeWAV(url: tempFile, samples: samples, sampleRate: sampleRate)

        // Load it back
        let signal = try AudioIO.load(path: tempFile.path, sr: nil, mono: true)

        XCTAssertEqual(signal.sampleRate, sampleRate)
        // AVFoundation may round to packet boundaries; allow small tolerance
        XCTAssertEqual(Double(signal.count), Double(numSamples), accuracy: 512)
        XCTAssertGreaterThan(signal.count, 0)
        // First sample should be near 0 (sin(0) = 0)
        XCTAssertEqual(signal[0], 0, accuracy: 0.01)
    }

    func testLoadWithResample() throws {
        let tempDir = FileManager.default.temporaryDirectory
        let tempFile = tempDir.appendingPathComponent("test_audio_resample_\(UUID().uuidString).wav")
        defer { try? FileManager.default.removeItem(at: tempFile) }

        let sampleRate = 44100
        let numSamples = 44100
        var samples = [Float](repeating: 0, count: numSamples)
        for i in 0..<numSamples {
            samples[i] = sin(Float(i) * 440.0 * 2.0 * .pi / Float(sampleRate))
        }

        try writeWAV(url: tempFile, samples: samples, sampleRate: sampleRate)

        // Load with resampling to 22050
        let signal = try AudioIO.load(path: tempFile.path, sr: 22050, mono: true)

        XCTAssertEqual(signal.sampleRate, 22050)
        // Should have roughly half the samples (with tolerance for rounding)
        XCTAssertEqual(Double(signal.count), 22050, accuracy: 512)
    }

    func testLoadWithOffset() throws {
        let tempDir = FileManager.default.temporaryDirectory
        let tempFile = tempDir.appendingPathComponent("test_audio_offset_\(UUID().uuidString).wav")
        defer { try? FileManager.default.removeItem(at: tempFile) }

        let sampleRate = 22050
        let numSamples = 22050 * 2  // 2 seconds
        var samples = [Float](repeating: 0, count: numSamples)
        for i in 0..<numSamples {
            samples[i] = Float(i) / Float(numSamples)  // Linear ramp
        }

        try writeWAV(url: tempFile, samples: samples, sampleRate: sampleRate)

        // Load with 1 second offset
        let signal = try AudioIO.load(path: tempFile.path, sr: nil, offset: 1.0)

        XCTAssertEqual(signal.sampleRate, sampleRate)
        // Should have roughly 1 second of samples (with tolerance)
        XCTAssertEqual(Double(signal.count), 22050, accuracy: 512)
        // First sample should be around 0.5 (midpoint of ramp)
        XCTAssertEqual(signal[0], 0.5, accuracy: 0.01)
    }

    func testLoadWithDuration() throws {
        let tempDir = FileManager.default.temporaryDirectory
        let tempFile = tempDir.appendingPathComponent("test_audio_dur_\(UUID().uuidString).wav")
        defer { try? FileManager.default.removeItem(at: tempFile) }

        let sampleRate = 22050
        let numSamples = 22050 * 3  // 3 seconds
        let samples = [Float](repeating: 0.5, count: numSamples)

        try writeWAV(url: tempFile, samples: samples, sampleRate: sampleRate)

        // Load only 1 second
        let signal = try AudioIO.load(path: tempFile.path, sr: nil, duration: 1.0)

        XCTAssertEqual(signal.sampleRate, sampleRate)
        // Should have roughly 1 second of samples (with tolerance for packet rounding)
        XCTAssertEqual(Double(signal.count), 22050, accuracy: 1024)
        // All samples should be 0.5
        XCTAssertEqual(signal[0], 0.5, accuracy: 0.01)
    }
}
