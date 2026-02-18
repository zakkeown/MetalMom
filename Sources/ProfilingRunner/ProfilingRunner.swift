#if os(macOS)
import Foundation
import MetalMomCore

@main
struct ProfilingRunner {
    static func main() throws {
        let args = CommandLine.arguments
        guard args.count >= 2 else {
            print("Usage: swift run ProfilingRunner <audio-file>")
            print("  e.g.: swift run ProfilingRunner ~/Desktop/moth_30s.m4a")
            Foundation.exit(1)
        }

        let filePath = (args[1] as NSString).expandingTildeInPath
        guard FileManager.default.fileExists(atPath: filePath) else {
            print("ERROR: File not found: \(filePath)")
            Foundation.exit(1)
        }

        print("MetalMom Profiling Runner")
        print("=========================")
        print("File: \(filePath)")
        print("Metal: \(MetalBackend.shared != nil ? "available" : "unavailable")")
        if let chip = MetalBackend.shared?.chipProfile {
            print("GPU: \(chip.gpuFamily), ~\(chip.estimatedCoreCount) cores")
        }
        print()

        // --- Load Audio ---
        let t0 = CFAbsoluteTimeGetCurrent()
        let signal = try AudioIO.load(path: filePath, sr: 22050, mono: true)
        let tLoad = CFAbsoluteTimeGetCurrent() - t0
        print("Load:           \(String(format: "%8.1f", tLoad * 1000)) ms  (\(signal.count) samples, \(signal.sampleRate) Hz)")

        // --- STFT ---
        let t1 = CFAbsoluteTimeGetCurrent()
        let stftMag = STFT.compute(signal: signal, nFFT: 2048, hopLength: 512)
        let tSTFT = CFAbsoluteTimeGetCurrent() - t1
        print("STFT:           \(String(format: "%8.1f", tSTFT * 1000)) ms  shape=\(stftMag.shape)")

        // --- Mel Spectrogram ---
        let t2 = CFAbsoluteTimeGetCurrent()
        let mel = MelSpectrogram.compute(signal: signal, nFFT: 2048, hopLength: 512)
        let tMel = CFAbsoluteTimeGetCurrent() - t2
        print("Mel:            \(String(format: "%8.1f", tMel * 1000)) ms  shape=\(mel.shape)")

        // --- MFCC (via FusedMFCC which routes GPU/CPU) ---
        let t3 = CFAbsoluteTimeGetCurrent()
        let mfcc = FusedMFCC.compute(signal: signal, nFFT: 2048, hopLength: 512, nMFCC: 13)
        let tMFCC = CFAbsoluteTimeGetCurrent() - t3
        print("MFCC:           \(String(format: "%8.1f", tMFCC * 1000)) ms  shape=\(mfcc.shape)")

        // --- Chroma ---
        let t4 = CFAbsoluteTimeGetCurrent()
        let chroma = Chroma.stft(signal: signal, nFFT: 2048, hopLength: 512)
        let tChroma = CFAbsoluteTimeGetCurrent() - t4
        print("Chroma:         \(String(format: "%8.1f", tChroma * 1000)) ms  shape=\(chroma.shape)")

        // --- Onset Strength ---
        let t5 = CFAbsoluteTimeGetCurrent()
        let onset = OnsetDetection.onsetStrength(signal: signal, nFFT: 2048, hopLength: 512)
        let tOnset = CFAbsoluteTimeGetCurrent() - t5
        print("Onset:          \(String(format: "%8.1f", tOnset * 1000)) ms  shape=\(onset.shape)")

        // --- Beat Tracking ---
        let t6 = CFAbsoluteTimeGetCurrent()
        let (tempo, beats) = BeatTracker.beatTrack(signal: signal, hopLength: 512)
        let tBeat = CFAbsoluteTimeGetCurrent() - t6
        print("Beat:           \(String(format: "%8.1f", tBeat * 1000)) ms  tempo=\(String(format: "%.1f", tempo)) BPM, \(beats.count) beats")

        // --- Total ---
        let total = tLoad + tSTFT + tMel + tMFCC + tChroma + tOnset + tBeat
        print()
        print("Total:          \(String(format: "%8.1f", total * 1000)) ms")
        print()
        print("Attach Instruments (Time Profiler + os_signpost) to see detailed breakdown.")
    }
}
#else
// ProfilingRunner is macOS-only (requires command-line arguments and tilde expansion).
@main
struct ProfilingRunner {
    static func main() {
        print("ProfilingRunner is only available on macOS.")
    }
}
#endif
