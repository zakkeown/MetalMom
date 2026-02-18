import XCTest
@testable import MetalMomCore

/// Performance benchmarks for core MetalMom operations.
///
/// Each test uses XCTest `measure {}` blocks which run the enclosed code 10 times
/// and report average wall-clock time. Results create Xcode performance baselines
/// that can be tracked across commits for regression detection.
///
/// All tests share a 5-second 440 Hz sine wave at 22050 Hz to provide meaningful,
/// comparable timings.
final class PerformanceBenchmarkTests: XCTestCase {

    // MARK: - Shared Test Signal

    /// 5-second 440 Hz sine wave at 22050 Hz — created once for all benchmarks.
    private static let signal: Signal = {
        SignalGen.tone(frequency: 440.0, sr: 22050, duration: 5.0)
    }()

    // MARK: - Spectral Benchmarks

    func testSTFTPerformance() {
        let sig = Self.signal
        measure {
            _ = STFT.compute(signal: sig)
        }
    }

    func testMelSpectrogramPerformance() {
        let sig = Self.signal
        measure {
            _ = MelSpectrogram.compute(signal: sig)
        }
    }

    func testMFCCPerformance() {
        let sig = Self.signal
        measure {
            _ = MFCC.compute(signal: sig)
        }
    }

    func testCQTPerformance() {
        let sig = Self.signal
        measure {
            _ = CQT.compute(signal: sig)
        }
    }

    // MARK: - Rhythm Benchmarks

    func testOnsetDetectionPerformance() {
        let sig = Self.signal
        measure {
            _ = OnsetDetection.detectOnsets(signal: sig)
        }
    }

    func testBeatTrackingPerformance() {
        let sig = Self.signal
        measure {
            _ = BeatTracker.beatTrack(
                signal: sig,
                sr: 22050,
                startBPM: 120.0,
                trimFirst: false,
                trimLast: false
            )
        }
    }

    // MARK: - Decomposition Benchmarks

    func testNMFPerformance() {
        // Pre-compute mel spectrogram outside the measure block so we only
        // benchmark the NMF decomposition itself.
        let mel = MelSpectrogram.compute(signal: Self.signal)
        measure {
            _ = NMF.decompose(mel, nComponents: 4, nIter: 50)
        }
    }
}
