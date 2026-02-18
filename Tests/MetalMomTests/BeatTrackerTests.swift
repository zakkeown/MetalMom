import XCTest
@testable import MetalMomCore

final class BeatTrackerTests: XCTestCase {

    // MARK: - Tempo Estimation Tests

    func testTempoEstimation120BPM() {
        // Generate clicks at 120 BPM (0.5s apart)
        let sr = 22050
        let duration = 5.0
        let n = Int(Double(sr) * duration)
        var samples = [Float](repeating: 0, count: n)

        let bpm: Float = 120.0
        let interval = 60.0 / Double(bpm)  // 0.5s

        var t = 0.0
        while t < duration {
            let idx = Int(t * Double(sr))
            for j in 0..<256 where idx + j < n {
                samples[idx + j] = sin(Float(j) * 1000.0 * 2.0 * .pi / Float(sr)) * 0.9
            }
            t += interval
        }

        let signal = Signal(data: samples, sampleRate: sr)

        let (tempo, beats) = BeatTracker.beatTrack(
            signal: signal,
            sr: sr,
            startBPM: 120.0,
            trimFirst: false,
            trimLast: false
        )

        // Tempo should be in a reasonable range around 120 BPM
        // Allow wide tolerance since ACF-based estimation is approximate
        XCTAssertGreaterThan(tempo, 60, "Tempo should be > 60 BPM")
        XCTAssertLessThan(tempo, 240, "Tempo should be < 240 BPM")

        // Should detect some beats
        XCTAssertGreaterThan(beats.count, 0, "Should detect at least one beat")
    }

    func testTempoEstimation90BPM() {
        let sr = 22050
        let duration = 6.0
        let n = Int(Double(sr) * duration)
        var samples = [Float](repeating: 0, count: n)

        let bpm: Float = 90.0
        let interval = 60.0 / Double(bpm)

        var t = 0.0
        while t < duration {
            let idx = Int(t * Double(sr))
            for j in 0..<256 where idx + j < n {
                samples[idx + j] = sin(Float(j) * 1000.0 * 2.0 * .pi / Float(sr)) * 0.9
            }
            t += interval
        }

        let signal = Signal(data: samples, sampleRate: sr)
        let (tempo, _) = BeatTracker.beatTrack(
            signal: signal,
            sr: sr,
            startBPM: 120.0,
            trimFirst: false,
            trimLast: false
        )

        // Should estimate something in the ballpark (allow octave errors)
        XCTAssertGreaterThan(tempo, 30, "Tempo should be > 30 BPM")
        XCTAssertLessThan(tempo, 300, "Tempo should be < 300 BPM")
    }

    // MARK: - Beat Tracking Tests

    func testBeatTrackReturnsNonEmptyBeats() {
        let sr = 22050
        let duration = 3.0
        let n = Int(Double(sr) * duration)
        var samples = [Float](repeating: 0, count: n)

        // Clicks at 120 BPM
        let interval = 0.5
        var t = 0.0
        while t < duration {
            let idx = Int(t * Double(sr))
            for j in 0..<256 where idx + j < n {
                samples[idx + j] = sin(Float(j) * 1000.0 * 2.0 * .pi / Float(sr)) * 0.9
            }
            t += interval
        }

        let signal = Signal(data: samples, sampleRate: sr)
        let (_, beats) = BeatTracker.beatTrack(signal: signal, sr: sr)

        XCTAssertGreaterThan(beats.count, 0, "Should detect beats from click track")
    }

    func testBeatFramesSorted() {
        let sr = 22050
        let duration = 4.0
        let n = Int(Double(sr) * duration)
        var samples = [Float](repeating: 0, count: n)

        // Clicks at 120 BPM
        var t = 0.0
        while t < duration {
            let idx = Int(t * Double(sr))
            for j in 0..<256 where idx + j < n {
                samples[idx + j] = sin(Float(j) * 1000.0 * 2.0 * .pi / Float(sr)) * 0.9
            }
            t += 0.5
        }

        let signal = Signal(data: samples, sampleRate: sr)
        let (_, beats) = BeatTracker.beatTrack(
            signal: signal, sr: sr, trimFirst: false, trimLast: false
        )

        // Beat frames should be in ascending order
        for i in 1..<beats.count {
            XCTAssertGreaterThan(beats[i], beats[i - 1],
                                  "Beat frames should be strictly increasing")
        }
    }

    func testBeatFramesNonNegative() {
        let sr = 22050
        let duration = 3.0
        let n = Int(Double(sr) * duration)
        var samples = [Float](repeating: 0, count: n)

        var t = 0.0
        while t < duration {
            let idx = Int(t * Double(sr))
            for j in 0..<256 where idx + j < n {
                samples[idx + j] = sin(Float(j) * 1000.0 * 2.0 * .pi / Float(sr)) * 0.9
            }
            t += 0.5
        }

        let signal = Signal(data: samples, sampleRate: sr)
        let (_, beats) = BeatTracker.beatTrack(
            signal: signal, sr: sr, trimFirst: false, trimLast: false
        )

        for i in 0..<beats.count {
            XCTAssertGreaterThanOrEqual(beats[i], 0, "Beat frames should be non-negative")
        }
    }

    func testBeatTrackSilentSignal() {
        let sr = 22050
        let signal = Signal(data: [Float](repeating: 0, count: sr * 2), sampleRate: sr)
        let (_, beats) = BeatTracker.beatTrack(signal: signal, sr: sr)

        // Silent signal may still produce beats from DP, but tempo should be low/zero
        // Just check no crash
        _ = beats
    }

    func testBeatTrackTrimming() {
        let sr = 22050
        let duration = 4.0
        let n = Int(Double(sr) * duration)
        var samples = [Float](repeating: 0, count: n)

        var t = 0.0
        while t < duration {
            let idx = Int(t * Double(sr))
            for j in 0..<256 where idx + j < n {
                samples[idx + j] = sin(Float(j) * 1000.0 * 2.0 * .pi / Float(sr)) * 0.9
            }
            t += 0.5
        }

        let signal = Signal(data: samples, sampleRate: sr)

        let (_, beatsNotTrimmed) = BeatTracker.beatTrack(
            signal: signal, sr: sr, trimFirst: false, trimLast: false
        )
        let (_, beatsTrimmed) = BeatTracker.beatTrack(
            signal: signal, sr: sr, trimFirst: true, trimLast: true
        )

        // Trimmed should have fewer beats (by up to 2)
        if beatsNotTrimmed.count > 2 {
            XCTAssertLessThanOrEqual(beatsTrimmed.count, beatsNotTrimmed.count,
                                      "Trimmed beats should be <= untrimmed")
        }
    }

    // MARK: - ACF Estimation Direct Tests

    func testEstimateTempoDirectACF() {
        // Create a synthetic onset envelope with peaks at 120 BPM
        let sr = 22050
        let hop = 512
        let framesPerSec = Float(sr) / Float(hop)
        let period = framesPerSec * 0.5  // 120 BPM => 0.5s period

        let nFrames = 200
        var envelope = [Float](repeating: 0.0, count: nFrames)

        // Place impulses at the expected period
        var f = 0
        while f < nFrames {
            envelope[f] = 1.0
            f += Int(round(period))
        }

        let tempo = BeatTracker.estimateTempo(
            envelope: envelope,
            sr: sr,
            hopLength: hop,
            startBPM: 120.0
        )

        // Should be close to 120 BPM (within 20% tolerance for ACF estimation)
        XCTAssertGreaterThan(tempo, 90, "Should estimate tempo > 90 BPM for 120 BPM input")
        XCTAssertLessThan(tempo, 160, "Should estimate tempo < 160 BPM for 120 BPM input")
    }

    func testEstimateTempoEmptyEnvelope() {
        let tempo = BeatTracker.estimateTempo(
            envelope: [],
            sr: 22050,
            hopLength: 512
        )
        XCTAssertEqual(tempo, 0, "Empty envelope should return 0 tempo")
    }

    func testEstimateTempoSingleFrame() {
        let tempo = BeatTracker.estimateTempo(
            envelope: [1.0],
            sr: 22050,
            hopLength: 512
        )
        XCTAssertEqual(tempo, 0, "Single frame should return 0 tempo")
    }
}
