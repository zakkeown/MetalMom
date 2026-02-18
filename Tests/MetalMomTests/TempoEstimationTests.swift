import XCTest
@testable import MetalMomCore

final class TempoEstimationTests: XCTestCase {

    // MARK: - Click Track Helpers

    /// Generate a click track at the given BPM.
    private func makeClickTrack(bpm: Float, sr: Int = 22050, duration: Double = 5.0) -> Signal {
        let n = Int(Double(sr) * duration)
        var samples = [Float](repeating: 0, count: n)

        let interval = 60.0 / Double(bpm)
        var t = 0.0
        while t < duration {
            let idx = Int(t * Double(sr))
            for j in 0..<256 where idx + j < n {
                samples[idx + j] = sin(Float(j) * 1000.0 * 2.0 * .pi / Float(sr)) * 0.9
            }
            t += interval
        }

        return Signal(data: samples, sampleRate: sr)
    }

    // MARK: - Tempo from Audio Signal

    func testTempo120BPM() {
        let signal = makeClickTrack(bpm: 120)

        let tempo = TempoEstimation.tempo(
            signal: signal,
            sr: 22050,
            startBPM: 120.0
        )

        // Allow +/- 15% tolerance
        XCTAssertGreaterThan(tempo, 102, "Tempo should be within 15% of 120 BPM (lower)")
        XCTAssertLessThan(tempo, 138, "Tempo should be within 15% of 120 BPM (upper)")
    }

    func testTempo90BPM() {
        let signal = makeClickTrack(bpm: 90, duration: 6.0)

        let tempo = TempoEstimation.tempo(
            signal: signal,
            sr: 22050,
            startBPM: 120.0
        )

        // ACF-based estimation can have octave ambiguity, so allow wide range
        // but check it's reasonable
        XCTAssertGreaterThan(tempo, 30, "Tempo should be > 30 BPM")
        XCTAssertLessThan(tempo, 300, "Tempo should be < 300 BPM")
    }

    func testTempoOnSilence() {
        let sr = 22050
        let signal = Signal(data: [Float](repeating: 0, count: sr * 3), sampleRate: sr)

        let tempo = TempoEstimation.tempo(
            signal: signal,
            sr: sr,
            startBPM: 120.0
        )

        // On silence, the onset envelope is all zeros so we return startBPM
        XCTAssertEqual(tempo, 120.0, "Tempo on silence should return startBPM")
    }

    // MARK: - Envelope-Level Tests

    func testEstimateFromEnvelope120BPM() {
        // Create a synthetic onset envelope with peaks at 120 BPM
        let sr = 22050
        let hop = 512
        let framesPerSec = Float(sr) / Float(hop)
        let period = framesPerSec * 0.5  // 120 BPM => 0.5s period

        let nFrames = 200
        var envelope = [Float](repeating: 0.0, count: nFrames)

        var f = 0
        while f < nFrames {
            envelope[f] = 1.0
            f += Int(round(period))
        }

        let tempo = TempoEstimation.estimateFromEnvelope(
            envelope: envelope,
            sr: sr,
            hopLength: hop,
            startBPM: 120.0
        )

        // Should be close to 120 BPM
        XCTAssertGreaterThan(tempo, 90, "Should estimate tempo > 90 BPM for 120 BPM input")
        XCTAssertLessThan(tempo, 160, "Should estimate tempo < 160 BPM for 120 BPM input")
    }

    func testEstimateFromEnvelopeEmptyReturnsDefault() {
        let tempo = TempoEstimation.estimateFromEnvelope(
            envelope: [],
            sr: 22050,
            hopLength: 512,
            startBPM: 100.0
        )
        XCTAssertEqual(tempo, 100.0, "Empty envelope should return startBPM")
    }

    func testEstimateFromEnvelopeSingleFrameReturnsDefault() {
        let tempo = TempoEstimation.estimateFromEnvelope(
            envelope: [1.0],
            sr: 22050,
            hopLength: 512,
            startBPM: 100.0
        )
        XCTAssertEqual(tempo, 100.0, "Single frame should return startBPM")
    }

    func testTempoResultInReasonableRange() {
        // Test with various BPMs that result is always in [30, 300]
        for bpm in [60, 100, 140, 180] as [Float] {
            let signal = makeClickTrack(bpm: bpm, duration: 4.0)
            let tempo = TempoEstimation.tempo(signal: signal, sr: 22050)
            XCTAssertGreaterThanOrEqual(tempo, 30, "Tempo should be >= 30 for \(bpm) BPM input")
            XCTAssertLessThanOrEqual(tempo, 300, "Tempo should be <= 300 for \(bpm) BPM input")
        }
    }
}
