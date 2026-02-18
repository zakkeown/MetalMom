import XCTest
@testable import MetalMomCore

final class ResampleTests: XCTestCase {
    func testResampleIdentity() {
        let signal = Signal(data: [1, 2, 3, 4, 5], sampleRate: 22050)
        let result = Resample.resample(signal: signal, targetRate: 22050)
        XCTAssertEqual(result.sampleRate, 22050)
        XCTAssertEqual(result.count, 5)
    }

    func testResampleDownsample() {
        // 1 second at 44100 -> 22050
        let n = 44100
        var samples = [Float](repeating: 0, count: n)
        for i in 0..<n {
            samples[i] = sin(Float(i) * 440.0 * 2.0 * .pi / 44100.0)
        }
        let signal = Signal(data: samples, sampleRate: 44100)
        let result = Resample.resample(signal: signal, targetRate: 22050)
        XCTAssertEqual(result.sampleRate, 22050)
        XCTAssertEqual(result.count, 22050, accuracy: 1)
    }

    func testResampleUpsample() {
        // 1 second at 22050 -> 44100
        let n = 22050
        var samples = [Float](repeating: 0, count: n)
        for i in 0..<n {
            samples[i] = sin(Float(i) * 440.0 * 2.0 * .pi / 22050.0)
        }
        let signal = Signal(data: samples, sampleRate: 22050)
        let result = Resample.resample(signal: signal, targetRate: 44100)
        XCTAssertEqual(result.sampleRate, 44100)
        XCTAssertEqual(result.count, 44100, accuracy: 1)
    }

    func testResampleRoundTripSNR() {
        // Round-trip: 22050 -> 44100 -> 22050 should have SNR > 40 dB
        let n = 22050
        var samples = [Float](repeating: 0, count: n)
        for i in 0..<n {
            // 440 Hz tone well below Nyquist
            samples[i] = sin(Float(i) * 440.0 * 2.0 * .pi / 22050.0)
        }
        let signal = Signal(data: samples, sampleRate: 22050)
        let up = Resample.resample(signal: signal, targetRate: 44100)
        let roundTrip = Resample.resample(signal: up, targetRate: 22050)

        // Compute SNR
        let outCount = min(signal.count, roundTrip.count)
        var signalPower: Float = 0
        var noisePower: Float = 0
        // Skip edges (filter transients)
        let skip = 1000
        for i in skip..<(outCount - skip) {
            let s = samples[i]
            let r = roundTrip[i]
            signalPower += s * s
            noisePower += (s - r) * (s - r)
        }
        let snr = 10.0 * log10f(signalPower / max(noisePower, 1e-20))
        XCTAssertGreaterThan(snr, 40.0, "Round-trip SNR should be > 40 dB, got \(snr) dB")
    }

    func testResampleEmpty() {
        let signal = Signal(data: [], sampleRate: 22050)
        let result = Resample.resample(signal: signal, targetRate: 44100)
        XCTAssertEqual(result.count, 0)
    }

    func testResamplePreservesEnergy() {
        let n = 22050
        var samples = [Float](repeating: 0, count: n)
        for i in 0..<n {
            samples[i] = sin(Float(i) * 440.0 * 2.0 * .pi / 22050.0)
        }
        let signal = Signal(data: samples, sampleRate: 22050)
        let resampled = Resample.resample(signal: signal, targetRate: 44100)

        // RMS should be approximately preserved
        var origRMS: Float = 0
        for s in samples { origRMS += s * s }
        origRMS = sqrtf(origRMS / Float(n))

        var resRMS: Float = 0
        for i in 0..<resampled.count { resRMS += resampled[i] * resampled[i] }
        resRMS = sqrtf(resRMS / Float(resampled.count))

        XCTAssertEqual(resRMS, origRMS, accuracy: 0.05)
    }
}
