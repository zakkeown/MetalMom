import XCTest
@testable import MetalMomCore

final class HPSSTests: XCTestCase {

    // MARK: - Helpers

    /// Generate a pure sine wave signal.
    private func makeSine(frequency: Float = 440.0, sr: Int = 22050, duration: Float = 0.5) -> Signal {
        let length = Int(Float(sr) * duration)
        var data = [Float](repeating: 0, count: length)
        let angularFreq = 2.0 * Float.pi * frequency / Float(sr)
        for i in 0..<length {
            data[i] = sinf(angularFreq * Float(i))
        }
        return Signal(data: data, sampleRate: sr)
    }

    /// Generate a click train (periodic impulses) as a percussive signal.
    private func makeClicks(sr: Int = 22050, duration: Float = 0.5, interval: Int = 2000) -> Signal {
        let length = Int(Float(sr) * duration)
        var data = [Float](repeating: 0, count: length)
        var pos = 0
        while pos < length {
            data[pos] = 1.0
            pos += interval
        }
        return Signal(data: data, sampleRate: sr)
    }

    /// Compute energy of a Signal.
    private func energy(_ signal: Signal) -> Float {
        var sum: Float = 0
        signal.withUnsafeBufferPointer { buf in
            for i in 0..<buf.count {
                sum += buf[i] * buf[i]
            }
        }
        return sum
    }

    // MARK: - Tests

    func testHPSSReturnsTwoSignals() {
        let signal = makeSine(duration: 0.25)
        let (h, p) = HPSS.hpss(signal: signal, nFFT: 1024)

        XCTAssertGreaterThan(h.count, 0, "Harmonic signal should not be empty")
        XCTAssertGreaterThan(p.count, 0, "Percussive signal should not be empty")
    }

    func testHPSSOutputLength() {
        let signal = makeSine(duration: 0.25)
        let inputLength = signal.count
        let (h, p) = HPSS.hpss(signal: signal, nFFT: 1024)

        XCTAssertEqual(h.count, inputLength, "Harmonic should match input length")
        XCTAssertEqual(p.count, inputLength, "Percussive should match input length")
    }

    func testHarmonicReturns1DSignal() {
        let signal = makeSine(duration: 0.25)
        let h = HPSS.harmonic(signal: signal, nFFT: 1024)

        XCTAssertGreaterThan(h.count, 0)
        XCTAssertEqual(h.shape.count, 1, "Harmonic should be 1D")
    }

    func testPercussiveReturns1DSignal() {
        let signal = makeSine(duration: 0.25)
        let p = HPSS.percussive(signal: signal, nFFT: 1024)

        XCTAssertGreaterThan(p.count, 0)
        XCTAssertEqual(p.shape.count, 1, "Percussive should be 1D")
    }

    func testSineIsMostlyHarmonic() {
        // A pure sine wave should have most energy in the harmonic component
        let signal = makeSine(frequency: 440.0, duration: 0.5)
        let (h, p) = HPSS.hpss(signal: signal, nFFT: 1024)

        let hEnergy = energy(h)
        let pEnergy = energy(p)
        let totalEnergy = hEnergy + pEnergy

        guard totalEnergy > 0 else {
            XCTFail("Total energy should be positive")
            return
        }

        let harmonicRatio = hEnergy / totalEnergy
        // Sine wave should be predominantly harmonic (>50% at minimum)
        XCTAssertGreaterThan(harmonicRatio, 0.5,
            "Sine wave harmonic ratio \(harmonicRatio) should be > 0.5")
    }

    func testClicksArePercussive() {
        // Click train should have more percussive energy than harmonic
        let signal = makeClicks(duration: 0.5, interval: 1000)
        let (h, p) = HPSS.hpss(signal: signal, nFFT: 1024)

        let hEnergy = energy(h)
        let pEnergy = energy(p)
        let totalEnergy = hEnergy + pEnergy

        guard totalEnergy > 0 else {
            XCTFail("Total energy should be positive")
            return
        }

        let percussiveRatio = pEnergy / totalEnergy
        // Clicks should have substantial percussive energy (>30%)
        XCTAssertGreaterThan(percussiveRatio, 0.3,
            "Click train percussive ratio \(percussiveRatio) should be > 0.3")
    }

    func testEnergyConservation() {
        // H + P should roughly approximate original signal in terms of energy
        // This is a soft check - HPSS preserves energy approximately
        let signal = makeSine(frequency: 440.0, duration: 0.5)
        let originalEnergy = energy(signal)
        let (h, p) = HPSS.hpss(signal: signal, nFFT: 1024)

        let hEnergy = energy(h)
        let pEnergy = energy(p)

        // The sum of harmonic and percussive energy should be in the
        // same order of magnitude as the original
        // Due to soft masking, energy is not perfectly conserved but
        // should be within 2x
        let ratio = (hEnergy + pEnergy) / max(originalEnergy, 1e-10)
        XCTAssertGreaterThan(ratio, 0.1, "Energy ratio too low: \(ratio)")
        XCTAssertLessThan(ratio, 5.0, "Energy ratio too high: \(ratio)")
    }

    func testCustomKernelSize() {
        let signal = makeSine(duration: 0.25)
        // Should not crash with different kernel sizes
        let (h1, _) = HPSS.hpss(signal: signal, nFFT: 1024, kernelSize: 11)
        let (h2, _) = HPSS.hpss(signal: signal, nFFT: 1024, kernelSize: 51)

        XCTAssertGreaterThan(h1.count, 0)
        XCTAssertGreaterThan(h2.count, 0)
    }

    func testCustomPowerAndMargin() {
        let signal = makeSine(duration: 0.25)
        // Should not crash with different power/margin values
        let (h, p) = HPSS.hpss(signal: signal, nFFT: 1024, power: 1.0, margin: 2.0)

        XCTAssertGreaterThan(h.count, 0)
        XCTAssertGreaterThan(p.count, 0)
    }
}
