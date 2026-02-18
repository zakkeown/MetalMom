import XCTest
@testable import MetalMomCore

final class TonnetzTests: XCTestCase {

    // MARK: - Helpers

    /// 1 second of 440 Hz sine at 22050 Hz.
    private func makeSineSignal(frequency: Float = 440.0, sr: Int = 22050,
                                 duration: Float = 1.0) -> Signal {
        let count = Int(Float(sr) * duration)
        let data = (0..<count).map { i in
            sinf(2.0 * .pi * frequency * Float(i) / Float(sr))
        }
        return Signal(data: data, sampleRate: sr)
    }

    // MARK: - Shape Tests

    func testTonnetzShape() {
        let signal = makeSineSignal()
        let result = Tonnetz.compute(signal: signal, nFFT: 2048)
        XCTAssertEqual(result.shape.count, 2, "Tonnetz should be 2D")
        XCTAssertEqual(result.shape[0], 6, "Should have 6 tonnetz dimensions")
        XCTAssertGreaterThan(result.shape[1], 0, "Should have at least 1 frame")
    }

    func testTonnetzShapeMatchesChromaFrames() {
        let signal = makeSineSignal()
        let chroma = Chroma.stft(signal: signal)
        let tonnetz = Tonnetz.compute(signal: signal)
        XCTAssertEqual(tonnetz.shape[1], chroma.shape[1],
                       "Tonnetz frame count should match chroma frame count")
    }

    func testTonnetzShapeCustomNFFT() {
        let signal = makeSineSignal()
        let result = Tonnetz.compute(signal: signal, nFFT: 1024)
        XCTAssertEqual(result.shape[0], 6)
        XCTAssertGreaterThan(result.shape[1], 0)
    }

    // MARK: - Value Tests

    func testTonnetzValuesFinite() {
        let signal = makeSineSignal()
        let result = Tonnetz.compute(signal: signal)
        for i in 0..<result.count {
            XCTAssertFalse(result[i].isNaN, "Tonnetz value should not be NaN")
            XCTAssertFalse(result[i].isInfinite, "Tonnetz value should not be infinite")
        }
    }

    func testTonnetzValueRange() {
        // Since chroma is L1-normalized and basis functions have max magnitude r,
        // each tonnetz dimension should be in [-r, r] where r is 1.0 or 0.5
        let signal = makeSineSignal()
        let result = Tonnetz.compute(signal: signal)
        let nFrames = result.shape[1]
        for d in 0..<6 {
            let maxR: Float = (d < 4) ? 1.0 : 0.5
            for f in 0..<nFrames {
                let val = abs(result[d * nFrames + f])
                XCTAssertLessThanOrEqual(val, maxR + 0.01,
                    "Tonnetz dim \(d) value should be bounded by radius \(maxR)")
            }
        }
    }

    func testTonnetzSilentSignal() {
        // Silent signal -> all zeros chroma -> all zeros tonnetz
        let signal = Signal(data: [Float](repeating: 0, count: 22050), sampleRate: 22050)
        let result = Tonnetz.compute(signal: signal)
        for i in 0..<result.count {
            XCTAssertEqual(result[i], 0, accuracy: 1e-6,
                          "Silent signal should produce zero tonnetz")
        }
    }

    func testTonnetzDifferentPitchesDiffer() {
        // Two different frequencies should produce different tonnetz features
        let signal440 = makeSineSignal(frequency: 440.0)
        let signal261 = makeSineSignal(frequency: 261.63)
        let tonnetz440 = Tonnetz.compute(signal: signal440)
        let tonnetz261 = Tonnetz.compute(signal: signal261)

        // Average each tonnetz dimension
        let nFrames = tonnetz440.shape[1]
        var different = false
        for d in 0..<6 {
            var avg440: Float = 0
            var avg261: Float = 0
            for f in 0..<nFrames {
                avg440 += tonnetz440[d * nFrames + f]
                avg261 += tonnetz261[d * nFrames + f]
            }
            avg440 /= Float(nFrames)
            avg261 /= Float(nFrames)
            if abs(avg440 - avg261) > 0.01 {
                different = true
                break
            }
        }
        XCTAssertTrue(different, "Different pitches should produce different tonnetz features")
    }
}
