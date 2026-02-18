import XCTest
@testable import MetalMomCore

final class ScalingTests: XCTestCase {

    // MARK: - amplitudeToDb

    func testAmplitudeToDbKnownValues() {
        // amplitude 1.0 with ref=1.0 → 0 dB
        let sig1 = Signal(data: [1.0], sampleRate: 22050)
        let db1 = Scaling.amplitudeToDb(sig1, ref: 1.0)
        XCTAssertEqual(db1[0], 0.0, accuracy: 1e-5, "1.0 amplitude → 0 dB")

        // amplitude 0.5 → 20*log10(0.5) ≈ -6.0206 dB
        let sig2 = Signal(data: [0.5], sampleRate: 22050)
        let db2 = Scaling.amplitudeToDb(sig2, ref: 1.0)
        XCTAssertEqual(db2[0], -6.0206, accuracy: 1e-3, "0.5 amplitude → ~-6.02 dB")

        // amplitude 10.0 → 20*log10(10) = 20 dB
        let sig3 = Signal(data: [10.0], sampleRate: 22050)
        let db3 = Scaling.amplitudeToDb(sig3, ref: 1.0)
        XCTAssertEqual(db3[0], 20.0, accuracy: 1e-3, "10.0 amplitude → 20 dB")
    }

    func testAmplitudeToDbMultipleValues() {
        let amplitudes: [Float] = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 10.0]
        let signal = Signal(data: amplitudes, sampleRate: 22050)
        let db = Scaling.amplitudeToDb(signal, ref: 1.0, topDb: nil)

        for i in 0..<amplitudes.count {
            let expected = 20.0 * log10(max(amplitudes[i], 1e-5))
            XCTAssertEqual(db[i], expected, accuracy: 1e-3,
                           "amplitudeToDb(\(amplitudes[i])) = \(db[i]), expected \(expected)")
        }
    }

    // MARK: - powerToDb

    func testPowerToDbKnownValues() {
        // power 1.0 with ref=1.0 → 0 dB
        let sig1 = Signal(data: [1.0], sampleRate: 22050)
        let db1 = Scaling.powerToDb(sig1, ref: 1.0)
        XCTAssertEqual(db1[0], 0.0, accuracy: 1e-5, "power 1.0 → 0 dB")

        // power 0.5 → 10*log10(0.5) ≈ -3.0103 dB
        let sig2 = Signal(data: [0.5], sampleRate: 22050)
        let db2 = Scaling.powerToDb(sig2, ref: 1.0)
        XCTAssertEqual(db2[0], -3.0103, accuracy: 1e-3, "power 0.5 → ~-3.01 dB")

        // power 100.0 → 10*log10(100) = 20 dB
        let sig3 = Signal(data: [100.0], sampleRate: 22050)
        let db3 = Scaling.powerToDb(sig3, ref: 1.0)
        XCTAssertEqual(db3[0], 20.0, accuracy: 1e-3, "power 100 → 20 dB")
    }

    // MARK: - Round-trip tests

    func testAmplitudeRoundTrip() {
        let original: [Float] = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]
        let signal = Signal(data: original, sampleRate: 22050)

        let db = Scaling.amplitudeToDb(signal, ref: 1.0, topDb: nil)
        let recovered = Scaling.dbToAmplitude(db, ref: 1.0)

        for i in 0..<original.count {
            XCTAssertEqual(recovered[i], original[i], accuracy: 1e-4,
                           "Round-trip amplitude[\(i)]: \(original[i]) → \(db[i]) dB → \(recovered[i])")
        }
    }

    func testPowerRoundTrip() {
        let original: [Float] = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        let signal = Signal(data: original, sampleRate: 22050)

        let db = Scaling.powerToDb(signal, ref: 1.0, topDb: nil)
        let recovered = Scaling.dbToPower(db, ref: 1.0)

        for i in 0..<original.count {
            XCTAssertEqual(recovered[i], original[i], accuracy: max(original[i] * 1e-4, 1e-5),
                           "Round-trip power[\(i)]: \(original[i]) → \(db[i]) dB → \(recovered[i])")
        }
    }

    // MARK: - amin clamping

    func testAminClamping() {
        // Very small values should be clamped to amin before log
        let signal = Signal(data: [1e-20, 0.0, 1e-30], sampleRate: 22050)

        // With amin=1e-5 (amplitude default), all should be clamped to amin
        let db = Scaling.amplitudeToDb(signal, ref: 1.0, amin: 1e-5, topDb: nil)
        let expectedDb = 20.0 * log10(Float(1e-5))  // = -100 dB

        for i in 0..<signal.count {
            XCTAssertEqual(db[i], expectedDb, accuracy: 1e-2,
                           "Clamped value at index \(i) should be \(expectedDb), got \(db[i])")
        }
    }

    func testPowerAminClamping() {
        let signal = Signal(data: [1e-20, 0.0], sampleRate: 22050)
        let db = Scaling.powerToDb(signal, ref: 1.0, amin: 1e-10, topDb: nil)
        let expectedDb = 10.0 * log10(Float(1e-10))  // = -100 dB

        for i in 0..<signal.count {
            XCTAssertEqual(db[i], expectedDb, accuracy: 1e-2,
                           "Power clamped value at index \(i) should be \(expectedDb), got \(db[i])")
        }
    }

    // MARK: - topDb clipping

    func testTopDbClipping() {
        // Signal with wide dynamic range
        let signal = Signal(data: [1.0, 0.1, 0.01, 0.001, 1e-5], sampleRate: 22050)
        let db = Scaling.amplitudeToDb(signal, ref: 1.0, topDb: 40.0)

        // Find max dB value
        var maxDb: Float = -Float.infinity
        for i in 0..<db.count {
            if db[i] > maxDb { maxDb = db[i] }
        }

        // All values should be within topDb of the max
        for i in 0..<db.count {
            XCTAssertGreaterThanOrEqual(db[i], maxDb - 40.0,
                                        "Value at \(i) should be >= \(maxDb - 40.0), got \(db[i])")
        }
    }

    func testTopDbNilNoClipping() {
        // Without topDb, very small values can go very negative
        let signal = Signal(data: [1.0, 1e-5], sampleRate: 22050)
        let db = Scaling.amplitudeToDb(signal, ref: 1.0, amin: 1e-10, topDb: nil)

        // 1e-5 → 20*log10(1e-5) = -100 dB, which is 100 dB below 0 dB
        // Without topDb clipping, this should remain at -100 dB
        XCTAssertEqual(db[0], 0.0, accuracy: 1e-3)
        XCTAssertEqual(db[1], -100.0, accuracy: 1e-2)
    }

    // MARK: - dbToAmplitude

    func testDbToAmplitudeKnownValues() {
        let signal = Signal(data: [0.0, -6.0206, 20.0], sampleRate: 22050)
        let amp = Scaling.dbToAmplitude(signal, ref: 1.0)

        XCTAssertEqual(amp[0], 1.0, accuracy: 1e-4, "0 dB → amplitude 1.0")
        XCTAssertEqual(amp[1], 0.5, accuracy: 1e-3, "-6.02 dB → amplitude 0.5")
        XCTAssertEqual(amp[2], 10.0, accuracy: 1e-3, "20 dB → amplitude 10.0")
    }

    // MARK: - dbToPower

    func testDbToPowerKnownValues() {
        let signal = Signal(data: [0.0, -3.0103, 20.0], sampleRate: 22050)
        let pwr = Scaling.dbToPower(signal, ref: 1.0)

        XCTAssertEqual(pwr[0], 1.0, accuracy: 1e-4, "0 dB → power 1.0")
        XCTAssertEqual(pwr[1], 0.5, accuracy: 1e-3, "-3.01 dB → power 0.5")
        XCTAssertEqual(pwr[2], 100.0, accuracy: 1e-1, "20 dB → power 100.0")
    }

    // MARK: - Relationship: power vs amplitude

    func testPowerAmplitudeRelationship() {
        // powerToDb(x^2) should equal amplitudeToDb(x) for positive x
        let amplitudes: [Float] = [0.1, 0.5, 1.0, 2.0, 5.0]
        let powers = amplitudes.map { $0 * $0 }

        let ampSignal = Signal(data: amplitudes, sampleRate: 22050)
        let pwrSignal = Signal(data: powers, sampleRate: 22050)

        let ampDb = Scaling.amplitudeToDb(ampSignal, ref: 1.0, topDb: nil)
        let pwrDb = Scaling.powerToDb(pwrSignal, ref: 1.0, topDb: nil)

        for i in 0..<amplitudes.count {
            XCTAssertEqual(ampDb[i], pwrDb[i], accuracy: 1e-3,
                           "amplitudeToDb(\(amplitudes[i])) = \(ampDb[i]) should equal powerToDb(\(powers[i])) = \(pwrDb[i])")
        }
    }

    // MARK: - Ref parameter

    func testRefParameter() {
        // With ref=2.0: 20*log10(1.0/2.0) = 20*log10(0.5) ≈ -6.02 dB
        let signal = Signal(data: [1.0], sampleRate: 22050)
        let db = Scaling.amplitudeToDb(signal, ref: 2.0)
        XCTAssertEqual(db[0], -6.0206, accuracy: 1e-3, "ref=2.0 should shift result")
    }

    // MARK: - Output shape preservation

    func testOutputShapePreservation() {
        let data: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        let signal = Signal(data: data, shape: [2, 3], sampleRate: 22050)
        let db = Scaling.amplitudeToDb(signal, ref: 1.0)

        XCTAssertEqual(db.shape, [2, 3], "Output shape should match input shape")
        XCTAssertEqual(db.count, 6, "Output count should match input count")
    }
}
