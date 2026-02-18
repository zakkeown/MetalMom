import XCTest
@testable import MetalMomCore

final class UnitsTests: XCTestCase {

    // MARK: - hzToMel known values

    func testZeroHz() {
        XCTAssertEqual(Units.hzToMel(0), 0.0, accuracy: 1e-6,
                       "0 Hz should map to 0 mel")
    }

    func testOneThousandHz() {
        // 1000 Hz is the boundary: 1000 / (200/3) = 15.0 mel
        XCTAssertEqual(Units.hzToMel(1000), 15.0, accuracy: 1e-5,
                       "1000 Hz should map to 15.0 mel (Slaney)")
    }

    func testBelowBoundary() {
        // 500 Hz → 500 / (200/3) = 7.5
        XCTAssertEqual(Units.hzToMel(500), 7.5, accuracy: 1e-5,
                       "500 Hz should map to 7.5 mel (linear region)")
    }

    func testAboveBoundary() {
        // 2000 Hz: above 1000 Hz, enters log region
        let mel = Units.hzToMel(2000)
        XCTAssertGreaterThan(mel, 15.0, "2000 Hz mel should be > 15.0")
        // librosa: hz_to_mel(2000, htk=False) ≈ 25.08
        XCTAssertEqual(mel, 25.08, accuracy: 0.1,
                       "2000 Hz should be approximately 25.08 mel")
    }

    // MARK: - melToHz known values

    func testZeroMel() {
        XCTAssertEqual(Units.melToHz(0), 0.0, accuracy: 1e-6,
                       "0 mel should map to 0 Hz")
    }

    func testFifteenMel() {
        // mel 15.0 → 1000 Hz (boundary)
        XCTAssertEqual(Units.melToHz(15.0), 1000.0, accuracy: 1e-3,
                       "15.0 mel should map to 1000 Hz")
    }

    func testLinearRegionMelToHz() {
        // mel 7.5 → 500 Hz
        XCTAssertEqual(Units.melToHz(7.5), 500.0, accuracy: 1e-3,
                       "7.5 mel should map to 500 Hz")
    }

    // MARK: - Round-trip tests

    func testRoundTripLinearRegion() {
        // Test values below 1000 Hz (linear region)
        let hzValues: [Float] = [0, 100, 250, 500, 750, 999]
        for hz in hzValues {
            let mel = Units.hzToMel(hz)
            let recovered = Units.melToHz(mel)
            XCTAssertEqual(recovered, hz, accuracy: 1e-3,
                           "Round-trip failed for \(hz) Hz (linear region)")
        }
    }

    func testRoundTripLogRegion() {
        // Test values above 1000 Hz (log region)
        let hzValues: [Float] = [1000, 2000, 4000, 8000, 11025]
        for hz in hzValues {
            let mel = Units.hzToMel(hz)
            let recovered = Units.melToHz(mel)
            XCTAssertEqual(recovered, hz, accuracy: hz * 1e-5,
                           "Round-trip failed for \(hz) Hz (log region)")
        }
    }

    func testRoundTripMelToHzToMel() {
        // Reverse direction: mel → Hz → mel
        let melValues: [Float] = [0, 5, 10, 15, 20, 30, 40]
        for mel in melValues {
            let hz = Units.melToHz(mel)
            let recovered = Units.hzToMel(hz)
            XCTAssertEqual(recovered, mel, accuracy: 1e-4,
                           "Round-trip failed for mel \(mel)")
        }
    }

    // MARK: - Vectorised

    func testVectorisedHzToMel() {
        let hzValues: [Float] = [0, 500, 1000, 2000]
        let mels = Units.hzToMel(hzValues)
        XCTAssertEqual(mels.count, 4)
        for i in 0..<hzValues.count {
            XCTAssertEqual(mels[i], Units.hzToMel(hzValues[i]), accuracy: 1e-6)
        }
    }

    func testVectorisedMelToHz() {
        let melValues: [Float] = [0, 7.5, 15.0, 25.0]
        let hzs = Units.melToHz(melValues)
        XCTAssertEqual(hzs.count, 4)
        for i in 0..<melValues.count {
            XCTAssertEqual(hzs[i], Units.melToHz(melValues[i]), accuracy: 1e-6)
        }
    }

    // MARK: - Monotonicity

    func testMonotonicallyIncreasingHzToMel() {
        let hzValues: [Float] = [0, 100, 500, 1000, 2000, 5000, 10000, 22050]
        let mels = Units.hzToMel(hzValues)
        for i in 1..<mels.count {
            XCTAssertGreaterThan(mels[i], mels[i - 1],
                                 "hzToMel should be monotonically increasing")
        }
    }

    func testMonotonicallyIncreasingMelToHz() {
        let melValues: [Float] = [0, 5, 10, 15, 20, 30, 40, 50]
        let hzs = Units.melToHz(melValues)
        for i in 1..<hzs.count {
            XCTAssertGreaterThan(hzs[i], hzs[i - 1],
                                 "melToHz should be monotonically increasing")
        }
    }
}
