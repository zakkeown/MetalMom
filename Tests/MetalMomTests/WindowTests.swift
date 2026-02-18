import XCTest
@testable import MetalMomCore

final class WindowTests: XCTestCase {

    // MARK: - Length

    func testHannWindowLength() {
        let window = Windows.hann(length: 512)
        XCTAssertEqual(window.count, 512)
    }

    func testHannWindowLengthOne() {
        let window = Windows.hann(length: 1)
        XCTAssertEqual(window.count, 1)
        XCTAssertEqual(window[0], 1.0)
    }

    func testHannWindowLengthZero() {
        let window = Windows.hann(length: 0)
        XCTAssertTrue(window.isEmpty)
    }

    // MARK: - Periodic endpoints (default)

    func testHannPeriodicFirstElementIsZero() {
        let window = Windows.hann(length: 8)
        XCTAssertEqual(window[0], 0.0, accuracy: 1e-7)
    }

    func testHannPeriodicLastElementIsNotZero() {
        // Periodic window: last element should NOT be zero
        let window = Windows.hann(length: 8)
        XCTAssertGreaterThan(window[7], 0.0)
    }

    // MARK: - Symmetric endpoints

    func testHannSymmetricEndpointsAreZero() {
        let window = Windows.hann(length: 8, periodic: false)
        XCTAssertEqual(window[0], 0.0, accuracy: 1e-7)
        XCTAssertEqual(window[7], 0.0, accuracy: 1e-7)
    }

    // MARK: - Peak value

    func testHannWindowPeakIsOne() {
        let window = Windows.hann(length: 256)
        let maxVal = window.max()!
        XCTAssertEqual(maxVal, 1.0, accuracy: 1e-6)
    }

    // MARK: - Known values (periodic, length 4)
    // Periodic Hann: w[n] = 0.5 * (1 - cos(2*pi*n/N))  for N = length
    //   w[0] = 0.5*(1 - cos(0))       = 0.0
    //   w[1] = 0.5*(1 - cos(pi/2))    = 0.5
    //   w[2] = 0.5*(1 - cos(pi))      = 1.0
    //   w[3] = 0.5*(1 - cos(3*pi/2))  = 0.5

    func testHannPeriodicKnownValues() {
        let window = Windows.hann(length: 4)
        XCTAssertEqual(window[0], 0.0, accuracy: 1e-7)
        XCTAssertEqual(window[1], 0.5, accuracy: 1e-6)
        XCTAssertEqual(window[2], 1.0, accuracy: 1e-6)
        XCTAssertEqual(window[3], 0.5, accuracy: 1e-6)
    }

    // MARK: - Known values (symmetric, length 5)
    // Symmetric Hann: w[n] = 0.5 * (1 - cos(2*pi*n/(N-1)))  for N = length
    //   w[0] = 0.5*(1 - cos(0))       = 0.0
    //   w[1] = 0.5*(1 - cos(pi/2))    = 0.5
    //   w[2] = 0.5*(1 - cos(pi))      = 1.0
    //   w[3] = 0.5*(1 - cos(3*pi/2))  = 0.5
    //   w[4] = 0.5*(1 - cos(2*pi))    = 0.0

    func testHannSymmetricKnownValues() {
        let window = Windows.hann(length: 5, periodic: false)
        XCTAssertEqual(window[0], 0.0, accuracy: 1e-7)
        XCTAssertEqual(window[1], 0.5, accuracy: 1e-6)
        XCTAssertEqual(window[2], 1.0, accuracy: 1e-6)
        XCTAssertEqual(window[3], 0.5, accuracy: 1e-6)
        XCTAssertEqual(window[4], 0.0, accuracy: 1e-7)
    }

    // MARK: - General Cosine

    func testGeneralCosineHannEquivalence() {
        // Hann = generalCosine with [0.5, 0.5]
        let hann = Windows.hann(length: 256)
        let gc = Windows.generalCosine(length: 256, coefficients: [0.5, 0.5])
        XCTAssertEqual(hann.count, gc.count)
        for i in 0..<hann.count {
            XCTAssertEqual(hann[i], gc[i], accuracy: 1e-6,
                           "Mismatch at index \(i)")
        }
    }

    func testGeneralCosineEdgeCases() {
        XCTAssertTrue(Windows.generalCosine(length: 0, coefficients: [0.5, 0.5]).isEmpty)
        let single = Windows.generalCosine(length: 1, coefficients: [0.5, 0.5])
        XCTAssertEqual(single.count, 1)
        XCTAssertEqual(single[0], 1.0)
        let empty = Windows.generalCosine(length: 5, coefficients: [])
        XCTAssertEqual(empty, [Float](repeating: 0, count: 5))
    }

    // MARK: - Hamming Window

    func testHammingLength() {
        let window = Windows.hamming(length: 512)
        XCTAssertEqual(window.count, 512)
    }

    func testHammingEdgeCases() {
        XCTAssertTrue(Windows.hamming(length: 0).isEmpty)
        let single = Windows.hamming(length: 1)
        XCTAssertEqual(single.count, 1)
        XCTAssertEqual(single[0], 1.0)
    }

    func testHammingSymmetricEndpoints() {
        // Hamming endpoints should be 0.54 - 0.46 = 0.08 (not zero)
        let window = Windows.hamming(length: 8, periodic: false)
        XCTAssertEqual(window[0], 0.08, accuracy: 1e-6)
        XCTAssertEqual(window[7], 0.08, accuracy: 1e-6)
    }

    func testHammingSymmetricSymmetry() {
        let window = Windows.hamming(length: 64, periodic: false)
        for i in 0..<32 {
            XCTAssertEqual(window[i], window[63 - i], accuracy: 1e-6,
                           "Hamming not symmetric at \(i)")
        }
    }

    func testHammingPeakIsOne() {
        let window = Windows.hamming(length: 256)
        let maxVal = window.max()!
        XCTAssertEqual(maxVal, 1.0, accuracy: 1e-6)
    }

    func testHammingPeriodicVsSymmetric() {
        let periodic = Windows.hamming(length: 8)
        let symmetric = Windows.hamming(length: 8, periodic: false)
        // They should differ
        var differ = false
        for i in 0..<8 {
            if abs(periodic[i] - symmetric[i]) > 1e-7 { differ = true; break }
        }
        XCTAssertTrue(differ, "Periodic and symmetric Hamming should differ")
    }

    // scipy.signal.windows.hamming(8, sym=False) — periodic:
    // [0.08, 0.21473088, 0.54, 0.86526912, 1.0, 0.86526912, 0.54, 0.21473088]
    func testHammingKnownValues() {
        let window = Windows.hamming(length: 8, periodic: true)
        let expected: [Float] = [0.08, 0.21473088, 0.54, 0.86526912,
                                 1.0, 0.86526912, 0.54, 0.21473088]
        XCTAssertEqual(window.count, expected.count)
        for i in 0..<window.count {
            XCTAssertEqual(window[i], expected[i], accuracy: 1e-4,
                           "Hamming mismatch at \(i): got \(window[i]), expected \(expected[i])")
        }
    }

    // MARK: - Blackman Window

    func testBlackmanLength() {
        let window = Windows.blackman(length: 512)
        XCTAssertEqual(window.count, 512)
    }

    func testBlackmanEdgeCases() {
        XCTAssertTrue(Windows.blackman(length: 0).isEmpty)
        let single = Windows.blackman(length: 1)
        XCTAssertEqual(single.count, 1)
        XCTAssertEqual(single[0], 1.0)
    }

    func testBlackmanSymmetricEndpoints() {
        // Blackman endpoints: 0.42 - 0.5 + 0.08 = 0.0
        let window = Windows.blackman(length: 64, periodic: false)
        XCTAssertEqual(window[0], 0.0, accuracy: 1e-6)
        XCTAssertEqual(window[63], 0.0, accuracy: 1e-6)
    }

    func testBlackmanSymmetricSymmetry() {
        let window = Windows.blackman(length: 64, periodic: false)
        for i in 0..<32 {
            XCTAssertEqual(window[i], window[63 - i], accuracy: 1e-6,
                           "Blackman not symmetric at \(i)")
        }
    }

    func testBlackmanPeakIsOne() {
        let window = Windows.blackman(length: 256)
        let maxVal = window.max()!
        XCTAssertEqual(maxVal, 1.0, accuracy: 1e-5)
    }

    func testBlackmanValuesInRange() {
        let window = Windows.blackman(length: 256)
        for (i, v) in window.enumerated() {
            XCTAssertGreaterThanOrEqual(v, -1e-6, "Blackman negative at \(i)")
            XCTAssertLessThanOrEqual(v, 1.0 + 1e-6, "Blackman > 1 at \(i)")
        }
    }

    // scipy.signal.windows.blackman(8, sym=False) — periodic:
    // [~0, 0.06644661, 0.34, 0.77355339, 1.0, 0.77355339, 0.34, 0.06644661]
    func testBlackmanKnownValues() {
        let window = Windows.blackman(length: 8, periodic: true)
        let expected: [Float] = [0.0, 0.06644661, 0.34, 0.77355339,
                                 1.0, 0.77355339, 0.34, 0.06644661]
        XCTAssertEqual(window.count, expected.count)
        for i in 0..<window.count {
            XCTAssertEqual(window[i], expected[i], accuracy: 1e-4,
                           "Blackman mismatch at \(i): got \(window[i]), expected \(expected[i])")
        }
    }

    // MARK: - Bartlett Window

    func testBartlettLength() {
        let window = Windows.bartlett(length: 512)
        XCTAssertEqual(window.count, 512)
    }

    func testBartlettEdgeCases() {
        XCTAssertTrue(Windows.bartlett(length: 0).isEmpty)
        let single = Windows.bartlett(length: 1)
        XCTAssertEqual(single.count, 1)
        XCTAssertEqual(single[0], 1.0)
    }

    func testBartlettSymmetricEndpointsAreZero() {
        let window = Windows.bartlett(length: 8, periodic: false)
        XCTAssertEqual(window[0], 0.0, accuracy: 1e-6)
        XCTAssertEqual(window[7], 0.0, accuracy: 1e-6)
    }

    func testBartlettSymmetricPeakAtCenter() {
        // For even-length symmetric Bartlett, peak should be 1.0
        let window = Windows.bartlett(length: 9, periodic: false)
        XCTAssertEqual(window[4], 1.0, accuracy: 1e-6)
    }

    func testBartlettSymmetricSymmetry() {
        let window = Windows.bartlett(length: 64, periodic: false)
        for i in 0..<32 {
            XCTAssertEqual(window[i], window[63 - i], accuracy: 1e-6,
                           "Bartlett not symmetric at \(i)")
        }
    }

    func testBartlettKnownValues() {
        // Symmetric Bartlett(5): [0, 0.5, 1.0, 0.5, 0]
        let window = Windows.bartlett(length: 5, periodic: false)
        let expected: [Float] = [0.0, 0.5, 1.0, 0.5, 0.0]
        for i in 0..<window.count {
            XCTAssertEqual(window[i], expected[i], accuracy: 1e-6,
                           "Bartlett mismatch at \(i)")
        }
    }

    func testBartlettValuesInRange() {
        let window = Windows.bartlett(length: 256)
        for (i, v) in window.enumerated() {
            XCTAssertGreaterThanOrEqual(v, -1e-6, "Bartlett negative at \(i)")
            XCTAssertLessThanOrEqual(v, 1.0 + 1e-6, "Bartlett > 1 at \(i)")
        }
    }

    // MARK: - Kaiser Window

    func testKaiserLength() {
        let window = Windows.kaiser(length: 512)
        XCTAssertEqual(window.count, 512)
    }

    func testKaiserEdgeCases() {
        XCTAssertTrue(Windows.kaiser(length: 0).isEmpty)
        let single = Windows.kaiser(length: 1)
        XCTAssertEqual(single.count, 1)
        XCTAssertEqual(single[0], 1.0)
    }

    func testKaiserCenterPeakIsOne() {
        // Kaiser peak at center = I0(beta)/I0(beta) = 1.0
        let window = Windows.kaiser(length: 65, periodic: false)
        XCTAssertEqual(window[32], 1.0, accuracy: 1e-5)
    }

    func testKaiserBetaZeroIsRectangular() {
        // beta = 0 => I0(0) = 1 everywhere => rectangular
        let window = Windows.kaiser(length: 64, beta: 0.0, periodic: false)
        for (i, v) in window.enumerated() {
            XCTAssertEqual(v, 1.0, accuracy: 1e-5,
                           "Kaiser beta=0 should be 1.0 at \(i), got \(v)")
        }
    }

    func testKaiserSymmetricSymmetry() {
        let window = Windows.kaiser(length: 64, periodic: false)
        for i in 0..<32 {
            XCTAssertEqual(window[i], window[63 - i], accuracy: 1e-5,
                           "Kaiser not symmetric at \(i)")
        }
    }

    func testKaiserValuesInRange() {
        let window = Windows.kaiser(length: 256)
        for (i, v) in window.enumerated() {
            XCTAssertGreaterThanOrEqual(v, -1e-6, "Kaiser negative at \(i)")
            XCTAssertLessThanOrEqual(v, 1.0 + 1e-5, "Kaiser > 1 at \(i)")
        }
    }

    func testKaiserPeriodicVsSymmetric() {
        let periodic = Windows.kaiser(length: 8)
        let symmetric = Windows.kaiser(length: 8, periodic: false)
        var differ = false
        for i in 0..<8 {
            if abs(periodic[i] - symmetric[i]) > 1e-7 { differ = true; break }
        }
        XCTAssertTrue(differ, "Periodic and symmetric Kaiser should differ")
    }

    // scipy.signal.windows.kaiser(8, 12.0, sym=True) — symmetric:
    // [5.27734413e-05, 3.27682884e-02, 3.30898637e-01, 8.88867664e-01,
    //  8.88867664e-01, 3.30898637e-01, 3.27682884e-02, 5.27734413e-05]
    func testKaiserKnownValuesSymmetric() {
        let window = Windows.kaiser(length: 8, beta: 12.0, periodic: false)
        let expected: [Float] = [5.27734413e-05, 3.27682884e-02, 3.30898637e-01, 8.88867664e-01,
                                 8.88867664e-01, 3.30898637e-01, 3.27682884e-02, 5.27734413e-05]
        XCTAssertEqual(window.count, expected.count)
        for i in 0..<window.count {
            XCTAssertEqual(window[i], expected[i], accuracy: 1e-3,
                           "Kaiser mismatch at \(i): got \(window[i]), expected \(expected[i])")
        }
    }

    // MARK: - Rectangular Window

    func testRectangularLength() {
        let window = Windows.rectangular(length: 512)
        XCTAssertEqual(window.count, 512)
    }

    func testRectangularEdgeCases() {
        XCTAssertTrue(Windows.rectangular(length: 0).isEmpty)
    }

    func testRectangularAllOnes() {
        let window = Windows.rectangular(length: 64)
        for (i, v) in window.enumerated() {
            XCTAssertEqual(v, 1.0, accuracy: 1e-7,
                           "Rectangular should be 1.0 at \(i)")
        }
    }

    // MARK: - Triangular Window

    func testTriangularLength() {
        let window = Windows.triangular(length: 512)
        XCTAssertEqual(window.count, 512)
    }

    func testTriangularEdgeCases() {
        XCTAssertTrue(Windows.triangular(length: 0).isEmpty)
        let single = Windows.triangular(length: 1)
        XCTAssertEqual(single.count, 1)
        XCTAssertEqual(single[0], 1.0)
    }

    func testTriangularEndpointsNotZero() {
        // Unlike Bartlett, triangular endpoints should NOT be zero
        let window = Windows.triangular(length: 8, periodic: false)
        XCTAssertGreaterThan(window[0], 0.0)
        XCTAssertGreaterThan(window[7], 0.0)
    }

    func testTriangularSymmetricSymmetry() {
        let window = Windows.triangular(length: 64, periodic: false)
        for i in 0..<32 {
            XCTAssertEqual(window[i], window[63 - i], accuracy: 1e-6,
                           "Triangular not symmetric at \(i)")
        }
    }

    func testTriangularValuesInRange() {
        let window = Windows.triangular(length: 256)
        for (i, v) in window.enumerated() {
            XCTAssertGreaterThanOrEqual(v, -1e-6, "Triangular negative at \(i)")
            XCTAssertLessThanOrEqual(v, 1.0 + 1e-6, "Triangular > 1 at \(i)")
        }
    }

    func testTriangularPeriodicVsSymmetric() {
        let periodic = Windows.triangular(length: 8)
        let symmetric = Windows.triangular(length: 8, periodic: false)
        var differ = false
        for i in 0..<8 {
            if abs(periodic[i] - symmetric[i]) > 1e-7 { differ = true; break }
        }
        XCTAssertTrue(differ, "Periodic and symmetric Triangular should differ")
    }
}
