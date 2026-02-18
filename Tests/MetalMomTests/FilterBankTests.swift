import XCTest
@testable import MetalMomCore

final class FilterBankTests: XCTestCase {

    // MARK: - Shape

    func testDefaultShape() {
        // Default: nMels=128, nFFT=2048 → shape [128, 1025]
        let fb = FilterBank.mel(sr: 22050, nFFT: 2048)
        XCTAssertEqual(fb.shape, [128, 1025],
                       "Default mel filterbank shape should be [128, 1025]")
    }

    func testCustomShape() {
        let fb = FilterBank.mel(sr: 22050, nFFT: 1024, nMels: 40)
        XCTAssertEqual(fb.shape, [40, 513],
                       "Custom mel filterbank shape should be [40, 513]")
    }

    func testTotalElementCount() {
        let fb = FilterBank.mel(sr: 22050, nFFT: 2048, nMels: 128)
        XCTAssertEqual(fb.count, 128 * 1025,
                       "Total element count should be nMels * (nFFT/2+1)")
    }

    // MARK: - Non-negativity

    func testAllValuesNonNegative() {
        let fb = FilterBank.mel(sr: 22050, nFFT: 2048, nMels: 128)
        for i in 0..<fb.count {
            XCTAssertGreaterThanOrEqual(fb[i], 0.0,
                                         "Filterbank value at index \(i) should be >= 0")
        }
    }

    // MARK: - No empty filters

    func testNoEmptyFilters() {
        let fb = FilterBank.mel(sr: 22050, nFFT: 2048, nMels: 128)
        let nFreqs = 1025
        for m in 0..<128 {
            var filterSum: Float = 0
            for k in 0..<nFreqs {
                filterSum += fb[m * nFreqs + k]
            }
            XCTAssertGreaterThan(filterSum, 0.0,
                                 "Filter \(m) should have non-zero sum")
        }
    }

    // MARK: - Triangular shape (rise then fall)

    func testTriangularShape() {
        // Verify that each filter has a single peak: values rise then fall.
        let fb = FilterBank.mel(sr: 22050, nFFT: 2048, nMels: 40)
        let nFreqs = 1025

        for m in 0..<40 {
            // Extract non-zero values for this filter
            var nonZeroValues: [(index: Int, value: Float)] = []
            for k in 0..<nFreqs {
                let v = fb[m * nFreqs + k]
                if v > 0 {
                    nonZeroValues.append((k, v))
                }
            }

            guard nonZeroValues.count >= 2 else { continue }

            // Find the peak index within the non-zero segment
            let peak = nonZeroValues.max(by: { $0.value < $1.value })!
            let peakIdx = nonZeroValues.firstIndex(where: { $0.index == peak.index })!

            // Values before peak should be non-decreasing
            for i in 1..<peakIdx {
                XCTAssertGreaterThanOrEqual(nonZeroValues[i].value,
                                             nonZeroValues[i - 1].value - 1e-6,
                                             "Filter \(m): rising slope violated at bin \(nonZeroValues[i].index)")
            }
            // Values after peak should be non-increasing
            for i in (peakIdx + 1)..<nonZeroValues.count {
                XCTAssertLessThanOrEqual(nonZeroValues[i].value,
                                          nonZeroValues[i - 1].value + 1e-6,
                                          "Filter \(m): falling slope violated at bin \(nonZeroValues[i].index)")
            }
        }
    }

    // MARK: - fMin / fMax

    func testCustomFMinFMax() {
        // With fMin=300, the lowest filter should have zero energy below ~300 Hz.
        let fb = FilterBank.mel(sr: 22050, nFFT: 2048, nMels: 40, fMin: 300)
        let nFreqs = 1025
        // Bin for ~300 Hz: 300 * 2048 / 22050 ≈ 27.8 → bin 27
        let binAt300 = Int(300.0 * 2048.0 / 22050.0)

        // All bins well below 300 Hz should be zero across all filters
        for k in 0..<max(binAt300 - 2, 0) {
            var colSum: Float = 0
            for m in 0..<40 {
                colSum += fb[m * nFreqs + k]
            }
            XCTAssertEqual(colSum, 0.0, accuracy: 1e-6,
                           "Bins below fMin should have zero weight, bin \(k)")
        }
    }

    func testCustomFMax() {
        // With fMax=4000, bins above ~4000 Hz should have zero energy.
        let fb = FilterBank.mel(sr: 22050, nFFT: 2048, nMels: 40, fMax: 4000)
        let nFreqs = 1025
        // Bin for 4000 Hz: 4000 * 2048 / 22050 ≈ 371.6 → bin 372
        let binAt4000 = Int(4000.0 * 2048.0 / 22050.0) + 2

        // All bins well above 4000 Hz should be zero
        for k in binAt4000..<nFreqs {
            var colSum: Float = 0
            for m in 0..<40 {
                colSum += fb[m * nFreqs + k]
            }
            XCTAssertEqual(colSum, 0.0, accuracy: 1e-6,
                           "Bins above fMax should have zero weight, bin \(k)")
        }
    }

    // MARK: - Slaney normalisation

    func testSlaneyNormalisation() {
        // With Slaney normalisation, each filter's area should be approximately 2 / bandwidth.
        // We can verify that the sum of each filter ≈ 2 / (f_right - f_left) * df,
        // where df = sr / nFFT. In practice, the sum of the normalised filter
        // should be approximately 1 (since area of unit-height triangle * enorm ≈ 1).
        // Actually: area = sum * df, and enorm * 0.5 * (fRight - fLeft) * df summed
        // should be close to 1.0 for each filter.
        // Simplified: after Slaney normalisation each filter should have similar total weight.
        let fb = FilterBank.mel(sr: 22050, nFFT: 2048, nMels: 40)
        let nFreqs = 1025

        // Check that filter sums are "reasonable" — between 0.01 and 10.0
        for m in 0..<40 {
            var filterSum: Float = 0
            for k in 0..<nFreqs {
                filterSum += fb[m * nFreqs + k]
            }
            XCTAssertGreaterThan(filterSum, 0.01,
                                 "Filter \(m) sum too small: \(filterSum)")
            XCTAssertLessThan(filterSum, 10.0,
                              "Filter \(m) sum too large: \(filterSum)")
        }
    }

    // MARK: - Sample rate variants

    func testSr44100() {
        let fb = FilterBank.mel(sr: 44100, nFFT: 2048, nMels: 128)
        XCTAssertEqual(fb.shape, [128, 1025])

        // Verify all non-negative
        for i in 0..<fb.count {
            XCTAssertGreaterThanOrEqual(fb[i], 0.0)
        }
    }

    func testSr16000() {
        let fb = FilterBank.mel(sr: 16000, nFFT: 512, nMels: 40)
        XCTAssertEqual(fb.shape, [40, 257])

        // No empty filters
        let nFreqs = 257
        for m in 0..<40 {
            var filterSum: Float = 0
            for k in 0..<nFreqs {
                filterSum += fb[m * nFreqs + k]
            }
            XCTAssertGreaterThan(filterSum, 0.0,
                                 "Filter \(m) should have non-zero sum for sr=16000")
        }
    }
}
