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
}
