import XCTest
@testable import MetalMomCore

final class SignalTests: XCTestCase {
    func testSignalFromArray() {
        let data: [Float] = [1.0, 2.0, 3.0, 4.0]
        let signal = Signal(data: data, sampleRate: 22050)
        XCTAssertEqual(signal.count, 4)
        XCTAssertEqual(signal.sampleRate, 22050)
        XCTAssertEqual(signal[0], 1.0)
        XCTAssertEqual(signal[3], 4.0)
    }

    func testSignalShape() {
        let data: [Float] = [1.0, 2.0, 3.0]
        let signal = Signal(data: data, sampleRate: 44100)
        XCTAssertEqual(signal.shape, [3])
    }

    func testSignal2D() {
        let data: [Float] = [1, 2, 3, 4, 5, 6]
        let signal = Signal(data: data, shape: [2, 3], sampleRate: 22050)
        XCTAssertEqual(signal.shape, [2, 3])
        XCTAssertEqual(signal.count, 6)
    }

    func testSignalDataPointerIsStable() {
        let data: [Float] = [1.0, 2.0, 3.0]
        let signal = Signal(data: data, sampleRate: 22050)
        // Pointer must be stable across multiple accesses (pinned storage)
        let ptr1 = signal.dataPointer
        let ptr2 = signal.dataPointer
        XCTAssertEqual(ptr1, ptr2)
        XCTAssertEqual(ptr1[0], 1.0)
        XCTAssertEqual(ptr1[2], 3.0)
    }
}
