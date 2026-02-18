import XCTest
@testable import MetalMomCore

final class ComplexSignalTests: XCTestCase {
    func testComplexSignalCreation() {
        // Interleaved: [real0, imag0, real1, imag1, ...]
        let data: [Float] = [1.0, 0.0, 0.0, 1.0, -1.0, 0.0]  // 3 complex values
        let signal = Signal(complexData: data, shape: [3], sampleRate: 22050)
        XCTAssertEqual(signal.dtype, .complex64)
        XCTAssertEqual(signal.shape, [3])
        XCTAssertEqual(signal.count, 6)  // raw float count
        XCTAssertEqual(signal.elementCount, 3)  // complex element count
    }

    func testComplexSignalRealImagAccess() {
        let data: [Float] = [1.0, 2.0, 3.0, 4.0]  // 2 complex values
        let signal = Signal(complexData: data, shape: [2], sampleRate: 22050)
        XCTAssertEqual(signal.realPart(at: 0), 1.0)
        XCTAssertEqual(signal.imagPart(at: 0), 2.0)
        XCTAssertEqual(signal.realPart(at: 1), 3.0)
        XCTAssertEqual(signal.imagPart(at: 1), 4.0)
    }

    func testComplexSignalPointerStability() {
        let data: [Float] = [1.0, 0.0, 2.0, 0.0]
        let signal = Signal(complexData: data, shape: [2], sampleRate: 22050)
        let ptr1 = signal.dataPointer
        let ptr2 = signal.dataPointer
        XCTAssertEqual(ptr1, ptr2)
    }

    func testComplexSignal2D() {
        // 2x3 complex matrix = 12 floats
        let data: [Float] = [1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0]
        let signal = Signal(complexData: data, shape: [2, 3], sampleRate: 22050)
        XCTAssertEqual(signal.dtype, .complex64)
        XCTAssertEqual(signal.shape, [2, 3])
        XCTAssertEqual(signal.count, 12)  // raw floats
        XCTAssertEqual(signal.elementCount, 6)  // complex elements
    }

    func testRealSignalDtype() {
        let signal = Signal(data: [1.0, 2.0, 3.0], sampleRate: 22050)
        XCTAssertEqual(signal.dtype, .float32)
        XCTAssertEqual(signal.count, 3)
        XCTAssertEqual(signal.elementCount, 3)
    }

    func testWithSplitComplex() {
        let data: [Float] = [1.0, 2.0, 3.0, 4.0]  // 2 complex values
        let signal = Signal(complexData: data, shape: [2], sampleRate: 22050)
        signal.withSplitComplex { split, count in
            XCTAssertEqual(count, 2)
            // Verify de-interleaved real/imag parts
            XCTAssertEqual(split.realp[0], 1.0)
            XCTAssertEqual(split.imagp[0], 2.0)
            XCTAssertEqual(split.realp[1], 3.0)
            XCTAssertEqual(split.imagp[1], 4.0)
        }
    }
}
