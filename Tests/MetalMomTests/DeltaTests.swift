import XCTest
@testable import MetalMomCore

final class DeltaTests: XCTestCase {

    // MARK: - Delta Shape Tests

    func testDeltaShape() {
        let data = Signal(data: [Float](repeating: 1, count: 200), shape: [4, 50], sampleRate: 22050)
        let result = Delta.compute(data: data)
        XCTAssertEqual(result.shape, [4, 50])
    }

    func testDeltaShapeSmallWidth() {
        let data = Signal(data: [Float](repeating: 1, count: 200), shape: [4, 50], sampleRate: 22050)
        let result = Delta.compute(data: data, width: 5)
        XCTAssertEqual(result.shape, [4, 50])
    }

    // MARK: - Delta Value Tests

    func testDeltaConstantInput() {
        // Delta of constant signal should be ~0
        let data = Signal(data: [Float](repeating: 5, count: 200), shape: [4, 50], sampleRate: 22050)
        let result = Delta.compute(data: data)
        for i in 0..<result.count {
            XCTAssertEqual(result[i], 0, accuracy: 1e-6)
        }
    }

    func testDeltaLinearRamp() {
        // Delta of linear ramp should be approximately constant
        let nFrames = 50
        var ramp = [Float](repeating: 0, count: nFrames)
        for i in 0..<nFrames {
            ramp[i] = Float(i)
        }
        let data = Signal(data: ramp, shape: [1, nFrames], sampleRate: 22050)
        let result = Delta.compute(data: data, width: 7)
        // All values should be close to 1.0 (interp mode gives constant slope for linear)
        let mid = nFrames / 2
        XCTAssertEqual(result[mid], 1.0, accuracy: 0.01)
        // Edge frames should also be 1.0 with interp mode (linear fit has exact slope)
        XCTAssertEqual(result[0], 1.0, accuracy: 0.01)
        XCTAssertEqual(result[nFrames - 1], 1.0, accuracy: 0.01)
    }

    func testDeltaDelta() {
        // Second-order delta of quadratic should be approximately constant
        let nFrames = 50
        var quadratic = [Float](repeating: 0, count: nFrames)
        for i in 0..<nFrames {
            quadratic[i] = Float(i * i)
        }
        let data = Signal(data: quadratic, shape: [1, nFrames], sampleRate: 22050)
        let result = Delta.compute(data: data, width: 7, order: 2)
        // Second derivative of x^2 should be approximately 2
        let mid = nFrames / 2
        XCTAssertEqual(result[mid], 2.0, accuracy: 0.5)
    }

    // MARK: - Stack Memory Shape Tests

    func testStackMemoryShape() {
        let data = Signal(data: [Float](repeating: 1, count: 40), shape: [4, 10], sampleRate: 22050)
        let result = Delta.stackMemory(data: data, nSteps: 3, delay: 1)
        XCTAssertEqual(result.shape, [12, 10])
    }

    func testStackMemoryShapeDefault() {
        let data = Signal(data: [Float](repeating: 1, count: 40), shape: [4, 10], sampleRate: 22050)
        let result = Delta.stackMemory(data: data)
        XCTAssertEqual(result.shape, [8, 10])  // nSteps=2 default -> 4*2 = 8
    }

    // MARK: - Stack Memory Value Tests

    func testStackMemoryValues() {
        // Simple 1-feature, 5-frame case
        let data = Signal(data: [1, 2, 3, 4, 5], shape: [1, 5], sampleRate: 22050)
        let result = Delta.stackMemory(data: data, nSteps: 2, delay: 1)
        // Step 0: no shift -> [1, 2, 3, 4, 5]
        XCTAssertEqual(result[0], 1)
        XCTAssertEqual(result[1], 2)
        XCTAssertEqual(result[2], 3)
        XCTAssertEqual(result[3], 4)
        XCTAssertEqual(result[4], 5)
        // Step 1: shift by 1, zero-padded -> [0, 1, 2, 3, 4]
        XCTAssertEqual(result[5], 0)  // t=0, src_t=-1 -> zero pad
        XCTAssertEqual(result[6], 1)  // t=1, src_t=0
        XCTAssertEqual(result[7], 2)  // t=2, src_t=1
        XCTAssertEqual(result[8], 3)  // t=3, src_t=2
        XCTAssertEqual(result[9], 4)  // t=4, src_t=3
    }

    func testStackMemoryThreeSteps() {
        // Match librosa: stack_memory([−3,−2,−1,0,1,2], n_steps=3)
        let data = Signal(data: [-3, -2, -1, 0, 1, 2], shape: [1, 6], sampleRate: 22050)
        let result = Delta.stackMemory(data: data, nSteps: 3, delay: 1)
        // Step 0: [-3, -2, -1, 0, 1, 2]
        XCTAssertEqual(result[0], -3)
        XCTAssertEqual(result[5], 2)
        // Step 1: [0, -3, -2, -1, 0, 1]
        XCTAssertEqual(result[6], 0)   // zero pad
        XCTAssertEqual(result[7], -3)
        XCTAssertEqual(result[11], 1)
        // Step 2: [0, 0, -3, -2, -1, 0]
        XCTAssertEqual(result[12], 0)  // zero pad
        XCTAssertEqual(result[13], 0)  // zero pad
        XCTAssertEqual(result[14], -3)
        XCTAssertEqual(result[17], 0)
    }

    func testStackMemoryNoShift() {
        // With nSteps=1, output should equal input
        let inputData: [Float] = [1, 2, 3, 4, 5, 6]
        let data = Signal(data: inputData, shape: [2, 3], sampleRate: 22050)
        let result = Delta.stackMemory(data: data, nSteps: 1, delay: 1)
        XCTAssertEqual(result.shape, [2, 3])
        for i in 0..<6 {
            XCTAssertEqual(result[i], inputData[i])
        }
    }

    // MARK: - Delta Finite Values

    func testDeltaValuesFinite() {
        let n = 22050
        let signal = Signal(data: (0..<n).map { sin(Float($0) * 440.0 * 2.0 * .pi / 22050.0) }, sampleRate: 22050)
        let stftResult = STFT.compute(signal: signal, nFFT: 2048)
        let result = Delta.compute(data: stftResult)
        for i in 0..<min(result.count, 100) {
            XCTAssertFalse(result[i].isNaN)
            XCTAssertFalse(result[i].isInfinite)
        }
    }
}
