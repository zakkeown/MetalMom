import XCTest
@testable import MetalMomCore
@testable import MetalMomBridge
import MetalMomCBridge

final class BridgeErrorTests: XCTestCase {

    // MARK: - Helpers

    /// Generate a 1-second 440 Hz sine wave at 22050 Hz.
    private func makeSineSignal(length: Int = 22050) -> [Float] {
        (0..<length).map { sinf(Float($0) * 2 * .pi * 440 / 22050) }
    }

    /// Default valid STFT parameters.
    private func makeDefaultParams() -> MMSTFTParams {
        MMSTFTParams(n_fft: 2048, hop_length: 512, win_length: 2048, center: 1)
    }

    // MARK: - Null / Invalid Input Tests

    func testNullContextReturnsError() {
        let signal = makeSineSignal()
        var params = makeDefaultParams()
        var out = MMBuffer()

        let status = MetalMomBridge.mm_stft(nil, signal, Int64(signal.count), 22050, &params, &out)
        XCTAssertEqual(status, MM_ERR_INVALID_INPUT,
                       "MetalMomBridge.mm_stft with nil context should return MM_ERR_INVALID_INPUT")
    }

    func testNullSignalReturnsError() {
        let ctx = MetalMomBridge.mm_init()!
        defer { MetalMomBridge.mm_destroy(ctx) }

        var params = makeDefaultParams()
        var out = MMBuffer()

        let status = MetalMomBridge.mm_stft(ctx, nil, 22050, 22050, &params, &out)
        XCTAssertEqual(status, MM_ERR_INVALID_INPUT,
                       "MetalMomBridge.mm_stft with nil signal should return MM_ERR_INVALID_INPUT")
    }

    func testZeroLengthReturnsError() {
        let ctx = MetalMomBridge.mm_init()!
        defer { MetalMomBridge.mm_destroy(ctx) }

        let signal = makeSineSignal()
        var params = makeDefaultParams()
        var out = MMBuffer()

        let status = MetalMomBridge.mm_stft(ctx, signal, 0, 22050, &params, &out)
        XCTAssertEqual(status, MM_ERR_INVALID_INPUT,
                       "MetalMomBridge.mm_stft with signalLength=0 should return MM_ERR_INVALID_INPUT")
    }

    func testNullOutputReturnsError() {
        let ctx = MetalMomBridge.mm_init()!
        defer { MetalMomBridge.mm_destroy(ctx) }

        let signal = makeSineSignal()
        var params = makeDefaultParams()

        let status = MetalMomBridge.mm_stft(ctx, signal, Int64(signal.count), 22050, &params, nil)
        XCTAssertEqual(status, MM_ERR_INVALID_INPUT,
                       "MetalMomBridge.mm_stft with nil output pointer should return MM_ERR_INVALID_INPUT")
    }

    // MARK: - Lifecycle Safety Tests (Bug Fix Verification)

    func testDoubleDestroyNoCrash() {
        // This was the bug: double MetalMomBridge.mm_destroy previously caused use-after-free.
        // Now uses a context registry -- second destroy is a no-op.
        let ctx = MetalMomBridge.mm_init()!
        MetalMomBridge.mm_destroy(ctx)
        MetalMomBridge.mm_destroy(ctx)  // Second destroy should be a safe no-op
        // If we reach here, the test passes (no crash).
    }

    func testDestroyNullIsSafe() {
        MetalMomBridge.mm_destroy(nil)
        // If we reach here, the test passes (no crash).
    }

    func testBufferFreeNullIsSafe() {
        MetalMomBridge.mm_buffer_free(nil)
        // If we reach here, the test passes (no crash).
    }

    func testBufferFreeEmptyBuffer() {
        // Create an MMBuffer with nil data and zero counts, then free it.
        var buf = MMBuffer()
        buf.data = nil
        buf.ndim = 0
        buf.dtype = 0
        buf.count = 0
        MetalMomBridge.mm_buffer_free(&buf)
        // Verify the buffer was cleaned up without crash.
        XCTAssertNil(buf.data, "After freeing, data should be nil")
        XCTAssertEqual(buf.count, 0, "After freeing, count should be 0")
    }

    // MARK: - Parameter Validation Tests

    func testNonPowerOfTwoReturnsError() {
        let ctx = MetalMomBridge.mm_init()!
        defer { MetalMomBridge.mm_destroy(ctx) }

        let signal = makeSineSignal()
        var params = MMSTFTParams(n_fft: 1000, hop_length: 512, win_length: 1000, center: 1)
        var out = MMBuffer()

        let status = MetalMomBridge.mm_stft(ctx, signal, Int64(signal.count), 22050, &params, &out)
        XCTAssertEqual(status, MM_ERR_INVALID_INPUT,
                       "MetalMomBridge.mm_stft with non-power-of-two n_fft should return MM_ERR_INVALID_INPUT")
    }

    // MARK: - Valid Operation Test

    func testValidSTFTReturnsOK() {
        let ctx = MetalMomBridge.mm_init()!
        defer { MetalMomBridge.mm_destroy(ctx) }

        let signal = makeSineSignal()
        var params = makeDefaultParams()
        var out = MMBuffer()

        let status = MetalMomBridge.mm_stft(ctx, signal, Int64(signal.count), 22050, &params, &out)
        XCTAssertEqual(status, MM_OK, "MetalMomBridge.mm_stft with valid params should return MM_OK")

        // Verify output has reasonable shape
        XCTAssertEqual(out.ndim, 2, "STFT output should be 2-dimensional")
        XCTAssertTrue(out.count > 0, "STFT output should have elements")
        XCTAssertNotNil(out.data, "STFT output data should not be nil")

        // Verify first dimension is n_fft/2 + 1 = 1025
        withUnsafePointer(to: &out.shape) { tuplePtr in
            tuplePtr.withMemoryRebound(to: Int64.self, capacity: 8) { shapePtr in
                XCTAssertEqual(shapePtr[0], 1025,
                               "First dimension should be n_fft/2 + 1")
                XCTAssertTrue(shapePtr[1] > 0,
                              "Second dimension (n_frames) should be positive")
            }
        }

        // Clean up the output buffer
        MetalMomBridge.mm_buffer_free(&out)
    }
}
