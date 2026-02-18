import XCTest
@testable import MetalMomCore
@testable import MetalMomBridge
import MetalMomCBridge

final class MemoryStressTests: XCTestCase {

    // MARK: - Helpers

    /// Return the current resident memory of this process in megabytes.
    private func currentMemoryMB() -> Float {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        if result == KERN_SUCCESS {
            return Float(info.resident_size) / (1024 * 1024)
        }
        return 0
    }

    /// Generate a 1-second 440 Hz sine wave at 22050 Hz.
    private func makeOneSecondSine(sampleRate: Int = 22050) -> [Float] {
        (0..<sampleRate).map { sinf(Float($0) * 2 * .pi * 440 / Float(sampleRate)) }
    }

    // MARK: - Test 1: Repeated STFT — No Leak

    /// Run STFT.compute 1000x on a 1-second signal. Measure memory before/after via
    /// mach_task_basic_info. Verify memory growth < 10 MB.
    /// This exercises Signal's UnsafeMutableBufferPointer dealloc path repeatedly.
    func testRepeatedSTFTNoLeak() {
        let sr = 22050
        let samples = makeOneSecondSine(sampleRate: sr)

        // Warm up — let any one-time allocations settle
        autoreleasepool {
            let warmup = Signal(data: samples, sampleRate: sr)
            _ = STFT.compute(signal: warmup, nFFT: 2048, hopLength: 512)
        }

        let memBefore = currentMemoryMB()

        for _ in 0..<1000 {
            autoreleasepool {
                let signal = Signal(data: samples, sampleRate: sr)
                let result = STFT.compute(signal: signal, nFFT: 2048, hopLength: 512)
                // Access a value to prevent dead-code elimination
                XCTAssertGreaterThan(result.count, 0)
            }
        }

        let memAfter = currentMemoryMB()
        let growth = memAfter - memBefore

        XCTAssertLessThan(growth, 10.0,
            "Memory grew by \(growth) MB over 1000 STFT iterations — possible leak "
            + "(before: \(memBefore) MB, after: \(memAfter) MB)")
    }

    // MARK: - Test 2: Bridge Call Cycle 500x

    /// Run 500 cycles of: mm_init -> mm_stft -> mm_buffer_free -> mm_destroy.
    /// Verify memory growth < 10 MB. This exercises the full C-bridge lifecycle:
    /// context creation, buffer allocation in mm_stft, buffer deallocation, context teardown.
    func testBridgeCallCycle500x() {
        let signal = makeOneSecondSine()
        var params = MMSTFTParams(n_fft: 2048, hop_length: 512, win_length: 2048, center: 1)

        // Warm up
        if let ctx = MetalMomBridge.mm_init() {
            var out = MMBuffer()
            _ = MetalMomBridge.mm_stft(ctx, signal, Int64(signal.count), 22050, &params, &out)
            MetalMomBridge.mm_buffer_free(&out)
            MetalMomBridge.mm_destroy(ctx)
        }

        let memBefore = currentMemoryMB()

        for _ in 0..<500 {
            autoreleasepool {
                guard let ctx = MetalMomBridge.mm_init() else {
                    XCTFail("mm_init returned nil")
                    return
                }

                var out = MMBuffer()
                let status = MetalMomBridge.mm_stft(
                    ctx, signal, Int64(signal.count), 22050, &params, &out
                )
                XCTAssertEqual(status, MM_OK)
                XCTAssertGreaterThan(out.count, 0)

                MetalMomBridge.mm_buffer_free(&out)
                MetalMomBridge.mm_destroy(ctx)
            }
        }

        let memAfter = currentMemoryMB()
        let growth = memAfter - memBefore

        XCTAssertLessThan(growth, 10.0,
            "Memory grew by \(growth) MB over 500 bridge cycles — possible leak "
            + "(before: \(memBefore) MB, after: \(memAfter) MB)")
    }

    // MARK: - Test 3: Long Signal STFT

    /// Create a 10-minute signal (22050 * 600 = 13.2M samples). Run STFT.compute once.
    /// Verify it completes without crash and produces valid output (correct shape, non-empty).
    func testLongSignalSTFT() {
        let sr = 22050
        let duration = 600 // 10 minutes in seconds
        let numSamples = sr * duration // 13,230,000 samples

        // Generate 10 minutes of 440 Hz sine
        var samples = [Float](repeating: 0, count: numSamples)
        let angularFreq = 2.0 * Float.pi * 440.0 / Float(sr)
        for i in 0..<numSamples {
            samples[i] = sinf(angularFreq * Float(i))
        }

        let signal = Signal(data: samples, sampleRate: sr)
        let nFFT = 2048
        let hopLength = 512

        let result = STFT.compute(signal: signal, nFFT: nFFT, hopLength: hopLength)

        // Verify shape
        let expectedNFreqs = nFFT / 2 + 1 // 1025
        XCTAssertEqual(result.shape.count, 2, "STFT output should be 2D")
        XCTAssertEqual(result.shape[0], expectedNFreqs,
            "Expected \(expectedNFreqs) frequency bins, got \(result.shape[0])")

        // With center=true, paddedLength = numSamples + nFFT
        let paddedLength = numSamples + nFFT
        let expectedFrames = 1 + (paddedLength - nFFT) / hopLength
        XCTAssertEqual(result.shape[1], expectedFrames,
            "Expected \(expectedFrames) frames, got \(result.shape[1])")

        // Verify data is non-trivial (not all zeros)
        let midFrame = result.shape[1] / 2
        let nFrames = result.shape[1]
        var maxVal: Float = 0
        for bin in 0..<expectedNFreqs {
            let val = result[bin * nFrames + midFrame]
            if val > maxVal {
                maxVal = val
            }
        }
        XCTAssertGreaterThan(maxVal, 0.0,
            "10-minute STFT should have non-zero energy in the middle frame")
    }

    // MARK: - Test 4: Rapid Context Churn 10000x

    /// Create and destroy 10,000 contexts sequentially (mm_init / mm_destroy).
    /// Verify memory growth < 5 MB. This ensures that context lifecycle (Unmanaged
    /// passRetained/release) does not leak the MMContextInternal or its SmartDispatcher.
    func testRapidContextChurn10000() {
        // Warm up
        if let ctx = MetalMomBridge.mm_init() {
            MetalMomBridge.mm_destroy(ctx)
        }

        let memBefore = currentMemoryMB()

        for _ in 0..<10_000 {
            autoreleasepool {
                guard let ctx = MetalMomBridge.mm_init() else {
                    XCTFail("mm_init returned nil during context churn")
                    return
                }
                MetalMomBridge.mm_destroy(ctx)
            }
        }

        let memAfter = currentMemoryMB()
        let growth = memAfter - memBefore

        XCTAssertLessThan(growth, 5.0,
            "Memory grew by \(growth) MB over 10,000 context create/destroy cycles — possible leak "
            + "(before: \(memBefore) MB, after: \(memAfter) MB)")
    }
}
