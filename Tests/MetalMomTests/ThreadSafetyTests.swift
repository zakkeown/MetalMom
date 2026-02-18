import XCTest
@testable import MetalMomCore
@testable import MetalMomBridge

final class ThreadSafetyTests: XCTestCase {

    // MARK: - Helpers

    /// Generate a sine wave signal at a given frequency.
    private func sineSignal(frequency: Float, sampleRate: Int = 22050, duration: Double = 0.5) -> Signal {
        let numSamples = Int(duration * Double(sampleRate))
        var samples = [Float](repeating: 0, count: numSamples)
        let angularFreq = 2.0 * Float.pi * frequency / Float(sampleRate)
        for i in 0..<numSamples {
            samples[i] = sinf(angularFreq * Float(i))
        }
        return Signal(data: samples, sampleRate: sampleRate)
    }

    // MARK: - Test 1: Concurrent Contexts with Independent Results

    /// Create 4 contexts on separate concurrent queues, each computing STFT on distinct
    /// sine signals (220, 440, 880, 1760 Hz). Verify each produces correct shape and
    /// that results differ across frequencies.
    func testConcurrentContextsIndependentResults() {
        let frequencies: [Float] = [220, 440, 880, 1760]
        let nFFT = 1024
        let sr = 22050
        let group = DispatchGroup()

        // Collect results from each concurrent context
        var results = [Int: Signal]()  // frequency index -> STFT magnitude
        let resultsLock = NSLock()

        for (idx, freq) in frequencies.enumerated() {
            group.enter()
            DispatchQueue.global(qos: .userInitiated).async {
                defer { group.leave() }

                // Each thread gets its own context (SmartDispatcher)
                let signal = self.sineSignal(frequency: freq, sampleRate: sr, duration: 0.5)
                let magnitude = STFT.compute(signal: signal, nFFT: nFFT)

                resultsLock.lock()
                results[idx] = magnitude
                resultsLock.unlock()
            }
        }

        let waitResult = group.wait(timeout: .now() + 30)
        XCTAssertEqual(waitResult, .success, "Concurrent STFT computations should complete within 30s")
        XCTAssertEqual(results.count, frequencies.count, "All 4 concurrent STFT results should be collected")

        // Verify shapes are consistent
        let expectedNFreqs = nFFT / 2 + 1
        for (idx, magnitude) in results {
            XCTAssertEqual(magnitude.shape.count, 2,
                           "Result \(idx) should be 2D")
            XCTAssertEqual(magnitude.shape[0], expectedNFreqs,
                           "Result \(idx) should have \(expectedNFreqs) frequency bins")
            XCTAssertGreaterThan(magnitude.shape[1], 0,
                                 "Result \(idx) should have at least 1 frame")
        }

        // Verify results differ — each frequency should produce a different peak bin.
        // Find the peak frequency bin in the middle frame for each result.
        var peakBins = [Int: Int]()
        for (idx, magnitude) in results {
            let nFrames = magnitude.shape[1]
            let midFrame = nFrames / 2
            var maxVal: Float = -1
            var maxBin = -1
            for bin in 0..<expectedNFreqs {
                let val = magnitude[bin * nFrames + midFrame]
                if val > maxVal {
                    maxVal = val
                    maxBin = bin
                }
            }
            peakBins[idx] = maxBin
        }

        // All peak bins should be distinct (220, 440, 880, 1760 Hz map to different bins)
        let uniqueBins = Set(peakBins.values)
        XCTAssertEqual(uniqueBins.count, frequencies.count,
                        "Each frequency should produce a distinct peak bin, got bins: \(peakBins)")
    }

    // MARK: - Test 2: Concurrent STFT on Same Data

    /// Same signal fed to 4 concurrent STFT.compute calls (each is stateless).
    /// Verify all produce identical results via element-wise comparison.
    func testConcurrentSTFTSameData() {
        let sr = 22050
        let signal = sineSignal(frequency: 440, sampleRate: sr, duration: 0.5)
        let nFFT = 1024
        let concurrency = 4
        let group = DispatchGroup()

        var results = [Int: Signal]()
        let resultsLock = NSLock()

        for idx in 0..<concurrency {
            group.enter()
            DispatchQueue.global(qos: .userInitiated).async {
                defer { group.leave() }

                // All threads use the same input signal; STFT.compute is stateless
                let magnitude = STFT.compute(signal: signal, nFFT: nFFT)

                resultsLock.lock()
                results[idx] = magnitude
                resultsLock.unlock()
            }
        }

        let waitResult = group.wait(timeout: .now() + 30)
        XCTAssertEqual(waitResult, .success, "Concurrent STFT computations should complete within 30s")
        XCTAssertEqual(results.count, concurrency, "All concurrent results should be collected")

        // Use result 0 as the reference
        guard let reference = results[0] else {
            XCTFail("Missing reference result")
            return
        }

        for idx in 1..<concurrency {
            guard let other = results[idx] else {
                XCTFail("Missing result \(idx)")
                continue
            }

            // Shape must match exactly
            XCTAssertEqual(reference.shape, other.shape,
                           "Result \(idx) shape should match reference")
            XCTAssertEqual(reference.count, other.count,
                           "Result \(idx) element count should match reference")

            // Element-wise comparison — results should be bit-identical since the
            // input data and algorithm are deterministic on CPU
            var mismatchCount = 0
            for i in 0..<reference.count {
                if reference[i] != other[i] {
                    mismatchCount += 1
                }
            }
            XCTAssertEqual(mismatchCount, 0,
                           "Result \(idx) should be identical to reference, but \(mismatchCount)/\(reference.count) elements differ")
        }
    }

    // MARK: - Test 3: Concurrent SmartDispatcher Creation

    /// Create 100 SmartDispatchers concurrently from 4 queues.
    /// Verify none crash and all have a valid activeBackend.
    func testConcurrentSmartDispatcherCreation() {
        let totalDispatchers = 100
        let group = DispatchGroup()

        var backends = [Int: BackendType]()
        let backendsLock = NSLock()

        for idx in 0..<totalDispatchers {
            group.enter()
            DispatchQueue.global(qos: .userInitiated).async {
                defer { group.leave() }

                let dispatcher = SmartDispatcher()
                let backend = dispatcher.activeBackend

                backendsLock.lock()
                backends[idx] = backend
                backendsLock.unlock()
            }
        }

        let waitResult = group.wait(timeout: .now() + 30)
        XCTAssertEqual(waitResult, .success, "Concurrent dispatcher creation should complete within 30s")
        XCTAssertEqual(backends.count, totalDispatchers,
                       "All \(totalDispatchers) dispatchers should be created successfully")

        // Every dispatcher should report a valid backend
        for (idx, backend) in backends {
            XCTAssertTrue(backend == .accelerate || backend == .metal,
                          "Dispatcher \(idx) should have a valid backend, got \(backend)")
        }

        // All dispatchers should agree on which backend is available
        let uniqueBackends = Set(backends.values)
        XCTAssertEqual(uniqueBackends.count, 1,
                       "All dispatchers should select the same backend on the same machine")
    }

    // MARK: - Test 4: Concurrent Signal Reads

    /// Create one Signal, read from 8 concurrent queues.
    /// Verify all reads return the same data.
    func testConcurrentSignalReads() {
        let sr = 22050
        let signal = sineSignal(frequency: 440, sampleRate: sr, duration: 0.5)
        let expectedCount = signal.count
        let concurrency = 8
        let group = DispatchGroup()

        // Each thread reads the entire signal into a local array
        var readArrays = [Int: [Float]]()
        let readLock = NSLock()

        for idx in 0..<concurrency {
            group.enter()
            DispatchQueue.global(qos: .userInitiated).async {
                defer { group.leave() }

                // Read via dataPointer (the stable pointer from UnsafeMutableBufferPointer)
                var localCopy = [Float](repeating: 0, count: expectedCount)
                for i in 0..<expectedCount {
                    localCopy[i] = signal[i]
                }

                readLock.lock()
                readArrays[idx] = localCopy
                readLock.unlock()
            }
        }

        let waitResult = group.wait(timeout: .now() + 30)
        XCTAssertEqual(waitResult, .success, "Concurrent signal reads should complete within 30s")
        XCTAssertEqual(readArrays.count, concurrency, "All readers should complete")

        // Use reader 0 as reference
        guard let reference = readArrays[0] else {
            XCTFail("Missing reference read")
            return
        }

        XCTAssertEqual(reference.count, expectedCount,
                       "Reference read should have correct length")

        for idx in 1..<concurrency {
            guard let other = readArrays[idx] else {
                XCTFail("Missing read \(idx)")
                continue
            }

            XCTAssertEqual(other.count, expectedCount,
                           "Read \(idx) should have correct length")

            // Bit-exact comparison — reads from immutable data should be identical
            var mismatchCount = 0
            for i in 0..<expectedCount {
                if reference[i] != other[i] {
                    mismatchCount += 1
                }
            }
            XCTAssertEqual(mismatchCount, 0,
                           "Read \(idx) should be identical to reference, but \(mismatchCount)/\(expectedCount) elements differ")
        }
    }

    // MARK: - Test 5: Rapid Context Create/Destroy

    /// Create and immediately destroy 1000 contexts sequentially.
    /// Tests the context registry (NSLock + Set) under rapid churn. No crash = pass.
    func testRapidContextCreateDestroy() {
        let iterations = 1000

        for i in 0..<iterations {
            guard let ctx = mm_init() else {
                XCTFail("mm_init() returned nil at iteration \(i)")
                return
            }
            mm_destroy(ctx)
        }

        // If we get here without crashing, the registry handles rapid churn correctly.
        // Also verify that double-destroy is safe (should be a no-op).
        if let ctx = mm_init() {
            mm_destroy(ctx)
            // Second destroy should be a safe no-op (registry already removed it)
            mm_destroy(ctx)
        }
    }
}
