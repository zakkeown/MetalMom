import XCTest
@testable import MetalMomCore

final class MetalSTFTTests: XCTestCase {

    // MARK: - GPU vs CPU Parity

    func testGPUSTFTProducesSameOutputAsCPU() throws {
        guard MetalBackend.shared != nil else {
            throw XCTSkip("Metal not available")
        }

        // Generate a 440 Hz sine wave, 1 second at 22050 Hz
        let sr = 22050
        let duration = 1.0
        let nSamples = Int(Double(sr) * duration)
        var signalData = [Float](repeating: 0, count: nSamples)
        for i in 0..<nSamples {
            signalData[i] = sin(2 * .pi * 440.0 * Float(i) / Float(sr))
        }
        let sig = Signal(data: signalData, sampleRate: sr)

        let input = STFTInput(signal: sig, nFFT: 2048, hopLength: 512, winLength: 2048, center: true)
        let stft = STFT()

        let cpuResult = stft.executeCPU(input)
        let gpuResult = stft.executeGPU(input)

        // Compare shapes
        XCTAssertEqual(cpuResult.magnitude.shape, gpuResult.magnitude.shape,
                       "Shape mismatch: cpu=\(cpuResult.magnitude.shape) gpu=\(gpuResult.magnitude.shape)")

        // Compare values with tolerance for float32 FFT differences
        let cpuData = cpuResult.magnitude.withUnsafeBufferPointer { Array($0) }
        let gpuData = gpuResult.magnitude.withUnsafeBufferPointer { Array($0) }

        XCTAssertEqual(cpuData.count, gpuData.count)

        var maxDiff: Float = 0
        var maxDiffIdx = 0
        for i in 0..<cpuData.count {
            let diff = abs(cpuData[i] - gpuData[i])
            if diff > maxDiff {
                maxDiff = diff
                maxDiffIdx = i
            }
        }

        // Allow small tolerance due to different FFT implementations (vDSP vs MPSGraph)
        XCTAssertLessThan(maxDiff, 1e-2,
            "Max difference \(maxDiff) at index \(maxDiffIdx): cpu=\(cpuData[maxDiffIdx]) gpu=\(gpuData[maxDiffIdx])")

        // Also check relative error for bins with significant magnitude.
        // Near-zero bins can have very high relative error due to floating-point
        // differences between vDSP and MPSGraph FFT implementations, so we only
        // check bins above a meaningful threshold.
        var maxRelError: Float = 0
        var relErrorCount = 0
        let magnitudeThreshold: Float = 1.0  // Only check bins with meaningful energy
        for i in 0..<cpuData.count {
            let cpuVal = cpuData[i]
            let gpuVal = gpuData[i]
            if cpuVal > magnitudeThreshold {
                let relErr = abs(cpuVal - gpuVal) / cpuVal
                if relErr > maxRelError {
                    maxRelError = relErr
                }
                relErrorCount += 1
            }
        }
        if relErrorCount > 0 {
            XCTAssertLessThan(maxRelError, 1e-3,
                "Max relative error \(maxRelError) over \(relErrorCount) significant bins")
        }
    }

    func testGPUSTFTSmallSignal() throws {
        guard MetalBackend.shared != nil else {
            throw XCTSkip("Metal not available")
        }

        // Very small signal — exercises edge case
        let sig = Signal(data: [Float](repeating: 0.5, count: 4096), sampleRate: 22050)
        let input = STFTInput(signal: sig, nFFT: 1024, hopLength: 256, winLength: 1024, center: true)
        let stft = STFT()

        let cpuResult = stft.executeCPU(input)
        let gpuResult = stft.executeGPU(input)

        XCTAssertEqual(cpuResult.magnitude.shape, gpuResult.magnitude.shape)

        let cpuData = cpuResult.magnitude.withUnsafeBufferPointer { Array($0) }
        let gpuData = gpuResult.magnitude.withUnsafeBufferPointer { Array($0) }

        for i in 0..<cpuData.count {
            XCTAssertEqual(cpuData[i], gpuData[i], accuracy: 1e-2,
                "Mismatch at index \(i): cpu=\(cpuData[i]) gpu=\(gpuData[i])")
        }
    }

    // MARK: - Shape Tests

    func testGPUSTFTShape() throws {
        guard MetalBackend.shared != nil else {
            throw XCTSkip("Metal not available")
        }

        let sig = Signal(data: [Float](repeating: 0.5, count: 22050), sampleRate: 22050)
        let input = STFTInput(signal: sig, nFFT: 1024, hopLength: 256, winLength: 1024, center: true)
        let result = STFT().executeGPU(input)

        let nFreqs = 1024 / 2 + 1  // 513
        XCTAssertEqual(result.magnitude.shape[0], nFreqs)
        XCTAssertGreaterThan(result.magnitude.shape[1], 0)
    }

    func testGPUSTFTEmptySignal() throws {
        guard MetalBackend.shared != nil else {
            throw XCTSkip("Metal not available")
        }

        // Signal shorter than nFFT should produce empty output
        let sig = Signal(data: [Float](repeating: 0, count: 100), sampleRate: 22050)
        let input = STFTInput(signal: sig, nFFT: 2048, hopLength: 512, winLength: 2048, center: false)
        let result = STFT().executeGPU(input)

        let nFreqs = 2048 / 2 + 1
        XCTAssertEqual(result.magnitude.shape[0], nFreqs)
        XCTAssertEqual(result.magnitude.shape[1], 0)
    }

    // MARK: - Dispatch Threshold

    func testDispatchThresholdUsesChipProfile() {
        if MetalBackend.shared != nil {
            XCTAssertLessThan(STFT.dispatchThreshold, Int.max,
                "With Metal available, threshold should be less than Int.max")
        } else {
            XCTAssertEqual(STFT.dispatchThreshold, Int.max,
                "Without Metal, threshold should be Int.max")
        }
    }

    // MARK: - Multi-frequency Parity

    func testGPUSTFTMultiFrequencyParity() throws {
        guard MetalBackend.shared != nil else {
            throw XCTSkip("Metal not available")
        }

        // Multi-frequency signal: 440 Hz + 1000 Hz + 2000 Hz
        let sr = 22050
        let nSamples = sr  // 1 second
        var signalData = [Float](repeating: 0, count: nSamples)
        for i in 0..<nSamples {
            let t = Float(i) / Float(sr)
            signalData[i] = sin(2 * .pi * 440.0 * t)
                          + 0.5 * sin(2 * .pi * 1000.0 * t)
                          + 0.3 * sin(2 * .pi * 2000.0 * t)
        }
        let sig = Signal(data: signalData, sampleRate: sr)

        let input = STFTInput(signal: sig, nFFT: 2048, hopLength: 512, winLength: 2048, center: true)
        let stft = STFT()

        let cpuResult = stft.executeCPU(input)
        let gpuResult = stft.executeGPU(input)

        XCTAssertEqual(cpuResult.magnitude.shape, gpuResult.magnitude.shape)

        let cpuData = cpuResult.magnitude.withUnsafeBufferPointer { Array($0) }
        let gpuData = gpuResult.magnitude.withUnsafeBufferPointer { Array($0) }

        var maxDiff: Float = 0
        for i in 0..<cpuData.count {
            let diff = abs(cpuData[i] - gpuData[i])
            if diff > maxDiff { maxDiff = diff }
        }
        XCTAssertLessThan(maxDiff, 1e-2,
            "Multi-frequency max difference \(maxDiff)")
    }

    // MARK: - Non-centered STFT

    func testGPUSTFTNonCentered() throws {
        guard MetalBackend.shared != nil else {
            throw XCTSkip("Metal not available")
        }

        let sr = 22050
        var signalData = [Float](repeating: 0, count: sr)
        for i in 0..<sr {
            signalData[i] = sin(2 * .pi * 440.0 * Float(i) / Float(sr))
        }
        let sig = Signal(data: signalData, sampleRate: sr)

        let input = STFTInput(signal: sig, nFFT: 2048, hopLength: 512, winLength: 2048, center: false)
        let stft = STFT()

        let cpuResult = stft.executeCPU(input)
        let gpuResult = stft.executeGPU(input)

        XCTAssertEqual(cpuResult.magnitude.shape, gpuResult.magnitude.shape)

        let cpuData = cpuResult.magnitude.withUnsafeBufferPointer { Array($0) }
        let gpuData = gpuResult.magnitude.withUnsafeBufferPointer { Array($0) }

        var maxDiff: Float = 0
        for i in 0..<cpuData.count {
            let diff = abs(cpuData[i] - gpuData[i])
            if diff > maxDiff { maxDiff = diff }
        }
        XCTAssertLessThan(maxDiff, 1e-2)
    }
}
