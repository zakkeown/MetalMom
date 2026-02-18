import XCTest
@testable import MetalMomCore

final class PCENTests: XCTestCase {

    // MARK: - PCEN Output Shape

    func testPCENOutputShapeMatchesInput() {
        // Create a 2D spectrogram [4 bands, 10 frames]
        let nBands = 4
        let nFrames = 10
        var data = [Float](repeating: 0, count: nBands * nFrames)
        for i in 0..<data.count {
            data[i] = Float.random(in: 0.1...10.0)
        }
        let spec = Signal(data: data, shape: [nBands, nFrames], sampleRate: 22050)

        let result = Scaling.pcen(spec)

        XCTAssertEqual(result.shape, [nBands, nFrames], "PCEN output shape should match input")
        XCTAssertEqual(result.count, nBands * nFrames, "PCEN output count should match input")
    }

    // MARK: - PCEN Output is Finite and Non-negative

    func testPCENOutputIsFiniteAndNonNegative() {
        let nBands = 8
        let nFrames = 20
        var data = [Float](repeating: 0, count: nBands * nFrames)
        for i in 0..<data.count {
            data[i] = Float.random(in: 0.01...100.0)
        }
        let spec = Signal(data: data, shape: [nBands, nFrames], sampleRate: 22050)

        let result = Scaling.pcen(spec)

        for i in 0..<result.count {
            XCTAssertTrue(result[i].isFinite, "PCEN output at index \(i) should be finite, got \(result[i])")
            // PCEN with default bias=2, power=0.5:
            // (S/(eps+M)^gain + bias)^power - bias^power
            // Since S >= 0, the term inside is >= bias, so result >= 0
            XCTAssertGreaterThanOrEqual(result[i], 0,
                "PCEN output at index \(i) should be non-negative, got \(result[i])")
        }
    }

    // MARK: - PCEN Reduces Dynamic Range

    func testPCENReducesDynamicRange() {
        // Create a spectrogram where the signal level increases dramatically.
        // PCEN should compress the dynamic range relative to the input.
        let nBands = 1
        let nFrames = 200
        var data = [Float](repeating: 0, count: nBands * nFrames)
        // Ramp from 0.1 up to 1000 — a 10000x dynamic range
        for frame in 0..<nFrames {
            data[frame] = 0.1 + 999.9 * Float(frame) / Float(nFrames - 1)
        }
        let spec = Signal(data: data, shape: [nBands, nFrames], sampleRate: 22050)

        // Compute max/min of input (skip first few frames to avoid IIR transient)
        let skip = 20
        var inputMax: Float = -Float.infinity
        var inputMin: Float = Float.infinity
        for i in skip..<spec.count {
            if spec[i] > inputMax { inputMax = spec[i] }
            if spec[i] < inputMin { inputMin = spec[i] }
        }
        let inputRatio = inputMax / inputMin

        // Apply PCEN
        let result = Scaling.pcen(spec)

        // Compute max/min of output (skip transient)
        var outputMax: Float = -Float.infinity
        var outputMin: Float = Float.infinity
        for i in skip..<result.count {
            if result[i] > outputMax { outputMax = result[i] }
            if result[i] > 1e-10 && result[i] < outputMin { outputMin = result[i] }
        }
        let outputRatio = outputMax / max(outputMin, 1e-10)

        // PCEN with adaptive gain should compress the ramp significantly
        XCTAssertLessThan(outputRatio, inputRatio,
            "PCEN should reduce dynamic range: input ratio=\(inputRatio), output ratio=\(outputRatio)")
    }

    // MARK: - PCEN with gain=0, power=1 approximates identity minus bias offset

    func testPCENGainZeroPowerOne() {
        // With gain=0: denominator = (eps + M)^0 = 1
        // With power=1: (S/1 + bias)^1 - bias^1 = S + bias - bias = S
        let nBands = 2
        let nFrames = 10
        var data = [Float](repeating: 0, count: nBands * nFrames)
        for i in 0..<data.count {
            data[i] = Float.random(in: 1.0...10.0)
        }
        let spec = Signal(data: data, shape: [nBands, nFrames], sampleRate: 22050)

        let result = Scaling.pcen(spec, gain: 0.0, bias: 2.0, power: 1.0, eps: 1e-6)

        // With gain=0, power=1: PCEN(S) = (S + bias) - bias = S
        for i in 0..<result.count {
            XCTAssertEqual(result[i], spec[i], accuracy: 1e-3,
                "PCEN with gain=0, power=1 should approximate identity at index \(i): got \(result[i]), expected \(spec[i])")
        }
    }

    // MARK: - PCEN Processes Each Channel Independently

    func testPCENChannelIndependence() {
        let nFrames = 20

        // Create two different bands
        let data1 = [Float](repeating: 5.0, count: nFrames)
        let data2 = [Float](repeating: 50.0, count: nFrames)

        // Process combined
        let combined = data1 + data2
        let specCombined = Signal(data: combined, shape: [2, nFrames], sampleRate: 22050)
        let resultCombined = Scaling.pcen(specCombined)

        // Process individually
        let spec1 = Signal(data: data1, shape: [1, nFrames], sampleRate: 22050)
        let result1 = Scaling.pcen(spec1)

        let spec2 = Signal(data: data2, shape: [1, nFrames], sampleRate: 22050)
        let result2 = Scaling.pcen(spec2)

        // Results should match
        for frame in 0..<nFrames {
            XCTAssertEqual(resultCombined[frame], result1[frame], accuracy: 1e-5,
                "Band 0 should match individual processing at frame \(frame)")
            XCTAssertEqual(resultCombined[nFrames + frame], result2[frame], accuracy: 1e-5,
                "Band 1 should match individual processing at frame \(frame)")
        }
    }

    // MARK: - A-Weighting Tests

    func testAWeightingAt1000Hz() {
        // A-weighting at 1000 Hz should be approximately 0 dB (reference point)
        let weights = Scaling.aWeighting(frequencies: [1000.0])
        XCTAssertEqual(weights[0], 0.0, accuracy: 0.1,
            "A-weighting at 1000 Hz should be ~0 dB, got \(weights[0])")
    }

    func testAWeightingAt100Hz() {
        // A-weighting at 100 Hz should be approximately -19.1 dB
        let weights = Scaling.aWeighting(frequencies: [100.0])
        XCTAssertEqual(weights[0], -19.1, accuracy: 1.0,
            "A-weighting at 100 Hz should be ~-19 dB, got \(weights[0])")
    }

    func testAWeightingAt10000Hz() {
        // A-weighting at 10000 Hz should be approximately -2.5 dB (rises then falls)
        let weights = Scaling.aWeighting(frequencies: [10000.0])
        // The actual value at 10 kHz is about -2.5 dB
        XCTAssertEqual(weights[0], -2.5, accuracy: 3.0,
            "A-weighting at 10000 Hz should be near +2 to -3 dB, got \(weights[0])")
    }

    // MARK: - C-Weighting is Flatter than A-Weighting

    func testCWeightingFlatterThanAWeighting() {
        // C-weighting should be flatter (closer to 0) than A-weighting across frequencies
        let frequencies: [Float] = [50, 100, 200, 500, 1000, 2000, 5000, 10000]
        let aWeights = Scaling.aWeighting(frequencies: frequencies)
        let cWeights = Scaling.cWeighting(frequencies: frequencies)

        // Sum of absolute deviations from 0 should be smaller for C than A
        var aDeviation: Float = 0
        var cDeviation: Float = 0
        for i in 0..<frequencies.count {
            aDeviation += abs(aWeights[i])
            cDeviation += abs(cWeights[i])
        }

        XCTAssertLessThan(cDeviation, aDeviation,
            "C-weighting should be flatter than A-weighting: C deviation=\(cDeviation), A deviation=\(aDeviation)")
    }

    // MARK: - B and D Weighting Produce Finite Results

    func testBWeightingProducesFiniteResults() {
        let frequencies: [Float] = [100, 500, 1000, 2000, 5000, 10000]
        let weights = Scaling.bWeighting(frequencies: frequencies)

        for i in 0..<frequencies.count {
            XCTAssertTrue(weights[i].isFinite,
                "B-weighting at \(frequencies[i]) Hz should be finite, got \(weights[i])")
        }
    }

    func testDWeightingProducesFiniteResults() {
        let frequencies: [Float] = [100, 500, 1000, 2000, 5000, 10000]
        let weights = Scaling.dWeighting(frequencies: frequencies)

        for i in 0..<frequencies.count {
            XCTAssertTrue(weights[i].isFinite,
                "D-weighting at \(frequencies[i]) Hz should be finite, got \(weights[i])")
        }
    }

    // MARK: - Weighting Curves Array Length

    func testWeightingCurvesArrayLength() {
        let frequencies: [Float] = [100, 500, 1000, 5000, 10000]
        let aW = Scaling.aWeighting(frequencies: frequencies)
        let bW = Scaling.bWeighting(frequencies: frequencies)
        let cW = Scaling.cWeighting(frequencies: frequencies)
        let dW = Scaling.dWeighting(frequencies: frequencies)

        XCTAssertEqual(aW.count, frequencies.count, "A-weighting output length should match input")
        XCTAssertEqual(bW.count, frequencies.count, "B-weighting output length should match input")
        XCTAssertEqual(cW.count, frequencies.count, "C-weighting output length should match input")
        XCTAssertEqual(dW.count, frequencies.count, "D-weighting output length should match input")
    }

    // MARK: - Apply A-Weighting to Spectrogram

    func testApplyAWeighting() {
        // Create a simple spectrogram [5 freq bins, 3 frames]
        let nFreqs = 5
        let nFrames = 3
        let data = [Float](repeating: 0.0, count: nFreqs * nFrames)  // All zeros (dB)
        let spec = Signal(data: data, shape: [nFreqs, nFrames], sampleRate: 22050)

        let result = Scaling.applyAWeighting(spec, sr: 22050, nFFT: 8)

        XCTAssertEqual(result.shape, [nFreqs, nFrames], "Output shape should match input")
        XCTAssertEqual(result.count, nFreqs * nFrames, "Output count should match input")

        // Since input is all zeros (dB), output should equal the A-weighting values
        // repeated across frames
        for i in 0..<result.count {
            XCTAssertTrue(result[i].isFinite || result[i] == -Float.infinity,
                "Apply A-weighting result should be finite or -inf at index \(i)")
        }
    }

    // MARK: - PCEN Empty Input

    func testPCENEmptyInput() {
        let spec = Signal(data: [], shape: [0, 0], sampleRate: 22050)
        let result = Scaling.pcen(spec)
        XCTAssertEqual(result.count, 0, "PCEN of empty input should be empty")
    }

    // MARK: - A-Weighting Monotonic Below 2kHz

    func testAWeightingMonotonicBelow2kHz() {
        // A-weighting should be monotonically increasing from ~20 Hz to ~2 kHz
        let frequencies: [Float] = [20, 50, 100, 200, 500, 1000, 2000]
        let weights = Scaling.aWeighting(frequencies: frequencies)

        for i in 1..<weights.count {
            XCTAssertGreaterThan(weights[i], weights[i - 1],
                "A-weighting should increase from \(frequencies[i-1]) to \(frequencies[i]) Hz: \(weights[i-1]) to \(weights[i])")
        }
    }
}
