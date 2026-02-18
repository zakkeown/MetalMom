import XCTest
@testable import MetalMomCore

final class NNFilterTests: XCTestCase {

    // MARK: - Output Shape Matches Input

    func testOutputShapeMatchesInput() {
        let nFeatures = 8
        let nFrames = 20
        let data = makeSpectrogram(rows: nFeatures, cols: nFrames)
        let S = Signal(data: data, shape: [nFeatures, nFrames], sampleRate: 22050)

        let result = NNFilter.filter(S, k: 5)

        XCTAssertEqual(result.shape, [nFeatures, nFrames],
            "Output shape should match input shape")
        XCTAssertEqual(result.count, nFeatures * nFrames)
    }

    // MARK: - Output Values Finite and Non-Negative

    func testOutputFiniteAndNonNegative() {
        let nFeatures = 10
        let nFrames = 15
        let data = makeSpectrogram(rows: nFeatures, cols: nFrames)
        let S = Signal(data: data, shape: [nFeatures, nFrames], sampleRate: 22050)

        let result = NNFilter.filter(S, k: 5)

        for i in 0..<result.count {
            XCTAssertTrue(result[i].isFinite,
                "Output[\(i)] should be finite, got \(result[i])")
            XCTAssertGreaterThanOrEqual(result[i], 0,
                "Output[\(i)] should be non-negative for non-negative input")
        }
    }

    // MARK: - Filtered Output Is Smoother

    func testFilteredOutputIsSmoother() {
        let nFeatures = 8
        let nFrames = 30
        // Create noisy spectrogram with some variation
        var data = [Float](repeating: 0, count: nFeatures * nFrames)
        var state: UInt64 = 42
        for i in 0..<data.count {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            let u = Float(state >> 33) / Float(1 << 31)
            data[i] = u + 0.5  // [0.5, 1.5] range
        }
        let S = Signal(data: data, shape: [nFeatures, nFrames], sampleRate: 22050)

        let result = NNFilter.filter(S, k: 10)

        // Compute frame-to-frame variance for input and output
        let inputVar = frameToFrameVariance(data, nFeatures: nFeatures, nFrames: nFrames)
        var resultData = [Float](repeating: 0, count: result.count)
        result.withUnsafeBufferPointer { buf in
            for i in 0..<result.count { resultData[i] = buf[i] }
        }
        let outputVar = frameToFrameVariance(resultData, nFeatures: nFeatures, nFrames: nFrames)

        XCTAssertLessThan(outputVar, inputVar,
            "Filtered output should be smoother: output var \(outputVar) vs input var \(inputVar)")
    }

    // MARK: - k=1 Without ExcludeSelf Returns Self

    func testK1WithoutExcludeSelfReturnsSelf() {
        let nFeatures = 5
        let nFrames = 10
        let data = makeSpectrogram(rows: nFeatures, cols: nFrames)
        let S = Signal(data: data, shape: [nFeatures, nFrames], sampleRate: 22050)

        // k=1, excludeSelf=false: each frame's nearest neighbor is itself (distance 0)
        let result = NNFilter.filter(S, k: 1, excludeSelf: false)

        S.withUnsafeBufferPointer { sBuf in
            result.withUnsafeBufferPointer { rBuf in
                for i in 0..<result.count {
                    XCTAssertEqual(rBuf[i], sBuf[i], accuracy: 1e-6,
                        "k=1 without excludeSelf should return self at index \(i)")
                }
            }
        }
    }

    // MARK: - Cosine vs Euclidean Both Valid

    func testCosineVsEuclideanBothValid() {
        let nFeatures = 8
        let nFrames = 15
        let data = makeSpectrogram(rows: nFeatures, cols: nFrames)
        let S = Signal(data: data, shape: [nFeatures, nFrames], sampleRate: 22050)

        let resultCosine = NNFilter.filter(S, k: 5, metric: .cosine)
        let resultEuclidean = NNFilter.filter(S, k: 5, metric: .euclidean)

        XCTAssertEqual(resultCosine.shape, [nFeatures, nFrames])
        XCTAssertEqual(resultEuclidean.shape, [nFeatures, nFrames])

        // Both should produce finite, non-negative values
        for i in 0..<resultCosine.count {
            XCTAssertTrue(resultCosine[i].isFinite, "Cosine result[\(i)] not finite")
            XCTAssertTrue(resultEuclidean[i].isFinite, "Euclidean result[\(i)] not finite")
            XCTAssertGreaterThanOrEqual(resultCosine[i], 0)
            XCTAssertGreaterThanOrEqual(resultEuclidean[i], 0)
        }

        // They should generally differ (different metrics find different neighbors)
        var differ = false
        resultCosine.withUnsafeBufferPointer { cBuf in
            resultEuclidean.withUnsafeBufferPointer { eBuf in
                for i in 0..<resultCosine.count {
                    if abs(cBuf[i] - eBuf[i]) > 1e-4 {
                        differ = true
                        break
                    }
                }
            }
        }
        XCTAssertTrue(differ, "Cosine and Euclidean metrics should generally produce different results")
    }

    // MARK: - Mean vs Median Both Work

    func testMeanVsMedianBothWork() {
        let nFeatures = 6
        let nFrames = 12
        let data = makeSpectrogram(rows: nFeatures, cols: nFrames)
        let S = Signal(data: data, shape: [nFeatures, nFrames], sampleRate: 22050)

        let resultMean = NNFilter.filter(S, k: 5, aggregate: .mean)
        let resultMedian = NNFilter.filter(S, k: 5, aggregate: .median)

        XCTAssertEqual(resultMean.shape, [nFeatures, nFrames])
        XCTAssertEqual(resultMedian.shape, [nFeatures, nFrames])

        for i in 0..<resultMean.count {
            XCTAssertTrue(resultMean[i].isFinite, "Mean result[\(i)] not finite")
            XCTAssertTrue(resultMedian[i].isFinite, "Median result[\(i)] not finite")
            XCTAssertGreaterThanOrEqual(resultMean[i], 0)
            XCTAssertGreaterThanOrEqual(resultMedian[i], 0)
        }
    }

    // MARK: - Repeating Pattern Preserved

    func testRepeatingPatternPreserved() {
        // Create a spectrogram with a repeating 4-frame pattern
        let nFeatures = 4
        let nFrames = 20
        let pattern: [Float] = [1, 0.5, 0.2, 0.8,   // frame 0
                                 0.3, 1, 0.6, 0.1,   // frame 1
                                 0.7, 0.2, 1, 0.4,   // frame 2
                                 0.5, 0.8, 0.3, 1]   // frame 3
        var data = [Float](repeating: 0, count: nFeatures * nFrames)
        // Data is [nFeatures, nFrames] row-major: S[f, t] = data[f * nFrames + t]
        for t in 0..<nFrames {
            let patternFrame = t % 4
            for f in 0..<nFeatures {
                data[f * nFrames + t] = pattern[patternFrame * nFeatures + f]
            }
        }

        let S = Signal(data: data, shape: [nFeatures, nFrames], sampleRate: 22050)
        let result = NNFilter.filter(S, k: 4, metric: .euclidean)

        // For a perfectly repeating pattern, filtering should return values close to the pattern
        // because the nearest neighbors are the identical repeating frames.
        // Check that the output preserves the pattern structure
        var maxDiff: Float = 0
        result.withUnsafeBufferPointer { buf in
            for t in 0..<nFrames {
                let patternFrame = t % 4
                for f in 0..<nFeatures {
                    let expected = pattern[patternFrame * nFeatures + f]
                    let actual = buf[f * nFrames + t]
                    let diff = abs(actual - expected)
                    if diff > maxDiff { maxDiff = diff }
                }
            }
        }

        // The neighbors of a repeating frame should be the other copies of itself,
        // so the average should be very close to the original value
        XCTAssertLessThan(maxDiff, 0.15,
            "Repeating pattern should be well-preserved, max diff: \(maxDiff)")
    }

    // MARK: - Edge Case: k >= nFrames (Clamps)

    func testKLargerThanNFrames() {
        let nFeatures = 4
        let nFrames = 5
        let data = makeSpectrogram(rows: nFeatures, cols: nFrames)
        let S = Signal(data: data, shape: [nFeatures, nFrames], sampleRate: 22050)

        // k=100 >> nFrames=5; should clamp gracefully
        let result = NNFilter.filter(S, k: 100)

        XCTAssertEqual(result.shape, [nFeatures, nFrames])
        for i in 0..<result.count {
            XCTAssertTrue(result[i].isFinite,
                "Result should be finite even with k > nFrames")
        }
    }

    // MARK: - Edge Case: Single Frame

    func testSingleFrame() {
        let nFeatures = 6
        let data: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        let S = Signal(data: data, shape: [nFeatures, 1], sampleRate: 22050)

        let result = NNFilter.filter(S, k: 5)

        XCTAssertEqual(result.shape, [nFeatures, 1])
        // Single frame: output should equal input (only neighbor is self)
        result.withUnsafeBufferPointer { buf in
            for i in 0..<nFeatures {
                XCTAssertEqual(buf[i], data[i], accuracy: 1e-6,
                    "Single frame should return self at index \(i)")
            }
        }
    }

    // MARK: - NearestNeighborIndices API

    func testNearestNeighborIndices() {
        let nFeatures = 4
        let nFrames = 8
        let k = 3
        let data = makeSpectrogram(rows: nFeatures, cols: nFrames)
        let S = Signal(data: data, shape: [nFeatures, nFrames], sampleRate: 22050)

        let indices = NNFilter.nearestNeighborIndices(S, k: k)

        XCTAssertEqual(indices.count, nFrames, "Should have one entry per frame")
        for t in 0..<nFrames {
            XCTAssertEqual(indices[t].count, k,
                "Frame \(t) should have \(k) neighbors")
            // All indices should be valid frame indices
            for idx in indices[t] {
                XCTAssertGreaterThanOrEqual(idx, 0)
                XCTAssertLessThan(idx, nFrames)
            }
            // With excludeSelf=true (default), self should not be in neighbors
            XCTAssertFalse(indices[t].contains(t),
                "Frame \(t) should not be in its own neighbor list (excludeSelf=true)")
        }
    }

    // MARK: - ExcludeSelf=false Includes Self

    func testExcludeSelfFalseIncludesSelf() {
        // Create spectrogram where each frame is unique but distinguishable
        let nFeatures = 3
        let nFrames = 5
        let data = makeSpectrogram(rows: nFeatures, cols: nFrames)
        let S = Signal(data: data, shape: [nFeatures, nFrames], sampleRate: 22050)

        let indices = NNFilter.nearestNeighborIndices(S, k: 1, excludeSelf: false)

        // When not excluding self, the nearest neighbor should be self (distance 0)
        for t in 0..<nFrames {
            XCTAssertEqual(indices[t].count, 1)
            XCTAssertEqual(indices[t][0], t,
                "With excludeSelf=false, k=1 nearest neighbor of frame \(t) should be itself")
        }
    }

    // MARK: - Helpers

    /// Create a non-negative spectrogram-like matrix.
    private func makeSpectrogram(rows: Int, cols: Int, seed: UInt64 = 42) -> [Float] {
        var state = seed
        var data = [Float](repeating: 0, count: rows * cols)
        for i in 0..<data.count {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            let u = Float(state >> 33) / Float(1 << 31)
            data[i] = u + 0.01  // ensure strictly positive
        }
        return data
    }

    /// Compute average frame-to-frame squared difference.
    private func frameToFrameVariance(_ data: [Float], nFeatures: Int, nFrames: Int) -> Float {
        guard nFrames > 1 else { return 0 }
        var totalVar: Float = 0
        for t in 1..<nFrames {
            for f in 0..<nFeatures {
                let diff = data[f * nFrames + t] - data[f * nFrames + (t - 1)]
                totalVar += diff * diff
            }
        }
        return totalVar / Float((nFrames - 1) * nFeatures)
    }
}
