import XCTest
@testable import MetalMomCore

final class DTWTests: XCTestCase {

    // MARK: - Accumulated Cost Matrix Shape

    func testAccumulatedCostShape() {
        let n = 5
        let m = 7
        let cost = makeConstantCost(n: n, m: m, value: 1.0)
        let result = DTW.compute(costMatrix: cost)

        XCTAssertEqual(result.accumulatedCost.shape, [n, m],
            "Accumulated cost should be [N, M]")
        XCTAssertEqual(result.accumulatedCost.count, n * m)
    }

    // MARK: - Identical Sequences Have Zero Total Cost

    func testIdenticalSequencesZeroCost() {
        let nFeatures = 4
        let nFrames = 10
        let data = makeFeatures(nFeatures: nFeatures, nFrames: nFrames)
        let X = Signal(data: data, shape: [nFeatures, nFrames], sampleRate: 22050)
        let Y = Signal(data: data, shape: [nFeatures, nFrames], sampleRate: 22050)

        let result = DTW.compute(X: X, Y: Y)

        // Small floating-point error from BLAS Euclidean distance (||x-x||^2 via gram matrix)
        XCTAssertEqual(result.totalCost, 0.0, accuracy: 1e-2,
            "Identical sequences should have near-zero DTW cost")
    }

    // MARK: - Warping Path Endpoints

    func testWarpingPathEndpoints() {
        let n = 6
        let m = 8
        let cost = makeRandomCost(n: n, m: m)
        let result = DTW.compute(costMatrix: cost)

        XCTAssertFalse(result.warpingPath.isEmpty, "Warping path should not be empty")

        let first = result.warpingPath.first!
        let last = result.warpingPath.last!

        XCTAssertEqual(first.0, 0, "Path should start at row 0")
        XCTAssertEqual(first.1, 0, "Path should start at col 0")
        XCTAssertEqual(last.0, n - 1, "Path should end at row N-1")
        XCTAssertEqual(last.1, m - 1, "Path should end at col M-1")
    }

    // MARK: - Warping Path Monotonicity

    func testWarpingPathMonotonicity() {
        let n = 10
        let m = 12
        let cost = makeRandomCost(n: n, m: m)
        let result = DTW.compute(costMatrix: cost)

        for k in 1..<result.warpingPath.count {
            let prev = result.warpingPath[k - 1]
            let curr = result.warpingPath[k]

            XCTAssertGreaterThanOrEqual(curr.0, prev.0,
                "Row indices must be non-decreasing: \(prev.0) -> \(curr.0)")
            XCTAssertGreaterThanOrEqual(curr.1, prev.1,
                "Col indices must be non-decreasing: \(prev.1) -> \(curr.1)")

            // Each step should advance at least one dimension
            let rowStep = curr.0 - prev.0
            let colStep = curr.1 - prev.1
            XCTAssertTrue(rowStep + colStep >= 1,
                "Each step must advance at least one dimension")
            XCTAssertLessThanOrEqual(rowStep, 1,
                "Row step must be 0 or 1")
            XCTAssertLessThanOrEqual(colStep, 1,
                "Col step must be 0 or 1")
        }
    }

    // MARK: - Time-Shifted Sequence Has Low Cost

    func testTimeShiftedLowCost() {
        let nFeatures = 4
        let nFrames = 20
        let data = makeFeatures(nFeatures: nFeatures, nFrames: nFrames)

        // X = data[0..<15], Y = data[5..<20] (shifted by 5 frames)
        let xLen = nFeatures * 15
        let yLen = nFeatures * 15

        var xData = [Float](repeating: 0, count: xLen)
        var yData = [Float](repeating: 0, count: yLen)
        for f in 0..<nFeatures {
            for t in 0..<15 {
                xData[f * 15 + t] = data[f * nFrames + t]
                yData[f * 15 + t] = data[f * nFrames + t + 5]
            }
        }

        let X = Signal(data: xData, shape: [nFeatures, 15], sampleRate: 22050)
        let Y = Signal(data: yData, shape: [nFeatures, 15], sampleRate: 22050)

        let resultShifted = DTW.compute(X: X, Y: Y)

        // Compare with random sequences which should have higher cost
        let randomData = makeFeatures(nFeatures: nFeatures, nFrames: 15, seed: 42)
        let Z = Signal(data: randomData, shape: [nFeatures, 15], sampleRate: 22050)
        let resultRandom = DTW.compute(X: X, Y: Z)

        XCTAssertLessThan(resultShifted.totalCost, resultRandom.totalCost,
            "Time-shifted sequence should have lower DTW cost than random")
    }

    // MARK: - DTW Cost is Symmetric

    func testCostSymmetry() {
        let nFeatures = 3
        let nFramesX = 8
        let nFramesY = 10
        let dataX = makeFeatures(nFeatures: nFeatures, nFrames: nFramesX, seed: 1)
        let dataY = makeFeatures(nFeatures: nFeatures, nFrames: nFramesY, seed: 2)
        let X = Signal(data: dataX, shape: [nFeatures, nFramesX], sampleRate: 22050)
        let Y = Signal(data: dataY, shape: [nFeatures, nFramesY], sampleRate: 22050)

        let resultAB = DTW.compute(X: X, Y: Y)
        let resultBA = DTW.compute(X: Y, Y: X)

        XCTAssertEqual(resultAB.totalCost, resultBA.totalCost, accuracy: 1e-4,
            "DTW(A,B) should equal DTW(B,A)")
    }

    // MARK: - Sakoe-Chiba Band Restricts Warping

    func testSakoeChibaBand() {
        let n = 10
        let m = 10
        let cost = makeRandomCost(n: n, m: m)

        let resultNoBand = DTW.compute(costMatrix: cost)
        let resultBand = DTW.compute(costMatrix: cost, bandWidth: 2)

        // Band-restricted cost should be >= unconstrained cost
        XCTAssertGreaterThanOrEqual(resultBand.totalCost, resultNoBand.totalCost - 1e-5,
            "Band-restricted DTW cost should be >= unconstrained cost")

        // Check that accumulated cost outside band is infinity
        resultBand.accumulatedCost.withUnsafeBufferPointer { buf in
            var foundInf = false
            for i in 0..<n {
                for j in 0..<m {
                    let val = buf[i * m + j]
                    if val.isInfinite {
                        foundInf = true
                    }
                }
            }
            // With a small band on a 10x10 matrix, some cells should be inf
            XCTAssertTrue(foundInf,
                "Band constraint should leave some cells as infinity")
        }
    }

    // MARK: - Feature-Based DTW

    func testFeatureBasedDTW() {
        let nFeatures = 8
        let nFramesX = 10
        let nFramesY = 12
        let dataX = makeFeatures(nFeatures: nFeatures, nFrames: nFramesX, seed: 10)
        let dataY = makeFeatures(nFeatures: nFeatures, nFrames: nFramesY, seed: 20)
        let X = Signal(data: dataX, shape: [nFeatures, nFramesX], sampleRate: 22050)
        let Y = Signal(data: dataY, shape: [nFeatures, nFramesY], sampleRate: 22050)

        let result = DTW.compute(X: X, Y: Y)

        XCTAssertEqual(result.accumulatedCost.shape, [nFramesX, nFramesY])
        XCTAssertGreaterThan(result.totalCost, 0,
            "Different feature sequences should have positive cost")

        // Check path
        XCTAssertFalse(result.warpingPath.isEmpty)
        XCTAssertEqual(result.warpingPath.first!.0, 0)
        XCTAssertEqual(result.warpingPath.first!.1, 0)
        XCTAssertEqual(result.warpingPath.last!.0, nFramesX - 1)
        XCTAssertEqual(result.warpingPath.last!.1, nFramesY - 1)
    }

    // MARK: - Edge Cases

    func testEdgeCase1x1() {
        let cost = Signal(data: [3.0], shape: [1, 1], sampleRate: 0)
        let result = DTW.compute(costMatrix: cost)

        XCTAssertEqual(result.totalCost, 3.0, accuracy: 1e-6)
        XCTAssertEqual(result.warpingPath.count, 1)
        XCTAssertEqual(result.warpingPath[0].0, 0)
        XCTAssertEqual(result.warpingPath[0].1, 0)
    }

    func testEdgeCase1xN() {
        let m = 5
        let data: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0]
        let cost = Signal(data: data, shape: [1, m], sampleRate: 0)
        let result = DTW.compute(costMatrix: cost)

        XCTAssertEqual(result.accumulatedCost.shape, [1, m])
        XCTAssertEqual(result.warpingPath.first!.0, 0)
        XCTAssertEqual(result.warpingPath.first!.1, 0)
        XCTAssertEqual(result.warpingPath.last!.0, 0)
        XCTAssertEqual(result.warpingPath.last!.1, m - 1)

        // Path should traverse all columns since N=1
        XCTAssertEqual(result.warpingPath.count, m)
        for (idx, step) in result.warpingPath.enumerated() {
            XCTAssertEqual(step.0, 0, "Row should always be 0 for 1xN")
            XCTAssertEqual(step.1, idx, "Column should match index for 1xN")
        }

        // Accumulated cost: 1, 3, 6, 10, 15
        XCTAssertEqual(result.totalCost, 15.0, accuracy: 1e-5)
    }

    func testEdgeCaseNx1() {
        let n = 5
        let data: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0]
        let cost = Signal(data: data, shape: [n, 1], sampleRate: 0)
        let result = DTW.compute(costMatrix: cost)

        XCTAssertEqual(result.accumulatedCost.shape, [n, 1])
        XCTAssertEqual(result.warpingPath.first!.0, 0)
        XCTAssertEqual(result.warpingPath.first!.1, 0)
        XCTAssertEqual(result.warpingPath.last!.0, n - 1)
        XCTAssertEqual(result.warpingPath.last!.1, 0)

        // Path should traverse all rows since M=1
        XCTAssertEqual(result.warpingPath.count, n)
        for (idx, step) in result.warpingPath.enumerated() {
            XCTAssertEqual(step.0, idx, "Row should match index for Nx1")
            XCTAssertEqual(step.1, 0, "Column should always be 0 for Nx1")
        }

        // Accumulated cost: 1, 3, 6, 10, 15
        XCTAssertEqual(result.totalCost, 15.0, accuracy: 1e-5)
    }

    // MARK: - Symmetric2 Step Pattern

    func testSymmetric2DifferentCost() {
        let n = 6
        let m = 8
        let cost = makeRandomCost(n: n, m: m)

        let resultStandard = DTW.compute(costMatrix: cost, stepPattern: .standard)
        let resultSym2 = DTW.compute(costMatrix: cost, stepPattern: .symmetric2)

        // Symmetric2 penalizes non-diagonal steps, so costs should generally differ
        // (unless the optimal path is purely diagonal, which is unlikely for n!=m)
        XCTAssertNotEqual(resultStandard.totalCost, resultSym2.totalCost,
            "Symmetric2 should produce different costs than standard for rectangular cost matrix")
    }

    func testSymmetric2WeightsNonDiagonal() {
        // Simple 2x3 cost matrix where non-diagonal weighting matters
        // C = [[1, 2, 3],
        //      [4, 5, 6]]
        let data: [Float] = [1, 2, 3, 4, 5, 6]
        let cost = Signal(data: data, shape: [2, 3], sampleRate: 0)

        let resultStd = DTW.compute(costMatrix: cost, stepPattern: .standard)
        let resultSym2 = DTW.compute(costMatrix: cost, stepPattern: .symmetric2)

        // Standard: D[0,0]=1, D[0,1]=3, D[0,2]=6, D[1,0]=5, D[1,1]=min(5+5,3+5,1+5)=6, D[1,2]=min(6+6,6+6,3+6)=9
        // Sym2: D[0,0]=1, D[0,1]=1+2*2=5, D[0,2]=5+2*3=11
        //       D[1,0]=1+2*4=9, D[1,1]=min(1+5, 9+2*5, 5+2*5)=6, D[1,2]=min(5+6, 6+2*6, 11+2*6)=11
        XCTAssertEqual(resultStd.totalCost, 9.0, accuracy: 1e-5,
            "Standard step pattern total cost for 2x3")
        XCTAssertEqual(resultSym2.totalCost, 11.0, accuracy: 1e-5,
            "Symmetric2 step pattern total cost for 2x3")
    }

    // MARK: - Total Cost Matches Accumulated Cost at (N-1, M-1)

    func testTotalCostMatchesCorner() {
        let n = 8
        let m = 6
        let cost = makeRandomCost(n: n, m: m)
        let result = DTW.compute(costMatrix: cost)

        result.accumulatedCost.withUnsafeBufferPointer { buf in
            let cornerValue = buf[(n - 1) * m + (m - 1)]
            XCTAssertEqual(result.totalCost, cornerValue, accuracy: 1e-6,
                "Total cost should equal accumulated cost at [N-1, M-1]")
        }
    }

    // MARK: - Accumulated Cost Non-Decreasing Along Path

    func testAccumulatedCostNonDecreasingAlongPath() {
        let n = 7
        let m = 9
        let cost = makeRandomCost(n: n, m: m)
        let result = DTW.compute(costMatrix: cost)

        result.accumulatedCost.withUnsafeBufferPointer { buf in
            for k in 1..<result.warpingPath.count {
                let prev = result.warpingPath[k - 1]
                let curr = result.warpingPath[k]
                let prevCost = buf[prev.0 * m + prev.1]
                let currCost = buf[curr.0 * m + curr.1]
                XCTAssertGreaterThanOrEqual(currCost, prevCost - 1e-6,
                    "Accumulated cost along path should be non-decreasing")
            }
        }
    }

    // MARK: - Helpers

    /// Create a constant cost matrix.
    private func makeConstantCost(n: Int, m: Int, value: Float) -> Signal {
        let data = [Float](repeating: value, count: n * m)
        return Signal(data: data, shape: [n, m], sampleRate: 0)
    }

    /// Create a random cost matrix with non-negative values.
    private func makeRandomCost(n: Int, m: Int) -> Signal {
        var data = [Float](repeating: 0, count: n * m)
        for i in 0..<data.count {
            // Simple deterministic pseudo-random
            let x = Float(i * 7 + 13)
            data[i] = abs(sinf(x)) * 10.0
        }
        return Signal(data: data, shape: [n, m], sampleRate: 0)
    }

    /// Create deterministic feature data.
    private func makeFeatures(nFeatures: Int, nFrames: Int, seed: Int = 0) -> [Float] {
        var data = [Float](repeating: 0, count: nFeatures * nFrames)
        for i in 0..<data.count {
            let x = Float(i + seed * 1000 + 1)
            data[i] = sinf(x * 0.1) * cosf(x * 0.07 + 0.5)
        }
        return data
    }
}
