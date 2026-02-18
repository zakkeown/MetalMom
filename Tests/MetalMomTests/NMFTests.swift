import XCTest
@testable import MetalMomCore

final class NMFTests: XCTestCase {

    // MARK: - Output Shapes

    func testOutputShapes() {
        let nFeatures = 10
        let nSamples = 20
        let nComponents = 3
        let data = makeNonNegativeMatrix(rows: nFeatures, cols: nSamples)
        let V = Signal(data: data, shape: [nFeatures, nSamples], sampleRate: 22050)

        let result = NMF.decompose(V, nComponents: nComponents, nIter: 50, seed: 42)

        XCTAssertEqual(result.W.shape, [nFeatures, nComponents],
            "W shape should be [nFeatures, nComponents]")
        XCTAssertEqual(result.H.shape, [nComponents, nSamples],
            "H shape should be [nComponents, nSamples]")
    }

    // MARK: - Non-negativity

    func testNonNegativity() {
        let nFeatures = 8
        let nSamples = 15
        let data = makeNonNegativeMatrix(rows: nFeatures, cols: nSamples)
        let V = Signal(data: data, shape: [nFeatures, nSamples], sampleRate: 22050)

        let result = NMF.decompose(V, nComponents: 4, nIter: 100, seed: 42)

        for i in 0..<result.W.count {
            XCTAssertGreaterThanOrEqual(result.W[i], 0,
                "W[\(i)] should be non-negative, got \(result.W[i])")
        }
        for i in 0..<result.H.count {
            XCTAssertGreaterThanOrEqual(result.H[i], 0,
                "H[\(i)] should be non-negative, got \(result.H[i])")
        }
    }

    // MARK: - Reconstruction Error Decreases

    func testReconstructionErrorDecreases() {
        let nFeatures = 10
        let nSamples = 20
        let nComponents = 3
        let data = makeNonNegativeMatrix(rows: nFeatures, cols: nSamples)
        let V = Signal(data: data, shape: [nFeatures, nSamples], sampleRate: 22050)

        let resultFew = NMF.decompose(V, nComponents: nComponents, nIter: 10, seed: 42)
        let errorFew = NMF.reconstructionError(V: V, W: resultFew.W, H: resultFew.H)

        let resultMany = NMF.decompose(V, nComponents: nComponents, nIter: 200, seed: 42)
        let errorMany = NMF.reconstructionError(V: V, W: resultMany.W, H: resultMany.H)

        XCTAssertLessThan(errorMany, errorFew,
            "More iterations should reduce error: \(errorMany) vs \(errorFew)")
    }

    // MARK: - Reasonable Reconstruction Error

    func testReasonableReconstructionError() {
        let nFeatures = 10
        let nSamples = 20
        let nComponents = 5
        let data = makeNonNegativeMatrix(rows: nFeatures, cols: nSamples)
        let V = Signal(data: data, shape: [nFeatures, nSamples], sampleRate: 22050)

        let result = NMF.decompose(V, nComponents: nComponents, nIter: 200, seed: 42)
        let error = NMF.reconstructionError(V: V, W: result.W, H: result.H)

        XCTAssertLessThan(error, 0.5,
            "Relative reconstruction error should be < 0.5, got \(error)")
    }

    // MARK: - Different nComponents

    func testDifferentNComponents() {
        let nFeatures = 12
        let nSamples = 18
        let data = makeNonNegativeMatrix(rows: nFeatures, cols: nSamples)
        let V = Signal(data: data, shape: [nFeatures, nSamples], sampleRate: 22050)

        for nc in [1, 3, 6, 12] {
            let result = NMF.decompose(V, nComponents: nc, nIter: 50, seed: 42)
            XCTAssertEqual(result.W.shape, [nFeatures, nc],
                "W shape should be [nFeatures, \(nc)]")
            XCTAssertEqual(result.H.shape, [nc, nSamples],
                "H shape should be [\(nc), nSamples]")
        }
    }

    // MARK: - More Components = Better Reconstruction

    func testMoreComponentsBetterReconstruction() {
        let nFeatures = 10
        let nSamples = 20
        let data = makeNonNegativeMatrix(rows: nFeatures, cols: nSamples)
        let V = Signal(data: data, shape: [nFeatures, nSamples], sampleRate: 22050)

        let resultLow = NMF.decompose(V, nComponents: 2, nIter: 200, seed: 42)
        let errorLow = NMF.reconstructionError(V: V, W: resultLow.W, H: resultLow.H)

        let resultHigh = NMF.decompose(V, nComponents: 8, nIter: 200, seed: 42)
        let errorHigh = NMF.reconstructionError(V: V, W: resultHigh.W, H: resultHigh.H)

        XCTAssertLessThan(errorHigh, errorLow,
            "More components should reduce error: \(errorHigh) vs \(errorLow)")
    }

    // MARK: - KL Divergence Objective

    func testKLDivergenceObjective() {
        let nFeatures = 10
        let nSamples = 15
        let nComponents = 4
        let data = makeNonNegativeMatrix(rows: nFeatures, cols: nSamples)
        let V = Signal(data: data, shape: [nFeatures, nSamples], sampleRate: 22050)

        let result = NMF.decompose(V, nComponents: nComponents, nIter: 100,
                                   objective: .klDivergence, seed: 42)

        // Check shapes
        XCTAssertEqual(result.W.shape, [nFeatures, nComponents])
        XCTAssertEqual(result.H.shape, [nComponents, nSamples])

        // Check non-negativity
        for i in 0..<result.W.count {
            XCTAssertGreaterThanOrEqual(result.W[i], 0,
                "KL W[\(i)] should be non-negative")
        }
        for i in 0..<result.H.count {
            XCTAssertGreaterThanOrEqual(result.H[i], 0,
                "KL H[\(i)] should be non-negative")
        }

        // Reconstruction error should be reasonable
        let error = NMF.reconstructionError(V: V, W: result.W, H: result.H)
        XCTAssertLessThan(error, 0.5,
            "KL reconstruction error should be < 0.5, got \(error)")
    }

    // MARK: - KL Divergence Error Decreases

    func testKLDivergenceErrorDecreases() {
        let nFeatures = 10
        let nSamples = 15
        let data = makeNonNegativeMatrix(rows: nFeatures, cols: nSamples)
        let V = Signal(data: data, shape: [nFeatures, nSamples], sampleRate: 22050)

        let resultFew = NMF.decompose(V, nComponents: 3, nIter: 10,
                                       objective: .klDivergence, seed: 42)
        let errorFew = NMF.reconstructionError(V: V, W: resultFew.W, H: resultFew.H)

        let resultMany = NMF.decompose(V, nComponents: 3, nIter: 200,
                                        objective: .klDivergence, seed: 42)
        let errorMany = NMF.reconstructionError(V: V, W: resultMany.W, H: resultMany.H)

        XCTAssertLessThan(errorMany, errorFew,
            "More KL iterations should reduce error: \(errorMany) vs \(errorFew)")
    }

    // MARK: - Single Column Input

    func testSingleColumnInput() {
        let nFeatures = 5
        let nSamples = 1
        let nComponents = 1
        let data: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0]
        let V = Signal(data: data, shape: [nFeatures, nSamples], sampleRate: 22050)

        let result = NMF.decompose(V, nComponents: nComponents, nIter: 50, seed: 42)

        XCTAssertEqual(result.W.shape, [nFeatures, nComponents])
        XCTAssertEqual(result.H.shape, [nComponents, nSamples])

        // All values non-negative
        for i in 0..<result.W.count {
            XCTAssertGreaterThanOrEqual(result.W[i], 0)
        }
        for i in 0..<result.H.count {
            XCTAssertGreaterThanOrEqual(result.H[i], 0)
        }
    }

    // MARK: - Seeded Determinism

    func testSeededDeterminism() {
        let nFeatures = 8
        let nSamples = 12
        let nComponents = 3
        let data = makeNonNegativeMatrix(rows: nFeatures, cols: nSamples, seed: 100)
        let V = Signal(data: data, shape: [nFeatures, nSamples], sampleRate: 22050)

        let result1 = NMF.decompose(V, nComponents: nComponents, nIter: 50, seed: 42)
        let result2 = NMF.decompose(V, nComponents: nComponents, nIter: 50, seed: 42)

        for i in 0..<result1.W.count {
            XCTAssertEqual(result1.W[i], result2.W[i], accuracy: 1e-6,
                "Seeded NMF should be deterministic: W[\(i)] = \(result1.W[i]) vs \(result2.W[i])")
        }
        for i in 0..<result1.H.count {
            XCTAssertEqual(result1.H[i], result2.H[i], accuracy: 1e-6,
                "Seeded NMF should be deterministic: H[\(i)] = \(result1.H[i]) vs \(result2.H[i])")
        }
    }

    // MARK: - Low-Rank Recovery

    func testLowRankRecovery() {
        // Create a rank-2 matrix V = W_true * H_true and check NMF recovers it well
        let nFeatures = 8
        let nSamples = 12

        // W_true: [8, 2] — two basis vectors
        let wTrue: [Float] = [
            1, 0, 2, 0, 3, 0, 4, 0,  // basis 1
            0, 1, 0, 2, 0, 3, 0, 4,  // basis 2
        ]
        // H_true: [2, 12] — activations
        var hTrue = [Float](repeating: 0, count: 2 * nSamples)
        for t in 0..<nSamples {
            hTrue[t] = Float(t) / Float(nSamples)      // basis 1 activation
            hTrue[nSamples + t] = 1.0 - Float(t) / Float(nSamples)  // basis 2 activation
        }

        // V = W_true * H_true: [8, 12]
        // W_true is column-interleaved as [f0c0, f0c1, f1c0, f1c1, ...]
        var vData = [Float](repeating: 0, count: nFeatures * nSamples)
        for f in 0..<nFeatures {
            for t in 0..<nSamples {
                var sum: Float = 0
                for k in 0..<2 {
                    sum += wTrue[f * 2 + k] * hTrue[k * nSamples + t]
                }
                vData[f * nSamples + t] = sum
            }
        }

        let V = Signal(data: vData, shape: [nFeatures, nSamples], sampleRate: 22050)
        let result = NMF.decompose(V, nComponents: 2, nIter: 300, seed: 42)
        let error = NMF.reconstructionError(V: V, W: result.W, H: result.H)

        // Should reconstruct a rank-2 matrix very well
        XCTAssertLessThan(error, 0.05,
            "NMF should reconstruct a rank-2 matrix well, got error \(error)")
    }

    // MARK: - Reconstruction Error With Finite Values

    func testReconstructionErrorFinite() {
        let nFeatures = 6
        let nSamples = 10
        let data = makeNonNegativeMatrix(rows: nFeatures, cols: nSamples)
        let V = Signal(data: data, shape: [nFeatures, nSamples], sampleRate: 22050)

        let result = NMF.decompose(V, nComponents: 3, nIter: 50, seed: 42)
        let error = NMF.reconstructionError(V: V, W: result.W, H: result.H)

        XCTAssertTrue(error.isFinite, "Reconstruction error should be finite, got \(error)")
        XCTAssertGreaterThanOrEqual(error, 0, "Reconstruction error should be non-negative")
    }

    // MARK: - All Values Finite

    func testAllValuesFinite() {
        let nFeatures = 10
        let nSamples = 20
        let data = makeNonNegativeMatrix(rows: nFeatures, cols: nSamples)
        let V = Signal(data: data, shape: [nFeatures, nSamples], sampleRate: 22050)

        let result = NMF.decompose(V, nComponents: 4, nIter: 200, seed: 42)

        for i in 0..<result.W.count {
            XCTAssertTrue(result.W[i].isFinite, "W[\(i)] should be finite, got \(result.W[i])")
        }
        for i in 0..<result.H.count {
            XCTAssertTrue(result.H[i].isFinite, "H[\(i)] should be finite, got \(result.H[i])")
        }
    }

    // MARK: - Helpers

    private func makeNonNegativeMatrix(rows: Int, cols: Int, seed: UInt64 = 42) -> [Float] {
        // Deterministic non-negative data using a simple LCG
        var state = seed
        var data = [Float](repeating: 0, count: rows * cols)
        for i in 0..<data.count {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            let u = Float(state >> 33) / Float(1 << 31)
            data[i] = u + 0.01  // ensure strictly positive
        }
        return data
    }
}
