import XCTest
@testable import MetalMomCore

final class RecurrenceTests: XCTestCase {

    // MARK: - Recurrence Matrix Shape

    func testRecurrenceMatrixShape() {
        let nFeatures = 8
        let nFrames = 20
        let data = makeFeatures(nFeatures: nFeatures, nFrames: nFrames)
        let features = Signal(data: data, shape: [nFeatures, nFrames], sampleRate: 22050)

        let result = Recurrence.recurrenceMatrix(features, mode: .knn(k: 5))

        XCTAssertEqual(result.shape, [nFrames, nFrames],
            "Recurrence matrix should be [nFrames, nFrames]")
        XCTAssertEqual(result.count, nFrames * nFrames)
    }

    // MARK: - kNN Mode Produces Binary Values

    func testKNNModeBinary() {
        let nFeatures = 4
        let nFrames = 10
        let data = makeFeatures(nFeatures: nFeatures, nFrames: nFrames)
        let features = Signal(data: data, shape: [nFeatures, nFrames], sampleRate: 22050)

        let result = Recurrence.recurrenceMatrix(features, mode: .knn(k: 3))

        result.withUnsafeBufferPointer { buf in
            for i in 0..<result.count {
                let v = buf[i]
                XCTAssertTrue(v == 0.0 || v == 1.0,
                    "kNN mode should produce binary values, got \(v) at index \(i)")
            }
        }
    }

    // MARK: - Threshold Mode Produces Binary Values

    func testThresholdModeBinary() {
        let nFeatures = 4
        let nFrames = 10
        let data = makeFeatures(nFeatures: nFeatures, nFrames: nFrames)
        let features = Signal(data: data, shape: [nFeatures, nFrames], sampleRate: 22050)

        // Use a generous threshold to get some recurrence points
        let result = Recurrence.recurrenceMatrix(features, mode: .threshold(5.0))

        result.withUnsafeBufferPointer { buf in
            for i in 0..<result.count {
                let v = buf[i]
                XCTAssertTrue(v == 0.0 || v == 1.0,
                    "Threshold mode should produce binary values, got \(v) at index \(i)")
            }
        }
    }

    // MARK: - Soft Mode Produces Non-Negative Distances

    func testSoftModeNonNegative() {
        let nFeatures = 4
        let nFrames = 10
        let data = makeFeatures(nFeatures: nFeatures, nFrames: nFrames)
        let features = Signal(data: data, shape: [nFeatures, nFrames], sampleRate: 22050)

        let result = Recurrence.recurrenceMatrix(features, mode: .soft)

        result.withUnsafeBufferPointer { buf in
            for i in 0..<result.count {
                XCTAssertGreaterThanOrEqual(buf[i], 0,
                    "Soft mode should produce non-negative values, got \(buf[i]) at index \(i)")
                XCTAssertTrue(buf[i].isFinite,
                    "Soft mode values should be finite")
            }
        }
    }

    // MARK: - Soft Mode Diagonal Is Zero (Self-Distance)

    func testSoftModeDiagonalZero() {
        let nFeatures = 4
        let nFrames = 10
        let data = makeFeatures(nFeatures: nFeatures, nFrames: nFrames)
        let features = Signal(data: data, shape: [nFeatures, nFrames], sampleRate: 22050)

        let result = Recurrence.recurrenceMatrix(features, mode: .soft)

        result.withUnsafeBufferPointer { buf in
            for i in 0..<nFrames {
                XCTAssertEqual(buf[i * nFrames + i], 0.0, accuracy: 1e-3,
                    "Diagonal should be ~zero (self-distance) at [\(i), \(i)]")
            }
        }
    }

    // MARK: - Symmetric Option Makes R Symmetric

    func testSymmetricOption() {
        let nFeatures = 4
        let nFrames = 15
        let data = makeFeatures(nFeatures: nFeatures, nFrames: nFrames)
        let features = Signal(data: data, shape: [nFeatures, nFrames], sampleRate: 22050)

        let result = Recurrence.recurrenceMatrix(features, mode: .knn(k: 3), symmetric: true)

        result.withUnsafeBufferPointer { buf in
            for i in 0..<nFrames {
                for j in 0..<nFrames {
                    XCTAssertEqual(buf[i * nFrames + j], buf[j * nFrames + i], accuracy: 1e-6,
                        "Symmetric recurrence matrix should have R[\(i),\(j)] == R[\(j),\(i)]")
                }
            }
        }
    }

    // MARK: - Cross-Similarity Shape

    func testCrossSimilarityShape() {
        let nFeatures = 6
        let nFramesA = 10
        let nFramesB = 15
        let dataA = makeFeatures(nFeatures: nFeatures, nFrames: nFramesA, seed: 42)
        let dataB = makeFeatures(nFeatures: nFeatures, nFrames: nFramesB, seed: 99)
        let sigA = Signal(data: dataA, shape: [nFeatures, nFramesA], sampleRate: 22050)
        let sigB = Signal(data: dataB, shape: [nFeatures, nFramesB], sampleRate: 22050)

        let result = Recurrence.crossSimilarity(sigA, sigB)

        XCTAssertEqual(result.shape, [nFramesA, nFramesB],
            "Cross-similarity should be [nFramesA, nFramesB]")
        XCTAssertEqual(result.count, nFramesA * nFramesB)
    }

    // MARK: - Cross-Similarity Non-Negative

    func testCrossSimilarityNonNegative() {
        let nFeatures = 4
        let nFramesA = 8
        let nFramesB = 12
        let dataA = makeFeatures(nFeatures: nFeatures, nFrames: nFramesA, seed: 42)
        let dataB = makeFeatures(nFeatures: nFeatures, nFrames: nFramesB, seed: 99)
        let sigA = Signal(data: dataA, shape: [nFeatures, nFramesA], sampleRate: 22050)
        let sigB = Signal(data: dataB, shape: [nFeatures, nFramesB], sampleRate: 22050)

        let result = Recurrence.crossSimilarity(sigA, sigB)

        result.withUnsafeBufferPointer { buf in
            for i in 0..<result.count {
                XCTAssertGreaterThanOrEqual(buf[i], 0,
                    "Cross-similarity distances should be non-negative")
                XCTAssertTrue(buf[i].isFinite, "Values should be finite")
            }
        }
    }

    // MARK: - RQA: Recurrence Rate in [0, 1]

    func testRQARecurrenceRateInRange() {
        let nFeatures = 4
        let nFrames = 15
        let data = makeFeatures(nFeatures: nFeatures, nFrames: nFrames)
        let features = Signal(data: data, shape: [nFeatures, nFrames], sampleRate: 22050)

        let rec = Recurrence.recurrenceMatrix(features, mode: .knn(k: 3))
        let result = Recurrence.rqa(rec)

        XCTAssertGreaterThanOrEqual(result.recurrenceRate, 0,
            "Recurrence rate should be >= 0")
        XCTAssertLessThanOrEqual(result.recurrenceRate, 1,
            "Recurrence rate should be <= 1")
    }

    // MARK: - RQA: Determinism in [0, 1]

    func testRQADeterminismInRange() {
        let nFeatures = 4
        let nFrames = 15
        let data = makeFeatures(nFeatures: nFeatures, nFrames: nFrames)
        let features = Signal(data: data, shape: [nFeatures, nFrames], sampleRate: 22050)

        let rec = Recurrence.recurrenceMatrix(features, mode: .knn(k: 3))
        let result = Recurrence.rqa(rec)

        XCTAssertGreaterThanOrEqual(result.determinism, 0,
            "Determinism should be >= 0")
        XCTAssertLessThanOrEqual(result.determinism, 1,
            "Determinism should be <= 1")
    }

    // MARK: - RQA: Laminarity in [0, 1]

    func testRQALaminarityInRange() {
        let nFeatures = 4
        let nFrames = 15
        let data = makeFeatures(nFeatures: nFeatures, nFrames: nFrames)
        let features = Signal(data: data, shape: [nFeatures, nFrames], sampleRate: 22050)

        let rec = Recurrence.recurrenceMatrix(features, mode: .knn(k: 3))
        let result = Recurrence.rqa(rec)

        XCTAssertGreaterThanOrEqual(result.laminarity, 0,
            "Laminarity should be >= 0")
        XCTAssertLessThanOrEqual(result.laminarity, 1,
            "Laminarity should be <= 1")
    }

    // MARK: - RQA: All-Ones Matrix Has High Recurrence Rate

    func testRQAAllOnesHighRecurrenceRate() {
        let n = 10
        // All-ones matrix (full recurrence)
        let data = [Float](repeating: 1.0, count: n * n)
        let matrix = Signal(data: data, shape: [n, n], sampleRate: 0)

        let result = Recurrence.rqa(matrix)

        XCTAssertGreaterThan(result.recurrenceRate, 0.9,
            "All-ones matrix should have high recurrence rate, got \(result.recurrenceRate)")
    }

    // MARK: - RQA: Identity Matrix Properties

    func testRQAIdentityMatrix() {
        let n = 10
        // Identity matrix: only main diagonal is 1
        var data = [Float](repeating: 0, count: n * n)
        for i in 0..<n {
            data[i * n + i] = 1.0
        }
        let matrix = Signal(data: data, shape: [n, n], sampleRate: 0)

        let result = Recurrence.rqa(matrix)

        // Identity matrix: no recurrence points off-diagonal
        XCTAssertEqual(result.recurrenceRate, 0.0, accuracy: 1e-6,
            "Identity matrix should have recurrence rate 0 (only diagonal is filled)")
        XCTAssertEqual(result.determinism, 0.0, accuracy: 1e-6,
            "No off-diagonal points means determinism 0")
    }

    // MARK: - Cosine Metric Works

    func testCosineMetricRecurrence() {
        let nFeatures = 4
        let nFrames = 10
        let data = makeFeatures(nFeatures: nFeatures, nFrames: nFrames)
        let features = Signal(data: data, shape: [nFeatures, nFrames], sampleRate: 22050)

        let result = Recurrence.recurrenceMatrix(features, mode: .knn(k: 3), metric: .cosine)

        XCTAssertEqual(result.shape, [nFrames, nFrames])
        result.withUnsafeBufferPointer { buf in
            for i in 0..<result.count {
                XCTAssertTrue(buf[i] == 0.0 || buf[i] == 1.0,
                    "kNN with cosine should produce binary values")
            }
        }
    }

    func testCosineMetricCrossSimilarity() {
        let nFeatures = 4
        let nFramesA = 8
        let nFramesB = 6
        let dataA = makeFeatures(nFeatures: nFeatures, nFrames: nFramesA, seed: 42)
        let dataB = makeFeatures(nFeatures: nFeatures, nFrames: nFramesB, seed: 99)
        let sigA = Signal(data: dataA, shape: [nFeatures, nFramesA], sampleRate: 22050)
        let sigB = Signal(data: dataB, shape: [nFeatures, nFramesB], sampleRate: 22050)

        let result = Recurrence.crossSimilarity(sigA, sigB, metric: .cosine)

        XCTAssertEqual(result.shape, [nFramesA, nFramesB])
        result.withUnsafeBufferPointer { buf in
            for i in 0..<result.count {
                XCTAssertGreaterThanOrEqual(buf[i], 0,
                    "Cosine distances should be non-negative")
                XCTAssertTrue(buf[i].isFinite)
            }
        }
    }

    // MARK: - Edge Case: Single Frame

    func testSingleFrame() {
        let nFeatures = 4
        let data: [Float] = [1.0, 2.0, 3.0, 4.0]
        let features = Signal(data: data, shape: [nFeatures, 1], sampleRate: 22050)

        let result = Recurrence.recurrenceMatrix(features, mode: .knn(k: 3))

        XCTAssertEqual(result.shape, [1, 1],
            "Single frame should produce 1x1 matrix")
    }

    // MARK: - Cross-Similarity Self vs Self Equals Recurrence Soft

    func testCrossSimilarityWithSelfEqualsSoftRecurrence() {
        let nFeatures = 4
        let nFrames = 8
        let data = makeFeatures(nFeatures: nFeatures, nFrames: nFrames)
        let features = Signal(data: data, shape: [nFeatures, nFrames], sampleRate: 22050)

        let rec = Recurrence.recurrenceMatrix(features, mode: .soft)
        let cross = Recurrence.crossSimilarity(features, features)

        // Cross-similarity of A with itself should equal the soft recurrence matrix
        XCTAssertEqual(rec.shape, cross.shape)
        rec.withUnsafeBufferPointer { recBuf in
            cross.withUnsafeBufferPointer { crossBuf in
                for i in 0..<rec.count {
                    XCTAssertEqual(recBuf[i], crossBuf[i], accuracy: 1e-4,
                        "Cross-similarity with self should match soft recurrence at index \(i)")
                }
            }
        }
    }

    // MARK: - kNN Recurrence Has Expected Row Sum

    func testKNNRecurrenceRowSum() {
        let nFeatures = 4
        let nFrames = 10
        let k = 3
        let data = makeFeatures(nFeatures: nFeatures, nFrames: nFrames)
        let features = Signal(data: data, shape: [nFeatures, nFrames], sampleRate: 22050)

        let result = Recurrence.recurrenceMatrix(features, mode: .knn(k: k))

        // Each row should sum to exactly k (each frame has k neighbors)
        result.withUnsafeBufferPointer { buf in
            for i in 0..<nFrames {
                var rowSum: Float = 0
                for j in 0..<nFrames {
                    rowSum += buf[i * nFrames + j]
                }
                XCTAssertEqual(rowSum, Float(k), accuracy: 1e-6,
                    "Row \(i) should sum to k=\(k), got \(rowSum)")
            }
        }
    }

    // MARK: - RQA Entropy Non-Negative

    func testRQAEntropyNonNegative() {
        let nFeatures = 4
        let nFrames = 20
        let data = makeFeatures(nFeatures: nFeatures, nFrames: nFrames)
        let features = Signal(data: data, shape: [nFeatures, nFrames], sampleRate: 22050)

        let rec = Recurrence.recurrenceMatrix(features, mode: .knn(k: 5))
        let result = Recurrence.rqa(rec)

        XCTAssertGreaterThanOrEqual(result.entropy, 0,
            "Entropy should be non-negative")
    }

    // MARK: - Helpers

    /// Create feature data [nFeatures, nFrames] with some variation.
    private func makeFeatures(nFeatures: Int, nFrames: Int, seed: UInt64 = 42) -> [Float] {
        var state = seed
        var data = [Float](repeating: 0, count: nFeatures * nFrames)
        for i in 0..<data.count {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            let u = Float(state >> 33) / Float(1 << 31)
            data[i] = u + 0.01
        }
        return data
    }
}
