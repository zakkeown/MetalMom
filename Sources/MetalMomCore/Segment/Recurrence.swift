import Foundation
import Accelerate

/// Recurrence matrix, cross-similarity matrix, and recurrence quantification analysis.
///
/// A recurrence matrix captures self-similarity in a sequence of feature vectors.
/// Cross-similarity captures similarity between two different sequences.
/// RQA extracts statistics (determinism, laminarity, etc.) from binary recurrence matrices.
public enum Recurrence {

    /// Mode for constructing a recurrence/self-similarity matrix.
    public enum Mode {
        /// k-nearest neighbors: R[i,j] = 1 if j is among the k nearest neighbors of i.
        case knn(k: Int)
        /// Distance threshold: R[i,j] = 1 if distance(i,j) < threshold.
        case threshold(Float)
        /// Soft (raw distances): R[i,j] = distance(i,j), no thresholding.
        case soft
    }

    /// Distance metric.
    public enum Metric {
        /// Euclidean (L2) distance.
        case euclidean
        /// Cosine distance: 1 - cosine_similarity.
        case cosine
    }

    /// RQA statistics extracted from a binary recurrence matrix.
    public struct RQAResult {
        /// Fraction of recurrence points (non-zero entries excluding main diagonal).
        public let recurrenceRate: Float
        /// Fraction of recurrence points forming diagonal lines (length >= lmin).
        public let determinism: Float
        /// Fraction of recurrence points forming vertical lines (length >= vmin).
        public let laminarity: Float
        /// Mean length of diagonal lines (length >= lmin).
        public let averageDiagonalLength: Float
        /// Mean length of vertical lines (length >= vmin), also called trapping time.
        public let averageVerticalLength: Float
        /// Length of the longest diagonal line (length >= lmin).
        public let longestDiagonalLine: Int
        /// Shannon entropy of the diagonal line length distribution.
        public let entropy: Float
    }

    // MARK: - Recurrence Matrix

    /// Compute self-recurrence matrix.
    ///
    /// - Parameters:
    ///   - features: Input 2D Signal, shape [nFeatures, nFrames]. Each column is a feature vector.
    ///   - mode: Thresholding mode. Default `.knn(k: 5)`.
    ///   - metric: Distance metric. Default `.euclidean`.
    ///   - symmetric: If true, use mutual nearest neighbors: R[i,j] = R[i,j] AND R[j,i]. Default false.
    /// - Returns: [nFrames, nFrames] Signal. Binary (0/1) for knn/threshold modes, non-negative distances for soft mode.
    public static func recurrenceMatrix(
        _ features: Signal,
        mode: Mode = .knn(k: 5),
        metric: Metric = .euclidean,
        symmetric: Bool = false
    ) -> Signal {
        precondition(features.shape.count == 2,
                     "Recurrence input must be 2D [nFeatures, nFrames]")
        let nFeatures = features.shape[0]
        let nFrames = features.shape[1]

        guard nFeatures > 0 && nFrames > 0 else {
            return Signal(data: [], shape: [0, 0], sampleRate: features.sampleRate)
        }

        // Single frame: return 1x1 matrix
        if nFrames == 1 {
            let val: Float
            switch mode {
            case .soft:
                val = 0.0  // self-distance is 0
            default:
                val = 0.0  // not a neighbor of itself for knn/threshold
            }
            return Signal(data: [val], shape: [1, 1], sampleRate: features.sampleRate)
        }

        // Compute pairwise distance matrix [nFrames, nFrames]
        let distances = features.withUnsafeBufferPointer { buf in
            computeDistanceMatrix(buf: buf, nFeatures: nFeatures, nFrames: nFrames, metric: metric)
        }

        // Apply mode to produce output matrix
        var result = [Float](repeating: 0, count: nFrames * nFrames)

        switch mode {
        case .knn(let k):
            let effectiveK = min(max(k, 1), nFrames - 1)
            // For each frame, find k nearest neighbors (excluding self)
            for i in 0..<nFrames {
                var pairs = [(Float, Int)]()
                pairs.reserveCapacity(nFrames - 1)
                for j in 0..<nFrames {
                    if j == i { continue }
                    pairs.append((distances[i * nFrames + j], j))
                }
                pairs.sort { $0.0 < $1.0 }
                let count = min(effectiveK, pairs.count)
                for p in 0..<count {
                    result[i * nFrames + pairs[p].1] = 1.0
                }
            }

            if symmetric {
                // Mutual kNN: R[i,j] = 1 only if both i->j and j->i
                var symResult = [Float](repeating: 0, count: nFrames * nFrames)
                for i in 0..<nFrames {
                    for j in 0..<nFrames {
                        if result[i * nFrames + j] > 0 && result[j * nFrames + i] > 0 {
                            symResult[i * nFrames + j] = 1.0
                        }
                    }
                }
                result = symResult
            }

        case .threshold(let thresh):
            for i in 0..<nFrames {
                for j in 0..<nFrames {
                    if i == j { continue }  // exclude diagonal
                    if distances[i * nFrames + j] < thresh {
                        result[i * nFrames + j] = 1.0
                    }
                }
            }

            if symmetric {
                // Already symmetric by construction (distance is symmetric),
                // but enforce for clarity
                for i in 0..<nFrames {
                    for j in (i + 1)..<nFrames {
                        let val = min(result[i * nFrames + j], result[j * nFrames + i])
                        result[i * nFrames + j] = val
                        result[j * nFrames + i] = val
                    }
                }
            }

        case .soft:
            // Raw distance values
            result = distances
        }

        return Signal(data: result, shape: [nFrames, nFrames], sampleRate: features.sampleRate)
    }

    // MARK: - Cross-Similarity

    /// Compute cross-similarity matrix between two feature sequences.
    ///
    /// - Parameters:
    ///   - featuresA: Input 2D Signal, shape [nFeatures, nFramesA].
    ///   - featuresB: Input 2D Signal, shape [nFeatures, nFramesB].
    ///   - metric: Distance metric. Default `.euclidean`.
    /// - Returns: [nFramesA, nFramesB] distance/similarity matrix.
    public static func crossSimilarity(
        _ featuresA: Signal,
        _ featuresB: Signal,
        metric: Metric = .euclidean
    ) -> Signal {
        precondition(featuresA.shape.count == 2 && featuresB.shape.count == 2,
                     "Cross-similarity inputs must be 2D [nFeatures, nFrames]")
        let nFeatA = featuresA.shape[0]
        let nFeatB = featuresB.shape[0]
        precondition(nFeatA == nFeatB,
                     "Feature dimensions must match: \(nFeatA) vs \(nFeatB)")

        let nFeatures = nFeatA
        let nFramesA = featuresA.shape[1]
        let nFramesB = featuresB.shape[1]

        guard nFeatures > 0 && nFramesA > 0 && nFramesB > 0 else {
            return Signal(data: [], shape: [nFramesA, nFramesB],
                          sampleRate: featuresA.sampleRate)
        }

        let result = featuresA.withUnsafeBufferPointer { bufA in
            featuresB.withUnsafeBufferPointer { bufB in
                computeCrossDistanceMatrix(
                    bufA: bufA, nFramesA: nFramesA,
                    bufB: bufB, nFramesB: nFramesB,
                    nFeatures: nFeatures, metric: metric
                )
            }
        }

        return Signal(data: result, shape: [nFramesA, nFramesB],
                      sampleRate: featuresA.sampleRate)
    }

    // MARK: - RQA

    /// Compute Recurrence Quantification Analysis statistics from a binary recurrence matrix.
    ///
    /// - Parameters:
    ///   - recurrenceMatrix: Binary [N, N] recurrence matrix (0/1 values).
    ///   - lmin: Minimum length for diagonal lines. Default 2.
    ///   - vmin: Minimum length for vertical lines. Default 2.
    /// - Returns: RQA statistics.
    public static func rqa(
        _ recurrenceMatrix: Signal,
        lmin: Int = 2,
        vmin: Int = 2
    ) -> RQAResult {
        precondition(recurrenceMatrix.shape.count == 2,
                     "RQA input must be 2D [N, N]")
        let n = recurrenceMatrix.shape[0]
        precondition(recurrenceMatrix.shape[1] == n,
                     "RQA input must be square, got [\(n), \(recurrenceMatrix.shape[1])]")

        guard n > 0 else {
            return RQAResult(
                recurrenceRate: 0, determinism: 0, laminarity: 0,
                averageDiagonalLength: 0, averageVerticalLength: 0,
                longestDiagonalLine: 0, entropy: 0
            )
        }

        return recurrenceMatrix.withUnsafeBufferPointer { buf in
            computeRQA(buf: buf, n: n, lmin: lmin, vmin: vmin)
        }
    }

    // MARK: - Distance Matrices (private)

    /// Compute pairwise distance matrix between frames of a single sequence.
    /// Layout: [nFrames, nFrames], row-major.
    private static func computeDistanceMatrix(
        buf: UnsafeBufferPointer<Float>,
        nFeatures: Int,
        nFrames: Int,
        metric: Metric
    ) -> [Float] {
        switch metric {
        case .euclidean:
            return euclideanDistanceMatrix(buf: buf, nFeatures: nFeatures, nFrames: nFrames)
        case .cosine:
            return cosineDistanceMatrix(buf: buf, nFeatures: nFeatures, nFrames: nFrames)
        }
    }

    /// Euclidean distance matrix using BLAS gram matrix.
    /// Input layout: [nFeatures, nFrames] row-major => S[f, t] = buf[f * nFrames + t].
    private static func euclideanDistanceMatrix(
        buf: UnsafeBufferPointer<Float>,
        nFeatures: Int,
        nFrames: Int
    ) -> [Float] {
        let base = buf.baseAddress!
        var dist = [Float](repeating: 0, count: nFrames * nFrames)

        // Precompute squared norms
        var sqNorms = [Float](repeating: 0, count: nFrames)
        for t in 0..<nFrames {
            var sum: Float = 0
            for f in 0..<nFeatures {
                let v = base[f * nFrames + t]
                sum += v * v
            }
            sqNorms[t] = sum
        }

        // Gram matrix G = X^T X where X is [nFeatures, nFrames]
        let gram = UnsafeMutablePointer<Float>.allocate(capacity: nFrames * nFrames)
        defer { gram.deallocate() }

        cblas_sgemm(
            CblasRowMajor, CblasTrans, CblasNoTrans,
            Int32(nFrames), Int32(nFrames), Int32(nFeatures),
            1.0, base, Int32(nFrames),
            base, Int32(nFrames),
            0.0, gram, Int32(nFrames)
        )

        for i in 0..<nFrames {
            for j in 0..<nFrames {
                let d2 = sqNorms[i] + sqNorms[j] - 2.0 * gram[i * nFrames + j]
                dist[i * nFrames + j] = sqrtf(max(d2, 0))
            }
        }

        return dist
    }

    /// Cosine distance matrix: d(i,j) = 1 - cos(frame_i, frame_j).
    private static func cosineDistanceMatrix(
        buf: UnsafeBufferPointer<Float>,
        nFeatures: Int,
        nFrames: Int
    ) -> [Float] {
        let base = buf.baseAddress!
        var dist = [Float](repeating: 0, count: nFrames * nFrames)

        // Compute L2 norms
        var norms = [Float](repeating: 0, count: nFrames)
        for t in 0..<nFrames {
            var sum: Float = 0
            for f in 0..<nFeatures {
                let v = base[f * nFrames + t]
                sum += v * v
            }
            norms[t] = sqrtf(max(sum, 1e-30))
        }

        // Gram matrix
        let gram = UnsafeMutablePointer<Float>.allocate(capacity: nFrames * nFrames)
        defer { gram.deallocate() }

        cblas_sgemm(
            CblasRowMajor, CblasTrans, CblasNoTrans,
            Int32(nFrames), Int32(nFrames), Int32(nFeatures),
            1.0, base, Int32(nFrames),
            base, Int32(nFrames),
            0.0, gram, Int32(nFrames)
        )

        for i in 0..<nFrames {
            for j in 0..<nFrames {
                let cosine = gram[i * nFrames + j] / (norms[i] * norms[j])
                dist[i * nFrames + j] = 1.0 - min(max(cosine, -1.0), 1.0)
            }
        }

        return dist
    }

    /// Compute cross-distance matrix between two feature sequences.
    /// Layout A: [nFeatures, nFramesA], Layout B: [nFeatures, nFramesB].
    /// Output: [nFramesA, nFramesB], row-major.
    private static func computeCrossDistanceMatrix(
        bufA: UnsafeBufferPointer<Float>,
        nFramesA: Int,
        bufB: UnsafeBufferPointer<Float>,
        nFramesB: Int,
        nFeatures: Int,
        metric: Metric
    ) -> [Float] {
        switch metric {
        case .euclidean:
            return crossEuclidean(bufA: bufA, nFramesA: nFramesA,
                                  bufB: bufB, nFramesB: nFramesB,
                                  nFeatures: nFeatures)
        case .cosine:
            return crossCosine(bufA: bufA, nFramesA: nFramesA,
                               bufB: bufB, nFramesB: nFramesB,
                               nFeatures: nFeatures)
        }
    }

    /// Cross Euclidean distance matrix using BLAS.
    private static func crossEuclidean(
        bufA: UnsafeBufferPointer<Float>,
        nFramesA: Int,
        bufB: UnsafeBufferPointer<Float>,
        nFramesB: Int,
        nFeatures: Int
    ) -> [Float] {
        let baseA = bufA.baseAddress!
        let baseB = bufB.baseAddress!

        // Squared norms for A
        var sqNormsA = [Float](repeating: 0, count: nFramesA)
        for t in 0..<nFramesA {
            var sum: Float = 0
            for f in 0..<nFeatures {
                let v = baseA[f * nFramesA + t]
                sum += v * v
            }
            sqNormsA[t] = sum
        }

        // Squared norms for B
        var sqNormsB = [Float](repeating: 0, count: nFramesB)
        for t in 0..<nFramesB {
            var sum: Float = 0
            for f in 0..<nFeatures {
                let v = baseB[f * nFramesB + t]
                sum += v * v
            }
            sqNormsB[t] = sum
        }

        // Cross gram matrix G = A^T B: [nFramesA, nFramesB]
        // A is [nFeatures, nFramesA], B is [nFeatures, nFramesB]
        // G = A^T * B: [nFramesA, nFeatures] x [nFeatures, nFramesB]
        // But A is row-major [nFeatures, nFramesA], so A transposed is [nFramesA, nFeatures]
        // with lda = nFramesA (stride in the original layout).
        let gram = UnsafeMutablePointer<Float>.allocate(capacity: nFramesA * nFramesB)
        defer { gram.deallocate() }

        cblas_sgemm(
            CblasRowMajor, CblasTrans, CblasNoTrans,
            Int32(nFramesA), Int32(nFramesB), Int32(nFeatures),
            1.0, baseA, Int32(nFramesA),
            baseB, Int32(nFramesB),
            0.0, gram, Int32(nFramesB)
        )

        var dist = [Float](repeating: 0, count: nFramesA * nFramesB)
        for i in 0..<nFramesA {
            for j in 0..<nFramesB {
                let d2 = sqNormsA[i] + sqNormsB[j] - 2.0 * gram[i * nFramesB + j]
                dist[i * nFramesB + j] = sqrtf(max(d2, 0))
            }
        }

        return dist
    }

    /// Cross cosine distance matrix.
    private static func crossCosine(
        bufA: UnsafeBufferPointer<Float>,
        nFramesA: Int,
        bufB: UnsafeBufferPointer<Float>,
        nFramesB: Int,
        nFeatures: Int
    ) -> [Float] {
        let baseA = bufA.baseAddress!
        let baseB = bufB.baseAddress!

        // Norms for A
        var normsA = [Float](repeating: 0, count: nFramesA)
        for t in 0..<nFramesA {
            var sum: Float = 0
            for f in 0..<nFeatures {
                let v = baseA[f * nFramesA + t]
                sum += v * v
            }
            normsA[t] = sqrtf(max(sum, 1e-30))
        }

        // Norms for B
        var normsB = [Float](repeating: 0, count: nFramesB)
        for t in 0..<nFramesB {
            var sum: Float = 0
            for f in 0..<nFeatures {
                let v = baseB[f * nFramesB + t]
                sum += v * v
            }
            normsB[t] = sqrtf(max(sum, 1e-30))
        }

        // Cross gram matrix
        let gram = UnsafeMutablePointer<Float>.allocate(capacity: nFramesA * nFramesB)
        defer { gram.deallocate() }

        cblas_sgemm(
            CblasRowMajor, CblasTrans, CblasNoTrans,
            Int32(nFramesA), Int32(nFramesB), Int32(nFeatures),
            1.0, baseA, Int32(nFramesA),
            baseB, Int32(nFramesB),
            0.0, gram, Int32(nFramesB)
        )

        var dist = [Float](repeating: 0, count: nFramesA * nFramesB)
        for i in 0..<nFramesA {
            for j in 0..<nFramesB {
                let cosine = gram[i * nFramesB + j] / (normsA[i] * normsB[j])
                dist[i * nFramesB + j] = 1.0 - min(max(cosine, -1.0), 1.0)
            }
        }

        return dist
    }

    // MARK: - RQA Computation (private)

    /// Compute RQA statistics from a binary recurrence matrix buf of size [n, n].
    private static func computeRQA(
        buf: UnsafeBufferPointer<Float>,
        n: Int,
        lmin: Int,
        vmin: Int
    ) -> RQAResult {
        // Count total recurrence points (excluding main diagonal)
        var totalPoints: Int = 0
        for i in 0..<n {
            for j in 0..<n {
                if i != j && buf[i * n + j] > 0.5 {
                    totalPoints += 1
                }
            }
        }

        let totalOffDiagonal = n * n - n
        let recurrenceRate: Float = totalOffDiagonal > 0
            ? Float(totalPoints) / Float(totalOffDiagonal)
            : 0

        // Diagonal line analysis
        // Scan all diagonals (excluding main diagonal k=0)
        var diagonalLineLengths = [Int]()
        var longestDiag = 0

        // Upper diagonals: k = 1..<n
        for k in 1..<n {
            var currentLength = 0
            let diagLen = n - k
            for d in 0..<diagLen {
                let i = d
                let j = d + k
                if buf[i * n + j] > 0.5 {
                    currentLength += 1
                } else {
                    if currentLength >= lmin {
                        diagonalLineLengths.append(currentLength)
                        if currentLength > longestDiag { longestDiag = currentLength }
                    }
                    currentLength = 0
                }
            }
            if currentLength >= lmin {
                diagonalLineLengths.append(currentLength)
                if currentLength > longestDiag { longestDiag = currentLength }
            }
        }

        // Lower diagonals: k = 1..<n
        for k in 1..<n {
            var currentLength = 0
            let diagLen = n - k
            for d in 0..<diagLen {
                let i = d + k
                let j = d
                if buf[i * n + j] > 0.5 {
                    currentLength += 1
                } else {
                    if currentLength >= lmin {
                        diagonalLineLengths.append(currentLength)
                        if currentLength > longestDiag { longestDiag = currentLength }
                    }
                    currentLength = 0
                }
            }
            if currentLength >= lmin {
                diagonalLineLengths.append(currentLength)
                if currentLength > longestDiag { longestDiag = currentLength }
            }
        }

        // Determinism: fraction of recurrence points in diagonal lines
        let diagLinePoints = diagonalLineLengths.reduce(0, +)
        let determinism: Float = totalPoints > 0
            ? Float(diagLinePoints) / Float(totalPoints)
            : 0

        // Average diagonal line length
        let avgDiagLength: Float = diagonalLineLengths.isEmpty
            ? 0
            : Float(diagLinePoints) / Float(diagonalLineLengths.count)

        // Entropy of diagonal line length distribution
        let entropy: Float
        if diagonalLineLengths.isEmpty {
            entropy = 0
        } else {
            // Build histogram of line lengths
            var histogram = [Int: Int]()
            for len in diagonalLineLengths {
                histogram[len, default: 0] += 1
            }
            let totalLines = Float(diagonalLineLengths.count)
            var ent: Float = 0
            for (_, count) in histogram {
                let p = Float(count) / totalLines
                if p > 0 {
                    ent -= p * log2f(p)
                }
            }
            entropy = ent
        }

        // Vertical line analysis
        var verticalLineLengths = [Int]()
        for j in 0..<n {
            var currentLength = 0
            for i in 0..<n {
                if i != j && buf[i * n + j] > 0.5 {
                    currentLength += 1
                } else {
                    if currentLength >= vmin {
                        verticalLineLengths.append(currentLength)
                    }
                    currentLength = 0
                }
            }
            if currentLength >= vmin {
                verticalLineLengths.append(currentLength)
            }
        }

        let vertLinePoints = verticalLineLengths.reduce(0, +)
        let laminarity: Float = totalPoints > 0
            ? Float(vertLinePoints) / Float(totalPoints)
            : 0

        let avgVertLength: Float = verticalLineLengths.isEmpty
            ? 0
            : Float(vertLinePoints) / Float(verticalLineLengths.count)

        return RQAResult(
            recurrenceRate: recurrenceRate,
            determinism: determinism,
            laminarity: laminarity,
            averageDiagonalLength: avgDiagLength,
            averageVerticalLength: avgVertLength,
            longestDiagonalLine: longestDiag,
            entropy: entropy
        )
    }
}
