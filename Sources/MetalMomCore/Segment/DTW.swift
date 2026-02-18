import Foundation
import Accelerate

/// Dynamic Time Warping for optimal alignment between two time-series sequences.
///
/// DTW finds the minimum-cost alignment path through a cost matrix by warping
/// the time axis. Commonly used for comparing audio/music sequences of different
/// lengths or tempos.
public enum DTW {

    /// Result of a DTW computation.
    public struct Result {
        /// Accumulated cost matrix, shape [N, M].
        public let accumulatedCost: Signal
        /// Optimal warping path from (0,0) to (N-1, M-1).
        /// Each element is (row, col) in the cost matrix.
        public let warpingPath: [(Int, Int)]
        /// Total alignment cost (value at D[N-1, M-1]).
        public let totalCost: Float
    }

    /// Step pattern for accumulated cost computation.
    public enum StepPattern {
        /// Standard: D[i,j] = C[i,j] + min(D[i-1,j], D[i,j-1], D[i-1,j-1])
        case standard
        /// Symmetric weighted: diagonal costs C[i,j], horizontal/vertical costs 2*C[i,j]
        case symmetric2
    }

    // MARK: - DTW from Pre-Computed Cost Matrix

    /// Compute DTW from a pre-computed cost matrix.
    ///
    /// - Parameters:
    ///   - costMatrix: 2D Signal of shape [N, M] containing local costs.
    ///   - stepPattern: Step pattern for accumulation. Default `.standard`.
    ///   - bandWidth: Sakoe-Chiba band width constraint. `nil` means no constraint.
    /// - Returns: DTW result with accumulated cost, warping path, and total cost.
    public static func compute(
        costMatrix: Signal,
        stepPattern: StepPattern = .standard,
        bandWidth: Int? = nil
    ) -> Result {
        precondition(costMatrix.shape.count == 2,
                     "Cost matrix must be 2D [N, M], got shape \(costMatrix.shape)")
        let n = costMatrix.shape[0]
        let m = costMatrix.shape[1]

        guard n > 0 && m > 0 else {
            return Result(
                accumulatedCost: Signal(data: [], shape: [0, 0], sampleRate: costMatrix.sampleRate),
                warpingPath: [],
                totalCost: 0
            )
        }

        return costMatrix.withUnsafeBufferPointer { buf in
            computeDTW(costBuf: buf, n: n, m: m,
                       stepPattern: stepPattern, bandWidth: bandWidth,
                       sampleRate: costMatrix.sampleRate)
        }
    }

    // MARK: - DTW from Two Feature Matrices

    /// Compute DTW between two feature matrices using Euclidean distance.
    ///
    /// - Parameters:
    ///   - X: 2D Signal of shape [nFeatures, N]. Each column is a feature vector.
    ///   - Y: 2D Signal of shape [nFeatures, M]. Each column is a feature vector.
    ///   - stepPattern: Step pattern for accumulation. Default `.standard`.
    ///   - bandWidth: Sakoe-Chiba band width constraint. `nil` means no constraint.
    /// - Returns: DTW result with accumulated cost, warping path, and total cost.
    public static func compute(
        X: Signal,
        Y: Signal,
        stepPattern: StepPattern = .standard,
        bandWidth: Int? = nil
    ) -> Result {
        precondition(X.shape.count == 2 && Y.shape.count == 2,
                     "Feature matrices must be 2D [nFeatures, nFrames]")
        precondition(X.shape[0] == Y.shape[0],
                     "Feature dimensions must match: \(X.shape[0]) vs \(Y.shape[0])")

        let nFeatures = X.shape[0]
        let nFramesX = X.shape[1]
        let nFramesY = Y.shape[1]

        guard nFeatures > 0 && nFramesX > 0 && nFramesY > 0 else {
            return Result(
                accumulatedCost: Signal(data: [], shape: [0, 0], sampleRate: X.sampleRate),
                warpingPath: [],
                totalCost: 0
            )
        }

        // Compute Euclidean distance cost matrix [nFramesX, nFramesY]
        let costData = X.withUnsafeBufferPointer { bufX in
            Y.withUnsafeBufferPointer { bufY in
                computeEuclideanCostMatrix(
                    bufX: bufX, nFramesX: nFramesX,
                    bufY: bufY, nFramesY: nFramesY,
                    nFeatures: nFeatures
                )
            }
        }

        let costMatrix = Signal(data: costData, shape: [nFramesX, nFramesY],
                                sampleRate: X.sampleRate)
        return compute(costMatrix: costMatrix, stepPattern: stepPattern,
                       bandWidth: bandWidth)
    }

    // MARK: - Private Implementation

    /// Core DTW computation: accumulate cost matrix and backtrack.
    private static func computeDTW(
        costBuf: UnsafeBufferPointer<Float>,
        n: Int,
        m: Int,
        stepPattern: StepPattern,
        bandWidth: Int?,
        sampleRate: Int
    ) -> Result {
        let inf: Float = Float.infinity

        // Accumulated cost matrix D[n, m]
        var D = [Float](repeating: inf, count: n * m)

        // Initialize D[0,0]
        D[0] = costBuf[0]

        // Initialize first column: D[i,0]
        for i in 1..<n {
            // Check band constraint
            if let bw = bandWidth {
                let rowNorm = Float(i) / Float(n)
                let colNorm = Float(0) / Float(m)
                if abs(rowNorm - colNorm) > Float(bw) / Float(max(n, m)) {
                    continue
                }
            }
            let cost = costBuf[i * m]
            switch stepPattern {
            case .standard:
                D[i * m] = D[(i - 1) * m] + cost
            case .symmetric2:
                // Vertical step: weight 2x
                D[i * m] = D[(i - 1) * m] + 2.0 * cost
            }
        }

        // Initialize first row: D[0,j]
        for j in 1..<m {
            if let bw = bandWidth {
                let rowNorm = Float(0) / Float(n)
                let colNorm = Float(j) / Float(m)
                if abs(rowNorm - colNorm) > Float(bw) / Float(max(n, m)) {
                    continue
                }
            }
            let cost = costBuf[j]
            switch stepPattern {
            case .standard:
                D[j] = D[j - 1] + cost
            case .symmetric2:
                // Horizontal step: weight 2x
                D[j] = D[j - 1] + 2.0 * cost
            }
        }

        // Fill the rest of D
        for i in 1..<n {
            for j in 1..<m {
                // Sakoe-Chiba band constraint
                if let bw = bandWidth {
                    let rowNorm = Float(i) / Float(n)
                    let colNorm = Float(j) / Float(m)
                    if abs(rowNorm - colNorm) > Float(bw) / Float(max(n, m)) {
                        continue
                    }
                }

                let cost = costBuf[i * m + j]
                let diag = D[(i - 1) * m + (j - 1)]
                let up = D[(i - 1) * m + j]
                let left = D[i * m + (j - 1)]

                switch stepPattern {
                case .standard:
                    D[i * m + j] = cost + min(diag, up, left)
                case .symmetric2:
                    // Diagonal: 1x weight, horizontal/vertical: 2x weight
                    let diagCost = diag + cost
                    let upCost = up + 2.0 * cost
                    let leftCost = left + 2.0 * cost
                    D[i * m + j] = min(diagCost, upCost, leftCost)
                }
            }
        }

        // Backtrack to find optimal warping path
        let path = backtrack(D: D, n: n, m: m, stepPattern: stepPattern)

        let totalCost = D[(n - 1) * m + (m - 1)]

        let accCost = Signal(data: D, shape: [n, m], sampleRate: sampleRate)

        return Result(accumulatedCost: accCost, warpingPath: path, totalCost: totalCost)
    }

    /// Backtrack from D[N-1, M-1] to D[0,0] to find the optimal warping path.
    private static func backtrack(
        D: [Float],
        n: Int,
        m: Int,
        stepPattern: StepPattern
    ) -> [(Int, Int)] {
        // Handle 1x1 case
        if n == 1 && m == 1 {
            return [(0, 0)]
        }

        var path = [(Int, Int)]()
        var i = n - 1
        var j = m - 1
        path.append((i, j))

        while i > 0 || j > 0 {
            if i == 0 {
                j -= 1
            } else if j == 0 {
                i -= 1
            } else {
                let diag = D[(i - 1) * m + (j - 1)]
                let up = D[(i - 1) * m + j]
                let left = D[i * m + (j - 1)]

                // Pick the predecessor with minimum accumulated cost
                if diag <= up && diag <= left {
                    i -= 1
                    j -= 1
                } else if up <= left {
                    i -= 1
                } else {
                    j -= 1
                }
            }
            path.append((i, j))
        }

        // Reverse to go from (0,0) to (N-1, M-1)
        path.reverse()
        return path
    }

    /// Compute Euclidean distance cost matrix between two feature sequences.
    /// X layout: [nFeatures, nFramesX], Y layout: [nFeatures, nFramesY].
    /// Output: [nFramesX, nFramesY].
    private static func computeEuclideanCostMatrix(
        bufX: UnsafeBufferPointer<Float>,
        nFramesX: Int,
        bufY: UnsafeBufferPointer<Float>,
        nFramesY: Int,
        nFeatures: Int
    ) -> [Float] {
        let baseX = bufX.baseAddress!
        let baseY = bufY.baseAddress!

        // Squared norms for X
        var sqNormsX = [Float](repeating: 0, count: nFramesX)
        for t in 0..<nFramesX {
            var sum: Float = 0
            for f in 0..<nFeatures {
                let v = baseX[f * nFramesX + t]
                sum += v * v
            }
            sqNormsX[t] = sum
        }

        // Squared norms for Y
        var sqNormsY = [Float](repeating: 0, count: nFramesY)
        for t in 0..<nFramesY {
            var sum: Float = 0
            for f in 0..<nFeatures {
                let v = baseY[f * nFramesY + t]
                sum += v * v
            }
            sqNormsY[t] = sum
        }

        // Cross gram matrix G = X^T Y: [nFramesX, nFramesY]
        let gram = UnsafeMutablePointer<Float>.allocate(capacity: nFramesX * nFramesY)
        defer { gram.deallocate() }

        cblas_sgemm(
            CblasRowMajor, CblasTrans, CblasNoTrans,
            Int32(nFramesX), Int32(nFramesY), Int32(nFeatures),
            1.0, baseX, Int32(nFramesX),
            baseY, Int32(nFramesY),
            0.0, gram, Int32(nFramesY)
        )

        var dist = [Float](repeating: 0, count: nFramesX * nFramesY)
        for i in 0..<nFramesX {
            for j in 0..<nFramesY {
                let d2 = sqNormsX[i] + sqNormsY[j] - 2.0 * gram[i * nFramesY + j]
                dist[i * nFramesY + j] = sqrtf(max(d2, 0))
            }
        }

        return dist
    }
}
