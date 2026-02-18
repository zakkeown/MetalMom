import Foundation
import Accelerate

/// Nearest-neighbor spectrogram filter.
///
/// Replaces each frame of a spectrogram with the average (or median) of its
/// k nearest neighbors, enhancing repeating structures while suppressing
/// non-repeating events. This is useful for background/foreground separation
/// (e.g., separating vocals from accompaniment).
///
/// Matches the functionality of `librosa.decompose.nn_filter`.
public enum NNFilter {

    /// Distance metric for nearest-neighbor computation.
    public enum DistanceMetric {
        /// Euclidean (L2) distance between frames.
        case euclidean
        /// Cosine distance: 1 - cosine_similarity.
        case cosine
    }

    /// Aggregation method for combining nearest neighbors.
    public enum Aggregate {
        /// Arithmetic mean of neighbors.
        case mean
        /// Element-wise median of neighbors.
        case median
    }

    // MARK: - Public API

    /// Nearest-neighbor filter for spectrograms.
    ///
    /// Replaces each frame with the aggregation of its k nearest neighbors,
    /// preserving repeating structure and smoothing transient events.
    ///
    /// - Parameters:
    ///   - spectrogram: Input 2D Signal, shape [nFeatures, nFrames].
    ///   - k: Number of nearest neighbors. Default 10. Clamped to available frames.
    ///   - metric: Distance metric for finding neighbors. Default `.cosine`.
    ///   - aggregate: Aggregation method. Default `.mean`.
    ///   - excludeSelf: If true, exclude the frame itself from its neighbor set. Default true.
    /// - Returns: Filtered spectrogram, same shape [nFeatures, nFrames].
    public static func filter(
        _ spectrogram: Signal,
        k: Int = 10,
        metric: DistanceMetric = .cosine,
        aggregate: Aggregate = .mean,
        excludeSelf: Bool = true
    ) -> Signal {
        precondition(spectrogram.shape.count == 2,
                     "NNFilter input must be 2D [nFeatures, nFrames]")
        let nFeatures = spectrogram.shape[0]
        let nFrames = spectrogram.shape[1]

        guard nFeatures > 0 && nFrames > 0 else {
            return Signal(data: [], shape: spectrogram.shape,
                          sampleRate: spectrogram.sampleRate)
        }

        // Single frame: return a copy
        if nFrames == 1 {
            var data = [Float](repeating: 0, count: nFeatures)
            spectrogram.withUnsafeBufferPointer { buf in
                for i in 0..<nFeatures { data[i] = buf[i] }
            }
            return Signal(data: data, shape: [nFeatures, 1],
                          sampleRate: spectrogram.sampleRate)
        }

        // Clamp k to the number of available neighbor candidates
        let maxK = excludeSelf ? nFrames - 1 : nFrames
        let effectiveK = min(max(k, 1), maxK)

        // Step 1: Compute kNN indices
        let indices = computeNNIndices(spectrogram, nFeatures: nFeatures,
                                        nFrames: nFrames, k: effectiveK,
                                        metric: metric, excludeSelf: excludeSelf)

        // Step 2: Aggregate neighbors
        var result = [Float](repeating: 0, count: nFeatures * nFrames)
        spectrogram.withUnsafeBufferPointer { buf in
            switch aggregate {
            case .mean:
                aggregateMean(buf: buf, indices: indices,
                              nFeatures: nFeatures, nFrames: nFrames,
                              k: effectiveK, result: &result)
            case .median:
                aggregateMedian(buf: buf, indices: indices,
                                nFeatures: nFeatures, nFrames: nFrames,
                                k: effectiveK, result: &result)
            }
        }

        return Signal(data: result, shape: [nFeatures, nFrames],
                       sampleRate: spectrogram.sampleRate)
    }

    /// Compute k-nearest-neighbor indices for each frame.
    ///
    /// - Parameters:
    ///   - spectrogram: Input 2D Signal, shape [nFeatures, nFrames].
    ///   - k: Number of nearest neighbors. Default 10. Clamped to available frames.
    ///   - metric: Distance metric. Default `.cosine`.
    ///   - excludeSelf: Exclude self from neighbor set. Default true.
    /// - Returns: Array of nFrames arrays, each containing k neighbor indices
    ///   sorted by distance (nearest first).
    public static func nearestNeighborIndices(
        _ spectrogram: Signal,
        k: Int = 10,
        metric: DistanceMetric = .cosine,
        excludeSelf: Bool = true
    ) -> [[Int]] {
        precondition(spectrogram.shape.count == 2,
                     "NNFilter input must be 2D [nFeatures, nFrames]")
        let nFeatures = spectrogram.shape[0]
        let nFrames = spectrogram.shape[1]

        guard nFeatures > 0 && nFrames > 0 else {
            return []
        }

        let maxK = excludeSelf ? max(nFrames - 1, 1) : nFrames
        let effectiveK = min(max(k, 1), maxK)

        let flatIndices = computeNNIndices(spectrogram, nFeatures: nFeatures,
                                            nFrames: nFrames, k: effectiveK,
                                            metric: metric, excludeSelf: excludeSelf)

        // Convert flat [k * nFrames] to [[Int]] of nFrames arrays, each of size k
        var result = [[Int]](repeating: [], count: nFrames)
        for t in 0..<nFrames {
            var neighbors = [Int](repeating: 0, count: effectiveK)
            for i in 0..<effectiveK {
                neighbors[i] = flatIndices[i * nFrames + t]
            }
            result[t] = neighbors
        }
        return result
    }

    // MARK: - Internal: kNN Index Computation

    /// Compute kNN indices stored as flat array [k, nFrames] (row-major).
    /// result[i * nFrames + t] = index of the i-th nearest neighbor of frame t.
    private static func computeNNIndices(
        _ spectrogram: Signal,
        nFeatures: Int,
        nFrames: Int,
        k: Int,
        metric: DistanceMetric,
        excludeSelf: Bool
    ) -> [Int] {
        // Compute pairwise distance matrix [nFrames, nFrames]
        let distances = spectrogram.withUnsafeBufferPointer { buf in
            computeDistanceMatrix(buf: buf, nFeatures: nFeatures,
                                  nFrames: nFrames, metric: metric)
        }

        // For each frame, find k nearest neighbors
        var indices = [Int](repeating: 0, count: k * nFrames)

        for t in 0..<nFrames {
            // Build (distance, index) pairs for sorting
            var pairs = [(Float, Int)]()
            pairs.reserveCapacity(nFrames)
            for j in 0..<nFrames {
                if excludeSelf && j == t { continue }
                pairs.append((distances[t * nFrames + j], j))
            }

            // Partial sort: we only need the k smallest
            pairs.sort { $0.0 < $1.0 }

            let count = min(k, pairs.count)
            for i in 0..<count {
                indices[i * nFrames + t] = pairs[i].1
            }
            // If count < k (shouldn't happen with proper clamping), fill rest with self
            for i in count..<k {
                indices[i * nFrames + t] = t
            }
        }

        return indices
    }

    // MARK: - Distance Matrix

    /// Compute pairwise distance matrix between frames.
    /// Returns flat [nFrames * nFrames] array, row-major.
    private static func computeDistanceMatrix(
        buf: UnsafeBufferPointer<Float>,
        nFeatures: Int,
        nFrames: Int,
        metric: DistanceMetric
    ) -> [Float] {
        switch metric {
        case .euclidean:
            return euclideanDistanceMatrix(buf: buf, nFeatures: nFeatures, nFrames: nFrames)
        case .cosine:
            return cosineDistanceMatrix(buf: buf, nFeatures: nFeatures, nFrames: nFrames)
        }
    }

    /// Euclidean distance matrix: d(i,j) = ||frame_i - frame_j||_2
    private static func euclideanDistanceMatrix(
        buf: UnsafeBufferPointer<Float>,
        nFeatures: Int,
        nFrames: Int
    ) -> [Float] {
        let base = buf.baseAddress!
        var dist = [Float](repeating: 0, count: nFrames * nFrames)

        // Precompute squared norms of each frame
        var sqNorms = [Float](repeating: 0, count: nFrames)
        for t in 0..<nFrames {
            var sum: Float = 0
            // Frame t is elements [f * nFrames + t] for f in 0..<nFeatures
            for f in 0..<nFeatures {
                let v = base[f * nFrames + t]
                sum += v * v
            }
            sqNorms[t] = sum
        }

        // Compute gram matrix G[i,j] = sum_f frame_i[f] * frame_j[f]
        // Frames are stored column-wise in row-major 2D: S[f, t] = buf[f * nFrames + t]
        // We need to extract frame vectors and compute dot products efficiently.
        //
        // d(i,j)^2 = ||i||^2 + ||j||^2 - 2 * dot(i, j)
        // Use cblas for the gram matrix: G = X^T X where X is [nFeatures, nFrames]
        let gram = UnsafeMutablePointer<Float>.allocate(capacity: nFrames * nFrames)
        defer { gram.deallocate() }

        // X^T X: [nFrames, nFrames] = [nFrames, nFeatures] x [nFeatures, nFrames]
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

    /// Cosine distance matrix: d(i,j) = 1 - cos(frame_i, frame_j)
    private static func cosineDistanceMatrix(
        buf: UnsafeBufferPointer<Float>,
        nFeatures: Int,
        nFrames: Int
    ) -> [Float] {
        let base = buf.baseAddress!
        var dist = [Float](repeating: 0, count: nFrames * nFrames)

        // Compute L2 norms of each frame
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
                // Clamp to [-1, 1] to handle numerical errors
                dist[i * nFrames + j] = 1.0 - min(max(cosine, -1.0), 1.0)
            }
        }

        return dist
    }

    // MARK: - Aggregation

    /// Mean aggregation: result[:,t] = mean(S[:, neighbors[t]])
    private static func aggregateMean(
        buf: UnsafeBufferPointer<Float>,
        indices: [Int],
        nFeatures: Int,
        nFrames: Int,
        k: Int,
        result: inout [Float]
    ) {
        let invK: Float = 1.0 / Float(k)
        let base = buf.baseAddress!

        for t in 0..<nFrames {
            for f in 0..<nFeatures {
                var sum: Float = 0
                for i in 0..<k {
                    let neighborIdx = indices[i * nFrames + t]
                    sum += base[f * nFrames + neighborIdx]
                }
                result[f * nFrames + t] = sum * invK
            }
        }
    }

    /// Median aggregation: result[f,t] = median of S[f, neighbors[t]]
    private static func aggregateMedian(
        buf: UnsafeBufferPointer<Float>,
        indices: [Int],
        nFeatures: Int,
        nFrames: Int,
        k: Int,
        result: inout [Float]
    ) {
        let base = buf.baseAddress!
        var scratch = [Float](repeating: 0, count: k)

        for t in 0..<nFrames {
            for f in 0..<nFeatures {
                for i in 0..<k {
                    let neighborIdx = indices[i * nFrames + t]
                    scratch[i] = base[f * nFrames + neighborIdx]
                }
                scratch.sort()
                let median: Float
                if k % 2 == 0 {
                    median = (scratch[k / 2 - 1] + scratch[k / 2]) / 2.0
                } else {
                    median = scratch[k / 2]
                }
                result[f * nFrames + t] = median
            }
        }
    }
}
