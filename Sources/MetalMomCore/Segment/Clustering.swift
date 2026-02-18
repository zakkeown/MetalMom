import Foundation
import Accelerate

/// Agglomerative temporal segmentation.
///
/// Merges adjacent segments bottom-up (by Euclidean distance between centroids)
/// until the target number of segments is reached. Only adjacent segments may
/// be merged, preserving temporal order.
public enum Clustering {

    /// Agglomerative temporal segmentation.
    ///
    /// Merges adjacent segments bottom-up until `nSegments` remain.
    ///
    /// - Parameters:
    ///   - features: Feature matrix [nFeatures, nFrames], row-major.
    ///   - nSegments: Target number of segments (clamped to 1...nFrames).
    /// - Returns: Array of segment boundary frame indices (length nSegments + 1),
    ///   starting at 0 and ending at nFrames.
    public static func agglomerative(
        features: Signal,
        nSegments: Int
    ) -> [Int] {
        precondition(features.shape.count == 2,
                     "Clustering input must be 2D [nFeatures, nFrames]")
        let nFeatures = features.shape[0]
        let nFrames = features.shape[1]

        guard nFeatures > 0 && nFrames > 0 else {
            return [0]
        }

        let k = min(max(nSegments, 1), nFrames)

        // If we want as many segments as frames, every frame is a boundary.
        if k >= nFrames {
            return Array(0...nFrames)
        }

        // Represent each segment as (start, end) half-open interval.
        // Initially every frame is its own segment.
        var segments: [(start: Int, end: Int)] = (0..<nFrames).map { ($0, $0 + 1) }

        // Pre-extract column vectors for fast centroid computation.
        // features is [nFeatures, nFrames] row-major: features[f, t] = data[f * nFrames + t].
        let featureData: [Float] = features.withUnsafeBufferPointer { buf in
            Array(buf)
        }

        // Helper: compute centroid (mean feature vector) for a segment [start, end).
        func centroid(start: Int, end: Int) -> [Float] {
            let len = end - start
            var result = [Float](repeating: 0, count: nFeatures)
            for f in 0..<nFeatures {
                var sum: Float = 0
                for t in start..<end {
                    sum += featureData[f * nFrames + t]
                }
                result[f] = sum / Float(len)
            }
            return result
        }

        // Helper: Euclidean distance between two vectors.
        func euclideanDist(_ a: [Float], _ b: [Float]) -> Float {
            var sumSq: Float = 0
            for i in 0..<a.count {
                let d = a[i] - b[i]
                sumSq += d * d
            }
            return sqrtf(sumSq)
        }

        // Cache centroids for each segment.
        var centroids = segments.map { centroid(start: $0.start, end: $0.end) }

        // Iteratively merge the closest adjacent pair until we reach k segments.
        while segments.count > k {
            // Find the adjacent pair with minimum distance.
            var minDist: Float = .greatestFiniteMagnitude
            var minIdx = 0

            for i in 0..<(segments.count - 1) {
                let d = euclideanDist(centroids[i], centroids[i + 1])
                if d < minDist {
                    minDist = d
                    minIdx = i
                }
            }

            // Merge segment minIdx and minIdx+1.
            let merged = (start: segments[minIdx].start,
                          end: segments[minIdx + 1].end)
            segments[minIdx] = merged
            segments.remove(at: minIdx + 1)

            // Update centroid for the merged segment.
            centroids[minIdx] = centroid(start: merged.start, end: merged.end)
            centroids.remove(at: minIdx + 1)
        }

        // Build boundary array: [seg0.start, seg1.start, ..., last.end].
        var boundaries = segments.map { $0.start }
        boundaries.append(segments.last!.end)
        return boundaries
    }

    /// Agglomerative segmentation returning per-frame segment labels.
    ///
    /// Each frame receives a label from `0..<nSegments`.
    ///
    /// - Parameters:
    ///   - features: Feature matrix [nFeatures, nFrames], row-major.
    ///   - nSegments: Target number of segments.
    /// - Returns: Array of length nFrames with segment labels.
    public static func agglomerativeLabels(
        features: Signal,
        nSegments: Int
    ) -> [Int] {
        let boundaries = agglomerative(features: features, nSegments: nSegments)
        let nFrames = features.shape.count >= 2 ? features.shape[1] : 0

        guard boundaries.count >= 2 else {
            return [Int](repeating: 0, count: nFrames)
        }

        var labels = [Int](repeating: 0, count: nFrames)
        for seg in 0..<(boundaries.count - 1) {
            let start = boundaries[seg]
            let end = boundaries[seg + 1]
            for t in start..<end {
                labels[t] = seg
            }
        }
        return labels
    }
}
