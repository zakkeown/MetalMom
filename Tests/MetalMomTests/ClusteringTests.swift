import XCTest
@testable import MetalMomCore

final class ClusteringTests: XCTestCase {

    // MARK: - Helpers

    /// Create a feature matrix [nFeatures, nFrames] with distinct clusters.
    /// Frames in cluster `c` are centered around `c * separation`.
    private func makeClustered(nFeatures: Int, framesPerCluster: Int,
                               nClusters: Int, separation: Float = 100.0) -> [Float] {
        let nFrames = framesPerCluster * nClusters
        var data = [Float](repeating: 0, count: nFeatures * nFrames)
        for c in 0..<nClusters {
            let center = Float(c) * separation
            for t in 0..<framesPerCluster {
                let frameIdx = c * framesPerCluster + t
                for f in 0..<nFeatures {
                    // Small noise so frames aren't identical
                    let noise = Float(f + t) * 0.01
                    data[f * nFrames + frameIdx] = center + noise
                }
            }
        }
        return data
    }

    /// Create simple sequential features [nFeatures, nFrames].
    private func makeFeatures(nFeatures: Int, nFrames: Int) -> [Float] {
        var data = [Float](repeating: 0, count: nFeatures * nFrames)
        for f in 0..<nFeatures {
            for t in 0..<nFrames {
                data[f * nFrames + t] = Float(f * nFrames + t) * 0.1
            }
        }
        return data
    }

    // MARK: - Boundary Properties

    func testBoundariesStartAtZeroAndEndAtNFrames() {
        let nFeatures = 4
        let nFrames = 20
        let data = makeFeatures(nFeatures: nFeatures, nFrames: nFrames)
        let features = Signal(data: data, shape: [nFeatures, nFrames], sampleRate: 22050)

        let boundaries = Clustering.agglomerative(features: features, nSegments: 3)

        XCTAssertEqual(boundaries.first, 0, "Boundaries must start at 0")
        XCTAssertEqual(boundaries.last, nFrames, "Boundaries must end at nFrames")
    }

    func testNumberOfBoundaries() {
        let nFeatures = 4
        let nFrames = 20
        let data = makeFeatures(nFeatures: nFeatures, nFrames: nFrames)
        let features = Signal(data: data, shape: [nFeatures, nFrames], sampleRate: 22050)

        for nSeg in [1, 2, 5, 10] {
            let boundaries = Clustering.agglomerative(features: features, nSegments: nSeg)
            XCTAssertEqual(boundaries.count, nSeg + 1,
                "Expected \(nSeg + 1) boundaries for \(nSeg) segments, got \(boundaries.count)")
        }
    }

    func testBoundariesAreSortedAndStrictlyIncreasing() {
        let nFeatures = 4
        let nFrames = 20
        let data = makeFeatures(nFeatures: nFeatures, nFrames: nFrames)
        let features = Signal(data: data, shape: [nFeatures, nFrames], sampleRate: 22050)

        let boundaries = Clustering.agglomerative(features: features, nSegments: 4)

        for i in 1..<boundaries.count {
            XCTAssertGreaterThan(boundaries[i], boundaries[i - 1],
                "Boundaries must be strictly increasing: \(boundaries)")
        }
    }

    // MARK: - Extreme Cases

    func testSingleSegment() {
        let nFeatures = 4
        let nFrames = 10
        let data = makeFeatures(nFeatures: nFeatures, nFrames: nFrames)
        let features = Signal(data: data, shape: [nFeatures, nFrames], sampleRate: 22050)

        let boundaries = Clustering.agglomerative(features: features, nSegments: 1)

        XCTAssertEqual(boundaries, [0, nFrames],
            "nSegments=1 should return [0, nFrames]")
    }

    func testNSegmentsEqualsNFrames() {
        let nFeatures = 4
        let nFrames = 5
        let data = makeFeatures(nFeatures: nFeatures, nFrames: nFrames)
        let features = Signal(data: data, shape: [nFeatures, nFrames], sampleRate: 22050)

        let boundaries = Clustering.agglomerative(features: features, nSegments: nFrames)

        // Every frame is its own segment: [0, 1, 2, 3, 4, 5]
        XCTAssertEqual(boundaries, Array(0...nFrames),
            "nSegments=nFrames should return every frame as a boundary")
    }

    func testNSegmentsExceedsNFrames() {
        let nFeatures = 4
        let nFrames = 5
        let data = makeFeatures(nFeatures: nFeatures, nFrames: nFrames)
        let features = Signal(data: data, shape: [nFeatures, nFrames], sampleRate: 22050)

        let boundaries = Clustering.agglomerative(features: features, nSegments: nFrames + 10)

        // Should be clamped to nFrames segments
        XCTAssertEqual(boundaries, Array(0...nFrames),
            "nSegments > nFrames should be clamped")
    }

    // MARK: - Distinct Clusters

    func testDistinctClustersDetected() {
        // Create 3 well-separated clusters of 5 frames each
        let nFeatures = 4
        let framesPerCluster = 5
        let nClusters = 3
        let nFrames = framesPerCluster * nClusters
        let data = makeClustered(nFeatures: nFeatures, framesPerCluster: framesPerCluster,
                                 nClusters: nClusters, separation: 100.0)
        let features = Signal(data: data, shape: [nFeatures, nFrames], sampleRate: 22050)

        let boundaries = Clustering.agglomerative(features: features, nSegments: 3)

        // With well-separated clusters, boundaries should fall at cluster transitions.
        // Cluster boundaries are at frames 5 and 10.
        XCTAssertEqual(boundaries.count, 4, "3 segments means 4 boundaries")
        XCTAssertEqual(boundaries[0], 0)
        XCTAssertEqual(boundaries[3], nFrames)

        // The internal boundaries should be at or near the cluster edges
        XCTAssertEqual(boundaries[1], 5,
            "First internal boundary should be at frame 5 (cluster edge)")
        XCTAssertEqual(boundaries[2], 10,
            "Second internal boundary should be at frame 10 (cluster edge)")
    }

    // MARK: - Labels

    func testLabelsCorrectCount() {
        let nFeatures = 4
        let nFrames = 20
        let nSeg = 4
        let data = makeFeatures(nFeatures: nFeatures, nFrames: nFrames)
        let features = Signal(data: data, shape: [nFeatures, nFrames], sampleRate: 22050)

        let labels = Clustering.agglomerativeLabels(features: features, nSegments: nSeg)

        XCTAssertEqual(labels.count, nFrames,
            "Labels array should have one entry per frame")
    }

    func testLabelsUniqueValues() {
        let nFeatures = 4
        let nFrames = 20
        let nSeg = 4
        let data = makeFeatures(nFeatures: nFeatures, nFrames: nFrames)
        let features = Signal(data: data, shape: [nFeatures, nFrames], sampleRate: 22050)

        let labels = Clustering.agglomerativeLabels(features: features, nSegments: nSeg)
        let unique = Set(labels)

        XCTAssertEqual(unique.count, nSeg,
            "Should have exactly \(nSeg) unique labels, got \(unique.count)")
    }

    func testLabelsNonNegativeAndBounded() {
        let nFeatures = 4
        let nFrames = 20
        let nSeg = 4
        let data = makeFeatures(nFeatures: nFeatures, nFrames: nFrames)
        let features = Signal(data: data, shape: [nFeatures, nFrames], sampleRate: 22050)

        let labels = Clustering.agglomerativeLabels(features: features, nSegments: nSeg)

        for (i, label) in labels.enumerated() {
            XCTAssertGreaterThanOrEqual(label, 0,
                "Label at frame \(i) should be >= 0, got \(label)")
            XCTAssertLessThan(label, nSeg,
                "Label at frame \(i) should be < \(nSeg), got \(label)")
        }
    }

    func testLabelsAreMonotonicallyNonDecreasing() {
        let nFeatures = 4
        let nFrames = 20
        let nSeg = 4
        let data = makeFeatures(nFeatures: nFeatures, nFrames: nFrames)
        let features = Signal(data: data, shape: [nFeatures, nFrames], sampleRate: 22050)

        let labels = Clustering.agglomerativeLabels(features: features, nSegments: nSeg)

        for i in 1..<labels.count {
            XCTAssertGreaterThanOrEqual(labels[i], labels[i - 1],
                "Labels should be monotonically non-decreasing")
        }
    }

    // MARK: - Two Frames

    func testTwoFramesTwoSegments() {
        let nFeatures = 2
        let nFrames = 2
        let data: [Float] = [0, 1, 0, 1] // 2 features x 2 frames
        let features = Signal(data: data, shape: [nFeatures, nFrames], sampleRate: 22050)

        let boundaries = Clustering.agglomerative(features: features, nSegments: 2)

        XCTAssertEqual(boundaries, [0, 1, 2],
            "2 frames, 2 segments: each frame is its own segment")
    }

    func testTwoFramesOneSegment() {
        let nFeatures = 2
        let nFrames = 2
        let data: [Float] = [0, 1, 0, 1]
        let features = Signal(data: data, shape: [nFeatures, nFrames], sampleRate: 22050)

        let boundaries = Clustering.agglomerative(features: features, nSegments: 1)

        XCTAssertEqual(boundaries, [0, 2],
            "2 frames, 1 segment: all merged")
    }
}
