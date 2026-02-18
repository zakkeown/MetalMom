import XCTest
@testable import MetalMomCore

final class KeyDetectionTests: XCTestCase {

    // MARK: - Helper

    /// Build activations where one index is dominant.
    private func makeDominant(index: Int, value: Float = 0.9, background: Float = 0.01) -> [Float] {
        var acts = [Float](repeating: background, count: 24)
        acts[index] = value
        return acts
    }

    // MARK: - 1. C Major Detection

    func testCMajorDetection() {
        // Index 3 = C major
        let activations = makeDominant(index: 3)
        let result = KeyDetection.detect(activations: activations)

        XCTAssertEqual(result.keyIndex, 3, "Should detect C major at index 3")
        XCTAssertEqual(result.keyLabel, "C major")
        XCTAssertTrue(result.isMajor)
        XCTAssertEqual(result.pitchClass, 3)
    }

    // MARK: - 2. A Minor Detection

    func testAMinorDetection() {
        // Index 12 = A minor
        let activations = makeDominant(index: 12)
        let result = KeyDetection.detect(activations: activations)

        XCTAssertEqual(result.keyIndex, 12, "Should detect A minor at index 12")
        XCTAssertEqual(result.keyLabel, "A minor")
        XCTAssertFalse(result.isMajor)
        XCTAssertEqual(result.pitchClass, 0)
    }

    // MARK: - 3. Confidence Value

    func testConfidenceValue() {
        let activations = makeDominant(index: 7, value: 0.75, background: 0.02)
        let result = KeyDetection.detect(activations: activations)

        XCTAssertEqual(result.confidence, 0.75, accuracy: 1e-6,
            "Confidence should equal the maximum activation value")
        XCTAssertEqual(result.keyIndex, 7)
    }

    // MARK: - 4. Key Label Mapping

    func testKeyLabelMapping() {
        let expectedLabels = [
            "A major", "A# major", "B major", "C major", "C# major", "D major",
            "D# major", "E major", "F major", "F# major", "G major", "G# major",
            "A minor", "A# minor", "B minor", "C minor", "C# minor", "D minor",
            "D# minor", "E minor", "F minor", "F# minor", "G minor", "G# minor"
        ]

        for i in 0..<24 {
            let label = KeyDetection.label(forIndex: i)
            XCTAssertEqual(label, expectedLabels[i],
                "Index \(i) should map to '\(expectedLabels[i])' but got '\(label)'")
        }
    }

    // MARK: - 5. isMajor Flag

    func testIsMajorFlag() {
        for i in 0..<12 {
            let activations = makeDominant(index: i)
            let result = KeyDetection.detect(activations: activations)
            XCTAssertTrue(result.isMajor,
                "Index \(i) should be major")
        }

        for i in 12..<24 {
            let activations = makeDominant(index: i)
            let result = KeyDetection.detect(activations: activations)
            XCTAssertFalse(result.isMajor,
                "Index \(i) should be minor")
        }
    }

    // MARK: - 6. Pitch Class

    func testPitchClass() {
        // C major = index 3, pitchClass = 3
        let cMajor = makeDominant(index: 3)
        XCTAssertEqual(KeyDetection.detect(activations: cMajor).pitchClass, 3)

        // A major = index 0, pitchClass = 0
        let aMajor = makeDominant(index: 0)
        XCTAssertEqual(KeyDetection.detect(activations: aMajor).pitchClass, 0)

        // G# minor = index 23, pitchClass = 11
        let gsMinor = makeDominant(index: 23)
        XCTAssertEqual(KeyDetection.detect(activations: gsMinor).pitchClass, 11)

        // For all indices, pitchClass should equal index % 12
        for i in 0..<24 {
            let acts = makeDominant(index: i)
            let result = KeyDetection.detect(activations: acts)
            XCTAssertEqual(result.pitchClass, i % 12,
                "Index \(i) should have pitchClass \(i % 12)")
        }
    }

    // MARK: - 7. Sequence Averaging

    func testSequenceAveraging() {
        // 3 frames: first two say C major (index 3), last says A minor (index 12)
        // Average should favor C major since it appears in 2/3 of frames
        var frame1 = [Float](repeating: 0.01, count: 24)
        frame1[3] = 0.9  // C major

        var frame2 = [Float](repeating: 0.01, count: 24)
        frame2[3] = 0.8  // C major

        var frame3 = [Float](repeating: 0.01, count: 24)
        frame3[12] = 0.7  // A minor

        let activations = frame1 + frame2 + frame3

        let result = KeyDetection.detectFromSequence(activations: activations, nFrames: 3)

        // Average for index 3: (0.9 + 0.8 + 0.01) / 3 = 0.57
        // Average for index 12: (0.01 + 0.01 + 0.7) / 3 = 0.24
        XCTAssertEqual(result.keyIndex, 3, "Should detect C major after averaging")
        XCTAssertEqual(result.keyLabel, "C major")

        // Verify the confidence is approximately the averaged value
        let expectedConfidence: Float = (0.9 + 0.8 + 0.01) / 3.0
        XCTAssertEqual(result.confidence, expectedConfidence, accuracy: 1e-5)
    }

    // MARK: - 8. Relative Key

    func testRelativeKey() {
        // C major (index 3) <-> A minor (index 12)
        XCTAssertEqual(KeyDetection.relativeKey(index: 3), 12,
            "Relative minor of C major should be A minor")
        XCTAssertEqual(KeyDetection.relativeKey(index: 12), 3,
            "Relative major of A minor should be C major")

        // G major (index 10) <-> E minor (index 19)
        XCTAssertEqual(KeyDetection.relativeKey(index: 10), 19,
            "Relative minor of G major should be E minor")
        XCTAssertEqual(KeyDetection.relativeKey(index: 19), 10,
            "Relative major of E minor should be G major")

        // A major (index 0) <-> F# minor (index 21)
        XCTAssertEqual(KeyDetection.relativeKey(index: 0), 21,
            "Relative minor of A major should be F# minor")
        XCTAssertEqual(KeyDetection.relativeKey(index: 21), 0,
            "Relative major of F# minor should be A major")

        // D major (index 5) <-> B minor (index 14)
        XCTAssertEqual(KeyDetection.relativeKey(index: 5), 14,
            "Relative minor of D major should be B minor")
        XCTAssertEqual(KeyDetection.relativeKey(index: 14), 5,
            "Relative major of B minor should be D major")
    }

    // MARK: - 9. Uniform Activations

    func testUniformActivations() {
        // All equal: should not crash, and should return a valid index
        let activations = [Float](repeating: 0.041666, count: 24)
        let result = KeyDetection.detect(activations: activations)

        XCTAssertGreaterThanOrEqual(result.keyIndex, 0)
        XCTAssertLessThan(result.keyIndex, 24)
        XCTAssertEqual(result.probabilities.count, 24)
        XCTAssertEqual(result.keyLabel, KeyDetection.keyLabels[result.keyIndex])
    }

    // MARK: - 10. Single Frame

    func testSingleFrame() {
        // nFrames=1 via detectFromSequence should give same result as detect
        let activations = makeDominant(index: 8, value: 0.95)  // F major

        let directResult = KeyDetection.detect(activations: activations)
        let seqResult = KeyDetection.detectFromSequence(activations: activations, nFrames: 1)

        XCTAssertEqual(directResult.keyIndex, seqResult.keyIndex)
        XCTAssertEqual(directResult.keyLabel, seqResult.keyLabel)
        XCTAssertEqual(directResult.confidence, seqResult.confidence, accuracy: 1e-6)
        XCTAssertEqual(directResult.isMajor, seqResult.isMajor)
        XCTAssertEqual(directResult.pitchClass, seqResult.pitchClass)

        XCTAssertEqual(seqResult.keyIndex, 8, "Should detect F major")
        XCTAssertEqual(seqResult.keyLabel, "F major")
    }
}
