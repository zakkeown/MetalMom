import XCTest
@testable import MetalMomCore

final class ChunkedInferenceTests: XCTestCase {

    // MARK: - Mock Inference Functions

    /// Identity function: output = input.
    private let identity: ChunkedInference.InferenceFn = { signal in signal }

    /// Scaling function: output = 2 * input.
    private let doubler: ChunkedInference.InferenceFn = { signal in
        let doubled = signal.withUnsafeBufferPointer { buf in
            buf.map { $0 * 2.0 }
        }
        return Signal(data: Array(doubled), shape: signal.shape, sampleRate: signal.sampleRate)
    }

    /// Offset function: output = input + 10. Useful for detecting overlap merge behavior.
    private let offsetter: ChunkedInference.InferenceFn = { signal in
        let offsetData = signal.withUnsafeBufferPointer { buf in
            buf.map { $0 + 10.0 }
        }
        return Signal(data: Array(offsetData), shape: signal.shape, sampleRate: signal.sampleRate)
    }

    // MARK: - Helper

    /// Extract all floats from a Signal.
    private func toArray(_ signal: Signal) -> [Float] {
        signal.withUnsafeBufferPointer { buf in
            Array(buf)
        }
    }

    // MARK: - Properties

    func testOverlapSizeComputed() {
        let chunked = ChunkedInference(chunkSize: 10, hopSize: 8, inferenceFn: identity)
        XCTAssertEqual(chunked.overlapSize, 2)

        let noOverlap = ChunkedInference(chunkSize: 10, hopSize: 10, inferenceFn: identity)
        XCTAssertEqual(noOverlap.overlapSize, 0)

        let maxOverlap = ChunkedInference(chunkSize: 10, hopSize: 1, inferenceFn: identity)
        XCTAssertEqual(maxOverlap.overlapSize, 9)
    }

    // MARK: - No overlap needed (input shorter than chunkSize)

    func testInputShorterThanChunkSize() throws {
        let chunked = ChunkedInference(chunkSize: 100, hopSize: 80, inferenceFn: identity)
        let inputData: [Float] = [1, 2, 3, 4, 5]
        let input = Signal(data: inputData, sampleRate: 22050)
        let output = try chunked.process(input: input)

        XCTAssertEqual(output.shape, [5])
        let result = toArray(output)
        for i in 0..<5 {
            XCTAssertEqual(result[i], inputData[i], accuracy: 1e-6)
        }
    }

    // MARK: - Exact chunks, no overlap

    func testExactChunksNoOverlap() throws {
        // 30 frames, chunkSize=10, hopSize=10 -> 3 exact chunks, no overlap.
        let chunked = ChunkedInference(
            chunkSize: 10, hopSize: 10, mergeStrategy: .overlapAdd, inferenceFn: identity
        )

        let inputData = (0..<30).map { Float($0) }
        let input = Signal(data: inputData, sampleRate: 22050)
        let output = try chunked.process(input: input)

        XCTAssertEqual(output.shape, [30])
        let result = toArray(output)
        for i in 0..<30 {
            XCTAssertEqual(result[i], inputData[i], accuracy: 1e-6,
                "Frame \(i) should be \(inputData[i]), got \(result[i])")
        }
    }

    // MARK: - Overlap-add merge (linear crossfade)

    func testOverlapAddMerge() throws {
        // chunkSize=10, hopSize=8 -> overlap=2.
        // With identity function, input is [0, 1, 2, ..., 17] (18 frames).
        // Chunk 0: frames [0..9], Chunk 1: frames [8..17]
        // Overlap at frames 8-9: crossfade between chunk0[8,9] and chunk1[0,1]
        // Since identity, both chunks have the same values in the overlap region.
        // Crossfade of equal values = same values.
        let chunked = ChunkedInference(
            chunkSize: 10, hopSize: 8, mergeStrategy: .overlapAdd, inferenceFn: identity
        )

        let inputData = (0..<18).map { Float($0) }
        let input = Signal(data: inputData, sampleRate: 22050)
        let output = try chunked.process(input: input)

        XCTAssertEqual(output.shape, [18])
        let result = toArray(output)
        for i in 0..<18 {
            XCTAssertEqual(result[i], inputData[i], accuracy: 1e-4,
                "Frame \(i): expected \(inputData[i]), got \(result[i])")
        }
    }

    func testOverlapAddCrossfadeBehavior() throws {
        // Use a function that outputs different values for different chunk starts,
        // so we can verify crossfade behavior.
        // Strategy: use the doubler. If chunk0 covers [0..9] and chunk1 covers [8..17],
        // then output chunk0 = [0,2,4,...,18] and output chunk1 = [16,18,...,34]
        // At overlap position 0 (frame 8): left=16, right=16, crossfade weight_right=0/2=0, so result=16
        // At overlap position 1 (frame 9): left=18, right=18, crossfade weight_right=1/2=0.5, so result=18
        // Since doubler(x) = 2*x, and overlap region has same input values, crossfade is trivial.
        let chunked = ChunkedInference(
            chunkSize: 10, hopSize: 8, mergeStrategy: .overlapAdd, inferenceFn: doubler
        )

        let inputData = (0..<18).map { Float($0) }
        let input = Signal(data: inputData, sampleRate: 22050)
        let output = try chunked.process(input: input)

        XCTAssertEqual(output.shape, [18])
        let result = toArray(output)
        // All frames should be 2 * input since the overlap values are identical.
        for i in 0..<18 {
            XCTAssertEqual(result[i], Float(i) * 2.0, accuracy: 1e-4,
                "Frame \(i): expected \(Float(i) * 2.0), got \(result[i])")
        }
    }

    // MARK: - Max-pool merge

    func testMaxPoolMerge() throws {
        // With identity on sequential data, overlap values are identical, so max = same.
        let chunked = ChunkedInference(
            chunkSize: 10, hopSize: 8, mergeStrategy: .maxPool, inferenceFn: identity
        )

        let inputData = (0..<18).map { Float($0) }
        let input = Signal(data: inputData, sampleRate: 22050)
        let output = try chunked.process(input: input)

        XCTAssertEqual(output.shape, [18])
        let result = toArray(output)
        for i in 0..<18 {
            XCTAssertEqual(result[i], inputData[i], accuracy: 1e-4,
                "Frame \(i): expected \(inputData[i]), got \(result[i])")
        }
    }

    func testMaxPoolTakesBiggerValue() throws {
        // Use offsetter (input + 10). Chunk0 outputs input+10, chunk1 also outputs input+10.
        // In overlap, both have same values -> max is the same.
        // To really test maxPool, we need chunks to produce different values.
        // Use a function that adds chunk-position-dependent offset.
        var callCount = 0
        let chunkBiased: ChunkedInference.InferenceFn = { signal in
            let bias: Float = callCount == 0 ? 100.0 : 200.0
            callCount += 1
            let biased = signal.withUnsafeBufferPointer { buf in
                buf.map { $0 + bias }
            }
            return Signal(data: Array(biased), shape: signal.shape, sampleRate: signal.sampleRate)
        }

        let chunked = ChunkedInference(
            chunkSize: 10, hopSize: 8, mergeStrategy: .maxPool, inferenceFn: chunkBiased
        )

        let inputData = (0..<18).map { Float($0) }
        let input = Signal(data: inputData, sampleRate: 22050)
        let output = try chunked.process(input: input)

        let result = toArray(output)
        // Non-overlap of chunk0 (frames 0-7): input + 100
        for i in 0..<8 {
            XCTAssertEqual(result[i], Float(i) + 100.0, accuracy: 1e-4)
        }
        // Overlap frames 8-9: max of (input+100) and (input+200). Since input is the same,
        // chunk1 value = input + 200 > chunk0 value = input + 100.
        for i in 8..<10 {
            XCTAssertEqual(result[i], Float(i) + 200.0, accuracy: 1e-4,
                "Overlap frame \(i): expected max = \(Float(i) + 200.0), got \(result[i])")
        }
        // Non-overlap of chunk1 (frames 10-17): input + 200
        for i in 10..<18 {
            XCTAssertEqual(result[i], Float(i) + 200.0, accuracy: 1e-4)
        }
    }

    // MARK: - Average merge

    func testAverageMerge() throws {
        // With identity, overlap has equal values -> average is same.
        let chunked = ChunkedInference(
            chunkSize: 10, hopSize: 8, mergeStrategy: .average, inferenceFn: identity
        )

        let inputData = (0..<18).map { Float($0) }
        let input = Signal(data: inputData, sampleRate: 22050)
        let output = try chunked.process(input: input)

        XCTAssertEqual(output.shape, [18])
        let result = toArray(output)
        for i in 0..<18 {
            XCTAssertEqual(result[i], inputData[i], accuracy: 1e-4)
        }
    }

    func testAverageMergeDifferentValues() throws {
        var callCount = 0
        let chunkBiased: ChunkedInference.InferenceFn = { signal in
            let bias: Float = callCount == 0 ? 0.0 : 10.0
            callCount += 1
            let biased = signal.withUnsafeBufferPointer { buf in
                buf.map { $0 + bias }
            }
            return Signal(data: Array(biased), shape: signal.shape, sampleRate: signal.sampleRate)
        }

        let chunked = ChunkedInference(
            chunkSize: 10, hopSize: 8, mergeStrategy: .average, inferenceFn: chunkBiased
        )

        let inputData = (0..<18).map { Float($0) }
        let input = Signal(data: inputData, sampleRate: 22050)
        let output = try chunked.process(input: input)

        let result = toArray(output)
        // Overlap at frames 8-9: average of (input + 0) and (input + 10)
        for i in 8..<10 {
            let expected = (Float(i) + Float(i) + 10.0) / 2.0
            XCTAssertEqual(result[i], expected, accuracy: 1e-4,
                "Overlap frame \(i): expected avg = \(expected), got \(result[i])")
        }
    }

    // MARK: - 1D input

    func test1DInput() throws {
        let chunked = ChunkedInference(
            chunkSize: 5, hopSize: 5, inferenceFn: identity
        )

        let inputData: [Float] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        let input = Signal(data: inputData, sampleRate: 44100)
        let output = try chunked.process(input: input)

        XCTAssertEqual(output.shape, [10])
        let result = toArray(output)
        for i in 0..<10 {
            XCTAssertEqual(result[i], inputData[i], accuracy: 1e-6)
        }
    }

    // MARK: - 2D input

    func test2DInput() throws {
        // Shape [6, 3] -> 6 frames, 3 features. chunkSize=3, hopSize=3 -> 2 exact chunks.
        let chunked = ChunkedInference(
            chunkSize: 3, hopSize: 3, inferenceFn: identity
        )

        let inputData: [Float] = [
            1, 2, 3,    // frame 0
            4, 5, 6,    // frame 1
            7, 8, 9,    // frame 2
            10, 11, 12, // frame 3
            13, 14, 15, // frame 4
            16, 17, 18  // frame 5
        ]
        let input = Signal(data: inputData, shape: [6, 3], sampleRate: 22050)
        let output = try chunked.process(input: input)

        XCTAssertEqual(output.shape, [6, 3])
        let result = toArray(output)
        for i in 0..<18 {
            XCTAssertEqual(result[i], inputData[i], accuracy: 1e-6,
                "Element \(i): expected \(inputData[i]), got \(result[i])")
        }
    }

    func test2DInputWithOverlap() throws {
        // Shape [8, 2] -> 8 frames, 2 features. chunkSize=5, hopSize=4 -> overlap=1.
        let chunked = ChunkedInference(
            chunkSize: 5, hopSize: 4, mergeStrategy: .overlapAdd, inferenceFn: identity
        )

        let inputData: [Float] = [
            1, 2,    // frame 0
            3, 4,    // frame 1
            5, 6,    // frame 2
            7, 8,    // frame 3
            9, 10,   // frame 4  -- overlap frame
            11, 12,  // frame 5
            13, 14,  // frame 6
            15, 16   // frame 7
        ]
        let input = Signal(data: inputData, shape: [8, 2], sampleRate: 22050)
        let output = try chunked.process(input: input)

        XCTAssertEqual(output.shape, [8, 2])
        let result = toArray(output)
        // With identity, overlap has same values, so crossfade of equal = same.
        for i in 0..<16 {
            XCTAssertEqual(result[i], inputData[i], accuracy: 1e-4,
                "Element \(i): expected \(inputData[i]), got \(result[i])")
        }
    }

    // MARK: - Single frame per chunk

    func testSingleFramePerChunk() throws {
        let chunked = ChunkedInference(
            chunkSize: 1, hopSize: 1, inferenceFn: identity
        )

        let inputData: [Float] = [10, 20, 30, 40, 50]
        let input = Signal(data: inputData, sampleRate: 22050)
        let output = try chunked.process(input: input)

        XCTAssertEqual(output.shape, [5])
        let result = toArray(output)
        for i in 0..<5 {
            XCTAssertEqual(result[i], inputData[i], accuracy: 1e-6)
        }
    }

    // MARK: - Large sequence

    func testLargeSequence() throws {
        let chunked = ChunkedInference(
            chunkSize: 100, hopSize: 80, mergeStrategy: .overlapAdd, inferenceFn: identity
        )

        let inputData = (0..<1000).map { Float($0) * 0.001 }
        let input = Signal(data: inputData, sampleRate: 22050)
        let output = try chunked.process(input: input)

        XCTAssertEqual(output.shape, [1000])
        let result = toArray(output)
        // With identity and overlap-add on equal values, result should match input.
        for i in 0..<1000 {
            XCTAssertEqual(result[i], inputData[i], accuracy: 1e-3,
                "Frame \(i): expected \(inputData[i]), got \(result[i])")
        }
    }

    // MARK: - Doubler preserves correctness

    func testDoublerNoOverlap() throws {
        let chunked = ChunkedInference(
            chunkSize: 5, hopSize: 5, inferenceFn: doubler
        )

        let inputData: [Float] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        let input = Signal(data: inputData, sampleRate: 22050)
        let output = try chunked.process(input: input)

        XCTAssertEqual(output.shape, [10])
        let result = toArray(output)
        for i in 0..<10 {
            XCTAssertEqual(result[i], inputData[i] * 2.0, accuracy: 1e-6,
                "Frame \(i): expected \(inputData[i] * 2.0), got \(result[i])")
        }
    }

    // MARK: - Sample rate preservation

    func testSampleRatePreserved() throws {
        let chunked = ChunkedInference(
            chunkSize: 5, hopSize: 5, inferenceFn: identity
        )

        let input = Signal(data: [Float](repeating: 1.0, count: 10), sampleRate: 48000)
        let output = try chunked.process(input: input)

        XCTAssertEqual(output.sampleRate, 48000)
    }

    // MARK: - Non-divisible input length

    func testNonDivisibleInputLength() throws {
        // 13 frames, chunkSize=5, hopSize=5 -> chunks at [0,5), [5,10), [10,13).
        // Last chunk has only 3 frames.
        let chunked = ChunkedInference(
            chunkSize: 5, hopSize: 5, inferenceFn: identity
        )

        let inputData = (0..<13).map { Float($0) }
        let input = Signal(data: inputData, sampleRate: 22050)
        let output = try chunked.process(input: input)

        XCTAssertEqual(output.shape, [13])
        let result = toArray(output)
        for i in 0..<13 {
            XCTAssertEqual(result[i], inputData[i], accuracy: 1e-6)
        }
    }

    // MARK: - Merge strategy selection

    func testMergeStrategyStoredCorrectly() {
        let oa = ChunkedInference(chunkSize: 10, hopSize: 8, mergeStrategy: .overlapAdd, inferenceFn: identity)
        XCTAssertEqual(oa.mergeStrategy, .overlapAdd)

        let mp = ChunkedInference(chunkSize: 10, hopSize: 8, mergeStrategy: .maxPool, inferenceFn: identity)
        XCTAssertEqual(mp.mergeStrategy, .maxPool)

        let avg = ChunkedInference(chunkSize: 10, hopSize: 8, mergeStrategy: .average, inferenceFn: identity)
        XCTAssertEqual(avg.mergeStrategy, .average)
    }
}

