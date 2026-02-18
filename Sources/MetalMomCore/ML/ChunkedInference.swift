import Accelerate
import Foundation

/// Strategies for merging overlapping regions between adjacent chunks.
public enum MergeStrategy {
    /// Linear crossfade in the overlap region (default).
    /// Left chunk fades out linearly, right chunk fades in linearly.
    case overlapAdd
    /// Take the maximum value at each position in the overlap.
    case maxPool
    /// Take the average of overlapping values.
    case average
}

/// Errors specific to chunked inference.
public enum ChunkedInferenceError: Error, Equatable {
    /// chunkSize must be > 0.
    case invalidChunkSize(Int)
    /// hopSize must be > 0 and <= chunkSize.
    case invalidHopSize(Int)
    /// Input signal has unsupported dimensionality (must be 1D or 2D).
    case unsupportedShape([Int])
    /// Inference produced inconsistent output shapes across chunks.
    case inconsistentOutputShapes(String)
}

/// Runs inference on long sequences by splitting into overlapping chunks
/// and merging the results.
///
/// This handles sequences that exceed a model's optimal or maximum input
/// length by:
/// 1. Splitting the input into chunks of `chunkSize` frames
/// 2. Overlapping adjacent chunks by `hopSize` frames (overlap = chunkSize - hopSize)
/// 3. Running inference on each chunk
/// 4. Merging overlapping regions using the specified strategy
public final class ChunkedInference {

    /// The inference callable -- either an InferenceEngine or EnsembleRunner.
    /// We use a closure to support both without tight coupling.
    public typealias InferenceFn = (Signal) throws -> Signal

    /// Number of frames per chunk.
    public let chunkSize: Int

    /// Number of frames to advance between chunks (chunkSize - overlap).
    public let hopSize: Int

    /// Strategy for merging overlapping regions.
    public let mergeStrategy: MergeStrategy

    /// The inference function to run on each chunk.
    private let inferenceFn: InferenceFn

    /// Overlap size in frames.
    public var overlapSize: Int { chunkSize - hopSize }

    /// Initialize chunked inference.
    ///
    /// - Parameters:
    ///   - chunkSize: Number of frames per chunk. Must be > 0.
    ///   - hopSize: Frames to advance between chunks. Must be > 0 and <= chunkSize.
    ///     The overlap is chunkSize - hopSize.
    ///   - mergeStrategy: How to combine overlapping regions.
    ///   - inferenceFn: Closure that runs inference on a single chunk Signal
    ///     and returns the output Signal.
    public init(
        chunkSize: Int,
        hopSize: Int,
        mergeStrategy: MergeStrategy = .overlapAdd,
        inferenceFn: @escaping InferenceFn
    ) {
        precondition(chunkSize > 0, "chunkSize must be > 0")
        precondition(hopSize > 0, "hopSize must be > 0")
        precondition(hopSize <= chunkSize, "hopSize must be <= chunkSize")
        self.chunkSize = chunkSize
        self.hopSize = hopSize
        self.mergeStrategy = mergeStrategy
        self.inferenceFn = inferenceFn
    }

    /// Convenience initializer with an InferenceEngine.
    public convenience init(
        chunkSize: Int,
        hopSize: Int,
        mergeStrategy: MergeStrategy = .overlapAdd,
        engine: InferenceEngine,
        inputName: String = "input",
        outputName: String = "output"
    ) {
        self.init(
            chunkSize: chunkSize,
            hopSize: hopSize,
            mergeStrategy: mergeStrategy,
            inferenceFn: { signal in
                try engine.predict(input: signal, inputName: inputName, outputName: outputName)
            }
        )
    }

    /// Convenience initializer with an EnsembleRunner.
    public convenience init(
        chunkSize: Int,
        hopSize: Int,
        mergeStrategy: MergeStrategy = .overlapAdd,
        ensemble: EnsembleRunner,
        inputName: String = "input",
        outputName: String = "output"
    ) {
        self.init(
            chunkSize: chunkSize,
            hopSize: hopSize,
            mergeStrategy: mergeStrategy,
            inferenceFn: { signal in
                try ensemble.predict(input: signal, inputName: inputName, outputName: outputName)
            }
        )
    }

    /// Run chunked inference on a sequence.
    ///
    /// The input Signal is expected to have shape [nFrames, nFeatures] (2D)
    /// or [nFrames] (1D). It will be split along the first axis (time/frame axis).
    ///
    /// - Parameter input: Full-length input Signal.
    /// - Returns: Full-length output Signal with chunks merged.
    public func process(input: Signal) throws -> Signal {
        // Determine dimensionality and frame count.
        let ndim = input.shape.count
        guard ndim == 1 || ndim == 2 else {
            throw ChunkedInferenceError.unsupportedShape(input.shape)
        }

        let nFrames = input.shape[0]
        let nFeatures = ndim == 2 ? input.shape[1] : 1
        let is2D = ndim == 2

        // Short-circuit: input fits in a single chunk.
        if nFrames <= chunkSize {
            return try inferenceFn(input)
        }

        // Calculate chunk boundaries.
        var chunkStarts: [Int] = []
        var start = 0
        while start < nFrames {
            chunkStarts.append(start)
            start += hopSize
        }

        // Extract and run inference on each chunk.
        var chunkOutputs: [Signal] = []
        chunkOutputs.reserveCapacity(chunkStarts.count)

        for startFrame in chunkStarts {
            let endFrame = min(startFrame + chunkSize, nFrames)
            let chunkFrames = endFrame - startFrame

            // Extract chunk data.
            let chunkData: [Float]
            let chunkShape: [Int]

            if is2D {
                let srcOffset = startFrame * nFeatures
                let srcEnd = endFrame * nFeatures
                chunkData = input.withUnsafeBufferPointer { buf in
                    Array(buf[srcOffset..<srcEnd])
                }
                chunkShape = [chunkFrames, nFeatures]
            } else {
                chunkData = input.withUnsafeBufferPointer { buf in
                    Array(buf[startFrame..<endFrame])
                }
                chunkShape = [chunkFrames]
            }

            let chunkSignal = Signal(data: chunkData, shape: chunkShape, sampleRate: input.sampleRate)
            let output = try inferenceFn(chunkSignal)
            chunkOutputs.append(output)
        }

        // Merge chunk outputs.
        return try mergeChunks(
            chunkOutputs: chunkOutputs,
            chunkStarts: chunkStarts,
            inputFrames: nFrames,
            sampleRate: input.sampleRate
        )
    }

    // MARK: - Private Merge Logic

    /// Merge overlapping chunk outputs into a single contiguous Signal.
    private func mergeChunks(
        chunkOutputs: [Signal],
        chunkStarts: [Int],
        inputFrames: Int,
        sampleRate: Int
    ) throws -> Signal {
        guard let firstOutput = chunkOutputs.first else {
            // Should never happen since we checked nFrames > 0 above.
            return Signal(data: [], shape: [0], sampleRate: sampleRate)
        }

        // Determine output dimensionality from the first chunk's output.
        let outNdim = firstOutput.shape.count
        let outFeatures: Int
        if outNdim == 2 {
            outFeatures = firstOutput.shape[1]
        } else if outNdim == 1 {
            outFeatures = 1
        } else {
            throw ChunkedInferenceError.inconsistentOutputShapes(
                "Output has unsupported dimensionality: \(firstOutput.shape)")
        }

        // Compute the ratio of output frames to input frames from the first chunk.
        // This handles cases where the model changes the frame count (e.g., downsampling).
        let firstInputChunkFrames = min(chunkSize, inputFrames)
        let firstOutputFrames = firstOutput.shape[0]
        let frameRatio = Double(firstOutputFrames) / Double(firstInputChunkFrames)

        // Total output frames.
        let totalOutputFrames = Int(round(Double(inputFrames) * frameRatio))

        // Compute output overlap size.
        let outputOverlap = overlapSize > 0
            ? Int(round(Double(overlapSize) * frameRatio))
            : 0
        // No overlap: just concatenate.
        if outputOverlap == 0 {
            return concatenateChunks(
                chunkOutputs: chunkOutputs,
                totalFrames: totalOutputFrames,
                nFeatures: outFeatures,
                ndim: outNdim,
                sampleRate: sampleRate
            )
        }

        // Allocate output buffer.
        let totalElements = totalOutputFrames * outFeatures
        var result = [Float](repeating: 0, count: totalElements)

        // Track how far we have written (in output frames).
        var writePos = 0

        for (i, output) in chunkOutputs.enumerated() {
            let outFrames = output.shape[0]

            if i == 0 {
                // First chunk: copy entirely.
                output.withUnsafeBufferPointer { buf in
                    let count = min(outFrames * outFeatures, totalElements)
                    result.withUnsafeMutableBufferPointer { dst in
                        dst.baseAddress!.update(from: buf.baseAddress!, count: count)
                    }
                }
                writePos = outFrames
            } else {
                // Subsequent chunks: merge overlap region, then copy non-overlap region.
                let overlapStart = writePos - outputOverlap
                let actualOverlap = min(outputOverlap, writePos, outFrames)

                if actualOverlap > 0 {
                    // Merge the overlap region.
                    mergeOverlapRegion(
                        result: &result,
                        resultOffset: overlapStart * outFeatures,
                        newChunk: output,
                        chunkOffset: 0,
                        overlapFrames: actualOverlap,
                        nFeatures: outFeatures
                    )
                }

                // Copy the non-overlap portion of this chunk.
                let nonOverlapStart = actualOverlap
                let nonOverlapFrames = outFrames - nonOverlapStart
                if nonOverlapFrames > 0 {
                    let dstOffset = (overlapStart + actualOverlap) * outFeatures
                    let srcOffset = nonOverlapStart * outFeatures
                    let count = min(
                        nonOverlapFrames * outFeatures,
                        totalElements - dstOffset
                    )
                    if count > 0 {
                        output.withUnsafeBufferPointer { buf in
                            result.withUnsafeMutableBufferPointer { dst in
                                dst.baseAddress!.advanced(by: dstOffset)
                                    .update(from: buf.baseAddress!.advanced(by: srcOffset), count: count)
                            }
                        }
                    }
                }

                writePos = overlapStart + outFrames
            }
        }

        // Trim or pad to exact output frame count.
        let finalCount = totalOutputFrames * outFeatures
        if result.count > finalCount {
            result = Array(result.prefix(finalCount))
        }

        let outShape: [Int] = outNdim == 2
            ? [totalOutputFrames, outFeatures]
            : [totalOutputFrames]

        return Signal(data: result, shape: outShape, sampleRate: sampleRate)
    }

    /// Concatenate chunk outputs without overlap merging.
    private func concatenateChunks(
        chunkOutputs: [Signal],
        totalFrames: Int,
        nFeatures: Int,
        ndim: Int,
        sampleRate: Int
    ) -> Signal {
        let totalElements = totalFrames * nFeatures
        var result = [Float](repeating: 0, count: totalElements)
        var offset = 0

        for output in chunkOutputs {
            let count = min(output.count, totalElements - offset)
            if count <= 0 { break }
            output.withUnsafeBufferPointer { buf in
                result.withUnsafeMutableBufferPointer { dst in
                    dst.baseAddress!.advanced(by: offset)
                        .update(from: buf.baseAddress!, count: count)
                }
            }
            offset += count
        }

        let outShape: [Int] = ndim == 2
            ? [totalFrames, nFeatures]
            : [totalFrames]

        return Signal(data: result, shape: outShape, sampleRate: sampleRate)
    }

    /// Merge the overlap region between the existing result and a new chunk.
    private func mergeOverlapRegion(
        result: inout [Float],
        resultOffset: Int,
        newChunk: Signal,
        chunkOffset: Int,
        overlapFrames: Int,
        nFeatures: Int
    ) {
        let overlapElements = overlapFrames * nFeatures

        switch mergeStrategy {
        case .overlapAdd:
            // Linear crossfade: left fades out, right fades in.
            newChunk.withUnsafeBufferPointer { buf in
                for f in 0..<overlapFrames {
                    let weightRight = Float(f) / Float(overlapFrames)
                    let weightLeft = 1.0 - weightRight
                    for feat in 0..<nFeatures {
                        let idx = f * nFeatures + feat
                        let resultIdx = resultOffset + idx
                        let chunkIdx = chunkOffset * nFeatures + idx
                        result[resultIdx] = weightLeft * result[resultIdx] + weightRight * buf[chunkIdx]
                    }
                }
            }

        case .maxPool:
            newChunk.withUnsafeBufferPointer { buf in
                for i in 0..<overlapElements {
                    let resultIdx = resultOffset + i
                    let chunkIdx = chunkOffset * nFeatures + i
                    result[resultIdx] = max(result[resultIdx], buf[chunkIdx])
                }
            }

        case .average:
            newChunk.withUnsafeBufferPointer { buf in
                for i in 0..<overlapElements {
                    let resultIdx = resultOffset + i
                    let chunkIdx = chunkOffset * nFeatures + i
                    result[resultIdx] = (result[resultIdx] + buf[chunkIdx]) / 2.0
                }
            }
        }
    }
}
