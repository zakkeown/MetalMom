import Accelerate
import CoreML
import Foundation

// MARK: - Errors

/// Errors specific to ensemble inference.
public enum EnsembleError: Error, Equatable, Sendable {
    /// No engines were provided (ensemble must have at least 1 model).
    case emptyEnsemble
    /// Model outputs have mismatched shapes across the ensemble.
    case shapeMismatch(String)
}

// MARK: - EnsembleRunner

/// Runs inference on an ensemble of CoreML models and combines their outputs.
///
/// madmom uses ensembles of 6-8 identical-architecture models for beat, onset,
/// and downbeat detection. Each model produces a probability sequence, and the
/// ensemble output is the average across all models.
public final class EnsembleRunner {

    /// Strategy for combining ensemble outputs.
    public enum CombineStrategy: Sendable {
        /// Average all model outputs element-wise (default for beat/onset).
        case mean
        /// Take the median across models for each element.
        case median
        /// Take element-wise maximum across models.
        case max
    }

    /// The loaded inference engines.
    public let engines: [InferenceEngine]

    /// The combination strategy.
    public let strategy: CombineStrategy

    // MARK: - Initializers

    /// Initialize with pre-loaded engines.
    ///
    /// - Parameters:
    ///   - engines: Array of inference engines (must not be empty).
    ///   - strategy: How to combine outputs (default: `.mean`).
    /// - Throws: `EnsembleError.emptyEnsemble` if `engines` is empty.
    public init(engines: [InferenceEngine], strategy: CombineStrategy = .mean) throws {
        guard !engines.isEmpty else {
            throw EnsembleError.emptyEnsemble
        }
        self.engines = engines
        self.strategy = strategy
    }

    /// Initialize by loading models from URLs.
    ///
    /// Each URL can be a `.mlmodelc` (compiled) or `.mlmodel` (will be compiled on load).
    ///
    /// - Parameters:
    ///   - modelURLs: URLs to model files (must not be empty).
    ///   - strategy: How to combine outputs (default: `.mean`).
    ///   - computeUnits: Which hardware to use for inference (default: `.all`).
    /// - Throws: `EnsembleError.emptyEnsemble` if `modelURLs` is empty, or model loading errors.
    public init(
        modelURLs: [URL],
        strategy: CombineStrategy = .mean,
        computeUnits: MLComputeUnits = .all
    ) throws {
        guard !modelURLs.isEmpty else {
            throw EnsembleError.emptyEnsemble
        }

        var loaded: [InferenceEngine] = []
        loaded.reserveCapacity(modelURLs.count)

        for url in modelURLs {
            let ext = url.pathExtension
            if ext == "mlmodelc" {
                loaded.append(try InferenceEngine(compiledModelURL: url, computeUnits: computeUnits))
            } else {
                loaded.append(try InferenceEngine(sourceModelURL: url, computeUnits: computeUnits))
            }
        }

        self.engines = loaded
        self.strategy = strategy
    }

    // MARK: - Properties

    /// Number of models in the ensemble.
    public var count: Int { engines.count }

    // MARK: - Prediction

    /// Run ensemble inference on a signal.
    ///
    /// Runs each model independently, then combines outputs using the
    /// configured strategy.
    ///
    /// - Parameters:
    ///   - input: Input signal.
    ///   - inputName: Model input feature name.
    ///   - outputName: Model output feature name.
    /// - Returns: Combined output signal.
    /// - Throws: Inference or ensemble errors.
    public func predict(
        input: Signal,
        inputName: String = "input",
        outputName: String = "output"
    ) throws -> Signal {
        let outputs = try runAllEngines(input: input, inputName: inputName, outputName: outputName)
        return try Self.combine(outputs: outputs, strategy: strategy)
    }

    /// Run ensemble inference and return individual model outputs alongside the combined result.
    ///
    /// Useful for analysis and debugging ensemble behavior.
    ///
    /// - Parameters:
    ///   - input: Input signal.
    ///   - inputName: Model input feature name.
    ///   - outputName: Model output feature name.
    /// - Returns: Tuple of (combined output, individual model outputs).
    /// - Throws: Inference or ensemble errors.
    public func predictWithDetails(
        input: Signal,
        inputName: String = "input",
        outputName: String = "output"
    ) throws -> (combined: Signal, individual: [Signal]) {
        let outputs = try runAllEngines(input: input, inputName: inputName, outputName: outputName)
        let combined = try Self.combine(outputs: outputs, strategy: strategy)
        return (combined: combined, individual: outputs)
    }

    // MARK: - Combination Logic

    /// Combine an array of signals using the specified strategy.
    ///
    /// All signals must have the same shape. The resulting signal inherits the shape
    /// and sample rate of the first signal.
    ///
    /// - Parameters:
    ///   - outputs: Array of signals to combine (must not be empty, must have matching shapes).
    ///   - strategy: How to combine the signals.
    /// - Returns: Combined signal.
    /// - Throws: `EnsembleError` if outputs are empty or have mismatched shapes.
    public static func combine(outputs: [Signal], strategy: CombineStrategy) throws -> Signal {
        guard let first = outputs.first else {
            throw EnsembleError.emptyEnsemble
        }

        let referenceShape = first.shape
        let outputCount = first.count
        let sampleRate = first.sampleRate

        // Validate all outputs have matching shapes.
        for (i, output) in outputs.enumerated() {
            if output.shape != referenceShape {
                throw EnsembleError.shapeMismatch(
                    "Output \(i) shape \(output.shape) doesn't match expected \(referenceShape)")
            }
        }

        // Single model: return a copy directly.
        if outputs.count == 1 {
            var data = [Float](repeating: 0, count: outputCount)
            first.withUnsafeBufferPointer { src in
                data.withUnsafeMutableBufferPointer { dst in
                    dst.baseAddress!.update(from: src.baseAddress!, count: outputCount)
                }
            }
            return Signal(data: data, shape: referenceShape, sampleRate: sampleRate)
        }

        switch strategy {
        case .mean:
            return combineMean(outputs: outputs, count: outputCount, shape: referenceShape, sampleRate: sampleRate)
        case .median:
            return combineMedian(outputs: outputs, count: outputCount, shape: referenceShape, sampleRate: sampleRate)
        case .max:
            return combineMax(outputs: outputs, count: outputCount, shape: referenceShape, sampleRate: sampleRate)
        }
    }

    // MARK: - Private Helpers

    /// Run all engines on the same input and collect outputs.
    private func runAllEngines(
        input: Signal,
        inputName: String,
        outputName: String
    ) throws -> [Signal] {
        var outputs: [Signal] = []
        outputs.reserveCapacity(engines.count)
        for engine in engines {
            let output = try engine.predict(input: input, inputName: inputName, outputName: outputName)
            outputs.append(output)
        }
        return outputs
    }

    /// Element-wise mean using vDSP.
    private static func combineMean(
        outputs: [Signal],
        count: Int,
        shape: [Int],
        sampleRate: Int
    ) -> Signal {
        var result = [Float](repeating: 0, count: count)
        let n = vDSP_Length(count)

        for output in outputs {
            output.withUnsafeBufferPointer { src in
                result.withUnsafeMutableBufferPointer { dst in
                    vDSP_vadd(src.baseAddress!, 1, dst.baseAddress!, 1, dst.baseAddress!, 1, n)
                }
            }
        }

        var divisor = Float(outputs.count)
        result.withUnsafeMutableBufferPointer { dst in
            vDSP_vsdiv(dst.baseAddress!, 1, &divisor, dst.baseAddress!, 1, n)
        }

        return Signal(data: result, shape: shape, sampleRate: sampleRate)
    }

    /// Element-wise median (scalar loop per element, efficient for typical ensemble sizes of 6-8).
    private static func combineMedian(
        outputs: [Signal],
        count: Int,
        shape: [Int],
        sampleRate: Int
    ) -> Signal {
        let modelCount = outputs.count
        var result = [Float](repeating: 0, count: count)
        var column = [Float](repeating: 0, count: modelCount)

        for i in 0..<count {
            for (j, output) in outputs.enumerated() {
                column[j] = output[i]
            }
            column.sort()

            if modelCount % 2 == 1 {
                result[i] = column[modelCount / 2]
            } else {
                result[i] = (column[modelCount / 2 - 1] + column[modelCount / 2]) / 2.0
            }
        }

        return Signal(data: result, shape: shape, sampleRate: sampleRate)
    }

    /// Element-wise maximum using vDSP pairwise reduction.
    private static func combineMax(
        outputs: [Signal],
        count: Int,
        shape: [Int],
        sampleRate: Int
    ) -> Signal {
        let n = vDSP_Length(count)

        // Start with a copy of the first output.
        var result = [Float](repeating: 0, count: count)
        outputs[0].withUnsafeBufferPointer { src in
            result.withUnsafeMutableBufferPointer { dst in
                dst.baseAddress!.update(from: src.baseAddress!, count: count)
            }
        }

        // Pairwise vmax with each subsequent output.
        for i in 1..<outputs.count {
            outputs[i].withUnsafeBufferPointer { src in
                result.withUnsafeMutableBufferPointer { dst in
                    vDSP_vmax(src.baseAddress!, 1, dst.baseAddress!, 1, dst.baseAddress!, 1, n)
                }
            }
        }

        return Signal(data: result, shape: shape, sampleRate: sampleRate)
    }
}
