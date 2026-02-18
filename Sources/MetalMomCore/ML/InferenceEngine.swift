import CoreML
import Foundation

// MARK: - Errors

/// Errors that can occur during CoreML inference.
public enum InferenceError: Error, Equatable, Sendable {
    /// The requested output feature was not found in model predictions.
    case missingOutput(String)
    /// The model file could not be loaded or is invalid.
    case invalidModel(String)
    /// Prediction failed at runtime.
    case predictionFailed(String)

    public static func == (lhs: InferenceError, rhs: InferenceError) -> Bool {
        switch (lhs, rhs) {
        case let (.missingOutput(a), .missingOutput(b)): return a == b
        case let (.invalidModel(a), .invalidModel(b)): return a == b
        case let (.predictionFailed(a), .predictionFailed(b)): return a == b
        default: return false
        }
    }
}

// MARK: - InferenceEngine

/// Loads a CoreML model and runs inference, converting between Signal and
/// MLMultiArray representations.
///
/// Supports both pre-compiled `.mlmodelc` bundles and uncompiled `.mlmodel` /
/// `.mlpackage` files (the latter are compiled on first load).
public final class InferenceEngine: @unchecked Sendable {
    private let model: MLModel

    /// The URL the model was loaded from (for diagnostics).
    public let modelURL: URL

    // MARK: - Initializers

    /// Initialize with a compiled CoreML model bundle (`.mlmodelc`).
    ///
    /// - Parameters:
    ///   - compiledModelURL: URL to a `.mlmodelc` directory.
    ///   - computeUnits: Which hardware to use for inference (default: `.all`).
    /// - Throws: If the model cannot be loaded.
    public init(compiledModelURL: URL, computeUnits: MLComputeUnits = .all) throws {
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        self.model = try MLModel(contentsOf: compiledModelURL, configuration: config)
        self.modelURL = compiledModelURL
    }

    /// Initialize from an uncompiled `.mlmodel` or `.mlpackage`.
    /// The model is compiled on first load (may take a moment).
    ///
    /// - Parameters:
    ///   - sourceModelURL: URL to an `.mlmodel` file or `.mlpackage` bundle.
    ///   - computeUnits: Which hardware to use for inference (default: `.all`).
    /// - Throws: If compilation or loading fails.
    public init(sourceModelURL: URL, computeUnits: MLComputeUnits = .all) throws {
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        let compiledURL = try MLModel.compileModel(at: sourceModelURL)
        self.model = try MLModel(contentsOf: compiledURL, configuration: config)
        self.modelURL = sourceModelURL
    }

    // MARK: - Model introspection

    /// Description of the model's expected inputs.
    public var inputDescription: MLModelDescription {
        model.modelDescription
    }

    /// Names of the model's input features.
    public var inputFeatureNames: Set<String> {
        Set(model.modelDescription.inputDescriptionsByName.keys)
    }

    /// Names of the model's output features.
    public var outputFeatureNames: Set<String> {
        Set(model.modelDescription.outputDescriptionsByName.keys)
    }

    // MARK: - Prediction (Signal API)

    /// Run inference with a Signal as input.
    ///
    /// The Signal's flat data is reshaped into an MLMultiArray according to `signal.shape`.
    ///
    /// - Parameters:
    ///   - input: Input signal whose data and shape are used.
    ///   - inputName: Name of the model's input feature (default: `"input"`).
    ///   - outputName: Name of the model's output feature (default: `"output"`).
    /// - Returns: A new Signal containing the model's output.
    /// - Throws: `InferenceError` on failure.
    public func predict(
        input: Signal,
        inputName: String = "input",
        outputName: String = "output"
    ) throws -> Signal {
        // Build MLMultiArray from the Signal's shape and data.
        let nsShape = input.shape.map { NSNumber(value: $0) }
        let mlArray = try MLMultiArray(shape: nsShape, dataType: .float32)

        // Fast copy via the MLMultiArray's mutable raw pointer.
        let destPtr = mlArray.dataPointer.assumingMemoryBound(to: Float.self)
        input.withUnsafeBufferPointer { src in
            destPtr.update(from: src.baseAddress!, count: input.count)
        }

        // Run prediction.
        let featureProvider = try MLDictionaryFeatureProvider(
            dictionary: [inputName: MLFeatureValue(multiArray: mlArray)]
        )
        let prediction: MLFeatureProvider
        do {
            prediction = try model.prediction(from: featureProvider)
        } catch {
            throw InferenceError.predictionFailed(error.localizedDescription)
        }

        // Extract the requested output.
        guard let outputValue = prediction.featureValue(for: outputName),
              let outputArray = outputValue.multiArrayValue else {
            throw InferenceError.missingOutput(outputName)
        }

        // Convert MLMultiArray → Signal.
        return mlMultiArrayToSignal(outputArray, sampleRate: input.sampleRate)
    }

    // MARK: - Prediction (raw data API)

    /// Run inference with raw float data and an explicit shape.
    ///
    /// - Parameters:
    ///   - data: Flat float array of input values.
    ///   - shape: Shape dimensions (must match `data.count`).
    ///   - inputName: Name of the model's input feature.
    ///   - outputName: Name of the model's output feature.
    ///   - sampleRate: Sample rate carried through to the output Signal.
    /// - Returns: A new Signal containing the model's output.
    public func predict(
        data: [Float],
        shape: [Int],
        inputName: String = "input",
        outputName: String = "output",
        sampleRate: Int = 0
    ) throws -> Signal {
        let signal = Signal(data: data, shape: shape, sampleRate: sampleRate)
        return try predict(input: signal, inputName: inputName, outputName: outputName)
    }

    // MARK: - Multi-output prediction

    /// Run inference and return all output features as a dictionary of Signals.
    ///
    /// Useful for models with more than one output head.
    ///
    /// - Parameters:
    ///   - input: Input signal.
    ///   - inputName: Name of the model's input feature.
    /// - Returns: Dictionary mapping output feature names to Signals.
    public func predictAll(
        input: Signal,
        inputName: String = "input"
    ) throws -> [String: Signal] {
        let nsShape = input.shape.map { NSNumber(value: $0) }
        let mlArray = try MLMultiArray(shape: nsShape, dataType: .float32)

        let destPtr = mlArray.dataPointer.assumingMemoryBound(to: Float.self)
        input.withUnsafeBufferPointer { src in
            destPtr.update(from: src.baseAddress!, count: input.count)
        }

        let featureProvider = try MLDictionaryFeatureProvider(
            dictionary: [inputName: MLFeatureValue(multiArray: mlArray)]
        )
        let prediction: MLFeatureProvider
        do {
            prediction = try model.prediction(from: featureProvider)
        } catch {
            throw InferenceError.predictionFailed(error.localizedDescription)
        }

        var results: [String: Signal] = [:]
        for name in outputFeatureNames {
            if let fv = prediction.featureValue(for: name),
               let arr = fv.multiArrayValue {
                results[name] = mlMultiArrayToSignal(arr, sampleRate: input.sampleRate)
            }
        }
        return results
    }

    // MARK: - Helpers

    /// Convert an MLMultiArray (any rank) to a flat-float Signal.
    ///
    /// Handles both float32 and float64 (Double) output arrays, converting
    /// to Float if necessary.
    private func mlMultiArrayToSignal(_ array: MLMultiArray, sampleRate: Int) -> Signal {
        let shape = (0..<array.shape.count).map { array.shape[$0].intValue }
        let count = shape.reduce(1, *)
        var data = [Float](repeating: 0, count: count)

        switch array.dataType {
        case .float32:
            let srcPtr = array.dataPointer.assumingMemoryBound(to: Float.self)
            data.withUnsafeMutableBufferPointer { dst in
                dst.baseAddress!.update(from: srcPtr, count: count)
            }
        case .double:
            let srcPtr = array.dataPointer.assumingMemoryBound(to: Double.self)
            data.withUnsafeMutableBufferPointer { dst in
                for i in 0..<count {
                    dst[i] = Float(srcPtr[i])
                }
            }
        default:
            // Fallback: use subscript access (slower but type-safe)
            for i in 0..<count {
                data[i] = array[i].floatValue
            }
        }

        return Signal(data: data, shape: shape, sampleRate: sampleRate)
    }
}
