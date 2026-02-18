import Foundation
import Metal

/// Wrapper for Metal compute shaders. Loads pipeline states on demand from
/// runtime-compiled shader source (since SPM does not bundle .metal files).
public final class MetalShaders {
    private let device: MTLDevice
    private let library: MTLLibrary
    private let reductionLibrary: MTLLibrary
    private let convolutionLibrary: MTLLibrary
    private var pipelineCache: [String: MTLComputePipelineState] = [:]
    private let lock = NSLock()

    /// Threadgroup size used by reduction kernels. Must match the shared memory
    /// array size declared in the Metal source (currently 256).
    private static let reductionThreadgroupSize: Int = 256

    /// Initialize with a Metal device. Compiles shader source at runtime.
    /// Returns `nil` if shader compilation fails.
    public init?(device: MTLDevice) {
        self.device = device

        // Compile elementwise shaders from embedded source string.
        do {
            self.library = try device.makeLibrary(source: MetalShaderSource.elementwise, options: nil)
        } catch {
            print("MetalShaders: failed to compile elementwise shaders: \(error)")
            return nil
        }

        // Compile reduction shaders from embedded source string.
        do {
            self.reductionLibrary = try device.makeLibrary(source: MetalShaderSource.reduction, options: nil)
        } catch {
            print("MetalShaders: failed to compile reduction shaders: \(error)")
            return nil
        }

        // Compile convolution shaders from embedded source string.
        do {
            self.convolutionLibrary = try device.makeLibrary(source: MetalShaderSource.convolution, options: nil)
        } catch {
            print("MetalShaders: failed to compile convolution shaders: \(error)")
            return nil
        }
    }

    // MARK: - Pipeline Management

    /// Get or create a compute pipeline for a named kernel function.
    /// Searches both the elementwise and reduction libraries.
    public func pipeline(for functionName: String) -> MTLComputePipelineState? {
        lock.lock()
        defer { lock.unlock() }

        if let cached = pipelineCache[functionName] {
            return cached
        }

        // Try elementwise library first, then reduction, then convolution.
        let function = library.makeFunction(name: functionName)
            ?? reductionLibrary.makeFunction(name: functionName)
            ?? convolutionLibrary.makeFunction(name: functionName)

        guard let function = function else {
            print("MetalShaders: no function named '\(functionName)'")
            return nil
        }
        guard let pipeline = try? device.makeComputePipelineState(function: function) else {
            print("MetalShaders: failed to create pipeline for '\(functionName)'")
            return nil
        }
        pipelineCache[functionName] = pipeline
        return pipeline
    }

    // MARK: - Dispatch Helper

    /// Dispatch an elementwise kernel with input/output buffers and optional extra constant buffers.
    ///
    /// Buffer layout:
    ///   - index 0: input float*
    ///   - index 1: output float*
    ///   - index 2: count (UInt32, set via setBytes)
    ///   - index 3+: extra buffers (e.g. amin, exponent, scale, bias)
    public func dispatchElementwise(
        commandBuffer: MTLCommandBuffer,
        pipeline: MTLComputePipelineState,
        inputBuffer: MTLBuffer,
        outputBuffer: MTLBuffer,
        count: Int,
        extraBuffers: [(MTLBuffer, Int)] = []
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(outputBuffer, offset: 0, index: 1)

        var countVal = UInt32(count)
        encoder.setBytes(&countVal, length: MemoryLayout<UInt32>.stride, index: 2)

        for (buffer, index) in extraBuffers {
            encoder.setBuffer(buffer, offset: 0, index: index)
        }

        let threadGroupSize = Swift.min(pipeline.maxTotalThreadsPerThreadgroup, 256)
        let gridSize = MTLSize(width: count, height: 1, depth: 1)
        let threadGroupDim = MTLSize(width: threadGroupSize, height: 1, depth: 1)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupDim)
        encoder.endEncoding()
    }

    // MARK: - Private Helpers

    /// Run a simple elementwise kernel (no extra parameters) and return the result.
    private func runSimple(kernelName: String, input: [Float]) -> [Float]? {
        if input.isEmpty { return [] }
        guard let pipeline = pipeline(for: kernelName) else { return nil }
        return runElementwise(pipeline: pipeline, input: input, count: input.count)
    }

    /// Run an elementwise kernel with one extra float constant at buffer index 3.
    private func runWithOneParam(kernelName: String, input: [Float], param: Float) -> [Float]? {
        if input.isEmpty { return [] }
        guard let pipeline = pipeline(for: kernelName) else { return nil }
        let count = input.count

        let inputBuf = input.withUnsafeBufferPointer { ptr in
            device.makeBuffer(bytes: ptr.baseAddress!, length: count * MemoryLayout<Float>.stride, options: .storageModeShared)
        }
        guard let inputBuf = inputBuf else { return nil }
        guard let outputBuf = device.makeBuffer(length: count * MemoryLayout<Float>.stride, options: .storageModeShared) else { return nil }

        var paramVal = param
        guard let paramBuf = device.makeBuffer(bytes: &paramVal, length: MemoryLayout<Float>.stride, options: .storageModeShared) else { return nil }

        guard let cmdBuf = MetalBackend.shared?.makeCommandBuffer() else { return nil }
        dispatchElementwise(
            commandBuffer: cmdBuf,
            pipeline: pipeline,
            inputBuffer: inputBuf,
            outputBuffer: outputBuf,
            count: count,
            extraBuffers: [(paramBuf, 3)]
        )
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        let resultPtr = outputBuf.contents().bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: resultPtr, count: count))
    }

    /// Run an elementwise kernel with two extra float constants at buffer indices 3 and 4.
    private func runWithTwoParams(kernelName: String, input: [Float], param1: Float, param2: Float) -> [Float]? {
        if input.isEmpty { return [] }
        guard let pipeline = pipeline(for: kernelName) else { return nil }
        let count = input.count

        let inputBuf = input.withUnsafeBufferPointer { ptr in
            device.makeBuffer(bytes: ptr.baseAddress!, length: count * MemoryLayout<Float>.stride, options: .storageModeShared)
        }
        guard let inputBuf = inputBuf else { return nil }
        guard let outputBuf = device.makeBuffer(length: count * MemoryLayout<Float>.stride, options: .storageModeShared) else { return nil }

        var p1 = param1
        var p2 = param2
        guard let paramBuf1 = device.makeBuffer(bytes: &p1, length: MemoryLayout<Float>.stride, options: .storageModeShared) else { return nil }
        guard let paramBuf2 = device.makeBuffer(bytes: &p2, length: MemoryLayout<Float>.stride, options: .storageModeShared) else { return nil }

        guard let cmdBuf = MetalBackend.shared?.makeCommandBuffer() else { return nil }
        dispatchElementwise(
            commandBuffer: cmdBuf,
            pipeline: pipeline,
            inputBuffer: inputBuf,
            outputBuffer: outputBuf,
            count: count,
            extraBuffers: [(paramBuf1, 3), (paramBuf2, 4)]
        )
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        let resultPtr = outputBuf.contents().bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: resultPtr, count: count))
    }

    /// Core helper for simple elementwise kernels (no extra parameters).
    private func runElementwise(pipeline: MTLComputePipelineState, input: [Float], count: Int) -> [Float]? {
        let inputBuf = input.withUnsafeBufferPointer { ptr in
            device.makeBuffer(bytes: ptr.baseAddress!, length: count * MemoryLayout<Float>.stride, options: .storageModeShared)
        }
        guard let inputBuf = inputBuf else { return nil }
        guard let outputBuf = device.makeBuffer(length: count * MemoryLayout<Float>.stride, options: .storageModeShared) else { return nil }

        guard let cmdBuf = MetalBackend.shared?.makeCommandBuffer() else { return nil }
        dispatchElementwise(
            commandBuffer: cmdBuf,
            pipeline: pipeline,
            inputBuffer: inputBuf,
            outputBuffer: outputBuf,
            count: count
        )
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        let resultPtr = outputBuf.contents().bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: resultPtr, count: count))
    }

    // MARK: - Convenience Functions

    /// Elementwise natural log on a float array using Metal.
    /// Returns `nil` if Metal execution fails.
    public func log(_ input: [Float]) -> [Float]? {
        runSimple(kernelName: "elementwise_log", input: input)
    }

    /// Elementwise exp on a float array using Metal.
    public func exp(_ input: [Float]) -> [Float]? {
        runSimple(kernelName: "elementwise_exp", input: input)
    }

    /// Elementwise power: each element raised to `exponent`.
    public func pow(_ input: [Float], exponent: Float) -> [Float]? {
        runWithOneParam(kernelName: "elementwise_pow", input: input, param: exponent)
    }

    /// Elementwise absolute value.
    public func abs(_ input: [Float]) -> [Float]? {
        runSimple(kernelName: "elementwise_abs", input: input)
    }

    /// Amplitude to dB conversion: `20 * log10(max(x, amin))`.
    public func amplitudeToDb(_ input: [Float], amin: Float = 1e-10) -> [Float]? {
        runWithOneParam(kernelName: "amplitude_to_db", input: input, param: amin)
    }

    /// Power to dB conversion: `10 * log10(max(x, amin))`.
    public func powerToDb(_ input: [Float], amin: Float = 1e-10) -> [Float]? {
        runWithOneParam(kernelName: "power_to_db", input: input, param: amin)
    }

    /// Elementwise log1p: `log(1 + x)`.
    public func log1p(_ input: [Float]) -> [Float]? {
        runSimple(kernelName: "elementwise_log1p", input: input)
    }

    /// Elementwise scale and bias: `x * scale + bias`.
    public func scaleBias(_ input: [Float], scale: Float, bias: Float) -> [Float]? {
        runWithTwoParams(kernelName: "elementwise_scale_bias", input: input, param1: scale, param2: bias)
    }

    // MARK: - Reduction Operations

    /// Parallel sum of a float array on GPU.
    /// Returns `nil` if Metal execution fails; returns `0` for empty input.
    public func sum(_ input: [Float]) -> Float? {
        if input.isEmpty { return 0 }
        return runReduction(kernelName: "reduce_sum", input: input, identity: 0) { partials in
            // CPU finish: sum all partial sums.
            var total: Float = 0
            for p in partials { total += p }
            return total
        }
    }

    /// Parallel max of a float array on GPU.
    /// Returns `nil` if Metal execution fails or input is empty.
    public func max(_ input: [Float]) -> Float? {
        if input.isEmpty { return nil }
        return runReduction(kernelName: "reduce_max", input: input, identity: -.infinity) { partials in
            partials.reduce(-Float.infinity) { Swift.max($0, $1) }
        }
    }

    /// Parallel min of a float array on GPU.
    /// Returns `nil` if Metal execution fails or input is empty.
    public func min(_ input: [Float]) -> Float? {
        if input.isEmpty { return nil }
        return runReduction(kernelName: "reduce_min", input: input, identity: .infinity) { partials in
            partials.reduce(Float.infinity) { Swift.min($0, $1) }
        }
    }

    /// Mean of a float array on GPU: `sum / count`.
    /// Returns `nil` if Metal execution fails or input is empty.
    public func mean(_ input: [Float]) -> Float? {
        if input.isEmpty { return nil }
        guard let s = sum(input) else { return nil }
        return s / Float(input.count)
    }

    /// Argmax — index of the maximum value in a float array.
    /// Returns `nil` if Metal execution fails or input is empty.
    public func argmax(_ input: [Float]) -> Int? {
        if input.isEmpty { return nil }
        return runArgmaxReduction(input: input)
    }

    // MARK: - Reduction Dispatch Helpers

    /// Run a scalar reduction kernel (sum, max, min).
    ///
    /// 1. Dispatch the kernel: each threadgroup reduces its chunk via shared memory
    ///    and writes one partial result.
    /// 2. Read the small `partials` array back to CPU.
    /// 3. Call `finish` to combine them into a single scalar.
    private func runReduction(
        kernelName: String,
        input: [Float],
        identity: Float,
        finish: ([Float]) -> Float
    ) -> Float? {
        let count = input.count
        guard let pipeline = pipeline(for: kernelName) else { return nil }

        let tgSize = Self.reductionThreadgroupSize
        let numGroups = (count + tgSize - 1) / tgSize

        // Allocate GPU buffers.
        let inputBuf = input.withUnsafeBufferPointer { ptr in
            device.makeBuffer(bytes: ptr.baseAddress!, length: count * MemoryLayout<Float>.stride, options: .storageModeShared)
        }
        guard let inputBuf = inputBuf else { return nil }
        guard let partialsBuf = device.makeBuffer(length: numGroups * MemoryLayout<Float>.stride, options: .storageModeShared) else { return nil }

        guard let cmdBuf = MetalBackend.shared?.makeCommandBuffer() else { return nil }
        guard let encoder = cmdBuf.makeComputeCommandEncoder() else { return nil }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputBuf, offset: 0, index: 0)
        encoder.setBuffer(partialsBuf, offset: 0, index: 1)
        var countVal = UInt32(count)
        encoder.setBytes(&countVal, length: MemoryLayout<UInt32>.stride, index: 2)

        let gridSize = MTLSize(width: numGroups * tgSize, height: 1, depth: 1)
        let threadGroupDim = MTLSize(width: tgSize, height: 1, depth: 1)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupDim)
        encoder.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        // Read partials back and finish on CPU.
        let partialsPtr = partialsBuf.contents().bindMemory(to: Float.self, capacity: numGroups)
        let partials = Array(UnsafeBufferPointer(start: partialsPtr, count: numGroups))
        return finish(partials)
    }

    /// Run the argmax reduction kernel.
    ///
    /// Like `runReduction` but produces both partial values and partial indices.
    /// The CPU finish finds the overall argmax among the threadgroup winners.
    private func runArgmaxReduction(input: [Float]) -> Int? {
        let count = input.count
        guard let pipeline = pipeline(for: "reduce_argmax") else { return nil }

        let tgSize = Self.reductionThreadgroupSize
        let numGroups = (count + tgSize - 1) / tgSize

        let inputBuf = input.withUnsafeBufferPointer { ptr in
            device.makeBuffer(bytes: ptr.baseAddress!, length: count * MemoryLayout<Float>.stride, options: .storageModeShared)
        }
        guard let inputBuf = inputBuf else { return nil }
        guard let partialValsBuf = device.makeBuffer(length: numGroups * MemoryLayout<Float>.stride, options: .storageModeShared) else { return nil }
        guard let partialIdxBuf = device.makeBuffer(length: numGroups * MemoryLayout<UInt32>.stride, options: .storageModeShared) else { return nil }

        guard let cmdBuf = MetalBackend.shared?.makeCommandBuffer() else { return nil }
        guard let encoder = cmdBuf.makeComputeCommandEncoder() else { return nil }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputBuf, offset: 0, index: 0)
        encoder.setBuffer(partialValsBuf, offset: 0, index: 1)
        encoder.setBuffer(partialIdxBuf, offset: 0, index: 2)
        var countVal = UInt32(count)
        encoder.setBytes(&countVal, length: MemoryLayout<UInt32>.stride, index: 3)

        let gridSize = MTLSize(width: numGroups * tgSize, height: 1, depth: 1)
        let threadGroupDim = MTLSize(width: tgSize, height: 1, depth: 1)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupDim)
        encoder.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        // Read partials and finish on CPU.
        let valsPtr = partialValsBuf.contents().bindMemory(to: Float.self, capacity: numGroups)
        let idxPtr = partialIdxBuf.contents().bindMemory(to: UInt32.self, capacity: numGroups)

        var bestVal: Float = -.infinity
        var bestIdx: UInt32 = 0
        for g in 0..<numGroups {
            if valsPtr[g] > bestVal {
                bestVal = valsPtr[g]
                bestIdx = idxPtr[g]
            }
        }
        return Int(bestIdx)
    }

    // MARK: - Convolution Operations

    /// 1D convolution (valid mode).
    /// Returns array of length `input.count - kernel.count + 1`.
    /// Returns `nil` if Metal execution fails or if `kernel` is longer than `input`.
    public func conv1d(_ input: [Float], kernel: [Float]) -> [Float]? {
        guard !input.isEmpty, !kernel.isEmpty, input.count >= kernel.count else { return nil }
        let outputLen = input.count - kernel.count + 1
        guard let pso = pipeline(for: "conv1d") else { return nil }

        let inputBuf = input.withUnsafeBufferPointer { ptr in
            device.makeBuffer(bytes: ptr.baseAddress!, length: input.count * MemoryLayout<Float>.stride, options: .storageModeShared)
        }
        guard let inputBuf = inputBuf else { return nil }

        let kernelBuf = kernel.withUnsafeBufferPointer { ptr in
            device.makeBuffer(bytes: ptr.baseAddress!, length: kernel.count * MemoryLayout<Float>.stride, options: .storageModeShared)
        }
        guard let kernelBuf = kernelBuf else { return nil }

        guard let outputBuf = device.makeBuffer(length: outputLen * MemoryLayout<Float>.stride, options: .storageModeShared) else { return nil }

        var inputLen = UInt32(input.count)
        var kernelSize = UInt32(kernel.count)

        guard let cmdBuf = MetalBackend.shared?.makeCommandBuffer() else { return nil }
        guard let encoder = cmdBuf.makeComputeCommandEncoder() else { return nil }

        encoder.setComputePipelineState(pso)
        encoder.setBuffer(inputBuf, offset: 0, index: 0)
        encoder.setBuffer(kernelBuf, offset: 0, index: 1)
        encoder.setBuffer(outputBuf, offset: 0, index: 2)
        encoder.setBytes(&inputLen, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.setBytes(&kernelSize, length: MemoryLayout<UInt32>.stride, index: 4)

        let threadGroupSize = Swift.min(pso.maxTotalThreadsPerThreadgroup, 256)
        let gridSize = MTLSize(width: outputLen, height: 1, depth: 1)
        let threadGroupDim = MTLSize(width: threadGroupSize, height: 1, depth: 1)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupDim)
        encoder.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        let resultPtr = outputBuf.contents().bindMemory(to: Float.self, capacity: outputLen)
        return Array(UnsafeBufferPointer(start: resultPtr, count: outputLen))
    }

    /// 1D convolution (same mode).
    /// Returns array of same length as input.
    /// Returns `nil` if Metal execution fails.
    public func conv1dSame(_ input: [Float], kernel: [Float]) -> [Float]? {
        guard !input.isEmpty, !kernel.isEmpty else { return nil }
        let outputLen = input.count
        guard let pso = pipeline(for: "conv1d_same") else { return nil }

        let inputBuf = input.withUnsafeBufferPointer { ptr in
            device.makeBuffer(bytes: ptr.baseAddress!, length: input.count * MemoryLayout<Float>.stride, options: .storageModeShared)
        }
        guard let inputBuf = inputBuf else { return nil }

        let kernelBuf = kernel.withUnsafeBufferPointer { ptr in
            device.makeBuffer(bytes: ptr.baseAddress!, length: kernel.count * MemoryLayout<Float>.stride, options: .storageModeShared)
        }
        guard let kernelBuf = kernelBuf else { return nil }

        guard let outputBuf = device.makeBuffer(length: outputLen * MemoryLayout<Float>.stride, options: .storageModeShared) else { return nil }

        var inputLen = UInt32(input.count)
        var kernelSize = UInt32(kernel.count)

        guard let cmdBuf = MetalBackend.shared?.makeCommandBuffer() else { return nil }
        guard let encoder = cmdBuf.makeComputeCommandEncoder() else { return nil }

        encoder.setComputePipelineState(pso)
        encoder.setBuffer(inputBuf, offset: 0, index: 0)
        encoder.setBuffer(kernelBuf, offset: 0, index: 1)
        encoder.setBuffer(outputBuf, offset: 0, index: 2)
        encoder.setBytes(&inputLen, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.setBytes(&kernelSize, length: MemoryLayout<UInt32>.stride, index: 4)

        let threadGroupSize = Swift.min(pso.maxTotalThreadsPerThreadgroup, 256)
        let gridSize = MTLSize(width: outputLen, height: 1, depth: 1)
        let threadGroupDim = MTLSize(width: threadGroupSize, height: 1, depth: 1)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupDim)
        encoder.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        let resultPtr = outputBuf.contents().bindMemory(to: Float.self, capacity: outputLen)
        return Array(UnsafeBufferPointer(start: resultPtr, count: outputLen))
    }

    /// 2D convolution (valid mode).
    /// Input: `[H, W]` row-major, Kernel: `[kH, kW]` row-major.
    /// Returns `[H-kH+1, W-kW+1]` row-major array.
    /// Returns `nil` if Metal execution fails or dimensions are invalid.
    public func conv2d(
        _ input: [Float], inputH: Int, inputW: Int,
        kernel: [Float], kernelH: Int, kernelW: Int
    ) -> [Float]? {
        guard inputH >= kernelH, inputW >= kernelW else { return nil }
        guard input.count == inputH * inputW else { return nil }
        guard kernel.count == kernelH * kernelW else { return nil }
        guard inputH > 0, inputW > 0, kernelH > 0, kernelW > 0 else { return nil }

        let outH = inputH - kernelH + 1
        let outW = inputW - kernelW + 1
        let outputLen = outH * outW

        guard let pso = pipeline(for: "conv2d") else { return nil }

        let inputBuf = input.withUnsafeBufferPointer { ptr in
            device.makeBuffer(bytes: ptr.baseAddress!, length: input.count * MemoryLayout<Float>.stride, options: .storageModeShared)
        }
        guard let inputBuf = inputBuf else { return nil }

        let kernelBuf = kernel.withUnsafeBufferPointer { ptr in
            device.makeBuffer(bytes: ptr.baseAddress!, length: kernel.count * MemoryLayout<Float>.stride, options: .storageModeShared)
        }
        guard let kernelBuf = kernelBuf else { return nil }

        guard let outputBuf = device.makeBuffer(length: outputLen * MemoryLayout<Float>.stride, options: .storageModeShared) else { return nil }

        var iH = UInt32(inputH)
        var iW = UInt32(inputW)
        var kH = UInt32(kernelH)
        var kW = UInt32(kernelW)

        guard let cmdBuf = MetalBackend.shared?.makeCommandBuffer() else { return nil }
        guard let encoder = cmdBuf.makeComputeCommandEncoder() else { return nil }

        encoder.setComputePipelineState(pso)
        encoder.setBuffer(inputBuf, offset: 0, index: 0)
        encoder.setBuffer(kernelBuf, offset: 0, index: 1)
        encoder.setBuffer(outputBuf, offset: 0, index: 2)
        encoder.setBytes(&iH, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.setBytes(&iW, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&kH, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&kW, length: MemoryLayout<UInt32>.stride, index: 6)

        let gridSize = MTLSize(width: outW, height: outH, depth: 1)
        let threadGroupDim = MTLSize(width: Swift.min(16, outW), height: Swift.min(16, outH), depth: 1)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupDim)
        encoder.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        let resultPtr = outputBuf.contents().bindMemory(to: Float.self, capacity: outputLen)
        return Array(UnsafeBufferPointer(start: resultPtr, count: outputLen))
    }
}
