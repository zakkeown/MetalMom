import Foundation
import Accelerate
import Metal
import MetalPerformanceShadersGraph

// MARK: - Input / Output Types

/// Input parameters for the STFT computation.
public struct STFTInput {
    public let signal: Signal
    public let nFFT: Int
    public let hopLength: Int
    public let winLength: Int
    public let center: Bool

    public init(signal: Signal, nFFT: Int, hopLength: Int, winLength: Int, center: Bool) {
        self.signal = signal
        self.nFFT = nFFT
        self.hopLength = hopLength
        self.winLength = winLength
        self.center = center
    }
}

/// Output of the STFT computation.
public struct STFTOutput {
    /// Magnitude spectrogram with shape [nFreqs, nFrames], row-major (C-order).
    public let magnitude: Signal
}

// MARK: - Helpers

/// Returns true if `n` is a power of two (and positive).
@inline(__always)
private func isPowerOfTwo(_ n: Int) -> Bool {
    n > 0 && (n & (n - 1)) == 0
}

/// Prepare a full-length window of exactly `nFFT` samples.
///
/// - If `winLength < nFFT`: zero-pad the window symmetrically to nFFT.
/// - If `winLength == nFFT`: use the window as-is.
/// - If `winLength > nFFT`: truncate the window to the center nFFT samples.
private func prepareWindow(winLength: Int, nFFT: Int) -> [Float] {
    let window = Windows.hann(length: winLength, periodic: true)
    if winLength < nFFT {
        let padBefore = (nFFT - winLength) / 2
        let padAfter = nFFT - winLength - padBefore
        return [Float](repeating: 0, count: padBefore) + window + [Float](repeating: 0, count: padAfter)
    } else if winLength > nFFT {
        // Truncate: take the center nFFT samples of the window
        let start = (winLength - nFFT) / 2
        return Array(window[start..<(start + nFFT)])
    } else {
        return window
    }
}

// MARK: - STFT ComputeOperation

/// Short-Time Fourier Transform using Accelerate/vDSP.
public struct STFT: ComputeOperation {
    public typealias Input = STFTInput
    public typealias Output = STFTOutput

    /// Minimum data size to prefer GPU over CPU. Uses ChipProfile when Metal is available.
    public static var dispatchThreshold: Int {
        MetalBackend.shared?.chipProfile.threshold(for: .stft) ?? Int.max
    }

    public init() {}

    public func executeCPU(_ input: STFTInput) -> STFTOutput {
        let nFFT = input.nFFT
        let hopLength = input.hopLength
        let winLength = input.winLength
        let nFreqs = nFFT / 2 + 1

        // --- 1. Optionally pad the signal ---
        let padded: [Float]
        if input.center {
            let padAmount = nFFT / 2
            padded = [Float](repeating: 0, count: padAmount)
                   + input.signal.withUnsafeBufferPointer { Array($0) }
                   + [Float](repeating: 0, count: padAmount)
        } else {
            padded = input.signal.withUnsafeBufferPointer { Array($0) }
        }

        let paddedLength = padded.count

        // --- 2. Compute number of frames ---
        guard paddedLength >= nFFT else {
            // Signal too short for even one frame
            let out = Signal(data: [], shape: [nFreqs, 0], sampleRate: input.signal.sampleRate)
            return STFTOutput(magnitude: out)
        }
        let nFrames = 1 + (paddedLength - nFFT) / hopLength

        // --- 3. Prepare window ---
        let fullWindow = prepareWindow(winLength: winLength, nFFT: nFFT)

        // --- 4. Set up vDSP FFT ---
        guard isPowerOfTwo(nFFT) else {
            // vDSP requires power-of-2 FFT sizes; return empty result
            let out = Signal(data: [], shape: [nFreqs, 0], sampleRate: input.signal.sampleRate)
            return STFTOutput(magnitude: out)
        }
        let log2n = vDSP_Length(log2(Double(nFFT)))
        guard let fftSetup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else {
            let out = Signal(data: [], shape: [nFreqs, 0], sampleRate: input.signal.sampleRate)
            return STFTOutput(magnitude: out)
        }
        defer { vDSP_destroy_fftsetup(fftSetup) }

        // --- 5. Allocate output buffer (row-major: [nFreqs, nFrames]) ---
        // We compute into a column-major temp buffer (fast per-frame writes),
        // then transpose to row-major at the end for NumPy/C-order compatibility.
        let totalElements = nFreqs * nFrames
        let tempPtr = UnsafeMutablePointer<Float>.allocate(capacity: totalElements)
        tempPtr.initialize(repeating: 0, count: totalElements)

        // Temporary buffer for each frame's FFT
        // vDSP_fft_zrip works on nFFT/2 complex pairs
        let halfN = nFFT / 2
        let realPart = UnsafeMutablePointer<Float>.allocate(capacity: halfN)
        let imagPart = UnsafeMutablePointer<Float>.allocate(capacity: halfN)
        defer {
            realPart.deallocate()
            imagPart.deallocate()
        }

        // Temporary buffer for the windowed frame
        let windowedFrame = UnsafeMutablePointer<Float>.allocate(capacity: nFFT)
        defer { windowedFrame.deallocate() }

        // --- 6. Process each frame ---
        padded.withUnsafeBufferPointer { paddedBuf in
            fullWindow.withUnsafeBufferPointer { winBuf in
                for frame in 0..<nFrames {
                    let start = frame * hopLength

                    // Apply window: windowed = signal[start..<start+nFFT] * window
                    vDSP_vmul(
                        paddedBuf.baseAddress! + start, 1,
                        winBuf.baseAddress!, 1,
                        windowedFrame, 1,
                        vDSP_Length(nFFT)
                    )

                    // Pack real data into split complex format for vDSP_fft_zrip.
                    // vDSP_fft_zrip expects even-indexed samples in realp and
                    // odd-indexed samples in imagp.
                    var splitComplex = DSPSplitComplex(realp: realPart, imagp: imagPart)
                    windowedFrame.withMemoryRebound(to: DSPComplex.self, capacity: halfN) { complexPtr in
                        vDSP_ctoz(complexPtr, 2, &splitComplex, 1, vDSP_Length(halfN))
                    }

                    // Forward FFT (in-place)
                    vDSP_fft_zrip(fftSetup, &splitComplex, 1, log2n, FFTDirection(kFFTDirection_Forward))

                    // --- 7. Extract magnitudes ---
                    // After vDSP_fft_zrip:
                    //   splitComplex.realp[0] = DC component (real, imag is 0)
                    //   splitComplex.imagp[0] = Nyquist component (real, imag is 0)
                    //   splitComplex.realp[k], imagp[k] for k=1..halfN-1 = complex bin k
                    //
                    // vDSP_fft_zrip includes a factor of 2 compared to the mathematical DFT.
                    // We apply a scale of 1/2 to normalize before computing magnitudes.

                    // Scale by 0.5 to normalize
                    var scale: Float = 0.5
                    vDSP_vsmul(realPart, 1, &scale, realPart, 1, vDSP_Length(halfN))
                    vDSP_vsmul(imagPart, 1, &scale, imagPart, 1, vDSP_Length(halfN))

                    let colOffset = frame * nFreqs

                    // DC bin (index 0): magnitude = |realp[0]| (imagp[0] holds Nyquist, not DC imag)
                    let dcVal = realPart[0]
                    tempPtr[colOffset + 0] = abs(dcVal)

                    // Nyquist bin (index nFreqs - 1 = halfN): magnitude = |imagp[0]|
                    // After scaling, imagp[0] holds the Nyquist real component / 2...
                    // Actually, vDSP packs DC in realp[0] and Nyquist in imagp[0].
                    // Both are purely real. After our 0.5 scaling, the values are correct.
                    let nyquistVal = imagPart[0]
                    tempPtr[colOffset + nFreqs - 1] = abs(nyquistVal)

                    // Bins 1..<halfN: magnitude = sqrt(real^2 + imag^2)
                    // Use vDSP_zvabs but we need to handle offset pointers for bins 1..<halfN
                    // vDSP_zvabs computes magnitude of split complex
                    var innerSplit = DSPSplitComplex(
                        realp: realPart + 1,
                        imagp: imagPart + 1
                    )
                    vDSP_zvabs(&innerSplit, 1, tempPtr + colOffset + 1, 1, vDSP_Length(halfN - 1))
                }
            }
        }

        // --- 8. Transpose from column-major [nFreqs x nFrames] to row-major ---
        // tempPtr layout: column-major [nFreqs, nFrames] = tempPtr[frame * nFreqs + freq]
        // outPtr layout: row-major [nFreqs, nFrames]     = outPtr[freq * nFrames + frame]
        let outPtr = UnsafeMutablePointer<Float>.allocate(capacity: totalElements)
        // vDSP_mtrans transposes an M x N matrix (row-major) to N x M (row-major).
        // Treating tempPtr as nFrames rows x nFreqs cols (row-major reading of column-major data),
        // we transpose to get nFreqs rows x nFrames cols.
        vDSP_mtrans(tempPtr, 1, outPtr, 1, vDSP_Length(nFreqs), vDSP_Length(nFrames))
        tempPtr.deallocate()

        let outBuffer = UnsafeMutableBufferPointer(start: outPtr, count: totalElements)
        let output = Signal(taking: outBuffer, shape: [nFreqs, nFrames],
                            sampleRate: input.signal.sampleRate)
        return STFTOutput(magnitude: output)
    }

    // MARK: - Power Spectrogram (avoiding sqrt roundtrip)

    /// Compute the power spectrogram (r^2 + i^2) directly from the FFT
    /// without the intermediate magnitude (sqrt) step.
    ///
    /// This avoids the precision loss of `sqrt(r^2 + i^2)` followed by
    /// squaring, which matters for downstream operations like mel spectrogram
    /// and MFCC that need power values.
    ///
    /// Returns a `Signal` with shape `[nFreqs, nFrames]` containing power values.
    public static func computePowerSpectrogram(
        signal: Signal,
        nFFT: Int,
        hopLength: Int? = nil,
        winLength: Int? = nil,
        center: Bool = true
    ) -> Signal {
        let state = Profiler.shared.begin("STFT.power")
        defer { Profiler.shared.end("STFT.power", state) }
        let hop = hopLength ?? (nFFT / 4)
        let win = winLength ?? nFFT
        let nFreqs = nFFT / 2 + 1

        // --- 1. Optionally pad the signal ---
        let padded: [Float]
        if center {
            let padAmount = nFFT / 2
            padded = [Float](repeating: 0, count: padAmount)
                   + signal.withUnsafeBufferPointer { Array($0) }
                   + [Float](repeating: 0, count: padAmount)
        } else {
            padded = signal.withUnsafeBufferPointer { Array($0) }
        }

        guard padded.count >= nFFT else {
            return Signal(data: [], shape: [nFreqs, 0], sampleRate: signal.sampleRate)
        }
        let nFrames = 1 + (padded.count - nFFT) / hop

        // --- 2. Prepare window ---
        let fullWindow = prepareWindow(winLength: win, nFFT: nFFT)

        // --- 3. Set up vDSP FFT ---
        guard isPowerOfTwo(nFFT) else {
            return Signal(data: [], shape: [nFreqs, 0], sampleRate: signal.sampleRate)
        }
        let log2n = vDSP_Length(log2(Double(nFFT)))
        guard let fftSetup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else {
            return Signal(data: [], shape: [nFreqs, 0], sampleRate: signal.sampleRate)
        }
        defer { vDSP_destroy_fftsetup(fftSetup) }

        let totalElements = nFreqs * nFrames
        let tempPtr = UnsafeMutablePointer<Float>.allocate(capacity: totalElements)
        tempPtr.initialize(repeating: 0, count: totalElements)

        let halfN = nFFT / 2
        let realPart = UnsafeMutablePointer<Float>.allocate(capacity: halfN)
        let imagPart = UnsafeMutablePointer<Float>.allocate(capacity: halfN)
        defer {
            realPart.deallocate()
            imagPart.deallocate()
        }

        let windowedFrame = UnsafeMutablePointer<Float>.allocate(capacity: nFFT)
        defer { windowedFrame.deallocate() }

        // --- 4. Process each frame ---
        padded.withUnsafeBufferPointer { paddedBuf in
            fullWindow.withUnsafeBufferPointer { winBuf in
                for frame in 0..<nFrames {
                    let start = frame * hop

                    // Apply window
                    vDSP_vmul(
                        paddedBuf.baseAddress! + start, 1,
                        winBuf.baseAddress!, 1,
                        windowedFrame, 1,
                        vDSP_Length(nFFT)
                    )

                    // Pack into split complex
                    var splitComplex = DSPSplitComplex(realp: realPart, imagp: imagPart)
                    windowedFrame.withMemoryRebound(to: DSPComplex.self, capacity: halfN) { complexPtr in
                        vDSP_ctoz(complexPtr, 2, &splitComplex, 1, vDSP_Length(halfN))
                    }

                    // Forward FFT
                    vDSP_fft_zrip(fftSetup, &splitComplex, 1, log2n, FFTDirection(kFFTDirection_Forward))

                    // Scale by 0.5 to normalize (vDSP factor-of-2)
                    var scale: Float = 0.5
                    vDSP_vsmul(realPart, 1, &scale, realPart, 1, vDSP_Length(halfN))
                    vDSP_vsmul(imagPart, 1, &scale, imagPart, 1, vDSP_Length(halfN))

                    let colOffset = frame * nFreqs

                    // DC bin: power = realp[0]^2 (imag is 0)
                    let dcVal = realPart[0]
                    tempPtr[colOffset + 0] = dcVal * dcVal

                    // Nyquist bin: power = imagp[0]^2 (imag is 0)
                    let nyquistVal = imagPart[0]
                    tempPtr[colOffset + nFreqs - 1] = nyquistVal * nyquistVal

                    // Bins 1..<halfN: power = real^2 + imag^2
                    // Use vDSP_zvmags which computes squared magnitudes directly
                    var innerSplit = DSPSplitComplex(
                        realp: realPart + 1,
                        imagp: imagPart + 1
                    )
                    vDSP_zvmags(&innerSplit, 1, tempPtr + colOffset + 1, 1, vDSP_Length(halfN - 1))
                }
            }
        }

        // --- 5. Transpose to row-major [nFreqs, nFrames] ---
        let outPtr = UnsafeMutablePointer<Float>.allocate(capacity: totalElements)
        vDSP_mtrans(tempPtr, 1, outPtr, 1, vDSP_Length(nFreqs), vDSP_Length(nFrames))
        tempPtr.deallocate()

        let outBuffer = UnsafeMutableBufferPointer(start: outPtr, count: totalElements)
        return Signal(taking: outBuffer, shape: [nFreqs, nFrames], sampleRate: signal.sampleRate)
    }
}

// MARK: - GPU Path (MPSGraph)

extension STFT {
    /// GPU-accelerated STFT using MPSGraph for batched real FFT + magnitude.
    ///
    /// The approach:
    /// 1. Prepare frames on CPU (padding, windowing, framing — memory-bound, not worth GPU dispatch)
    /// 2. Use MPSGraph `realToHermiteanFFT` for batched FFT on all frames at once
    /// 3. Compute magnitudes via MPSGraph ops: sqrt(real^2 + imag^2)
    /// 4. Transpose [nFrames, nFreqs] -> [nFreqs, nFrames] on GPU
    /// 5. Read results back to CPU
    public func executeGPU(_ input: STFTInput) -> STFTOutput {
        guard let metalBackend = MetalBackend.shared else {
            // Fallback to CPU if Metal unavailable
            return executeCPU(input)
        }

        let nFFT = input.nFFT
        let hopLength = input.hopLength
        let winLength = input.winLength
        let nFreqs = nFFT / 2 + 1

        // --- 1. Optionally pad the signal ---
        let padded: [Float]
        if input.center {
            let padAmount = nFFT / 2
            padded = [Float](repeating: 0, count: padAmount)
                   + input.signal.withUnsafeBufferPointer { Array($0) }
                   + [Float](repeating: 0, count: padAmount)
        } else {
            padded = input.signal.withUnsafeBufferPointer { Array($0) }
        }

        let paddedLength = padded.count

        // --- 2. Compute number of frames ---
        guard paddedLength >= nFFT else {
            let out = Signal(data: [], shape: [nFreqs, 0], sampleRate: input.signal.sampleRate)
            return STFTOutput(magnitude: out)
        }
        let nFrames = 1 + (paddedLength - nFFT) / hopLength

        // --- 3. Prepare window ---
        let fullWindow = prepareWindow(winLength: winLength, nFFT: nFFT)

        // --- 4. Frame + window on CPU -> [nFrames, nFFT] ---
        var framedData = [Float](repeating: 0, count: nFrames * nFFT)
        padded.withUnsafeBufferPointer { paddedBuf in
            fullWindow.withUnsafeBufferPointer { winBuf in
                framedData.withUnsafeMutableBufferPointer { outBuf in
                    for frame in 0..<nFrames {
                        let srcStart = frame * hopLength
                        let dstStart = frame * nFFT
                        vDSP_vmul(
                            paddedBuf.baseAddress! + srcStart, 1,
                            winBuf.baseAddress!, 1,
                            outBuf.baseAddress! + dstStart, 1,
                            vDSP_Length(nFFT)
                        )
                    }
                }
            }
        }

        // --- 5. Build MPSGraph for batched real FFT + magnitude ---
        let graph = MPSGraph()

        // Input placeholder: [nFrames, nFFT] real float32
        let inputTensor = graph.placeholder(
            shape: [nFrames as NSNumber, nFFT as NSNumber],
            dataType: .float32,
            name: "input_frames"
        )

        // FFT descriptor: forward transform, no scaling (matches vDSP after our manual normalization)
        let fftDescriptor = MPSGraphFFTDescriptor()
        fftDescriptor.inverse = false
        fftDescriptor.scalingMode = .none

        // Real-to-Hermitean FFT along axis 1 (the nFFT dimension).
        // Input:  [nFrames, nFFT] float32
        // Output: [nFrames, nFFT/2+1] complexFloat32
        let fftResult = graph.realToHermiteanFFT(
            inputTensor,
            axes: [1],
            descriptor: fftDescriptor,
            name: "fft"
        )

        // Extract real and imaginary parts: each [nFrames, nFreqs] float32
        let realPart = graph.realPartOfTensor(tensor: fftResult, name: "real")
        let imagPart = graph.imaginaryPartOfTensor(tensor: fftResult, name: "imag")

        // Magnitude: sqrt(real^2 + imag^2)
        let realSq = graph.multiplication(realPart, realPart, name: "real_sq")
        let imagSq = graph.multiplication(imagPart, imagPart, name: "imag_sq")
        let sumSq = graph.addition(realSq, imagSq, name: "sum_sq")
        let magnitude = graph.squareRoot(with: sumSq, name: "magnitude")

        // Transpose [nFrames, nFreqs] -> [nFreqs, nFrames]
        let transposed = graph.transposeTensor(magnitude, dimension: 0, withDimension: 1, name: "transposed")

        // --- 6. Create input MTLBuffer and MPSGraphTensorData ---
        let inputByteCount = framedData.count * MemoryLayout<Float>.stride
        guard let inputBuffer = framedData.withUnsafeBufferPointer({ buf in
            metalBackend.device.makeBuffer(
                bytes: buf.baseAddress!,
                length: inputByteCount,
                options: .storageModeShared
            )
        }) else {
            // Metal buffer allocation failed — fall back to CPU
            return executeCPU(input)
        }

        let inputMPSData = MPSGraphTensorData(
            inputBuffer,
            shape: [nFrames as NSNumber, nFFT as NSNumber],
            dataType: .float32
        )

        // --- 7. Execute graph ---
        let results = graph.run(
            with: metalBackend.commandQueue,
            feeds: [inputTensor: inputMPSData],
            targetTensors: [transposed],
            targetOperations: nil
        )

        // --- 8. Read results back ---
        guard let outputMPSData = results[transposed] else {
            // Unexpected missing result — fall back to CPU
            return executeCPU(input)
        }

        let outputCount = nFreqs * nFrames
        let outPtr = UnsafeMutablePointer<Float>.allocate(capacity: outputCount)
        outputMPSData.mpsndarray().readBytes(outPtr, strideBytes: nil)

        let outBuffer = UnsafeMutableBufferPointer(start: outPtr, count: outputCount)
        let output = Signal(taking: outBuffer, shape: [nFreqs, nFrames],
                            sampleRate: input.signal.sampleRate)
        return STFTOutput(magnitude: output)
    }
}

// MARK: - Convenience API

extension STFT {
    /// Convenience method to compute STFT magnitude spectrogram.
    public static func compute(
        signal: Signal,
        nFFT: Int = 2048,
        hopLength: Int? = nil,
        winLength: Int? = nil,
        center: Bool = true
    ) -> Signal {
        let state = Profiler.shared.begin("STFT")
        defer { Profiler.shared.end("STFT", state) }
        let hop = hopLength ?? nFFT / 4
        let win = winLength ?? nFFT
        let input = STFTInput(signal: signal, nFFT: nFFT, hopLength: hop, winLength: win, center: center)
        let op = STFT()
        let dispatcher = SmartDispatcher()
        return dispatcher.dispatch(op, input: input, dataSize: signal.count).magnitude
    }

    /// Compute complex STFT returning interleaved real/imag Signal with dtype `.complex64`.
    /// Shape is [nFreqs, nFrames] (logical complex elements); raw float storage is 2x that.
    public static func computeComplex(
        signal: Signal,
        nFFT: Int = 2048,
        hopLength: Int? = nil,
        winLength: Int? = nil,
        center: Bool = true
    ) -> Signal {
        let hop = hopLength ?? nFFT / 4
        let win = winLength ?? nFFT
        let nFreqs = nFFT / 2 + 1

        // --- 1. Optionally pad the signal ---
        let padded: [Float]
        if center {
            let padAmount = nFFT / 2
            padded = [Float](repeating: 0, count: padAmount)
                   + signal.withUnsafeBufferPointer { Array($0) }
                   + [Float](repeating: 0, count: padAmount)
        } else {
            padded = signal.withUnsafeBufferPointer { Array($0) }
        }

        let paddedLength = padded.count

        // --- 2. Compute number of frames ---
        guard paddedLength >= nFFT else {
            return Signal(complexData: [], shape: [nFreqs, 0], sampleRate: signal.sampleRate)
        }
        let nFrames = 1 + (paddedLength - nFFT) / hop

        // --- 3. Prepare window ---
        let fullWindow = prepareWindow(winLength: win, nFFT: nFFT)

        // --- 4. Set up vDSP FFT ---
        guard isPowerOfTwo(nFFT) else {
            return Signal(complexData: [], shape: [nFreqs, 0], sampleRate: signal.sampleRate)
        }
        let log2n = vDSP_Length(log2(Double(nFFT)))
        guard let fftSetup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else {
            return Signal(complexData: [], shape: [nFreqs, 0], sampleRate: signal.sampleRate)
        }
        defer { vDSP_destroy_fftsetup(fftSetup) }

        // --- 5. Allocate complex output buffer ---
        // Column-major temp: for each frame, nFreqs interleaved pairs (2 * nFreqs floats).
        // Layout: tempPtr[(frame * nFreqs + freq) * 2 + 0] = real
        //         tempPtr[(frame * nFreqs + freq) * 2 + 1] = imag
        let totalComplex = nFreqs * nFrames
        let totalFloats = totalComplex * 2
        let tempPtr = UnsafeMutablePointer<Float>.allocate(capacity: totalFloats)
        tempPtr.initialize(repeating: 0, count: totalFloats)

        let halfN = nFFT / 2
        let realPart = UnsafeMutablePointer<Float>.allocate(capacity: halfN)
        let imagPart = UnsafeMutablePointer<Float>.allocate(capacity: halfN)
        defer {
            realPart.deallocate()
            imagPart.deallocate()
        }

        let windowedFrame = UnsafeMutablePointer<Float>.allocate(capacity: nFFT)
        defer { windowedFrame.deallocate() }

        // --- 6. Process each frame ---
        padded.withUnsafeBufferPointer { paddedBuf in
            fullWindow.withUnsafeBufferPointer { winBuf in
                for frame in 0..<nFrames {
                    let start = frame * hop

                    // Apply window
                    vDSP_vmul(
                        paddedBuf.baseAddress! + start, 1,
                        winBuf.baseAddress!, 1,
                        windowedFrame, 1,
                        vDSP_Length(nFFT)
                    )

                    // Pack into split complex for vDSP_fft_zrip
                    var splitComplex = DSPSplitComplex(realp: realPart, imagp: imagPart)
                    windowedFrame.withMemoryRebound(to: DSPComplex.self, capacity: halfN) { complexPtr in
                        vDSP_ctoz(complexPtr, 2, &splitComplex, 1, vDSP_Length(halfN))
                    }

                    // Forward FFT (in-place)
                    vDSP_fft_zrip(fftSetup, &splitComplex, 1, log2n, FFTDirection(kFFTDirection_Forward))

                    // Scale by 0.5 to normalize (vDSP_fft_zrip has factor-of-2)
                    var scale: Float = 0.5
                    vDSP_vsmul(realPart, 1, &scale, realPart, 1, vDSP_Length(halfN))
                    vDSP_vsmul(imagPart, 1, &scale, imagPart, 1, vDSP_Length(halfN))

                    // --- 7. Store interleaved complex output (column-major) ---
                    // Column offset in complex elements
                    let colBase = frame * nFreqs

                    // DC bin (index 0): real = realp[0], imag = 0
                    tempPtr[(colBase + 0) * 2 + 0] = realPart[0]
                    tempPtr[(colBase + 0) * 2 + 1] = 0.0

                    // Bins 1..<halfN: real = realp[k], imag = imagp[k]
                    for k in 1..<halfN {
                        tempPtr[(colBase + k) * 2 + 0] = realPart[k]
                        tempPtr[(colBase + k) * 2 + 1] = imagPart[k]
                    }

                    // Nyquist bin (index nFreqs-1 = halfN): real = imagp[0], imag = 0
                    tempPtr[(colBase + nFreqs - 1) * 2 + 0] = imagPart[0]
                    tempPtr[(colBase + nFreqs - 1) * 2 + 1] = 0.0
                }
            }
        }

        // --- 8. Transpose from column-major to row-major ---
        // tempPtr is column-major: element (freq, frame) at tempPtr[(frame * nFreqs + freq) * 2]
        // We need row-major: element (freq, frame) at outPtr[(freq * nFrames + frame) * 2]
        //
        // Treat each complex element as a pair of floats. We transpose the [nFrames, nFreqs]
        // matrix of pairs to [nFreqs, nFrames] by transposing real and imaginary planes separately.
        let outPtr = UnsafeMutablePointer<Float>.allocate(capacity: totalFloats)

        // Extract real plane, transpose, store back
        // temp real plane: tempPtr[i*2] for i in 0..<totalComplex
        // We need a temporary plane buffer for the transpose
        let realPlane = UnsafeMutablePointer<Float>.allocate(capacity: totalComplex)
        let imagPlane = UnsafeMutablePointer<Float>.allocate(capacity: totalComplex)
        let realPlaneOut = UnsafeMutablePointer<Float>.allocate(capacity: totalComplex)
        let imagPlaneOut = UnsafeMutablePointer<Float>.allocate(capacity: totalComplex)
        defer {
            realPlane.deallocate()
            imagPlane.deallocate()
            realPlaneOut.deallocate()
            imagPlaneOut.deallocate()
        }

        // De-interleave: extract real and imag planes from interleaved temp buffer
        // vDSP_ctoz splits interleaved [r0,i0,r1,i1,...] into split complex {realp, imagp}
        var splitPlane = DSPSplitComplex(realp: realPlane, imagp: imagPlane)
        tempPtr.withMemoryRebound(to: DSPComplex.self, capacity: totalComplex) { complexPtr in
            vDSP_ctoz(complexPtr, 2, &splitPlane, 1, vDSP_Length(totalComplex))
        }

        // Transpose each plane: [nFrames rows x nFreqs cols] -> [nFreqs rows x nFrames cols]
        vDSP_mtrans(realPlane, 1, realPlaneOut, 1, vDSP_Length(nFreqs), vDSP_Length(nFrames))
        vDSP_mtrans(imagPlane, 1, imagPlaneOut, 1, vDSP_Length(nFreqs), vDSP_Length(nFrames))

        // Re-interleave into outPtr using vDSP_ztoc
        var splitOut = DSPSplitComplex(realp: realPlaneOut, imagp: imagPlaneOut)
        outPtr.withMemoryRebound(to: DSPComplex.self, capacity: totalComplex) { complexPtr in
            vDSP_ztoc(&splitOut, 1, complexPtr, 2, vDSP_Length(totalComplex))
        }

        tempPtr.deallocate()

        let outBuffer = UnsafeMutableBufferPointer(start: outPtr, count: totalFloats)
        return Signal(taking: outBuffer, shape: [nFreqs, nFrames],
                      sampleRate: signal.sampleRate, dtype: .complex64)
    }
}

// MARK: - Inverse STFT

extension STFT {
    /// Inverse STFT: reconstruct time-domain signal from complex spectrogram via overlap-add.
    ///
    /// - Parameters:
    ///   - complexSTFT: Complex spectrogram with shape [nFreqs, nFrames], dtype `.complex64`.
    ///                  Row-major layout: for freq `k`, frame `f`, the real part is at
    ///                  raw index `2 * (k * nFrames + f)`, imag at `2 * (k * nFrames + f) + 1`.
    ///   - hopLength: Hop length in samples. Defaults to `(nFreqs - 1) * 2 / 4`.
    ///   - winLength: Window length. Defaults to `nFFT`.
    ///   - center: If `true` (default), trims `nFFT / 2` samples from each end of the output
    ///             (undoing the padding applied by the forward STFT).
    ///   - length: If specified, truncates or zero-pads the output to this length.
    /// - Returns: Reconstructed 1D real-valued Signal.
    public static func inverse(
        complexSTFT: Signal,
        hopLength: Int? = nil,
        winLength: Int? = nil,
        center: Bool = true,
        length: Int? = nil
    ) -> Signal {
        precondition(complexSTFT.dtype == .complex64, "iSTFT requires complex input")
        precondition(complexSTFT.shape.count == 2, "iSTFT requires 2D input [nFreqs, nFrames]")

        let nFreqs = complexSTFT.shape[0]
        let nFrames = complexSTFT.shape[1]
        let nFFT = (nFreqs - 1) * 2
        let hop = hopLength ?? nFFT / 4
        let win = winLength ?? nFFT
        let halfN = nFFT / 2

        // Compute expected output length (length of the padded signal)
        let expectedLength = nFFT + (nFrames - 1) * hop

        // --- 1. Prepare synthesis window ---
        let fullWindow = prepareWindow(winLength: win, nFFT: nFFT)

        // --- 2. Compute window normalization (sum of squared windows at each output sample) ---
        // This is the COLA (Constant Overlap-Add) normalization factor.
        var windowSum = [Float](repeating: 0, count: expectedLength)
        for frame in 0..<nFrames {
            let start = frame * hop
            for i in 0..<nFFT {
                windowSum[start + i] += fullWindow[i] * fullWindow[i]
            }
        }

        // --- 3. Set up vDSP inverse FFT ---
        guard isPowerOfTwo(nFFT) else {
            return Signal(data: [Float](repeating: 0, count: expectedLength),
                         sampleRate: complexSTFT.sampleRate)
        }
        let log2n = vDSP_Length(log2(Double(nFFT)))
        guard let fftSetup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else {
            return Signal(data: [Float](repeating: 0, count: expectedLength),
                         sampleRate: complexSTFT.sampleRate)
        }
        defer { vDSP_destroy_fftsetup(fftSetup) }

        // --- 4. Allocate output and temporary buffers ---
        var output = [Float](repeating: 0, count: expectedLength)
        let realPart = UnsafeMutablePointer<Float>.allocate(capacity: halfN)
        let imagPart = UnsafeMutablePointer<Float>.allocate(capacity: halfN)
        let frameBuffer = UnsafeMutablePointer<Float>.allocate(capacity: nFFT)
        defer {
            realPart.deallocate()
            imagPart.deallocate()
            frameBuffer.deallocate()
        }

        // --- 5. Process each frame: inverse FFT + window + overlap-add ---
        complexSTFT.withUnsafeBufferPointer { rawBuf in
            fullWindow.withUnsafeBufferPointer { winBuf in
                for frame in 0..<nFrames {
                    // Extract this frame's complex bins from row-major [nFreqs, nFrames] layout.
                    // For freq k, frame f: real at rawBuf[2 * (k * nFrames + f)],
                    //                      imag at rawBuf[2 * (k * nFrames + f) + 1]

                    // Pack into vDSP split complex format:
                    // - realp[0] = DC real, imagp[0] = Nyquist real
                    // - realp[k] = bin k real, imagp[k] = bin k imag, for k=1..<halfN

                    // DC bin (freq 0)
                    let dcIdx = 2 * (0 * nFrames + frame)
                    realPart[0] = rawBuf[dcIdx]  // DC real

                    // Nyquist bin (freq nFreqs-1 = halfN)
                    let nyquistIdx = 2 * ((nFreqs - 1) * nFrames + frame)
                    imagPart[0] = rawBuf[nyquistIdx]  // Nyquist real goes into imagp[0]

                    // Bins 1..<halfN
                    for k in 1..<halfN {
                        let idx = 2 * (k * nFrames + frame)
                        realPart[k] = rawBuf[idx]
                        imagPart[k] = rawBuf[idx + 1]
                    }

                    // Scale by 2.0 to undo the 0.5 normalization from forward FFT.
                    // The forward STFT scaled by 0.5 to normalize the vDSP factor-of-2.
                    // The inverse FFT via vDSP_fft_zrip(kFFTDirection_Inverse) produces
                    // values that are nFFT times the true IDFT (plus the factor-of-2 from
                    // the packed format). So:
                    // - Forward scaled by 0.5 to get true DFT values
                    // - We need to multiply by 2.0 to restore the packed format values
                    // - Then inverse FFT gives us nFFT * true signal (with factor-of-2)
                    // - We scale by 1/(2*nFFT) after inverse to get true signal
                    // Actually:
                    //   Forward: vDSP produces 2*DFT, we scale by 0.5 -> DFT values
                    //   Inverse: vDSP_fft_zrip(inverse) expects the 2*DFT format
                    //   So we multiply by 2.0 to get back to vDSP's native format
                    //   Then inverse gives us nFFT * signal (times 2 from packed format)
                    //   So we divide by (2 * nFFT) afterward
                    var scaleUp: Float = 2.0
                    vDSP_vsmul(realPart, 1, &scaleUp, realPart, 1, vDSP_Length(halfN))
                    vDSP_vsmul(imagPart, 1, &scaleUp, imagPart, 1, vDSP_Length(halfN))

                    // Inverse FFT (in-place)
                    var splitComplex = DSPSplitComplex(realp: realPart, imagp: imagPart)
                    vDSP_fft_zrip(fftSetup, &splitComplex, 1, log2n, FFTDirection(kFFTDirection_Inverse))

                    // Unpack split complex to interleaved real signal
                    // vDSP_ztoc converts split complex back to interleaved pairs
                    frameBuffer.withMemoryRebound(to: DSPComplex.self, capacity: halfN) { complexPtr in
                        vDSP_ztoc(&splitComplex, 1, complexPtr, 2, vDSP_Length(halfN))
                    }

                    // Scale by 1/(2*nFFT) to get the true inverse DFT result
                    // vDSP_fft_zrip inverse output = 2 * nFFT * true_signal
                    var scaleDown: Float = 1.0 / Float(2 * nFFT)
                    vDSP_vsmul(frameBuffer, 1, &scaleDown, frameBuffer, 1, vDSP_Length(nFFT))

                    // Apply synthesis window
                    vDSP_vmul(frameBuffer, 1, winBuf.baseAddress!, 1, frameBuffer, 1, vDSP_Length(nFFT))

                    // Overlap-add into output
                    let start = frame * hop
                    for i in 0..<nFFT {
                        output[start + i] += frameBuffer[i]
                    }
                }
            }
        }

        // --- 6. Normalize by window overlap ---
        // Divide by the sum of squared windows (COLA normalization).
        // Avoid division by zero for samples with no window coverage.
        for i in 0..<expectedLength {
            if windowSum[i] > 1e-10 {
                output[i] /= windowSum[i]
            }
        }

        // --- 7. Trim center padding and apply length ---
        // When center=True, the forward STFT padded nFFT/2 on each side.
        // We take from offset nFFT/2 in the reconstruction buffer.
        // When length is specified, we take exactly that many samples from offset.
        // When not specified, we trim nFFT/2 from each end.
        var result: [Float]
        if center {
            let trimStart = nFFT / 2
            if let length = length {
                // Take `length` samples starting at trimStart from the full buffer
                let end = min(trimStart + length, expectedLength)
                result = Array(output[trimStart..<end])
                // Pad with zeros if the buffer doesn't have enough samples
                if result.count < length {
                    result += [Float](repeating: 0, count: length - result.count)
                }
            } else {
                // Default: trim nFFT/2 from each end
                let trimEnd = expectedLength - nFFT / 2
                if trimEnd > trimStart {
                    result = Array(output[trimStart..<trimEnd])
                } else {
                    result = []
                }
            }
        } else {
            result = output
            if let length = length {
                if result.count > length {
                    result = Array(result.prefix(length))
                } else if result.count < length {
                    result += [Float](repeating: 0, count: length - result.count)
                }
            }
        }

        return Signal(data: result, sampleRate: complexSTFT.sampleRate)
    }
}
