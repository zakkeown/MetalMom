import Foundation
import Metal
import MetalPerformanceShaders
import MetalPerformanceShadersGraph
import Accelerate

/// Fused MFCC pipeline: STFT -> power -> mel filterbank -> log -> DCT
/// All stages in a single GPU command buffer when Metal is available.
/// Falls back to per-step CPU pipeline otherwise.
public enum FusedMFCC {

    /// Compute MFCC using fused GPU pipeline when available.
    /// Falls back to per-step CPU pipeline otherwise.
    ///
    /// Returns a `Signal` with shape `[nMFCC, nFrames]`, row-major.
    ///
    /// - Parameters:
    ///   - signal: Input audio signal (1D).
    ///   - sr: Sample rate override. If `nil`, uses `signal.sampleRate`.
    ///   - nFFT: FFT window size. Default 2048.
    ///   - hopLength: Hop length in samples. Default `nFFT / 4`.
    ///   - winLength: Window length. Default `nFFT`.
    ///   - center: If `true`, pad signal so frames are centred. Default `true`.
    ///   - nMels: Number of mel bands. Default 128.
    ///   - nMFCC: Number of MFCC coefficients to return. Default 20.
    ///   - fMin: Lowest frequency (Hz) for mel filterbank. Default 0.0.
    ///   - fMax: Highest frequency (Hz). If `nil`, uses `sr / 2`.
    /// - Returns: MFCC `Signal` with shape `[nMFCC, nFrames]`.
    public static func compute(
        signal: Signal,
        sr: Int? = nil,
        nFFT: Int = 2048,
        hopLength: Int? = nil,
        winLength: Int? = nil,
        center: Bool = true,
        nMels: Int = 128,
        nMFCC: Int = 20,
        fMin: Float = 0.0,
        fMax: Float? = nil
    ) -> Signal {
        let sampleRate = sr ?? signal.sampleRate
        let hop = hopLength ?? (nFFT / 4)
        let win = winLength ?? nFFT
        let maxFreq = fMax ?? Float(sampleRate) / 2.0

        // Decide: fused GPU or standard CPU
        let dataSize = signal.count
        let useGPU = MetalBackend.shared != nil
            && dataSize >= (MetalBackend.shared?.chipProfile.threshold(for: .stft) ?? Int.max)

        if useGPU, let result = computeGPU(
            signal: signal, sr: sampleRate, nFFT: nFFT, hopLength: hop,
            winLength: win, center: center, nMels: nMels, nMFCC: nMFCC,
            fMin: fMin, fMax: maxFreq
        ) {
            return result
        }

        // Fallback: standard per-step CPU computation
        return computeCPU(
            signal: signal, sr: sampleRate, nFFT: nFFT, hopLength: hop,
            winLength: win, center: center, nMels: nMels, nMFCC: nMFCC,
            fMin: fMin, fMax: maxFreq
        )
    }

    // MARK: - GPU Fused Path

    /// Fused GPU pipeline: STFT -> power -> mel -> log(dB) -> DCT in a single MPSGraph execution.
    /// Returns `nil` if Metal resources cannot be allocated.
    static func computeGPU(
        signal: Signal, sr: Int, nFFT: Int, hopLength: Int,
        winLength: Int, center: Bool, nMels: Int, nMFCC: Int,
        fMin: Float, fMax: Float
    ) -> Signal? {
        guard let backend = MetalBackend.shared else { return nil }
        let device = backend.device
        let nFreqs = nFFT / 2 + 1

        // Step 1: Frame + window on CPU (memory-bound, fast)
        let padded: [Float]
        if center {
            let padAmount = nFFT / 2
            padded = [Float](repeating: 0, count: padAmount)
                   + signal.withUnsafeBufferPointer { Array($0) }
                   + [Float](repeating: 0, count: padAmount)
        } else {
            padded = signal.withUnsafeBufferPointer { Array($0) }
        }

        guard padded.count >= nFFT else { return nil }
        let nFrames = 1 + (padded.count - nFFT) / hopLength

        let window = Windows.hann(length: winLength, periodic: true)
        let fullWindow: [Float]
        if winLength < nFFT {
            let padBefore = (nFFT - winLength) / 2
            let padAfter = nFFT - winLength - padBefore
            fullWindow = [Float](repeating: 0, count: padBefore) + window + [Float](repeating: 0, count: padAfter)
        } else {
            fullWindow = window
        }

        // Create framed+windowed data: [nFrames, nFFT]
        var framedData = [Float](repeating: 0, count: nFrames * nFFT)
        framedData.withUnsafeMutableBufferPointer { outBuf in
            padded.withUnsafeBufferPointer { paddedBuf in
                fullWindow.withUnsafeBufferPointer { winBuf in
                    for frame in 0..<nFrames {
                        let start = frame * hopLength
                        let offset = frame * nFFT
                        vDSP_vmul(
                            paddedBuf.baseAddress! + start, 1,
                            winBuf.baseAddress!, 1,
                            outBuf.baseAddress! + offset, 1,
                            vDSP_Length(nFFT)
                        )
                    }
                }
            }
        }

        // Step 2: Build MPSGraph for fused FFT -> power -> mel -> log -> DCT
        let graph = MPSGraph()

        // Input: framed data [nFrames, nFFT]
        let inputTensor = graph.placeholder(
            shape: [nFrames as NSNumber, nFFT as NSNumber],
            dataType: .float32,
            name: "frames"
        )

        // FFT -> complex [nFrames, nFreqs]
        let fftDesc = MPSGraphFFTDescriptor()
        fftDesc.inverse = false
        fftDesc.scalingMode = .none
        let fftResult = graph.realToHermiteanFFT(inputTensor, axes: [1], descriptor: fftDesc, name: "fft")

        // Magnitude squared (power spectrogram)
        let realPart = graph.realPartOfTensor(tensor: fftResult, name: "real")
        let imagPart = graph.imaginaryPartOfTensor(tensor: fftResult, name: "imag")
        let powerSpec = graph.addition(
            graph.multiplication(realPart, realPart, name: nil),
            graph.multiplication(imagPart, imagPart, name: nil),
            name: "power_spec"
        ) // [nFrames, nFreqs]

        // Transpose to [nFreqs, nFrames] for matmul with mel filterbank
        let powerSpecT = graph.transposeTensor(powerSpec, dimension: 0, withDimension: 1, name: "power_T")

        // Mel filterbank: [nMels, nFreqs] @ [nFreqs, nFrames] = [nMels, nFrames]
        let melFB = FilterBank.mel(sr: sr, nFFT: nFFT, nMels: nMels, fMin: fMin, fMax: fMax)
        let melWeights: [Float] = melFB.withUnsafeBufferPointer { Array($0) }
        let melConst = graph.constant(
            Data(bytes: melWeights, count: melWeights.count * MemoryLayout<Float>.stride),
            shape: [nMels as NSNumber, nFreqs as NSNumber],
            dataType: .float32
        )
        let melSpec = graph.matrixMultiplication(
            primary: melConst,
            secondary: powerSpecT,
            name: "mel_spec"
        ) // [nMels, nFrames]

        // Log: 10 * log10(max(melSpec, 1e-10))
        // = 10 / ln(10) * ln(max(melSpec, 1e-10))
        let amin = graph.constant(1e-10, dataType: .float32)
        let clampedMel = graph.maximum(melSpec, amin, name: "clamped")
        let logMel = graph.logarithm(with: clampedMel, name: "log_mel")
        let dbScaleValue: Double = 10.0 / log(10.0)
        let dbScale = graph.constant(dbScaleValue, dataType: .float32)
        let melDb = graph.multiplication(logMel, dbScale, name: "mel_db") // [nMels, nFrames]

        // topDb clipping: clip to (max - 80) dB, matching Scaling.powerToDb default
        let maxDb = graph.reductionMaximum(with: melDb, axes: [0, 1], name: "max_db")
        let topDbConst = graph.constant(80.0, dataType: .float32)
        let lowerBound = graph.subtraction(maxDb, topDbConst, name: "lower_bound")
        let clippedMelDb = graph.maximum(melDb, lowerBound, name: "clipped_mel_db")

        // DCT-II (first nMFCC coefficients) via matrix multiply
        // dctMatrix [nMFCC, nMels] @ clippedMelDb [nMels, nFrames] = [nMFCC, nFrames]
        var dctMatrix = [Float](repeating: 0, count: nMFCC * nMels)
        let normK = sqrtf(2.0 / Float(nMels))
        let norm0 = sqrtf(1.0 / Float(nMels))
        for k in 0..<nMFCC {
            let norm = (k == 0) ? norm0 : normK
            for n in 0..<nMels {
                let angle = Float.pi * Float(k) * (2.0 * Float(n) + 1.0) / (2.0 * Float(nMels))
                dctMatrix[k * nMels + n] = cosf(angle) * norm
            }
        }

        let dctConst = graph.constant(
            Data(bytes: dctMatrix, count: dctMatrix.count * MemoryLayout<Float>.stride),
            shape: [nMFCC as NSNumber, nMels as NSNumber],
            dataType: .float32
        )
        let mfcc = graph.matrixMultiplication(primary: dctConst, secondary: clippedMelDb, name: "mfcc")
        // mfcc: [nMFCC, nFrames]

        // Execute the graph
        guard let inputBuf = device.makeBuffer(
            bytes: framedData,
            length: framedData.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ) else { return nil }

        let inputMPSData = MPSGraphTensorData(
            inputBuf,
            shape: [nFrames as NSNumber, nFFT as NSNumber],
            dataType: .float32
        )

        let results = graph.run(
            with: backend.commandQueue,
            feeds: [inputTensor: inputMPSData],
            targetTensors: [mfcc],
            targetOperations: nil
        )

        guard let outputMPSData = results[mfcc] else { return nil }
        let outputCount = nMFCC * nFrames
        let outPtr = UnsafeMutablePointer<Float>.allocate(capacity: outputCount)
        let stridePtr: UnsafeMutablePointer<Int>? = nil
        outputMPSData.mpsndarray().readBytes(outPtr, strideBytes: stridePtr)

        let outBuffer = UnsafeMutableBufferPointer(start: outPtr, count: outputCount)
        return Signal(taking: outBuffer, shape: [nMFCC, nFrames], sampleRate: sr)
    }

    // MARK: - CPU Fallback

    /// CPU path using existing per-step computation: MelSpectrogram -> powerToDb -> DCT.
    static func computeCPU(
        signal: Signal, sr: Int, nFFT: Int, hopLength: Int,
        winLength: Int, center: Bool, nMels: Int, nMFCC: Int,
        fMin: Float, fMax: Float
    ) -> Signal {
        // Use existing MFCC computation which is already well-tested
        return MFCC.compute(
            signal: signal,
            sr: sr,
            nMFCC: nMFCC,
            nFFT: nFFT,
            hopLength: hopLength,
            winLength: winLength,
            nMels: nMels,
            fMin: fMin,
            fMax: fMax,
            center: center
        )
    }
}
