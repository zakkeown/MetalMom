import Foundation
import MetalMomCore
import MetalMomCBridge

// MARK: - Internal Context

/// Internal context holding the SmartDispatcher.
final class MMContextInternal {
    let dispatcher: SmartDispatcher

    init() {
        self.dispatcher = SmartDispatcher()
    }
}

/// Thread-safe registry of live context pointers to prevent double-free.
private let contextRegistryLock = NSLock()
private var contextRegistry = Set<UnsafeMutableRawPointer>()

// MARK: - Lifecycle

@_cdecl("mm_init")
public func mm_init() -> UnsafeMutableRawPointer? {
    let ctx = MMContextInternal()
    let ptr = Unmanaged.passRetained(ctx).toOpaque()
    contextRegistryLock.lock()
    contextRegistry.insert(ptr)
    contextRegistryLock.unlock()
    return ptr
}

@_cdecl("mm_destroy")
public func mm_destroy(_ ctx: UnsafeMutableRawPointer?) {
    guard let ctx = ctx else { return }
    contextRegistryLock.lock()
    let wasRegistered = contextRegistry.remove(ctx) != nil
    contextRegistryLock.unlock()
    guard wasRegistered else { return }  // Already destroyed — no-op
    Unmanaged<MMContextInternal>.fromOpaque(ctx).release()
}

// MARK: - STFT

@_cdecl("mm_stft")
public func mm_stft(
    _ ctx: UnsafeMutableRawPointer?,
    _ signalData: UnsafePointer<Float>?,
    _ signalLength: Int64,
    _ sampleRate: Int32,
    _ params: UnsafePointer<MMSTFTParams>?,
    _ out: UnsafeMutablePointer<MMBuffer>?
) -> Int32 {
    guard let ctx = ctx,
          let signalData = signalData,
          signalLength > 0,
          let params = params,
          let out = out else {
        return MM_ERR_INVALID_INPUT
    }

    let context = Unmanaged<MMContextInternal>.fromOpaque(ctx).takeUnretainedValue()
    let p = params.pointee

    // Validate STFT parameters
    let nFFT = Int(p.n_fft)
    guard nFFT > 0, nFFT & (nFFT - 1) == 0 else {
        return MM_ERR_INVALID_INPUT  // nFFT must be a positive power of 2
    }

    // Copy input data into a Signal
    let length = Int(signalLength)
    let inputArray = Array(UnsafeBufferPointer(start: signalData, count: length))
    let signal = Signal(data: inputArray, sampleRate: Int(sampleRate))

    // Build STFT input
    let hopLength = Int(p.hop_length)
    let winLength = Int(p.win_length)
    let center = p.center != 0

    let stftInput = STFTInput(
        signal: signal,
        nFFT: nFFT,
        hopLength: hopLength,
        winLength: winLength,
        center: center
    )

    // Compute STFT via dispatcher
    let stft = STFT()
    let result = context.dispatcher.dispatch(stft, input: stftInput, dataSize: signal.count)
    let magnitude = result.magnitude

    // Allocate output buffer and copy magnitude data
    let count = magnitude.count
    guard count > 0 else {
        // Valid but empty result
        out.pointee.data = nil
        out.pointee.ndim = Int32(magnitude.shape.count)
        let emptyShape = magnitude.shape
        withUnsafeMutablePointer(to: &out.pointee.shape) { tuplePtr in
            tuplePtr.withMemoryRebound(to: Int64.self, capacity: 8) { shapePtr in
                for i in 0..<8 {
                    shapePtr[i] = 0
                }
                for i in 0..<min(emptyShape.count, 8) {
                    shapePtr[i] = Int64(emptyShape[i])
                }
            }
        }
        out.pointee.dtype = 0
        out.pointee.count = 0
        return MM_OK
    }

    let outData = UnsafeMutablePointer<Float>.allocate(capacity: count)
    magnitude.withUnsafeBufferPointer { srcBuf in
        outData.initialize(from: srcBuf.baseAddress!, count: count)
    }

    // Fill MMBuffer struct
    out.pointee.data = outData
    out.pointee.ndim = Int32(magnitude.shape.count)
    // Write shape values via tuple indexing
    let shape = magnitude.shape
    withUnsafeMutablePointer(to: &out.pointee.shape) { tuplePtr in
        tuplePtr.withMemoryRebound(to: Int64.self, capacity: 8) { shapePtr in
            // Zero all 8 dimensions first
            for i in 0..<8 {
                shapePtr[i] = 0
            }
            for i in 0..<min(shape.count, 8) {
                shapePtr[i] = Int64(shape[i])
            }
        }
    }
    out.pointee.dtype = 0  // float32
    out.pointee.count = Int64(count)

    return MM_OK
}

// MARK: - iSTFT

@_cdecl("mm_istft")
public func mm_istft(
    _ ctx: UnsafeMutableRawPointer?,
    _ stftData: UnsafePointer<Float>?,
    _ stftCount: Int64,
    _ nFreqs: Int32,
    _ nFrames: Int32,
    _ sampleRate: Int32,
    _ hopLength: Int32,
    _ winLength: Int32,
    _ center: Int32,
    _ outputLength: Int64,
    _ out: UnsafeMutablePointer<MMBuffer>?
) -> Int32 {
    guard let ctx = ctx,
          let stftData = stftData,
          stftCount > 0,
          let out = out else {
        return MM_ERR_INVALID_INPUT
    }

    let _ = Unmanaged<MMContextInternal>.fromOpaque(ctx).takeUnretainedValue()

    // Build complex Signal from raw interleaved float data
    // stftCount is the total number of raw floats (2 * nFreqs * nFrames)
    let count = Int(stftCount)
    let inputArray = Array(UnsafeBufferPointer(start: stftData, count: count))
    let complexSTFT = Signal(complexData: inputArray,
                             shape: [Int(nFreqs), Int(nFrames)],
                             sampleRate: Int(sampleRate))

    // Compute inverse STFT
    let hop = Int(hopLength)
    let win = Int(winLength)
    let isCentered = center != 0
    let reqLength: Int? = outputLength > 0 ? Int(outputLength) : nil

    let result = STFT.inverse(
        complexSTFT: complexSTFT,
        hopLength: hop,
        winLength: win,
        center: isCentered,
        length: reqLength
    )

    // Fill output MMBuffer
    let outCount = result.count
    guard outCount > 0 else {
        out.pointee.data = nil
        out.pointee.ndim = 1
        withUnsafeMutablePointer(to: &out.pointee.shape) { tuplePtr in
            tuplePtr.withMemoryRebound(to: Int64.self, capacity: 8) { shapePtr in
                for i in 0..<8 { shapePtr[i] = 0 }
            }
        }
        out.pointee.dtype = 0
        out.pointee.count = 0
        return MM_OK
    }

    let outData = UnsafeMutablePointer<Float>.allocate(capacity: outCount)
    result.withUnsafeBufferPointer { srcBuf in
        outData.initialize(from: srcBuf.baseAddress!, count: outCount)
    }

    out.pointee.data = outData
    out.pointee.ndim = 1
    withUnsafeMutablePointer(to: &out.pointee.shape) { tuplePtr in
        tuplePtr.withMemoryRebound(to: Int64.self, capacity: 8) { shapePtr in
            for i in 0..<8 { shapePtr[i] = 0 }
            shapePtr[0] = Int64(outCount)
        }
    }
    out.pointee.dtype = 0  // float32
    out.pointee.count = Int64(outCount)

    return MM_OK
}

// MARK: - dB Scaling

@_cdecl("mm_amplitude_to_db")
public func mm_amplitude_to_db(
    _ ctx: UnsafeMutableRawPointer?,
    _ data: UnsafePointer<Float>?,
    _ count: Int64,
    _ ref: Float,
    _ amin: Float,
    _ topDb: Float,
    _ out: UnsafeMutablePointer<MMBuffer>?
) -> Int32 {
    guard let data = data, count > 0, let out = out else {
        return MM_ERR_INVALID_INPUT
    }

    let length = Int(count)
    let inputArray = Array(UnsafeBufferPointer(start: data, count: length))
    let signal = Signal(data: inputArray, sampleRate: 0)

    // topDb <= 0 means no clipping (Python passes 0.0 for None)
    let topDbOpt: Float? = topDb > 0 ? topDb : nil
    let result = Scaling.amplitudeToDb(signal, ref: ref, amin: amin, topDb: topDbOpt)

    let outCount = result.count
    let outData = UnsafeMutablePointer<Float>.allocate(capacity: outCount)
    result.withUnsafeBufferPointer { srcBuf in
        outData.initialize(from: srcBuf.baseAddress!, count: outCount)
    }

    out.pointee.data = outData
    out.pointee.ndim = 1
    withUnsafeMutablePointer(to: &out.pointee.shape) { tuplePtr in
        tuplePtr.withMemoryRebound(to: Int64.self, capacity: 8) { shapePtr in
            for i in 0..<8 { shapePtr[i] = 0 }
            shapePtr[0] = Int64(outCount)
        }
    }
    out.pointee.dtype = 0
    out.pointee.count = Int64(outCount)

    return MM_OK
}

@_cdecl("mm_power_to_db")
public func mm_power_to_db(
    _ ctx: UnsafeMutableRawPointer?,
    _ data: UnsafePointer<Float>?,
    _ count: Int64,
    _ ref: Float,
    _ amin: Float,
    _ topDb: Float,
    _ out: UnsafeMutablePointer<MMBuffer>?
) -> Int32 {
    guard let data = data, count > 0, let out = out else {
        return MM_ERR_INVALID_INPUT
    }

    let length = Int(count)
    let inputArray = Array(UnsafeBufferPointer(start: data, count: length))
    let signal = Signal(data: inputArray, sampleRate: 0)

    let topDbOpt: Float? = topDb > 0 ? topDb : nil
    let result = Scaling.powerToDb(signal, ref: ref, amin: amin, topDb: topDbOpt)

    let outCount = result.count
    let outData = UnsafeMutablePointer<Float>.allocate(capacity: outCount)
    result.withUnsafeBufferPointer { srcBuf in
        outData.initialize(from: srcBuf.baseAddress!, count: outCount)
    }

    out.pointee.data = outData
    out.pointee.ndim = 1
    withUnsafeMutablePointer(to: &out.pointee.shape) { tuplePtr in
        tuplePtr.withMemoryRebound(to: Int64.self, capacity: 8) { shapePtr in
            for i in 0..<8 { shapePtr[i] = 0 }
            shapePtr[0] = Int64(outCount)
        }
    }
    out.pointee.dtype = 0
    out.pointee.count = Int64(outCount)

    return MM_OK
}

// MARK: - Mel Spectrogram

@_cdecl("mm_mel_spectrogram")
public func mm_mel_spectrogram(
    _ ctx: UnsafeMutableRawPointer?,
    _ signalData: UnsafePointer<Float>?,
    _ signalLength: Int64,
    _ sampleRate: Int32,
    _ nFFT: Int32,
    _ hopLength: Int32,
    _ winLength: Int32,
    _ center: Int32,
    _ power: Float,
    _ nMels: Int32,
    _ fMin: Float,
    _ fMax: Float,
    _ out: UnsafeMutablePointer<MMBuffer>?
) -> Int32 {
    guard let ctx = ctx,
          let signalData = signalData,
          signalLength > 0,
          let out = out else {
        return MM_ERR_INVALID_INPUT
    }

    let _ = Unmanaged<MMContextInternal>.fromOpaque(ctx).takeUnretainedValue()

    // Copy input data into a Signal
    let length = Int(signalLength)
    let inputArray = Array(UnsafeBufferPointer(start: signalData, count: length))
    let signal = Signal(data: inputArray, sampleRate: Int(sampleRate))

    // fMax <= 0 means use sr/2 (Python passes 0.0 for None)
    let fMaxOpt: Float? = fMax > 0 ? fMax : nil
    // hopLength <= 0 means use default (nFFT/4)
    let hopOpt: Int? = hopLength > 0 ? Int(hopLength) : nil
    // winLength <= 0 means use default (nFFT)
    let winOpt: Int? = winLength > 0 ? Int(winLength) : nil

    let result = MelSpectrogram.compute(
        signal: signal,
        sr: Int(sampleRate),
        nFFT: Int(nFFT),
        hopLength: hopOpt,
        winLength: winOpt,
        center: center != 0,
        power: power,
        nMels: Int(nMels),
        fMin: fMin,
        fMax: fMaxOpt
    )

    // Fill output MMBuffer
    let outCount = result.count
    guard outCount > 0 else {
        out.pointee.data = nil
        out.pointee.ndim = Int32(result.shape.count)
        withUnsafeMutablePointer(to: &out.pointee.shape) { tuplePtr in
            tuplePtr.withMemoryRebound(to: Int64.self, capacity: 8) { shapePtr in
                for i in 0..<8 { shapePtr[i] = 0 }
                for i in 0..<min(result.shape.count, 8) {
                    shapePtr[i] = Int64(result.shape[i])
                }
            }
        }
        out.pointee.dtype = 0
        out.pointee.count = 0
        return MM_OK
    }

    let outData = UnsafeMutablePointer<Float>.allocate(capacity: outCount)
    result.withUnsafeBufferPointer { srcBuf in
        outData.initialize(from: srcBuf.baseAddress!, count: outCount)
    }

    out.pointee.data = outData
    out.pointee.ndim = Int32(result.shape.count)
    let shape = result.shape
    withUnsafeMutablePointer(to: &out.pointee.shape) { tuplePtr in
        tuplePtr.withMemoryRebound(to: Int64.self, capacity: 8) { shapePtr in
            for i in 0..<8 { shapePtr[i] = 0 }
            for i in 0..<min(shape.count, 8) {
                shapePtr[i] = Int64(shape[i])
            }
        }
    }
    out.pointee.dtype = 0  // float32
    out.pointee.count = Int64(outCount)

    return MM_OK
}

// MARK: - MFCC

@_cdecl("mm_mfcc")
public func mm_mfcc(
    _ ctx: UnsafeMutableRawPointer?,
    _ signalData: UnsafePointer<Float>?,
    _ signalLength: Int64,
    _ sampleRate: Int32,
    _ nMFCC: Int32,
    _ nFFT: Int32,
    _ hopLength: Int32,
    _ winLength: Int32,
    _ nMels: Int32,
    _ fMin: Float,
    _ fMax: Float,
    _ center: Int32,
    _ out: UnsafeMutablePointer<MMBuffer>?
) -> Int32 {
    guard let ctx = ctx,
          let signalData = signalData,
          signalLength > 0,
          let out = out else {
        return MM_ERR_INVALID_INPUT
    }

    let _ = Unmanaged<MMContextInternal>.fromOpaque(ctx).takeUnretainedValue()

    // Copy input data into a Signal
    let length = Int(signalLength)
    let inputArray = Array(UnsafeBufferPointer(start: signalData, count: length))
    let signal = Signal(data: inputArray, sampleRate: Int(sampleRate))

    // fMax <= 0 means use sr/2 (Python passes 0.0 for None)
    let fMaxOpt: Float? = fMax > 0 ? fMax : nil
    // hopLength <= 0 means use default (nFFT/4)
    let hopOpt: Int? = hopLength > 0 ? Int(hopLength) : nil
    // winLength <= 0 means use default (nFFT)
    let winOpt: Int? = winLength > 0 ? Int(winLength) : nil

    let result = MFCC.compute(
        signal: signal,
        sr: Int(sampleRate),
        nMFCC: Int(nMFCC),
        nFFT: Int(nFFT),
        hopLength: hopOpt,
        winLength: winOpt,
        nMels: Int(nMels),
        fMin: fMin,
        fMax: fMaxOpt,
        center: center != 0
    )

    // Fill output MMBuffer
    let outCount = result.count
    guard outCount > 0 else {
        out.pointee.data = nil
        out.pointee.ndim = Int32(result.shape.count)
        withUnsafeMutablePointer(to: &out.pointee.shape) { tuplePtr in
            tuplePtr.withMemoryRebound(to: Int64.self, capacity: 8) { shapePtr in
                for i in 0..<8 { shapePtr[i] = 0 }
                for i in 0..<min(result.shape.count, 8) {
                    shapePtr[i] = Int64(result.shape[i])
                }
            }
        }
        out.pointee.dtype = 0
        out.pointee.count = 0
        return MM_OK
    }

    let outData = UnsafeMutablePointer<Float>.allocate(capacity: outCount)
    result.withUnsafeBufferPointer { srcBuf in
        outData.initialize(from: srcBuf.baseAddress!, count: outCount)
    }

    out.pointee.data = outData
    out.pointee.ndim = Int32(result.shape.count)
    let shape = result.shape
    withUnsafeMutablePointer(to: &out.pointee.shape) { tuplePtr in
        tuplePtr.withMemoryRebound(to: Int64.self, capacity: 8) { shapePtr in
            for i in 0..<8 { shapePtr[i] = 0 }
            for i in 0..<min(shape.count, 8) {
                shapePtr[i] = Int64(shape[i])
            }
        }
    }
    out.pointee.dtype = 0  // float32
    out.pointee.count = Int64(outCount)

    return MM_OK
}

// MARK: - Chroma STFT

@_cdecl("mm_chroma_stft")
public func mm_chroma_stft(
    _ ctx: UnsafeMutableRawPointer?,
    _ signalData: UnsafePointer<Float>?,
    _ signalLength: Int64,
    _ sampleRate: Int32,
    _ nFFT: Int32,
    _ hopLength: Int32,
    _ winLength: Int32,
    _ nChroma: Int32,
    _ center: Int32,
    _ out: UnsafeMutablePointer<MMBuffer>?
) -> Int32 {
    guard let ctx = ctx,
          let signalData = signalData,
          signalLength > 0,
          let out = out else {
        return MM_ERR_INVALID_INPUT
    }

    let _ = Unmanaged<MMContextInternal>.fromOpaque(ctx).takeUnretainedValue()

    // Copy input data into a Signal
    let length = Int(signalLength)
    let inputArray = Array(UnsafeBufferPointer(start: signalData, count: length))
    let signal = Signal(data: inputArray, sampleRate: Int(sampleRate))

    // hopLength <= 0 means use default (nFFT/4)
    let hopOpt: Int? = hopLength > 0 ? Int(hopLength) : nil
    // winLength <= 0 means use default (nFFT)
    let winOpt: Int? = winLength > 0 ? Int(winLength) : nil

    let result = Chroma.stft(
        signal: signal,
        sr: Int(sampleRate),
        nFFT: Int(nFFT),
        hopLength: hopOpt,
        winLength: winOpt,
        nChroma: Int(nChroma),
        center: center != 0
    )

    // Fill output MMBuffer
    let outCount = result.count
    guard outCount > 0 else {
        out.pointee.data = nil
        out.pointee.ndim = Int32(result.shape.count)
        withUnsafeMutablePointer(to: &out.pointee.shape) { tuplePtr in
            tuplePtr.withMemoryRebound(to: Int64.self, capacity: 8) { shapePtr in
                for i in 0..<8 { shapePtr[i] = 0 }
                for i in 0..<min(result.shape.count, 8) {
                    shapePtr[i] = Int64(result.shape[i])
                }
            }
        }
        out.pointee.dtype = 0
        out.pointee.count = 0
        return MM_OK
    }

    let outData = UnsafeMutablePointer<Float>.allocate(capacity: outCount)
    result.withUnsafeBufferPointer { srcBuf in
        outData.initialize(from: srcBuf.baseAddress!, count: outCount)
    }

    out.pointee.data = outData
    out.pointee.ndim = Int32(result.shape.count)
    let shape = result.shape
    withUnsafeMutablePointer(to: &out.pointee.shape) { tuplePtr in
        tuplePtr.withMemoryRebound(to: Int64.self, capacity: 8) { shapePtr in
            for i in 0..<8 { shapePtr[i] = 0 }
            for i in 0..<min(shape.count, 8) {
                shapePtr[i] = Int64(shape[i])
            }
        }
    }
    out.pointee.dtype = 0  // float32
    out.pointee.count = Int64(outCount)

    return MM_OK
}

// MARK: - Spectral Descriptors (shared helper)

/// Fill an MMBuffer from a Signal result. Shared by all spectral descriptor bridge functions.
private func fillBuffer(_ result: Signal, _ out: UnsafeMutablePointer<MMBuffer>) -> Int32 {
    let outCount = result.count
    guard outCount > 0 else {
        out.pointee.data = nil
        out.pointee.ndim = Int32(result.shape.count)
        withUnsafeMutablePointer(to: &out.pointee.shape) { tuplePtr in
            tuplePtr.withMemoryRebound(to: Int64.self, capacity: 8) { shapePtr in
                for i in 0..<8 { shapePtr[i] = 0 }
                for i in 0..<min(result.shape.count, 8) {
                    shapePtr[i] = Int64(result.shape[i])
                }
            }
        }
        out.pointee.dtype = 0
        out.pointee.count = 0
        return MM_OK
    }

    let outData = UnsafeMutablePointer<Float>.allocate(capacity: outCount)
    result.withUnsafeBufferPointer { srcBuf in
        outData.initialize(from: srcBuf.baseAddress!, count: outCount)
    }

    out.pointee.data = outData
    out.pointee.ndim = Int32(result.shape.count)
    let shape = result.shape
    withUnsafeMutablePointer(to: &out.pointee.shape) { tuplePtr in
        tuplePtr.withMemoryRebound(to: Int64.self, capacity: 8) { shapePtr in
            for i in 0..<8 { shapePtr[i] = 0 }
            for i in 0..<min(shape.count, 8) {
                shapePtr[i] = Int64(shape[i])
            }
        }
    }
    out.pointee.dtype = 0  // float32
    out.pointee.count = Int64(outCount)

    return MM_OK
}

// MARK: - Spectral Centroid

@_cdecl("mm_spectral_centroid")
public func mm_spectral_centroid(
    _ ctx: UnsafeMutableRawPointer?,
    _ signalData: UnsafePointer<Float>?,
    _ signalLength: Int64,
    _ sampleRate: Int32,
    _ nFFT: Int32,
    _ hopLength: Int32,
    _ winLength: Int32,
    _ center: Int32,
    _ out: UnsafeMutablePointer<MMBuffer>?
) -> Int32 {
    guard let ctx = ctx,
          let signalData = signalData,
          signalLength > 0,
          let out = out else {
        return MM_ERR_INVALID_INPUT
    }

    let _ = Unmanaged<MMContextInternal>.fromOpaque(ctx).takeUnretainedValue()

    let length = Int(signalLength)
    let inputArray = Array(UnsafeBufferPointer(start: signalData, count: length))
    let signal = Signal(data: inputArray, sampleRate: Int(sampleRate))

    let hopOpt: Int? = hopLength > 0 ? Int(hopLength) : nil
    let winOpt: Int? = winLength > 0 ? Int(winLength) : nil

    let spec = STFT.compute(signal: signal, nFFT: Int(nFFT),
                            hopLength: hopOpt, winLength: winOpt,
                            center: center != 0)
    let result = SpectralDescriptors.centroid(spectrogram: spec, sr: Int(sampleRate),
                                              nFFT: Int(nFFT))
    return fillBuffer(result, out)
}

// MARK: - Spectral Bandwidth

@_cdecl("mm_spectral_bandwidth")
public func mm_spectral_bandwidth(
    _ ctx: UnsafeMutableRawPointer?,
    _ signalData: UnsafePointer<Float>?,
    _ signalLength: Int64,
    _ sampleRate: Int32,
    _ nFFT: Int32,
    _ hopLength: Int32,
    _ winLength: Int32,
    _ center: Int32,
    _ p: Float,
    _ out: UnsafeMutablePointer<MMBuffer>?
) -> Int32 {
    guard let ctx = ctx,
          let signalData = signalData,
          signalLength > 0,
          let out = out else {
        return MM_ERR_INVALID_INPUT
    }

    let _ = Unmanaged<MMContextInternal>.fromOpaque(ctx).takeUnretainedValue()

    let length = Int(signalLength)
    let inputArray = Array(UnsafeBufferPointer(start: signalData, count: length))
    let signal = Signal(data: inputArray, sampleRate: Int(sampleRate))

    let hopOpt: Int? = hopLength > 0 ? Int(hopLength) : nil
    let winOpt: Int? = winLength > 0 ? Int(winLength) : nil

    let spec = STFT.compute(signal: signal, nFFT: Int(nFFT),
                            hopLength: hopOpt, winLength: winOpt,
                            center: center != 0)
    let result = SpectralDescriptors.bandwidth(spectrogram: spec, sr: Int(sampleRate),
                                               nFFT: Int(nFFT), p: p)
    return fillBuffer(result, out)
}

// MARK: - Spectral Contrast

@_cdecl("mm_spectral_contrast")
public func mm_spectral_contrast(
    _ ctx: UnsafeMutableRawPointer?,
    _ signalData: UnsafePointer<Float>?,
    _ signalLength: Int64,
    _ sampleRate: Int32,
    _ nFFT: Int32,
    _ hopLength: Int32,
    _ winLength: Int32,
    _ center: Int32,
    _ nBands: Int32,
    _ fMin: Float,
    _ out: UnsafeMutablePointer<MMBuffer>?
) -> Int32 {
    guard let ctx = ctx,
          let signalData = signalData,
          signalLength > 0,
          let out = out else {
        return MM_ERR_INVALID_INPUT
    }

    let _ = Unmanaged<MMContextInternal>.fromOpaque(ctx).takeUnretainedValue()

    let length = Int(signalLength)
    let inputArray = Array(UnsafeBufferPointer(start: signalData, count: length))
    let signal = Signal(data: inputArray, sampleRate: Int(sampleRate))

    let hopOpt: Int? = hopLength > 0 ? Int(hopLength) : nil
    let winOpt: Int? = winLength > 0 ? Int(winLength) : nil

    let spec = STFT.compute(signal: signal, nFFT: Int(nFFT),
                            hopLength: hopOpt, winLength: winOpt,
                            center: center != 0)
    let result = SpectralDescriptors.contrast(spectrogram: spec, sr: Int(sampleRate),
                                              nFFT: Int(nFFT), nBands: Int(nBands),
                                              fMin: fMin)
    return fillBuffer(result, out)
}

// MARK: - Spectral Rolloff

@_cdecl("mm_spectral_rolloff")
public func mm_spectral_rolloff(
    _ ctx: UnsafeMutableRawPointer?,
    _ signalData: UnsafePointer<Float>?,
    _ signalLength: Int64,
    _ sampleRate: Int32,
    _ nFFT: Int32,
    _ hopLength: Int32,
    _ winLength: Int32,
    _ center: Int32,
    _ rollPercent: Float,
    _ out: UnsafeMutablePointer<MMBuffer>?
) -> Int32 {
    guard let ctx = ctx,
          let signalData = signalData,
          signalLength > 0,
          let out = out else {
        return MM_ERR_INVALID_INPUT
    }

    let _ = Unmanaged<MMContextInternal>.fromOpaque(ctx).takeUnretainedValue()

    let length = Int(signalLength)
    let inputArray = Array(UnsafeBufferPointer(start: signalData, count: length))
    let signal = Signal(data: inputArray, sampleRate: Int(sampleRate))

    let hopOpt: Int? = hopLength > 0 ? Int(hopLength) : nil
    let winOpt: Int? = winLength > 0 ? Int(winLength) : nil

    let spec = STFT.compute(signal: signal, nFFT: Int(nFFT),
                            hopLength: hopOpt, winLength: winOpt,
                            center: center != 0)
    let result = SpectralDescriptors.rolloff(spectrogram: spec, sr: Int(sampleRate),
                                             nFFT: Int(nFFT), rollPercent: rollPercent)
    return fillBuffer(result, out)
}

// MARK: - Spectral Flatness

@_cdecl("mm_spectral_flatness")
public func mm_spectral_flatness(
    _ ctx: UnsafeMutableRawPointer?,
    _ signalData: UnsafePointer<Float>?,
    _ signalLength: Int64,
    _ sampleRate: Int32,
    _ nFFT: Int32,
    _ hopLength: Int32,
    _ winLength: Int32,
    _ center: Int32,
    _ out: UnsafeMutablePointer<MMBuffer>?
) -> Int32 {
    guard let ctx = ctx,
          let signalData = signalData,
          signalLength > 0,
          let out = out else {
        return MM_ERR_INVALID_INPUT
    }

    let _ = Unmanaged<MMContextInternal>.fromOpaque(ctx).takeUnretainedValue()

    let length = Int(signalLength)
    let inputArray = Array(UnsafeBufferPointer(start: signalData, count: length))
    let signal = Signal(data: inputArray, sampleRate: Int(sampleRate))

    let hopOpt: Int? = hopLength > 0 ? Int(hopLength) : nil
    let winOpt: Int? = winLength > 0 ? Int(winLength) : nil

    let spec = STFT.compute(signal: signal, nFFT: Int(nFFT),
                            hopLength: hopOpt, winLength: winOpt,
                            center: center != 0)
    let result = SpectralDescriptors.flatness(spectrogram: spec)
    return fillBuffer(result, out)
}

// MARK: - RMS Energy

@_cdecl("mm_rms")
public func mm_rms(
    _ ctx: UnsafeMutableRawPointer?,
    _ signalData: UnsafePointer<Float>?,
    _ signalLength: Int64,
    _ sampleRate: Int32,
    _ frameLength: Int32,
    _ hopLength: Int32,
    _ center: Int32,
    _ out: UnsafeMutablePointer<MMBuffer>?
) -> Int32 {
    guard let ctx = ctx,
          let signalData = signalData,
          signalLength > 0,
          let out = out else {
        return MM_ERR_INVALID_INPUT
    }

    let _ = Unmanaged<MMContextInternal>.fromOpaque(ctx).takeUnretainedValue()

    let length = Int(signalLength)
    let inputArray = Array(UnsafeBufferPointer(start: signalData, count: length))
    let signal = Signal(data: inputArray, sampleRate: Int(sampleRate))

    let result = RMS.compute(
        signal: signal,
        frameLength: Int(frameLength),
        hopLength: Int(hopLength),
        center: center != 0
    )

    return fillBuffer(result, out)
}

// MARK: - Zero-Crossing Rate

@_cdecl("mm_zero_crossing_rate")
public func mm_zero_crossing_rate(
    _ ctx: UnsafeMutableRawPointer?,
    _ signalData: UnsafePointer<Float>?,
    _ signalLength: Int64,
    _ sampleRate: Int32,
    _ frameLength: Int32,
    _ hopLength: Int32,
    _ center: Int32,
    _ out: UnsafeMutablePointer<MMBuffer>?
) -> Int32 {
    guard let ctx = ctx,
          let signalData = signalData,
          signalLength > 0,
          let out = out else {
        return MM_ERR_INVALID_INPUT
    }

    let _ = Unmanaged<MMContextInternal>.fromOpaque(ctx).takeUnretainedValue()

    let length = Int(signalLength)
    let inputArray = Array(UnsafeBufferPointer(start: signalData, count: length))
    let signal = Signal(data: inputArray, sampleRate: Int(sampleRate))

    let result = ZeroCrossing.rate(
        signal: signal,
        frameLength: Int(frameLength),
        hopLength: Int(hopLength),
        center: center != 0
    )

    return fillBuffer(result, out)
}

// MARK: - Tonnetz

@_cdecl("mm_tonnetz")
public func mm_tonnetz(
    _ ctx: UnsafeMutableRawPointer?,
    _ signalData: UnsafePointer<Float>?,
    _ signalLength: Int64,
    _ sampleRate: Int32,
    _ nFFT: Int32,
    _ hopLength: Int32,
    _ winLength: Int32,
    _ nChroma: Int32,
    _ center: Int32,
    _ out: UnsafeMutablePointer<MMBuffer>?
) -> Int32 {
    guard let ctx = ctx,
          let signalData = signalData,
          signalLength > 0,
          let out = out else {
        return MM_ERR_INVALID_INPUT
    }

    let _ = Unmanaged<MMContextInternal>.fromOpaque(ctx).takeUnretainedValue()

    // Copy input data into a Signal
    let length = Int(signalLength)
    let inputArray = Array(UnsafeBufferPointer(start: signalData, count: length))
    let signal = Signal(data: inputArray, sampleRate: Int(sampleRate))

    // hopLength <= 0 means use default (nFFT/4)
    let hopOpt: Int? = hopLength > 0 ? Int(hopLength) : nil
    // winLength <= 0 means use default (nFFT)
    let winOpt: Int? = winLength > 0 ? Int(winLength) : nil

    let result = Tonnetz.compute(
        signal: signal,
        sr: Int(sampleRate),
        nFFT: Int(nFFT),
        hopLength: hopOpt,
        winLength: winOpt,
        nChroma: Int(nChroma),
        center: center != 0
    )

    return fillBuffer(result, out)
}

// MARK: - Delta Features

@_cdecl("mm_delta")
public func mm_delta(
    _ ctx: UnsafeMutableRawPointer?,
    _ data: UnsafePointer<Float>?,
    _ count: Int64,
    _ nFeatures: Int32,
    _ nFrames: Int32,
    _ width: Int32,
    _ order: Int32,
    _ out: UnsafeMutablePointer<MMBuffer>?
) -> Int32 {
    guard let data = data, count > 0, let out = out else {
        return MM_ERR_INVALID_INPUT
    }

    let length = Int(count)
    let inputArray = Array(UnsafeBufferPointer(start: data, count: length))
    let signal = Signal(data: inputArray, shape: [Int(nFeatures), Int(nFrames)], sampleRate: 0)

    let result = Delta.compute(data: signal, width: Int(width), order: Int(order))
    return fillBuffer(result, out)
}

// MARK: - Stack Memory

@_cdecl("mm_stack_memory")
public func mm_stack_memory(
    _ ctx: UnsafeMutableRawPointer?,
    _ data: UnsafePointer<Float>?,
    _ count: Int64,
    _ nFeatures: Int32,
    _ nFrames: Int32,
    _ nSteps: Int32,
    _ delay: Int32,
    _ out: UnsafeMutablePointer<MMBuffer>?
) -> Int32 {
    guard let data = data, count > 0, let out = out else {
        return MM_ERR_INVALID_INPUT
    }

    let length = Int(count)
    let inputArray = Array(UnsafeBufferPointer(start: data, count: length))
    let signal = Signal(data: inputArray, shape: [Int(nFeatures), Int(nFrames)], sampleRate: 0)

    let result = Delta.stackMemory(data: signal, nSteps: Int(nSteps), delay: Int(delay))
    return fillBuffer(result, out)
}

// MARK: - Poly Features

@_cdecl("mm_poly_features")
public func mm_poly_features(
    _ ctx: UnsafeMutableRawPointer?,
    _ data: UnsafePointer<Float>?,
    _ count: Int64,
    _ nFeatures: Int32,
    _ nFrames: Int32,
    _ order: Int32,
    _ sr: Int32,
    _ nFFT: Int32,
    _ out: UnsafeMutablePointer<MMBuffer>?
) -> Int32 {
    guard let data = data, count > 0, let out = out else {
        return MM_ERR_INVALID_INPUT
    }

    let length = Int(count)
    let inputArray = Array(UnsafeBufferPointer(start: data, count: length))
    let signal = Signal(data: inputArray, shape: [Int(nFeatures), Int(nFrames)], sampleRate: 0)

    let result = PolyFeatures.compute(data: signal, order: Int(order), sr: Int(sr), nFFT: Int(nFFT))
    return fillBuffer(result, out)
}

// MARK: - Audio Info

@_cdecl("mm_get_duration")
public func mm_get_duration(
    _ ctx: UnsafeMutableRawPointer?,
    _ path: UnsafePointer<CChar>?,
    _ outDuration: UnsafeMutablePointer<Float>?
) -> Int32 {
    guard let path = path, let outDuration = outDuration else {
        return MM_ERR_INVALID_INPUT
    }
    do {
        let dur = try AudioIO.getDuration(path: String(cString: path))
        outDuration.pointee = Float(dur)
        return MM_OK
    } catch {
        return MM_ERR_INVALID_INPUT
    }
}

@_cdecl("mm_get_sample_rate")
public func mm_get_sample_rate(
    _ ctx: UnsafeMutableRawPointer?,
    _ path: UnsafePointer<CChar>?,
    _ outSR: UnsafeMutablePointer<Int32>?
) -> Int32 {
    guard let path = path, let outSR = outSR else {
        return MM_ERR_INVALID_INPUT
    }
    do {
        let sr = try AudioIO.getSampleRate(path: String(cString: path))
        outSR.pointee = Int32(sr)
        return MM_OK
    } catch {
        return MM_ERR_INVALID_INPUT
    }
}

// MARK: - Audio Loading

@_cdecl("mm_load")
public func mm_load(
    _ ctx: UnsafeMutableRawPointer?,
    _ path: UnsafePointer<CChar>?,
    _ sr: Int32,
    _ mono: Int32,
    _ offset: Float,
    _ duration: Float,
    _ out: UnsafeMutablePointer<MMBuffer>?,
    _ outSR: UnsafeMutablePointer<Int32>?
) -> Int32 {
    guard let path = path, let out = out else {
        return MM_ERR_INVALID_INPUT
    }

    let pathStr = String(cString: path)
    let targetSR: Int? = sr > 0 ? Int(sr) : nil
    let dur: Double? = duration > 0 ? Double(duration) : nil

    do {
        let result = try AudioIO.load(
            path: pathStr,
            sr: targetSR,
            mono: mono != 0,
            offset: Double(offset),
            duration: dur
        )

        if let outSR = outSR {
            outSR.pointee = Int32(result.sampleRate)
        }

        return fillBuffer(result, out)
    } catch {
        return MM_ERR_INVALID_INPUT
    }
}

// MARK: - Resampling

@_cdecl("mm_resample")
public func mm_resample(
    _ ctx: UnsafeMutableRawPointer?,
    _ signalData: UnsafePointer<Float>?,
    _ signalLength: Int64,
    _ sourceSR: Int32,
    _ targetSR: Int32,
    _ out: UnsafeMutablePointer<MMBuffer>?
) -> Int32 {
    guard let signalData = signalData, signalLength > 0, let out = out else {
        return MM_ERR_INVALID_INPUT
    }

    let length = Int(signalLength)
    let inputArray = Array(UnsafeBufferPointer(start: signalData, count: length))
    let signal = Signal(data: inputArray, sampleRate: Int(sourceSR))

    let result = Resample.resample(signal: signal, targetRate: Int(targetSR))
    return fillBuffer(result, out)
}

// MARK: - Signal Generation

@_cdecl("mm_tone")
public func mm_tone(
    _ ctx: UnsafeMutableRawPointer?,
    _ frequency: Float,
    _ sr: Int32,
    _ length: Int64,
    _ phi: Float,
    _ out: UnsafeMutablePointer<MMBuffer>?
) -> Int32 {
    guard let out = out else { return MM_ERR_INVALID_INPUT }
    let result = SignalGen.tone(frequency: frequency, sr: Int(sr),
                                 length: Int(length), phi: phi)
    return fillBuffer(result, out)
}

@_cdecl("mm_chirp")
public func mm_chirp(
    _ ctx: UnsafeMutableRawPointer?,
    _ fmin: Float,
    _ fmax: Float,
    _ sr: Int32,
    _ length: Int64,
    _ linear: Int32,
    _ out: UnsafeMutablePointer<MMBuffer>?
) -> Int32 {
    guard let out = out else { return MM_ERR_INVALID_INPUT }
    let result = SignalGen.chirp(fmin: fmin, fmax: fmax, sr: Int(sr),
                                  length: Int(length), linear: linear != 0)
    return fillBuffer(result, out)
}

@_cdecl("mm_clicks")
public func mm_clicks(
    _ ctx: UnsafeMutableRawPointer?,
    _ times: UnsafePointer<Float>?,
    _ nTimes: Int32,
    _ sr: Int32,
    _ length: Int64,
    _ clickFreq: Float,
    _ clickDuration: Float,
    _ out: UnsafeMutablePointer<MMBuffer>?
) -> Int32 {
    guard let out = out else { return MM_ERR_INVALID_INPUT }

    var clickTimes: [Float]? = nil
    if let t = times, nTimes > 0 {
        clickTimes = Array(UnsafeBufferPointer(start: t, count: Int(nTimes)))
    }

    let len: Int? = length > 0 ? Int(length) : nil
    let result = SignalGen.clicks(times: clickTimes, sr: Int(sr), length: len,
                                    clickFreq: clickFreq, clickDuration: clickDuration)
    return fillBuffer(result, out)
}

// MARK: - Onset Evaluation

@_cdecl("mm_onset_evaluate")
public func mm_onset_evaluate(
    _ ctx: UnsafeMutableRawPointer?,
    _ reference: UnsafePointer<Float>?,
    _ nRef: Int32,
    _ estimated: UnsafePointer<Float>?,
    _ nEst: Int32,
    _ window: Float,
    _ outPrecision: UnsafeMutablePointer<Float>?,
    _ outRecall: UnsafeMutablePointer<Float>?,
    _ outFMeasure: UnsafeMutablePointer<Float>?
) -> Int32 {
    guard let outPrecision = outPrecision,
          let outRecall = outRecall,
          let outFMeasure = outFMeasure else {
        return MM_ERR_INVALID_INPUT
    }

    let refArray: [Float]
    if let r = reference, nRef > 0 {
        refArray = Array(UnsafeBufferPointer(start: r, count: Int(nRef)))
    } else {
        refArray = []
    }

    let estArray: [Float]
    if let e = estimated, nEst > 0 {
        estArray = Array(UnsafeBufferPointer(start: e, count: Int(nEst)))
    } else {
        estArray = []
    }

    let result = OnsetEval.evaluate(reference: refArray, estimated: estArray, window: window)
    outPrecision.pointee = result.precision
    outRecall.pointee = result.recall
    outFMeasure.pointee = result.fMeasure

    return MM_OK
}

// MARK: - Beat Evaluation

@_cdecl("mm_beat_evaluate")
public func mm_beat_evaluate(
    _ ctx: UnsafeMutableRawPointer?,
    _ reference: UnsafePointer<Float>?,
    _ nRef: Int32,
    _ estimated: UnsafePointer<Float>?,
    _ nEst: Int32,
    _ fmeasureWindow: Float,
    _ outFMeasure: UnsafeMutablePointer<Float>?,
    _ outCemgil: UnsafeMutablePointer<Float>?,
    _ outPScore: UnsafeMutablePointer<Float>?,
    _ outCmlC: UnsafeMutablePointer<Float>?,
    _ outCmlT: UnsafeMutablePointer<Float>?,
    _ outAmlC: UnsafeMutablePointer<Float>?,
    _ outAmlT: UnsafeMutablePointer<Float>?
) -> Int32 {
    guard let outFMeasure = outFMeasure,
          let outCemgil = outCemgil,
          let outPScore = outPScore,
          let outCmlC = outCmlC,
          let outCmlT = outCmlT,
          let outAmlC = outAmlC,
          let outAmlT = outAmlT else {
        return MM_ERR_INVALID_INPUT
    }

    let refArray: [Float]
    if let r = reference, nRef > 0 {
        refArray = Array(UnsafeBufferPointer(start: r, count: Int(nRef)))
    } else {
        refArray = []
    }

    let estArray: [Float]
    if let e = estimated, nEst > 0 {
        estArray = Array(UnsafeBufferPointer(start: e, count: Int(nEst)))
    } else {
        estArray = []
    }

    let result = BeatEval.evaluate(reference: refArray, estimated: estArray,
                                   fMeasureWindow: fmeasureWindow)
    outFMeasure.pointee = result.fMeasure
    outCemgil.pointee = result.cemgil
    outPScore.pointee = result.pScore
    outCmlC.pointee = result.cmlC
    outCmlT.pointee = result.cmlT
    outAmlC.pointee = result.amlC
    outAmlT.pointee = result.amlT

    return MM_OK
}

// MARK: - Tempo Evaluation

@_cdecl("mm_tempo_evaluate")
public func mm_tempo_evaluate(
    _ ctx: UnsafeMutableRawPointer?,
    _ refTempo: Float,
    _ estTempo: Float,
    _ tolerance: Float,
    _ outPScore: UnsafeMutablePointer<Float>?
) -> Int32 {
    guard let outPScore = outPScore else {
        return MM_ERR_INVALID_INPUT
    }

    let tol = tolerance > 0 ? tolerance : 0.08
    outPScore.pointee = TempoEval.pScore(referenceTempo: refTempo, estimatedTempo: estTempo,
                                         tolerance: tol)

    return MM_OK
}

// MARK: - Chord Evaluation

@_cdecl("mm_chord_accuracy")
public func mm_chord_accuracy(
    _ ctx: UnsafeMutableRawPointer?,
    _ reference: UnsafePointer<Int32>?,
    _ estimated: UnsafePointer<Int32>?,
    _ n: Int32,
    _ outAccuracy: UnsafeMutablePointer<Float>?
) -> Int32 {
    guard let outAccuracy = outAccuracy else {
        return MM_ERR_INVALID_INPUT
    }

    let refArray: [Int32]
    if let r = reference, n > 0 {
        refArray = Array(UnsafeBufferPointer(start: r, count: Int(n)))
    } else {
        refArray = []
    }

    let estArray: [Int32]
    if let e = estimated, n > 0 {
        estArray = Array(UnsafeBufferPointer(start: e, count: Int(n)))
    } else {
        estArray = []
    }

    outAccuracy.pointee = ChordEval.accuracy(reference: refArray, estimated: estArray)

    return MM_OK
}

// MARK: - Onset Strength

@_cdecl("mm_onset_strength")
public func mm_onset_strength(
    _ ctx: UnsafeMutableRawPointer?,
    _ signalData: UnsafePointer<Float>?,
    _ signalLength: Int64,
    _ sampleRate: Int32,
    _ nFFT: Int32,
    _ hopLength: Int32,
    _ nMels: Int32,
    _ fMin: Float,
    _ fMax: Float,
    _ center: Int32,
    _ aggregate: Int32,
    _ out: UnsafeMutablePointer<MMBuffer>?
) -> Int32 {
    guard let signalData = signalData, signalLength > 0, let out = out else {
        return MM_ERR_INVALID_INPUT
    }

    let length = Int(signalLength)
    let inputArray = Array(UnsafeBufferPointer(start: signalData, count: length))
    let signal = Signal(data: inputArray, sampleRate: Int(sampleRate))

    let hopOpt: Int? = hopLength > 0 ? Int(hopLength) : nil
    let fMaxOpt: Float? = fMax > 0 ? fMax : nil

    let result = OnsetDetection.onsetStrength(
        signal: signal,
        sr: Int(sampleRate),
        nFFT: Int(nFFT),
        hopLength: hopOpt,
        nMels: Int(nMels),
        fmin: fMin,
        fmax: fMaxOpt,
        center: center != 0,
        aggregate: aggregate != 0
    )

    return fillBuffer(result, out)
}

// MARK: - Onset Detection (Peak Picking)

@_cdecl("mm_onset_detect")
public func mm_onset_detect(
    _ ctx: UnsafeMutableRawPointer?,
    _ signalData: UnsafePointer<Float>?,
    _ signalLength: Int64,
    _ sampleRate: Int32,
    _ nFFT: Int32,
    _ hopLength: Int32,
    _ nMels: Int32,
    _ fMin: Float,
    _ fMax: Float,
    _ center: Int32,
    _ preMax: Int32,
    _ postMax: Int32,
    _ preAvg: Int32,
    _ postAvg: Int32,
    _ delta: Float,
    _ wait: Int32,
    _ backtrack: Int32,
    _ out: UnsafeMutablePointer<MMBuffer>?
) -> Int32 {
    guard let signalData = signalData, signalLength > 0, let out = out else {
        return MM_ERR_INVALID_INPUT
    }

    let length = Int(signalLength)
    let inputArray = Array(UnsafeBufferPointer(start: signalData, count: length))
    let signal = Signal(data: inputArray, sampleRate: Int(sampleRate))

    let hopOpt: Int? = hopLength > 0 ? Int(hopLength) : nil
    let fMaxOpt: Float? = fMax > 0 ? fMax : nil

    let result = OnsetDetection.detectOnsets(
        signal: signal,
        sr: Int(sampleRate),
        nFFT: Int(nFFT),
        hopLength: hopOpt,
        nMels: Int(nMels),
        fmin: fMin,
        fmax: fMaxOpt,
        center: center != 0,
        preMax: Int(preMax),
        postMax: Int(postMax),
        preAvg: Int(preAvg),
        postAvg: Int(postAvg),
        delta: delta,
        wait: Int(wait),
        doBacktrack: backtrack != 0
    )

    return fillBuffer(result, out)
}

// MARK: - Tempo Estimation

@_cdecl("mm_tempo")
public func mm_tempo(
    _ ctx: UnsafeMutableRawPointer?,
    _ signalData: UnsafePointer<Float>?,
    _ signalLength: Int64,
    _ sampleRate: Int32,
    _ hopLength: Int32,
    _ nFFT: Int32,
    _ nMels: Int32,
    _ fMin: Float,
    _ fMax: Float,
    _ startBPM: Float,
    _ center: Int32,
    _ outTempo: UnsafeMutablePointer<Float>?
) -> Int32 {
    guard let signalData = signalData,
          signalLength > 0,
          let outTempo = outTempo else {
        return MM_ERR_INVALID_INPUT
    }

    let length = Int(signalLength)
    let inputArray = Array(UnsafeBufferPointer(start: signalData, count: length))
    let signal = Signal(data: inputArray, sampleRate: Int(sampleRate))

    let hopOpt: Int? = hopLength > 0 ? Int(hopLength) : nil
    let fMaxOpt: Float? = fMax > 0 ? fMax : nil

    let tempo = TempoEstimation.tempo(
        signal: signal,
        sr: Int(sampleRate),
        hopLength: hopOpt,
        nFFT: Int(nFFT),
        nMels: Int(nMels),
        fmin: fMin,
        fmax: fMaxOpt,
        startBPM: startBPM,
        center: center != 0
    )

    outTempo.pointee = tempo
    return MM_OK
}

// MARK: - Beat Tracking

@_cdecl("mm_beat_track")
public func mm_beat_track(
    _ ctx: UnsafeMutableRawPointer?,
    _ signalData: UnsafePointer<Float>?,
    _ signalLength: Int64,
    _ sampleRate: Int32,
    _ hopLength: Int32,
    _ nFFT: Int32,
    _ nMels: Int32,
    _ fMin: Float,
    _ fMax: Float,
    _ startBPM: Float,
    _ trim: Int32,
    _ outTempo: UnsafeMutablePointer<Float>?,
    _ outBeats: UnsafeMutablePointer<MMBuffer>?
) -> Int32 {
    guard let signalData = signalData,
          signalLength > 0,
          let outTempo = outTempo,
          let outBeats = outBeats else {
        return MM_ERR_INVALID_INPUT
    }

    let length = Int(signalLength)
    let inputArray = Array(UnsafeBufferPointer(start: signalData, count: length))
    let signal = Signal(data: inputArray, sampleRate: Int(sampleRate))

    let hopOpt: Int? = hopLength > 0 ? Int(hopLength) : nil
    let fMaxOpt: Float? = fMax > 0 ? fMax : nil
    let doTrim = trim != 0

    let (tempo, beats) = BeatTracker.beatTrack(
        signal: signal,
        sr: Int(sampleRate),
        hopLength: hopOpt,
        nFFT: Int(nFFT),
        nMels: Int(nMels),
        fmin: fMin,
        fmax: fMaxOpt,
        startBPM: startBPM,
        trimFirst: doTrim,
        trimLast: doTrim
    )

    outTempo.pointee = tempo
    return fillBuffer(beats, outBeats)
}

// MARK: - Tempogram (Autocorrelation)

@_cdecl("mm_tempogram")
public func mm_tempogram(
    _ ctx: UnsafeMutableRawPointer?,
    _ signalData: UnsafePointer<Float>?,
    _ signalLength: Int64,
    _ sampleRate: Int32,
    _ hopLength: Int32,
    _ nFFT: Int32,
    _ nMels: Int32,
    _ fMin: Float,
    _ fMax: Float,
    _ center: Int32,
    _ winLength: Int32,
    _ out: UnsafeMutablePointer<MMBuffer>?
) -> Int32 {
    guard let signalData = signalData,
          signalLength > 0,
          let out = out else {
        return MM_ERR_INVALID_INPUT
    }

    let length = Int(signalLength)
    let inputArray = Array(UnsafeBufferPointer(start: signalData, count: length))
    let signal = Signal(data: inputArray, sampleRate: Int(sampleRate))

    let hopOpt: Int? = hopLength > 0 ? Int(hopLength) : nil
    let fMaxOpt: Float? = fMax > 0 ? fMax : nil

    let result = Tempogram.autocorrelation(
        signal: signal,
        sr: Int(sampleRate),
        hopLength: hopOpt,
        nFFT: Int(nFFT),
        nMels: Int(nMels),
        fmin: fMin,
        fmax: fMaxOpt,
        center: center != 0,
        winLength: Int(winLength)
    )

    return fillBuffer(result, out)
}

// MARK: - Tempogram (Fourier)

@_cdecl("mm_fourier_tempogram")
public func mm_fourier_tempogram(
    _ ctx: UnsafeMutableRawPointer?,
    _ signalData: UnsafePointer<Float>?,
    _ signalLength: Int64,
    _ sampleRate: Int32,
    _ hopLength: Int32,
    _ nFFT: Int32,
    _ nMels: Int32,
    _ fMin: Float,
    _ fMax: Float,
    _ center: Int32,
    _ winLength: Int32,
    _ out: UnsafeMutablePointer<MMBuffer>?
) -> Int32 {
    guard let signalData = signalData,
          signalLength > 0,
          let out = out else {
        return MM_ERR_INVALID_INPUT
    }

    let length = Int(signalLength)
    let inputArray = Array(UnsafeBufferPointer(start: signalData, count: length))
    let signal = Signal(data: inputArray, sampleRate: Int(sampleRate))

    let hopOpt: Int? = hopLength > 0 ? Int(hopLength) : nil
    let fMaxOpt: Float? = fMax > 0 ? fMax : nil

    let result = Tempogram.fourier(
        signal: signal,
        sr: Int(sampleRate),
        hopLength: hopOpt,
        nFFT: Int(nFFT),
        nMels: Int(nMels),
        fmin: fMin,
        fmax: fMaxOpt,
        center: center != 0,
        winLength: Int(winLength)
    )

    return fillBuffer(result, out)
}

// MARK: - Predominant Local Pulse (PLP)

@_cdecl("mm_plp")
public func mm_plp(
    _ ctx: UnsafeMutableRawPointer?,
    _ signalData: UnsafePointer<Float>?,
    _ signalLength: Int64,
    _ sampleRate: Int32,
    _ hopLength: Int32,
    _ nFFT: Int32,
    _ nMels: Int32,
    _ fMin: Float,
    _ fMax: Float,
    _ center: Int32,
    _ winLength: Int32,
    _ tempoMin: Float,
    _ tempoMax: Float,
    _ out: UnsafeMutablePointer<MMBuffer>?
) -> Int32 {
    guard let signalData = signalData,
          signalLength > 0,
          let out = out else {
        return MM_ERR_INVALID_INPUT
    }

    let length = Int(signalLength)
    let inputArray = Array(UnsafeBufferPointer(start: signalData, count: length))
    let signal = Signal(data: inputArray, sampleRate: Int(sampleRate))

    let hopOpt: Int? = hopLength > 0 ? Int(hopLength) : nil
    let fMaxOpt: Float? = fMax > 0 ? fMax : nil

    let result = BeatTracker.plp(
        signal: signal,
        sr: Int(sampleRate),
        hopLength: hopOpt,
        nFFT: Int(nFFT),
        nMels: Int(nMels),
        fmin: fMin,
        fmax: fMaxOpt,
        center: center != 0,
        winLength: Int(winLength),
        tempoMin: tempoMin,
        tempoMax: tempoMax
    )

    return fillBuffer(result, out)
}

// MARK: - YIN Pitch Estimation

@_cdecl("mm_yin")
public func mm_yin(
    _ ctx: UnsafeMutableRawPointer?,
    _ signalData: UnsafePointer<Float>?,
    _ signalLength: Int64,
    _ sampleRate: Int32,
    _ fMin: Float,
    _ fMax: Float,
    _ frameLength: Int32,
    _ hopLength: Int32,
    _ troughThreshold: Float,
    _ center: Int32,
    _ out: UnsafeMutablePointer<MMBuffer>?
) -> Int32 {
    guard let signalData = signalData,
          signalLength > 0,
          let out = out else {
        return MM_ERR_INVALID_INPUT
    }

    let length = Int(signalLength)
    let inputArray = Array(UnsafeBufferPointer(start: signalData, count: length))
    let signal = Signal(data: inputArray, sampleRate: Int(sampleRate))

    // hopLength <= 0 means use default (frameLength/4)
    let hopOpt: Int? = hopLength > 0 ? Int(hopLength) : nil

    let result = YIN.yin(
        signal: signal,
        fMin: fMin,
        fMax: fMax,
        sr: Int(sampleRate),
        frameLength: Int(frameLength),
        hopLength: hopOpt,
        troughThreshold: troughThreshold,
        center: center != 0
    )

    return fillBuffer(result, out)
}

// MARK: - pYIN Probabilistic Pitch Estimation

@_cdecl("mm_pyin")
public func mm_pyin(
    _ ctx: UnsafeMutableRawPointer?,
    _ signalData: UnsafePointer<Float>?,
    _ signalLength: Int64,
    _ sampleRate: Int32,
    _ fMin: Float,
    _ fMax: Float,
    _ frameLength: Int32,
    _ hopLength: Int32,
    _ nThresholds: Int32,
    _ betaAlpha: Float,
    _ betaBeta: Float,
    _ resolution: Float,
    _ switchProb: Float,
    _ center: Int32,
    _ out: UnsafeMutablePointer<MMBuffer>?
) -> Int32 {
    guard let signalData = signalData,
          signalLength > 0,
          let out = out else {
        return MM_ERR_INVALID_INPUT
    }

    let length = Int(signalLength)
    let inputArray = Array(UnsafeBufferPointer(start: signalData, count: length))
    let signal = Signal(data: inputArray, sampleRate: Int(sampleRate))

    // hopLength <= 0 means use default (frameLength/4)
    let hopOpt: Int? = hopLength > 0 ? Int(hopLength) : nil

    let result = YIN.pyin(
        signal: signal,
        fMin: fMin,
        fMax: fMax,
        sr: Int(sampleRate),
        frameLength: Int(frameLength),
        hopLength: hopOpt,
        nThresholds: Int(nThresholds),
        betaAlpha: betaAlpha,
        betaBeta: betaBeta,
        resolution: resolution,
        switchProb: switchProb,
        center: center != 0
    )

    return fillBuffer(result, out)
}

// MARK: - Piptrack (Parabolic Interpolation Pitch Tracking)

@_cdecl("mm_piptrack")
public func mm_piptrack(
    _ ctx: UnsafeMutableRawPointer?,
    _ signalData: UnsafePointer<Float>?,
    _ signalLength: Int64,
    _ sampleRate: Int32,
    _ nFFT: Int32,
    _ hopLength: Int32,
    _ winLength: Int32,
    _ fMin: Float,
    _ fMax: Float,
    _ threshold: Float,
    _ center: Int32,
    _ out: UnsafeMutablePointer<MMBuffer>?
) -> Int32 {
    guard let signalData = signalData,
          signalLength > 0,
          let out = out else {
        return MM_ERR_INVALID_INPUT
    }

    let length = Int(signalLength)
    let inputArray = Array(UnsafeBufferPointer(start: signalData, count: length))
    let signal = Signal(data: inputArray, sampleRate: Int(sampleRate))

    // hopLength <= 0 means use default (nFFT/4)
    let hopOpt: Int? = hopLength > 0 ? Int(hopLength) : nil
    // winLength <= 0 means use default (nFFT)
    let winOpt: Int? = winLength > 0 ? Int(winLength) : nil

    let result = Piptrack.piptrack(
        signal: signal,
        sr: Int(sampleRate),
        nFFT: Int(nFFT),
        hopLength: hopOpt,
        winLength: winOpt,
        fMin: fMin,
        fMax: fMax,
        threshold: threshold,
        center: center != 0
    )

    return fillBuffer(result, out)
}

// MARK: - Tuning Estimation

@_cdecl("mm_estimate_tuning")
public func mm_estimate_tuning(
    _ ctx: UnsafeMutableRawPointer?,
    _ signalData: UnsafePointer<Float>?,
    _ signalLength: Int64,
    _ sampleRate: Int32,
    _ nFFT: Int32,
    _ hopLength: Int32,
    _ winLength: Int32,
    _ resolution: Float,
    _ binsPerOctave: Int32,
    _ center: Int32,
    _ outTuning: UnsafeMutablePointer<Float>?
) -> Int32 {
    guard let signalData = signalData,
          signalLength > 0,
          let outTuning = outTuning else {
        return MM_ERR_INVALID_INPUT
    }

    let length = Int(signalLength)
    let inputArray = Array(UnsafeBufferPointer(start: signalData, count: length))
    let signal = Signal(data: inputArray, sampleRate: Int(sampleRate))

    // hopLength <= 0 means use default (nFFT/4)
    let hopOpt: Int? = hopLength > 0 ? Int(hopLength) : nil
    // winLength <= 0 means use default (nFFT)
    let winOpt: Int? = winLength > 0 ? Int(winLength) : nil

    let tuning = Tuning.estimateTuning(
        signal: signal,
        sr: Int(sampleRate),
        nFFT: Int(nFFT),
        hopLength: hopOpt,
        winLength: winOpt,
        resolution: resolution,
        binsPerOctave: Int(binsPerOctave),
        center: center != 0
    )

    outTuning.pointee = tuning
    return MM_OK
}

// MARK: - HPSS (Harmonic-Percussive Source Separation)

@_cdecl("mm_hpss")
public func mm_hpss(
    _ ctx: UnsafeMutableRawPointer?,
    _ signalData: UnsafePointer<Float>?,
    _ signalLength: Int64,
    _ sampleRate: Int32,
    _ nFFT: Int32,
    _ hopLength: Int32,
    _ winLength: Int32,
    _ center: Int32,
    _ kernelSize: Int32,
    _ power: Float,
    _ margin: Float,
    _ out: UnsafeMutablePointer<MMBuffer>?
) -> Int32 {
    guard let signalData = signalData,
          signalLength > 0,
          let out = out else {
        return MM_ERR_INVALID_INPUT
    }

    let length = Int(signalLength)
    let inputArray = Array(UnsafeBufferPointer(start: signalData, count: length))
    let signal = Signal(data: inputArray, sampleRate: Int(sampleRate))

    let hopOpt: Int? = hopLength > 0 ? Int(hopLength) : nil
    let winOpt: Int? = winLength > 0 ? Int(winLength) : nil

    let (h, p) = HPSS.hpss(
        signal: signal,
        sr: Int(sampleRate),
        nFFT: Int(nFFT),
        hopLength: hopOpt,
        winLength: winOpt,
        center: center != 0,
        kernelSize: Int(kernelSize),
        power: power,
        margin: margin
    )

    // Pack into shape [2, signalLength]: row 0 = harmonic, row 1 = percussive
    let outLength = h.count
    let totalCount = 2 * outLength
    let outData = UnsafeMutablePointer<Float>.allocate(capacity: totalCount)

    h.withUnsafeBufferPointer { hBuf in
        outData.initialize(from: hBuf.baseAddress!, count: outLength)
    }
    p.withUnsafeBufferPointer { pBuf in
        (outData + outLength).initialize(from: pBuf.baseAddress!, count: outLength)
    }

    out.pointee.data = outData
    out.pointee.ndim = 2
    withUnsafeMutablePointer(to: &out.pointee.shape) { tuplePtr in
        tuplePtr.withMemoryRebound(to: Int64.self, capacity: 8) { shapePtr in
            for i in 0..<8 { shapePtr[i] = 0 }
            shapePtr[0] = 2
            shapePtr[1] = Int64(outLength)
        }
    }
    out.pointee.dtype = 0
    out.pointee.count = Int64(totalCount)

    return MM_OK
}

@_cdecl("mm_harmonic")
public func mm_harmonic(
    _ ctx: UnsafeMutableRawPointer?,
    _ signalData: UnsafePointer<Float>?,
    _ signalLength: Int64,
    _ sampleRate: Int32,
    _ nFFT: Int32,
    _ hopLength: Int32,
    _ winLength: Int32,
    _ center: Int32,
    _ kernelSize: Int32,
    _ power: Float,
    _ margin: Float,
    _ out: UnsafeMutablePointer<MMBuffer>?
) -> Int32 {
    guard let signalData = signalData,
          signalLength > 0,
          let out = out else {
        return MM_ERR_INVALID_INPUT
    }

    let length = Int(signalLength)
    let inputArray = Array(UnsafeBufferPointer(start: signalData, count: length))
    let signal = Signal(data: inputArray, sampleRate: Int(sampleRate))

    let hopOpt: Int? = hopLength > 0 ? Int(hopLength) : nil
    let winOpt: Int? = winLength > 0 ? Int(winLength) : nil

    let result = HPSS.harmonic(
        signal: signal,
        sr: Int(sampleRate),
        nFFT: Int(nFFT),
        hopLength: hopOpt,
        winLength: winOpt,
        center: center != 0,
        kernelSize: Int(kernelSize),
        power: power,
        margin: margin
    )

    return fillBuffer(result, out)
}

@_cdecl("mm_percussive")
public func mm_percussive(
    _ ctx: UnsafeMutableRawPointer?,
    _ signalData: UnsafePointer<Float>?,
    _ signalLength: Int64,
    _ sampleRate: Int32,
    _ nFFT: Int32,
    _ hopLength: Int32,
    _ winLength: Int32,
    _ center: Int32,
    _ kernelSize: Int32,
    _ power: Float,
    _ margin: Float,
    _ out: UnsafeMutablePointer<MMBuffer>?
) -> Int32 {
    guard let signalData = signalData,
          signalLength > 0,
          let out = out else {
        return MM_ERR_INVALID_INPUT
    }

    let length = Int(signalLength)
    let inputArray = Array(UnsafeBufferPointer(start: signalData, count: length))
    let signal = Signal(data: inputArray, sampleRate: Int(sampleRate))

    let hopOpt: Int? = hopLength > 0 ? Int(hopLength) : nil
    let winOpt: Int? = winLength > 0 ? Int(winLength) : nil

    let result = HPSS.percussive(
        signal: signal,
        sr: Int(sampleRate),
        nFFT: Int(nFFT),
        hopLength: hopOpt,
        winLength: winOpt,
        center: center != 0,
        kernelSize: Int(kernelSize),
        power: power,
        margin: margin
    )

    return fillBuffer(result, out)
}

// MARK: - Time Stretching

@_cdecl("mm_time_stretch")
public func mm_time_stretch(
    _ ctx: UnsafeMutableRawPointer?,
    _ signalData: UnsafePointer<Float>?,
    _ signalLength: Int64,
    _ sampleRate: Int32,
    _ rate: Float,
    _ nFFT: Int32,
    _ hopLength: Int32,
    _ out: UnsafeMutablePointer<MMBuffer>?
) -> Int32 {
    guard let signalData = signalData,
          signalLength > 0,
          rate > 0,
          let out = out else {
        return MM_ERR_INVALID_INPUT
    }

    let length = Int(signalLength)
    let inputArray = Array(UnsafeBufferPointer(start: signalData, count: length))
    let signal = Signal(data: inputArray, sampleRate: Int(sampleRate))

    let hopOpt: Int? = hopLength > 0 ? Int(hopLength) : nil

    let result = TimeStretch.timeStretch(
        signal: signal,
        rate: rate,
        sr: Int(sampleRate),
        nFFT: Int(nFFT),
        hopLength: hopOpt
    )

    return fillBuffer(result, out)
}

// MARK: - Pitch Shifting

@_cdecl("mm_pitch_shift")
public func mm_pitch_shift(
    _ ctx: UnsafeMutableRawPointer?,
    _ signalData: UnsafePointer<Float>?,
    _ signalLength: Int64,
    _ sampleRate: Int32,
    _ nSteps: Float,
    _ binsPerOctave: Int32,
    _ nFFT: Int32,
    _ hopLength: Int32,
    _ out: UnsafeMutablePointer<MMBuffer>?
) -> Int32 {
    guard let signalData = signalData,
          signalLength > 0,
          let out = out else {
        return MM_ERR_INVALID_INPUT
    }

    let length = Int(signalLength)
    let inputArray = Array(UnsafeBufferPointer(start: signalData, count: length))
    let signal = Signal(data: inputArray, sampleRate: Int(sampleRate))

    let hopOpt: Int? = hopLength > 0 ? Int(hopLength) : nil

    let result = PitchShift.pitchShift(
        signal: signal,
        sr: Int(sampleRate),
        nSteps: nSteps,
        binsPerOctave: Int(binsPerOctave),
        nFFT: Int(nFFT),
        hopLength: hopOpt
    )

    return fillBuffer(result, out)
}

// MARK: - Trim (Silence Trimming)

@_cdecl("mm_trim")
public func mm_trim(
    _ ctx: UnsafeMutableRawPointer?,
    _ signalData: UnsafePointer<Float>?,
    _ signalLength: Int64,
    _ sampleRate: Int32,
    _ topDb: Float,
    _ frameLength: Int32,
    _ hopLength: Int32,
    _ out: UnsafeMutablePointer<MMBuffer>?,
    _ outStart: UnsafeMutablePointer<Int64>?,
    _ outEnd: UnsafeMutablePointer<Int64>?
) -> Int32 {
    guard let signalData = signalData,
          signalLength > 0,
          let out = out,
          let outStart = outStart,
          let outEnd = outEnd else {
        return MM_ERR_INVALID_INPUT
    }

    let length = Int(signalLength)
    let inputArray = Array(UnsafeBufferPointer(start: signalData, count: length))
    let signal = Signal(data: inputArray, sampleRate: Int(sampleRate))

    let (trimmed, startIdx, endIdx) = Trim.trim(
        signal: signal,
        topDb: topDb,
        frameLength: Int(frameLength),
        hopLength: Int(hopLength)
    )

    outStart.pointee = Int64(startIdx)
    outEnd.pointee = Int64(endIdx)

    return fillBuffer(trimmed, out)
}

// MARK: - Split (Non-Silent Interval Detection)

@_cdecl("mm_split")
public func mm_split(
    _ ctx: UnsafeMutableRawPointer?,
    _ signalData: UnsafePointer<Float>?,
    _ signalLength: Int64,
    _ sampleRate: Int32,
    _ topDb: Float,
    _ frameLength: Int32,
    _ hopLength: Int32,
    _ out: UnsafeMutablePointer<MMBuffer>?
) -> Int32 {
    guard let signalData = signalData,
          signalLength > 0,
          let out = out else {
        return MM_ERR_INVALID_INPUT
    }

    let length = Int(signalLength)
    let inputArray = Array(UnsafeBufferPointer(start: signalData, count: length))
    let signal = Signal(data: inputArray, sampleRate: Int(sampleRate))

    let intervals = Split.split(
        signal: signal,
        topDb: topDb,
        frameLength: Int(frameLength),
        hopLength: Int(hopLength)
    )

    let nIntervals = intervals.count
    if nIntervals == 0 {
        // Return empty buffer with shape [0, 2]
        let emptySignal = Signal(data: [], shape: [0, 2], sampleRate: Int(sampleRate))
        return fillBuffer(emptySignal, out)
    }

    // Pack intervals as flat float array: [start0, end0, start1, end1, ...]
    var flatData = [Float](repeating: 0, count: nIntervals * 2)
    for (i, interval) in intervals.enumerated() {
        flatData[i * 2] = Float(interval.start)
        flatData[i * 2 + 1] = Float(interval.end)
    }

    let result = Signal(data: flatData, shape: [nIntervals, 2], sampleRate: Int(sampleRate))
    return fillBuffer(result, out)
}

// MARK: - Preemphasis

@_cdecl("mm_preemphasis")
public func mm_preemphasis(
    _ ctx: UnsafeMutableRawPointer?,
    _ signalData: UnsafePointer<Float>?,
    _ signalLength: Int64,
    _ sampleRate: Int32,
    _ coef: Float,
    _ out: UnsafeMutablePointer<MMBuffer>?
) -> Int32 {
    guard let signalData = signalData,
          signalLength > 0,
          let out = out else {
        return MM_ERR_INVALID_INPUT
    }

    let length = Int(signalLength)
    let inputArray = Array(UnsafeBufferPointer(start: signalData, count: length))
    let signal = Signal(data: inputArray, sampleRate: Int(sampleRate))

    let result = Preemphasis.preemphasis(signal: signal, coef: coef)
    return fillBuffer(result, out)
}

// MARK: - Deemphasis

@_cdecl("mm_deemphasis")
public func mm_deemphasis(
    _ ctx: UnsafeMutableRawPointer?,
    _ signalData: UnsafePointer<Float>?,
    _ signalLength: Int64,
    _ sampleRate: Int32,
    _ coef: Float,
    _ out: UnsafeMutablePointer<MMBuffer>?
) -> Int32 {
    guard let signalData = signalData,
          signalLength > 0,
          let out = out else {
        return MM_ERR_INVALID_INPUT
    }

    let length = Int(signalLength)
    let inputArray = Array(UnsafeBufferPointer(start: signalData, count: length))
    let signal = Signal(data: inputArray, sampleRate: Int(sampleRate))

    let result = Preemphasis.deemphasis(signal: signal, coef: coef)
    return fillBuffer(result, out)
}

// MARK: - Neural Beat Decode

@_cdecl("mm_neural_beat_decode")
public func mm_neural_beat_decode(
    _ ctx: UnsafeMutableRawPointer?,
    _ activations: UnsafePointer<Float>?,
    _ nFrames: Int32,
    _ fps: Float,
    _ minBPM: Float,
    _ maxBPM: Float,
    _ transitionLambda: Float,
    _ threshold: Float,
    _ trim: Int32,
    _ outTempo: UnsafeMutablePointer<Float>?,
    _ outBeats: UnsafeMutablePointer<MMBuffer>?
) -> Int32 {
    guard let activations = activations,
          nFrames > 0,
          let outTempo = outTempo,
          let outBeats = outBeats else {
        return MM_ERR_INVALID_INPUT
    }

    let actArray = Array(UnsafeBufferPointer(start: activations, count: Int(nFrames)))

    let (tempo, beats) = NeuralBeatTracker.decode(
        activations: actArray,
        fps: fps,
        minBPM: minBPM,
        maxBPM: maxBPM,
        transitionLambda: transitionLambda,
        threshold: threshold,
        trim: trim != 0
    )

    outTempo.pointee = tempo

    let floatBeats = beats.map { Float($0) }
    let result = Signal(data: floatBeats.isEmpty ? [] : floatBeats,
                        shape: [floatBeats.count], sampleRate: 0)
    return fillBuffer(result, outBeats)
}

// MARK: - Neural Onset Detect

@_cdecl("mm_neural_onset_detect")
public func mm_neural_onset_detect(
    _ ctx: UnsafeMutableRawPointer?,
    _ activations: UnsafePointer<Float>?,
    _ nFrames: Int32,
    _ fps: Float,
    _ threshold: Float,
    _ preMax: Int32,
    _ postMax: Int32,
    _ preAvg: Int32,
    _ postAvg: Int32,
    _ combineMethod: Int32,   // 0=fixed, 1=adaptive, 2=combined
    _ wait: Int32,
    _ out: UnsafeMutablePointer<MMBuffer>?
) -> Int32 {
    guard let activations = activations,
          nFrames > 0,
          let out = out else {
        return MM_ERR_INVALID_INPUT
    }

    let actArray = Array(UnsafeBufferPointer(start: activations, count: Int(nFrames)))

    let method: NeuralOnsetDetector.ThresholdMethod
    switch combineMethod {
    case 0: method = .fixed
    case 2: method = .combined
    default: method = .adaptive
    }

    let onsetFrames = NeuralOnsetDetector.detect(
        activations: actArray,
        fps: fps,
        threshold: threshold,
        preMax: Int(preMax),
        postMax: Int(postMax),
        preAvg: Int(preAvg),
        postAvg: Int(postAvg),
        combineMethod: method,
        wait: Int(wait)
    )

    let floatFrames = onsetFrames.map { Float($0) }
    let result = Signal(data: floatFrames.isEmpty ? [] : floatFrames,
                        shape: [floatFrames.count], sampleRate: 0)
    return fillBuffer(result, out)
}

// MARK: - Downbeat Detection

@_cdecl("mm_downbeat_detect")
public func mm_downbeat_detect(
    _ ctx: UnsafeMutableRawPointer?,
    _ activations: UnsafePointer<Float>?,
    _ nFrames: Int32,
    _ fps: Float,
    _ beatsPerBar: Int32,
    _ minBPM: Float,
    _ maxBPM: Float,
    _ transitionLambda: Float,
    _ outBeats: UnsafeMutablePointer<MMBuffer>?,
    _ outDownbeats: UnsafeMutablePointer<MMBuffer>?
) -> Int32 {
    guard let activations = activations,
          nFrames > 0,
          let outBeats = outBeats,
          let outDownbeats = outDownbeats else {
        return MM_ERR_INVALID_INPUT
    }

    let totalCount = Int(nFrames) * 3
    let actArray = Array(UnsafeBufferPointer(start: activations, count: totalCount))

    let (beatFrames, downbeatFrames, _) = Downbeat.decode(
        activations: actArray,
        nFrames: Int(nFrames),
        fps: fps,
        beatsPerBar: Int(beatsPerBar),
        minBPM: minBPM,
        maxBPM: maxBPM,
        transitionLambda: transitionLambda
    )

    // Fill beat frames buffer
    let floatBeats = beatFrames.map { Float($0) }
    let beatsSignal = Signal(data: floatBeats.isEmpty ? [] : floatBeats,
                             shape: [floatBeats.count], sampleRate: 0)
    let beatsStatus = fillBuffer(beatsSignal, outBeats)
    guard beatsStatus == MM_OK else { return beatsStatus }

    // Fill downbeat frames buffer
    let floatDownbeats = downbeatFrames.map { Float($0) }
    let downbeatsSignal = Signal(data: floatDownbeats.isEmpty ? [] : floatDownbeats,
                                 shape: [floatDownbeats.count], sampleRate: 0)
    return fillBuffer(downbeatsSignal, outDownbeats)
}

// MARK: - Key Detection

@_cdecl("mm_key_detect")
public func mm_key_detect(
    _ ctx: UnsafeMutableRawPointer?,
    _ activations: UnsafePointer<Float>?,
    _ nFrames: Int32,
    _ outKeyIndex: UnsafeMutablePointer<Int32>?,
    _ outConfidence: UnsafeMutablePointer<Float>?,
    _ outProbabilities: UnsafeMutablePointer<MMBuffer>?
) -> Int32 {
    guard let activations = activations,
          nFrames > 0,
          let outKeyIndex = outKeyIndex,
          let outConfidence = outConfidence,
          let outProbabilities = outProbabilities else {
        return MM_ERR_INVALID_INPUT
    }

    let totalCount = Int(nFrames) * 24
    let actArray = Array(UnsafeBufferPointer(start: activations, count: totalCount))

    let result: KeyDetection.KeyResult
    if nFrames == 1 {
        result = KeyDetection.detect(activations: actArray)
    } else {
        result = KeyDetection.detectFromSequence(activations: actArray, nFrames: Int(nFrames))
    }

    outKeyIndex.pointee = Int32(result.keyIndex)
    outConfidence.pointee = result.confidence

    // Fill probabilities buffer (24 floats)
    let probSignal = Signal(data: result.probabilities, shape: [24], sampleRate: 0)
    return fillBuffer(probSignal, outProbabilities)
}

// MARK: - Chord Recognition

@_cdecl("mm_chord_detect")
public func mm_chord_detect(
    _ ctx: UnsafeMutableRawPointer?,
    _ activations: UnsafePointer<Float>?,
    _ nFrames: Int32,
    _ nClasses: Int32,
    _ transitionScores: UnsafePointer<Float>?,
    _ selfTransitionBias: Float,
    _ out: UnsafeMutablePointer<MMBuffer>?
) -> Int32 {
    guard let activations = activations,
          nFrames > 0,
          nClasses > 0,
          let out = out else {
        return MM_ERR_INVALID_INPUT
    }

    let totalCount = Int(nFrames) * Int(nClasses)
    let actArray = Array(UnsafeBufferPointer(start: activations, count: totalCount))

    var transArray: [Float]? = nil
    if let ts = transitionScores {
        let transCount = Int(nClasses) * Int(nClasses)
        transArray = Array(UnsafeBufferPointer(start: ts, count: transCount))
    }

    let events = ChordRecognition.decode(
        activations: actArray,
        nFrames: Int(nFrames),
        nClasses: Int(nClasses),
        transitionScores: transArray,
        selfTransitionBias: selfTransitionBias
    )

    let nEvents = events.count
    if nEvents == 0 {
        let emptySignal = Signal(data: [], shape: [0, 3], sampleRate: 0)
        return fillBuffer(emptySignal, out)
    }

    // Pack as [nEvents, 3] float array: (startFrame, endFrame, chordIndex)
    var flatData = [Float](repeating: 0, count: nEvents * 3)
    for (i, event) in events.enumerated() {
        flatData[i * 3 + 0] = Float(event.startFrame)
        flatData[i * 3 + 1] = Float(event.endFrame)
        flatData[i * 3 + 2] = Float(event.chordIndex)
    }

    let result = Signal(data: flatData, shape: [nEvents, 3], sampleRate: 0)
    return fillBuffer(result, out)
}

// MARK: - Piano Transcription

@_cdecl("mm_piano_transcribe")
public func mm_piano_transcribe(
    _ ctx: UnsafeMutableRawPointer?,
    _ activations: UnsafePointer<Float>?,
    _ nFrames: Int32,
    _ threshold: Float,
    _ minDuration: Int32,
    _ useHMM: Int32,
    _ out: UnsafeMutablePointer<MMBuffer>?
) -> Int32 {
    guard let activations = activations,
          nFrames > 0,
          let out = out else {
        return MM_ERR_INVALID_INPUT
    }

    let totalCount = Int(nFrames) * 88
    let actArray = Array(UnsafeBufferPointer(start: activations, count: totalCount))

    let events: [PianoTranscription.NoteEvent]
    if useHMM != 0 {
        events = PianoTranscription.detectHMM(
            activations: actArray,
            nFrames: Int(nFrames),
            onsetProb: 0.01,
            offsetProb: 0.05,
            minDuration: Int(minDuration)
        )
    } else {
        events = PianoTranscription.detect(
            activations: actArray,
            nFrames: Int(nFrames),
            threshold: threshold,
            minDuration: Int(minDuration)
        )
    }

    let nEvents = events.count
    if nEvents == 0 {
        let emptySignal = Signal(data: [], shape: [0, 4], sampleRate: 0)
        return fillBuffer(emptySignal, out)
    }

    // Pack as [nEvents, 4] float array: (midiNote, onsetFrame, offsetFrame, velocity)
    var flatData = [Float](repeating: 0, count: nEvents * 4)
    for (i, event) in events.enumerated() {
        flatData[i * 4 + 0] = Float(event.midiNote)
        flatData[i * 4 + 1] = Float(event.onsetFrame)
        flatData[i * 4 + 2] = Float(event.offsetFrame)
        flatData[i * 4 + 3] = event.velocity
    }

    let result = Signal(data: flatData, shape: [nEvents, 4], sampleRate: 0)
    return fillBuffer(result, out)
}

// MARK: - CQT (Constant-Q Transform)

@_cdecl("mm_cqt")
public func mm_cqt(
    _ ctx: UnsafeMutableRawPointer?,
    _ signalData: UnsafePointer<Float>?,
    _ signalLength: Int64,
    _ sampleRate: Int32,
    _ hopLength: Int32,
    _ fMin: Float,
    _ fMax: Float,
    _ binsPerOctave: Int32,
    _ nFFT: Int32,
    _ center: Int32,
    _ out: UnsafeMutablePointer<MMBuffer>?
) -> Int32 {
    guard let ctx = ctx,
          let signalData = signalData,
          signalLength > 0,
          let out = out else {
        return MM_ERR_INVALID_INPUT
    }

    let _ = Unmanaged<MMContextInternal>.fromOpaque(ctx).takeUnretainedValue()

    let length = Int(signalLength)
    let inputArray = Array(UnsafeBufferPointer(start: signalData, count: length))
    let signal = Signal(data: inputArray, sampleRate: Int(sampleRate))

    let hopOpt: Int? = hopLength > 0 ? Int(hopLength) : nil
    let fMaxOpt: Float? = fMax > 0 ? fMax : nil
    let nFFTOpt: Int? = nFFT > 0 ? Int(nFFT) : nil

    let result = CQT.compute(
        signal: signal,
        hopLength: hopOpt,
        fMin: fMin,
        fMax: fMaxOpt,
        binsPerOctave: Int(binsPerOctave),
        nFFT: nFFTOpt,
        center: center != 0
    )

    return fillBuffer(result, out)
}

// MARK: - VQT (Variable-Q Transform)

@_cdecl("mm_vqt")
public func mm_vqt(
    _ ctx: UnsafeMutableRawPointer?,
    _ signalData: UnsafePointer<Float>?,
    _ signalLength: Int64,
    _ sampleRate: Int32,
    _ hopLength: Int32,
    _ fMin: Float,
    _ fMax: Float,
    _ binsPerOctave: Int32,
    _ gamma: Float,
    _ nFFT: Int32,
    _ center: Int32,
    _ out: UnsafeMutablePointer<MMBuffer>?
) -> Int32 {
    guard let ctx = ctx,
          let signalData = signalData,
          signalLength > 0,
          let out = out else {
        return MM_ERR_INVALID_INPUT
    }

    let _ = Unmanaged<MMContextInternal>.fromOpaque(ctx).takeUnretainedValue()

    let length = Int(signalLength)
    let inputArray = Array(UnsafeBufferPointer(start: signalData, count: length))
    let signal = Signal(data: inputArray, sampleRate: Int(sampleRate))

    let hopOpt: Int? = hopLength > 0 ? Int(hopLength) : nil
    let fMaxOpt: Float? = fMax > 0 ? fMax : nil
    let nFFTOpt: Int? = nFFT > 0 ? Int(nFFT) : nil

    let result = CQT.vqt(
        signal: signal,
        hopLength: hopOpt,
        fMin: fMin,
        fMax: fMaxOpt,
        binsPerOctave: Int(binsPerOctave),
        gamma: gamma,
        nFFT: nFFTOpt,
        center: center != 0
    )

    return fillBuffer(result, out)
}

// MARK: - Hybrid CQT

@_cdecl("mm_hybrid_cqt")
public func mm_hybrid_cqt(
    _ ctx: UnsafeMutableRawPointer?,
    _ signalData: UnsafePointer<Float>?,
    _ signalLength: Int64,
    _ sampleRate: Int32,
    _ hopLength: Int32,
    _ fMin: Float,
    _ fMax: Float,
    _ binsPerOctave: Int32,
    _ nFFT: Int32,
    _ center: Int32,
    _ out: UnsafeMutablePointer<MMBuffer>?
) -> Int32 {
    guard let ctx = ctx,
          let signalData = signalData,
          signalLength > 0,
          let out = out else {
        return MM_ERR_INVALID_INPUT
    }

    let _ = Unmanaged<MMContextInternal>.fromOpaque(ctx).takeUnretainedValue()

    let length = Int(signalLength)
    let inputArray = Array(UnsafeBufferPointer(start: signalData, count: length))
    let signal = Signal(data: inputArray, sampleRate: Int(sampleRate))

    let hopOpt: Int? = hopLength > 0 ? Int(hopLength) : nil
    let fMaxOpt: Float? = fMax > 0 ? fMax : nil
    let nFFTOpt: Int? = nFFT > 0 ? Int(nFFT) : nil

    let result = CQT.hybridCQT(
        signal: signal,
        hopLength: hopOpt,
        fMin: fMin,
        fMax: fMaxOpt,
        binsPerOctave: Int(binsPerOctave),
        nFFT: nFFTOpt,
        center: center != 0
    )

    return fillBuffer(result, out)
}

// MARK: - Reassigned Spectrogram

@_cdecl("mm_reassigned_spectrogram")
public func mm_reassigned_spectrogram(
    _ ctx: UnsafeMutableRawPointer?,
    _ signalData: UnsafePointer<Float>?,
    _ signalLength: Int64,
    _ sampleRate: Int32,
    _ nFFT: Int32,
    _ hopLength: Int32,
    _ winLength: Int32,
    _ center: Int32,
    _ out: UnsafeMutablePointer<MMBuffer>?
) -> Int32 {
    guard let ctx = ctx,
          let signalData = signalData,
          signalLength > 0,
          let out = out else {
        return MM_ERR_INVALID_INPUT
    }

    let _ = Unmanaged<MMContextInternal>.fromOpaque(ctx).takeUnretainedValue()

    let length = Int(signalLength)
    let inputArray = Array(UnsafeBufferPointer(start: signalData, count: length))
    let signal = Signal(data: inputArray, sampleRate: Int(sampleRate))

    let hopOpt: Int? = hopLength > 0 ? Int(hopLength) : nil
    let winOpt: Int? = winLength > 0 ? Int(winLength) : nil

    let (mag, freqs, times) = ReassignedSpectrogram.compute(
        signal: signal,
        sr: Int(sampleRate),
        nFFT: Int(nFFT),
        hopLength: hopOpt,
        winLength: winOpt,
        center: center != 0
    )

    // Pack three outputs into shape [3, nFreqs, nFrames]
    let nFreqs = mag.shape[0]
    let nFrames = mag.shape.count > 1 ? mag.shape[1] : 0
    let planeSize = nFreqs * nFrames
    let totalCount = 3 * planeSize

    guard planeSize > 0 else {
        // Empty result
        let emptySignal = Signal(data: [], shape: [3, nFreqs, 0], sampleRate: Int(sampleRate))
        return fillBuffer(emptySignal, out)
    }

    let outData = UnsafeMutablePointer<Float>.allocate(capacity: totalCount)

    mag.withUnsafeBufferPointer { magBuf in
        outData.initialize(from: magBuf.baseAddress!, count: planeSize)
    }
    freqs.withUnsafeBufferPointer { freqBuf in
        (outData + planeSize).initialize(from: freqBuf.baseAddress!, count: planeSize)
    }
    times.withUnsafeBufferPointer { timeBuf in
        (outData + 2 * planeSize).initialize(from: timeBuf.baseAddress!, count: planeSize)
    }

    out.pointee.data = outData
    out.pointee.ndim = 3
    withUnsafeMutablePointer(to: &out.pointee.shape) { tuplePtr in
        tuplePtr.withMemoryRebound(to: Int64.self, capacity: 8) { shapePtr in
            for i in 0..<8 { shapePtr[i] = 0 }
            shapePtr[0] = 3
            shapePtr[1] = Int64(nFreqs)
            shapePtr[2] = Int64(nFrames)
        }
    }
    out.pointee.dtype = 0
    out.pointee.count = Int64(totalCount)

    return MM_OK
}

// MARK: - Phase Vocoder

@_cdecl("mm_phase_vocoder")
public func mm_phase_vocoder(
    _ ctx: UnsafeMutableRawPointer?,
    _ stftData: UnsafePointer<Float>?,
    _ stftCount: Int64,
    _ nFreqs: Int32,
    _ nFrames: Int32,
    _ sampleRate: Int32,
    _ rate: Float,
    _ hopLength: Int32,
    _ out: UnsafeMutablePointer<MMBuffer>?
) -> Int32 {
    guard let stftData = stftData,
          stftCount > 0,
          nFreqs > 0,
          nFrames > 0,
          rate > 0,
          let out = out else {
        return MM_ERR_INVALID_INPUT
    }

    // Build complex Signal from interleaved float data
    let count = Int(stftCount)
    let inputArray = Array(UnsafeBufferPointer(start: stftData, count: count))
    let complexSTFT = Signal(complexData: inputArray, shape: [Int(nFreqs), Int(nFrames)],
                              sampleRate: Int(sampleRate))

    let hopOpt: Int? = hopLength > 0 ? Int(hopLength) : nil

    let result = PhaseVocoder.phaseVocoder(
        complexSTFT: complexSTFT,
        rate: rate,
        hopLength: hopOpt
    )

    return fillBuffer(result, out)
}

// MARK: - Griffin-Lim

@_cdecl("mm_griffinlim")
public func mm_griffinlim(
    _ ctx: UnsafeMutableRawPointer?,
    _ magData: UnsafePointer<Float>?,
    _ magCount: Int64,
    _ nFreqs: Int32,
    _ nFrames: Int32,
    _ sampleRate: Int32,
    _ nIter: Int32,
    _ hopLength: Int32,
    _ winLength: Int32,
    _ center: Int32,
    _ outputLength: Int64,
    _ out: UnsafeMutablePointer<MMBuffer>?
) -> Int32 {
    guard let magData = magData,
          magCount > 0,
          nFreqs > 0,
          nFrames > 0,
          nIter > 0,
          let out = out else {
        return MM_ERR_INVALID_INPUT
    }

    let count = Int(magCount)
    let inputArray = Array(UnsafeBufferPointer(start: magData, count: count))
    let magnitude = Signal(data: inputArray, shape: [Int(nFreqs), Int(nFrames)],
                            sampleRate: Int(sampleRate))

    let hopOpt: Int? = hopLength > 0 ? Int(hopLength) : nil
    let winOpt: Int? = winLength > 0 ? Int(winLength) : nil
    let lenOpt: Int? = outputLength > 0 ? Int(outputLength) : nil

    let result = PhaseVocoder.griffinLim(
        magnitude: magnitude,
        nIter: Int(nIter),
        hopLength: hopOpt,
        winLength: winOpt,
        center: center != 0,
        length: lenOpt
    )

    return fillBuffer(result, out)
}

// MARK: - Griffin-Lim CQT

@_cdecl("mm_griffinlim_cqt")
public func mm_griffinlim_cqt(
    _ ctx: UnsafeMutableRawPointer?,
    _ magData: UnsafePointer<Float>?,
    _ magCount: Int64,
    _ nBins: Int32,
    _ nFrames: Int32,
    _ sr: Int32,
    _ nIter: Int32,
    _ hopLength: Int32,
    _ fmin: Float,
    _ binsPerOctave: Int32,
    _ out: UnsafeMutablePointer<MMBuffer>?
) -> Int32 {
    guard let magData = magData,
          magCount > 0,
          nBins > 0,
          nFrames > 0,
          nIter > 0,
          let out = out else {
        return MM_ERR_INVALID_INPUT
    }

    let count = Int(magCount)
    let inputArray = Array(UnsafeBufferPointer(start: magData, count: count))
    let magnitude = Signal(data: inputArray, shape: [Int(nBins), Int(nFrames)],
                            sampleRate: Int(sr))

    let hopOpt: Int? = hopLength > 0 ? Int(hopLength) : nil

    let result = PhaseVocoder.griffinLimCQT(
        magnitude: magnitude,
        sr: Int(sr),
        nIter: Int(nIter),
        hopLength: hopOpt,
        fMin: fmin,
        binsPerOctave: Int(binsPerOctave)
    )

    return fillBuffer(result, out)
}

// MARK: - PCEN

@_cdecl("mm_pcen")
public func mm_pcen(
    _ ctx: UnsafeMutableRawPointer?,
    _ data: UnsafePointer<Float>?,
    _ count: Int64,
    _ nBands: Int32,
    _ nFrames: Int32,
    _ sampleRate: Int32,
    _ hopLength: Int32,
    _ gain: Float,
    _ bias: Float,
    _ power: Float,
    _ timeConstant: Float,
    _ eps: Float,
    _ out: UnsafeMutablePointer<MMBuffer>?
) -> Int32 {
    guard let data = data, count > 0,
          nBands > 0, nFrames > 0,
          let out = out else {
        return MM_ERR_INVALID_INPUT
    }

    let length = Int(count)
    let inputArray = Array(UnsafeBufferPointer(start: data, count: length))
    let signal = Signal(data: inputArray, shape: [Int(nBands), Int(nFrames)],
                        sampleRate: Int(sampleRate))

    let hopOpt = hopLength > 0 ? Int(hopLength) : 512

    let result = Scaling.pcen(
        signal,
        sr: Int(sampleRate),
        hopLength: hopOpt,
        gain: gain,
        bias: bias,
        power: power,
        timeConstant: timeConstant,
        eps: eps
    )

    return fillBuffer(result, out)
}

// MARK: - A-Weighting

@_cdecl("mm_a_weighting")
public func mm_a_weighting(
    _ ctx: UnsafeMutableRawPointer?,
    _ frequencies: UnsafePointer<Float>?,
    _ nFreqs: Int32,
    _ out: UnsafeMutablePointer<MMBuffer>?
) -> Int32 {
    guard let frequencies = frequencies, nFreqs > 0,
          let out = out else {
        return MM_ERR_INVALID_INPUT
    }

    let freqArray = Array(UnsafeBufferPointer(start: frequencies, count: Int(nFreqs)))
    let weights = Scaling.aWeighting(frequencies: freqArray)

    let result = Signal(data: weights, shape: [Int(nFreqs)], sampleRate: 0)
    return fillBuffer(result, out)
}

// MARK: - Chroma CQT

@_cdecl("mm_chroma_cqt")
public func mm_chroma_cqt(
    _ ctx: UnsafeMutableRawPointer?,
    _ signalData: UnsafePointer<Float>?,
    _ signalLength: Int64,
    _ sampleRate: Int32,
    _ hopLength: Int32,
    _ fMin: Float,
    _ binsPerOctave: Int32,
    _ nOctaves: Int32,
    _ nChroma: Int32,
    _ norm: Float,
    _ out: UnsafeMutablePointer<MMBuffer>?
) -> Int32 {
    guard let ctx = ctx,
          let signalData = signalData,
          signalLength > 0,
          let out = out else {
        return MM_ERR_INVALID_INPUT
    }

    let _ = Unmanaged<MMContextInternal>.fromOpaque(ctx).takeUnretainedValue()

    let length = Int(signalLength)
    let inputArray = Array(UnsafeBufferPointer(start: signalData, count: length))
    let signal = Signal(data: inputArray, sampleRate: Int(sampleRate))

    let hopOpt: Int? = hopLength > 0 ? Int(hopLength) : nil
    // norm <= 0 signals no normalization
    let normOpt: Float? = norm > 0 ? norm : nil

    let result = Chroma.cqt(
        signal: signal,
        sr: Int(sampleRate),
        hopLength: hopOpt,
        fMin: fMin,
        binsPerOctave: Int(binsPerOctave),
        nOctaves: Int(nOctaves),
        nChroma: Int(nChroma),
        norm: normOpt
    )

    return fillBuffer(result, out)
}

// MARK: - Chroma VQT

@_cdecl("mm_chroma_vqt")
public func mm_chroma_vqt(
    _ ctx: UnsafeMutableRawPointer?,
    _ signalData: UnsafePointer<Float>?,
    _ signalLength: Int64,
    _ sampleRate: Int32,
    _ hopLength: Int32,
    _ fMin: Float,
    _ binsPerOctave: Int32,
    _ nOctaves: Int32,
    _ gamma: Float,
    _ nChroma: Int32,
    _ norm: Float,
    _ out: UnsafeMutablePointer<MMBuffer>?
) -> Int32 {
    guard let ctx = ctx,
          let signalData = signalData,
          signalLength > 0,
          let out = out else {
        return MM_ERR_INVALID_INPUT
    }

    let _ = Unmanaged<MMContextInternal>.fromOpaque(ctx).takeUnretainedValue()

    let length = Int(signalLength)
    let inputArray = Array(UnsafeBufferPointer(start: signalData, count: length))
    let signal = Signal(data: inputArray, sampleRate: Int(sampleRate))

    let hopOpt: Int? = hopLength > 0 ? Int(hopLength) : nil
    // norm <= 0 signals no normalization
    let normOpt: Float? = norm > 0 ? norm : nil

    let result = Chroma.vqt(
        signal: signal,
        sr: Int(sampleRate),
        hopLength: hopOpt,
        fMin: fMin,
        binsPerOctave: Int(binsPerOctave),
        nOctaves: Int(nOctaves),
        gamma: gamma,
        nChroma: Int(nChroma),
        norm: normOpt
    )

    return fillBuffer(result, out)
}

// MARK: - Chroma CENS

@_cdecl("mm_chroma_cens")
public func mm_chroma_cens(
    _ ctx: UnsafeMutableRawPointer?,
    _ signalData: UnsafePointer<Float>?,
    _ signalLength: Int64,
    _ sampleRate: Int32,
    _ hopLength: Int32,
    _ fMin: Float,
    _ binsPerOctave: Int32,
    _ nOctaves: Int32,
    _ nChroma: Int32,
    _ winLenSmooth: Int32,
    _ out: UnsafeMutablePointer<MMBuffer>?
) -> Int32 {
    guard let ctx = ctx,
          let signalData = signalData,
          signalLength > 0,
          let out = out else {
        return MM_ERR_INVALID_INPUT
    }

    let _ = Unmanaged<MMContextInternal>.fromOpaque(ctx).takeUnretainedValue()

    let length = Int(signalLength)
    let inputArray = Array(UnsafeBufferPointer(start: signalData, count: length))
    let signal = Signal(data: inputArray, sampleRate: Int(sampleRate))

    let hopOpt: Int? = hopLength > 0 ? Int(hopLength) : nil

    let result = Chroma.cens(
        signal: signal,
        sr: Int(sampleRate),
        hopLength: hopOpt,
        fMin: fMin,
        binsPerOctave: Int(binsPerOctave),
        nOctaves: Int(nOctaves),
        nChroma: Int(nChroma),
        winLenSmooth: Int(winLenSmooth)
    )

    return fillBuffer(result, out)
}

// MARK: - Mel to Audio (Feature Inversion)

@_cdecl("mm_mel_to_audio")
public func mm_mel_to_audio(
    _ ctx: UnsafeMutableRawPointer?,
    _ melData: UnsafePointer<Float>?,
    _ melCount: Int64,
    _ nMels: Int32,
    _ nFrames: Int32,
    _ sampleRate: Int32,
    _ nFFT: Int32,
    _ hopLength: Int32,
    _ winLength: Int32,
    _ center: Int32,
    _ nIter: Int32,
    _ power: Float,
    _ fMin: Float,
    _ fMax: Float,
    _ outputLength: Int64,
    _ out: UnsafeMutablePointer<MMBuffer>?
) -> Int32 {
    guard let melData = melData,
          melCount > 0,
          nMels > 0,
          nFrames > 0,
          let out = out else {
        return MM_ERR_INVALID_INPUT
    }

    let count = Int(melCount)
    let inputArray = Array(UnsafeBufferPointer(start: melData, count: count))
    let melSpec = Signal(data: inputArray, shape: [Int(nMels), Int(nFrames)],
                         sampleRate: Int(sampleRate))

    let hopOpt: Int? = hopLength > 0 ? Int(hopLength) : nil
    let winOpt: Int? = winLength > 0 ? Int(winLength) : nil
    let lenOpt: Int? = outputLength > 0 ? Int(outputLength) : nil
    let fMaxOpt: Float? = fMax > 0 ? fMax : nil

    let result = Inversion.melToAudio(
        melSpectrogram: melSpec,
        sr: Int(sampleRate),
        nFFT: Int(nFFT),
        hopLength: hopOpt,
        winLength: winOpt,
        center: center != 0,
        power: power,
        nIter: Int(nIter),
        fMin: fMin,
        fMax: fMaxOpt,
        length: lenOpt
    )

    return fillBuffer(result, out)
}

// MARK: - MFCC to Mel (Feature Inversion)

@_cdecl("mm_mfcc_to_mel")
public func mm_mfcc_to_mel(
    _ ctx: UnsafeMutableRawPointer?,
    _ mfccData: UnsafePointer<Float>?,
    _ mfccCount: Int64,
    _ nMFCC: Int32,
    _ nFrames: Int32,
    _ sampleRate: Int32,
    _ nMels: Int32,
    _ out: UnsafeMutablePointer<MMBuffer>?
) -> Int32 {
    guard let mfccData = mfccData,
          mfccCount > 0,
          nMFCC > 0,
          nFrames > 0,
          nMels > 0,
          let out = out else {
        return MM_ERR_INVALID_INPUT
    }

    let count = Int(mfccCount)
    let inputArray = Array(UnsafeBufferPointer(start: mfccData, count: count))
    let mfcc = Signal(data: inputArray, shape: [Int(nMFCC), Int(nFrames)],
                      sampleRate: Int(sampleRate))

    let result = Inversion.mfccToMel(mfcc: mfcc, nMels: Int(nMels))

    return fillBuffer(result, out)
}

// MARK: - NMF (Non-negative Matrix Factorization)

@_cdecl("mm_nmf")
public func mm_nmf(
    _ ctx: UnsafeMutableRawPointer?,
    _ data: UnsafePointer<Float>?,
    _ nFeatures: Int32,
    _ nSamples: Int32,
    _ sampleRate: Int32,
    _ nComponents: Int32,
    _ nIter: Int32,
    _ objectiveType: Int32,
    _ outW: UnsafeMutablePointer<MMBuffer>?,
    _ outH: UnsafeMutablePointer<MMBuffer>?
) -> Int32 {
    guard let data = data,
          nFeatures > 0,
          nSamples > 0,
          nComponents > 0,
          nIter > 0,
          let outW = outW,
          let outH = outH else {
        return MM_ERR_INVALID_INPUT
    }

    let totalCount = Int(nFeatures) * Int(nSamples)
    let inputArray = Array(UnsafeBufferPointer(start: data, count: totalCount))
    let V = Signal(data: inputArray, shape: [Int(nFeatures), Int(nSamples)],
                   sampleRate: Int(sampleRate))

    let objective: NMF.Objective = objectiveType == 1 ? .klDivergence : .euclidean

    let result = NMF.decompose(
        V,
        nComponents: Int(nComponents),
        nIter: Int(nIter),
        objective: objective
    )

    let wStatus = fillBuffer(result.W, outW)
    guard wStatus == MM_OK else { return wStatus }
    return fillBuffer(result.H, outH)
}

// MARK: - Nearest-Neighbor Filter

@_cdecl("mm_nn_filter")
public func mm_nn_filter(
    _ ctx: UnsafeMutableRawPointer?,
    _ data: UnsafePointer<Float>?,
    _ nFeatures: Int32,
    _ nFrames: Int32,
    _ sampleRate: Int32,
    _ k: Int32,
    _ metricType: Int32,
    _ aggregateType: Int32,
    _ excludeSelf: Int32,
    _ out: UnsafeMutablePointer<MMBuffer>?
) -> Int32 {
    guard let data = data,
          nFeatures > 0,
          nFrames > 0,
          k > 0,
          let out = out else {
        return MM_ERR_INVALID_INPUT
    }

    let totalCount = Int(nFeatures) * Int(nFrames)
    let inputArray = Array(UnsafeBufferPointer(start: data, count: totalCount))
    let spectrogram = Signal(data: inputArray, shape: [Int(nFeatures), Int(nFrames)],
                             sampleRate: Int(sampleRate))

    let metric: NNFilter.DistanceMetric = metricType == 1 ? .euclidean : .cosine
    let aggregate: NNFilter.Aggregate = aggregateType == 1 ? .median : .mean

    let result = NNFilter.filter(
        spectrogram,
        k: Int(k),
        metric: metric,
        aggregate: aggregate,
        excludeSelf: excludeSelf != 0
    )

    return fillBuffer(result, out)
}

// MARK: - Recurrence Matrix

@_cdecl("mm_recurrence_matrix")
public func mm_recurrence_matrix(
    _ ctx: UnsafeMutableRawPointer?,
    _ data: UnsafePointer<Float>?,
    _ count: Int64,
    _ nFeatures: Int32,
    _ nFrames: Int32,
    _ mode: Int32,          // 0=knn, 1=threshold, 2=soft
    _ modeParam: Float,     // k (as float) for knn, threshold value for threshold, ignored for soft
    _ metricType: Int32,    // 0=euclidean, 1=cosine
    _ symmetric: Int32,     // 0=false, 1=true
    _ out: UnsafeMutablePointer<MMBuffer>?
) -> Int32 {
    guard let data = data, count > 0, let out = out else {
        return MM_ERR_INVALID_INPUT
    }

    let length = Int(count)
    let inputArray = Array(UnsafeBufferPointer(start: data, count: length))
    let signal = Signal(data: inputArray, shape: [Int(nFeatures), Int(nFrames)], sampleRate: 0)

    let recMode: Recurrence.Mode
    switch mode {
    case 1:
        recMode = .threshold(modeParam)
    case 2:
        recMode = .soft
    default:
        recMode = .knn(k: Int(modeParam))
    }

    let metric: Recurrence.Metric = metricType == 1 ? .cosine : .euclidean

    let result = Recurrence.recurrenceMatrix(
        signal,
        mode: recMode,
        metric: metric,
        symmetric: symmetric != 0
    )

    return fillBuffer(result, out)
}

@_cdecl("mm_cross_similarity")
public func mm_cross_similarity(
    _ ctx: UnsafeMutableRawPointer?,
    _ dataA: UnsafePointer<Float>?,
    _ countA: Int64,
    _ dataB: UnsafePointer<Float>?,
    _ countB: Int64,
    _ nFeatures: Int32,
    _ nFramesA: Int32,
    _ nFramesB: Int32,
    _ metricType: Int32,    // 0=euclidean, 1=cosine
    _ out: UnsafeMutablePointer<MMBuffer>?
) -> Int32 {
    guard let dataA = dataA, countA > 0,
          let dataB = dataB, countB > 0,
          let out = out else {
        return MM_ERR_INVALID_INPUT
    }

    let lengthA = Int(countA)
    let lengthB = Int(countB)
    let arrayA = Array(UnsafeBufferPointer(start: dataA, count: lengthA))
    let arrayB = Array(UnsafeBufferPointer(start: dataB, count: lengthB))
    let signalA = Signal(data: arrayA, shape: [Int(nFeatures), Int(nFramesA)], sampleRate: 0)
    let signalB = Signal(data: arrayB, shape: [Int(nFeatures), Int(nFramesB)], sampleRate: 0)

    let metric: Recurrence.Metric = metricType == 1 ? .cosine : .euclidean

    let result = Recurrence.crossSimilarity(signalA, signalB, metric: metric)

    return fillBuffer(result, out)
}

// MARK: - Dynamic Time Warping

@_cdecl("mm_dtw")
public func mm_dtw(
    _ ctx: UnsafeMutableRawPointer?,
    _ data: UnsafePointer<Float>?,
    _ count: Int64,
    _ rows: Int32,
    _ cols: Int32,
    _ stepPattern: Int32,    // 0=standard, 1=symmetric2
    _ bandWidth: Int32,      // 0=no constraint, >0=Sakoe-Chiba band width
    _ out: UnsafeMutablePointer<MMBuffer>?
) -> Int32 {
    guard let data = data, count > 0, let out = out else {
        return MM_ERR_INVALID_INPUT
    }

    let length = Int(count)
    let n = Int(rows)
    let m = Int(cols)
    guard n > 0 && m > 0 && n * m == length else {
        return MM_ERR_INVALID_INPUT
    }

    let inputArray = Array(UnsafeBufferPointer(start: data, count: length))
    let costMatrix = Signal(data: inputArray, shape: [n, m], sampleRate: 0)

    let pattern: DTW.StepPattern = stepPattern == 1 ? .symmetric2 : .standard
    let bw: Int? = bandWidth > 0 ? Int(bandWidth) : nil

    let result = DTW.compute(costMatrix: costMatrix, stepPattern: pattern, bandWidth: bw)

    return fillBuffer(result.accumulatedCost, out)
}

// MARK: - Agglomerative Segmentation

@_cdecl("mm_agglomerative")
public func mm_agglomerative(
    _ ctx: UnsafeMutableRawPointer?,
    _ data: UnsafePointer<Float>?,
    _ count: Int64,
    _ nFeatures: Int32,
    _ nFrames: Int32,
    _ nSegments: Int32,
    _ out: UnsafeMutablePointer<MMBuffer>?
) -> Int32 {
    guard let data = data, count > 0, let out = out else {
        return MM_ERR_INVALID_INPUT
    }

    let length = Int(count)
    let inputArray = Array(UnsafeBufferPointer(start: data, count: length))
    let signal = Signal(data: inputArray, shape: [Int(nFeatures), Int(nFrames)], sampleRate: 0)

    let boundaries = Clustering.agglomerative(features: signal, nSegments: Int(nSegments))

    let result = Signal(data: boundaries.map { Float($0) },
                        shape: [boundaries.count],
                        sampleRate: 0)
    return fillBuffer(result, out)
}

// MARK: - Viterbi Decoding (HMM)

@_cdecl("mm_viterbi")
public func mm_viterbi(
    _ ctx: UnsafeMutableRawPointer?,
    _ logObsData: UnsafePointer<Float>?,
    _ nFrames: Int32,
    _ nStates: Int32,
    _ logInitial: UnsafePointer<Float>?,
    _ logTransition: UnsafePointer<Float>?,
    _ out: UnsafeMutablePointer<MMBuffer>?
) -> Int32 {
    guard let logObsData = logObsData,
          nFrames > 0, nStates > 0,
          let logInitial = logInitial,
          let logTransition = logTransition,
          let out = out else {
        return MM_ERR_INVALID_INPUT
    }

    let frames = Int(nFrames)
    let states = Int(nStates)

    // Convert flat log_obs [nFrames * nStates] to [[Float]]
    let obsFlat = Array(UnsafeBufferPointer(start: logObsData, count: frames * states))
    var logObs = [[Float]]()
    logObs.reserveCapacity(frames)
    for t in 0..<frames {
        let start = t * states
        logObs.append(Array(obsFlat[start..<start + states]))
    }

    // Convert flat log_initial [nStates] to [Float]
    let logInit = Array(UnsafeBufferPointer(start: logInitial, count: states))

    // Convert flat log_transition [nStates * nStates] to [[Float]]
    let transFlat = Array(UnsafeBufferPointer(start: logTransition, count: states * states))
    var logTrans = [[Float]]()
    logTrans.reserveCapacity(states)
    for i in 0..<states {
        let start = i * states
        logTrans.append(Array(transFlat[start..<start + states]))
    }

    // Run Viterbi
    let result = HMM.viterbi(
        logObservations: logObs,
        logInitial: logInit,
        logTransition: logTrans
    )

    // Return path as float-encoded indices
    let pathFloats = result.path.map { Float($0) }
    let pathSignal = Signal(data: pathFloats, shape: [frames], sampleRate: 0)
    return fillBuffer(pathSignal, out)
}

// MARK: - Viterbi Decoding (CRF / Discriminative)

@_cdecl("mm_viterbi_discriminative")
public func mm_viterbi_discriminative(
    _ ctx: UnsafeMutableRawPointer?,
    _ unaryData: UnsafePointer<Float>?,
    _ nFrames: Int32,
    _ nStates: Int32,
    _ pairwiseData: UnsafePointer<Float>?,
    _ out: UnsafeMutablePointer<MMBuffer>?
) -> Int32 {
    guard let unaryData = unaryData,
          nFrames > 0, nStates > 0,
          let pairwiseData = pairwiseData,
          let out = out else {
        return MM_ERR_INVALID_INPUT
    }

    let frames = Int(nFrames)
    let states = Int(nStates)

    // Convert flat unary [nFrames * nStates] to [[Float]]
    let unaryFlat = Array(UnsafeBufferPointer(start: unaryData, count: frames * states))
    var unaryScores = [[Float]]()
    unaryScores.reserveCapacity(frames)
    for t in 0..<frames {
        let start = t * states
        unaryScores.append(Array(unaryFlat[start..<start + states]))
    }

    // Convert flat pairwise [nStates * nStates] to [[Float]]
    let pairFlat = Array(UnsafeBufferPointer(start: pairwiseData, count: states * states))
    var pairwiseScores = [[Float]]()
    pairwiseScores.reserveCapacity(states)
    for i in 0..<states {
        let start = i * states
        pairwiseScores.append(Array(pairFlat[start..<start + states]))
    }

    // Run CRF Viterbi
    let result = CRF.viterbiDecode(
        unaryScores: unaryScores,
        pairwiseScores: pairwiseScores
    )

    // Return path as float-encoded indices
    let pathFloats = result.path.map { Float($0) }
    let pathSignal = Signal(data: pathFloats, shape: [frames], sampleRate: 0)
    return fillBuffer(pathSignal, out)
}

// MARK: - Unit Conversions

@_cdecl("mm_hz_to_midi")
public func mm_hz_to_midi(
    _ ctx: UnsafeMutableRawPointer?,
    _ data: UnsafePointer<Float>?,
    _ count: Int64,
    _ out: UnsafeMutablePointer<MMBuffer>?
) -> Int32 {
    guard let data = data, count > 0, let out = out else {
        return MM_ERR_INVALID_INPUT
    }

    let length = Int(count)
    let inputArray = Array(UnsafeBufferPointer(start: data, count: length))
    let result = Units.hzToMidi(inputArray)
    let signal = Signal(data: result, shape: [length], sampleRate: 0)
    return fillBuffer(signal, out)
}

@_cdecl("mm_midi_to_hz")
public func mm_midi_to_hz(
    _ ctx: UnsafeMutableRawPointer?,
    _ data: UnsafePointer<Float>?,
    _ count: Int64,
    _ out: UnsafeMutablePointer<MMBuffer>?
) -> Int32 {
    guard let data = data, count > 0, let out = out else {
        return MM_ERR_INVALID_INPUT
    }

    let length = Int(count)
    let inputArray = Array(UnsafeBufferPointer(start: data, count: length))
    let result = Units.midiToHz(inputArray)
    let signal = Signal(data: result, shape: [length], sampleRate: 0)
    return fillBuffer(signal, out)
}

@_cdecl("mm_times_to_frames")
public func mm_times_to_frames(
    _ ctx: UnsafeMutableRawPointer?,
    _ data: UnsafePointer<Float>?,
    _ count: Int64,
    _ sr: Int32,
    _ hopLength: Int32,
    _ out: UnsafeMutablePointer<MMBuffer>?
) -> Int32 {
    guard let data = data, count > 0, let out = out else {
        return MM_ERR_INVALID_INPUT
    }

    let length = Int(count)
    let inputArray = Array(UnsafeBufferPointer(start: data, count: length))
    let frames = Units.timesToFrames(inputArray, sr: Int(sr), hopLength: Int(hopLength))
    let result = frames.map { Float($0) }
    let signal = Signal(data: result, shape: [length], sampleRate: 0)
    return fillBuffer(signal, out)
}

@_cdecl("mm_frames_to_time")
public func mm_frames_to_time(
    _ ctx: UnsafeMutableRawPointer?,
    _ data: UnsafePointer<Float>?,
    _ count: Int64,
    _ sr: Int32,
    _ hopLength: Int32,
    _ out: UnsafeMutablePointer<MMBuffer>?
) -> Int32 {
    guard let data = data, count > 0, let out = out else {
        return MM_ERR_INVALID_INPUT
    }

    let length = Int(count)
    let inputArray = Array(UnsafeBufferPointer(start: data, count: length))
    let frames = inputArray.map { Int($0) }
    let result = Units.framesToTime(frames, sr: Int(sr), hopLength: Int(hopLength))
    let signal = Signal(data: result, shape: [length], sampleRate: 0)
    return fillBuffer(signal, out)
}

@_cdecl("mm_fft_frequencies")
public func mm_fft_frequencies(
    _ ctx: UnsafeMutableRawPointer?,
    _ sr: Int32,
    _ nFFT: Int32,
    _ out: UnsafeMutablePointer<MMBuffer>?
) -> Int32 {
    guard sr > 0, nFFT > 0, let out = out else {
        return MM_ERR_INVALID_INPUT
    }

    let result = Units.fftFrequencies(sr: Int(sr), nFFT: Int(nFFT))
    let signal = Signal(data: result, shape: [result.count], sampleRate: 0)
    return fillBuffer(signal, out)
}

@_cdecl("mm_mel_frequencies")
public func mm_mel_frequencies(
    _ ctx: UnsafeMutableRawPointer?,
    _ nMels: Int32,
    _ fMin: Float,
    _ fMax: Float,
    _ out: UnsafeMutablePointer<MMBuffer>?
) -> Int32 {
    guard nMels > 0, let out = out else {
        return MM_ERR_INVALID_INPUT
    }

    let result = Units.melFrequencies(nMels: Int(nMels), fMin: fMin, fMax: fMax)
    let signal = Signal(data: result, shape: [result.count], sampleRate: 0)
    return fillBuffer(signal, out)
}

// MARK: - Semitone Bandpass Filterbank

@_cdecl("mm_semitone_filterbank")
public func mm_semitone_filterbank(
    _ ctx: UnsafeMutableRawPointer?,
    _ signalData: UnsafePointer<Float>?,
    _ signalLength: Int64,
    _ sampleRate: Int32,
    _ midiLow: Int32,
    _ midiHigh: Int32,
    _ order: Int32,
    _ out: UnsafeMutablePointer<MMBuffer>?
) -> Int32 {
    guard let signalData = signalData,
          signalLength > 0,
          sampleRate > 0,
          midiLow >= 0,
          midiHigh >= midiLow,
          order > 0,
          let out = out else {
        return MM_ERR_INVALID_INPUT
    }

    let length = Int(signalLength)
    let inputArray = Array(UnsafeBufferPointer(start: signalData, count: length))
    let signal = Signal(data: inputArray, sampleRate: Int(sampleRate))

    let result = SemitoneBandpass.filterbank(
        signal: signal,
        sr: Int(sampleRate),
        midiLow: Int(midiLow),
        midiHigh: Int(midiHigh),
        order: Int(order)
    )

    return fillBuffer(result, out)
}

@_cdecl("mm_semitone_frequencies")
public func mm_semitone_frequencies(
    _ ctx: UnsafeMutableRawPointer?,
    _ midiLow: Int32,
    _ midiHigh: Int32,
    _ out: UnsafeMutablePointer<MMBuffer>?
) -> Int32 {
    guard midiLow >= 0,
          midiHigh >= midiLow,
          let out = out else {
        return MM_ERR_INVALID_INPUT
    }

    let freqs = SemitoneBandpass.semitoneFrequencies(
        midiLow: Int(midiLow),
        midiHigh: Int(midiHigh)
    )
    let signal = Signal(data: freqs, shape: [freqs.count], sampleRate: 0)
    return fillBuffer(signal, out)
}

// MARK: - Memory

@_cdecl("mm_buffer_free")
public func mm_buffer_free(_ buf: UnsafeMutablePointer<MMBuffer>?) {
    guard let buf = buf else { return }
    if let data = buf.pointee.data {
        data.deallocate()
    }
    buf.pointee.data = nil
    buf.pointee.ndim = 0
    buf.pointee.dtype = 0
    buf.pointee.count = 0
    withUnsafeMutablePointer(to: &buf.pointee.shape) { tuplePtr in
        tuplePtr.withMemoryRebound(to: Int64.self, capacity: 8) { shapePtr in
            for i in 0..<8 {
                shapePtr[i] = 0
            }
        }
    }
}
