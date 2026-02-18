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

// MARK: - Lifecycle

@_cdecl("mm_init")
public func mm_init() -> UnsafeMutableRawPointer? {
    let ctx = MMContextInternal()
    return Unmanaged.passRetained(ctx).toOpaque()
}

@_cdecl("mm_destroy")
public func mm_destroy(_ ctx: UnsafeMutableRawPointer?) {
    guard let ctx = ctx else { return }
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

    // Copy input data into a Signal
    let length = Int(signalLength)
    let inputArray = Array(UnsafeBufferPointer(start: signalData, count: length))
    let signal = Signal(data: inputArray, sampleRate: Int(sampleRate))

    // Build STFT input
    let nFFT = Int(p.n_fft)
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
