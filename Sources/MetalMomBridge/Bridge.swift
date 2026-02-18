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
