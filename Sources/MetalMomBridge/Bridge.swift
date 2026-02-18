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
