import Accelerate
import Foundation

/// Distinguishes real-valued from complex-valued signal data.
public enum SignalDType: Equatable {
    case float32      // Real-valued: N floats
    case complex64    // Complex-valued: 2N floats (interleaved real, imag, real, imag, ...)
}

/// Core data type wrapping audio data with shape metadata.
/// Uses manually-allocated UnsafeMutableBufferPointer for stable pointer addresses
/// safe to pass to Accelerate, Metal, and across the C ABI.
public final class Signal {
    private let storage: UnsafeMutableBufferPointer<Float>
    public let shape: [Int]
    public let sampleRate: Int
    public let dtype: SignalDType

    /// Total number of raw floats in storage.
    /// For complex signals this is 2x the number of complex elements.
    public var count: Int { storage.count }

    /// Number of logical elements.
    /// For `.float32`, same as `count`. For `.complex64`, half of `count` (each element is a real/imag pair).
    public var elementCount: Int {
        switch dtype {
        case .float32:
            return storage.count
        case .complex64:
            return storage.count / 2
        }
    }

    // MARK: - Real-valued initializers

    /// Create a 1D signal from a Float array (copies into pinned storage).
    public init(data: [Float], sampleRate: Int) {
        let ptr = UnsafeMutablePointer<Float>.allocate(capacity: data.count)
        ptr.initialize(from: data, count: data.count)
        self.storage = UnsafeMutableBufferPointer(start: ptr, count: data.count)
        self.shape = [data.count]
        self.sampleRate = sampleRate
        self.dtype = .float32
    }

    /// Create an N-dimensional signal from a flat Float array with explicit shape.
    public init(data: [Float], shape: [Int], sampleRate: Int) {
        precondition(data.count == shape.reduce(1, *),
                     "Data count \(data.count) doesn't match shape \(shape)")
        let ptr = UnsafeMutablePointer<Float>.allocate(capacity: data.count)
        ptr.initialize(from: data, count: data.count)
        self.storage = UnsafeMutableBufferPointer(start: ptr, count: data.count)
        self.shape = shape
        self.sampleRate = sampleRate
        self.dtype = .float32
    }

    /// Create from pre-allocated buffer (takes ownership, caller must not free).
    public init(taking buffer: UnsafeMutableBufferPointer<Float>, shape: [Int], sampleRate: Int,
                dtype: SignalDType = .float32) {
        switch dtype {
        case .float32:
            precondition(buffer.count == shape.reduce(1, *),
                         "Buffer count \(buffer.count) doesn't match shape \(shape)")
        case .complex64:
            precondition(buffer.count == shape.reduce(1, *) * 2,
                         "Complex buffer count \(buffer.count) doesn't match shape \(shape) * 2")
        }
        self.storage = buffer
        self.shape = shape
        self.sampleRate = sampleRate
        self.dtype = dtype
    }

    // MARK: - Complex-valued initializers

    /// Create a complex-valued signal from interleaved real/imag float data.
    /// `shape` describes the complex element dimensions (not the raw float count).
    /// For shape [3], `complexData` must contain 6 floats: [r0, i0, r1, i1, r2, i2].
    public init(complexData: [Float], shape: [Int], sampleRate: Int) {
        let elementCount = shape.reduce(1, *)
        precondition(complexData.count == elementCount * 2,
                     "Complex data count \(complexData.count) doesn't match shape \(shape) * 2 = \(elementCount * 2)")
        let ptr = UnsafeMutablePointer<Float>.allocate(capacity: complexData.count)
        ptr.initialize(from: complexData, count: complexData.count)
        self.storage = UnsafeMutableBufferPointer(start: ptr, count: complexData.count)
        self.shape = shape
        self.sampleRate = sampleRate
        self.dtype = .complex64
    }

    deinit {
        storage.baseAddress?.deinitialize(count: storage.count)
        storage.baseAddress?.deallocate()
    }

    // MARK: - Subscript access

    public subscript(index: Int) -> Float {
        get { storage[index] }
        set { storage[index] = newValue }
    }

    // MARK: - Complex element access

    /// Returns the real part of the complex element at the given index.
    /// Precondition: signal must be `.complex64` and index must be in bounds.
    public func realPart(at index: Int) -> Float {
        precondition(dtype == .complex64, "realPart(at:) requires complex64 dtype")
        precondition(index >= 0 && index < elementCount, "Index \(index) out of bounds for \(elementCount) complex elements")
        return storage[index * 2]
    }

    /// Returns the imaginary part of the complex element at the given index.
    /// Precondition: signal must be `.complex64` and index must be in bounds.
    public func imagPart(at index: Int) -> Float {
        precondition(dtype == .complex64, "imagPart(at:) requires complex64 dtype")
        precondition(index >= 0 && index < elementCount, "Index \(index) out of bounds for \(elementCount) complex elements")
        return storage[index * 2 + 1]
    }

    /// Provides a DSPSplitComplex view of interleaved complex data for Accelerate operations.
    /// De-interleaves data into temporary real and imaginary arrays, calls the body closure,
    /// then deallocates the temporary storage.
    /// The `Int` parameter is the number of complex elements.
    public func withSplitComplex<R>(_ body: (DSPSplitComplex, Int) throws -> R) rethrows -> R {
        precondition(dtype == .complex64, "withSplitComplex requires complex64 dtype")
        let n = elementCount
        let realp = UnsafeMutablePointer<Float>.allocate(capacity: n)
        let imagp = UnsafeMutablePointer<Float>.allocate(capacity: n)
        defer {
            realp.deinitialize(count: n)
            realp.deallocate()
            imagp.deinitialize(count: n)
            imagp.deallocate()
        }

        // De-interleave using vDSP_ctoz: converts interleaved complex to split complex
        var split = DSPSplitComplex(realp: realp, imagp: imagp)
        storage.baseAddress!.withMemoryRebound(to: DSPComplex.self, capacity: n) { complexPtr in
            vDSP_ctoz(complexPtr, 2, &split, 1, vDSP_Length(n))
        }

        return try body(split, n)
    }

    // MARK: - Pointer access

    /// Stable pointer to underlying data. Safe because storage is manually allocated.
    public var dataPointer: UnsafePointer<Float> {
        UnsafePointer(storage.baseAddress!)
    }

    /// Mutable pointer to underlying data.
    public var mutableDataPointer: UnsafeMutablePointer<Float> {
        storage.baseAddress!
    }

    /// Access underlying storage for Accelerate operations.
    public func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Float>) throws -> R) rethrows -> R {
        try body(UnsafeBufferPointer(storage))
    }

    public func withUnsafeMutableBufferPointer<R>(_ body: (UnsafeMutableBufferPointer<Float>) throws -> R) rethrows -> R {
        try body(storage)
    }
}
