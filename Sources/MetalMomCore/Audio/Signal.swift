import Foundation

/// Core data type wrapping audio data with shape metadata.
/// Uses manually-allocated UnsafeMutableBufferPointer for stable pointer addresses
/// safe to pass to Accelerate, Metal, and across the C ABI.
public final class Signal {
    private let storage: UnsafeMutableBufferPointer<Float>
    public let shape: [Int]
    public let sampleRate: Int

    public var count: Int { storage.count }

    /// Create a 1D signal from a Float array (copies into pinned storage).
    public init(data: [Float], sampleRate: Int) {
        let ptr = UnsafeMutablePointer<Float>.allocate(capacity: data.count)
        ptr.initialize(from: data, count: data.count)
        self.storage = UnsafeMutableBufferPointer(start: ptr, count: data.count)
        self.shape = [data.count]
        self.sampleRate = sampleRate
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
    }

    /// Create from pre-allocated buffer (takes ownership, caller must not free).
    public init(taking buffer: UnsafeMutableBufferPointer<Float>, shape: [Int], sampleRate: Int) {
        precondition(buffer.count == shape.reduce(1, *))
        self.storage = buffer
        self.shape = shape
        self.sampleRate = sampleRate
    }

    deinit {
        storage.baseAddress?.deinitialize(count: storage.count)
        storage.baseAddress?.deallocate()
    }

    public subscript(index: Int) -> Float {
        get { storage[index] }
        set { storage[index] = newValue }
    }

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
