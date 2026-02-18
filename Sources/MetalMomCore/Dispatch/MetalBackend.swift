import Foundation
import Metal

/// Metal GPU backend.  Holds the device, command queue, and shader library.
/// The singleton is `nil` when Metal is unavailable (e.g. Linux CI).
public final class MetalBackend {

    /// Shared singleton — `nil` if Metal is unavailable.
    public static let shared: MetalBackend? = {
        guard let device = MTLCreateSystemDefaultDevice() else { return nil }
        return MetalBackend(device: device)
    }()

    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue
    public let chipProfile: ChipProfile

    // Shader library, loaded on first access.
    private var _defaultLibrary: MTLLibrary?

    /// The default Metal shader library for this package.
    /// Returns `nil` until .metal shader files are added (Task 10.3+).
    public var defaultLibrary: MTLLibrary? {
        if _defaultLibrary == nil {
            _defaultLibrary = device.makeDefaultLibrary()
        }
        return _defaultLibrary
    }

    private init?(device: MTLDevice) {
        self.device = device
        guard let queue = device.makeCommandQueue() else { return nil }
        self.commandQueue = queue
        self.chipProfile = ChipProfile(device: device)
    }

    /// Shared MetalShaders instance — compiled on first access.
    /// Returns `nil` if shader compilation fails.
    public lazy var shaders: MetalShaders? = MetalShaders(device: device)

    /// Create a command buffer for a batch of GPU operations.
    public func makeCommandBuffer() -> MTLCommandBuffer? {
        commandQueue.makeCommandBuffer()
    }
}
