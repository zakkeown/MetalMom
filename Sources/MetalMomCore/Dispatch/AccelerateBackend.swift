import Foundation

/// Marker for the Accelerate (CPU) backend. Holds no state —
/// Accelerate functions are all stateless and thread-safe.
public final class AccelerateBackend: Sendable {
    public static let shared = AccelerateBackend()
    private init() {}
}
