import Foundation

/// Routes compute operations to the optimal backend based on data size.
/// Prefers Metal when available; falls back to Accelerate (CPU).
public final class SmartDispatcher {
    public let activeBackend: BackendType
    public let metalBackend: MetalBackend?

    public init() {
        if let metal = MetalBackend.shared {
            self.metalBackend = metal
            self.activeBackend = .metal
        } else {
            self.metalBackend = nil
            self.activeBackend = .accelerate
        }
    }

    /// Determine which backend to use for a given data size and operation threshold.
    public func routingDecision(dataSize: Int, operationThreshold: Int) -> BackendType {
        guard activeBackend == .metal else { return .accelerate }
        return dataSize >= operationThreshold ? .metal : .accelerate
    }

    /// Execute a compute operation, routing to the appropriate backend.
    public func dispatch<Op: ComputeOperation>(_ op: Op, input: Op.Input, dataSize: Int) -> Op.Output {
        let decision = routingDecision(dataSize: dataSize, operationThreshold: Op.dispatchThreshold)
        switch decision {
        case .accelerate:
            return op.executeCPU(input)
        case .metal:
            return op.executeGPU(input)
        }
    }
}
