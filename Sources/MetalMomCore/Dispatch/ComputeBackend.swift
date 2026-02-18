import Foundation

/// Identifies which compute backend to use.
public enum BackendType: Equatable, Sendable {
    case accelerate
    case metal
}

/// Protocol for compute operations that support both CPU and GPU paths.
public protocol ComputeOperation {
    associatedtype Input
    associatedtype Output

    /// Data size threshold above which GPU is preferred. Int.max = always CPU.
    static var dispatchThreshold: Int { get }

    func executeCPU(_ input: Input) -> Output

    /// GPU path. During Phases 1-9 this calls fatalError().
    /// Phase 10 fills in real implementations.
    func executeGPU(_ input: Input) -> Output
}

/// Default GPU implementation — fatalError until Phase 10.
extension ComputeOperation {
    public func executeGPU(_ input: Input) -> Output {
        fatalError("\(type(of: self)).executeGPU not yet implemented — Phase 10")
    }
}
