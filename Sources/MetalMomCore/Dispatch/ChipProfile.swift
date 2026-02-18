import Foundation
import Metal

/// GPU chip profiling — detects capabilities and provides dispatch thresholds.
public struct ChipProfile: Sendable {
    public let gpuFamily: GPUFamily
    public let estimatedCoreCount: Int
    public let maxBufferLength: Int
    public let hasNonUniformThreadgroups: Bool

    /// Mapped GPU families based on Metal's MTLGPUFamily API.
    /// Each case corresponds to the *highest* MTLGPUFamily the device reports.
    public enum GPUFamily: Comparable, Sendable {
        case apple7   // M1 family
        case apple8   // M2 family
        case apple9   // M3 / M4 family (current SDK maps both here)
        case unknown
    }

    public init(device: MTLDevice) {
        // Detect GPU family — probe from highest to lowest.
        // As of Xcode 16 / macOS 15, M4 still reports apple9 as highest.
        if device.supportsFamily(.apple9) {
            self.gpuFamily = .apple9
        } else if device.supportsFamily(.apple8) {
            self.gpuFamily = .apple8
        } else if device.supportsFamily(.apple7) {
            self.gpuFamily = .apple7
        } else {
            self.gpuFamily = .unknown
        }

        // Estimate GPU core count from family.
        // Exact core count is not exposed by the Metal API.
        // These are base-chip counts for the standard (non-Pro/Max/Ultra) die.
        switch self.gpuFamily {
        case .apple7:  self.estimatedCoreCount = 8   // M1 base
        case .apple8:  self.estimatedCoreCount = 10  // M2 base
        case .apple9:  self.estimatedCoreCount = 10  // M3 / M4 base
        case .unknown: self.estimatedCoreCount = 8
        }

        self.maxBufferLength = device.maxBufferLength
        self.hasNonUniformThreadgroups = device.supportsFamily(.apple4)
    }

    /// Minimum data size (in elements) to prefer GPU over CPU for a given
    /// operation type.  Lower values mean we hand off to Metal sooner.
    public func threshold(for operation: OperationType) -> Int {
        switch operation {
        case .stft:
            return gpuFamily >= .apple9 ? 8192 : 16384
        case .matmul:
            return gpuFamily >= .apple9 ? 4096 : 8192
        case .elementwise:
            // Elementwise: GPU only for very large arrays (data-transfer overhead)
            return gpuFamily >= .apple9 ? 65536 : 131072
        case .reduction:
            return gpuFamily >= .apple9 ? 32768 : 65536
        case .convolution:
            return gpuFamily >= .apple9 ? 8192 : 16384
        }
    }

    public enum OperationType: Sendable {
        case stft
        case matmul
        case elementwise
        case reduction
        case convolution
    }
}
