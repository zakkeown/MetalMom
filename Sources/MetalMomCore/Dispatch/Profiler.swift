import Foundation
import os

/// Permanent instrumentation infrastructure using OSSignposter.
/// Signposts have near-zero overhead when Instruments is not attached.
public final class Profiler: @unchecked Sendable {
    public static let shared = Profiler()

    private let signposter: OSSignposter

    /// Categories for grouping signpost activity (for future per-category signposters).
    public enum Category {
        case bridge
        case engine
        case gpu
    }

    private init() {
        self.signposter = OSSignposter(
            subsystem: "com.metalmom.engine",
            category: .pointsOfInterest
        )
    }

    /// Begin an interval and return the state token needed to end it.
    @inline(__always)
    public func begin(_ name: StaticString) -> OSSignpostIntervalState {
        signposter.beginInterval(name)
    }

    /// End a previously-started interval.
    @inline(__always)
    public func end(_ name: StaticString, _ state: OSSignpostIntervalState) {
        signposter.endInterval(name, state)
    }

    /// Emit a single point event (no duration).
    @inline(__always)
    public func event(_ name: StaticString) {
        signposter.emitEvent(name)
    }
}
