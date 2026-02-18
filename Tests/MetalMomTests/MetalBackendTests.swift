import XCTest
@testable import MetalMomCore

final class MetalBackendTests: XCTestCase {

    // MARK: - MetalBackend

    func testMetalBackendInitialization() throws {
        guard let backend = MetalBackend.shared else {
            throw XCTSkip("No Metal device available (CI runner?)")
        }
        XCTAssertNotNil(backend.device)
        XCTAssertNotNil(backend.commandQueue)
    }

    func testMetalDeviceName() throws {
        guard let backend = MetalBackend.shared else {
            throw XCTSkip("No Metal device available")
        }
        XCTAssertFalse(backend.device.name.isEmpty)
    }

    func testCommandBufferCreation() throws {
        guard let backend = MetalBackend.shared else {
            throw XCTSkip("No Metal device available")
        }
        let cmdBuf = backend.makeCommandBuffer()
        XCTAssertNotNil(cmdBuf)
    }

    func testDefaultLibraryIsNilWithoutShaders() throws {
        guard let backend = MetalBackend.shared else {
            throw XCTSkip("No Metal device available")
        }
        // Until .metal shader files are added (Task 10.3+), the library
        // may be nil.  This test documents current expected state.
        // It is not a failure if a library happens to exist.
        _ = backend.defaultLibrary  // just ensure no crash
    }

    // MARK: - ChipProfile

    func testChipProfileDetection() throws {
        guard let backend = MetalBackend.shared else {
            throw XCTSkip("No Metal device available")
        }
        let profile = backend.chipProfile
        XCTAssertGreaterThan(profile.estimatedCoreCount, 0)
        XCTAssertGreaterThan(profile.maxBufferLength, 0)
        XCTAssertNotEqual(profile.gpuFamily, .unknown)
    }

    func testChipProfileThresholdsArePositive() throws {
        guard let backend = MetalBackend.shared else {
            throw XCTSkip("No Metal device available")
        }
        let profile = backend.chipProfile
        XCTAssertGreaterThan(profile.threshold(for: .stft), 0)
        XCTAssertGreaterThan(profile.threshold(for: .matmul), 0)
        XCTAssertGreaterThan(profile.threshold(for: .elementwise), 0)
        XCTAssertGreaterThan(profile.threshold(for: .reduction), 0)
        XCTAssertGreaterThan(profile.threshold(for: .convolution), 0)
    }

    func testHasNonUniformThreadgroups() throws {
        guard let backend = MetalBackend.shared else {
            throw XCTSkip("No Metal device available")
        }
        // All Apple Silicon (M1+) supports non-uniform threadgroups
        XCTAssertTrue(backend.chipProfile.hasNonUniformThreadgroups)
    }

    func testGPUFamilyComparable() {
        // Verify ordering: apple7 < apple8 < apple9
        XCTAssertLessThan(ChipProfile.GPUFamily.apple7, .apple8)
        XCTAssertLessThan(ChipProfile.GPUFamily.apple8, .apple9)
    }

    // MARK: - SmartDispatcher integration

    func testSmartDispatcherWithMetal() {
        let dispatcher = SmartDispatcher()
        if MetalBackend.shared != nil {
            XCTAssertEqual(dispatcher.activeBackend, .metal)
            XCTAssertNotNil(dispatcher.metalBackend)
        } else {
            XCTAssertEqual(dispatcher.activeBackend, .accelerate)
            XCTAssertNil(dispatcher.metalBackend)
        }
    }

    func testRoutingDecisionCPUForSmallData() {
        let dispatcher = SmartDispatcher()
        // Small data should always route to CPU
        let decision = dispatcher.routingDecision(dataSize: 100, operationThreshold: 10000)
        XCTAssertEqual(decision, .accelerate)
    }

    func testRoutingDecisionGPUForLargeData() throws {
        let dispatcher = SmartDispatcher()
        guard dispatcher.activeBackend == .metal else {
            throw XCTSkip("Metal not available")
        }
        // Large data should route to GPU when Metal is active
        let decision = dispatcher.routingDecision(dataSize: 100_000, operationThreshold: 1000)
        XCTAssertEqual(decision, .metal)
    }

    func testRoutingDecisionAtExactThreshold() throws {
        let dispatcher = SmartDispatcher()
        guard dispatcher.activeBackend == .metal else {
            throw XCTSkip("Metal not available")
        }
        // At exact threshold, should route to GPU (>=)
        let decision = dispatcher.routingDecision(dataSize: 5000, operationThreshold: 5000)
        XCTAssertEqual(decision, .metal)
    }
}
