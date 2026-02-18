import XCTest
@testable import MetalMomCore

final class DispatchTests: XCTestCase {
    func testDispatcherSelectsAvailableBackend() {
        let dispatcher = SmartDispatcher()
        if MetalBackend.shared != nil {
            // On Apple Silicon, Metal is available
            XCTAssertEqual(dispatcher.activeBackend, .metal)
            XCTAssertNotNil(dispatcher.metalBackend)
        } else {
            // On non-Metal platforms (Linux CI)
            XCTAssertEqual(dispatcher.activeBackend, .accelerate)
            XCTAssertNil(dispatcher.metalBackend)
        }
    }

    func testDispatcherRoutesSmallDataToCPU() {
        let dispatcher = SmartDispatcher()
        // With threshold = Int.max, all work goes to CPU regardless of backend
        let decision = dispatcher.routingDecision(dataSize: 1000, operationThreshold: Int.max)
        XCTAssertEqual(decision, .accelerate)
    }
}
