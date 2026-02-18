import XCTest
@testable import MetalMomCore

final class DispatchTests: XCTestCase {
    func testDispatcherDefaultsToAccelerate() {
        let dispatcher = SmartDispatcher()
        XCTAssertEqual(dispatcher.activeBackend, .accelerate)
    }

    func testDispatcherRoutesSmallDataToCPU() {
        let dispatcher = SmartDispatcher()
        // With threshold = Int.max (GPU not available), all work goes to CPU
        let decision = dispatcher.routingDecision(dataSize: 1000, operationThreshold: Int.max)
        XCTAssertEqual(decision, .accelerate)
    }
}
