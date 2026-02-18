import XCTest
@testable import MetalMomCore

final class MetalMomTests: XCTestCase {
    func testVersion() {
        XCTAssertEqual(MetalMom.version, "0.1.0")
    }
}
