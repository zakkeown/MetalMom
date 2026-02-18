import XCTest
import Metal

final class MetalSmokeTests: XCTestCase {
    func testMetalDeviceAvailable() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("No Metal device available (CI runner?)")
        }
        XCTAssertFalse(device.name.isEmpty)
        print("Metal device: \(device.name)")

        let queue = device.makeCommandQueue()
        XCTAssertNotNil(queue, "Failed to create command queue")
    }

    func testGPUFamilyDetection() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("No Metal device available")
        }
        // All Apple Silicon supports at least family apple3
        let supportsApple3 = device.supportsFamily(.apple3)
        XCTAssertTrue(supportsApple3, "Expected Apple GPU family 3+")
        print("GPU core count: estimated via family support")
    }
}
