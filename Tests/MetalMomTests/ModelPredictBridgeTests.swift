import XCTest
@testable import MetalMomBridge
import MetalMomCBridge

final class ModelPredictBridgeTests: XCTestCase {

    /// Path to the test_identity.mlmodel fixture.
    private var mlmodelPath: String {
        let thisFile = URL(fileURLWithPath: #filePath)
        let testsDir = thisFile
            .deletingLastPathComponent()  // MetalMomTests/
            .deletingLastPathComponent()  // Tests/
        return testsDir.appendingPathComponent("fixtures/test_identity.mlmodel").path
    }

    /// Path to the test_identity.mlmodelc fixture.
    private var mlmodelcPath: String {
        let thisFile = URL(fileURLWithPath: #filePath)
        let testsDir = thisFile
            .deletingLastPathComponent()  // MetalMomTests/
            .deletingLastPathComponent()  // Tests/
        return testsDir.appendingPathComponent("fixtures/test_identity.mlmodelc").path
    }

    // MARK: - Identity Model Tests

    func testIdentityModelMLModel() throws {
        let ctx = MetalMomBridge.mm_init()!
        defer { MetalMomBridge.mm_destroy(ctx) }

        // test_identity.mlmodel expects input shape [1, 4]
        let inputData: [Float] = [1.0, 2.0, 3.0, 4.0]
        var inputShape: [Int32] = [1, 4]
        var out = MMBuffer()

        let status = mlmodelPath.withCString { pathPtr in
            MetalMomBridge.mm_model_predict(
                ctx,
                pathPtr,
                inputData,
                &inputShape,
                Int32(inputShape.count),
                Int32(inputData.count),
                &out
            )
        }

        XCTAssertEqual(status, MM_OK, "mm_model_predict should succeed with .mlmodel")
        XCTAssertEqual(out.count, 4, "Output should have 4 elements")

        // Verify output matches input (identity model)
        let outPtr = out.data!
        for i in 0..<4 {
            XCTAssertEqual(outPtr[i], inputData[i], accuracy: 1e-5,
                           "Output[\(i)] should match input[\(i)]")
        }

        MetalMomBridge.mm_buffer_free(&out)
    }

    func testIdentityModelMLModelC() throws {
        let ctx = MetalMomBridge.mm_init()!
        defer { MetalMomBridge.mm_destroy(ctx) }

        // test_identity.mlmodelc expects input shape [1, 4]
        let inputData: [Float] = [1.0, 2.0, 3.0, 4.0]
        var inputShape: [Int32] = [1, 4]
        var out = MMBuffer()

        let status = mlmodelcPath.withCString { pathPtr in
            MetalMomBridge.mm_model_predict(
                ctx,
                pathPtr,
                inputData,
                &inputShape,
                Int32(inputShape.count),
                Int32(inputData.count),
                &out
            )
        }

        XCTAssertEqual(status, MM_OK, "mm_model_predict should succeed with .mlmodelc")
        XCTAssertEqual(out.count, 4, "Output should have 4 elements")

        let outPtr = out.data!
        for i in 0..<4 {
            XCTAssertEqual(outPtr[i], inputData[i], accuracy: 1e-5,
                           "Output[\(i)] should match input[\(i)]")
        }

        MetalMomBridge.mm_buffer_free(&out)
    }

    // MARK: - Error Handling Tests

    func testNilInputDataReturnsError() {
        let ctx = MetalMomBridge.mm_init()!
        defer { MetalMomBridge.mm_destroy(ctx) }

        var inputShape: [Int32] = [1, 1, 3, 1, 1]
        var out = MMBuffer()

        let status = mlmodelPath.withCString { pathPtr in
            MetalMomBridge.mm_model_predict(
                ctx,
                pathPtr,
                nil,
                &inputShape,
                Int32(inputShape.count),
                3,
                &out
            )
        }

        XCTAssertEqual(status, MM_ERR_INVALID_INPUT)
    }

    func testNilModelPathReturnsError() {
        let ctx = MetalMomBridge.mm_init()!
        defer { MetalMomBridge.mm_destroy(ctx) }

        let inputData: [Float] = [1.0, 2.0, 3.0]
        var inputShape: [Int32] = [1, 1, 3, 1, 1]
        var out = MMBuffer()

        let status = MetalMomBridge.mm_model_predict(
            ctx,
            nil,
            inputData,
            &inputShape,
            Int32(inputShape.count),
            Int32(inputData.count),
            &out
        )

        XCTAssertEqual(status, MM_ERR_INVALID_INPUT)
    }

    func testInvalidModelPathReturnsError() {
        let ctx = MetalMomBridge.mm_init()!
        defer { MetalMomBridge.mm_destroy(ctx) }

        let inputData: [Float] = [1.0, 2.0, 3.0]
        var inputShape: [Int32] = [1, 1, 3, 1, 1]
        var out = MMBuffer()

        let status = "/nonexistent/path.mlmodel".withCString { pathPtr in
            MetalMomBridge.mm_model_predict(
                ctx,
                pathPtr,
                inputData,
                &inputShape,
                Int32(inputShape.count),
                Int32(inputData.count),
                &out
            )
        }

        XCTAssertEqual(status, MM_ERR_INTERNAL,
                       "Invalid model path should return MM_ERR_INTERNAL")
    }

    func testZeroInputCountReturnsError() {
        let ctx = MetalMomBridge.mm_init()!
        defer { MetalMomBridge.mm_destroy(ctx) }

        let inputData: [Float] = [1.0, 2.0, 3.0]
        var inputShape: [Int32] = [1, 1, 3, 1, 1]
        var out = MMBuffer()

        let status = mlmodelPath.withCString { pathPtr in
            MetalMomBridge.mm_model_predict(
                ctx,
                pathPtr,
                inputData,
                &inputShape,
                Int32(inputShape.count),
                0,  // zero count
                &out
            )
        }

        XCTAssertEqual(status, MM_ERR_INVALID_INPUT)
    }
}
