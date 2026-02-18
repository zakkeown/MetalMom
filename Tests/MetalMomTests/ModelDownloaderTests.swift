import XCTest
@testable import MetalMomCore

final class ModelDownloaderTests: XCTestCase {

    func testModelFamilyRawValues() {
        // Verify all model families have valid raw values for URL construction
        for family in ModelFamily.allCases {
            XCTAssertFalse(family.rawValue.isEmpty,
                "Model family \(family) should have a non-empty raw value")
        }
    }

    func testModelFamilyCount() {
        // Ensure we have all 9 model families
        XCTAssertEqual(ModelFamily.allCases.count, 9)
    }

    func testDefaultCacheDirectory() {
        let downloader = ModelDownloader.shared
        let cacheDir = downloader.cacheDirectory
        XCTAssertTrue(cacheDir.path.contains("MetalMom"),
            "Cache directory should be in MetalMom namespace")
        XCTAssertTrue(cacheDir.path.contains("Models"),
            "Cache directory should contain Models subdirectory")
    }

    func testIsCachedReturnsFalseForMissingModel() {
        let downloader = ModelDownloader.shared
        XCTAssertFalse(downloader.isCached(.rnnBeatProcessor),
            "Should return false for model not yet downloaded")
    }

    func testClearCacheDoesNotThrow() {
        let downloader = ModelDownloader.shared
        XCTAssertNoThrow(try downloader.clearCache())
    }

    func testRepositoryURL() {
        let downloader = ModelDownloader.shared
        XCTAssertTrue(downloader.repositoryURL.absoluteString.contains("huggingface.co"),
            "Repository URL should point to Hugging Face")
    }
}
