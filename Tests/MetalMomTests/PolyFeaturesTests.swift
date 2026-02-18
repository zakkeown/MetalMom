import XCTest
@testable import MetalMomCore

final class PolyFeaturesTests: XCTestCase {

    func testPolyFeaturesShapeOrder1() {
        let data = Signal(data: [Float](repeating: 1, count: 40), shape: [4, 10], sampleRate: 22050)
        let result = PolyFeatures.compute(data: data, order: 1)
        XCTAssertEqual(result.shape, [2, 10])
    }

    func testPolyFeaturesShapeOrder2() {
        let data = Signal(data: [Float](repeating: 1, count: 40), shape: [4, 10], sampleRate: 22050)
        let result = PolyFeatures.compute(data: data, order: 2)
        XCTAssertEqual(result.shape, [3, 10])
    }

    func testPolyFeaturesConstant() {
        // Constant column -> slope should be ~0, intercept should be the constant
        // Use sr=nFFT so freq[i] = i (freqScale=1)
        var vals = [Float](repeating: 0, count: 20)
        for t in 0..<5 {
            for f in 0..<4 {
                vals[f * 5 + t] = 3.0
            }
        }
        let data = Signal(data: vals, shape: [4, 5], sampleRate: 22050)
        let result = PolyFeatures.compute(data: data, order: 1, sr: 4, nFFT: 4)
        // Slope (coeff 0, highest power) should be ~0
        for t in 0..<5 {
            XCTAssertEqual(result[t], 0, accuracy: 1e-4, "Slope should be ~0 for constant")
        }
        // Intercept (coeff 1) should be ~3.0
        for t in 0..<5 {
            XCTAssertEqual(result[5 + t], 3.0, accuracy: 1e-4, "Intercept should be ~3.0")
        }
    }

    func testPolyFeaturesLinear() {
        // With sr=nFFT=1, freq[i] = i, so y=x gives polyfit slope=1, intercept=0
        var vals = [Float](repeating: 0, count: 20)
        for t in 0..<5 {
            for f in 0..<4 {
                vals[f * 5 + t] = Float(f)
            }
        }
        let data = Signal(data: vals, shape: [4, 5], sampleRate: 22050)
        // Use sr=nFFT so freqScale=1, freq[i]=i
        let result = PolyFeatures.compute(data: data, order: 1, sr: 4, nFFT: 4)
        // Slope should be 1.0
        for t in 0..<5 {
            XCTAssertEqual(result[t], 1.0, accuracy: 1e-4, "Slope should be 1.0 for y=x")
        }
        // Intercept should be 0.0
        for t in 0..<5 {
            XCTAssertEqual(result[5 + t], 0.0, accuracy: 1e-4, "Intercept should be 0.0 for y=x")
        }
    }

    func testPolyFeaturesLinearWithFreqScale() {
        // With default sr=22050, nFFT=2048, freq[i] = i * 22050/2048 ~ i * 10.77
        // y = freq[i] should give slope=1, intercept=0 when fitting against freq
        let nFeatures = 4
        let nFrames = 3
        var vals = [Float](repeating: 0, count: nFeatures * nFrames)
        let freqScale = Float(22050) / Float(2048)
        for t in 0..<nFrames {
            for f in 0..<nFeatures {
                vals[f * nFrames + t] = Float(f) * freqScale  // y = freq[f]
            }
        }
        let data = Signal(data: vals, shape: [nFeatures, nFrames], sampleRate: 22050)
        let result = PolyFeatures.compute(data: data, order: 1, sr: 22050, nFFT: 2048)
        // Slope should be 1.0 (y = 1.0 * freq + 0.0)
        for t in 0..<nFrames {
            XCTAssertEqual(result[t], 1.0, accuracy: 1e-3, "Slope should be 1.0")
        }
        // Intercept should be ~0.0
        for t in 0..<nFrames {
            XCTAssertEqual(result[nFrames + t], 0.0, accuracy: 1e-2, "Intercept should be ~0.0")
        }
    }

    func testPolyFeaturesQuadratic() {
        // Use sr=nFFT so freqScale=1, freq[i]=i
        // y = x^2 for x = [0, 1, 2, 3, 4]
        var vals = [Float](repeating: 0, count: 15)
        for t in 0..<3 {
            for f in 0..<5 {
                vals[f * 3 + t] = Float(f * f)
            }
        }
        let data = Signal(data: vals, shape: [5, 3], sampleRate: 22050)
        let result = PolyFeatures.compute(data: data, order: 2, sr: 5, nFFT: 5)
        XCTAssertEqual(result.shape, [3, 3])
        // coeff[0] (x^2 term) should be 1.0
        for t in 0..<3 {
            XCTAssertEqual(result[t], 1.0, accuracy: 1e-3, "x^2 coeff should be 1.0")
        }
        // coeff[1] (x term) should be ~0.0
        for t in 0..<3 {
            XCTAssertEqual(result[3 + t], 0.0, accuracy: 1e-3, "x coeff should be 0.0")
        }
        // coeff[2] (constant) should be ~0.0
        for t in 0..<3 {
            XCTAssertEqual(result[6 + t], 0.0, accuracy: 1e-3, "constant should be 0.0")
        }
    }

    func testPolyFeaturesValuesFinite() {
        let data = Signal(data: (0..<100).map { Float($0) * 0.1 }, shape: [10, 10], sampleRate: 22050)
        let result = PolyFeatures.compute(data: data, order: 2)
        for i in 0..<result.count {
            XCTAssertFalse(result[i].isNaN, "Value at \(i) is NaN")
            XCTAssertFalse(result[i].isInfinite, "Value at \(i) is infinite")
        }
    }

    func testPolyFeaturesLargeNFeatures() {
        // Test with realistic spectrogram size (1025 frequency bins)
        // This tests numerical stability with large frequency values
        let nFeatures = 1025
        let nFrames = 3
        let sr = 22050
        let nFFT = 2048
        let freqScale = Float(sr) / Float(nFFT)
        var vals = [Float](repeating: 0, count: nFeatures * nFrames)
        for t in 0..<nFrames {
            for f in 0..<nFeatures {
                // y = freq[f] = f * freqScale
                vals[f * nFrames + t] = Float(f) * freqScale
            }
        }
        let data = Signal(data: vals, shape: [nFeatures, nFrames], sampleRate: sr)
        let result = PolyFeatures.compute(data: data, order: 1, sr: sr, nFFT: nFFT)
        XCTAssertEqual(result.shape, [2, nFrames])
        // Slope should be 1.0 (y = freq, fitting against freq)
        for t in 0..<nFrames {
            XCTAssertEqual(result[t], 1.0, accuracy: 1e-2, "Slope should be ~1.0")
        }
        // Intercept should be ~0.0
        for t in 0..<nFrames {
            XCTAssertEqual(result[nFrames + t], 0.0, accuracy: 1e-1, "Intercept should be ~0.0")
        }
        // All values should be finite
        for i in 0..<result.count {
            XCTAssertFalse(result[i].isNaN)
            XCTAssertFalse(result[i].isInfinite)
        }
    }
}
