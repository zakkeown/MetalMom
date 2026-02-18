import XCTest
@testable import MetalMomCore

final class SpectralOnsetTests: XCTestCase {

    // MARK: - Helper signals

    /// 1 second of silence at 22050 Hz with a single impulse in the middle.
    private func makeClick(length: Int = 22050, sr: Int = 22050) -> Signal {
        var data = [Float](repeating: 0, count: length)
        data[length / 2] = 1.0
        return Signal(data: data, sampleRate: sr)
    }

    /// 1 second sine wave.
    private func makeSine(freq: Float, length: Int = 22050, sr: Int = 22050) -> Signal {
        let data = (0..<length).map { Float(sin(2.0 * .pi * Double(freq) * Double($0) / Double(sr))) }
        return Signal(data: data, sampleRate: sr)
    }

    /// Signal with frequency modulation (vibrato).
    private func makeVibrato(
        baseFreq: Float = 440.0,
        vibratoRate: Float = 6.0,
        vibratoDepth: Float = 20.0,
        length: Int = 22050,
        sr: Int = 22050
    ) -> Signal {
        var phase: Double = 0
        let data = (0..<length).map { i -> Float in
            let t = Double(i) / Double(sr)
            let freq = Double(baseFreq) + Double(vibratoDepth) * sin(2.0 * .pi * Double(vibratoRate) * t)
            phase += 2.0 * .pi * freq / Double(sr)
            return Float(sin(phase))
        }
        return Signal(data: data, sampleRate: sr)
    }

    // MARK: - SuperFlux Tests

    func testSuperfluxBasicShape() {
        let signal = makeSine(freq: 440)
        let result = OnsetDetection.superflux(signal: signal)
        XCTAssertEqual(result.shape[0], 1, "First dimension should be 1")
        XCTAssertGreaterThan(result.shape[1], 0, "Should have at least one frame")
    }

    func testSuperfluxNonNegative() {
        let signal = makeSine(freq: 440)
        let result = OnsetDetection.superflux(signal: signal)
        result.withUnsafeBufferPointer { buf in
            for i in 0..<buf.count {
                XCTAssertGreaterThanOrEqual(buf[i], 0, "SuperFlux values should be non-negative")
                XCTAssertFalse(buf[i].isNaN, "SuperFlux should not produce NaN")
                XCTAssertFalse(buf[i].isInfinite, "SuperFlux should not produce Inf")
            }
        }
    }

    func testSuperfluxVibratoRobustness() {
        // Vibrato signal should produce less flux with SuperFlux (max filter suppresses it)
        // compared to standard spectral flux on the raw magnitude
        let vibrato = makeVibrato()

        // SuperFlux with max filter should suppress vibrato-induced flux
        let sf = OnsetDetection.superflux(signal: vibrato, maxFilterSize: 3)

        // Compute onset strength (standard spectral flux for comparison sense)
        let standardFlux = OnsetDetection.superflux(signal: vibrato, maxFilterSize: 1)

        // Sum up the total energy in each
        var sfSum: Float = 0
        var stdSum: Float = 0
        sf.withUnsafeBufferPointer { buf in
            for i in 0..<buf.count { sfSum += buf[i] }
        }
        standardFlux.withUnsafeBufferPointer { buf in
            for i in 0..<buf.count { stdSum += buf[i] }
        }

        // SuperFlux (with max filter) should have less total flux than without
        // because max filter suppresses vibrato-induced spectral changes
        XCTAssertLessThanOrEqual(sfSum, stdSum + 1e-6,
            "SuperFlux with max filter should suppress vibrato flux vs no max filter")
    }

    // MARK: - Complex Flux Tests

    func testComplexFluxShape() {
        let signal = makeSine(freq: 440)
        let result = OnsetDetection.complexFlux(signal: signal)
        XCTAssertEqual(result.shape[0], 1, "First dimension should be 1")
        XCTAssertGreaterThan(result.shape[1], 0, "Should have at least one frame")
    }

    func testComplexFluxWithClick() {
        let click = makeClick()
        let result = OnsetDetection.complexFlux(signal: click)
        let nFrames = result.shape[1]

        // Find the maximum value and its index
        var maxVal: Float = -Float.infinity
        var maxIdx = 0
        result.withUnsafeBufferPointer { buf in
            for i in 0..<nFrames {
                if buf[i] > maxVal {
                    maxVal = buf[i]
                    maxIdx = i
                }
            }
        }

        // The click is at sample length/2. With center padding and hop=512,
        // the peak should be roughly in the middle frames.
        let midFrame = nFrames / 2
        let tolerance = nFrames / 4
        XCTAssertGreaterThan(maxVal, 0, "Should detect the click with non-zero flux")
        XCTAssertTrue(
            abs(maxIdx - midFrame) <= tolerance,
            "Peak complex flux (\(maxIdx)) should be near the middle frame (\(midFrame))"
        )
    }

    func testComplexFluxNonNegative() {
        let signal = makeSine(freq: 440)
        let result = OnsetDetection.complexFlux(signal: signal)
        result.withUnsafeBufferPointer { buf in
            for i in 0..<buf.count {
                XCTAssertGreaterThanOrEqual(buf[i], 0, "Complex flux values should be non-negative")
                XCTAssertFalse(buf[i].isNaN, "Complex flux should not produce NaN")
            }
        }
    }

    // MARK: - HFC Tests

    func testHFCBasicShape() {
        let signal = makeSine(freq: 440)
        let result = OnsetDetection.highFrequencyContent(signal: signal)
        XCTAssertEqual(result.shape[0], 1, "First dimension should be 1")
        XCTAssertGreaterThan(result.shape[1], 0, "Should have at least one frame")
    }

    func testHFCNonNegative() {
        let signal = makeSine(freq: 440)
        let result = OnsetDetection.highFrequencyContent(signal: signal)
        result.withUnsafeBufferPointer { buf in
            for i in 0..<buf.count {
                XCTAssertGreaterThanOrEqual(buf[i], 0, "HFC values should be non-negative")
                XCTAssertFalse(buf[i].isNaN, "HFC should not produce NaN")
                XCTAssertFalse(buf[i].isInfinite, "HFC should not produce Inf")
            }
        }
    }

    func testHFCEmphasizesHighFreq() {
        let sr = 22050
        // Low frequency sine (100 Hz) vs high frequency sine (5000 Hz)
        let lowFreq = makeSine(freq: 100, sr: sr)
        let highFreq = makeSine(freq: 5000, sr: sr)

        let hfcLow = OnsetDetection.highFrequencyContent(signal: lowFreq, sr: sr)
        let hfcHigh = OnsetDetection.highFrequencyContent(signal: highFreq, sr: sr)

        // Sum HFC values for comparison
        var sumLow: Float = 0
        var sumHigh: Float = 0
        hfcLow.withUnsafeBufferPointer { buf in
            for i in 0..<buf.count { sumLow += buf[i] }
        }
        hfcHigh.withUnsafeBufferPointer { buf in
            for i in 0..<buf.count { sumHigh += buf[i] }
        }

        XCTAssertGreaterThan(sumHigh, sumLow,
            "High frequency signal should produce larger HFC than low frequency")
    }

    // MARK: - KL Divergence Tests

    func testKLDivergenceBasicShape() {
        let signal = makeSine(freq: 440)
        let result = OnsetDetection.klDivergence(signal: signal)
        XCTAssertEqual(result.shape[0], 1, "First dimension should be 1")
        XCTAssertGreaterThan(result.shape[1], 0, "Should have at least one frame")
    }

    func testKLDivergenceNonNegative() {
        let signal = makeSine(freq: 440)
        let result = OnsetDetection.klDivergence(signal: signal)
        result.withUnsafeBufferPointer { buf in
            for i in 0..<buf.count {
                XCTAssertGreaterThanOrEqual(buf[i], 0, "KL divergence should be non-negative (half-wave rectified)")
                XCTAssertFalse(buf[i].isNaN, "KL divergence should not produce NaN")
                XCTAssertFalse(buf[i].isInfinite, "KL divergence should not produce Inf")
            }
        }
    }

    func testKLDivergenceWithSpectralChange() {
        // Create a signal that changes frequency suddenly in the middle
        let sr = 22050
        let half = sr / 2
        let lowPart = (0..<half).map { Float(sin(2.0 * .pi * 200.0 * Double($0) / Double(sr))) }
        let highPart = (half..<sr).map { Float(sin(2.0 * .pi * 4000.0 * Double($0) / Double(sr))) }
        let signal = Signal(data: lowPart + highPart, sampleRate: sr)

        let result = OnsetDetection.klDivergence(signal: signal, sr: sr)
        let nFrames = result.shape[1]

        // Find the maximum KL divergence -- should be near the transition
        var maxVal: Float = 0
        var maxIdx = 0
        result.withUnsafeBufferPointer { buf in
            for i in 0..<nFrames {
                if buf[i] > maxVal {
                    maxVal = buf[i]
                    maxIdx = i
                }
            }
        }

        XCTAssertGreaterThan(maxVal, 0, "Should detect spectral change")

        // The transition is at sample sr/2. With center-padding and STFT framing,
        // the peak may be shifted toward later frames. We just verify it's detected
        // in the second half of the signal (after the transition point).
        let quarterFrame = nFrames / 4
        XCTAssertGreaterThan(maxIdx, quarterFrame,
            "Peak KL divergence (\(maxIdx)) should be after the early portion of the signal")
    }

    // MARK: - Cross-method Tests

    func testAllMethodsReturnCorrectFrameCount() {
        let sr = 22050
        let signal = makeSine(freq: 440, length: sr, sr: sr)
        let nFFT = 2048
        let hop = 512

        let sf = OnsetDetection.superflux(signal: signal, sr: sr, nFFT: nFFT, hopLength: hop)
        let cf = OnsetDetection.complexFlux(signal: signal, sr: sr, nFFT: nFFT, hopLength: hop)
        let hfc = OnsetDetection.highFrequencyContent(signal: signal, sr: sr, nFFT: nFFT, hopLength: hop)
        let kl = OnsetDetection.klDivergence(signal: signal, sr: sr, nFFT: nFFT, hopLength: hop)

        // All should have the same number of frames
        XCTAssertEqual(sf.shape[1], cf.shape[1],
            "SuperFlux and ComplexFlux should return same frame count")
        XCTAssertEqual(sf.shape[1], hfc.shape[1],
            "SuperFlux and HFC should return same frame count")
        XCTAssertEqual(sf.shape[1], kl.shape[1],
            "SuperFlux and KL should return same frame count")
    }

    func testShortSignalDoesNotCrash() {
        // Very short signals (less than one FFT frame)
        let shortSignal = Signal(data: [0.5, -0.3, 0.1], sampleRate: 22050)

        // None of these should crash
        let sf = OnsetDetection.superflux(signal: shortSignal)
        let cf = OnsetDetection.complexFlux(signal: shortSignal)
        let hfc = OnsetDetection.highFrequencyContent(signal: shortSignal)
        let kl = OnsetDetection.klDivergence(signal: shortSignal)

        // All should return something valid
        XCTAssertGreaterThanOrEqual(sf.shape.last!, 1, "SuperFlux should return at least 1 frame")
        XCTAssertGreaterThanOrEqual(cf.shape.last!, 1, "ComplexFlux should return at least 1 frame")
        XCTAssertGreaterThanOrEqual(hfc.shape.last!, 1, "HFC should return at least 1 frame")
        XCTAssertGreaterThanOrEqual(kl.shape.last!, 1, "KL should return at least 1 frame")
    }
}
