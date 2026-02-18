import XCTest
@testable import MetalMomCore

/// Parameterized tests that sweep across hundreds of parameter combinations to find
/// edge cases in core MetalMom operations. Each test uses `XCTContext.runActivity` to
/// group sub-tests for clear failure reporting.
final class ParameterSweepTests: XCTestCase {

    // MARK: - Shared Test Signals

    /// 1-second 440 Hz sine at 22050 Hz (default test signal).
    private static let defaultSignal: Signal = {
        SignalGen.tone(frequency: 440.0, sr: 22050, duration: 1.0)
    }()

    /// Helper: generate a 440 Hz sine at a given sample rate and length.
    private func makeSine(sampleRate: Int, length: Int) -> Signal {
        SignalGen.tone(frequency: 440.0, sr: sampleRate, length: length)
    }

    /// Helper: check that all values in a Signal are finite.
    private func assertAllFinite(_ signal: Signal, label: String, file: StaticString = #file, line: UInt = #line) {
        signal.withUnsafeBufferPointer { buf in
            for i in 0..<buf.count {
                if !buf[i].isFinite {
                    XCTFail("\(label): non-finite value at index \(i) (\(buf[i]))", file: file, line: line)
                    return
                }
            }
        }
    }

    // MARK: - 1. STFT Parameter Sweep (60 combos)

    func testSTFTParameterSweep() {
        let nFFTs = [256, 512, 1024, 2048, 4096]
        let hopDivisors = [4, 2]  // hopLength = nFFT / divisor
        let centerValues = [true, false]
        let signalLengths = [4096, 22050, 44100]

        var comboCount = 0

        for nFFT in nFFTs {
            for hopDiv in hopDivisors {
                let hop = nFFT / hopDiv
                for center in centerValues {
                    for sigLen in signalLengths {
                        comboCount += 1
                        let label = "nFFT=\(nFFT) hop=\(hop) center=\(center) len=\(sigLen)"

                        XCTContext.runActivity(named: label) { _ in
                            let signal = makeSine(sampleRate: 22050, length: sigLen)
                            let result = STFT.compute(
                                signal: signal,
                                nFFT: nFFT,
                                hopLength: hop,
                                winLength: nFFT,
                                center: center
                            )

                            // Shape checks
                            XCTAssertEqual(result.shape.count, 2,
                                "\(label): expected 2D output")
                            XCTAssertEqual(result.shape[0], nFFT / 2 + 1,
                                "\(label): nFreqs should be nFFT/2+1")

                            let nFrames = result.shape[1]

                            // For center=true, signal is padded so we always get frames
                            // For center=false with very short signals, nFrames may be 0
                            if center {
                                XCTAssertGreaterThan(nFrames, 0,
                                    "\(label): should produce at least 1 frame with center=true")
                            } else {
                                if sigLen >= nFFT {
                                    XCTAssertGreaterThan(nFrames, 0,
                                        "\(label): signal >= nFFT should produce frames")
                                }
                                // sigLen < nFFT with center=false may yield 0 frames; that is valid
                            }

                            // Verify expected frame count
                            if center {
                                let paddedLen = sigLen + nFFT
                                let expectedFrames = 1 + (paddedLen - nFFT) / hop
                                XCTAssertEqual(nFrames, expectedFrames,
                                    "\(label): frame count mismatch")
                            }

                            // All values finite
                            assertAllFinite(result, label: label)
                        }
                    }
                }
            }
        }

        XCTAssertEqual(comboCount, 60, "Expected 60 STFT parameter combinations")
    }

    // MARK: - 2. Mel Spectrogram Parameter Sweep (10 combos)

    func testMelParameterSweep() {
        let nMelsValues = [1, 20, 40, 128, 256]
        let nFFTs = [512, 2048]

        var comboCount = 0

        for nMels in nMelsValues {
            for nFFT in nFFTs {
                comboCount += 1
                let label = "nMels=\(nMels) nFFT=\(nFFT)"

                XCTContext.runActivity(named: label) { _ in
                    let result = MelSpectrogram.compute(
                        signal: Self.defaultSignal,
                        nFFT: nFFT,
                        nMels: nMels
                    )

                    XCTAssertEqual(result.shape.count, 2,
                        "\(label): expected 2D output")
                    XCTAssertEqual(result.shape[0], nMels,
                        "\(label): first dim should be nMels")
                    XCTAssertGreaterThan(result.shape[1], 0,
                        "\(label): should produce at least 1 frame")

                    assertAllFinite(result, label: label)
                }
            }
        }

        XCTAssertEqual(comboCount, 10, "Expected 10 Mel parameter combinations")
    }

    // MARK: - 3. MFCC Parameter Sweep (8 combos)

    func testMFCCParameterSweep() {
        let nMFCCs = [1, 13, 20, 40]
        let nMelsValues = [40, 128]

        var comboCount = 0

        for nMFCC in nMFCCs {
            for nMels in nMelsValues {
                comboCount += 1
                let label = "nMFCC=\(nMFCC) nMels=\(nMels)"

                XCTContext.runActivity(named: label) { _ in
                    let result = MFCC.compute(
                        signal: Self.defaultSignal,
                        nMFCC: nMFCC,
                        nMels: nMels
                    )

                    let expectedCoeffs = min(nMFCC, nMels)

                    XCTAssertEqual(result.shape.count, 2,
                        "\(label): expected 2D output")
                    XCTAssertEqual(result.shape[0], expectedCoeffs,
                        "\(label): first dim should be min(nMFCC, nMels)")
                    XCTAssertGreaterThan(result.shape[1], 0,
                        "\(label): should produce at least 1 frame")

                    assertAllFinite(result, label: label)
                }
            }
        }

        XCTAssertEqual(comboCount, 8, "Expected 8 MFCC parameter combinations")
    }

    // MARK: - 4. YIN Parameter Sweep (9 combos)

    func testYINParameterSweep() {
        let freqRanges: [(fMin: Float, fMax: Float)] = [
            (50, 500),
            (100, 2000),
            (200, 4000)
        ]
        let frameLengths = [1024, 2048, 4096]

        var comboCount = 0

        for (fMin, fMax) in freqRanges {
            for frameLength in frameLengths {
                comboCount += 1
                let label = "fMin=\(fMin) fMax=\(fMax) frameLength=\(frameLength)"

                XCTContext.runActivity(named: label) { _ in
                    let result = YIN.yin(
                        signal: Self.defaultSignal,
                        fMin: fMin,
                        fMax: fMax,
                        frameLength: frameLength
                    )

                    XCTAssertGreaterThan(result.count, 0,
                        "\(label): should produce pitch estimates")

                    // All pitch values should be non-negative and finite
                    result.withUnsafeBufferPointer { buf in
                        for i in 0..<buf.count {
                            XCTAssertTrue(buf[i].isFinite,
                                "\(label): non-finite pitch at index \(i)")
                            XCTAssertGreaterThanOrEqual(buf[i], 0,
                                "\(label): negative pitch at index \(i)")
                        }
                    }
                }
            }
        }

        XCTAssertEqual(comboCount, 9, "Expected 9 YIN parameter combinations")
    }

    // MARK: - 5. HPSS Parameter Sweep (16 combos)

    func testHPSSParameterSweep() {
        let kernelSizes = [1, 15, 31, 65]  // all odd
        let powers: [Float] = [1.0, 2.0]
        let margins: [Float] = [1.0, 3.0]

        // Use a shorter signal to keep HPSS runtime reasonable
        let shortSignal = makeSine(sampleRate: 22050, length: 11025)  // 0.5 seconds

        var comboCount = 0

        for kernelSize in kernelSizes {
            for power in powers {
                for margin in margins {
                    comboCount += 1
                    let label = "kernel=\(kernelSize) power=\(power) margin=\(margin)"

                    XCTContext.runActivity(named: label) { _ in
                        let (harmonic, percussive) = HPSS.hpss(
                            signal: shortSignal,
                            kernelSize: kernelSize,
                            power: power,
                            margin: margin
                        )

                        // Both components should have the same shape as the input
                        XCTAssertEqual(harmonic.count, shortSignal.count,
                            "\(label): harmonic length should match input")
                        XCTAssertEqual(percussive.count, shortSignal.count,
                            "\(label): percussive length should match input")

                        // Both should be finite
                        assertAllFinite(harmonic, label: "\(label) harmonic")
                        assertAllFinite(percussive, label: "\(label) percussive")
                    }
                }
            }
        }

        XCTAssertEqual(comboCount, 16, "Expected 16 HPSS parameter combinations")
    }

    // MARK: - 6. Resample Parameter Sweep (12 combos)

    func testResampleParameterSweep() {
        let sampleRates = [8000, 16000, 22050, 44100]

        var comboCount = 0

        for sourceSR in sampleRates {
            for targetSR in sampleRates {
                guard sourceSR != targetSR else { continue }
                comboCount += 1
                let label = "source=\(sourceSR) target=\(targetSR)"

                XCTContext.runActivity(named: label) { _ in
                    // Generate 0.5-second signal at the source rate
                    let sigLen = sourceSR / 2
                    let signal = makeSine(sampleRate: sourceSR, length: sigLen)

                    let result = Resample.resample(signal: signal, targetRate: targetSR)

                    XCTAssertEqual(result.sampleRate, targetSR,
                        "\(label): sample rate should match target")
                    XCTAssertGreaterThan(result.count, 0,
                        "\(label): should produce output samples")

                    // Check approximate expected length
                    let expectedLen = Int(ceil(Double(sigLen) * Double(targetSR) / Double(sourceSR)))
                    let tolerance = max(2, expectedLen / 100)  // allow 1% or 2 samples
                    XCTAssertEqual(result.count, expectedLen, accuracy: tolerance,
                        "\(label): output length should be approximately \(expectedLen)")

                    assertAllFinite(result, label: label)
                }
            }
        }

        XCTAssertEqual(comboCount, 12, "Expected 12 Resample parameter combinations")
    }

    // MARK: - 7. CQT Parameter Sweep (6 combos)

    func testCQTParameterSweep() {
        let binsPerOctaveValues = [12, 24, 36]
        let fMinValues: [Float] = [32.70, 65.41]

        var comboCount = 0

        for bpo in binsPerOctaveValues {
            for fMin in fMinValues {
                comboCount += 1
                let label = "binsPerOctave=\(bpo) fMin=\(fMin)"

                XCTContext.runActivity(named: label) { _ in
                    let result = CQT.compute(
                        signal: Self.defaultSignal,
                        fMin: fMin,
                        binsPerOctave: bpo
                    )

                    XCTAssertGreaterThan(result.count, 0,
                        "\(label): should produce output")
                    XCTAssertEqual(result.shape.count, 2,
                        "\(label): should be 2D [nBins, nFrames]")
                    XCTAssertGreaterThan(result.shape[0], 0,
                        "\(label): should have CQT bins")
                    XCTAssertGreaterThan(result.shape[1], 0,
                        "\(label): should have frames")

                    assertAllFinite(result, label: label)
                }
            }
        }

        XCTAssertEqual(comboCount, 6, "Expected 6 CQT parameter combinations")
    }

    // MARK: - 8. Chroma Parameter Sweep (9 combos)

    func testChromaParameterSweep() {
        let nFFTs = [512, 1024, 2048]
        let signalLengths = [4096, 11025, 22050]

        var comboCount = 0

        for nFFT in nFFTs {
            for sigLen in signalLengths {
                comboCount += 1
                let label = "nFFT=\(nFFT) sigLen=\(sigLen)"

                XCTContext.runActivity(named: label) { _ in
                    let signal = makeSine(sampleRate: 22050, length: sigLen)

                    let result = Chroma.stft(
                        signal: signal,
                        nFFT: nFFT
                    )

                    XCTAssertEqual(result.shape.count, 2,
                        "\(label): should be 2D")
                    XCTAssertEqual(result.shape[0], 12,
                        "\(label): should have 12 chroma bins")
                    XCTAssertGreaterThan(result.shape[1], 0,
                        "\(label): should produce frames")

                    assertAllFinite(result, label: label)
                }
            }
        }

        XCTAssertEqual(comboCount, 9, "Expected 9 Chroma parameter combinations")
    }

    // MARK: - 9. Onset Strength Parameter Sweep (3 signal types)

    func testOnsetStrengthParameterSweep() {
        let sr = 22050

        // Signal type 1: Pure sine
        let sine = SignalGen.tone(frequency: 440.0, sr: sr, duration: 1.0)

        // Signal type 2: Random noise
        var noiseData = [Float](repeating: 0, count: sr)
        for i in 0..<sr {
            // Simple deterministic pseudo-random via sine mixing
            noiseData[i] = sinf(Float(i) * 0.123) * cosf(Float(i) * 0.456) * sinf(Float(i) * 0.789)
        }
        let noise = Signal(data: noiseData, sampleRate: sr)

        // Signal type 3: Silence
        let silence = Signal(data: [Float](repeating: 0, count: sr), sampleRate: sr)

        let signals: [(name: String, signal: Signal)] = [
            ("sine", sine),
            ("noise", noise),
            ("silence", silence)
        ]

        for (name, signal) in signals {
            XCTContext.runActivity(named: "onset_\(name)") { _ in
                let result = OnsetDetection.onsetStrength(signal: signal)

                XCTAssertGreaterThan(result.count, 0,
                    "\(name): should produce onset strength values")

                assertAllFinite(result, label: "onset_\(name)")

                // Silence should produce near-zero onset strength
                if name == "silence" {
                    result.withUnsafeBufferPointer { buf in
                        var maxVal: Float = 0
                        for i in 0..<buf.count {
                            maxVal = max(maxVal, abs(buf[i]))
                        }
                        XCTAssertLessThan(maxVal, 1e-3,
                            "silence: onset strength should be near zero, got max=\(maxVal)")
                    }
                }
            }
        }
    }

    // MARK: - 10. Time Stretch Parameter Sweep (5 combos)

    func testTimeStretchParameterSweep() {
        let rates: [Float] = [0.5, 1.0, 1.5, 2.0, 4.0]

        // Use a shorter signal so time stretching is fast
        let shortSignal = makeSine(sampleRate: 22050, length: 11025)  // 0.5 seconds

        for rate in rates {
            let label = "rate=\(rate)"

            XCTContext.runActivity(named: label) { _ in
                let result = TimeStretch.timeStretch(
                    signal: shortSignal,
                    rate: rate
                )

                // All values should be finite
                assertAllFinite(result, label: label)

                // Output length should be approximately inputLength / rate
                // Allow +/- 20% tolerance due to STFT framing effects
                let inputLen = shortSignal.count
                let expectedLen = Float(inputLen) / rate
                let actualLen = Float(result.count)

                let ratio = actualLen / expectedLen
                XCTAssertGreaterThan(ratio, 0.8,
                    "\(label): output too short. expected ~\(Int(expectedLen)), got \(result.count)")
                XCTAssertLessThan(ratio, 1.2,
                    "\(label): output too long. expected ~\(Int(expectedLen)), got \(result.count)")
            }
        }
    }
}

// MARK: - XCTAssertEqual with accuracy for Int

private func XCTAssertEqual(
    _ expression1: Int,
    _ expression2: Int,
    accuracy: Int,
    _ message: String = "",
    file: StaticString = #file,
    line: UInt = #line
) {
    let diff = abs(expression1 - expression2)
    XCTAssertTrue(diff <= accuracy,
        "\(message) - expected \(expression2) +/- \(accuracy), got \(expression1) (diff=\(diff))",
        file: file, line: line)
}
