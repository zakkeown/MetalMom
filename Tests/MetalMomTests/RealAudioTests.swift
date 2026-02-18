import XCTest
@testable import MetalMomCore

/// Tests that exercise the full audio pipeline with synthetic but realistic signals.
///
/// These tests verify that MetalMom produces physically meaningful results for
/// signals with known spectral, temporal, and harmonic properties.
final class RealAudioTests: XCTestCase {

    // MARK: - Helpers

    /// Seeded random number generator for reproducible white noise.
    struct SeededRNG: RandomNumberGenerator {
        private var state: UInt64

        init(seed: UInt64) {
            self.state = seed
        }

        mutating func next() -> UInt64 {
            // xorshift64 — simple, fast, deterministic
            state ^= state << 13
            state ^= state >> 7
            state ^= state << 17
            return state
        }
    }

    /// Generate white noise using a seeded RNG.
    private func whiteNoise(count: Int, seed: UInt64 = 42) -> [Float] {
        var rng = SeededRNG(seed: seed)
        return (0..<count).map { _ in Float.random(in: -1...1, using: &rng) }
    }

    /// Generate a linear chirp: frequency sweeps linearly from f0 to f1 over duration seconds.
    private func linearChirp(f0: Float, f1: Float, duration: Float, sr: Int) -> [Float] {
        let n = Int(duration * Float(sr))
        return (0..<n).map { i in
            let t = Float(i) / Float(sr)
            // Instantaneous frequency: f(t) = f0 + (f1 - f0) * t / duration
            // Phase integral: phi(t) = 2 * pi * (f0 * t + (f1 - f0) * t^2 / (2 * duration))
            let phase = 2.0 * Float.pi * (f0 * t + (f1 - f0) * t * t / (2.0 * duration))
            return sinf(phase)
        }
    }

    /// Generate a short sine burst (click) at a given sample offset.
    private func sineBurst(at offset: Int, frequency: Float, burstSamples: Int, sr: Int) -> (offset: Int, data: [Float]) {
        let data = (0..<burstSamples).map { i in
            sinf(2.0 * Float.pi * frequency * Float(i) / Float(sr))
        }
        return (offset, data)
    }

    // MARK: - Test 1: White Noise Flat Spectrum

    func testWhiteNoiseFlat() {
        let sr = 22050
        let duration = 2.0
        let n = Int(duration) * sr
        let noise = whiteNoise(count: n, seed: 42)
        let signal = Signal(data: noise, sampleRate: sr)

        // Compute mel spectrogram (power, 128 mel bands)
        let melSpec = MelSpectrogram.compute(signal: signal, sr: sr)
        let nMels = melSpec.shape[0]
        let nFrames = melSpec.shape[1]

        XCTAssertGreaterThan(nMels, 0, "Mel spectrogram should have mel bands")
        XCTAssertGreaterThan(nFrames, 0, "Mel spectrogram should have frames")

        // Sum energy across time for each mel band
        var bandEnergies = [Float](repeating: 0, count: nMels)
        melSpec.withUnsafeBufferPointer { buf in
            for m in 0..<nMels {
                for f in 0..<nFrames {
                    bandEnergies[m] += buf[m * nFrames + f]
                }
            }
        }

        // Find max and min band energies (excluding bands that might be near zero
        // at the very edges of the mel scale)
        let trimmedBands = Array(bandEnergies[2..<(nMels - 2)])
        let maxEnergy = trimmedBands.max() ?? 0
        let minEnergy = trimmedBands.min() ?? 0

        XCTAssertGreaterThan(minEnergy, 0, "All mel bands should have some energy for white noise")

        let ratio = maxEnergy / minEnergy
        // White noise should have roughly flat spectrum; allow up to 10x variation
        // across mel bands (mel bands at extremes can have slight energy differences)
        XCTAssertLessThan(ratio, 10.0,
            "White noise mel band energy ratio (max/min = \(ratio)) should be < 10 for a roughly flat spectrum")
    }

    // MARK: - Test 2: Chirp Spectral Centroid Increases

    func testChirpSpectralCentroidIncreases() {
        let sr = 22050
        let chirp = linearChirp(f0: 200, f1: 4000, duration: 2.0, sr: sr)
        let signal = Signal(data: chirp, sampleRate: sr)

        // Compute magnitude spectrogram, then spectral centroid
        let stftMag = STFT.compute(signal: signal, nFFT: 2048)
        let centroid = SpectralDescriptors.centroid(spectrogram: stftMag, sr: sr, nFFT: 2048)

        let nFrames = centroid.count
        XCTAssertGreaterThan(nFrames, 4, "Need enough frames for quarter comparison")

        // Compare first quarter mean vs last quarter mean
        let q = nFrames / 4
        var firstQuarterSum: Float = 0
        var lastQuarterSum: Float = 0

        centroid.withUnsafeBufferPointer { buf in
            for i in 0..<q {
                firstQuarterSum += buf[i]
            }
            for i in (nFrames - q)..<nFrames {
                lastQuarterSum += buf[i]
            }
        }

        let firstQuarterMean = firstQuarterSum / Float(q)
        let lastQuarterMean = lastQuarterSum / Float(q)

        XCTAssertGreaterThan(lastQuarterMean, firstQuarterMean,
            "Chirp spectral centroid should increase: first quarter mean = \(firstQuarterMean) Hz, " +
            "last quarter mean = \(lastQuarterMean) Hz")
    }

    // MARK: - Test 3: Click Train Onset Detection

    func testClickTrainOnsetDetection() {
        let sr = 22050
        let duration: Float = 3.0
        let n = Int(duration * Float(sr))
        var samples = [Float](repeating: 0, count: n)

        // Place 10 clicks at 0.3s intervals, starting at 0.15s
        // Each click is a 1ms burst of 1kHz sine
        let burstSamples = Int(0.001 * Float(sr))  // ~22 samples for 1ms at 22050 Hz
        let clickCount = 10

        for i in 0..<clickCount {
            let clickTime = 0.15 + Float(i) * 0.3
            let offset = Int(clickTime * Float(sr))
            let burst = sineBurst(at: offset, frequency: 1000.0, burstSamples: burstSamples, sr: sr)
            for j in 0..<burst.data.count {
                let idx = burst.offset + j
                if idx < n {
                    samples[idx] = burst.data[j] * 0.9  // amplitude 0.9
                }
            }
        }

        let signal = Signal(data: samples, sampleRate: sr)

        // Run onset detection with sensitive parameters for short bursts
        let onsetFrames = OnsetDetection.detectOnsets(
            signal: signal,
            sr: sr,
            hopLength: 512,
            delta: 0.03,
            wait: 5
        )

        let detectedCount = onsetFrames.count
        // Require at least 60% of clicks found (>= 6 out of 10)
        XCTAssertGreaterThanOrEqual(detectedCount, 6,
            "Click train onset detection found \(detectedCount) onsets, expected >= 6 out of \(clickCount) clicks")
    }

    // MARK: - Test 4: Harmonic Complex Chroma

    func testHarmonicComplexChroma() {
        let sr = 22050
        let duration: Float = 1.0
        let n = Int(duration * Float(sr))

        // A4 = 440 Hz plus 5 harmonics (880, 1320, 1760, 2200, 2640)
        let fundamentals: [Float] = [440.0, 880.0, 1320.0, 1760.0, 2200.0, 2640.0]
        var samples = [Float](repeating: 0, count: n)
        for freq in fundamentals {
            for i in 0..<n {
                let t = Float(i) / Float(sr)
                samples[i] += sinf(2.0 * Float.pi * freq * t)
            }
        }

        // Normalize to [-1, 1]
        let maxAbs = samples.map { abs($0) }.max() ?? 1.0
        if maxAbs > 0 {
            for i in 0..<n {
                samples[i] /= maxAbs
            }
        }

        let signal = Signal(data: samples, sampleRate: sr)

        // Compute chroma features (12 bins, C-based: C=0, C#=1, ..., A=9, A#=10, B=11)
        let chroma = Chroma.stft(signal: signal, sr: sr, nFFT: 2048)

        let nChroma = chroma.shape[0]
        let nFrames = chroma.shape[1]
        XCTAssertEqual(nChroma, 12, "Chroma should have 12 bins")
        XCTAssertGreaterThan(nFrames, 0, "Chroma should have frames")

        // Sum chroma energy across all frames for each pitch class
        var chromaEnergy = [Float](repeating: 0, count: nChroma)
        chroma.withUnsafeBufferPointer { buf in
            for c in 0..<nChroma {
                for f in 0..<nFrames {
                    chromaEnergy[c] += buf[c * nFrames + f]
                }
            }
        }

        // In standard C-based chroma: A is at index 9
        let aIndex = 9
        let aEnergy = chromaEnergy[aIndex]

        // Find the index with maximum energy
        var maxIndex = 0
        var maxEnergy: Float = -1
        for c in 0..<nChroma {
            if chromaEnergy[c] > maxEnergy {
                maxEnergy = chromaEnergy[c]
                maxIndex = c
            }
        }

        let pitchNames = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        XCTAssertEqual(maxIndex, aIndex,
            "Harmonic complex at A4 should peak at chroma bin A (index \(aIndex)), " +
            "but peaked at \(pitchNames[maxIndex]) (index \(maxIndex)). " +
            "Energies: \(zip(pitchNames, chromaEnergy).map { "\($0): \($1)" }.joined(separator: ", "))")

        // Also verify A has significantly more energy than the average of non-A bins
        let otherEnergy = (chromaEnergy.reduce(0, +) - aEnergy) / Float(nChroma - 1)
        XCTAssertGreaterThan(aEnergy, otherEnergy * 1.5,
            "A chroma energy (\(aEnergy)) should be well above the average of other bins (\(otherEnergy))")
    }

    // MARK: - Test 5: Full Pipeline No Crash

    func testFullPipelineNoCrash() {
        let sr = 22050
        let duration: Float = 3.0
        let n = Int(duration * Float(sr))
        var rng = SeededRNG(seed: 123)

        // Generate music-like mixture: bass + melody + noise floor
        var samples = [Float](repeating: 0, count: n)

        // Bass: 110 Hz sustained tone
        for i in 0..<n {
            let t = Float(i) / Float(sr)
            samples[i] += 0.3 * sinf(2.0 * Float.pi * 110.0 * t)
        }

        // Melody: random note changes every 0.5s, frequencies from A3 to A5
        let noteFrequencies: [Float] = [220, 261.6, 293.7, 329.6, 349.2, 392.0, 440.0, 523.3, 587.3, 659.3, 698.5, 784.0, 880.0]
        let samplesPerNote = sr / 2  // 0.5s per note
        for noteIdx in 0..<(n / samplesPerNote) {
            let freqIdx = Int(rng.next() % UInt64(noteFrequencies.count))
            let freq = noteFrequencies[freqIdx]
            let start = noteIdx * samplesPerNote
            let end = min(start + samplesPerNote, n)
            for i in start..<end {
                let t = Float(i) / Float(sr)
                samples[i] += 0.2 * sinf(2.0 * Float.pi * freq * t)
            }
        }

        // Noise floor
        for i in 0..<n {
            samples[i] += 0.02 * Float.random(in: -1...1, using: &rng)
        }

        // Normalize
        let maxAbs = samples.map { abs($0) }.max() ?? 1.0
        if maxAbs > 0 {
            for i in 0..<n {
                samples[i] /= maxAbs
            }
        }

        let signal = Signal(data: samples, sampleRate: sr)

        // --- Run each stage of the pipeline and verify ---

        // STFT
        let stftResult = STFT.compute(signal: signal, nFFT: 2048)
        XCTAssertEqual(stftResult.shape.count, 2, "STFT should produce 2D output")
        XCTAssertEqual(stftResult.shape[0], 1025, "STFT should have nFFT/2+1 frequency bins")
        XCTAssertGreaterThan(stftResult.shape[1], 0, "STFT should have frames")
        assertAllFinite(stftResult, name: "STFT")

        // Mel spectrogram
        let melResult = MelSpectrogram.compute(signal: signal, sr: sr)
        XCTAssertEqual(melResult.shape[0], 128, "Mel spectrogram should have 128 mel bands")
        XCTAssertGreaterThan(melResult.shape[1], 0, "Mel spectrogram should have frames")
        assertAllFinite(melResult, name: "MelSpectrogram")

        // MFCC
        let mfccResult = MFCC.compute(signal: signal, sr: sr)
        XCTAssertEqual(mfccResult.shape[0], 20, "MFCC should have 20 coefficients by default")
        XCTAssertGreaterThan(mfccResult.shape[1], 0, "MFCC should have frames")
        assertAllFinite(mfccResult, name: "MFCC")

        // Onset detection
        let onsets = OnsetDetection.detectOnsets(signal: signal, sr: sr)
        // Just verify it returns something and values are finite
        assertAllFinite(onsets, name: "OnsetDetection")

        // Beat tracking
        let (tempo, beats) = BeatTracker.beatTrack(
            signal: signal,
            sr: sr,
            startBPM: 120.0,
            trimFirst: false,
            trimLast: false
        )
        XCTAssertFalse(tempo.isNaN, "Tempo should not be NaN")
        XCTAssertFalse(tempo.isInfinite, "Tempo should not be infinite")
        XCTAssertGreaterThan(tempo, 0, "Tempo should be positive for a signal with energy")
        assertAllFinite(beats, name: "BeatTracker")
    }

    // MARK: - Assertion Helpers

    /// Assert that all elements in a Signal are finite (not NaN, not Inf).
    private func assertAllFinite(_ signal: Signal, name: String, file: StaticString = #file, line: UInt = #line) {
        signal.withUnsafeBufferPointer { buf in
            for i in 0..<buf.count {
                if buf[i].isNaN || buf[i].isInfinite {
                    XCTFail("\(name) contains non-finite value at index \(i): \(buf[i])", file: file, line: line)
                    return
                }
            }
        }
    }
}
