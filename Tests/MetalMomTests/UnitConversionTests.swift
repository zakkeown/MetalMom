import XCTest
@testable import MetalMomCore

final class UnitConversionTests: XCTestCase {

    // MARK: - Hz <-> MIDI round-trip

    func testHzToMidiA4() {
        let midi = Units.hzToMidi(440.0)
        XCTAssertEqual(midi, 69.0, accuracy: 1e-5, "A4 (440 Hz) should map to MIDI 69")
    }

    func testMidiToHzA4() {
        let hz = Units.midiToHz(69.0)
        XCTAssertEqual(hz, 440.0, accuracy: 1e-3, "MIDI 69 should map to 440 Hz")
    }

    func testHzMidiRoundTrip() {
        let hzValues: [Float] = [27.5, 65.41, 130.81, 261.63, 440.0, 880.0, 4186.01]
        for hz in hzValues {
            let midi = Units.hzToMidi(hz)
            let recovered = Units.midiToHz(midi)
            XCTAssertEqual(recovered, hz, accuracy: hz * 1e-4,
                           "Hz->MIDI->Hz round-trip failed for \(hz)")
        }
    }

    func testMidiHzRoundTrip() {
        let midiValues: [Float] = [21, 36, 48, 60, 69, 72, 84, 96, 108]
        for midi in midiValues {
            let hz = Units.midiToHz(midi)
            let recovered = Units.hzToMidi(hz)
            XCTAssertEqual(recovered, midi, accuracy: 1e-4,
                           "MIDI->Hz->MIDI round-trip failed for \(midi)")
        }
    }

    func testHzToMidiVectorized() {
        let hzArray: [Float] = [440.0, 880.0, 220.0]
        let midiArray = Units.hzToMidi(hzArray)
        XCTAssertEqual(midiArray.count, 3)
        XCTAssertEqual(midiArray[0], 69.0, accuracy: 1e-5)
        XCTAssertEqual(midiArray[1], 81.0, accuracy: 1e-5)
        XCTAssertEqual(midiArray[2], 57.0, accuracy: 1e-5)
    }

    func testMidiToHzVectorized() {
        let midiArray: [Float] = [69.0, 81.0, 57.0]
        let hzArray = Units.midiToHz(midiArray)
        XCTAssertEqual(hzArray.count, 3)
        XCTAssertEqual(hzArray[0], 440.0, accuracy: 1e-2)
        XCTAssertEqual(hzArray[1], 880.0, accuracy: 1e-1)
        XCTAssertEqual(hzArray[2], 220.0, accuracy: 1e-2)
    }

    // MARK: - Hz <-> Note name

    func testHzToNoteA4() {
        let note = Units.hzToNote(440.0)
        XCTAssertEqual(note, "A4", "440 Hz should be A4")
    }

    func testHzToNoteC4() {
        // C4 = MIDI 60 = 261.63 Hz
        let note = Units.hzToNote(261.63)
        XCTAssertEqual(note, "C4", "~261.63 Hz should be C4")
    }

    func testNoteToHzA4() {
        let hz = Units.noteToHz("A4")
        XCTAssertEqual(hz, 440.0, accuracy: 1e-2, "A4 should be 440 Hz")
    }

    func testNoteToHzC4() {
        let hz = Units.noteToHz("C4")
        XCTAssertEqual(hz, 261.63, accuracy: 0.1, "C4 should be ~261.63 Hz")
    }

    func testNoteToHzSharp() {
        let hz = Units.noteToHz("C#4")
        let expected = Units.midiToHz(61.0)
        XCTAssertEqual(hz, expected, accuracy: 1e-2, "C#4 should match MIDI 61")
    }

    func testNoteToHzFlat() {
        let hz = Units.noteToHz("Db4")
        let expected = Units.midiToHz(61.0) // Db4 = C#4 = MIDI 61
        XCTAssertEqual(hz, expected, accuracy: 1e-2, "Db4 should match C#4")
    }

    // MARK: - MIDI <-> Note round-trip

    func testMidiToNote() {
        XCTAssertEqual(Units.midiToNote(69.0), "A4")
        XCTAssertEqual(Units.midiToNote(60.0), "C4")
        XCTAssertEqual(Units.midiToNote(0.0), "C-1")
        XCTAssertEqual(Units.midiToNote(127.0), "G9")
    }

    func testNoteToMidi() {
        XCTAssertEqual(Units.noteToMidi("A4"), 69.0, accuracy: 1e-5)
        XCTAssertEqual(Units.noteToMidi("C4"), 60.0, accuracy: 1e-5)
        XCTAssertEqual(Units.noteToMidi("C-1"), 0.0, accuracy: 1e-5)
        XCTAssertEqual(Units.noteToMidi("G9"), 127.0, accuracy: 1e-5)
    }

    func testMidiNoteRoundTrip() {
        for midi in stride(from: Float(0), through: 127, by: 1) {
            let note = Units.midiToNote(midi)
            let recovered = Units.noteToMidi(note)
            XCTAssertEqual(recovered, midi, accuracy: 1e-5,
                           "MIDI->Note->MIDI round-trip failed for MIDI \(midi), note=\(note)")
        }
    }

    // MARK: - Hz -> Octave

    func testHzToOctA4() {
        // A4 = 440 Hz, tuning = 440
        // log2(440 / (440/16)) = log2(16) = 4.0
        let oct = Units.hzToOct(440.0)
        XCTAssertEqual(oct, 4.0, accuracy: 1e-5, "A4 should be octave ~4.0")
    }

    func testHzToOctA5() {
        // A5 = 880 Hz
        // log2(880 / 27.5) = log2(32) = 5.0
        let oct = Units.hzToOct(880.0)
        XCTAssertEqual(oct, 5.0, accuracy: 1e-5, "A5 should be octave ~5.0")
    }

    func testHzToOctZero() {
        let oct = Units.hzToOct(0.0)
        XCTAssertTrue(oct.isNaN, "0 Hz should produce NaN octave")
    }

    func testHzToOctVectorized() {
        let hzArray: [Float] = [440.0, 880.0, 220.0]
        let octArray = Units.hzToOct(hzArray)
        XCTAssertEqual(octArray.count, 3)
        XCTAssertEqual(octArray[0], 4.0, accuracy: 1e-5)
        XCTAssertEqual(octArray[1], 5.0, accuracy: 1e-5)
        XCTAssertEqual(octArray[2], 3.0, accuracy: 1e-5)
    }

    // MARK: - Time <-> Frame round-trip

    func testTimesToFrames() {
        let times: [Float] = [0.0, 0.5, 1.0, 2.0]
        let sr = 22050
        let hopLength = 512
        let frames = Units.timesToFrames(times, sr: sr, hopLength: hopLength)
        // frame = floor(time * sr / hop)
        XCTAssertEqual(frames[0], 0)
        XCTAssertEqual(frames[1], Int(floorf(0.5 * 22050.0 / 512.0)))
        XCTAssertEqual(frames[2], Int(floorf(1.0 * 22050.0 / 512.0)))
        XCTAssertEqual(frames[3], Int(floorf(2.0 * 22050.0 / 512.0)))
    }

    func testFramesToTime() {
        let frames = [0, 21, 43, 86]
        let sr = 22050
        let hopLength = 512
        let times = Units.framesToTime(frames, sr: sr, hopLength: hopLength)
        for i in 0..<frames.count {
            let expected = Float(frames[i]) * Float(hopLength) / Float(sr)
            XCTAssertEqual(times[i], expected, accuracy: 1e-6,
                           "framesToTime mismatch at index \(i)")
        }
    }

    func testTimeFrameRoundTrip() {
        let sr = 22050
        let hopLength = 512
        // Start from frames, go to time, then back to frames
        let originalFrames = [0, 10, 50, 100, 200]
        let times = Units.framesToTime(originalFrames, sr: sr, hopLength: hopLength)
        let recoveredFrames = Units.timesToFrames(times, sr: sr, hopLength: hopLength)
        for i in 0..<originalFrames.count {
            XCTAssertEqual(recoveredFrames[i], originalFrames[i],
                           "Time->Frame round-trip failed at index \(i)")
        }
    }

    // MARK: - Time <-> Sample round-trip

    func testTimesToSamples() {
        let times: [Float] = [0.0, 0.5, 1.0]
        let sr = 22050
        let samples = Units.timesToSamples(times, sr: sr)
        XCTAssertEqual(samples[0], 0)
        XCTAssertEqual(samples[1], 11025)
        XCTAssertEqual(samples[2], 22050)
    }

    func testSamplesToTime() {
        let samples = [0, 11025, 22050]
        let sr = 22050
        let times = Units.samplesToTime(samples, sr: sr)
        XCTAssertEqual(times[0], 0.0, accuracy: 1e-6)
        XCTAssertEqual(times[1], 0.5, accuracy: 1e-6)
        XCTAssertEqual(times[2], 1.0, accuracy: 1e-6)
    }

    func testTimeSampleRoundTrip() {
        let sr = 44100
        let originalSamples = [0, 22050, 44100, 88200]
        let times = Units.samplesToTime(originalSamples, sr: sr)
        let recoveredSamples = Units.timesToSamples(times, sr: sr)
        for i in 0..<originalSamples.count {
            XCTAssertEqual(recoveredSamples[i], originalSamples[i],
                           "Time->Sample round-trip failed at index \(i)")
        }
    }

    // MARK: - Frame <-> Sample round-trip

    func testFramesToSamples() {
        let frames = [0, 1, 10, 100]
        let hopLength = 512
        let samples = Units.framesToSamples(frames, hopLength: hopLength)
        XCTAssertEqual(samples[0], 0)
        XCTAssertEqual(samples[1], 512)
        XCTAssertEqual(samples[2], 5120)
        XCTAssertEqual(samples[3], 51200)
    }

    func testSamplesToFrames() {
        let samples = [0, 512, 5120, 51200]
        let hopLength = 512
        let frames = Units.samplesToFrames(samples, hopLength: hopLength)
        XCTAssertEqual(frames[0], 0)
        XCTAssertEqual(frames[1], 1)
        XCTAssertEqual(frames[2], 10)
        XCTAssertEqual(frames[3], 100)
    }

    func testFrameSampleRoundTrip() {
        let hopLength = 512
        let originalFrames = [0, 5, 10, 50, 100]
        let samples = Units.framesToSamples(originalFrames, hopLength: hopLength)
        let recoveredFrames = Units.samplesToFrames(samples, hopLength: hopLength)
        for i in 0..<originalFrames.count {
            XCTAssertEqual(recoveredFrames[i], originalFrames[i],
                           "Frame->Sample round-trip failed at index \(i)")
        }
    }

    // MARK: - FFT Frequencies

    func testFFTFrequencies() {
        let freqs = Units.fftFrequencies(sr: 22050, nFFT: 2048)
        XCTAssertEqual(freqs.count, 1025, "nFFT/2 + 1 bins expected")
        XCTAssertEqual(freqs[0], 0.0, accuracy: 1e-6, "First bin should be 0 Hz")
        XCTAssertEqual(freqs.last!, 11025.0, accuracy: 1e-2, "Last bin should be sr/2")
        // Check monotonically increasing
        for i in 1..<freqs.count {
            XCTAssertGreaterThan(freqs[i], freqs[i - 1],
                                 "FFT frequencies should be monotonically increasing")
        }
    }

    func testFFTFrequenciesStep() {
        let sr = 44100
        let nFFT = 4096
        let freqs = Units.fftFrequencies(sr: sr, nFFT: nFFT)
        let expectedStep = Float(sr) / Float(nFFT)
        XCTAssertEqual(freqs[1] - freqs[0], expectedStep, accuracy: 1e-5,
                       "Frequency step should be sr/nFFT")
    }

    // MARK: - Mel Frequencies

    func testMelFrequencies() {
        let freqs = Units.melFrequencies(nMels: 128, fMin: 0.0, fMax: 11025.0)
        XCTAssertEqual(freqs.count, 128)
        // Should be sorted
        for i in 1..<freqs.count {
            XCTAssertGreaterThanOrEqual(freqs[i], freqs[i - 1],
                                        "Mel frequencies should be sorted")
        }
        // First >= fMin
        XCTAssertGreaterThanOrEqual(freqs[0], 0.0, "First mel freq should be >= fMin")
        // Last <= fMax (with tolerance)
        XCTAssertLessThanOrEqual(freqs.last!, 11025.0 + 1.0, "Last mel freq should be <= fMax")
    }

    func testMelFrequenciesSmall() {
        let freqs = Units.melFrequencies(nMels: 10, fMin: 200.0, fMax: 8000.0)
        XCTAssertEqual(freqs.count, 10)
        XCTAssertEqual(freqs[0], 200.0, accuracy: 1.0, "First should be near fMin")
        XCTAssertEqual(freqs.last!, 8000.0, accuracy: 1.0, "Last should be near fMax")
    }

    // MARK: - Edge cases

    func testHzToMidiZero() {
        let midi = Units.hzToMidi(0.0)
        XCTAssertTrue(midi.isNaN, "0 Hz should produce NaN MIDI")
    }

    func testHzToMidiNegative() {
        let midi = Units.hzToMidi(-100.0)
        XCTAssertTrue(midi.isNaN, "Negative Hz should produce NaN MIDI")
    }

    func testHzToNoteZero() {
        let note = Units.hzToNote(0.0)
        XCTAssertEqual(note, "nan", "0 Hz should produce 'nan' note name")
    }

    func testNoteToMidiInvalid() {
        let midi = Units.noteToMidi("")
        XCTAssertTrue(midi.isNaN, "Empty string should produce NaN MIDI")
        let midi2 = Units.noteToMidi("XYZ")
        XCTAssertTrue(midi2.isNaN, "Invalid note should produce NaN MIDI")
    }

    func testNegativeTime() {
        // Negative time should produce negative frame index
        let frames = Units.timesToFrames([-1.0], sr: 22050, hopLength: 512)
        XCTAssertLessThan(frames[0], 0, "Negative time should produce negative frame")
    }
}
