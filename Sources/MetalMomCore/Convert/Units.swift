import Foundation

/// Hz / mel scale conversion utilities.
///
/// Uses the Slaney (O'Shaughnessy) formula, matching librosa's default
/// (`htk=False`):
///   - Below 1000 Hz: linear mapping (200 mels per 1000 Hz)
///   - Above 1000 Hz: logarithmic mapping
public enum Units {

    // MARK: - Scalar conversions

    /// Convert a frequency in Hz to the mel scale (Slaney formula).
    ///
    /// - Parameter hz: Frequency in Hz.
    /// - Returns: Corresponding value on the mel scale.
    public static func hzToMel(_ hz: Float) -> Float {
        let f_sp: Float = 200.0 / 3.0  // ~66.667 Hz per mel below 1000 Hz
        var mel = hz / f_sp

        let minLogHz: Float = 1000.0
        let minLogMel = minLogHz / f_sp  // 15.0
        let logstep = logf(6.4) / 27.0  // step size for log region

        if hz >= minLogHz {
            mel = minLogMel + logf(hz / minLogHz) / logstep
        }
        return mel
    }

    /// Convert a mel-scale value back to Hz (inverse of `hzToMel`).
    ///
    /// - Parameter mel: Value on the mel scale.
    /// - Returns: Corresponding frequency in Hz.
    public static func melToHz(_ mel: Float) -> Float {
        let f_sp: Float = 200.0 / 3.0
        let minLogHz: Float = 1000.0
        let minLogMel = minLogHz / f_sp  // 15.0
        let logstep = logf(6.4) / 27.0

        if mel < minLogMel {
            return mel * f_sp
        } else {
            return minLogHz * expf((mel - minLogMel) * logstep)
        }
    }

    // MARK: - Double-precision scalar conversions

    /// Convert a frequency in Hz to the mel scale using Double precision.
    ///
    /// Matches librosa's float64 mel computation for filterbank generation.
    /// - Parameter hz: Frequency in Hz.
    /// - Returns: Corresponding value on the mel scale.
    public static func hzToMelD(_ hz: Double) -> Double {
        let f_sp: Double = 200.0 / 3.0
        var mel = hz / f_sp

        let minLogHz: Double = 1000.0
        let minLogMel = minLogHz / f_sp  // 15.0
        let logstep = log(6.4) / 27.0

        if hz >= minLogHz {
            mel = minLogMel + log(hz / minLogHz) / logstep
        }
        return mel
    }

    /// Convert a mel-scale value back to Hz using Double precision.
    ///
    /// Matches librosa's float64 mel computation for filterbank generation.
    /// - Parameter mel: Value on the mel scale.
    /// - Returns: Corresponding frequency in Hz.
    public static func melToHzD(_ mel: Double) -> Double {
        let f_sp: Double = 200.0 / 3.0
        let minLogHz: Double = 1000.0
        let minLogMel = minLogHz / f_sp  // 15.0
        let logstep = log(6.4) / 27.0

        if mel < minLogMel {
            return mel * f_sp
        } else {
            return minLogHz * exp((mel - minLogMel) * logstep)
        }
    }

    // MARK: - Vectorised conversions

    /// Convert an array of Hz values to mel scale.
    public static func hzToMel(_ hz: [Float]) -> [Float] {
        hz.map { hzToMel($0) }
    }

    /// Convert an array of mel values to Hz.
    public static func melToHz(_ mel: [Float]) -> [Float] {
        mel.map { melToHz($0) }
    }

    // MARK: - Hz / MIDI conversions

    /// Convert frequency in Hz to MIDI note number.
    ///
    /// A4 = 440 Hz maps to MIDI 69. Formula: `12 * log2(hz / 440) + 69`.
    /// Returns `Float.nan` for non-positive Hz values.
    public static func hzToMidi(_ hz: Float) -> Float {
        guard hz > 0 else { return .nan }
        return 12.0 * log2f(hz / 440.0) + 69.0
    }

    /// Convert MIDI note number to frequency in Hz.
    ///
    /// MIDI 69 maps to 440 Hz. Formula: `440 * 2^((midi - 69) / 12)`.
    public static func midiToHz(_ midi: Float) -> Float {
        return 440.0 * powf(2.0, (midi - 69.0) / 12.0)
    }

    /// Convert an array of Hz values to MIDI note numbers.
    public static func hzToMidi(_ hz: [Float]) -> [Float] {
        hz.map { hzToMidi($0) }
    }

    /// Convert an array of MIDI note numbers to Hz.
    public static func midiToHz(_ midi: [Float]) -> [Float] {
        midi.map { midiToHz($0) }
    }

    // MARK: - Hz / Note name conversions

    /// Note names indexed by pitch class (0 = C, 11 = B).
    private static let noteNames = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    /// Convert a MIDI note number to a note name string (e.g. 69 -> "A4").
    ///
    /// Rounds to the nearest integer MIDI number. MIDI 0 = C-1.
    public static func midiToNote(_ midi: Float) -> String {
        let rounded = Int(roundf(midi))
        let pitchClass = ((rounded % 12) + 12) % 12  // handle negatives
        let octave = (rounded / 12) - 1
        return "\(noteNames[pitchClass])\(octave)"
    }

    /// Convert a note name string to a MIDI note number (e.g. "A4" -> 69.0).
    ///
    /// Supported formats: "C4", "C#4", "Db4", "C-1", "C#-1".
    /// Returns `Float.nan` for invalid note strings.
    public static func noteToMidi(_ note: String) -> Float {
        guard !note.isEmpty else { return .nan }

        var idx = note.startIndex

        // Parse note letter
        let letter = note[idx]
        idx = note.index(after: idx)

        let baseMap: [Character: Int] = [
            "C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11
        ]
        guard let base = baseMap[Character(letter.uppercased())] else { return .nan }

        // Parse accidental
        var accidental = 0
        if idx < note.endIndex {
            let ch = note[idx]
            if ch == "#" || ch == "\u{266F}" {
                accidental = 1
                idx = note.index(after: idx)
            } else if ch == "b" || ch == "\u{266D}" {
                accidental = -1
                idx = note.index(after: idx)
            }
        }

        // Parse octave (may be negative, e.g. "-1")
        let octaveStr = String(note[idx...])
        guard let octave = Int(octaveStr) else { return .nan }

        let midi = (octave + 1) * 12 + base + accidental
        return Float(midi)
    }

    /// Convert frequency in Hz to a note name string (e.g. 440.0 -> "A4").
    ///
    /// Returns "nan" for non-positive Hz values.
    public static func hzToNote(_ hz: Float) -> String {
        guard hz > 0 else { return "nan" }
        return midiToNote(hzToMidi(hz))
    }

    /// Convert a note name string to frequency in Hz (e.g. "A4" -> 440.0).
    ///
    /// Returns `Float.nan` for invalid note strings.
    public static func noteToHz(_ note: String) -> Float {
        let midi = noteToMidi(note)
        guard !midi.isNaN else { return .nan }
        return midiToHz(midi)
    }

    // MARK: - Hz / Octave conversions

    /// Convert frequency in Hz to octave number.
    ///
    /// Uses `log2(hz / (tuning / 16))` where tuning defaults to A4 = 440 Hz.
    /// A4 (440 Hz) maps to approximately octave 4.0.
    /// Returns `Float.nan` for non-positive Hz values.
    public static func hzToOct(_ hz: Float, tuning: Float = 440.0) -> Float {
        guard hz > 0 else { return .nan }
        return log2f(hz / (tuning / 16.0))
    }

    /// Convert an array of Hz values to octave numbers.
    public static func hzToOct(_ hz: [Float], tuning: Float = 440.0) -> [Float] {
        hz.map { hzToOct($0, tuning: tuning) }
    }

    // MARK: - Time / Frame conversions

    /// Convert time values in seconds to frame indices.
    ///
    /// Formula: `frame = floor(time * sr / hopLength)`
    ///
    /// - Parameters:
    ///   - times: Array of time values in seconds.
    ///   - sr: Sample rate in Hz.
    ///   - hopLength: Hop length in samples.
    ///   - nFFT: FFT size (unused, kept for librosa compat). Default 0.
    /// - Returns: Array of frame indices.
    public static func timesToFrames(_ times: [Float], sr: Int, hopLength: Int, nFFT: Int = 0) -> [Int] {
        let srF = Float(sr)
        let hopF = Float(hopLength)
        return times.map { Int(floorf($0 * srF / hopF)) }
    }

    /// Convert frame indices to time values in seconds.
    ///
    /// Formula: `time = frame * hopLength / sr`
    ///
    /// - Parameters:
    ///   - frames: Array of frame indices.
    ///   - sr: Sample rate in Hz.
    ///   - hopLength: Hop length in samples.
    ///   - nFFT: FFT size (unused, kept for librosa compat). Default 0.
    /// - Returns: Array of time values in seconds.
    public static func framesToTime(_ frames: [Int], sr: Int, hopLength: Int, nFFT: Int = 0) -> [Float] {
        let srF = Float(sr)
        let hopF = Float(hopLength)
        return frames.map { Float($0) * hopF / srF }
    }

    // MARK: - Time / Sample conversions

    /// Convert time values in seconds to sample indices.
    ///
    /// Formula: `sample = floor(time * sr)`
    public static func timesToSamples(_ times: [Float], sr: Int) -> [Int] {
        let srF = Float(sr)
        return times.map { Int(floorf($0 * srF)) }
    }

    /// Convert sample indices to time values in seconds.
    ///
    /// Formula: `time = sample / sr`
    public static func samplesToTime(_ samples: [Int], sr: Int) -> [Float] {
        let srF = Float(sr)
        return samples.map { Float($0) / srF }
    }

    // MARK: - Frame / Sample conversions

    /// Convert frame indices to sample indices.
    ///
    /// Formula: `sample = frame * hopLength`
    public static func framesToSamples(_ frames: [Int], hopLength: Int, nFFT: Int = 0) -> [Int] {
        frames.map { $0 * hopLength }
    }

    /// Convert sample indices to frame indices.
    ///
    /// Formula: `frame = floor(sample / hopLength)`
    public static func samplesToFrames(_ samples: [Int], hopLength: Int, nFFT: Int = 0) -> [Int] {
        samples.map { $0 / hopLength }
    }

    // MARK: - Frequency bin generation

    /// Generate an array of FFT bin center frequencies.
    ///
    /// Returns `[0, sr/nFFT, 2*sr/nFFT, ..., sr/2]` with `nFFT/2 + 1` elements.
    ///
    /// - Parameters:
    ///   - sr: Sample rate in Hz.
    ///   - nFFT: FFT window size.
    /// - Returns: Array of center frequencies for each FFT bin.
    public static func fftFrequencies(sr: Int, nFFT: Int) -> [Float] {
        let nBins = nFFT / 2 + 1
        let step = Float(sr) / Float(nFFT)
        return (0..<nBins).map { Float($0) * step }
    }

    /// Generate an array of mel-spaced frequencies.
    ///
    /// Returns `nMels` frequencies evenly spaced on the mel scale between
    /// `fMin` and `fMax`, then converted back to Hz.
    ///
    /// - Parameters:
    ///   - nMels: Number of mel frequencies to generate.
    ///   - fMin: Minimum frequency in Hz. Default 0.
    ///   - fMax: Maximum frequency in Hz. Default 11025.
    /// - Returns: Array of frequencies in Hz, mel-spaced.
    public static func melFrequencies(nMels: Int, fMin: Float = 0.0, fMax: Float = 11025.0) -> [Float] {
        let melMin = hzToMel(fMin)
        let melMax = hzToMel(fMax)
        let melStep = (melMax - melMin) / Float(nMels - 1)
        return (0..<nMels).map { melToHz(melMin + Float($0) * melStep) }
    }
}
