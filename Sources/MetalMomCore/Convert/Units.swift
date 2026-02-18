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

    // MARK: - Vectorised conversions

    /// Convert an array of Hz values to mel scale.
    public static func hzToMel(_ hz: [Float]) -> [Float] {
        hz.map { hzToMel($0) }
    }

    /// Convert an array of mel values to Hz.
    public static func melToHz(_ mel: [Float]) -> [Float] {
        mel.map { melToHz($0) }
    }
}
