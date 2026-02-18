import Foundation
import Accelerate

// MARK: - Input / Output Types

/// Input parameters for the STFT computation.
public struct STFTInput {
    public let signal: Signal
    public let nFFT: Int
    public let hopLength: Int
    public let winLength: Int
    public let center: Bool

    public init(signal: Signal, nFFT: Int, hopLength: Int, winLength: Int, center: Bool) {
        self.signal = signal
        self.nFFT = nFFT
        self.hopLength = hopLength
        self.winLength = winLength
        self.center = center
    }
}

/// Output of the STFT computation.
public struct STFTOutput {
    /// Magnitude spectrogram with shape [nFreqs, nFrames], column-major.
    public let magnitude: Signal
}

// MARK: - STFT ComputeOperation

/// Short-Time Fourier Transform using Accelerate/vDSP.
public struct STFT: ComputeOperation {
    public typealias Input = STFTInput
    public typealias Output = STFTOutput

    /// CPU-only until Phase 10.
    public static var dispatchThreshold: Int { Int.max }

    public init() {}

    public func executeCPU(_ input: STFTInput) -> STFTOutput {
        let nFFT = input.nFFT
        let hopLength = input.hopLength
        let winLength = input.winLength
        let nFreqs = nFFT / 2 + 1

        // --- 1. Optionally pad the signal ---
        let padded: [Float]
        if input.center {
            let padAmount = nFFT / 2
            padded = [Float](repeating: 0, count: padAmount)
                   + input.signal.withUnsafeBufferPointer { Array($0) }
                   + [Float](repeating: 0, count: padAmount)
        } else {
            padded = input.signal.withUnsafeBufferPointer { Array($0) }
        }

        let paddedLength = padded.count

        // --- 2. Compute number of frames ---
        guard paddedLength >= nFFT else {
            // Signal too short for even one frame
            let out = Signal(data: [], shape: [nFreqs, 0], sampleRate: input.signal.sampleRate)
            return STFTOutput(magnitude: out)
        }
        let nFrames = 1 + (paddedLength - nFFT) / hopLength

        // --- 3. Prepare window ---
        let window = Windows.hann(length: winLength, periodic: true)
        // If winLength < nFFT, we zero-pad the window to nFFT (center it).
        // If winLength == nFFT, no padding needed.
        let fullWindow: [Float]
        if winLength < nFFT {
            let padBefore = (nFFT - winLength) / 2
            let padAfter = nFFT - winLength - padBefore
            fullWindow = [Float](repeating: 0, count: padBefore)
                       + window
                       + [Float](repeating: 0, count: padAfter)
        } else {
            fullWindow = window
        }

        // --- 4. Set up vDSP FFT ---
        let log2n = vDSP_Length(log2(Double(nFFT)))
        guard let fftSetup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else {
            fatalError("Failed to create FFT setup for nFFT=\(nFFT)")
        }
        defer { vDSP_destroy_fftsetup(fftSetup) }

        // --- 5. Allocate output buffer (column-major: [nFreqs, nFrames]) ---
        let totalElements = nFreqs * nFrames
        let outPtr = UnsafeMutablePointer<Float>.allocate(capacity: totalElements)
        outPtr.initialize(repeating: 0, count: totalElements)

        // Temporary buffer for each frame's FFT
        // vDSP_fft_zrip works on nFFT/2 complex pairs
        let halfN = nFFT / 2
        let realPart = UnsafeMutablePointer<Float>.allocate(capacity: halfN)
        let imagPart = UnsafeMutablePointer<Float>.allocate(capacity: halfN)
        defer {
            realPart.deallocate()
            imagPart.deallocate()
        }

        // Temporary buffer for the windowed frame
        let windowedFrame = UnsafeMutablePointer<Float>.allocate(capacity: nFFT)
        defer { windowedFrame.deallocate() }

        // --- 6. Process each frame ---
        padded.withUnsafeBufferPointer { paddedBuf in
            fullWindow.withUnsafeBufferPointer { winBuf in
                for frame in 0..<nFrames {
                    let start = frame * hopLength

                    // Apply window: windowed = signal[start..<start+nFFT] * window
                    vDSP_vmul(
                        paddedBuf.baseAddress! + start, 1,
                        winBuf.baseAddress!, 1,
                        windowedFrame, 1,
                        vDSP_Length(nFFT)
                    )

                    // Pack real data into split complex format for vDSP_fft_zrip.
                    // vDSP_fft_zrip expects even-indexed samples in realp and
                    // odd-indexed samples in imagp.
                    var splitComplex = DSPSplitComplex(realp: realPart, imagp: imagPart)
                    windowedFrame.withMemoryRebound(to: DSPComplex.self, capacity: halfN) { complexPtr in
                        vDSP_ctoz(complexPtr, 2, &splitComplex, 1, vDSP_Length(halfN))
                    }

                    // Forward FFT (in-place)
                    vDSP_fft_zrip(fftSetup, &splitComplex, 1, log2n, FFTDirection(kFFTDirection_Forward))

                    // --- 7. Extract magnitudes ---
                    // After vDSP_fft_zrip:
                    //   splitComplex.realp[0] = DC component (real, imag is 0)
                    //   splitComplex.imagp[0] = Nyquist component (real, imag is 0)
                    //   splitComplex.realp[k], imagp[k] for k=1..halfN-1 = complex bin k
                    //
                    // vDSP_fft_zrip includes a factor of 2 compared to the mathematical DFT.
                    // We apply a scale of 1/2 to normalize before computing magnitudes.

                    // Scale by 0.5 to normalize
                    var scale: Float = 0.5
                    vDSP_vsmul(realPart, 1, &scale, realPart, 1, vDSP_Length(halfN))
                    vDSP_vsmul(imagPart, 1, &scale, imagPart, 1, vDSP_Length(halfN))

                    let colOffset = frame * nFreqs

                    // DC bin (index 0): magnitude = |realp[0]| (imagp[0] holds Nyquist, not DC imag)
                    let dcVal = realPart[0]
                    outPtr[colOffset + 0] = abs(dcVal)

                    // Nyquist bin (index nFreqs - 1 = halfN): magnitude = |imagp[0]|
                    // After scaling, imagp[0] holds the Nyquist real component / 2...
                    // Actually, vDSP packs DC in realp[0] and Nyquist in imagp[0].
                    // Both are purely real. After our 0.5 scaling, the values are correct.
                    let nyquistVal = imagPart[0]
                    outPtr[colOffset + nFreqs - 1] = abs(nyquistVal)

                    // Bins 1..<halfN: magnitude = sqrt(real^2 + imag^2)
                    // Use vDSP_zvabs but we need to handle offset pointers for bins 1..<halfN
                    // vDSP_zvabs computes magnitude of split complex
                    var innerSplit = DSPSplitComplex(
                        realp: realPart + 1,
                        imagp: imagPart + 1
                    )
                    vDSP_zvabs(&innerSplit, 1, outPtr + colOffset + 1, 1, vDSP_Length(halfN - 1))
                }
            }
        }

        let outBuffer = UnsafeMutableBufferPointer(start: outPtr, count: totalElements)
        let output = Signal(taking: outBuffer, shape: [nFreqs, nFrames],
                            sampleRate: input.signal.sampleRate)
        return STFTOutput(magnitude: output)
    }
}

// MARK: - Convenience API

extension STFT {
    /// Convenience method to compute STFT magnitude spectrogram.
    public static func compute(
        signal: Signal,
        nFFT: Int = 2048,
        hopLength: Int? = nil,
        winLength: Int? = nil,
        center: Bool = true
    ) -> Signal {
        let hop = hopLength ?? nFFT / 4
        let win = winLength ?? nFFT
        let input = STFTInput(signal: signal, nFFT: nFFT, hopLength: hop, winLength: win, center: center)
        let op = STFT()
        let dispatcher = SmartDispatcher()
        return dispatcher.dispatch(op, input: input, dataSize: signal.count).magnitude
    }
}
