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
    /// Magnitude spectrogram with shape [nFreqs, nFrames], row-major (C-order).
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

        // --- 5. Allocate output buffer (row-major: [nFreqs, nFrames]) ---
        // We compute into a column-major temp buffer (fast per-frame writes),
        // then transpose to row-major at the end for NumPy/C-order compatibility.
        let totalElements = nFreqs * nFrames
        let tempPtr = UnsafeMutablePointer<Float>.allocate(capacity: totalElements)
        tempPtr.initialize(repeating: 0, count: totalElements)

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
                    tempPtr[colOffset + 0] = abs(dcVal)

                    // Nyquist bin (index nFreqs - 1 = halfN): magnitude = |imagp[0]|
                    // After scaling, imagp[0] holds the Nyquist real component / 2...
                    // Actually, vDSP packs DC in realp[0] and Nyquist in imagp[0].
                    // Both are purely real. After our 0.5 scaling, the values are correct.
                    let nyquistVal = imagPart[0]
                    tempPtr[colOffset + nFreqs - 1] = abs(nyquistVal)

                    // Bins 1..<halfN: magnitude = sqrt(real^2 + imag^2)
                    // Use vDSP_zvabs but we need to handle offset pointers for bins 1..<halfN
                    // vDSP_zvabs computes magnitude of split complex
                    var innerSplit = DSPSplitComplex(
                        realp: realPart + 1,
                        imagp: imagPart + 1
                    )
                    vDSP_zvabs(&innerSplit, 1, tempPtr + colOffset + 1, 1, vDSP_Length(halfN - 1))
                }
            }
        }

        // --- 8. Transpose from column-major [nFreqs x nFrames] to row-major ---
        // tempPtr layout: column-major [nFreqs, nFrames] = tempPtr[frame * nFreqs + freq]
        // outPtr layout: row-major [nFreqs, nFrames]     = outPtr[freq * nFrames + frame]
        let outPtr = UnsafeMutablePointer<Float>.allocate(capacity: totalElements)
        // vDSP_mtrans transposes an M x N matrix (row-major) to N x M (row-major).
        // Treating tempPtr as nFrames rows x nFreqs cols (row-major reading of column-major data),
        // we transpose to get nFreqs rows x nFrames cols.
        vDSP_mtrans(tempPtr, 1, outPtr, 1, vDSP_Length(nFreqs), vDSP_Length(nFrames))
        tempPtr.deallocate()

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

    /// Compute complex STFT returning interleaved real/imag Signal with dtype `.complex64`.
    /// Shape is [nFreqs, nFrames] (logical complex elements); raw float storage is 2x that.
    public static func computeComplex(
        signal: Signal,
        nFFT: Int = 2048,
        hopLength: Int? = nil,
        winLength: Int? = nil,
        center: Bool = true
    ) -> Signal {
        let hop = hopLength ?? nFFT / 4
        let win = winLength ?? nFFT
        let nFreqs = nFFT / 2 + 1

        // --- 1. Optionally pad the signal ---
        let padded: [Float]
        if center {
            let padAmount = nFFT / 2
            padded = [Float](repeating: 0, count: padAmount)
                   + signal.withUnsafeBufferPointer { Array($0) }
                   + [Float](repeating: 0, count: padAmount)
        } else {
            padded = signal.withUnsafeBufferPointer { Array($0) }
        }

        let paddedLength = padded.count

        // --- 2. Compute number of frames ---
        guard paddedLength >= nFFT else {
            return Signal(complexData: [], shape: [nFreqs, 0], sampleRate: signal.sampleRate)
        }
        let nFrames = 1 + (paddedLength - nFFT) / hop

        // --- 3. Prepare window ---
        let window = Windows.hann(length: win, periodic: true)
        let fullWindow: [Float]
        if win < nFFT {
            let padBefore = (nFFT - win) / 2
            let padAfter = nFFT - win - padBefore
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

        // --- 5. Allocate complex output buffer ---
        // Column-major temp: for each frame, nFreqs interleaved pairs (2 * nFreqs floats).
        // Layout: tempPtr[(frame * nFreqs + freq) * 2 + 0] = real
        //         tempPtr[(frame * nFreqs + freq) * 2 + 1] = imag
        let totalComplex = nFreqs * nFrames
        let totalFloats = totalComplex * 2
        let tempPtr = UnsafeMutablePointer<Float>.allocate(capacity: totalFloats)
        tempPtr.initialize(repeating: 0, count: totalFloats)

        let halfN = nFFT / 2
        let realPart = UnsafeMutablePointer<Float>.allocate(capacity: halfN)
        let imagPart = UnsafeMutablePointer<Float>.allocate(capacity: halfN)
        defer {
            realPart.deallocate()
            imagPart.deallocate()
        }

        let windowedFrame = UnsafeMutablePointer<Float>.allocate(capacity: nFFT)
        defer { windowedFrame.deallocate() }

        // --- 6. Process each frame ---
        padded.withUnsafeBufferPointer { paddedBuf in
            fullWindow.withUnsafeBufferPointer { winBuf in
                for frame in 0..<nFrames {
                    let start = frame * hop

                    // Apply window
                    vDSP_vmul(
                        paddedBuf.baseAddress! + start, 1,
                        winBuf.baseAddress!, 1,
                        windowedFrame, 1,
                        vDSP_Length(nFFT)
                    )

                    // Pack into split complex for vDSP_fft_zrip
                    var splitComplex = DSPSplitComplex(realp: realPart, imagp: imagPart)
                    windowedFrame.withMemoryRebound(to: DSPComplex.self, capacity: halfN) { complexPtr in
                        vDSP_ctoz(complexPtr, 2, &splitComplex, 1, vDSP_Length(halfN))
                    }

                    // Forward FFT (in-place)
                    vDSP_fft_zrip(fftSetup, &splitComplex, 1, log2n, FFTDirection(kFFTDirection_Forward))

                    // Scale by 0.5 to normalize (vDSP_fft_zrip has factor-of-2)
                    var scale: Float = 0.5
                    vDSP_vsmul(realPart, 1, &scale, realPart, 1, vDSP_Length(halfN))
                    vDSP_vsmul(imagPart, 1, &scale, imagPart, 1, vDSP_Length(halfN))

                    // --- 7. Store interleaved complex output (column-major) ---
                    // Column offset in complex elements
                    let colBase = frame * nFreqs

                    // DC bin (index 0): real = realp[0], imag = 0
                    tempPtr[(colBase + 0) * 2 + 0] = realPart[0]
                    tempPtr[(colBase + 0) * 2 + 1] = 0.0

                    // Bins 1..<halfN: real = realp[k], imag = imagp[k]
                    for k in 1..<halfN {
                        tempPtr[(colBase + k) * 2 + 0] = realPart[k]
                        tempPtr[(colBase + k) * 2 + 1] = imagPart[k]
                    }

                    // Nyquist bin (index nFreqs-1 = halfN): real = imagp[0], imag = 0
                    tempPtr[(colBase + nFreqs - 1) * 2 + 0] = imagPart[0]
                    tempPtr[(colBase + nFreqs - 1) * 2 + 1] = 0.0
                }
            }
        }

        // --- 8. Transpose from column-major to row-major ---
        // tempPtr is column-major: element (freq, frame) at tempPtr[(frame * nFreqs + freq) * 2]
        // We need row-major: element (freq, frame) at outPtr[(freq * nFrames + frame) * 2]
        //
        // Treat each complex element as a pair of floats. We transpose the [nFrames, nFreqs]
        // matrix of pairs to [nFreqs, nFrames] by transposing real and imaginary planes separately.
        let outPtr = UnsafeMutablePointer<Float>.allocate(capacity: totalFloats)

        // Extract real plane, transpose, store back
        // temp real plane: tempPtr[i*2] for i in 0..<totalComplex
        // We need a temporary plane buffer for the transpose
        let realPlane = UnsafeMutablePointer<Float>.allocate(capacity: totalComplex)
        let imagPlane = UnsafeMutablePointer<Float>.allocate(capacity: totalComplex)
        let realPlaneOut = UnsafeMutablePointer<Float>.allocate(capacity: totalComplex)
        let imagPlaneOut = UnsafeMutablePointer<Float>.allocate(capacity: totalComplex)
        defer {
            realPlane.deallocate()
            imagPlane.deallocate()
            realPlaneOut.deallocate()
            imagPlaneOut.deallocate()
        }

        // De-interleave: extract real and imag planes from interleaved temp buffer
        // vDSP_ctoz splits interleaved [r0,i0,r1,i1,...] into split complex {realp, imagp}
        var splitPlane = DSPSplitComplex(realp: realPlane, imagp: imagPlane)
        tempPtr.withMemoryRebound(to: DSPComplex.self, capacity: totalComplex) { complexPtr in
            vDSP_ctoz(complexPtr, 2, &splitPlane, 1, vDSP_Length(totalComplex))
        }

        // Transpose each plane: [nFrames rows x nFreqs cols] -> [nFreqs rows x nFrames cols]
        vDSP_mtrans(realPlane, 1, realPlaneOut, 1, vDSP_Length(nFreqs), vDSP_Length(nFrames))
        vDSP_mtrans(imagPlane, 1, imagPlaneOut, 1, vDSP_Length(nFreqs), vDSP_Length(nFrames))

        // Re-interleave into outPtr using vDSP_ztoc
        var splitOut = DSPSplitComplex(realp: realPlaneOut, imagp: imagPlaneOut)
        outPtr.withMemoryRebound(to: DSPComplex.self, capacity: totalComplex) { complexPtr in
            vDSP_ztoc(&splitOut, 1, complexPtr, 2, vDSP_Length(totalComplex))
        }

        tempPtr.deallocate()

        let outBuffer = UnsafeMutableBufferPointer(start: outPtr, count: totalFloats)
        return Signal(taking: outBuffer, shape: [nFreqs, nFrames],
                      sampleRate: signal.sampleRate, dtype: .complex64)
    }
}
