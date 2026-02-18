import Foundation
import Metal
import MetalPerformanceShaders

/// GPU matrix multiplication using Metal Performance Shaders (MPS).
///
/// Wraps `MPSMatrixMultiplication` to compute `C = A @ B` for row-major
/// float32 matrices.  Used primarily for mel filterbank application:
///
///     melFilterbank [nMels, nFreqs] @ powerSpectrogram [nFreqs, nFrames] = [nMels, nFrames]
///
public final class MetalMatmul {

    /// Compute C = A @ B using MPS on the GPU.
    ///
    /// A: [M, K], B: [K, N], C: [M, N] -- all row-major float32.
    ///
    /// Returns `nil` if Metal is unavailable or buffer allocation fails.
    public static func multiply(
        a: [Float], aRows: Int, aCols: Int,
        b: [Float], bRows: Int, bCols: Int
    ) -> [Float]? {
        guard let backend = MetalBackend.shared else { return nil }
        let M = aRows
        let K = aCols
        let N = bCols
        precondition(bRows == K, "Inner dimensions must match: aCols=\(K) != bRows=\(bRows)")
        precondition(a.count == M * K, "a.count=\(a.count) != \(M)*\(K)")
        precondition(b.count == K * N, "b.count=\(b.count) != \(K)*\(N)")

        let device = backend.device

        // Create Metal buffers with shared storage (unified memory on Apple Silicon).
        guard let aBuf = a.withUnsafeBufferPointer({ ptr in
            device.makeBuffer(
                bytes: ptr.baseAddress!,
                length: M * K * MemoryLayout<Float>.stride,
                options: .storageModeShared
            )
        }),
        let bBuf = b.withUnsafeBufferPointer({ ptr in
            device.makeBuffer(
                bytes: ptr.baseAddress!,
                length: K * N * MemoryLayout<Float>.stride,
                options: .storageModeShared
            )
        }),
        let cBuf = device.makeBuffer(
            length: M * N * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )
        else { return nil }

        // MPS matrix descriptors — row bytes = columns * stride (row-major, no padding).
        let aDesc = MPSMatrixDescriptor(
            rows: M, columns: K,
            rowBytes: K * MemoryLayout<Float>.stride,
            dataType: .float32
        )
        let bDesc = MPSMatrixDescriptor(
            rows: K, columns: N,
            rowBytes: N * MemoryLayout<Float>.stride,
            dataType: .float32
        )
        let cDesc = MPSMatrixDescriptor(
            rows: M, columns: N,
            rowBytes: N * MemoryLayout<Float>.stride,
            dataType: .float32
        )

        let matA = MPSMatrix(buffer: aBuf, descriptor: aDesc)
        let matB = MPSMatrix(buffer: bBuf, descriptor: bDesc)
        let matC = MPSMatrix(buffer: cBuf, descriptor: cDesc)

        // Create kernel: C = alpha * A @ B + beta * C  (alpha=1, beta=0)
        let matmul = MPSMatrixMultiplication(
            device: device,
            transposeLeft: false,
            transposeRight: false,
            resultRows: M,
            resultColumns: N,
            interiorColumns: K,
            alpha: 1.0,
            beta: 0.0
        )

        guard let cmdBuf = backend.makeCommandBuffer() else { return nil }
        matmul.encode(commandBuffer: cmdBuf, leftMatrix: matA, rightMatrix: matB, resultMatrix: matC)
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        // Read back result from shared memory.
        let resultPtr = cBuf.contents().bindMemory(to: Float.self, capacity: M * N)
        return Array(UnsafeBufferPointer(start: resultPtr, count: M * N))
    }

    /// Encode a matrix multiply into an existing command buffer.
    ///
    /// This avoids extra copies when matrices are already in Metal buffers,
    /// enabling pipeline fusion (e.g. STFT -> power -> matmul -> dB on GPU).
    public static func encode(
        commandBuffer: MTLCommandBuffer,
        leftMatrix: MPSMatrix,
        rightMatrix: MPSMatrix,
        resultMatrix: MPSMatrix,
        device: MTLDevice
    ) {
        let matmul = MPSMatrixMultiplication(
            device: device,
            transposeLeft: false,
            transposeRight: false,
            resultRows: leftMatrix.rows,
            resultColumns: rightMatrix.columns,
            interiorColumns: leftMatrix.columns,
            alpha: 1.0,
            beta: 0.0
        )
        matmul.encode(
            commandBuffer: commandBuffer,
            leftMatrix: leftMatrix,
            rightMatrix: rightMatrix,
            resultMatrix: resultMatrix
        )
    }
}
