import Foundation
import Accelerate

/// Non-negative Matrix Factorization (NMF).
///
/// Decomposes a non-negative matrix V into the product of two non-negative
/// matrices W and H such that V ~ W * H.
///
/// - V: [nFeatures, nSamples] (e.g., magnitude spectrogram)
/// - W: [nFeatures, nComponents] (basis / spectral templates)
/// - H: [nComponents, nSamples] (activations / temporal weights)
public enum NMF {

    /// The distance metric / divergence to minimize.
    public enum Objective {
        /// Frobenius norm (Euclidean distance): minimize ||V - WH||_F^2
        case euclidean
        /// Generalized KL divergence: minimize D_KL(V || WH)
        case klDivergence
    }

    /// Result of NMF decomposition.
    public struct Result {
        /// Basis matrix, shape [nFeatures, nComponents].
        public let W: Signal
        /// Activation matrix, shape [nComponents, nSamples].
        public let H: Signal
    }

    // MARK: - Public API

    /// Perform NMF: decompose V ~ W * H using multiplicative update rules.
    ///
    /// - Parameters:
    ///   - V: Non-negative input matrix, shape [nFeatures, nSamples].
    ///   - nComponents: Number of components (rank of the factorization).
    ///   - nIter: Number of multiplicative update iterations. Default 200.
    ///   - objective: Distance metric to minimize. Default `.euclidean`.
    ///   - eps: Small constant to avoid division by zero. Default 1e-8.
    ///   - seed: Optional random seed for reproducible initializations.
    /// - Returns: `Result` containing W and H.
    public static func decompose(
        _ V: Signal,
        nComponents: Int = 8,
        nIter: Int = 200,
        objective: Objective = .euclidean,
        eps: Float = 1e-8,
        seed: UInt64? = nil
    ) -> Result {
        precondition(V.shape.count == 2, "NMF input must be 2D [nFeatures, nSamples]")
        let nFeatures = V.shape[0]
        let nSamples = V.shape[1]
        precondition(nFeatures > 0 && nSamples > 0, "NMF input dimensions must be positive")
        precondition(nComponents > 0 && nComponents <= min(nFeatures, nSamples),
                     "nComponents must be in [1, min(nFeatures, nSamples)]")

        // Initialize W [nFeatures, nComponents] and H [nComponents, nSamples]
        // with random non-negative values scaled by sqrt(mean(V) / nComponents)
        var rng: RandomNumberGenerator
        if let seed = seed {
            rng = SeededRNG(seed: seed)
        } else {
            rng = SystemRandomNumberGenerator() as RandomNumberGenerator
        }

        let scale: Float = V.withUnsafeBufferPointer { buf in
            var mean: Float = 0
            vDSP_meanv(buf.baseAddress!, 1, &mean, vDSP_Length(buf.count))
            return sqrtf(max(mean, eps) / Float(nComponents))
        }

        let wCount = nFeatures * nComponents
        let hCount = nComponents * nSamples

        let wPtr = UnsafeMutablePointer<Float>.allocate(capacity: wCount)
        let hPtr = UnsafeMutablePointer<Float>.allocate(capacity: hCount)

        for i in 0..<wCount {
            wPtr[i] = abs(Float.random(in: 0.0..<1.0, using: &rng)) * scale + eps
        }
        for i in 0..<hCount {
            hPtr[i] = abs(Float.random(in: 0.0..<1.0, using: &rng)) * scale + eps
        }

        // Run multiplicative updates
        V.withUnsafeBufferPointer { vBuf in
            let vBase = vBuf.baseAddress!

            switch objective {
            case .euclidean:
                euclideanUpdates(
                    vBase: vBase, wPtr: wPtr, hPtr: hPtr,
                    nFeatures: nFeatures, nSamples: nSamples,
                    nComponents: nComponents, nIter: nIter, eps: eps
                )
            case .klDivergence:
                klDivergenceUpdates(
                    vBase: vBase, wPtr: wPtr, hPtr: hPtr,
                    nFeatures: nFeatures, nSamples: nSamples,
                    nComponents: nComponents, nIter: nIter, eps: eps
                )
            }
        }

        let wBuffer = UnsafeMutableBufferPointer(start: wPtr, count: wCount)
        let hBuffer = UnsafeMutableBufferPointer(start: hPtr, count: hCount)

        let W = Signal(taking: wBuffer, shape: [nFeatures, nComponents], sampleRate: V.sampleRate)
        let H = Signal(taking: hBuffer, shape: [nComponents, nSamples], sampleRate: V.sampleRate)

        return Result(W: W, H: H)
    }

    /// Compute relative reconstruction error: ||V - W*H||_F / ||V||_F.
    ///
    /// - Parameters:
    ///   - V: Original matrix, shape [nFeatures, nSamples].
    ///   - W: Basis matrix, shape [nFeatures, nComponents].
    ///   - H: Activation matrix, shape [nComponents, nSamples].
    /// - Returns: Relative Frobenius-norm reconstruction error.
    public static func reconstructionError(
        V: Signal, W: Signal, H: Signal
    ) -> Float {
        let nFeatures = V.shape[0]
        let nSamples = V.shape[1]
        let nComponents = W.shape[1]

        precondition(V.shape == [nFeatures, nSamples])
        precondition(W.shape == [nFeatures, nComponents])
        precondition(H.shape == [nComponents, nSamples])

        let whCount = nFeatures * nSamples

        // Compute WH
        let whPtr = UnsafeMutablePointer<Float>.allocate(capacity: whCount)
        defer { whPtr.deallocate() }

        W.withUnsafeBufferPointer { wBuf in
            H.withUnsafeBufferPointer { hBuf in
                vDSP_mmul(
                    wBuf.baseAddress!, 1,
                    hBuf.baseAddress!, 1,
                    whPtr, 1,
                    vDSP_Length(nFeatures),
                    vDSP_Length(nSamples),
                    vDSP_Length(nComponents)
                )
            }
        }

        // Compute ||V - WH||_F and ||V||_F
        var vNorm: Float = 0
        var diffNorm: Float = 0

        V.withUnsafeBufferPointer { vBuf in
            let vBase = vBuf.baseAddress!

            // diff = V - WH
            let diffPtr = UnsafeMutablePointer<Float>.allocate(capacity: whCount)
            defer { diffPtr.deallocate() }

            // diff = V
            diffPtr.initialize(from: vBase, count: whCount)
            // diff = V - WH
            var minusOne: Float = -1.0
            vDSP_vsma(whPtr, 1, &minusOne, diffPtr, 1, diffPtr, 1, vDSP_Length(whCount))

            // ||diff||_F^2
            vDSP_dotpr(diffPtr, 1, diffPtr, 1, &diffNorm, vDSP_Length(whCount))
            // ||V||_F^2
            vDSP_dotpr(vBase, 1, vBase, 1, &vNorm, vDSP_Length(whCount))
        }

        guard vNorm > 0 else { return 0 }
        return sqrtf(diffNorm / vNorm)
    }

    // MARK: - Euclidean Updates

    /// Multiplicative updates minimizing Frobenius norm ||V - WH||_F^2.
    ///
    /// H <- H * (W^T V) / (W^T W H + eps)
    /// W <- W * (V H^T) / (W H H^T + eps)
    private static func euclideanUpdates(
        vBase: UnsafePointer<Float>,
        wPtr: UnsafeMutablePointer<Float>,
        hPtr: UnsafeMutablePointer<Float>,
        nFeatures: Int, nSamples: Int, nComponents: Int,
        nIter: Int, eps: Float
    ) {
        let wCount = nFeatures * nComponents
        let hCount = nComponents * nSamples

        // Scratch buffers
        let wtv = UnsafeMutablePointer<Float>.allocate(capacity: nComponents * nSamples)  // W^T V
        let wtw = UnsafeMutablePointer<Float>.allocate(capacity: nComponents * nComponents) // W^T W
        let wtwh = UnsafeMutablePointer<Float>.allocate(capacity: nComponents * nSamples)   // W^T W H
        let vht = UnsafeMutablePointer<Float>.allocate(capacity: nFeatures * nComponents)   // V H^T
        let hht = UnsafeMutablePointer<Float>.allocate(capacity: nComponents * nComponents) // H H^T
        let whht = UnsafeMutablePointer<Float>.allocate(capacity: nFeatures * nComponents)  // W H H^T

        defer {
            wtv.deallocate()
            wtw.deallocate()
            wtwh.deallocate()
            vht.deallocate()
            hht.deallocate()
            whht.deallocate()
        }

        for _ in 0..<nIter {
            // --- Update H ---
            // W^T V: [nComponents, nSamples] = [nComponents, nFeatures] x [nFeatures, nSamples]
            cblas_sgemm(
                CblasRowMajor, CblasTrans, CblasNoTrans,
                Int32(nComponents), Int32(nSamples), Int32(nFeatures),
                1.0, wPtr, Int32(nComponents),
                vBase, Int32(nSamples),
                0.0, wtv, Int32(nSamples)
            )

            // W^T W: [nComponents, nComponents] = [nComponents, nFeatures] x [nFeatures, nComponents]
            cblas_sgemm(
                CblasRowMajor, CblasTrans, CblasNoTrans,
                Int32(nComponents), Int32(nComponents), Int32(nFeatures),
                1.0, wPtr, Int32(nComponents),
                wPtr, Int32(nComponents),
                0.0, wtw, Int32(nComponents)
            )

            // W^T W H: [nComponents, nSamples] = [nComponents, nComponents] x [nComponents, nSamples]
            vDSP_mmul(
                wtw, 1,
                hPtr, 1,
                wtwh, 1,
                vDSP_Length(nComponents),
                vDSP_Length(nSamples),
                vDSP_Length(nComponents)
            )

            // H <- H * (W^T V) / (W^T W H + eps)
            for i in 0..<hCount {
                hPtr[i] = hPtr[i] * wtv[i] / (wtwh[i] + eps)
            }

            // --- Update W ---
            // V H^T: [nFeatures, nComponents] = [nFeatures, nSamples] x [nSamples, nComponents]
            cblas_sgemm(
                CblasRowMajor, CblasNoTrans, CblasTrans,
                Int32(nFeatures), Int32(nComponents), Int32(nSamples),
                1.0, vBase, Int32(nSamples),
                hPtr, Int32(nSamples),
                0.0, vht, Int32(nComponents)
            )

            // H H^T: [nComponents, nComponents] = [nComponents, nSamples] x [nSamples, nComponents]
            cblas_sgemm(
                CblasRowMajor, CblasNoTrans, CblasTrans,
                Int32(nComponents), Int32(nComponents), Int32(nSamples),
                1.0, hPtr, Int32(nSamples),
                hPtr, Int32(nSamples),
                0.0, hht, Int32(nComponents)
            )

            // W H H^T: [nFeatures, nComponents] = [nFeatures, nComponents] x [nComponents, nComponents]
            vDSP_mmul(
                wPtr, 1,
                hht, 1,
                whht, 1,
                vDSP_Length(nFeatures),
                vDSP_Length(nComponents),
                vDSP_Length(nComponents)
            )

            // W <- W * (V H^T) / (W H H^T + eps)
            for i in 0..<wCount {
                wPtr[i] = wPtr[i] * vht[i] / (whht[i] + eps)
            }
        }
    }

    // MARK: - KL Divergence Updates

    /// Multiplicative updates minimizing generalized KL divergence D_KL(V || WH).
    ///
    /// H <- H * (W^T (V / (WH + eps))) / (colsum(W) + eps)
    /// W <- W * ((V / (WH + eps)) H^T) / (rowsum(H) + eps)
    private static func klDivergenceUpdates(
        vBase: UnsafePointer<Float>,
        wPtr: UnsafeMutablePointer<Float>,
        hPtr: UnsafeMutablePointer<Float>,
        nFeatures: Int, nSamples: Int, nComponents: Int,
        nIter: Int, eps: Float
    ) {
        let vCount = nFeatures * nSamples

        // Scratch
        let wh = UnsafeMutablePointer<Float>.allocate(capacity: vCount)        // W H
        let ratio = UnsafeMutablePointer<Float>.allocate(capacity: vCount)     // V / (WH + eps)
        let numerH = UnsafeMutablePointer<Float>.allocate(capacity: nComponents * nSamples)  // W^T ratio
        let numerW = UnsafeMutablePointer<Float>.allocate(capacity: nFeatures * nComponents)  // ratio H^T
        let colSumW = UnsafeMutablePointer<Float>.allocate(capacity: nComponents)  // column sums of W
        let rowSumH = UnsafeMutablePointer<Float>.allocate(capacity: nComponents)  // row sums of H

        defer {
            wh.deallocate()
            ratio.deallocate()
            numerH.deallocate()
            numerW.deallocate()
            colSumW.deallocate()
            rowSumH.deallocate()
        }

        for _ in 0..<nIter {
            // WH: [nFeatures, nSamples]
            vDSP_mmul(
                wPtr, 1,
                hPtr, 1,
                wh, 1,
                vDSP_Length(nFeatures),
                vDSP_Length(nSamples),
                vDSP_Length(nComponents)
            )

            // ratio = V / (WH + eps)
            for i in 0..<vCount {
                ratio[i] = vBase[i] / (wh[i] + eps)
            }

            // --- Update H ---
            // numerH = W^T ratio: [nComponents, nSamples]
            cblas_sgemm(
                CblasRowMajor, CblasTrans, CblasNoTrans,
                Int32(nComponents), Int32(nSamples), Int32(nFeatures),
                1.0, wPtr, Int32(nComponents),
                ratio, Int32(nSamples),
                0.0, numerH, Int32(nSamples)
            )

            // colSumW[k] = sum over features of W[:, k]
            for k in 0..<nComponents {
                var s: Float = 0
                for f in 0..<nFeatures {
                    s += wPtr[f * nComponents + k]
                }
                colSumW[k] = s + eps
            }

            // H <- H * numerH / colSumW (broadcast colSumW across samples)
            for k in 0..<nComponents {
                let denom = colSumW[k]
                for t in 0..<nSamples {
                    let idx = k * nSamples + t
                    hPtr[idx] = hPtr[idx] * numerH[idx] / denom
                }
            }

            // Recompute WH with updated H for W update
            vDSP_mmul(
                wPtr, 1,
                hPtr, 1,
                wh, 1,
                vDSP_Length(nFeatures),
                vDSP_Length(nSamples),
                vDSP_Length(nComponents)
            )

            // Recompute ratio
            for i in 0..<vCount {
                ratio[i] = vBase[i] / (wh[i] + eps)
            }

            // --- Update W ---
            // numerW = ratio H^T: [nFeatures, nComponents]
            cblas_sgemm(
                CblasRowMajor, CblasNoTrans, CblasTrans,
                Int32(nFeatures), Int32(nComponents), Int32(nSamples),
                1.0, ratio, Int32(nSamples),
                hPtr, Int32(nSamples),
                0.0, numerW, Int32(nComponents)
            )

            // rowSumH[k] = sum over samples of H[k, :]
            for k in 0..<nComponents {
                var s: Float = 0
                for t in 0..<nSamples {
                    s += hPtr[k * nSamples + t]
                }
                rowSumH[k] = s + eps
            }

            // W <- W * numerW / rowSumH (broadcast rowSumH across features)
            for f in 0..<nFeatures {
                for k in 0..<nComponents {
                    let idx = f * nComponents + k
                    wPtr[idx] = wPtr[idx] * numerW[idx] / rowSumH[k]
                }
            }
        }
    }
}

// MARK: - Seeded RNG (for reproducible initialization)

/// A simple deterministic random number generator for reproducible NMF initializations.
private struct SeededRNG: RandomNumberGenerator {
    private var state: UInt64

    init(seed: UInt64) {
        self.state = seed
    }

    mutating func next() -> UInt64 {
        // SplitMix64
        state &+= 0x9e3779b97f4a7c15
        var z = state
        z = (z ^ (z >> 30)) &* 0xbf58476d1ce4e5b9
        z = (z ^ (z >> 27)) &* 0x94d049bb133111eb
        return z ^ (z >> 31)
    }
}
