import XCTest
import Accelerate
@testable import MetalMomCore

/// Benchmarks to calibrate CPU vs GPU dispatch thresholds.
/// Run with: swift test --filter ThresholdCalibrationTests -v
/// These are not normal unit tests -- they measure performance to find crossover points.
final class ThresholdCalibrationTests: XCTestCase {

    // MARK: - Helpers

    private func getBackend() throws -> MetalBackend {
        guard let backend = MetalBackend.shared else {
            throw XCTSkip("Metal not available")
        }
        return backend
    }

    private func getShaders() throws -> MetalShaders {
        let backend = try getBackend()
        guard let shaders = backend.shaders else {
            throw XCTSkip("Metal shader compilation failed")
        }
        return shaders
    }

    // MARK: - STFT Threshold

    func testSTFTCrossover() throws {
        _ = try getBackend()

        let sizes = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
        let nFFT = 2048
        let hopLength = 512

        print("\n=== STFT Crossover ===")
        print("Signal Length | CPU (ms) | GPU (ms) | Winner")
        print("-------------|----------|----------|-------")

        let stft = STFT()

        for size in sizes {
            let signal = Signal(data: [Float](repeating: 0.5, count: size), sampleRate: 22050)
            let input = STFTInput(
                signal: signal, nFFT: nFFT, hopLength: hopLength,
                winLength: nFFT, center: true
            )

            // Warmup
            _ = stft.executeCPU(input)
            _ = stft.executeGPU(input)

            // Benchmark CPU
            let cpuStart = CFAbsoluteTimeGetCurrent()
            for _ in 0..<5 {
                _ = stft.executeCPU(input)
            }
            let cpuTime = (CFAbsoluteTimeGetCurrent() - cpuStart) / 5.0 * 1000

            // Benchmark GPU
            let gpuStart = CFAbsoluteTimeGetCurrent()
            for _ in 0..<5 {
                _ = stft.executeGPU(input)
            }
            let gpuTime = (CFAbsoluteTimeGetCurrent() - gpuStart) / 5.0 * 1000

            let winner = cpuTime < gpuTime ? "CPU" : "GPU"
            print(String(format: "%12d | %8.2f | %8.2f | %@", size, cpuTime, gpuTime, winner))
        }
    }

    // MARK: - Elementwise Threshold

    func testElementwiseCrossover() throws {
        let shaders = try getShaders()

        let sizes = [1024, 4096, 16384, 65536, 262144, 1048576, 4194304]

        print("\n=== Elementwise Log Crossover ===")
        print("Array Size   | CPU (ms) | GPU (ms) | Winner")
        print("-------------|----------|----------|-------")

        for size in sizes {
            let data = (0..<size).map { _ in Float.random(in: 0.1...10.0) }

            // Warmup
            _ = shaders.log(data)
            var cpuResult = [Float](repeating: 0, count: size)
            var count = Int32(size)
            vvlogf(&cpuResult, data, &count)

            // Benchmark CPU
            let cpuStart = CFAbsoluteTimeGetCurrent()
            for _ in 0..<10 {
                vvlogf(&cpuResult, data, &count)
            }
            let cpuTime = (CFAbsoluteTimeGetCurrent() - cpuStart) / 10.0 * 1000

            // Benchmark GPU
            let gpuStart = CFAbsoluteTimeGetCurrent()
            for _ in 0..<10 {
                _ = shaders.log(data)
            }
            let gpuTime = (CFAbsoluteTimeGetCurrent() - gpuStart) / 10.0 * 1000

            let winner = cpuTime < gpuTime ? "CPU" : "GPU"
            print(String(format: "%12d | %8.2f | %8.2f | %@", size, cpuTime, gpuTime, winner))
        }
    }

    // MARK: - Matrix Multiply Threshold

    func testMatmulCrossover() throws {
        _ = try getBackend()

        // Test various matrix sizes relevant to mel spectrogram
        let configs: [(m: Int, k: Int, n: Int)] = [
            (32, 64, 10),
            (64, 128, 20),
            (128, 512, 50),
            (128, 1025, 100),
            (128, 1025, 200),
            (128, 1025, 500),
            (256, 2048, 100),
        ]

        print("\n=== Matrix Multiply Crossover ===")
        print("M x K x N         | CPU (ms) | GPU (ms) | Winner")
        print("-------------------|----------|----------|-------")

        for config in configs {
            let a = (0..<(config.m * config.k)).map { _ in Float.random(in: -1...1) }
            let b = (0..<(config.k * config.n)).map { _ in Float.random(in: -1...1) }

            // Warmup
            _ = MetalMatmul.multiply(
                a: a, aRows: config.m, aCols: config.k,
                b: b, bRows: config.k, bCols: config.n
            )
            var cpuResult = [Float](repeating: 0, count: config.m * config.n)
            vDSP_mmul(
                a, 1, b, 1, &cpuResult, 1,
                vDSP_Length(config.m), vDSP_Length(config.n), vDSP_Length(config.k)
            )

            // Benchmark CPU
            let cpuStart = CFAbsoluteTimeGetCurrent()
            for _ in 0..<10 {
                vDSP_mmul(
                    a, 1, b, 1, &cpuResult, 1,
                    vDSP_Length(config.m), vDSP_Length(config.n), vDSP_Length(config.k)
                )
            }
            let cpuTime = (CFAbsoluteTimeGetCurrent() - cpuStart) / 10.0 * 1000

            // Benchmark GPU
            let gpuStart = CFAbsoluteTimeGetCurrent()
            for _ in 0..<10 {
                _ = MetalMatmul.multiply(
                    a: a, aRows: config.m, aCols: config.k,
                    b: b, bRows: config.k, bCols: config.n
                )
            }
            let gpuTime = (CFAbsoluteTimeGetCurrent() - gpuStart) / 10.0 * 1000

            let winner = cpuTime < gpuTime ? "CPU" : "GPU"
            let label = "\(config.m)x\(config.k)x\(config.n)"
            print(String(format: "%-18@ | %8.2f | %8.2f | %@", label as NSString, cpuTime, gpuTime, winner))
        }
    }

    // MARK: - Reduction Threshold

    func testReductionCrossover() throws {
        let shaders = try getShaders()

        let sizes = [1024, 4096, 16384, 65536, 262144, 1048576]

        print("\n=== Reduction Sum Crossover ===")
        print("Array Size   | CPU (ms) | GPU (ms) | Winner")
        print("-------------|----------|----------|-------")

        for size in sizes {
            let data = (0..<size).map { _ in Float.random(in: 0...1) }

            // Warmup
            _ = shaders.sum(data)
            var cpuSum: Float = 0
            vDSP_sve(data, 1, &cpuSum, vDSP_Length(size))

            // Benchmark CPU
            let cpuStart = CFAbsoluteTimeGetCurrent()
            for _ in 0..<20 {
                vDSP_sve(data, 1, &cpuSum, vDSP_Length(size))
            }
            let cpuTime = (CFAbsoluteTimeGetCurrent() - cpuStart) / 20.0 * 1000

            // Benchmark GPU
            let gpuStart = CFAbsoluteTimeGetCurrent()
            for _ in 0..<20 {
                _ = shaders.sum(data)
            }
            let gpuTime = (CFAbsoluteTimeGetCurrent() - gpuStart) / 20.0 * 1000

            let winner = cpuTime < gpuTime ? "CPU" : "GPU"
            print(String(format: "%12d | %8.2f | %8.2f | %@", size, cpuTime, gpuTime, winner))
        }
    }

    // MARK: - Convolution Threshold

    func testConvolutionCrossover() throws {
        let shaders = try getShaders()

        let sizes = [1024, 4096, 16384, 65536, 262144]
        let kernelSize = 64

        print("\n=== Conv1D Crossover (kernel=64) ===")
        print("Input Size   | CPU (ms) | GPU (ms) | Winner")
        print("-------------|----------|----------|-------")

        let kernel = (0..<kernelSize).map { _ in Float.random(in: -1...1) }

        for size in sizes {
            let data = (0..<size).map { _ in Float.random(in: -1...1) }

            // Warmup
            _ = shaders.conv1d(data, kernel: kernel)

            // CPU using vDSP_conv
            let outputLen = size - kernelSize + 1
            var cpuResult = [Float](repeating: 0, count: outputLen)

            let cpuStart = CFAbsoluteTimeGetCurrent()
            for _ in 0..<10 {
                vDSP_conv(data, 1, kernel, 1, &cpuResult, 1,
                          vDSP_Length(outputLen), vDSP_Length(kernelSize))
            }
            let cpuTime = (CFAbsoluteTimeGetCurrent() - cpuStart) / 10.0 * 1000

            // GPU
            let gpuStart = CFAbsoluteTimeGetCurrent()
            for _ in 0..<10 {
                _ = shaders.conv1d(data, kernel: kernel)
            }
            let gpuTime = (CFAbsoluteTimeGetCurrent() - gpuStart) / 10.0 * 1000

            let winner = cpuTime < gpuTime ? "CPU" : "GPU"
            print(String(format: "%12d | %8.2f | %8.2f | %@", size, cpuTime, gpuTime, winner))
        }
    }

    // MARK: - Summary

    func testPrintCurrentThresholds() throws {
        let backend = try getBackend()

        let profile = backend.chipProfile
        print("\n=== Current ChipProfile Thresholds ===")
        print("GPU Family: \(profile.gpuFamily)")
        print("Estimated Cores: \(profile.estimatedCoreCount)")
        print("STFT threshold: \(profile.threshold(for: .stft))")
        print("Matmul threshold: \(profile.threshold(for: .matmul))")
        print("Elementwise threshold: \(profile.threshold(for: .elementwise))")
        print("Reduction threshold: \(profile.threshold(for: .reduction))")
        print("Convolution threshold: \(profile.threshold(for: .convolution))")
    }
}
