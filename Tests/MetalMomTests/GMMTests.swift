import XCTest
@testable import MetalMomCore

final class GMMTests: XCTestCase {

    // MARK: - Test 1: Single Gaussian (K=1)

    func testSingleGaussianLogPDF() {
        // Single 1D Gaussian: mean=0, variance=1 (standard normal).
        // log N(x=0 | mu=0, sigma^2=1) = -0.5 * [1*log(2*pi) + log(1) + 0]
        //                                = -0.5 * log(2*pi)
        //                                ≈ -0.9189385
        let logPDF = GMM.logGaussianPDF(
            observation: [0.0],
            mean: [0.0],
            covariance: [1.0]
        )
        let expected: Float = -0.5 * log(2.0 * Float.pi)
        XCTAssertEqual(logPDF, expected, accuracy: 1e-4,
                       "Standard normal PDF at x=0 should be -0.5*log(2*pi)")

        // Single Gaussian GMM with weight=1 should give the same result as logGaussianPDF.
        let params = GMM.Parameters(
            means: [[0.0]],
            covariances: [[1.0]],
            weights: [1.0]
        )
        let logLL = GMM.logLikelihood(observation: [0.0], params: params)
        XCTAssertEqual(logLL, expected, accuracy: 1e-4,
                       "K=1 GMM logLikelihood should equal logGaussianPDF")
    }

    func testSingleGaussian2D() {
        // 2D Gaussian: mean=[1, 2], covariance=[0.5, 2.0]
        // Observation: [1, 2] (at the mean)
        // log N([1,2] | [1,2], [0.5, 2.0]) = -0.5 * [2*log(2*pi) + log(0.5) + log(2.0) + 0 + 0]
        //                                    = -0.5 * [2*log(2*pi) + log(1.0)]
        //                                    = -0.5 * 2*log(2*pi)
        //                                    = -log(2*pi)
        //                                    ≈ -1.8378770
        let logPDF = GMM.logGaussianPDF(
            observation: [1.0, 2.0],
            mean: [1.0, 2.0],
            covariance: [0.5, 2.0]
        )
        let expected: Float = -0.5 * (2.0 * log(2.0 * Float.pi) + log(0.5) + log(2.0))
        XCTAssertEqual(logPDF, expected, accuracy: 1e-4)
    }

    // MARK: - Test 2: Two-Component GMM

    func testTwoComponentGMMLogLikelihood() {
        // Two 1D Gaussian components:
        //   Component 0: weight=0.3, mean=0.0, variance=1.0
        //   Component 1: weight=0.7, mean=3.0, variance=0.5
        // Observation: x=1.0
        //
        // log N(1 | 0, 1) = -0.5 * [log(2*pi) + log(1) + (1-0)^2/1]
        //                  = -0.5 * [log(2*pi) + 1]
        //                  ≈ -1.4189385
        //
        // log N(1 | 3, 0.5) = -0.5 * [log(2*pi) + log(0.5) + (1-3)^2/0.5]
        //                    = -0.5 * [log(2*pi) + log(0.5) + 8]
        //                    ≈ -4.5725789
        //
        // log P(x) = logSumExp(log(0.3) + logN0, log(0.7) + logN1)

        let params = GMM.Parameters(
            means: [[0.0], [3.0]],
            covariances: [[1.0], [0.5]],
            weights: [0.3, 0.7]
        )

        let logN0: Float = -0.5 * (log(2.0 * Float.pi) + 0 + 1.0)
        let logN1: Float = -0.5 * (log(2.0 * Float.pi) + log(0.5) + 8.0)
        let logP0 = log(Float(0.3)) + logN0
        let logP1 = log(Float(0.7)) + logN1
        let maxVal = max(logP0, logP1)
        let expectedLogLL = maxVal + log(exp(logP0 - maxVal) + exp(logP1 - maxVal))

        let actualLogLL = GMM.logLikelihood(observation: [1.0], params: params)
        XCTAssertEqual(actualLogLL, expectedLogLL, accuracy: 1e-4,
                       "Two-component GMM log-likelihood should match manual computation")
    }

    // MARK: - Test 3: Responsibilities Sum to 1

    func testResponsibilitiesSumToOne() {
        let params = GMM.Parameters(
            means: [[0.0, 0.0], [3.0, 3.0], [-2.0, 1.0]],
            covariances: [[1.0, 1.0], [0.5, 0.5], [2.0, 2.0]],
            weights: [0.2, 0.5, 0.3]
        )

        let observations: [[Float]] = [
            [0.0, 0.0],
            [3.0, 3.0],
            [-2.0, 1.0],
            [1.0, 1.5],
            [10.0, -5.0]
        ]

        for (i, obs) in observations.enumerated() {
            let logResp = GMM.logResponsibilities(observation: obs, params: params)
            XCTAssertEqual(logResp.count, 3, "Should have K=3 responsibilities")

            // exp(logResp) should sum to 1.0
            let respSum = logResp.reduce(Float(0)) { $0 + exp($1) }
            XCTAssertEqual(respSum, 1.0, accuracy: 1e-4,
                           "Responsibilities should sum to 1.0 for observation \(i)")

            // All log responsibilities should be <= 0 (probabilities <= 1)
            for k in 0..<3 {
                XCTAssertLessThanOrEqual(logResp[k], 1e-5,
                    "Log responsibility should be <= 0")
            }
        }
    }

    // MARK: - Test 4: Batch Consistency

    func testBatchConsistencyWithIndividual() {
        let params = GMM.Parameters(
            means: [[1.0, 2.0], [-1.0, 0.0]],
            covariances: [[0.5, 1.0], [1.0, 0.5]],
            weights: [0.4, 0.6]
        )

        let observations: [[Float]] = [
            [0.0, 0.0],
            [1.0, 2.0],
            [-1.0, 0.0],
            [5.0, -3.0],
            [0.5, 1.0]
        ]

        let batchResults = GMM.logLikelihoodBatch(observations: observations, params: params)
        XCTAssertEqual(batchResults.count, observations.count)

        for (i, obs) in observations.enumerated() {
            let individual = GMM.logLikelihood(observation: obs, params: params)
            XCTAssertEqual(batchResults[i], individual, accuracy: 1e-6,
                           "Batch result should match individual for observation \(i)")
        }
    }

    // MARK: - Test 5: Observation at Mean (Degenerate Case)

    func testObservationAtMean() {
        // When x == mu, the Mahalanobis distance is 0.
        // log N(mu | mu, sigma^2) = -0.5 * [D*log(2*pi) + sum(log(sigma^2))]
        let D = 3
        let mean: [Float] = [1.0, 2.0, 3.0]
        let covariance: [Float] = [0.5, 1.0, 2.0]

        let logPDF = GMM.logGaussianPDF(
            observation: mean,
            mean: mean,
            covariance: covariance
        )

        var expectedLogDet: Float = 0
        for d in 0..<D {
            expectedLogDet += log(covariance[d])
        }
        let expected: Float = -0.5 * (Float(D) * log(2.0 * Float.pi) + expectedLogDet)

        XCTAssertEqual(logPDF, expected, accuracy: 1e-4,
                       "At the mean, Mahalanobis distance should be zero")
    }

    func testObservationAtMeanIsMaxPDF() {
        // The PDF is maximized when x == mu (for any given covariance).
        let mean: [Float] = [2.0, -1.0]
        let covariance: [Float] = [1.0, 1.0]

        let logPDFatMean = GMM.logGaussianPDF(
            observation: mean,
            mean: mean,
            covariance: covariance
        )

        // Nearby points should have lower log PDF
        let offsets: [[Float]] = [[2.1, -1.0], [2.0, -0.9], [2.5, -1.5], [1.0, 0.0]]
        for offset in offsets {
            let logPDFoff = GMM.logGaussianPDF(
                observation: offset,
                mean: mean,
                covariance: covariance
            )
            XCTAssertGreaterThan(logPDFatMean, logPDFoff,
                "PDF at mean should be greater than at \(offset)")
        }
    }

    // MARK: - Test 6: High-Dimensional (D=10, K=3)

    func testHighDimensional() {
        let D = 10
        let K = 3

        // Create deterministic parameters
        var means = [[Float]]()
        var covariances = [[Float]]()
        var weights = [Float]()

        for k in 0..<K {
            let mean = (0..<D).map { Float($0) * Float(k + 1) * 0.1 }
            let cov = [Float](repeating: Float(k + 1) * 0.5, count: D)
            means.append(mean)
            covariances.append(cov)
            weights.append(Float(k + 1))  // [1, 2, 3] -> normalized to [1/6, 2/6, 3/6]
        }
        // Normalize weights
        let wSum = weights.reduce(0, +)
        weights = weights.map { $0 / wSum }

        let params = GMM.Parameters(means: means, covariances: covariances, weights: weights)
        XCTAssertEqual(params.nComponents, K)
        XCTAssertEqual(params.dimension, D)

        // Evaluate at a test point
        let obs = [Float](repeating: 0.5, count: D)
        let logLL = GMM.logLikelihood(observation: obs, params: params)

        XCTAssertFalse(logLL.isNaN, "High-dimensional logLikelihood should not be NaN")
        XCTAssertFalse(logLL.isInfinite, "High-dimensional logLikelihood should be finite")

        // Responsibilities should still sum to 1
        let logResp = GMM.logResponsibilities(observation: obs, params: params)
        XCTAssertEqual(logResp.count, K)
        let respSum = logResp.reduce(Float(0)) { $0 + exp($1) }
        XCTAssertEqual(respSum, 1.0, accuracy: 1e-4,
                       "Responsibilities should sum to 1.0 for D=10, K=3")
    }

    // MARK: - Test 7: Uniform Weights with Equal Observations

    func testUniformWeightsEqualResponsibilities() {
        // Three identical components (same mean, same covariance) with equal weights.
        // For any observation, responsibilities should all be equal (1/3 each).
        let K = 3
        let mean: [Float] = [0.0, 0.0]
        let cov: [Float] = [1.0, 1.0]
        let weight: Float = 1.0 / Float(K)

        let params = GMM.Parameters(
            means: [[Float]](repeating: mean, count: K),
            covariances: [[Float]](repeating: cov, count: K),
            weights: [Float](repeating: weight, count: K)
        )

        let obs: [Float] = [1.5, -0.5]
        let logResp = GMM.logResponsibilities(observation: obs, params: params)

        let expectedLogResp = log(Float(1.0) / Float(K))
        for k in 0..<K {
            XCTAssertEqual(logResp[k], expectedLogResp, accuracy: 1e-4,
                           "Uniform identical components should give equal responsibilities")
        }
    }

    func testUniformWeightsDifferentMeans() {
        // Equal weights, different means. Observation closer to component 0 should
        // give it higher responsibility.
        let params = GMM.Parameters(
            means: [[0.0], [10.0], [20.0]],
            covariances: [[1.0], [1.0], [1.0]],
            weights: [Float(1.0 / 3.0), Float(1.0 / 3.0), Float(1.0 / 3.0)]
        )

        let obs: [Float] = [0.5]  // much closer to component 0
        let logResp = GMM.logResponsibilities(observation: obs, params: params)

        // Component 0 should dominate
        XCTAssertGreaterThan(logResp[0], logResp[1],
            "Closer component should have higher responsibility")
        XCTAssertGreaterThan(logResp[1], logResp[2],
            "Responsibility should decrease with distance")

        // With means so far apart, component 0 should have nearly all the mass
        XCTAssertEqual(exp(logResp[0]), 1.0, accuracy: 1e-4,
                       "Component 0 should have nearly all responsibility for x=0.5")
    }

    // MARK: - Additional: Log-weight initialization

    func testLogWeightInitialization() {
        // Verify both init paths produce the same logWeights.
        let weights: [Float] = [0.2, 0.5, 0.3]
        let logWeights = weights.map { log($0) }

        let params1 = GMM.Parameters(
            means: [[0], [1], [2]],
            covariances: [[1], [1], [1]],
            weights: weights
        )
        let params2 = GMM.Parameters(
            means: [[0], [1], [2]],
            covariances: [[1], [1], [1]],
            logWeights: logWeights
        )

        for k in 0..<3 {
            XCTAssertEqual(params1.logWeights[k], params2.logWeights[k], accuracy: 1e-6,
                           "Both init paths should produce identical logWeights")
        }
    }

    // MARK: - Symmetry Test

    func testSymmetricGaussianPDF() {
        // For a symmetric Gaussian, N(x | mu, sigma^2) == N(2*mu - x | mu, sigma^2)
        let mean: [Float] = [3.0, 5.0]
        let cov: [Float] = [1.0, 2.0]
        let obs: [Float] = [4.0, 6.0]
        let mirrorObs: [Float] = [2.0, 4.0]  // 2*mean - obs

        let logPDF1 = GMM.logGaussianPDF(observation: obs, mean: mean, covariance: cov)
        let logPDF2 = GMM.logGaussianPDF(observation: mirrorObs, mean: mean, covariance: cov)

        XCTAssertEqual(logPDF1, logPDF2, accuracy: 1e-4,
                       "Gaussian PDF should be symmetric about the mean")
    }
}
