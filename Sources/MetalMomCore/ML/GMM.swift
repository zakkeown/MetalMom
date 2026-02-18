import Foundation
import Accelerate

/// Gaussian Mixture Model evaluation with diagonal covariance.
///
/// Used by neural onset/beat/chord pipelines where model activations are
/// evaluated against trained GMMs to produce observation likelihoods for
/// HMM/CRF decoding. All methods operate in the log domain for numerical
/// stability.
public enum GMM {

    // MARK: - GMM Parameters

    /// Parameters for a Gaussian Mixture Model with diagonal covariance.
    public struct Parameters {
        /// Component means: [K][D] where K = number of components, D = feature dimension.
        public let means: [[Float]]
        /// Component diagonal covariances: [K][D].
        public let covariances: [[Float]]
        /// Component log-weights (log prior probabilities): [K]. Must sum to ~1.0 in exp domain.
        public let logWeights: [Float]

        /// Number of components K.
        public var nComponents: Int { means.count }
        /// Feature dimension D.
        public var dimension: Int { means.isEmpty ? 0 : means[0].count }

        /// Initialize from weights in the probability domain.
        ///
        /// - Parameters:
        ///   - means: Component means [K][D].
        ///   - covariances: Diagonal covariances [K][D].
        ///   - weights: Component weights (prior probabilities) [K]. Should sum to 1.0.
        public init(means: [[Float]], covariances: [[Float]], weights: [Float]) {
            self.means = means
            self.covariances = covariances
            self.logWeights = weights.map { log($0) }
        }

        /// Initialize from weights already in the log domain.
        ///
        /// - Parameters:
        ///   - means: Component means [K][D].
        ///   - covariances: Diagonal covariances [K][D].
        ///   - logWeights: Log component weights [K].
        public init(means: [[Float]], covariances: [[Float]], logWeights: [Float]) {
            self.means = means
            self.covariances = covariances
            self.logWeights = logWeights
        }
    }

    // MARK: - Log-Likelihood

    /// Compute the log-likelihood of a single observation under the GMM.
    ///
    /// log P(x) = logSumExp_k [ log(w_k) + log N(x | mu_k, sigma_k^2) ]
    ///
    /// where log N(x | mu, sigma^2) = -0.5 * [ D*log(2*pi) + sum(log(sigma^2)) + sum((x-mu)^2 / sigma^2) ]
    ///
    /// - Parameters:
    ///   - observation: Feature vector of dimension D.
    ///   - params: GMM parameters.
    /// - Returns: Log-likelihood log P(x | GMM).
    public static func logLikelihood(observation: [Float], params: Parameters) -> Float {
        let K = params.nComponents
        var componentLogProbs = [Float](repeating: 0, count: K)

        for k in 0..<K {
            let logPDF = logGaussianPDF(
                observation: observation,
                mean: params.means[k],
                covariance: params.covariances[k]
            )
            componentLogProbs[k] = params.logWeights[k] + logPDF
        }

        return logSumExp(componentLogProbs)
    }

    /// Compute log-likelihoods for a batch of observations.
    ///
    /// - Parameters:
    ///   - observations: [N][D] array of feature vectors.
    ///   - params: GMM parameters.
    /// - Returns: [N] array of log-likelihoods.
    public static func logLikelihoodBatch(observations: [[Float]], params: Parameters) -> [Float] {
        observations.map { logLikelihood(observation: $0, params: params) }
    }

    // MARK: - Per-Component Scores

    /// Compute log responsibilities (posterior component probabilities) for an observation.
    ///
    /// log r_k = log(w_k) + log N(x | mu_k, sigma_k^2) - log P(x)
    ///
    /// - Parameters:
    ///   - observation: Feature vector of dimension D.
    ///   - params: GMM parameters.
    /// - Returns: [K] array of log responsibilities (sum to ~0 in log domain, ~1 in exp domain).
    public static func logResponsibilities(observation: [Float], params: Parameters) -> [Float] {
        let K = params.nComponents
        var componentLogProbs = [Float](repeating: 0, count: K)

        for k in 0..<K {
            let logPDF = logGaussianPDF(
                observation: observation,
                mean: params.means[k],
                covariance: params.covariances[k]
            )
            componentLogProbs[k] = params.logWeights[k] + logPDF
        }

        let logEvidence = logSumExp(componentLogProbs)

        // log r_k = componentLogProbs[k] - logEvidence
        return componentLogProbs.map { $0 - logEvidence }
    }

    // MARK: - Component Log-PDF

    /// Compute log probability density for a single Gaussian component (diagonal covariance).
    ///
    /// log N(x | mu, sigma^2) = -0.5 * [ D*log(2*pi) + sum(log(sigma^2)) + sum((x-mu)^2 / sigma^2) ]
    ///
    /// - Parameters:
    ///   - observation: Feature vector [D].
    ///   - mean: Component mean [D].
    ///   - covariance: Diagonal covariance [D] (variance per dimension).
    /// - Returns: Log PDF value.
    public static func logGaussianPDF(observation: [Float], mean: [Float], covariance: [Float]) -> Float {
        let D = observation.count
        let log2pi = log(2.0 * Float.pi)

        // Compute sum(log(sigma^2)) and sum((x - mu)^2 / sigma^2) using vDSP where helpful
        var diff = [Float](repeating: 0, count: D)
        var logVar: Float = 0
        var mahal: Float = 0

        // diff = observation - mean
        vDSP_vsub(mean, 1, observation, 1, &diff, 1, vDSP_Length(D))

        // diff^2
        var diffSq = [Float](repeating: 0, count: D)
        vDSP_vsq(diff, 1, &diffSq, 1, vDSP_Length(D))

        // diffSq / covariance (element-wise)
        var scaled = [Float](repeating: 0, count: D)
        vDSP_vdiv(covariance, 1, diffSq, 1, &scaled, 1, vDSP_Length(D))

        // sum of scaled
        vDSP_sve(scaled, 1, &mahal, vDSP_Length(D))

        // sum of log(covariance)
        var logCov = [Float](repeating: 0, count: D)
        var Dlen = Int32(D)
        vvlogf(&logCov, covariance, &Dlen)
        vDSP_sve(logCov, 1, &logVar, vDSP_Length(D))

        return -0.5 * (Float(D) * log2pi + logVar + mahal)
    }

    // MARK: - Private Helpers

    /// Numerically stable log-sum-exp of an array of values.
    ///
    /// Computes log(sum(exp(values))) without overflow.
    private static func logSumExp(_ values: [Float]) -> Float {
        guard let maxVal = values.max(), maxVal > -.infinity else { return -.infinity }
        var sum: Float = 0
        for v in values {
            sum += exp(v - maxVal)
        }
        return maxVal + log(sum)
    }
}
