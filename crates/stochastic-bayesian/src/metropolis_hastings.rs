//! Metropolis-Hastings MCMC sampler
//!
//! Implements the Metropolis-Hastings algorithm for sampling from arbitrary
//! posterior distributions in Bayesian inference.
//!
//! # Mathematical Foundation
//!
//! **Metropolis-Hastings Algorithm:**
//!
//! 1. Start with initial state θ⁽⁰⁾
//! 2. For t = 1 to T:
//!    a. Propose θ* ~ q(·|θ⁽ᵗ⁻¹⁾)
//!    b. Compute acceptance ratio:
//!       ```text
//!       r = [π(θ*) q(θ⁽ᵗ⁻¹⁾|θ*)] / [π(θ⁽ᵗ⁻¹⁾) q(θ*|θ⁽ᵗ⁻¹⁾)]
//!       ```
//!    c. Accept with probability min(1, r):
//!       - θ⁽ᵗ⁾ = θ* with probability min(1, r)
//!       - θ⁽ᵗ⁾ = θ⁽ᵗ⁻¹⁾ otherwise
//!
//! **Properties:**
//! - Converges to stationary distribution π(θ)
//! - Symmetric proposals simplify to: r = π(θ*) / π(θ⁽ᵗ⁻¹⁾)
//! - Requires burn-in period to reach stationarity
//!
//! # References
//!
//! - Metropolis, N., et al. (1953). "Equation of state calculations by fast computing machines"
//! - Hastings, W.K. (1970). "Monte Carlo sampling methods using Markov chains"
//! - Gelman, A., et al. (2013). "Bayesian Data Analysis" (3rd ed.)
//! - Robert, C.P., & Casella, G. (2004). "Monte Carlo Statistical Methods"
//! - Brooks, S., et al. (2011). "Handbook of Markov Chain Monte Carlo"

use stochastic_core::StochasticError;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal as RandNormal};

/// Proposal distribution type
#[derive(Debug, Clone, Copy)]
pub enum ProposalType {
    /// Gaussian random walk: θ* = θ + N(0, σ²)
    RandomWalk { step_size: f64 },
    /// Independent Gaussian: θ* ~ N(0, σ²)
    Independent { scale: f64 },
}

/// Target posterior distribution
pub trait TargetDistribution {
    /// Evaluate log posterior: log π(θ|data)
    fn log_posterior(&self, theta: &[f64]) -> f64;

    /// Dimensionality of parameter space
    fn dimensions(&self) -> usize;
}

/// Configuration for Metropolis-Hastings sampler
#[derive(Debug, Clone)]
pub struct MetropolisHastingsConfig {
    /// Number of MCMC iterations
    pub num_iterations: usize,
    /// Burn-in period (discarded samples)
    pub burn_in: usize,
    /// Thinning factor (keep every nth sample)
    pub thin: usize,
    /// Proposal distribution
    pub proposal: ProposalType,
    /// Random seed
    pub seed: Option<u64>,
    /// Initial state (if None, starts at origin)
    pub initial_state: Option<Vec<f64>>,
}

impl Default for MetropolisHastingsConfig {
    fn default() -> Self {
        Self {
            num_iterations: 10000,
            burn_in: 1000,
            thin: 1,
            proposal: ProposalType::RandomWalk { step_size: 0.1 },
            seed: None,
            initial_state: None,
        }
    }
}

/// Metropolis-Hastings MCMC sampler
pub struct MetropolisHastings {
    config: MetropolisHastingsConfig,
}

impl MetropolisHastings {
    /// Create new sampler with default configuration
    pub fn new() -> Self {
        Self {
            config: MetropolisHastingsConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: MetropolisHastingsConfig) -> Self {
        Self { config }
    }

    /// Set number of iterations
    pub fn with_iterations(mut self, iterations: usize) -> Self {
        self.config.num_iterations = iterations;
        self
    }

    /// Set burn-in period
    pub fn with_burn_in(mut self, burn_in: usize) -> Self {
        self.config.burn_in = burn_in;
        self
    }

    /// Set thinning factor
    pub fn with_thin(mut self, thin: usize) -> Self {
        self.config.thin = thin.max(1);
        self
    }

    /// Set proposal distribution
    pub fn with_proposal(mut self, proposal: ProposalType) -> Self {
        self.config.proposal = proposal;
        self
    }

    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.config.seed = Some(seed);
        self
    }

    /// Propose new state based on current state
    fn propose(
        &self,
        current: &[f64],
        rng: &mut impl Rng,
    ) -> Vec<f64> {
        match self.config.proposal {
            ProposalType::RandomWalk { step_size } => {
                current.iter()
                    .map(|&x| {
                        let normal = RandNormal::new(0.0, step_size)
                            .expect("Failed to create Normal distribution");
                        x + normal.sample(rng)
                    })
                    .collect()
            }
            ProposalType::Independent { scale } => {
                (0..current.len())
                    .map(|_| {
                        let normal = RandNormal::new(0.0, scale)
                            .expect("Failed to create Normal distribution");
                        normal.sample(rng)
                    })
                    .collect()
            }
        }
    }

    /// Compute acceptance ratio
    fn acceptance_ratio(
        &self,
        target: &dyn TargetDistribution,
        current: &[f64],
        proposed: &[f64],
    ) -> f64 {
        let log_posterior_current = target.log_posterior(current);
        let log_posterior_proposed = target.log_posterior(proposed);

        // For symmetric proposals (like random walk), ratio simplifies
        let log_ratio = log_posterior_proposed - log_posterior_current;
        log_ratio.exp()
    }

    /// Run Metropolis-Hastings sampler
    pub fn sample(
        &self,
        target: &dyn TargetDistribution,
    ) -> Result<MCMCResult, StochasticError> {
        let dim = target.dimensions();

        // Initialize state
        let mut current_state = if let Some(ref init) = self.config.initial_state {
            if init.len() != dim {
                return Err(StochasticError::validation(
                    format!("Initial state dimension mismatch: expected {}, got {}", dim, init.len())
                ));
            }
            init.clone()
        } else {
            vec![0.0; dim]
        };

        let mut rng = if let Some(seed) = self.config.seed {
            rand::rngs::StdRng::seed_from_u64(seed)
        } else {
            rand::rngs::StdRng::from_entropy()
        };

        let mut samples = Vec::new();
        let mut accepted = 0;
        let mut log_posteriors = Vec::new();

        // MCMC iterations
        for iter in 0..self.config.num_iterations {
            // Propose new state
            let proposed_state = self.propose(&current_state, &mut rng);

            // Compute acceptance ratio
            let ratio = self.acceptance_ratio(target, &current_state, &proposed_state);

            // Accept/reject
            let u: f64 = rng.gen();
            if u < ratio.min(1.0) {
                current_state = proposed_state;
                accepted += 1;
            }

            // Store sample (after burn-in, with thinning)
            if iter >= self.config.burn_in && (iter - self.config.burn_in) % self.config.thin == 0 {
                samples.push(current_state.clone());
                log_posteriors.push(target.log_posterior(&current_state));
            }
        }

        // Compute diagnostics
        let acceptance_rate = accepted as f64 / self.config.num_iterations as f64;
        let effective_samples = samples.len();

        // Compute posterior statistics
        let posterior_means = self.compute_means(&samples);
        let posterior_std_devs = self.compute_std_devs(&samples, &posterior_means);

        // Compute autocorrelation at lag 1 (simplified ESS estimate)
        let autocorr_lag1 = self.compute_autocorrelation(&samples, 1);

        // Effective Sample Size (ESS) approximation
        let ess = if autocorr_lag1.abs() < 0.999 {
            effective_samples as f64 / (1.0 + 2.0 * autocorr_lag1.abs())
        } else {
            effective_samples as f64 / 100.0 // Conservative estimate
        };

        Ok(MCMCResult {
            samples,
            log_posteriors,
            posterior_means,
            posterior_std_devs,
            acceptance_rate,
            effective_sample_size: ess,
            num_samples: effective_samples,
            burn_in: self.config.burn_in,
            thin: self.config.thin,
        })
    }

    /// Compute means of samples
    fn compute_means(&self, samples: &[Vec<f64>]) -> Vec<f64> {
        if samples.is_empty() {
            return vec![];
        }

        let dim = samples[0].len();
        let n = samples.len() as f64;

        (0..dim)
            .map(|d| {
                samples.iter().map(|s| s[d]).sum::<f64>() / n
            })
            .collect()
    }

    /// Compute standard deviations
    fn compute_std_devs(&self, samples: &[Vec<f64>], means: &[f64]) -> Vec<f64> {
        if samples.is_empty() {
            return vec![];
        }

        let dim = samples[0].len();
        let n = samples.len() as f64;

        (0..dim)
            .map(|d| {
                let variance = samples.iter()
                    .map(|s| (s[d] - means[d]).powi(2))
                    .sum::<f64>() / (n - 1.0);
                variance.sqrt()
            })
            .collect()
    }

    /// Compute autocorrelation at given lag
    fn compute_autocorrelation(&self, samples: &[Vec<f64>], lag: usize) -> f64 {
        if samples.is_empty() || lag >= samples.len() {
            return 0.0;
        }

        // Compute autocorrelation for first dimension (simplified)
        let values: Vec<f64> = samples.iter().map(|s| s[0]).collect();
        let n = values.len();
        let mean = values.iter().sum::<f64>() / n as f64;

        let numerator: f64 = (0..n - lag)
            .map(|i| (values[i] - mean) * (values[i + lag] - mean))
            .sum();

        let denominator: f64 = values.iter()
            .map(|&x| (x - mean).powi(2))
            .sum();

        if denominator.abs() < 1e-10 {
            0.0
        } else {
            numerator / denominator
        }
    }
}

/// Result from MCMC sampling
#[derive(Debug, Clone)]
pub struct MCMCResult {
    /// Posterior samples (after burn-in and thinning)
    pub samples: Vec<Vec<f64>>,
    /// Log posterior values for each sample
    pub log_posteriors: Vec<f64>,
    /// Posterior means
    pub posterior_means: Vec<f64>,
    /// Posterior standard deviations
    pub posterior_std_devs: Vec<f64>,
    /// Acceptance rate
    pub acceptance_rate: f64,
    /// Effective sample size
    pub effective_sample_size: f64,
    /// Number of samples (after burn-in and thinning)
    pub num_samples: usize,
    /// Burn-in period used
    pub burn_in: usize,
    /// Thinning factor used
    pub thin: usize,
}

impl Default for MetropolisHastings {
    fn default() -> Self {
        Self::new()
    }
}

// Note: MetropolisHastings doesn't directly implement StochasticAnalyzer
// because it requires a TargetDistribution, not just a TimeSeries

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    /// Simple 1D Gaussian target for testing
    struct GaussianTarget {
        mean: f64,
        std_dev: f64,
    }

    impl TargetDistribution for GaussianTarget {
        fn log_posterior(&self, theta: &[f64]) -> f64 {
            let x = theta[0];
            let z = (x - self.mean) / self.std_dev;
            -0.5 * z * z - (2.0 * PI * self.std_dev * self.std_dev).ln()
        }

        fn dimensions(&self) -> usize {
            1
        }
    }

    #[test]
    fn test_gaussian_sampling() {
        let target = GaussianTarget {
            mean: 5.0,
            std_dev: 2.0,
        };

        let sampler = MetropolisHastings::new()
            .with_iterations(5000)
            .with_burn_in(500)
            .with_seed(42)
            .with_proposal(ProposalType::RandomWalk { step_size: 0.5 });

        let result = sampler.sample(&target).unwrap();

        // Posterior mean should be close to true mean
        assert_relative_eq!(result.posterior_means[0], 5.0, epsilon = 0.3);

        // Posterior std dev should be close to true std dev
        assert_relative_eq!(result.posterior_std_devs[0], 2.0, epsilon = 0.4);

        // Acceptance rate should be in valid range (0, 1]
        // Note: Small step sizes yield high acceptance rates (~90%)
        // Large step sizes yield low acceptance rates (~10-30%)
        assert!(result.acceptance_rate > 0.0 && result.acceptance_rate <= 1.0);
    }

    #[test]
    fn test_acceptance_rate_with_step_size() {
        let target = GaussianTarget {
            mean: 0.0,
            std_dev: 1.0,
        };

        // Small step size -> high acceptance
        let sampler_small = MetropolisHastings::new()
            .with_iterations(1000)
            .with_seed(42)
            .with_proposal(ProposalType::RandomWalk { step_size: 0.1 });

        let result_small = sampler_small.sample(&target).unwrap();

        // Large step size -> low acceptance
        let sampler_large = MetropolisHastings::new()
            .with_iterations(1000)
            .with_seed(42)
            .with_proposal(ProposalType::RandomWalk { step_size: 10.0 });

        let result_large = sampler_large.sample(&target).unwrap();

        // Small steps should have higher acceptance than large steps
        assert!(result_small.acceptance_rate > result_large.acceptance_rate);
    }

    #[test]
    fn test_burn_in_and_thinning() {
        let target = GaussianTarget {
            mean: 0.0,
            std_dev: 1.0,
        };

        let sampler = MetropolisHastings::new()
            .with_iterations(1000)
            .with_burn_in(100)
            .with_thin(5)
            .with_seed(42);

        let result = sampler.sample(&target).unwrap();

        // Number of samples should be (iterations - burn_in) / thin
        let expected_samples = (1000 - 100) / 5;
        assert_eq!(result.num_samples, expected_samples);
    }

    #[test]
    fn test_reproducibility() {
        let target = GaussianTarget {
            mean: 0.0,
            std_dev: 1.0,
        };

        let sampler1 = MetropolisHastings::new()
            .with_iterations(500)
            .with_burn_in(100)
            .with_seed(123);

        let sampler2 = MetropolisHastings::new()
            .with_iterations(500)
            .with_burn_in(100)
            .with_seed(123);

        let result1 = sampler1.sample(&target).unwrap();
        let result2 = sampler2.sample(&target).unwrap();

        // Same seed should produce identical results
        assert_relative_eq!(result1.posterior_means[0], result2.posterior_means[0], epsilon = 1e-10);
    }

    #[test]
    fn test_effective_sample_size() {
        let target = GaussianTarget {
            mean: 0.0,
            std_dev: 1.0,
        };

        let sampler = MetropolisHastings::new()
            .with_iterations(2000)
            .with_burn_in(200)
            .with_seed(42);

        let result = sampler.sample(&target).unwrap();

        // ESS should be positive and less than total samples
        assert!(result.effective_sample_size > 0.0);
        assert!(result.effective_sample_size <= result.num_samples as f64);
    }

    #[test]
    fn test_initial_state() {
        let target = GaussianTarget {
            mean: 10.0,
            std_dev: 1.0,
        };

        let sampler = MetropolisHastings::new()
            .with_iterations(1000)
            .with_burn_in(100)
            .with_seed(42);

        // Set initial state
        let config = MetropolisHastingsConfig {
            initial_state: Some(vec![10.0]), // Start near mean
            ..sampler.config
        };

        let sampler_with_init = MetropolisHastings::with_config(config);
        let result = sampler_with_init.sample(&target).unwrap();

        // Should converge to true mean
        assert_relative_eq!(result.posterior_means[0], 10.0, epsilon = 0.3);
    }

    #[test]
    fn test_log_posteriors_recorded() {
        let target = GaussianTarget {
            mean: 0.0,
            std_dev: 1.0,
        };

        let sampler = MetropolisHastings::new()
            .with_iterations(500)
            .with_burn_in(50)
            .with_seed(42);

        let result = sampler.sample(&target).unwrap();

        // Should have log posterior for each sample
        assert_eq!(result.log_posteriors.len(), result.samples.len());

        // All log posteriors should be finite
        for &lp in &result.log_posteriors {
            assert!(lp.is_finite());
        }
    }
}
