//! Dirichlet-Multinomial Model for Bayesian inference
//!
//! Implements Bayesian inference for categorical/multinomial data using the
//! Dirichlet-Multinomial conjugate prior model.
//!
//! # Mathematical Foundation
//!
//! **Prior Distribution** (Dirichlet):
//! ```text
//! p(θ|α) = [Γ(Σαᵢ) / Πᵢ Γ(αᵢ)] × Πᵢ θᵢ^(αᵢ-1)
//! ```
//!
//! **Likelihood** (Multinomial):
//! ```text
//! p(x|θ) = (n! / Πᵢ xᵢ!) × Πᵢ θᵢ^xᵢ
//! ```
//!
//! **Posterior Distribution** (Dirichlet):
//! ```text
//! p(θ|x,α) = Dirichlet(α₁ + x₁, α₂ + x₂, ..., αₖ + xₖ)
//! ```
//!
//! **Jeffreys Prior** (non-informative):
//! ```text
//! αᵢ = 0.5  for all i
//! ```
//!
//! **Uniform Prior**:
//! ```text
//! αᵢ = 1.0  for all i
//! ```
//!
//! # References
//!
//! - Gelman, A., et al. (2013). "Bayesian Data Analysis" (3rd ed.)
//! - Murphy, K.P. (2012). "Machine Learning: A Probabilistic Perspective"
//! - Bishop, C.M. (2006). "Pattern Recognition and Machine Learning"
//! - Frigyik, B.A., Kapila, A., & Gupta, M.R. (2010). "Introduction to the Dirichlet Distribution"
//! - Minka, T. (2000). "Estimating a Dirichlet distribution"

use stochastic_core::{AnalysisResult, StochasticAnalyzer, StochasticError, TimeSeries, Domain};
use statrs::distribution::{Dirichlet, Continuous};
use statrs::function::gamma::ln_gamma;
use rand::Rng;
use rand::distributions::Distribution;
use std::collections::HashMap;

/// Prior distribution type for Dirichlet
#[derive(Debug, Clone)]
pub enum PriorType {
    /// Uniform prior: αᵢ = 1.0 (no prior knowledge)
    Uniform,
    /// Jeffreys prior: αᵢ = 0.5 (non-informative)
    Jeffreys,
    /// Custom concentrations for each category
    Custom(Vec<f64>),
}

/// Configuration for Dirichlet-Multinomial model
#[derive(Debug, Clone)]
pub struct DirichletMultinomialConfig {
    /// Prior distribution
    pub prior: PriorType,
    /// Number of posterior samples to draw
    pub num_samples: usize,
    /// Confidence level for credible intervals
    pub confidence_level: f64,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
}

impl Default for DirichletMultinomialConfig {
    fn default() -> Self {
        Self {
            prior: PriorType::Uniform,
            num_samples: 10000,
            confidence_level: 0.95,
            seed: None,
        }
    }
}

/// Dirichlet-Multinomial Bayesian model
pub struct DirichletMultinomial {
    config: DirichletMultinomialConfig,
}

impl DirichletMultinomial {
    /// Create new model with default configuration
    pub fn new() -> Self {
        Self {
            config: DirichletMultinomialConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: DirichletMultinomialConfig) -> Self {
        Self { config }
    }

    /// Set prior distribution
    pub fn with_prior(mut self, prior: PriorType) -> Self {
        self.config.prior = prior;
        self
    }

    /// Set number of posterior samples
    pub fn with_samples(mut self, num_samples: usize) -> Self {
        self.config.num_samples = num_samples;
        self
    }

    /// Set confidence level
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.config.confidence_level = confidence;
        self
    }

    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.config.seed = Some(seed);
        self
    }

    /// Get prior concentrations for k categories
    fn get_prior_concentrations(&self, k: usize) -> Vec<f64> {
        match &self.config.prior {
            PriorType::Uniform => vec![1.0; k],
            PriorType::Jeffreys => vec![0.5; k],
            PriorType::Custom(alpha) => {
                if alpha.len() != k {
                    vec![1.0; k] // Fallback to uniform
                } else {
                    alpha.clone()
                }
            }
        }
    }

    /// Compute posterior concentrations (Bayesian update)
    fn posterior_concentrations(&self, prior_alpha: &[f64], counts: &[usize]) -> Vec<f64> {
        prior_alpha
            .iter()
            .zip(counts.iter())
            .map(|(&alpha, &count)| alpha + count as f64)
            .collect()
    }

    /// Compute log evidence (marginal likelihood) for model comparison
    fn log_evidence(&self, prior_alpha: &[f64], counts: &[usize]) -> f64 {
        let posterior_alpha = self.posterior_concentrations(prior_alpha, counts);
        let n: usize = counts.iter().sum();

        // Log-evidence = log B(α+x) - log B(α) + log multinomial coefficient
        let log_beta_prior: f64 = prior_alpha.iter().map(|&a| ln_gamma(a)).sum::<f64>()
            - ln_gamma(prior_alpha.iter().sum::<f64>());

        let log_beta_posterior: f64 = posterior_alpha.iter().map(|&a| ln_gamma(a)).sum::<f64>()
            - ln_gamma(posterior_alpha.iter().sum::<f64>());

        // Multinomial coefficient: log(n! / Πᵢ xᵢ!)
        let log_multinomial_coef = ln_gamma((n + 1) as f64)
            - counts.iter().map(|&x| ln_gamma((x + 1) as f64)).sum::<f64>();

        log_beta_posterior - log_beta_prior + log_multinomial_coef
    }

    /// Sample from posterior Dirichlet distribution
    fn sample_posterior(&self, posterior_alpha: &[f64], num_samples: usize) -> Vec<Vec<f64>> {
        let mut rng = if let Some(seed) = self.config.seed {
            rand::rngs::StdRng::seed_from_u64(seed)
        } else {
            rand::rngs::StdRng::from_entropy()
        };

        let dirichlet = Dirichlet::new(posterior_alpha)
            .expect("Failed to create Dirichlet distribution");

        (0..num_samples)
            .map(|_| dirichlet.sample(&mut rng))
            .collect()
    }

    /// Compute credible intervals from posterior samples
    fn credible_intervals(&self, samples: &[Vec<f64>], k: usize) -> Vec<(f64, f64, f64)> {
        let alpha = (1.0 - self.config.confidence_level) / 2.0;
        let lower_idx = (alpha * samples.len() as f64) as usize;
        let upper_idx = ((1.0 - alpha) * samples.len() as f64) as usize;

        (0..k)
            .map(|i| {
                let mut category_samples: Vec<f64> = samples.iter()
                    .map(|s| s[i])
                    .collect();
                category_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());

                let mean = category_samples.iter().sum::<f64>() / category_samples.len() as f64;
                let lower = category_samples[lower_idx];
                let upper = category_samples[upper_idx];

                (lower, mean, upper)
            })
            .collect()
    }

    /// Perform Bayesian inference
    pub fn infer(&self, data: &TimeSeries) -> Result<AnalysisResult, StochasticError> {
        self.validate(data)?;

        // Count frequencies
        let mut category_counts: HashMap<i64, usize> = HashMap::new();
        for &value in &data.values() {
            let category = value.round() as i64;
            *category_counts.entry(category).or_insert(0) += 1;
        }

        // Sort categories for consistent ordering
        let mut categories: Vec<i64> = category_counts.keys().copied().collect();
        categories.sort();
        let k = categories.len();

        let counts: Vec<usize> = categories.iter()
            .map(|cat| category_counts[cat])
            .collect();

        // Set up prior
        let prior_alpha = self.get_prior_concentrations(k);

        // Compute posterior
        let posterior_alpha = self.posterior_concentrations(&prior_alpha, &counts);

        // Compute log evidence
        let log_evidence = self.log_evidence(&prior_alpha, &counts);

        // Sample from posterior
        let samples = self.sample_posterior(&posterior_alpha, self.config.num_samples);

        // Compute credible intervals
        let intervals = self.credible_intervals(&samples, k);

        // Posterior means (also the MAP estimates for Dirichlet)
        let posterior_mean: Vec<f64> = posterior_alpha.iter()
            .map(|&a| a / posterior_alpha.iter().sum::<f64>())
            .collect();

        // Effective sample size (prior strength)
        let effective_prior_n: f64 = prior_alpha.iter().sum();
        let effective_posterior_n: f64 = posterior_alpha.iter().sum();

        // Model uncertainty (posterior variance for each probability)
        let posterior_variance: Vec<f64> = posterior_alpha.iter()
            .map(|&a| {
                let sum_alpha: f64 = posterior_alpha.iter().sum();
                let p = a / sum_alpha;
                p * (1.0 - p) / (sum_alpha + 1.0)
            })
            .collect();

        let interpretation = format!(
            "Bayesian inference completed with {} categories. \
             Posterior updated from prior with effective sample size {:.1} → {:.1}. \
             Log evidence = {:.2}.",
            k, effective_prior_n, effective_posterior_n, log_evidence
        );

        let mut result = AnalysisResult::new(self.name())
            .with_metric("num_categories", k as f64)
            .with_metric("sample_size", data.len() as f64)
            .with_metric("log_evidence", log_evidence)
            .with_metric("effective_prior_n", effective_prior_n)
            .with_metric("effective_posterior_n", effective_posterior_n)
            .with_metric("num_posterior_samples", self.config.num_samples as f64)
            .with_interpretation(interpretation)
            .with_metadata("prior", format!("{:?}", self.config.prior))
            .with_metadata("confidence_level", format!("{:.2}", self.config.confidence_level));

        // Add category-specific metrics
        for (i, &cat) in categories.iter().enumerate() {
            let (lower, mean, upper) = intervals[i];
            result = result
                .with_metric(&format!("category_{}_count", cat), counts[i] as f64)
                .with_metric(&format!("category_{}_posterior_mean", cat), posterior_mean[i])
                .with_metric(&format!("category_{}_credible_lower", cat), lower)
                .with_metric(&format!("category_{}_credible_upper", cat), upper)
                .with_metric(&format!("category_{}_variance", cat), posterior_variance[i]);
        }

        Ok(result)
    }
}

impl Default for DirichletMultinomial {
    fn default() -> Self {
        Self::new()
    }
}

impl StochasticAnalyzer for DirichletMultinomial {
    fn name(&self) -> &str {
        "Dirichlet-Multinomial Bayesian Model"
    }

    fn analyze(&self, data: &TimeSeries) -> Result<AnalysisResult, StochasticError> {
        self.infer(data)
    }

    fn required_sample_size(&self) -> usize {
        // Need sufficient data for meaningful Bayesian inference
        10
    }

    fn validate(&self, data: &TimeSeries) -> Result<bool, StochasticError> {
        if data.len() < self.required_sample_size() {
            return Err(StochasticError::insufficient_data(
                self.required_sample_size(),
                data.len(),
            ));
        }

        // Model works best with discrete/categorical data
        match data.domain {
            Domain::Continuous => {
                return Err(StochasticError::validation(
                    "Dirichlet-Multinomial model is designed for discrete/categorical data. \
                     Consider discretizing continuous data first."
                ));
            }
            _ => {}
        }

        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_uniform_prior() {
        let model = DirichletMultinomial::new();
        let prior = model.get_prior_concentrations(5);

        assert_eq!(prior.len(), 5);
        for &alpha in &prior {
            assert_relative_eq!(alpha, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_jeffreys_prior() {
        let model = DirichletMultinomial::new()
            .with_prior(PriorType::Jeffreys);
        let prior = model.get_prior_concentrations(5);

        assert_eq!(prior.len(), 5);
        for &alpha in &prior {
            assert_relative_eq!(alpha, 0.5, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_custom_prior() {
        let custom_alpha = vec![2.0, 3.0, 1.5];
        let model = DirichletMultinomial::new()
            .with_prior(PriorType::Custom(custom_alpha.clone()));
        let prior = model.get_prior_concentrations(3);

        assert_eq!(prior, custom_alpha);
    }

    #[test]
    fn test_posterior_update() {
        let model = DirichletMultinomial::new();
        let prior = vec![1.0, 1.0, 1.0]; // Uniform
        let counts = vec![5, 3, 2]; // Observed data

        let posterior = model.posterior_concentrations(&prior, &counts);

        assert_eq!(posterior, vec![6.0, 4.0, 3.0]);
    }

    #[test]
    fn test_simple_inference() {
        // Data: 5 zeros, 3 ones, 2 twos
        let mut values = Vec::new();
        values.extend(vec![0.0; 5]);
        values.extend(vec![1.0; 3]);
        values.extend(vec![2.0; 2]);

        let data = TimeSeries::from_values(values).with_domain(Domain::Discrete);
        let model = DirichletMultinomial::new().with_samples(1000).with_seed(42);

        let result = model.analyze(&data).unwrap();

        let num_cats = result.metrics.get("num_categories").unwrap();
        assert_eq!(*num_cats, 3.0);

        // Check posterior means are reasonable
        let mean_0 = result.metrics.get("category_0_posterior_mean").unwrap();
        let mean_1 = result.metrics.get("category_1_posterior_mean").unwrap();
        let mean_2 = result.metrics.get("category_2_posterior_mean").unwrap();

        // Posterior means should reflect observed frequencies
        // 5/(5+3+2) ≈ 0.5, 3/10 = 0.3, 2/10 = 0.2
        assert!(*mean_0 > 0.4 && *mean_0 < 0.6);
        assert!(*mean_1 > 0.2 && *mean_1 < 0.4);
        assert!(*mean_2 > 0.1 && *mean_2 < 0.3);
    }

    #[test]
    fn test_credible_intervals() {
        let values = vec![0.0; 100]; // All same category
        let data = TimeSeries::from_values(values).with_domain(Domain::Discrete);

        let model = DirichletMultinomial::new()
            .with_samples(1000)
            .with_confidence(0.95)
            .with_seed(42);

        let result = model.analyze(&data).unwrap();

        let lower = result.metrics.get("category_0_credible_lower").unwrap();
        let upper = result.metrics.get("category_0_credible_upper").unwrap();
        let mean = result.metrics.get("category_0_posterior_mean").unwrap();

        // Credible interval should contain mean
        assert!(*lower <= *mean && *mean <= *upper);

        // For all data in one category, posterior mean should be close to 1.0
        assert!(*mean > 0.95);
    }

    #[test]
    fn test_log_evidence() {
        let values = vec![0.0, 1.0, 0.0, 1.0];
        let data = TimeSeries::from_values(values).with_domain(Domain::Discrete);

        let model = DirichletMultinomial::new().with_seed(42);
        let result = model.analyze(&data).unwrap();

        let log_ev = result.metrics.get("log_evidence").unwrap();

        // Log evidence should be finite
        assert!(log_ev.is_finite());
    }

    #[test]
    fn test_posterior_variance() {
        let values = vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0];
        let data = TimeSeries::from_values(values).with_domain(Domain::Discrete);

        let model = DirichletMultinomial::new().with_seed(42);
        let result = model.analyze(&data).unwrap();

        let var_0 = result.metrics.get("category_0_variance").unwrap();
        let var_1 = result.metrics.get("category_1_variance").unwrap();
        let var_2 = result.metrics.get("category_2_variance").unwrap();

        // Variances should be positive and reasonable
        assert!(*var_0 > 0.0 && *var_0 < 0.1);
        assert!(*var_1 > 0.0 && *var_1 < 0.1);
        assert!(*var_2 > 0.0 && *var_2 < 0.1);
    }

    #[test]
    fn test_insufficient_data() {
        let data = TimeSeries::from_values(vec![1.0, 2.0]).with_domain(Domain::Discrete);
        let model = DirichletMultinomial::new();

        let result = model.analyze(&data);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("insufficient"));
    }

    #[test]
    fn test_continuous_data_rejection() {
        let data = TimeSeries::from_values(vec![1.5, 2.7, 3.9]).with_domain(Domain::Continuous);
        let model = DirichletMultinomial::new();

        let result = model.analyze(&data);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("discrete"));
    }

    #[test]
    fn test_effective_sample_size() {
        let values = vec![0.0; 50];
        let data = TimeSeries::from_values(values).with_domain(Domain::Discrete);

        let uniform_model = DirichletMultinomial::new().with_seed(42);
        let result_uniform = uniform_model.analyze(&data).unwrap();

        let jeffreys_model = DirichletMultinomial::new()
            .with_prior(PriorType::Jeffreys)
            .with_seed(42);
        let result_jeffreys = jeffreys_model.analyze(&data).unwrap();

        let eff_n_uniform = result_uniform.metrics.get("effective_prior_n").unwrap();
        let eff_n_jeffreys = result_jeffreys.metrics.get("effective_prior_n").unwrap();

        // Uniform: α=1.0, Jeffreys: α=0.5
        // For 1 category: uniform has effective N=1.0, Jeffreys has effective N=0.5
        assert_relative_eq!(*eff_n_uniform, 1.0, epsilon = 0.01);
        assert_relative_eq!(*eff_n_jeffreys, 0.5, epsilon = 0.01);
    }

    #[test]
    fn test_reproducibility_with_seed() {
        let values = vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0];
        let data = TimeSeries::from_values(values.clone()).with_domain(Domain::Discrete);

        let model1 = DirichletMultinomial::new().with_seed(123).with_samples(100);
        let model2 = DirichletMultinomial::new().with_seed(123).with_samples(100);

        let result1 = model1.analyze(&data).unwrap();
        let result2 = model2.analyze(&data).unwrap();

        // Results with same seed should be identical
        let mean1 = result1.metrics.get("category_0_posterior_mean").unwrap();
        let mean2 = result2.metrics.get("category_0_posterior_mean").unwrap();

        assert_relative_eq!(*mean1, *mean2, epsilon = 1e-10);
    }
}
