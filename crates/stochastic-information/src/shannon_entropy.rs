//! Shannon Entropy implementation
//!
//! Measures the average information content (uncertainty) in a random variable.
//! Higher entropy indicates more randomness/unpredictability.
//!
//! # Mathematical Foundation
//!
//! **Discrete Shannon Entropy:**
//! ```text
//! H(X) = -Σᵢ p(xᵢ) log₂ p(xᵢ)
//! ```
//!
//! **Differential Entropy** (continuous):
//! ```text
//! h(X) = -∫ f(x) log f(x) dx
//! ```
//!
//! **Properties:**
//! - H(X) ≥ 0
//! - H(X) ≤ log₂ k (uniform maximum for k outcomes)
//! - H(X) = 0 iff X is deterministic
//!
//! # References
//!
//! - Shannon, C.E. (1948). "A Mathematical Theory of Communication"
//! - Cover, T.M., & Thomas, J.A. (2006). "Elements of Information Theory" (2nd ed.)
//! - Jaynes, E.T. (2003). "Probability Theory: The Logic of Science"
//! - Paninski, L. (2003). "Estimation of Entropy and Mutual Information"
//! - Gao, S., Ver Steeg, G., & Galstyan, A. (2015). "Efficient Estimation of Mutual Information for Strongly Dependent Variables"

use stochastic_core::{AnalysisResult, StochasticAnalyzer, StochasticError, TimeSeries, Domain};
use std::collections::HashMap;

/// Configuration for Shannon Entropy calculation
#[derive(Debug, Clone)]
pub struct ShannonEntropyConfig {
    /// Number of bins for continuous data discretization
    /// If None, uses Sturges' rule: k = ceil(1 + log2(n))
    pub num_bins: Option<usize>,
    /// Base of logarithm (2 = bits, e = nats, 10 = dits)
    pub log_base: LogBase,
    /// Whether to normalize entropy to [0, 1]
    pub normalize: bool,
}

/// Logarithm base for entropy calculation
#[derive(Debug, Clone, Copy)]
pub enum LogBase {
    /// Binary logarithm (bits)
    Two,
    /// Natural logarithm (nats)
    E,
    /// Decimal logarithm (dits/hartleys)
    Ten,
}

impl Default for ShannonEntropyConfig {
    fn default() -> Self {
        Self {
            num_bins: None,
            log_base: LogBase::Two,
            normalize: false,
        }
    }
}

/// Shannon Entropy analyzer
pub struct ShannonEntropy {
    config: ShannonEntropyConfig,
}

impl ShannonEntropy {
    /// Create a new Shannon Entropy analyzer with default configuration
    pub fn new() -> Self {
        Self {
            config: ShannonEntropyConfig::default(),
        }
    }

    /// Create a Shannon Entropy analyzer with custom configuration
    pub fn with_config(config: ShannonEntropyConfig) -> Self {
        Self { config }
    }

    /// Set the number of bins for discretization
    pub fn with_bins(mut self, num_bins: usize) -> Self {
        self.config.num_bins = Some(num_bins);
        self
    }

    /// Set the logarithm base
    pub fn with_log_base(mut self, log_base: LogBase) -> Self {
        self.config.log_base = log_base;
        self
    }

    /// Enable/disable normalization
    pub fn with_normalize(mut self, normalize: bool) -> Self {
        self.config.normalize = normalize;
        self
    }

    /// Compute optimal number of bins using Sturges' rule
    fn optimal_bins(&self, n: usize) -> usize {
        if let Some(bins) = self.config.num_bins {
            return bins;
        }
        // Sturges' rule: k = ceil(1 + log2(n))
        let k = 1.0 + (n as f64).log2();
        k.ceil() as usize
    }

    /// Bin continuous data into discrete probability distribution
    fn bin_data(&self, values: &[f64], num_bins: usize) -> HashMap<usize, f64> {
        let min = values.iter().copied().fold(f64::INFINITY, f64::min);
        let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        let bin_width = if max == min {
            1.0
        } else {
            (max - min) / num_bins as f64
        };

        let mut bin_counts: HashMap<usize, usize> = HashMap::new();
        let n = values.len();

        for &value in values {
            let bin_index = if max == min {
                0
            } else if value == max {
                num_bins - 1
            } else {
                ((value - min) / bin_width).floor() as usize
            };
            *bin_counts.entry(bin_index).or_insert(0) += 1;
        }

        // Convert counts to probabilities
        let mut probabilities = HashMap::new();
        for (bin, count) in bin_counts {
            probabilities.insert(bin, count as f64 / n as f64);
        }

        probabilities
    }

    /// Compute entropy from probability distribution
    fn entropy_from_probabilities(&self, probabilities: &HashMap<usize, f64>) -> f64 {
        let mut entropy = 0.0;

        for &p in probabilities.values() {
            if p > 0.0 {
                let log_p = match self.config.log_base {
                    LogBase::Two => p.log2(),
                    LogBase::E => p.ln(),
                    LogBase::Ten => p.log10(),
                };
                entropy -= p * log_p;
            }
        }

        entropy
    }

    /// Get logarithm base value
    fn get_log_base_value(&self) -> f64 {
        match self.config.log_base {
            LogBase::Two => 2.0,
            LogBase::E => std::f64::consts::E,
            LogBase::Ten => 10.0,
        }
    }

    /// Compute maximum possible entropy (uniform distribution)
    fn max_entropy(&self, num_bins: usize) -> f64 {
        match self.config.log_base {
            LogBase::Two => (num_bins as f64).log2(),
            LogBase::E => (num_bins as f64).ln(),
            LogBase::Ten => (num_bins as f64).log10(),
        }
    }

    /// Calculate Shannon Entropy
    pub fn calculate(&self, data: &TimeSeries) -> Result<AnalysisResult, StochasticError> {
        self.validate(data)?;

        let values = data.values();
        let n = values.len();

        let (entropy, num_bins, probabilities) = match data.domain {
            Domain::Discrete | Domain::Categorical => {
                // For discrete data, use exact value counts
                let mut value_counts: HashMap<i64, usize> = HashMap::new();
                for &v in &values {
                    let discrete_val = v.round() as i64;
                    *value_counts.entry(discrete_val).or_insert(0) += 1;
                }

                let num_unique = value_counts.len();

                // Convert to probabilities
                let mut probs = HashMap::new();
                for (i, (_, count)) in value_counts.iter().enumerate() {
                    probs.insert(i, *count as f64 / n as f64);
                }

                let h = self.entropy_from_probabilities(&probs);
                (h, num_unique, probs)
            }
            Domain::Continuous => {
                // For continuous data, discretize into bins
                let num_bins = self.optimal_bins(n);
                let probs = self.bin_data(&values, num_bins);
                let h = self.entropy_from_probabilities(&probs);
                (h, num_bins, probs)
            }
        };

        // Normalize if requested
        let normalized_entropy = if self.config.normalize {
            let max_h = self.max_entropy(num_bins);
            if max_h > 0.0 {
                entropy / max_h
            } else {
                0.0
            }
        } else {
            entropy
        };

        // Compute additional metrics
        let max_entropy = self.max_entropy(num_bins);
        let redundancy = if max_entropy > 0.0 {
            1.0 - (entropy / max_entropy)
        } else {
            0.0
        };

        // Efficiency (0 = minimum info, 1 = maximum info)
        let efficiency = if max_entropy > 0.0 {
            entropy / max_entropy
        } else {
            0.0
        };

        // Compute Gini-Simpson index (diversity measure)
        let gini_simpson: f64 = probabilities.values().map(|&p| p * p).sum();
        let diversity = 1.0 - gini_simpson;

        let log_base_str = match self.config.log_base {
            LogBase::Two => "bits",
            LogBase::E => "nats",
            LogBase::Ten => "dits",
        };

        let interpretation = if entropy < 0.5 * max_entropy {
            format!(
                "Low entropy ({:.3} {}) indicates low randomness/high predictability. \
                 System has significant order (redundancy = {:.1}%)",
                entropy,
                log_base_str,
                redundancy * 100.0
            )
        } else if entropy > 0.9 * max_entropy {
            format!(
                "High entropy ({:.3} {}) indicates high randomness/unpredictability. \
                 System approaches maximum disorder (efficiency = {:.1}%)",
                entropy,
                log_base_str,
                efficiency * 100.0
            )
        } else {
            format!(
                "Medium entropy ({:.3} {}) indicates moderate randomness. \
                 System has balance between order and disorder (efficiency = {:.1}%)",
                entropy,
                log_base_str,
                efficiency * 100.0
            )
        };

        let mut result = AnalysisResult::new(self.name())
            .with_metric("entropy", entropy)
            .with_metric("normalized_entropy", normalized_entropy)
            .with_metric("max_entropy", max_entropy)
            .with_metric("redundancy", redundancy)
            .with_metric("efficiency", efficiency)
            .with_metric("diversity", diversity)
            .with_metric("gini_simpson_index", gini_simpson)
            .with_metric("num_bins", num_bins as f64)
            .with_metric("sample_size", n as f64)
            .with_interpretation(interpretation)
            .with_metadata("log_base", format!("{:?}", self.config.log_base))
            .with_metadata("units", log_base_str);

        // Add probability distribution as metadata
        if probabilities.len() <= 20 {
            // Only include if not too large
            let prob_str: Vec<String> = probabilities
                .iter()
                .map(|(k, v)| format!("bin_{}: {:.4}", k, v))
                .collect();
            result = result.with_metadata("probability_distribution", prob_str.join(", "));
        }

        Ok(result)
    }
}

impl Default for ShannonEntropy {
    fn default() -> Self {
        Self::new()
    }
}

impl StochasticAnalyzer for ShannonEntropy {
    fn name(&self) -> &str {
        "Shannon Entropy"
    }

    fn analyze(&self, data: &TimeSeries) -> Result<AnalysisResult, StochasticError> {
        self.calculate(data)
    }

    fn required_sample_size(&self) -> usize {
        // Entropy estimation requires at least 10 samples
        // More samples provide better estimates
        10
    }

    fn validate(&self, data: &TimeSeries) -> Result<bool, StochasticError> {
        if data.len() < self.required_sample_size() {
            return Err(StochasticError::insufficient_data(
                self.required_sample_size(),
                data.len(),
            ));
        }

        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_uniform_distribution_max_entropy() {
        // Uniform distribution should have maximum entropy
        let values: Vec<f64> = (0..100).map(|i| (i % 10) as f64).collect();
        let data = TimeSeries::from_values(values).with_domain(Domain::Discrete);

        let analyzer = ShannonEntropy::new();
        let result = analyzer.analyze(&data).unwrap();

        let entropy = result.metrics.get("entropy").unwrap();
        let max_entropy = result.metrics.get("max_entropy").unwrap();

        // Uniform distribution over 10 values should have entropy ≈ log2(10) ≈ 3.32
        assert_relative_eq!(*entropy, *max_entropy, epsilon = 0.01);
        assert_relative_eq!(*entropy, 10.0_f64.log2(), epsilon = 0.01);
    }

    #[test]
    fn test_deterministic_zero_entropy() {
        // Constant data should have zero entropy
        let values = vec![5.0; 100];
        let data = TimeSeries::from_values(values);

        let analyzer = ShannonEntropy::new();
        let result = analyzer.analyze(&data).unwrap();

        let entropy = result.metrics.get("entropy").unwrap();
        assert_relative_eq!(*entropy, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_binary_distribution() {
        // Binary distribution with equal probabilities
        let mut values = Vec::new();
        for _ in 0..50 {
            values.push(0.0);
        }
        for _ in 0..50 {
            values.push(1.0);
        }
        let data = TimeSeries::from_values(values).with_domain(Domain::Discrete);

        let analyzer = ShannonEntropy::new();
        let result = analyzer.analyze(&data).unwrap();

        let entropy = result.metrics.get("entropy").unwrap();

        // Binary with p=0.5 should have entropy = 1 bit
        assert_relative_eq!(*entropy, 1.0, epsilon = 0.01);
    }

    #[test]
    fn test_biased_binary() {
        // Biased binary: 90% zeros, 10% ones
        let mut values = Vec::new();
        for _ in 0..90 {
            values.push(0.0);
        }
        for _ in 0..10 {
            values.push(1.0);
        }
        let data = TimeSeries::from_values(values).with_domain(Domain::Discrete);

        let analyzer = ShannonEntropy::new();
        let result = analyzer.analyze(&data).unwrap();

        let entropy = result.metrics.get("entropy").unwrap();

        // H(0.9, 0.1) = -0.9*log2(0.9) - 0.1*log2(0.1) ≈ 0.469
        let expected = -0.9 * 0.9_f64.log2() - 0.1 * 0.1_f64.log2();
        assert_relative_eq!(*entropy, expected, epsilon = 0.01);
    }

    #[test]
    fn test_log_base_nats() {
        let values = vec![0.0, 1.0, 0.0, 1.0]; // Binary
        let data = TimeSeries::from_values(values).with_domain(Domain::Discrete);

        let config = ShannonEntropyConfig {
            num_bins: None,
            log_base: LogBase::E,
            normalize: false,
        };
        let analyzer = ShannonEntropy::with_config(config);
        let result = analyzer.analyze(&data).unwrap();

        let entropy = result.metrics.get("entropy").unwrap();

        // Binary with p=0.5 in nats = ln(2) ≈ 0.693
        assert_relative_eq!(*entropy, 2.0_f64.ln(), epsilon = 0.01);
    }

    #[test]
    fn test_normalization() {
        let values: Vec<f64> = (0..100).map(|i| (i % 5) as f64).collect();
        let data = TimeSeries::from_values(values).with_domain(Domain::Discrete);

        let config = ShannonEntropyConfig {
            num_bins: None,
            log_base: LogBase::Two,
            normalize: true,
        };
        let analyzer = ShannonEntropy::with_config(config);
        let result = analyzer.analyze(&data).unwrap();

        let normalized = result.metrics.get("normalized_entropy").unwrap();

        // Uniform distribution should have normalized entropy = 1.0
        assert_relative_eq!(*normalized, 1.0, epsilon = 0.01);
    }

    #[test]
    fn test_redundancy() {
        // Constant data has maximum redundancy
        let values = vec![1.0; 50];
        let data = TimeSeries::from_values(values);

        let analyzer = ShannonEntropy::new();
        let result = analyzer.analyze(&data).unwrap();

        let redundancy = result.metrics.get("redundancy").unwrap();
        assert_relative_eq!(*redundancy, 1.0, epsilon = 0.01);
    }

    #[test]
    fn test_diversity_gini() {
        // Uniform distribution has maximum diversity
        let values: Vec<f64> = (0..100).map(|i| (i % 10) as f64).collect();
        let data = TimeSeries::from_values(values).with_domain(Domain::Discrete);

        let analyzer = ShannonEntropy::new();
        let result = analyzer.analyze(&data).unwrap();

        let diversity = result.metrics.get("diversity").unwrap();
        let gini = result.metrics.get("gini_simpson_index").unwrap();

        // For uniform p=0.1 over 10 values:
        // Gini-Simpson = 10 * (0.1)^2 = 0.1
        // Diversity = 1 - 0.1 = 0.9
        assert_relative_eq!(*gini, 0.1, epsilon = 0.01);
        assert_relative_eq!(*diversity, 0.9, epsilon = 0.01);
    }

    #[test]
    fn test_continuous_binning() {
        // Continuous uniform data
        let values: Vec<f64> = (0..1000).map(|i| i as f64 / 10.0).collect();
        let data = TimeSeries::from_values(values).with_domain(Domain::Continuous);

        let analyzer = ShannonEntropy::new().with_bins(10);
        let result = analyzer.analyze(&data).unwrap();

        let entropy = result.metrics.get("entropy").unwrap();
        let num_bins = result.metrics.get("num_bins").unwrap();

        assert_eq!(*num_bins, 10.0);
        // Should be close to log2(10) for uniform distribution
        assert!(*entropy > 3.0 && *entropy < 3.5);
    }

    #[test]
    fn test_sturges_rule() {
        let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let data = TimeSeries::from_values(values).with_domain(Domain::Continuous);

        let analyzer = ShannonEntropy::new(); // No bins specified
        let result = analyzer.analyze(&data).unwrap();

        let num_bins = result.metrics.get("num_bins").unwrap();

        // Sturges for n=100: k = ceil(1 + log2(100)) = ceil(7.64) = 8
        assert_eq!(*num_bins, 8.0);
    }

    #[test]
    fn test_insufficient_data() {
        let data = TimeSeries::from_values(vec![1.0, 2.0]);
        let analyzer = ShannonEntropy::new();

        let result = analyzer.analyze(&data);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("insufficient"));
    }
}
