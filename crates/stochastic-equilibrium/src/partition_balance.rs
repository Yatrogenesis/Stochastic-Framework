//! Partition Balance Analysis
//!
//! Implements chi-squared goodness-of-fit tests for assessing balance across range partitions.
//! Divides the data range into equal-width bins and tests whether observations are uniformly
//! distributed across partitions, detecting systematic biases in the generating process.
//!
//! # Mathematical Foundation
//!
//! ## Range Partitioning
//!
//! Given n observations {x₁, x₂, ..., xₙ} with range [min, max], divide into m partitions:
//!
//! ```text
//! Partition j: [min + (j-1)w, min + jw)  for j = 1, 2, ..., m
//! where w = (max - min) / m
//! ```
//!
//! ## Chi-Squared Balance Statistic
//!
//! The balance test statistic is:
//!
//! ```text
//! χ²_balance = Σⱼ₌₁ᵐ (nⱼ - n/m)² / (n/m)
//! ```
//!
//! where:
//! - nⱼ = observed count in partition j
//! - n/m = expected count per partition (uniform distribution)
//! - m = number of partitions
//!
//! Under H₀ (uniform distribution), χ²_balance ~ χ²(m-1)
//!
//! ## Partition Quality Metrics
//!
//! - **Balance coefficient**: BC = 1 - (max_count - min_count) / n
//! - **Partition entropy**: H = -Σⱼ (nⱼ/n) log(nⱼ/n)
//! - **Normalized entropy**: H_norm = H / log(m) ∈ [0, 1]
//!
//! ## Interpretation
//!
//! - χ²_balance small, p > 0.05: Data is balanced across partitions
//! - χ²_balance large, p < 0.05: Significant imbalance detected
//! - H_norm ≈ 1: Maximum entropy (uniform distribution)
//! - H_norm < 0.9: Potential clustering or bias
//!
//! # References
//!
//! - NIST SP 800-22 Rev. 1a (2010). "A Statistical Test Suite for Random Number Generators
//!   and Testing for Randomness." Section 2.3: Frequency Test within a Block.
//! - Knuth, D.E. (1998). "The Art of Computer Programming, Vol. 2: Seminumerical Algorithms."
//!   3rd ed., Addison-Wesley. Section 3.3.1: Chi-Square Test.
//! - Marsaglia, G. (1996). "DIEHARD: A Battery of Tests of Randomness."
//! - Rukhin, A., et al. (2010). "Statistical Testing of Random Number Generators."
//!   NIST Special Publication.
//! - Gentle, J.E. (2003). "Random Number Generation and Monte Carlo Methods."
//!   2nd ed., Springer. Section 7.3: Testing for Randomness.

use statrs::distribution::{ChiSquared, ContinuousCDF};
use stochastic_core::{AnalysisResult, StochasticAnalyzer, StochasticError, TimeSeries};

/// Configuration for partition balance analysis
#[derive(Debug, Clone)]
pub struct PartitionConfig {
    /// Number of partitions to divide the range into (default: 10)
    pub num_partitions: usize,
    /// Minimum observations per partition for valid chi-squared test (default: 5)
    pub min_observations_per_partition: usize,
    /// Significance level for hypothesis testing (default: 0.05)
    pub significance_level: f64,
}

impl Default for PartitionConfig {
    fn default() -> Self {
        Self {
            num_partitions: 10,
            min_observations_per_partition: 5,
            significance_level: 0.05,
        }
    }
}

impl PartitionConfig {
    /// Create new configuration with defaults
    pub fn new() -> Self {
        Self::default()
    }

    /// Set number of partitions
    pub fn with_num_partitions(mut self, partitions: usize) -> Self {
        self.num_partitions = partitions.max(2);
        self
    }

    /// Set minimum observations per partition
    pub fn with_min_observations(mut self, min_obs: usize) -> Self {
        self.min_observations_per_partition = min_obs.max(1);
        self
    }

    /// Set significance level
    pub fn with_significance_level(mut self, alpha: f64) -> Self {
        self.significance_level = alpha.clamp(0.001, 0.5);
        self
    }
}

/// Partition Balance Analyzer
pub struct PartitionBalanceAnalyzer {
    config: PartitionConfig,
}

impl PartitionBalanceAnalyzer {
    /// Create new analyzer with default configuration
    pub fn new() -> Self {
        Self {
            config: PartitionConfig::default(),
        }
    }

    /// Create analyzer with custom configuration
    pub fn with_config(config: PartitionConfig) -> Self {
        Self { config }
    }

    /// Set number of partitions
    pub fn with_num_partitions(mut self, partitions: usize) -> Self {
        self.config.num_partitions = partitions.max(2);
        self
    }

    /// Compute range partitions and bin counts
    fn compute_partitions(&self, data: &[f64]) -> (Vec<f64>, Vec<usize>, f64, f64) {
        let _n = data.len();
        let m = self.config.num_partitions;

        // Find data range
        let min_val = data.iter().copied().fold(f64::INFINITY, f64::min);
        let max_val = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        // Add small epsilon to max to make interval closed on right
        let epsilon = (max_val - min_val) * 1e-10 + 1e-10;
        let adjusted_max = max_val + epsilon;

        // Compute partition width
        let width = (adjusted_max - min_val) / m as f64;

        // Initialize partition boundaries and counts
        let mut boundaries = Vec::with_capacity(m + 1);
        for i in 0..=m {
            boundaries.push(min_val + i as f64 * width);
        }

        let mut counts = vec![0; m];

        // Bin the data
        for &value in data {
            let partition_idx = if value >= adjusted_max {
                m - 1
            } else {
                ((value - min_val) / width).floor() as usize
            };
            let partition_idx = partition_idx.min(m - 1);
            counts[partition_idx] += 1;
        }

        (boundaries, counts, min_val, max_val)
    }

    /// Compute chi-squared balance statistic
    fn compute_chi_squared(&self, counts: &[usize], n: usize) -> f64 {
        let m = counts.len();
        let expected = n as f64 / m as f64;

        counts
            .iter()
            .map(|&observed| {
                let diff = observed as f64 - expected;
                (diff * diff) / expected
            })
            .sum()
    }

    /// Compute p-value for chi-squared statistic
    fn compute_p_value(&self, chi_squared: f64, degrees_of_freedom: usize) -> Result<f64, StochasticError> {
        if degrees_of_freedom == 0 {
            return Err(StochasticError::numerical(
                "Degrees of freedom must be positive".to_string(),
            ));
        }

        let dist = ChiSquared::new(degrees_of_freedom as f64)
            .map_err(|e| StochasticError::numerical(format!("Failed to create chi-squared distribution: {}", e)))?;

        Ok(1.0 - dist.cdf(chi_squared))
    }

    /// Compute balance coefficient
    fn compute_balance_coefficient(&self, counts: &[usize], n: usize) -> f64 {
        if counts.is_empty() {
            return 0.0;
        }

        let max_count = *counts.iter().max().unwrap() as f64;
        let min_count = *counts.iter().min().unwrap() as f64;

        1.0 - (max_count - min_count) / n as f64
    }

    /// Compute Shannon entropy of partition distribution
    fn compute_entropy(&self, counts: &[usize], n: usize) -> f64 {
        let n_f64 = n as f64;

        counts
            .iter()
            .filter(|&&count| count > 0)
            .map(|&count| {
                let p = count as f64 / n_f64;
                -p * p.ln()
            })
            .sum()
    }

    /// Compute normalized entropy (0 to 1)
    fn compute_normalized_entropy(&self, entropy: f64, num_partitions: usize) -> f64 {
        if num_partitions <= 1 {
            return 0.0;
        }

        let max_entropy = (num_partitions as f64).ln();
        if max_entropy == 0.0 {
            0.0
        } else {
            entropy / max_entropy
        }
    }

    /// Compute partition variance
    fn compute_partition_variance(&self, counts: &[usize], expected: f64) -> f64 {
        if counts.is_empty() {
            return 0.0;
        }

        let sum_sq_diff: f64 = counts
            .iter()
            .map(|&count| {
                let diff = count as f64 - expected;
                diff * diff
            })
            .sum();

        sum_sq_diff / counts.len() as f64
    }

    /// Validate configuration and data
    fn validate_config(&self, data_len: usize) -> Result<(), StochasticError> {
        let m = self.config.num_partitions;
        let min_obs = self.config.min_observations_per_partition;

        // Check minimum data requirement
        let required_min = m * min_obs;
        if data_len < required_min {
            return Err(StochasticError::validation(format!(
                "Insufficient data for {} partitions with {} min observations each. Need {} points, have {}",
                m, min_obs, required_min, data_len
            )));
        }

        // Check partition count is reasonable
        if m < 2 {
            return Err(StochasticError::validation(
                "Number of partitions must be at least 2".to_string(),
            ));
        }

        if m > data_len / 2 {
            return Err(StochasticError::validation(format!(
                "Too many partitions ({}) for data size ({}). Use at most {}",
                m,
                data_len,
                data_len / 2
            )));
        }

        Ok(())
    }

    /// Generate interpretation of results
    fn interpret_results(&self, p_value: f64, normalized_entropy: f64, balance_coeff: f64) -> String {
        let alpha = self.config.significance_level;

        let balance_assessment = if p_value > alpha {
            format!(
                "BALANCED: Data is uniformly distributed across partitions (p = {:.4} > α = {:.4})",
                p_value, alpha
            )
        } else {
            format!(
                "IMBALANCED: Significant deviation from uniform distribution detected (p = {:.4} < α = {:.4})",
                p_value, alpha
            )
        };

        let entropy_assessment = if normalized_entropy > 0.95 {
            "Excellent entropy (near-maximum uniformity)"
        } else if normalized_entropy > 0.85 {
            "Good entropy (acceptable uniformity)"
        } else if normalized_entropy > 0.70 {
            "Moderate entropy (some clustering detected)"
        } else {
            "Low entropy (significant clustering or bias)"
        };

        let balance_assessment_detailed = if balance_coeff > 0.9 {
            "Excellent balance"
        } else if balance_coeff > 0.75 {
            "Good balance"
        } else if balance_coeff > 0.5 {
            "Moderate imbalance"
        } else {
            "Significant imbalance"
        };

        format!(
            "{}\n{} (H_norm = {:.4}, BC = {:.4}).\n{}",
            balance_assessment, entropy_assessment, normalized_entropy, balance_coeff, balance_assessment_detailed
        )
    }
}

impl Default for PartitionBalanceAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl StochasticAnalyzer for PartitionBalanceAnalyzer {
    fn name(&self) -> &str {
        "Partition Balance (Chi-Squared)"
    }

    fn analyze(&self, data: &TimeSeries) -> Result<AnalysisResult, StochasticError> {
        self.validate(data)?;

        let values = data.values();
        let n = values.len();
        let m = self.config.num_partitions;

        // Validate configuration
        self.validate_config(n)?;

        // Compute partitions and counts
        let (_boundaries, counts, min_val, max_val) = self.compute_partitions(&values);

        // Verify minimum observations per partition
        let min_count = *counts.iter().min().unwrap();
        if min_count < self.config.min_observations_per_partition {
            return Err(StochasticError::validation(format!(
                "Partition with only {} observations (minimum required: {}). Consider reducing num_partitions or increasing data size.",
                min_count, self.config.min_observations_per_partition
            )));
        }

        // Compute chi-squared statistic
        let chi_squared = self.compute_chi_squared(&counts, n);
        let df = m - 1;
        let p_value = self.compute_p_value(chi_squared, df)?;

        // Compute additional metrics
        let expected = n as f64 / m as f64;
        let entropy = self.compute_entropy(&counts, n);
        let normalized_entropy = self.compute_normalized_entropy(entropy, m);
        let balance_coeff = self.compute_balance_coefficient(&counts, n);
        let partition_variance = self.compute_partition_variance(&counts, expected);

        // Generate interpretation
        let interpretation = self.interpret_results(p_value, normalized_entropy, balance_coeff);

        // Build result
        Ok(AnalysisResult::new(self.name())
            .with_metric("chi_squared", chi_squared)
            .with_metric("degrees_of_freedom", df as f64)
            .with_p_value(p_value)
            .with_metric("num_partitions", m as f64)
            .with_metric("entropy", entropy)
            .with_metric("normalized_entropy", normalized_entropy)
            .with_metric("balance_coefficient", balance_coeff)
            .with_metric("partition_variance", partition_variance)
            .with_metric("expected_per_partition", expected)
            .with_metric("min_count", min_count as f64)
            .with_metric("max_count", *counts.iter().max().unwrap() as f64)
            .with_metric("data_range", max_val - min_val)
            .with_metadata("range_min", min_val.to_string())
            .with_metadata("range_max", max_val.to_string())
            .with_metadata("partition_width", ((max_val - min_val) / m as f64).to_string())
            .with_interpretation(interpretation))
    }

    fn required_sample_size(&self) -> usize {
        // Need at least min_observations per partition
        self.config.num_partitions * self.config.min_observations_per_partition
    }

    fn validate(&self, data: &TimeSeries) -> Result<bool, StochasticError> {
        let required = self.required_sample_size();
        if data.len() < required {
            return Err(StochasticError::insufficient_data(required, data.len()));
        }

        let values = data.values();

        // Check for NaN or infinite values
        if values.iter().any(|v| !v.is_finite()) {
            return Err(StochasticError::validation(
                "Data contains NaN or infinite values".to_string(),
            ));
        }

        // Check for constant data
        let first = values[0];
        if values.iter().all(|&v| (v - first).abs() < 1e-10) {
            return Err(StochasticError::validation(
                "Data is constant, cannot perform partition analysis".to_string(),
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
    fn test_config_defaults() {
        let config = PartitionConfig::default();
        assert_eq!(config.num_partitions, 10);
        assert_eq!(config.min_observations_per_partition, 5);
        assert_eq!(config.significance_level, 0.05);
    }

    #[test]
    fn test_config_builder() {
        let config = PartitionConfig::new()
            .with_num_partitions(20)
            .with_min_observations(10)
            .with_significance_level(0.01);

        assert_eq!(config.num_partitions, 20);
        assert_eq!(config.min_observations_per_partition, 10);
        assert_eq!(config.significance_level, 0.01);
    }

    #[test]
    fn test_analyzer_creation() {
        let analyzer = PartitionBalanceAnalyzer::new();
        assert_eq!(analyzer.name(), "Partition Balance (Chi-Squared)");
        assert_eq!(analyzer.config.num_partitions, 10);
    }

    #[test]
    fn test_analyzer_with_config() {
        let config = PartitionConfig::new().with_num_partitions(15);
        let analyzer = PartitionBalanceAnalyzer::with_config(config);
        assert_eq!(analyzer.config.num_partitions, 15);
    }

    #[test]
    fn test_uniform_distribution() {
        // Uniformly distributed data should have high p-value
        let data: Vec<f64> = (0..1000).map(|i| i as f64).collect();
        let ts = TimeSeries::from_values(data);
        let analyzer = PartitionBalanceAnalyzer::new().with_num_partitions(10);

        let result = analyzer.analyze(&ts).unwrap();
        let p_value = result.p_value.unwrap();

        // Uniform distribution should not be rejected
        assert!(p_value > 0.05, "p-value {} should be > 0.05 for uniform data", p_value);
    }

    #[test]
    fn test_biased_distribution() {
        // Create heavily biased data (most values in one partition)
        let mut data = Vec::new();
        // 900 values in range [0, 10)
        for _ in 0..900 {
            data.push(5.0);
        }
        // 100 values spread across [10, 100)
        for i in 0..100 {
            data.push(10.0 + i as f64);
        }

        let ts = TimeSeries::from_values(data);
        let analyzer = PartitionBalanceAnalyzer::new().with_num_partitions(10);

        let result = analyzer.analyze(&ts).unwrap();
        let p_value = result.p_value.unwrap();

        // Biased distribution should be rejected
        assert!(p_value < 0.05, "p-value {} should be < 0.05 for biased data", p_value);
    }

    #[test]
    fn test_partition_computation() {
        let data = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let analyzer = PartitionBalanceAnalyzer::new().with_num_partitions(5);

        let (boundaries, counts, min_val, max_val) = analyzer.compute_partitions(&data);

        assert_eq!(boundaries.len(), 6); // m+1 boundaries
        assert_eq!(counts.len(), 5);
        assert_relative_eq!(min_val, 0.0, epsilon = 1e-10);
        assert_relative_eq!(max_val, 9.0, epsilon = 1e-10);
        assert_eq!(counts.iter().sum::<usize>(), 10);
    }

    #[test]
    fn test_chi_squared_computation() {
        let analyzer = PartitionBalanceAnalyzer::new();

        // Perfect uniform: all partitions have 10 observations
        let counts = vec![10, 10, 10, 10, 10];
        let chi_sq = analyzer.compute_chi_squared(&counts, 50);
        assert_relative_eq!(chi_sq, 0.0, epsilon = 1e-10);

        // Imbalanced: [20, 10, 10, 5, 5]
        let counts = vec![20, 10, 10, 5, 5];
        let chi_sq = analyzer.compute_chi_squared(&counts, 50);
        assert!(chi_sq > 0.0);
    }

    #[test]
    fn test_balance_coefficient() {
        let analyzer = PartitionBalanceAnalyzer::new();

        // Perfect balance
        let counts = vec![10, 10, 10, 10];
        let bc = analyzer.compute_balance_coefficient(&counts, 40);
        assert_relative_eq!(bc, 1.0, epsilon = 1e-10);

        // Maximum imbalance
        let counts = vec![40, 0, 0, 0];
        let bc = analyzer.compute_balance_coefficient(&counts, 40);
        assert_relative_eq!(bc, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_entropy_computation() {
        let analyzer = PartitionBalanceAnalyzer::new();

        // Uniform distribution has maximum entropy
        let counts = vec![25, 25, 25, 25];
        let entropy = analyzer.compute_entropy(&counts, 100);
        let normalized = analyzer.compute_normalized_entropy(entropy, 4);
        assert!(normalized > 0.99); // Very close to 1.0

        // Concentrated distribution has low entropy
        let counts = vec![100, 0, 0, 0];
        let entropy = analyzer.compute_entropy(&counts, 100);
        let normalized = analyzer.compute_normalized_entropy(entropy, 4);
        assert_relative_eq!(normalized, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_validation_insufficient_data() {
        let analyzer = PartitionBalanceAnalyzer::new().with_num_partitions(10);
        let data = TimeSeries::from_values(vec![1.0; 30]); // Only 30 points, need 50

        let result = analyzer.validate(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_validation_constant_data() {
        let analyzer = PartitionBalanceAnalyzer::new();
        let data = TimeSeries::from_values(vec![5.0; 100]);

        let result = analyzer.validate(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_validation_nan_data() {
        let analyzer = PartitionBalanceAnalyzer::new();
        let mut values = vec![1.0; 100];
        values[50] = f64::NAN;
        let data = TimeSeries::from_values(values);

        let result = analyzer.validate(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_required_sample_size() {
        let analyzer = PartitionBalanceAnalyzer::new()
            .with_num_partitions(10);

        let required = analyzer.required_sample_size();
        assert_eq!(required, 50); // 10 partitions * 5 min obs
    }

    #[test]
    fn test_p_value_computation() {
        let analyzer = PartitionBalanceAnalyzer::new();

        // Small chi-squared should give high p-value
        let p_value = analyzer.compute_p_value(1.0, 9).unwrap();
        assert!(p_value > 0.9);

        // Large chi-squared should give low p-value
        let p_value = analyzer.compute_p_value(30.0, 9).unwrap();
        assert!(p_value < 0.01);
    }

    #[test]
    fn test_analyze_returns_all_metrics() {
        let data: Vec<f64> = (0..500).map(|i| i as f64).collect();
        let ts = TimeSeries::from_values(data);
        let analyzer = PartitionBalanceAnalyzer::new();

        let result = analyzer.analyze(&ts).unwrap();

        assert!(result.metrics.contains_key("chi_squared"));
        assert!(result.metrics.contains_key("degrees_of_freedom"));
        assert!(result.metrics.contains_key("entropy"));
        assert!(result.metrics.contains_key("normalized_entropy"));
        assert!(result.metrics.contains_key("balance_coefficient"));
        assert!(result.metrics.contains_key("partition_variance"));
        assert!(result.p_value.is_some());
        assert!(!result.interpretation.is_empty());
    }
}
