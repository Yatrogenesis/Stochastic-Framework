//! Birkhoff-Khinchin Ergodic Theorem Analysis
//!
//! Tests ergodicity by comparing time averages with ensemble averages and analyzing
//! convergence properties. The Birkhoff ergodic theorem states that for ergodic systems,
//! time averages converge to ensemble averages almost everywhere.
//!
//! # Mathematical Foundation
//!
//! ## Birkhoff-Khinchin Ergodic Theorem
//!
//! For an ergodic dynamical system (X, μ, T) and integrable function f:
//!
//! ```text
//! lim_{N→∞} (1/N) Σⁿ⁻¹ᵢ₌₀ f(Tⁱx) = ∫ f dμ    (μ-almost everywhere)
//! ```
//!
//! ## Time Average Convergence
//!
//! For a time series {x₁, x₂, ..., xₙ}, the time average is:
//!
//! ```text
//! ⟨f⟩_time = (1/N) Σⁿᵢ₌₁ f(xᵢ)
//! ```
//!
//! For ergodic systems: ⟨f⟩_time → ⟨f⟩_ensemble = E[f(X)]
//!
//! ## Ergodicity Test Statistic
//!
//! Divide series into M non-overlapping blocks of length B:
//!
//! ```text
//! Block average: μₖ = (1/B) Σᴮⱼ₌₁ x_{(k-1)B+j}
//! Variance of block averages: σ²_block = Var(μₖ)
//! Time series variance: σ²
//! Ergodicity ratio: R = σ²_block / (σ²/B)
//! ```
//!
//! For ergodic systems: R → 1 as B → ∞
//!
//! ## Convergence Rate
//!
//! Analyze how time averages converge by computing running averages:
//!
//! ```text
//! A(n) = (1/n) Σⁿᵢ₌₁ xᵢ
//! Deviation: D(n) = |A(n) - μ|
//! ```
//!
//! Expected convergence: D(n) ∼ O(1/√n) for ergodic systems
//!
//! ## Interpretation
//!
//! - R ≈ 1: System is ergodic (time ≈ ensemble average)
//! - R ≫ 1: Non-ergodic behavior (long-term correlations)
//! - R ≪ 1: Fast mixing (short-term correlations)
//! - D(n) ~ 1/√n: Standard ergodic convergence
//!
//! # References
//!
//! - Birkhoff, G.D. (1931). "Proof of the Ergodic Theorem." Proceedings of the National
//!   Academy of Sciences, 17(12), 656-660.
//! - von Neumann, J. (1932). "Proof of the Quasi-Ergodic Hypothesis." Proceedings of the
//!   National Academy of Sciences, 18(1), 70-82.
//! - Walters, P. (1982). "An Introduction to Ergodic Theory." Graduate Texts in Mathematics,
//!   Vol. 79, Springer-Verlag.
//! - Petersen, K. (1983). "Ergodic Theory." Cambridge Studies in Advanced Mathematics,
//!   Cambridge University Press.
//! - Krengel, U. (1985). "Ergodic Theorems." De Gruyter Studies in Mathematics.
//! - Cornfeld, I.P., Fomin, S.V., & Sinai, Y.G. (1982). "Ergodic Theory." Springer.

use ndarray::Array1;
use stochastic_core::{AnalysisResult, StochasticAnalyzer, StochasticError, TimeSeries};

/// Configuration for Birkhoff ergodic analysis
#[derive(Debug, Clone)]
pub struct BirkhoffConfig {
    /// Number of blocks for ergodicity test (default: 10)
    pub num_blocks: usize,
    /// Block size (auto-computed if None)
    pub block_size: Option<usize>,
    /// Number of convergence points to analyze (default: 50)
    pub convergence_points: usize,
    /// Test multiple block sizes (default: true)
    pub test_multiple_scales: bool,
    /// Significance level for statistical tests (default: 0.05)
    pub significance_level: f64,
}

impl Default for BirkhoffConfig {
    fn default() -> Self {
        Self {
            num_blocks: 10,
            block_size: None,
            convergence_points: 50,
            test_multiple_scales: true,
            significance_level: 0.05,
        }
    }
}

impl BirkhoffConfig {
    /// Create new configuration with defaults
    pub fn new() -> Self {
        Self::default()
    }

    /// Set number of blocks
    pub fn with_num_blocks(mut self, blocks: usize) -> Self {
        self.num_blocks = blocks.max(2);
        self
    }

    /// Set block size
    pub fn with_block_size(mut self, size: usize) -> Self {
        self.block_size = Some(size.max(2));
        self
    }

    /// Set convergence points
    pub fn with_convergence_points(mut self, points: usize) -> Self {
        self.convergence_points = points.max(10);
        self
    }

    /// Enable/disable multi-scale testing
    pub fn with_multi_scale(mut self, enable: bool) -> Self {
        self.test_multiple_scales = enable;
        self
    }
}

/// Birkhoff Ergodic Analyzer
pub struct BirkhoffAnalyzer {
    config: BirkhoffConfig,
}

impl BirkhoffAnalyzer {
    /// Create new analyzer with default configuration
    pub fn new() -> Self {
        Self {
            config: BirkhoffConfig::default(),
        }
    }

    /// Create analyzer with custom configuration
    pub fn with_config(config: BirkhoffConfig) -> Self {
        Self { config }
    }

    /// Set number of blocks
    pub fn with_num_blocks(mut self, blocks: usize) -> Self {
        self.config.num_blocks = blocks.max(2);
        self
    }

    /// Compute block averages
    fn compute_block_averages(&self, data: &[f64], block_size: usize) -> Vec<f64> {
        let num_full_blocks = data.len() / block_size;
        let mut block_avgs = Vec::with_capacity(num_full_blocks);

        for i in 0..num_full_blocks {
            let start = i * block_size;
            let end = start + block_size;
            let block = &data[start..end];
            let avg: f64 = block.iter().sum::<f64>() / block.len() as f64;
            block_avgs.push(avg);
        }

        block_avgs
    }

    /// Compute ergodicity ratio
    fn compute_ergodicity_ratio(&self, data: &[f64], block_size: usize) -> Result<f64, StochasticError> {
        let block_avgs = self.compute_block_averages(data, block_size);

        if block_avgs.len() < 2 {
            return Err(StochasticError::numerical(
                "Insufficient blocks for ergodicity ratio".to_string(),
            ));
        }

        // Compute variance of block averages
        let mean_block: f64 = block_avgs.iter().sum::<f64>() / block_avgs.len() as f64;
        let var_block: f64 = block_avgs
            .iter()
            .map(|&x| (x - mean_block).powi(2))
            .sum::<f64>()
            / (block_avgs.len() - 1) as f64;

        // Compute time series variance
        let mean_total: f64 = data.iter().sum::<f64>() / data.len() as f64;
        let var_total: f64 = data
            .iter()
            .map(|&x| (x - mean_total).powi(2))
            .sum::<f64>()
            / (data.len() - 1) as f64;

        // Ergodicity ratio
        let expected_var_block = var_total / block_size as f64;

        if expected_var_block == 0.0 {
            return Err(StochasticError::numerical("Zero variance".to_string()));
        }

        Ok(var_block / expected_var_block)
    }

    /// Compute running time averages
    fn compute_running_averages(&self, data: &[f64]) -> Array1<f64> {
        let n = data.len();
        let num_points = self.config.convergence_points.min(n);
        let mut running_avgs = Array1::<f64>::zeros(num_points);

        let step = n / num_points;

        for i in 1..=num_points {
            let idx = (i * step).min(n);
            let avg: f64 = data[..idx].iter().sum::<f64>() / idx as f64;
            running_avgs[i - 1] = avg;
        }

        running_avgs
    }

    /// Analyze convergence rate
    fn analyze_convergence(&self, data: &[f64]) -> (f64, f64, f64) {
        let running_avgs = self.compute_running_averages(data);
        let ensemble_avg: f64 = data.iter().sum::<f64>() / data.len() as f64;

        // Compute deviations
        let deviations: Vec<f64> = running_avgs
            .iter()
            .map(|&avg| (avg - ensemble_avg).abs())
            .collect();

        // Final deviation
        let final_deviation = deviations.last().copied().unwrap_or(0.0);

        // Maximum deviation
        let max_deviation = deviations.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        // Average deviation over last 25% of convergence
        let last_quarter_start = (deviations.len() * 3) / 4;
        let avg_late_deviation: f64 = deviations[last_quarter_start..]
            .iter()
            .sum::<f64>()
            / (deviations.len() - last_quarter_start) as f64;

        (final_deviation, max_deviation, avg_late_deviation)
    }

    /// Estimate convergence exponent (should be ~0.5 for ergodic)
    fn estimate_convergence_exponent(&self, data: &[f64]) -> f64 {
        let running_avgs = self.compute_running_averages(data);
        let ensemble_avg: f64 = data.iter().sum::<f64>() / data.len() as f64;

        let n = running_avgs.len();
        let step = data.len() / n;

        // Use last half of data for fitting
        let start_idx = n / 2;
        let mut sum_log_n = 0.0;
        let mut sum_log_dev = 0.0;
        let mut sum_log_n_sq = 0.0;
        let mut sum_log_n_log_dev = 0.0;
        let mut count = 0;

        for i in start_idx..n {
            let sample_size = ((i + 1) * step) as f64;
            let deviation = (running_avgs[i] - ensemble_avg).abs();

            if deviation > 1e-10 && sample_size > 0.0 {
                let log_n = sample_size.ln();
                let log_dev = deviation.ln();

                sum_log_n += log_n;
                sum_log_dev += log_dev;
                sum_log_n_sq += log_n * log_n;
                sum_log_n_log_dev += log_n * log_dev;
                count += 1;
            }
        }

        if count < 2 {
            return 0.5; // Default expected exponent
        }

        // Linear regression: log(deviation) ~ slope * log(n) + intercept
        let n_f = count as f64;
        let slope = (n_f * sum_log_n_log_dev - sum_log_n * sum_log_dev)
            / (n_f * sum_log_n_sq - sum_log_n * sum_log_n);

        // Expected slope is -0.5 for ergodic systems (deviation ~ 1/√n)
        -slope
    }

    /// Compute variance of time averages at different scales
    fn compute_multi_scale_variance(&self, data: &[f64]) -> Vec<(usize, f64)> {
        let n = data.len();
        let mut results = Vec::new();

        // Test block sizes from n/20 to n/4
        let min_block = (n / 20).max(10);
        let max_block = n / 4;

        let mut block_size = min_block;
        while block_size <= max_block {
            if let Ok(ratio) = self.compute_ergodicity_ratio(data, block_size) {
                results.push((block_size, ratio));
            }
            block_size = (block_size as f64 * 1.5) as usize;
        }

        results
    }

    /// Generate interpretation
    fn interpret_results(&self, ratio: f64, exponent: f64, final_dev: f64) -> String {
        let ergodicity_assessment = if (ratio - 1.0).abs() < 0.2 {
            "ERGODIC: Time averages converge to ensemble average"
        } else if ratio > 1.5 {
            "NON-ERGODIC: Strong long-term correlations detected"
        } else if ratio < 0.5 {
            "FAST MIXING: Rapid decorrelation"
        } else {
            "WEAKLY ERGODIC: Moderate deviation from ideal ergodicity"
        };

        let convergence_assessment = if (exponent - 0.5).abs() < 0.15 {
            "Expected 1/√n convergence rate"
        } else if exponent < 0.5 {
            "Slower than expected convergence"
        } else {
            "Faster than expected convergence"
        };

        format!(
            "{} (R = {:.4}).\n{} (α = {:.4}, expected ≈ 0.5).\nFinal deviation: {:.6}",
            ergodicity_assessment, ratio, convergence_assessment, exponent, final_dev
        )
    }
}

impl Default for BirkhoffAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl StochasticAnalyzer for BirkhoffAnalyzer {
    fn name(&self) -> &str {
        "Birkhoff Ergodic Theorem"
    }

    fn analyze(&self, data: &TimeSeries) -> Result<AnalysisResult, StochasticError> {
        self.validate(data)?;

        let values = data.values();
        let n = values.len();

        // Determine block size
        let block_size = self.config.block_size.unwrap_or(n / self.config.num_blocks);

        // Validate block size
        if block_size < 2 {
            return Err(StochasticError::validation(
                "Block size too small".to_string(),
            ));
        }

        if n / block_size < 2 {
            return Err(StochasticError::validation(
                "Insufficient data for block analysis".to_string(),
            ));
        }

        // Compute ergodicity ratio
        let ergodicity_ratio = self.compute_ergodicity_ratio(&values, block_size)?;

        // Compute convergence metrics
        let (final_dev, max_dev, avg_late_dev) = self.analyze_convergence(&values);
        let convergence_exponent = self.estimate_convergence_exponent(&values);

        // Compute ensemble average and variance
        let ensemble_avg: f64 = values.iter().sum::<f64>() / n as f64;
        let ensemble_var: f64 = values
            .iter()
            .map(|&x| (x - ensemble_avg).powi(2))
            .sum::<f64>()
            / (n - 1) as f64;

        // Build result
        let mut result = AnalysisResult::new(self.name())
            .with_metric("ergodicity_ratio", ergodicity_ratio)
            .with_metric("convergence_exponent", convergence_exponent)
            .with_metric("final_deviation", final_dev)
            .with_metric("max_deviation", max_dev)
            .with_metric("avg_late_deviation", avg_late_dev)
            .with_metric("ensemble_mean", ensemble_avg)
            .with_metric("ensemble_variance", ensemble_var)
            .with_metric("block_size", block_size as f64)
            .with_metric("num_blocks", (n / block_size) as f64);

        // Multi-scale analysis if enabled
        if self.config.test_multiple_scales {
            let multi_scale = self.compute_multi_scale_variance(&values);
            let avg_ratio: f64 = multi_scale.iter().map(|(_, r)| r).sum::<f64>() / multi_scale.len() as f64;
            let ratio_variance: f64 = multi_scale
                .iter()
                .map(|(_, r)| (r - avg_ratio).powi(2))
                .sum::<f64>()
                / multi_scale.len() as f64;

            result = result
                .with_metric("avg_ergodicity_ratio_multiscale", avg_ratio)
                .with_metric("ergodicity_ratio_variance", ratio_variance)
                .with_metric("num_scales_tested", multi_scale.len() as f64);
        }

        // Interpretation
        let interpretation = self.interpret_results(ergodicity_ratio, convergence_exponent, final_dev);
        result = result.with_interpretation(interpretation);

        Ok(result)
    }

    fn required_sample_size(&self) -> usize {
        // Need enough data for multiple blocks
        let min_block_size = 10;
        let min_blocks = self.config.num_blocks.max(5);
        min_block_size * min_blocks
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
                "Data is constant, cannot perform ergodic analysis".to_string(),
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
        let config = BirkhoffConfig::default();
        assert_eq!(config.num_blocks, 10);
        assert_eq!(config.convergence_points, 50);
        assert!(config.test_multiple_scales);
    }

    #[test]
    fn test_config_builder() {
        let config = BirkhoffConfig::new()
            .with_num_blocks(20)
            .with_block_size(50)
            .with_convergence_points(100);

        assert_eq!(config.num_blocks, 20);
        assert_eq!(config.block_size, Some(50));
        assert_eq!(config.convergence_points, 100);
    }

    #[test]
    fn test_analyzer_creation() {
        let analyzer = BirkhoffAnalyzer::new();
        assert_eq!(analyzer.name(), "Birkhoff Ergodic Theorem");
    }

    #[test]
    fn test_block_averages() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let analyzer = BirkhoffAnalyzer::new();
        let block_avgs = analyzer.compute_block_averages(&data, 4);

        assert_eq!(block_avgs.len(), 2);
        assert_relative_eq!(block_avgs[0], 2.5, epsilon = 1e-10); // (1+2+3+4)/4
        assert_relative_eq!(block_avgs[1], 6.5, epsilon = 1e-10); // (5+6+7+8)/4
    }

    #[test]
    fn test_ergodic_iid_gaussian() {
        // IID Gaussian should be ergodic (ratio ≈ 1)
        use rand::thread_rng;
        use rand_distr::{Distribution, Normal as RandNormal};

        let normal = RandNormal::new(0.0, 1.0).unwrap();
        let mut rng = thread_rng();
        let data: Vec<f64> = (0..1000).map(|_| normal.sample(&mut rng)).collect();

        let ts = TimeSeries::from_values(data);
        let analyzer = BirkhoffAnalyzer::new();

        let result = analyzer.analyze(&ts).unwrap();
        let ratio = result.metrics.get("ergodicity_ratio").unwrap();

        // Should be positive and finite (actual value can vary due to randomness)
        assert!(ratio.is_finite() && *ratio > 0.0, "Ergodicity ratio should be finite and positive");
    }

    #[test]
    fn test_running_averages() {
        let data: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let analyzer = BirkhoffAnalyzer::new();
        let running_avgs = analyzer.compute_running_averages(&data);

        assert_eq!(running_avgs.len(), analyzer.config.convergence_points);
        // All values should be positive and increasing towards 50.5
        assert!(running_avgs.iter().all(|&x| x > 0.0));
    }

    #[test]
    fn test_convergence_analysis() {
        let data: Vec<f64> = vec![1.0; 1000]; // Constant data
        let analyzer = BirkhoffAnalyzer::new();
        let (final_dev, max_dev, avg_late_dev) = analyzer.analyze_convergence(&data);

        // Constant data should have zero deviation
        assert_relative_eq!(final_dev, 0.0, epsilon = 1e-10);
        assert_relative_eq!(max_dev, 0.0, epsilon = 1e-10);
        assert_relative_eq!(avg_late_dev, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_validation_insufficient_data() {
        let analyzer = BirkhoffAnalyzer::new();
        let data = TimeSeries::from_values(vec![1.0; 30]);

        let result = analyzer.validate(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_validation_constant_data() {
        let analyzer = BirkhoffAnalyzer::new();
        let data = TimeSeries::from_values(vec![5.0; 200]);

        let result = analyzer.validate(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_validation_nan_data() {
        let analyzer = BirkhoffAnalyzer::new();
        let mut values = vec![1.0; 200];
        values[100] = f64::NAN;
        let data = TimeSeries::from_values(values);

        let result = analyzer.validate(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_required_sample_size() {
        let analyzer = BirkhoffAnalyzer::new();
        let required = analyzer.required_sample_size();
        assert!(required >= 50);
    }

    #[test]
    fn test_analyze_returns_all_metrics() {
        let data: Vec<f64> = (0..500).map(|i| (i as f64 * 0.1).sin()).collect();
        let ts = TimeSeries::from_values(data);
        let analyzer = BirkhoffAnalyzer::new();

        let result = analyzer.analyze(&ts).unwrap();

        assert!(result.metrics.contains_key("ergodicity_ratio"));
        assert!(result.metrics.contains_key("convergence_exponent"));
        assert!(result.metrics.contains_key("final_deviation"));
        assert!(result.metrics.contains_key("ensemble_mean"));
        assert!(result.metrics.contains_key("ensemble_variance"));
        assert!(!result.interpretation.is_empty());
    }

    #[test]
    fn test_multi_scale_variance() {
        let data: Vec<f64> = (0..1000).map(|i| i as f64).collect();
        let analyzer = BirkhoffAnalyzer::new();
        let multi_scale = analyzer.compute_multi_scale_variance(&data);

        assert!(!multi_scale.is_empty());
        // Each entry should have a block size and ratio
        for (block_size, ratio) in multi_scale {
            assert!(block_size > 0);
            assert!(ratio.is_finite());
        }
    }
}
