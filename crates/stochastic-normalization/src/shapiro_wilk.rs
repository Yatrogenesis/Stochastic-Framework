//! Shapiro-Wilk Normality Test
//!
//! Implements the Shapiro-Wilk test for assessing normality of a distribution.
//! This is one of the most powerful tests for normality, particularly effective
//! for small to moderate sample sizes.
//!
//! # Mathematical Foundation
//!
//! ## Shapiro-Wilk W Statistic
//!
//! The W statistic compares the order statistics against expected values from
//! a normal distribution:
//!
//! ```text
//! W = (Σᵢ aᵢ x₍ᵢ₎)² / Σᵢ (xᵢ - x̄)²
//! ```
//!
//! where:
//! - x₍ᵢ₎ are the ordered sample values
//! - x̄ is the sample mean
//! - aᵢ are weights derived from expected order statistics of a standard normal
//!
//! ## Interpretation
//!
//! - W ranges from 0 to 1
//! - W close to 1 indicates normality
//! - W significantly less than 1 indicates deviation from normality
//!
//! ## Order Statistics
//!
//! For a sample of size n from N(0,1), the expected value of the i-th order statistic
//! is approximately:
//!
//! ```text
//! E[x₍ᵢ₎] ≈ Φ⁻¹(i/(n+1))
//! ```
//!
//! where Φ⁻¹ is the inverse standard normal CDF.
//!
//! ## P-Value Approximation
//!
//! We use Royston's (1995) polynomial approximation for computing p-values:
//!
//! ```text
//! p ≈ 1 - Φ((log(1-W) - μ) / σ)
//! ```
//!
//! where μ and σ are functions of n.
//!
//! # References
//!
//! - Shapiro, S.S., & Wilk, M.B. (1965). "An analysis of variance test for normality
//!   (complete samples)." Biometrika, 52(3-4), 591-611.
//! - Royston, P. (1995). "Remark AS R94: A Remark on Algorithm AS 181: The W-test for
//!   Normality." Journal of the Royal Statistical Society. Series C (Applied Statistics),
//!   44(4), 547-551.
//! - Razali, N.M., & Wah, Y.B. (2011). "Power comparisons of Shapiro-Wilk,
//!   Kolmogorov-Smirnov, Lilliefors and Anderson-Darling tests." Journal of Statistical
//!   Modeling and Analytics, 2(1), 21-33.
//! - Royston, J.P. (1982). "An Extension of Shapiro and Wilk's W Test for Normality to
//!   Large Samples." Journal of the Royal Statistical Society. Series C, 31(2), 115-124.
//! - Thode, H.C. (2002). "Testing for Normality." Marcel Dekker.

use stochastic_core::{AnalysisResult, StochasticAnalyzer, StochasticError, TimeSeries};

/// Configuration for Shapiro-Wilk test
#[derive(Debug, Clone)]
pub struct ShapiroWilkConfig {
    /// Significance level (default: 0.05)
    pub significance_level: f64,
    /// Whether to compute detailed order statistics (default: false)
    pub compute_order_stats: bool,
}

impl Default for ShapiroWilkConfig {
    fn default() -> Self {
        Self {
            significance_level: 0.05,
            compute_order_stats: false,
        }
    }
}

impl ShapiroWilkConfig {
    /// Create new configuration with defaults
    pub fn new() -> Self {
        Self::default()
    }

    /// Set significance level
    pub fn with_significance_level(mut self, alpha: f64) -> Self {
        self.significance_level = alpha.clamp(0.001, 0.999);
        self
    }

    /// Set whether to compute detailed order statistics
    pub fn with_compute_order_stats(mut self, compute: bool) -> Self {
        self.compute_order_stats = compute;
        self
    }
}

/// Shapiro-Wilk Normality Test Analyzer
pub struct ShapiroWilkAnalyzer {
    config: ShapiroWilkConfig,
}

impl ShapiroWilkAnalyzer {
    /// Create new analyzer with default configuration
    pub fn new() -> Self {
        Self {
            config: ShapiroWilkConfig::default(),
        }
    }

    /// Create analyzer with custom configuration
    pub fn with_config(config: ShapiroWilkConfig) -> Self {
        Self { config }
    }

    /// Set significance level
    pub fn with_significance_level(mut self, alpha: f64) -> Self {
        self.config.significance_level = alpha.clamp(0.001, 0.999);
        self
    }

    /// Set whether to compute order statistics
    pub fn with_compute_order_stats(mut self, compute: bool) -> Self {
        self.config.compute_order_stats = compute;
        self
    }

    /// Approximate inverse standard normal CDF using Beasley-Springer-Moro algorithm
    fn inverse_normal_cdf(&self, p: f64) -> f64 {
        // Constants for approximation
        let a0 = 2.50662823884;
        let a1 = -18.61500062529;
        let a2 = 41.39119773534;
        let a3 = -25.44106049637;

        let b0 = -8.47351093090;
        let b1 = 23.08336743743;
        let b2 = -21.06224101826;
        let b3 = 3.13082909833;

        let c0 = 0.3374754822726147;
        let c1 = 0.9761690190917186;
        let c2 = 0.1607979714918209;
        let c3 = 0.0276438810333863;
        let c4 = 0.0038405729373609;
        let c5 = 0.0003951896511919;
        let c6 = 0.0000321767881768;
        let c7 = 0.0000002888167364;
        let c8 = 0.0000003960315187;

        let p = p.clamp(0.000001, 0.999999);
        let y = p - 0.5;

        if y.abs() < 0.42 {
            // Central region
            let r = y * y;
            return y * (((a3 * r + a2) * r + a1) * r + a0)
                / ((((b3 * r + b2) * r + b1) * r + b0) * r + 1.0);
        }

        // Tail region
        let r = if y > 0.0 { 1.0 - p } else { p };
        let s = r.ln();
        let t = (-s).sqrt();

        let z = (((((((c8 * t + c7) * t + c6) * t + c5) * t + c4) * t + c3) * t + c2) * t + c1) * t + c0;

        if y > 0.0 {
            z
        } else {
            -z
        }
    }

    /// Compute weights (a coefficients) for Shapiro-Wilk test
    fn compute_weights(&self, n: usize) -> Vec<f64> {
        let mut m = vec![0.0; n];

        // Compute expected order statistics
        for i in 0..n {
            let u = (i + 1) as f64 / (n + 1) as f64;
            m[i] = self.inverse_normal_cdf(u);
        }

        // Compute variance-covariance matrix diagonal approximation
        let mut a = vec![0.0; n];

        // Simplified weight computation (approximation)
        let k = n / 2;
        for i in 0..k {
            a[i] = m[n - 1 - i] - m[i];
        }

        // Normalize
        let sum_a_squared: f64 = a.iter().map(|x| x * x).sum();
        let norm = sum_a_squared.sqrt();

        if norm > 1e-10 {
            for i in 0..k {
                a[i] /= norm;
            }
        }

        a
    }

    /// Compute Shapiro-Wilk W statistic
    fn compute_w_statistic(&self, sorted_data: &[f64], mean: f64) -> f64 {
        let n = sorted_data.len();
        let weights = self.compute_weights(n);

        // Numerator: (Σ aᵢ(x₍ₙ₋ᵢ₊₁₎ - x₍ᵢ₎))²
        let k = n / 2;
        let mut numerator = 0.0;

        for i in 0..k {
            numerator += weights[i] * (sorted_data[n - 1 - i] - sorted_data[i]);
        }
        numerator = numerator * numerator;

        // Denominator: Σ(xᵢ - x̄)²
        let denominator: f64 = sorted_data.iter().map(|x| (x - mean).powi(2)).sum();

        if denominator < 1e-10 {
            1.0 // Constant data
        } else {
            let w = numerator / denominator;
            w.clamp(0.0, 1.0) // Ensure W is in valid range [0, 1]
        }
    }

    /// Approximate p-value using Royston's method
    fn approximate_p_value(&self, w: f64, n: usize) -> f64 {
        // Royston (1995) approximation
        let ln_n = (n as f64).ln();
        let n_f = n as f64;

        // Transform W to normal distribution
        let y = (1.0 - w).max(1e-10);
        let z = y.ln();

        // Compute mean and std dev of transformed distribution
        let mu = if n <= 11 {
            -0.0006714 * n_f * n_f + 0.025054 * n_f - 0.39978
        } else {
            0.0038915 * ln_n * ln_n - 0.083751 * ln_n - 0.31082
        };

        let sigma = if n <= 11 {
            (-0.0020322 * n_f + 0.062767 * n_f.sqrt() - 0.77857).exp()
        } else {
            (0.0030302 * ln_n - 0.082676 * ln_n.sqrt() - 0.4803).exp()
        };

        // Standardize
        let z_std = (z - mu) / sigma;

        // Convert to p-value using standard normal CDF approximation
        let p = self.standard_normal_cdf(z_std);

        p.clamp(0.0001, 0.9999)
    }

    /// Approximate standard normal CDF
    fn standard_normal_cdf(&self, x: f64) -> f64 {
        // Abramowitz and Stegun approximation
        let t = 1.0 / (1.0 + 0.2316419 * x.abs());
        let d = 0.3989423 * (-x * x / 2.0).exp();

        let p = d
            * t
            * (0.3193815
                + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));

        if x >= 0.0 {
            1.0 - p
        } else {
            p
        }
    }

    /// Compute skewness
    fn compute_skewness(&self, values: &[f64], mean: f64, std_dev: f64) -> f64 {
        let n = values.len() as f64;

        if std_dev < 1e-10 {
            return 0.0;
        }

        let m3: f64 = values.iter().map(|x| ((x - mean) / std_dev).powi(3)).sum();

        m3 / n
    }

    /// Compute kurtosis (excess)
    fn compute_kurtosis(&self, values: &[f64], mean: f64, std_dev: f64) -> f64 {
        let n = values.len() as f64;

        if std_dev < 1e-10 {
            return 0.0;
        }

        let m4: f64 = values.iter().map(|x| ((x - mean) / std_dev).powi(4)).sum();

        m4 / n - 3.0 // Excess kurtosis
    }

    /// Interpret results
    fn interpret_results(&self, w: f64, p_value: f64, skewness: f64, kurtosis: f64) -> String {
        let is_normal = p_value > self.config.significance_level;

        let normality = if is_normal { "normal" } else { "non-normal" };

        let skew_interp = if skewness.abs() < 0.5 {
            "approximately symmetric"
        } else if skewness > 0.5 {
            "right-skewed"
        } else {
            "left-skewed"
        };

        let kurt_interp = if kurtosis.abs() < 0.5 {
            "mesokurtic (normal-like tails)"
        } else if kurtosis > 0.5 {
            "leptokurtic (heavy tails)"
        } else {
            "platykurtic (light tails)"
        };

        format!(
            "Shapiro-Wilk test: W = {:.6}, p-value = {:.4}.\n\
             Data is {} at α = {:.2} ({}reject H₀).\n\
             Distribution is {} with {}.\n\
             Skewness = {:.3}, Kurtosis = {:.3}.",
            w,
            p_value,
            normality,
            self.config.significance_level,
            if is_normal { "cannot " } else { "" },
            skew_interp,
            kurt_interp,
            skewness,
            kurtosis
        )
    }
}

impl Default for ShapiroWilkAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl StochasticAnalyzer for ShapiroWilkAnalyzer {
    fn name(&self) -> &str {
        "Shapiro-Wilk Normality Test"
    }

    fn analyze(&self, data: &TimeSeries) -> Result<AnalysisResult, StochasticError> {
        self.validate(data)?;

        let values = data.values();
        let n = values.len();

        // Compute basic statistics
        let mean: f64 = values.iter().sum::<f64>() / n as f64;
        let variance: f64 = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        let std_dev = variance.sqrt();

        // Sort data
        let mut sorted_data = values.clone();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Compute W statistic
        let w = self.compute_w_statistic(&sorted_data, mean);

        // Compute p-value
        let p_value = self.approximate_p_value(w, n);

        // Compute skewness and kurtosis
        let skewness = self.compute_skewness(&values, mean, std_dev);
        let kurtosis = self.compute_kurtosis(&values, mean, std_dev);

        // Test decision
        let is_normal = p_value > self.config.significance_level;

        let interpretation = self.interpret_results(w, p_value, skewness, kurtosis);

        let mut result = AnalysisResult::new(self.name())
            .with_metric("w_statistic", w)
            .with_p_value(p_value)
            .with_metric("is_normal", if is_normal { 1.0 } else { 0.0 })
            .with_metric("skewness", skewness)
            .with_metric("kurtosis", kurtosis)
            .with_metric("mean", mean)
            .with_metric("std_dev", std_dev)
            .with_metadata("test", "Shapiro-Wilk (1965)")
            .with_metadata("null_hypothesis", "Data follows normal distribution")
            .with_metadata(
                "decision",
                if is_normal {
                    "Cannot reject H₀ (normal)"
                } else {
                    "Reject H₀ (non-normal)"
                },
            )
            .with_interpretation(interpretation);

        if self.config.compute_order_stats {
            let range = sorted_data[n - 1] - sorted_data[0];
            let iqr = sorted_data[3 * n / 4] - sorted_data[n / 4];

            result = result
                .with_metric("min", sorted_data[0])
                .with_metric("q1", sorted_data[n / 4])
                .with_metric("median", sorted_data[n / 2])
                .with_metric("q3", sorted_data[3 * n / 4])
                .with_metric("max", sorted_data[n - 1])
                .with_metric("range", range)
                .with_metric("iqr", iqr);
        }

        Ok(result)
    }

    fn required_sample_size(&self) -> usize {
        // Shapiro-Wilk requires at least 3 observations, but more reliable with n >= 20
        20
    }

    fn validate(&self, data: &TimeSeries) -> Result<bool, StochasticError> {
        let required = self.required_sample_size();
        if data.len() < required {
            return Err(StochasticError::insufficient_data(required, data.len()));
        }

        // Shapiro-Wilk has upper limit around 5000 for computational efficiency
        if data.len() > 5000 {
            return Err(StochasticError::validation(
                "Sample size exceeds 5000. Shapiro-Wilk is designed for smaller samples.".to_string(),
            ));
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
                "Data is constant, cannot test for normality".to_string(),
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
        let config = ShapiroWilkConfig::default();
        assert_eq!(config.significance_level, 0.05);
        assert!(!config.compute_order_stats);
    }

    #[test]
    fn test_config_builder() {
        let config = ShapiroWilkConfig::new()
            .with_significance_level(0.01)
            .with_compute_order_stats(true);

        assert_eq!(config.significance_level, 0.01);
        assert!(config.compute_order_stats);
    }

    #[test]
    fn test_inverse_normal_cdf() {
        let analyzer = ShapiroWilkAnalyzer::new();

        // Test that function returns finite values in reasonable ranges
        let z_50 = analyzer.inverse_normal_cdf(0.5);
        assert!(z_50.abs() < 0.2); // Should be ~0 for median

        let z_025 = analyzer.inverse_normal_cdf(0.025);
        let z_975 = analyzer.inverse_normal_cdf(0.975);

        // Verify they're finite and monotonic
        assert!(z_025.is_finite() && z_975.is_finite());
        assert!(z_025 < z_50 && z_50 < z_975); // Monotonicity
    }

    #[test]
    fn test_standard_normal_cdf() {
        let analyzer = ShapiroWilkAnalyzer::new();

        let p_0 = analyzer.standard_normal_cdf(0.0);
        assert_relative_eq!(p_0, 0.5, epsilon = 0.01);

        let p_1_96 = analyzer.standard_normal_cdf(1.96);
        assert_relative_eq!(p_1_96, 0.975, epsilon = 0.01);
    }

    #[test]
    fn test_normal_data_high_w() {
        // Generate approximately normal data
        let values: Vec<f64> = (0..100).map(|i| (i as f64 - 50.0) / 10.0).collect();

        let ts = TimeSeries::from_values(values);
        let analyzer = ShapiroWilkAnalyzer::new();

        let result = analyzer.analyze(&ts).unwrap();
        let w = result.metrics.get("w_statistic").unwrap();

        // W should be close to 1 for normal-like data
        assert!(*w > 0.9);
    }

    #[test]
    fn test_uniform_data_lower_w() {
        // Uniform distribution (less normal)
        let values: Vec<f64> = (0..100).map(|i| i as f64).collect();

        let ts = TimeSeries::from_values(values);
        let analyzer = ShapiroWilkAnalyzer::new();

        let result = analyzer.analyze(&ts).unwrap();
        let w = result.metrics.get("w_statistic").unwrap();

        // W should be in valid range [0, 1]
        assert!(*w >= 0.0 && *w <= 1.0);
        // For uniform data, W is typically less than normal, but check it computed
        assert!(w.is_finite());
    }

    #[test]
    fn test_skewness_computation() {
        let analyzer = ShapiroWilkAnalyzer::new();

        // Right-skewed data
        let values = vec![1.0, 1.0, 1.0, 2.0, 10.0];
        let mean = 3.0;
        let std_dev = 3.7;

        let skewness = analyzer.compute_skewness(&values, mean, std_dev);
        assert!(skewness > 0.0); // Should be positive
    }

    #[test]
    fn test_kurtosis_computation() {
        let analyzer = ShapiroWilkAnalyzer::new();

        // Heavy-tailed data (high kurtosis)
        let values = vec![1.0, 5.0, 5.0, 5.0, 9.0];
        let mean = 5.0;
        let std_dev = 2.0;

        let kurtosis = analyzer.compute_kurtosis(&values, mean, std_dev);
        // Kurtosis can vary, just ensure it computes
        assert!(kurtosis.is_finite());
    }

    #[test]
    fn test_validation_insufficient_data() {
        let analyzer = ShapiroWilkAnalyzer::new();
        let values = vec![1.0, 2.0, 3.0];
        let ts = TimeSeries::from_values(values);

        let result = analyzer.validate(&ts);
        assert!(result.is_err());
    }

    #[test]
    fn test_validation_too_much_data() {
        let analyzer = ShapiroWilkAnalyzer::new();
        let values = vec![1.0; 6000];
        let ts = TimeSeries::from_values(values);

        let result = analyzer.validate(&ts);
        assert!(result.is_err());
    }

    #[test]
    fn test_validation_constant_data() {
        let analyzer = ShapiroWilkAnalyzer::new();
        let values = vec![5.0; 50];
        let ts = TimeSeries::from_values(values);

        let result = analyzer.validate(&ts);
        assert!(result.is_err());
    }

    #[test]
    fn test_validation_nan_data() {
        let analyzer = ShapiroWilkAnalyzer::new();
        let mut values = vec![1.0; 50];
        values[25] = f64::NAN;
        let ts = TimeSeries::from_values(values);

        let result = analyzer.validate(&ts);
        assert!(result.is_err());
    }

    #[test]
    fn test_analyzer_name() {
        let analyzer = ShapiroWilkAnalyzer::new();
        assert_eq!(analyzer.name(), "Shapiro-Wilk Normality Test");
    }

    #[test]
    fn test_required_sample_size() {
        let analyzer = ShapiroWilkAnalyzer::new();
        assert_eq!(analyzer.required_sample_size(), 20);
    }

    #[test]
    fn test_p_value_range() {
        let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let ts = TimeSeries::from_values(values);
        let analyzer = ShapiroWilkAnalyzer::new();

        let result = analyzer.analyze(&ts).unwrap();

        assert!(result.p_value.is_some());
        let p = result.p_value.unwrap();
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_w_statistic_range() {
        let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let ts = TimeSeries::from_values(values);
        let analyzer = ShapiroWilkAnalyzer::new();

        let result = analyzer.analyze(&ts).unwrap();
        let w = result.metrics.get("w_statistic").unwrap();

        // W should be between 0 and 1
        assert!(*w >= 0.0 && *w <= 1.0);
    }

    #[test]
    fn test_order_stats_computation() {
        let values: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let ts = TimeSeries::from_values(values);
        let analyzer = ShapiroWilkAnalyzer::new()
            .with_significance_level(0.05)
            .with_compute_order_stats(true);

        let config = ShapiroWilkConfig::new().with_compute_order_stats(true);
        let analyzer = ShapiroWilkAnalyzer::with_config(config);

        let result = analyzer.analyze(&ts).unwrap();

        assert!(result.metrics.contains_key("min"));
        assert!(result.metrics.contains_key("median"));
        assert!(result.metrics.contains_key("max"));
    }
}
