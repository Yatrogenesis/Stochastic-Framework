//! Mixing Properties and Autocorrelation Analysis
//!
//! Analyzes temporal independence through autocorrelation functions, mixing coefficients,
//! and statistical tests for serial dependence. Assesses how quickly a system "forgets"
//! its initial state and whether observations become independent over time.
//!
//! # Mathematical Foundation
//!
//! ## Autocorrelation Function
//!
//! For a stationary time series {Xₜ} with mean μ and variance σ²:
//!
//! ```text
//! ρ(k) = Cov(Xₜ, Xₜ₊ₖ) / Var(Xₜ)
//!      = E[(Xₜ - μ)(Xₜ₊ₖ - μ)] / σ²
//! ```
//!
//! Properties:
//! - ρ(0) = 1
//! - |ρ(k)| ≤ 1
//! - ρ(-k) = ρ(k) for stationary processes
//!
//! ## Mixing Coefficient (α-mixing)
//!
//! For σ-algebras F₋∞ᵗ and Fₜ₊ₖ∞:
//!
//! ```text
//! α(k) = sup |P(A ∩ B) - P(A)P(B)|
//! ```
//!
//! where A ∈ F₋∞⁰, B ∈ Fₖ∞
//!
//! Empirical approximation:
//! ```text
//! α̂(k) ≈ max_bins |P̂(Iᵢ ∩ Jⱼ₊ₖ) - P̂(Iᵢ)P̂(Jⱼ)|
//! ```
//!
//! ## Ljung-Box Q Statistic
//!
//! Test for autocorrelation across multiple lags:
//!
//! ```text
//! Q = n(n+2) Σₖ₌₁ʰ [ρ̂²(k) / (n-k)]
//! ```
//!
//! Under H₀ (no autocorrelation): Q ~ χ²(h)
//!
//! ## Decay Rate Analysis
//!
//! Fit exponential decay to autocorrelation:
//!
//! ```text
//! ρ(k) ≈ A exp(-k/τ)
//! ```
//!
//! where τ is the correlation time (decay rate)
//!
//! ## Interpretation
//!
//! - ρ(k) → 0 as k → ∞: Mixing property holds
//! - α(k) → 0 as k → ∞: Strong mixing
//! - Q large, p < 0.05: Significant autocorrelation
//! - Small τ: Fast mixing (short memory)
//! - Large τ: Slow mixing (long memory)
//!
//! # References
//!
//! - Rosenblatt, M. (1956). "A Central Limit Theorem and a Strong Mixing Condition."
//!   Proceedings of the National Academy of Sciences, 42(1), 43-47.
//! - Bradley, R.C. (2005). "Basic Properties of Strong Mixing Conditions. A Survey and
//!   Some Open Questions." Probability Surveys, 2, 107-144.
//! - Box, G.E.P., & Pierce, D.A. (1970). "Distribution of Residual Autocorrelations in
//!   Autoregressive-Integrated Moving Average Time Series Models." Journal of the
//!   American Statistical Association, 65(332), 1509-1526.
//! - Ljung, G.M., & Box, G.E.P. (1978). "On a Measure of Lack of Fit in Time Series
//!   Models." Biometrika, 65(2), 297-303.
//! - Bradley, R.C. (1986). "Basic Properties of Strong Mixing Conditions." Progress in
//!   Probability and Statistics, 11, 165-192.
//! - Doukhan, P. (1994). "Mixing: Properties and Examples." Lecture Notes in Statistics,
//!   Vol. 85, Springer-Verlag.

use ndarray::Array1;
use statrs::distribution::{ChiSquared, ContinuousCDF};
use stochastic_core::{AnalysisResult, StochasticAnalyzer, StochasticError, TimeSeries};

/// Configuration for mixing properties analysis
#[derive(Debug, Clone)]
pub struct MixingConfig {
    /// Maximum lag for autocorrelation (default: 20)
    pub max_lag: usize,
    /// Number of lags for Ljung-Box test (default: 10)
    pub ljung_box_lags: usize,
    /// Number of bins for mixing coefficient estimation (default: 5)
    pub num_bins: usize,
    /// Estimate decay rate (default: true)
    pub estimate_decay_rate: bool,
    /// Significance level for tests (default: 0.05)
    pub significance_level: f64,
}

impl Default for MixingConfig {
    fn default() -> Self {
        Self {
            max_lag: 20,
            ljung_box_lags: 10,
            num_bins: 5,
            estimate_decay_rate: true,
            significance_level: 0.05,
        }
    }
}

impl MixingConfig {
    /// Create new configuration with defaults
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum lag
    pub fn with_max_lag(mut self, lag: usize) -> Self {
        self.max_lag = lag.max(1);
        self
    }

    /// Set Ljung-Box lags
    pub fn with_ljung_box_lags(mut self, lags: usize) -> Self {
        self.ljung_box_lags = lags.max(1);
        self
    }

    /// Set number of bins for mixing coefficient
    pub fn with_num_bins(mut self, bins: usize) -> Self {
        self.num_bins = bins.max(2);
        self
    }

    /// Enable/disable decay rate estimation
    pub fn with_decay_rate(mut self, enable: bool) -> Self {
        self.estimate_decay_rate = enable;
        self
    }
}

/// Mixing Properties Analyzer
pub struct MixingAnalyzer {
    config: MixingConfig,
}

impl MixingAnalyzer {
    /// Create new analyzer with default configuration
    pub fn new() -> Self {
        Self {
            config: MixingConfig::default(),
        }
    }

    /// Create analyzer with custom configuration
    pub fn with_config(config: MixingConfig) -> Self {
        Self { config }
    }

    /// Set maximum lag
    pub fn with_max_lag(mut self, lag: usize) -> Self {
        self.config.max_lag = lag.max(1);
        self
    }

    /// Compute autocorrelation function
    fn compute_autocorrelation(&self, data: &[f64], max_lag: usize) -> Array1<f64> {
        let n = data.len();
        let mean: f64 = data.iter().sum::<f64>() / n as f64;

        // Compute variance
        let variance: f64 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;

        if variance == 0.0 {
            return Array1::zeros(max_lag + 1);
        }

        let mut acf = Array1::<f64>::zeros(max_lag + 1);
        acf[0] = 1.0; // ρ(0) = 1

        for k in 1..=max_lag {
            if k >= n {
                break;
            }

            let mut covariance = 0.0;
            for t in 0..(n - k) {
                covariance += (data[t] - mean) * (data[t + k] - mean);
            }
            covariance /= n as f64;

            acf[k] = covariance / variance;
        }

        acf
    }

    /// Compute Ljung-Box Q statistic
    fn compute_ljung_box(&self, acf: &Array1<f64>, n: usize, h: usize) -> (f64, f64) {
        let mut q = 0.0;

        for k in 1..=h.min(acf.len() - 1) {
            if n > k {
                q += acf[k].powi(2) / (n - k) as f64;
            }
        }

        q *= n as f64 * (n as f64 + 2.0);

        // Compute p-value using chi-squared distribution
        let dist = ChiSquared::new(h as f64).unwrap();
        let p_value = 1.0 - dist.cdf(q);

        (q, p_value)
    }

    /// Estimate mixing coefficient (simplified α-mixing)
    fn estimate_mixing_coefficient(&self, data: &[f64], lag: usize) -> f64 {
        let n = data.len();
        if lag >= n {
            return 0.0;
        }

        let num_bins = self.config.num_bins;

        // Determine bin edges
        let min_val = data.iter().copied().fold(f64::INFINITY, f64::min);
        let max_val = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let bin_width = (max_val - min_val) / num_bins as f64;

        if bin_width == 0.0 {
            return 0.0;
        }

        // Discretize data into bins
        let bin_data: Vec<usize> = data
            .iter()
            .map(|&x| {
                let bin = ((x - min_val) / bin_width).floor() as usize;
                bin.min(num_bins - 1)
            })
            .collect();

        // Compute joint and marginal probabilities
        let n_pairs = n - lag;
        let mut joint_counts = vec![vec![0; num_bins]; num_bins];
        let mut marginal_t = vec![0; num_bins];
        let mut marginal_t_lag = vec![0; num_bins];

        for t in 0..n_pairs {
            let i = bin_data[t];
            let j = bin_data[t + lag];
            joint_counts[i][j] += 1;
            marginal_t[i] += 1;
            marginal_t_lag[j] += 1;
        }

        // Compute mixing coefficient
        let mut max_diff: f64 = 0.0;
        let n_pairs_f = n_pairs as f64;

        for i in 0..num_bins {
            for j in 0..num_bins {
                let p_joint = joint_counts[i][j] as f64 / n_pairs_f;
                let p_i = marginal_t[i] as f64 / n_pairs_f;
                let p_j = marginal_t_lag[j] as f64 / n_pairs_f;
                let diff = (p_joint - p_i * p_j).abs();
                max_diff = max_diff.max(diff);
            }
        }

        max_diff
    }

    /// Estimate decay rate by fitting exponential to ACF
    fn estimate_decay_rate(&self, acf: &Array1<f64>) -> Option<f64> {
        // Fit ρ(k) ≈ exp(-k/τ) using linear regression on log(|ρ(k)|)
        let mut sum_k = 0.0;
        let mut sum_log_rho = 0.0;
        let mut sum_k_sq = 0.0;
        let mut sum_k_log_rho = 0.0;
        let mut count = 0;

        for k in 1..acf.len() {
            let rho = acf[k].abs();
            if rho > 1e-10 {
                let log_rho = rho.ln();
                let k_f = k as f64;

                sum_k += k_f;
                sum_log_rho += log_rho;
                sum_k_sq += k_f * k_f;
                sum_k_log_rho += k_f * log_rho;
                count += 1;
            }
        }

        if count < 3 {
            return None;
        }

        let n_f = count as f64;
        let slope = (n_f * sum_k_log_rho - sum_k * sum_log_rho) / (n_f * sum_k_sq - sum_k * sum_k);

        // τ = -1/slope
        if slope < 0.0 {
            Some(-1.0 / slope)
        } else {
            None
        }
    }

    /// Compute effective sample size accounting for autocorrelation
    fn compute_effective_sample_size(&self, acf: &Array1<f64>, n: usize) -> f64 {
        // Bartlett's formula: n_eff = n / (1 + 2*Σρ(k))
        let mut sum_acf = 0.0;
        for k in 1..acf.len() {
            sum_acf += acf[k];
        }

        let factor = 1.0 + 2.0 * sum_acf;
        if factor > 0.0 {
            n as f64 / factor
        } else {
            n as f64
        }
    }

    /// Compute integrated autocorrelation time
    fn compute_integrated_autocorr_time(&self, acf: &Array1<f64>) -> f64 {
        // τ_int = 1/2 + Σₖ₌₁^∞ ρ(k)
        let sum_acf: f64 = acf.iter().skip(1).sum();
        0.5 + sum_acf
    }

    /// Generate interpretation
    fn interpret_results(
        &self,
        lb_p_value: f64,
        max_acf: f64,
        tau: Option<f64>,
        alpha_decay: f64,
    ) -> String {
        let significance = self.config.significance_level;

        let autocorr_assessment = if lb_p_value > significance {
            format!(
                "NO SIGNIFICANT AUTOCORRELATION: Ljung-Box test p = {:.4} > α = {:.4}",
                lb_p_value, significance
            )
        } else {
            format!(
                "SIGNIFICANT AUTOCORRELATION DETECTED: Ljung-Box test p = {:.4} < α = {:.4}",
                lb_p_value, significance
            )
        };

        let mixing_assessment = if max_acf < 0.3 {
            "Strong mixing (weak correlations)"
        } else if max_acf < 0.6 {
            "Moderate mixing"
        } else {
            "Weak mixing (strong persistent correlations)"
        };

        let mut interpretation = format!(
            "{}\n{} (max |ρ(k)| = {:.4}).",
            autocorr_assessment, mixing_assessment, max_acf
        );

        if let Some(tau_val) = tau {
            let decay_quality = if tau_val < 5.0 {
                "fast"
            } else if tau_val < 20.0 {
                "moderate"
            } else {
                "slow"
            };
            interpretation.push_str(&format!(
                "\nDecay time: τ = {:.2} lags ({} decorrelation).",
                tau_val, decay_quality
            ));
        }

        interpretation.push_str(&format!(
            "\nMixing coefficient α̂ = {:.4} (smaller is better).",
            alpha_decay
        ));

        interpretation
    }
}

impl Default for MixingAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl StochasticAnalyzer for MixingAnalyzer {
    fn name(&self) -> &str {
        "Mixing Properties & Autocorrelation"
    }

    fn analyze(&self, data: &TimeSeries) -> Result<AnalysisResult, StochasticError> {
        self.validate(data)?;

        let values = data.values();
        let n = values.len();

        // Determine maximum lag
        let max_lag = self.config.max_lag.min(n / 4);
        if max_lag < 1 {
            return Err(StochasticError::validation(
                "Data too short for autocorrelation analysis".to_string(),
            ));
        }

        // Compute autocorrelation function
        let acf = self.compute_autocorrelation(&values, max_lag);

        // Ljung-Box test
        let lb_lags = self.config.ljung_box_lags.min(max_lag);
        let (ljung_box_q, ljung_box_p) = self.compute_ljung_box(&acf, n, lb_lags);

        // Find maximum absolute autocorrelation (excluding lag 0)
        let max_acf = acf
            .iter()
            .skip(1)
            .map(|&x| x.abs())
            .fold(0.0, f64::max);

        // First significant lag (where |ρ(k)| < 2/√n)
        let threshold = 2.0 / (n as f64).sqrt();
        let first_insignificant_lag = acf
            .iter()
            .skip(1)
            .position(|&x| x.abs() < threshold)
            .map(|pos| pos + 1)
            .unwrap_or(max_lag);

        // Compute effective sample size
        let effective_n = self.compute_effective_sample_size(&acf, n);

        // Compute integrated autocorrelation time
        let tau_int = self.compute_integrated_autocorr_time(&acf);

        // Estimate decay rate
        let decay_tau = if self.config.estimate_decay_rate {
            self.estimate_decay_rate(&acf)
        } else {
            None
        };

        // Estimate mixing coefficient at lag 1 and max_lag/2
        let alpha_lag1 = self.estimate_mixing_coefficient(&values, 1);
        let alpha_mid = self.estimate_mixing_coefficient(&values, max_lag / 2);

        // Build result
        let mut result = AnalysisResult::new(self.name())
            .with_metric("ljung_box_q", ljung_box_q)
            .with_p_value(ljung_box_p)
            .with_metric("max_autocorrelation", max_acf)
            .with_metric("first_insignificant_lag", first_insignificant_lag as f64)
            .with_metric("effective_sample_size", effective_n)
            .with_metric("sample_size_ratio", effective_n / n as f64)
            .with_metric("integrated_autocorr_time", tau_int)
            .with_metric("mixing_coeff_lag1", alpha_lag1)
            .with_metric("mixing_coeff_mid", alpha_mid)
            .with_metric("max_lag_tested", max_lag as f64);

        // Add ACF values for first several lags
        for k in 1..=5.min(max_lag) {
            result = result.with_metric(&format!("acf_lag_{}", k), acf[k]);
        }

        // Add decay rate if estimated
        if let Some(tau) = decay_tau {
            result = result.with_metric("decay_rate", tau);
        }

        // Interpretation
        let interpretation = self.interpret_results(ljung_box_p, max_acf, decay_tau, alpha_mid);
        result = result.with_interpretation(interpretation);

        Ok(result)
    }

    fn required_sample_size(&self) -> usize {
        // Need at least 4 * max_lag for meaningful autocorrelation
        let min_per_lag = 4;
        (self.config.max_lag * min_per_lag).max(100)
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
                "Data is constant, cannot perform autocorrelation analysis".to_string(),
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
        let config = MixingConfig::default();
        assert_eq!(config.max_lag, 20);
        assert_eq!(config.ljung_box_lags, 10);
        assert_eq!(config.num_bins, 5);
        assert!(config.estimate_decay_rate);
    }

    #[test]
    fn test_config_builder() {
        let config = MixingConfig::new()
            .with_max_lag(30)
            .with_ljung_box_lags(15)
            .with_num_bins(10);

        assert_eq!(config.max_lag, 30);
        assert_eq!(config.ljung_box_lags, 15);
        assert_eq!(config.num_bins, 10);
    }

    #[test]
    fn test_analyzer_creation() {
        let analyzer = MixingAnalyzer::new();
        assert_eq!(analyzer.name(), "Mixing Properties & Autocorrelation");
    }

    #[test]
    fn test_autocorrelation_constant() {
        let data = vec![5.0; 100];
        let analyzer = MixingAnalyzer::new();
        let acf = analyzer.compute_autocorrelation(&data, 10);

        // Constant data has undefined autocorrelation (returns 0)
        assert_eq!(acf[0], 0.0);
    }

    #[test]
    fn test_autocorrelation_white_noise() {
        // White noise should have ACF ≈ 0 for k > 0
        use rand::thread_rng;
        use rand_distr::{Distribution, Normal as RandNormal};

        let normal = RandNormal::new(0.0, 1.0).unwrap();
        let mut rng = thread_rng();
        let data: Vec<f64> = (0..1000).map(|_| normal.sample(&mut rng)).collect();

        let analyzer = MixingAnalyzer::new();
        let acf = analyzer.compute_autocorrelation(&data, 20);

        assert_relative_eq!(acf[0], 1.0, epsilon = 1e-10);

        // ACF at other lags should be small
        let max_acf = acf.iter().skip(1).map(|&x| x.abs()).fold(0.0, f64::max);
        assert!(max_acf < 0.15, "White noise ACF should be small, got {}", max_acf);
    }

    #[test]
    fn test_autocorrelation_perfect_correlation() {
        // Data with perfect lag-1 correlation
        let mut data = Vec::new();
        for i in 0..100 {
            data.push(i as f64);
            data.push(i as f64);
        }

        let analyzer = MixingAnalyzer::new();
        let acf = analyzer.compute_autocorrelation(&data, 5);

        assert_relative_eq!(acf[0], 1.0, epsilon = 1e-10);
        // Should have strong correlation at odd lags
        assert!(acf[1].abs() > 0.5);
    }

    #[test]
    fn test_ljung_box_white_noise() {
        use rand::thread_rng;
        use rand_distr::{Distribution, Normal as RandNormal};

        let normal = RandNormal::new(0.0, 1.0).unwrap();
        let mut rng = thread_rng();
        let data: Vec<f64> = (0..500).map(|_| normal.sample(&mut rng)).collect();

        let ts = TimeSeries::from_values(data);
        let analyzer = MixingAnalyzer::new();

        let result = analyzer.analyze(&ts).unwrap();
        let p_value = result.p_value.unwrap();

        // White noise should not reject H0 (no autocorrelation)
        assert!(p_value > 0.05, "White noise should have p > 0.05, got {}", p_value);
    }

    #[test]
    fn test_ljung_box_correlated() {
        // AR(1) process with strong correlation
        let mut data = vec![0.0];
        for i in 1..500 {
            data.push(0.9 * data[i - 1] + 0.1);
        }

        let ts = TimeSeries::from_values(data);
        let analyzer = MixingAnalyzer::new();

        let result = analyzer.analyze(&ts).unwrap();
        let p_value = result.p_value.unwrap();

        // Strong autocorrelation should reject H0
        assert!(p_value < 0.05, "Correlated data should have p < 0.05, got {}", p_value);
    }

    #[test]
    fn test_mixing_coefficient() {
        let data: Vec<f64> = (0..200).map(|i| i as f64).collect();
        let analyzer = MixingAnalyzer::new();

        let alpha_1 = analyzer.estimate_mixing_coefficient(&data, 1);
        let alpha_10 = analyzer.estimate_mixing_coefficient(&data, 10);

        // Both should be valid
        assert!(alpha_1.is_finite());
        assert!(alpha_10.is_finite());
        assert!(alpha_1 >= 0.0);
        assert!(alpha_10 >= 0.0);
    }

    #[test]
    fn test_decay_rate_estimation() {
        // Create data with exponential decay autocorrelation
        let analyzer = MixingAnalyzer::new();

        // Simulate ACF with known decay
        let mut acf = Array1::<f64>::zeros(20);
        acf[0] = 1.0;
        for k in 1..20 {
            acf[k] = (-(k as f64) / 5.0).exp(); // τ = 5
        }

        let tau = analyzer.estimate_decay_rate(&acf);
        assert!(tau.is_some());

        let tau_val = tau.unwrap();
        // Should be close to 5
        assert!((tau_val - 5.0).abs() < 2.0, "Estimated τ = {}, expected ≈ 5", tau_val);
    }

    #[test]
    fn test_effective_sample_size() {
        let analyzer = MixingAnalyzer::new();

        // No autocorrelation: n_eff = n
        let acf_uncorr = Array1::<f64>::zeros(10);
        let n_eff = analyzer.compute_effective_sample_size(&acf_uncorr, 100);
        assert_relative_eq!(n_eff, 100.0, epsilon = 0.1);

        // Strong autocorrelation: n_eff < n
        let mut acf_corr = Array1::<f64>::zeros(10);
        acf_corr.fill(0.5);
        let n_eff_corr = analyzer.compute_effective_sample_size(&acf_corr, 100);
        assert!(n_eff_corr < 100.0);
    }

    #[test]
    fn test_integrated_autocorr_time() {
        let analyzer = MixingAnalyzer::new();

        let acf = Array1::from_vec(vec![1.0, 0.5, 0.25, 0.125, 0.0625]);
        let tau_int = analyzer.compute_integrated_autocorr_time(&acf);

        // τ_int = 0.5 + 0.5 + 0.25 + 0.125 + 0.0625 = 1.4375
        assert_relative_eq!(tau_int, 1.4375, epsilon = 1e-10);
    }

    #[test]
    fn test_validation_insufficient_data() {
        let analyzer = MixingAnalyzer::new();
        let data = TimeSeries::from_values(vec![1.0; 50]);

        let result = analyzer.validate(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_validation_constant_data() {
        let analyzer = MixingAnalyzer::new();
        let data = TimeSeries::from_values(vec![5.0; 200]);

        let result = analyzer.validate(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_validation_nan_data() {
        let analyzer = MixingAnalyzer::new();
        let mut values = vec![1.0; 200];
        values[100] = f64::NAN;
        let data = TimeSeries::from_values(values);

        let result = analyzer.validate(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_required_sample_size() {
        let analyzer = MixingAnalyzer::new();
        let required = analyzer.required_sample_size();
        assert!(required >= 80); // 20 lags * 4
    }

    #[test]
    fn test_analyze_returns_all_metrics() {
        use rand::thread_rng;
        use rand_distr::{Distribution, Normal as RandNormal};

        let normal = RandNormal::new(0.0, 1.0).unwrap();
        let mut rng = thread_rng();
        let data: Vec<f64> = (0..500).map(|_| normal.sample(&mut rng)).collect();

        let ts = TimeSeries::from_values(data);
        let analyzer = MixingAnalyzer::new();

        let result = analyzer.analyze(&ts).unwrap();

        assert!(result.metrics.contains_key("ljung_box_q"));
        assert!(result.metrics.contains_key("max_autocorrelation"));
        assert!(result.metrics.contains_key("effective_sample_size"));
        assert!(result.metrics.contains_key("mixing_coeff_lag1"));
        assert!(result.metrics.contains_key("acf_lag_1"));
        assert!(result.p_value.is_some());
        assert!(!result.interpretation.is_empty());
    }
}
