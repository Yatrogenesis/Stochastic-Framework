//! Parity Distribution Test
//!
//! Analyzes the distribution of even and odd values in numerical sequences.
//! Implements chi-squared tests for parity balance and digit-level parity analysis
//! for integer data, detecting systematic biases in low-order bits.
//!
//! # Mathematical Foundation
//!
//! ## Parity Definition
//!
//! For a value x:
//! - Even: x mod 2 = 0
//! - Odd: x mod 2 = 1 (or x mod 2 ≠ 0 for floating point)
//!
//! ## Chi-Squared Parity Statistic
//!
//! For n observations with n_even even values and n_odd odd values:
//!
//! ```text
//! χ²_parity = (n_even - n_odd)² / n
//! ```
//!
//! Under H₀ (equal probability of even/odd), χ²_parity ~ χ²(1)
//!
//! Alternative formulation:
//!
//! ```text
//! χ²_parity = [(n_even - n/2)² + (n_odd - n/2)²] / (n/2)
//!           = 2 * [(n_even - n/2)² / (n/2)]
//! ```
//!
//! ## Digit Parity Analysis
//!
//! For integer sequences, analyze parity of individual decimal digits:
//!
//! ```text
//! d_k(x) = ⌊x / 10^k⌋ mod 10    for k = 0, 1, 2, ...
//! ```
//!
//! Test each digit position for parity balance using χ²(1) test.
//!
//! ## Runs Test for Parity Sequences
//!
//! Convert sequence to binary parity sequence: {0, 1, 0, 1, ...}
//! Count runs (consecutive identical values):
//!
//! ```text
//! Expected runs: μ_R = 1 + 2*n_0*n_1/n
//! Variance: σ²_R = (μ_R - 1)*(μ_R - 2)/(n - 1)
//! Z = (R - μ_R) / σ_R ~ N(0,1)
//! ```
//!
//! ## Interpretation
//!
//! - |n_even - n_odd| small: Balanced parity distribution
//! - χ²_parity small, p > 0.05: No evidence of parity bias
//! - Runs test |Z| < 1.96: No evidence of parity clustering
//! - All digit positions balanced: No low-order bit bias
//!
//! # References
//!
//! - NIST SP 800-22 Rev. 1a (2010). "A Statistical Test Suite for Random Number Generators
//!   and Testing for Randomness." Section 2.3: Frequency Test.
//! - Knuth, D.E. (1998). "The Art of Computer Programming, Vol. 2: Seminumerical Algorithms."
//!   3rd ed., Addison-Wesley. Section 3.3.2: Poker Test and Serial Test.
//! - Marsaglia, G. (1996). "DIEHARD: A Battery of Tests of Randomness."
//! - Rukhin, A., et al. (2010). "Statistical Testing of Random Number Generators."
//! - L'Ecuyer, P., & Simard, R. (2007). "TestU01: A C Library for Empirical Testing of
//!   Random Number Generators." ACM Transactions on Mathematical Software, 33(4), Article 22.

use statrs::distribution::{ChiSquared, ContinuousCDF, Normal};
use stochastic_core::{AnalysisResult, StochasticAnalyzer, StochasticError, TimeSeries};

/// Configuration for parity test analysis
#[derive(Debug, Clone)]
pub struct ParityConfig {
    /// Test parity of individual digits for integer-like data (default: false)
    pub test_digit_parity: bool,
    /// Maximum number of digit positions to test (default: 5)
    pub max_digit_positions: usize,
    /// Perform runs test on parity sequence (default: true)
    pub perform_runs_test: bool,
    /// Significance level for hypothesis testing (default: 0.05)
    pub significance_level: f64,
    /// Tolerance for considering values as integers (default: 1e-6)
    pub integer_tolerance: f64,
}

impl Default for ParityConfig {
    fn default() -> Self {
        Self {
            test_digit_parity: false,
            max_digit_positions: 5,
            perform_runs_test: true,
            significance_level: 0.05,
            integer_tolerance: 1e-6,
        }
    }
}

impl ParityConfig {
    /// Create new configuration with defaults
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable digit parity testing
    pub fn with_digit_parity(mut self, enable: bool) -> Self {
        self.test_digit_parity = enable;
        self
    }

    /// Set maximum digit positions to test
    pub fn with_max_digit_positions(mut self, max_digits: usize) -> Self {
        self.max_digit_positions = max_digits.max(1).min(10);
        self
    }

    /// Enable/disable runs test
    pub fn with_runs_test(mut self, enable: bool) -> Self {
        self.perform_runs_test = enable;
        self
    }

    /// Set significance level
    pub fn with_significance_level(mut self, alpha: f64) -> Self {
        self.significance_level = alpha.clamp(0.001, 0.5);
        self
    }
}

/// Parity Test Analyzer
pub struct ParityTestAnalyzer {
    config: ParityConfig,
}

impl ParityTestAnalyzer {
    /// Create new analyzer with default configuration
    pub fn new() -> Self {
        Self {
            config: ParityConfig::default(),
        }
    }

    /// Create analyzer with custom configuration
    pub fn with_config(config: ParityConfig) -> Self {
        Self { config }
    }

    /// Enable digit parity testing
    pub fn with_digit_parity(mut self) -> Self {
        self.config.test_digit_parity = true;
        self
    }

    /// Check if value is even (for floating point, round to nearest integer first)
    fn is_even(&self, value: f64) -> bool {
        let rounded = value.round();
        (rounded.abs() % 2.0) < 0.5
    }

    /// Check if data appears to be integer-valued
    fn is_integer_data(&self, data: &[f64]) -> bool {
        data.iter()
            .all(|&v| (v - v.round()).abs() < self.config.integer_tolerance)
    }

    /// Compute parity counts
    fn compute_parity_counts(&self, data: &[f64]) -> (usize, usize) {
        let mut n_even = 0;
        let mut n_odd = 0;

        for &value in data {
            if self.is_even(value) {
                n_even += 1;
            } else {
                n_odd += 1;
            }
        }

        (n_even, n_odd)
    }

    /// Compute chi-squared parity statistic
    fn compute_chi_squared_parity(&self, n_even: usize, n_odd: usize, n: usize) -> f64 {
        let expected = n as f64 / 2.0;
        let chi_sq_even = (n_even as f64 - expected).powi(2) / expected;
        let chi_sq_odd = (n_odd as f64 - expected).powi(2) / expected;
        chi_sq_even + chi_sq_odd
    }

    /// Compute p-value for chi-squared test with 1 degree of freedom
    fn compute_p_value(&self, chi_squared: f64) -> Result<f64, StochasticError> {
        let dist = ChiSquared::new(1.0)
            .map_err(|e| StochasticError::numerical(format!("Failed to create chi-squared distribution: {}", e)))?;

        Ok(1.0 - dist.cdf(chi_squared))
    }

    /// Extract digit at position k (k=0 is ones place)
    fn extract_digit(&self, value: f64, position: usize) -> Option<u8> {
        let abs_val = value.abs().round() as i64;
        let divisor = 10_i64.pow(position as u32);
        let digit = (abs_val / divisor) % 10;
        Some(digit as u8)
    }

    /// Test parity of specific digit position
    fn test_digit_position_parity(&self, data: &[f64], position: usize) -> Result<(f64, f64, usize, usize), StochasticError> {
        let mut n_even = 0;
        let mut n_odd = 0;
        let mut valid_count = 0;

        for &value in data {
            if let Some(digit) = self.extract_digit(value, position) {
                valid_count += 1;
                if digit % 2 == 0 {
                    n_even += 1;
                } else {
                    n_odd += 1;
                }
            }
        }

        if valid_count == 0 {
            return Err(StochasticError::numerical(
                "No valid digits found at position".to_string(),
            ));
        }

        let chi_squared = self.compute_chi_squared_parity(n_even, n_odd, valid_count);
        let p_value = self.compute_p_value(chi_squared)?;

        Ok((chi_squared, p_value, n_even, n_odd))
    }

    /// Perform runs test on parity sequence
    fn perform_runs_test(&self, data: &[f64]) -> Result<(f64, f64, usize), StochasticError> {
        // Convert to parity sequence
        let parity_seq: Vec<bool> = data.iter().map(|&v| self.is_even(v)).collect();
        let n = parity_seq.len();

        // Count 0s and 1s
        let n_0 = parity_seq.iter().filter(|&&p| p).count();
        let n_1 = n - n_0;

        if n_0 == 0 || n_1 == 0 {
            return Err(StochasticError::numerical(
                "All values have same parity, cannot perform runs test".to_string(),
            ));
        }

        // Count runs
        let mut runs = 1;
        for i in 1..n {
            if parity_seq[i] != parity_seq[i - 1] {
                runs += 1;
            }
        }

        // Compute expected runs and variance
        let n_f = n as f64;
        let n_0_f = n_0 as f64;
        let n_1_f = n_1 as f64;

        let expected_runs = 1.0 + (2.0 * n_0_f * n_1_f) / n_f;
        let variance = (2.0 * n_0_f * n_1_f * (2.0 * n_0_f * n_1_f - n_f)) / (n_f * n_f * (n_f - 1.0));
        let std_dev = variance.sqrt();

        let z_score = if std_dev > 0.0 {
            (runs as f64 - expected_runs) / std_dev
        } else {
            0.0
        };

        // Compute p-value using standard normal distribution
        let normal = Normal::new(0.0, 1.0)
            .map_err(|e| StochasticError::numerical(format!("Failed to create normal distribution: {}", e)))?;

        let p_value = 2.0 * (1.0 - normal.cdf(z_score.abs()));

        Ok((z_score, p_value, runs))
    }

    /// Compute parity balance ratio
    fn compute_balance_ratio(&self, n_even: usize, n_odd: usize) -> f64 {
        let min = n_even.min(n_odd) as f64;
        let max = n_even.max(n_odd) as f64;

        if max == 0.0 {
            0.0
        } else {
            min / max
        }
    }

    /// Generate interpretation of results
    fn interpret_results(
        &self,
        p_value: f64,
        balance_ratio: f64,
        runs_p_value: Option<f64>,
        digit_results: &[(usize, f64)],
    ) -> String {
        let alpha = self.config.significance_level;

        let parity_assessment = if p_value > alpha {
            format!(
                "BALANCED: Even/odd parity is balanced (p = {:.4} > α = {:.4})",
                p_value, alpha
            )
        } else {
            format!(
                "IMBALANCED: Significant parity imbalance detected (p = {:.4} < α = {:.4})",
                p_value, alpha
            )
        };

        let balance_quality = if balance_ratio > 0.95 {
            "Excellent balance"
        } else if balance_ratio > 0.85 {
            "Good balance"
        } else if balance_ratio > 0.70 {
            "Moderate imbalance"
        } else {
            "Significant imbalance"
        };

        let mut interpretation = format!("{}\n{} (ratio = {:.4}).", parity_assessment, balance_quality, balance_ratio);

        if let Some(runs_p) = runs_p_value {
            let runs_assessment = if runs_p > alpha {
                "No significant clustering"
            } else {
                "Parity clustering detected"
            };
            interpretation.push_str(&format!("\nRuns test: {} (p = {:.4}).", runs_assessment, runs_p));
        }

        if !digit_results.is_empty() {
            let failed_digits: Vec<usize> = digit_results
                .iter()
                .filter(|(_, p)| *p < alpha)
                .map(|(pos, _)| *pos)
                .collect();

            if failed_digits.is_empty() {
                interpretation.push_str("\nAll digit positions show balanced parity.");
            } else {
                interpretation.push_str(&format!(
                    "\nDigit parity imbalance at positions: {:?}",
                    failed_digits
                ));
            }
        }

        interpretation
    }
}

impl Default for ParityTestAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl StochasticAnalyzer for ParityTestAnalyzer {
    fn name(&self) -> &str {
        "Parity Distribution Test"
    }

    fn analyze(&self, data: &TimeSeries) -> Result<AnalysisResult, StochasticError> {
        self.validate(data)?;

        let values = data.values();
        let n = values.len();

        // Compute basic parity counts
        let (n_even, n_odd) = self.compute_parity_counts(&values);

        // Compute chi-squared statistic
        let chi_squared = self.compute_chi_squared_parity(n_even, n_odd, n);
        let p_value = self.compute_p_value(chi_squared)?;

        // Compute balance ratio
        let balance_ratio = self.compute_balance_ratio(n_even, n_odd);

        // Initialize result
        let mut result = AnalysisResult::new(self.name())
            .with_metric("chi_squared", chi_squared)
            .with_p_value(p_value)
            .with_metric("n_even", n_even as f64)
            .with_metric("n_odd", n_odd as f64)
            .with_metric("balance_ratio", balance_ratio)
            .with_metric("even_proportion", n_even as f64 / n as f64)
            .with_metric("odd_proportion", n_odd as f64 / n as f64);

        // Perform runs test if enabled
        let runs_p_value = if self.config.perform_runs_test {
            match self.perform_runs_test(&values) {
                Ok((z_score, p_val, runs)) => {
                    result = result
                        .with_metric("runs_z_score", z_score)
                        .with_metric("runs_p_value", p_val)
                        .with_metric("runs_count", runs as f64);
                    Some(p_val)
                }
                Err(_) => None,
            }
        } else {
            None
        };

        // Test digit parity if enabled and data is integer-like
        let mut digit_results = Vec::new();
        if self.config.test_digit_parity && self.is_integer_data(&values) {
            for position in 0..self.config.max_digit_positions {
                if let Ok((digit_chi_sq, digit_p_val, digit_even, digit_odd)) =
                    self.test_digit_position_parity(&values, position)
                {
                    result = result
                        .with_metric(&format!("digit_{}_chi_squared", position), digit_chi_sq)
                        .with_metric(&format!("digit_{}_p_value", position), digit_p_val)
                        .with_metric(&format!("digit_{}_even_count", position), digit_even as f64)
                        .with_metric(&format!("digit_{}_odd_count", position), digit_odd as f64);

                    digit_results.push((position, digit_p_val));
                }
            }
        }

        // Generate interpretation
        let interpretation = self.interpret_results(p_value, balance_ratio, runs_p_value, &digit_results);
        result = result.with_interpretation(interpretation);

        Ok(result)
    }

    fn required_sample_size(&self) -> usize {
        // Need at least 30 observations for chi-squared test
        // More for runs test to be meaningful
        50
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

        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_config_defaults() {
        let config = ParityConfig::default();
        assert!(!config.test_digit_parity);
        assert_eq!(config.max_digit_positions, 5);
        assert!(config.perform_runs_test);
        assert_eq!(config.significance_level, 0.05);
    }

    #[test]
    fn test_config_builder() {
        let config = ParityConfig::new()
            .with_digit_parity(true)
            .with_max_digit_positions(3)
            .with_runs_test(false)
            .with_significance_level(0.01);

        assert!(config.test_digit_parity);
        assert_eq!(config.max_digit_positions, 3);
        assert!(!config.perform_runs_test);
        assert_eq!(config.significance_level, 0.01);
    }

    #[test]
    fn test_analyzer_creation() {
        let analyzer = ParityTestAnalyzer::new();
        assert_eq!(analyzer.name(), "Parity Distribution Test");
    }

    #[test]
    fn test_is_even() {
        let analyzer = ParityTestAnalyzer::new();
        assert!(analyzer.is_even(0.0));
        assert!(analyzer.is_even(2.0));
        assert!(analyzer.is_even(4.0));
        assert!(analyzer.is_even(-2.0));
        assert!(!analyzer.is_even(1.0));
        assert!(!analyzer.is_even(3.0));
        assert!(!analyzer.is_even(-3.0));

        // Floating point values should be rounded
        assert!(analyzer.is_even(2.1));
        assert!(!analyzer.is_even(2.6));
    }

    #[test]
    fn test_parity_counts() {
        let analyzer = ParityTestAnalyzer::new();
        let data = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let (n_even, n_odd) = analyzer.compute_parity_counts(&data);

        assert_eq!(n_even, 3); // 0, 2, 4
        assert_eq!(n_odd, 3);  // 1, 3, 5
    }

    #[test]
    fn test_balanced_parity() {
        // Perfectly balanced even/odd distribution
        let data: Vec<f64> = (0..1000).map(|i| i as f64).collect();
        let ts = TimeSeries::from_values(data);
        let analyzer = ParityTestAnalyzer::new();

        let result = analyzer.analyze(&ts).unwrap();
        let p_value = result.p_value.unwrap();

        // Should have high p-value (not reject H0)
        assert!(p_value > 0.05, "p-value {} should be > 0.05 for balanced parity", p_value);
    }

    #[test]
    fn test_imbalanced_parity() {
        // Heavily biased toward even numbers
        let mut data = Vec::new();
        for i in 0..900 {
            data.push((i * 2) as f64); // Even numbers
        }
        for i in 0..100 {
            data.push((i * 2 + 1) as f64); // Odd numbers
        }

        let ts = TimeSeries::from_values(data);
        let analyzer = ParityTestAnalyzer::new();

        let result = analyzer.analyze(&ts).unwrap();
        let p_value = result.p_value.unwrap();

        // Should have low p-value (reject H0)
        assert!(p_value < 0.001, "p-value {} should be < 0.001 for imbalanced parity", p_value);
    }

    #[test]
    fn test_chi_squared_computation() {
        let analyzer = ParityTestAnalyzer::new();

        // Perfect balance
        let chi_sq = analyzer.compute_chi_squared_parity(50, 50, 100);
        assert_relative_eq!(chi_sq, 0.0, epsilon = 1e-10);

        // Imbalanced
        let chi_sq = analyzer.compute_chi_squared_parity(80, 20, 100);
        assert!(chi_sq > 0.0);
    }

    #[test]
    fn test_balance_ratio() {
        let analyzer = ParityTestAnalyzer::new();

        // Perfect balance
        let ratio = analyzer.compute_balance_ratio(50, 50);
        assert_relative_eq!(ratio, 1.0, epsilon = 1e-10);

        // Imbalanced
        let ratio = analyzer.compute_balance_ratio(80, 20);
        assert_relative_eq!(ratio, 0.25, epsilon = 1e-10);
    }

    #[test]
    fn test_is_integer_data() {
        let analyzer = ParityTestAnalyzer::new();

        let integers = vec![1.0, 2.0, 3.0, 4.0];
        assert!(analyzer.is_integer_data(&integers));

        let floats = vec![1.1, 2.2, 3.3];
        assert!(!analyzer.is_integer_data(&floats));
    }

    #[test]
    fn test_extract_digit() {
        let analyzer = ParityTestAnalyzer::new();

        assert_eq!(analyzer.extract_digit(12345.0, 0), Some(5)); // Ones place
        assert_eq!(analyzer.extract_digit(12345.0, 1), Some(4)); // Tens place
        assert_eq!(analyzer.extract_digit(12345.0, 2), Some(3)); // Hundreds place
        assert_eq!(analyzer.extract_digit(12345.0, 3), Some(2)); // Thousands place
        assert_eq!(analyzer.extract_digit(12345.0, 4), Some(1)); // Ten-thousands place
    }

    #[test]
    fn test_digit_parity() {
        let data: Vec<f64> = (0..1000).map(|i| i as f64).collect();
        let ts = TimeSeries::from_values(data);
        let config = ParityConfig::new().with_digit_parity(true);
        let analyzer = ParityTestAnalyzer::with_config(config);

        let result = analyzer.analyze(&ts).unwrap();

        // Should have digit parity metrics
        assert!(result.metrics.contains_key("digit_0_p_value"));
        assert!(result.metrics.contains_key("digit_1_p_value"));
    }

    #[test]
    fn test_runs_test_alternating() {
        // Perfectly alternating even/odd should have maximum runs
        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let analyzer = ParityTestAnalyzer::new();

        let (z_score, p_value, runs) = analyzer.perform_runs_test(&data).unwrap();

        // Alternating sequence has maximum runs
        assert_eq!(runs, 100); // Each value is a run
        assert!(p_value < 0.05, "Alternating pattern should be detected");
    }

    #[test]
    fn test_runs_test_clustered() {
        // All even then all odd
        let mut data = Vec::new();
        for i in 0..50 {
            data.push((i * 2) as f64); // Even
        }
        for i in 0..50 {
            data.push((i * 2 + 1) as f64); // Odd
        }

        let analyzer = ParityTestAnalyzer::new();
        let (z_score, p_value, runs) = analyzer.perform_runs_test(&data).unwrap();

        // Should have only 2 runs (one even block, one odd block)
        assert_eq!(runs, 2);
        assert!(p_value < 0.05, "Clustering should be detected");
    }

    #[test]
    fn test_validation_insufficient_data() {
        let analyzer = ParityTestAnalyzer::new();
        let data = TimeSeries::from_values(vec![1.0; 30]); // Less than required 50

        let result = analyzer.validate(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_validation_nan_data() {
        let analyzer = ParityTestAnalyzer::new();
        let mut values = vec![1.0; 100];
        values[50] = f64::NAN;
        let data = TimeSeries::from_values(values);

        let result = analyzer.validate(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_required_sample_size() {
        let analyzer = ParityTestAnalyzer::new();
        assert_eq!(analyzer.required_sample_size(), 50);
    }

    #[test]
    fn test_analyze_returns_all_metrics() {
        let data: Vec<f64> = (0..200).map(|i| i as f64).collect();
        let ts = TimeSeries::from_values(data);
        let analyzer = ParityTestAnalyzer::new();

        let result = analyzer.analyze(&ts).unwrap();

        assert!(result.metrics.contains_key("chi_squared"));
        assert!(result.metrics.contains_key("n_even"));
        assert!(result.metrics.contains_key("n_odd"));
        assert!(result.metrics.contains_key("balance_ratio"));
        assert!(result.metrics.contains_key("runs_z_score"));
        assert!(result.metrics.contains_key("runs_p_value"));
        assert!(result.p_value.is_some());
        assert!(!result.interpretation.is_empty());
    }
}
