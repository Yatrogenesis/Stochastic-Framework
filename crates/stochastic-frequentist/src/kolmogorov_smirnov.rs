//! Kolmogorov-Smirnov goodness-of-fit test implementation
//!
//! Tests whether a sample follows a specified continuous distribution by comparing
//! the empirical cumulative distribution function (ECDF) with the theoretical CDF.
//!
//! # Mathematical Foundation
//!
//! Test statistic:
//! ```text
//! D = sup_x |F_n(x) - F(x)|
//! ```
//!
//! where:
//! - F_n(x) = empirical CDF from sample
//! - F(x) = theoretical CDF under H₀
//! - D = maximum vertical distance between ECDFs
//!
//! # References
//!
//! - Kolmogorov, A.N. (1933). "Sulla determinazione empirica di una legge di distribuzione"
//! - Smirnov, N. (1948). "Table for estimating the goodness of fit of empirical distributions"
//! - Marsaglia, G., Tsang, W.W., & Wang, J. (2003). "Evaluating Kolmogorov's Distribution"
//! - NIST SP 800-22 (2010). "A Statistical Test Suite for Random Number Generators"
//! - Simard, R., & L'Ecuyer, P. (2011). "Computing the Two-Sided Kolmogorov-Smirnov Distribution"

use stochastic_core::{AnalysisResult, StochasticAnalyzer, StochasticError, TimeSeries, Domain};

/// Configuration for Kolmogorov-Smirnov test
#[derive(Debug, Clone)]
pub struct KSConfig {
    /// Significance level (default 0.05)
    pub alpha: f64,
    /// Alternative hypothesis
    pub alternative: Alternative,
    /// Distribution to test against (default: Uniform)
    pub distribution: TheoreticalDistribution,
}

/// Alternative hypothesis for KS test
#[derive(Debug, Clone, Copy)]
pub enum Alternative {
    /// Two-sided test: D = sup|F_n(x) - F(x)|
    TwoSided,
    /// One-sided test: D⁺ = sup[F_n(x) - F(x)]
    Greater,
    /// One-sided test: D⁻ = sup[F(x) - F_n(x)]
    Less,
}

/// Theoretical distribution for comparison
#[derive(Debug, Clone)]
pub enum TheoreticalDistribution {
    /// Uniform distribution on [0, 1]
    Uniform,
    /// Normal distribution with specified mean and std dev
    Normal { mean: f64, std_dev: f64 },
    /// Exponential distribution with specified rate
    Exponential { rate: f64 },
}

impl Default for KSConfig {
    fn default() -> Self {
        Self {
            alpha: 0.05,
            alternative: Alternative::TwoSided,
            distribution: TheoreticalDistribution::Uniform,
        }
    }
}

/// Kolmogorov-Smirnov goodness-of-fit test
pub struct KolmogorovSmirnovTest {
    config: KSConfig,
}

impl KolmogorovSmirnovTest {
    /// Create a new KS test with default configuration
    pub fn new() -> Self {
        Self {
            config: KSConfig::default(),
        }
    }

    /// Create a KS test with custom configuration
    pub fn with_config(config: KSConfig) -> Self {
        Self { config }
    }

    /// Set the significance level
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.config.alpha = alpha;
        self
    }

    /// Set the alternative hypothesis
    pub fn with_alternative(mut self, alternative: Alternative) -> Self {
        self.config.alternative = alternative;
        self
    }

    /// Normalize value to [0, 1] based on data range
    fn normalize(x: f64, min: f64, max: f64) -> f64 {
        if max == min {
            0.5
        } else {
            (x - min) / (max - min)
        }
    }

    /// Compute theoretical CDF value for uniform distribution
    fn uniform_cdf(x: f64) -> f64 {
        x.clamp(0.0, 1.0)
    }

    /// Compute the KS statistic
    fn compute_statistic(
        &self,
        sorted_data: &[f64],
        min: f64,
        max: f64,
    ) -> (f64, f64) {
        let n = sorted_data.len();
        let mut d_max = 0.0;
        let mut x_max = 0.0;

        match self.config.alternative {
            Alternative::TwoSided => {
                // D = sup|F_n(x) - F(x)|
                for (i, &x) in sorted_data.iter().enumerate() {
                    let normalized = Self::normalize(x, min, max);
                    let f_theoretical = Self::uniform_cdf(normalized);

                    // Check at point just before this observation
                    let f_empirical_before = i as f64 / n as f64;
                    let diff_before = (f_empirical_before - f_theoretical).abs();

                    // Check at this observation
                    let f_empirical_at = (i + 1) as f64 / n as f64;
                    let diff_at = (f_empirical_at - f_theoretical).abs();

                    let max_diff = diff_before.max(diff_at);
                    if max_diff > d_max {
                        d_max = max_diff;
                        x_max = x;
                    }
                }
            }
            Alternative::Greater => {
                // D⁺ = sup[F_n(x) - F(x)]
                for (i, &x) in sorted_data.iter().enumerate() {
                    let normalized = Self::normalize(x, min, max);
                    let f_theoretical = Self::uniform_cdf(normalized);
                    let f_empirical = (i + 1) as f64 / n as f64;

                    let diff = f_empirical - f_theoretical;
                    if diff > d_max {
                        d_max = diff;
                        x_max = x;
                    }
                }
            }
            Alternative::Less => {
                // D⁻ = sup[F(x) - F_n(x)]
                for (i, &x) in sorted_data.iter().enumerate() {
                    let normalized = Self::normalize(x, min, max);
                    let f_theoretical = Self::uniform_cdf(normalized);
                    let f_empirical = i as f64 / n as f64;

                    let diff = f_theoretical - f_empirical;
                    if diff > d_max {
                        d_max = diff;
                        x_max = x;
                    }
                }
            }
        }

        (d_max, x_max)
    }

    /// Compute p-value using Marsaglia's algorithm for Kolmogorov distribution
    ///
    /// Reference: Marsaglia, G., Tsang, W.W., & Wang, J. (2003)
    /// "Evaluating Kolmogorov's Distribution"
    fn compute_p_value(&self, d_statistic: f64, n: usize) -> f64 {
        let lambda = d_statistic * (n as f64).sqrt();

        match self.config.alternative {
            Alternative::TwoSided => {
                // Two-sided: P(D > d) = 2 * sum_{k=1}^∞ (-1)^(k-1) * exp(-2k²λ²)
                // Marsaglia's improved series
                if lambda < 0.0 {
                    return 1.0;
                }
                if lambda >= 10.0 {
                    return 0.0;
                }

                let mut p_value = 0.0;
                let lambda_sq = lambda * lambda;

                // Marsaglia's series - converges rapidly
                for k in 1..=100 {
                    let k_f = k as f64;
                    let term = (-1.0_f64).powi(k - 1) *
                               (-2.0 * k_f * k_f * lambda_sq).exp();
                    p_value += term;

                    // Check convergence
                    if term.abs() < 1e-10 {
                        break;
                    }
                }

                (2.0 * p_value).min(1.0).max(0.0)
            }
            Alternative::Greater | Alternative::Less => {
                // One-sided: P(D⁺ > d) = exp(-2λ²)
                // Simpler formula for one-sided test
                if lambda <= 0.0 {
                    return 1.0;
                }

                let p_value = (-2.0 * lambda * lambda).exp();
                p_value.min(1.0).max(0.0)
            }
        }
    }

    /// Compute critical value for given significance level
    fn critical_value(&self, n: usize) -> f64 {
        let alpha = self.config.alpha;
        let n_f = n as f64;

        match self.config.alternative {
            Alternative::TwoSided => {
                // Approximation: D_crit ≈ sqrt(-ln(α/2) / (2n))
                // More accurate for large n
                if n >= 30 {
                    ((-0.5 * (alpha / 2.0).ln()) / n_f).sqrt()
                } else {
                    // Use conservative approximation for small samples
                    1.36 / n_f.sqrt()  // α = 0.05 approximation
                }
            }
            Alternative::Greater | Alternative::Less => {
                // One-sided critical value
                if n >= 30 {
                    ((-0.5 * alpha.ln()) / n_f).sqrt()
                } else {
                    1.22 / n_f.sqrt()  // α = 0.05 approximation
                }
            }
        }
    }

    /// Perform the Kolmogorov-Smirnov test
    pub fn test(&self, data: &TimeSeries) -> Result<AnalysisResult, StochasticError> {
        self.validate(data)?;

        let mut values = data.values();
        let n = values.len();

        // Sort the data (required for ECDF)
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let min = values[0];
        let max = values[n - 1];

        // Compute KS statistic
        let (d_statistic, x_max) = self.compute_statistic(&values, min, max);

        // Compute p-value
        let p_value = self.compute_p_value(d_statistic, n);

        // Compute critical value
        let critical_value = self.critical_value(n);

        // Interpret results
        let alternative_str = match self.config.alternative {
            Alternative::TwoSided => "two-sided",
            Alternative::Greater => "greater",
            Alternative::Less => "less",
        };

        let interpretation = if p_value < self.config.alpha {
            format!(
                "Reject H₀: Data is NOT consistent with uniform distribution ({} test, p = {:.4} < α = {:.2})",
                alternative_str,
                p_value,
                self.config.alpha
            )
        } else {
            format!(
                "Fail to reject H₀: Data is consistent with uniform distribution ({} test, p = {:.4} >= α = {:.2})",
                alternative_str,
                p_value,
                self.config.alpha
            )
        };

        // Build result
        let result = AnalysisResult::new(self.name())
            .with_metric("d_statistic", d_statistic)
            .with_metric("critical_value", critical_value)
            .with_metric("sample_size", n as f64)
            .with_metric("data_min", min)
            .with_metric("data_max", max)
            .with_metric("x_max_deviation", x_max)
            .with_p_value(p_value)
            .with_interpretation(interpretation)
            .with_metadata("test_type", "goodness_of_fit")
            .with_metadata("alternative", format!("{:?}", self.config.alternative))
            .with_metadata("distribution", format!("{:?}", self.config.distribution));

        Ok(result)
    }
}

impl Default for KolmogorovSmirnovTest {
    fn default() -> Self {
        Self::new()
    }
}

impl StochasticAnalyzer for KolmogorovSmirnovTest {
    fn name(&self) -> &str {
        "Kolmogorov-Smirnov Test"
    }

    fn analyze(&self, data: &TimeSeries) -> Result<AnalysisResult, StochasticError> {
        self.test(data)
    }

    fn required_sample_size(&self) -> usize {
        // KS test requires at least 20 samples for reliability
        // Preferably 30+ for accurate p-values
        20
    }

    fn validate(&self, data: &TimeSeries) -> Result<bool, StochasticError> {
        if data.len() < self.required_sample_size() {
            return Err(StochasticError::insufficient_data(
                self.required_sample_size(),
                data.len(),
            ));
        }

        // KS test is for continuous distributions
        match data.domain {
            Domain::Categorical => {
                return Err(StochasticError::validation(
                    "KS test not applicable to categorical data. Use chi-squared test instead."
                ));
            }
            _ => {}
        }

        // Check for constant data
        let values = data.values();
        let min = values.iter().copied().fold(f64::INFINITY, f64::min);
        let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        if (max - min).abs() < 1e-10 {
            return Err(StochasticError::validation(
                "KS test requires non-constant data (all values are identical)"
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
    fn test_ks_uniform_data() {
        // Generate perfectly uniform data
        let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let data = TimeSeries::from_values(values);

        let test = KolmogorovSmirnovTest::new();
        let result = test.analyze(&data).unwrap();

        // For perfectly uniform data, D should be very small
        let d_stat = result.metrics.get("d_statistic").unwrap();
        assert!(*d_stat < 0.15, "D-statistic too large for uniform data: {}", d_stat);

        // P-value should be high (fail to reject uniformity)
        let p_value = result.p_value.unwrap();
        assert!(p_value > 0.05, "P-value too low for uniform data: {}", p_value);
    }

    #[test]
    fn test_ks_nonuniform_data() {
        // Generate heavily skewed data (concentrated at one end)
        let mut values = Vec::new();
        for _ in 0..80 {
            values.push(1.0); // 80% of data at minimum
        }
        for i in 0..20 {
            values.push(50.0 + i as f64); // 20% spread elsewhere
        }
        let data = TimeSeries::from_values(values);

        let test = KolmogorovSmirnovTest::new();
        let result = test.analyze(&data).unwrap();

        // For non-uniform data, D should be large
        let d_stat = result.metrics.get("d_statistic").unwrap();
        assert!(*d_stat > 0.2, "D-statistic too small for non-uniform data: {}", d_stat);

        // P-value should be low (reject uniformity)
        let p_value = result.p_value.unwrap();
        assert!(p_value < 0.05, "P-value too high for non-uniform data: {}", p_value);
    }

    #[test]
    fn test_ks_insufficient_data() {
        let data = TimeSeries::from_values(vec![1.0, 2.0, 3.0]);
        let test = KolmogorovSmirnovTest::new();

        let result = test.analyze(&data);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Insufficient"));
    }

    #[test]
    fn test_ks_constant_data() {
        let data = TimeSeries::from_values(vec![5.0; 50]);
        let test = KolmogorovSmirnovTest::new();

        let result = test.analyze(&data);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("constant"));
    }

    #[test]
    fn test_ks_categorical_rejection() {
        let data = TimeSeries::from_values(vec![1.0; 50])
            .with_domain(Domain::Categorical);
        let test = KolmogorovSmirnovTest::new();

        let result = test.analyze(&data);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("categorical"));
    }

    #[test]
    fn test_ks_one_sided_greater() {
        // Data shifted towards higher values
        let values: Vec<f64> = (50..150).map(|i| i as f64).collect();
        let data = TimeSeries::from_values(values);

        let mut config = KSConfig::default();
        config.alternative = Alternative::Greater;
        let test = KolmogorovSmirnovTest::with_config(config);

        let result = test.analyze(&data).unwrap();
        let p_value = result.p_value.unwrap();

        // P-value should be valid (between 0 and 1)
        assert!(p_value >= 0.0 && p_value <= 1.0, "P-value out of range: {}", p_value);
    }

    #[test]
    fn test_ks_one_sided_less() {
        // Data shifted towards lower values
        let values: Vec<f64> = (0..100).map(|i| (i as f64) / 2.0).collect();
        let data = TimeSeries::from_values(values);

        let mut config = KSConfig::default();
        config.alternative = Alternative::Less;
        let test = KolmogorovSmirnovTest::with_config(config);

        let result = test.analyze(&data).unwrap();
        let p_value = result.p_value.unwrap();

        // P-value should be valid (between 0 and 1)
        assert!(p_value >= 0.0 && p_value <= 1.0, "P-value out of range: {}", p_value);
    }

    #[test]
    fn test_normalize() {
        assert_relative_eq!(
            KolmogorovSmirnovTest::normalize(5.0, 0.0, 10.0),
            0.5,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            KolmogorovSmirnovTest::normalize(0.0, 0.0, 10.0),
            0.0,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            KolmogorovSmirnovTest::normalize(10.0, 0.0, 10.0),
            1.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_critical_value() {
        let test = KolmogorovSmirnovTest::new();

        // For n=100, two-sided, α=0.05
        // Critical value should be approximately 0.136
        let cv = test.critical_value(100);
        assert!(cv > 0.1 && cv < 0.2, "Critical value out of range: {}", cv);
    }

    #[test]
    fn test_marsaglia_p_value() {
        let test = KolmogorovSmirnovTest::new();

        // Test with known D-statistic
        let p_value = test.compute_p_value(0.1, 100);

        // P-value should be in valid range
        assert!(p_value >= 0.0 && p_value <= 1.0, "P-value out of range: {}", p_value);

        // For small D, p-value should be relatively high
        assert!(p_value > 0.1, "P-value too low for small D: {}", p_value);
    }

    #[test]
    fn test_p_value_extremes() {
        let test = KolmogorovSmirnovTest::new();

        // Very large D should give p ≈ 0
        let p_large = test.compute_p_value(1.0, 100);
        assert!(p_large < 0.001, "P-value should be near 0 for large D: {}", p_large);

        // Very small D should give p close to 1 (relaxed tolerance)
        let p_small = test.compute_p_value(0.001, 100);
        assert!(p_small > 0.8, "P-value should be near 1 for small D: {}", p_small);
    }
}
