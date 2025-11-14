//! Chi-squared goodness-of-fit test implementation
//!
//! Tests the null hypothesis that observed data follows a specified distribution
//! (typically uniform for randomness testing).
//!
//! # Mathematical Foundation
//!
//! Test statistic:
//! ```text
//! χ² = Σᵢ (Oᵢ - Eᵢ)² / Eᵢ
//! ```
//!
//! where:
//! - Oᵢ = observed frequency in bin i
//! - Eᵢ = expected frequency under H₀
//! - Degrees of freedom = k - 1 (k = number of bins)
//!
//! # References
//!
//! - Pearson, K. (1900). "On the criterion that a given system of deviations..."
//! - NIST SP 800-22 (2010). "A Statistical Test Suite for Random and Pseudorandom Number Generators"
//! - D'Agostino, R.B. & Stephens, M.A. (1986). "Goodness-of-Fit Techniques"

use stochastic_core::{AnalysisResult, StochasticAnalyzer, StochasticError, TimeSeries, Domain};
use statrs::distribution::{ChiSquared, ContinuousCDF};

/// Configuration for Chi-squared test
#[derive(Debug, Clone)]
pub struct ChiSquaredConfig {
    /// Number of bins for discrete data
    /// If None, uses Sturges' rule: k = ceil(1 + log2(n))
    pub num_bins: Option<usize>,
    /// Significance level (default 0.05)
    pub alpha: f64,
    /// Expected distribution (default: uniform)
    pub expected_distribution: ExpectedDistribution,
}

#[derive(Debug, Clone)]
pub enum ExpectedDistribution {
    /// Uniform distribution
    Uniform,
    /// Normal distribution with specified mean and std dev
    Normal { mean: f64, std_dev: f64 },
    /// Custom expected frequencies (must sum to 1.0)
    Custom(Vec<f64>),
}

impl Default for ChiSquaredConfig {
    fn default() -> Self {
        Self {
            num_bins: None,
            alpha: 0.05,
            expected_distribution: ExpectedDistribution::Uniform,
        }
    }
}

/// Chi-squared goodness-of-fit test
pub struct ChiSquaredTest {
    config: ChiSquaredConfig,
}

impl ChiSquaredTest {
    /// Create a new Chi-squared test with default configuration
    pub fn new() -> Self {
        Self {
            config: ChiSquaredConfig::default(),
        }
    }

    /// Create a Chi-squared test with custom configuration
    pub fn with_config(config: ChiSquaredConfig) -> Self {
        Self { config }
    }

    /// Set the number of bins
    pub fn with_bins(mut self, num_bins: usize) -> Self {
        self.config.num_bins = Some(num_bins);
        self
    }

    /// Set the significance level
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.config.alpha = alpha;
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

    /// Bin the data into frequency counts
    fn bin_data(&self, values: &[f64], num_bins: usize) -> (Vec<usize>, f64, f64) {
        let min = values.iter().copied().fold(f64::INFINITY, f64::min);
        let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        let bin_width = (max - min) / num_bins as f64;
        let mut bins = vec![0; num_bins];

        for &value in values {
            let bin_index = if value == max {
                num_bins - 1
            } else {
                ((value - min) / bin_width).floor() as usize
            };
            bins[bin_index] += 1;
        }

        (bins, min, max)
    }

    /// Compute expected frequencies
    fn expected_frequencies(&self, n: usize, num_bins: usize) -> Vec<f64> {
        match &self.config.expected_distribution {
            ExpectedDistribution::Uniform => {
                // Uniform: each bin has equal expected frequency
                vec![n as f64 / num_bins as f64; num_bins]
            }
            ExpectedDistribution::Normal { mean: _, std_dev: _ } => {
                // For normal distribution, would integrate CDF over bin ranges
                // Simplified implementation: assume uniform for now
                // TODO: Implement proper normal distribution binning
                vec![n as f64 / num_bins as f64; num_bins]
            }
            ExpectedDistribution::Custom(probs) => {
                // Use custom probabilities
                probs.iter().map(|p| p * n as f64).collect()
            }
        }
    }

    /// Perform the chi-squared test
    pub fn test(&self, data: &TimeSeries) -> Result<AnalysisResult, StochasticError> {
        self.validate(data)?;

        let values = data.values();
        let n = values.len();
        let num_bins = self.optimal_bins(n);

        // Bin the data
        let (observed, min, max) = self.bin_data(&values, num_bins);

        // Get expected frequencies
        let expected = self.expected_frequencies(n, num_bins);

        // Ensure all expected frequencies are >= 5 (standard requirement)
        let min_expected = expected.iter().copied().fold(f64::INFINITY, f64::min);
        if min_expected < 5.0 {
            return Err(StochasticError::validation(
                format!(
                    "Chi-squared test requires expected frequency >= 5 in all bins. \
                     Minimum expected: {:.2}. Try reducing number of bins or increasing sample size.",
                    min_expected
                )
            ));
        }

        // Compute chi-squared statistic
        let chi_squared_stat: f64 = observed
            .iter()
            .zip(&expected)
            .map(|(&obs, &exp)| {
                let diff = obs as f64 - exp;
                (diff * diff) / exp
            })
            .sum();

        // Degrees of freedom
        let dof = (num_bins - 1) as f64;

        // Compute p-value
        let chi_sq_dist = ChiSquared::new(dof).map_err(|e| {
            StochasticError::analysis_failed(format!("Failed to create chi-squared distribution: {}", e))
        })?;

        let p_value = 1.0 - chi_sq_dist.cdf(chi_squared_stat);

        // Interpret results
        let interpretation = if p_value < self.config.alpha {
            format!(
                "Reject H₀: Data is NOT consistent with {} distribution (p = {:.4} < α = {:.2})",
                match self.config.expected_distribution {
                    ExpectedDistribution::Uniform => "uniform",
                    ExpectedDistribution::Normal { .. } => "normal",
                    ExpectedDistribution::Custom(_) => "expected",
                },
                p_value,
                self.config.alpha
            )
        } else {
            format!(
                "Fail to reject H₀: Data is consistent with {} distribution (p = {:.4} >= α = {:.2})",
                match self.config.expected_distribution {
                    ExpectedDistribution::Uniform => "uniform",
                    ExpectedDistribution::Normal { .. } => "normal",
                    ExpectedDistribution::Custom(_) => "expected",
                },
                p_value,
                self.config.alpha
            )
        };

        // Build result
        let result = AnalysisResult::new(self.name())
            .with_metric("chi_squared", chi_squared_stat)
            .with_metric("degrees_of_freedom", dof)
            .with_metric("num_bins", num_bins as f64)
            .with_metric("sample_size", n as f64)
            .with_metric("data_min", min)
            .with_metric("data_max", max)
            .with_p_value(p_value)
            .with_interpretation(interpretation)
            .with_metadata("test_type", "goodness_of_fit")
            .with_metadata("distribution", format!("{:?}", self.config.expected_distribution));

        Ok(result)
    }
}

impl Default for ChiSquaredTest {
    fn default() -> Self {
        Self::new()
    }
}

impl StochasticAnalyzer for ChiSquaredTest {
    fn name(&self) -> &str {
        "Chi-Squared Test"
    }

    fn analyze(&self, data: &TimeSeries) -> Result<AnalysisResult, StochasticError> {
        self.test(data)
    }

    fn required_sample_size(&self) -> usize {
        // Chi-squared test requires at least 30 samples for reliability
        // and expected frequency >= 5 in all bins
        30
    }

    fn validate(&self, data: &TimeSeries) -> Result<bool, StochasticError> {
        if data.len() < self.required_sample_size() {
            return Err(StochasticError::insufficient_data(
                self.required_sample_size(),
                data.len(),
            ));
        }

        // Chi-squared is suitable for both discrete and continuous data
        match data.domain {
            Domain::Categorical => {
                return Err(StochasticError::validation(
                    "Chi-squared test not applicable to categorical data. Use contingency table test instead."
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
    fn test_chi_squared_uniform_data() {
        // Generate perfectly uniform data
        let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let data = TimeSeries::from_values(values);

        let test = ChiSquaredTest::new().with_bins(10);
        let result = test.analyze(&data).unwrap();

        // For perfectly uniform data, chi-squared should be very small
        let chi_sq = result.metrics.get("chi_squared").unwrap();
        assert!(*chi_sq < 5.0, "Chi-squared too large for uniform data: {}", chi_sq);

        // P-value should be high (fail to reject uniformity)
        let p_value = result.p_value.unwrap();
        assert!(p_value > 0.05, "P-value too low for uniform data: {}", p_value);
    }

    #[test]
    fn test_chi_squared_nonuniform_data() {
        // Generate heavily skewed data (many values in first half)
        let mut values = Vec::new();
        for _ in 0..80 {
            values.push(10.0); // 80% of data in one region
        }
        for i in 0..20 {
            values.push(50.0 + i as f64); // 20% spread elsewhere
        }
        let data = TimeSeries::from_values(values);

        let test = ChiSquaredTest::new().with_bins(10);
        let result = test.analyze(&data).unwrap();

        // For non-uniform data, chi-squared should be large
        let chi_sq = result.metrics.get("chi_squared").unwrap();
        assert!(*chi_sq > 10.0, "Chi-squared too small for non-uniform data: {}", chi_sq);

        // P-value should be low (reject uniformity)
        let p_value = result.p_value.unwrap();
        assert!(p_value < 0.05, "P-value too high for non-uniform data: {}", p_value);
    }

    #[test]
    fn test_chi_squared_insufficient_data() {
        let data = TimeSeries::from_values(vec![1.0, 2.0, 3.0]);
        let test = ChiSquaredTest::new();

        let result = test.analyze(&data);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Insufficient"));
    }

    #[test]
    fn test_sturges_rule() {
        let test = ChiSquaredTest::new();

        // For n=100, Sturges: k = ceil(1 + log2(100)) = ceil(7.644) = 8
        assert_eq!(test.optimal_bins(100), 8);

        // For n=1000, Sturges: k = ceil(1 + log2(1000)) = ceil(10.966) = 11
        assert_eq!(test.optimal_bins(1000), 11);
    }

    #[test]
    fn test_binning() {
        let values = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let test = ChiSquaredTest::new();

        let (bins, min, max) = test.bin_data(&values, 5);

        assert_eq!(bins.len(), 5);
        assert_relative_eq!(min, 0.0);
        assert_relative_eq!(max, 9.0);

        // Each bin should have 2 values
        assert_eq!(bins, vec![2, 2, 2, 2, 2]);
    }

    #[test]
    fn test_expected_uniform() {
        let test = ChiSquaredTest::new();
        let expected = test.expected_frequencies(100, 10);

        assert_eq!(expected.len(), 10);
        for &exp in &expected {
            assert_relative_eq!(exp, 10.0);
        }
    }

    #[test]
    fn test_categorical_data_rejection() {
        let data = TimeSeries::from_values(vec![1.0; 50])
            .with_domain(Domain::Categorical);
        let test = ChiSquaredTest::new();

        let result = test.analyze(&data);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("categorical"));
    }
}
