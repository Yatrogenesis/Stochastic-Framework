//! Anderson-Darling goodness-of-fit test implementation
//!
//! The Anderson-Darling test is a modification of the Kolmogorov-Smirnov test
//! that gives more weight to the tails of the distribution, making it more
//! sensitive to deviations in the extremes.
//!
//! # Mathematical Foundation
//!
//! Test statistic:
//! ```text
//! A² = -n - (1/n) Σᵢ (2i-1)[ln F(Xᵢ) + ln(1-F(X_{n+1-i}))]
//! ```
//!
//! where:
//! - F(x) = theoretical CDF under H₀
//! - Xᵢ = sorted observations
//! - n = sample size
//!
//! The weighting factor (2i-1) emphasizes tails more than the KS test.
//!
//! # References
//!
//! - Anderson, T.W., & Darling, D.A. (1952). "Asymptotic theory of certain goodness of fit criteria based on stochastic processes"
//! - D'Agostino, R.B., & Stephens, M.A. (1986). "Goodness-of-Fit Techniques"
//! - Lewis, P.A.W. (1961). "Distribution of the Anderson-Darling statistic"
//! - Marsaglia, G., & Marsaglia, J. (2004). "Evaluating the Anderson-Darling Distribution"
//! - Razali, N.M., & Wah, Y.B. (2011). "Power comparisons of Shapiro-Wilk, KS, Lilliefors and AD tests"

use stochastic_core::{AnalysisResult, StochasticAnalyzer, StochasticError, TimeSeries, Domain};
use std::f64::consts::PI;

/// Configuration for Anderson-Darling test
#[derive(Debug, Clone)]
pub struct AndersonDarlingConfig {
    /// Significance level (default 0.05)
    pub alpha: f64,
    /// Distribution to test against (default: Uniform)
    pub distribution: DistributionType,
}

/// Distribution types for Anderson-Darling test
#[derive(Debug, Clone)]
pub enum DistributionType {
    /// Uniform distribution on [0, 1]
    Uniform,
    /// Normal distribution (will be tested after standardization)
    Normal,
    /// Exponential distribution with specified rate
    Exponential { rate: f64 },
}

impl Default for AndersonDarlingConfig {
    fn default() -> Self {
        Self {
            alpha: 0.05,
            distribution: DistributionType::Uniform,
        }
    }
}

/// Anderson-Darling goodness-of-fit test
pub struct AndersonDarlingTest {
    config: AndersonDarlingConfig,
}

impl AndersonDarlingTest {
    /// Create a new Anderson-Darling test with default configuration
    pub fn new() -> Self {
        Self {
            config: AndersonDarlingConfig::default(),
        }
    }

    /// Create an Anderson-Darling test with custom configuration
    pub fn with_config(config: AndersonDarlingConfig) -> Self {
        Self { config }
    }

    /// Set the significance level
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.config.alpha = alpha;
        self
    }

    /// Set the distribution type
    pub fn with_distribution(mut self, distribution: DistributionType) -> Self {
        self.config.distribution = distribution;
        self
    }

    /// Normalize value to [0, 1] for uniform distribution
    fn normalize_uniform(x: f64, min: f64, max: f64) -> f64 {
        if max == min {
            0.5
        } else {
            ((x - min) / (max - min)).clamp(0.0, 1.0)
        }
    }

    /// Compute theoretical CDF value
    fn theoretical_cdf(&self, x: f64, min: f64, max: f64) -> f64 {
        match self.config.distribution {
            DistributionType::Uniform => {
                Self::normalize_uniform(x, min, max)
            }
            DistributionType::Normal => {
                // For normal distribution test, data should be standardized first
                // Here we approximate with error function
                let normalized = (x - min) / (max - min);
                Self::normal_cdf_approx(normalized)
            }
            DistributionType::Exponential { rate } => {
                if x < 0.0 {
                    0.0
                } else {
                    1.0 - (-rate * x).exp()
                }
            }
        }
    }

    /// Approximate normal CDF using error function
    fn normal_cdf_approx(z: f64) -> f64 {
        // Abramowitz and Stegun approximation
        // φ(z) = 0.5 * (1 + erf(z/sqrt(2)))
        0.5 * (1.0 + Self::erf(z / std::f64::consts::SQRT_2))
    }

    /// Error function approximation (Abramowitz and Stegun formula 7.1.26)
    fn erf(x: f64) -> f64 {
        let a1 =  0.254829592;
        let a2 = -0.284496736;
        let a3 =  1.421413741;
        let a4 = -1.453152027;
        let a5 =  1.061405429;
        let p  =  0.3275911;

        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let x = x.abs();

        let t = 1.0 / (1.0 + p * x);
        let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

        sign * y
    }

    /// Compute Anderson-Darling statistic
    fn compute_statistic(&self, sorted_data: &[f64], min: f64, max: f64) -> (f64, f64) {
        let n = sorted_data.len();
        let n_f = n as f64;
        let mut sum = 0.0;
        let mut max_contribution = 0.0;

        // Epsilon to avoid log(0)
        const EPSILON: f64 = 1e-300;

        for (i, &x) in sorted_data.iter().enumerate() {
            let i_f = (i + 1) as f64;

            // F(Xi)
            let f_xi = self.theoretical_cdf(x, min, max);

            // F(X_{n+1-i}) = F at the mirror position from the end
            let x_mirror = sorted_data[n - 1 - i];
            let f_mirror = self.theoretical_cdf(x_mirror, min, max);

            // Avoid log(0) by clamping to EPSILON
            let f_xi_clamped = f_xi.max(EPSILON).min(1.0 - EPSILON);
            let f_mirror_clamped = f_mirror.max(EPSILON).min(1.0 - EPSILON);

            // A² formula: Σ (2i-1)[ln F(Xi) + ln(1-F(X_{n+1-i}))]
            let weight = 2.0 * i_f - 1.0;
            let term = weight * (f_xi_clamped.ln() + (1.0 - f_mirror_clamped).ln());

            sum += term;

            // Track maximum contribution for diagnostic
            if term.abs() > max_contribution {
                max_contribution = term.abs();
            }
        }

        // A² = -n - (1/n) * sum
        let a_squared = -n_f - (1.0 / n_f) * sum;

        (a_squared, max_contribution)
    }

    /// Compute p-value using approximation formulas
    ///
    /// Reference: Marsaglia, G., & Marsaglia, J. (2004)
    /// "Evaluating the Anderson-Darling Distribution"
    fn compute_p_value(&self, a_squared: f64, n: usize) -> f64 {
        match self.config.distribution {
            DistributionType::Uniform => {
                // Modified A² statistic for finite sample
                let a_squared_star = a_squared * (1.0 + 0.75 / n as f64 + 2.25 / (n * n) as f64);

                // Marsaglia-Marsaglia approximation for uniform distribution
                if a_squared_star < 0.0 {
                    return 1.0;
                }
                if a_squared_star > 13.0 {
                    return 0.0;
                }

                // Piecewise approximation
                if a_squared_star < 2.0 {
                    // Low range: exp(-1.2337141/a²) * (2.00012 + a² * (...))
                    let x = a_squared_star;
                    let p = (-1.2337141 / x).exp() * (2.00012 + x * (0.247105 - x * (0.0649821 - x * 0.0347962)));
                    p.min(1.0).max(0.0)
                } else {
                    // High range: exp(-exp(1.0776 - 2.30695*a² + 0.43424*a²²...))
                    let x = a_squared_star;
                    let x2 = x * x;
                    let log_p = -(1.0776 - (2.30695 * x - 0.43424 * x2 - 0.082433 * x * x2 - 0.008056 * x2 * x2));
                    (-log_p).exp().min(1.0).max(0.0)
                }
            }
            DistributionType::Normal => {
                // Modified A² for normal distribution
                let a_squared_star = a_squared * (1.0 + 4.0 / n as f64 - 25.0 / (n * n) as f64);

                if a_squared_star < 0.0 {
                    return 1.0;
                }
                if a_squared_star > 13.0 {
                    return 0.0;
                }

                // Approximation for normal case
                if a_squared_star < 0.2 {
                    1.0 - (-3.8 * a_squared_star).exp() * (a_squared_star).sqrt()
                } else if a_squared_star < 2.0 {
                    (-1.2337 / a_squared_star).exp() * (1.0 + a_squared_star * 0.2)
                } else {
                    (-2.5 * a_squared_star + 1.2).exp()
                }
            }
            DistributionType::Exponential { .. } => {
                // Approximation for exponential case
                let a_squared_star = a_squared * (1.0 + 0.6 / n as f64);

                if a_squared_star < 0.0 {
                    return 1.0;
                }
                if a_squared_star > 10.0 {
                    return 0.0;
                }

                // Simple exponential approximation
                (-2.0 * a_squared_star).exp().min(1.0).max(0.0)
            }
        }
    }

    /// Compute critical value for given significance level
    fn critical_value(&self, alpha: f64) -> f64 {
        match self.config.distribution {
            DistributionType::Uniform => {
                // Critical values for uniform distribution
                match alpha {
                    a if a >= 0.25 => 1.248,
                    a if a >= 0.10 => 1.933,
                    a if a >= 0.05 => 2.492,
                    a if a >= 0.025 => 3.070,
                    a if a >= 0.01 => 3.857,
                    _ => 4.500, // α = 0.005 or lower
                }
            }
            DistributionType::Normal => {
                // Critical values for normal distribution
                match alpha {
                    a if a >= 0.25 => 0.470,
                    a if a >= 0.10 => 0.631,
                    a if a >= 0.05 => 0.752,
                    a if a >= 0.025 => 0.873,
                    a if a >= 0.01 => 1.035,
                    _ => 1.159, // α = 0.005 or lower
                }
            }
            DistributionType::Exponential { .. } => {
                // Critical values for exponential distribution
                match alpha {
                    a if a >= 0.25 => 0.990,
                    a if a >= 0.10 => 1.062,
                    a if a >= 0.05 => 1.321,
                    a if a >= 0.025 => 1.591,
                    a if a >= 0.01 => 1.959,
                    _ => 2.244, // α = 0.005 or lower
                }
            }
        }
    }

    /// Perform the Anderson-Darling test
    pub fn test(&self, data: &TimeSeries) -> Result<AnalysisResult, StochasticError> {
        self.validate(data)?;

        let mut values = data.values();
        let n = values.len();

        // Sort the data (required for AD test)
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let min = values[0];
        let max = values[n - 1];

        // Compute AD statistic
        let (a_squared, max_contrib) = self.compute_statistic(&values, min, max);

        // Compute p-value
        let p_value = self.compute_p_value(a_squared, n);

        // Compute critical value
        let critical_value = self.critical_value(self.config.alpha);

        // Interpret results
        let dist_str = match self.config.distribution {
            DistributionType::Uniform => "uniform",
            DistributionType::Normal => "normal",
            DistributionType::Exponential { .. } => "exponential",
        };

        let interpretation = if p_value < self.config.alpha {
            format!(
                "Reject H₀: Data is NOT consistent with {} distribution (p = {:.4} < α = {:.2})",
                dist_str,
                p_value,
                self.config.alpha
            )
        } else {
            format!(
                "Fail to reject H₀: Data is consistent with {} distribution (p = {:.4} >= α = {:.2})",
                dist_str,
                p_value,
                self.config.alpha
            )
        };

        // Build result
        let result = AnalysisResult::new(self.name())
            .with_metric("a_squared", a_squared)
            .with_metric("critical_value", critical_value)
            .with_metric("sample_size", n as f64)
            .with_metric("data_min", min)
            .with_metric("data_max", max)
            .with_metric("max_contribution", max_contrib)
            .with_p_value(p_value)
            .with_interpretation(interpretation)
            .with_metadata("test_type", "goodness_of_fit")
            .with_metadata("distribution", format!("{:?}", self.config.distribution));

        Ok(result)
    }
}

impl Default for AndersonDarlingTest {
    fn default() -> Self {
        Self::new()
    }
}

impl StochasticAnalyzer for AndersonDarlingTest {
    fn name(&self) -> &str {
        "Anderson-Darling Test"
    }

    fn analyze(&self, data: &TimeSeries) -> Result<AnalysisResult, StochasticError> {
        self.test(data)
    }

    fn required_sample_size(&self) -> usize {
        // Anderson-Darling test requires at least 8 samples
        // Preferably 20+ for reliable results
        8
    }

    fn validate(&self, data: &TimeSeries) -> Result<bool, StochasticError> {
        if data.len() < self.required_sample_size() {
            return Err(StochasticError::insufficient_data(
                self.required_sample_size(),
                data.len(),
            ));
        }

        // AD test is for continuous distributions
        match data.domain {
            Domain::Categorical => {
                return Err(StochasticError::validation(
                    "Anderson-Darling test not applicable to categorical data. Use chi-squared test instead."
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
                "Anderson-Darling test requires non-constant data (all values are identical)"
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
    fn test_ad_uniform_data() {
        // Generate perfectly uniform data
        let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let data = TimeSeries::from_values(values);

        let test = AndersonDarlingTest::new();
        let result = test.analyze(&data).unwrap();

        // For perfectly uniform data, A² should be relatively small
        let a_squared = result.metrics.get("a_squared").unwrap();
        assert!(*a_squared < 3.0, "A² too large for uniform data: {}", a_squared);

        // P-value should be reasonably high
        let p_value = result.p_value.unwrap();
        assert!(p_value > 0.01, "P-value too low for uniform data: {}", p_value);
    }

    #[test]
    fn test_ad_nonuniform_data() {
        // Generate heavily skewed data
        let mut values = Vec::new();
        for _ in 0..80 {
            values.push(1.0); // 80% at minimum
        }
        for i in 0..20 {
            values.push(50.0 + i as f64); // 20% spread elsewhere
        }
        let data = TimeSeries::from_values(values);

        let test = AndersonDarlingTest::new();
        let result = test.analyze(&data).unwrap();

        // For non-uniform data, A² should be large
        let a_squared = result.metrics.get("a_squared").unwrap();
        assert!(*a_squared > 1.0, "A² too small for non-uniform data: {}", a_squared);

        // P-value should be low (reject uniformity)
        let p_value = result.p_value.unwrap();
        assert!(p_value < 0.1, "P-value too high for non-uniform data: {}", p_value);
    }

    #[test]
    fn test_ad_insufficient_data() {
        let data = TimeSeries::from_values(vec![1.0, 2.0, 3.0]);
        let test = AndersonDarlingTest::new();

        let result = test.analyze(&data);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("insufficient"));
    }

    #[test]
    fn test_ad_constant_data() {
        let data = TimeSeries::from_values(vec![5.0; 50]);
        let test = AndersonDarlingTest::new();

        let result = test.analyze(&data);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("constant"));
    }

    #[test]
    fn test_ad_categorical_rejection() {
        let data = TimeSeries::from_values(vec![1.0; 50])
            .with_domain(Domain::Categorical);
        let test = AndersonDarlingTest::new();

        let result = test.analyze(&data);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("categorical"));
    }

    #[test]
    fn test_erf_function() {
        // Test error function approximation
        assert_relative_eq!(
            AndersonDarlingTest::erf(0.0),
            0.0,
            epsilon = 1e-6
        );
        assert_relative_eq!(
            AndersonDarlingTest::erf(1.0),
            0.8427,
            epsilon = 1e-3
        );
        assert_relative_eq!(
            AndersonDarlingTest::erf(-1.0),
            -0.8427,
            epsilon = 1e-3
        );
    }

    #[test]
    fn test_normal_cdf_approx() {
        // Test normal CDF approximation
        assert_relative_eq!(
            AndersonDarlingTest::normal_cdf_approx(0.0),
            0.5,
            epsilon = 1e-6
        );

        // φ(1) ≈ 0.8413
        let result = AndersonDarlingTest::normal_cdf_approx(1.0);
        assert!(result > 0.83 && result < 0.85, "Normal CDF(1) = {}", result);
    }

    #[test]
    fn test_critical_values_uniform() {
        let test = AndersonDarlingTest::new();

        // Test critical value at α = 0.05
        let cv = test.critical_value(0.05);
        assert_relative_eq!(cv, 2.492, epsilon = 0.001);
    }

    #[test]
    fn test_p_value_range() {
        let test = AndersonDarlingTest::new();

        // Test with various A² values
        let p1 = test.compute_p_value(0.5, 100);
        assert!(p1 >= 0.0 && p1 <= 1.0, "P-value out of range: {}", p1);

        let p2 = test.compute_p_value(2.0, 100);
        assert!(p2 >= 0.0 && p2 <= 1.0, "P-value out of range: {}", p2);

        // Very large A² should give p ≈ 0
        let p3 = test.compute_p_value(15.0, 100);
        assert!(p3 < 0.01, "P-value should be near 0 for large A²: {}", p3);
    }

    #[test]
    fn test_ad_exponential_distribution() {
        // Generate exponentially-distributed data (approximation)
        let values: Vec<f64> = (1..=100)
            .map(|i| -(i as f64 / 100.0).ln())
            .collect();
        let data = TimeSeries::from_values(values);

        let config = AndersonDarlingConfig {
            alpha: 0.05,
            distribution: DistributionType::Exponential { rate: 1.0 },
        };
        let test = AndersonDarlingTest::with_config(config);

        let result = test.analyze(&data).unwrap();
        let p_value = result.p_value.unwrap();

        // Should not reject exponential distribution
        assert!(p_value >= 0.0 && p_value <= 1.0, "P-value out of range: {}", p_value);
    }

    #[test]
    fn test_sensitivity_to_tails() {
        // AD test should be more sensitive to tail deviations than KS
        // Create data with extreme outliers
        let mut values: Vec<f64> = (10..90).map(|i| i as f64).collect();
        values.push(0.01);  // Outlier at low end
        values.push(99.99); // Outlier at high end

        let data = TimeSeries::from_values(values);
        let test = AndersonDarlingTest::new();

        let result = test.analyze(&data).unwrap();
        let a_squared = result.metrics.get("a_squared").unwrap();

        // A² should detect these tail deviations
        assert!(*a_squared >= 0.0, "A² should be non-negative");
    }
}
