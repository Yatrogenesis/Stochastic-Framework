//! Finite Field Analysis (GF(p))
//!
//! Analyzes time series data through the lens of finite field arithmetic over GF(p),
//! where p is a prime modulus. This framework examines modular distribution properties,
//! multiplicative group structure, and field element statistics.
//!
//! # Mathematical Foundation
//!
//! ## Finite Field GF(p)
//!
//! A finite field GF(p) for prime p consists of elements {0, 1, 2, ..., p-1} with:
//!
//! - **Addition**: (a + b) mod p
//! - **Multiplication**: (a × b) mod p
//! - **Multiplicative Group**: GF(p)* = {1, 2, ..., p-1} forms a cyclic group of order p-1
//!
//! ## Order of Elements
//!
//! The order of element a ∈ GF(p)* is the smallest positive integer k such that:
//!
//! ```text
//! a^k ≡ 1 (mod p)
//! ```
//!
//! By Lagrange's theorem, the order divides φ(p) = p-1.
//!
//! ## Primitive Roots
//!
//! An element g is a primitive root modulo p if its order equals p-1:
//!
//! ```text
//! ord_p(g) = p - 1
//! ```
//!
//! Primitive roots generate the entire multiplicative group GF(p)*.
//!
//! ## Distribution Tests
//!
//! We test uniformity of reduced values using the chi-squared statistic:
//!
//! ```text
//! χ² = Σᵢ (Oᵢ - Eᵢ)² / Eᵢ
//! ```
//!
//! where Oᵢ is observed frequency and Eᵢ = n/p is expected frequency for uniform distribution.
//!
//! # References
//!
//! - Lidl, R., & Niederreiter, H. (1997). "Finite Fields" (Encyclopedia of Mathematics and its
//!   Applications, Vol. 20). Cambridge University Press.
//! - Shparlinski, I.E. (2013). "Cryptographic Applications of Analytic Number Theory: Complexity
//!   Lower Bounds and Pseudorandomness." Birkhäuser.
//! - Bach, E., & Shallit, J. (1996). "Algorithmic Number Theory, Vol. 1: Efficient Algorithms."
//!   MIT Press.
//! - Gauss, C.F. (1801). "Disquisitiones Arithmeticae" (foundational classic).
//! - Shoup, V. (2009). "A Computational Introduction to Number Theory and Algebra" (2nd ed.).
//!   Cambridge University Press.

use stochastic_core::{AnalysisResult, StochasticAnalyzer, StochasticError, TimeSeries};
use std::collections::HashMap;

/// Configuration for finite field analysis
#[derive(Debug, Clone)]
pub struct FiniteFieldConfig {
    /// Prime modulus p for GF(p) (default: 97)
    pub modulus: u64,
    /// Significance level for chi-squared test (default: 0.05)
    pub significance_level: f64,
    /// Whether to compute element orders (default: true)
    pub compute_orders: bool,
    /// Maximum modulus to test for primality (default: 10000)
    pub max_modulus: u64,
}

impl Default for FiniteFieldConfig {
    fn default() -> Self {
        Self {
            modulus: 97,
            significance_level: 0.05,
            compute_orders: true,
            max_modulus: 10000,
        }
    }
}

impl FiniteFieldConfig {
    /// Create new configuration with defaults
    pub fn new() -> Self {
        Self::default()
    }

    /// Set prime modulus
    pub fn with_modulus(mut self, p: u64) -> Self {
        self.modulus = p;
        self
    }

    /// Set significance level
    pub fn with_significance_level(mut self, alpha: f64) -> Self {
        self.significance_level = alpha.clamp(0.001, 0.999);
        self
    }

    /// Set whether to compute element orders
    pub fn with_compute_orders(mut self, compute: bool) -> Self {
        self.compute_orders = compute;
        self
    }
}

/// Finite Field Analyzer for GF(p)
pub struct FiniteFieldAnalyzer {
    config: FiniteFieldConfig,
}

impl FiniteFieldAnalyzer {
    /// Create new analyzer with default configuration
    pub fn new() -> Self {
        Self {
            config: FiniteFieldConfig::default(),
        }
    }

    /// Create analyzer with custom configuration
    pub fn with_config(config: FiniteFieldConfig) -> Self {
        Self { config }
    }

    /// Set prime modulus
    pub fn with_modulus(mut self, p: u64) -> Self {
        self.config.modulus = p;
        self
    }

    /// Check if a number is prime using trial division
    fn is_prime(&self, n: u64) -> bool {
        if n < 2 {
            return false;
        }
        if n == 2 || n == 3 {
            return true;
        }
        if n % 2 == 0 || n % 3 == 0 {
            return false;
        }

        let mut i = 5u64;
        while i * i <= n {
            if n % i == 0 || n % (i + 2) == 0 {
                return false;
            }
            i += 6;
        }
        true
    }

    /// Reduce values modulo p
    fn reduce_modulo(&self, values: &[f64]) -> Vec<u64> {
        let p = self.config.modulus;
        values
            .iter()
            .map(|&v| {
                let abs_val = v.abs();
                let truncated = abs_val.trunc() as u64;
                truncated % p
            })
            .collect()
    }

    /// Compute frequency distribution in GF(p)
    fn compute_distribution(&self, reduced: &[u64]) -> HashMap<u64, usize> {
        let mut freq = HashMap::new();
        for &val in reduced {
            *freq.entry(val).or_insert(0) += 1;
        }
        freq
    }

    /// Chi-squared test for uniform distribution
    fn chi_squared_test(&self, freq: &HashMap<u64, usize>, n: usize) -> (f64, f64, bool) {
        let p = self.config.modulus;
        let expected = n as f64 / p as f64;

        let mut chi_squared = 0.0;
        for i in 0..p {
            let observed = *freq.get(&i).unwrap_or(&0) as f64;
            chi_squared += (observed - expected).powi(2) / expected;
        }

        // Degrees of freedom: p - 1
        let df = (p - 1) as f64;

        // Critical value for chi-squared distribution (approximation)
        let critical_value = self.chi_squared_critical(df, self.config.significance_level);

        let is_uniform = chi_squared <= critical_value;

        (chi_squared, critical_value, is_uniform)
    }

    /// Approximate chi-squared critical value using Wilson-Hilferty transformation
    fn chi_squared_critical(&self, df: f64, alpha: f64) -> f64 {
        // Z-score for alpha (using standard normal approximation)
        let z = match alpha {
            a if a <= 0.001 => 3.291,
            a if a <= 0.01 => 2.576,
            a if a <= 0.05 => 1.960,
            a if a <= 0.10 => 1.645,
            _ => 1.282,
        };

        // Wilson-Hilferty transformation
        let term = 1.0 - 2.0 / (9.0 * df) + z * (2.0 / (9.0 * df)).sqrt();
        df * term.powi(3)
    }

    /// Compute modular exponentiation: base^exp mod modulus
    fn mod_exp(&self, mut base: u64, mut exp: u64, modulus: u64) -> u64 {
        let mut result = 1u64;
        base %= modulus;

        while exp > 0 {
            if exp % 2 == 1 {
                result = (result as u128 * base as u128 % modulus as u128) as u64;
            }
            exp >>= 1;
            base = (base as u128 * base as u128 % modulus as u128) as u64;
        }

        result
    }

    /// Compute multiplicative order of element a in GF(p)*
    fn compute_order(&self, a: u64, p: u64) -> Option<u64> {
        if a == 0 || a >= p {
            return None;
        }

        let phi = p - 1; // For prime p, φ(p) = p - 1

        // Find divisors of φ(p)
        let divisors = self.find_divisors(phi);

        // Order must divide φ(p), so check divisors
        for &d in &divisors {
            if self.mod_exp(a, d, p) == 1 {
                return Some(d);
            }
        }

        None
    }

    /// Find all divisors of n
    fn find_divisors(&self, n: u64) -> Vec<u64> {
        let mut divisors = Vec::new();
        let sqrt_n = (n as f64).sqrt() as u64;

        for i in 1..=sqrt_n {
            if n % i == 0 {
                divisors.push(i);
                if i != n / i {
                    divisors.push(n / i);
                }
            }
        }

        divisors.sort_unstable();
        divisors
    }

    /// Count primitive roots (elements of order p-1)
    fn count_primitive_roots(&self, reduced: &[u64]) -> usize {
        let p = self.config.modulus;
        let target_order = p - 1;

        reduced
            .iter()
            .filter(|&&a| a != 0)
            .filter(|&&a| {
                if let Some(order) = self.compute_order(a, p) {
                    order == target_order
                } else {
                    false
                }
            })
            .count()
    }

    /// Compute order statistics for non-zero elements
    fn compute_order_statistics(&self, reduced: &[u64]) -> HashMap<u64, usize> {
        let p = self.config.modulus;
        let mut order_freq = HashMap::new();

        for &a in reduced {
            if a == 0 {
                continue;
            }

            if let Some(order) = self.compute_order(a, p) {
                *order_freq.entry(order).or_insert(0) += 1;
            }
        }

        order_freq
    }

    /// Interpret results
    fn interpret_results(
        &self,
        chi_squared: f64,
        is_uniform: bool,
        primitive_count: usize,
        zero_count: usize,
        n: usize,
    ) -> String {
        let p = self.config.modulus;
        let uniformity = if is_uniform {
            "uniform"
        } else {
            "non-uniform"
        };

        let primitive_pct = if n > zero_count {
            100.0 * primitive_count as f64 / (n - zero_count) as f64
        } else {
            0.0
        };

        format!(
            "Distribution in GF({}) is {} (χ² = {:.4}). \
             Found {} primitive roots ({:.2}% of non-zero elements). \
             Zero elements: {} ({:.2}%). \
             {} distribution suggests {} randomness in modular arithmetic structure.",
            p,
            uniformity,
            chi_squared,
            primitive_count,
            primitive_pct,
            zero_count,
            100.0 * zero_count as f64 / n as f64,
            if is_uniform { "Uniform" } else { "Non-uniform" },
            if is_uniform { "good" } else { "poor" }
        )
    }
}

impl Default for FiniteFieldAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl StochasticAnalyzer for FiniteFieldAnalyzer {
    fn name(&self) -> &str {
        "Finite Field GF(p)"
    }

    fn analyze(&self, data: &TimeSeries) -> Result<AnalysisResult, StochasticError> {
        self.validate(data)?;

        let p = self.config.modulus;

        // Validate prime modulus
        if !self.is_prime(p) {
            return Err(StochasticError::validation(format!(
                "Modulus {} is not prime. Finite field GF(p) requires prime p.",
                p
            )));
        }

        let values = data.values();
        let n = values.len();

        // Reduce values modulo p
        let reduced = self.reduce_modulo(&values);

        // Compute distribution
        let freq = self.compute_distribution(&reduced);

        // Chi-squared test for uniformity
        let (chi_squared, critical_value, is_uniform) = self.chi_squared_test(&freq, n);

        // Count zeros
        let zero_count = *freq.get(&0).unwrap_or(&0);

        // Compute order statistics if enabled
        let (primitive_count, order_stats) = if self.config.compute_orders && p < self.config.max_modulus {
            let primitive = self.count_primitive_roots(&reduced);
            let orders = self.compute_order_statistics(&reduced);
            (primitive, Some(orders))
        } else {
            (0, None)
        };

        // Most common element
        let most_common = freq
            .iter()
            .max_by_key(|(_, &count)| count)
            .map(|(&elem, &count)| (elem, count))
            .unwrap_or((0, 0));

        // Compute entropy of distribution
        let entropy: f64 = freq
            .values()
            .map(|&count| {
                let prob = count as f64 / n as f64;
                if prob > 0.0 {
                    -prob * prob.log2()
                } else {
                    0.0
                }
            })
            .sum();

        let max_entropy = (p as f64).log2();
        let normalized_entropy = entropy / max_entropy;

        let interpretation = self.interpret_results(
            chi_squared,
            is_uniform,
            primitive_count,
            zero_count,
            n,
        );

        let mut result = AnalysisResult::new(self.name())
            .with_metric("modulus", p as f64)
            .with_metric("chi_squared", chi_squared)
            .with_metric("critical_value", critical_value)
            .with_metric("is_uniform", if is_uniform { 1.0 } else { 0.0 })
            .with_metric("zero_count", zero_count as f64)
            .with_metric("zero_percentage", 100.0 * zero_count as f64 / n as f64)
            .with_metric("unique_values", freq.len() as f64)
            .with_metric("coverage", 100.0 * freq.len() as f64 / p as f64)
            .with_metric("most_common_element", most_common.0 as f64)
            .with_metric("most_common_count", most_common.1 as f64)
            .with_metric("entropy", entropy)
            .with_metric("normalized_entropy", normalized_entropy)
            .with_metadata("field", format!("GF({})", p))
            .with_metadata("is_prime", "true");

        if self.config.compute_orders && order_stats.is_some() {
            result = result
                .with_metric("primitive_root_count", primitive_count as f64)
                .with_metric(
                    "primitive_root_percentage",
                    if n > zero_count {
                        100.0 * primitive_count as f64 / (n - zero_count) as f64
                    } else {
                        0.0
                    },
                );
        }

        result = result.with_interpretation(interpretation);

        Ok(result)
    }

    fn required_sample_size(&self) -> usize {
        // Need at least 5 expected counts per bin for chi-squared test
        (self.config.modulus * 5).max(100) as usize
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

        // Validate modulus
        if self.config.modulus < 2 {
            return Err(StochasticError::validation(
                "Modulus must be at least 2".to_string(),
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
        let config = FiniteFieldConfig::default();
        assert_eq!(config.modulus, 97);
        assert_eq!(config.significance_level, 0.05);
        assert!(config.compute_orders);
    }

    #[test]
    fn test_config_builder() {
        let config = FiniteFieldConfig::new()
            .with_modulus(53)
            .with_significance_level(0.01)
            .with_compute_orders(false);

        assert_eq!(config.modulus, 53);
        assert_eq!(config.significance_level, 0.01);
        assert!(!config.compute_orders);
    }

    #[test]
    fn test_is_prime() {
        let analyzer = FiniteFieldAnalyzer::new();

        assert!(analyzer.is_prime(2));
        assert!(analyzer.is_prime(3));
        assert!(analyzer.is_prime(5));
        assert!(analyzer.is_prime(97));
        assert!(analyzer.is_prime(101));

        assert!(!analyzer.is_prime(1));
        assert!(!analyzer.is_prime(4));
        assert!(!analyzer.is_prime(100));
    }

    #[test]
    fn test_reduce_modulo() {
        let analyzer = FiniteFieldAnalyzer::new().with_modulus(7);
        let values = vec![1.0, 8.0, 15.0, 22.0, -3.0, 7.0];
        let reduced = analyzer.reduce_modulo(&values);

        assert_eq!(reduced, vec![1, 1, 1, 1, 3, 0]);
    }

    #[test]
    fn test_mod_exp() {
        let analyzer = FiniteFieldAnalyzer::new();

        // 2^10 mod 13 = 1024 mod 13 = 10
        assert_eq!(analyzer.mod_exp(2, 10, 13), 10);

        // 3^4 mod 7 = 81 mod 7 = 4
        assert_eq!(analyzer.mod_exp(3, 4, 7), 4);

        // 5^0 mod 11 = 1
        assert_eq!(analyzer.mod_exp(5, 0, 11), 1);
    }

    #[test]
    fn test_compute_order() {
        let analyzer = FiniteFieldAnalyzer::new();

        // In GF(7), 3 is a primitive root with order 6
        let order = analyzer.compute_order(3, 7);
        assert_eq!(order, Some(6));

        // In GF(7), 2 has order 3
        let order = analyzer.compute_order(2, 7);
        assert_eq!(order, Some(3));

        // 0 has no order
        let order = analyzer.compute_order(0, 7);
        assert_eq!(order, None);
    }

    #[test]
    fn test_find_divisors() {
        let analyzer = FiniteFieldAnalyzer::new();

        let divisors = analyzer.find_divisors(12);
        assert_eq!(divisors, vec![1, 2, 3, 4, 6, 12]);

        let divisors = analyzer.find_divisors(7);
        assert_eq!(divisors, vec![1, 7]);
    }

    #[test]
    fn test_uniform_distribution() {
        let mut values = Vec::new();
        for i in 0..500 {
            values.push((i % 5) as f64);
        }

        let ts = TimeSeries::from_values(values);
        let analyzer = FiniteFieldAnalyzer::new().with_modulus(5);

        let result = analyzer.analyze(&ts);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert!(result.metrics.contains_key("chi_squared"));
        assert!(result.metrics.contains_key("is_uniform"));
    }

    #[test]
    fn test_non_uniform_distribution() {
        let mut values = vec![1.0; 400];
        values.extend(vec![2.0; 100]);

        let ts = TimeSeries::from_values(values);
        let analyzer = FiniteFieldAnalyzer::new().with_modulus(5);

        let result = analyzer.analyze(&ts);
        assert!(result.is_ok());

        let result = result.unwrap();
        let is_uniform = result.metrics.get("is_uniform").unwrap();
        assert_relative_eq!(*is_uniform, 0.0); // Should be non-uniform
    }

    #[test]
    fn test_validation_non_prime() {
        let values: Vec<f64> = (0..500).map(|x| x as f64).collect();
        let ts = TimeSeries::from_values(values);
        let analyzer = FiniteFieldAnalyzer::new().with_modulus(100); // Not prime

        let result = analyzer.analyze(&ts);
        assert!(result.is_err());
    }

    #[test]
    fn test_validation_insufficient_data() {
        let values = vec![1.0, 2.0, 3.0];
        let ts = TimeSeries::from_values(values);
        let analyzer = FiniteFieldAnalyzer::new();

        let result = analyzer.validate(&ts);
        assert!(result.is_err());
    }

    #[test]
    fn test_validation_nan_data() {
        let mut values = vec![1.0; 500];
        values[250] = f64::NAN;
        let ts = TimeSeries::from_values(values);
        let analyzer = FiniteFieldAnalyzer::new();

        let result = analyzer.validate(&ts);
        assert!(result.is_err());
    }

    #[test]
    fn test_entropy_computation() {
        // Uniform distribution should have high entropy
        let values: Vec<f64> = (0..1000).map(|x| (x % 7) as f64).collect();
        let ts = TimeSeries::from_values(values);
        let analyzer = FiniteFieldAnalyzer::new().with_modulus(7);

        let result = analyzer.analyze(&ts).unwrap();
        let normalized_entropy = result.metrics.get("normalized_entropy").unwrap();

        // Should be close to 1.0 for uniform distribution
        assert!(*normalized_entropy > 0.95);
    }

    #[test]
    fn test_coverage_metric() {
        let values: Vec<f64> = (0..300).map(|x| (x % 3) as f64).collect();
        let ts = TimeSeries::from_values(values);
        let analyzer = FiniteFieldAnalyzer::new().with_modulus(11);

        let result = analyzer.analyze(&ts).unwrap();
        let coverage = result.metrics.get("coverage").unwrap();

        // Only 3 out of 11 values used
        assert_relative_eq!(*coverage, 100.0 * 3.0 / 11.0, epsilon = 0.1);
    }

    #[test]
    fn test_analyzer_name() {
        let analyzer = FiniteFieldAnalyzer::new();
        assert_eq!(analyzer.name(), "Finite Field GF(p)");
    }

    #[test]
    fn test_required_sample_size() {
        let analyzer = FiniteFieldAnalyzer::new().with_modulus(11);
        let required = analyzer.required_sample_size();
        assert!(required >= 55); // At least 5 * modulus
    }
}
