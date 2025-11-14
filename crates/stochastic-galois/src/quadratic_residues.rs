//! Quadratic Residue Analysis
//!
//! Analyzes the distribution of quadratic residues and non-residues modulo prime p
//! using the Legendre symbol. Tests whether sequences exhibit balanced QR/NQR patterns
//! expected in random or pseudorandom data.
//!
//! # Mathematical Foundation
//!
//! ## Legendre Symbol
//!
//! For odd prime p and integer a, the Legendre symbol (a/p) is defined as:
//!
//! ```text
//! (a/p) = {  1  if a is a quadratic residue mod p and a ≢ 0 (mod p)
//!         { -1  if a is a quadratic non-residue mod p
//!         {  0  if a ≡ 0 (mod p)
//! ```
//!
//! ## Euler's Criterion
//!
//! The Legendre symbol can be computed efficiently using Euler's criterion:
//!
//! ```text
//! (a/p) ≡ a^((p-1)/2) (mod p)
//! ```
//!
//! ## Quadratic Residues
//!
//! An integer a is a quadratic residue (QR) modulo p if there exists x such that:
//!
//! ```text
//! x² ≡ a (mod p)
//! ```
//!
//! ## Distribution Properties
//!
//! For prime p, exactly (p-1)/2 non-zero elements are quadratic residues, and
//! (p-1)/2 are non-residues. We test this balance using chi-squared:
//!
//! ```text
//! χ² = Σᵢ (Oᵢ - Eᵢ)² / Eᵢ
//! ```
//!
//! where we expect equal counts of QR and NQR in random sequences.
//!
//! ## Law of Quadratic Reciprocity
//!
//! For distinct odd primes p and q:
//!
//! ```text
//! (p/q)(q/p) = (-1)^((p-1)(q-1)/4)
//! ```
//!
//! # References
//!
//! - Ireland, K., & Rosen, M. (1990). "A Classical Introduction to Modern Number Theory"
//!   (2nd ed.). Graduate Texts in Mathematics, Vol. 84. Springer-Verlag.
//! - Bach, E., & Shallit, J. (1996). "Algorithmic Number Theory, Vol. 1: Efficient Algorithms."
//!   MIT Press.
//! - Shparlinski, I.E. (2013). "Cryptographic Applications of Analytic Number Theory."
//!   Birkhäuser.
//! - Gauss, C.F. (1801). "Disquisitiones Arithmeticae" (foundational classic on QR).
//! - Cohen, H. (1993). "A Course in Computational Algebraic Number Theory." Springer.

use stochastic_core::{AnalysisResult, StochasticAnalyzer, StochasticError, TimeSeries};
use std::collections::HashMap;

/// Configuration for quadratic residue analysis
#[derive(Debug, Clone)]
pub struct QuadraticResidueConfig {
    /// Prime modulus p (default: 97)
    pub modulus: u64,
    /// Significance level for chi-squared test (default: 0.05)
    pub significance_level: f64,
    /// Whether to compute consecutive patterns (default: true)
    pub analyze_patterns: bool,
}

impl Default for QuadraticResidueConfig {
    fn default() -> Self {
        Self {
            modulus: 97,
            significance_level: 0.05,
            analyze_patterns: true,
        }
    }
}

impl QuadraticResidueConfig {
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

    /// Set whether to analyze consecutive patterns
    pub fn with_analyze_patterns(mut self, analyze: bool) -> Self {
        self.analyze_patterns = analyze;
        self
    }
}

/// Quadratic Residue Analyzer
pub struct QuadraticResidueAnalyzer {
    config: QuadraticResidueConfig,
}

impl QuadraticResidueAnalyzer {
    /// Create new analyzer with default configuration
    pub fn new() -> Self {
        Self {
            config: QuadraticResidueConfig::default(),
        }
    }

    /// Create analyzer with custom configuration
    pub fn with_config(config: QuadraticResidueConfig) -> Self {
        Self { config }
    }

    /// Set prime modulus
    pub fn with_modulus(mut self, p: u64) -> Self {
        self.config.modulus = p;
        self
    }

    /// Check if a number is prime
    fn is_prime(&self, n: u64) -> bool {
        if n < 2 {
            return false;
        }
        if n == 2 {
            return true;
        }
        if n % 2 == 0 {
            return false;
        }

        let mut i = 3u64;
        while i * i <= n {
            if n % i == 0 {
                return false;
            }
            i += 2;
        }
        true
    }

    /// Compute modular exponentiation: base^exp mod modulus
    fn mod_exp(&self, mut base: u64, mut exp: u64, modulus: u64) -> u64 {
        let mut result = 1u64;
        base %= modulus;

        while exp > 0 {
            if exp % 2 == 1 {
                result = ((result as u128 * base as u128) % modulus as u128) as u64;
            }
            exp >>= 1;
            base = ((base as u128 * base as u128) % modulus as u128) as u64;
        }

        result
    }

    /// Compute Legendre symbol (a/p) using Euler's criterion
    /// Returns: 1 (QR), -1 (NQR), or 0 (multiple of p)
    fn legendre_symbol(&self, a: u64, p: u64) -> i8 {
        if a % p == 0 {
            return 0;
        }

        let exp = (p - 1) / 2;
        let result = self.mod_exp(a, exp, p);

        if result == 1 {
            1 // Quadratic residue
        } else if result == p - 1 {
            -1 // Quadratic non-residue (p-1 ≡ -1 mod p)
        } else {
            0 // Should not happen for prime p
        }
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

    /// Classify elements as QR, NQR, or zero
    fn classify_residues(&self, reduced: &[u64]) -> (usize, usize, usize) {
        let p = self.config.modulus;
        let mut qr_count = 0;
        let mut nqr_count = 0;
        let mut zero_count = 0;

        for &a in reduced {
            match self.legendre_symbol(a, p) {
                1 => qr_count += 1,
                -1 => nqr_count += 1,
                0 => zero_count += 1,
                _ => unreachable!(),
            }
        }

        (qr_count, nqr_count, zero_count)
    }

    /// Chi-squared test for QR/NQR balance
    fn chi_squared_balance_test(&self, qr_count: usize, nqr_count: usize) -> (f64, f64, bool) {
        let total = qr_count + nqr_count;
        if total == 0 {
            return (0.0, 0.0, true);
        }

        let expected = total as f64 / 2.0;
        let chi_squared = (qr_count as f64 - expected).powi(2) / expected
            + (nqr_count as f64 - expected).powi(2) / expected;

        // Degrees of freedom = 1 (two categories - 1)
        let critical_value = self.chi_squared_critical(1.0, self.config.significance_level);

        let is_balanced = chi_squared <= critical_value;

        (chi_squared, critical_value, is_balanced)
    }

    /// Approximate chi-squared critical value
    fn chi_squared_critical(&self, _df: f64, alpha: f64) -> f64 {
        // For df=1 and common alpha values
        match alpha {
            a if a <= 0.001 => 10.828,
            a if a <= 0.01 => 6.635,
            a if a <= 0.05 => 3.841,
            a if a <= 0.10 => 2.706,
            _ => 1.642,
        }
    }

    /// Analyze consecutive QR/NQR patterns
    fn analyze_consecutive_patterns(&self, reduced: &[u64]) -> HashMap<String, usize> {
        let p = self.config.modulus;
        let mut patterns = HashMap::new();

        for window in reduced.windows(2) {
            let s1 = self.legendre_symbol(window[0], p);
            let s2 = self.legendre_symbol(window[1], p);

            let pattern = match (s1, s2) {
                (1, 1) => "QR-QR",
                (1, -1) => "QR-NQR",
                (-1, 1) => "NQR-QR",
                (-1, -1) => "NQR-NQR",
                _ => "Zero", // Skip patterns with zeros
            };

            if pattern != "Zero" {
                *patterns.entry(pattern.to_string()).or_insert(0) += 1;
            }
        }

        patterns
    }

    /// Compute run lengths of consecutive QRs or NQRs
    fn compute_run_lengths(&self, reduced: &[u64]) -> (Vec<usize>, Vec<usize>) {
        let p = self.config.modulus;
        let mut qr_runs = Vec::new();
        let mut nqr_runs = Vec::new();

        let mut current_run = 0;
        let mut current_type: Option<i8> = None;

        for &a in reduced {
            let symbol = self.legendre_symbol(a, p);

            if symbol == 0 {
                // Reset on zero
                if current_run > 0 {
                    match current_type {
                        Some(1) => qr_runs.push(current_run),
                        Some(-1) => nqr_runs.push(current_run),
                        _ => {}
                    }
                }
                current_run = 0;
                current_type = None;
            } else if current_type == Some(symbol) {
                current_run += 1;
            } else {
                if current_run > 0 {
                    match current_type {
                        Some(1) => qr_runs.push(current_run),
                        Some(-1) => nqr_runs.push(current_run),
                        _ => {}
                    }
                }
                current_run = 1;
                current_type = Some(symbol);
            }
        }

        // Add final run
        if current_run > 0 {
            match current_type {
                Some(1) => qr_runs.push(current_run),
                Some(-1) => nqr_runs.push(current_run),
                _ => {}
            }
        }

        (qr_runs, nqr_runs)
    }

    /// Compute average run length
    fn average_run_length(&self, runs: &[usize]) -> f64 {
        if runs.is_empty() {
            0.0
        } else {
            runs.iter().sum::<usize>() as f64 / runs.len() as f64
        }
    }

    /// Interpret results
    fn interpret_results(
        &self,
        qr_count: usize,
        nqr_count: usize,
        zero_count: usize,
        chi_squared: f64,
        is_balanced: bool,
        avg_qr_run: f64,
        avg_nqr_run: f64,
    ) -> String {
        let p = self.config.modulus;
        let total = qr_count + nqr_count + zero_count;

        let balance = if is_balanced { "balanced" } else { "imbalanced" };

        format!(
            "Quadratic residue analysis modulo {}:\n\
             Found {} QRs ({:.2}%), {} NQRs ({:.2}%), {} zeros ({:.2}%).\n\
             Distribution is {} (χ² = {:.4}).\n\
             Average QR run length: {:.2}, NQR run length: {:.2}.\n\
             {} balance suggests {} pseudorandomness in quadratic structure.",
            p,
            qr_count,
            100.0 * qr_count as f64 / total as f64,
            nqr_count,
            100.0 * nqr_count as f64 / total as f64,
            zero_count,
            100.0 * zero_count as f64 / total as f64,
            balance,
            chi_squared,
            avg_qr_run,
            avg_nqr_run,
            if is_balanced { "Balanced" } else { "Imbalanced" },
            if is_balanced { "good" } else { "poor" }
        )
    }
}

impl Default for QuadraticResidueAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl StochasticAnalyzer for QuadraticResidueAnalyzer {
    fn name(&self) -> &str {
        "Quadratic Residue Analysis"
    }

    fn analyze(&self, data: &TimeSeries) -> Result<AnalysisResult, StochasticError> {
        self.validate(data)?;

        let p = self.config.modulus;

        // Validate prime modulus and odd
        if !self.is_prime(p) {
            return Err(StochasticError::validation(format!(
                "Modulus {} is not prime. Legendre symbol requires prime p.",
                p
            )));
        }

        if p == 2 {
            return Err(StochasticError::validation(
                "Modulus must be odd prime (p ≠ 2) for meaningful QR analysis.".to_string(),
            ));
        }

        let values = data.values();
        let n = values.len();

        // Reduce values modulo p
        let reduced = self.reduce_modulo(&values);

        // Classify as QR, NQR, or zero
        let (qr_count, nqr_count, zero_count) = self.classify_residues(&reduced);

        // Chi-squared test for balance
        let (chi_squared, critical_value, is_balanced) =
            self.chi_squared_balance_test(qr_count, nqr_count);

        // Analyze patterns if enabled
        let (patterns, avg_qr_run, avg_nqr_run) = if self.config.analyze_patterns {
            let patterns = self.analyze_consecutive_patterns(&reduced);
            let (qr_runs, nqr_runs) = self.compute_run_lengths(&reduced);
            let avg_qr = self.average_run_length(&qr_runs);
            let avg_nqr = self.average_run_length(&nqr_runs);
            (Some(patterns), avg_qr, avg_nqr)
        } else {
            (None, 0.0, 0.0)
        };

        let interpretation = self.interpret_results(
            qr_count,
            nqr_count,
            zero_count,
            chi_squared,
            is_balanced,
            avg_qr_run,
            avg_nqr_run,
        );

        let mut result = AnalysisResult::new(self.name())
            .with_metric("modulus", p as f64)
            .with_metric("qr_count", qr_count as f64)
            .with_metric("nqr_count", nqr_count as f64)
            .with_metric("zero_count", zero_count as f64)
            .with_metric("qr_percentage", 100.0 * qr_count as f64 / n as f64)
            .with_metric("nqr_percentage", 100.0 * nqr_count as f64 / n as f64)
            .with_metric("chi_squared", chi_squared)
            .with_metric("critical_value", critical_value)
            .with_metric("is_balanced", if is_balanced { 1.0 } else { 0.0 })
            .with_metadata("test", "Legendre Symbol (Euler's Criterion)")
            .with_metadata("expected_ratio", "1:1 (QR:NQR)");

        if self.config.analyze_patterns {
            result = result
                .with_metric("avg_qr_run_length", avg_qr_run)
                .with_metric("avg_nqr_run_length", avg_nqr_run);

            if let Some(ref pats) = patterns {
                for (pattern, count) in pats {
                    result = result.with_metric(
                        &format!("pattern_{}", pattern.replace('-', "_")),
                        *count as f64,
                    );
                }
            }
        }

        result = result.with_interpretation(interpretation);

        Ok(result)
    }

    fn required_sample_size(&self) -> usize {
        // Need sufficient samples for chi-squared test
        100
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
        if self.config.modulus < 3 {
            return Err(StochasticError::validation(
                "Modulus must be at least 3 (odd prime)".to_string(),
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
        let config = QuadraticResidueConfig::default();
        assert_eq!(config.modulus, 97);
        assert_eq!(config.significance_level, 0.05);
        assert!(config.analyze_patterns);
    }

    #[test]
    fn test_config_builder() {
        let config = QuadraticResidueConfig::new()
            .with_modulus(53)
            .with_significance_level(0.01)
            .with_analyze_patterns(false);

        assert_eq!(config.modulus, 53);
        assert_eq!(config.significance_level, 0.01);
        assert!(!config.analyze_patterns);
    }

    #[test]
    fn test_is_prime() {
        let analyzer = QuadraticResidueAnalyzer::new();

        assert!(analyzer.is_prime(3));
        assert!(analyzer.is_prime(7));
        assert!(analyzer.is_prime(97));

        assert!(!analyzer.is_prime(1));
        assert!(!analyzer.is_prime(4));
        assert!(!analyzer.is_prime(100));
    }

    #[test]
    fn test_legendre_symbol_qr() {
        let analyzer = QuadraticResidueAnalyzer::new();

        // In GF(7), 1, 2, 4 are QRs (1²=1, 3²=2, 2²=4)
        assert_eq!(analyzer.legendre_symbol(1, 7), 1);
        assert_eq!(analyzer.legendre_symbol(2, 7), 1);
        assert_eq!(analyzer.legendre_symbol(4, 7), 1);
    }

    #[test]
    fn test_legendre_symbol_nqr() {
        let analyzer = QuadraticResidueAnalyzer::new();

        // In GF(7), 3, 5, 6 are NQRs
        assert_eq!(analyzer.legendre_symbol(3, 7), -1);
        assert_eq!(analyzer.legendre_symbol(5, 7), -1);
        assert_eq!(analyzer.legendre_symbol(6, 7), -1);
    }

    #[test]
    fn test_legendre_symbol_zero() {
        let analyzer = QuadraticResidueAnalyzer::new();

        // 0 and multiples of p
        assert_eq!(analyzer.legendre_symbol(0, 7), 0);
        assert_eq!(analyzer.legendre_symbol(7, 7), 0);
        assert_eq!(analyzer.legendre_symbol(14, 7), 0);
    }

    #[test]
    fn test_classify_residues() {
        let analyzer = QuadraticResidueAnalyzer::new().with_modulus(7);
        let reduced = vec![1, 2, 3, 4, 5, 6, 0];

        let (qr, nqr, zero) = analyzer.classify_residues(&reduced);

        assert_eq!(qr, 3); // 1, 2, 4
        assert_eq!(nqr, 3); // 3, 5, 6
        assert_eq!(zero, 1); // 0
    }

    #[test]
    fn test_balanced_distribution() {
        // Create balanced QR/NQR distribution
        let mut values = Vec::new();
        for _ in 0..50 {
            values.push(1.0); // QR in GF(7)
            values.push(3.0); // NQR in GF(7)
        }

        let ts = TimeSeries::from_values(values);
        let analyzer = QuadraticResidueAnalyzer::new().with_modulus(7);

        let result = analyzer.analyze(&ts).unwrap();
        let is_balanced = result.metrics.get("is_balanced").unwrap();

        assert_relative_eq!(*is_balanced, 1.0); // Should be balanced
    }

    #[test]
    fn test_imbalanced_distribution() {
        // Create imbalanced distribution (all QRs)
        let values = vec![1.0; 200]; // 1 is QR in GF(7)

        let ts = TimeSeries::from_values(values);
        let analyzer = QuadraticResidueAnalyzer::new().with_modulus(7);

        let result = analyzer.analyze(&ts).unwrap();
        let is_balanced = result.metrics.get("is_balanced").unwrap();

        assert_relative_eq!(*is_balanced, 0.0); // Should be imbalanced
    }

    #[test]
    fn test_consecutive_patterns() {
        let analyzer = QuadraticResidueAnalyzer::new().with_modulus(7);
        let reduced = vec![1, 1, 3, 3, 2, 5]; // QR, QR, NQR, NQR, QR, NQR

        let patterns = analyzer.analyze_consecutive_patterns(&reduced);

        assert_eq!(*patterns.get("QR-QR").unwrap(), 1);
        assert_eq!(*patterns.get("QR-NQR").unwrap(), 2);
        assert_eq!(*patterns.get("NQR-NQR").unwrap(), 1);
        assert_eq!(*patterns.get("NQR-QR").unwrap(), 1);
    }

    #[test]
    fn test_run_lengths() {
        let analyzer = QuadraticResidueAnalyzer::new().with_modulus(7);
        let reduced = vec![1, 1, 1, 3, 3, 2, 5, 5, 5]; // 3 QRs, 2 NQRs, 1 QR, 3 NQRs

        let (qr_runs, nqr_runs) = analyzer.compute_run_lengths(&reduced);

        assert_eq!(qr_runs, vec![3, 1]);
        assert_eq!(nqr_runs, vec![2, 3]);
    }

    #[test]
    fn test_average_run_length() {
        let analyzer = QuadraticResidueAnalyzer::new();

        let runs = vec![1, 2, 3, 4];
        let avg = analyzer.average_run_length(&runs);
        assert_relative_eq!(avg, 2.5);

        let empty: Vec<usize> = vec![];
        let avg = analyzer.average_run_length(&empty);
        assert_relative_eq!(avg, 0.0);
    }

    #[test]
    fn test_validation_non_prime() {
        let values: Vec<f64> = (0..200).map(|x| x as f64).collect();
        let ts = TimeSeries::from_values(values);
        let analyzer = QuadraticResidueAnalyzer::new().with_modulus(100);

        let result = analyzer.analyze(&ts);
        assert!(result.is_err());
    }

    #[test]
    fn test_validation_even_prime() {
        let values: Vec<f64> = (0..200).map(|x| x as f64).collect();
        let ts = TimeSeries::from_values(values);
        let analyzer = QuadraticResidueAnalyzer::new().with_modulus(2);

        let result = analyzer.analyze(&ts);
        assert!(result.is_err());
    }

    #[test]
    fn test_validation_insufficient_data() {
        let values = vec![1.0, 2.0, 3.0];
        let ts = TimeSeries::from_values(values);
        let analyzer = QuadraticResidueAnalyzer::new();

        let result = analyzer.validate(&ts);
        assert!(result.is_err());
    }

    #[test]
    fn test_analyzer_name() {
        let analyzer = QuadraticResidueAnalyzer::new();
        assert_eq!(analyzer.name(), "Quadratic Residue Analysis");
    }

    #[test]
    fn test_mod_exp() {
        let analyzer = QuadraticResidueAnalyzer::new();

        // Test Euler's criterion: 2^3 mod 7 = 8 mod 7 = 1 (2 is QR in GF(7))
        assert_eq!(analyzer.mod_exp(2, 3, 7), 1);

        // 3^3 mod 7 = 27 mod 7 = 6 ≡ -1 (3 is NQR in GF(7))
        assert_eq!(analyzer.mod_exp(3, 3, 7), 6);
    }
}
