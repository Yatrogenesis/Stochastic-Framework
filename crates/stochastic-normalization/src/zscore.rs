//! Z-Score Normalization and Outlier Detection
//!
//! Implements Z-score standardization and robust outlier detection methods,
//! including both classical and modified Z-score approaches. Detects anomalous
//! values that deviate significantly from the central tendency.
//!
//! # Mathematical Foundation
//!
//! ## Standard Z-Score
//!
//! The Z-score (or standard score) transforms data to have mean 0 and standard deviation 1:
//!
//! ```text
//! z = (x - μ) / σ
//! ```
//!
//! where μ is the mean and σ is the standard deviation.
//!
//! ## Outlier Detection
//!
//! A value is considered an outlier if:
//!
//! ```text
//! |z| > k
//! ```
//!
//! where k is a threshold (typically 2.5, 3, or 3.5).
//!
//! ## Modified Z-Score (Robust)
//!
//! For non-normal data or in the presence of outliers, the modified Z-score uses
//! median absolute deviation (MAD):
//!
//! ```text
//! M = median(x)
//! MAD = median(|xᵢ - M|)
//! Modified Z-score: Mᵢ = 0.6745(xᵢ - M) / MAD
//! ```
//!
//! The factor 0.6745 is the 75th percentile of the standard normal distribution,
//! making MAD comparable to standard deviation for normal distributions.
//!
//! ## Outlier Classification
//!
//! - **Mild outliers**: k < |z| ≤ k₁ (e.g., 2.5 < |z| ≤ 3.5)
//! - **Extreme outliers**: |z| > k₁ (e.g., |z| > 3.5)
//!
//! # References
//!
//! - Rousseeuw, P.J., & Hubert, M. (2011). "Robust statistics for outlier detection."
//!   Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery, 1(1), 73-79.
//! - Iglewicz, B., & Hoaglin, D.C. (1993). "How to Detect and Handle Outliers."
//!   ASQC Basic References in Quality Control, Vol. 16. ASQC Press.
//! - Leys, C., Ley, C., Klein, O., Bernard, P., & Licata, L. (2013). "Detecting outliers:
//!   Do not use standard deviation around the mean, use absolute deviation around the median."
//!   Journal of Experimental Social Psychology, 49(4), 764-766.
//! - Huber, P.J., & Ronchetti, E.M. (2009). "Robust Statistics" (2nd ed.). Wiley.
//! - Barnett, V., & Lewis, T. (1994). "Outliers in Statistical Data" (3rd ed.). Wiley.

use stochastic_core::{AnalysisResult, StochasticAnalyzer, StochasticError, TimeSeries};

/// Method for outlier detection
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OutlierMethod {
    /// Standard Z-score using mean and std dev
    Standard,
    /// Modified Z-score using median and MAD (robust)
    Modified,
}

/// Configuration for Z-score analysis
#[derive(Debug, Clone)]
pub struct ZScoreConfig {
    /// Outlier detection method (default: Standard)
    pub method: OutlierMethod,
    /// Threshold for outlier detection (default: 3.0)
    pub threshold: f64,
    /// Threshold for extreme outliers (default: 3.5)
    pub extreme_threshold: f64,
    /// Whether to compute normalized values (default: true)
    pub compute_normalized: bool,
}

impl Default for ZScoreConfig {
    fn default() -> Self {
        Self {
            method: OutlierMethod::Standard,
            threshold: 3.0,
            extreme_threshold: 3.5,
            compute_normalized: true,
        }
    }
}

impl ZScoreConfig {
    /// Create new configuration with defaults
    pub fn new() -> Self {
        Self::default()
    }

    /// Set outlier detection method
    pub fn with_method(mut self, method: OutlierMethod) -> Self {
        self.method = method;
        self
    }

    /// Set outlier threshold
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold.max(0.0);
        self
    }

    /// Set extreme outlier threshold
    pub fn with_extreme_threshold(mut self, threshold: f64) -> Self {
        self.extreme_threshold = threshold.max(0.0);
        self
    }

    /// Set whether to compute normalized values
    pub fn with_compute_normalized(mut self, compute: bool) -> Self {
        self.compute_normalized = compute;
        self
    }
}

/// Z-Score Analyzer for normalization and outlier detection
pub struct ZScoreAnalyzer {
    config: ZScoreConfig,
}

impl ZScoreAnalyzer {
    /// Create new analyzer with default configuration
    pub fn new() -> Self {
        Self {
            config: ZScoreConfig::default(),
        }
    }

    /// Create analyzer with custom configuration
    pub fn with_config(config: ZScoreConfig) -> Self {
        Self { config }
    }

    /// Set outlier detection method
    pub fn with_method(mut self, method: OutlierMethod) -> Self {
        self.config.method = method;
        self
    }

    /// Set outlier threshold
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.config.threshold = threshold.max(0.0);
        self
    }

    /// Set extreme outlier threshold
    pub fn with_extreme_threshold(mut self, threshold: f64) -> Self {
        self.config.extreme_threshold = threshold.max(0.0);
        self
    }

    /// Compute standard Z-scores
    fn compute_standard_zscores(&self, values: &[f64]) -> Vec<f64> {
        let n = values.len() as f64;
        let mean: f64 = values.iter().sum::<f64>() / n;
        let variance: f64 = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let std_dev = variance.sqrt();

        if std_dev < 1e-10 {
            // Constant data
            vec![0.0; values.len()]
        } else {
            values.iter().map(|x| (x - mean) / std_dev).collect()
        }
    }

    /// Compute median
    fn compute_median(&self, values: &[f64]) -> f64 {
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n = sorted.len();
        if n % 2 == 0 {
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        } else {
            sorted[n / 2]
        }
    }

    /// Compute median absolute deviation (MAD)
    fn compute_mad(&self, values: &[f64], median: f64) -> f64 {
        let deviations: Vec<f64> = values.iter().map(|x| (x - median).abs()).collect();
        self.compute_median(&deviations)
    }

    /// Compute modified Z-scores (robust)
    fn compute_modified_zscores(&self, values: &[f64]) -> Vec<f64> {
        let median = self.compute_median(values);
        let mad = self.compute_mad(values, median);

        // Factor 0.6745 is the 75th percentile of standard normal
        let scale_factor = 0.6745;

        if mad < 1e-10 {
            // Constant data or very low variability
            vec![0.0; values.len()]
        } else {
            values
                .iter()
                .map(|x| scale_factor * (x - median) / mad)
                .collect()
        }
    }

    /// Compute Z-scores based on configured method
    fn compute_zscores(&self, values: &[f64]) -> Vec<f64> {
        match self.config.method {
            OutlierMethod::Standard => self.compute_standard_zscores(values),
            OutlierMethod::Modified => self.compute_modified_zscores(values),
        }
    }

    /// Detect outliers based on Z-scores
    fn detect_outliers(&self, zscores: &[f64]) -> (Vec<usize>, Vec<usize>) {
        let mut mild_outliers = Vec::new();
        let mut extreme_outliers = Vec::new();

        for (i, &z) in zscores.iter().enumerate() {
            let abs_z = z.abs();
            if abs_z > self.config.extreme_threshold {
                extreme_outliers.push(i);
            } else if abs_z > self.config.threshold {
                mild_outliers.push(i);
            }
        }

        (mild_outliers, extreme_outliers)
    }

    /// Compute statistics on Z-scores
    fn compute_zscore_statistics(&self, zscores: &[f64]) -> (f64, f64, f64, f64) {
        let n = zscores.len() as f64;
        let mean: f64 = zscores.iter().sum::<f64>() / n;

        let variance: f64 = zscores.iter().map(|z| (z - mean).powi(2)).sum::<f64>() / n;
        let std_dev = variance.sqrt();

        let max_abs = zscores.iter().map(|z| z.abs()).fold(0.0f64, f64::max);

        let kurtosis = if std_dev > 1e-10 {
            let fourth_moment: f64 = zscores
                .iter()
                .map(|z| ((z - mean) / std_dev).powi(4))
                .sum::<f64>()
                / n;
            fourth_moment - 3.0 // Excess kurtosis
        } else {
            0.0
        };

        (mean, std_dev, max_abs, kurtosis)
    }

    /// Interpret results
    fn interpret_results(
        &self,
        method: OutlierMethod,
        mild_count: usize,
        extreme_count: usize,
        max_abs_z: f64,
        n: usize,
    ) -> String {
        let method_name = match method {
            OutlierMethod::Standard => "standard Z-score",
            OutlierMethod::Modified => "modified Z-score (MAD-based)",
        };

        let total_outliers = mild_count + extreme_count;
        let outlier_pct = 100.0 * total_outliers as f64 / n as f64;

        let quality = if outlier_pct < 1.0 {
            "excellent"
        } else if outlier_pct < 5.0 {
            "good"
        } else if outlier_pct < 10.0 {
            "moderate"
        } else {
            "poor"
        };

        format!(
            "Using {}, detected {} mild outliers and {} extreme outliers ({:.2}% total).\n\
             Maximum |z| = {:.3}. Data quality: {}.\n\
             {} method is {} for detecting anomalous values in {} data.",
            method_name,
            mild_count,
            extreme_count,
            outlier_pct,
            max_abs_z,
            quality,
            if method == OutlierMethod::Modified {
                "Robust"
            } else {
                "Standard"
            },
            if method == OutlierMethod::Modified {
                "recommended"
            } else {
                "appropriate"
            },
            if method == OutlierMethod::Modified {
                "non-normal or contaminated"
            } else {
                "approximately normal"
            }
        )
    }
}

impl Default for ZScoreAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl StochasticAnalyzer for ZScoreAnalyzer {
    fn name(&self) -> &str {
        match self.config.method {
            OutlierMethod::Standard => "Z-Score (Standard)",
            OutlierMethod::Modified => "Z-Score (Modified/MAD)",
        }
    }

    fn analyze(&self, data: &TimeSeries) -> Result<AnalysisResult, StochasticError> {
        self.validate(data)?;

        let values = data.values();
        let n = values.len();

        // Compute Z-scores
        let zscores = self.compute_zscores(&values);

        // Detect outliers
        let (mild_outliers, extreme_outliers) = self.detect_outliers(&zscores);

        // Compute Z-score statistics
        let (z_mean, z_std, max_abs_z, kurtosis) = self.compute_zscore_statistics(&zscores);

        // Compute original statistics for reference
        let orig_mean: f64 = values.iter().sum::<f64>() / n as f64;
        let orig_variance: f64 = values
            .iter()
            .map(|x| (x - orig_mean).powi(2))
            .sum::<f64>()
            / n as f64;
        let orig_std = orig_variance.sqrt();

        let interpretation = self.interpret_results(
            self.config.method,
            mild_outliers.len(),
            extreme_outliers.len(),
            max_abs_z,
            n,
        );

        let mut result = AnalysisResult::new(self.name())
            .with_metric("mild_outlier_count", mild_outliers.len() as f64)
            .with_metric("extreme_outlier_count", extreme_outliers.len() as f64)
            .with_metric("total_outlier_count", (mild_outliers.len() + extreme_outliers.len()) as f64)
            .with_metric(
                "outlier_percentage",
                100.0 * (mild_outliers.len() + extreme_outliers.len()) as f64 / n as f64,
            )
            .with_metric("max_abs_zscore", max_abs_z)
            .with_metric("zscore_mean", z_mean)
            .with_metric("zscore_std", z_std)
            .with_metric("zscore_kurtosis", kurtosis)
            .with_metric("original_mean", orig_mean)
            .with_metric("original_std", orig_std)
            .with_metric("threshold", self.config.threshold)
            .with_metric("extreme_threshold", self.config.extreme_threshold)
            .with_metadata(
                "method",
                match self.config.method {
                    OutlierMethod::Standard => "Standard (mean, std)",
                    OutlierMethod::Modified => "Modified (median, MAD)",
                },
            )
            .with_interpretation(interpretation);

        // Store outlier indices as metadata
        if !mild_outliers.is_empty() {
            result = result.with_metadata(
                "mild_outlier_indices",
                format!("{:?}", &mild_outliers[..mild_outliers.len().min(10)]),
            );
        }

        if !extreme_outliers.is_empty() {
            result = result.with_metadata(
                "extreme_outlier_indices",
                format!("{:?}", &extreme_outliers[..extreme_outliers.len().min(10)]),
            );
        }

        Ok(result)
    }

    fn required_sample_size(&self) -> usize {
        // Need at least 30 for meaningful statistics
        30
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

        // Check if all values are identical
        let first = values[0];
        if values.iter().all(|&v| (v - first).abs() < 1e-10) {
            return Err(StochasticError::validation(
                "Data is constant, Z-scores undefined".to_string(),
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
        let config = ZScoreConfig::default();
        assert_eq!(config.method, OutlierMethod::Standard);
        assert_eq!(config.threshold, 3.0);
        assert_eq!(config.extreme_threshold, 3.5);
    }

    #[test]
    fn test_config_builder() {
        let config = ZScoreConfig::new()
            .with_method(OutlierMethod::Modified)
            .with_threshold(2.5)
            .with_extreme_threshold(3.0);

        assert_eq!(config.method, OutlierMethod::Modified);
        assert_eq!(config.threshold, 2.5);
        assert_eq!(config.extreme_threshold, 3.0);
    }

    #[test]
    fn test_standard_zscore_computation() {
        let analyzer = ZScoreAnalyzer::new();
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let zscores = analyzer.compute_standard_zscores(&values);

        // Mean = 3, std ≈ 1.414
        // Z-scores should sum to ~0 and have std ~1
        let sum: f64 = zscores.iter().sum();
        assert!(sum.abs() < 1e-10);
    }

    #[test]
    fn test_median_computation() {
        let analyzer = ZScoreAnalyzer::new();

        // Odd length
        let values = vec![1.0, 3.0, 2.0, 5.0, 4.0];
        let median = analyzer.compute_median(&values);
        assert_relative_eq!(median, 3.0);

        // Even length
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let median = analyzer.compute_median(&values);
        assert_relative_eq!(median, 2.5);
    }

    #[test]
    fn test_mad_computation() {
        let analyzer = ZScoreAnalyzer::new();
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let median = 3.0;

        let mad = analyzer.compute_mad(&values, median);
        assert_relative_eq!(mad, 1.0); // Deviations: [2, 1, 0, 1, 2], median = 1
    }

    #[test]
    fn test_modified_zscore_computation() {
        let analyzer = ZScoreAnalyzer::new().with_method(OutlierMethod::Modified);
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let zscores = analyzer.compute_modified_zscores(&values);

        // Should have reasonable values
        assert!(zscores.iter().all(|&z| z.abs() < 10.0));
    }

    #[test]
    fn test_outlier_detection_standard() {
        let analyzer = ZScoreAnalyzer::new().with_threshold(2.0).with_extreme_threshold(3.0);

        let mut values = vec![0.0; 100];
        values.extend(vec![10.0, -10.0, 15.0]); // Clear outliers

        let ts = TimeSeries::from_values(values);
        let result = analyzer.analyze(&ts).unwrap();

        let outlier_count = result.metrics.get("total_outlier_count").unwrap();
        assert!(*outlier_count > 0.0); // Should detect outliers
    }

    #[test]
    fn test_no_outliers_normal_data() {
        let analyzer = ZScoreAnalyzer::new();

        // Standard normal-like data
        let values: Vec<f64> = (0..100).map(|i| (i as f64 - 50.0) / 20.0).collect();

        let ts = TimeSeries::from_values(values);
        let result = analyzer.analyze(&ts).unwrap();

        let outlier_count = result.metrics.get("total_outlier_count").unwrap();
        assert_relative_eq!(*outlier_count, 0.0); // Should have no outliers
    }

    #[test]
    fn test_modified_zscore_robust() {
        let analyzer = ZScoreAnalyzer::new()
            .with_method(OutlierMethod::Modified)
            .with_threshold(3.5);

        let mut values: Vec<f64> = (1..=30).map(|i| i as f64).collect();
        values.extend(vec![100.0]); // Extreme outlier

        let ts = TimeSeries::from_values(values);
        let result = analyzer.analyze(&ts).unwrap();

        let outlier_count = result.metrics.get("total_outlier_count").unwrap();
        assert!(*outlier_count >= 1.0); // Should detect the outlier
    }

    #[test]
    fn test_validation_constant_data() {
        let analyzer = ZScoreAnalyzer::new();
        let values = vec![5.0; 50];
        let ts = TimeSeries::from_values(values);

        let result = analyzer.validate(&ts);
        assert!(result.is_err());
    }

    #[test]
    fn test_validation_insufficient_data() {
        let analyzer = ZScoreAnalyzer::new();
        let values = vec![1.0, 2.0, 3.0];
        let ts = TimeSeries::from_values(values);

        let result = analyzer.validate(&ts);
        assert!(result.is_err());
    }

    #[test]
    fn test_validation_nan_data() {
        let analyzer = ZScoreAnalyzer::new();
        let mut values = vec![1.0; 50];
        values[25] = f64::NAN;
        let ts = TimeSeries::from_values(values);

        let result = analyzer.validate(&ts);
        assert!(result.is_err());
    }

    #[test]
    fn test_analyzer_name() {
        let std_analyzer = ZScoreAnalyzer::new();
        assert_eq!(std_analyzer.name(), "Z-Score (Standard)");

        let mod_analyzer = ZScoreAnalyzer::new().with_method(OutlierMethod::Modified);
        assert_eq!(mod_analyzer.name(), "Z-Score (Modified/MAD)");
    }

    #[test]
    fn test_required_sample_size() {
        let analyzer = ZScoreAnalyzer::new();
        assert_eq!(analyzer.required_sample_size(), 30);
    }

    #[test]
    fn test_zscore_statistics() {
        let analyzer = ZScoreAnalyzer::new();
        let values: Vec<f64> = (0..100).map(|i| i as f64).collect();

        let zscores = analyzer.compute_standard_zscores(&values);
        let (mean, std, _max, _kurt) = analyzer.compute_zscore_statistics(&zscores);

        // Z-scores should have mean ≈ 0 and std ≈ 1
        assert!(mean.abs() < 1e-10);
        assert_relative_eq!(std, 1.0, epsilon = 1e-2);
    }
}
