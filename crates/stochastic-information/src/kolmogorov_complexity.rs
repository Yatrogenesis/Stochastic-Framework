//! Kolmogorov Complexity approximation via compression
//!
//! Approximates the algorithmic complexity of data using lossless compression.
//! Lower compressed size indicates higher compressibility (more structure/patterns).
//! Higher compressed size indicates lower compressibility (more randomness).
//!
//! # Mathematical Foundation
//!
//! **Kolmogorov Complexity** (theoretical):
//! ```text
//! K(x) = min{|p| : U(p) = x}
//! ```
//! where:
//! - K(x) = minimum length of a program p that produces string x
//! - U = universal Turing machine
//! - |p| = length of program p
//!
//! **Compression-based approximation:**
//! ```text
//! K(x) ≈ C(x)
//! ```
//! where C(x) = compressed size using DEFLATE/gzip
//!
//! **Properties:**
//! - K(x) is incomputable (halting problem)
//! - Compression provides upper bound: C(x) ≥ K(x)
//! - Random data is incompressible: C(random) ≈ |random|
//! - Patterned data is compressible: C(pattern) << |pattern|
//!
//! # References
//!
//! - Kolmogorov, A.N. (1965). "Three approaches to the quantitative definition of information"
//! - Li, M., & Vitányi, P. (2008). "An Introduction to Kolmogorov Complexity and Its Applications" (3rd ed.)
//! - Cilibrasi, R., & Vitányi, P. (2005). "Clustering by compression"
//! - Bennet, C.H., et al. (1998). "Information distance"
//! - Delahaye, J.P., & Zenil, H. (2012). "Numerical evaluation of algorithmic complexity for short strings"

use stochastic_core::{AnalysisResult, StochasticAnalyzer, StochasticError, TimeSeries};
use flate2::write::DeflateEncoder;
use flate2::Compression;
use std::io::Write;

/// Configuration for Kolmogorov Complexity approximation
#[derive(Debug, Clone)]
pub struct KolmogorovComplexityConfig {
    /// Compression level (0-9, where 9 is best compression)
    pub compression_level: u32,
    /// Whether to include metadata in analysis
    pub include_metadata: bool,
    /// Encoding precision (decimal places for float conversion)
    pub precision: usize,
}

impl Default for KolmogorovComplexityConfig {
    fn default() -> Self {
        Self {
            compression_level: 9, // Maximum compression
            include_metadata: true,
            precision: 6, // 6 decimal places
        }
    }
}

/// Kolmogorov Complexity approximation analyzer
pub struct KolmogorovComplexity {
    config: KolmogorovComplexityConfig,
}

impl KolmogorovComplexity {
    /// Create a new Kolmogorov Complexity analyzer
    pub fn new() -> Self {
        Self {
            config: KolmogorovComplexityConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: KolmogorovComplexityConfig) -> Self {
        Self { config }
    }

    /// Set compression level (0-9)
    pub fn with_compression_level(mut self, level: u32) -> Self {
        self.config.compression_level = level.min(9);
        self
    }

    /// Set encoding precision
    pub fn with_precision(mut self, precision: usize) -> Self {
        self.config.precision = precision;
        self
    }

    /// Convert float data to byte string for compression
    fn encode_data(&self, values: &[f64]) -> Vec<u8> {
        let mut bytes = Vec::new();

        for &value in values {
            // Format float with specified precision
            let formatted = format!("{:.prec$}", value, prec = self.config.precision);
            bytes.extend_from_slice(formatted.as_bytes());
            bytes.push(b','); // Separator
        }

        bytes
    }

    /// Compress data using DEFLATE
    fn compress(&self, data: &[u8]) -> Result<Vec<u8>, StochasticError> {
        let compression = Compression::new(self.config.compression_level);
        let mut encoder = DeflateEncoder::new(Vec::new(), compression);

        encoder.write_all(data).map_err(|e| {
            StochasticError::analysis_failed(format!("Compression failed: {}", e))
        })?;

        encoder.finish().map_err(|e| {
            StochasticError::analysis_failed(format!("Compression finalization failed: {}", e))
        })
    }

    /// Calculate compression ratio
    fn compression_ratio(original_size: usize, compressed_size: usize) -> f64 {
        if original_size == 0 {
            return 0.0;
        }
        compressed_size as f64 / original_size as f64
    }

    /// Calculate normalized compression distance between two sequences
    pub fn normalized_compression_distance(
        &self,
        x: &TimeSeries,
        y: &TimeSeries,
    ) -> Result<f64, StochasticError> {
        let x_data = self.encode_data(&x.values());
        let y_data = self.encode_data(&y.values());

        let c_x = self.compress(&x_data)?.len();
        let c_y = self.compress(&y_data)?.len();

        // Concatenate x and y
        let mut xy_data = x_data.clone();
        xy_data.extend_from_slice(&y_data);
        let c_xy = self.compress(&xy_data)?.len();

        // NCD(x,y) = (C(xy) - min(C(x), C(y))) / max(C(x), C(y))
        let min_c = c_x.min(c_y) as f64;
        let max_c = c_x.max(c_y) as f64;

        if max_c == 0.0 {
            return Ok(0.0);
        }

        let ncd = (c_xy as f64 - min_c) / max_c;
        Ok(ncd.clamp(0.0, 1.0))
    }

    /// Calculate Kolmogorov Complexity approximation
    pub fn calculate(&self, data: &TimeSeries) -> Result<AnalysisResult, StochasticError> {
        self.validate(data)?;

        let values = data.values();
        let n = values.len();

        // Encode data as byte string
        let encoded = self.encode_data(&values);
        let original_size = encoded.len();

        // Compress
        let compressed = self.compress(&encoded)?;
        let compressed_size = compressed.len();

        // Calculate metrics
        let ratio = Self::compression_ratio(original_size, compressed_size);
        let space_saving = 1.0 - ratio;
        let bits_per_value = (compressed_size * 8) as f64 / n as f64;

        // Normalized complexity (0 = maximally compressible, 1 = incompressible)
        let normalized_complexity = ratio;

        // Estimate algorithmic randomness (0 = highly patterned, 1 = random)
        let randomness_score = ratio;

        // Interpretation thresholds
        let interpretation = if ratio > 0.95 {
            format!(
                "Very high complexity (ratio = {:.3}). Data is nearly incompressible, \
                 indicating high randomness or noise. Minimal patterns detected.",
                ratio
            )
        } else if ratio > 0.7 {
            format!(
                "High complexity (ratio = {:.3}). Data is mostly incompressible, \
                 suggesting significant randomness. Few patterns present.",
                ratio
            )
        } else if ratio > 0.4 {
            format!(
                "Medium complexity (ratio = {:.3}). Data shows moderate structure. \
                 {:.1}% space saving achieved through compression.",
                ratio,
                space_saving * 100.0
            )
        } else if ratio > 0.2 {
            format!(
                "Low complexity (ratio = {:.3}). Data is highly compressible, \
                 indicating strong patterns or repetition. {:.1}% space saving.",
                ratio,
                space_saving * 100.0
            )
        } else {
            format!(
                "Very low complexity (ratio = {:.3}). Data is extremely compressible, \
                 suggesting near-deterministic patterns. {:.1}% space saving.",
                ratio,
                space_saving * 100.0
            )
        };

        // Compute Shannon entropy for comparison
        let mut value_counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        for &v in &values {
            let key = format!("{:.prec$}", v, prec = self.config.precision);
            *value_counts.entry(key).or_insert(0) += 1;
        }

        let mut shannon_entropy = 0.0;
        for &count in value_counts.values() {
            let p = count as f64 / n as f64;
            shannon_entropy -= p * p.log2();
        }

        let result = AnalysisResult::new(self.name())
            .with_metric("original_size_bytes", original_size as f64)
            .with_metric("compressed_size_bytes", compressed_size as f64)
            .with_metric("compression_ratio", ratio)
            .with_metric("space_saving", space_saving)
            .with_metric("normalized_complexity", normalized_complexity)
            .with_metric("randomness_score", randomness_score)
            .with_metric("bits_per_value", bits_per_value)
            .with_metric("shannon_entropy", shannon_entropy)
            .with_metric("sample_size", n as f64)
            .with_interpretation(interpretation)
            .with_metadata("compression_algorithm", "DEFLATE")
            .with_metadata("compression_level", format!("{}", self.config.compression_level))
            .with_metadata("precision", format!("{} decimal places", self.config.precision));

        Ok(result)
    }
}

impl Default for KolmogorovComplexity {
    fn default() -> Self {
        Self::new()
    }
}

impl StochasticAnalyzer for KolmogorovComplexity {
    fn name(&self) -> &str {
        "Kolmogorov Complexity (Compression Approximation)"
    }

    fn analyze(&self, data: &TimeSeries) -> Result<AnalysisResult, StochasticError> {
        self.calculate(data)
    }

    fn required_sample_size(&self) -> usize {
        // Compression-based methods need reasonable sample size
        // Too small: compression overhead dominates
        // Too large: better approximation
        20
    }

    fn validate(&self, data: &TimeSeries) -> Result<bool, StochasticError> {
        if data.len() < self.required_sample_size() {
            return Err(StochasticError::insufficient_data(
                self.required_sample_size(),
                data.len(),
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
    fn test_constant_data_high_compression() {
        // Constant data should compress very well
        let values = vec![1.0; 100];
        let data = TimeSeries::from_values(values);

        let analyzer = KolmogorovComplexity::new();
        let result = analyzer.analyze(&data).unwrap();

        let ratio = result.metrics.get("compression_ratio").unwrap();

        // Constant data should have very low compression ratio
        assert!(*ratio < 0.3, "Constant data should compress well, got ratio: {}", ratio);
    }

    #[test]
    fn test_random_data_low_compression() {
        // Random data should not compress well
        // Simulate random with varying values
        let values: Vec<f64> = (0..100)
            .map(|i| {
                // Pseudo-random variation
                let x = (i as f64 * 0.12345).sin();
                let y = (i as f64 * 0.98765).cos();
                x * 1000.0 + y * 500.0
            })
            .collect();

        let data = TimeSeries::from_values(values);

        let analyzer = KolmogorovComplexity::new();
        let result = analyzer.analyze(&data).unwrap();

        let ratio = result.metrics.get("compression_ratio").unwrap();

        // Random-ish data should have higher compression ratio
        assert!(*ratio > 0.5, "Random data should not compress well, got ratio: {}", ratio);
    }

    #[test]
    fn test_periodic_data_medium_compression() {
        // Periodic data should compress moderately
        let values: Vec<f64> = (0..100)
            .map(|i| ((i % 10) as f64).sin())
            .collect();

        let data = TimeSeries::from_values(values);

        let analyzer = KolmogorovComplexity::new();
        let result = analyzer.analyze(&data).unwrap();

        let ratio = result.metrics.get("compression_ratio").unwrap();

        // Periodic data should compress decently
        assert!(*ratio < 0.8, "Periodic data should compress, got ratio: {}", ratio);
    }

    #[test]
    fn test_compression_levels() {
        let values: Vec<f64> = (0..100).map(|i| (i % 7) as f64).collect();
        let data = TimeSeries::from_values(values);

        // Test different compression levels
        let low_comp = KolmogorovComplexity::new().with_compression_level(1);
        let high_comp = KolmogorovComplexity::new().with_compression_level(9);

        let result_low = low_comp.analyze(&data).unwrap();
        let result_high = high_comp.analyze(&data).unwrap();

        let size_low = result_low.metrics.get("compressed_size_bytes").unwrap();
        let size_high = result_high.metrics.get("compressed_size_bytes").unwrap();

        // Higher compression level should produce smaller or equal size
        assert!(*size_high <= *size_low + 5.0, // Allow small tolerance
                "High compression should be <= low compression");
    }

    #[test]
    fn test_space_saving() {
        let values = vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]; // Repeating pattern
        let values = values.into_iter().cycle().take(100).collect();
        let data = TimeSeries::from_values(values);

        let analyzer = KolmogorovComplexity::new();
        let result = analyzer.analyze(&data).unwrap();

        let space_saving = result.metrics.get("space_saving").unwrap();

        // Repeating pattern should save significant space
        assert!(*space_saving > 0.3, "Repeating pattern should save space: {}", space_saving);
    }

    #[test]
    fn test_normalized_complexity_range() {
        let values: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let data = TimeSeries::from_values(values);

        let analyzer = KolmogorovComplexity::new();
        let result = analyzer.analyze(&data).unwrap();

        let normalized = result.metrics.get("normalized_complexity").unwrap();

        // Normalized complexity should be in [0, 1]
        assert!(*normalized >= 0.0 && *normalized <= 1.0,
                "Normalized complexity out of range: {}", normalized);
    }

    #[test]
    fn test_bits_per_value() {
        let values = vec![1.0; 50];
        let data = TimeSeries::from_values(values);

        let analyzer = KolmogorovComplexity::new();
        let result = analyzer.analyze(&data).unwrap();

        let bits_per_value = result.metrics.get("bits_per_value").unwrap();

        // For constant data, bits per value should be low
        assert!(*bits_per_value < 10.0, "Constant data should need few bits per value: {}", bits_per_value);
    }

    #[test]
    fn test_precision_effect() {
        let values: Vec<f64> = vec![1.123456789, 2.987654321, 3.456789012];
        let values = values.into_iter().cycle().take(30).collect();
        let data = TimeSeries::from_values(values);

        let low_precision = KolmogorovComplexity::new().with_precision(2);
        let high_precision = KolmogorovComplexity::new().with_precision(8);

        let result_low = low_precision.analyze(&data).unwrap();
        let result_high = high_precision.analyze(&data).unwrap();

        let size_low = result_low.metrics.get("original_size_bytes").unwrap();
        let size_high = result_high.metrics.get("original_size_bytes").unwrap();

        // Higher precision should result in larger original size
        assert!(*size_high > *size_low, "High precision should increase data size");
    }

    #[test]
    fn test_insufficient_data() {
        let data = TimeSeries::from_values(vec![1.0, 2.0]);
        let analyzer = KolmogorovComplexity::new();

        let result = analyzer.analyze(&data);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("insufficient"));
    }

    #[test]
    fn test_normalized_compression_distance() {
        // Similar sequences should have low NCD
        let x = TimeSeries::from_values(vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
        let y = TimeSeries::from_values(vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);

        let analyzer = KolmogorovComplexity::new();
        let ncd = analyzer.normalized_compression_distance(&x, &y).unwrap();

        // Identical sequences should have NCD close to 0
        assert!(ncd < 0.2, "NCD for identical sequences should be low: {}", ncd);
    }

    #[test]
    fn test_ncd_dissimilar_sequences() {
        let x = TimeSeries::from_values(vec![1.0; 30]);
        let y: Vec<f64> = (0..30).map(|i| (i as f64 * 0.789).sin()).collect();
        let y = TimeSeries::from_values(y);

        let analyzer = KolmogorovComplexity::new();
        let ncd = analyzer.normalized_compression_distance(&x, &y).unwrap();

        // Different sequences should have higher NCD
        assert!(ncd > 0.1, "NCD for different sequences should be higher: {}", ncd);
        assert!(ncd >= 0.0 && ncd <= 1.0, "NCD should be in [0,1]: {}", ncd);
    }

    #[test]
    fn test_shannon_entropy_comparison() {
        // Uniform distribution
        let values: Vec<f64> = (0..100).map(|i| (i % 10) as f64).collect();
        let data = TimeSeries::from_values(values);

        let analyzer = KolmogorovComplexity::new();
        let result = analyzer.analyze(&data).unwrap();

        let shannon = result.metrics.get("shannon_entropy").unwrap();

        // Uniform over 10 values should have entropy close to log2(10) ≈ 3.32
        assert!(*shannon > 3.0 && *shannon < 3.5,
                "Shannon entropy for uniform distribution: {}", shannon);
    }
}
