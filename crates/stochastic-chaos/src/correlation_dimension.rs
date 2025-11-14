//! Correlation Dimension Analysis
//!
//! Implements the Grassberger-Procaccia algorithm for computing the correlation
//! dimension, a measure of the fractal dimension of an attractor in phase space.
//! The correlation dimension characterizes the complexity and self-similarity
//! of chaotic attractors.
//!
//! # Mathematical Foundation
//!
//! ## Correlation Integral
//!
//! The correlation integral C(r) counts the fraction of point pairs closer than
//! distance r:
//!
//! ```text
//! C(r) = lim_{N→∞} (1/N²) Σᵢ Σⱼ Θ(r - ||Xᵢ - Xⱼ||)
//! ```
//!
//! where Θ is the Heaviside step function and Xᵢ are phase space vectors.
//!
//! ## Correlation Dimension
//!
//! For self-similar (fractal) attractors, C(r) scales as a power law:
//!
//! ```text
//! C(r) ~ r^ν   for r → 0
//! ```
//!
//! The correlation dimension ν is the slope of log C(r) vs log r:
//!
//! ```text
//! ν = lim_{r→0} d(log C(r)) / d(log r)
//! ```
//!
//! ## Interpretation
//!
//! - ν ≈ 0: Point attractor (fixed point)
//! - ν ≈ 1: Limit cycle (periodic orbit)
//! - Non-integer ν: Fractal (strange) attractor
//! - ν ≥ 2: Higher-dimensional dynamics
//!
//! For chaotic systems: ν ≤ attractor dimension ≤ box-counting dimension
//!
//! ## Theiler Window
//!
//! To avoid temporal correlations, exclude pairs with |i - j| < w (Theiler window),
//! where w is typically the autocorrelation time.
//!
//! # References
//!
//! - Grassberger, P., & Procaccia, I. (1983). "Measuring the strangeness of strange
//!   attractors." Physica D, 9(1-2), 189-208.
//! - Grassberger, P., & Procaccia, I. (2004). "Measuring the strangeness of strange
//!   attractors." Physica D, 189(3-4), 13-28. [50th anniversary edition]
//! - Theiler, J. (1990). "Estimating fractal dimension." Journal of the Optical
//!   Society of America A, 7(6), 1055-1073.
//! - Kantz, H., & Schreiber, T. (2004). "Nonlinear Time Series Analysis" (2nd ed.).
//!   Cambridge University Press.
//! - Eckmann, J.P., & Ruelle, D. (1992). "Fundamental limitations for estimating
//!   dimensions and Lyapunov exponents in dynamical systems." Physica D, 56(2-3), 185-187.

use ndarray::Array2;
use stochastic_core::{AnalysisResult, StochasticAnalyzer, StochasticError, TimeSeries};

/// Configuration for correlation dimension analysis
#[derive(Debug, Clone)]
pub struct CorrelationConfig {
    /// Embedding dimension (default: 5)
    pub embedding_dimension: usize,
    /// Time delay for embedding (default: 1)
    pub time_delay: usize,
    /// Minimum radius for correlation integral (default: auto)
    pub min_radius: Option<f64>,
    /// Maximum radius for correlation integral (default: auto)
    pub max_radius: Option<f64>,
    /// Number of radius values to test (default: 30)
    pub num_radii: usize,
    /// Theiler window to avoid temporal correlations (default: 0)
    pub theiler_window: usize,
    /// Minimum number of points for linear fit (default: 10)
    pub min_fit_points: usize,
}

impl Default for CorrelationConfig {
    fn default() -> Self {
        Self {
            embedding_dimension: 5,
            time_delay: 1,
            min_radius: None,
            max_radius: None,
            num_radii: 30,
            theiler_window: 0,
            min_fit_points: 10,
        }
    }
}

impl CorrelationConfig {
    /// Create new configuration with defaults
    pub fn new() -> Self {
        Self::default()
    }

    /// Set embedding dimension
    pub fn with_embedding_dimension(mut self, dim: usize) -> Self {
        self.embedding_dimension = dim.max(2);
        self
    }

    /// Set time delay
    pub fn with_time_delay(mut self, delay: usize) -> Self {
        self.time_delay = delay.max(1);
        self
    }

    /// Set minimum radius
    pub fn with_min_radius(mut self, radius: f64) -> Self {
        self.min_radius = Some(radius.max(0.0));
        self
    }

    /// Set maximum radius
    pub fn with_max_radius(mut self, radius: f64) -> Self {
        self.max_radius = Some(radius.max(0.0));
        self
    }

    /// Set number of radii to test
    pub fn with_num_radii(mut self, num: usize) -> Self {
        self.num_radii = num.max(10);
        self
    }

    /// Set Theiler window
    pub fn with_theiler_window(mut self, window: usize) -> Self {
        self.theiler_window = window;
        self
    }

    /// Set minimum fit points
    pub fn with_min_fit_points(mut self, points: usize) -> Self {
        self.min_fit_points = points.max(5);
        self
    }
}

/// Correlation Dimension Analyzer using Grassberger-Procaccia algorithm
pub struct CorrelationDimensionAnalyzer {
    config: CorrelationConfig,
}

impl CorrelationDimensionAnalyzer {
    /// Create new analyzer with default configuration
    pub fn new() -> Self {
        Self {
            config: CorrelationConfig::default(),
        }
    }

    /// Create analyzer with custom configuration
    pub fn with_config(config: CorrelationConfig) -> Self {
        Self { config }
    }

    /// Set embedding dimension
    pub fn with_embedding_dimension(mut self, dim: usize) -> Self {
        self.config.embedding_dimension = dim.max(2);
        self
    }

    /// Set time delay
    pub fn with_time_delay(mut self, delay: usize) -> Self {
        self.config.time_delay = delay.max(1);
        self
    }

    /// Set Theiler window
    pub fn with_theiler_window(mut self, window: usize) -> Self {
        self.config.theiler_window = window;
        self
    }

    /// Reconstruct phase space using time delay embedding
    fn reconstruct_phase_space(&self, data: &[f64]) -> Array2<f64> {
        let n = data.len();
        let m = self.config.embedding_dimension;
        let tau = self.config.time_delay;

        let num_vectors = n - (m - 1) * tau;
        let mut phase_space = Array2::zeros((num_vectors, m));

        for i in 0..num_vectors {
            for j in 0..m {
                phase_space[[i, j]] = data[i + j * tau];
            }
        }

        phase_space
    }

    /// Compute Euclidean distance between two vectors
    fn euclidean_distance(&self, v1: ndarray::ArrayView1<f64>, v2: ndarray::ArrayView1<f64>) -> f64 {
        v1.iter()
            .zip(v2.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Compute all pairwise distances in phase space
    fn compute_distances(&self, phase_space: &Array2<f64>) -> Vec<f64> {
        let num_points = phase_space.nrows();
        let mut distances = Vec::new();
        let theiler = self.config.theiler_window;

        for i in 0..num_points {
            for j in (i + 1 + theiler)..num_points {
                let dist = self.euclidean_distance(phase_space.row(i), phase_space.row(j));
                distances.push(dist);
            }
        }

        distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
        distances
    }

    /// Determine radius range automatically
    fn determine_radius_range(&self, distances: &[f64]) -> (f64, f64) {
        if distances.is_empty() {
            return (1e-6, 1.0);
        }

        let min_dist = distances[0].max(1e-10);
        let max_dist = distances[distances.len() - 1];

        let min_r = self.config.min_radius.unwrap_or_else(|| {
            // Start at 1% of minimum distance or minimum non-zero distance
            let min_nonzero = distances.iter().find(|&&d| d > 1e-10).copied().unwrap_or(min_dist);
            min_nonzero * 0.01
        });

        let max_r = self.config.max_radius.unwrap_or_else(|| {
            // End at 10% of maximum distance
            max_dist * 0.1
        });

        (min_r.max(1e-10), max_r.max(min_r * 2.0))
    }

    /// Compute correlation integral for a given radius
    fn correlation_integral(&self, distances: &[f64], radius: f64) -> f64 {
        if distances.is_empty() {
            return 0.0;
        }

        // Count pairs within radius using binary search
        let count = distances.iter().take_while(|&&d| d < radius).count();

        // Normalize by total number of pairs
        count as f64 / distances.len() as f64
    }

    /// Compute correlation integrals for multiple radii
    fn compute_correlation_curve(&self, distances: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let (min_r, max_r) = self.determine_radius_range(distances);
        let num_radii = self.config.num_radii;

        let mut radii = Vec::with_capacity(num_radii);
        let mut correlations = Vec::with_capacity(num_radii);

        // Logarithmic spacing
        let log_min = min_r.ln();
        let log_max = max_r.ln();
        let log_step = (log_max - log_min) / (num_radii - 1) as f64;

        for i in 0..num_radii {
            let log_r = log_min + i as f64 * log_step;
            let r = log_r.exp();
            let c = self.correlation_integral(distances, r);

            if c > 0.0 && c < 1.0 {
                radii.push(r);
                correlations.push(c);
            }
        }

        (radii, correlations)
    }

    /// Find scaling region in log-log plot
    fn find_scaling_region(&self, log_radii: &[f64], log_correlations: &[f64]) -> Option<(usize, usize)> {
        if log_radii.len() < self.config.min_fit_points {
            return None;
        }

        let n = log_radii.len();
        let min_points = self.config.min_fit_points;

        // Try different windows and find the one with best R²
        let mut best_r_squared = 0.0;
        let mut best_region = None;

        for start in 0..=(n.saturating_sub(min_points)) {
            for end in (start + min_points)..=n {
                let window_log_r = &log_radii[start..end];
                let window_log_c = &log_correlations[start..end];

                let (_, r_squared) = self.linear_fit(window_log_r, window_log_c);

                if r_squared > best_r_squared && r_squared > 0.95 {
                    best_r_squared = r_squared;
                    best_region = Some((start, end));
                }
            }
        }

        best_region
    }

    /// Perform linear regression
    fn linear_fit(&self, x: &[f64], y: &[f64]) -> (f64, f64) {
        let n = x.len().min(y.len());
        if n < 2 {
            return (0.0, 0.0);
        }

        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xx = 0.0;
        let mut sum_xy = 0.0;
        let mut count = 0.0;

        for i in 0..n {
            if x[i].is_finite() && y[i].is_finite() {
                sum_x += x[i];
                sum_y += y[i];
                sum_xx += x[i] * x[i];
                sum_xy += x[i] * y[i];
                count += 1.0;
            }
        }

        if count < 2.0 {
            return (0.0, 0.0);
        }

        let slope = (count * sum_xy - sum_x * sum_y) / (count * sum_xx - sum_x * sum_x);

        // Compute R-squared
        let mean_y = sum_y / count;
        let mut ss_tot = 0.0;
        let mut ss_res = 0.0;

        for i in 0..n {
            if x[i].is_finite() && y[i].is_finite() {
                let y_pred = slope * (x[i] - sum_x / count) + mean_y;
                ss_tot += (y[i] - mean_y).powi(2);
                ss_res += (y[i] - y_pred).powi(2);
            }
        }

        let r_squared = if ss_tot > 1e-10 {
            1.0 - (ss_res / ss_tot)
        } else {
            0.0
        };

        (slope, r_squared)
    }

    /// Compute correlation dimension from correlation curve
    fn compute_correlation_dimension(
        &self,
        radii: &[f64],
        correlations: &[f64],
    ) -> Result<DimensionResult, StochasticError> {
        if radii.is_empty() || correlations.is_empty() {
            return Err(StochasticError::analysis_failed(
                "No valid correlation data".to_string(),
            ));
        }

        // Convert to log scale
        let log_radii: Vec<f64> = radii.iter().map(|r| r.ln()).collect();
        let log_correlations: Vec<f64> = correlations.iter().map(|c| c.ln()).collect();

        // Find scaling region
        let scaling_region = self.find_scaling_region(&log_radii, &log_correlations);

        let (dimension, r_squared, region_size) = if let Some((start, end)) = scaling_region {
            let window_log_r = &log_radii[start..end];
            let window_log_c = &log_correlations[start..end];
            let (slope, r_sq) = self.linear_fit(window_log_r, window_log_c);
            (slope, r_sq, end - start)
        } else {
            // Use entire curve if no good scaling region found
            let (slope, r_sq) = self.linear_fit(&log_radii, &log_correlations);
            (slope, r_sq, log_radii.len())
        };

        Ok(DimensionResult {
            correlation_dimension: dimension,
            r_squared,
            num_radii: radii.len(),
            scaling_region_size: region_size,
            min_radius: *radii.first().unwrap(),
            max_radius: *radii.last().unwrap(),
        })
    }

    /// Compute correlation dimension for multiple embedding dimensions
    fn compute_for_multiple_dimensions(
        &self,
        data: &[f64],
        max_dim: usize,
    ) -> Vec<(usize, f64, f64)> {
        let mut results = Vec::new();

        for m in 2..=max_dim {
            let mut config = self.config.clone();
            config.embedding_dimension = m;
            let analyzer = Self::with_config(config);

            let phase_space = analyzer.reconstruct_phase_space(data);
            let distances = analyzer.compute_distances(&phase_space);
            let (radii, correlations) = analyzer.compute_correlation_curve(&distances);

            if let Ok(result) = analyzer.compute_correlation_dimension(&radii, &correlations) {
                results.push((m, result.correlation_dimension, result.r_squared));
            }
        }

        results
    }

    /// Interpret correlation dimension
    fn interpret_dimension(&self, dimension: f64, r_squared: f64, embedding_dim: usize) -> String {
        let attractor_type = if dimension < 0.5 {
            "point attractor (fixed point)"
        } else if dimension < 1.5 {
            "limit cycle (periodic orbit)"
        } else if dimension < 2.5 {
            "low-dimensional chaotic attractor"
        } else if dimension < 3.5 {
            "moderate-dimensional chaotic attractor"
        } else {
            "high-dimensional or noisy dynamics"
        };

        let is_fractal = (dimension - dimension.round()).abs() > 0.1;
        let fractal_note = if is_fractal {
            "Non-integer dimension indicates fractal (strange) attractor."
        } else {
            "Near-integer dimension."
        };

        let fit_quality = if r_squared > 0.95 {
            "excellent"
        } else if r_squared > 0.90 {
            "good"
        } else if r_squared > 0.80 {
            "moderate"
        } else {
            "poor"
        };

        format!(
            "Correlation dimension ν = {:.3} suggests {}. {} \
             Embedding dimension m = {}. Linear fit quality: {} (R² = {:.4}). \
             For reliable results, the correlation dimension should saturate as m increases.",
            dimension, attractor_type, fractal_note, embedding_dim, fit_quality, r_squared
        )
    }
}

/// Result of correlation dimension computation
struct DimensionResult {
    correlation_dimension: f64,
    r_squared: f64,
    num_radii: usize,
    scaling_region_size: usize,
    min_radius: f64,
    max_radius: f64,
}

impl Default for CorrelationDimensionAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl StochasticAnalyzer for CorrelationDimensionAnalyzer {
    fn name(&self) -> &str {
        "Correlation Dimension (Grassberger-Procaccia)"
    }

    fn analyze(&self, data: &TimeSeries) -> Result<AnalysisResult, StochasticError> {
        self.validate(data)?;

        let values = data.values();

        // Step 1: Reconstruct phase space
        let phase_space = self.reconstruct_phase_space(&values);
        let num_phase_vectors = phase_space.nrows();

        // Step 2: Compute pairwise distances
        let distances = self.compute_distances(&phase_space);
        let num_pairs = distances.len();

        // Step 3: Compute correlation curve
        let (radii, correlations) = self.compute_correlation_curve(&distances);

        // Step 4: Compute correlation dimension
        let result = self.compute_correlation_dimension(&radii, &correlations)?;

        // Step 5: Compute for multiple dimensions to check saturation
        let multi_dim_results = self.compute_for_multiple_dimensions(&values, self.config.embedding_dimension.min(7));

        // Build analysis result
        let interpretation = self.interpret_dimension(
            result.correlation_dimension,
            result.r_squared,
            self.config.embedding_dimension,
        );

        let mut analysis_result = AnalysisResult::new(self.name())
            .with_metric("correlation_dimension", result.correlation_dimension)
            .with_metric("r_squared", result.r_squared)
            .with_metric("embedding_dimension", self.config.embedding_dimension as f64)
            .with_metric("time_delay", self.config.time_delay as f64)
            .with_metric("num_phase_vectors", num_phase_vectors as f64)
            .with_metric("num_pairs", num_pairs as f64)
            .with_metric("num_radii_tested", result.num_radii as f64)
            .with_metric("scaling_region_points", result.scaling_region_size as f64)
            .with_metric("min_radius", result.min_radius)
            .with_metric("max_radius", result.max_radius)
            .with_metadata("algorithm", "Grassberger & Procaccia (1983)")
            .with_metadata("theiler_window", self.config.theiler_window.to_string())
            .with_interpretation(interpretation);

        // Add dimensions for multiple embedding dimensions
        for (m, dim, r_sq) in multi_dim_results {
            analysis_result = analysis_result
                .with_metric(&format!("dimension_m{}", m), dim)
                .with_metric(&format!("r_squared_m{}", m), r_sq);
        }

        Ok(analysis_result)
    }

    fn required_sample_size(&self) -> usize {
        // Need sufficient points for reliable statistics
        // Rule of thumb: at least 10^(D+1) points for dimension D
        let m = self.config.embedding_dimension;
        let tau = self.config.time_delay;
        let base_requirement = 10_usize.pow((m / 2 + 1).min(4) as u32);
        let embedding_requirement = (m - 1) * tau + base_requirement;
        embedding_requirement.max(500)
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

        // Check for constant series
        let first = values[0];
        if values.iter().all(|&v| (v - first).abs() < 1e-10) {
            return Err(StochasticError::validation(
                "Data is constant, cannot compute correlation dimension".to_string(),
            ));
        }

        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::Array1;

    #[test]
    fn test_correlation_config_defaults() {
        let config = CorrelationConfig::default();
        assert_eq!(config.embedding_dimension, 5);
        assert_eq!(config.time_delay, 1);
        assert_eq!(config.num_radii, 30);
        assert_eq!(config.theiler_window, 0);
    }

    #[test]
    fn test_correlation_config_builder() {
        let config = CorrelationConfig::new()
            .with_embedding_dimension(4)
            .with_time_delay(2)
            .with_theiler_window(5);

        assert_eq!(config.embedding_dimension, 4);
        assert_eq!(config.time_delay, 2);
        assert_eq!(config.theiler_window, 5);
    }

    #[test]
    fn test_analyzer_creation() {
        let analyzer = CorrelationDimensionAnalyzer::new();
        assert_eq!(analyzer.name(), "Correlation Dimension (Grassberger-Procaccia)");
        assert_eq!(analyzer.config.embedding_dimension, 5);
    }

    #[test]
    fn test_phase_space_reconstruction() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let analyzer = CorrelationDimensionAnalyzer::new()
            .with_embedding_dimension(3)
            .with_time_delay(2);

        let phase_space = analyzer.reconstruct_phase_space(&data);

        assert_eq!(phase_space.nrows(), 6); // 10 - (3-1)*2 = 6
        assert_eq!(phase_space.ncols(), 3);
        assert_eq!(phase_space[[0, 0]], 1.0);
        assert_eq!(phase_space[[0, 1]], 3.0);
        assert_eq!(phase_space[[0, 2]], 5.0);
    }

    #[test]
    fn test_euclidean_distance() {
        let analyzer = CorrelationDimensionAnalyzer::new();
        let v1 = Array1::from_vec(vec![0.0, 0.0, 0.0]);
        let v2 = Array1::from_vec(vec![3.0, 4.0, 0.0]);

        let dist = analyzer.euclidean_distance(v1.view(), v2.view());
        assert_relative_eq!(dist, 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_compute_distances() {
        let data: Vec<f64> = (0..20).map(|x| x as f64).collect();
        let analyzer = CorrelationDimensionAnalyzer::new()
            .with_embedding_dimension(3)
            .with_time_delay(1);

        let phase_space = analyzer.reconstruct_phase_space(&data);
        let distances = analyzer.compute_distances(&phase_space);

        assert!(!distances.is_empty());
        assert!(distances.iter().all(|d| *d >= 0.0));
        // Check sorted
        for i in 1..distances.len() {
            assert!(distances[i] >= distances[i - 1]);
        }
    }

    #[test]
    fn test_correlation_integral() {
        let distances = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let analyzer = CorrelationDimensionAnalyzer::new();

        let c_2_5 = analyzer.correlation_integral(&distances, 2.5);
        assert_relative_eq!(c_2_5, 0.4, epsilon = 1e-10); // 2 out of 5

        let c_5_5 = analyzer.correlation_integral(&distances, 5.5);
        assert_relative_eq!(c_5_5, 1.0, epsilon = 1e-10); // All 5
    }

    #[test]
    fn test_linear_fit() {
        let analyzer = CorrelationDimensionAnalyzer::new();

        // Perfect line: y = 2x
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];

        let (slope, r_squared) = analyzer.linear_fit(&x, &y);

        assert_relative_eq!(slope, 2.0, epsilon = 1e-6);
        assert_relative_eq!(r_squared, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_validation_insufficient_data() {
        let analyzer = CorrelationDimensionAnalyzer::new();
        let data = TimeSeries::from_values(vec![1.0, 2.0, 3.0]);

        let result = analyzer.validate(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_validation_constant_data() {
        let analyzer = CorrelationDimensionAnalyzer::new();
        let data = TimeSeries::from_values(vec![1.0; 1000]);

        let result = analyzer.validate(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_validation_nan_data() {
        let analyzer = CorrelationDimensionAnalyzer::new();
        let mut values = vec![1.0; 1000];
        values[500] = f64::NAN;
        let data = TimeSeries::from_values(values);

        let result = analyzer.validate(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_required_sample_size() {
        let analyzer = CorrelationDimensionAnalyzer::new()
            .with_embedding_dimension(5)
            .with_time_delay(3);

        let required = analyzer.required_sample_size();
        assert!(required >= 500);
    }

    #[test]
    fn test_logistic_map_chaotic() {
        // Logistic map with r=4 (chaotic)
        let mut data = Vec::with_capacity(2000);
        let mut x = 0.1;
        for _ in 0..2000 {
            data.push(x);
            x = 4.0 * x * (1.0 - x);
        }

        let ts = TimeSeries::from_values(data);
        let analyzer = CorrelationDimensionAnalyzer::new()
            .with_embedding_dimension(5)
            .with_time_delay(1)
            .with_theiler_window(2);

        let result = analyzer.analyze(&ts);
        assert!(result.is_ok());

        let result = result.unwrap();
        let dimension = result.metrics.get("correlation_dimension").unwrap();

        // Logistic map at r=4 has correlation dimension close to 1
        assert!(*dimension > 0.5 && *dimension < 2.0);
    }

    #[test]
    fn test_sine_wave_periodic() {
        // Sine wave should have dimension close to 1 (limit cycle)
        let data: Vec<f64> = (0..2000).map(|i| (i as f64 * 0.1).sin()).collect();

        let ts = TimeSeries::from_values(data);
        let analyzer = CorrelationDimensionAnalyzer::new()
            .with_embedding_dimension(4)
            .with_time_delay(5);

        let result = analyzer.analyze(&ts);
        assert!(result.is_ok());

        let result = result.unwrap();
        let dimension = result.metrics.get("correlation_dimension").unwrap();

        // Should be close to 1 for limit cycle
        assert!(*dimension > 0.5 && *dimension < 2.0);
    }

    #[test]
    fn test_analyze_returns_all_metrics() {
        let mut data = Vec::with_capacity(1500);
        let mut x = 0.1;
        for _ in 0..1500 {
            data.push(x);
            x = 3.8 * x * (1.0 - x);
        }

        let ts = TimeSeries::from_values(data);
        let analyzer = CorrelationDimensionAnalyzer::new();

        let result = analyzer.analyze(&ts).unwrap();

        assert!(result.metrics.contains_key("correlation_dimension"));
        assert!(result.metrics.contains_key("r_squared"));
        assert!(result.metrics.contains_key("embedding_dimension"));
        assert!(result.metrics.contains_key("time_delay"));
        assert!(result.metrics.contains_key("num_phase_vectors"));
        assert!(result.metrics.contains_key("num_pairs"));
        assert!(!result.interpretation.is_empty());
    }

    #[test]
    fn test_theiler_window_reduces_pairs() {
        let data: Vec<f64> = (0..100).map(|x| x as f64).collect();

        let analyzer1 = CorrelationDimensionAnalyzer::new()
            .with_embedding_dimension(3)
            .with_time_delay(1)
            .with_theiler_window(0);

        let analyzer2 = CorrelationDimensionAnalyzer::new()
            .with_embedding_dimension(3)
            .with_time_delay(1)
            .with_theiler_window(10);

        let phase_space = analyzer1.reconstruct_phase_space(&data);
        let distances1 = analyzer1.compute_distances(&phase_space);
        let distances2 = analyzer2.compute_distances(&phase_space);

        assert!(distances2.len() < distances1.len());
    }

    #[test]
    fn test_correlation_curve_properties() {
        let mut data = Vec::with_capacity(800);
        let mut x = 0.1;
        for _ in 0..800 {
            data.push(x);
            x = 3.9 * x * (1.0 - x);
        }

        let analyzer = CorrelationDimensionAnalyzer::new()
            .with_embedding_dimension(3)
            .with_time_delay(1);

        let phase_space = analyzer.reconstruct_phase_space(&data);
        let distances = analyzer.compute_distances(&phase_space);
        let (radii, correlations) = analyzer.compute_correlation_curve(&distances);

        assert!(!radii.is_empty());
        assert_eq!(radii.len(), correlations.len());

        // Correlations should be between 0 and 1
        assert!(correlations.iter().all(|&c| c > 0.0 && c < 1.0));

        // Correlations should increase with radius
        for i in 1..correlations.len() {
            assert!(correlations[i] >= correlations[i - 1]);
        }
    }

    #[test]
    fn test_interpret_dimension_types() {
        let analyzer = CorrelationDimensionAnalyzer::new();

        let interp_point = analyzer.interpret_dimension(0.3, 0.98, 5);
        assert!(interp_point.contains("point attractor"));

        let interp_cycle = analyzer.interpret_dimension(1.2, 0.97, 5);
        assert!(interp_cycle.contains("limit cycle"));

        let interp_chaotic = analyzer.interpret_dimension(2.1, 0.96, 5);
        assert!(interp_chaotic.contains("chaotic"));
    }

    #[test]
    fn test_multiple_dimensions_computed() {
        let mut data = Vec::with_capacity(1500);
        let mut x = 0.1;
        for _ in 0..1500 {
            data.push(x);
            x = 3.9 * x * (1.0 - x);
        }

        let ts = TimeSeries::from_values(data);
        let analyzer = CorrelationDimensionAnalyzer::new()
            .with_embedding_dimension(5)
            .with_time_delay(1);

        let result = analyzer.analyze(&ts).unwrap();

        // Should have dimensions for m=2,3,4,5
        assert!(result.metrics.contains_key("dimension_m2"));
        assert!(result.metrics.contains_key("dimension_m3"));
        assert!(result.metrics.contains_key("dimension_m4"));
        assert!(result.metrics.contains_key("dimension_m5"));
    }

    #[test]
    fn test_radius_range_determination() {
        let distances = vec![0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0];
        let analyzer = CorrelationDimensionAnalyzer::new();

        let (min_r, max_r) = analyzer.determine_radius_range(&distances);

        assert!(min_r > 0.0);
        assert!(max_r > min_r);
        assert!(max_r <= 10.0);
    }

    #[test]
    fn test_scaling_region_finder() {
        let analyzer = CorrelationDimensionAnalyzer::new();

        // Create perfect power law: log(C) = 2 * log(r)
        let log_radii: Vec<f64> = (0..30).map(|i| -3.0 + i as f64 * 0.2).collect();
        let log_correlations: Vec<f64> = log_radii.iter().map(|r| 2.0 * r).collect();

        let region = analyzer.find_scaling_region(&log_radii, &log_correlations);

        assert!(region.is_some());
        let (start, end) = region.unwrap();
        assert!(end > start);
        assert!(end - start >= analyzer.config.min_fit_points);
    }

    #[test]
    fn test_henon_map() {
        // Henon map: x_{n+1} = 1 - ax_n^2 + y_n, y_{n+1} = bx_n
        // Has correlation dimension ≈ 1.26
        let mut data = Vec::with_capacity(3000);
        let mut x = 0.1;
        let mut y = 0.3;
        let a = 1.4;
        let b = 0.3;

        for _ in 0..3000 {
            data.push(x);
            let x_new = 1.0 - a * x * x + y;
            let y_new = b * x;
            x = x_new;
            y = y_new;
        }

        let ts = TimeSeries::from_values(data);
        let analyzer = CorrelationDimensionAnalyzer::new()
            .with_embedding_dimension(4)
            .with_time_delay(1);

        let result = analyzer.analyze(&ts);
        assert!(result.is_ok());

        let result = result.unwrap();
        let dimension = result.metrics.get("correlation_dimension").unwrap();

        // Henon map has fractal dimension around 1.26
        assert!(*dimension > 1.0 && *dimension < 2.0);
    }
}
