//! Lyapunov Exponent Analysis
//!
//! Computes the largest Lyapunov exponent using the Rosenstein et al. (1993) algorithm.
//! The Lyapunov exponent quantifies the rate of separation of infinitesimally close
//! trajectories in phase space, characterizing the sensitivity to initial conditions
//! that is a hallmark of chaotic systems.
//!
//! # Mathematical Foundation
//!
//! ## Lyapunov Exponent Definition
//!
//! The largest Lyapunov exponent λ₁ is defined as:
//!
//! ```text
//! λ₁ = lim_{t→∞} lim_{δ(0)→0} (1/t) ln[δ(t)/δ(0)]
//! ```
//!
//! where δ(t) is the distance between two initially close trajectories at time t.
//!
//! ## Rosenstein Algorithm
//!
//! 1. **Phase Space Reconstruction**: Embed the time series using delay coordinates
//!    X(i) = [x(i), x(i+τ), ..., x(i+(m-1)τ)]
//!
//! 2. **Nearest Neighbor Search**: For each point X(i), find its nearest neighbor X(j)
//!    such that |i - j| > mean_period to avoid temporal correlations
//!
//! 3. **Distance Evolution**: Track divergence d(i,t) = |X(i+t) - X(j+t)|
//!
//! 4. **Average Logarithmic Divergence**:
//!    y(t) = (1/Δt) <ln[d(i,t)]>
//!
//! 5. **Linear Fit**: λ₁ is the slope of y(t) vs t in the linear scaling region
//!
//! ## Interpretation
//!
//! - λ₁ > 0: Chaotic dynamics (exponential divergence)
//! - λ₁ ≈ 0: Periodic or quasi-periodic behavior
//! - λ₁ < 0: Stable fixed point (convergence)
//!
//! # References
//!
//! - Rosenstein, M.T., Collins, J.J., & De Luca, C.J. (1993). "A practical method for
//!   calculating largest Lyapunov exponents from small data sets." Physica D, 65(1-2), 117-134.
//! - Wolf, A., Swift, J.B., Swinney, H.L., & Vastano, J.A. (1985). "Determining Lyapunov
//!   exponents from a time series." Physica D, 16(3), 285-317.
//! - Kantz, H. (1994). "A robust method to estimate the maximal Lyapunov exponent of a
//!   time series." Physics Letters A, 185(1), 77-87.
//! - Eckmann, J.P., & Ruelle, D. (1985). "Ergodic theory of chaos and strange attractors."
//!   Reviews of Modern Physics, 57(3), 617-656.
//! -Sprott, J.C. (2003). "Chaos and Time-Series Analysis." Oxford University Press.

use ndarray::{Array1, Array2};
use stochastic_core::{AnalysisResult, StochasticAnalyzer, StochasticError, TimeSeries};

/// Configuration for Lyapunov exponent analysis
#[derive(Debug, Clone)]
pub struct LyapunovConfig {
    /// Embedding dimension (default: 3)
    pub embedding_dimension: usize,
    /// Time delay for embedding (default: 1)
    pub time_delay: usize,
    /// Minimum temporal separation for nearest neighbors (default: data.len()/10)
    pub min_temporal_separation: Option<usize>,
    /// Maximum evolution time to track (default: data.len()/10)
    pub max_evolution_time: Option<usize>,
    /// Number of points to use for linear fit (default: 20)
    pub fit_points: usize,
}

impl Default for LyapunovConfig {
    fn default() -> Self {
        Self {
            embedding_dimension: 3,
            time_delay: 1,
            min_temporal_separation: None,
            max_evolution_time: None,
            fit_points: 20,
        }
    }
}

impl LyapunovConfig {
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

    /// Set minimum temporal separation
    pub fn with_min_temporal_separation(mut self, separation: usize) -> Self {
        self.min_temporal_separation = Some(separation);
        self
    }

    /// Set maximum evolution time
    pub fn with_max_evolution_time(mut self, time: usize) -> Self {
        self.max_evolution_time = Some(time);
        self
    }

    /// Set number of points for linear fit
    pub fn with_fit_points(mut self, points: usize) -> Self {
        self.fit_points = points.max(5);
        self
    }
}

/// Lyapunov Exponent Analyzer using Rosenstein algorithm
pub struct LyapunovAnalyzer {
    config: LyapunovConfig,
}

impl LyapunovAnalyzer {
    /// Create new analyzer with default configuration
    pub fn new() -> Self {
        Self {
            config: LyapunovConfig::default(),
        }
    }

    /// Create analyzer with custom configuration
    pub fn with_config(config: LyapunovConfig) -> Self {
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

    /// Perform phase space reconstruction using time delay embedding
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

    /// Find nearest neighbor for each point in phase space
    fn find_nearest_neighbors(
        &self,
        phase_space: &Array2<f64>,
        min_separation: usize,
    ) -> Vec<(usize, f64)> {
        let num_points = phase_space.nrows();
        let mut neighbors = Vec::with_capacity(num_points);

        for i in 0..num_points {
            let mut min_dist = f64::MAX;
            let mut nearest_idx = 0;

            for j in 0..num_points {
                // Skip if temporal separation is too small
                if i.abs_diff(j) < min_separation {
                    continue;
                }

                let dist = self.euclidean_distance(phase_space.row(i), phase_space.row(j));

                if dist < min_dist && dist > 0.0 {
                    min_dist = dist;
                    nearest_idx = j;
                }
            }

            neighbors.push((nearest_idx, min_dist));
        }

        neighbors
    }

    /// Compute Euclidean distance between two vectors
    fn euclidean_distance(&self, v1: ndarray::ArrayView1<f64>, v2: ndarray::ArrayView1<f64>) -> f64 {
        v1.iter()
            .zip(v2.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Track divergence of nearest neighbors over time
    fn track_divergence(
        &self,
        phase_space: &Array2<f64>,
        neighbors: &[(usize, f64)],
        max_time: usize,
    ) -> Array1<f64> {
        let num_points = phase_space.nrows();
        let mut divergence = Array1::<f64>::zeros(max_time);
        let mut counts = Array1::<f64>::zeros(max_time);

        for (i, &(j, _)) in neighbors.iter().enumerate() {
            for t in 0..max_time {
                let i_t = i + t;
                let j_t = j + t;

                if i_t >= num_points || j_t >= num_points {
                    break;
                }

                let dist = self.euclidean_distance(phase_space.row(i_t), phase_space.row(j_t));

                if dist > 0.0 {
                    divergence[t] += dist.ln();
                    counts[t] += 1.0;
                }
            }
        }

        // Average the logarithmic divergence
        for t in 0..max_time {
            if counts[t] > 0.0 {
                divergence[t] /= counts[t];
            }
        }

        divergence
    }

    /// Fit linear regression to compute Lyapunov exponent
    fn compute_lyapunov_exponent(&self, divergence: &Array1<f64>, sampling_rate: f64) -> (f64, f64) {
        let n = self.config.fit_points.min(divergence.len());
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xx = 0.0;
        let mut sum_xy = 0.0;

        for i in 0..n {
            let x = i as f64 / sampling_rate;
            let y = divergence[i];

            if !y.is_finite() {
                continue;
            }

            sum_x += x;
            sum_y += y;
            sum_xx += x * x;
            sum_xy += x * y;
        }

        let n_valid = n as f64;
        let slope = (n_valid * sum_xy - sum_x * sum_y) / (n_valid * sum_xx - sum_x * sum_x);
        let intercept = (sum_y - slope * sum_x) / n_valid;

        // Compute R-squared
        let mean_y = sum_y / n_valid;
        let mut ss_tot = 0.0;
        let mut ss_res = 0.0;

        for i in 0..n {
            let x = i as f64 / sampling_rate;
            let y = divergence[i];

            if !y.is_finite() {
                continue;
            }

            let y_pred = slope * x + intercept;
            ss_tot += (y - mean_y).powi(2);
            ss_res += (y - y_pred).powi(2);
        }

        let r_squared = if ss_tot > 0.0 {
            1.0 - (ss_res / ss_tot)
        } else {
            0.0
        };

        (slope, r_squared)
    }

    /// Compute embedding parameters automatically if not specified
    fn determine_parameters(&self, data_len: usize) -> (usize, usize) {
        let min_separation = self
            .config
            .min_temporal_separation
            .unwrap_or_else(|| (data_len / 10).max(1));

        let max_time = self
            .config
            .max_evolution_time
            .unwrap_or_else(|| (data_len / 10).max(10));

        (min_separation, max_time)
    }

    /// Validate configuration parameters
    fn validate_config(&self, data_len: usize) -> Result<(), StochasticError> {
        let m = self.config.embedding_dimension;
        let tau = self.config.time_delay;

        let required_length = (m - 1) * tau + 1;
        if data_len < required_length {
            return Err(StochasticError::validation(format!(
                "Data length {} insufficient for embedding (dimension={}, delay={}). Need at least {} points.",
                data_len, m, tau, required_length
            )));
        }

        if self.config.fit_points < 5 {
            return Err(StochasticError::validation(
                "fit_points must be at least 5 for reliable linear regression".to_string(),
            ));
        }

        Ok(())
    }

    /// Interpret the Lyapunov exponent value
    fn interpret_lyapunov(&self, lambda: f64, r_squared: f64) -> String {
        let dynamics = if lambda > 0.01 {
            "chaotic (positive Lyapunov exponent)"
        } else if lambda > -0.01 {
            "near-zero (periodic or quasi-periodic)"
        } else {
            "stable (negative Lyapunov exponent)"
        };

        let reliability = if r_squared > 0.95 {
            "excellent"
        } else if r_squared > 0.85 {
            "good"
        } else if r_squared > 0.70 {
            "moderate"
        } else {
            "poor"
        };

        format!(
            "System exhibits {} dynamics. λ₁ = {:.6} bits/iteration. \
             Linear fit quality: {} (R² = {:.4}). \
             Positive values indicate sensitivity to initial conditions and chaotic behavior.",
            dynamics, lambda, reliability, r_squared
        )
    }
}

impl Default for LyapunovAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl StochasticAnalyzer for LyapunovAnalyzer {
    fn name(&self) -> &str {
        "Lyapunov Exponent (Rosenstein)"
    }

    fn analyze(&self, data: &TimeSeries) -> Result<AnalysisResult, StochasticError> {
        self.validate(data)?;

        let values = data.values();
        let data_len = values.len();

        // Validate configuration
        self.validate_config(data_len)?;

        // Determine parameters
        let (min_separation, max_time) = self.determine_parameters(data_len);

        // Get sampling rate or use default of 1.0
        let sampling_rate = data.sampling_rate.unwrap_or(1.0);

        // Step 1: Reconstruct phase space
        let phase_space = self.reconstruct_phase_space(&values);

        // Step 2: Find nearest neighbors
        let neighbors = self.find_nearest_neighbors(&phase_space, min_separation);

        // Step 3: Track divergence
        let divergence = self.track_divergence(&phase_space, &neighbors, max_time);

        // Step 4: Compute Lyapunov exponent
        let (lambda, r_squared) = self.compute_lyapunov_exponent(&divergence, sampling_rate);

        // Compute additional metrics
        let num_phase_points = phase_space.nrows();
        let mean_neighbor_dist: f64 = neighbors.iter().map(|(_, d)| d).sum::<f64>() / neighbors.len() as f64;

        // Build result
        let interpretation = self.interpret_lyapunov(lambda, r_squared);

        Ok(AnalysisResult::new(self.name())
            .with_metric("lyapunov_exponent", lambda)
            .with_metric("r_squared", r_squared)
            .with_metric("embedding_dimension", self.config.embedding_dimension as f64)
            .with_metric("time_delay", self.config.time_delay as f64)
            .with_metric("num_phase_points", num_phase_points as f64)
            .with_metric("mean_neighbor_distance", mean_neighbor_dist)
            .with_metadata("algorithm", "Rosenstein et al. (1993)")
            .with_metadata("min_temporal_separation", min_separation.to_string())
            .with_metadata("max_evolution_time", max_time.to_string())
            .with_interpretation(interpretation))
    }

    fn required_sample_size(&self) -> usize {
        // Need sufficient points for embedding and neighbor search
        let m = self.config.embedding_dimension;
        let tau = self.config.time_delay;
        let min_embedded = (m - 1) * tau + 50;
        min_embedded.max(100)
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
                "Data is constant, cannot compute Lyapunov exponent".to_string(),
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
    fn test_lyapunov_config_defaults() {
        let config = LyapunovConfig::default();
        assert_eq!(config.embedding_dimension, 3);
        assert_eq!(config.time_delay, 1);
        assert_eq!(config.fit_points, 20);
    }

    #[test]
    fn test_lyapunov_config_builder() {
        let config = LyapunovConfig::new()
            .with_embedding_dimension(5)
            .with_time_delay(2)
            .with_fit_points(15);

        assert_eq!(config.embedding_dimension, 5);
        assert_eq!(config.time_delay, 2);
        assert_eq!(config.fit_points, 15);
    }

    #[test]
    fn test_analyzer_creation() {
        let analyzer = LyapunovAnalyzer::new();
        assert_eq!(analyzer.name(), "Lyapunov Exponent (Rosenstein)");
        assert_eq!(analyzer.config.embedding_dimension, 3);
    }

    #[test]
    fn test_analyzer_with_config() {
        let config = LyapunovConfig::new().with_embedding_dimension(4);
        let analyzer = LyapunovAnalyzer::with_config(config);
        assert_eq!(analyzer.config.embedding_dimension, 4);
    }

    #[test]
    fn test_phase_space_reconstruction() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let analyzer = LyapunovAnalyzer::new()
            .with_embedding_dimension(3)
            .with_time_delay(1);

        let phase_space = analyzer.reconstruct_phase_space(&data);

        assert_eq!(phase_space.nrows(), 6); // 8 - (3-1)*1 = 6
        assert_eq!(phase_space.ncols(), 3);
        assert_eq!(phase_space[[0, 0]], 1.0);
        assert_eq!(phase_space[[0, 1]], 2.0);
        assert_eq!(phase_space[[0, 2]], 3.0);
    }

    #[test]
    fn test_euclidean_distance() {
        let analyzer = LyapunovAnalyzer::new();
        let v1 = Array1::from_vec(vec![0.0, 0.0, 0.0]);
        let v2 = Array1::from_vec(vec![3.0, 4.0, 0.0]);

        let dist = analyzer.euclidean_distance(v1.view(), v2.view());
        assert_relative_eq!(dist, 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_validation_insufficient_data() {
        let analyzer = LyapunovAnalyzer::new();
        let data = TimeSeries::from_values(vec![1.0, 2.0, 3.0]);

        let result = analyzer.validate(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_validation_constant_data() {
        let analyzer = LyapunovAnalyzer::new();
        let data = TimeSeries::from_values(vec![1.0; 200]);

        let result = analyzer.validate(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_validation_nan_data() {
        let analyzer = LyapunovAnalyzer::new();
        let mut values = vec![1.0; 200];
        values[100] = f64::NAN;
        let data = TimeSeries::from_values(values);

        let result = analyzer.validate(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_required_sample_size() {
        let analyzer = LyapunovAnalyzer::new()
            .with_embedding_dimension(5)
            .with_time_delay(3);

        let required = analyzer.required_sample_size();
        assert!(required >= 100); // Should be at least 100
    }

    #[test]
    fn test_logistic_map_chaotic() {
        // Logistic map: x_{n+1} = r * x_n * (1 - x_n) with r=4 (chaotic)
        let mut data = Vec::with_capacity(1000);
        let mut x = 0.1;
        let r = 4.0;

        for _ in 0..1000 {
            data.push(x);
            x = r * x * (1.0 - x);
        }

        let ts = TimeSeries::from_values(data).with_sampling_rate(1.0);
        let analyzer = LyapunovAnalyzer::new()
            .with_embedding_dimension(3)
            .with_time_delay(1);

        let result = analyzer.analyze(&ts);
        assert!(result.is_ok());

        let result = result.unwrap();
        let lambda = result.metrics.get("lyapunov_exponent").unwrap();

        // Logistic map at r=4 has λ ≈ ln(2) ≈ 0.693
        assert!(*lambda > 0.0, "Chaotic system should have positive λ");
    }

    #[test]
    fn test_sine_wave_periodic() {
        // Sine wave is periodic, should have λ ≈ 0
        let mut data = Vec::with_capacity(1000);
        for i in 0..1000 {
            data.push((i as f64 * 0.1).sin());
        }

        let ts = TimeSeries::from_values(data).with_sampling_rate(1.0);
        let analyzer = LyapunovAnalyzer::new()
            .with_embedding_dimension(3)
            .with_time_delay(5);

        let result = analyzer.analyze(&ts);
        assert!(result.is_ok());

        let result = result.unwrap();
        let lambda = result.metrics.get("lyapunov_exponent").unwrap();

        // Periodic systems should have λ near 0 or negative
        assert!(
            *lambda < 0.1,
            "Periodic system should have near-zero or negative λ, got {}",
            lambda
        );
    }

    #[test]
    fn test_analyze_returns_all_metrics() {
        let mut data = Vec::with_capacity(500);
        let mut x = 0.1;
        for _ in 0..500 {
            data.push(x);
            x = 3.9 * x * (1.0 - x);
        }

        let ts = TimeSeries::from_values(data).with_sampling_rate(1.0);
        let analyzer = LyapunovAnalyzer::new();

        let result = analyzer.analyze(&ts).unwrap();

        assert!(result.metrics.contains_key("lyapunov_exponent"));
        assert!(result.metrics.contains_key("r_squared"));
        assert!(result.metrics.contains_key("embedding_dimension"));
        assert!(result.metrics.contains_key("time_delay"));
        assert!(result.metrics.contains_key("num_phase_points"));
        assert!(result.metrics.contains_key("mean_neighbor_distance"));
        assert!(!result.interpretation.is_empty());
    }

    #[test]
    fn test_nearest_neighbors_temporal_separation() {
        let data: Vec<f64> = (0..100).map(|x| x as f64).collect();
        let analyzer = LyapunovAnalyzer::new();
        let phase_space = analyzer.reconstruct_phase_space(&data);

        let min_sep = 10;
        let neighbors = analyzer.find_nearest_neighbors(&phase_space, min_sep);

        // Verify temporal separation constraint
        for (i, &(j, _)) in neighbors.iter().enumerate() {
            let separation = i.abs_diff(j);
            assert!(
                separation >= min_sep,
                "Neighbor separation {} < min_sep {}",
                separation,
                min_sep
            );
        }
    }

    #[test]
    fn test_divergence_tracking() {
        let data: Vec<f64> = (0..200).map(|x| (x as f64 * 0.1).sin()).collect();
        let analyzer = LyapunovAnalyzer::new();
        let phase_space = analyzer.reconstruct_phase_space(&data);
        let neighbors = analyzer.find_nearest_neighbors(&phase_space, 10);

        let divergence = analyzer.track_divergence(&phase_space, &neighbors, 50);

        assert_eq!(divergence.len(), 50);
        assert!(divergence.iter().all(|&d| d.is_finite()));
    }

    #[test]
    fn test_linear_fit_computation() {
        let mut data = Vec::with_capacity(300);
        let mut x = 0.1;
        for _ in 0..300 {
            data.push(x);
            x = 3.8 * x * (1.0 - x);
        }

        let ts = TimeSeries::from_values(data);
        let analyzer = LyapunovAnalyzer::new();

        let values = ts.values();
        let phase_space = analyzer.reconstruct_phase_space(&values);
        let neighbors = analyzer.find_nearest_neighbors(&phase_space, 20);
        let divergence = analyzer.track_divergence(&phase_space, &neighbors, 40);

        let (lambda, r_squared) = analyzer.compute_lyapunov_exponent(&divergence, 1.0);

        assert!(lambda.is_finite());
        assert!(r_squared >= 0.0 && r_squared <= 1.0);
    }

    #[test]
    fn test_interpret_lyapunov_chaotic() {
        let analyzer = LyapunovAnalyzer::new();
        let interpretation = analyzer.interpret_lyapunov(0.5, 0.95);

        assert!(interpretation.contains("chaotic"));
        assert!(interpretation.contains("excellent") || interpretation.contains("good"));
    }

    #[test]
    fn test_interpret_lyapunov_periodic() {
        let analyzer = LyapunovAnalyzer::new();
        let interpretation = analyzer.interpret_lyapunov(0.005, 0.90);

        assert!(interpretation.contains("periodic") || interpretation.contains("quasi-periodic"));
    }

    #[test]
    fn test_config_validation() {
        let analyzer = LyapunovAnalyzer::new()
            .with_embedding_dimension(5)
            .with_time_delay(10);

        // Too short for embedding
        let result = analyzer.validate_config(50);
        assert!(result.is_ok());

        // Way too short
        let result = analyzer.validate_config(10);
        assert!(result.is_err());
    }
}
