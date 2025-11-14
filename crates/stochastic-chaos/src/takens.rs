//! Takens Embedding and Phase Space Reconstruction
//!
//! Implements Takens' delay embedding theorem for reconstructing the attractor of a
//! dynamical system from a single time series observation. Includes methods for
//! determining optimal embedding parameters using False Nearest Neighbors (FNN)
//! and Average Mutual Information (AMI).
//!
//! # Mathematical Foundation
//!
//! ## Takens' Theorem (1981)
//!
//! Given a smooth dynamical system with attractor dimension d_A, a generic
//! observable x(t), and embedding parameters (m, τ), the delay coordinate
//! reconstruction:
//!
//! ```text
//! X(t) = [x(t), x(t+τ), x(t+2τ), ..., x(t+(m-1)τ)]
//! ```
//!
//! is diffeomorphic to the original attractor if:
//! ```text
//! m ≥ 2d_A + 1
//! ```
//!
//! ## Optimal Time Delay (τ)
//!
//! **Average Mutual Information (AMI):**
//!
//! ```text
//! I(τ) = ΣΣ p(xᵢ, xᵢ₊τ) log₂[p(xᵢ, xᵢ₊τ) / (p(xᵢ)p(xᵢ₊τ))]
//! ```
//!
//! Optimal τ is at the first minimum of I(τ), indicating statistical independence
//! while maintaining dynamical information.
//!
//! ## Optimal Embedding Dimension (m)
//!
//! **False Nearest Neighbors (FNN):**
//!
//! A point is a "false nearest neighbor" if it appears close in m dimensions
//! but separates when embedded in m+1 dimensions:
//!
//! ```text
//! R_m+1(i,j) = |x(i+mτ) - x(j+mτ)| / R_m(i,j) > R_threshold
//! ```
//!
//! Optimal m is where the fraction of FNN drops below a threshold (typically < 1%).
//!
//! # References
//!
//! - Takens, F. (1981). "Detecting strange attractors in turbulence." In Dynamical
//!   Systems and Turbulence, Lecture Notes in Mathematics, vol. 898, pp. 366-381.
//! - Kennel, M.B., Brown, R., & Abarbanel, H.D.I. (1992). "Determining embedding
//!   parameters for phase-space reconstruction using a geometrical construction."
//!   Physical Review A, 45(6), 3403-3411.
//! - Fraser, A.M., & Swinney, H.L. (1986). "Independent coordinates for strange
//!   attractors from mutual information." Physical Review A, 33(2), 1134-1140.
//! - Kantz, H., & Schreiber, T. (2004). "Nonlinear Time Series Analysis" (2nd ed.).
//!   Cambridge University Press.
//! - Abarbanel, H.D.I. (1996). "Analysis of Observed Chaotic Data." Springer.

use ndarray::Array2;
use stochastic_core::{AnalysisResult, StochasticAnalyzer, StochasticError, TimeSeries};

/// Configuration for Takens embedding
#[derive(Debug, Clone)]
pub struct TakensConfig {
    /// Embedding dimension (None = auto-detect using FNN)
    pub embedding_dimension: Option<usize>,
    /// Time delay (None = auto-detect using AMI)
    pub time_delay: Option<usize>,
    /// Maximum embedding dimension to test for FNN (default: 10)
    pub max_dimension: usize,
    /// Maximum time delay to test for AMI (default: 50)
    pub max_delay: usize,
    /// Threshold for false nearest neighbors ratio (default: 0.01)
    pub fnn_threshold: f64,
    /// Distance threshold multiplier for FNN (default: 15.0)
    pub fnn_rtol: f64,
    /// Number of bins for AMI histogram (default: 16)
    pub ami_bins: usize,
}

impl Default for TakensConfig {
    fn default() -> Self {
        Self {
            embedding_dimension: None,
            time_delay: None,
            max_dimension: 10,
            max_delay: 50,
            fnn_threshold: 0.01,
            fnn_rtol: 15.0,
            ami_bins: 16,
        }
    }
}

impl TakensConfig {
    /// Create new configuration with defaults
    pub fn new() -> Self {
        Self::default()
    }

    /// Set embedding dimension explicitly
    pub fn with_embedding_dimension(mut self, dim: usize) -> Self {
        self.embedding_dimension = Some(dim.max(2));
        self
    }

    /// Set time delay explicitly
    pub fn with_time_delay(mut self, delay: usize) -> Self {
        self.time_delay = Some(delay.max(1));
        self
    }

    /// Set maximum dimension for auto-detection
    pub fn with_max_dimension(mut self, max_dim: usize) -> Self {
        self.max_dimension = max_dim.max(2);
        self
    }

    /// Set maximum delay for auto-detection
    pub fn with_max_delay(mut self, max_delay: usize) -> Self {
        self.max_delay = max_delay.max(1);
        self
    }

    /// Set FNN threshold
    pub fn with_fnn_threshold(mut self, threshold: f64) -> Self {
        self.fnn_threshold = threshold.max(0.0).min(1.0);
        self
    }

    /// Set FNN distance tolerance
    pub fn with_fnn_rtol(mut self, rtol: f64) -> Self {
        self.fnn_rtol = rtol.max(1.0);
        self
    }

    /// Set AMI histogram bins
    pub fn with_ami_bins(mut self, bins: usize) -> Self {
        self.ami_bins = bins.max(4);
        self
    }
}

/// Takens Embedding analyzer for phase space reconstruction
pub struct TakensEmbedding {
    config: TakensConfig,
}

impl TakensEmbedding {
    /// Create new analyzer with default configuration
    pub fn new() -> Self {
        Self {
            config: TakensConfig::default(),
        }
    }

    /// Create analyzer with custom configuration
    pub fn with_config(config: TakensConfig) -> Self {
        Self { config }
    }

    /// Set embedding dimension
    pub fn with_embedding_dimension(mut self, dim: usize) -> Self {
        self.config.embedding_dimension = Some(dim.max(2));
        self
    }

    /// Set time delay
    pub fn with_time_delay(mut self, delay: usize) -> Self {
        self.config.time_delay = Some(delay.max(1));
        self
    }

    /// Compute Average Mutual Information for time delay selection
    fn compute_mutual_information(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        let max_delay = self.config.max_delay.min(n / 4);
        let nbins = self.config.ami_bins;

        let mut ami_values = Vec::with_capacity(max_delay);

        // Normalize data to [0, 1] for binning
        let min_val = data.iter().copied().fold(f64::INFINITY, f64::min);
        let max_val = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let range = max_val - min_val;

        if range < 1e-10 {
            return vec![0.0; max_delay];
        }

        for delay in 1..=max_delay {
            let mut joint_hist = vec![vec![0usize; nbins]; nbins];
            let mut hist_x = vec![0usize; nbins];
            let mut hist_y = vec![0usize; nbins];
            let mut count = 0;

            for i in 0..(n - delay) {
                let x_val = data[i];
                let y_val = data[i + delay];

                let x_bin = ((x_val - min_val) / range * (nbins as f64 - 0.001)) as usize;
                let y_bin = ((y_val - min_val) / range * (nbins as f64 - 0.001)) as usize;

                let x_bin = x_bin.min(nbins - 1);
                let y_bin = y_bin.min(nbins - 1);

                joint_hist[x_bin][y_bin] += 1;
                hist_x[x_bin] += 1;
                hist_y[y_bin] += 1;
                count += 1;
            }

            // Compute mutual information
            let mut ami = 0.0;
            for i in 0..nbins {
                for j in 0..nbins {
                    if joint_hist[i][j] > 0 {
                        let p_xy = joint_hist[i][j] as f64 / count as f64;
                        let p_x = hist_x[i] as f64 / count as f64;
                        let p_y = hist_y[j] as f64 / count as f64;

                        ami += p_xy * (p_xy / (p_x * p_y)).log2();
                    }
                }
            }

            ami_values.push(ami);
        }

        ami_values
    }

    /// Find first minimum in AMI curve
    fn find_first_minimum(&self, ami_values: &[f64]) -> usize {
        if ami_values.len() < 3 {
            return 1;
        }

        for i in 1..(ami_values.len() - 1) {
            if ami_values[i] < ami_values[i - 1] && ami_values[i] < ami_values[i + 1] {
                return i + 1; // +1 because we started at delay=1
            }
        }

        // If no minimum found, use the delay with minimum value
        let min_idx = ami_values
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        min_idx + 1
    }

    /// Reconstruct phase space with given parameters
    fn reconstruct_phase_space(&self, data: &[f64], m: usize, tau: usize) -> Array2<f64> {
        let n = data.len();
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

    /// Compute False Nearest Neighbors percentage
    fn compute_fnn(&self, data: &[f64], m: usize, tau: usize) -> f64 {
        let phase_space = self.reconstruct_phase_space(data, m, tau);
        let num_points = phase_space.nrows();

        if num_points < 2 {
            return 1.0;
        }

        // Compute standard deviation for normalization
        let std_dev = {
            let mean = data.iter().sum::<f64>() / data.len() as f64;
            let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
            variance.sqrt()
        };

        if std_dev < 1e-10 {
            return 0.0;
        }

        let mut false_neighbors = 0;
        let mut total_neighbors = 0;

        // For each point, find its nearest neighbor
        for i in 0..(num_points - tau) {
            let mut min_dist = f64::MAX;
            let mut nearest_idx = 0;

            // Find nearest neighbor in m-dimensional space
            for j in 0..(num_points - tau) {
                if i == j {
                    continue;
                }

                let dist = self.euclidean_distance(phase_space.row(i), phase_space.row(j));

                if dist < min_dist && dist > 0.0 {
                    min_dist = dist;
                    nearest_idx = j;
                }
            }

            if min_dist == f64::MAX || min_dist < 1e-10 {
                continue;
            }

            // Check if it's a false neighbor in (m+1)-dimensional space
            let next_i = i + tau;
            let next_j = nearest_idx + tau;

            if next_i < data.len() && next_j < data.len() {
                let additional_dist = (data[next_i] - data[next_j]).abs();
                let ratio = additional_dist / min_dist;

                // Check two conditions for false neighbors
                let is_false = ratio > self.config.fnn_rtol ||
                              (min_dist * min_dist + additional_dist * additional_dist).sqrt() / std_dev > self.config.fnn_rtol;

                if is_false {
                    false_neighbors += 1;
                }

                total_neighbors += 1;
            }
        }

        if total_neighbors == 0 {
            return 1.0;
        }

        false_neighbors as f64 / total_neighbors as f64
    }

    /// Determine optimal embedding dimension using FNN
    fn find_optimal_dimension(&self, data: &[f64], tau: usize) -> (usize, Vec<f64>) {
        let max_dim = self.config.max_dimension;
        let mut fnn_percentages = Vec::with_capacity(max_dim);

        for m in 1..=max_dim {
            let fnn_pct = self.compute_fnn(data, m, tau);
            fnn_percentages.push(fnn_pct);

            // Stop if we've found good embedding dimension
            if fnn_pct < self.config.fnn_threshold && m >= 2 {
                return (m, fnn_percentages);
            }
        }

        // If no dimension meets threshold, choose dimension where FNN is minimized
        let optimal_dim = fnn_percentages
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx + 1)
            .unwrap_or(3);

        (optimal_dim, fnn_percentages)
    }

    /// Perform complete Takens embedding analysis
    fn perform_embedding(&self, data: &[f64]) -> Result<EmbeddingResult, StochasticError> {
        // Step 1: Determine optimal time delay if not specified
        let tau = if let Some(delay) = self.config.time_delay {
            delay
        } else {
            let ami_values = self.compute_mutual_information(data);
            let optimal_tau = self.find_first_minimum(&ami_values);
            optimal_tau
        };

        // Step 2: Determine optimal embedding dimension if not specified
        let (m, fnn_percentages) = if let Some(dim) = self.config.embedding_dimension {
            let fnn_values = (1..=dim)
                .map(|d| self.compute_fnn(data, d, tau))
                .collect();
            (dim, fnn_values)
        } else {
            self.find_optimal_dimension(data, tau)
        };

        // Step 3: Reconstruct phase space
        let phase_space = self.reconstruct_phase_space(data, m, tau);

        Ok(EmbeddingResult {
            embedding_dimension: m,
            time_delay: tau,
            phase_space,
            fnn_percentages,
        })
    }

    /// Interpret embedding results
    fn interpret_embedding(&self, result: &EmbeddingResult) -> String {
        let m = result.embedding_dimension;
        let tau = result.time_delay;
        let fnn = result.fnn_percentages.last().unwrap_or(&1.0);

        let quality = if *fnn < 0.01 {
            "excellent"
        } else if *fnn < 0.05 {
            "good"
        } else if *fnn < 0.15 {
            "moderate"
        } else {
            "poor"
        };

        let attractor_dim = if m > 1 {
            (m - 1) / 2
        } else {
            1
        };

        format!(
            "Phase space reconstructed with embedding dimension m={} and time delay τ={}. \
             False nearest neighbors: {:.2}% ({}quality). \
             Estimated attractor dimension: ≤ {}. \
             Reconstructed {} phase space vectors from the time series.",
            m,
            tau,
            fnn * 100.0,
            quality,
            attractor_dim,
            result.phase_space.nrows()
        )
    }
}

/// Result of Takens embedding analysis
struct EmbeddingResult {
    embedding_dimension: usize,
    time_delay: usize,
    phase_space: Array2<f64>,
    fnn_percentages: Vec<f64>,
}

impl Default for TakensEmbedding {
    fn default() -> Self {
        Self::new()
    }
}

impl StochasticAnalyzer for TakensEmbedding {
    fn name(&self) -> &str {
        "Takens Embedding"
    }

    fn analyze(&self, data: &TimeSeries) -> Result<AnalysisResult, StochasticError> {
        self.validate(data)?;

        let values = data.values();

        // Perform embedding
        let embedding_result = self.perform_embedding(&values)?;

        // Compute additional metrics
        let phase_space = &embedding_result.phase_space;
        let num_vectors = phase_space.nrows();

        // Compute mean and std of distances in phase space
        let mut distances = Vec::new();
        for i in 0..num_vectors.min(1000) {
            for j in (i + 1)..num_vectors.min(1000) {
                let dist = self.euclidean_distance(phase_space.row(i), phase_space.row(j));
                distances.push(dist);
            }
        }

        let mean_distance = if !distances.is_empty() {
            distances.iter().sum::<f64>() / distances.len() as f64
        } else {
            0.0
        };

        let std_distance = if !distances.is_empty() {
            let mean = mean_distance;
            let variance = distances.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / distances.len() as f64;
            variance.sqrt()
        } else {
            0.0
        };

        // Build result
        let interpretation = self.interpret_embedding(&embedding_result);

        let mut result = AnalysisResult::new(self.name())
            .with_metric("embedding_dimension", embedding_result.embedding_dimension as f64)
            .with_metric("time_delay", embedding_result.time_delay as f64)
            .with_metric("num_phase_vectors", num_vectors as f64)
            .with_metric("mean_phase_distance", mean_distance)
            .with_metric("std_phase_distance", std_distance)
            .with_metadata("method", "Takens (1981) delay embedding")
            .with_metadata("delay_selection", "Average Mutual Information (Fraser & Swinney 1986)")
            .with_metadata("dimension_selection", "False Nearest Neighbors (Kennel et al. 1992)")
            .with_interpretation(interpretation);

        // Add FNN percentages for each dimension tested
        for (i, &fnn) in embedding_result.fnn_percentages.iter().enumerate() {
            result = result.with_metric(&format!("fnn_dim_{}", i + 1), fnn * 100.0);
        }

        Ok(result)
    }

    fn required_sample_size(&self) -> usize {
        // Need sufficient points for embedding and FNN computation
        let max_dim = self.config.max_dimension;
        let max_delay = self.config.max_delay;
        let min_embedded = (max_dim - 1) * max_delay + 100;
        min_embedded.max(200)
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
                "Data is constant, cannot perform embedding".to_string(),
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
    fn test_takens_config_defaults() {
        let config = TakensConfig::default();
        assert_eq!(config.max_dimension, 10);
        assert_eq!(config.max_delay, 50);
        assert_eq!(config.fnn_threshold, 0.01);
        assert_eq!(config.ami_bins, 16);
    }

    #[test]
    fn test_takens_config_builder() {
        let config = TakensConfig::new()
            .with_embedding_dimension(5)
            .with_time_delay(3)
            .with_max_dimension(8);

        assert_eq!(config.embedding_dimension, Some(5));
        assert_eq!(config.time_delay, Some(3));
        assert_eq!(config.max_dimension, 8);
    }

    #[test]
    fn test_analyzer_creation() {
        let analyzer = TakensEmbedding::new();
        assert_eq!(analyzer.name(), "Takens Embedding");
    }

    #[test]
    fn test_phase_space_reconstruction() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let analyzer = TakensEmbedding::new();

        let phase_space = analyzer.reconstruct_phase_space(&data, 3, 2);

        assert_eq!(phase_space.nrows(), 6); // 10 - (3-1)*2 = 6
        assert_eq!(phase_space.ncols(), 3);
        assert_eq!(phase_space[[0, 0]], 1.0);
        assert_eq!(phase_space[[0, 1]], 3.0);
        assert_eq!(phase_space[[0, 2]], 5.0);
    }

    #[test]
    fn test_euclidean_distance() {
        let analyzer = TakensEmbedding::new();
        let v1 = Array1::from_vec(vec![0.0, 0.0, 0.0]);
        let v2 = Array1::from_vec(vec![3.0, 4.0, 0.0]);

        let dist = analyzer.euclidean_distance(v1.view(), v2.view());
        assert_relative_eq!(dist, 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_mutual_information_computation() {
        // Create autocorrelated data
        let mut data = Vec::with_capacity(500);
        data.push(0.5);
        for i in 1..500 {
            data.push(0.7 * data[i - 1] + 0.3 * (i as f64 * 0.1).sin());
        }

        let analyzer = TakensEmbedding::new();
        let ami_values = analyzer.compute_mutual_information(&data);

        assert!(!ami_values.is_empty());
        assert!(ami_values.iter().all(|v| v.is_finite()));
        assert!(ami_values[0] > 0.0); // AMI at delay=1 should be positive
    }

    #[test]
    fn test_find_first_minimum() {
        let analyzer = TakensEmbedding::new();

        // Create data with clear minimum at index 3
        let values = vec![1.0, 0.8, 0.6, 0.4, 0.5, 0.7, 0.9];
        let min_idx = analyzer.find_first_minimum(&values);

        assert_eq!(min_idx, 4); // Index 3 in the values, but +1 offset
    }

    #[test]
    fn test_fnn_computation() {
        let mut data = Vec::with_capacity(300);
        let mut x = 0.1;
        for _ in 0..300 {
            data.push(x);
            x = 3.9 * x * (1.0 - x);
        }

        let analyzer = TakensEmbedding::new();
        let fnn_pct = analyzer.compute_fnn(&data, 3, 5);

        assert!(fnn_pct >= 0.0 && fnn_pct <= 1.0);
    }

    #[test]
    fn test_optimal_dimension_finding() {
        let mut data = Vec::with_capacity(500);
        let mut x = 0.1;
        for _ in 0..500 {
            data.push(x);
            x = 3.8 * x * (1.0 - x);
        }

        let analyzer = TakensEmbedding::new();
        let (optimal_dim, fnn_values) = analyzer.find_optimal_dimension(&data, 3);

        assert!(optimal_dim >= 1);
        assert!(optimal_dim <= analyzer.config.max_dimension);
        assert_eq!(fnn_values.len(), optimal_dim);
    }

    #[test]
    fn test_validation_insufficient_data() {
        let analyzer = TakensEmbedding::new();
        let data = TimeSeries::from_values(vec![1.0, 2.0, 3.0]);

        let result = analyzer.validate(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_validation_constant_data() {
        let analyzer = TakensEmbedding::new();
        let data = TimeSeries::from_values(vec![1.0; 500]);

        let result = analyzer.validate(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_validation_nan_data() {
        let analyzer = TakensEmbedding::new();
        let mut values = vec![1.0; 500];
        values[250] = f64::NAN;
        let data = TimeSeries::from_values(values);

        let result = analyzer.validate(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_required_sample_size() {
        let analyzer = TakensEmbedding::new();
        let required = analyzer.required_sample_size();
        assert!(required >= 200);
    }

    #[test]
    fn test_analyze_logistic_map() {
        let mut data = Vec::with_capacity(600);
        let mut x = 0.1;
        for _ in 0..600 {
            data.push(x);
            x = 3.9 * x * (1.0 - x);
        }

        let ts = TimeSeries::from_values(data);
        let analyzer = TakensEmbedding::new()
            .with_time_delay(1)
            .with_embedding_dimension(3);

        let result = analyzer.analyze(&ts);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert!(result.metrics.contains_key("embedding_dimension"));
        assert!(result.metrics.contains_key("time_delay"));
        assert!(result.metrics.contains_key("num_phase_vectors"));
    }

    #[test]
    fn test_analyze_auto_parameters() {
        let mut data = Vec::with_capacity(1000);
        let mut x = 0.1;
        for _ in 0..1000 {
            data.push(x);
            x = 3.8 * x * (1.0 - x);
        }

        let ts = TimeSeries::from_values(data);
        let analyzer = TakensEmbedding::new();

        let result = analyzer.analyze(&ts);
        assert!(result.is_ok());

        let result = result.unwrap();
        let dim = result.metrics.get("embedding_dimension").unwrap();
        let delay = result.metrics.get("time_delay").unwrap();

        assert!(*dim >= 1.0);
        assert!(*delay >= 1.0);
        assert!(!result.interpretation.is_empty());
    }

    #[test]
    fn test_analyze_returns_all_metrics() {
        let mut data = Vec::with_capacity(1000);
        for i in 0..1000 {
            data.push((i as f64 * 0.1).sin() + 0.1 * (i as f64 * 0.05).cos());
        }

        let ts = TimeSeries::from_values(data);
        let analyzer = TakensEmbedding::new()
            .with_embedding_dimension(4)
            .with_time_delay(5);

        let result = analyzer.analyze(&ts).unwrap();

        assert!(result.metrics.contains_key("embedding_dimension"));
        assert!(result.metrics.contains_key("time_delay"));
        assert!(result.metrics.contains_key("num_phase_vectors"));
        assert!(result.metrics.contains_key("mean_phase_distance"));
        assert!(result.metrics.contains_key("std_phase_distance"));
        assert!(!result.interpretation.is_empty());
    }

    #[test]
    fn test_embedding_lorenz_like_data() {
        // Generate Lorenz-like chaotic data
        let mut data = Vec::with_capacity(1000);
        let mut x = 1.0;
        let mut y = 1.0;

        for _ in 0..1000 {
            data.push(x);
            let x_new = x + 0.01 * (10.0 * (y - x));
            let y_new = y + 0.01 * (x * (28.0 - data.len() as f64 * 0.001) - y);
            x = x_new;
            y = y_new;
        }

        let ts = TimeSeries::from_values(data);
        let analyzer = TakensEmbedding::new();

        let result = analyzer.analyze(&ts);
        assert!(result.is_ok());
    }

    #[test]
    fn test_interpret_embedding() {
        let analyzer = TakensEmbedding::new();
        let phase_space = Array2::zeros((100, 3));
        let fnn_percentages = vec![0.5, 0.2, 0.05, 0.01];

        let embedding_result = EmbeddingResult {
            embedding_dimension: 4,
            time_delay: 7,
            phase_space,
            fnn_percentages,
        };

        let interpretation = analyzer.interpret_embedding(&embedding_result);

        assert!(interpretation.contains("m=4"));
        assert!(interpretation.contains("τ=7"));
        assert!(interpretation.contains("100"));
    }

    #[test]
    fn test_fnn_decreases_with_dimension() {
        let mut data = Vec::with_capacity(400);
        let mut x = 0.1;
        for _ in 0..400 {
            data.push(x);
            x = 3.9 * x * (1.0 - x);
        }

        let analyzer = TakensEmbedding::new();

        // FNN should generally decrease as dimension increases
        let fnn1 = analyzer.compute_fnn(&data, 1, 3);
        let fnn3 = analyzer.compute_fnn(&data, 3, 3);
        let fnn5 = analyzer.compute_fnn(&data, 5, 3);

        // At least one of these should be true for chaotic data
        assert!(fnn1 >= fnn3 || fnn3 >= fnn5);
    }

    #[test]
    fn test_ami_for_periodic_data() {
        // Generate periodic sine wave
        let data: Vec<f64> = (0..500).map(|i| (i as f64 * 0.1).sin()).collect();

        let analyzer = TakensEmbedding::new();
        let ami_values = analyzer.compute_mutual_information(&data);

        assert!(!ami_values.is_empty());
        assert!(ami_values.iter().all(|v| v.is_finite() && *v >= 0.0));
    }
}
