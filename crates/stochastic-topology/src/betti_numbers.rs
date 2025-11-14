//! Betti Number Analysis
//!
//! Computes Betti numbers to count topological features (connected components,
//! loops, voids) at different scales in data.
//!
//! # Mathematical Foundation
//!
//! ## Betti Numbers
//!
//! The k-th Betti number βₖ counts the number of independent k-dimensional holes:
//!
//! ```text
//! β₀ = number of connected components
//! β₁ = number of 1-dimensional holes (loops/cycles)
//! β₂ = number of 2-dimensional voids (cavities)
//! ```
//!
//! Formally, βₖ = rank(Hₖ) where Hₖ is the k-th homology group.
//!
//! ## Euler Characteristic
//!
//! The Euler characteristic χ relates Betti numbers:
//!
//! ```text
//! χ = β₀ - β₁ + β₂ - β₃ + ...
//! ```
//!
//! For a 2D surface:
//! - Sphere: χ = 2 (β₀=1, β₁=0, β₂=1)
//! - Torus: χ = 0 (β₀=1, β₁=2, β₂=1)
//! - Double torus: χ = -2
//!
//! ## Homology Groups
//!
//! The homology group Hₖ(X) is computed from chain complex:
//!
//! ```text
//! ... → Cₖ₊₁ --∂ₖ₊₁--> Cₖ --∂ₖ--> Cₖ₋₁ → ...
//! ```
//!
//! where Cₖ is the group of k-chains and ∂ₖ is the boundary operator.
//!
//! The k-th homology is:
//!
//! ```text
//! Hₖ = ker(∂ₖ) / im(∂ₖ₊₁)
//! ```
//!
//! ## Scale-dependent Features
//!
//! Betti numbers depend on the scale ε at which we examine the data:
//! - Small ε: many components, few loops
//! - Large ε: few components, loops appear and disappear
//!
//! # References
//!
//! - Carlsson, G. (2009). "Topology and data." Bulletin of the AMS, 46(2), 255-308.
//! - Otter, N., Porter, M. A., Tillmann, U., Grindrod, P., & Harrington, H. A. (2017).
//!   "A roadmap for the computation of persistent homology." EPJ Data Science, 6(1), 17.
//! - Chazal, F., de Silva, V., Glisse, M., & Oudot, S. (2021). "The Structure and
//!   Stability of Persistence Modules." Springer.
//! - Wasserman, L. (2018). "Topological Data Analysis." Annual Review of Statistics
//!   and Its Application, 5, 501-532.
//! - Munch, E. (2017). "A user's guide to topological data analysis." Journal of
//!   Learning Analytics, 4(2), 47-61.
//! - Bubenik, P. (2015). "Statistical topological data analysis using persistence
//!   landscapes." Journal of Machine Learning Research, 16(1), 77-102.

use ndarray::Array2;
use petgraph::unionfind::UnionFind;
use std::collections::HashSet;
use stochastic_core::{AnalysisResult, StochasticAnalyzer, StochasticError, TimeSeries};

/// Betti numbers at a specific scale
#[derive(Debug, Clone, PartialEq)]
pub struct BettiNumbers {
    /// β₀: Number of connected components
    pub beta_0: usize,
    /// β₁: Number of loops/cycles
    pub beta_1: usize,
    /// β₂: Number of voids/cavities
    pub beta_2: usize,
    /// Scale (epsilon) at which these were computed
    pub scale: f64,
}

impl BettiNumbers {
    /// Create new Betti numbers
    pub fn new(beta_0: usize, beta_1: usize, beta_2: usize, scale: f64) -> Self {
        Self {
            beta_0,
            beta_1,
            beta_2,
            scale,
        }
    }

    /// Compute Euler characteristic χ = β₀ - β₁ + β₂
    pub fn euler_characteristic(&self) -> i32 {
        self.beta_0 as i32 - self.beta_1 as i32 + self.beta_2 as i32
    }

    /// Total number of topological features
    pub fn total_features(&self) -> usize {
        self.beta_0 + self.beta_1 + self.beta_2
    }

    /// Check if topologically trivial (single component, no holes)
    pub fn is_trivial(&self) -> bool {
        self.beta_0 == 1 && self.beta_1 == 0 && self.beta_2 == 0
    }

    /// Topological complexity measure
    pub fn complexity(&self) -> f64 {
        (self.beta_1 + 2 * self.beta_2) as f64
    }
}

/// Configuration for Betti number analysis
#[derive(Debug, Clone)]
pub struct BettiConfig {
    /// Number of scales to analyze (default: 20)
    pub num_scales: usize,
    /// Maximum distance for analysis (default: auto)
    pub max_distance: Option<f64>,
    /// Minimum number of points per component (default: 1)
    pub min_component_size: usize,
    /// Whether to normalize distances (default: true)
    pub normalize: bool,
}

impl Default for BettiConfig {
    fn default() -> Self {
        Self {
            num_scales: 20,
            max_distance: None,
            min_component_size: 1,
            normalize: true,
        }
    }
}

impl BettiConfig {
    /// Create new configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set number of scales
    pub fn with_num_scales(mut self, num: usize) -> Self {
        self.num_scales = num.max(5);
        self
    }

    /// Set maximum distance
    pub fn with_max_distance(mut self, dist: f64) -> Self {
        self.max_distance = Some(dist);
        self
    }

    /// Set minimum component size
    pub fn with_min_component_size(mut self, size: usize) -> Self {
        self.min_component_size = size;
        self
    }

    /// Set normalization flag
    pub fn with_normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }
}

/// Betti Number Analyzer
pub struct BettiAnalyzer {
    config: BettiConfig,
}

impl BettiAnalyzer {
    /// Create new analyzer with default configuration
    pub fn new() -> Self {
        Self {
            config: BettiConfig::default(),
        }
    }

    /// Create analyzer with custom configuration
    pub fn with_config(config: BettiConfig) -> Self {
        Self { config }
    }

    /// Compute Euclidean distance
    fn euclidean_distance(&self, p1: ndarray::ArrayView1<f64>, p2: ndarray::ArrayView1<f64>) -> f64 {
        p1.iter()
            .zip(p2.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Embed time series into 2D using time-delay embedding
    fn embed_timeseries(&self, values: &[f64]) -> Array2<f64> {
        let n = values.len();
        let tau = (n / 10).max(1);
        let m = n - tau;

        let mut points = Array2::zeros((m, 2));
        for i in 0..m {
            points[[i, 0]] = values[i];
            points[[i, 1]] = values[i + tau];
        }

        // Normalize if requested
        if self.config.normalize {
            self.normalize_points(&mut points);
        }

        points
    }

    /// Normalize points to [0, 1] range
    fn normalize_points(&self, points: &mut Array2<f64>) {
        let (nrows, ncols) = (points.nrows(), points.ncols());

        for col in 0..ncols {
            let mut min_val = f64::MAX;
            let mut max_val = f64::MIN;

            for row in 0..nrows {
                let val = points[[row, col]];
                min_val = min_val.min(val);
                max_val = max_val.max(val);
            }

            let range = max_val - min_val;
            if range > 1e-10 {
                for row in 0..nrows {
                    points[[row, col]] = (points[[row, col]] - min_val) / range;
                }
            }
        }
    }

    /// Compute pairwise distances
    fn compute_distances(&self, points: &Array2<f64>) -> Array2<f64> {
        let n = points.nrows();
        let mut distances = Array2::zeros((n, n));

        for i in 0..n {
            for j in i + 1..n {
                let dist = self.euclidean_distance(points.row(i), points.row(j));
                distances[[i, j]] = dist;
                distances[[j, i]] = dist;
            }
        }

        distances
    }

    /// Compute β₀ (connected components) at given scale
    fn compute_beta_0(&self, distances: &Array2<f64>, epsilon: f64) -> usize {
        let n = distances.nrows();
        let mut uf = UnionFind::new(n);

        // Connect points within distance epsilon
        for i in 0..n {
            for j in i + 1..n {
                if distances[[i, j]] <= epsilon {
                    uf.union(i, j);
                }
            }
        }

        // Count distinct components
        let mut roots = HashSet::new();
        for i in 0..n {
            roots.insert(uf.find(i));
        }

        roots.len()
    }

    /// Compute β₁ (loops) at given scale using cycle rank
    fn compute_beta_1(&self, distances: &Array2<f64>, epsilon: f64) -> usize {
        let n = distances.nrows();
        let mut uf = UnionFind::new(n);
        let mut edge_count = 0;
        let vertex_count = n;

        // Build graph at this scale
        for i in 0..n {
            for j in i + 1..n {
                if distances[[i, j]] <= epsilon {
                    let root_i = uf.find(i);
                    let root_j = uf.find(j);

                    edge_count += 1;

                    if root_i != root_j {
                        uf.union(i, j);
                    }
                }
            }
        }

        // Count connected components
        let mut components = HashSet::new();
        for i in 0..n {
            components.insert(uf.find(i));
        }
        let num_components = components.len();

        // Cycle rank = edges - vertices + components
        // This gives β₁ for the graph
        let beta_1 = if edge_count >= vertex_count {
            edge_count - vertex_count + num_components
        } else {
            0
        };

        beta_1
    }

    /// Compute β₂ (voids) - simplified estimation
    fn compute_beta_2(&self, _distances: &Array2<f64>, _epsilon: f64) -> usize {
        // For 2D embeddings, β₂ is typically 0
        // A more sophisticated implementation would require 3D+ embeddings
        0
    }

    /// Compute Betti numbers at multiple scales
    fn compute_multi_scale_betti(&self, distances: &Array2<f64>) -> Vec<BettiNumbers> {
        // Determine maximum distance
        let max_dist = if let Some(max_d) = self.config.max_distance {
            max_d
        } else {
            let mut max_d: f64 = 0.0;
            for i in 0..distances.nrows() {
                for j in i + 1..distances.ncols() {
                    max_d = max_d.max(distances[[i, j]]);
                }
            }
            max_d * 0.75
        };

        // Generate scales
        let scales: Vec<f64> = (0..self.config.num_scales)
            .map(|i| {
                let t = (i + 1) as f64 / self.config.num_scales as f64;
                t * max_dist
            })
            .collect();

        // Compute Betti numbers at each scale
        scales
            .iter()
            .map(|&epsilon| {
                let beta_0 = self.compute_beta_0(distances, epsilon);
                let beta_1 = self.compute_beta_1(distances, epsilon);
                let beta_2 = self.compute_beta_2(distances, epsilon);

                BettiNumbers::new(beta_0, beta_1, beta_2, epsilon)
            })
            .collect()
    }

    /// Find scale with maximum topological complexity
    fn find_optimal_scale(&self, betti_sequence: &[BettiNumbers]) -> (usize, f64) {
        let mut max_complexity = 0.0;
        let mut max_idx = 0;

        for (idx, betti) in betti_sequence.iter().enumerate() {
            let complexity = betti.complexity();
            if complexity > max_complexity {
                max_complexity = complexity;
                max_idx = idx;
            }
        }

        (max_idx, max_complexity)
    }

    /// Interpret Betti number sequence
    fn interpret_betti_sequence(&self, sequence: &[BettiNumbers]) -> String {
        if sequence.is_empty() {
            return "No Betti numbers computed.".to_string();
        }

        let (optimal_idx, max_complexity) = self.find_optimal_scale(sequence);
        let optimal_betti = &sequence[optimal_idx];

        let avg_beta_0 = sequence.iter().map(|b| b.beta_0).sum::<usize>() as f64 / sequence.len() as f64;
        let avg_beta_1 = sequence.iter().map(|b| b.beta_1).sum::<usize>() as f64 / sequence.len() as f64;

        let euler = optimal_betti.euler_characteristic();

        format!(
            "Multi-scale Betti analysis: optimal scale ε={:.4} with β₀={}, β₁={}, β₂={} (χ={}). \
             Average across scales: β₀={:.2}, β₁={:.2}. Maximum topological complexity: {:.2}. \
             β₀ counts connected components, β₁ counts loops/cycles, β₂ counts voids.",
            optimal_betti.scale,
            optimal_betti.beta_0,
            optimal_betti.beta_1,
            optimal_betti.beta_2,
            euler,
            avg_beta_0,
            avg_beta_1,
            max_complexity
        )
    }
}

impl Default for BettiAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl StochasticAnalyzer for BettiAnalyzer {
    fn name(&self) -> &str {
        "Betti Numbers"
    }

    fn analyze(&self, data: &TimeSeries) -> Result<AnalysisResult, StochasticError> {
        self.validate(data)?;

        let values = data.values();

        // Embed into 2D point cloud
        let points = self.embed_timeseries(&values);

        // Compute distances
        let distances = self.compute_distances(&points);

        // Compute Betti numbers at multiple scales
        let betti_sequence = self.compute_multi_scale_betti(&distances);

        // Find optimal scale
        let (optimal_idx, max_complexity) = self.find_optimal_scale(&betti_sequence);
        let optimal_betti = &betti_sequence[optimal_idx];

        // Compute statistics
        let avg_beta_0 = betti_sequence.iter().map(|b| b.beta_0).sum::<usize>() as f64
            / betti_sequence.len() as f64;
        let avg_beta_1 = betti_sequence.iter().map(|b| b.beta_1).sum::<usize>() as f64
            / betti_sequence.len() as f64;

        let max_beta_0 = betti_sequence.iter().map(|b| b.beta_0).max().unwrap_or(0);
        let max_beta_1 = betti_sequence.iter().map(|b| b.beta_1).max().unwrap_or(0);

        let euler_char = optimal_betti.euler_characteristic();

        let interpretation = self.interpret_betti_sequence(&betti_sequence);

        Ok(AnalysisResult::new(self.name())
            .with_metric("optimal_beta_0", optimal_betti.beta_0 as f64)
            .with_metric("optimal_beta_1", optimal_betti.beta_1 as f64)
            .with_metric("optimal_beta_2", optimal_betti.beta_2 as f64)
            .with_metric("optimal_scale", optimal_betti.scale)
            .with_metric("euler_characteristic", euler_char as f64)
            .with_metric("avg_beta_0", avg_beta_0)
            .with_metric("avg_beta_1", avg_beta_1)
            .with_metric("max_beta_0", max_beta_0 as f64)
            .with_metric("max_beta_1", max_beta_1 as f64)
            .with_metric("topological_complexity", max_complexity)
            .with_metric("num_scales", betti_sequence.len() as f64)
            .with_metric("num_points", points.nrows() as f64)
            .with_metadata("normalize", self.config.normalize.to_string())
            .with_interpretation(interpretation))
    }

    fn required_sample_size(&self) -> usize {
        30
    }

    fn validate(&self, data: &TimeSeries) -> Result<bool, StochasticError> {
        if data.len() < self.required_sample_size() {
            return Err(StochasticError::insufficient_data(
                self.required_sample_size(),
                data.len(),
            ));
        }

        let values = data.values();

        if values.iter().any(|v| !v.is_finite()) {
            return Err(StochasticError::validation(
                "Data contains NaN or infinite values".to_string(),
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
    fn test_betti_numbers_creation() {
        let betti = BettiNumbers::new(1, 2, 0, 0.5);
        assert_eq!(betti.beta_0, 1);
        assert_eq!(betti.beta_1, 2);
        assert_eq!(betti.beta_2, 0);
        assert_eq!(betti.scale, 0.5);
    }

    #[test]
    fn test_euler_characteristic() {
        let betti = BettiNumbers::new(1, 2, 1, 1.0);
        assert_eq!(betti.euler_characteristic(), 0); // 1 - 2 + 1 = 0 (torus)
    }

    #[test]
    fn test_total_features() {
        let betti = BettiNumbers::new(2, 3, 1, 1.0);
        assert_eq!(betti.total_features(), 6);
    }

    #[test]
    fn test_is_trivial() {
        let trivial = BettiNumbers::new(1, 0, 0, 1.0);
        let nontrivial = BettiNumbers::new(1, 1, 0, 1.0);

        assert!(trivial.is_trivial());
        assert!(!nontrivial.is_trivial());
    }

    #[test]
    fn test_complexity() {
        let betti = BettiNumbers::new(1, 2, 1, 1.0);
        assert_relative_eq!(betti.complexity(), 4.0, epsilon = 1e-10); // 2 + 2*1
    }

    #[test]
    fn test_config_defaults() {
        let config = BettiConfig::default();
        assert_eq!(config.num_scales, 20);
        assert!(config.normalize);
        assert_eq!(config.min_component_size, 1);
    }

    #[test]
    fn test_config_builder() {
        let config = BettiConfig::new()
            .with_num_scales(15)
            .with_max_distance(2.0)
            .with_normalize(false);

        assert_eq!(config.num_scales, 15);
        assert_eq!(config.max_distance, Some(2.0));
        assert!(!config.normalize);
    }

    #[test]
    fn test_analyzer_creation() {
        let analyzer = BettiAnalyzer::new();
        assert_eq!(analyzer.name(), "Betti Numbers");
    }

    #[test]
    fn test_euclidean_distance() {
        let analyzer = BettiAnalyzer::new();
        let p1 = Array1::from_vec(vec![0.0, 0.0]);
        let p2 = Array1::from_vec(vec![3.0, 4.0]);

        let dist = analyzer.euclidean_distance(p1.view(), p2.view());
        assert_relative_eq!(dist, 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_validation_insufficient_data() {
        let analyzer = BettiAnalyzer::new();
        let data = TimeSeries::from_values(vec![1.0, 2.0, 3.0]);

        assert!(analyzer.validate(&data).is_err());
    }

    #[test]
    fn test_validation_nan_data() {
        let analyzer = BettiAnalyzer::new();
        let mut values = vec![1.0; 50];
        values[25] = f64::NAN;
        let data = TimeSeries::from_values(values);

        assert!(analyzer.validate(&data).is_err());
    }

    #[test]
    fn test_compute_beta_0_single_component() {
        let analyzer = BettiAnalyzer::new();
        let mut distances = Array2::zeros((4, 4));

        // All points very close - should be one component
        for i in 0..4 {
            for j in 0..4 {
                distances[[i, j]] = 0.1;
            }
        }
        for i in 0..4 {
            distances[[i, i]] = 0.0;
        }

        let beta_0 = analyzer.compute_beta_0(&distances, 0.5);
        assert_eq!(beta_0, 1);
    }

    #[test]
    fn test_compute_beta_0_multiple_components() {
        let analyzer = BettiAnalyzer::new();
        let mut distances = Array2::zeros((4, 4));

        // Two separate pairs
        distances[[0, 1]] = 0.1;
        distances[[1, 0]] = 0.1;
        distances[[2, 3]] = 0.1;
        distances[[3, 2]] = 0.1;

        // Large distances between pairs
        for i in 0..2 {
            for j in 2..4 {
                distances[[i, j]] = 10.0;
                distances[[j, i]] = 10.0;
            }
        }

        let beta_0 = analyzer.compute_beta_0(&distances, 0.5);
        assert_eq!(beta_0, 2);
    }

    #[test]
    fn test_circle_data_topology() {
        // Generate points on a circle
        let mut data = Vec::new();
        for i in 0..100 {
            let angle = 2.0 * std::f64::consts::PI * (i as f64) / 100.0;
            data.push(angle.cos() + angle.sin());
        }

        let ts = TimeSeries::from_values(data);
        let analyzer = BettiAnalyzer::new();

        let result = analyzer.analyze(&ts);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert!(result.metrics.contains_key("optimal_beta_0"));
        assert!(result.metrics.contains_key("optimal_beta_1"));
    }

    #[test]
    fn test_linear_data_topology() {
        // Linear data should have minimal topological features
        let data: Vec<f64> = (0..100).map(|x| x as f64).collect();
        let ts = TimeSeries::from_values(data);
        let analyzer = BettiAnalyzer::new();

        let result = analyzer.analyze(&ts).unwrap();

        // Linear data at large scale should be connected
        let beta_0 = result.metrics.get("optimal_beta_0").unwrap();
        assert!(*beta_0 <= 2.0);
    }

    #[test]
    fn test_analyze_returns_all_metrics() {
        let data: Vec<f64> = (0..80).map(|x| (x as f64 * 0.1).sin()).collect();
        let ts = TimeSeries::from_values(data);
        let analyzer = BettiAnalyzer::new();

        let result = analyzer.analyze(&ts).unwrap();

        assert!(result.metrics.contains_key("optimal_beta_0"));
        assert!(result.metrics.contains_key("optimal_beta_1"));
        assert!(result.metrics.contains_key("optimal_beta_2"));
        assert!(result.metrics.contains_key("optimal_scale"));
        assert!(result.metrics.contains_key("euler_characteristic"));
        assert!(result.metrics.contains_key("avg_beta_0"));
        assert!(result.metrics.contains_key("avg_beta_1"));
        assert!(result.metrics.contains_key("topological_complexity"));
        assert!(!result.interpretation.is_empty());
    }

    #[test]
    fn test_normalize_points() {
        let analyzer = BettiAnalyzer::new();
        let mut points = Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 5.0, 10.0, 10.0, 20.0]).unwrap();

        analyzer.normalize_points(&mut points);

        // Check that values are in [0, 1]
        for i in 0..points.nrows() {
            for j in 0..points.ncols() {
                let val = points[[i, j]];
                assert!(val >= 0.0 && val <= 1.0);
            }
        }

        // Check min and max
        assert_relative_eq!(points[[0, 0]], 0.0, epsilon = 1e-10);
        assert_relative_eq!(points[[2, 0]], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_multi_scale_betti() {
        let analyzer = BettiAnalyzer::new();
        let data: Vec<f64> = (0..50).map(|x| x as f64).collect();
        let ts = TimeSeries::from_values(data);

        let points = analyzer.embed_timeseries(&ts.values());
        let distances = analyzer.compute_distances(&points);
        let betti_seq = analyzer.compute_multi_scale_betti(&distances);

        assert_eq!(betti_seq.len(), analyzer.config.num_scales);

        // Beta_0 should decrease as scale increases
        let first_beta_0 = betti_seq[0].beta_0;
        let last_beta_0 = betti_seq[betti_seq.len() - 1].beta_0;
        assert!(first_beta_0 >= last_beta_0);
    }

    #[test]
    fn test_embed_timeseries() {
        let analyzer = BettiAnalyzer::new();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        let embedded = analyzer.embed_timeseries(&data);

        assert!(embedded.nrows() > 0);
        assert_eq!(embedded.ncols(), 2);
    }

    #[test]
    fn test_periodic_data() {
        // Periodic sine wave
        let data: Vec<f64> = (0..150).map(|x| (x as f64 * 0.1).sin()).collect();
        let ts = TimeSeries::from_values(data);
        let analyzer = BettiAnalyzer::new();

        let result = analyzer.analyze(&ts);
        assert!(result.is_ok());

        let result = result.unwrap();
        let complexity = result.metrics.get("topological_complexity").unwrap();
        assert!(*complexity >= 0.0);
    }
}
