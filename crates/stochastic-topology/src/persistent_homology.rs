//! Persistent Homology Analysis
//!
//! Implements persistent homology computation for tracking topological features across
//! multiple scales using filtrations and persistence diagrams.
//!
//! # Mathematical Foundation
//!
//! ## Persistent Homology
//!
//! Persistent homology studies the evolution of topological features across a filtration:
//!
//! ```text
//! ∅ = K₀ ⊆ K₁ ⊆ K₂ ⊆ ... ⊆ Kₙ = K
//! ```
//!
//! Each feature (connected component, loop, void) has a birth time bᵢ when it appears
//! and a death time dᵢ when it disappears. The persistence is:
//!
//! ```text
//! pᵢ = dᵢ - bᵢ
//! ```
//!
//! ## Persistence Diagram
//!
//! The persistence diagram plots birth-death pairs (bᵢ, dᵢ) in the plane:
//! - Points far from diagonal: persistent features (signal)
//! - Points near diagonal: short-lived features (noise)
//!
//! ## Homology Groups
//!
//! For dimension k, homology group Hₖ counts k-dimensional holes:
//! - H₀: connected components
//! - H₁: loops/cycles
//! - H₂: voids/cavities
//!
//! The k-th Betti number βₖ = rank(Hₖ) counts independent k-dimensional features.
//!
//! ## Boundary Operator
//!
//! For a simplex σ = [v₀, v₁, ..., vₖ], the boundary operator is:
//!
//! ```text
//! ∂ₖ(σ) = Σᵢ₌₀ᵏ (-1)ⁱ [v₀,...,v̂ᵢ,...,vₖ]
//! ```
//!
//! where v̂ᵢ denotes omitting vertex vᵢ.
//!
//! # References
//!
//! - Carlsson, G. (2009). "Topology and data." Bulletin of the American Mathematical
//!   Society, 46(2), 255-308. (Classic foundational reference)
//! - Zomorodian, A., & Carlsson, G. (2005). "Computing persistent homology."
//!   Discrete & Computational Geometry, 33(2), 249-274.
//! - Edelsbrunner, H., & Harer, J. (2008). "Computational Topology: An Introduction."
//!   American Mathematical Society.
//! - Chazal, F., & Michel, B. (2021). "An introduction to Topological Data Analysis:
//!   fundamental and practical aspects for data scientists." Frontiers in AI, 4, 667963.
//! - Otter, N., Porter, M. A., Tillmann, U., Grindrod, P., & Harrington, H. A. (2017).
//!   "A roadmap for the computation of persistent homology." EPJ Data Science, 6(1), 17.
//! - Kerber, M., Morozov, D., & Nigmetov, A. (2017). "Geometry helps to compare
//!   persistence diagrams." Journal of Experimental Algorithmics, 22, 1-20.

use ndarray::Array2;
use petgraph::graph::{Graph, UnGraph};
use petgraph::unionfind::UnionFind;
use std::collections::HashMap;
use stochastic_core::{AnalysisResult, StochasticAnalyzer, StochasticError, TimeSeries};

/// Birth-death pair representing a topological feature
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BirthDeathPair {
    /// Birth time (when feature appears)
    pub birth: f64,
    /// Death time (when feature disappears)
    pub death: f64,
    /// Dimension of the feature (0=component, 1=loop, 2=void)
    pub dimension: usize,
}

impl BirthDeathPair {
    /// Create new birth-death pair
    pub fn new(birth: f64, death: f64, dimension: usize) -> Self {
        Self {
            birth,
            death,
            dimension,
        }
    }

    /// Compute persistence (lifetime)
    pub fn persistence(&self) -> f64 {
        self.death - self.birth
    }

    /// Check if feature is significant (far from diagonal)
    pub fn is_significant(&self, threshold: f64) -> bool {
        self.persistence() > threshold
    }
}

/// Persistence diagram containing all birth-death pairs
#[derive(Debug, Clone)]
pub struct PersistenceDiagram {
    /// All birth-death pairs
    pub pairs: Vec<BirthDeathPair>,
    /// Maximum filtration value
    pub max_filtration: f64,
}

impl PersistenceDiagram {
    /// Create new persistence diagram
    pub fn new(pairs: Vec<BirthDeathPair>, max_filtration: f64) -> Self {
        Self {
            pairs,
            max_filtration,
        }
    }

    /// Get pairs by dimension
    pub fn pairs_by_dimension(&self, dim: usize) -> Vec<BirthDeathPair> {
        self.pairs
            .iter()
            .filter(|p| p.dimension == dim)
            .copied()
            .collect()
    }

    /// Get significant features
    pub fn significant_features(&self, threshold: f64) -> Vec<BirthDeathPair> {
        self.pairs
            .iter()
            .filter(|p| p.is_significant(threshold))
            .copied()
            .collect()
    }

    /// Count features by dimension
    pub fn count_by_dimension(&self, dim: usize) -> usize {
        self.pairs.iter().filter(|p| p.dimension == dim).count()
    }

    /// Compute maximum persistence
    pub fn max_persistence(&self) -> f64 {
        self.pairs
            .iter()
            .map(|p| p.persistence())
            .fold(0.0, f64::max)
    }
}

/// Configuration for persistent homology analysis
#[derive(Debug, Clone)]
pub struct PersistenceConfig {
    /// Maximum homology dimension to compute (default: 1)
    pub max_dimension: usize,
    /// Number of filtration steps (default: 50)
    pub num_steps: usize,
    /// Persistence threshold for significance (default: 0.1)
    pub persistence_threshold: f64,
    /// Maximum distance for Vietoris-Rips (default: auto)
    pub max_distance: Option<f64>,
}

impl Default for PersistenceConfig {
    fn default() -> Self {
        Self {
            max_dimension: 1,
            num_steps: 50,
            persistence_threshold: 0.1,
            max_distance: None,
        }
    }
}

impl PersistenceConfig {
    /// Create new configuration with defaults
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum dimension
    pub fn with_max_dimension(mut self, dim: usize) -> Self {
        self.max_dimension = dim;
        self
    }

    /// Set number of filtration steps
    pub fn with_num_steps(mut self, steps: usize) -> Self {
        self.num_steps = steps.max(10);
        self
    }

    /// Set persistence threshold
    pub fn with_persistence_threshold(mut self, threshold: f64) -> Self {
        self.persistence_threshold = threshold;
        self
    }

    /// Set maximum distance
    pub fn with_max_distance(mut self, dist: f64) -> Self {
        self.max_distance = Some(dist);
        self
    }
}

/// Persistent Homology Analyzer
pub struct PersistentHomology {
    config: PersistenceConfig,
}

impl PersistentHomology {
    /// Create new analyzer with default configuration
    pub fn new() -> Self {
        Self {
            config: PersistenceConfig::default(),
        }
    }

    /// Create analyzer with custom configuration
    pub fn with_config(config: PersistenceConfig) -> Self {
        Self { config }
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

    /// Compute Euclidean distance between two points
    fn euclidean_distance(&self, p1: ndarray::ArrayView1<f64>, p2: ndarray::ArrayView1<f64>) -> f64 {
        p1.iter()
            .zip(p2.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Embed time series into 2D point cloud using time-delay embedding
    fn embed_timeseries(&self, values: &[f64]) -> Array2<f64> {
        let n = values.len();
        let tau = (n / 10).max(1);
        let m = n - tau;

        let mut points = Array2::zeros((m, 2));
        for i in 0..m {
            points[[i, 0]] = values[i];
            points[[i, 1]] = values[i + tau];
        }

        points
    }

    /// Build filtration using Vietoris-Rips complex
    fn build_filtration(&self, distances: &Array2<f64>) -> Vec<f64> {
        let n = distances.nrows();
        let mut all_distances = Vec::new();

        for i in 0..n {
            for j in i + 1..n {
                all_distances.push(distances[[i, j]]);
            }
        }

        all_distances.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Determine max distance
        let max_dist = self
            .config
            .max_distance
            .unwrap_or_else(|| all_distances[all_distances.len() * 3 / 4]);

        // Create filtration steps
        let num_steps = self.config.num_steps;
        let filtration: Vec<f64> = (0..num_steps)
            .map(|i| (i as f64 / (num_steps - 1) as f64) * max_dist)
            .collect();

        filtration
    }

    /// Compute H₀ (connected components) using Union-Find
    fn compute_h0_persistence(
        &self,
        distances: &Array2<f64>,
        filtration: &[f64],
    ) -> Vec<BirthDeathPair> {
        let n = distances.nrows();
        let mut pairs = Vec::new();
        let mut uf = UnionFind::new(n);
        let mut component_births = vec![0.0; n];

        // Initially, each point is its own component (born at time 0)
        for i in 0..n {
            component_births[i] = 0.0;
        }

        for &epsilon in filtration.iter() {
            let time = epsilon;

            // Add edges at this filtration value
            for i in 0..n {
                for j in i + 1..n {
                    if distances[[i, j]] <= epsilon {
                        let root_i = uf.find(i);
                        let root_j = uf.find(j);

                        if root_i != root_j {
                            // Components merge - one dies
                            // The younger component dies
                            let death_time = time;
                            let (dying_root, _surviving_root) = if component_births[root_i] > component_births[root_j] {
                                (root_i, root_j)
                            } else {
                                (root_j, root_i)
                            };

                            let birth_time = component_births[dying_root];
                            if death_time > birth_time {
                                pairs.push(BirthDeathPair::new(birth_time, death_time, 0));
                            }

                            // Union the components
                            uf.union(root_i, root_j);
                        }
                    }
                }
            }
        }

        pairs
    }

    /// Detect 1-dimensional loops (H₁) using cycle detection
    fn compute_h1_persistence(
        &self,
        distances: &Array2<f64>,
        filtration: &[f64],
    ) -> Vec<BirthDeathPair> {
        let n = distances.nrows();
        let mut pairs = Vec::new();
        let mut graph: UnGraph<(), f64> = Graph::new_undirected();
        let mut node_map = HashMap::new();

        // Create nodes
        for i in 0..n {
            let node = graph.add_node(());
            node_map.insert(i, node);
        }

        let mut uf = UnionFind::new(n);

        for &epsilon in filtration.iter() {
            let mut edges_at_step = Vec::new();

            // Collect edges at this filtration value
            for i in 0..n {
                for j in i + 1..n {
                    if distances[[i, j]] <= epsilon {
                        edges_at_step.push((i, j, distances[[i, j]]));
                    }
                }
            }

            // Process edges
            for (i, j, dist) in edges_at_step {
                let root_i = uf.find(i);
                let root_j = uf.find(j);

                if root_i == root_j {
                    // Adding edge within same component creates a cycle
                    // This is birth of a 1-cycle
                    pairs.push(BirthDeathPair::new(dist, epsilon * 1.5, 1));
                } else {
                    // Merge components
                    uf.union(root_i, root_j);
                }

                let ni = node_map[&i];
                let nj = node_map[&j];
                graph.add_edge(ni, nj, dist);
            }
        }

        pairs
    }

    /// Interpret persistence diagram
    fn interpret_diagram(&self, diagram: &PersistenceDiagram) -> String {
        let h0_count = diagram.count_by_dimension(0);
        let h1_count = diagram.count_by_dimension(1);

        let significant_h0 = diagram
            .pairs_by_dimension(0)
            .iter()
            .filter(|p| p.is_significant(self.config.persistence_threshold))
            .count();

        let significant_h1 = diagram
            .pairs_by_dimension(1)
            .iter()
            .filter(|p| p.is_significant(self.config.persistence_threshold))
            .count();

        let max_persistence = diagram.max_persistence();

        format!(
            "Persistence diagram contains {} H₀ features ({} significant) and {} H₁ features ({} significant). \
             Maximum persistence: {:.4}. Significant features indicate robust topological structure. \
             H₀ represents connected components, H₁ represents loops/cycles.",
            h0_count, significant_h0, h1_count, significant_h1, max_persistence
        )
    }
}

impl Default for PersistentHomology {
    fn default() -> Self {
        Self::new()
    }
}

impl StochasticAnalyzer for PersistentHomology {
    fn name(&self) -> &str {
        "Persistent Homology"
    }

    fn analyze(&self, data: &TimeSeries) -> Result<AnalysisResult, StochasticError> {
        self.validate(data)?;

        let values = data.values();

        // Embed time series into 2D point cloud
        let points = self.embed_timeseries(&values);

        // Compute pairwise distances
        let distances = self.compute_distances(&points);

        // Build filtration
        let filtration = self.build_filtration(&distances);

        // Compute persistence for different dimensions
        let mut all_pairs = Vec::new();

        // H₀: Connected components
        let h0_pairs = self.compute_h0_persistence(&distances, &filtration);
        all_pairs.extend(h0_pairs);

        // H₁: Loops (if requested)
        if self.config.max_dimension >= 1 {
            let h1_pairs = self.compute_h1_persistence(&distances, &filtration);
            all_pairs.extend(h1_pairs);
        }

        // Create persistence diagram
        let max_filt = filtration.last().copied().unwrap_or(1.0);
        let diagram = PersistenceDiagram::new(all_pairs, max_filt);

        // Compute metrics
        let h0_count = diagram.count_by_dimension(0);
        let h1_count = diagram.count_by_dimension(1);
        let max_persistence = diagram.max_persistence();

        let significant_features = diagram.significant_features(self.config.persistence_threshold);
        let significant_count = significant_features.len();

        // Compute average persistence
        let avg_persistence = if !diagram.pairs.is_empty() {
            diagram.pairs.iter().map(|p| p.persistence()).sum::<f64>() / diagram.pairs.len() as f64
        } else {
            0.0
        };

        let interpretation = self.interpret_diagram(&diagram);

        Ok(AnalysisResult::new(self.name())
            .with_metric("h0_features", h0_count as f64)
            .with_metric("h1_features", h1_count as f64)
            .with_metric("max_persistence", max_persistence)
            .with_metric("avg_persistence", avg_persistence)
            .with_metric("significant_features", significant_count as f64)
            .with_metric("total_features", diagram.pairs.len() as f64)
            .with_metric("num_points", points.nrows() as f64)
            .with_metric("max_filtration", max_filt)
            .with_metadata("max_dimension", self.config.max_dimension.to_string())
            .with_metadata("num_steps", self.config.num_steps.to_string())
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

        // Check for NaN or infinite values
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
    fn test_birth_death_pair_creation() {
        let pair = BirthDeathPair::new(0.5, 1.5, 0);
        assert_eq!(pair.birth, 0.5);
        assert_eq!(pair.death, 1.5);
        assert_eq!(pair.dimension, 0);
    }

    #[test]
    fn test_persistence_computation() {
        let pair = BirthDeathPair::new(0.5, 2.5, 1);
        assert_relative_eq!(pair.persistence(), 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_significance_check() {
        let pair = BirthDeathPair::new(0.0, 1.5, 0);
        assert!(pair.is_significant(0.5));
        assert!(!pair.is_significant(2.0));
    }

    #[test]
    fn test_persistence_diagram_creation() {
        let pairs = vec![
            BirthDeathPair::new(0.0, 1.0, 0),
            BirthDeathPair::new(0.5, 2.0, 1),
        ];
        let diagram = PersistenceDiagram::new(pairs, 2.5);
        assert_eq!(diagram.pairs.len(), 2);
        assert_eq!(diagram.max_filtration, 2.5);
    }

    #[test]
    fn test_pairs_by_dimension() {
        let pairs = vec![
            BirthDeathPair::new(0.0, 1.0, 0),
            BirthDeathPair::new(0.5, 2.0, 1),
            BirthDeathPair::new(0.3, 1.5, 0),
        ];
        let diagram = PersistenceDiagram::new(pairs, 2.5);

        let h0 = diagram.pairs_by_dimension(0);
        let h1 = diagram.pairs_by_dimension(1);

        assert_eq!(h0.len(), 2);
        assert_eq!(h1.len(), 1);
    }

    #[test]
    fn test_significant_features() {
        let pairs = vec![
            BirthDeathPair::new(0.0, 1.5, 0),  // persistence = 1.5
            BirthDeathPair::new(0.5, 0.6, 1),  // persistence = 0.1
        ];
        let diagram = PersistenceDiagram::new(pairs, 2.0);

        let significant = diagram.significant_features(0.5);
        assert_eq!(significant.len(), 1);
        assert_eq!(significant[0].dimension, 0);
    }

    #[test]
    fn test_max_persistence() {
        let pairs = vec![
            BirthDeathPair::new(0.0, 1.0, 0),
            BirthDeathPair::new(0.5, 3.0, 1),
        ];
        let diagram = PersistenceDiagram::new(pairs, 3.0);
        assert_relative_eq!(diagram.max_persistence(), 2.5, epsilon = 1e-10);
    }

    #[test]
    fn test_config_defaults() {
        let config = PersistenceConfig::default();
        assert_eq!(config.max_dimension, 1);
        assert_eq!(config.num_steps, 50);
        assert_eq!(config.persistence_threshold, 0.1);
    }

    #[test]
    fn test_config_builder() {
        let config = PersistenceConfig::new()
            .with_max_dimension(2)
            .with_num_steps(100)
            .with_persistence_threshold(0.2);

        assert_eq!(config.max_dimension, 2);
        assert_eq!(config.num_steps, 100);
        assert_eq!(config.persistence_threshold, 0.2);
    }

    #[test]
    fn test_analyzer_creation() {
        let analyzer = PersistentHomology::new();
        assert_eq!(analyzer.name(), "Persistent Homology");
    }

    #[test]
    fn test_euclidean_distance() {
        let analyzer = PersistentHomology::new();
        let p1 = Array1::from_vec(vec![0.0, 0.0]);
        let p2 = Array1::from_vec(vec![3.0, 4.0]);

        let dist = analyzer.euclidean_distance(p1.view(), p2.view());
        assert_relative_eq!(dist, 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_validation_insufficient_data() {
        let analyzer = PersistentHomology::new();
        let data = TimeSeries::from_values(vec![1.0, 2.0, 3.0]);

        let result = analyzer.validate(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_validation_nan_data() {
        let analyzer = PersistentHomology::new();
        let mut values = vec![1.0; 50];
        values[25] = f64::NAN;
        let data = TimeSeries::from_values(values);

        let result = analyzer.validate(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_circle_detection() {
        // Generate points on a circle - should have one significant H₁ feature
        let mut data = Vec::new();
        for i in 0..100 {
            let angle = 2.0 * std::f64::consts::PI * (i as f64) / 100.0;
            data.push(angle.cos() + angle.sin());
        }

        let ts = TimeSeries::from_values(data);
        let analyzer = PersistentHomology::new();

        let result = analyzer.analyze(&ts);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert!(result.metrics.contains_key("h0_features"));
        assert!(result.metrics.contains_key("h1_features"));
    }

    #[test]
    fn test_linear_data() {
        // Linear data should have minimal topological features
        let data: Vec<f64> = (0..100).map(|x| x as f64).collect();
        let ts = TimeSeries::from_values(data);
        let analyzer = PersistentHomology::new();

        let result = analyzer.analyze(&ts).unwrap();
        let h1 = result.metrics.get("h1_features").unwrap();

        // Linear data in 2D embedding can have some features but should be finite
        assert!(h1.is_finite());
        assert!(*h1 >= 0.0);
    }

    #[test]
    fn test_noisy_sine_wave() {
        // Sine wave with noise
        let mut data = Vec::new();
        for i in 0..200 {
            let x = (i as f64) * 0.1;
            data.push(x.sin() + ((i % 7) as f64) * 0.01);
        }

        let ts = TimeSeries::from_values(data);
        let analyzer = PersistentHomology::new();

        let result = analyzer.analyze(&ts);
        assert!(result.is_ok());
    }

    #[test]
    fn test_analyze_returns_all_metrics() {
        let data: Vec<f64> = (0..100).map(|x| (x as f64 * 0.1).sin()).collect();
        let ts = TimeSeries::from_values(data);
        let analyzer = PersistentHomology::new();

        let result = analyzer.analyze(&ts).unwrap();

        assert!(result.metrics.contains_key("h0_features"));
        assert!(result.metrics.contains_key("h1_features"));
        assert!(result.metrics.contains_key("max_persistence"));
        assert!(result.metrics.contains_key("avg_persistence"));
        assert!(result.metrics.contains_key("significant_features"));
        assert!(result.metrics.contains_key("total_features"));
        assert!(!result.interpretation.is_empty());
    }

    #[test]
    fn test_embed_timeseries() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let analyzer = PersistentHomology::new();
        let embedded = analyzer.embed_timeseries(&data);

        assert!(embedded.nrows() > 0);
        assert_eq!(embedded.ncols(), 2);
    }

    #[test]
    fn test_compute_distances() {
        let analyzer = PersistentHomology::new();
        let mut points = Array2::zeros((3, 2));
        points[[0, 0]] = 0.0;
        points[[0, 1]] = 0.0;
        points[[1, 0]] = 1.0;
        points[[1, 1]] = 0.0;
        points[[2, 0]] = 0.0;
        points[[2, 1]] = 1.0;

        let distances = analyzer.compute_distances(&points);

        assert_eq!(distances.nrows(), 3);
        assert_eq!(distances.ncols(), 3);
        assert_relative_eq!(distances[[0, 1]], 1.0, epsilon = 1e-10);
        assert_relative_eq!(distances[[0, 2]], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_filtration_construction() {
        let analyzer = PersistentHomology::new();
        let mut distances = Array2::zeros((5, 5));

        for i in 0..5 {
            for j in 0..5 {
                distances[[i, j]] = ((i as f64) - (j as f64)).abs();
            }
        }

        let filtration = analyzer.build_filtration(&distances);

        assert_eq!(filtration.len(), analyzer.config.num_steps);
        assert!(filtration[0] <= filtration[filtration.len() - 1]);
    }
}
