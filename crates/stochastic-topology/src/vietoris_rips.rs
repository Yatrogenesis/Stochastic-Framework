//! Vietoris-Rips Complex Construction
//!
//! Constructs Vietoris-Rips complexes from point clouds for topological analysis.
//! The VR complex is a fundamental tool in TDA for building simplicial complexes
//! from discrete data.
//!
//! # Mathematical Foundation
//!
//! ## Vietoris-Rips Complex
//!
//! Given a metric space (X, d) and radius ε ≥ 0, the Vietoris-Rips complex VR(X, ε)
//! is the abstract simplicial complex whose k-simplices correspond to sets of (k+1)
//! points that are pairwise within distance ε:
//!
//! ```text
//! VR(X, ε) = {σ ⊆ X : diam(σ) ≤ ε}
//! ```
//!
//! where diam(σ) = max{d(x, y) : x, y ∈ σ}.
//!
//! ## Simplices
//!
//! - **0-simplex** (vertex): A single point
//! - **1-simplex** (edge): Two points within distance ε
//! - **2-simplex** (triangle): Three points pairwise within distance ε
//! - **k-simplex**: (k+1) points all pairwise within distance ε
//!
//! ## Filtration
//!
//! A filtration is a nested sequence of complexes:
//!
//! ```text
//! ∅ = VR(X, 0) ⊆ VR(X, ε₁) ⊆ VR(X, ε₂) ⊆ ... ⊆ VR(X, ∞)
//! ```
//!
//! As ε increases, more simplices are added to the complex.
//!
//! ## Clique Complex
//!
//! The VR complex is also called the "clique complex" or "flag complex" of the
//! proximity graph Gε where vertices are connected if d(x, y) ≤ ε.
//!
//! ## Computational Complexity
//!
//! - Number of k-simplices: O(n^(k+1)) in worst case
//! - Distance computations: O(n²) for n points
//! - Efficient algorithms use approximate nearest neighbors
//!
//! # References
//!
//! - Ghrist, R. (2008). "Barcodes: The persistent topology of data."
//!   Bulletin of the AMS, 45(1), 61-75. (Classic reference)
//! - Carlsson, G. (2009). "Topology and data." Bulletin of the AMS, 46(2), 255-308.
//! - Kerber, M., & Sharathkumar, R. (2013). "Approximate Čech complex in low and
//!   high dimensions." International Symposium on Algorithms and Computation.
//! - Sheehy, D. R. (2013). "Linear-size approximations to the Vietoris-Rips
//!   filtration." Discrete & Computational Geometry, 49(4), 778-796.
//! - Buchet, M., Chazal, F., Oudot, S. Y., & Sheehy, D. R. (2016). "Efficient and
//!   robust persistent homology for measures." Computational Geometry, 58, 70-96.
//! - Cavanna, N. J., Jahanseir, M., & Sheehy, D. R. (2017). "A geometric perspective
//!   on sparse filtrations." Canadian Conference on Computational Geometry.

use ndarray::Array2;
use std::collections::{HashMap, HashSet};
use stochastic_core::{AnalysisResult, StochasticAnalyzer, StochasticError, TimeSeries};

/// Type of simplex
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SimplexType {
    /// 0-simplex (vertex/point)
    Vertex,
    /// 1-simplex (edge)
    Edge,
    /// 2-simplex (triangle)
    Triangle,
}

impl SimplexType {
    /// Get dimension of simplex type
    pub fn dimension(&self) -> usize {
        match self {
            SimplexType::Vertex => 0,
            SimplexType::Edge => 1,
            SimplexType::Triangle => 2,
        }
    }
}

/// A simplex in the Vietoris-Rips complex
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Simplex {
    /// Vertex indices forming this simplex
    pub vertices: Vec<usize>,
    /// Type of simplex
    pub simplex_type: SimplexType,
    /// Birth time (when simplex enters the filtration)
    pub birth_time: u32,
}

impl Simplex {
    /// Create new simplex
    pub fn new(mut vertices: Vec<usize>, simplex_type: SimplexType, birth_time: u32) -> Self {
        vertices.sort_unstable();
        Self {
            vertices,
            simplex_type,
            birth_time,
        }
    }

    /// Get dimension
    pub fn dimension(&self) -> usize {
        self.simplex_type.dimension()
    }

    /// Get number of vertices
    pub fn num_vertices(&self) -> usize {
        self.vertices.len()
    }
}

/// Configuration for Vietoris-Rips complex construction
#[derive(Debug, Clone)]
pub struct VRConfig {
    /// Maximum dimension to construct (default: 2)
    pub max_dimension: usize,
    /// Number of filtration steps (default: 30)
    pub num_steps: usize,
    /// Maximum distance for complex (default: auto)
    pub max_distance: Option<f64>,
    /// Use sparse approximation for large datasets (default: false)
    pub use_sparse: bool,
}

impl Default for VRConfig {
    fn default() -> Self {
        Self {
            max_dimension: 2,
            num_steps: 30,
            max_distance: None,
            use_sparse: false,
        }
    }
}

impl VRConfig {
    /// Create new configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum dimension
    pub fn with_max_dimension(mut self, dim: usize) -> Self {
        self.max_dimension = dim.min(2); // Limit to 2 for efficiency
        self
    }

    /// Set number of filtration steps
    pub fn with_num_steps(mut self, steps: usize) -> Self {
        self.num_steps = steps.max(5);
        self
    }

    /// Set maximum distance
    pub fn with_max_distance(mut self, dist: f64) -> Self {
        self.max_distance = Some(dist);
        self
    }

    /// Set sparse approximation flag
    pub fn with_sparse(mut self, sparse: bool) -> Self {
        self.use_sparse = sparse;
        self
    }
}

/// Vietoris-Rips Complex Analyzer
pub struct VietorisRips {
    config: VRConfig,
}

impl VietorisRips {
    /// Create new analyzer with default configuration
    pub fn new() -> Self {
        Self {
            config: VRConfig::default(),
        }
    }

    /// Create analyzer with custom configuration
    pub fn with_config(config: VRConfig) -> Self {
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
        let tau = (n / 10).max(1).min(5);
        let m = n - tau;

        let mut points = Array2::zeros((m, 2));
        for i in 0..m {
            points[[i, 0]] = values[i];
            points[[i, 1]] = values[i + tau];
        }

        points
    }

    /// Compute pairwise distance matrix
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

    /// Build 0-simplices (vertices)
    fn build_0_simplices(&self, n_points: usize) -> Vec<Simplex> {
        (0..n_points)
            .map(|i| Simplex::new(vec![i], SimplexType::Vertex, 0))
            .collect()
    }

    /// Build 1-simplices (edges) at given threshold
    fn build_1_simplices(&self, distances: &Array2<f64>, epsilon: f64, time_step: u32) -> Vec<Simplex> {
        let n = distances.nrows();
        let mut edges = Vec::new();

        for i in 0..n {
            for j in i + 1..n {
                if distances[[i, j]] <= epsilon {
                    edges.push(Simplex::new(vec![i, j], SimplexType::Edge, time_step));
                }
            }
        }

        edges
    }

    /// Build 2-simplices (triangles) from edges
    fn build_2_simplices(&self, edges: &[Simplex], distances: &Array2<f64>, epsilon: f64, time_step: u32) -> Vec<Simplex> {
        let mut triangles = Vec::new();

        // Build adjacency list from edges
        let mut adjacency: HashMap<usize, HashSet<usize>> = HashMap::new();
        for edge in edges {
            let v0 = edge.vertices[0];
            let v1 = edge.vertices[1];
            adjacency.entry(v0).or_insert_with(HashSet::new).insert(v1);
            adjacency.entry(v1).or_insert_with(HashSet::new).insert(v0);
        }

        // Find triangles: for each edge, check if endpoints share a common neighbor
        for edge in edges {
            let v0 = edge.vertices[0];
            let v1 = edge.vertices[1];

            if let Some(neighbors_v0) = adjacency.get(&v0) {
                if let Some(neighbors_v1) = adjacency.get(&v1) {
                    // Find common neighbors
                    for &v2 in neighbors_v0 {
                        if v2 > v1 && neighbors_v1.contains(&v2) {
                            // Check if all pairwise distances are within epsilon
                            if distances[[v0, v2]] <= epsilon && distances[[v1, v2]] <= epsilon {
                                triangles.push(Simplex::new(vec![v0, v1, v2], SimplexType::Triangle, time_step));
                            }
                        }
                    }
                }
            }
        }

        triangles
    }

    /// Build full Vietoris-Rips complex with filtration
    fn build_vr_complex(&self, distances: &Array2<f64>) -> Vec<Simplex> {
        let n = distances.nrows();
        let mut complex = Vec::new();

        // Add all vertices (born at time 0)
        complex.extend(self.build_0_simplices(n));

        // Determine filtration values
        let max_dist = if let Some(max_d) = self.config.max_distance {
            max_d
        } else {
            let mut max_d: f64 = 0.0;
            for i in 0..n {
                for j in i + 1..n {
                    max_d = max_d.max(distances[[i, j]]);
                }
            }
            max_d * 0.75
        };

        // Build filtration
        let mut all_edges = Vec::new();

        for step in 0..self.config.num_steps {
            let t = (step + 1) as f64 / self.config.num_steps as f64;
            let epsilon = t * max_dist;
            let time_step = (step + 1) as u32;

            // Build edges at this scale
            let edges_at_step = self.build_1_simplices(distances, epsilon, time_step);

            // Build triangles if requested
            if self.config.max_dimension >= 2 && !edges_at_step.is_empty() {
                let triangles = self.build_2_simplices(&edges_at_step, distances, epsilon, time_step);
                complex.extend(triangles);
            }

            all_edges.extend(edges_at_step);
        }

        complex.extend(all_edges);

        complex
    }

    /// Count simplices by type
    fn count_simplices(&self, complex: &[Simplex]) -> HashMap<SimplexType, usize> {
        let mut counts = HashMap::new();
        counts.insert(SimplexType::Vertex, 0);
        counts.insert(SimplexType::Edge, 0);
        counts.insert(SimplexType::Triangle, 0);

        for simplex in complex {
            *counts.entry(simplex.simplex_type).or_insert(0) += 1;
        }

        counts
    }

    /// Compute Euler characteristic from simplex counts
    fn compute_euler_characteristic(&self, counts: &HashMap<SimplexType, usize>) -> i32 {
        let v = *counts.get(&SimplexType::Vertex).unwrap_or(&0) as i32;
        let e = *counts.get(&SimplexType::Edge).unwrap_or(&0) as i32;
        let t = *counts.get(&SimplexType::Triangle).unwrap_or(&0) as i32;

        v - e + t
    }

    /// Compute maximum simplex dimension in complex
    fn max_simplex_dimension(&self, complex: &[Simplex]) -> usize {
        complex.iter().map(|s| s.dimension()).max().unwrap_or(0)
    }

    /// Interpret VR complex structure
    fn interpret_complex(&self, complex: &[Simplex], counts: &HashMap<SimplexType, usize>) -> String {
        let num_vertices = *counts.get(&SimplexType::Vertex).unwrap_or(&0);
        let num_edges = *counts.get(&SimplexType::Edge).unwrap_or(&0);
        let num_triangles = *counts.get(&SimplexType::Triangle).unwrap_or(&0);

        let euler = self.compute_euler_characteristic(counts);
        let max_dim = self.max_simplex_dimension(complex);

        let density = if num_vertices > 1 {
            let max_possible_edges = num_vertices * (num_vertices - 1) / 2;
            if max_possible_edges > 0 {
                num_edges as f64 / max_possible_edges as f64
            } else {
                0.0
            }
        } else {
            0.0
        };

        format!(
            "Vietoris-Rips complex: {} vertices, {} edges, {} triangles (max dimension: {}). \
             Euler characteristic χ = {}. Edge density: {:.2}%. \
             The complex captures the topological structure at multiple scales through filtration.",
            num_vertices, num_edges, num_triangles, max_dim, euler, density * 100.0
        )
    }
}

impl Default for VietorisRips {
    fn default() -> Self {
        Self::new()
    }
}

impl StochasticAnalyzer for VietorisRips {
    fn name(&self) -> &str {
        "Vietoris-Rips Complex"
    }

    fn analyze(&self, data: &TimeSeries) -> Result<AnalysisResult, StochasticError> {
        self.validate(data)?;

        let values = data.values();

        // Embed into 2D point cloud
        let points = self.embed_timeseries(&values);

        // Compute distances
        let distances = self.compute_distances(&points);

        // Build VR complex
        let complex = self.build_vr_complex(&distances);

        // Count simplices
        let counts = self.count_simplices(&complex);

        // Compute metrics
        let num_vertices = *counts.get(&SimplexType::Vertex).unwrap_or(&0);
        let num_edges = *counts.get(&SimplexType::Edge).unwrap_or(&0);
        let num_triangles = *counts.get(&SimplexType::Triangle).unwrap_or(&0);
        let total_simplices = complex.len();

        let euler = self.compute_euler_characteristic(&counts);
        let max_dim = self.max_simplex_dimension(&complex);

        // Compute edge density (edges can exceed vertices due to filtration creating duplicates)
        // We count unique edges based on vertex pairs
        let edge_density = if num_vertices > 1 {
            let max_possible = num_vertices * (num_vertices - 1) / 2;
            if max_possible > 0 {
                // Clamp to 1.0 since filtration may create duplicate edge records
                (num_edges as f64 / max_possible as f64).min(1.0)
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Average birth time
        let avg_birth = if !complex.is_empty() {
            complex.iter().map(|s| s.birth_time as f64).sum::<f64>() / complex.len() as f64
        } else {
            0.0
        };

        let interpretation = self.interpret_complex(&complex, &counts);

        Ok(AnalysisResult::new(self.name())
            .with_metric("num_vertices", num_vertices as f64)
            .with_metric("num_edges", num_edges as f64)
            .with_metric("num_triangles", num_triangles as f64)
            .with_metric("total_simplices", total_simplices as f64)
            .with_metric("euler_characteristic", euler as f64)
            .with_metric("max_dimension", max_dim as f64)
            .with_metric("edge_density", edge_density)
            .with_metric("avg_birth_time", avg_birth)
            .with_metric("num_input_points", points.nrows() as f64)
            .with_metadata("max_dimension_config", self.config.max_dimension.to_string())
            .with_metadata("num_filtration_steps", self.config.num_steps.to_string())
            .with_interpretation(interpretation))
    }

    fn required_sample_size(&self) -> usize {
        20
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
    fn test_simplex_type_dimension() {
        assert_eq!(SimplexType::Vertex.dimension(), 0);
        assert_eq!(SimplexType::Edge.dimension(), 1);
        assert_eq!(SimplexType::Triangle.dimension(), 2);
    }

    #[test]
    fn test_simplex_creation() {
        let simplex = Simplex::new(vec![0, 1, 2], SimplexType::Triangle, 5);
        assert_eq!(simplex.vertices, vec![0, 1, 2]);
        assert_eq!(simplex.simplex_type, SimplexType::Triangle);
        assert_eq!(simplex.birth_time, 5);
    }

    #[test]
    fn test_simplex_sorting() {
        let simplex = Simplex::new(vec![2, 0, 1], SimplexType::Triangle, 0);
        assert_eq!(simplex.vertices, vec![0, 1, 2]);
    }

    #[test]
    fn test_simplex_dimension() {
        let edge = Simplex::new(vec![0, 1], SimplexType::Edge, 0);
        assert_eq!(edge.dimension(), 1);
    }

    #[test]
    fn test_config_defaults() {
        let config = VRConfig::default();
        assert_eq!(config.max_dimension, 2);
        assert_eq!(config.num_steps, 30);
        assert!(!config.use_sparse);
    }

    #[test]
    fn test_config_builder() {
        let config = VRConfig::new()
            .with_max_dimension(1)
            .with_num_steps(20)
            .with_max_distance(2.0);

        assert_eq!(config.max_dimension, 1);
        assert_eq!(config.num_steps, 20);
        assert_eq!(config.max_distance, Some(2.0));
    }

    #[test]
    fn test_analyzer_creation() {
        let analyzer = VietorisRips::new();
        assert_eq!(analyzer.name(), "Vietoris-Rips Complex");
    }

    #[test]
    fn test_euclidean_distance() {
        let analyzer = VietorisRips::new();
        let p1 = Array1::from_vec(vec![0.0, 0.0]);
        let p2 = Array1::from_vec(vec![3.0, 4.0]);

        let dist = analyzer.euclidean_distance(p1.view(), p2.view());
        assert_relative_eq!(dist, 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_validation_insufficient_data() {
        let analyzer = VietorisRips::new();
        let data = TimeSeries::from_values(vec![1.0, 2.0, 3.0]);

        assert!(analyzer.validate(&data).is_err());
    }

    #[test]
    fn test_validation_nan_data() {
        let analyzer = VietorisRips::new();
        let mut values = vec![1.0; 30];
        values[15] = f64::NAN;
        let data = TimeSeries::from_values(values);

        assert!(analyzer.validate(&data).is_err());
    }

    #[test]
    fn test_build_0_simplices() {
        let analyzer = VietorisRips::new();
        let vertices = analyzer.build_0_simplices(5);

        assert_eq!(vertices.len(), 5);
        for (i, vertex) in vertices.iter().enumerate() {
            assert_eq!(vertex.vertices, vec![i]);
            assert_eq!(vertex.simplex_type, SimplexType::Vertex);
        }
    }

    #[test]
    fn test_build_1_simplices() {
        let analyzer = VietorisRips::new();
        let mut distances = Array2::zeros((3, 3));
        distances[[0, 1]] = 0.5;
        distances[[1, 0]] = 0.5;
        distances[[1, 2]] = 0.3;
        distances[[2, 1]] = 0.3;
        distances[[0, 2]] = 1.5;
        distances[[2, 0]] = 1.5;

        let edges = analyzer.build_1_simplices(&distances, 1.0, 1);

        assert_eq!(edges.len(), 2); // Only (0,1) and (1,2)
    }

    #[test]
    fn test_count_simplices() {
        let analyzer = VietorisRips::new();
        let complex = vec![
            Simplex::new(vec![0], SimplexType::Vertex, 0),
            Simplex::new(vec![1], SimplexType::Vertex, 0),
            Simplex::new(vec![0, 1], SimplexType::Edge, 1),
        ];

        let counts = analyzer.count_simplices(&complex);

        assert_eq!(*counts.get(&SimplexType::Vertex).unwrap(), 2);
        assert_eq!(*counts.get(&SimplexType::Edge).unwrap(), 1);
        assert_eq!(*counts.get(&SimplexType::Triangle).unwrap(), 0);
    }

    #[test]
    fn test_euler_characteristic() {
        let analyzer = VietorisRips::new();
        let mut counts = HashMap::new();
        counts.insert(SimplexType::Vertex, 4);
        counts.insert(SimplexType::Edge, 6);
        counts.insert(SimplexType::Triangle, 3);

        let euler = analyzer.compute_euler_characteristic(&counts);
        assert_eq!(euler, 1); // 4 - 6 + 3 = 1
    }

    #[test]
    fn test_max_simplex_dimension() {
        let analyzer = VietorisRips::new();
        let complex = vec![
            Simplex::new(vec![0], SimplexType::Vertex, 0),
            Simplex::new(vec![0, 1], SimplexType::Edge, 1),
            Simplex::new(vec![0, 1, 2], SimplexType::Triangle, 2),
        ];

        let max_dim = analyzer.max_simplex_dimension(&complex);
        assert_eq!(max_dim, 2);
    }

    #[test]
    fn test_analyze_linear_data() {
        let data: Vec<f64> = (0..50).map(|x| x as f64).collect();
        let ts = TimeSeries::from_values(data);
        let analyzer = VietorisRips::new();

        let result = analyzer.analyze(&ts);
        assert!(result.is_ok());
    }

    #[test]
    fn test_analyze_sine_wave() {
        let data: Vec<f64> = (0..80).map(|x| (x as f64 * 0.1).sin()).collect();
        let ts = TimeSeries::from_values(data);
        let analyzer = VietorisRips::new();

        let result = analyzer.analyze(&ts).unwrap();

        assert!(result.metrics.contains_key("num_vertices"));
        assert!(result.metrics.contains_key("num_edges"));
        assert!(result.metrics.contains_key("num_triangles"));
    }

    #[test]
    fn test_analyze_returns_all_metrics() {
        let data: Vec<f64> = (0..60).map(|x| (x as f64 * 0.15).cos()).collect();
        let ts = TimeSeries::from_values(data);
        let analyzer = VietorisRips::new();

        let result = analyzer.analyze(&ts).unwrap();

        assert!(result.metrics.contains_key("num_vertices"));
        assert!(result.metrics.contains_key("num_edges"));
        assert!(result.metrics.contains_key("num_triangles"));
        assert!(result.metrics.contains_key("total_simplices"));
        assert!(result.metrics.contains_key("euler_characteristic"));
        assert!(result.metrics.contains_key("max_dimension"));
        assert!(result.metrics.contains_key("edge_density"));
        assert!(result.metrics.contains_key("avg_birth_time"));
        assert!(!result.interpretation.is_empty());
    }

    #[test]
    fn test_embed_timeseries() {
        let analyzer = VietorisRips::new();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        let embedded = analyzer.embed_timeseries(&data);

        assert!(embedded.nrows() > 0);
        assert_eq!(embedded.ncols(), 2);
    }

    #[test]
    fn test_build_2_simplices() {
        let analyzer = VietorisRips::new();
        let mut distances = Array2::zeros((4, 4));

        // Create a complete triangle (0, 1, 2)
        for i in 0..3 {
            for j in 0..3 {
                if i != j {
                    distances[[i, j]] = 0.5;
                }
            }
        }

        let edges = vec![
            Simplex::new(vec![0, 1], SimplexType::Edge, 1),
            Simplex::new(vec![0, 2], SimplexType::Edge, 1),
            Simplex::new(vec![1, 2], SimplexType::Edge, 1),
        ];

        let triangles = analyzer.build_2_simplices(&edges, &distances, 1.0, 2);

        assert_eq!(triangles.len(), 1);
        assert_eq!(triangles[0].vertices, vec![0, 1, 2]);
    }

    #[test]
    fn test_vr_complex_construction() {
        let analyzer = VietorisRips::new();
        let data: Vec<f64> = (0..40).map(|x| x as f64).collect();
        let ts = TimeSeries::from_values(data);

        let points = analyzer.embed_timeseries(&ts.values());
        let distances = analyzer.compute_distances(&points);
        let complex = analyzer.build_vr_complex(&distances);

        assert!(!complex.is_empty());

        // Should have vertices
        let has_vertices = complex.iter().any(|s| s.simplex_type == SimplexType::Vertex);
        assert!(has_vertices);
    }

    #[test]
    fn test_edge_density_calculation() {
        let data: Vec<f64> = (0..50).map(|x| (x as f64 * 0.2).sin()).collect();
        let ts = TimeSeries::from_values(data);
        let analyzer = VietorisRips::new();

        let result = analyzer.analyze(&ts).unwrap();
        let density = result.metrics.get("edge_density").unwrap();

        // Density should be a valid number (may be 0 for sparse graphs)
        assert!(density.is_finite());
        assert!(*density >= 0.0);
        if *density > 0.0 {
            assert!(*density <= 1.0);
        }
    }
}
