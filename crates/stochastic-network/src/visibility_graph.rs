//! Visibility Graph Analysis
//!
//! Implements Natural Visibility Graph (NVG) and Horizontal Visibility Graph (HVG)
//! algorithms for converting time series into complex networks. These methods map
//! temporal correlations into topological features, enabling network-based analysis
//! of time series data.
//!
//! # Mathematical Foundation
//!
//! ## Natural Visibility Graph
//!
//! Two data points (tᵢ, yᵢ) and (tⱼ, yⱼ) are mutually visible if for all tₖ between
//! tᵢ and tⱼ, the following geometric condition holds:
//!
//! ```text
//! y(tₖ) < y(tⱼ) + (y(tᵢ) - y(tⱼ)) × (tₖ - tⱼ) / (tᵢ - tⱼ)
//! ```
//!
//! This condition ensures that the line of sight between points i and j is not
//! blocked by any intermediate point k.
//!
//! ## Horizontal Visibility Graph
//!
//! A simpler criterion where two points (tᵢ, yᵢ) and (tⱼ, yⱼ) are connected if:
//!
//! ```text
//! y(tₖ) < min(y(tᵢ), y(tⱼ))  for all tₖ ∈ (tᵢ, tⱼ)
//! ```
//!
//! ## Network Properties
//!
//! For a graph G = (V, E):
//!
//! **Degree Distribution**: P(k) = probability that a node has degree k
//!
//! **Average Degree**: ⟨k⟩ = (1/N) Σᵢ kᵢ
//!
//! **Degree Centrality**: C_D(v) = k(v) / (N - 1)
//!
//! **Network Density**: ρ = 2|E| / (N(N-1))
//!
//! ## Theoretical Results
//!
//! - Random series → exponential degree distribution
//! - Periodic series → regular graph structure
//! - Fractal series → scale-free degree distribution
//! - Chaotic series → specific topological signatures
//!
//! # References
//!
//! - Lacasa, L., Luque, B., Ballesteros, F., Luque, J., & Nuño, J.C. (2008).
//!   "From time series to complex networks: The visibility graph."
//!   Proceedings of the National Academy of Sciences, 105(13), 4972-4975.
//! - Luque, B., Lacasa, L., Ballesteros, F., & Luque, J. (2009).
//!   "Horizontal visibility graphs: Exact results for random time series."
//!   Physical Review E, 80(4), 046103.
//! - Lacasa, L., & Toral, R. (2010). "Description of stochastic and chaotic
//!   series using visibility graphs." Physical Review E, 82(3), 036120.
//! - Donner, R.V., Zou, Y., Donges, J.F., Marwan, N., & Kurths, J. (2010).
//!   "Recurrence networks—a novel paradigm for nonlinear time series analysis."
//!   New Journal of Physics, 12(3), 033025.
//! - Zou, Y., Donner, R.V., Marwan, N., Donges, J.F., & Kurths, J. (2019).
//!   "Complex network approaches to nonlinear time series analysis."
//!   Physics Reports, 787, 1-97.

use petgraph::graph::{Graph, NodeIndex};
use petgraph::algo::{dijkstra, connected_components};
use petgraph::Undirected;
use std::collections::HashMap;
use stochastic_core::{AnalysisResult, StochasticAnalyzer, StochasticError, TimeSeries};

/// Configuration for visibility graph analysis
#[derive(Debug, Clone)]
pub struct VisibilityConfig {
    /// Use horizontal visibility (simpler) instead of natural visibility
    pub horizontal: bool,
    /// Create directed graph (default: false for undirected)
    pub directed: bool,
    /// Number of bins for degree distribution histogram
    pub degree_bins: usize,
    /// Calculate network diameter (computationally expensive for large networks)
    pub calculate_diameter: bool,
    /// Maximum degree to consider in distribution (for very large networks)
    pub max_degree: Option<usize>,
}

impl Default for VisibilityConfig {
    fn default() -> Self {
        Self {
            horizontal: false,
            directed: false,
            degree_bins: 20,
            calculate_diameter: true,
            max_degree: None,
        }
    }
}

impl VisibilityConfig {
    /// Create new configuration with defaults
    pub fn new() -> Self {
        Self::default()
    }

    /// Use horizontal visibility graph
    pub fn horizontal(mut self) -> Self {
        self.horizontal = true;
        self
    }

    /// Use natural visibility graph (default)
    pub fn natural(mut self) -> Self {
        self.horizontal = false;
        self
    }

    /// Create directed graph
    pub fn directed(mut self) -> Self {
        self.directed = true;
        self
    }

    /// Create undirected graph (default)
    pub fn undirected(mut self) -> Self {
        self.directed = false;
        self
    }

    /// Set number of bins for degree distribution
    pub fn with_degree_bins(mut self, bins: usize) -> Self {
        self.degree_bins = bins.max(5);
        self
    }

    /// Enable/disable diameter calculation
    pub fn with_diameter_calculation(mut self, calculate: bool) -> Self {
        self.calculate_diameter = calculate;
        self
    }

    /// Set maximum degree to consider
    pub fn with_max_degree(mut self, max_degree: usize) -> Self {
        self.max_degree = Some(max_degree);
        self
    }
}

/// Visibility Graph Analyzer
pub struct VisibilityGraphAnalyzer {
    config: VisibilityConfig,
}

impl VisibilityGraphAnalyzer {
    /// Create new analyzer with default configuration
    pub fn new() -> Self {
        Self {
            config: VisibilityConfig::default(),
        }
    }

    /// Create analyzer with custom configuration
    pub fn with_config(config: VisibilityConfig) -> Self {
        Self { config }
    }

    /// Use horizontal visibility graph
    pub fn horizontal(mut self) -> Self {
        self.config.horizontal = true;
        self
    }

    /// Use natural visibility graph
    pub fn natural(mut self) -> Self {
        self.config.horizontal = false;
        self
    }

    /// Create directed graph
    pub fn directed(mut self) -> Self {
        self.config.directed = true;
        self
    }

    /// Build natural visibility graph from time series
    ///
    /// Two points (i, yᵢ) and (j, yⱼ) with i < j are connected if all intermediate
    /// points k satisfy: y_k < y_j + (y_i - y_j) * (k - j) / (i - j)
    fn build_natural_visibility_graph(&self, data: &[f64]) -> Graph<usize, (), Undirected> {
        let n = data.len();
        let mut graph = Graph::new_undirected();

        // Create nodes
        let nodes: Vec<NodeIndex> = (0..n).map(|i| graph.add_node(i)).collect();

        // Check visibility between all pairs
        for i in 0..n {
            for j in (i + 1)..n {
                if self.is_visible_natural(data, i, j) {
                    graph.add_edge(nodes[i], nodes[j], ());
                }
            }
        }

        graph
    }

    /// Check if two points have natural visibility
    fn is_visible_natural(&self, data: &[f64], i: usize, j: usize) -> bool {
        if i >= j {
            return false;
        }

        let yi = data[i];
        let yj = data[j];
        let ti = i as f64;
        let tj = j as f64;

        // Check all intermediate points
        for k in (i + 1)..j {
            let yk = data[k];
            let tk = k as f64;

            // Line of sight equation: y_k should be below the line connecting (i, yi) and (j, yj)
            let threshold = yj + (yi - yj) * (tk - tj) / (ti - tj);

            if yk >= threshold {
                return false; // Blocked by point k
            }
        }

        true
    }

    /// Build horizontal visibility graph from time series
    ///
    /// Two points (i, yᵢ) and (j, yⱼ) are connected if all intermediate points k
    /// satisfy: y_k < min(y_i, y_j)
    fn build_horizontal_visibility_graph(&self, data: &[f64]) -> Graph<usize, (), Undirected> {
        let n = data.len();
        let mut graph = Graph::new_undirected();

        // Create nodes
        let nodes: Vec<NodeIndex> = (0..n).map(|i| graph.add_node(i)).collect();

        // Check horizontal visibility between all pairs
        for i in 0..n {
            for j in (i + 1)..n {
                if self.is_visible_horizontal(data, i, j) {
                    graph.add_edge(nodes[i], nodes[j], ());
                }
            }
        }

        graph
    }

    /// Check if two points have horizontal visibility
    fn is_visible_horizontal(&self, data: &[f64], i: usize, j: usize) -> bool {
        if i >= j {
            return false;
        }

        let min_height = data[i].min(data[j]);

        // Check all intermediate points
        for k in (i + 1)..j {
            if data[k] >= min_height {
                return false; // Blocked by point k
            }
        }

        true
    }

    /// Calculate degree distribution of the graph
    fn degree_distribution(&self, graph: &Graph<usize, (), Undirected>) -> HashMap<usize, usize> {
        let mut distribution = HashMap::new();

        for node in graph.node_indices() {
            let degree = graph.neighbors(node).count();
            *distribution.entry(degree).or_insert(0) += 1;
        }

        distribution
    }

    /// Calculate average degree
    fn average_degree(&self, graph: &Graph<usize, (), Undirected>) -> f64 {
        if graph.node_count() == 0 {
            return 0.0;
        }

        let total_degree: usize = graph.node_indices()
            .map(|node| graph.neighbors(node).count())
            .sum();

        total_degree as f64 / graph.node_count() as f64
    }

    /// Calculate network density
    ///
    /// ρ = 2|E| / (N(N-1)) for undirected graphs
    fn network_density(&self, graph: &Graph<usize, (), Undirected>) -> f64 {
        let n = graph.node_count();
        if n <= 1 {
            return 0.0;
        }

        let edges = graph.edge_count();
        let max_edges = n * (n - 1) / 2;

        edges as f64 / max_edges as f64
    }

    /// Calculate network diameter (longest shortest path)
    fn network_diameter(&self, graph: &Graph<usize, (), Undirected>) -> Option<usize> {
        if graph.node_count() == 0 {
            return None;
        }

        let mut max_distance = 0;

        for node in graph.node_indices() {
            let distances = dijkstra(&graph, node, None, |_| 1);

            if let Some(&dist) = distances.values().max() {
                max_distance = max_distance.max(dist);
            }
        }

        Some(max_distance)
    }

    /// Calculate average path length
    fn average_path_length(&self, graph: &Graph<usize, (), Undirected>) -> f64 {
        if graph.node_count() <= 1 {
            return 0.0;
        }

        let mut total_distance = 0;
        let mut count = 0;

        for node in graph.node_indices() {
            let distances = dijkstra(&graph, node, None, |_| 1);

            for &dist in distances.values() {
                if dist > 0 {
                    total_distance += dist;
                    count += 1;
                }
            }
        }

        if count == 0 {
            0.0
        } else {
            total_distance as f64 / count as f64
        }
    }

    /// Calculate degree distribution statistics
    fn degree_statistics(&self, distribution: &HashMap<usize, usize>) -> (f64, f64, usize, usize) {
        if distribution.is_empty() {
            return (0.0, 0.0, 0, 0);
        }

        let total_nodes: usize = distribution.values().sum();

        // Calculate mean degree
        let mean: f64 = distribution.iter()
            .map(|(&degree, &count)| degree * count)
            .sum::<usize>() as f64 / total_nodes as f64;

        // Calculate standard deviation
        let variance: f64 = distribution.iter()
            .map(|(&degree, &count)| {
                let diff = degree as f64 - mean;
                diff * diff * count as f64
            })
            .sum::<f64>() / total_nodes as f64;
        let std_dev = variance.sqrt();

        // Find min and max degree
        let min_degree = *distribution.keys().min().unwrap_or(&0);
        let max_degree = *distribution.keys().max().unwrap_or(&0);

        (mean, std_dev, min_degree, max_degree)
    }

    /// Get the graph for external analysis
    pub fn build_graph(&self, data: &TimeSeries) -> Result<Graph<usize, (), Undirected>, StochasticError> {
        let values = data.values();

        let graph = if self.config.horizontal {
            self.build_horizontal_visibility_graph(&values)
        } else {
            self.build_natural_visibility_graph(&values)
        };

        Ok(graph)
    }

    /// Get degree sequence
    pub fn degree_sequence(&self, graph: &Graph<usize, (), Undirected>) -> Vec<usize> {
        graph.node_indices()
            .map(|node| graph.neighbors(node).count())
            .collect()
    }

    /// Calculate transitivity (global clustering coefficient)
    ///
    /// T = 3 × (number of triangles) / (number of connected triples)
    fn transitivity(&self, graph: &Graph<usize, (), Undirected>) -> f64 {
        let mut triangles = 0;
        let mut triples = 0;

        for node in graph.node_indices() {
            let neighbors: Vec<_> = graph.neighbors(node).collect();
            let degree = neighbors.len();

            if degree < 2 {
                continue;
            }

            // Count connected triples centered at this node
            triples += degree * (degree - 1) / 2;

            // Count triangles involving this node
            for i in 0..neighbors.len() {
                for j in (i + 1)..neighbors.len() {
                    if graph.contains_edge(neighbors[i], neighbors[j]) {
                        triangles += 1;
                    }
                }
            }
        }

        if triples == 0 {
            0.0
        } else {
            3.0 * triangles as f64 / triples as f64
        }
    }

    /// Number of connected components
    fn num_connected_components(&self, graph: &Graph<usize, (), Undirected>) -> usize {
        connected_components(&graph)
    }
}

impl Default for VisibilityGraphAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl StochasticAnalyzer for VisibilityGraphAnalyzer {
    fn name(&self) -> &str {
        if self.config.horizontal {
            "Horizontal Visibility Graph"
        } else {
            "Natural Visibility Graph"
        }
    }

    fn analyze(&self, data: &TimeSeries) -> Result<AnalysisResult, StochasticError> {
        self.validate(data)?;

        let values = data.values();

        // Build the visibility graph
        let graph = if self.config.horizontal {
            self.build_horizontal_visibility_graph(&values)
        } else {
            self.build_natural_visibility_graph(&values)
        };

        // Calculate network metrics
        let avg_degree = self.average_degree(&graph);
        let density = self.network_density(&graph);
        let avg_path_length = self.average_path_length(&graph);
        let transitivity = self.transitivity(&graph);
        let num_components = self.num_connected_components(&graph);

        // Calculate degree distribution
        let degree_dist = self.degree_distribution(&graph);
        let (mean_deg, std_deg, min_deg, max_deg) = self.degree_statistics(&degree_dist);

        // Optional diameter calculation
        let diameter = if self.config.calculate_diameter && graph.node_count() < 1000 {
            self.network_diameter(&graph)
        } else {
            None
        };

        // Create result
        let mut result = AnalysisResult::new(self.name())
            .with_metric("nodes", graph.node_count() as f64)
            .with_metric("edges", graph.edge_count() as f64)
            .with_metric("average_degree", avg_degree)
            .with_metric("network_density", density)
            .with_metric("average_path_length", avg_path_length)
            .with_metric("transitivity", transitivity)
            .with_metric("connected_components", num_components as f64)
            .with_metric("degree_mean", mean_deg)
            .with_metric("degree_std", std_deg)
            .with_metric("degree_min", min_deg as f64)
            .with_metric("degree_max", max_deg as f64);

        if let Some(d) = diameter {
            result = result.with_metric("diameter", d as f64);
        }

        // Add interpretation
        let interpretation = self.interpret_results(&result);
        result = result.with_interpretation(interpretation);

        Ok(result)
    }

    fn required_sample_size(&self) -> usize {
        10 // Minimum for meaningful graph
    }

    fn validate(&self, data: &TimeSeries) -> Result<bool, StochasticError> {
        if data.len() < self.required_sample_size() {
            return Err(StochasticError::InsufficientData {
                required: self.required_sample_size(),
                actual: data.len(),
            });
        }

        if data.values().iter().all(|&v| v.is_nan()) {
            return Err(StochasticError::validation("All values are NaN"));
        }

        Ok(true)
    }
}

impl VisibilityGraphAnalyzer {
    /// Interpret the analysis results
    fn interpret_results(&self, result: &AnalysisResult) -> String {
        let avg_degree = result.metrics.get("average_degree").copied().unwrap_or(0.0);
        let density = result.metrics.get("network_density").copied().unwrap_or(0.0);
        let transitivity = result.metrics.get("transitivity").copied().unwrap_or(0.0);
        let components = result.metrics.get("connected_components").copied().unwrap_or(1.0) as usize;

        let mut interp = String::new();

        // Connectivity interpretation
        if components > 1 {
            interp.push_str(&format!(
                "Network has {} disconnected components, indicating distinct regimes. ",
                components
            ));
        } else {
            interp.push_str("Network is fully connected. ");
        }

        // Density interpretation
        if density > 0.5 {
            interp.push_str("High network density suggests strong temporal correlations. ");
        } else if density < 0.1 {
            interp.push_str("Low network density indicates sparse connectivity. ");
        }

        // Degree interpretation
        if avg_degree > 10.0 {
            interp.push_str("High average degree indicates many visibility connections. ");
        } else if avg_degree < 3.0 {
            interp.push_str("Low average degree suggests limited visibility between points. ");
        }

        // Transitivity interpretation
        if transitivity > 0.3 {
            interp.push_str("High transitivity indicates clustering and local structure. ");
        }

        // General interpretation based on graph type
        if self.config.horizontal {
            interp.push_str(
                "HVG typically shows exponential degree distribution for random series, "
            );
            interp.push_str("and specific patterns for periodic or chaotic dynamics.");
        } else {
            interp.push_str(
                "NVG can reveal fractal properties through scale-free degree distributions. "
            );
            interp.push_str("Random series show exponential, periodic series show regular structure.");
        }

        interp
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_visibility_config_defaults() {
        let config = VisibilityConfig::default();
        assert!(!config.horizontal);
        assert!(!config.directed);
        assert_eq!(config.degree_bins, 20);
        assert!(config.calculate_diameter);
    }

    #[test]
    fn test_visibility_config_builders() {
        let config = VisibilityConfig::new()
            .horizontal()
            .directed()
            .with_degree_bins(30)
            .with_diameter_calculation(false);

        assert!(config.horizontal);
        assert!(config.directed);
        assert_eq!(config.degree_bins, 30);
        assert!(!config.calculate_diameter);
    }

    #[test]
    fn test_natural_visibility_simple_line() {
        // Increasing line: all points should see each other
        let data = TimeSeries::from_values(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        let analyzer = VisibilityGraphAnalyzer::new().natural();

        let result = analyzer.analyze(&data).unwrap();

        let edges = result.metrics.get("edges").unwrap();
        // In an increasing sequence, each point sees all subsequent points
        assert!(*edges > 0.0);
    }

    #[test]
    fn test_horizontal_visibility_simple() {
        let data = TimeSeries::from_values(vec![1.0, 3.0, 2.0, 4.0, 1.0, 3.0, 2.0, 4.0, 1.0, 2.0]);
        let analyzer = VisibilityGraphAnalyzer::new().horizontal();

        let result = analyzer.analyze(&data).unwrap();

        assert_eq!(result.metrics.get("nodes").unwrap(), &10.0);
        assert!(result.metrics.get("edges").unwrap() > &0.0);
    }

    #[test]
    fn test_visibility_periodic_series() {
        // Sine wave should create specific graph structure
        let n = 50;
        let data: Vec<f64> = (0..n)
            .map(|i| (i as f64 * 2.0 * std::f64::consts::PI / 10.0).sin())
            .collect();
        let ts = TimeSeries::from_values(data);

        let analyzer = VisibilityGraphAnalyzer::new();
        let result = analyzer.analyze(&ts).unwrap();

        assert_eq!(result.metrics.get("nodes").unwrap(), &(n as f64));
        assert!(result.metrics.get("average_degree").unwrap() > &2.0);
    }

    #[test]
    fn test_natural_vs_horizontal() {
        let data = TimeSeries::from_values(vec![1.0, 5.0, 3.0, 6.0, 2.0, 4.0, 3.0, 5.0, 2.0, 6.0]);

        let natural = VisibilityGraphAnalyzer::new().natural();
        let horizontal = VisibilityGraphAnalyzer::new().horizontal();

        let nat_result = natural.analyze(&data).unwrap();
        let hor_result = horizontal.analyze(&data).unwrap();

        // Natural visibility typically creates denser graphs than horizontal
        let nat_edges = nat_result.metrics.get("edges").unwrap();
        let hor_edges = hor_result.metrics.get("edges").unwrap();

        assert!(nat_edges >= hor_edges);
    }

    #[test]
    fn test_degree_distribution() {
        let data = TimeSeries::from_values(vec![1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0, 2.0]);
        let analyzer = VisibilityGraphAnalyzer::new();

        let graph = analyzer.build_graph(&data).unwrap();
        let dist = analyzer.degree_distribution(&graph);

        // Should have some degree distribution
        assert!(!dist.is_empty());

        // Sum of counts should equal number of nodes
        let total: usize = dist.values().sum();
        assert_eq!(total, 10);
    }

    #[test]
    fn test_network_density() {
        // Complete visibility (monotonic increasing)
        let data = TimeSeries::from_values(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        let analyzer = VisibilityGraphAnalyzer::new();

        let graph = analyzer.build_graph(&data).unwrap();
        let density = analyzer.network_density(&graph);

        // Monotonic series should have some connectivity
        assert!(density > 0.0);
    }

    #[test]
    fn test_average_path_length() {
        let data = TimeSeries::from_values(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let analyzer = VisibilityGraphAnalyzer::new();

        let graph = analyzer.build_graph(&data).unwrap();
        let avg_path = analyzer.average_path_length(&graph);

        // Should have reasonable path length
        assert!(avg_path > 0.0);
        assert!(avg_path < 10.0);
    }

    #[test]
    fn test_transitivity() {
        let data = TimeSeries::from_values(vec![1.0, 2.0, 3.0, 2.0, 1.0]);
        let analyzer = VisibilityGraphAnalyzer::new();

        let graph = analyzer.build_graph(&data).unwrap();
        let trans = analyzer.transitivity(&graph);

        // Transitivity should be between 0 and 1
        assert!(trans >= 0.0);
        assert!(trans <= 1.0);
    }

    #[test]
    fn test_connected_components() {
        let data = TimeSeries::from_values(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let analyzer = VisibilityGraphAnalyzer::new();

        let graph = analyzer.build_graph(&data).unwrap();
        let components = analyzer.num_connected_components(&graph);

        // Monotonic series should be fully connected
        assert_eq!(components, 1);
    }

    #[test]
    fn test_insufficient_data() {
        let data = TimeSeries::from_values(vec![1.0, 2.0]);
        let analyzer = VisibilityGraphAnalyzer::new();

        let result = analyzer.validate(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_random_series() {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let data: Vec<f64> = (0..100).map(|_| rng.gen_range(0.0..10.0)).collect();
        let ts = TimeSeries::from_values(data);

        let analyzer = VisibilityGraphAnalyzer::new();
        let result = analyzer.analyze(&ts).unwrap();

        // Random series should have specific properties
        assert_eq!(result.metrics.get("nodes").unwrap(), &100.0);
        assert!(result.metrics.get("average_degree").unwrap() > &2.0);

        // Should be connected
        assert_eq!(result.metrics.get("connected_components").unwrap(), &1.0);
    }
}
