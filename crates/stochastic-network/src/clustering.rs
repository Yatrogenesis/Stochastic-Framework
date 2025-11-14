//! Clustering and Community Detection
//!
//! Implements clustering coefficient calculations and community detection algorithms
//! for network analysis. These methods identify local and global structural patterns
//! in networks, including small-world properties and modular organization.
//!
//! # Mathematical Foundation
//!
//! ## Local Clustering Coefficient
//!
//! For a node v with degree k_v, the local clustering coefficient is:
//!
//! ```text
//! C(v) = 2e_v / [k_v(k_v - 1)]
//! ```
//!
//! where e_v is the number of edges between neighbors of v.
//! Measures the fraction of triangles that exist out of possible triangles.
//!
//! ## Average Clustering Coefficient
//!
//! ```text
//! ⟨C⟩ = (1/N) Σ_v C(v)
//! ```
//!
//! Average over all nodes with degree ≥ 2.
//!
//! ## Global Clustering (Transitivity)
//!
//! ```text
//! C_global = 3 × (number of triangles) / (number of connected triples)
//! ```
//!
//! where a connected triple is three nodes with at least two edges.
//!
//! ## Modularity
//!
//! For a partition of nodes into communities c:
//!
//! ```text
//! Q = (1/2m) Σ_ij [A_ij - k_i k_j / 2m] δ(c_i, c_j)
//! ```
//!
//! where:
//! - m = total number of edges
//! - A_ij = adjacency matrix
//! - k_i = degree of node i
//! - δ(c_i, c_j) = 1 if nodes i and j are in same community, 0 otherwise
//!
//! ## Small-World Coefficient
//!
//! ```text
//! σ = (C / C_random) / (L / L_random)
//! ```
//!
//! where C is clustering coefficient and L is average path length.
//! σ >> 1 indicates small-world properties.
//!
//! # References
//!
//! - Watts, D.J., & Strogatz, S.H. (1998). "Collective dynamics of 'small-world'
//!   networks." Nature, 393(6684), 440-442.
//! - Newman, M.E.J., & Girvan, M. (2004). "Finding and evaluating community structure
//!   in networks." Physical Review E, 69(2), 026113.
//! - Blondel, V.D., Guillaume, J.L., Lambiotte, R., & Lefebvre, E. (2008).
//!   "Fast unfolding of communities in large networks."
//!   Journal of Statistical Mechanics: Theory and Experiment, 2008(10), P10008.
//! - Fortunato, S. (2010). "Community detection in graphs."
//!   Physics Reports, 486(3-5), 75-174.
//! - Raghavan, U.N., Albert, R., & Kumara, S. (2007). "Near linear time algorithm
//!   to detect community structures in large-scale networks." Physical Review E, 76(3), 036106.
//! - Newman, M.E.J. (2006). "Modularity and community structure in networks."
//!   Proceedings of the National Academy of Sciences, 103(23), 8577-8582.

use petgraph::graph::{Graph, NodeIndex};
use petgraph::algo::dijkstra;
use petgraph::Undirected;
use std::collections::{HashMap, HashSet};
use stochastic_core::{AnalysisResult, StochasticAnalyzer, StochasticError, TimeSeries};

/// Configuration for clustering analysis
#[derive(Debug, Clone)]
pub struct ClusteringConfig {
    /// Calculate local clustering coefficients
    pub calculate_local: bool,
    /// Calculate global clustering (transitivity)
    pub calculate_global: bool,
    /// Perform community detection
    pub detect_communities: bool,
    /// Maximum iterations for community detection
    pub max_iterations: usize,
    /// Calculate small-world metrics
    pub calculate_small_world: bool,
    /// Number of random graphs for small-world comparison
    pub num_random_graphs: usize,
}

impl Default for ClusteringConfig {
    fn default() -> Self {
        Self {
            calculate_local: true,
            calculate_global: true,
            detect_communities: true,
            max_iterations: 100,
            calculate_small_world: false, // Computationally expensive
            num_random_graphs: 10,
        }
    }
}

impl ClusteringConfig {
    /// Create new configuration with defaults
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable/disable local clustering
    pub fn with_local_clustering(mut self, enable: bool) -> Self {
        self.calculate_local = enable;
        self
    }

    /// Enable/disable global clustering
    pub fn with_global_clustering(mut self, enable: bool) -> Self {
        self.calculate_global = enable;
        self
    }

    /// Enable/disable community detection
    pub fn with_community_detection(mut self, enable: bool) -> Self {
        self.detect_communities = enable;
        self
    }

    /// Set maximum iterations for community detection
    pub fn with_max_iterations(mut self, max_iter: usize) -> Self {
        self.max_iterations = max_iter.max(10);
        self
    }

    /// Enable/disable small-world calculation
    pub fn with_small_world(mut self, enable: bool) -> Self {
        self.calculate_small_world = enable;
        self
    }

    /// Set number of random graphs for comparison
    pub fn with_num_random_graphs(mut self, num: usize) -> Self {
        self.num_random_graphs = num.max(1);
        self
    }
}

/// Clustering and Community Detection Analyzer
pub struct ClusteringAnalyzer {
    config: ClusteringConfig,
}

impl ClusteringAnalyzer {
    /// Create new analyzer with default configuration
    pub fn new() -> Self {
        Self {
            config: ClusteringConfig::default(),
        }
    }

    /// Create analyzer with custom configuration
    pub fn with_config(config: ClusteringConfig) -> Self {
        Self { config }
    }

    /// Calculate local clustering coefficient for a single node
    ///
    /// C(v) = 2e_v / [k_v(k_v - 1)]
    pub fn local_clustering_coefficient(
        &self,
        graph: &Graph<usize, (), Undirected>,
        node: NodeIndex,
    ) -> f64 {
        let neighbors: Vec<_> = graph.neighbors(node).collect();
        let degree = neighbors.len();

        if degree < 2 {
            return 0.0;
        }

        // Count edges between neighbors
        let mut edges_between_neighbors = 0;
        for i in 0..neighbors.len() {
            for j in (i + 1)..neighbors.len() {
                if graph.contains_edge(neighbors[i], neighbors[j]) {
                    edges_between_neighbors += 1;
                }
            }
        }

        let max_edges = degree * (degree - 1) / 2;
        edges_between_neighbors as f64 / max_edges as f64
    }

    /// Calculate local clustering coefficients for all nodes
    pub fn local_clustering_coefficients(
        &self,
        graph: &Graph<usize, (), Undirected>,
    ) -> HashMap<NodeIndex, f64> {
        graph
            .node_indices()
            .map(|node| {
                let coeff = self.local_clustering_coefficient(graph, node);
                (node, coeff)
            })
            .collect()
    }

    /// Calculate average clustering coefficient
    ///
    /// ⟨C⟩ = (1/N) Σ C(v)
    pub fn average_clustering_coefficient(&self, graph: &Graph<usize, (), Undirected>) -> f64 {
        let coefficients = self.local_clustering_coefficients(graph);

        if coefficients.is_empty() {
            return 0.0;
        }

        // Average over nodes with degree >= 2
        let valid_coeffs: Vec<f64> = coefficients
            .iter()
            .filter(|(node, _)| graph.neighbors(**node).count() >= 2)
            .map(|(_, &coeff)| coeff)
            .collect();

        if valid_coeffs.is_empty() {
            return 0.0;
        }

        valid_coeffs.iter().sum::<f64>() / valid_coeffs.len() as f64
    }

    /// Calculate global clustering coefficient (transitivity)
    ///
    /// C_global = 3 × triangles / triples
    pub fn global_clustering_coefficient(&self, graph: &Graph<usize, (), Undirected>) -> f64 {
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

    /// Count triangles in the graph
    pub fn count_triangles(&self, graph: &Graph<usize, (), Undirected>) -> usize {
        let mut triangles = 0;

        for node in graph.node_indices() {
            let neighbors: Vec<_> = graph.neighbors(node).collect();

            for i in 0..neighbors.len() {
                for j in (i + 1)..neighbors.len() {
                    if graph.contains_edge(neighbors[i], neighbors[j]) {
                        triangles += 1;
                    }
                }
            }
        }

        triangles / 3 // Each triangle counted 3 times
    }

    /// Detect communities using label propagation algorithm
    ///
    /// Simple and fast community detection method
    pub fn label_propagation(
        &self,
        graph: &Graph<usize, (), Undirected>,
    ) -> HashMap<NodeIndex, usize> {
        let mut labels: HashMap<NodeIndex, usize> = graph
            .node_indices()
            .enumerate()
            .map(|(i, node)| (node, i))
            .collect();

        let mut changed = true;
        let mut iterations = 0;

        while changed && iterations < self.config.max_iterations {
            changed = false;
            iterations += 1;

            // Shuffle node order for better convergence
            let nodes: Vec<_> = graph.node_indices().collect();

            for &node in &nodes {
                // Count labels of neighbors
                let mut label_counts: HashMap<usize, usize> = HashMap::new();

                for neighbor in graph.neighbors(node) {
                    *label_counts.entry(labels[&neighbor]).or_insert(0) += 1;
                }

                if let Some((&most_common_label, _)) =
                    label_counts.iter().max_by_key(|(_, &count)| count)
                {
                    if labels[&node] != most_common_label {
                        labels.insert(node, most_common_label);
                        changed = true;
                    }
                }
            }
        }

        labels
    }

    /// Calculate modularity for a given community partition
    ///
    /// Q = (1/2m) Σ [A_ij - k_i k_j / 2m] δ(c_i, c_j)
    pub fn modularity(
        &self,
        graph: &Graph<usize, (), Undirected>,
        communities: &HashMap<NodeIndex, usize>,
    ) -> f64 {
        let m = graph.edge_count() as f64;
        if m == 0.0 {
            return 0.0;
        }

        let mut q = 0.0;

        for node_i in graph.node_indices() {
            let k_i = graph.neighbors(node_i).count() as f64;
            let c_i = communities[&node_i];

            for node_j in graph.node_indices() {
                let k_j = graph.neighbors(node_j).count() as f64;
                let c_j = communities[&node_j];

                if c_i != c_j {
                    continue;
                }

                let a_ij = if graph.contains_edge(node_i, node_j) {
                    1.0
                } else {
                    0.0
                };

                q += a_ij - (k_i * k_j) / (2.0 * m);
            }
        }

        q / (2.0 * m)
    }

    /// Get number of communities in a partition
    pub fn num_communities(&self, communities: &HashMap<NodeIndex, usize>) -> usize {
        let unique_labels: HashSet<_> = communities.values().collect();
        unique_labels.len()
    }

    /// Get community sizes
    pub fn community_sizes(&self, communities: &HashMap<NodeIndex, usize>) -> HashMap<usize, usize> {
        let mut sizes = HashMap::new();

        for &community in communities.values() {
            *sizes.entry(community).or_insert(0) += 1;
        }

        sizes
    }

    /// Calculate average path length of the graph
    pub fn average_path_length(&self, graph: &Graph<usize, (), Undirected>) -> f64 {
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

    /// Calculate small-world coefficient
    ///
    /// σ = (C / C_random) / (L / L_random)
    pub fn small_world_coefficient(
        &self,
        graph: &Graph<usize, (), Undirected>,
    ) -> Option<f64> {
        let c = self.average_clustering_coefficient(graph);
        let l = self.average_path_length(graph);

        if l == 0.0 {
            return None;
        }

        // Generate random graphs with same degree sequence
        let mut c_random_sum = 0.0;
        let mut l_random_sum = 0.0;

        for _ in 0..self.config.num_random_graphs {
            if let Some(random_graph) = self.generate_random_graph(graph) {
                c_random_sum += self.average_clustering_coefficient(&random_graph);
                l_random_sum += self.average_path_length(&random_graph);
            }
        }

        let c_random = c_random_sum / self.config.num_random_graphs as f64;
        let l_random = l_random_sum / self.config.num_random_graphs as f64;

        if c_random == 0.0 || l_random == 0.0 {
            return None;
        }

        let sigma = (c / c_random) / (l / l_random);
        Some(sigma)
    }

    /// Generate random graph with same number of nodes and edges
    ///
    /// Simple Erdős-Rényi random graph
    fn generate_random_graph(
        &self,
        original: &Graph<usize, (), Undirected>,
    ) -> Option<Graph<usize, (), Undirected>> {
        use rand::Rng;

        let n = original.node_count();
        let m = original.edge_count();

        if n == 0 {
            return None;
        }

        let mut graph = Graph::new_undirected();
        let nodes: Vec<_> = (0..n).map(|i| graph.add_node(i)).collect();

        let mut rng = rand::thread_rng();
        let mut edges_added = 0;

        // Add random edges
        while edges_added < m {
            let i = rng.gen_range(0..n);
            let j = rng.gen_range(0..n);

            if i != j && !graph.contains_edge(nodes[i], nodes[j]) {
                graph.add_edge(nodes[i], nodes[j], ());
                edges_added += 1;
            }
        }

        Some(graph)
    }

    /// Build a graph from time series for analysis
    pub fn build_sample_graph(&self, data: &TimeSeries) -> Graph<usize, (), Undirected> {
        let values = data.values();
        let n = values.len();
        let mut graph = Graph::new_undirected();

        // Create nodes
        let nodes: Vec<NodeIndex> = (0..n).map(|i| graph.add_node(i)).collect();

        // Connect adjacent points
        for i in 0..(n - 1) {
            graph.add_edge(nodes[i], nodes[i + 1], ());
        }

        // Add long-range connections based on similarity
        let threshold = data.statistics().std_dev * 0.5;

        for i in 0..n {
            for j in (i + 2)..n.min(i + 10) {
                let diff = (values[i] - values[j]).abs();
                if diff < threshold {
                    graph.add_edge(nodes[i], nodes[j], ());
                }
            }
        }

        graph
    }

    /// Calculate clustering distribution
    pub fn clustering_distribution(
        &self,
        graph: &Graph<usize, (), Undirected>,
    ) -> Vec<f64> {
        self.local_clustering_coefficients(graph)
            .values()
            .copied()
            .collect()
    }
}

impl Default for ClusteringAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl StochasticAnalyzer for ClusteringAnalyzer {
    fn name(&self) -> &str {
        "Network Clustering Analysis"
    }

    fn analyze(&self, data: &TimeSeries) -> Result<AnalysisResult, StochasticError> {
        self.validate(data)?;

        // Build graph from time series
        let graph = self.build_sample_graph(data);

        let mut result = AnalysisResult::new(self.name())
            .with_metric("nodes", graph.node_count() as f64)
            .with_metric("edges", graph.edge_count() as f64);

        // Local clustering
        if self.config.calculate_local {
            let avg_clustering = self.average_clustering_coefficient(&graph);
            result = result.with_metric("average_clustering_coefficient", avg_clustering);

            let clustering_coeffs = self.local_clustering_coefficients(&graph);
            let max_clustering = clustering_coeffs.values()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            result = result.with_metric("max_clustering_coefficient", max_clustering);
        }

        // Global clustering
        if self.config.calculate_global {
            let global_clustering = self.global_clustering_coefficient(&graph);
            let triangles = self.count_triangles(&graph);

            result = result
                .with_metric("global_clustering_coefficient", global_clustering)
                .with_metric("triangles", triangles as f64);
        }

        // Community detection
        if self.config.detect_communities {
            let communities = self.label_propagation(&graph);
            let modularity = self.modularity(&graph, &communities);
            let num_communities = self.num_communities(&communities);

            result = result
                .with_metric("modularity", modularity)
                .with_metric("num_communities", num_communities as f64);

            let sizes = self.community_sizes(&communities);
            if let Some(&largest_community) = sizes.values().max() {
                result = result.with_metric("largest_community_size", largest_community as f64);
            }
        }

        // Small-world metrics
        if self.config.calculate_small_world && graph.node_count() < 100 {
            if let Some(sigma) = self.small_world_coefficient(&graph) {
                result = result.with_metric("small_world_coefficient", sigma);
            }
        }

        // Additional metrics
        let avg_path_length = self.average_path_length(&graph);
        result = result.with_metric("average_path_length", avg_path_length);

        // Interpretation
        let interpretation = self.interpret_results(&result);
        result = result.with_interpretation(interpretation);

        Ok(result)
    }

    fn required_sample_size(&self) -> usize {
        5 // Minimum for meaningful clustering
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

impl ClusteringAnalyzer {
    /// Interpret the analysis results
    fn interpret_results(&self, result: &AnalysisResult) -> String {
        let mut interp = String::new();

        // Clustering interpretation
        if let Some(&avg_clustering) = result.metrics.get("average_clustering_coefficient") {
            if avg_clustering > 0.6 {
                interp.push_str("High clustering coefficient indicates strong local connectivity and transitivity. ");
            } else if avg_clustering < 0.2 {
                interp.push_str("Low clustering suggests sparse local structure. ");
            } else {
                interp.push_str("Moderate clustering indicates balanced local and global connectivity. ");
            }
        }

        // Modularity interpretation
        if let Some(&modularity) = result.metrics.get("modularity") {
            if modularity > 0.3 {
                interp.push_str("High modularity indicates clear community structure. ");
            } else if modularity < 0.1 {
                interp.push_str("Low modularity suggests weak or no community structure. ");
            }

            if let Some(&num_comm) = result.metrics.get("num_communities") {
                interp.push_str(&format!(
                    "Network partitioned into {} communities. ",
                    num_comm as usize
                ));
            }
        }

        // Small-world interpretation
        if let Some(&sigma) = result.metrics.get("small_world_coefficient") {
            if sigma > 1.5 {
                interp.push_str("Small-world properties detected: high clustering with short paths. ");
            }
        }

        interp.push_str(
            "Clustering measures local cohesion; communities reveal modular organization."
        );

        interp
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_clustering_config_defaults() {
        let config = ClusteringConfig::default();
        assert!(config.calculate_local);
        assert!(config.calculate_global);
        assert!(config.detect_communities);
        assert_eq!(config.max_iterations, 100);
    }

    #[test]
    fn test_local_clustering_triangle() {
        // Triangle graph: all nodes should have C = 1.0
        let mut graph = Graph::new_undirected();
        let n0 = graph.add_node(0);
        let n1 = graph.add_node(1);
        let n2 = graph.add_node(2);

        graph.add_edge(n0, n1, ());
        graph.add_edge(n1, n2, ());
        graph.add_edge(n2, n0, ());

        let analyzer = ClusteringAnalyzer::new();

        assert_relative_eq!(
            analyzer.local_clustering_coefficient(&graph, n0),
            1.0,
            epsilon = 1e-6
        );
        assert_relative_eq!(
            analyzer.local_clustering_coefficient(&graph, n1),
            1.0,
            epsilon = 1e-6
        );
        assert_relative_eq!(
            analyzer.local_clustering_coefficient(&graph, n2),
            1.0,
            epsilon = 1e-6
        );
    }

    #[test]
    fn test_local_clustering_star() {
        // Star graph: center has C = 0, leaves have C = 0
        let mut graph = Graph::new_undirected();
        let center = graph.add_node(0);
        let mut leaves = Vec::new();

        for i in 1..5 {
            let node = graph.add_node(i);
            graph.add_edge(center, node, ());
            leaves.push(node);
        }

        let analyzer = ClusteringAnalyzer::new();

        assert_relative_eq!(
            analyzer.local_clustering_coefficient(&graph, center),
            0.0,
            epsilon = 1e-6
        );

        for &leaf in &leaves {
            // Leaves have degree 1, so C = 0
            assert_relative_eq!(
                analyzer.local_clustering_coefficient(&graph, leaf),
                0.0,
                epsilon = 1e-6
            );
        }
    }

    #[test]
    fn test_average_clustering_coefficient() {
        let mut graph = Graph::new_undirected();
        let nodes: Vec<_> = (0..4).map(|i| graph.add_node(i)).collect();

        // Create a graph with one triangle and one isolated edge
        graph.add_edge(nodes[0], nodes[1], ());
        graph.add_edge(nodes[1], nodes[2], ());
        graph.add_edge(nodes[2], nodes[0], ());

        let analyzer = ClusteringAnalyzer::new();
        let avg_c = analyzer.average_clustering_coefficient(&graph);

        // Triangle nodes all have C = 1.0
        assert_relative_eq!(avg_c, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_global_clustering_coefficient() {
        // Triangle graph
        let mut graph = Graph::new_undirected();
        let n0 = graph.add_node(0);
        let n1 = graph.add_node(1);
        let n2 = graph.add_node(2);

        graph.add_edge(n0, n1, ());
        graph.add_edge(n1, n2, ());
        graph.add_edge(n2, n0, ());

        let analyzer = ClusteringAnalyzer::new();
        let global_c = analyzer.global_clustering_coefficient(&graph);

        // Triangle should have high global clustering
        // The formula is: 3 * triangles / triples
        // For a triangle: 3 * 1 / 3 = 1.0 (since there are 3 connected triples)
        assert!(global_c > 0.9);
    }

    #[test]
    fn test_count_triangles() {
        // Two triangles sharing an edge
        let mut graph = Graph::new_undirected();
        let nodes: Vec<_> = (0..4).map(|i| graph.add_node(i)).collect();

        // Triangle 1: 0-1-2
        graph.add_edge(nodes[0], nodes[1], ());
        graph.add_edge(nodes[1], nodes[2], ());
        graph.add_edge(nodes[2], nodes[0], ());

        // Triangle 2: 1-2-3
        graph.add_edge(nodes[1], nodes[3], ());
        graph.add_edge(nodes[2], nodes[3], ());

        let analyzer = ClusteringAnalyzer::new();
        let triangles = analyzer.count_triangles(&graph);

        assert_eq!(triangles, 2);
    }

    #[test]
    fn test_label_propagation_complete_graph() {
        // Complete graph should converge to single community
        let mut graph = Graph::new_undirected();
        let nodes: Vec<_> = (0..5).map(|i| graph.add_node(i)).collect();

        for i in 0..5 {
            for j in (i + 1)..5 {
                graph.add_edge(nodes[i], nodes[j], ());
            }
        }

        let analyzer = ClusteringAnalyzer::new();
        let communities = analyzer.label_propagation(&graph);
        let num_communities = analyzer.num_communities(&communities);

        // Should converge to 1 community
        assert_eq!(num_communities, 1);
    }

    #[test]
    fn test_label_propagation_disconnected() {
        // Two disconnected triangles
        let mut graph = Graph::new_undirected();

        // Triangle 1
        let t1_n0 = graph.add_node(0);
        let t1_n1 = graph.add_node(1);
        let t1_n2 = graph.add_node(2);
        graph.add_edge(t1_n0, t1_n1, ());
        graph.add_edge(t1_n1, t1_n2, ());
        graph.add_edge(t1_n2, t1_n0, ());

        // Triangle 2
        let t2_n0 = graph.add_node(3);
        let t2_n1 = graph.add_node(4);
        let t2_n2 = graph.add_node(5);
        graph.add_edge(t2_n0, t2_n1, ());
        graph.add_edge(t2_n1, t2_n2, ());
        graph.add_edge(t2_n2, t2_n0, ());

        let analyzer = ClusteringAnalyzer::new();
        let communities = analyzer.label_propagation(&graph);
        let num_communities = analyzer.num_communities(&communities);

        // Should detect 2 communities
        assert_eq!(num_communities, 2);
    }

    #[test]
    fn test_modularity_single_community() {
        let mut graph = Graph::new_undirected();
        let nodes: Vec<_> = (0..4).map(|i| graph.add_node(i)).collect();

        graph.add_edge(nodes[0], nodes[1], ());
        graph.add_edge(nodes[1], nodes[2], ());
        graph.add_edge(nodes[2], nodes[3], ());

        // All nodes in same community
        let communities: HashMap<_, _> = nodes.iter().map(|&n| (n, 0)).collect();

        let analyzer = ClusteringAnalyzer::new();
        let modularity = analyzer.modularity(&graph, &communities);

        // Single community should have low modularity
        assert!(modularity < 0.5);
    }

    #[test]
    fn test_num_communities() {
        let mut communities = HashMap::new();
        communities.insert(NodeIndex::new(0), 0);
        communities.insert(NodeIndex::new(1), 0);
        communities.insert(NodeIndex::new(2), 1);
        communities.insert(NodeIndex::new(3), 1);
        communities.insert(NodeIndex::new(4), 2);

        let analyzer = ClusteringAnalyzer::new();
        let num = analyzer.num_communities(&communities);

        assert_eq!(num, 3);
    }

    #[test]
    fn test_community_sizes() {
        let mut communities = HashMap::new();
        communities.insert(NodeIndex::new(0), 0);
        communities.insert(NodeIndex::new(1), 0);
        communities.insert(NodeIndex::new(2), 0);
        communities.insert(NodeIndex::new(3), 1);
        communities.insert(NodeIndex::new(4), 1);

        let analyzer = ClusteringAnalyzer::new();
        let sizes = analyzer.community_sizes(&communities);

        assert_eq!(sizes[&0], 3);
        assert_eq!(sizes[&1], 2);
    }

    #[test]
    fn test_average_path_length() {
        // Line graph: 0-1-2-3
        let mut graph = Graph::new_undirected();
        let nodes: Vec<_> = (0..4).map(|i| graph.add_node(i)).collect();

        for i in 0..3 {
            graph.add_edge(nodes[i], nodes[i + 1], ());
        }

        let analyzer = ClusteringAnalyzer::new();
        let avg_path = analyzer.average_path_length(&graph);

        // Average should be between 1 and 3
        assert!(avg_path >= 1.0);
        assert!(avg_path <= 3.0);
    }

    #[test]
    fn test_analyzer_with_timeseries() {
        let data = TimeSeries::from_values(vec![1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0]);
        let analyzer = ClusteringAnalyzer::new();

        let result = analyzer.analyze(&data).unwrap();

        assert!(result.metrics.contains_key("average_clustering_coefficient"));
        assert!(result.metrics.contains_key("global_clustering_coefficient"));
        assert!(result.metrics.contains_key("modularity"));
    }

    #[test]
    fn test_insufficient_data() {
        let data = TimeSeries::from_values(vec![1.0, 2.0]);
        let analyzer = ClusteringAnalyzer::new();

        let result = analyzer.validate(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_clustering_distribution() {
        let mut graph = Graph::new_undirected();
        let nodes: Vec<_> = (0..4).map(|i| graph.add_node(i)).collect();

        // Triangle plus one node
        graph.add_edge(nodes[0], nodes[1], ());
        graph.add_edge(nodes[1], nodes[2], ());
        graph.add_edge(nodes[2], nodes[0], ());
        graph.add_edge(nodes[0], nodes[3], ());

        let analyzer = ClusteringAnalyzer::new();
        let dist = analyzer.clustering_distribution(&graph);

        assert_eq!(dist.len(), 4);
    }

    #[test]
    fn test_generate_random_graph() {
        let mut original = Graph::new_undirected();
        let nodes: Vec<_> = (0..5).map(|i| original.add_node(i)).collect();

        original.add_edge(nodes[0], nodes[1], ());
        original.add_edge(nodes[1], nodes[2], ());
        original.add_edge(nodes[2], nodes[3], ());

        let analyzer = ClusteringAnalyzer::new();
        let random = analyzer.generate_random_graph(&original);

        assert!(random.is_some());
        let random_graph = random.unwrap();
        assert_eq!(random_graph.node_count(), original.node_count());
        assert_eq!(random_graph.edge_count(), original.edge_count());
    }
}
