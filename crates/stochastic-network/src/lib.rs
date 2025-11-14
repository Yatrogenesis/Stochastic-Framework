//! # Network Theory Framework
//!
//! Advanced network analysis for time series data using graph theory and complex
//! network methods. This framework converts time series into networks and analyzes
//! their topological properties, centrality measures, and community structure.
//!
//! ## Modules
//!
//! ### Visibility Graphs
//!
//! - **Natural Visibility Graph (NVG)**: Converts time series to networks based on
//!   geometric visibility between data points. Reveals fractal and chaotic properties.
//! - **Horizontal Visibility Graph (HVG)**: Simpler visibility criterion, computationally
//!   faster, with well-understood theoretical properties for random and periodic series.
//!
//! ### Centrality Measures
//!
//! Identifies important nodes in networks:
//! - **Degree Centrality**: Based on number of connections
//! - **Closeness Centrality**: Based on distance to other nodes
//! - **Betweenness Centrality**: Based on control of information flow
//! - **Eigenvector Centrality**: Based on influence (connections to influential nodes)
//!
//! ### Clustering and Communities
//!
//! Detects local and global structural patterns:
//! - **Clustering Coefficient**: Measures local transitivity and triangle formation
//! - **Community Detection**: Identifies modular organization using label propagation
//! - **Modularity**: Quantifies quality of community structure
//! - **Small-World Properties**: Detects networks with high clustering and short paths
//!
//! ## Mathematical Foundation
//!
//! ### Visibility Graph Construction
//!
//! **Natural Visibility**: Points (tᵢ, yᵢ) and (tⱼ, yⱼ) are connected if:
//! ```text
//! ∀k ∈ (i,j): y(tₖ) < y(tⱼ) + (y(tᵢ) - y(tⱼ)) × (tₖ - tⱼ) / (tᵢ - tⱼ)
//! ```
//!
//! **Horizontal Visibility**: Points are connected if:
//! ```text
//! ∀k ∈ (i,j): y(tₖ) < min(y(tᵢ), y(tⱼ))
//! ```
//!
//! ### Network Metrics
//!
//! **Degree Distribution**: P(k) characterizes network topology
//! - Random series → exponential P(k)
//! - Fractal series → power-law P(k)
//! - Periodic series → delta function P(k)
//!
//! **Centrality Measures**:
//! ```text
//! Degree:      C_D(v) = k(v) / (N - 1)
//! Closeness:   C_C(v) = (N - 1) / Σ_u d(v,u)
//! Betweenness: C_B(v) = Σ_{s≠v≠t} σ_st(v) / σ_st
//! Eigenvector: x_v = (1/λ) Σ_u A_vu x_u
//! ```
//!
//! **Clustering**:
//! ```text
//! Local:  C(v) = 2e_v / [k_v(k_v-1)]
//! Global: C = 3 × triangles / triples
//! ```
//!
//! **Modularity**:
//! ```text
//! Q = (1/2m) Σ_ij [A_ij - k_i k_j / 2m] δ(c_i, c_j)
//! ```
//!
//! ## Examples
//!
//! ### Basic Visibility Graph Analysis
//!
//! ```no_run
//! use stochastic_network::{VisibilityGraphAnalyzer, VisibilityConfig};
//! use stochastic_core::{TimeSeries, StochasticAnalyzer};
//!
//! // Create time series
//! let data = TimeSeries::from_values(vec![1.0, 3.0, 2.0, 4.0, 1.0, 5.0, 2.0, 3.0, 1.0, 4.0]);
//!
//! // Analyze with natural visibility graph
//! let analyzer = VisibilityGraphAnalyzer::new();
//! let result = analyzer.analyze(&data).unwrap();
//!
//! println!("Network has {} nodes and {} edges",
//!          result.metrics["nodes"], result.metrics["edges"]);
//! println!("Average degree: {}", result.metrics["average_degree"]);
//! ```
//!
//! ### Horizontal Visibility Graph
//!
//! ```no_run
//! use stochastic_network::VisibilityGraphAnalyzer;
//! use stochastic_core::{TimeSeries, StochasticAnalyzer};
//!
//! let data = TimeSeries::from_values(vec![1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
//!
//! // Use horizontal visibility (faster)
//! let analyzer = VisibilityGraphAnalyzer::new().horizontal();
//! let result = analyzer.analyze(&data).unwrap();
//!
//! println!("Horizontal VG density: {}", result.metrics["network_density"]);
//! ```
//!
//! ### Centrality Analysis
//!
//! ```no_run
//! use stochastic_network::CentralityAnalyzer;
//! use stochastic_core::{TimeSeries, StochasticAnalyzer};
//!
//! let data = TimeSeries::from_values(vec![1.0, 5.0, 3.0, 7.0, 2.0, 6.0, 4.0, 8.0, 3.0, 5.0]);
//!
//! let analyzer = CentralityAnalyzer::new().with_top_k(3);
//! let result = analyzer.analyze(&data).unwrap();
//!
//! println!("Degree centrality: {}", result.metrics["degree_centrality_avg"]);
//! println!("Closeness centrality: {}", result.metrics["closeness_centrality_avg"]);
//! ```
//!
//! ### Community Detection
//!
//! ```no_run
//! use stochastic_network::ClusteringAnalyzer;
//! use stochastic_core::{TimeSeries, StochasticAnalyzer};
//!
//! let data = TimeSeries::from_values(
//!     (0..50).map(|i| (i as f64 * 0.1).sin()).collect()
//! );
//!
//! let analyzer = ClusteringAnalyzer::new();
//! let result = analyzer.analyze(&data).unwrap();
//!
//! println!("Clustering coefficient: {}",
//!          result.metrics["average_clustering_coefficient"]);
//! println!("Number of communities: {}", result.metrics["num_communities"]);
//! println!("Modularity: {}", result.metrics["modularity"]);
//! ```
//!
//! ## Key References
//!
//! ### Visibility Graphs
//! - Lacasa et al. (2008). "From time series to complex networks: The visibility graph."
//!   PNAS, 105(13), 4972-4975.
//! - Luque et al. (2009). "Horizontal visibility graphs." Physical Review E, 80(4), 046103.
//! - Zou et al. (2019). "Complex network approaches to nonlinear time series analysis."
//!   Physics Reports, 787, 1-97.
//!
//! ### Centrality
//! - Freeman (1977). "A set of measures of centrality based on betweenness." Sociometry, 40(1), 35-41.
//! - Brandes (2001). "A faster algorithm for betweenness centrality."
//!   Journal of Mathematical Sociology, 25(2), 163-177.
//! - Newman (2010). "Networks: An Introduction." Oxford University Press.
//!
//! ### Clustering and Communities
//! - Watts & Strogatz (1998). "Collective dynamics of 'small-world' networks." Nature, 393, 440-442.
//! - Newman & Girvan (2004). "Finding and evaluating community structure."
//!   Physical Review E, 69(2), 026113.
//! - Fortunato (2010). "Community detection in graphs." Physics Reports, 486(3-5), 75-174.
//!
//! ## Applications
//!
//! - **Financial Markets**: Detect market regimes and critical transitions
//! - **Climate Data**: Identify climate network patterns and teleconnections
//! - **Physiological Signals**: Analyze heart rate variability and brain networks
//! - **Seismic Activity**: Characterize earthquake sequences through network topology
//! - **Turbulence**: Study intermittency and cascade processes

pub mod visibility_graph;
pub mod centrality;
pub mod clustering;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

// Re-export main types and analyzers
pub use visibility_graph::{VisibilityGraphAnalyzer, VisibilityConfig};
pub use centrality::{CentralityAnalyzer, CentralityConfig};
pub use clustering::{ClusteringAnalyzer, ClusteringConfig};

/// Get version information
pub fn version() -> &'static str {
    VERSION
}

#[cfg(test)]
mod tests {
    use super::*;
    use stochastic_core::{StochasticAnalyzer, TimeSeries};

    #[test]
    fn test_version() {
        let ver = version();
        assert!(!ver.is_empty());
        assert_eq!(ver, VERSION);
    }

    #[test]
    fn test_visibility_graph_integration() {
        let data = TimeSeries::from_values(vec![1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0]);

        // Natural visibility
        let nat_analyzer = VisibilityGraphAnalyzer::new();
        let nat_result = nat_analyzer.analyze(&data).unwrap();
        assert!(nat_result.metrics.contains_key("nodes"));
        assert!(nat_result.metrics.contains_key("edges"));
        assert_eq!(nat_result.metrics["nodes"], 10.0);

        // Horizontal visibility
        let hor_analyzer = VisibilityGraphAnalyzer::new().horizontal();
        let hor_result = hor_analyzer.analyze(&data).unwrap();
        assert!(hor_result.metrics.contains_key("network_density"));
    }

    #[test]
    fn test_centrality_integration() {
        let data = TimeSeries::from_values(vec![1.0, 5.0, 3.0, 7.0, 2.0, 6.0, 4.0, 8.0]);

        let analyzer = CentralityAnalyzer::new();
        let result = analyzer.analyze(&data).unwrap();

        assert!(result.metrics.contains_key("degree_centrality_avg"));
        assert!(result.metrics.contains_key("closeness_centrality_avg"));
        assert!(result.metrics.contains_key("eigenvector_centrality_avg"));
    }

    #[test]
    fn test_clustering_integration() {
        let data = TimeSeries::from_values(vec![
            1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0,
        ]);

        let analyzer = ClusteringAnalyzer::new();
        let result = analyzer.analyze(&data).unwrap();

        assert!(result.metrics.contains_key("average_clustering_coefficient"));
        assert!(result.metrics.contains_key("global_clustering_coefficient"));
        assert!(result.metrics.contains_key("modularity"));
        assert!(result.metrics.contains_key("num_communities"));
    }

    #[test]
    fn test_all_analyzers_with_same_data() {
        let data = TimeSeries::from_values((0..20).map(|i| (i as f64 * 0.3).sin()).collect());

        // Visibility graph
        let vg_analyzer = VisibilityGraphAnalyzer::new();
        let vg_result = vg_analyzer.analyze(&data);
        assert!(vg_result.is_ok());

        // Centrality
        let cent_analyzer = CentralityAnalyzer::new();
        let cent_result = cent_analyzer.analyze(&data);
        assert!(cent_result.is_ok());

        // Clustering
        let clust_analyzer = ClusteringAnalyzer::new();
        let clust_result = clust_analyzer.analyze(&data);
        assert!(clust_result.is_ok());
    }

    #[test]
    fn test_periodic_series() {
        // Sine wave - periodic series
        let data: Vec<f64> = (0..30)
            .map(|i| (i as f64 * 2.0 * std::f64::consts::PI / 10.0).sin())
            .collect();
        let ts = TimeSeries::from_values(data);

        let analyzer = VisibilityGraphAnalyzer::new();
        let result = analyzer.analyze(&ts).unwrap();

        // Periodic series should create specific network structure
        assert!(result.metrics["average_degree"] > 2.0);
        assert!(result.metrics["network_density"] > 0.0);
    }

    #[test]
    fn test_random_series() {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let data: Vec<f64> = (0..50).map(|_| rng.gen_range(0.0..10.0)).collect();
        let ts = TimeSeries::from_values(data);

        // All analyzers should work with random data
        let vg = VisibilityGraphAnalyzer::new().analyze(&ts);
        let cent = CentralityAnalyzer::new().analyze(&ts);
        let clust = ClusteringAnalyzer::new().analyze(&ts);

        assert!(vg.is_ok());
        assert!(cent.is_ok());
        assert!(clust.is_ok());
    }

    #[test]
    fn test_analyzer_names() {
        let vg = VisibilityGraphAnalyzer::new();
        assert_eq!(vg.name(), "Natural Visibility Graph");

        let hvg = VisibilityGraphAnalyzer::new().horizontal();
        assert_eq!(hvg.name(), "Horizontal Visibility Graph");

        let cent = CentralityAnalyzer::new();
        assert_eq!(cent.name(), "Network Centrality Analysis");

        let clust = ClusteringAnalyzer::new();
        assert_eq!(clust.name(), "Network Clustering Analysis");
    }

    #[test]
    fn test_required_sample_sizes() {
        let vg = VisibilityGraphAnalyzer::new();
        assert_eq!(vg.required_sample_size(), 10);

        let cent = CentralityAnalyzer::new();
        assert_eq!(cent.required_sample_size(), 5);

        let clust = ClusteringAnalyzer::new();
        assert_eq!(clust.required_sample_size(), 5);
    }
}
