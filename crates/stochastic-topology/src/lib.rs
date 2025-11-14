//! # Topological Data Analysis Framework
//!
//! This crate provides topological data analysis (TDA) tools for the STOCHASTIC framework,
//! enabling the study of shape and structure in complex datasets through algebraic topology.
//!
//! ## Modules
//!
//! - **persistent_homology**: Persistent homology computation using filtrations
//! - **betti_numbers**: Betti number analysis for topological features
//! - **vietoris_rips**: Vietoris-Rips complex construction
//!
//! ## Overview
//!
//! Topological Data Analysis provides methods to extract global geometric and topological
//! features from data. The key insight is that the "shape" of data often reveals important
//! structural properties that traditional statistics might miss.
//!
//! ### Key Concepts
//!
//! - **Persistent Homology**: Tracks topological features across multiple scales
//! - **Betti Numbers**: Count topological features (components, loops, voids)
//! - **Simplicial Complexes**: Build topological representations from point clouds
//!
//! ## References
//!
//! - Carlsson, G. (2009). "Topology and data." Bulletin of the AMS, 46(2), 255-308.
//! - Edelsbrunner, H., & Harer, J. (2008). "Computational Topology: An Introduction."
//! - Chazal, F., & Michel, B. (2021). "An introduction to Topological Data Analysis."
//!
//! ## Example
//!
//! ```ignore
//! use stochastic_topology::{PersistentHomology, PersistenceConfig};
//! use stochastic_core::TimeSeries;
//!
//! let data = TimeSeries::from_values(vec![1.0, 2.0, 3.0, 2.0, 1.0]);
//! let analyzer = PersistentHomology::new();
//! let result = analyzer.analyze(&data)?;
//! ```

pub mod persistent_homology;
pub mod betti_numbers;
pub mod vietoris_rips;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

// Re-export main types for convenience
pub use persistent_homology::{PersistentHomology, PersistenceConfig, PersistenceDiagram, BirthDeathPair};
pub use betti_numbers::{BettiAnalyzer, BettiConfig, BettiNumbers};
pub use vietoris_rips::{VietorisRips, VRConfig, SimplexType};

/// Get version information
pub fn version() -> &'static str {
    VERSION
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        let ver = version();
        assert!(!ver.is_empty());
        assert_eq!(ver, VERSION);
    }
}
