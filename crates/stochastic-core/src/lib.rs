//! # STOCHASTIC Core
//!
//! Core types, traits, and error handling for the STOCHASTIC framework.
//!
//! This crate provides the fundamental building blocks used by all analysis
//! frameworks in the STOCHASTIC ecosystem.
//!
//! ## Main Components
//!
//! - **Types**: `TimeSeries`, `DataPoint`, `AnalysisResult`, `AnalysisConfig`
//! - **Traits**: `StochasticAnalyzer`, `IncrementalAnalyzer`, `ConfigurableAnalyzer`
//! - **Errors**: `StochasticError` and `Result` type alias
//!
//! ## Example
//!
//! ```
//! use stochastic_core::{TimeSeries, AnalysisConfig};
//!
//! // Create a time series from values
//! let data = TimeSeries::from_values(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
//!
//! // Get basic statistics
//! let stats = data.statistics();
//! println!("Mean: {}, Std Dev: {}", stats.mean, stats.std_dev);
//!
//! // Configure analysis parameters
//! let config = AnalysisConfig::new()
//!     .with_confidence_level(0.95)
//!     .with_bootstrap_iterations(10000);
//! ```

pub mod error;
pub mod traits;
pub mod types;

// Re-export main types for convenience
pub use error::{Result, StochasticError};
pub use traits::{
    ConfigurableAnalyzer, IncrementalAnalyzer, ParallelAnalyzer, PlotType, StochasticAnalyzer,
    VisualizationData, Visualizable,
};
pub use types::{
    AnalysisConfig, AnalysisResult, DataPoint, Domain, Statistics, TimeSeries,
};

/// Version of the stochastic-core crate
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

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

    #[test]
    fn test_timeseries_creation() {
        let ts = TimeSeries::from_values(vec![1.0, 2.0, 3.0]);
        assert_eq!(ts.len(), 3);
        assert!(!ts.is_empty());
    }

    #[test]
    fn test_analysis_result_creation() {
        let result = AnalysisResult::new("Test")
            .with_metric("test_metric", 42.0)
            .with_p_value(0.05);

        assert_eq!(result.framework, "Test");
        assert_eq!(result.p_value, Some(0.05));
    }

    #[test]
    fn test_config_defaults() {
        let config = AnalysisConfig::default();
        assert_eq!(config.confidence_level, 0.95);
        assert_eq!(config.bootstrap_iterations, 10000);
        assert_eq!(config.max_iterations, 1000);
    }
}
