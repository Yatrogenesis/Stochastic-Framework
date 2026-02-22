//! Chaos Theory Framework
//!
//! Implements nonlinear dynamics and chaos theory methods for time series analysis,
//! including Lyapunov exponents, Takens embedding, and correlation dimension.

pub mod lyapunov;
pub mod takens;
pub mod correlation_dimension;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

pub use lyapunov::{LyapunovAnalyzer, LyapunovConfig};
pub use takens::{TakensEmbedding, TakensConfig};
pub use correlation_dimension::{CorrelationDimensionAnalyzer, CorrelationConfig};
