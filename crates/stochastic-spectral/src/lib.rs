//! Spectral Analysis Framework
//!
//! Implements frequency-domain analysis techniques for time series data.
//!
//! # Modules
//! - `fft`: Fast Fourier Transform analysis
//! - `psd`: Power Spectral Density estimation
//! - `coherence`: Coherence analysis between signals

pub mod fft;
pub mod psd;
pub mod coherence;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

// Re-export key types
pub use fft::{FFTAnalyzer, FFTConfig};
pub use psd::{PSDAnalyzer, PSDConfig, PSDMethod};
pub use coherence::{CoherenceAnalyzer, CoherenceConfig};
