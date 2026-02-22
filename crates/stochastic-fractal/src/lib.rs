//! Multifractal analysis for STOCHASTIC framework
//!
//! Implements Detrended Fluctuation Analysis (DFA) and Hurst exponent calculation.

pub mod dfa;

pub use dfa::DFA;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
