//! Frequentist statistical analysis for STOCHASTIC framework
//!
//! Implements chi-squared test, Kolmogorov-Smirnov test, and Anderson-Darling test.

pub mod chi_squared;
pub mod kolmogorov_smirnov;

pub use chi_squared::ChiSquaredTest;
pub use kolmogorov_smirnov::KolmogorovSmirnovTest;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
