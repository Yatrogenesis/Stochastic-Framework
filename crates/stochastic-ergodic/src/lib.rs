//! # Ergodic Theory Framework
//!
//! Implements ergodic analysis methods for time series including the Birkhoff-Khinchin
//! ergodic theorem and mixing properties. Tests whether time averages converge to
//! ensemble averages and analyzes temporal correlation decay.
//!
//! ## Modules
//!
//! - **birkhoff**: Birkhoff-Khinchin ergodic theorem and time average convergence
//! - **mixing**: Mixing properties, autocorrelation, and temporal independence
//!
//! ## References
//!
//! - Birkhoff, G.D. (1931). "Proof of the Ergodic Theorem." Proc. Natl. Acad. Sci. USA
//! - Walters, P. (1982). "An Introduction to Ergodic Theory." Springer
//! - Petersen, K. (1983). "Ergodic Theory." Cambridge University Press
//! - Bradley, R.C. (2005). "Basic Properties of Strong Mixing Conditions." Progress in Probability
//! - Rosenblatt, M. (1956). "A Central Limit Theorem and a Strong Mixing Condition"

pub mod birkhoff;
pub mod mixing;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

pub use birkhoff::{BirkhoffAnalyzer, BirkhoffConfig};
pub use mixing::{MixingAnalyzer, MixingConfig};
