//! # Range Equilibrium Analysis Framework
//!
//! Statistical tests for assessing balance and uniformity in numerical sequences.
//! Implements partition-based chi-squared tests and parity analysis for detecting
//! deviations from equilibrium distributions.
//!
//! ## Modules
//!
//! - **partition_balance**: Range partitioning and chi-squared balance tests
//! - **parity_test**: Even/odd parity distribution analysis
//!
//! ## References
//!
//! - NIST SP 800-22 Rev. 1a (2010). "A Statistical Test Suite for Random Number Generators"
//! - Knuth, D.E. (1998). "The Art of Computer Programming, Vol. 2: Seminumerical Algorithms"
//! - Marsaglia, G. (1996). "DIEHARD: A Battery of Tests of Randomness"

pub mod partition_balance;
pub mod parity_test;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

pub use partition_balance::{PartitionBalanceAnalyzer, PartitionConfig};
pub use parity_test::{ParityTestAnalyzer, ParityConfig};
