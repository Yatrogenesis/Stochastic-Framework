//! Galois Field Theory Framework
//!
//! Implements finite field arithmetic and quadratic residue analysis for number-theoretic
//! properties in random sequences. Provides tools for analyzing modular arithmetic patterns
//! and distribution properties in finite fields GF(p).

pub mod finite_field;
pub mod quadratic_residues;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

pub use finite_field::{FiniteFieldAnalyzer, FiniteFieldConfig};
pub use quadratic_residues::{QuadraticResidueAnalyzer, QuadraticResidueConfig};
