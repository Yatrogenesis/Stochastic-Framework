//! # Quantum Information Framework
//!
//! Implements quantum information theory measures for analyzing system complexity,
//! integration, and information content. Based on density matrix formalism and
//! Integrated Information Theory (IIT).
//!
//! ## Modules
//!
//! - **density_matrix**: Density matrix ρ construction and analysis
//! - **von_neumann_entropy**: Von Neumann entropy S(ρ) = -Tr(ρ log ρ)
//! - **integrated_information**: Integrated Information Φ (Tononi 2004)
//!
//! ## References
//!
//! - Nielsen, M.A., & Chuang, I.L. (2000). "Quantum Computation and Quantum Information"
//! - Tononi, G. (2004). "An information integration theory of consciousness"
//! - Wilde, M.M. (2013). "Quantum Information Theory"
//! - Oizumi, M., et al. (2014). "From the phenomenology to the mechanisms of consciousness"
//! - Tononi, G., et al. (2016). "Integrated information theory: from consciousness to its physical substrate"
//! - Albantakis, L., et al. (2023). "Integrated information theory (IIT) 4.0"

pub mod density_matrix;
pub mod von_neumann_entropy;
pub mod integrated_information;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

pub use density_matrix::{DensityMatrix, DensityConfig};
pub use von_neumann_entropy::{VonNeumannEntropy, VNConfig};
pub use integrated_information::{IntegratedInformation, IITConfig};
