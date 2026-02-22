//! Information theory for STOCHASTIC framework
//!
//! Implements Shannon Entropy, Mutual Information, and Kolmogorov Complexity approximation.

pub mod shannon_entropy;
pub mod mutual_information;
pub mod kolmogorov_complexity;

pub use shannon_entropy::ShannonEntropy;
pub use mutual_information::MutualInformation;
pub use kolmogorov_complexity::KolmogorovComplexity;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
