//! Bayesian inference for STOCHASTIC framework
//!
//! Implements Dirichlet-Multinomial model and Metropolis-Hastings MCMC sampler.

pub mod dirichlet_multinomial;
pub mod metropolis_hastings;

pub use dirichlet_multinomial::DirichletMultinomial;
pub use metropolis_hastings::{MetropolisHastings, TargetDistribution, MCMCResult};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
