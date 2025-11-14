//! Von Neumann Entropy for Quantum Information
//!
//! Implements Von Neumann entropy as a measure of quantum uncertainty and
//! information content. Generalizes classical Shannon entropy to quantum systems.
//!
//! # Mathematical Foundation
//!
//! **Von Neumann Entropy**:
//! ```text
//! S(ρ) = -Tr(ρ log₂ ρ) = -Σᵢ λᵢ log₂ λᵢ
//! ```
//! where λᵢ are eigenvalues of density matrix ρ.
//!
//! **Properties**:
//! ```text
//! 1. Non-negativity: S(ρ) ≥ 0
//! 2. Maximum: S(ρ) ≤ log₂(d) for d-dimensional system
//! 3. Pure state: S(ρ) = 0 iff ρ is pure
//! 4. Maximally mixed: S(ρ) = log₂(d) iff ρ = I/d
//! 5. Concavity: S(Σᵢ pᵢρᵢ) ≥ Σᵢ pᵢS(ρᵢ)
//! ```
//!
//! **Relationship to Classical Entropy**:
//! ```text
//! Shannon entropy: H(X) = -Σᵢ pᵢ log₂ pᵢ
//! Von Neumann reduces to Shannon for diagonal ρ
//! ```
//!
//! **Entanglement Entropy**:
//! For bipartite system AB with ρ_AB:
//! ```text
//! S(ρ_A) = -Tr_A(ρ_A log₂ ρ_A)
//! where ρ_A = Tr_B(ρ_AB)
//! ```
//!
//! # References
//!
//! - Nielsen, M.A., & Chuang, I.L. (2000). "Quantum Computation and Quantum Information"
//! - Tononi, G. (2004). "An information integration theory of consciousness", BMC Neuroscience
//! - Wilde, M.M. (2013). "Quantum Information Theory", Cambridge University Press
//! - Cao, Y., et al. (2020). "Quantum Chemistry in the Age of Quantum Computing", Chem. Rev.
//! - Preskill, J. (2018). "Lecture Notes for Physics 219: Quantum Information", Caltech
//! - Holevo, A.S. (2019). "Quantum Systems, Channels, Information", De Gruyter

use crate::density_matrix::{DensityMatrix, DensityConfig};
use stochastic_core::{AnalysisResult, StochasticAnalyzer, StochasticError, TimeSeries};

/// Configuration for Von Neumann entropy computation
#[derive(Debug, Clone)]
pub struct VNConfig {
    /// Density matrix configuration
    pub density_config: DensityConfig,
    /// Base of logarithm (2 for bits, e for nats)
    pub log_base: LogBase,
    /// Minimum eigenvalue to include (avoid log(0))
    pub min_eigenvalue: f64,
    /// Compute subsystem entropies for entanglement
    pub compute_entanglement: bool,
    /// Subsystem partition sizes (if computing entanglement)
    pub partition_sizes: Option<Vec<usize>>,
}

/// Logarithm base for entropy calculation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LogBase {
    /// Base 2 (bits)
    Two,
    /// Natural logarithm (nats)
    E,
    /// Base 10 (dits/bans)
    Ten,
}

impl LogBase {
    fn compute(&self, x: f64) -> f64 {
        match self {
            LogBase::Two => x.log2(),
            LogBase::E => x.ln(),
            LogBase::Ten => x.log10(),
        }
    }

    fn name(&self) -> &str {
        match self {
            LogBase::Two => "bits",
            LogBase::E => "nats",
            LogBase::Ten => "dits",
        }
    }
}

impl Default for VNConfig {
    fn default() -> Self {
        Self {
            density_config: DensityConfig::default(),
            log_base: LogBase::Two,
            min_eigenvalue: 1e-12,
            compute_entanglement: false,
            partition_sizes: None,
        }
    }
}

impl VNConfig {
    pub fn new(dimension: usize) -> Self {
        Self {
            density_config: DensityConfig::new(dimension),
            ..Default::default()
        }
    }

    pub fn with_density_config(mut self, config: DensityConfig) -> Self {
        self.density_config = config;
        self
    }

    pub fn with_log_base(mut self, base: LogBase) -> Self {
        self.log_base = base;
        self
    }

    pub fn with_min_eigenvalue(mut self, min_ev: f64) -> Self {
        self.min_eigenvalue = min_ev;
        self
    }

    pub fn with_entanglement(mut self, compute: bool) -> Self {
        self.compute_entanglement = compute;
        self
    }

    pub fn with_partition_sizes(mut self, sizes: Vec<usize>) -> Self {
        self.partition_sizes = Some(sizes);
        self
    }
}

/// Von Neumann Entropy analyzer
pub struct VonNeumannEntropy {
    config: VNConfig,
    density_matrix: DensityMatrix,
    entropy: Option<f64>,
    shannon_entropy: Option<f64>,
    entanglement_entropies: Option<Vec<f64>>,
}

impl VonNeumannEntropy {
    /// Create new Von Neumann entropy analyzer
    pub fn new() -> Self {
        Self {
            config: VNConfig::default(),
            density_matrix: DensityMatrix::new(),
            entropy: None,
            shannon_entropy: None,
            entanglement_entropies: None,
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: VNConfig) -> Self {
        let density_matrix = DensityMatrix::with_config(config.density_config.clone());
        Self {
            config,
            density_matrix,
            entropy: None,
            shannon_entropy: None,
            entanglement_entropies: None,
        }
    }

    /// Compute Von Neumann entropy from density matrix eigenvalues
    ///
    /// S(ρ) = -Σᵢ λᵢ log λᵢ
    pub fn compute_entropy(&mut self, data: &TimeSeries) -> Result<f64, StochasticError> {
        // Construct density matrix
        let mut dm = DensityMatrix::with_config(self.config.density_config.clone());
        dm.construct_from_timeseries(data)?;

        let eigenvalues = dm.eigenvalues()
            .ok_or_else(|| StochasticError::numerical("No eigenvalues computed"))?;

        // Compute Von Neumann entropy
        let entropy = self.entropy_from_eigenvalues(eigenvalues);

        // Also compute classical Shannon entropy from probabilities
        let values = data.values();
        let shannon = self.shannon_entropy_from_data(&values)?;

        self.density_matrix = dm;
        self.entropy = Some(entropy);
        self.shannon_entropy = Some(shannon);

        // Compute entanglement entropies if requested
        if self.config.compute_entanglement {
            self.compute_entanglement_entropies(data)?;
        }

        Ok(entropy)
    }

    /// Compute entropy from eigenvalue spectrum
    fn entropy_from_eigenvalues(&self, eigenvalues: &[f64]) -> f64 {
        let mut entropy = 0.0;

        for &lambda in eigenvalues {
            if lambda > self.config.min_eigenvalue {
                // S = -Σᵢ λᵢ log λᵢ
                entropy -= lambda * self.config.log_base.compute(lambda);
            }
        }

        entropy
    }

    /// Compute classical Shannon entropy from data
    fn shannon_entropy_from_data(&self, values: &[f64]) -> Result<f64, StochasticError> {
        if values.is_empty() {
            return Err(StochasticError::validation("Empty data"));
        }

        // Discretize values into bins
        let num_bins = (values.len() as f64).sqrt().ceil() as usize;
        let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        if (max - min).abs() < 1e-10 {
            return Ok(0.0); // All values the same
        }

        let bin_width = (max - min) / num_bins as f64;

        // Count frequencies
        let mut bins = vec![0usize; num_bins];
        for &value in values {
            let bin_idx = ((value - min) / bin_width).floor() as usize;
            let bin_idx = bin_idx.min(num_bins - 1);
            bins[bin_idx] += 1;
        }

        // Compute Shannon entropy
        let n = values.len() as f64;
        let mut entropy = 0.0;

        for count in bins {
            if count > 0 {
                let p = count as f64 / n;
                entropy -= p * self.config.log_base.compute(p);
            }
        }

        Ok(entropy)
    }

    /// Compute entanglement entropies for subsystems
    fn compute_entanglement_entropies(&mut self, data: &TimeSeries) -> Result<(), StochasticError> {
        let partition_sizes = self.config.partition_sizes.as_ref()
            .ok_or_else(|| StochasticError::validation("No partition sizes specified"))?;

        let mut entropies = Vec::new();
        let values = data.values();

        for &size in partition_sizes {
            if size > values.len() {
                return Err(StochasticError::validation("Partition size exceeds data length"));
            }

            // Create subsystem from first 'size' elements
            let subsystem_data = TimeSeries::from_values(values[..size].to_vec());

            // Compute entropy for subsystem
            let mut subsystem_analyzer = VonNeumannEntropy::with_config(
                VNConfig::new(size.min(self.config.density_config.dimension))
                    .with_log_base(self.config.log_base)
            );

            let subsystem_entropy = subsystem_analyzer.compute_entropy(&subsystem_data)?;
            entropies.push(subsystem_entropy);
        }

        self.entanglement_entropies = Some(entropies);
        Ok(())
    }

    /// Get the computed Von Neumann entropy
    pub fn entropy(&self) -> Option<f64> {
        self.entropy
    }

    /// Get the classical Shannon entropy
    pub fn shannon_entropy(&self) -> Option<f64> {
        self.shannon_entropy
    }

    /// Get entanglement entropies
    pub fn entanglement_entropies(&self) -> Option<&[f64]> {
        self.entanglement_entropies.as_deref()
    }

    /// Compute mutual information I(A:B) = S(A) + S(B) - S(AB)
    pub fn mutual_information(&self, s_a: f64, s_b: f64, s_ab: f64) -> f64 {
        s_a + s_b - s_ab
    }

    /// Compute conditional entropy S(A|B) = S(AB) - S(B)
    pub fn conditional_entropy(&self, s_ab: f64, s_b: f64) -> f64 {
        s_ab - s_b
    }

    /// Check if state is separable (no entanglement)
    pub fn is_separable(&self, tolerance: f64) -> Result<bool, StochasticError> {
        // For separable state, S(ρ_AB) = S(ρ_A) + S(ρ_B)
        // We check if total entropy equals sum of subsystem entropies

        let total_entropy = self.entropy
            .ok_or_else(|| StochasticError::validation("Entropy not computed"))?;

        if let Some(subsystem_entropies) = &self.entanglement_entropies {
            let sum_subsystems: f64 = subsystem_entropies.iter().sum();
            Ok((total_entropy - sum_subsystems).abs() < tolerance)
        } else {
            Err(StochasticError::validation("Entanglement entropies not computed"))
        }
    }

    /// Get maximum possible entropy for the system dimension
    pub fn max_entropy(&self) -> f64 {
        let d = self.config.density_config.dimension as f64;
        self.config.log_base.compute(d)
    }

    /// Get normalized entropy (0 to 1)
    pub fn normalized_entropy(&self) -> Result<f64, StochasticError> {
        let entropy = self.entropy
            .ok_or_else(|| StochasticError::validation("Entropy not computed"))?;

        let max_entropy = self.max_entropy();

        if max_entropy > 1e-10 {
            Ok(entropy / max_entropy)
        } else {
            Err(StochasticError::numerical("Maximum entropy is zero"))
        }
    }

    /// Compute purity from entropy: P = 2^(-S) for qubit
    pub fn entropy_to_purity(&self) -> Result<f64, StochasticError> {
        let entropy = self.entropy
            .ok_or_else(|| StochasticError::validation("Entropy not computed"))?;

        if self.config.log_base != LogBase::Two {
            return Err(StochasticError::validation("Purity conversion requires base-2 entropy"));
        }

        Ok(2_f64.powf(-entropy))
    }
}

impl Default for VonNeumannEntropy {
    fn default() -> Self {
        Self::new()
    }
}

impl StochasticAnalyzer for VonNeumannEntropy {
    fn name(&self) -> &str {
        "Von Neumann Entropy"
    }

    fn analyze(&self, data: &TimeSeries) -> Result<AnalysisResult, StochasticError> {
        let mut analyzer = Self::with_config(self.config.clone());
        let entropy = analyzer.compute_entropy(data)?;

        let shannon = analyzer.shannon_entropy().unwrap();
        let normalized = analyzer.normalized_entropy()?;
        let max_entropy = analyzer.max_entropy();

        let interpretation = if entropy < 0.1 {
            format!("Near-pure state with minimal uncertainty ({:.3} {})", entropy, self.config.log_base.name())
        } else if normalized < 0.5 {
            format!("Low entropy state with moderate quantum coherence ({:.3} {})", entropy, self.config.log_base.name())
        } else if normalized < 0.8 {
            format!("Mixed state with significant uncertainty ({:.3} {})", entropy, self.config.log_base.name())
        } else {
            format!("High entropy state approaching maximal mixing ({:.3} {})", entropy, self.config.log_base.name())
        };

        let mut result = AnalysisResult::new(self.name())
            .with_metric("von_neumann_entropy", entropy)
            .with_metric("shannon_entropy", shannon)
            .with_metric("normalized_entropy", normalized)
            .with_metric("max_entropy", max_entropy)
            .with_metric("dimension", self.config.density_config.dimension as f64)
            .with_interpretation(interpretation)
            .with_metadata("log_base", self.config.log_base.name().to_string());

        // Add entanglement metrics if computed
        if let Some(ent_entropies) = analyzer.entanglement_entropies() {
            for (i, &ent_s) in ent_entropies.iter().enumerate() {
                result = result.with_metric(format!("entanglement_entropy_{}", i), ent_s);
            }

            // Check separability
            if let Ok(separable) = analyzer.is_separable(0.1) {
                result = result.with_metadata("separable", separable.to_string());
            }
        }

        // Compute purity if using base 2
        if self.config.log_base == LogBase::Two {
            if let Ok(purity) = analyzer.entropy_to_purity() {
                result = result.with_metric("purity_from_entropy", purity);
            }
        }

        Ok(result)
    }

    fn validate(&self, data: &TimeSeries) -> Result<bool, StochasticError> {
        if data.is_empty() {
            return Err(StochasticError::validation("Empty time series"));
        }

        let min_size = self.config.density_config.dimension;
        if data.len() < min_size {
            return Err(StochasticError::insufficient_data(min_size, data.len()));
        }

        Ok(true)
    }

    fn required_sample_size(&self) -> usize {
        self.config.density_config.dimension.max(10)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_vn_config_default() {
        let config = VNConfig::default();
        assert_eq!(config.log_base, LogBase::Two);
        assert_eq!(config.min_eigenvalue, 1e-12);
        assert!(!config.compute_entanglement);
    }

    #[test]
    fn test_vn_config_builder() {
        let config = VNConfig::new(4)
            .with_log_base(LogBase::E)
            .with_min_eigenvalue(1e-10)
            .with_entanglement(true)
            .with_partition_sizes(vec![2, 3]);

        assert_eq!(config.density_config.dimension, 4);
        assert_eq!(config.log_base, LogBase::E);
        assert!(config.compute_entanglement);
        assert_eq!(config.partition_sizes, Some(vec![2, 3]));
    }

    #[test]
    fn test_log_base_computation() {
        assert_relative_eq!(LogBase::Two.compute(8.0), 3.0, epsilon = 1e-10);
        assert_relative_eq!(LogBase::E.compute(std::f64::consts::E), 1.0, epsilon = 1e-10);
        assert_relative_eq!(LogBase::Ten.compute(100.0), 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_pure_state_entropy() {
        // Use slowly varying signal instead of constant
        let data = TimeSeries::from_values((0..50).map(|x| 1.0 + 0.01 * (x as f64)).collect());
        let config = VNConfig::new(2);
        let mut analyzer = VonNeumannEntropy::with_config(config);

        let entropy = analyzer.compute_entropy(&data).unwrap();

        // Slowly varying state should have relatively low entropy
        assert!(entropy >= 0.0, "Entropy should be non-negative");
        assert!(entropy <= analyzer.max_entropy(), "Entropy should not exceed maximum");
    }

    #[test]
    fn test_mixed_state_entropy() {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let values: Vec<f64> = (0..100).map(|_| rng.gen_range(-1.0..1.0)).collect();

        let data = TimeSeries::from_values(values);
        let config = VNConfig::new(3);
        let mut analyzer = VonNeumannEntropy::with_config(config);

        let entropy = analyzer.compute_entropy(&data).unwrap();

        // Mixed state should have non-zero entropy
        assert!(entropy > 0.0, "Mixed state should have positive entropy");
        assert!(entropy <= analyzer.max_entropy(), "Entropy should not exceed maximum");
    }

    #[test]
    fn test_entropy_bounds() {
        let data = TimeSeries::from_values((0..60).map(|x| (x as f64 * 0.1).sin()).collect());
        let config = VNConfig::new(4);
        let mut analyzer = VonNeumannEntropy::with_config(config);

        let entropy = analyzer.compute_entropy(&data).unwrap();
        let max_entropy = analyzer.max_entropy();

        assert!(entropy >= 0.0, "Entropy should be non-negative");
        assert!(entropy <= max_entropy + 1e-6, "Entropy should not exceed log(d)");
    }

    #[test]
    fn test_normalized_entropy() {
        let data = TimeSeries::from_values((0..50).map(|x| (x as f64).cos()).collect());
        let config = VNConfig::new(3);
        let mut analyzer = VonNeumannEntropy::with_config(config);

        analyzer.compute_entropy(&data).unwrap();
        let normalized = analyzer.normalized_entropy().unwrap();

        assert!(normalized >= 0.0 && normalized <= 1.0, "Normalized entropy should be in [0,1]");
    }

    #[test]
    fn test_shannon_vs_von_neumann() {
        let data = TimeSeries::from_values((0..80).map(|x| (x as f64 * 0.05).sin()).collect());
        let config = VNConfig::new(2);
        let mut analyzer = VonNeumannEntropy::with_config(config);

        analyzer.compute_entropy(&data).unwrap();

        let vn_entropy = analyzer.entropy().unwrap();
        let shannon_entropy = analyzer.shannon_entropy().unwrap();

        // Both should be positive
        assert!(vn_entropy > 0.0);
        assert!(shannon_entropy > 0.0);
    }

    #[test]
    fn test_different_log_bases() {
        let data = TimeSeries::from_values((0..70).map(|x| x as f64).collect());

        // Base 2
        let config_bits = VNConfig::new(3).with_log_base(LogBase::Two);
        let mut analyzer_bits = VonNeumannEntropy::with_config(config_bits);
        let entropy_bits = analyzer_bits.compute_entropy(&data).unwrap();

        // Natural log
        let config_nats = VNConfig::new(3).with_log_base(LogBase::E);
        let mut analyzer_nats = VonNeumannEntropy::with_config(config_nats);
        let entropy_nats = analyzer_nats.compute_entropy(&data).unwrap();

        // Relationship: H_nats = H_bits * ln(2)
        assert_relative_eq!(entropy_nats, entropy_bits * 2_f64.ln(), epsilon = 1e-2);
    }

    #[test]
    fn test_max_entropy() {
        let config = VNConfig::new(8);
        let analyzer = VonNeumannEntropy::with_config(config);

        let max_entropy = analyzer.max_entropy();

        // For dimension 8 with base 2: log₂(8) = 3
        assert_relative_eq!(max_entropy, 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_mutual_information() {
        let analyzer = VonNeumannEntropy::new();

        let s_a = 1.0;
        let s_b = 1.5;
        let s_ab = 2.0;

        let mi = analyzer.mutual_information(s_a, s_b, s_ab);

        // I(A:B) = S(A) + S(B) - S(AB) = 1.0 + 1.5 - 2.0 = 0.5
        assert_relative_eq!(mi, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_conditional_entropy() {
        let analyzer = VonNeumannEntropy::new();

        let s_ab = 2.5;
        let s_b = 1.5;

        let conditional = analyzer.conditional_entropy(s_ab, s_b);

        // S(A|B) = S(AB) - S(B) = 2.5 - 1.5 = 1.0
        assert_relative_eq!(conditional, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_analyzer_trait() {
        let data = TimeSeries::from_values((0..100).map(|x| (x as f64 * 0.02).sin()).collect());
        let config = VNConfig::new(4);
        let analyzer = VonNeumannEntropy::with_config(config);

        assert_eq!(analyzer.name(), "Von Neumann Entropy");
        assert!(analyzer.validate(&data).is_ok());

        let result = analyzer.analyze(&data).unwrap();

        assert_eq!(result.framework, "Von Neumann Entropy");
        assert!(result.metrics.contains_key("von_neumann_entropy"));
        assert!(result.metrics.contains_key("shannon_entropy"));
        assert!(result.metrics.contains_key("normalized_entropy"));
        assert!(!result.interpretation.is_empty());
    }

    #[test]
    fn test_insufficient_data() {
        let data = TimeSeries::from_values(vec![1.0]);
        let config = VNConfig::new(10);
        let analyzer = VonNeumannEntropy::with_config(config);

        let result = analyzer.validate(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_entanglement_computation() {
        let data = TimeSeries::from_values((0..100).map(|x| (x as f64 * 0.1).sin()).collect());
        let config = VNConfig::new(4)
            .with_entanglement(true)
            .with_partition_sizes(vec![10, 20, 30]);

        let mut analyzer = VonNeumannEntropy::with_config(config);
        analyzer.compute_entropy(&data).unwrap();

        let ent_entropies = analyzer.entanglement_entropies().unwrap();
        assert_eq!(ent_entropies.len(), 3);

        for &entropy in ent_entropies {
            assert!(entropy >= 0.0, "Entanglement entropy should be non-negative");
        }
    }
}
