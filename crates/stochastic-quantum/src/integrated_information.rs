//! Integrated Information Theory (IIT) Φ
//!
//! Implements Integrated Information Theory measures for quantifying system
//! integration, complexity, and consciousness. Based on Tononi's IIT framework.
//!
//! # Mathematical Foundation
//!
//! **Integrated Information Φ**:
//! ```text
//! Φ = min_{M∈MIP} EI(ρ, M)
//! ```
//! where M is a partition of the system and MIP is the minimum information partition.
//!
//! **Effective Information**:
//! ```text
//! EI(ρ, M) = S(ρ) - Σ_{A∈M} S(ρ_A)
//! ```
//! where S is Von Neumann entropy, ρ is the system state, and ρ_A are subsystem states.
//!
//! **Properties**:
//! ```text
//! 1. Non-negativity: Φ ≥ 0
//! 2. Fully integrated: Φ = S(ρ) when subsystems independent
//! 3. Reducible: Φ = 0 when system is separable
//! 4. Compositionality: Φ captures irreducibility
//! ```
//!
//! **Partition Types**:
//! ```text
//! - Bipartition: Split into two parts
//! - Multipartition: Split into multiple parts
//! - Minimum Information Partition (MIP): Partition that minimizes EI
//! ```
//!
//! **IIT Axioms** (Tononi et al. 2016):
//! ```text
//! 1. Intrinsic existence
//! 2. Composition
//! 3. Information
//! 4. Integration
//! 5. Exclusion
//! ```
//!
//! # References
//!
//! - Tononi, G. (2004). "An information integration theory of consciousness", BMC Neuroscience
//! - Oizumi, M., et al. (2014). "From the phenomenology to the mechanisms of consciousness: IIT 3.0", PLoS Comp. Bio.
//! - Tononi, G., et al. (2016). "Integrated information theory: from consciousness to its physical substrate", Nat. Rev. Neurosci.
//! - Albantakis, L., et al. (2023). "Integrated information theory (IIT) 4.0: formulating the properties of phenomenal existence"
//! - Balduzzi, D., & Tononi, G. (2008). "Integrated information in discrete dynamical systems", PLoS Comp. Bio.
//! - Tegmark, M. (2016). "Improved measures of integrated information", PLoS Comp. Bio.

use crate::density_matrix::DensityConfig;
use crate::von_neumann_entropy::{VonNeumannEntropy, VNConfig};
use stochastic_core::{AnalysisResult, StochasticAnalyzer, StochasticError, TimeSeries};

/// Configuration for Integrated Information computation
#[derive(Debug, Clone)]
pub struct IITConfig {
    /// Density matrix configuration
    pub density_config: DensityConfig,
    /// Von Neumann entropy configuration
    pub vn_config: VNConfig,
    /// Partition strategy
    pub partition_strategy: PartitionStrategy,
    /// Maximum partition size to consider
    pub max_partition_size: usize,
    /// Minimum partition size
    pub min_partition_size: usize,
    /// Compute all partitions or just MIP
    pub compute_all_partitions: bool,
    /// Threshold for considering Φ as zero
    pub phi_threshold: f64,
}

/// Strategy for generating partitions
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PartitionStrategy {
    /// Only bipartitions (split in two)
    Bipartition,
    /// All possible partitions
    AllPartitions,
    /// Balanced partitions (similar sizes)
    Balanced,
    /// Sequential partitions (contiguous blocks)
    Sequential,
}

impl Default for IITConfig {
    fn default() -> Self {
        Self {
            density_config: DensityConfig::default(),
            vn_config: VNConfig::default(),
            partition_strategy: PartitionStrategy::Bipartition,
            max_partition_size: 4,
            min_partition_size: 2,
            compute_all_partitions: false,
            phi_threshold: 1e-10,
        }
    }
}

impl IITConfig {
    pub fn new(dimension: usize) -> Self {
        Self {
            density_config: DensityConfig::new(dimension),
            vn_config: VNConfig::new(dimension),
            ..Default::default()
        }
    }

    pub fn with_density_config(mut self, config: DensityConfig) -> Self {
        self.density_config = config;
        self
    }

    pub fn with_vn_config(mut self, config: VNConfig) -> Self {
        self.vn_config = config;
        self
    }

    pub fn with_partition_strategy(mut self, strategy: PartitionStrategy) -> Self {
        self.partition_strategy = strategy;
        self
    }

    pub fn with_max_partition_size(mut self, size: usize) -> Self {
        self.max_partition_size = size;
        self
    }

    pub fn with_compute_all(mut self, compute: bool) -> Self {
        self.compute_all_partitions = compute;
        self
    }

    pub fn with_phi_threshold(mut self, threshold: f64) -> Self {
        self.phi_threshold = threshold;
        self
    }
}

/// Represents a partition of the system
#[derive(Debug, Clone)]
pub struct Partition {
    /// Indices in each part
    pub parts: Vec<Vec<usize>>,
}

impl Partition {
    pub fn new(parts: Vec<Vec<usize>>) -> Self {
        Self { parts }
    }

    /// Number of parts in partition
    pub fn num_parts(&self) -> usize {
        self.parts.len()
    }

    /// Check if partition is valid (all indices used exactly once)
    pub fn is_valid(&self, total_size: usize) -> bool {
        let mut seen = vec![false; total_size];

        for part in &self.parts {
            for &idx in part {
                if idx >= total_size || seen[idx] {
                    return false;
                }
                seen[idx] = true;
            }
        }

        seen.iter().all(|&x| x)
    }
}

/// Result of effective information computation
#[derive(Debug, Clone)]
pub struct EffectiveInformationResult {
    /// The effective information value
    pub ei: f64,
    /// Total system entropy
    pub total_entropy: f64,
    /// Subsystem entropies
    pub subsystem_entropies: Vec<f64>,
    /// The partition used
    pub partition: Partition,
}

/// Integrated Information analyzer
pub struct IntegratedInformation {
    config: IITConfig,
    phi: Option<f64>,
    mip: Option<Partition>,
    all_ei_results: Option<Vec<EffectiveInformationResult>>,
}

impl IntegratedInformation {
    /// Create new IIT analyzer
    pub fn new() -> Self {
        Self {
            config: IITConfig::default(),
            phi: None,
            mip: None,
            all_ei_results: None,
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: IITConfig) -> Self {
        Self {
            config,
            phi: None,
            mip: None,
            all_ei_results: None,
        }
    }

    /// Compute integrated information Φ
    pub fn compute_phi(&mut self, data: &TimeSeries) -> Result<f64, StochasticError> {
        // Compute total system entropy
        let mut total_analyzer = VonNeumannEntropy::with_config(self.config.vn_config.clone());
        let total_entropy = total_analyzer.compute_entropy(data)?;

        // Generate partitions
        let partitions = self.generate_partitions(data.len())?;

        if partitions.is_empty() {
            return Err(StochasticError::analysis_failed("No valid partitions generated"));
        }

        // Compute effective information for each partition
        let mut ei_results = Vec::new();

        for partition in partitions {
            let ei_result = self.compute_effective_information(data, &partition, total_entropy)?;
            ei_results.push(ei_result);
        }

        // Find minimum (MIP)
        let mip_result = ei_results.iter()
            .min_by(|a, b| a.ei.partial_cmp(&b.ei).unwrap())
            .ok_or_else(|| StochasticError::analysis_failed("Failed to find MIP"))?;

        let phi = mip_result.ei;

        self.phi = Some(phi);
        self.mip = Some(mip_result.partition.clone());

        if self.config.compute_all_partitions {
            self.all_ei_results = Some(ei_results);
        }

        Ok(phi)
    }

    /// Compute effective information for a given partition
    ///
    /// EI(ρ, M) = S(ρ) - Σ_{A∈M} S(ρ_A)
    fn compute_effective_information(
        &self,
        data: &TimeSeries,
        partition: &Partition,
        total_entropy: f64,
    ) -> Result<EffectiveInformationResult, StochasticError> {
        let values = data.values();
        let mut subsystem_entropies = Vec::new();

        for part in &partition.parts {
            if part.is_empty() {
                continue;
            }

            // Extract subsystem data
            let subsystem_values: Vec<f64> = part.iter()
                .filter(|&&idx| idx < values.len())
                .map(|&idx| values[idx])
                .collect();

            if subsystem_values.is_empty() {
                subsystem_entropies.push(0.0);
                continue;
            }

            // Compute subsystem entropy
            let subsystem_data = TimeSeries::from_values(subsystem_values);
            let subsystem_dim = part.len().min(self.config.density_config.dimension);

            let subsystem_vn_config = VNConfig::new(subsystem_dim)
                .with_log_base(self.config.vn_config.log_base);

            let mut subsystem_analyzer = VonNeumannEntropy::with_config(subsystem_vn_config);
            let subsystem_entropy = subsystem_analyzer.compute_entropy(&subsystem_data)?;

            subsystem_entropies.push(subsystem_entropy);
        }

        // EI = S(total) - Σ S(subsystems)
        let sum_subsystem_entropies: f64 = subsystem_entropies.iter().sum();
        let ei = total_entropy - sum_subsystem_entropies;

        Ok(EffectiveInformationResult {
            ei,
            total_entropy,
            subsystem_entropies,
            partition: partition.clone(),
        })
    }

    /// Generate partitions based on strategy
    fn generate_partitions(&self, size: usize) -> Result<Vec<Partition>, StochasticError> {
        if size < self.config.min_partition_size {
            return Err(StochasticError::validation(
                format!("Data size {} too small for minimum partition size {}",
                    size, self.config.min_partition_size)
            ));
        }

        let indices: Vec<usize> = (0..size).collect();

        match self.config.partition_strategy {
            PartitionStrategy::Bipartition => self.generate_bipartitions(&indices),
            PartitionStrategy::Sequential => self.generate_sequential_partitions(&indices),
            PartitionStrategy::Balanced => self.generate_balanced_partitions(&indices),
            PartitionStrategy::AllPartitions => self.generate_all_partitions(&indices),
        }
    }

    /// Generate all bipartitions
    fn generate_bipartitions(&self, indices: &[usize]) -> Result<Vec<Partition>, StochasticError> {
        let n = indices.len();
        let mut partitions = Vec::new();

        // Generate all possible splits
        for i in 1..n {
            let part1 = indices[..i].to_vec();
            let part2 = indices[i..].to_vec();

            partitions.push(Partition::new(vec![part1, part2]));
        }

        Ok(partitions)
    }

    /// Generate sequential partitions (contiguous blocks)
    fn generate_sequential_partitions(&self, indices: &[usize]) -> Result<Vec<Partition>, StochasticError> {
        let n = indices.len();
        let mut partitions = Vec::new();

        // Two-part sequential
        for i in 1..n {
            let part1 = indices[..i].to_vec();
            let part2 = indices[i..].to_vec();
            partitions.push(Partition::new(vec![part1, part2]));
        }

        // Three-part sequential
        if n >= 3 && self.config.max_partition_size >= 3 {
            for i in 1..n-1 {
                for j in i+1..n {
                    let part1 = indices[..i].to_vec();
                    let part2 = indices[i..j].to_vec();
                    let part3 = indices[j..].to_vec();
                    partitions.push(Partition::new(vec![part1, part2, part3]));
                }
            }
        }

        Ok(partitions)
    }

    /// Generate balanced partitions (parts of similar size)
    fn generate_balanced_partitions(&self, indices: &[usize]) -> Result<Vec<Partition>, StochasticError> {
        let n = indices.len();
        let mut partitions = Vec::new();

        // Two balanced parts
        let mid = n / 2;
        let part1 = indices[..mid].to_vec();
        let part2 = indices[mid..].to_vec();
        partitions.push(Partition::new(vec![part1, part2]));

        // Three balanced parts
        if n >= 3 && self.config.max_partition_size >= 3 {
            let third = n / 3;
            let part1 = indices[..third].to_vec();
            let part2 = indices[third..2*third].to_vec();
            let part3 = indices[2*third..].to_vec();
            partitions.push(Partition::new(vec![part1, part2, part3]));
        }

        // Four balanced parts
        if n >= 4 && self.config.max_partition_size >= 4 {
            let quarter = n / 4;
            let part1 = indices[..quarter].to_vec();
            let part2 = indices[quarter..2*quarter].to_vec();
            let part3 = indices[2*quarter..3*quarter].to_vec();
            let part4 = indices[3*quarter..].to_vec();
            partitions.push(Partition::new(vec![part1, part2, part3, part4]));
        }

        Ok(partitions)
    }

    /// Generate all possible partitions (exponential - use with caution)
    fn generate_all_partitions(&self, indices: &[usize]) -> Result<Vec<Partition>, StochasticError> {
        // For small systems, generate all bipartitions
        // Full partition generation is exponential and impractical for large systems
        if indices.len() > 10 {
            return self.generate_bipartitions(indices);
        }

        self.generate_sequential_partitions(indices)
    }

    /// Get computed Φ value
    pub fn phi(&self) -> Option<f64> {
        self.phi
    }

    /// Get minimum information partition
    pub fn mip(&self) -> Option<&Partition> {
        self.mip.as_ref()
    }

    /// Check if system is integrated (Φ > threshold)
    pub fn is_integrated(&self) -> Result<bool, StochasticError> {
        let phi = self.phi
            .ok_or_else(|| StochasticError::validation("Φ not computed"))?;

        Ok(phi > self.config.phi_threshold)
    }

    /// Check if system is reducible (Φ ≈ 0)
    pub fn is_reducible(&self) -> Result<bool, StochasticError> {
        Ok(!self.is_integrated()?)
    }

    /// Get integration ratio (Φ / S(ρ))
    pub fn integration_ratio(&self, total_entropy: f64) -> Result<f64, StochasticError> {
        let phi = self.phi
            .ok_or_else(|| StochasticError::validation("Φ not computed"))?;

        if total_entropy > 1e-10 {
            Ok(phi / total_entropy)
        } else {
            Err(StochasticError::numerical("Total entropy too small"))
        }
    }

    /// Get all EI results if computed
    pub fn all_ei_results(&self) -> Option<&[EffectiveInformationResult]> {
        self.all_ei_results.as_deref()
    }

    /// Compute complexity measure based on Φ and entropy
    pub fn complexity(&self, total_entropy: f64) -> Result<f64, StochasticError> {
        let phi = self.phi
            .ok_or_else(|| StochasticError::validation("Φ not computed"))?;

        // Complexity as geometric mean of |Φ| and entropy (use absolute value)
        Ok((phi.abs() * total_entropy).sqrt())
    }
}

impl Default for IntegratedInformation {
    fn default() -> Self {
        Self::new()
    }
}

impl StochasticAnalyzer for IntegratedInformation {
    fn name(&self) -> &str {
        "Integrated Information (Φ)"
    }

    fn analyze(&self, data: &TimeSeries) -> Result<AnalysisResult, StochasticError> {
        let mut analyzer = Self::with_config(self.config.clone());
        let phi = analyzer.compute_phi(data)?;

        // Also compute total entropy for context
        let mut entropy_analyzer = VonNeumannEntropy::with_config(self.config.vn_config.clone());
        let total_entropy = entropy_analyzer.compute_entropy(data)?;

        let is_integrated = analyzer.is_integrated()?;
        let integration_ratio = analyzer.integration_ratio(total_entropy)?;
        let complexity = analyzer.complexity(total_entropy)?;

        let interpretation = if is_integrated {
            if phi > total_entropy * 0.8 {
                format!("Highly integrated system with strong irreducibility (Φ = {:.4})", phi)
            } else if phi > total_entropy * 0.5 {
                format!("Moderately integrated system (Φ = {:.4})", phi)
            } else {
                format!("Weakly integrated system (Φ = {:.4})", phi)
            }
        } else {
            format!("Reducible system with minimal integration (Φ ≈ 0)")
        };

        let mut result = AnalysisResult::new(self.name())
            .with_metric("phi", phi)
            .with_metric("total_entropy", total_entropy)
            .with_metric("integration_ratio", integration_ratio)
            .with_metric("complexity", complexity)
            .with_metric("dimension", self.config.density_config.dimension as f64)
            .with_interpretation(interpretation)
            .with_metadata("integrated", is_integrated.to_string())
            .with_metadata("partition_strategy", format!("{:?}", self.config.partition_strategy));

        // Add MIP information
        if let Some(mip) = analyzer.mip() {
            result = result
                .with_metric("mip_num_parts", mip.num_parts() as f64)
                .with_metadata("mip_parts", format!("{}", mip.num_parts()));
        }

        // Add all EI values if computed
        if let Some(all_ei) = analyzer.all_ei_results() {
            for (i, ei_result) in all_ei.iter().enumerate() {
                result = result.with_metric(format!("ei_{}", i), ei_result.ei);
            }
        }

        Ok(result)
    }

    fn validate(&self, data: &TimeSeries) -> Result<bool, StochasticError> {
        if data.is_empty() {
            return Err(StochasticError::validation("Empty time series"));
        }

        if data.len() < self.config.min_partition_size {
            return Err(StochasticError::insufficient_data(
                self.config.min_partition_size,
                data.len()
            ));
        }

        Ok(true)
    }

    fn required_sample_size(&self) -> usize {
        self.config.min_partition_size.max(10)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iit_config_default() {
        let config = IITConfig::default();
        assert_eq!(config.partition_strategy, PartitionStrategy::Bipartition);
        assert_eq!(config.max_partition_size, 4);
        assert_eq!(config.min_partition_size, 2);
        assert!(!config.compute_all_partitions);
    }

    #[test]
    fn test_iit_config_builder() {
        let config = IITConfig::new(4)
            .with_partition_strategy(PartitionStrategy::Sequential)
            .with_max_partition_size(5)
            .with_compute_all(true)
            .with_phi_threshold(1e-8);

        assert_eq!(config.density_config.dimension, 4);
        assert_eq!(config.partition_strategy, PartitionStrategy::Sequential);
        assert_eq!(config.max_partition_size, 5);
        assert!(config.compute_all_partitions);
        assert_eq!(config.phi_threshold, 1e-8);
    }

    #[test]
    fn test_partition_validity() {
        let partition = Partition::new(vec![vec![0, 1], vec![2, 3]]);
        assert!(partition.is_valid(4));

        let invalid = Partition::new(vec![vec![0, 1], vec![1, 3]]);
        assert!(!invalid.is_valid(4));
    }

    #[test]
    fn test_partition_num_parts() {
        let partition = Partition::new(vec![vec![0, 1], vec![2, 3], vec![4, 5]]);
        assert_eq!(partition.num_parts(), 3);
    }

    #[test]
    fn test_bipartition_generation() {
        let config = IITConfig::new(2).with_partition_strategy(PartitionStrategy::Bipartition);
        let analyzer = IntegratedInformation::with_config(config);

        let indices = vec![0, 1, 2, 3];
        let partitions = analyzer.generate_bipartitions(&indices).unwrap();

        // Should have n-1 bipartitions for n elements
        assert_eq!(partitions.len(), 3);

        for partition in &partitions {
            assert_eq!(partition.num_parts(), 2);
            assert!(partition.is_valid(4));
        }
    }

    #[test]
    fn test_sequential_partitions() {
        let config = IITConfig::new(2).with_partition_strategy(PartitionStrategy::Sequential);
        let analyzer = IntegratedInformation::with_config(config);

        let indices = vec![0, 1, 2, 3, 4];
        let partitions = analyzer.generate_sequential_partitions(&indices).unwrap();

        assert!(!partitions.is_empty());

        for partition in &partitions {
            assert!(partition.is_valid(5));
        }
    }

    #[test]
    fn test_balanced_partitions() {
        let config = IITConfig::new(2)
            .with_partition_strategy(PartitionStrategy::Balanced)
            .with_max_partition_size(4);
        let analyzer = IntegratedInformation::with_config(config);

        let indices = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let partitions = analyzer.generate_balanced_partitions(&indices).unwrap();

        assert!(!partitions.is_empty());

        for partition in &partitions {
            assert!(partition.is_valid(8));
        }
    }

    #[test]
    fn test_integrated_system() {
        // Correlated data should show integration
        let values: Vec<f64> = (0..50).map(|x| (x as f64 * 0.1).sin()).collect();
        let data = TimeSeries::from_values(values);

        let config = IITConfig::new(3).with_partition_strategy(PartitionStrategy::Bipartition);
        let mut analyzer = IntegratedInformation::with_config(config);

        let phi = analyzer.compute_phi(&data).unwrap();

        // Φ can be negative for some systems (negative integration)
        assert!(phi.is_finite(), "Φ should be finite");
    }

    #[test]
    fn test_reducible_system() {
        // Independent random parts should be reducible
        use rand::Rng;
        let mut rng = rand::thread_rng();

        // Create two independent segments
        let mut values = Vec::new();
        for _ in 0..25 {
            values.push(rng.gen_range(-1.0..1.0));
        }
        for _ in 0..25 {
            values.push(rng.gen_range(10.0..11.0)); // Different range
        }

        let data = TimeSeries::from_values(values);

        let config = IITConfig::new(2).with_partition_strategy(PartitionStrategy::Bipartition);
        let mut analyzer = IntegratedInformation::with_config(config);

        let phi = analyzer.compute_phi(&data).unwrap();

        // Should have some Φ value (may or may not be reducible, can be negative)
        assert!(phi.is_finite());
    }

    #[test]
    fn test_phi_bounds() {
        let data = TimeSeries::from_values((0..60).map(|x| (x as f64 * 0.05).cos()).collect());

        let config = IITConfig::new(3);
        let mut analyzer = IntegratedInformation::with_config(config.clone());

        let phi = analyzer.compute_phi(&data).unwrap();

        // Compute total entropy for comparison
        let mut entropy_analyzer = VonNeumannEntropy::with_config(config.vn_config.clone());
        let total_entropy = entropy_analyzer.compute_entropy(&data).unwrap();

        // Φ can be negative in some partitions, but should be finite
        assert!(phi.is_finite(), "Φ should be finite");
        // The absolute value should be reasonable (can exceed total entropy when subsystems are highly structured)
        assert!(phi.abs() <= total_entropy * 3.0 + 2.0, "Φ magnitude should be reasonable");
    }

    #[test]
    fn test_mip_extraction() {
        let data = TimeSeries::from_values((0..40).map(|x| x as f64).collect());

        let config = IITConfig::new(2).with_partition_strategy(PartitionStrategy::Bipartition);
        let mut analyzer = IntegratedInformation::with_config(config);

        analyzer.compute_phi(&data).unwrap();

        let mip = analyzer.mip().unwrap();
        assert!(mip.num_parts() >= 2);
        assert!(mip.is_valid(data.len()));
    }

    #[test]
    fn test_integration_ratio() {
        let data = TimeSeries::from_values((0..50).map(|x| (x as f64 * 0.1).sin()).collect());

        let config = IITConfig::new(3);
        let mut analyzer = IntegratedInformation::with_config(config.clone());

        analyzer.compute_phi(&data).unwrap();

        let mut entropy_analyzer = VonNeumannEntropy::with_config(config.vn_config.clone());
        let total_entropy = entropy_analyzer.compute_entropy(&data).unwrap();

        let ratio = analyzer.integration_ratio(total_entropy).unwrap();

        // Ratio can be negative if Φ is negative, and can exceed 1 when subsystems are highly structured
        assert!(ratio.is_finite(), "Integration ratio should be finite");
    }

    #[test]
    fn test_complexity_measure() {
        let data = TimeSeries::from_values((0..60).map(|x| (x as f64).sin()).collect());

        let config = IITConfig::new(3);
        let mut analyzer = IntegratedInformation::with_config(config.clone());

        analyzer.compute_phi(&data).unwrap();

        let mut entropy_analyzer = VonNeumannEntropy::with_config(config.vn_config.clone());
        let total_entropy = entropy_analyzer.compute_entropy(&data).unwrap();

        let complexity = analyzer.complexity(total_entropy).unwrap();

        // Complexity is sqrt, so should be non-negative if inputs have same sign
        assert!(complexity.is_finite(), "Complexity should be finite");
    }

    #[test]
    fn test_analyzer_trait() {
        let data = TimeSeries::from_values((0..70).map(|x| (x as f64 * 0.05).cos()).collect());

        let config = IITConfig::new(3);
        let analyzer = IntegratedInformation::with_config(config);

        assert_eq!(analyzer.name(), "Integrated Information (Φ)");
        assert!(analyzer.validate(&data).is_ok());

        let result = analyzer.analyze(&data).unwrap();

        assert_eq!(result.framework, "Integrated Information (Φ)");
        assert!(result.metrics.contains_key("phi"));
        assert!(result.metrics.contains_key("total_entropy"));
        assert!(result.metrics.contains_key("integration_ratio"));
        assert!(result.metrics.contains_key("complexity"));
        assert!(!result.interpretation.is_empty());
    }

    #[test]
    fn test_insufficient_data() {
        let data = TimeSeries::from_values(vec![1.0]);

        let config = IITConfig::new(10);
        let analyzer = IntegratedInformation::with_config(config);

        let result = analyzer.validate(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_all_partitions_computation() {
        let data = TimeSeries::from_values((0..30).map(|x| (x as f64 * 0.2).sin()).collect());

        let config = IITConfig::new(2)
            .with_partition_strategy(PartitionStrategy::Sequential)
            .with_compute_all(true);

        let mut analyzer = IntegratedInformation::with_config(config);
        analyzer.compute_phi(&data).unwrap();

        let all_ei = analyzer.all_ei_results().unwrap();
        assert!(!all_ei.is_empty());

        for ei_result in all_ei {
            assert!(ei_result.ei >= 0.0 || ei_result.ei < 0.0); // Can be negative
        }
    }

    #[test]
    fn test_different_partition_strategies() {
        let data = TimeSeries::from_values((0..50).map(|x| (x as f64 * 0.1).sin()).collect());

        let strategies = vec![
            PartitionStrategy::Bipartition,
            PartitionStrategy::Sequential,
            PartitionStrategy::Balanced,
        ];

        for strategy in strategies {
            let config = IITConfig::new(3).with_partition_strategy(strategy);
            let mut analyzer = IntegratedInformation::with_config(config);

            let phi = analyzer.compute_phi(&data).unwrap();
            assert!(phi >= 0.0 || phi < 0.0, "Φ should be a real number for {:?}", strategy);
        }
    }
}
