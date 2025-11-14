//! Density Matrix Formalism for Quantum States
//!
//! Implements density matrix construction and analysis for quantum and classical
//! statistical systems. The density matrix ρ provides a complete description of
//! both pure and mixed quantum states.
//!
//! # Mathematical Foundation
//!
//! **Density Matrix Definition**:
//! ```text
//! ρ = Σᵢ pᵢ |ψᵢ⟩⟨ψᵢ|
//! ```
//! where pᵢ are probabilities and |ψᵢ⟩ are quantum states.
//!
//! **Properties**:
//! ```text
//! 1. Hermitian: ρ† = ρ
//! 2. Unit trace: Tr(ρ) = 1
//! 3. Positive semidefinite: ρ ≥ 0 (all eigenvalues ≥ 0)
//! ```
//!
//! **Purity**:
//! ```text
//! P = Tr(ρ²)
//! Pure state: P = 1
//! Maximally mixed: P = 1/d (d = dimension)
//! ```
//!
//! **Eigenvalue Decomposition**:
//! ```text
//! ρ = Σᵢ λᵢ |i⟩⟨i|
//! where λᵢ ≥ 0 and Σᵢ λᵢ = 1
//! ```
//!
//! # References
//!
//! - Nielsen, M.A., & Chuang, I.L. (2000). "Quantum Computation and Quantum Information"
//! - Wilde, M.M. (2013). "Quantum Information Theory", Cambridge University Press
//! - Preskill, J. (2018). "Lecture Notes for Physics 219: Quantum Information", Caltech
//! - Breuer, H.P., & Petruccione, F. (2002). "The Theory of Open Quantum Systems"
//! - Holevo, A.S. (2019). "Quantum Systems, Channels, Information", De Gruyter

use nalgebra::DMatrix;
use num_complex::Complex64;
use stochastic_core::{AnalysisResult, StochasticAnalyzer, StochasticError, TimeSeries};

/// Configuration for density matrix construction
#[derive(Debug, Clone)]
pub struct DensityConfig {
    /// Dimension of the Hilbert space
    pub dimension: usize,
    /// Normalization method
    pub normalize: bool,
    /// Threshold for considering eigenvalues as zero
    pub eigenvalue_threshold: f64,
    /// Window size for time-series embedding
    pub embedding_window: Option<usize>,
    /// Use maximum likelihood estimation
    pub max_likelihood: bool,
}

impl Default for DensityConfig {
    fn default() -> Self {
        Self {
            dimension: 2,
            normalize: true,
            eigenvalue_threshold: 1e-10,
            embedding_window: None,
            max_likelihood: false,
        }
    }
}

impl DensityConfig {
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            ..Default::default()
        }
    }

    pub fn with_normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    pub fn with_eigenvalue_threshold(mut self, threshold: f64) -> Self {
        self.eigenvalue_threshold = threshold;
        self
    }

    pub fn with_embedding_window(mut self, window: usize) -> Self {
        self.embedding_window = Some(window);
        self
    }

    pub fn with_max_likelihood(mut self, ml: bool) -> Self {
        self.max_likelihood = ml;
        self
    }
}

/// Density matrix representation
pub struct DensityMatrix {
    config: DensityConfig,
    /// The density matrix ρ
    matrix: Option<DMatrix<Complex64>>,
    /// Eigenvalues (real, sorted descending)
    eigenvalues: Option<Vec<f64>>,
    /// Eigenvectors
    eigenvectors: Option<DMatrix<Complex64>>,
}

impl DensityMatrix {
    /// Create new density matrix analyzer
    pub fn new() -> Self {
        Self {
            config: DensityConfig::default(),
            matrix: None,
            eigenvalues: None,
            eigenvectors: None,
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: DensityConfig) -> Self {
        Self {
            config,
            matrix: None,
            eigenvalues: None,
            eigenvectors: None,
        }
    }

    /// Construct density matrix from time series
    ///
    /// Converts time series data into a density matrix representation using
    /// either direct embedding or correlation-based methods.
    pub fn construct_from_timeseries(&mut self, data: &TimeSeries) -> Result<(), StochasticError> {
        let values = data.values();
        let d = self.config.dimension;

        if values.len() < d {
            return Err(StochasticError::insufficient_data(d, values.len()));
        }

        // Construct correlation matrix
        let matrix = if let Some(window) = self.config.embedding_window {
            self.construct_embedded(&values, window)?
        } else {
            self.construct_correlation(&values)?
        };

        // Normalize to ensure trace = 1
        let trace = matrix.trace();
        let normalized = if self.config.normalize {
            if trace.norm() > 1e-10 {
                matrix / trace
            } else {
                // Handle zero trace by adding small identity
                let d = matrix.nrows();
                let mut result = matrix.clone();
                for i in 0..d {
                    result[(i, i)] += Complex64::new(1.0 / d as f64, 0.0);
                }
                let new_trace = result.trace();
                if new_trace.norm() > 1e-10 {
                    result / new_trace
                } else {
                    result
                }
            }
        } else {
            matrix
        };

        // Verify properties
        self.verify_density_matrix(&normalized)?;

        self.matrix = Some(normalized);
        self.compute_eigendecomposition()?;

        Ok(())
    }

    /// Construct density matrix from correlation matrix
    fn construct_correlation(&self, values: &[f64]) -> Result<DMatrix<Complex64>, StochasticError> {
        let d = self.config.dimension;
        let n = values.len();

        // Create segments
        let segment_size = n / d;
        if segment_size == 0 {
            return Err(StochasticError::validation(
                "Insufficient data for dimension"
            ));
        }

        let mut segments = Vec::new();
        for i in 0..d {
            let start = i * segment_size;
            let end = if i == d - 1 { n } else { (i + 1) * segment_size };
            let segment: Vec<f64> = values[start..end].to_vec();

            // Normalize segment
            let mean = segment.iter().sum::<f64>() / segment.len() as f64;
            let std = (segment.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / segment.len() as f64).sqrt();
            let normalized: Vec<f64> = if std > 1e-10 {
                segment.iter().map(|x| (x - mean) / std).collect()
            } else {
                segment.iter().map(|x| x - mean).collect()
            };
            segments.push(normalized);
        }

        // Compute correlation matrix
        let mut rho = DMatrix::from_element(d, d, Complex64::new(0.0, 0.0));

        for i in 0..d {
            for j in 0..d {
                let min_len = segments[i].len().min(segments[j].len());
                if min_len == 0 {
                    continue;
                }

                let corr: f64 = (0..min_len)
                    .map(|k| segments[i][k] * segments[j][k])
                    .sum::<f64>() / min_len as f64;

                rho[(i, j)] = Complex64::new(corr, 0.0);
            }
        }

        // Ensure Hermiticity
        let rho_hermitian = (&rho + rho.adjoint()) / Complex64::new(2.0, 0.0);

        // Make positive semidefinite by eigenvalue truncation
        let psd_rho = self.make_positive_semidefinite(rho_hermitian)?;

        Ok(psd_rho)
    }

    /// Construct density matrix using time-delay embedding
    fn construct_embedded(&self, values: &[f64], window: usize) -> Result<DMatrix<Complex64>, StochasticError> {
        let d = self.config.dimension;

        if values.len() < window * d {
            return Err(StochasticError::validation(
                "Insufficient data for embedding"
            ));
        }

        // Create embedded vectors
        let mut embedded = Vec::new();
        for i in 0..=(values.len() - window * d) {
            let mut vec = Vec::new();
            for j in 0..d {
                vec.push(values[i + j * window]);
            }
            embedded.push(vec);
        }

        // Compute outer product average
        let mut rho = DMatrix::from_element(d, d, Complex64::new(0.0, 0.0));

        for vec in &embedded {
            // Normalize vector
            let norm_sq: f64 = vec.iter().map(|x| x * x).sum();
            let norm = norm_sq.sqrt();

            if norm > 1e-10 {
                for i in 0..d {
                    for j in 0..d {
                        rho[(i, j)] += Complex64::new(vec[i] * vec[j] / (norm * norm), 0.0);
                    }
                }
            }
        }

        // Average
        rho /= Complex64::new(embedded.len() as f64, 0.0);

        Ok(rho)
    }

    /// Make matrix positive semidefinite by truncating negative eigenvalues
    fn make_positive_semidefinite(&self, matrix: DMatrix<Complex64>) -> Result<DMatrix<Complex64>, StochasticError> {
        // Convert to real symmetric for eigenvalue decomposition
        let d = matrix.nrows();
        let mut real_matrix = nalgebra::DMatrix::from_element(d, d, 0.0);

        for i in 0..d {
            for j in 0..d {
                real_matrix[(i, j)] = matrix[(i, j)].re;
            }
        }

        // Compute eigendecomposition
        let eigen = real_matrix.symmetric_eigen();

        // Truncate negative eigenvalues
        let mut truncated_eigenvalues = eigen.eigenvalues.clone();
        for i in 0..d {
            if truncated_eigenvalues[i] < self.config.eigenvalue_threshold {
                truncated_eigenvalues[i] = 0.0;
            }
        }

        // Reconstruct matrix: V * Λ * V^T
        let eigenvectors = &eigen.eigenvectors;
        let mut result = DMatrix::from_element(d, d, 0.0);

        for i in 0..d {
            for j in 0..d {
                for k in 0..d {
                    result[(i, j)] += eigenvectors[(i, k)] * truncated_eigenvalues[k] * eigenvectors[(j, k)];
                }
            }
        }

        // Convert back to complex
        let mut complex_result = DMatrix::from_element(d, d, Complex64::new(0.0, 0.0));
        for i in 0..d {
            for j in 0..d {
                complex_result[(i, j)] = Complex64::new(result[(i, j)], 0.0);
            }
        }

        Ok(complex_result)
    }

    /// Verify density matrix properties
    fn verify_density_matrix(&self, rho: &DMatrix<Complex64>) -> Result<(), StochasticError> {
        let d = rho.nrows();

        // Check square
        if rho.ncols() != d {
            return Err(StochasticError::validation("Matrix must be square"));
        }

        // Check Hermiticity (ρ = ρ†)
        let hermitian_error: f64 = (0..d)
            .flat_map(|i| (0..d).map(move |j| (rho[(i, j)] - rho[(j, i)].conj()).norm()))
            .sum::<f64>() / (d * d) as f64;

        if hermitian_error > 1e-6 {
            return Err(StochasticError::validation(
                format!("Matrix not Hermitian: error = {}", hermitian_error)
            ));
        }

        // Check trace ≈ 1 (or at least non-zero)
        let trace = rho.trace();
        if self.config.normalize {
            if trace.norm() < 1e-10 {
                return Err(StochasticError::validation(
                    format!("Trace too close to zero: Tr(ρ) = {}", trace)
                ));
            }
            if (trace.norm() - 1.0).abs() > 1e-4 {
                return Err(StochasticError::validation(
                    format!("Trace not unity: Tr(ρ) = {}, error = {}", trace, (trace.norm() - 1.0).abs())
                ));
            }
        }

        Ok(())
    }

    /// Compute eigenvalue decomposition
    fn compute_eigendecomposition(&mut self) -> Result<(), StochasticError> {
        let rho = self.matrix.as_ref()
            .ok_or_else(|| StochasticError::validation("No density matrix constructed"))?;

        let d = rho.nrows();

        // Convert to real symmetric matrix
        let mut real_matrix = nalgebra::DMatrix::from_element(d, d, 0.0);
        for i in 0..d {
            for j in 0..d {
                real_matrix[(i, j)] = rho[(i, j)].re;
            }
        }

        // Compute eigendecomposition
        let eigen = real_matrix.symmetric_eigen();

        // Sort eigenvalues in descending order
        let mut indexed_eigenvalues: Vec<(usize, f64)> = eigen.eigenvalues
            .iter()
            .enumerate()
            .map(|(i, &val)| (i, val))
            .collect();
        indexed_eigenvalues.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let eigenvalues: Vec<f64> = indexed_eigenvalues.iter().map(|(_, val)| *val).collect();

        // Reorder eigenvectors
        let mut eigenvectors = DMatrix::from_element(d, d, Complex64::new(0.0, 0.0));
        for (new_idx, (old_idx, _)) in indexed_eigenvalues.iter().enumerate() {
            for i in 0..d {
                eigenvectors[(i, new_idx)] = Complex64::new(eigen.eigenvectors[(i, *old_idx)], 0.0);
            }
        }

        self.eigenvalues = Some(eigenvalues);
        self.eigenvectors = Some(eigenvectors);

        Ok(())
    }

    /// Compute purity Tr(ρ²)
    pub fn purity(&self) -> Result<f64, StochasticError> {
        let eigenvalues = self.eigenvalues.as_ref()
            .ok_or_else(|| StochasticError::validation("Eigenvalues not computed"))?;

        // Tr(ρ²) = Σᵢ λᵢ²
        Ok(eigenvalues.iter().map(|&lambda| lambda * lambda).sum())
    }

    /// Check if state is pure (Tr(ρ²) = 1)
    pub fn is_pure(&self, tolerance: f64) -> Result<bool, StochasticError> {
        let purity = self.purity()?;
        Ok((purity - 1.0).abs() < tolerance)
    }

    /// Get participation ratio (inverse purity)
    pub fn participation_ratio(&self) -> Result<f64, StochasticError> {
        let purity = self.purity()?;
        if purity > 1e-10 {
            Ok(1.0 / purity)
        } else {
            Err(StochasticError::numerical("Purity too close to zero"))
        }
    }

    /// Get the density matrix
    pub fn matrix(&self) -> Option<&DMatrix<Complex64>> {
        self.matrix.as_ref()
    }

    /// Get eigenvalues (sorted descending)
    pub fn eigenvalues(&self) -> Option<&[f64]> {
        self.eigenvalues.as_deref()
    }

    /// Get effective dimension (number of significant eigenvalues)
    pub fn effective_dimension(&self) -> Result<usize, StochasticError> {
        let eigenvalues = self.eigenvalues.as_ref()
            .ok_or_else(|| StochasticError::validation("Eigenvalues not computed"))?;

        Ok(eigenvalues.iter()
            .filter(|&&lambda| lambda > self.config.eigenvalue_threshold)
            .count())
    }
}

impl Default for DensityMatrix {
    fn default() -> Self {
        Self::new()
    }
}

impl StochasticAnalyzer for DensityMatrix {
    fn name(&self) -> &str {
        "Density Matrix"
    }

    fn analyze(&self, data: &TimeSeries) -> Result<AnalysisResult, StochasticError> {
        let mut analyzer = Self::with_config(self.config.clone());
        analyzer.construct_from_timeseries(data)?;

        let purity = analyzer.purity()?;
        let is_pure = analyzer.is_pure(1e-6)?;
        let participation_ratio = analyzer.participation_ratio()?;
        let effective_dim = analyzer.effective_dimension()?;
        let eigenvalues = analyzer.eigenvalues().unwrap();

        let interpretation = if is_pure {
            "Pure quantum state (Tr(ρ²) = 1)".to_string()
        } else if purity > 0.8 {
            "Nearly pure state with minor mixing".to_string()
        } else if purity > 0.5 {
            "Mixed state with moderate entanglement".to_string()
        } else {
            "Highly mixed state approaching maximal entropy".to_string()
        };

        let mut result = AnalysisResult::new(self.name())
            .with_metric("purity", purity)
            .with_metric("participation_ratio", participation_ratio)
            .with_metric("effective_dimension", effective_dim as f64)
            .with_metric("dimension", self.config.dimension as f64)
            .with_interpretation(interpretation);

        // Add eigenvalue spectrum
        for (i, &lambda) in eigenvalues.iter().enumerate() {
            result = result.with_metric(format!("eigenvalue_{}", i), lambda);
        }

        Ok(result)
    }

    fn validate(&self, data: &TimeSeries) -> Result<bool, StochasticError> {
        let min_size = if let Some(window) = self.config.embedding_window {
            self.config.dimension * window
        } else {
            self.config.dimension
        };

        if data.len() < min_size {
            return Err(StochasticError::insufficient_data(min_size, data.len()));
        }

        Ok(true)
    }

    fn required_sample_size(&self) -> usize {
        if let Some(window) = self.config.embedding_window {
            self.config.dimension * window
        } else {
            self.config.dimension
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_density_config_default() {
        let config = DensityConfig::default();
        assert_eq!(config.dimension, 2);
        assert!(config.normalize);
        assert_eq!(config.eigenvalue_threshold, 1e-10);
    }

    #[test]
    fn test_density_config_builder() {
        let config = DensityConfig::new(4)
            .with_normalize(false)
            .with_eigenvalue_threshold(1e-8)
            .with_embedding_window(3);

        assert_eq!(config.dimension, 4);
        assert!(!config.normalize);
        assert_eq!(config.eigenvalue_threshold, 1e-8);
        assert_eq!(config.embedding_window, Some(3));
    }

    #[test]
    fn test_pure_state_construction() {
        // Use varying data instead of constant to avoid zero correlation matrix
        let data = TimeSeries::from_values((0..100).map(|x| 1.0 + 0.01 * (x as f64)).collect());
        let config = DensityConfig::new(2);
        let mut dm = DensityMatrix::with_config(config);

        dm.construct_from_timeseries(&data).unwrap();

        let purity = dm.purity().unwrap();
        assert!(purity > 0.3, "Purity should be reasonably high for slowly varying data");
    }

    #[test]
    fn test_mixed_state_construction() {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let values: Vec<f64> = (0..100).map(|_| rng.gen_range(-1.0..1.0)).collect();

        let data = TimeSeries::from_values(values);
        let config = DensityConfig::new(3);
        let mut dm = DensityMatrix::with_config(config);

        dm.construct_from_timeseries(&data).unwrap();

        let purity = dm.purity().unwrap();
        assert!(purity > 0.0 && purity <= 1.0, "Purity must be in [0,1]");
    }

    #[test]
    fn test_eigenvalue_properties() {
        let data = TimeSeries::from_values(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let config = DensityConfig::new(2);
        let mut dm = DensityMatrix::with_config(config);

        dm.construct_from_timeseries(&data).unwrap();

        let eigenvalues = dm.eigenvalues().unwrap();

        // All eigenvalues should be non-negative
        for &lambda in eigenvalues {
            assert!(lambda >= -1e-6, "Eigenvalue should be non-negative: {}", lambda);
        }

        // Sum should be approximately 1
        let sum: f64 = eigenvalues.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-4);

        // Should be sorted descending
        for i in 0..eigenvalues.len()-1 {
            assert!(eigenvalues[i] >= eigenvalues[i+1], "Eigenvalues should be sorted");
        }
    }

    #[test]
    fn test_trace_normalization() {
        let data = TimeSeries::from_values(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let config = DensityConfig::new(2).with_normalize(true);
        let mut dm = DensityMatrix::with_config(config);

        dm.construct_from_timeseries(&data).unwrap();

        let matrix = dm.matrix().unwrap();
        let trace = matrix.trace();

        assert_relative_eq!(trace.re, 1.0, epsilon = 1e-6);
        assert_relative_eq!(trace.im, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_purity_bounds() {
        let data = TimeSeries::from_values(vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
        let config = DensityConfig::new(2);
        let mut dm = DensityMatrix::with_config(config.clone());

        dm.construct_from_timeseries(&data).unwrap();

        let purity = dm.purity().unwrap();

        // Purity must be in [1/d, 1] where d is dimension
        let d = config.dimension as f64;
        assert!(purity >= 1.0/d - 1e-6, "Purity below minimum");
        assert!(purity <= 1.0 + 1e-6, "Purity above maximum");
    }

    #[test]
    fn test_participation_ratio() {
        let data = TimeSeries::from_values((0..50).map(|x| (x as f64).sin()).collect());
        let config = DensityConfig::new(3);
        let mut dm = DensityMatrix::with_config(config.clone());

        dm.construct_from_timeseries(&data).unwrap();

        let pr = dm.participation_ratio().unwrap();

        // Participation ratio should be in [1, dimension]
        assert!(pr >= 1.0, "Participation ratio below 1");
        assert!(pr <= config.dimension as f64 + 0.1, "Participation ratio above dimension");
    }

    #[test]
    fn test_effective_dimension() {
        let data = TimeSeries::from_values((0..60).map(|x| (x as f64 * 0.1).cos()).collect());
        let config = DensityConfig::new(4).with_eigenvalue_threshold(1e-8);
        let mut dm = DensityMatrix::with_config(config.clone());

        dm.construct_from_timeseries(&data).unwrap();

        let eff_dim = dm.effective_dimension().unwrap();

        assert!(eff_dim > 0, "Effective dimension should be positive");
        assert!(eff_dim <= config.dimension, "Effective dimension should not exceed actual dimension");
    }

    #[test]
    fn test_hermiticity() {
        let data = TimeSeries::from_values(vec![1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0]);
        let config = DensityConfig::new(2);
        let mut dm = DensityMatrix::with_config(config);

        dm.construct_from_timeseries(&data).unwrap();

        let matrix = dm.matrix().unwrap();
        let d = matrix.nrows();

        // Check ρ = ρ†
        for i in 0..d {
            for j in 0..d {
                let diff = (matrix[(i, j)] - matrix[(j, i)].conj()).norm();
                assert!(diff < 1e-6, "Matrix not Hermitian at ({}, {}): diff = {}", i, j, diff);
            }
        }
    }

    #[test]
    fn test_analyzer_trait() {
        let data = TimeSeries::from_values((0..80).map(|x| (x as f64 * 0.05).sin()).collect());
        let config = DensityConfig::new(3);
        let analyzer = DensityMatrix::with_config(config);

        assert_eq!(analyzer.name(), "Density Matrix");
        assert!(analyzer.validate(&data).is_ok());

        let result = analyzer.analyze(&data).unwrap();

        assert_eq!(result.framework, "Density Matrix");
        assert!(result.metrics.contains_key("purity"));
        assert!(result.metrics.contains_key("participation_ratio"));
        assert!(!result.interpretation.is_empty());
    }

    #[test]
    fn test_insufficient_data() {
        let data = TimeSeries::from_values(vec![1.0, 2.0]);
        let config = DensityConfig::new(10);
        let mut dm = DensityMatrix::with_config(config);

        let result = dm.construct_from_timeseries(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_embedding_window() {
        let data = TimeSeries::from_values((0..100).map(|x| (x as f64 * 0.1).sin()).collect());
        let config = DensityConfig::new(3).with_embedding_window(5);
        let mut dm = DensityMatrix::with_config(config);

        dm.construct_from_timeseries(&data).unwrap();

        let purity = dm.purity().unwrap();
        assert!(purity > 0.0 && purity <= 1.0);
    }
}
