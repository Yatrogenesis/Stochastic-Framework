//! Mutual Information implementation
//!
//! Measures the mutual dependence between two random variables.
//! Quantifies the reduction in uncertainty about one variable given knowledge of the other.
//!
//! # Mathematical Foundation
//!
//! **Mutual Information:**
//! ```text
//! I(X; Y) = Σₓ Σᵧ p(x,y) log[p(x,y) / (p(x)p(y))]
//! ```
//!
//! **Alternative formulation:**
//! ```text
//! I(X; Y) = H(X) + H(Y) - H(X,Y)
//! ```
//!
//! where:
//! - H(X) = entropy of X
//! - H(Y) = entropy of Y
//! - H(X,Y) = joint entropy
//!
//! **Properties:**
//! - I(X; Y) ≥ 0
//! - I(X; Y) = 0 iff X and Y are independent
//! - I(X; Y) = H(X) = H(Y) iff X and Y are deterministically related
//! - I(X; Y) = I(Y; X) (symmetric)
//!
//! # References
//!
//! - Shannon, C.E. (1948). "A Mathematical Theory of Communication"
//! - Cover, T.M., & Thomas, J.A. (2006). "Elements of Information Theory"
//! - Kraskov, A., Stögbauer, H., & Grassberger, P. (2004). "Estimating mutual information"
//! - Gao, S., Ver Steeg, G., & Galstyan, A. (2015). "Efficient Estimation of Mutual Information for Strongly Dependent Variables"
//! - Khan, S., et al. (2007). "Relative performance of mutual information estimation methods for quantifying the dependence among short and noisy data"

use stochastic_core::{AnalysisResult, StochasticAnalyzer, StochasticError, TimeSeries};
use std::collections::HashMap;

/// Configuration for Mutual Information calculation
#[derive(Debug, Clone)]
pub struct MutualInformationConfig {
    /// Number of bins for discretization (X variable)
    pub num_bins_x: Option<usize>,
    /// Number of bins for discretization (Y variable)
    pub num_bins_y: Option<usize>,
    /// Base of logarithm (2 = bits, e = nats)
    pub log_base: LogBase,
    /// Whether to normalize MI to [0, 1]
    pub normalize: bool,
}

/// Logarithm base for mutual information
#[derive(Debug, Clone, Copy)]
pub enum LogBase {
    /// Binary logarithm (bits)
    Two,
    /// Natural logarithm (nats)
    E,
}

impl Default for MutualInformationConfig {
    fn default() -> Self {
        Self {
            num_bins_x: None,
            num_bins_y: None,
            log_base: LogBase::Two,
            normalize: false,
        }
    }
}

/// Mutual Information analyzer
pub struct MutualInformation {
    config: MutualInformationConfig,
}

impl MutualInformation {
    /// Create a new Mutual Information analyzer
    pub fn new() -> Self {
        Self {
            config: MutualInformationConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: MutualInformationConfig) -> Self {
        Self { config }
    }

    /// Set number of bins for both variables
    pub fn with_bins(mut self, bins: usize) -> Self {
        self.config.num_bins_x = Some(bins);
        self.config.num_bins_y = Some(bins);
        self
    }

    /// Set number of bins separately
    pub fn with_bins_separate(mut self, bins_x: usize, bins_y: usize) -> Self {
        self.config.num_bins_x = Some(bins_x);
        self.config.num_bins_y = Some(bins_y);
        self
    }

    /// Set logarithm base
    pub fn with_log_base(mut self, log_base: LogBase) -> Self {
        self.config.log_base = log_base;
        self
    }

    /// Enable normalization
    pub fn with_normalize(mut self, normalize: bool) -> Self {
        self.config.normalize = normalize;
        self
    }

    /// Compute optimal bins using Sturges' rule
    fn optimal_bins(&self, n: usize, for_x: bool) -> usize {
        let bins_option = if for_x {
            self.config.num_bins_x
        } else {
            self.config.num_bins_y
        };

        if let Some(bins) = bins_option {
            return bins;
        }

        // Sturges' rule: k = ceil(1 + log2(n))
        // For MI, use slightly fewer bins to avoid sparsity
        let k = 1.0 + (n as f64).log2() * 0.8;
        k.ceil().max(2.0) as usize
    }

    /// Discretize continuous data
    fn discretize(&self, values: &[f64], num_bins: usize) -> Vec<usize> {
        let min = values.iter().copied().fold(f64::INFINITY, f64::min);
        let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        let bin_width = if max == min {
            1.0
        } else {
            (max - min) / num_bins as f64
        };

        values
            .iter()
            .map(|&v| {
                if max == min {
                    0
                } else if v == max {
                    num_bins - 1
                } else {
                    ((v - min) / bin_width).floor() as usize
                }
            })
            .collect()
    }

    /// Compute entropy from counts
    fn entropy_from_counts(&self, counts: &HashMap<usize, usize>, total: usize) -> f64 {
        let mut h = 0.0;

        for &count in counts.values() {
            if count > 0 {
                let p = count as f64 / total as f64;
                let log_p = match self.config.log_base {
                    LogBase::Two => p.log2(),
                    LogBase::E => p.ln(),
                };
                h -= p * log_p;
            }
        }

        h
    }

    /// Compute joint entropy from 2D counts
    fn joint_entropy(&self, joint_counts: &HashMap<(usize, usize), usize>, total: usize) -> f64 {
        let mut h = 0.0;

        for &count in joint_counts.values() {
            if count > 0 {
                let p = count as f64 / total as f64;
                let log_p = match self.config.log_base {
                    LogBase::Two => p.log2(),
                    LogBase::E => p.ln(),
                };
                h -= p * log_p;
            }
        }

        h
    }

    /// Calculate mutual information between two time series
    pub fn calculate(
        &self,
        x_data: &TimeSeries,
        y_data: &TimeSeries,
    ) -> Result<AnalysisResult, StochasticError> {
        // Validate both inputs
        self.validate(x_data)?;
        self.validate(y_data)?;

        if x_data.len() != y_data.len() {
            return Err(StochasticError::validation(
                format!(
                    "Time series must have equal length. X: {}, Y: {}",
                    x_data.len(),
                    y_data.len()
                )
            ));
        }

        let n = x_data.len();
        let x_values = x_data.values();
        let y_values = y_data.values();

        // Determine number of bins
        let num_bins_x = self.optimal_bins(n, true);
        let num_bins_y = self.optimal_bins(n, false);

        // Discretize data
        let x_discrete = self.discretize(&x_values, num_bins_x);
        let y_discrete = self.discretize(&y_values, num_bins_y);

        // Compute marginal distributions
        let mut x_counts: HashMap<usize, usize> = HashMap::new();
        let mut y_counts: HashMap<usize, usize> = HashMap::new();
        let mut joint_counts: HashMap<(usize, usize), usize> = HashMap::new();

        for i in 0..n {
            let x_bin = x_discrete[i];
            let y_bin = y_discrete[i];

            *x_counts.entry(x_bin).or_insert(0) += 1;
            *y_counts.entry(y_bin).or_insert(0) += 1;
            *joint_counts.entry((x_bin, y_bin)).or_insert(0) += 1;
        }

        // Compute entropies
        let h_x = self.entropy_from_counts(&x_counts, n);
        let h_y = self.entropy_from_counts(&y_counts, n);
        let h_xy = self.joint_entropy(&joint_counts, n);

        // Compute mutual information: I(X;Y) = H(X) + H(Y) - H(X,Y)
        let mi = h_x + h_y - h_xy;

        // Normalized mutual information (various definitions)
        let nmi_arithmetic = if h_x + h_y > 0.0 {
            2.0 * mi / (h_x + h_y)
        } else {
            0.0
        };

        let nmi_geometric = if h_x * h_y > 0.0 {
            mi / (h_x * h_y).sqrt()
        } else {
            0.0
        };

        let nmi_min = if h_x.min(h_y) > 0.0 {
            mi / h_x.min(h_y)
        } else {
            0.0
        };

        let nmi_max = if h_x.max(h_y) > 0.0 {
            mi / h_x.max(h_y)
        } else {
            0.0
        };

        // Information quality ratio (efficiency of shared information)
        let information_quality = if h_xy > 0.0 {
            mi / h_xy
        } else {
            0.0
        };

        // Variation of information (metric distance)
        let variation_of_information = h_xy - mi;

        // Conditional entropies
        let h_x_given_y = h_xy - h_y;
        let h_y_given_x = h_xy - h_x;

        let log_base_str = match self.config.log_base {
            LogBase::Two => "bits",
            LogBase::E => "nats",
        };

        let interpretation = if mi < 0.01 {
            format!(
                "Near-zero MI ({:.4} {}) indicates X and Y are approximately independent. \
                 No significant mutual dependence detected.",
                mi, log_base_str
            )
        } else if mi > 0.9 * h_x.min(h_y) {
            format!(
                "High MI ({:.4} {}) indicates strong dependence between X and Y. \
                 Variables share substantial information (NMI_min = {:.3})",
                mi, log_base_str, nmi_min
            )
        } else {
            format!(
                "Moderate MI ({:.4} {}) indicates partial dependence. \
                 X and Y share some information but retain independence (NMI = {:.3})",
                mi, log_base_str, nmi_arithmetic
            )
        };

        let result = AnalysisResult::new(self.name())
            .with_metric("mutual_information", mi)
            .with_metric("entropy_x", h_x)
            .with_metric("entropy_y", h_y)
            .with_metric("joint_entropy", h_xy)
            .with_metric("conditional_entropy_x_given_y", h_x_given_y)
            .with_metric("conditional_entropy_y_given_x", h_y_given_x)
            .with_metric("nmi_arithmetic", nmi_arithmetic)
            .with_metric("nmi_geometric", nmi_geometric)
            .with_metric("nmi_min", nmi_min)
            .with_metric("nmi_max", nmi_max)
            .with_metric("information_quality", information_quality)
            .with_metric("variation_of_information", variation_of_information)
            .with_metric("num_bins_x", num_bins_x as f64)
            .with_metric("num_bins_y", num_bins_y as f64)
            .with_metric("sample_size", n as f64)
            .with_interpretation(interpretation)
            .with_metadata("log_base", format!("{:?}", self.config.log_base))
            .with_metadata("units", log_base_str);

        Ok(result)
    }
}

impl Default for MutualInformation {
    fn default() -> Self {
        Self::new()
    }
}

impl StochasticAnalyzer for MutualInformation {
    fn name(&self) -> &str {
        "Mutual Information"
    }

    fn analyze(&self, _data: &TimeSeries) -> Result<AnalysisResult, StochasticError> {
        Err(StochasticError::validation(
            "Mutual Information requires two time series. Use calculate(x, y) method instead."
        ))
    }

    fn required_sample_size(&self) -> usize {
        // MI estimation requires more samples due to joint distribution
        30
    }

    fn validate(&self, data: &TimeSeries) -> Result<bool, StochasticError> {
        if data.len() < self.required_sample_size() {
            return Err(StochasticError::insufficient_data(
                self.required_sample_size(),
                data.len(),
            ));
        }

        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use stochastic_core::Domain;
    use approx::assert_relative_eq;

    #[test]
    fn test_independent_variables_zero_mi() {
        // X and Y are independent
        let x: Vec<f64> = (0..100).map(|i| (i % 10) as f64).collect();
        let y: Vec<f64> = (0..100).map(|i| ((i * 3) % 7) as f64).collect();

        let x_data = TimeSeries::from_values(x).with_domain(Domain::Discrete);
        let y_data = TimeSeries::from_values(y).with_domain(Domain::Discrete);

        let analyzer = MutualInformation::new();
        let result = analyzer.calculate(&x_data, &y_data).unwrap();

        let mi = result.metrics.get("mutual_information").unwrap();

        // Independent variables should have very low MI
        assert!(*mi < 0.1, "MI for independent variables too high: {}", mi);
    }

    #[test]
    fn test_identical_variables_max_mi() {
        // X = Y (perfect dependence)
        let x: Vec<f64> = (0..100).map(|i| (i % 10) as f64).collect();
        let y = x.clone();

        let x_data = TimeSeries::from_values(x).with_domain(Domain::Discrete);
        let y_data = TimeSeries::from_values(y).with_domain(Domain::Discrete);

        let analyzer = MutualInformation::new();
        let result = analyzer.calculate(&x_data, &y_data).unwrap();

        let mi = result.metrics.get("mutual_information").unwrap();
        let h_x = result.metrics.get("entropy_x").unwrap();
        let h_y = result.metrics.get("entropy_y").unwrap();

        // For X = Y: I(X;Y) = H(X) = H(Y)
        assert_relative_eq!(*mi, *h_x, epsilon = 0.01);
        assert_relative_eq!(*mi, *h_y, epsilon = 0.01);
    }

    #[test]
    fn test_linear_relationship() {
        // Y = 2X + 1 (deterministic linear)
        let x: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&v| 2.0 * v + 1.0).collect();

        let x_data = TimeSeries::from_values(x).with_domain(Domain::Continuous);
        let y_data = TimeSeries::from_values(y).with_domain(Domain::Continuous);

        let analyzer = MutualInformation::new().with_bins(10);
        let result = analyzer.calculate(&x_data, &y_data).unwrap();

        let mi = result.metrics.get("mutual_information").unwrap();
        let h_x = result.metrics.get("entropy_x").unwrap();

        // Deterministic relationship: MI should be high (close to H(X))
        assert!(*mi > 0.9 * h_x, "MI for deterministic relationship too low");
    }

    #[test]
    fn test_symmetry() {
        // I(X;Y) = I(Y;X)
        let x: Vec<f64> = (0..100).map(|i| (i % 7) as f64).collect();
        let y: Vec<f64> = (0..100).map(|i| (i % 5) as f64).collect();

        let x_data = TimeSeries::from_values(x.clone()).with_domain(Domain::Discrete);
        let y_data = TimeSeries::from_values(y.clone()).with_domain(Domain::Discrete);

        let analyzer = MutualInformation::new();

        let result_xy = analyzer.calculate(&x_data, &y_data).unwrap();
        let result_yx = analyzer.calculate(&y_data, &x_data).unwrap();

        let mi_xy = result_xy.metrics.get("mutual_information").unwrap();
        let mi_yx = result_yx.metrics.get("mutual_information").unwrap();

        assert_relative_eq!(*mi_xy, *mi_yx, epsilon = 1e-10);
    }

    #[test]
    fn test_unequal_length_error() {
        // x has 30 elements, y has 35 elements (both > required_sample_size)
        let x_vec = vec![1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0,
                         1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0,
                         1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0];
        let y_vec = vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0,
                         1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0,
                         1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0,
                         1.0, 2.0, 1.0, 2.0, 1.0];  // 35 elements (different length)
        let x = TimeSeries::from_values(x_vec);
        let y = TimeSeries::from_values(y_vec);

        let analyzer = MutualInformation::new();
        let result = analyzer.calculate(&x, &y);

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("equal length"));
    }

    #[test]
    fn test_insufficient_data() {
        let x = TimeSeries::from_values(vec![1.0, 2.0]);
        let y = TimeSeries::from_values(vec![1.0, 2.0]);

        let analyzer = MutualInformation::new();
        let result = analyzer.calculate(&x, &y);

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Insufficient"));
    }

    #[test]
    fn test_conditional_entropies() {
        let x: Vec<f64> = (0..100).map(|i| (i % 5) as f64).collect();
        let y = x.clone(); // Y = X

        let x_data = TimeSeries::from_values(x).with_domain(Domain::Discrete);
        let y_data = TimeSeries::from_values(y).with_domain(Domain::Discrete);

        let analyzer = MutualInformation::new();
        let result = analyzer.calculate(&x_data, &y_data).unwrap();

        let h_x_given_y = result.metrics.get("conditional_entropy_x_given_y").unwrap();
        let h_y_given_x = result.metrics.get("conditional_entropy_y_given_x").unwrap();

        // For X = Y: H(X|Y) = H(Y|X) = 0
        assert_relative_eq!(*h_x_given_y, 0.0, epsilon = 0.01);
        assert_relative_eq!(*h_y_given_x, 0.0, epsilon = 0.01);
    }

    #[test]
    fn test_normalized_mi_range() {
        let x: Vec<f64> = (0..100).map(|i| (i % 8) as f64).collect();
        let y: Vec<f64> = (0..100).map(|i| (i % 6) as f64).collect();

        let x_data = TimeSeries::from_values(x).with_domain(Domain::Discrete);
        let y_data = TimeSeries::from_values(y).with_domain(Domain::Discrete);

        let analyzer = MutualInformation::new();
        let result = analyzer.calculate(&x_data, &y_data).unwrap();

        let nmi_arith = result.metrics.get("nmi_arithmetic").unwrap();
        let nmi_geom = result.metrics.get("nmi_geometric").unwrap();

        // Normalized MI should be in [0, 1]
        assert!(*nmi_arith >= 0.0 && *nmi_arith <= 1.0);
        assert!(*nmi_geom >= 0.0 && *nmi_geom <= 1.0);
    }

    #[test]
    fn test_variation_of_information() {
        let x: Vec<f64> = (0..100).map(|i| (i % 5) as f64).collect();
        let y = x.clone();

        let x_data = TimeSeries::from_values(x).with_domain(Domain::Discrete);
        let y_data = TimeSeries::from_values(y).with_domain(Domain::Discrete);

        let analyzer = MutualInformation::new();
        let result = analyzer.calculate(&x_data, &y_data).unwrap();

        let vi = result.metrics.get("variation_of_information").unwrap();

        // For X = Y: VI = H(X,Y) - I(X;Y) = H(X) - H(X) = 0
        assert_relative_eq!(*vi, 0.0, epsilon = 0.01);
    }

    #[test]
    fn test_log_base_nats() {
        let x: Vec<f64> = vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
                               0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
                               0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]; // Binary
        let y = x.clone();

        let x_data = TimeSeries::from_values(x).with_domain(Domain::Discrete);
        let y_data = TimeSeries::from_values(y).with_domain(Domain::Discrete);

        let config = MutualInformationConfig {
            num_bins_x: None,
            num_bins_y: None,
            log_base: LogBase::E,
            normalize: false,
        };
        let analyzer = MutualInformation::with_config(config);
        let result = analyzer.calculate(&x_data, &y_data).unwrap();

        let mi = result.metrics.get("mutual_information").unwrap();

        // For binary X=Y in nats: I(X;X) = H(X) = ln(2)
        assert_relative_eq!(*mi, 2.0_f64.ln(), epsilon = 0.01);
    }
}
