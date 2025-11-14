//! Detrended Fluctuation Analysis (DFA) implementation
//!
//! DFA quantifies long-range correlations in time series by analyzing how
//! fluctuations scale with window size. The Hurst exponent H characterizes
//! the persistence or anti-persistence of the process.
//!
//! # Mathematical Foundation
//!
//! **DFA Algorithm:**
//!
//! 1. **Integration**: Y(i) = Σₖ₌₁ⁱ [x(k) - x̄]
//! 2. **Segmentation**: Divide into non-overlapping boxes of size n
//! 3. **Local Detrending**: Fit polynomial in each box, compute residuals
//! 4. **Fluctuation**: F(n) = √(1/N Σᵢ[Y(i) - yₙ(i)]²)
//! 5. **Scaling**: log F(n) vs log n → slope = H (Hurst exponent)
//!
//! **Hurst Exponent Interpretation:**
//! - H > 0.5: Persistent (positive correlations, trending)
//! - H = 0.5: Random walk (uncorrelated, Brownian motion)
//! - H < 0.5: Anti-persistent (negative correlations, mean-reverting)
//!
//! **Fractal Dimension:**
//! ```text
//! D = 2 - H
//! ```
//!
//! # References
//!
//! - Peng, C.K., et al. (1994). "Mosaic organization of DNA nucleotides"
//! - Kantelhardt, J.W., et al. (2001). "Detecting long-range correlations with detrended fluctuation analysis"
//! - Hurst, H.E. (1951). "Long-term storage capacity of reservoirs"
//! - Mandelbrot, B.B., & Van Ness, J.W. (1968). "Fractional Brownian motions"
//! - Hardstone, R., et al. (2012). "Detrended fluctuation analysis: a scale-free view on neuronal oscillations"

use stochastic_core::{AnalysisResult, StochasticAnalyzer, StochasticError, TimeSeries};
use ndarray::{Array1, ArrayView1};

/// Detrending order for DFA
#[derive(Debug, Clone, Copy)]
pub enum DetrendOrder {
    /// Linear detrending (DFA1)
    Linear,
    /// Quadratic detrending (DFA2)
    Quadratic,
    /// Cubic detrending (DFA3)
    Cubic,
}

/// Configuration for DFA
#[derive(Debug, Clone)]
pub struct DFAConfig {
    /// Detrending polynomial order
    pub detrend_order: DetrendOrder,
    /// Minimum box size (default: 4)
    pub min_box_size: usize,
    /// Maximum box size (default: N/4)
    pub max_box_size: Option<usize>,
    /// Number of box sizes to test
    pub num_box_sizes: usize,
}

impl Default for DFAConfig {
    fn default() -> Self {
        Self {
            detrend_order: DetrendOrder::Linear,
            min_box_size: 4,
            max_box_size: None,
            num_box_sizes: 20,
        }
    }
}

/// Detrended Fluctuation Analysis analyzer
pub struct DFA {
    config: DFAConfig,
}

impl DFA {
    /// Create new DFA analyzer with default configuration
    pub fn new() -> Self {
        Self {
            config: DFAConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: DFAConfig) -> Self {
        Self { config }
    }

    /// Set detrending order
    pub fn with_detrend_order(mut self, order: DetrendOrder) -> Self {
        self.config.detrend_order = order;
        self
    }

    /// Set minimum box size
    pub fn with_min_box_size(mut self, size: usize) -> Self {
        self.config.min_box_size = size.max(2);
        self
    }

    /// Set number of box sizes
    pub fn with_num_box_sizes(mut self, num: usize) -> Self {
        self.config.num_box_sizes = num.max(5);
        self
    }

    /// Integrate time series (cumulative sum after removing mean)
    fn integrate(&self, values: &[f64]) -> Array1<f64> {
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let mut integrated = Array1::zeros(values.len());
        let mut cumsum = 0.0;

        for (i, &val) in values.iter().enumerate() {
            cumsum += val - mean;
            integrated[i] = cumsum;
        }

        integrated
    }

    /// Fit polynomial to segment and compute residuals
    fn detrend_segment(&self, segment: ArrayView1<f64>) -> f64 {
        let n = segment.len();
        if n == 0 {
            return 0.0;
        }

        let order = match self.config.detrend_order {
            DetrendOrder::Linear => 1,
            DetrendOrder::Quadratic => 2,
            DetrendOrder::Cubic => 3,
        };

        // Simple polynomial fit using least squares
        let (coeffs, _) = self.polynomial_fit(segment, order);

        // Compute residuals
        let mut residual_sum = 0.0;
        for (i, &y) in segment.iter().enumerate() {
            let x = i as f64;
            let y_fit = self.poly_eval(&coeffs, x);
            let residual = y - y_fit;
            residual_sum += residual * residual;
        }

        residual_sum
    }

    /// Polynomial fit using least squares
    fn polynomial_fit(&self, y: ArrayView1<f64>, order: usize) -> (Vec<f64>, f64) {
        let n = y.len();
        let m = order + 1;

        // Build design matrix X (Vandermonde)
        let mut x_matrix = vec![vec![0.0; m]; n];
        for i in 0..n {
            let x = i as f64;
            for j in 0..m {
                x_matrix[i][j] = x.powi(j as i32);
            }
        }

        // Normal equations: X^T X β = X^T y
        let mut xtx = vec![vec![0.0; m]; m];
        let mut xty = vec![0.0; m];

        for i in 0..m {
            for j in 0..m {
                for k in 0..n {
                    xtx[i][j] += x_matrix[k][i] * x_matrix[k][j];
                }
            }
            for k in 0..n {
                xty[i] += x_matrix[k][i] * y[k];
            }
        }

        // Solve using Gaussian elimination
        let coeffs = self.solve_linear_system(&xtx, &xty);

        // Compute R²
        let y_mean = y.iter().sum::<f64>() / n as f64;
        let mut ss_tot = 0.0;
        let mut ss_res = 0.0;

        for (i, &y_val) in y.iter().enumerate() {
            let x = i as f64;
            let y_fit = self.poly_eval(&coeffs, x);
            ss_tot += (y_val - y_mean).powi(2);
            ss_res += (y_val - y_fit).powi(2);
        }

        let r_squared = if ss_tot > 0.0 {
            1.0 - (ss_res / ss_tot)
        } else {
            0.0
        };

        (coeffs, r_squared)
    }

    /// Evaluate polynomial at x
    fn poly_eval(&self, coeffs: &[f64], x: f64) -> f64 {
        coeffs.iter()
            .enumerate()
            .map(|(i, &c)| c * x.powi(i as i32))
            .sum()
    }

    /// Solve linear system Ax = b using Gaussian elimination
    fn solve_linear_system(&self, a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
        let n = b.len();
        let mut aug = vec![vec![0.0; n + 1]; n];

        // Create augmented matrix
        for i in 0..n {
            for j in 0..n {
                aug[i][j] = a[i][j];
            }
            aug[i][n] = b[i];
        }

        // Forward elimination
        for i in 0..n {
            // Partial pivoting
            let mut max_row = i;
            for k in i + 1..n {
                if aug[k][i].abs() > aug[max_row][i].abs() {
                    max_row = k;
                }
            }
            aug.swap(i, max_row);

            // Eliminate column
            for k in i + 1..n {
                if aug[i][i].abs() < 1e-10 {
                    continue;
                }
                let factor = aug[k][i] / aug[i][i];
                for j in i..=n {
                    aug[k][j] -= factor * aug[i][j];
                }
            }
        }

        // Back substitution
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            if aug[i][i].abs() < 1e-10 {
                x[i] = 0.0;
                continue;
            }

            x[i] = aug[i][n];
            for j in i + 1..n {
                x[i] -= aug[i][j] * x[j];
            }
            x[i] /= aug[i][i];
        }

        x
    }

    /// Compute fluctuation function F(n) for given box size
    fn fluctuation(&self, integrated: &Array1<f64>, box_size: usize) -> f64 {
        let n = integrated.len();
        let num_boxes = n / box_size;

        if num_boxes == 0 {
            return 0.0;
        }

        let mut total_variance = 0.0;
        let mut valid_boxes = 0;

        // Forward direction
        for i in 0..num_boxes {
            let start = i * box_size;
            let end = (i + 1) * box_size;
            let segment = integrated.slice(ndarray::s![start..end]);
            total_variance += self.detrend_segment(segment);
            valid_boxes += 1;
        }

        // Backward direction (for remaining data)
        if n % box_size >= box_size / 2 {
            let _remaining = n % box_size;
            let start = n - box_size;
            let segment = integrated.slice(ndarray::s![start..]);
            total_variance += self.detrend_segment(segment);
            valid_boxes += 1;
        }

        if valid_boxes == 0 {
            return 0.0;
        }

        (total_variance / valid_boxes as f64).sqrt()
    }

    /// Generate logarithmically-spaced box sizes
    fn generate_box_sizes(&self, n: usize) -> Vec<usize> {
        let min_size = self.config.min_box_size;
        let max_size = self.config.max_box_size.unwrap_or(n / 4).min(n / 2);

        if max_size <= min_size {
            return vec![min_size];
        }

        let log_min = (min_size as f64).ln();
        let log_max = (max_size as f64).ln();
        let step = (log_max - log_min) / (self.config.num_box_sizes - 1) as f64;

        (0..self.config.num_box_sizes)
            .map(|i| {
                let log_size = log_min + i as f64 * step;
                log_size.exp().round() as usize
            })
            .filter(|&size| size >= min_size && size <= max_size)
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect::<Vec<_>>()
    }

    /// Perform DFA and compute Hurst exponent
    pub fn analyze_dfa(&self, data: &TimeSeries) -> Result<AnalysisResult, StochasticError> {
        self.validate(data)?;

        let values = data.values();
        let n = values.len();

        // Step 1: Integrate
        let integrated = self.integrate(&values);

        // Step 2: Generate box sizes
        let mut box_sizes = self.generate_box_sizes(n);
        box_sizes.sort_unstable();

        if box_sizes.len() < 3 {
            return Err(StochasticError::validation(
                "Insufficient box sizes for DFA analysis. Try larger dataset or adjust parameters."
            ));
        }

        // Step 3: Compute fluctuations for each box size
        let mut log_boxes = Vec::new();
        let mut log_fluctuations = Vec::new();

        for &box_size in &box_sizes {
            let f = self.fluctuation(&integrated, box_size);
            if f > 0.0 {
                log_boxes.push((box_size as f64).ln());
                log_fluctuations.push(f.ln());
            }
        }

        if log_boxes.len() < 3 {
            return Err(StochasticError::validation(
                "Insufficient valid fluctuation measurements for DFA."
            ));
        }

        // Step 4: Fit line to log-log plot → slope = Hurst exponent
        let _log_boxes_arr = Array1::from_vec(log_boxes.clone());
        let log_fluct_arr = Array1::from_vec(log_fluctuations.clone());

        let (coeffs, r_squared) = self.polynomial_fit(log_fluct_arr.view(), 1);
        let hurst_exponent = coeffs[1]; // Slope

        // Fractal dimension
        let fractal_dimension = 2.0 - hurst_exponent;

        // Interpret scaling behavior
        let scaling_behavior = if hurst_exponent > 0.6 {
            "Strongly persistent (trending, long-range positive correlations)"
        } else if hurst_exponent > 0.5 {
            "Weakly persistent (slight trending)"
        } else if (hurst_exponent - 0.5).abs() < 0.05 {
            "Random walk (uncorrelated, Brownian motion)"
        } else if hurst_exponent > 0.4 {
            "Weakly anti-persistent (slight mean-reversion)"
        } else {
            "Strongly anti-persistent (mean-reverting, oscillatory)"
        };

        let interpretation = format!(
            "Hurst exponent H = {:.3} indicates {}. \
             Fractal dimension D = {:.3}. \
             Scaling fit quality R² = {:.3}.",
            hurst_exponent, scaling_behavior, fractal_dimension, r_squared
        );

        let result = AnalysisResult::new(self.name())
            .with_metric("hurst_exponent", hurst_exponent)
            .with_metric("fractal_dimension", fractal_dimension)
            .with_metric("scaling_r_squared", r_squared)
            .with_metric("num_box_sizes", box_sizes.len() as f64)
            .with_metric("min_box_size", box_sizes[0] as f64)
            .with_metric("max_box_size", box_sizes[box_sizes.len() - 1] as f64)
            .with_metric("sample_size", n as f64)
            .with_interpretation(interpretation)
            .with_metadata("detrend_order", format!("{:?}", self.config.detrend_order))
            .with_metadata("scaling_behavior", scaling_behavior);

        Ok(result)
    }
}

impl Default for DFA {
    fn default() -> Self {
        Self::new()
    }
}

impl StochasticAnalyzer for DFA {
    fn name(&self) -> &str {
        "Detrended Fluctuation Analysis (DFA)"
    }

    fn analyze(&self, data: &TimeSeries) -> Result<AnalysisResult, StochasticError> {
        self.analyze_dfa(data)
    }

    fn required_sample_size(&self) -> usize {
        // DFA requires sufficient data for multiple box sizes
        // Minimum: 100 points for meaningful analysis
        100
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
    use approx::assert_relative_eq;

    #[test]
    fn test_white_noise_hurst() {
        // White noise should have H ≈ 0.5
        let values: Vec<f64> = (0..1000)
            .map(|i| ((i as f64 * 0.123).sin() * (i as f64 * 0.789).cos()))
            .collect();
        let data = TimeSeries::from_values(values);

        let dfa = DFA::new();
        let result = dfa.analyze(&data).unwrap();

        let hurst = result.metrics.get("hurst_exponent").unwrap();

        // Should be close to 0.5 for uncorrelated noise
        assert!(*hurst > 0.3 && *hurst < 0.7,
                "Hurst exponent for noise should be ~0.5, got {}", hurst);
    }

    #[test]
    fn test_trending_data_hurst() {
        // Trending data should have H > 0.5
        let values: Vec<f64> = (0..1000)
            .map(|i| i as f64 + (i as f64 * 0.01).sin())
            .collect();
        let data = TimeSeries::from_values(values);

        let dfa = DFA::new();
        let result = dfa.analyze(&data).unwrap();

        let hurst = result.metrics.get("hurst_exponent").unwrap();

        // Trending data should have H > 0.5
        assert!(*hurst > 0.5, "Hurst for trending data should be > 0.5, got {}", hurst);
    }

    #[test]
    fn test_fractal_dimension() {
        let values: Vec<f64> = (0..500).map(|i| (i as f64 * 0.1).sin()).collect();
        let data = TimeSeries::from_values(values);

        let dfa = DFA::new();
        let result = dfa.analyze(&data).unwrap();

        let hurst = result.metrics.get("hurst_exponent").unwrap();
        let fractal_dim = result.metrics.get("fractal_dimension").unwrap();

        // D = 2 - H
        assert_relative_eq!(*fractal_dim, 2.0 - hurst, epsilon = 1e-10);
    }

    #[test]
    fn test_detrending_orders() {
        let values: Vec<f64> = (0..500).map(|i| i as f64).collect();
        let data = TimeSeries::from_values(values);

        let dfa1 = DFA::new().with_detrend_order(DetrendOrder::Linear);
        let dfa2 = DFA::new().with_detrend_order(DetrendOrder::Quadratic);

        let result1 = dfa1.analyze(&data).unwrap();
        let result2 = dfa2.analyze(&data).unwrap();

        // Both should produce valid Hurst exponents
        let h1 = result1.metrics.get("hurst_exponent").unwrap();
        let h2 = result2.metrics.get("hurst_exponent").unwrap();

        assert!(*h1 > 0.0 && *h1 < 2.0);
        assert!(*h2 > 0.0 && *h2 < 2.0);
    }

    #[test]
    fn test_box_sizes_generation() {
        let values: Vec<f64> = (0..1000).map(|i| i as f64).collect();
        let data = TimeSeries::from_values(values);

        let dfa = DFA::new()
            .with_min_box_size(10)
            .with_num_box_sizes(15);

        let result = dfa.analyze(&data).unwrap();

        let num_boxes = result.metrics.get("num_box_sizes").unwrap();
        let min_box = result.metrics.get("min_box_size").unwrap();
        let max_box = result.metrics.get("max_box_size").unwrap();

        assert!(*num_boxes >= 3.0);
        assert_relative_eq!(*min_box, 10.0, epsilon = 1.0);
        assert!(*max_box <= 1000.0 / 2.0);
    }

    #[test]
    fn test_insufficient_data() {
        let data = TimeSeries::from_values(vec![1.0; 50]);
        let dfa = DFA::new();

        let result = dfa.analyze(&data);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("insufficient"));
    }

    #[test]
    fn test_r_squared_quality() {
        let values: Vec<f64> = (0..1000).map(|i| (i as f64 * 0.05).sin()).collect();
        let data = TimeSeries::from_values(values);

        let dfa = DFA::new();
        let result = dfa.analyze(&data).unwrap();

        let r_squared = result.metrics.get("scaling_r_squared").unwrap();

        // R² should be reasonably high for good fit
        assert!(*r_squared > 0.5, "R² should indicate good fit, got {}", r_squared);
    }
}
