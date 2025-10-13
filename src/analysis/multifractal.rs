/// Multifractal Analysis - Detrended Fluctuation Analysis (DFA)
/// Computes Hurst exponent, fractal dimension, and normalized entropy
use crate::models::{LotteryDataset, MultifractalResult};
use anyhow::Result;

pub struct MultifractalAnalyzer;

impl MultifractalAnalyzer {
    /// Perform multifractal analysis using DFA
    pub fn analyze(dataset: &LotteryDataset) -> Result<MultifractalResult> {
        log::info!("Starting multifractal analysis with DFA");

        // Create time series from draw data
        let time_series = Self::create_time_series(dataset);

        // Compute Hurst exponent using DFA
        let hurst_exponent = Self::compute_hurst_exponent(&time_series);

        // Compute fractal dimension: D = 2 - H
        let fractal_dimension = 2.0 - hurst_exponent;

        // Compute normalized entropy
        let normalized_entropy = Self::compute_normalized_entropy(dataset);

        // Interpret persistence
        let is_persistent = hurst_exponent > 0.5;
        let interpretation = Self::interpret_hurst(hurst_exponent);

        log::info!(
            "Multifractal Analysis: H={:.4}, D={:.4}, entropy={:.4}, persistent={}",
            hurst_exponent,
            fractal_dimension,
            normalized_entropy,
            is_persistent
        );

        Ok(MultifractalResult {
            hurst_exponent,
            fractal_dimension,
            normalized_entropy,
            is_persistent,
            interpretation,
        })
    }

    /// Create time series from lottery draws
    fn create_time_series(dataset: &LotteryDataset) -> Vec<f64> {
        dataset
            .draws
            .iter()
            .map(|draw| {
                let sum: u32 = draw.numbers.iter().sum();
                sum as f64 / draw.numbers.len() as f64
            })
            .collect()
    }

    /// Compute Hurst exponent using simplified DFA
    /// H > 0.5: persistent (trend-reinforcing)
    /// H = 0.5: random walk (Brownian motion)
    /// H < 0.5: anti-persistent (mean-reverting)
    fn compute_hurst_exponent(time_series: &[f64]) -> f64 {
        if time_series.len() < 4 {
            return 0.5; // Default to random walk
        }

        // Compute cumulative sum (integrate the series)
        let mean = time_series.iter().sum::<f64>() / time_series.len() as f64;
        let mut cumsum = Vec::new();
        let mut sum = 0.0;
        for &val in time_series {
            sum += val - mean;
            cumsum.push(sum);
        }

        // Use multiple window sizes for DFA
        let window_sizes: Vec<usize> = vec![4, 8, 16]
            .into_iter()
            .filter(|&w| w < time_series.len() / 2)
            .collect();

        if window_sizes.is_empty() {
            return 0.5;
        }

        let mut log_scales = Vec::new();
        let mut log_fluctuations = Vec::new();

        for &window_size in &window_sizes {
            let fluctuation = Self::compute_fluctuation(&cumsum, window_size);
            if fluctuation > 0.0 {
                log_scales.push((window_size as f64).ln());
                log_fluctuations.push(fluctuation.ln());
            }
        }

        if log_scales.len() < 2 {
            return 0.5;
        }

        // Compute Hurst exponent as slope of log-log plot
        let hurst = Self::compute_slope(&log_scales, &log_fluctuations);

        // Clamp to reasonable range [0, 1]
        hurst.max(0.0).min(1.0)
    }

    /// Compute fluctuation for a given window size
    fn compute_fluctuation(cumsum: &[f64], window_size: usize) -> f64 {
        let n_windows = cumsum.len() / window_size;
        if n_windows == 0 {
            return 0.0;
        }

        let mut total_variance = 0.0;

        for i in 0..n_windows {
            let start = i * window_size;
            let end = start + window_size;
            let window = &cumsum[start..end];

            // Fit linear trend to window
            let (slope, intercept) = Self::fit_linear(window);

            // Compute variance of detrended data
            let mut variance = 0.0;
            for (j, &val) in window.iter().enumerate() {
                let trend = slope * j as f64 + intercept;
                variance += (val - trend).powi(2);
            }
            variance /= window_size as f64;
            total_variance += variance;
        }

        (total_variance / n_windows as f64).sqrt()
    }

    /// Fit linear trend using least squares
    fn fit_linear(data: &[f64]) -> (f64, f64) {
        let n = data.len() as f64;
        if n < 2.0 {
            return (0.0, data.get(0).copied().unwrap_or(0.0));
        }

        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xy = 0.0;
        let mut sum_xx = 0.0;

        for (i, &y) in data.iter().enumerate() {
            let x = i as f64;
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_xx += x * x;
        }

        let denominator = n * sum_xx - sum_x * sum_x;
        if denominator.abs() < 1e-10 {
            return (0.0, sum_y / n);
        }

        let slope = (n * sum_xy - sum_x * sum_y) / denominator;
        let intercept = (sum_y - slope * sum_x) / n;

        (slope, intercept)
    }

    /// Compute slope of linear regression
    fn compute_slope(x: &[f64], y: &[f64]) -> f64 {
        let (slope, _) = Self::fit_linear_xy(x, y);
        slope
    }

    /// Fit linear trend for two arrays
    fn fit_linear_xy(x: &[f64], y: &[f64]) -> (f64, f64) {
        let n = x.len().min(y.len()) as f64;
        if n < 2.0 {
            return (0.0, 0.0);
        }

        let mean_x = x.iter().take(n as usize).sum::<f64>() / n;
        let mean_y = y.iter().take(n as usize).sum::<f64>() / n;

        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for i in 0..(n as usize) {
            let dx = x[i] - mean_x;
            let dy = y[i] - mean_y;
            numerator += dx * dy;
            denominator += dx * dx;
        }

        if denominator.abs() < 1e-10 {
            return (0.0, mean_y);
        }

        let slope = numerator / denominator;
        let intercept = mean_y - slope * mean_x;

        (slope, intercept)
    }

    /// Compute normalized Shannon entropy
    fn compute_normalized_entropy(dataset: &LotteryDataset) -> f64 {
        use std::collections::HashMap;

        let mut frequency_map = HashMap::new();
        let mut total_count = 0;

        for draw in &dataset.draws {
            for &number in &draw.numbers {
                *frequency_map.entry(number).or_insert(0) += 1;
                total_count += 1;
            }
        }

        if total_count == 0 {
            return 0.0;
        }

        // Compute Shannon entropy: H = -Σ p_i log₂(p_i)
        let mut entropy = 0.0;
        for &count in frequency_map.values() {
            if count > 0 {
                let p = count as f64 / total_count as f64;
                entropy -= p * p.log2();
            }
        }

        // Normalize by maximum entropy (log₂(number of categories))
        let max_entropy = ((dataset.max_number - dataset.min_number + 1) as f64).log2();
        if max_entropy > 0.0 {
            entropy / max_entropy
        } else {
            0.0
        }
    }

    /// Interpret Hurst exponent
    fn interpret_hurst(h: f64) -> String {
        if h > 0.6 {
            format!("Strongly persistent (H={:.3}): Past trends likely to continue", h)
        } else if h > 0.5 {
            format!("Weakly persistent (H={:.3}): Slight trend reinforcement", h)
        } else if h > 0.4 {
            format!("Weakly anti-persistent (H={:.3}): Slight mean reversion", h)
        } else {
            format!("Strongly anti-persistent (H={:.3}): Strong mean reversion", h)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::LotteryDraw;
    use chrono::Utc;

    #[test]
    fn test_multifractal_analysis() {
        let draws = vec![
            LotteryDraw {
                id: Some(1),
                numbers: vec![1, 2, 3, 4, 5],
                draw_date: Utc::now(),
                game_name: "Test".to_string(),
            },
            LotteryDraw {
                id: Some(2),
                numbers: vec![6, 7, 8, 9, 10],
                draw_date: Utc::now(),
                game_name: "Test".to_string(),
            },
            LotteryDraw {
                id: Some(3),
                numbers: vec![11, 12, 13, 14, 15],
                draw_date: Utc::now(),
                game_name: "Test".to_string(),
            },
            LotteryDraw {
                id: Some(4),
                numbers: vec![16, 17, 18, 19, 20],
                draw_date: Utc::now(),
                game_name: "Test".to_string(),
            },
        ];

        let dataset = LotteryDataset {
            draws,
            min_number: 1,
            max_number: 28,
            numbers_per_draw: 5,
        };

        let result = MultifractalAnalyzer::analyze(&dataset);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert!(result.hurst_exponent >= 0.0 && result.hurst_exponent <= 1.0);
        assert!(result.fractal_dimension >= 1.0 && result.fractal_dimension <= 2.0);
        assert!(result.normalized_entropy >= 0.0 && result.normalized_entropy <= 1.0);
        assert!(!result.interpretation.is_empty());
    }

    #[test]
    fn test_hurst_exponent_bounds() {
        let time_series = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let h = MultifractalAnalyzer::compute_hurst_exponent(&time_series);
        assert!(h >= 0.0 && h <= 1.0);
    }

    #[test]
    fn test_fractal_dimension_relationship() {
        let draws = vec![
            LotteryDraw {
                id: Some(1),
                numbers: vec![1, 5, 10, 15, 20],
                draw_date: Utc::now(),
                game_name: "Test".to_string(),
            },
            LotteryDraw {
                id: Some(2),
                numbers: vec![2, 6, 11, 16, 21],
                draw_date: Utc::now(),
                game_name: "Test".to_string(),
            },
        ];

        let dataset = LotteryDataset {
            draws,
            min_number: 1,
            max_number: 28,
            numbers_per_draw: 5,
        };

        let result = MultifractalAnalyzer::analyze(&dataset).unwrap();

        // D = 2 - H
        let expected_d = 2.0 - result.hurst_exponent;
        assert!((result.fractal_dimension - expected_d).abs() < 1e-6);
    }

    #[test]
    fn test_normalized_entropy() {
        let draws = vec![
            LotteryDraw {
                id: Some(1),
                numbers: vec![1, 1, 1, 1, 1], // Low entropy
                draw_date: Utc::now(),
                game_name: "Test".to_string(),
            },
        ];

        let dataset = LotteryDataset {
            draws,
            min_number: 1,
            max_number: 28,
            numbers_per_draw: 5,
        };

        let entropy = MultifractalAnalyzer::compute_normalized_entropy(&dataset);
        assert!(entropy >= 0.0 && entropy <= 1.0);
        // All same number = 0 entropy
        assert!(entropy < 0.01);
    }

    #[test]
    fn test_fit_linear() {
        // Perfect line: y = 2x + 1
        let data = vec![1.0, 3.0, 5.0, 7.0, 9.0];
        let (slope, intercept) = MultifractalAnalyzer::fit_linear(&data);
        assert!((slope - 2.0).abs() < 0.01);
        assert!((intercept - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_interpret_hurst() {
        let persistent = MultifractalAnalyzer::interpret_hurst(0.7);
        assert!(persistent.contains("persistent"));

        let random = MultifractalAnalyzer::interpret_hurst(0.5);
        assert!(random.contains("persistent"));

        let anti_persistent = MultifractalAnalyzer::interpret_hurst(0.3);
        assert!(anti_persistent.contains("anti-persistent"));
    }
}
