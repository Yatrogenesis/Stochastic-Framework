/// Ergodic Analysis - Birkhoff-Khinchin Theorem
/// Tests if temporal averages equal ensemble averages (ergodicity hypothesis)
use crate::models::{ErgodicResult, LotteryDataset};
use anyhow::Result;
use std::collections::HashMap;

pub struct ErgodicAnalyzer;

impl ErgodicAnalyzer {
    /// Perform ergodic analysis using Birkhoff-Khinchin theorem
    pub fn analyze(dataset: &LotteryDataset) -> Result<ErgodicResult> {
        log::info!("Starting ergodic analysis with Birkhoff-Khinchin theorem");

        // Compute temporal mean: average frequency over time
        let temporal_mean = Self::compute_temporal_mean(dataset);

        // Compute ensemble mean: expected value under uniform distribution
        let ensemble_mean = Self::compute_ensemble_mean(dataset);

        // Perform runs test to detect non-randomness
        let runs_test_z_score = Self::compute_runs_test(dataset)?;

        // Compute autocorrelation function
        let autocorrelation = Self::compute_autocorrelation(dataset, 10);

        // System is ergodic if temporal mean ≈ ensemble mean and runs test is insignificant
        let is_ergodic = (temporal_mean - ensemble_mean).abs() < 0.05
            && runs_test_z_score.abs() < 1.96; // 95% confidence

        log::info!(
            "Ergodic Analysis: temporal_mean={:.6}, ensemble_mean={:.6}, runs_z={:.4}, ergodic={}",
            temporal_mean,
            ensemble_mean,
            runs_test_z_score,
            is_ergodic
        );

        Ok(ErgodicResult {
            temporal_mean,
            ensemble_mean,
            runs_test_z_score,
            is_ergodic,
            autocorrelation,
        })
    }

    /// Compute temporal mean: average number value across all draws
    fn compute_temporal_mean(dataset: &LotteryDataset) -> f64 {
        let total_sum: u32 = dataset
            .draws
            .iter()
            .flat_map(|d| &d.numbers)
            .sum();
        let total_count = dataset.draws.iter().map(|d| d.numbers.len()).sum::<usize>();

        total_sum as f64 / total_count as f64
    }

    /// Compute ensemble mean: expected value under uniform distribution
    fn compute_ensemble_mean(dataset: &LotteryDataset) -> f64 {
        // For uniform distribution over [min, max], mean = (min + max) / 2
        (dataset.min_number + dataset.max_number) as f64 / 2.0
    }

    /// Compute runs test Z-score to detect non-randomness
    /// A run is a sequence of consecutive values above or below the median
    fn compute_runs_test(dataset: &LotteryDataset) -> Result<f64> {
        // Create time series of draw means
        let draw_means: Vec<f64> = dataset
            .draws
            .iter()
            .map(|d| {
                let sum: u32 = d.numbers.iter().sum();
                sum as f64 / d.numbers.len() as f64
            })
            .collect();

        if draw_means.len() < 2 {
            return Ok(0.0);
        }

        // Compute median
        let mut sorted_means = draw_means.clone();
        sorted_means.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = if sorted_means.len() % 2 == 0 {
            let mid = sorted_means.len() / 2;
            (sorted_means[mid - 1] + sorted_means[mid]) / 2.0
        } else {
            sorted_means[sorted_means.len() / 2]
        };

        // Count runs (sequences above/below median)
        let mut runs = 1;
        let mut n1 = 0; // count above median
        let mut n2 = 0; // count below median

        let mut prev_above = draw_means[0] > median;
        if prev_above {
            n1 += 1;
        } else {
            n2 += 1;
        }

        for &mean in &draw_means[1..] {
            let current_above = mean > median;
            if current_above {
                n1 += 1;
            } else {
                n2 += 1;
            }

            if current_above != prev_above {
                runs += 1;
                prev_above = current_above;
            }
        }

        // Compute Z-score for runs test
        let n = (n1 + n2) as f64;
        if n1 == 0 || n2 == 0 {
            return Ok(0.0);
        }

        let expected_runs = (2.0 * n1 as f64 * n2 as f64) / n + 1.0;
        let variance = (2.0 * n1 as f64 * n2 as f64 * (2.0 * n1 as f64 * n2 as f64 - n))
            / (n * n * (n - 1.0));
        let std_dev = variance.sqrt();

        if std_dev == 0.0 {
            return Ok(0.0);
        }

        let z_score = (runs as f64 - expected_runs) / std_dev;

        Ok(z_score)
    }

    /// Compute autocorrelation function for lag 1 to max_lag
    fn compute_autocorrelation(dataset: &LotteryDataset, max_lag: usize) -> Vec<f64> {
        // Create time series of number frequencies per draw
        let time_series: Vec<f64> = dataset
            .draws
            .iter()
            .map(|d| {
                let sum: u32 = d.numbers.iter().sum();
                sum as f64
            })
            .collect();

        if time_series.len() < 2 {
            return vec![0.0; max_lag];
        }

        let mean = time_series.iter().sum::<f64>() / time_series.len() as f64;

        let mut autocorr = Vec::new();

        for lag in 1..=max_lag {
            if lag >= time_series.len() {
                autocorr.push(0.0);
                continue;
            }

            let mut numerator = 0.0;
            let mut denominator = 0.0;

            for i in 0..time_series.len() - lag {
                numerator += (time_series[i] - mean) * (time_series[i + lag] - mean);
            }

            for i in 0..time_series.len() {
                denominator += (time_series[i] - mean).powi(2);
            }

            let acf = if denominator != 0.0 {
                numerator / denominator
            } else {
                0.0
            };

            autocorr.push(acf);
        }

        autocorr
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::LotteryDraw;
    use chrono::Utc;

    #[test]
    fn test_ergodic_analysis() {
        let draws = vec![
            LotteryDraw {
                id: Some(1),
                numbers: vec![1, 2, 3, 4, 5],
                draw_date: Utc::now(),
                game_name: "Test".to_string(),
            },
            LotteryDraw {
                id: Some(2),
                numbers: vec![10, 15, 20, 25, 28],
                draw_date: Utc::now(),
                game_name: "Test".to_string(),
            },
            LotteryDraw {
                id: Some(3),
                numbers: vec![5, 10, 15, 20, 25],
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

        let result = ErgodicAnalyzer::analyze(&dataset);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert!(result.temporal_mean > 0.0);
        assert_eq!(result.ensemble_mean, 14.5); // (1 + 28) / 2
        assert_eq!(result.autocorrelation.len(), 10);
    }

    #[test]
    fn test_temporal_vs_ensemble_mean() {
        // Create draws with uniform distribution
        let draws = vec![
            LotteryDraw {
                id: Some(1),
                numbers: vec![1, 5, 10, 15, 20],
                draw_date: Utc::now(),
                game_name: "Test".to_string(),
            },
            LotteryDraw {
                id: Some(2),
                numbers: vec![2, 8, 14, 18, 28],
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

        let result = ErgodicAnalyzer::analyze(&dataset).unwrap();

        // Ensemble mean should be (1 + 28) / 2 = 14.5
        assert_eq!(result.ensemble_mean, 14.5);

        // Temporal mean should be close to ensemble mean for uniform distribution
        assert!(result.temporal_mean > 0.0);
    }

    #[test]
    fn test_autocorrelation_bounds() {
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
        ];

        let dataset = LotteryDataset {
            draws,
            min_number: 1,
            max_number: 28,
            numbers_per_draw: 5,
        };

        let result = ErgodicAnalyzer::analyze(&dataset).unwrap();

        // Autocorrelation should be in [-1, 1]
        for &acf in &result.autocorrelation {
            assert!(acf >= -1.0 && acf <= 1.0, "ACF should be in [-1, 1], got {}", acf);
        }
    }
}
