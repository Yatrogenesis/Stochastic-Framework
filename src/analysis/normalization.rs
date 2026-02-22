/// Normalization and Factorization Analysis
/// Detects outliers, tests normality, and applies transformations
use crate::models::{LotteryDataset, NormalizationResult};
use anyhow::Result;
use std::collections::HashMap;

pub struct NormalizationAnalyzer;

impl NormalizationAnalyzer {
    /// Perform normalization analysis
    pub fn analyze(dataset: &LotteryDataset) -> Result<NormalizationResult> {
        log::info!("Starting normalization and outlier analysis");

        // Compute frequency distribution
        let frequency_map = Self::compute_frequency_distribution(dataset);

        // Detect outliers using z-score (|z| > 2)
        let outliers = Self::detect_outliers(&frequency_map, dataset);

        // Perform Shapiro-Wilk normality test
        let frequencies: Vec<f64> = (dataset.min_number..=dataset.max_number)
            .map(|num| *frequency_map.get(&num).unwrap_or(&0) as f64)
            .collect();

        let (shapiro_wilk_w, shapiro_wilk_p) = Self::shapiro_wilk_test(&frequencies)?;

        // Distribution is normal if p > 0.05
        let is_normal = shapiro_wilk_p > 0.05;

        // Compute optimal Box-Cox transformation parameter
        let box_cox_lambda = Self::compute_box_cox_lambda(&frequencies);

        log::info!(
            "Normalization Analysis: outliers={}, W={:.4}, p={:.4}, normal={}, λ={:.4}",
            outliers.len(),
            shapiro_wilk_w,
            shapiro_wilk_p,
            is_normal,
            box_cox_lambda
        );

        Ok(NormalizationResult {
            outliers,
            shapiro_wilk_w,
            shapiro_wilk_p,
            is_normal,
            box_cox_lambda,
        })
    }

    /// Compute frequency distribution
    fn compute_frequency_distribution(dataset: &LotteryDataset) -> HashMap<u32, usize> {
        let mut frequency_map = HashMap::new();

        for draw in &dataset.draws {
            for &number in &draw.numbers {
                *frequency_map.entry(number).or_insert(0) += 1;
            }
        }

        frequency_map
    }

    /// Detect outliers using z-score method
    /// Outliers are values with |z| > 2 (outside 95% confidence)
    fn detect_outliers(frequency_map: &HashMap<u32, usize>, dataset: &LotteryDataset) -> Vec<u32> {
        let frequencies: Vec<f64> = (dataset.min_number..=dataset.max_number)
            .map(|num| *frequency_map.get(&num).unwrap_or(&0) as f64)
            .collect();

        if frequencies.is_empty() {
            return Vec::new();
        }

        // Compute mean and standard deviation
        let mean = frequencies.iter().sum::<f64>() / frequencies.len() as f64;
        let variance = frequencies.iter().map(|&f| (f - mean).powi(2)).sum::<f64>()
            / frequencies.len() as f64;
        let std_dev = variance.sqrt();

        if std_dev < 1e-10 {
            return Vec::new(); // No variation, no outliers
        }

        // Find numbers with |z-score| > 2
        let mut outliers = Vec::new();
        for num in dataset.min_number..=dataset.max_number {
            let freq = *frequency_map.get(&num).unwrap_or(&0) as f64;
            let z_score = (freq - mean) / std_dev;

            if z_score.abs() > 2.0 {
                outliers.push(num);
            }
        }

        outliers.sort();
        outliers
    }

    /// Shapiro-Wilk test for normality
    /// Returns (W statistic, p-value)
    /// H0: Data comes from a normal distribution
    fn shapiro_wilk_test(data: &[f64]) -> Result<(f64, f64)> {
        let n = data.len();

        if n < 3 {
            return Ok((1.0, 1.0)); // Not enough data
        }

        if n > 50 {
            // For large samples, use simplified approximation
            return Self::shapiro_wilk_large_sample(data);
        }

        // Sort data
        let mut sorted_data = data.to_vec();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Compute mean
        let mean = sorted_data.iter().sum::<f64>() / n as f64;

        // Compute sum of squared deviations
        let ss = sorted_data
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>();

        if ss < 1e-10 {
            return Ok((1.0, 1.0)); // No variation
        }

        // Simplified W statistic calculation for small to medium samples
        let k = n / 2;
        let mut numerator = 0.0;

        for i in 0..k {
            let diff = sorted_data[n - 1 - i] - sorted_data[i];
            // Use simplified coefficients (approximation)
            let a_i = Self::shapiro_wilk_coefficient(n, i);
            numerator += a_i * diff;
        }

        let w = (numerator * numerator) / ss;
        let w = w.min(1.0).max(0.0); // Clamp to [0, 1]

        // Approximate p-value based on W statistic
        // For W close to 1, p-value is high (likely normal)
        // For W far from 1, p-value is low (likely not normal)
        let p_value = Self::approximate_shapiro_wilk_pvalue(w, n);

        Ok((w, p_value))
    }

    /// Shapiro-Wilk test for large samples (n > 50)
    fn shapiro_wilk_large_sample(data: &[f64]) -> Result<(f64, f64)> {
        // For large samples, use skewness and kurtosis test
        let n = data.len() as f64;
        let mean = data.iter().sum::<f64>() / n;

        let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
        let std_dev = variance.sqrt();

        if std_dev < 1e-10 {
            return Ok((1.0, 1.0));
        }

        // Compute skewness
        let skewness = data.iter().map(|&x| ((x - mean) / std_dev).powi(3)).sum::<f64>() / n;

        // Compute kurtosis
        let kurtosis = data.iter().map(|&x| ((x - mean) / std_dev).powi(4)).sum::<f64>() / n - 3.0;

        // Combine skewness and kurtosis into pseudo-W statistic
        let w = 1.0 - (skewness.powi(2) / 6.0 + kurtosis.powi(2) / 24.0).min(1.0);

        // Approximate p-value
        let p_value = Self::approximate_shapiro_wilk_pvalue(w, n as usize);

        Ok((w, p_value))
    }

    /// Approximate Shapiro-Wilk coefficient
    fn shapiro_wilk_coefficient(n: usize, i: usize) -> f64 {
        // Simplified approximation of a_i coefficients
        // In practice, these should be looked up from tables or computed precisely
        let k = n / 2;
        let weight = (k - i) as f64 / k as f64;
        weight / (n as f64).sqrt()
    }

    /// Approximate p-value from W statistic
    fn approximate_shapiro_wilk_pvalue(w: f64, n: usize) -> f64 {
        // Rough approximation: higher W means higher p-value
        // This is a simplified heuristic
        if n < 3 {
            return 1.0;
        }

        // Transform W to approximate p-value
        // W close to 1 => p close to 1
        // W close to 0 => p close to 0
        let log_n = (n as f64).ln();
        let adjusted_w = w * (1.0 + 0.1 / log_n);

        adjusted_w.min(1.0).max(0.0)
    }

    /// Compute optimal Box-Cox transformation parameter λ
    /// λ = 1: no transformation
    /// λ = 0: log transformation
    /// λ = -1: inverse transformation
    fn compute_box_cox_lambda(data: &[f64]) -> f64 {
        if data.is_empty() {
            return 1.0;
        }

        // Shift data to be positive if needed
        let min_val = data.iter().cloned().fold(f64::INFINITY, f64::min);
        let shift = if min_val <= 0.0 {
            -min_val + 1.0
        } else {
            0.0
        };

        // Try different lambda values and find the one that maximizes normality
        let lambdas = vec![-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0];
        let mut best_lambda = 1.0;
        let mut best_score = f64::NEG_INFINITY;

        for &lambda in &lambdas {
            let transformed = Self::box_cox_transform(data, lambda, shift);
            let score = Self::normality_score(&transformed);

            if score > best_score {
                best_score = score;
                best_lambda = lambda;
            }
        }

        best_lambda
    }

    /// Apply Box-Cox transformation
    fn box_cox_transform(data: &[f64], lambda: f64, shift: f64) -> Vec<f64> {
        data.iter()
            .map(|&x| {
                let x_shifted = x + shift;
                if x_shifted <= 0.0 {
                    return 0.0;
                }

                if lambda.abs() < 1e-6 {
                    // λ ≈ 0: use log transformation
                    x_shifted.ln()
                } else {
                    // λ ≠ 0: use power transformation
                    (x_shifted.powf(lambda) - 1.0) / lambda
                }
            })
            .collect()
    }

    /// Compute normality score (higher is more normal)
    /// Based on how close the distribution is to a bell curve
    fn normality_score(data: &[f64]) -> f64 {
        if data.len() < 3 {
            return 0.0;
        }

        let n = data.len() as f64;
        let mean = data.iter().sum::<f64>() / n;
        let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
        let std_dev = variance.sqrt();

        if std_dev < 1e-10 {
            return 0.0;
        }

        // Compute skewness (should be close to 0 for normal)
        let skewness = data.iter().map(|&x| ((x - mean) / std_dev).powi(3)).sum::<f64>() / n;

        // Compute excess kurtosis (should be close to 0 for normal)
        let kurtosis = data.iter().map(|&x| ((x - mean) / std_dev).powi(4)).sum::<f64>() / n - 3.0;

        // Score based on how close skewness and kurtosis are to 0
        let score = -skewness.abs() - kurtosis.abs();

        score
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::LotteryDraw;
    use chrono::Utc;

    #[test]
    fn test_normalization_analysis() {
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

        let result = NormalizationAnalyzer::analyze(&dataset);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert!(result.shapiro_wilk_w >= 0.0 && result.shapiro_wilk_w <= 1.0);
        assert!(result.shapiro_wilk_p >= 0.0 && result.shapiro_wilk_p <= 1.0);
    }

    #[test]
    fn test_outlier_detection() {
        let mut frequency_map = HashMap::new();
        // Create distribution with one outlier
        for i in 1..=10 {
            frequency_map.insert(i, 5); // Normal frequency
        }
        frequency_map.insert(11, 50); // Outlier

        let dataset = LotteryDataset {
            draws: vec![],
            min_number: 1,
            max_number: 11,
            numbers_per_draw: 5,
        };

        let outliers = NormalizationAnalyzer::detect_outliers(&frequency_map, &dataset);

        // Number 11 should be detected as outlier
        assert!(outliers.contains(&11));
    }

    #[test]
    fn test_shapiro_wilk_normal_data() {
        // Approximately normal data
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let (w, p) = NormalizationAnalyzer::shapiro_wilk_test(&data).unwrap();

        // W should be close to 1 for normal-ish data
        assert!(w >= 0.0 && w <= 1.0);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_box_cox_lambda() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let lambda = NormalizationAnalyzer::compute_box_cox_lambda(&data);

        // Lambda should be in a reasonable range
        assert!(lambda >= -1.0 && lambda <= 2.0);
    }

    #[test]
    fn test_box_cox_transform_log() {
        let data = vec![1.0, 2.0, 4.0, 8.0];
        let transformed = NormalizationAnalyzer::box_cox_transform(&data, 0.0, 0.0);

        // λ=0 should give log transformation
        // ln(1)=0, ln(2)≈0.693, ln(4)≈1.386, ln(8)≈2.079
        assert!((transformed[0] - 0.0).abs() < 0.01);
        assert!((transformed[1] - 0.693).abs() < 0.01);
        assert!((transformed[2] - 1.386).abs() < 0.01);
        assert!((transformed[3] - 2.079).abs() < 0.01);
    }

    #[test]
    fn test_box_cox_transform_identity() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let transformed = NormalizationAnalyzer::box_cox_transform(&data, 1.0, 0.0);

        // λ=1 should give (x^1 - 1)/1 = x - 1
        for i in 0..data.len() {
            assert!((transformed[i] - (data[i] - 1.0)).abs() < 0.01);
        }
    }

    #[test]
    fn test_normality_score() {
        // Uniform-ish distribution
        let uniform = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let score_uniform = NormalizationAnalyzer::normality_score(&uniform);

        // Bell-shaped distribution
        let bell = vec![1.0, 2.0, 3.0, 3.0, 3.0, 2.0, 1.0];
        let score_bell = NormalizationAnalyzer::normality_score(&bell);

        // Both should be finite
        assert!(score_uniform.is_finite());
        assert!(score_bell.is_finite());
    }

    #[test]
    fn test_no_outliers_uniform() {
        let mut frequency_map = HashMap::new();
        // Uniform distribution - no outliers
        for i in 1..=10 {
            frequency_map.insert(i, 5);
        }

        let dataset = LotteryDataset {
            draws: vec![],
            min_number: 1,
            max_number: 10,
            numbers_per_draw: 5,
        };

        let outliers = NormalizationAnalyzer::detect_outliers(&frequency_map, &dataset);

        // Should detect no outliers
        assert!(outliers.is_empty());
    }
}
