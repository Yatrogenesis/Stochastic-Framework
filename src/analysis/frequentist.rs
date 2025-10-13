/// Frequentist Analysis - χ² Goodness-of-Fit Test
/// Based on Pearson's chi-squared test and Central Limit Theorem
use crate::models::{FrequentistResult, LotteryDataset};
use anyhow::Result;
use statrs::distribution::{ChiSquared, ContinuousCDF};
use std::collections::HashMap;

pub struct FrequentistAnalyzer;

impl FrequentistAnalyzer {
    /// Perform frequentist analysis using chi-squared goodness-of-fit test
    pub fn analyze(dataset: &LotteryDataset) -> Result<FrequentistResult> {
        // Count frequency of each number
        let frequency_map = Self::compute_frequency_distribution(dataset);

        // Expected frequency under uniform distribution
        let total_draws = dataset.draws.iter().map(|d| d.numbers.len()).sum::<usize>();
        let number_count = (dataset.max_number - dataset.min_number + 1) as usize;
        let expected_frequency = total_draws as f64 / number_count as f64;

        // Compute chi-squared statistic
        let chi_squared = Self::compute_chi_squared(&frequency_map, expected_frequency, dataset);

        // Degrees of freedom: k - 1 where k is number of categories
        let dof = number_count - 1;

        // Compute p-value
        let chi_dist = ChiSquared::new(dof as f64)?;
        let p_value = 1.0 - chi_dist.cdf(chi_squared);

        // Test decision: reject H0 if p < 0.05
        let is_uniform = p_value > 0.05;

        // Convert frequency map to sorted vector
        let mut frequency_distribution: Vec<(u32, usize)> = frequency_map
            .iter()
            .map(|(&k, &v)| (k, v))
            .collect();
        frequency_distribution.sort_by_key(|&(k, _)| k);

        log::info!(
            "Frequentist Analysis: χ² = {:.4}, p-value = {:.4}, uniform = {}",
            chi_squared,
            p_value,
            is_uniform
        );

        Ok(FrequentistResult {
            chi_squared,
            p_value,
            degrees_of_freedom: dof,
            frequency_distribution,
            is_uniform,
        })
    }

    /// Compute frequency distribution of all numbers
    fn compute_frequency_distribution(dataset: &LotteryDataset) -> HashMap<u32, usize> {
        let mut frequency_map = HashMap::new();

        for draw in &dataset.draws {
            for &number in &draw.numbers {
                *frequency_map.entry(number).or_insert(0) += 1;
            }
        }

        frequency_map
    }

    /// Compute chi-squared statistic: Σ((O - E)² / E)
    fn compute_chi_squared(
        frequency_map: &HashMap<u32, usize>,
        expected_frequency: f64,
        dataset: &LotteryDataset,
    ) -> f64 {
        let mut chi_squared = 0.0;

        for number in dataset.min_number..=dataset.max_number {
            let observed = *frequency_map.get(&number).unwrap_or(&0) as f64;
            let difference = observed - expected_frequency;
            chi_squared += (difference * difference) / expected_frequency;
        }

        chi_squared
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::LotteryDraw;
    use chrono::Utc;

    #[test]
    fn test_frequentist_analysis() {
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
        ];

        let dataset = LotteryDataset {
            draws,
            min_number: 1,
            max_number: 28,
            numbers_per_draw: 5,
        };

        let result = FrequentistAnalyzer::analyze(&dataset);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert!(result.chi_squared > 0.0);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    }
}
