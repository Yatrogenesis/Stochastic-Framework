/// Bayesian Inference Analysis - Dirichlet Distribution
/// Uses Jeffreys non-informative prior (α = 0.5 for each category)
use crate::models::{BayesianResult, LotteryDataset};
use anyhow::Result;
use special::Gamma;
use std::collections::HashMap;

pub struct BayesianAnalyzer;

impl BayesianAnalyzer {
    /// Perform Bayesian inference using Dirichlet-Multinomial conjugate prior
    pub fn analyze(dataset: &LotteryDataset) -> Result<BayesianResult> {
        log::info!("Starting Bayesian analysis with Dirichlet prior");

        // Jeffreys non-informative prior: α = 0.5
        let alpha_prior = 0.5;

        // Count frequency of each number
        let frequency_map = Self::compute_frequency_distribution(dataset);

        // Compute posterior probabilities: Dirichlet(α + counts)
        let posterior_probabilities = Self::compute_posterior_probabilities(
            &frequency_map,
            dataset,
            alpha_prior,
        );

        // Calculate 95% credible intervals
        let credible_intervals = Self::compute_credible_intervals(
            &frequency_map,
            dataset,
            alpha_prior,
            0.95,
        );

        // Get top numbers by posterior probability
        let mut sorted_posteriors = posterior_probabilities.clone();
        sorted_posteriors.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let top_numbers: Vec<u32> = sorted_posteriors
            .iter()
            .take(10)
            .map(|&(num, _)| num)
            .collect();

        log::info!(
            "Bayesian Analysis complete: Top number = {} with posterior = {:.6}",
            top_numbers[0],
            sorted_posteriors[0].1
        );

        Ok(BayesianResult {
            posterior_probabilities,
            credible_intervals,
            top_numbers,
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

    /// Compute posterior probabilities using Dirichlet distribution
    /// Posterior: Dirichlet(α + counts)
    /// Mean of Dirichlet: (α_i + n_i) / (Σα + N)
    fn compute_posterior_probabilities(
        frequency_map: &HashMap<u32, usize>,
        dataset: &LotteryDataset,
        alpha_prior: f64,
    ) -> Vec<(u32, f64)> {
        let number_count = (dataset.max_number - dataset.min_number + 1) as usize;
        let total_observations: usize = dataset.draws.iter().map(|d| d.numbers.len()).sum();
        let total_alpha = alpha_prior * number_count as f64;
        let denominator = total_alpha + total_observations as f64;

        let mut posteriors = Vec::new();

        for number in dataset.min_number..=dataset.max_number {
            let count = *frequency_map.get(&number).unwrap_or(&0) as f64;
            let posterior_mean = (alpha_prior + count) / denominator;
            posteriors.push((number, posterior_mean));
        }

        posteriors
    }

    /// Compute 95% credible intervals using Beta approximation
    /// For Dirichlet, the marginal for each category is Beta(α_i + n_i, Σα_j + Σn_j - α_i - n_i)
    fn compute_credible_intervals(
        frequency_map: &HashMap<u32, usize>,
        dataset: &LotteryDataset,
        alpha_prior: f64,
        confidence: f64,
    ) -> Vec<(u32, f64, f64)> {
        let number_count = (dataset.max_number - dataset.min_number + 1) as usize;
        let total_observations: usize = dataset.draws.iter().map(|d| d.numbers.len()).sum();
        let total_alpha = alpha_prior * number_count as f64;

        let mut intervals = Vec::new();
        let tail_probability = (1.0 - confidence) / 2.0;

        for number in dataset.min_number..=dataset.max_number {
            let count = *frequency_map.get(&number).unwrap_or(&0) as f64;

            // Parameters for Beta distribution
            let alpha_post = alpha_prior + count;
            let beta_post = total_alpha + total_observations as f64 - alpha_post;

            // Approximate credible interval using normal approximation for large samples
            // For Beta(α, β), mean = α/(α+β), var = αβ/((α+β)²(α+β+1))
            let mean = alpha_post / (alpha_post + beta_post);
            let variance = (alpha_post * beta_post)
                / ((alpha_post + beta_post).powi(2) * (alpha_post + beta_post + 1.0));
            let std_dev = variance.sqrt();

            // 95% CI using normal approximation: mean ± 1.96 * std_dev
            let z_score = 1.96; // for 95% confidence
            let lower = (mean - z_score * std_dev).max(0.0);
            let upper = (mean + z_score * std_dev).min(1.0);

            intervals.push((number, lower, upper));
        }

        intervals
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::LotteryDraw;
    use chrono::Utc;

    #[test]
    fn test_bayesian_analysis() {
        let draws = vec![
            LotteryDraw {
                id: Some(1),
                numbers: vec![1, 2, 3, 4, 5],
                draw_date: Utc::now(),
                game_name: "Test".to_string(),
            },
            LotteryDraw {
                id: Some(2),
                numbers: vec![1, 2, 6, 7, 8],
                draw_date: Utc::now(),
                game_name: "Test".to_string(),
            },
            LotteryDraw {
                id: Some(3),
                numbers: vec![1, 9, 10, 11, 12],
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

        let result = BayesianAnalyzer::analyze(&dataset);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(result.posterior_probabilities.len(), 28);
        assert_eq!(result.credible_intervals.len(), 28);
        assert_eq!(result.top_numbers.len(), 10);

        // Check that probabilities sum to approximately 1.0
        let sum: f64 = result.posterior_probabilities.iter().map(|&(_, p)| p).sum();
        assert!((sum - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_credible_intervals_ordering() {
        let draws = vec![
            LotteryDraw {
                id: Some(1),
                numbers: vec![1, 2, 3, 4, 5],
                draw_date: Utc::now(),
                game_name: "Test".to_string(),
            },
        ];

        let dataset = LotteryDataset {
            draws,
            min_number: 1,
            max_number: 10,
            numbers_per_draw: 5,
        };

        let result = BayesianAnalyzer::analyze(&dataset).unwrap();

        // Check that lower bound <= upper bound for all intervals
        for &(_, lower, upper) in &result.credible_intervals {
            assert!(lower <= upper, "Lower bound should be <= upper bound");
            assert!(lower >= 0.0 && lower <= 1.0, "Lower bound should be in [0, 1]");
            assert!(upper >= 0.0 && upper <= 1.0, "Upper bound should be in [0, 1]");
        }
    }
}
