/// Critical Path Analysis - Multi-Objective Optimization
/// Combines multiple criteria to recommend optimal lottery combinations
use crate::models::{
    BayesianResult, CombinationProperties, ConfidenceLevel, CriticalPathResult,
    FrequentistResult, LotteryDataset,
};
use anyhow::Result;
use rayon::prelude::*;
use std::collections::HashSet;

pub struct CriticalPathAnalyzer;

impl CriticalPathAnalyzer {
    // Scoring weights
    const WEIGHT_BAYESIAN: f64 = 0.35; // 35% - Bayesian posterior (inverse)
    const WEIGHT_FREQUENCY: f64 = 0.35; // 35% - Frequency (inverse)
    const WEIGHT_RANGE_EQUILIBRIUM: f64 = 0.15; // 15% - Range balance
    const WEIGHT_PRIME_DIVERSITY: f64 = 0.15; // 15% - Prime diversity

    /// Perform critical path analysis with multi-objective optimization
    pub fn analyze(
        dataset: &LotteryDataset,
        frequentist: &FrequentistResult,
        bayesian: &BayesianResult,
        confidence_level: ConfidenceLevel,
    ) -> Result<CriticalPathResult> {
        log::info!(
            "Starting critical path analysis with confidence level {:?}",
            confidence_level
        );

        // Generate candidate combinations
        let num_candidates = Self::calculate_num_candidates(confidence_level);
        let candidates = Self::generate_candidate_combinations(
            dataset,
            frequentist,
            bayesian,
            num_candidates,
        );

        log::info!("Generated {} candidate combinations", candidates.len());

        // Score all candidates in parallel
        let scored_candidates: Vec<(Vec<u32>, f64)> = candidates
            .par_iter()
            .map(|combination| {
                let score = Self::score_combination(combination, frequentist, bayesian, dataset);
                (combination.clone(), score)
            })
            .collect();

        // Find best combination
        let best = scored_candidates
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .ok_or_else(|| anyhow::anyhow!("No valid combinations found"))?;

        let recommended_combination = best.0.clone();
        let score = best.1;

        // Calculate properties of recommended combination
        let properties = Self::calculate_combination_properties(&recommended_combination, dataset);

        log::info!(
            "Critical Path: Recommended combination {:?} with score {:.4}",
            recommended_combination,
            score
        );

        Ok(CriticalPathResult {
            recommended_combination,
            score,
            properties,
        })
    }

    /// Calculate number of candidates based on confidence level
    fn calculate_num_candidates(confidence_level: ConfidenceLevel) -> usize {
        match confidence_level {
            ConfidenceLevel::Low => 1000,
            ConfidenceLevel::Medium => 5000,
            ConfidenceLevel::High => 10000,
            ConfidenceLevel::VeryHigh => 20000,
        }
    }

    /// Generate candidate combinations using various strategies
    fn generate_candidate_combinations(
        dataset: &LotteryDataset,
        frequentist: &FrequentistResult,
        bayesian: &BayesianResult,
        num_candidates: usize,
    ) -> Vec<Vec<u32>> {
        let mut candidates = Vec::new();

        // Strategy 1: Top Bayesian numbers
        candidates.extend(Self::generate_top_bayesian(bayesian, dataset.numbers_per_draw, 100));

        // Strategy 2: Least frequent numbers (inverse frequency)
        candidates.extend(Self::generate_least_frequent(
            frequentist,
            dataset,
            100,
        ));

        // Strategy 3: Balanced range combinations
        candidates.extend(Self::generate_balanced_range(dataset, 100));

        // Strategy 4: Prime-rich combinations
        candidates.extend(Self::generate_prime_rich(dataset, 100));

        // Strategy 5: Random diverse combinations
        candidates.extend(Self::generate_random_diverse(
            dataset,
            num_candidates.saturating_sub(candidates.len()),
        ));

        candidates
    }

    /// Generate combinations using top Bayesian posterior probabilities
    fn generate_top_bayesian(
        bayesian: &BayesianResult,
        numbers_per_draw: usize,
        count: usize,
    ) -> Vec<Vec<u32>> {
        let mut combinations = Vec::new();

        // Sort by posterior probability
        let mut sorted_posteriors = bayesian.posterior_probabilities.clone();
        sorted_posteriors.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Generate combinations with varying number of top numbers
        for i in 0..count.min(20) {
            let offset = i / 4;
            let mut combo = Vec::new();

            for j in 0..numbers_per_draw {
                if offset + j < sorted_posteriors.len() {
                    combo.push(sorted_posteriors[offset + j].0);
                }
            }

            if combo.len() == numbers_per_draw {
                combo.sort();
                combinations.push(combo);
            }
        }

        combinations
    }

    /// Generate combinations using least frequent numbers
    fn generate_least_frequent(
        frequentist: &FrequentistResult,
        dataset: &LotteryDataset,
        count: usize,
    ) -> Vec<Vec<u32>> {
        let mut combinations = Vec::new();

        // Sort by frequency (ascending)
        let mut sorted_freq = frequentist.frequency_distribution.clone();
        sorted_freq.sort_by_key(|&(_, freq)| freq);

        // Generate combinations with varying number of least frequent numbers
        for i in 0..count.min(20) {
            let offset = i / 4;
            let mut combo = Vec::new();

            for j in 0..dataset.numbers_per_draw {
                if offset + j < sorted_freq.len() {
                    combo.push(sorted_freq[offset + j].0);
                }
            }

            if combo.len() == dataset.numbers_per_draw {
                combo.sort();
                combinations.push(combo);
            }
        }

        combinations
    }

    /// Generate combinations with balanced range distribution
    fn generate_balanced_range(dataset: &LotteryDataset, count: usize) -> Vec<Vec<u32>> {
        let mut combinations = Vec::new();
        let numbers_per_draw = dataset.numbers_per_draw;

        // Try to balance across Low (1-9), Medium (10-19), High (20-28)
        let low_numbers: Vec<u32> = (1..=9.min(dataset.max_number)).collect();
        let medium_numbers: Vec<u32> = (10..=19.min(dataset.max_number)).collect();
        let high_numbers: Vec<u32> = (20..=dataset.max_number).collect();

        for _ in 0..count.min(100) {
            let n_low = numbers_per_draw / 3;
            let n_medium = numbers_per_draw / 3;
            let n_high = numbers_per_draw - n_low - n_medium;

            let mut combo = Vec::new();

            // Pick from each range
            for _ in 0..n_low {
                if let Some(&num) = low_numbers
                    .iter()
                    .find(|&&n| !combo.contains(&n))
                {
                    combo.push(num);
                }
            }

            for _ in 0..n_medium {
                if let Some(&num) = medium_numbers
                    .iter()
                    .find(|&&n| !combo.contains(&n))
                {
                    combo.push(num);
                }
            }

            for _ in 0..n_high {
                if let Some(&num) = high_numbers
                    .iter()
                    .find(|&&n| !combo.contains(&n))
                {
                    combo.push(num);
                }
            }

            if combo.len() == numbers_per_draw {
                combo.sort();
                combinations.push(combo);
            }
        }

        combinations
    }

    /// Generate combinations rich in prime numbers
    fn generate_prime_rich(dataset: &LotteryDataset, count: usize) -> Vec<Vec<u32>> {
        let mut combinations = Vec::new();
        let primes: Vec<u32> = (dataset.min_number..=dataset.max_number)
            .filter(|&n| Self::is_prime(n))
            .collect();

        if primes.len() < dataset.numbers_per_draw {
            return combinations;
        }

        for _ in 0..count.min(50) {
            let mut combo = Vec::new();

            // Include at least 60% primes
            let n_primes = (dataset.numbers_per_draw as f64 * 0.6).ceil() as usize;

            for &prime in primes.iter().take(n_primes) {
                if !combo.contains(&prime) {
                    combo.push(prime);
                }
            }

            // Fill remaining with non-primes
            for num in dataset.min_number..=dataset.max_number {
                if combo.len() >= dataset.numbers_per_draw {
                    break;
                }
                if !combo.contains(&num) && !Self::is_prime(num) {
                    combo.push(num);
                }
            }

            if combo.len() == dataset.numbers_per_draw {
                combo.sort();
                combinations.push(combo);
            }
        }

        combinations
    }

    /// Generate random diverse combinations
    fn generate_random_diverse(dataset: &LotteryDataset, count: usize) -> Vec<Vec<u32>> {
        let mut combinations = Vec::new();
        let all_numbers: Vec<u32> = (dataset.min_number..=dataset.max_number).collect();

        for seed in 0..count {
            let mut combo = Vec::new();

            // Use deterministic pseudo-random based on seed
            let mut state = seed as u64;

            while combo.len() < dataset.numbers_per_draw {
                // Simple LCG pseudo-random
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                let index = (state as usize) % all_numbers.len();
                let num = all_numbers[index];

                if !combo.contains(&num) {
                    combo.push(num);
                }
            }

            combo.sort();
            combinations.push(combo);
        }

        combinations
    }

    /// Score a combination using multi-objective criteria
    fn score_combination(
        combination: &[u32],
        frequentist: &FrequentistResult,
        bayesian: &BayesianResult,
        dataset: &LotteryDataset,
    ) -> f64 {
        let mut score = 0.0;

        // 1. Bayesian score (inverse of posterior - we want underrepresented numbers)
        let bayesian_score = Self::compute_bayesian_score(combination, bayesian);
        score += Self::WEIGHT_BAYESIAN * bayesian_score;

        // 2. Frequency score (inverse - we want less frequent numbers)
        let frequency_score = Self::compute_frequency_score(combination, frequentist);
        score += Self::WEIGHT_FREQUENCY * frequency_score;

        // 3. Range equilibrium score
        let range_score = Self::compute_range_equilibrium_score(combination);
        score += Self::WEIGHT_RANGE_EQUILIBRIUM * range_score;

        // 4. Prime diversity score
        let prime_score = Self::compute_prime_diversity_score(combination);
        score += Self::WEIGHT_PRIME_DIVERSITY * prime_score;

        score
    }

    /// Compute Bayesian score (prefer lower posterior probabilities)
    fn compute_bayesian_score(combination: &[u32], bayesian: &BayesianResult) -> f64 {
        let mut total_posterior = 0.0;

        for &num in combination {
            if let Some(&(_, prob)) = bayesian
                .posterior_probabilities
                .iter()
                .find(|&&(n, _)| n == num)
            {
                total_posterior += prob;
            }
        }

        let avg_posterior = total_posterior / combination.len() as f64;

        // Invert: lower posterior = higher score
        1.0 - avg_posterior
    }

    /// Compute frequency score (prefer less frequent numbers)
    fn compute_frequency_score(combination: &[u32], frequentist: &FrequentistResult) -> f64 {
        let mut total_freq = 0.0;
        let max_freq = frequentist
            .frequency_distribution
            .iter()
            .map(|&(_, f)| f)
            .max()
            .unwrap_or(1) as f64;

        for &num in combination {
            if let Some(&(_, freq)) = frequentist
                .frequency_distribution
                .iter()
                .find(|&&(n, _)| n == num)
            {
                total_freq += freq as f64 / max_freq;
            }
        }

        let avg_freq = total_freq / combination.len() as f64;

        // Invert: lower frequency = higher score
        1.0 - avg_freq
    }

    /// Compute range equilibrium score (prefer balanced distribution)
    fn compute_range_equilibrium_score(combination: &[u32]) -> f64 {
        let mut low = 0;
        let mut medium = 0;
        let mut high = 0;

        for &num in combination {
            match num {
                1..=9 => low += 1,
                10..=19 => medium += 1,
                20..=28 => high += 1,
                _ => {}
            }
        }

        // Perfect balance would be equal distribution
        let total = combination.len() as f64;
        let ideal = total / 3.0;

        let low_diff = (low as f64 - ideal).abs();
        let medium_diff = (medium as f64 - ideal).abs();
        let high_diff = (high as f64 - ideal).abs();

        let total_diff = low_diff + medium_diff + high_diff;

        // Lower difference = higher score
        1.0 / (1.0 + total_diff)
    }

    /// Compute prime diversity score
    fn compute_prime_diversity_score(combination: &[u32]) -> f64 {
        let prime_count = combination.iter().filter(|&&n| Self::is_prime(n)).count();
        let non_prime_count = combination.len() - prime_count;

        // Ideal is 40-60% primes
        let prime_ratio = prime_count as f64 / combination.len() as f64;

        if prime_ratio >= 0.4 && prime_ratio <= 0.6 {
            1.0
        } else if prime_ratio < 0.4 {
            prime_ratio / 0.4
        } else {
            (1.0 - prime_ratio) / 0.4
        }
    }

    /// Calculate properties of a combination
    fn calculate_combination_properties(
        combination: &[u32],
        dataset: &LotteryDataset,
    ) -> CombinationProperties {
        let sum: u32 = combination.iter().sum();
        let mean = sum as f64 / combination.len() as f64;

        // Median
        let mut sorted = combination.to_vec();
        sorted.sort();
        let median = if sorted.len() % 2 == 0 {
            let mid = sorted.len() / 2;
            (sorted[mid - 1] + sorted[mid]) as f64 / 2.0
        } else {
            sorted[sorted.len() / 2] as f64
        };

        // Standard deviation
        let variance = combination
            .iter()
            .map(|&x| (x as f64 - mean).powi(2))
            .sum::<f64>()
            / combination.len() as f64;
        let std_dev = variance.sqrt();

        // Range
        let range = sorted[sorted.len() - 1] - sorted[0];

        // Even/odd count
        let even_count = combination.iter().filter(|&&n| n % 2 == 0).count();
        let odd_count = combination.len() - even_count;

        // Prime count
        let primes: Vec<u32> = combination
            .iter()
            .filter(|&&n| Self::is_prime(n))
            .copied()
            .collect();
        let prime_count = primes.len();

        // Range distribution
        let low_count = combination.iter().filter(|&&n| n >= 1 && n <= 9).count();
        let medium_count = combination.iter().filter(|&&n| n >= 10 && n <= 19).count();
        let high_count = combination.iter().filter(|&&n| n >= 20 && n <= 28).count();

        CombinationProperties {
            sum,
            mean,
            median,
            std_dev,
            range,
            even_count,
            odd_count,
            prime_count,
            primes,
            low_count,
            medium_count,
            high_count,
        }
    }

    /// Check if a number is prime
    fn is_prime(n: u32) -> bool {
        if n < 2 {
            return false;
        }
        if n == 2 {
            return true;
        }
        if n % 2 == 0 {
            return false;
        }

        let sqrt_n = (n as f64).sqrt() as u32;
        for i in (3..=sqrt_n).step_by(2) {
            if n % i == 0 {
                return false;
            }
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{BayesianResult, FrequentistResult, LotteryDraw};
    use chrono::Utc;

    fn create_test_dataset() -> LotteryDataset {
        LotteryDataset {
            draws: vec![
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
            ],
            min_number: 1,
            max_number: 28,
            numbers_per_draw: 5,
        }
    }

    fn create_test_frequentist() -> FrequentistResult {
        FrequentistResult {
            chi_squared: 10.0,
            p_value: 0.5,
            degrees_of_freedom: 27,
            frequency_distribution: (1..=28).map(|n| (n, 1)).collect(),
            is_uniform: true,
        }
    }

    fn create_test_bayesian() -> BayesianResult {
        BayesianResult {
            posterior_probabilities: (1..=28).map(|n| (n, 1.0 / 28.0)).collect(),
            credible_intervals: (1..=28).map(|n| (n, 0.01, 0.05)).collect(),
            top_numbers: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        }
    }

    #[test]
    fn test_critical_path_analysis() {
        let dataset = create_test_dataset();
        let frequentist = create_test_frequentist();
        let bayesian = create_test_bayesian();

        let result = CriticalPathAnalyzer::analyze(
            &dataset,
            &frequentist,
            &bayesian,
            ConfidenceLevel::Low,
        );

        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(result.recommended_combination.len(), 5);
        assert!(result.score > 0.0);
    }

    #[test]
    fn test_combination_properties() {
        let combination = vec![1, 5, 10, 15, 20];
        let dataset = create_test_dataset();

        let props = CriticalPathAnalyzer::calculate_combination_properties(&combination, &dataset);

        assert_eq!(props.sum, 51);
        assert_eq!(props.mean, 10.2);
        assert_eq!(props.median, 10.0);
        assert_eq!(props.range, 19);
        assert_eq!(props.even_count, 2); // 10, 20
        assert_eq!(props.odd_count, 3); // 1, 5, 15
    }

    #[test]
    fn test_is_prime() {
        assert!(CriticalPathAnalyzer::is_prime(2));
        assert!(CriticalPathAnalyzer::is_prime(3));
        assert!(CriticalPathAnalyzer::is_prime(5));
        assert!(CriticalPathAnalyzer::is_prime(7));
        assert!(CriticalPathAnalyzer::is_prime(11));
        assert!(CriticalPathAnalyzer::is_prime(13));

        assert!(!CriticalPathAnalyzer::is_prime(1));
        assert!(!CriticalPathAnalyzer::is_prime(4));
        assert!(!CriticalPathAnalyzer::is_prime(6));
        assert!(!CriticalPathAnalyzer::is_prime(8));
        assert!(!CriticalPathAnalyzer::is_prime(9));
    }

    #[test]
    fn test_range_equilibrium_score() {
        // Perfectly balanced: 2 low, 2 medium, 1 high
        let balanced = vec![1, 5, 10, 15, 20];
        let score_balanced = CriticalPathAnalyzer::compute_range_equilibrium_score(&balanced);

        // Unbalanced: all low
        let unbalanced = vec![1, 2, 3, 4, 5];
        let score_unbalanced = CriticalPathAnalyzer::compute_range_equilibrium_score(&unbalanced);

        // Balanced should have higher score
        assert!(score_balanced > score_unbalanced);
    }

    #[test]
    fn test_prime_diversity_score() {
        // 3 primes out of 5 = 60% (ideal)
        let ideal = vec![2, 3, 5, 6, 8];
        let score_ideal = CriticalPathAnalyzer::compute_prime_diversity_score(&ideal);

        // All primes = 100% (too many)
        let too_many = vec![2, 3, 5, 7, 11];
        let score_too_many = CriticalPathAnalyzer::compute_prime_diversity_score(&too_many);

        // No primes = 0% (too few)
        let too_few = vec![4, 6, 8, 10, 12];
        let score_too_few = CriticalPathAnalyzer::compute_prime_diversity_score(&too_few);

        // Ideal should have highest score
        assert!(score_ideal >= score_too_many);
        assert!(score_ideal >= score_too_few);
    }

    #[test]
    fn test_generate_combinations() {
        let dataset = create_test_dataset();
        let frequentist = create_test_frequentist();
        let bayesian = create_test_bayesian();

        let candidates =
            CriticalPathAnalyzer::generate_candidate_combinations(&dataset, &frequentist, &bayesian, 100);

        assert!(!candidates.is_empty());

        // All combinations should have correct length
        for combo in &candidates {
            assert_eq!(combo.len(), dataset.numbers_per_draw);
        }

        // All combinations should be sorted
        for combo in &candidates {
            let mut sorted = combo.clone();
            sorted.sort();
            assert_eq!(*combo, sorted);
        }
    }

    #[test]
    fn test_confidence_level_affects_candidates() {
        let low_count = CriticalPathAnalyzer::calculate_num_candidates(ConfidenceLevel::Low);
        let high_count = CriticalPathAnalyzer::calculate_num_candidates(ConfidenceLevel::High);
        let very_high_count =
            CriticalPathAnalyzer::calculate_num_candidates(ConfidenceLevel::VeryHigh);

        assert!(high_count > low_count);
        assert!(very_high_count > high_count);
    }
}
