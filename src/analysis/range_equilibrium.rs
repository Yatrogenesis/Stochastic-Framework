/// Range Equilibrium Analysis
/// Analyzes the distribution across Low, Medium, High ranges and parity
use crate::models::{LotteryDataset, RangeEquilibriumResult};
use anyhow::Result;
use statrs::distribution::{ChiSquared, ContinuousCDF};

pub struct RangeEquilibriumAnalyzer;

#[derive(Debug, Clone, Copy, PartialEq)]
enum Range {
    Low,    // 1-9
    Medium, // 10-19
    High,   // 20-28
}

impl RangeEquilibriumAnalyzer {
    /// Perform range equilibrium analysis
    pub fn analyze(dataset: &LotteryDataset) -> Result<RangeEquilibriumResult> {
        log::info!("Starting range equilibrium analysis");

        // Count numbers in each range
        let (low_count, medium_count, high_count) = Self::count_by_range(dataset);
        let total_count = (low_count + medium_count + high_count) as f64;

        // Calculate percentages
        let low_percentage = (low_count as f64 / total_count) * 100.0;
        let medium_percentage = (medium_count as f64 / total_count) * 100.0;
        let high_percentage = (high_count as f64 / total_count) * 100.0;

        // Count even/odd numbers
        let (even_count, odd_count) = Self::count_parity(dataset);
        let total_parity = (even_count + odd_count) as f64;
        let even_percentage = (even_count as f64 / total_parity) * 100.0;
        let odd_percentage = (odd_count as f64 / total_parity) * 100.0;

        // Chi-squared test for range balance
        let (chi_squared_ranges, p_value_ranges) =
            Self::chi_squared_test_ranges(low_count, medium_count, high_count)?;

        // System is balanced if p-value > 0.05 (fail to reject null hypothesis of uniformity)
        let is_balanced = p_value_ranges > 0.05;

        log::info!(
            "Range Equilibrium: Low={:.2}%, Med={:.2}%, High={:.2}%, χ²={:.4}, p={:.4}, balanced={}",
            low_percentage,
            medium_percentage,
            high_percentage,
            chi_squared_ranges,
            p_value_ranges,
            is_balanced
        );

        Ok(RangeEquilibriumResult {
            low_percentage,
            medium_percentage,
            high_percentage,
            even_percentage,
            odd_percentage,
            chi_squared_ranges,
            p_value_ranges,
            is_balanced,
        })
    }

    /// Determine which range a number belongs to
    fn get_range(number: u32) -> Range {
        match number {
            1..=9 => Range::Low,
            10..=19 => Range::Medium,
            20..=28 => Range::High,
            _ => Range::Medium, // Default fallback
        }
    }

    /// Count numbers in each range
    fn count_by_range(dataset: &LotteryDataset) -> (usize, usize, usize) {
        let mut low = 0;
        let mut medium = 0;
        let mut high = 0;

        for draw in &dataset.draws {
            for &number in &draw.numbers {
                match Self::get_range(number) {
                    Range::Low => low += 1,
                    Range::Medium => medium += 1,
                    Range::High => high += 1,
                }
            }
        }

        (low, medium, high)
    }

    /// Count even and odd numbers
    fn count_parity(dataset: &LotteryDataset) -> (usize, usize) {
        let mut even = 0;
        let mut odd = 0;

        for draw in &dataset.draws {
            for &number in &draw.numbers {
                if number % 2 == 0 {
                    even += 1;
                } else {
                    odd += 1;
                }
            }
        }

        (even, odd)
    }

    /// Chi-squared test for range uniformity
    /// H0: Numbers are uniformly distributed across ranges
    fn chi_squared_test_ranges(
        low: usize,
        medium: usize,
        high: usize,
    ) -> Result<(f64, f64)> {
        // Total observations
        let total = (low + medium + high) as f64;

        // Expected counts under uniform distribution
        // Low: 9 numbers (1-9), Medium: 10 numbers (10-19), High: 9 numbers (20-28)
        // Total: 28 numbers
        let expected_low = total * (9.0 / 28.0);
        let expected_medium = total * (10.0 / 28.0);
        let expected_high = total * (9.0 / 28.0);

        // Chi-squared statistic: Σ((O - E)² / E)
        let chi_squared = ((low as f64 - expected_low).powi(2) / expected_low)
            + ((medium as f64 - expected_medium).powi(2) / expected_medium)
            + ((high as f64 - expected_high).powi(2) / expected_high);

        // Degrees of freedom: k - 1 = 3 - 1 = 2
        let dof = 2.0;
        let chi_dist = ChiSquared::new(dof)?;
        let p_value = 1.0 - chi_dist.cdf(chi_squared);

        Ok((chi_squared, p_value))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::LotteryDraw;
    use chrono::Utc;

    #[test]
    fn test_range_equilibrium_analysis() {
        let draws = vec![
            LotteryDraw {
                id: Some(1),
                numbers: vec![1, 10, 20, 5, 15],
                draw_date: Utc::now(),
                game_name: "Test".to_string(),
            },
            LotteryDraw {
                id: Some(2),
                numbers: vec![2, 11, 21, 6, 16],
                draw_date: Utc::now(),
                game_name: "Test".to_string(),
            },
            LotteryDraw {
                id: Some(3),
                numbers: vec![3, 12, 22, 7, 17],
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

        let result = RangeEquilibriumAnalyzer::analyze(&dataset);
        assert!(result.is_ok());

        let result = result.unwrap();

        // Check percentages sum to 100%
        let total = result.low_percentage + result.medium_percentage + result.high_percentage;
        assert!((total - 100.0).abs() < 0.01);

        // Check parity percentages sum to 100%
        let parity_total = result.even_percentage + result.odd_percentage;
        assert!((parity_total - 100.0).abs() < 0.01);

        // Check chi-squared is non-negative
        assert!(result.chi_squared_ranges >= 0.0);

        // Check p-value is in [0, 1]
        assert!(result.p_value_ranges >= 0.0 && result.p_value_ranges <= 1.0);
    }

    #[test]
    fn test_range_classification() {
        assert_eq!(RangeEquilibriumAnalyzer::get_range(1), Range::Low);
        assert_eq!(RangeEquilibriumAnalyzer::get_range(5), Range::Low);
        assert_eq!(RangeEquilibriumAnalyzer::get_range(9), Range::Low);
        assert_eq!(RangeEquilibriumAnalyzer::get_range(10), Range::Medium);
        assert_eq!(RangeEquilibriumAnalyzer::get_range(15), Range::Medium);
        assert_eq!(RangeEquilibriumAnalyzer::get_range(19), Range::Medium);
        assert_eq!(RangeEquilibriumAnalyzer::get_range(20), Range::High);
        assert_eq!(RangeEquilibriumAnalyzer::get_range(25), Range::High);
        assert_eq!(RangeEquilibriumAnalyzer::get_range(28), Range::High);
    }

    #[test]
    fn test_parity_counting() {
        let draws = vec![
            LotteryDraw {
                id: Some(1),
                numbers: vec![2, 4, 6, 8, 10], // All even
                draw_date: Utc::now(),
                game_name: "Test".to_string(),
            },
            LotteryDraw {
                id: Some(2),
                numbers: vec![1, 3, 5, 7, 9], // All odd
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

        let result = RangeEquilibriumAnalyzer::analyze(&dataset).unwrap();

        // Should be 50% even, 50% odd
        assert!((result.even_percentage - 50.0).abs() < 0.01);
        assert!((result.odd_percentage - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_balanced_distribution() {
        // Create perfectly balanced draws across ranges
        let draws = vec![
            LotteryDraw {
                id: Some(1),
                numbers: vec![1, 2, 10, 11, 20],
                draw_date: Utc::now(),
                game_name: "Test".to_string(),
            },
            LotteryDraw {
                id: Some(2),
                numbers: vec![3, 4, 12, 13, 21],
                draw_date: Utc::now(),
                game_name: "Test".to_string(),
            },
            LotteryDraw {
                id: Some(3),
                numbers: vec![5, 6, 14, 15, 22],
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

        let result = RangeEquilibriumAnalyzer::analyze(&dataset).unwrap();

        // All percentages should be positive
        assert!(result.low_percentage > 0.0);
        assert!(result.medium_percentage > 0.0);
        assert!(result.high_percentage > 0.0);
    }
}
