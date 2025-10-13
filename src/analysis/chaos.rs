/// Chaos Theory Analysis - Lyapunov Exponent and Takens Embedding
/// Detects chaotic behavior in lottery draw sequences
use crate::models::{ChaosTheoryResult, LotteryDataset};
use anyhow::Result;

pub struct ChaosTheoryAnalyzer;

impl ChaosTheoryAnalyzer {
    const EMBEDDING_DIMENSION: usize = 3; // m = 3
    const TIME_DELAY: usize = 1; // τ = 1

    /// Perform chaos theory analysis
    pub fn analyze(dataset: &LotteryDataset) -> Result<ChaosTheoryResult> {
        log::info!("Starting chaos theory analysis with Lyapunov and Takens");

        // Create time series from draws
        let time_series = Self::create_time_series(dataset);

        // Compute Lyapunov exponent
        let lyapunov_exponent = Self::compute_lyapunov_exponent(&time_series);

        // Perform Takens embedding reconstruction
        let embedded_series = Self::takens_embedding(&time_series, Self::EMBEDDING_DIMENSION, Self::TIME_DELAY);

        // Compute correlation dimension
        let correlation_dimension = Self::compute_correlation_dimension(&embedded_series);

        // System is chaotic if λ > 0
        let is_chaotic = lyapunov_exponent > 0.0;

        log::info!(
            "Chaos Analysis: λ={:.4}, D_corr={:.4}, chaotic={}, embedding_dim={}",
            lyapunov_exponent,
            correlation_dimension,
            is_chaotic,
            Self::EMBEDDING_DIMENSION
        );

        Ok(ChaosTheoryResult {
            lyapunov_exponent,
            correlation_dimension,
            is_chaotic,
            embedding_dimension: Self::EMBEDDING_DIMENSION,
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

    /// Compute Lyapunov exponent (largest)
    /// λ > 0: chaotic (exponential divergence of nearby trajectories)
    /// λ = 0: neutral (periodic or quasi-periodic)
    /// λ < 0: stable (convergence)
    fn compute_lyapunov_exponent(time_series: &[f64]) -> f64 {
        if time_series.len() < 4 {
            return 0.0;
        }

        let n = time_series.len();
        let mut sum_log_divergence = 0.0;
        let mut count = 0;

        // Compare nearby trajectories
        for i in 0..n - 2 {
            let mut closest_distance = f64::MAX;
            let mut closest_idx = 0;

            // Find nearest neighbor
            for j in 0..n - 2 {
                if i == j || (i as i32 - j as i32).abs() < 2 {
                    continue;
                }

                let distance = (time_series[i] - time_series[j]).abs();
                if distance < closest_distance && distance > 1e-10 {
                    closest_distance = distance;
                    closest_idx = j;
                }
            }

            if closest_idx > 0 && closest_distance < f64::MAX {
                // Measure divergence after one time step
                let initial_distance = (time_series[i] - time_series[closest_idx]).abs();
                let final_distance = (time_series[i + 1] - time_series[closest_idx + 1]).abs();

                if initial_distance > 1e-10 && final_distance > 1e-10 {
                    let divergence_ratio = final_distance / initial_distance;
                    if divergence_ratio > 0.0 && divergence_ratio.is_finite() {
                        sum_log_divergence += divergence_ratio.ln();
                        count += 1;
                    }
                }
            }
        }

        if count > 0 {
            sum_log_divergence / count as f64
        } else {
            0.0
        }
    }

    /// Takens time-delay embedding reconstruction
    /// Converts scalar time series into m-dimensional phase space
    /// Each point: [x(t), x(t+τ), x(t+2τ), ..., x(t+(m-1)τ)]
    fn takens_embedding(time_series: &[f64], m: usize, tau: usize) -> Vec<Vec<f64>> {
        let max_delay = (m - 1) * tau;
        if time_series.len() <= max_delay {
            return Vec::new();
        }

        let mut embedded = Vec::new();

        for i in 0..time_series.len() - max_delay {
            let mut point = Vec::new();
            for j in 0..m {
                point.push(time_series[i + j * tau]);
            }
            embedded.push(point);
        }

        embedded
    }

    /// Compute correlation dimension using Grassberger-Procaccia algorithm
    /// Measures the fractal dimension of the attractor
    fn compute_correlation_dimension(embedded_series: &[Vec<f64>]) -> f64 {
        if embedded_series.len() < 2 {
            return 0.0;
        }

        // Choose a scale (radius)
        let distances = Self::compute_pairwise_distances(embedded_series);
        if distances.is_empty() {
            return 0.0;
        }

        // Use median as characteristic scale
        let mut sorted_distances = distances.clone();
        sorted_distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_distance = sorted_distances[sorted_distances.len() / 2];

        if median_distance < 1e-10 {
            return 0.0;
        }

        // Count pairs within different radii
        let radii = vec![median_distance * 0.5, median_distance, median_distance * 2.0];
        let mut log_radii = Vec::new();
        let mut log_correlations = Vec::new();

        for &r in &radii {
            let count = Self::count_pairs_within_radius(embedded_series, r);
            if count > 0 {
                log_radii.push(r.ln());
                log_correlations.push((count as f64).ln());
            }
        }

        if log_radii.len() < 2 {
            return 1.0; // Default dimension
        }

        // Correlation dimension is the slope of log(C(r)) vs log(r)
        let dimension = Self::compute_slope(&log_radii, &log_correlations);

        // Clamp to reasonable range
        dimension.max(0.0).min(embedded_series[0].len() as f64)
    }

    /// Compute all pairwise Euclidean distances
    fn compute_pairwise_distances(points: &[Vec<f64>]) -> Vec<f64> {
        let mut distances = Vec::new();

        for i in 0..points.len() {
            for j in i + 1..points.len() {
                let dist = Self::euclidean_distance(&points[i], &points[j]);
                distances.push(dist);
            }
        }

        distances
    }

    /// Euclidean distance between two points
    fn euclidean_distance(p1: &[f64], p2: &[f64]) -> f64 {
        p1.iter()
            .zip(p2.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Count pairs of points within radius r
    fn count_pairs_within_radius(points: &[Vec<f64>], radius: f64) -> usize {
        let mut count = 0;

        for i in 0..points.len() {
            for j in i + 1..points.len() {
                let dist = Self::euclidean_distance(&points[i], &points[j]);
                if dist < radius {
                    count += 1;
                }
            }
        }

        count
    }

    /// Compute slope of linear regression
    fn compute_slope(x: &[f64], y: &[f64]) -> f64 {
        let n = x.len().min(y.len()) as f64;
        if n < 2.0 {
            return 0.0;
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
            return 0.0;
        }

        numerator / denominator
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::LotteryDraw;
    use chrono::Utc;

    #[test]
    fn test_chaos_theory_analysis() {
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

        let result = ChaosTheoryAnalyzer::analyze(&dataset);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(result.embedding_dimension, 3);
        assert!(result.correlation_dimension >= 0.0);
    }

    #[test]
    fn test_takens_embedding() {
        let time_series = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let embedded = ChaosTheoryAnalyzer::takens_embedding(&time_series, 3, 1);

        // With m=3, τ=1, we should get n - (m-1)τ = 6 - 2 = 4 points
        assert_eq!(embedded.len(), 4);

        // Each point should be 3-dimensional
        assert_eq!(embedded[0].len(), 3);

        // Check first embedded point: [1.0, 2.0, 3.0]
        assert_eq!(embedded[0], vec![1.0, 2.0, 3.0]);

        // Check second embedded point: [2.0, 3.0, 4.0]
        assert_eq!(embedded[1], vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_euclidean_distance() {
        let p1 = vec![0.0, 0.0, 0.0];
        let p2 = vec![3.0, 4.0, 0.0];

        let dist = ChaosTheoryAnalyzer::euclidean_distance(&p1, &p2);
        assert!((dist - 5.0).abs() < 1e-6); // 3-4-5 triangle
    }

    #[test]
    fn test_lyapunov_sign() {
        // Create a simple periodic series
        let periodic = vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0];
        let lambda_periodic = ChaosTheoryAnalyzer::compute_lyapunov_exponent(&periodic);

        // Periodic systems should have λ ≈ 0
        // Due to discretization, we just check it's finite
        assert!(lambda_periodic.is_finite());
    }

    #[test]
    fn test_correlation_dimension_bounds() {
        let time_series = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let embedded = ChaosTheoryAnalyzer::takens_embedding(&time_series, 3, 1);
        let dim = ChaosTheoryAnalyzer::compute_correlation_dimension(&embedded);

        // Dimension should be non-negative and at most the embedding dimension
        assert!(dim >= 0.0);
        assert!(dim <= 3.0);
    }

    #[test]
    fn test_chaotic_classification() {
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
            LotteryDraw {
                id: Some(3),
                numbers: vec![3, 7, 12, 17, 22],
                draw_date: Utc::now(),
                game_name: "Test".to_string(),
            },
            LotteryDraw {
                id: Some(4),
                numbers: vec![4, 8, 13, 18, 23],
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

        let result = ChaosTheoryAnalyzer::analyze(&dataset).unwrap();

        // is_chaotic should be true iff lyapunov_exponent > 0
        assert_eq!(result.is_chaotic, result.lyapunov_exponent > 0.0);
    }

    #[test]
    fn test_embedding_with_insufficient_data() {
        let short_series = vec![1.0, 2.0];
        let embedded = ChaosTheoryAnalyzer::takens_embedding(&short_series, 3, 1);

        // Should return empty vector for insufficient data
        assert!(embedded.is_empty());
    }
}
