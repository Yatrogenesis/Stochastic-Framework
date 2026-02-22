/// Galois Field Analysis - GF(29) Finite Geometry
/// Uses generator g=2 and analyzes quadratic residues
use crate::models::{GaloisFieldResult, LotteryDataset};
use anyhow::Result;

pub struct GaloisFieldAnalyzer;

impl GaloisFieldAnalyzer {
    const FIELD_ORDER: u32 = 29; // Prime field GF(29)
    const GENERATOR: u32 = 2; // Primitive root modulo 29

    /// Perform Galois field analysis
    pub fn analyze(dataset: &LotteryDataset) -> Result<GaloisFieldResult> {
        log::info!("Starting Galois field GF(29) analysis");

        // Compute quadratic residues in GF(29)
        let quadratic_residues = Self::compute_quadratic_residues();

        // Compute combinatorial dimension
        // For choosing 5 numbers from 28: log₂(C(28,5))
        let combinatorial_dimension = Self::compute_combinatorial_dimension(
            dataset.max_number as usize,
            dataset.numbers_per_draw,
        );

        log::info!(
            "Galois Field Analysis: field_order={}, generator={}, QR_count={}, comb_dim={:.4}",
            Self::FIELD_ORDER,
            Self::GENERATOR,
            quadratic_residues.len(),
            combinatorial_dimension
        );

        Ok(GaloisFieldResult {
            field_order: Self::FIELD_ORDER,
            generator: Self::GENERATOR,
            quadratic_residues,
            combinatorial_dimension,
        })
    }

    /// Compute quadratic residues in GF(29)
    /// QR = {a² mod p : a ∈ [1, p-1]}
    fn compute_quadratic_residues() -> Vec<u32> {
        let mut residues = Vec::new();
        let p = Self::FIELD_ORDER;

        for a in 1..p {
            let residue = (a * a) % p;
            if !residues.contains(&residue) {
                residues.push(residue);
            }
        }

        residues.sort();
        residues
    }

    /// Compute combinatorial dimension: log₂(C(n, k))
    /// Uses formula: log₂(n!/(k!(n-k)!))
    fn compute_combinatorial_dimension(n: usize, k: usize) -> f64 {
        if k > n {
            return 0.0;
        }

        // Use logarithm properties to avoid overflow
        // log(C(n,k)) = log(n!) - log(k!) - log((n-k)!)
        let log_factorial = |x: usize| -> f64 {
            if x <= 1 {
                return 0.0;
            }
            (1..=x).map(|i| (i as f64).ln()).sum::<f64>()
        };

        let log_comb = log_factorial(n) - log_factorial(k) - log_factorial(n - k);

        // Convert from natural log to log base 2
        log_comb / std::f64::consts::LN_2
    }

    /// Check if a number is a quadratic residue
    #[allow(dead_code)]
    fn is_quadratic_residue(n: u32) -> bool {
        let p = Self::FIELD_ORDER;
        let n = n % p;

        if n == 0 {
            return true;
        }

        // Using Euler's criterion: a^((p-1)/2) ≡ 1 (mod p) iff a is a QR
        Self::mod_pow(n, (p - 1) / 2, p) == 1
    }

    /// Modular exponentiation: base^exp mod m
    fn mod_pow(mut base: u32, mut exp: u32, modulus: u32) -> u32 {
        if modulus == 1 {
            return 0;
        }

        let mut result = 1;
        base %= modulus;

        while exp > 0 {
            if exp % 2 == 1 {
                result = (result as u64 * base as u64 % modulus as u64) as u32;
            }
            exp >>= 1;
            base = (base as u64 * base as u64 % modulus as u64) as u32;
        }

        result
    }

    /// Verify generator is primitive root
    #[allow(dead_code)]
    fn verify_generator() -> bool {
        let g = Self::GENERATOR;
        let p = Self::FIELD_ORDER;

        // A generator must have order p-1
        // Check if g^((p-1)/q) ≢ 1 (mod p) for all prime divisors q of p-1
        // For p=29, p-1=28=2²×7
        let prime_divisors = vec![2, 7];

        for q in prime_divisors {
            if Self::mod_pow(g, (p - 1) / q, p) == 1 {
                return false;
            }
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::LotteryDraw;
    use chrono::Utc;

    #[test]
    fn test_galois_field_analysis() {
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
            max_number: 28,
            numbers_per_draw: 5,
        };

        let result = GaloisFieldAnalyzer::analyze(&dataset);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(result.field_order, 29);
        assert_eq!(result.generator, 2);
        assert!(result.quadratic_residues.len() > 0);
        assert!(result.combinatorial_dimension > 0.0);
    }

    #[test]
    fn test_quadratic_residues() {
        let residues = GaloisFieldAnalyzer::compute_quadratic_residues();

        // In GF(29), there should be (29-1)/2 = 14 non-zero quadratic residues
        assert_eq!(residues.len(), 14);

        // All residues should be in range [1, 28]
        for &r in &residues {
            assert!(r >= 1 && r < 29);
        }

        // Residues should be sorted
        for i in 1..residues.len() {
            assert!(residues[i] > residues[i - 1]);
        }
    }

    #[test]
    fn test_quadratic_residue_known_values() {
        // Known QRs in GF(29): 1, 4, 5, 6, 7, 9, 13, 16, 20, 22, 23, 24, 25, 28
        let residues = GaloisFieldAnalyzer::compute_quadratic_residues();

        // Check some known QRs
        assert!(residues.contains(&1)); // 1² = 1
        assert!(residues.contains(&4)); // 2² = 4
        assert!(residues.contains(&9)); // 3² = 9
        assert!(residues.contains(&16)); // 4² = 16
        assert!(residues.contains(&25)); // 5² = 25
    }

    #[test]
    fn test_combinatorial_dimension() {
        // C(28, 5) = 98,280
        // log₂(98,280) ≈ 16.58
        let dim = GaloisFieldAnalyzer::compute_combinatorial_dimension(28, 5);
        assert!(dim > 16.0 && dim < 17.0);

        // C(5, 5) = 1, log₂(1) = 0
        let dim_trivial = GaloisFieldAnalyzer::compute_combinatorial_dimension(5, 5);
        assert!((dim_trivial - 0.0).abs() < 0.001);

        // C(10, 2) = 45, log₂(45) ≈ 5.49
        let dim_small = GaloisFieldAnalyzer::compute_combinatorial_dimension(10, 2);
        assert!(dim_small > 5.0 && dim_small < 6.0);
    }

    #[test]
    fn test_mod_pow() {
        // 2^10 mod 29 = 1024 mod 29 = 7
        assert_eq!(GaloisFieldAnalyzer::mod_pow(2, 10, 29), 7);

        // 3^5 mod 7 = 243 mod 7 = 5
        assert_eq!(GaloisFieldAnalyzer::mod_pow(3, 5, 7), 5);

        // a^0 mod m = 1
        assert_eq!(GaloisFieldAnalyzer::mod_pow(5, 0, 13), 1);

        // a^1 mod m = a mod m
        assert_eq!(GaloisFieldAnalyzer::mod_pow(7, 1, 11), 7);
    }

    #[test]
    fn test_generator_verification() {
        // Generator 2 should be a primitive root modulo 29
        assert!(GaloisFieldAnalyzer::verify_generator());
    }

    #[test]
    fn test_is_quadratic_residue() {
        // 1 is always a QR (1² = 1)
        assert!(GaloisFieldAnalyzer::is_quadratic_residue(1));

        // 4 is a QR (2² = 4)
        assert!(GaloisFieldAnalyzer::is_quadratic_residue(4));

        // 0 is considered a QR
        assert!(GaloisFieldAnalyzer::is_quadratic_residue(0));

        // Test with the actual computed QRs
        let residues = GaloisFieldAnalyzer::compute_quadratic_residues();
        for &r in &residues {
            assert!(GaloisFieldAnalyzer::is_quadratic_residue(r));
        }
    }
}
