/// Core data models for the LOTTO Analysis Framework
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

/// Represents a single lottery drawing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LotteryDraw {
    pub id: Option<u64>,
    pub numbers: Vec<u32>,
    pub draw_date: DateTime<Utc>,
    pub game_name: String,
}

/// Dataset containing multiple lottery draws
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LotteryDataset {
    pub draws: Vec<LotteryDraw>,
    pub min_number: u32,
    pub max_number: u32,
    pub numbers_per_draw: usize,
}

/// Result of frequentist analysis (χ² test)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrequentistResult {
    pub chi_squared: f64,
    pub p_value: f64,
    pub degrees_of_freedom: usize,
    pub frequency_distribution: Vec<(u32, usize)>,
    pub is_uniform: bool,
}

/// Result of Bayesian inference with Dirichlet distribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BayesianResult {
    pub posterior_probabilities: Vec<(u32, f64)>,
    pub credible_intervals: Vec<(u32, f64, f64)>,  // (number, lower_95%, upper_95%)
    pub top_numbers: Vec<u32>,
}

/// Result of ergodic analysis (Birkhoff-Khinchin theorem)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErgodicResult {
    pub temporal_mean: f64,
    pub ensemble_mean: f64,
    pub runs_test_z_score: f64,
    pub is_ergodic: bool,
    pub autocorrelation: Vec<f64>,
}

/// Result of range equilibrium analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RangeEquilibriumResult {
    pub low_percentage: f64,
    pub medium_percentage: f64,
    pub high_percentage: f64,
    pub even_percentage: f64,
    pub odd_percentage: f64,
    pub chi_squared_ranges: f64,
    pub p_value_ranges: f64,
    pub is_balanced: bool,
}

/// Result of Galois field (finite geometry) analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GaloisFieldResult {
    pub field_order: u32,
    pub generator: u32,
    pub quadratic_residues: Vec<u32>,
    pub combinatorial_dimension: f64,
}

/// Result of multifractal analysis (DFA, Hurst exponent)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultifractalResult {
    pub hurst_exponent: f64,
    pub fractal_dimension: f64,
    pub normalized_entropy: f64,
    pub is_persistent: bool,
    pub interpretation: String,
}

/// Result of chaos theory analysis (Lyapunov, Takens)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChaosTheoryResult {
    pub lyapunov_exponent: f64,
    pub correlation_dimension: f64,
    pub is_chaotic: bool,
    pub embedding_dimension: usize,
}

/// Result of factorization and normalization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizationResult {
    pub outliers: Vec<u32>,
    pub shapiro_wilk_w: f64,
    pub shapiro_wilk_p: f64,
    pub is_normal: bool,
    pub box_cox_lambda: f64,
}

/// Critical path result with multi-objective optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CriticalPathResult {
    pub recommended_combination: Vec<u32>,
    pub score: f64,
    pub properties: CombinationProperties,
}

/// Properties of a recommended combination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CombinationProperties {
    pub sum: u32,
    pub mean: f64,
    pub median: f64,
    pub std_dev: f64,
    pub range: u32,
    pub even_count: usize,
    pub odd_count: usize,
    pub prime_count: usize,
    pub primes: Vec<u32>,
    pub low_count: usize,
    pub medium_count: usize,
    pub high_count: usize,
}

/// Complete analysis result containing all 9 frameworks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompleteAnalysisResult {
    pub dataset_info: DatasetInfo,
    pub frequentist: FrequentistResult,
    pub bayesian: BayesianResult,
    pub ergodic: ErgodicResult,
    pub range_equilibrium: RangeEquilibriumResult,
    pub galois_field: GaloisFieldResult,
    pub multifractal: MultifractalResult,
    pub chaos_theory: ChaosTheoryResult,
    pub normalization: NormalizationResult,
    pub critical_path: CriticalPathResult,
    pub confidence_level: f64,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetInfo {
    pub total_draws: usize,
    pub total_numbers_extracted: usize,
    pub date_range: (DateTime<Utc>, DateTime<Utc>),
}

/// Confidence level thresholds
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConfidenceLevel {
    Low = 80,      // 0.80
    Medium = 90,   // 0.90
    High = 95,     // 0.95
    VeryHigh = 99, // 0.99
}

impl ConfidenceLevel {
    pub fn as_f64(&self) -> f64 {
        match self {
            ConfidenceLevel::Low => 0.80,
            ConfidenceLevel::Medium => 0.90,
            ConfidenceLevel::High => 0.95,
            ConfidenceLevel::VeryHigh => 0.99,
        }
    }
}
