/// Main analysis module containing all 9 mathematical frameworks
pub mod frequentist;
pub mod bayesian;
pub mod ergodic;
pub mod range_equilibrium;
pub mod galois;
pub mod multifractal;
pub mod chaos;
pub mod normalization;
pub mod critical_path;

use crate::models::*;
use anyhow::Result;
use rayon::prelude::*;

/// Main analyzer that coordinates all 9 frameworks
pub struct LottoAnalyzer {
    pub dataset: LotteryDataset,
}

impl LottoAnalyzer {
    pub fn new(dataset: LotteryDataset) -> Self {
        Self { dataset }
    }

    /// Run complete analysis with all 9 frameworks
    pub fn analyze(&self, confidence_level: ConfidenceLevel) -> Result<CompleteAnalysisResult> {
        log::info!("Starting complete analysis with {} draws", self.dataset.draws.len());

        // Run all analyses sequentially (can be optimized with rayon later)
        let frequentist_result = frequentist::FrequentistAnalyzer::analyze(&self.dataset)?;
        let bayesian_result = bayesian::BayesianAnalyzer::analyze(&self.dataset)?;
        let ergodic_result = ergodic::ErgodicAnalyzer::analyze(&self.dataset)?;
        let range_result = range_equilibrium::RangeEquilibriumAnalyzer::analyze(&self.dataset)?;
        let galois_result = galois::GaloisFieldAnalyzer::analyze(&self.dataset)?;
        let multifractal_result = multifractal::MultifractalAnalyzer::analyze(&self.dataset)?;
        let chaos_result = chaos::ChaosTheoryAnalyzer::analyze(&self.dataset)?;
        let normalization_result = normalization::NormalizationAnalyzer::analyze(&self.dataset)?;

        // Critical path uses results from all previous analyses
        let critical_path_result = critical_path::CriticalPathAnalyzer::analyze(
            &self.dataset,
            &frequentist_result,
            &bayesian_result,
            confidence_level,
        )?;

        let dataset_info = DatasetInfo {
            total_draws: self.dataset.draws.len(),
            total_numbers_extracted: self.dataset.draws.iter()
                .map(|d| d.numbers.len())
                .sum(),
            date_range: self.get_date_range(),
        };

        Ok(CompleteAnalysisResult {
            dataset_info,
            frequentist: frequentist_result,
            bayesian: bayesian_result,
            ergodic: ergodic_result,
            range_equilibrium: range_result,
            galois_field: galois_result,
            multifractal: multifractal_result,
            chaos_theory: chaos_result,
            normalization: normalization_result,
            critical_path: critical_path_result,
            confidence_level: confidence_level.as_f64(),
            timestamp: chrono::Utc::now(),
        })
    }

    fn get_date_range(&self) -> (chrono::DateTime<chrono::Utc>, chrono::DateTime<chrono::Utc>) {
        let dates: Vec<_> = self.dataset.draws.iter().map(|d| d.draw_date).collect();
        let min_date = *dates.iter().min().unwrap();
        let max_date = *dates.iter().max().unwrap();
        (min_date, max_date)
    }
}
