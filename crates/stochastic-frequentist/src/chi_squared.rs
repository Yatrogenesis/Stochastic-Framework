//! Chi-squared test implementation

use stochastic_core::{AnalysisResult, StochasticAnalyzer, StochasticError, TimeSeries};

pub struct ChiSquaredTest {
    significance_level: f64,
}

impl ChiSquaredTest {
    pub fn new() -> Self {
        Self {
            significance_level: 0.05,
        }
    }
}

impl Default for ChiSquaredTest {
    fn default() -> Self {
        Self::new()
    }
}

impl StochasticAnalyzer for ChiSquaredTest {
    fn name(&self) -> &str {
        "Chi-Squared Test"
    }

    fn analyze(&self, _data: &TimeSeries) -> Result<AnalysisResult, StochasticError> {
        // TODO: Implement chi-squared test
        Ok(AnalysisResult::new(self.name())
            .with_interpretation("Chi-squared test not yet implemented"))
    }

    fn required_sample_size(&self) -> usize {
        30
    }
}
