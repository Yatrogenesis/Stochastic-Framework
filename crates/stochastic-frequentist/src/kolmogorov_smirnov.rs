//! Kolmogorov-Smirnov test implementation

use stochastic_core::{AnalysisResult, StochasticAnalyzer, StochasticError, TimeSeries};

pub struct KolmogorovSmirnovTest {
    significance_level: f64,
}

impl KolmogorovSmirnovTest {
    pub fn new() -> Self {
        Self {
            significance_level: 0.05,
        }
    }
}

impl Default for KolmogorovSmirnovTest {
    fn default() -> Self {
        Self::new()
    }
}

impl StochasticAnalyzer for KolmogorovSmirnovTest {
    fn name(&self) -> &str {
        "Kolmogorov-Smirnov Test"
    }

    fn analyze(&self, _data: &TimeSeries) -> Result<AnalysisResult, StochasticError> {
        // TODO: Implement KS test
        Ok(AnalysisResult::new(self.name())
            .with_interpretation("KS test not yet implemented"))
    }

    fn required_sample_size(&self) -> usize {
        20
    }
}
