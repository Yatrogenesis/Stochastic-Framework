use crate::{AnalysisResult, StochasticError, TimeSeries};

/// Main trait for all stochastic analysis frameworks
///
/// All analysis methods in the STOCHASTIC framework implement this trait,
/// providing a unified interface for different mathematical methodologies.
///
/// # Thread Safety
///
/// Implementations must be `Send + Sync` to allow parallel execution
/// across multiple threads.
pub trait StochasticAnalyzer: Send + Sync {
    /// Returns the name of this analyzer/framework
    fn name(&self) -> &str;

    /// Perform the analysis on the given time series
    ///
    /// # Arguments
    ///
    /// * `data` - The time series to analyze
    ///
    /// # Returns
    ///
    /// * `Ok(AnalysisResult)` - The analysis results
    /// * `Err(StochasticError)` - If the analysis fails
    ///
    /// # Example
    ///
    /// ```ignore
    /// use stochastic_core::{StochasticAnalyzer, TimeSeries};
    ///
    /// let data = TimeSeries::from_values(vec![1.0, 2.0, 3.0]);
    /// let analyzer = MyAnalyzer::new();
    /// let result = analyzer.analyze(&data)?;
    /// println!("P-value: {:?}", result.p_value);
    /// ```
    fn analyze(&self, data: &TimeSeries) -> Result<AnalysisResult, StochasticError>;

    /// Validate that the data is suitable for this analysis
    ///
    /// This method checks if the input data meets the requirements
    /// for this particular analysis method (e.g., minimum sample size,
    /// domain constraints, etc.)
    ///
    /// # Arguments
    ///
    /// * `data` - The time series to validate
    ///
    /// # Returns
    ///
    /// * `Ok(true)` - Data is valid
    /// * `Ok(false)` - Data is invalid but no error occurred
    /// * `Err(StochasticError)` - Validation error with details
    fn validate(&self, data: &TimeSeries) -> Result<bool, StochasticError> {
        // Default implementation: check minimum sample size
        let min_size = self.required_sample_size();
        if data.len() < min_size {
            return Err(StochasticError::insufficient_data(min_size, data.len()));
        }
        Ok(true)
    }

    /// Returns the minimum required sample size for this analysis
    ///
    /// Different statistical methods have different sample size requirements
    /// to ensure reliable results.
    ///
    /// # Returns
    ///
    /// Minimum number of data points required
    fn required_sample_size(&self) -> usize {
        // Conservative default: at least 30 samples (CLT threshold)
        30
    }

    /// Optional: Provide recommendations based on the analysis
    ///
    /// Some analyzers may provide actionable recommendations based
    /// on their results.
    ///
    /// # Arguments
    ///
    /// * `result` - The analysis result to base recommendations on
    ///
    /// # Returns
    ///
    /// A vector of recommendation strings, or empty if none
    fn recommend(&self, _result: &AnalysisResult) -> Vec<String> {
        Vec::new()
    }
}

/// Trait for analyzers that support incremental/online updates
///
/// Some analyses can be updated incrementally as new data arrives,
/// which is more efficient than recomputing from scratch.
pub trait IncrementalAnalyzer: StochasticAnalyzer {
    /// Update the analysis with a new data point
    ///
    /// # Arguments
    ///
    /// * `value` - New data point value
    ///
    /// # Returns
    ///
    /// * `Ok(())` - Update successful
    /// * `Err(StochasticError)` - Update failed
    fn update(&mut self, value: f64) -> Result<(), StochasticError>;

    /// Reset the analyzer to initial state
    fn reset(&mut self);

    /// Get current analysis result without full recomputation
    fn current_result(&self) -> Result<AnalysisResult, StochasticError>;
}

/// Trait for analyzers that can be configured with parameters
pub trait ConfigurableAnalyzer: StochasticAnalyzer {
    /// Type of configuration this analyzer uses
    type Config;

    /// Create a new analyzer with custom configuration
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration parameters
    ///
    /// # Returns
    ///
    /// New analyzer instance
    fn with_config(config: Self::Config) -> Self;

    /// Get the current configuration
    fn config(&self) -> &Self::Config;
}

/// Trait for analyzers that support parallel execution
pub trait ParallelAnalyzer: StochasticAnalyzer {
    /// Analyze multiple time series in parallel
    ///
    /// # Arguments
    ///
    /// * `datasets` - Slice of time series to analyze
    ///
    /// # Returns
    ///
    /// Vector of analysis results, one per input dataset
    fn analyze_batch(&self, datasets: &[TimeSeries]) -> Result<Vec<AnalysisResult>, StochasticError> {
        // Default implementation using rayon
        datasets
            .iter()
            .map(|ts| self.analyze(ts))
            .collect()
    }
}

/// Trait for analyzers that can visualize their results
pub trait Visualizable {
    /// Generate visualization data
    ///
    /// Returns data suitable for plotting (e.g., for CLI or GUI)
    fn visualization_data(&self, result: &AnalysisResult) -> VisualizationData;
}

/// Data structure for visualization
#[derive(Debug, Clone)]
pub struct VisualizationData {
    pub plot_type: PlotType,
    pub x_values: Vec<f64>,
    pub y_values: Vec<f64>,
    pub labels: Vec<String>,
    pub title: String,
    pub x_label: String,
    pub y_label: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlotType {
    Line,
    Scatter,
    Histogram,
    BarChart,
    Heatmap,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AnalysisResult;

    // Mock analyzer for testing
    struct MockAnalyzer {
        min_samples: usize,
    }

    impl StochasticAnalyzer for MockAnalyzer {
        fn name(&self) -> &str {
            "Mock Analyzer"
        }

        fn analyze(&self, data: &TimeSeries) -> Result<AnalysisResult, StochasticError> {
            self.validate(data)?;
            Ok(AnalysisResult::new(self.name())
                .with_metric("sample_size", data.len() as f64)
                .with_interpretation("Mock analysis completed"))
        }

        fn required_sample_size(&self) -> usize {
            self.min_samples
        }
    }

    #[test]
    fn test_analyzer_validation() {
        let analyzer = MockAnalyzer { min_samples: 10 };

        // Valid data
        let valid_data = TimeSeries::from_values(vec![1.0; 15]);
        assert!(analyzer.validate(&valid_data).is_ok());

        // Invalid data
        let invalid_data = TimeSeries::from_values(vec![1.0; 5]);
        assert!(analyzer.validate(&invalid_data).is_err());
    }

    #[test]
    fn test_analyzer_analysis() {
        let analyzer = MockAnalyzer { min_samples: 5 };
        let data = TimeSeries::from_values(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        let result = analyzer.analyze(&data).unwrap();
        assert_eq!(result.framework, "Mock Analyzer");
        assert_eq!(result.metrics.get("sample_size"), Some(&5.0));
    }
}
