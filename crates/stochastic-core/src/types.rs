use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Represents a single data point in a time series
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataPoint {
    /// Optional timestamp for the data point
    pub timestamp: Option<chrono::DateTime<chrono::Utc>>,
    /// The value of the data point
    pub value: f64,
    /// Additional metadata associated with this point
    pub metadata: HashMap<String, String>,
}

impl DataPoint {
    /// Create a new data point with just a value
    pub fn new(value: f64) -> Self {
        Self {
            timestamp: None,
            value,
            metadata: HashMap::new(),
        }
    }

    /// Create a data point with timestamp
    pub fn with_timestamp(value: f64, timestamp: chrono::DateTime<chrono::Utc>) -> Self {
        Self {
            timestamp: Some(timestamp),
            value,
            metadata: HashMap::new(),
        }
    }

    /// Add metadata to this data point
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

/// Domain of a time series
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Domain {
    /// Discrete values (integers)
    Discrete,
    /// Continuous values (real numbers)
    Continuous,
    /// Categorical data
    Categorical,
}

/// Represents a time series of data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeries {
    /// The data points in the time series
    pub points: Vec<DataPoint>,
    /// Optional sampling rate (Hz)
    pub sampling_rate: Option<f64>,
    /// Domain of the data
    pub domain: Domain,
    /// Name/description of the time series
    pub name: Option<String>,
}

impl TimeSeries {
    /// Create a new time series from a vector of values
    pub fn from_values(values: Vec<f64>) -> Self {
        Self {
            points: values.into_iter().map(DataPoint::new).collect(),
            sampling_rate: None,
            domain: Domain::Continuous,
            name: None,
        }
    }

    /// Create a new time series with domain specification
    pub fn new(points: Vec<DataPoint>, domain: Domain) -> Self {
        Self {
            points,
            sampling_rate: None,
            domain,
            name: None,
        }
    }

    /// Load time series from CSV file
    pub fn from_csv<P: AsRef<Path>>(path: P) -> crate::Result<Self> {
        let mut reader = csv::Reader::from_path(path)?;
        let mut points = Vec::new();

        for result in reader.records() {
            let record = result?;
            if let Some(value_str) = record.get(0) {
                let value: f64 = value_str
                    .parse()
                    .map_err(|_| crate::error::StochasticError::validation(
                        format!("Failed to parse value: {}", value_str)
                    ))?;
                points.push(DataPoint::new(value));
            }
        }

        Ok(Self {
            points,
            sampling_rate: None,
            domain: Domain::Continuous,
            name: None,
        })
    }

    /// Get the length of the time series
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Check if the time series is empty
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Get values as a vector
    pub fn values(&self) -> Vec<f64> {
        self.points.iter().map(|p| p.value).collect()
    }

    /// Set the sampling rate
    pub fn with_sampling_rate(mut self, rate: f64) -> Self {
        self.sampling_rate = Some(rate);
        self
    }

    /// Set the name
    pub fn with_name(mut self, name: String) -> Self {
        self.name = Some(name);
        self
    }

    /// Set the domain
    pub fn with_domain(mut self, domain: Domain) -> Self {
        self.domain = domain;
        self
    }

    /// Get basic statistics
    pub fn statistics(&self) -> Statistics {
        let values = self.values();
        Statistics::compute(&values)
    }
}

/// Basic statistical measures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Statistics {
    pub mean: f64,
    pub median: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
    pub count: usize,
}

impl Statistics {
    /// Compute statistics from a slice of values
    pub fn compute(values: &[f64]) -> Self {
        let count = values.len();
        if count == 0 {
            return Self {
                mean: 0.0,
                median: 0.0,
                std_dev: 0.0,
                min: 0.0,
                max: 0.0,
                count: 0,
            };
        }

        let mean = values.iter().sum::<f64>() / count as f64;
        let variance = values.iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f64>() / count as f64;
        let std_dev = variance.sqrt();

        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = if count % 2 == 0 {
            (sorted[count / 2 - 1] + sorted[count / 2]) / 2.0
        } else {
            sorted[count / 2]
        };

        let min = sorted[0];
        let max = sorted[count - 1];

        Self {
            mean,
            median,
            std_dev,
            min,
            max,
            count,
        }
    }
}

/// Configuration for analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisConfig {
    /// Confidence level (e.g., 0.95 for 95%)
    pub confidence_level: f64,
    /// Number of bootstrap iterations
    pub bootstrap_iterations: usize,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
    /// Maximum number of iterations for iterative algorithms
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            confidence_level: 0.95,
            bootstrap_iterations: 10000,
            seed: None,
            max_iterations: 1000,
            tolerance: 1e-6,
        }
    }
}

impl AnalysisConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_confidence_level(mut self, level: f64) -> Self {
        self.confidence_level = level;
        self
    }

    pub fn with_bootstrap_iterations(mut self, iterations: usize) -> Self {
        self.bootstrap_iterations = iterations;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    pub fn with_max_iterations(mut self, max_iter: usize) -> Self {
        self.max_iterations = max_iter;
        self
    }

    pub fn with_tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }
}

/// Result of a statistical analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisResult {
    /// Name of the framework that produced this result
    pub framework: String,
    /// Computed metrics
    pub metrics: HashMap<String, f64>,
    /// P-value (if applicable)
    pub p_value: Option<f64>,
    /// Confidence interval (if applicable)
    pub confidence_interval: Option<(f64, f64)>,
    /// Human-readable interpretation
    pub interpretation: String,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl AnalysisResult {
    pub fn new(framework: impl Into<String>) -> Self {
        Self {
            framework: framework.into(),
            metrics: HashMap::new(),
            p_value: None,
            confidence_interval: None,
            interpretation: String::new(),
            metadata: HashMap::new(),
        }
    }

    pub fn with_metric(mut self, key: impl Into<String>, value: f64) -> Self {
        self.metrics.insert(key.into(), value);
        self
    }

    pub fn with_p_value(mut self, p_value: f64) -> Self {
        self.p_value = Some(p_value);
        self
    }

    pub fn with_confidence_interval(mut self, lower: f64, upper: f64) -> Self {
        self.confidence_interval = Some((lower, upper));
        self
    }

    pub fn with_interpretation(mut self, interpretation: impl Into<String>) -> Self {
        self.interpretation = interpretation.into();
        self
    }

    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_datapoint_creation() {
        let dp = DataPoint::new(42.0);
        assert_eq!(dp.value, 42.0);
        assert!(dp.timestamp.is_none());
        assert!(dp.metadata.is_empty());
    }

    #[test]
    fn test_timeseries_from_values() {
        let ts = TimeSeries::from_values(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(ts.len(), 5);
        assert_eq!(ts.values(), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_statistics() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = Statistics::compute(&values);
        assert_eq!(stats.mean, 3.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);
        assert_eq!(stats.count, 5);
    }

    #[test]
    fn test_analysis_config_builder() {
        let config = AnalysisConfig::new()
            .with_confidence_level(0.99)
            .with_bootstrap_iterations(5000)
            .with_seed(42);

        assert_eq!(config.confidence_level, 0.99);
        assert_eq!(config.bootstrap_iterations, 5000);
        assert_eq!(config.seed, Some(42));
    }

    #[test]
    fn test_analysis_result_builder() {
        let result = AnalysisResult::new("Test Framework")
            .with_metric("statistic", 3.14)
            .with_p_value(0.05)
            .with_interpretation("Test passed");

        assert_eq!(result.framework, "Test Framework");
        assert_eq!(result.metrics.get("statistic"), Some(&3.14));
        assert_eq!(result.p_value, Some(0.05));
    }
}
