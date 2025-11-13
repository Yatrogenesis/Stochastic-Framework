use thiserror::Error;

/// Main error type for the STOCHASTIC framework
#[derive(Error, Debug)]
pub enum StochasticError {
    #[error("Insufficient data: need at least {required} samples, got {actual}")]
    InsufficientData { required: usize, actual: usize },

    #[error("Invalid parameter: {param} = {value}, expected {constraint}")]
    InvalidParameter {
        param: String,
        value: String,
        constraint: String,
    },

    #[error("Analysis failed: {reason}")]
    AnalysisFailed { reason: String },

    #[error("Numerical error: {details}")]
    NumericalError { details: String },

    #[error("Data validation failed: {message}")]
    ValidationError { message: String },

    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("CSV parsing error: {0}")]
    CsvError(#[from] csv::Error),

    #[error("Serialization error: {0}")]
    SerdeError(#[from] serde_json::Error),

    #[error("Domain mismatch: expected {expected:?}, got {actual:?}")]
    DomainMismatch { expected: String, actual: String },

    #[error("Convergence failure: {algorithm} did not converge after {iterations} iterations")]
    ConvergenceError {
        algorithm: String,
        iterations: usize,
    },

    #[error("Matrix operation error: {operation}")]
    MatrixError { operation: String },

    #[error("Unsupported operation: {message}")]
    UnsupportedOperation { message: String },
}

/// Result type alias for STOCHASTIC framework operations
pub type Result<T> = std::result::Result<T, StochasticError>;

impl StochasticError {
    /// Create an insufficient data error
    pub fn insufficient_data(required: usize, actual: usize) -> Self {
        StochasticError::InsufficientData { required, actual }
    }

    /// Create an invalid parameter error
    pub fn invalid_param(param: impl Into<String>, value: impl Into<String>, constraint: impl Into<String>) -> Self {
        StochasticError::InvalidParameter {
            param: param.into(),
            value: value.into(),
            constraint: constraint.into(),
        }
    }

    /// Create an analysis failed error
    pub fn analysis_failed(reason: impl Into<String>) -> Self {
        StochasticError::AnalysisFailed {
            reason: reason.into(),
        }
    }

    /// Create a numerical error
    pub fn numerical(details: impl Into<String>) -> Self {
        StochasticError::NumericalError {
            details: details.into(),
        }
    }

    /// Create a validation error
    pub fn validation(message: impl Into<String>) -> Self {
        StochasticError::ValidationError {
            message: message.into(),
        }
    }

    /// Create a convergence error
    pub fn convergence(algorithm: impl Into<String>, iterations: usize) -> Self {
        StochasticError::ConvergenceError {
            algorithm: algorithm.into(),
            iterations,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insufficient_data_error() {
        let err = StochasticError::insufficient_data(100, 50);
        assert!(matches!(err, StochasticError::InsufficientData { .. }));
        assert!(err.to_string().contains("need at least 100"));
    }

    #[test]
    fn test_invalid_parameter_error() {
        let err = StochasticError::invalid_param("alpha", "0.0", "> 0");
        assert!(matches!(err, StochasticError::InvalidParameter { .. }));
    }

    #[test]
    fn test_error_display() {
        let err = StochasticError::analysis_failed("chi-squared test failed");
        let msg = err.to_string();
        assert!(msg.contains("Analysis failed"));
        assert!(msg.contains("chi-squared"));
    }
}
