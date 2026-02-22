//! Normalization and Outlier Detection Framework
//!
//! Implements statistical normalization techniques and outlier detection methods,
//! including Z-score analysis and Shapiro-Wilk normality testing. Provides tools
//! for data quality assessment and distributional analysis.

pub mod zscore;
pub mod shapiro_wilk;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

pub use zscore::{ZScoreAnalyzer, ZScoreConfig, OutlierMethod};
pub use shapiro_wilk::{ShapiroWilkAnalyzer, ShapiroWilkConfig};
