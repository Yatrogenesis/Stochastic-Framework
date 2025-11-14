//! Coherence Analysis
//!
//! Measures frequency-domain correlation between two signals using magnitude-squared coherence.
//! Quantifies the linear relationship between signals as a function of frequency.
//!
//! # Mathematical Foundation
//!
//! **Magnitude-Squared Coherence:**
//! ```text
//! Cxy²(f) = |Pxy(f)|² / [Pxx(f) · Pyy(f)]
//! ```
//!
//! where:
//! - Pxy(f) = cross power spectral density
//! - Pxx(f) = auto power spectral density of x
//! - Pyy(f) = auto power spectral density of y
//!
//! **Properties:**
//! - 0 ≤ Cxy²(f) ≤ 1
//! - Cxy²(f) = 1: perfect linear relationship at frequency f
//! - Cxy²(f) = 0: no linear relationship at frequency f
//! - Analogous to correlation coefficient R² in time domain
//!
//! **Cross Power Spectral Density:**
//! ```text
//! Pxy(f) = E[X(f) · Y*(f)]
//! ```
//! where Y*(f) is complex conjugate of Y(f)
//!
//! # References
//!
//! - Carter, G.C. (1987). "Coherence and time delay estimation"
//! - Bendat, J.S., & Piersol, A.G. (2010). "Random Data: Analysis and Measurement Procedures" (4th ed.)
//! - Welch, P.D. (1967). "The use of FFT for estimation of power spectra"
//! - Halliday, D.M., et al. (1995). "A framework for the analysis of mixed time series/point process data"
//! - Nolte, G., et al. (2004). "Identifying true brain interaction from EEG data using the imaginary part of coherency"

use stochastic_core::{AnalysisResult, StochasticAnalyzer, StochasticError, TimeSeries};
use rustfft::{FftPlanner, num_complex::Complex};
use std::f64::consts::PI;

/// Configuration for coherence analysis
#[derive(Debug, Clone)]
pub struct CoherenceConfig {
    /// Sampling frequency (Hz)
    pub sampling_freq: f64,
    /// Segment length for Welch's method
    pub segment_length: usize,
    /// Overlap between segments
    pub overlap: usize,
    /// Window function
    pub window: WindowType,
    /// Return only positive frequencies
    pub one_sided: bool,
}

/// Window function
#[derive(Debug, Clone, Copy)]
pub enum WindowType {
    Hann,
    Hamming,
    Blackman,
    Rectangular,
}

impl Default for CoherenceConfig {
    fn default() -> Self {
        Self {
            sampling_freq: 1.0,
            segment_length: 256,
            overlap: 128,
            window: WindowType::Hann,
            one_sided: true,
        }
    }
}

/// Coherence Analyzer
pub struct CoherenceAnalyzer {
    config: CoherenceConfig,
}

impl CoherenceAnalyzer {
    /// Create new coherence analyzer
    pub fn new() -> Self {
        Self {
            config: CoherenceConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: CoherenceConfig) -> Self {
        Self { config }
    }

    /// Set sampling frequency
    pub fn with_sampling_freq(mut self, freq: f64) -> Self {
        self.config.sampling_freq = freq;
        self
    }

    /// Set segment parameters
    pub fn with_segments(mut self, length: usize, overlap: usize) -> Self {
        self.config.segment_length = length;
        self.config.overlap = overlap;
        self
    }

    /// Apply window function
    fn apply_window(&self, signal: &[f64]) -> Vec<f64> {
        let n = signal.len();
        match self.config.window {
            WindowType::Rectangular => signal.to_vec(),
            WindowType::Hann => {
                signal.iter()
                    .enumerate()
                    .map(|(i, &x)| x * (0.5 - 0.5 * (2.0 * PI * i as f64 / (n - 1) as f64).cos()))
                    .collect()
            }
            WindowType::Hamming => {
                signal.iter()
                    .enumerate()
                    .map(|(i, &x)| x * (0.54 - 0.46 * (2.0 * PI * i as f64 / (n - 1) as f64).cos()))
                    .collect()
            }
            WindowType::Blackman => {
                let a0 = 0.42;
                let a1 = 0.5;
                let a2 = 0.08;
                signal.iter()
                    .enumerate()
                    .map(|(i, &x)| {
                        let t = i as f64 / (n - 1) as f64;
                        x * (a0 - a1 * (2.0 * PI * t).cos() + a2 * (4.0 * PI * t).cos())
                    })
                    .collect()
            }
        }
    }

    /// Compute FFT of segment
    fn compute_fft(&self, segment: &[f64]) -> Vec<Complex<f64>> {
        let windowed = self.apply_window(segment);
        
        let mut buffer: Vec<Complex<f64>> = windowed.iter()
            .map(|&x| Complex::new(x, 0.0))
            .collect();

        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(buffer.len());
        fft.process(&mut buffer);

        buffer
    }

    /// Compute magnitude-squared coherence between two signals
    pub fn compute_coherence(
        &self,
        signal_x: &[f64],
        signal_y: &[f64],
    ) -> Result<Vec<f64>, StochasticError> {
        if signal_x.len() != signal_y.len() {
            return Err(StochasticError::validation(
                format!("Signals must have same length: {} vs {}", signal_x.len(), signal_y.len())
            ));
        }

        if self.config.segment_length > signal_x.len() {
            return Err(StochasticError::invalid_param(
                "segment_length",
                self.config.segment_length.to_string(),
                format!("<= signal length ({})", signal_x.len())
            ));
        }

        let step = self.config.segment_length - self.config.overlap;
        let mut num_segments = 0;

        // Initialize accumulators
        let mut pxx_sum = vec![Complex::new(0.0, 0.0); self.config.segment_length];
        let mut pyy_sum = vec![Complex::new(0.0, 0.0); self.config.segment_length];
        let mut pxy_sum = vec![Complex::new(0.0, 0.0); self.config.segment_length];

        // Process overlapping segments
        let mut start = 0;
        while start + self.config.segment_length <= signal_x.len() {
            let segment_x = &signal_x[start..start + self.config.segment_length];
            let segment_y = &signal_y[start..start + self.config.segment_length];

            let fft_x = self.compute_fft(segment_x);
            let fft_y = self.compute_fft(segment_y);

            // Accumulate cross and auto spectra
            for i in 0..self.config.segment_length {
                // Pxx = X · X*
                pxx_sum[i] += fft_x[i] * fft_x[i].conj();
                // Pyy = Y · Y*
                pyy_sum[i] += fft_y[i] * fft_y[i].conj();
                // Pxy = X · Y*
                pxy_sum[i] += fft_x[i] * fft_y[i].conj();
            }

            num_segments += 1;
            start += step;
        }

        if num_segments == 0 {
            return Err(StochasticError::validation(
                "No valid segments for coherence calculation"
            ));
        }

        // Average and compute coherence
        let coherence: Vec<f64> = (0..self.config.segment_length)
            .map(|i| {
                let pxx = (pxx_sum[i] / num_segments as f64).norm();
                let pyy = (pyy_sum[i] / num_segments as f64).norm();
                let pxy = (pxy_sum[i] / num_segments as f64).norm();

                if pxx * pyy > 0.0 {
                    let coh_sq = (pxy * pxy) / (pxx * pyy);
                    coh_sq.min(1.0) // Ensure ≤ 1 due to numerical errors
                } else {
                    0.0
                }
            })
            .collect();

        Ok(coherence)
    }

    /// Get frequency bins
    pub fn frequency_bins(&self) -> Vec<f64> {
        let freq_resolution = self.config.sampling_freq / self.config.segment_length as f64;
        (0..self.config.segment_length)
            .map(|k| k as f64 * freq_resolution)
            .collect()
    }

    /// Compute mean coherence over all frequencies
    pub fn mean_coherence(&self, coherence: &[f64]) -> f64 {
        let max_len = if self.config.one_sided {
            coherence.len() / 2
        } else {
            coherence.len()
        };

        coherence[..max_len].iter().sum::<f64>() / max_len as f64
    }

    /// Find peak coherence frequency
    pub fn peak_coherence(&self, coherence: &[f64]) -> (f64, f64) {
        let max_len = if self.config.one_sided {
            coherence.len() / 2
        } else {
            coherence.len()
        };

        let freqs = self.frequency_bins();

        let (max_idx, max_coh) = coherence[1..max_len]
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, &c)| (idx + 1, c))
            .unwrap_or((0, 0.0));

        (freqs[max_idx], max_coh)
    }

    /// Compute coherence in frequency band
    pub fn band_coherence(&self, coherence: &[f64], low_freq: f64, high_freq: f64) -> f64 {
        let freqs = self.frequency_bins();
        let max_len = if self.config.one_sided {
            coherence.len() / 2
        } else {
            coherence.len()
        };

        let band_values: Vec<f64> = freqs[..max_len].iter()
            .zip(coherence[..max_len].iter())
            .filter_map(|(&f, &c)| {
                if f >= low_freq && f <= high_freq {
                    Some(c)
                } else {
                    None
                }
            })
            .collect();

        if band_values.is_empty() {
            0.0
        } else {
            band_values.iter().sum::<f64>() / band_values.len() as f64
        }
    }

    /// Analyze coherence between two time series
    pub fn analyze_two_signals(
        &self,
        signal_x: &TimeSeries,
        signal_y: &TimeSeries,
    ) -> Result<AnalysisResult, StochasticError> {
        // Validate both signals
        self.validate(signal_x)?;
        self.validate(signal_y)?;

        if signal_x.len() != signal_y.len() {
            return Err(StochasticError::validation(
                format!("Signals must have equal length: {} vs {}", signal_x.len(), signal_y.len())
            ));
        }

        let x = signal_x.values();
        let y = signal_y.values();

        let coherence = self.compute_coherence(&x, &y)?;
        
        let mean_coh = self.mean_coherence(&coherence);
        let (peak_freq, peak_coh) = self.peak_coherence(&coherence);
        
        let nyquist = self.config.sampling_freq / 2.0;
        let freq_resolution = self.config.sampling_freq / self.config.segment_length as f64;

        // Compute band coherences
        let low_band = self.band_coherence(&coherence, 0.0, nyquist * 0.3);
        let mid_band = self.band_coherence(&coherence, nyquist * 0.3, nyquist * 0.6);
        let high_band = self.band_coherence(&coherence, nyquist * 0.6, nyquist);

        let interpretation = format!(
            "Coherence analysis completed. Mean coherence: {:.3}. \
             Peak coherence: {:.3} at {:.2} Hz. \
             Frequency resolution: {:.3} Hz.",
            mean_coh, peak_coh, peak_freq, freq_resolution
        );

        Ok(AnalysisResult::new(self.name())
            .with_metric("mean_coherence", mean_coh)
            .with_metric("peak_coherence", peak_coh)
            .with_metric("peak_coherence_frequency_hz", peak_freq)
            .with_metric("low_band_coherence", low_band)
            .with_metric("mid_band_coherence", mid_band)
            .with_metric("high_band_coherence", high_band)
            .with_metric("frequency_resolution_hz", freq_resolution)
            .with_metric("nyquist_frequency_hz", nyquist)
            .with_interpretation(interpretation)
            .with_metadata("segment_length", format!("{}", self.config.segment_length))
            .with_metadata("overlap", format!("{}", self.config.overlap))
            .with_metadata("window", format!("{:?}", self.config.window)))
    }
}

impl Default for CoherenceAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl StochasticAnalyzer for CoherenceAnalyzer {
    fn name(&self) -> &str {
        "Magnitude-Squared Coherence"
    }

    fn analyze(&self, _data: &TimeSeries) -> Result<AnalysisResult, StochasticError> {
        // For single signal, cannot compute coherence
        Err(StochasticError::validation(
            "Coherence requires two signals. Use analyze_two_signals() instead."
        ))
    }

    fn required_sample_size(&self) -> usize {
        self.config.segment_length
    }

    fn validate(&self, data: &TimeSeries) -> Result<bool, StochasticError> {
        if data.len() < self.required_sample_size() {
            return Err(StochasticError::insufficient_data(
                self.required_sample_size(),
                data.len(),
            ));
        }

        if self.config.sampling_freq <= 0.0 {
            return Err(StochasticError::invalid_param(
                "sampling_freq",
                self.config.sampling_freq.to_string(),
                "> 0"
            ));
        }

        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_perfect_coherence() {
        // Two identical signals should have coherence ≈ 1
        let fs = 100.0;
        let n = 1024;
        let freq = 10.0;

        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * freq * i as f64 / fs).sin())
            .collect();

        let data_x = TimeSeries::from_values(signal.clone());
        let data_y = TimeSeries::from_values(signal);

        let analyzer = CoherenceAnalyzer::new()
            .with_sampling_freq(fs)
            .with_segments(256, 128);

        let result = analyzer.analyze_two_signals(&data_x, &data_y).unwrap();
        let mean_coh = result.metrics.get("mean_coherence").unwrap();

        // Should be close to 1.0
        assert!(*mean_coh > 0.95);
    }

    #[test]
    fn test_zero_coherence_independent() {
        // Two independent noise signals should have low coherence
        let fs = 100.0;
        let n = 2048;

        let signal_x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.123).sin()).collect();
        let signal_y: Vec<f64> = (0..n).map(|i| (i as f64 * 0.456).cos()).collect();

        let data_x = TimeSeries::from_values(signal_x);
        let data_y = TimeSeries::from_values(signal_y);

        let analyzer = CoherenceAnalyzer::new()
            .with_sampling_freq(fs)
            .with_segments(512, 256);

        let result = analyzer.analyze_two_signals(&data_x, &data_y).unwrap();
        let mean_coh = result.metrics.get("mean_coherence").unwrap();

        // Should be low (close to 0)
        assert!(*mean_coh < 0.3);
    }

    #[test]
    fn test_coherence_at_specific_frequency() {
        let fs = 100.0;
        let n = 2048;
        let common_freq = 15.0;

        // Both signals have component at 15 Hz, plus unique components
        let signal_x: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / fs;
                (2.0 * PI * common_freq * t).sin() + 0.5 * (2.0 * PI * 5.0 * t).sin()
            })
            .collect();

        let signal_y: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / fs;
                (2.0 * PI * common_freq * t).sin() + 0.5 * (2.0 * PI * 25.0 * t).sin()
            })
            .collect();

        let data_x = TimeSeries::from_values(signal_x);
        let data_y = TimeSeries::from_values(signal_y);

        let analyzer = CoherenceAnalyzer::new()
            .with_sampling_freq(fs)
            .with_segments(512, 256);

        let result = analyzer.analyze_two_signals(&data_x, &data_y).unwrap();
        let peak_freq = result.metrics.get("peak_coherence_frequency_hz").unwrap();
        let peak_coh = result.metrics.get("peak_coherence").unwrap();

        // Peak coherence should be near 15 Hz
        assert_relative_eq!(*peak_freq, common_freq, epsilon = 2.0);
        // Peak coherence should be high
        assert!(*peak_coh > 0.7);
    }

    #[test]
    fn test_coherence_bounds() {
        let fs = 100.0;
        let n = 1024;

        let signal_x: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
        let signal_y: Vec<f64> = (0..n).map(|i| (i as f64 * 1.1).cos()).collect();

        let data_x = TimeSeries::from_values(signal_x);
        let data_y = TimeSeries::from_values(signal_y);

        let analyzer = CoherenceAnalyzer::new()
            .with_sampling_freq(fs)
            .with_segments(256, 128);

        let coherence = analyzer.compute_coherence(&data_x.values(), &data_y.values()).unwrap();

        // All coherence values should be in [0, 1]
        for &c in &coherence {
            assert!(c >= 0.0 && c <= 1.0);
        }
    }

    #[test]
    fn test_phase_shifted_signals() {
        // Phase shift doesn't affect magnitude-squared coherence
        let fs = 100.0;
        let n = 1024;
        let freq = 10.0;
        let phase_shift = PI / 4.0;

        let signal_x: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * freq * i as f64 / fs).sin())
            .collect();

        let signal_y: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * freq * i as f64 / fs + phase_shift).sin())
            .collect();

        let data_x = TimeSeries::from_values(signal_x);
        let data_y = TimeSeries::from_values(signal_y);

        let analyzer = CoherenceAnalyzer::new()
            .with_sampling_freq(fs)
            .with_segments(256, 128);

        let result = analyzer.analyze_two_signals(&data_x, &data_y).unwrap();
        let mean_coh = result.metrics.get("mean_coherence").unwrap();

        // Should still have high coherence despite phase shift
        assert!(*mean_coh > 0.9);
    }

    #[test]
    fn test_different_length_signals() {
        let data_x = TimeSeries::from_values(vec![1.0; 100]);
        let data_y = TimeSeries::from_values(vec![1.0; 200]);

        let analyzer = CoherenceAnalyzer::new();

        let result = analyzer.analyze_two_signals(&data_x, &data_y);
        assert!(result.is_err());
    }

    #[test]
    fn test_band_coherence() {
        let fs = 100.0;
        let n = 2048;

        // Strong coherence at low frequencies, weak at high
        let low_freq = 5.0;
        let high_freq = 30.0;

        let signal_x: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / fs;
                (2.0 * PI * low_freq * t).sin() + 0.5 * (2.0 * PI * high_freq * t).sin()
            })
            .collect();

        let signal_y: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / fs;
                (2.0 * PI * low_freq * t).sin() + (t * 234.567).sin()
            })
            .collect();

        let data_x = TimeSeries::from_values(signal_x);
        let data_y = TimeSeries::from_values(signal_y);

        let analyzer = CoherenceAnalyzer::new()
            .with_sampling_freq(fs)
            .with_segments(512, 256);

        let result = analyzer.analyze_two_signals(&data_x, &data_y).unwrap();
        
        let low_band = result.metrics.get("low_band_coherence").unwrap();
        let high_band = result.metrics.get("high_band_coherence").unwrap();

        // Low band should have higher coherence
        assert!(*low_band > *high_band);
    }

    #[test]
    fn test_frequency_bins() {
        let fs = 100.0;
        let segment_length = 200;

        let config = CoherenceConfig {
            sampling_freq: fs,
            segment_length,
            ..Default::default()
        };

        let analyzer = CoherenceAnalyzer::with_config(config);
        let freqs = analyzer.frequency_bins();

        assert_eq!(freqs.len(), segment_length);
        
        // First frequency should be 0 (DC)
        assert_relative_eq!(freqs[0], 0.0, epsilon = 1e-10);
        
        // Frequency resolution
        let df = fs / segment_length as f64;
        assert_relative_eq!(freqs[1], df, epsilon = 1e-10);
    }

    #[test]
    fn test_window_functions() {
        let signal = vec![1.0; 64];
        
        for window in &[WindowType::Hann, WindowType::Hamming, 
                        WindowType::Blackman, WindowType::Rectangular] {
            let mut config = CoherenceConfig::default();
            config.window = *window;
            let analyzer = CoherenceAnalyzer::with_config(config);
            
            let windowed = analyzer.apply_window(&signal);
            assert_eq!(windowed.len(), signal.len());
        }
    }

    #[test]
    fn test_insufficient_data() {
        let data_x = TimeSeries::from_values(vec![1.0; 50]);
        let data_y = TimeSeries::from_values(vec![1.0; 50]);

        let analyzer = CoherenceAnalyzer::new()
            .with_segments(256, 128);

        let result = analyzer.analyze_two_signals(&data_x, &data_y);
        assert!(result.is_err());
    }

    #[test]
    fn test_overlapping_segments() {
        let fs = 100.0;
        let n = 2000;

        let signal_x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
        let signal_y: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();

        let data_x = TimeSeries::from_values(signal_x);
        let data_y = TimeSeries::from_values(signal_y);

        // Test with different overlaps
        for overlap in &[0, 100, 200] {
            let analyzer = CoherenceAnalyzer::new()
                .with_sampling_freq(fs)
                .with_segments(400, *overlap);

            let result = analyzer.analyze_two_signals(&data_x, &data_y);
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_scaled_signal() {
        // Scaling one signal shouldn't affect coherence
        let fs = 100.0;
        let n = 1024;
        let freq = 10.0;

        let signal_x: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * freq * i as f64 / fs).sin())
            .collect();

        let signal_y: Vec<f64> = signal_x.iter().map(|&x| x * 5.0).collect();

        let data_x = TimeSeries::from_values(signal_x);
        let data_y = TimeSeries::from_values(signal_y);

        let analyzer = CoherenceAnalyzer::new()
            .with_sampling_freq(fs)
            .with_segments(256, 128);

        let result = analyzer.analyze_two_signals(&data_x, &data_y).unwrap();
        let mean_coh = result.metrics.get("mean_coherence").unwrap();

        // Should still have high coherence
        assert!(*mean_coh > 0.95);
    }
}
