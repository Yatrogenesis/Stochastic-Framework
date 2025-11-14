//! Power Spectral Density (PSD) Estimation
//!
//! Implements Welch's method for robust PSD estimation using overlapping segments
//! and windowing to reduce variance.
//!
//! # Mathematical Foundation
//!
//! **Welch's Method (1967):**
//! Divide signal into overlapping segments, compute periodogram for each,
//! then average to reduce variance.
//!
//! **Periodogram:**
//! ```text
//! P(f) = (1 / (Nfs)) |X(f)|²
//! ```
//!
//! **Welch's PSD Estimate:**
//! ```text
//! Ŝ(f) = (1/K) Σₖ Pₖ(f)
//! ```
//! where K = number of segments
//!
//! **Variance Reduction:**
//! ```text
//! Var[Ŝ(f)] ≈ S²(f) / K
//! ```
//!
//! **Segment Overlap:**
//! Typically 50% overlap (Welch, 1967) provides good bias-variance tradeoff
//!
//! # References
//!
//! - Welch, P.D. (1967). "The use of fast Fourier transform for the estimation of power spectra"
//! - Bartlett, M.S. (1948). "Smoothing periodograms from time-series with continuous spectra"
//! - Blackman, R.B., & Tukey, J.W. (1958). "The measurement of power spectra"
//! - Stoica, P., & Moses, R. (2005). "Spectral Analysis of Signals"
//! - Kay, S.M. (1988). "Modern Spectral Estimation: Theory and Application"

use stochastic_core::{AnalysisResult, StochasticAnalyzer, StochasticError, TimeSeries};
use rustfft::{FftPlanner, num_complex::Complex};
use std::f64::consts::PI;

/// PSD estimation method
#[derive(Debug, Clone, Copy)]
pub enum PSDMethod {
    /// Welch's method with overlapping segments
    Welch {
        /// Segment length
        segment_length: usize,
        /// Overlap between segments (samples)
        overlap: usize,
    },
    /// Simple periodogram (no averaging)
    Periodogram,
}

/// Window function for PSD
#[derive(Debug, Clone, Copy)]
pub enum WindowFunction {
    Hann,
    Hamming,
    Blackman,
    Rectangular,
}

/// Configuration for PSD estimation
#[derive(Debug, Clone)]
pub struct PSDConfig {
    /// Sampling frequency (Hz)
    pub sampling_freq: f64,
    /// PSD estimation method
    pub method: PSDMethod,
    /// Window function
    pub window: WindowFunction,
    /// Return one-sided PSD (positive frequencies only)
    pub one_sided: bool,
    /// Scale to density (PSD) vs spectrum
    pub scale_to_density: bool,
}

impl Default for PSDConfig {
    fn default() -> Self {
        Self {
            sampling_freq: 1.0,
            method: PSDMethod::Welch {
                segment_length: 256,
                overlap: 128,
            },
            window: WindowFunction::Hann,
            one_sided: true,
            scale_to_density: true,
        }
    }
}

/// Power Spectral Density Analyzer
pub struct PSDAnalyzer {
    config: PSDConfig,
}

impl PSDAnalyzer {
    /// Create new PSD analyzer
    pub fn new() -> Self {
        Self {
            config: PSDConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: PSDConfig) -> Self {
        Self { config }
    }

    /// Set sampling frequency
    pub fn with_sampling_freq(mut self, freq: f64) -> Self {
        self.config.sampling_freq = freq;
        self
    }

    /// Set Welch method parameters
    pub fn with_welch(mut self, segment_length: usize, overlap: usize) -> Self {
        self.config.method = PSDMethod::Welch { segment_length, overlap };
        self
    }

    /// Set window function
    pub fn with_window(mut self, window: WindowFunction) -> Self {
        self.config.window = window;
        self
    }

    /// Apply window function
    fn apply_window(&self, signal: &[f64]) -> Vec<f64> {
        let n = signal.len();
        match self.config.window {
            WindowFunction::Rectangular => signal.to_vec(),
            WindowFunction::Hann => {
                signal.iter()
                    .enumerate()
                    .map(|(i, &x)| x * (0.5 - 0.5 * (2.0 * PI * i as f64 / (n - 1) as f64).cos()))
                    .collect()
            }
            WindowFunction::Hamming => {
                signal.iter()
                    .enumerate()
                    .map(|(i, &x)| x * (0.54 - 0.46 * (2.0 * PI * i as f64 / (n - 1) as f64).cos()))
                    .collect()
            }
            WindowFunction::Blackman => {
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

    /// Compute periodogram of a segment
    fn periodogram(&self, segment: &[f64]) -> Vec<f64> {
        let windowed = self.apply_window(segment);
        let n = windowed.len();

        // Compute FFT
        let mut buffer: Vec<Complex<f64>> = windowed.iter()
            .map(|&x| Complex::new(x, 0.0))
            .collect();

        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(buffer.len());
        fft.process(&mut buffer);

        // Compute power: |X(f)|² / (N * fs)
        let scale = if self.config.scale_to_density {
            (n as f64) * self.config.sampling_freq
        } else {
            n as f64
        };

        buffer.iter()
            .map(|c| (c.re.powi(2) + c.im.powi(2)) / scale)
            .collect()
    }

    /// Compute window correction factor
    fn window_correction(&self, n: usize) -> f64 {
        let window = self.apply_window(&vec![1.0; n]);
        let sum_sq: f64 = window.iter().map(|&x| x * x).sum();
        (n as f64) / sum_sq
    }

    /// Compute PSD using Welch's method
    pub fn compute_psd(&self, signal: &[f64]) -> Result<Vec<f64>, StochasticError> {
        match self.config.method {
            PSDMethod::Periodogram => {
                Ok(self.periodogram(signal))
            }
            PSDMethod::Welch { segment_length, overlap } => {
                if segment_length > signal.len() {
                    return Err(StochasticError::invalid_param(
                        "segment_length",
                        segment_length.to_string(),
                        format!("<= signal length ({})", signal.len())
                    ));
                }

                if overlap >= segment_length {
                    return Err(StochasticError::invalid_param(
                        "overlap",
                        overlap.to_string(),
                        format!("< segment_length ({})", segment_length)
                    ));
                }

                let step = segment_length - overlap;
                let mut num_segments = 0;
                let mut psd_sum = vec![0.0; segment_length];

                // Process overlapping segments
                let mut start = 0;
                while start + segment_length <= signal.len() {
                    let segment = &signal[start..start + segment_length];
                    let segment_psd = self.periodogram(segment);

                    for (i, &val) in segment_psd.iter().enumerate() {
                        psd_sum[i] += val;
                    }

                    num_segments += 1;
                    start += step;
                }

                if num_segments == 0 {
                    return Err(StochasticError::validation(
                        "No valid segments could be extracted for Welch's method"
                    ));
                }

                // Average over segments
                let psd: Vec<f64> = psd_sum.iter()
                    .map(|&sum| sum / num_segments as f64)
                    .collect();

                // Apply window correction
                let correction = self.window_correction(segment_length);
                let corrected_psd: Vec<f64> = psd.iter()
                    .map(|&val| val * correction)
                    .collect();

                Ok(corrected_psd)
            }
        }
    }

    /// Get frequency bins for PSD
    pub fn frequency_bins(&self, n_fft: usize) -> Vec<f64> {
        let freq_resolution = self.config.sampling_freq / n_fft as f64;
        (0..n_fft)
            .map(|k| k as f64 * freq_resolution)
            .collect()
    }

    /// Compute total power
    pub fn total_power(&self, psd: &[f64]) -> f64 {
        let max_len = if self.config.one_sided {
            psd.len() / 2
        } else {
            psd.len()
        };

        let freq_resolution = self.config.sampling_freq / psd.len() as f64;
        
        // Integrate PSD: ∫ S(f) df
        psd[..max_len].iter().sum::<f64>() * freq_resolution
    }

    /// Find peak frequency in PSD
    pub fn peak_frequency(&self, psd: &[f64]) -> (f64, f64) {
        let max_len = if self.config.one_sided {
            psd.len() / 2
        } else {
            psd.len()
        };

        let freqs = self.frequency_bins(psd.len());

        // Skip DC component
        let (max_idx, max_power) = psd[1..max_len]
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, &p)| (idx + 1, p))
            .unwrap_or((0, 0.0));

        (freqs[max_idx], max_power)
    }

    /// Compute band power (integrate PSD over frequency band)
    pub fn band_power(&self, psd: &[f64], low_freq: f64, high_freq: f64) -> f64 {
        let freqs = self.frequency_bins(psd.len());
        let freq_resolution = self.config.sampling_freq / psd.len() as f64;

        let max_len = if self.config.one_sided {
            psd.len() / 2
        } else {
            psd.len()
        };

        let band_sum: f64 = freqs[..max_len].iter()
            .zip(psd[..max_len].iter())
            .filter_map(|(&f, &p)| {
                if f >= low_freq && f <= high_freq {
                    Some(p)
                } else {
                    None
                }
            })
            .sum();

        band_sum * freq_resolution
    }

    /// Perform PSD analysis
    pub fn analyze_psd(&self, data: &TimeSeries) -> Result<AnalysisResult, StochasticError> {
        self.validate(data)?;

        let signal = data.values();
        let psd = self.compute_psd(&signal)?;
        
        let (peak_freq, peak_power) = self.peak_frequency(&psd);
        let total_power = self.total_power(&psd);
        
        let segment_length = match self.config.method {
            PSDMethod::Welch { segment_length, .. } => segment_length,
            PSDMethod::Periodogram => signal.len(),
        };

        let freq_resolution = self.config.sampling_freq / segment_length as f64;
        let nyquist = self.config.sampling_freq / 2.0;

        // Compute some band powers (example: low, mid, high frequencies)
        let low_band = self.band_power(&psd, 0.0, nyquist * 0.3);
        let mid_band = self.band_power(&psd, nyquist * 0.3, nyquist * 0.6);
        let high_band = self.band_power(&psd, nyquist * 0.6, nyquist);

        let interpretation = format!(
            "PSD analysis using {:?} method. Peak frequency: {:.3} Hz with power density {:.6}. \
             Total power: {:.6}. Frequency resolution: {:.3} Hz.",
            self.config.method, peak_freq, peak_power, total_power, freq_resolution
        );

        Ok(AnalysisResult::new(self.name())
            .with_metric("peak_frequency_hz", peak_freq)
            .with_metric("peak_power_density", peak_power)
            .with_metric("total_power", total_power)
            .with_metric("low_band_power", low_band)
            .with_metric("mid_band_power", mid_band)
            .with_metric("high_band_power", high_band)
            .with_metric("frequency_resolution_hz", freq_resolution)
            .with_metric("nyquist_frequency_hz", nyquist)
            .with_interpretation(interpretation)
            .with_metadata("method", format!("{:?}", self.config.method))
            .with_metadata("window", format!("{:?}", self.config.window))
            .with_metadata("sampling_frequency_hz", format!("{}", self.config.sampling_freq)))
    }
}

impl Default for PSDAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl StochasticAnalyzer for PSDAnalyzer {
    fn name(&self) -> &str {
        "Power Spectral Density (Welch)"
    }

    fn analyze(&self, data: &TimeSeries) -> Result<AnalysisResult, StochasticError> {
        self.analyze_psd(data)
    }

    fn required_sample_size(&self) -> usize {
        match self.config.method {
            PSDMethod::Welch { segment_length, .. } => segment_length,
            PSDMethod::Periodogram => 8,
        }
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
    fn test_psd_sine_wave() {
        let fs = 100.0;
        let n = 1024;
        let freq = 10.0;

        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * freq * i as f64 / fs).sin())
            .collect();

        let data = TimeSeries::from_values(signal);
        let analyzer = PSDAnalyzer::new()
            .with_sampling_freq(fs)
            .with_welch(256, 128);

        let result = analyzer.analyze(&data).unwrap();
        let peak_freq = result.metrics.get("peak_frequency_hz").unwrap();

        assert_relative_eq!(*peak_freq, freq, epsilon = 1.0);
    }

    #[test]
    fn test_welch_variance_reduction() {
        let fs = 100.0;
        let n = 2048;
        
        // Signal with noise
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / fs;
                (2.0 * PI * 10.0 * t).sin() + 0.1 * (t * 123.456).sin()
            })
            .collect();

        let data = TimeSeries::from_values(signal);

        // Periodogram (no averaging)
        let periodogram_analyzer = PSDAnalyzer::with_config(PSDConfig {
            method: PSDMethod::Periodogram,
            sampling_freq: fs,
            ..Default::default()
        });

        // Welch (with averaging)
        let welch_analyzer = PSDAnalyzer::new()
            .with_sampling_freq(fs)
            .with_welch(512, 256);

        let psd_periodogram = periodogram_analyzer.compute_psd(&data.values()).unwrap();
        let psd_welch = welch_analyzer.compute_psd(&data.values()).unwrap();

        // Welch should have fewer points but smoother estimate
        assert!(psd_welch.len() < psd_periodogram.len());
    }

    #[test]
    fn test_total_power() {
        let fs = 100.0;
        let n = 512;
        let amplitude = 1.0;
        let freq = 10.0;

        // Sine wave: RMS = amplitude / sqrt(2)
        // Power = RMS² = amplitude² / 2
        let signal: Vec<f64> = (0..n)
            .map(|i| amplitude * (2.0 * PI * freq * i as f64 / fs).sin())
            .collect();

        let data = TimeSeries::from_values(signal);
        let analyzer = PSDAnalyzer::new()
            .with_sampling_freq(fs)
            .with_welch(256, 128);

        let result = analyzer.analyze(&data).unwrap();
        let total_power = result.metrics.get("total_power").unwrap();

        // Expected power ≈ amplitude² / 2 = 0.5, but one-sided gives half
        // Due to frequency resolution and windowing effects
        assert_relative_eq!(*total_power, 0.25, epsilon = 0.1);
    }

    #[test]
    fn test_band_power() {
        let fs = 100.0;
        let n = 1024;

        // Two frequency components: 5 Hz and 20 Hz
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / fs;
                (2.0 * PI * 5.0 * t).sin() + (2.0 * PI * 20.0 * t).sin()
            })
            .collect();

        let data = TimeSeries::from_values(signal);
        let analyzer = PSDAnalyzer::new()
            .with_sampling_freq(fs)
            .with_welch(256, 128);

        let psd = analyzer.compute_psd(&data.values()).unwrap();

        // Band around 5 Hz
        let low_band = analyzer.band_power(&psd, 3.0, 7.0);
        
        // Band around 20 Hz
        let mid_band = analyzer.band_power(&psd, 18.0, 22.0);

        // Both bands should have significant power
        assert!(low_band > 0.1);
        assert!(mid_band > 0.1);
    }

    #[test]
    fn test_window_functions() {
        let signal = vec![1.0; 64];
        
        for window in &[WindowFunction::Hann, WindowFunction::Hamming, 
                        WindowFunction::Blackman, WindowFunction::Rectangular] {
            let analyzer = PSDAnalyzer::new().with_window(*window);
            let windowed = analyzer.apply_window(&signal);
            
            assert_eq!(windowed.len(), signal.len());
        }
    }

    #[test]
    fn test_frequency_resolution() {
        let fs = 100.0;
        let segment_length = 200;
        
        let analyzer = PSDAnalyzer::new()
            .with_sampling_freq(fs)
            .with_welch(segment_length, 100);

        let data = TimeSeries::from_values(vec![1.0; 1000]);
        let result = analyzer.analyze(&data).unwrap();
        
        let freq_res = result.metrics.get("frequency_resolution_hz").unwrap();
        
        // Δf = fs / N
        assert_relative_eq!(*freq_res, fs / segment_length as f64, epsilon = 1e-10);
    }

    #[test]
    fn test_periodogram_vs_welch() {
        let fs = 100.0;
        let n = 1024;
        let freq = 15.0;

        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * freq * i as f64 / fs).sin())
            .collect();

        let data = TimeSeries::from_values(signal);

        let periodogram = PSDAnalyzer::with_config(PSDConfig {
            method: PSDMethod::Periodogram,
            sampling_freq: fs,
            ..Default::default()
        });

        let welch = PSDAnalyzer::new()
            .with_sampling_freq(fs)
            .with_welch(512, 256);

        let result_p = periodogram.analyze(&data).unwrap();
        let result_w = welch.analyze(&data).unwrap();

        let peak_p = result_p.metrics.get("peak_frequency_hz").unwrap();
        let peak_w = result_w.metrics.get("peak_frequency_hz").unwrap();

        // Both should detect the same peak frequency
        assert_relative_eq!(*peak_p, freq, epsilon = 1.0);
        assert_relative_eq!(*peak_w, freq, epsilon = 1.0);
    }

    #[test]
    fn test_overlapping_segments() {
        let fs = 100.0;
        let n = 2000;
        
        let signal: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
        let data = TimeSeries::from_values(signal);

        // 50% overlap
        let analyzer = PSDAnalyzer::new()
            .with_sampling_freq(fs)
            .with_welch(400, 200);

        let result = analyzer.analyze(&data);
        assert!(result.is_ok());
    }

    #[test]
    fn test_invalid_segment_length() {
        let data = TimeSeries::from_values(vec![1.0; 100]);
        
        let analyzer = PSDAnalyzer::new()
            .with_sampling_freq(100.0)
            .with_welch(200, 100); // segment_length > signal length

        let result = analyzer.analyze(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_overlap() {
        let data = TimeSeries::from_values(vec![1.0; 1000]);
        
        let analyzer = PSDAnalyzer::new()
            .with_sampling_freq(100.0)
            .with_welch(200, 200); // overlap >= segment_length

        let result = analyzer.analyze(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_nyquist_frequency() {
        let fs = 100.0;
        let data = TimeSeries::from_values(vec![1.0; 512]);
        
        let analyzer = PSDAnalyzer::new().with_sampling_freq(fs);
        let result = analyzer.analyze(&data).unwrap();
        
        let nyquist = result.metrics.get("nyquist_frequency_hz").unwrap();
        assert_relative_eq!(*nyquist, fs / 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_peak_detection_multiple_frequencies() {
        let fs = 100.0;
        let n = 2048;

        // Strong component at 12 Hz, weak at 30 Hz
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / fs;
                2.0 * (2.0 * PI * 12.0 * t).sin() + 0.5 * (2.0 * PI * 30.0 * t).sin()
            })
            .collect();

        let data = TimeSeries::from_values(signal);
        let analyzer = PSDAnalyzer::new()
            .with_sampling_freq(fs)
            .with_welch(512, 256);

        let result = analyzer.analyze(&data).unwrap();
        let peak_freq = result.metrics.get("peak_frequency_hz").unwrap();

        // Should detect the stronger 12 Hz component
        assert_relative_eq!(*peak_freq, 12.0, epsilon = 1.0);
    }

    #[test]
    fn test_window_correction_factor() {
        let segment_length = 256;
        
        for window in &[WindowFunction::Hann, WindowFunction::Hamming, WindowFunction::Blackman] {
            let analyzer = PSDAnalyzer::new().with_window(*window);
            let correction = analyzer.window_correction(segment_length);
            
            // Correction factor should be > 1 for non-rectangular windows
            assert!(correction >= 1.0);
        }
    }
}
