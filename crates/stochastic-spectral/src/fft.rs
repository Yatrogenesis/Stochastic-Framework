//! Fast Fourier Transform (FFT) Analysis
//!
//! Implements FFT-based spectral analysis for time series data using the Cooley-Tukey
//! algorithm via the RustFFT library.
//!
//! # Mathematical Foundation
//!
//! **Discrete Fourier Transform (DFT):**
//! ```text
//! X[k] = Σ_{n=0}^{N-1} x[n] · e^(-i2πkn/N)
//! ```
//!
//! **Fast Fourier Transform (Cooley-Tukey, 1965):**
//! Reduces computational complexity from O(N²) to O(N log N) using divide-and-conquer.
//!
//! **Power Spectrum:**
//! ```text
//! P[k] = |X[k]|² / N
//! ```
//!
//! **Frequency Resolution:**
//! ```text
//! Δf = fs / N
//! ```
//! where fs = sampling frequency, N = number of samples
//!
//! **Nyquist Frequency:**
//! ```text
//! fₙ = fs / 2
//! ```
//!
//! # References
//!
//! - Cooley, J.W., & Tukey, J.W. (1965). "An algorithm for the machine calculation of complex Fourier series"
//! - Brigham, E.O. (1988). "The Fast Fourier Transform and Its Applications"
//! - Oppenheim, A.V., & Schafer, R.W. (2009). "Discrete-Time Signal Processing" (3rd ed.)
//! - Lyons, R.G. (2011). "Understanding Digital Signal Processing" (3rd ed.)
//! - Smith, S.W. (2011). "Digital Signal Processing: A Practical Guide for Engineers and Scientists"

use stochastic_core::{AnalysisResult, StochasticAnalyzer, StochasticError, TimeSeries};
use rustfft::{FftPlanner, num_complex::Complex};

use std::f64::consts::PI;

/// Configuration for FFT analysis
#[derive(Debug, Clone)]
pub struct FFTConfig {
    /// Sampling frequency (Hz)
    pub sampling_freq: f64,
    /// Apply windowing function to reduce spectral leakage
    pub window: WindowType,
    /// Zero-padding factor (multiplies signal length)
    pub zero_padding: usize,
    /// Return only positive frequencies (up to Nyquist)
    pub one_sided: bool,
}

/// Window functions for spectral leakage reduction
#[derive(Debug, Clone, Copy)]
pub enum WindowType {
    /// Rectangular window (no windowing)
    Rectangular,
    /// Hann window (von Hann, 1967)
    Hann,
    /// Hamming window (Hamming, 1989)
    Hamming,
    /// Blackman window (Blackman & Tukey, 1958)
    Blackman,
}

impl Default for FFTConfig {
    fn default() -> Self {
        Self {
            sampling_freq: 1.0,
            window: WindowType::Hann,
            zero_padding: 1,
            one_sided: true,
        }
    }
}

/// FFT Analyzer
pub struct FFTAnalyzer {
    config: FFTConfig,
}

impl FFTAnalyzer {
    /// Create new FFT analyzer with default configuration
    pub fn new() -> Self {
        Self {
            config: FFTConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: FFTConfig) -> Self {
        Self { config }
    }

    /// Set sampling frequency (Hz)
    pub fn with_sampling_freq(mut self, freq: f64) -> Self {
        self.config.sampling_freq = freq;
        self
    }

    /// Set window type
    pub fn with_window(mut self, window: WindowType) -> Self {
        self.config.window = window;
        self
    }

    /// Set zero-padding factor
    pub fn with_zero_padding(mut self, factor: usize) -> Self {
        self.config.zero_padding = factor.max(1);
        self
    }

    /// Apply window function to signal
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

    /// Compute FFT of signal
    pub fn compute_fft(&self, signal: &[f64]) -> Result<Vec<Complex<f64>>, StochasticError> {
        if signal.is_empty() {
            return Err(StochasticError::validation("Signal cannot be empty"));
        }

        // Apply window
        let windowed = self.apply_window(signal);

        // Zero-padding
        let padded_len = windowed.len() * self.config.zero_padding;
        let mut buffer: Vec<Complex<f64>> = windowed.iter()
            .map(|&x| Complex::new(x, 0.0))
            .collect();
        buffer.resize(padded_len, Complex::new(0.0, 0.0));

        // Compute FFT
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(buffer.len());
        fft.process(&mut buffer);

        Ok(buffer)
    }

    /// Compute magnitude spectrum
    pub fn magnitude_spectrum(&self, fft_result: &[Complex<f64>]) -> Vec<f64> {
        fft_result.iter()
            .map(|c| (c.re.powi(2) + c.im.powi(2)).sqrt())
            .collect()
    }

    /// Compute power spectrum
    pub fn power_spectrum(&self, fft_result: &[Complex<f64>]) -> Vec<f64> {
        let n = fft_result.len() as f64;
        fft_result.iter()
            .map(|c| (c.re.powi(2) + c.im.powi(2)) / n)
            .collect()
    }

    /// Compute phase spectrum (radians)
    pub fn phase_spectrum(&self, fft_result: &[Complex<f64>]) -> Vec<f64> {
        fft_result.iter()
            .map(|c| c.im.atan2(c.re))
            .collect()
    }

    /// Get frequency bins (Hz)
    pub fn frequency_bins(&self, n_samples: usize) -> Vec<f64> {
        let n_fft = n_samples * self.config.zero_padding;
        let freq_resolution = self.config.sampling_freq / n_fft as f64;
        
        (0..n_fft)
            .map(|k| k as f64 * freq_resolution)
            .collect()
    }

    /// Find dominant frequency (fundamental frequency)
    pub fn dominant_frequency(&self, fft_result: &[Complex<f64>], n_samples: usize) -> (f64, f64) {
        let power = self.power_spectrum(fft_result);
        let freqs = self.frequency_bins(n_samples);

        // Skip DC component (k=0)
        let max_len = if self.config.one_sided {
            fft_result.len() / 2
        } else {
            fft_result.len()
        };

        let (max_idx, max_power) = power[1..max_len]
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, &p)| (idx + 1, p))
            .unwrap_or((0, 0.0));

        (freqs[max_idx], max_power)
    }

    /// Compute spectral centroid (center of mass of spectrum)
    pub fn spectral_centroid(&self, fft_result: &[Complex<f64>], n_samples: usize) -> f64 {
        let power = self.power_spectrum(fft_result);
        let freqs = self.frequency_bins(n_samples);

        let max_len = if self.config.one_sided {
            fft_result.len() / 2
        } else {
            fft_result.len()
        };

        let weighted_sum: f64 = freqs[..max_len].iter()
            .zip(power[..max_len].iter())
            .map(|(&f, &p)| f * p)
            .sum();

        let total_power: f64 = power[..max_len].iter().sum();

        if total_power > 0.0 {
            weighted_sum / total_power
        } else {
            0.0
        }
    }

    /// Compute spectral bandwidth (spread around centroid)
    pub fn spectral_bandwidth(&self, fft_result: &[Complex<f64>], n_samples: usize) -> f64 {
        let power = self.power_spectrum(fft_result);
        let freqs = self.frequency_bins(n_samples);
        let centroid = self.spectral_centroid(fft_result, n_samples);

        let max_len = if self.config.one_sided {
            fft_result.len() / 2
        } else {
            fft_result.len()
        };

        let variance: f64 = freqs[..max_len].iter()
            .zip(power[..max_len].iter())
            .map(|(&f, &p)| (f - centroid).powi(2) * p)
            .sum();

        let total_power: f64 = power[..max_len].iter().sum();

        if total_power > 0.0 {
            (variance / total_power).sqrt()
        } else {
            0.0
        }
    }

    /// Compute total spectral energy
    pub fn spectral_energy(&self, fft_result: &[Complex<f64>]) -> f64 {
        let power = self.power_spectrum(fft_result);
        
        let max_len = if self.config.one_sided {
            fft_result.len() / 2
        } else {
            fft_result.len()
        };

        power[..max_len].iter().sum()
    }

    /// Perform complete FFT analysis
    pub fn analyze_spectrum(&self, data: &TimeSeries) -> Result<AnalysisResult, StochasticError> {
        self.validate(data)?;

        let signal = data.values();
        let n = signal.len();

        // Compute FFT
        let fft_result = self.compute_fft(&signal)?;
        
        // Compute spectral features
        let (dominant_freq, dominant_power) = self.dominant_frequency(&fft_result, n);
        let centroid = self.spectral_centroid(&fft_result, n);
        let bandwidth = self.spectral_bandwidth(&fft_result, n);
        let energy = self.spectral_energy(&fft_result);

        // Nyquist frequency
        let nyquist = self.config.sampling_freq / 2.0;

        // Frequency resolution
        let n_fft = n * self.config.zero_padding;
        let freq_resolution = self.config.sampling_freq / n_fft as f64;

        let interpretation = format!(
            "FFT analysis completed. Dominant frequency: {:.3} Hz with power {:.6}. \
             Spectral centroid: {:.3} Hz, bandwidth: {:.3} Hz. \
             Frequency resolution: {:.3} Hz, Nyquist: {:.3} Hz.",
            dominant_freq, dominant_power, centroid, bandwidth, freq_resolution, nyquist
        );

        Ok(AnalysisResult::new(self.name())
            .with_metric("dominant_frequency_hz", dominant_freq)
            .with_metric("dominant_power", dominant_power)
            .with_metric("spectral_centroid_hz", centroid)
            .with_metric("spectral_bandwidth_hz", bandwidth)
            .with_metric("spectral_energy", energy)
            .with_metric("nyquist_frequency_hz", nyquist)
            .with_metric("frequency_resolution_hz", freq_resolution)
            .with_metric("fft_size", n_fft as f64)
            .with_metric("zero_padding_factor", self.config.zero_padding as f64)
            .with_interpretation(interpretation)
            .with_metadata("window_type", format!("{:?}", self.config.window))
            .with_metadata("sampling_frequency_hz", format!("{}", self.config.sampling_freq)))
    }
}

impl Default for FFTAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl StochasticAnalyzer for FFTAnalyzer {
    fn name(&self) -> &str {
        "Fast Fourier Transform (FFT)"
    }

    fn analyze(&self, data: &TimeSeries) -> Result<AnalysisResult, StochasticError> {
        self.analyze_spectrum(data)
    }

    fn required_sample_size(&self) -> usize {
        8 // Minimum for meaningful FFT
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
    fn test_fft_sine_wave() {
        // Pure sine wave at 5 Hz
        let fs = 100.0; // 100 Hz sampling
        let n = 256;
        let freq = 5.0;
        
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * freq * i as f64 / fs).sin())
            .collect();

        let data = TimeSeries::from_values(signal);
        let analyzer = FFTAnalyzer::new().with_sampling_freq(fs);
        
        let result = analyzer.analyze(&data).unwrap();
        let dominant_freq = result.metrics.get("dominant_frequency_hz").unwrap();

        // Should detect 5 Hz frequency
        assert_relative_eq!(*dominant_freq, freq, epsilon = 0.5);
    }

    #[test]
    fn test_fft_multiple_frequencies() {
        // Signal with 3 Hz and 7 Hz components
        let fs = 100.0;
        let n = 256;
        
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / fs;
                (2.0 * PI * 3.0 * t).sin() + 0.5 * (2.0 * PI * 7.0 * t).sin()
            })
            .collect();

        let data = TimeSeries::from_values(signal);
        let analyzer = FFTAnalyzer::new().with_sampling_freq(fs);
        
        let fft_result = analyzer.compute_fft(&data.values()).unwrap();
        let (dominant_freq, _) = analyzer.dominant_frequency(&fft_result, n);

        // Dominant should be 3 Hz (larger amplitude)
        assert_relative_eq!(dominant_freq, 3.0, epsilon = 0.5);
    }

    #[test]
    fn test_nyquist_frequency() {
        let fs = 100.0;
        let data = TimeSeries::from_values(vec![1.0; 64]);
        let analyzer = FFTAnalyzer::new().with_sampling_freq(fs);
        
        let result = analyzer.analyze(&data).unwrap();
        let nyquist = result.metrics.get("nyquist_frequency_hz").unwrap();

        assert_relative_eq!(*nyquist, fs / 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_window_functions() {
        let signal = vec![1.0, 2.0, 3.0, 2.0, 1.0];
        
        // Test each window type
        for window in &[WindowType::Rectangular, WindowType::Hann, 
                        WindowType::Hamming, WindowType::Blackman] {
            let analyzer = FFTAnalyzer::new().with_window(*window);
            let windowed = analyzer.apply_window(&signal);
            
            // Windowed signal should have same length
            assert_eq!(windowed.len(), signal.len());
            
            // Non-rectangular windows should reduce edge values
            if !matches!(window, WindowType::Rectangular) {
                assert!(windowed[0].abs() <= signal[0].abs());
                assert!(windowed[windowed.len()-1].abs() <= signal[signal.len()-1].abs());
            }
        }
    }

    #[test]
    fn test_zero_padding() {
        let signal = vec![1.0; 32];
        let data = TimeSeries::from_values(signal);
        
        let analyzer = FFTAnalyzer::new()
            .with_zero_padding(4)
            .with_sampling_freq(100.0);
        
        let result = analyzer.analyze(&data).unwrap();
        let fft_size = result.metrics.get("fft_size").unwrap();
        
        // FFT size should be 32 * 4 = 128
        assert_eq!(*fft_size, 128.0);
    }

    #[test]
    fn test_spectral_centroid() {
        // White noise-like signal (broad spectrum)
        let signal: Vec<f64> = (0..128).map(|i| (i as f64 * 0.1).sin()).collect();
        let data = TimeSeries::from_values(signal);
        
        let analyzer = FFTAnalyzer::new().with_sampling_freq(100.0);
        let result = analyzer.analyze(&data).unwrap();
        
        let centroid = result.metrics.get("spectral_centroid_hz").unwrap();
        
        // Centroid should be positive and less than Nyquist
        assert!(*centroid > 0.0 && *centroid < 50.0);
    }

    #[test]
    fn test_spectral_bandwidth() {
        let fs = 100.0;
        let n = 128;
        
        // Narrow-band signal (pure tone)
        let narrow: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 10.0 * i as f64 / fs).sin())
            .collect();
        
        // Broad-band signal (multiple frequencies)
        let broad: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / fs;
                (2.0 * PI * 5.0 * t).sin() 
                + (2.0 * PI * 15.0 * t).sin()
                + (2.0 * PI * 25.0 * t).sin()
            })
            .collect();

        let analyzer = FFTAnalyzer::new().with_sampling_freq(fs);
        
        let narrow_data = TimeSeries::from_values(narrow);
        let broad_data = TimeSeries::from_values(broad);
        
        let narrow_result = analyzer.analyze(&narrow_data).unwrap();
        let broad_result = analyzer.analyze(&broad_data).unwrap();
        
        let narrow_bw = narrow_result.metrics.get("spectral_bandwidth_hz").unwrap();
        let broad_bw = broad_result.metrics.get("spectral_bandwidth_hz").unwrap();
        
        // Broad signal should have larger bandwidth
        assert!(*broad_bw > *narrow_bw);
    }

    #[test]
    fn test_spectral_energy() {
        let fs = 100.0;
        let n = 128;
        
        // Low amplitude signal
        let low: Vec<f64> = (0..n)
            .map(|i| 0.1 * (2.0 * PI * 10.0 * i as f64 / fs).sin())
            .collect();
        
        // High amplitude signal
        let high: Vec<f64> = (0..n)
            .map(|i| 1.0 * (2.0 * PI * 10.0 * i as f64 / fs).sin())
            .collect();

        let analyzer = FFTAnalyzer::new().with_sampling_freq(fs);
        
        let low_data = TimeSeries::from_values(low);
        let high_data = TimeSeries::from_values(high);
        
        let low_result = analyzer.analyze(&low_data).unwrap();
        let high_result = analyzer.analyze(&high_data).unwrap();
        
        let low_energy = low_result.metrics.get("spectral_energy").unwrap();
        let high_energy = high_result.metrics.get("spectral_energy").unwrap();
        
        // High amplitude signal should have more energy
        assert!(*high_energy > *low_energy);
    }

    #[test]
    fn test_frequency_resolution() {
        let fs = 100.0;
        let n = 100;
        
        let data = TimeSeries::from_values(vec![1.0; n]);
        let analyzer = FFTAnalyzer::new().with_sampling_freq(fs);
        
        let result = analyzer.analyze(&data).unwrap();
        let freq_res = result.metrics.get("frequency_resolution_hz").unwrap();
        
        // Frequency resolution should be fs / N
        assert_relative_eq!(*freq_res, fs / n as f64, epsilon = 1e-10);
    }

    #[test]
    fn test_dc_component() {
        // Signal with DC offset
        let signal: Vec<f64> = vec![5.0; 64];
        let data = TimeSeries::from_values(signal);
        
        let analyzer = FFTAnalyzer::new().with_sampling_freq(100.0);
        let fft_result = analyzer.compute_fft(&data.values()).unwrap();
        
        // DC component (k=0) should be dominant for constant signal
        let power = analyzer.power_spectrum(&fft_result);
        assert!(power[0] > power[1]);
    }

    #[test]
    fn test_insufficient_data() {
        let data = TimeSeries::from_values(vec![1.0, 2.0]);
        let analyzer = FFTAnalyzer::new();
        
        let result = analyzer.analyze(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_sampling_frequency() {
        let data = TimeSeries::from_values(vec![1.0; 64]);
        let analyzer = FFTAnalyzer::new().with_sampling_freq(-10.0);
        
        let result = analyzer.analyze(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_phase_spectrum() {
        let fs = 100.0;
        let n = 64;
        let freq = 10.0;
        
        // Sine wave (90° phase shift from cosine)
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * freq * i as f64 / fs).sin())
            .collect();

        let data = TimeSeries::from_values(signal);
        let analyzer = FFTAnalyzer::new().with_sampling_freq(fs);
        
        let fft_result = analyzer.compute_fft(&data.values()).unwrap();
        let phase = analyzer.phase_spectrum(&fft_result);
        
        // Phase spectrum should exist for all frequency bins
        assert_eq!(phase.len(), fft_result.len());
        
        // All phase values should be in [-π, π]
        for &p in &phase {
            assert!(p >= -PI && p <= PI);
        }
    }
}
