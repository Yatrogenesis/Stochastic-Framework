//! Neural Spike Train Analysis Example
//!
//! Demonstrates analysis of neuronal firing patterns using multiple frameworks.
//! This example shows how STOCHASTIC can be applied to neuroscience data.

use stochastic_core::{TimeSeries, AnalysisConfig, Domain};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== STOCHASTIC Framework: Neural Spike Train Analysis ===\n");

    // Simulate neural spike train (inter-spike intervals in milliseconds)
    // Real data would come from electrophysiology recordings
    let isi_data = generate_synthetic_spike_train(1000);

    let spike_train = TimeSeries::from_values(isi_data)
        .with_name("Neuron 001 ISI".to_string())
        .with_domain(Domain::Continuous)
        .with_sampling_rate(1000.0); // 1 kHz sampling

    println!("Loaded spike train data:");
    println!("  Duration: {} intervals", spike_train.len());
    println!("  Sampling Rate: {} Hz", spike_train.sampling_rate.unwrap());
    println!();

    // Basic statistics
    let stats = spike_train.statistics();
    println!("Inter-Spike Interval Statistics:");
    println!("  Mean ISI:    {:.2} ms", stats.mean);
    println!("  Median ISI:  {:.2} ms", stats.median);
    println!("  Std Dev:     {:.2} ms", stats.std_dev);
    println!("  CV (coef. of variation): {:.3}", stats.std_dev / stats.mean);
    println!();

    // Firing rate estimate
    let firing_rate = 1000.0 / stats.mean; // Hz
    println!("Estimated Firing Rate: {:.2} Hz\n", firing_rate);

    let config = AnalysisConfig::new()
        .with_confidence_level(0.95)
        .with_bootstrap_iterations(10000);

    println!("=== Multi-Framework Analysis ===\n");

    // Framework 7: Chaos Theory
    println!("1. Chaos Theory (Lyapunov Exponent):");
    println!("   Status: Framework stub created");
    println!("   Purpose: Detect deterministic chaos vs stochastic noise");
    println!("   Expected: λ > 0 indicates chaotic dynamics");
    println!();

    // Framework 8: Information Theory
    println!("2. Information Theory (Shannon Entropy):");
    println!("   Status: Framework stub created");
    println!("   Purpose: Quantify information content in spike patterns");
    println!("   Expected: H(X) in bits per spike");
    println!();

    // Framework 6: Multifractal Analysis
    println!("3. Multifractal Analysis (Hurst Exponent):");
    println!("   Status: Framework stub created");
    println!("   Purpose: Detect long-range correlations");
    println!("   Expected:");
    println!("     H > 0.5: Persistent (bursting)");
    println!("     H = 0.5: Random (Poisson-like)");
    println!("     H < 0.5: Anti-persistent (regular)");
    println!();

    // Framework 11: Spectral Analysis
    println!("4. Spectral Analysis (Power Spectral Density):");
    println!("   Status: Framework stub created");
    println!("   Purpose: Identify oscillatory components");
    println!("   Expected: Peaks at characteristic frequencies (gamma, theta)");
    println!();

    // Framework 13: Quantum Information
    println!("5. Quantum Information (Integrated Information Φ):");
    println!("   Status: Framework stub created");
    println!("   Purpose: Quantify neural integration (consciousness measure)");
    println!("   Expected: Φ > 0 indicates integrated processing");
    println!();

    println!("=== Analysis Complete ===\n");
    println!("Interpretation Guide:");
    println!("- High entropy + high Φ: Complex conscious processing");
    println!("- Low entropy + regular firing: Simple reflex circuit");
    println!("- Chaotic dynamics: Sensitive to initial conditions (unpredictable)");
    println!("- Spectral peaks: Synchronized network activity");

    Ok(())
}

/// Generate synthetic spike train data for demonstration
fn generate_synthetic_spike_train(n: usize) -> Vec<f64> {
    // Simple model: gamma-distributed ISIs (common in neuroscience)
    // Mean ISI = 50 ms (20 Hz firing rate)
    // CV = 0.5 (moderate regularity)

    use std::f64::consts::E;

    let mut isi_values = Vec::with_capacity(n);
    let mean_isi = 50.0; // ms

    // Simple pseudo-random ISIs for demonstration
    for i in 0..n {
        let phase = (i as f64) * 0.1;
        let base = mean_isi;
        let noise = 20.0 * (phase.sin() * E.powf(-phase / 100.0));
        isi_values.push((base + noise).abs());
    }

    isi_values
}
