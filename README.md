# STOCHASTIC Framework

**Multi-Framework Analysis for Stochastic Process Pattern Detection**

[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0008--6093--8267-green)](https://orcid.org/0009-0008-6093-8267)

## Overview

**STOCHASTIC** (**S**tochastic **T**esting and **O**rder **CH**assis **A**nalysis for **S**ystematic **T**emporal **I**nference and **C**lassification) is a comprehensive Rust framework for detecting emergent order in stochastic processes using 13 independent mathematical methodologies.

### Research Context

This framework was developed as part of consciousness AI research to validate pattern detection methodologies on certified random number generators before applying them to neural dynamics and quantum systems.

**Key Insight**: If a methodology cannot distinguish order from certified randomness, it's suitable for detecting genuine emergent patterns in complex systems.

## Applications

- **Consciousness AI**: Pattern detection in neuroplastic operating systems
- **Quantum Information**: Integrated information (Φ) analysis
- **Neuroscience**: Neural spike train analysis
- **Complex Systems**: Emergent behavior detection
- **Financial Markets**: Time series anomaly detection
- **Climate Science**: Long-range dependency analysis
- **Validation**: RNG certification and testing

## 13 Mathematical Frameworks

### 1. Frequentist Statistics

Classical hypothesis testing for distributional conformity:

- **χ² goodness-of-fit test**: Tests uniformity of discrete distributions
- **Kolmogorov-Smirnov test**: Non-parametric test for continuous distributions
- **Anderson-Darling test**: Enhanced sensitivity to tail deviations

**Applications**: Basic randomness testing, distribution validation

### 2. Bayesian Inference

Probabilistic reasoning with prior knowledge integration:

- **Dirichlet conjugate prior**: Natural prior for categorical distributions
- **MCMC sampling**: Metropolis-Hastings and Gibbs sampling
- **Credible intervals**: Bayesian uncertainty quantification
- **Posterior predictive checks**: Model validation

**Applications**: Parameter estimation under uncertainty, adaptive models

### 3. Ergodic Theory

Analysis of long-term statistical behavior:

- **Birkhoff-Khinchin theorem**: Time average ≈ ensemble average
- **Temporal vs ensemble averaging**: Ergodicity testing
- **Mixing properties**: Weak, strong, and exponential mixing
- **Poincaré recurrence**: Return time statistics

**Applications**: Stationarity testing, long-range correlations

### 4. Range Equilibrium Analysis

Statistical balance across data partitions:

- **Distribution balance**: χ² test across temporal/spatial partitions
- **Parity analysis**: Even/odd ratio testing
- **Range-specific bias**: Local deviation detection
- **Entropy balance**: Information content uniformity

**Applications**: Bias detection, data quality assessment

### 5. Galois Field Theory

Algebraic structure in finite domains:

- **GF(p) finite field operations**: Modular arithmetic over primes
- **Quadratic residues**: Legendre symbols and patterns
- **Multiplicative group structure**: Generator detection
- **Polynomial rings**: Irreducible polynomial analysis

**Applications**: Cryptographic RNG validation, algebraic patterns

### 6. Multifractal Analysis

Scale-invariant structure detection:

- **Detrended Fluctuation Analysis (DFA)**: Long-range correlation quantification
- **Hurst exponent H**: Persistence (H > 0.5) vs anti-persistence (H < 0.5)
- **Wavelet transform analysis**: Multi-resolution decomposition
- **Multifractal spectrum f(α)**: Singularity strength distribution

**Applications**: Financial time series, physiological signals, climate data

### 7. Chaos Theory

Deterministic nonlinear dynamics:

- **Lyapunov exponents**: Sensitivity to initial conditions
- **Takens embedding**: State space reconstruction from time series
- **Strange attractor detection**: Fractal dimension estimation
- **0-1 test for chaos**: Discriminates chaos from noise

**Applications**: Neuronal dynamics, weather prediction, cardiac rhythms

### 8. Information Theory

Quantification of information content and complexity:

- **Shannon entropy**: H(X) = -Σ p(x) log p(x)
- **Mutual information**: I(X;Y) dependency measure
- **Transfer entropy**: Directional information flow
- **Kolmogorov complexity**: Algorithmic information content

**Applications**: Neural coding, communication systems, data compression

### 9. Normalization & Outlier Detection

Data preprocessing and quality control:

- **Z-score normalization**: Standardization to mean=0, std=1
- **Shapiro-Wilk test**: Normality assessment
- **Box-Cox transformation**: Variance stabilization
- **Robust outlier detection**: MAD, IQR methods

**Applications**: Data cleaning, feature engineering, preprocessing

### 10. Topological Data Analysis (TDA)

Shape and connectivity in high-dimensional data:

- **Persistent homology**: Multi-scale topological feature extraction
- **Betti numbers**: Counting connected components, holes, voids
- **Mapper algorithm**: Simplicial complex construction
- **Persistence diagrams**: Birth-death time visualization

**Applications**: Neuroscience (brain networks), materials science, sensor networks

### 11. Spectral Analysis

Frequency-domain characterization:

- **Fast Fourier Transform (FFT)**: Frequency spectrum computation
- **Power Spectral Density (PSD)**: Energy distribution across frequencies
- **Coherence analysis**: Cross-spectral correlation
- **Periodogram smoothing**: Welch's method

**Applications**: Signal processing, EEG/MEG analysis, vibration analysis

### 12. Network Theory

Graph-theoretic structure in time series:

- **Visibility graph construction**: Natural and horizontal visibility algorithms
- **Centrality measures**: Betweenness, closeness, eigenvector centrality
- **Community detection**: Louvain, modularity optimization
- **Network motifs**: Recurring subgraph patterns

**Applications**: Social networks, brain connectivity, epidemic modeling

### 13. Quantum Information Theory

Quantum-inspired metrics for classical systems:

- **Density matrix representation**: ρ = |ψ⟩⟨ψ|
- **Entanglement measures**: Von Neumann entropy, negativity
- **Integrated Information (Φ)**: IIT 3.0 consciousness metric
- **Quantum discord**: Non-classical correlations

**Applications**: Consciousness research, quantum computing, neural integration

## Architecture

```
STOCHASTIC-Framework/
├── crates/
│   ├── stochastic-core/          # Core types, traits, errors
│   ├── stochastic-frequentist/   # Framework 1: χ², KS test
│   ├── stochastic-bayesian/      # Framework 2: Bayesian inference
│   ├── stochastic-ergodic/       # Framework 3: Ergodic theory
│   ├── stochastic-equilibrium/   # Framework 4: Range equilibrium
│   ├── stochastic-galois/        # Framework 5: Galois fields
│   ├── stochastic-fractal/       # Framework 6: Multifractal
│   ├── stochastic-chaos/         # Framework 7: Chaos theory
│   ├── stochastic-information/   # Framework 8: Information theory
│   ├── stochastic-normalization/ # Framework 9: Normalization
│   ├── stochastic-topology/      # Framework 10: TDA
│   ├── stochastic-spectral/      # Framework 11: Spectral
│   ├── stochastic-network/       # Framework 12: Networks
│   ├── stochastic-quantum/       # Framework 13: Quantum info
│   └── stochastic-optimization/  # Multi-objective optimization
├── cli/                          # Command-line interface
├── gui/                          # Desktop GUI (egui)
├── examples/                     # Usage examples
└── docs/                         # Documentation
```

## Installation

### From source

```bash
git clone https://github.com/Yatrogenesis/STOCHASTIC-Framework.git
cd STOCHASTIC-Framework
cargo build --release
```

### As library dependency

```toml
[dependencies]
stochastic-core = "1.0"
stochastic-bayesian = "1.0"
stochastic-spectral = "1.0"
# ... other frameworks as needed
```

## Quick Start

### Basic Analysis Pipeline

```rust
use stochastic_core::{TimeSeries, AnalysisConfig};
use stochastic_bayesian::BayesianAnalyzer;
use stochastic_spectral::SpectralAnalyzer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load time series data
    let data = TimeSeries::from_csv("neural_spike_train.csv")?;

    // Configure analysis
    let config = AnalysisConfig::default()
        .with_confidence_level(0.95)
        .with_bootstrap_iterations(10000);

    // Bayesian analysis
    let bayesian = BayesianAnalyzer::new(config.clone());
    let bayes_results = bayesian.analyze(&data)?;
    println!("Posterior probabilities: {:?}", bayes_results.posteriors);

    // Spectral analysis
    let spectral = SpectralAnalyzer::new(config);
    let spec_results = spectral.analyze(&data)?;
    println!("Dominant frequencies: {:?}", spec_results.peaks);

    Ok(())
}
```

### Multi-Framework Validation

```rust
use stochastic_core::StochasticAnalyzer;

fn validate_randomness(data: &TimeSeries) -> AnalysisReport {
    let analyzers: Vec<Box<dyn StochasticAnalyzer>> = vec![
        Box::new(FrequentistAnalyzer::default()),
        Box::new(BayesianAnalyzer::default()),
        Box::new(ErgodicAnalyzer::default()),
        Box::new(FractalAnalyzer::default()),
        Box::new(InformationAnalyzer::default()),
    ];

    let results: Vec<AnalysisResult> = analyzers
        .iter()
        .map(|analyzer| analyzer.analyze(data))
        .collect::<Result<Vec<_>, _>>()?;

    AnalysisReport::aggregate(results)
}
```

### Command-Line Interface

```bash
# Analyze single dataset with all frameworks
stochastic analyze --input data.csv --all-frameworks --output report.json

# Run specific framework
stochastic analyze --input data.csv --framework bayesian --plot

# Compare multiple datasets
stochastic compare dataset1.csv dataset2.csv --frameworks spectral,chaos

# Generate synthetic test data
stochastic generate --distribution uniform --size 10000 --output test.csv

# Validate RNG output
stochastic validate-rng --input rng_output.bin --certification NIST-SP800-22
```

## Examples

### 1. Neural Spike Train Analysis

```rust
// examples/neural_spike_trains.rs
use stochastic_chaos::LyapunovAnalyzer;
use stochastic_information::EntropyAnalyzer;

let spike_train = TimeSeries::from_neurodata("recordings/neuron_001.nex")?;

// Detect chaotic dynamics
let chaos = LyapunovAnalyzer::new(embedding_dim: 3, delay: 10);
let lyapunov_exp = chaos.largest_exponent(&spike_train)?;
println!("Largest Lyapunov: {:.4} (chaos={:?})",
         lyapunov_exp, lyapunov_exp > 0.0);

// Information content
let entropy = EntropyAnalyzer::new();
let shannon_h = entropy.shannon_entropy(&spike_train)?;
println!("Shannon entropy: {:.4} bits", shannon_h);
```

### 2. Quantum System Analysis

```rust
// examples/quantum_system.rs
use stochastic_quantum::IntegratedInformationAnalyzer;

let quantum_state = TimeSeries::from_quantum_measurements("entangled_pair.csv")?;

// Compute Φ (integrated information)
let phi_analyzer = IntegratedInformationAnalyzer::new();
let phi = phi_analyzer.compute_phi(&quantum_state)?;
println!("Integrated Information Φ: {:.6}", phi);
```

### 3. Financial Time Series

```rust
// examples/financial_timeseries.rs
use stochastic_fractal::HurstAnalyzer;

let stock_prices = TimeSeries::from_csv("stock_data.csv")?;

let hurst = HurstAnalyzer::new().compute(&stock_prices)?;
match hurst {
    h if h > 0.5 => println!("Persistent (trending): H={:.3}", h),
    h if h < 0.5 => println!("Anti-persistent (mean-reverting): H={:.3}", h),
    _ => println!("Random walk: H≈0.5"),
}
```

### 4. RNG Validation

```rust
// examples/rng_validation.rs
use stochastic_frequentist::ChiSquaredTest;

// Validate lottery/RNG as baseline for true randomness
let lottery_data = TimeSeries::from_csv("lottery_draws.csv")?;

let chi2 = ChiSquaredTest::new();
let result = chi2.test_uniformity(&lottery_data)?;

if result.p_value > 0.05 {
    println!("✓ Data consistent with uniform randomness (p={:.4})", result.p_value);
} else {
    println!("✗ Significant deviation from randomness detected (p={:.4})", result.p_value);
}
```

## Performance

- **Parallel execution**: All analyzers support multi-threading via `rayon`
- **Benchmarks**: See `tests/benchmarks/` for performance metrics
- **Optimization**: Critical paths use SIMD where available

Typical analysis times (10,000 data points, Intel i7-12700K):
- Frequentist tests: ~5ms
- Bayesian MCMC: ~200ms (10k iterations)
- Spectral FFT: ~15ms
- Topological persistence: ~800ms
- Quantum Φ: ~1.5s (depends on partition count)

## Citation

If you use this framework in academic research, please cite:

```bibtex
@software{molina_stochastic_2025,
  author = {Molina, Francisco},
  title = {{STOCHASTIC Framework: Multi-Framework Analysis for
           Stochastic Process Pattern Detection}},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/Yatrogenesis/STOCHASTIC-Framework},
  version = {1.0.0},
  doi = {10.5281/zenodo.XXXXXXX}
}
```

## Theoretical Foundations

Each framework is grounded in established mathematical theory:

1. **Frequentist**: Pearson (1900), Kolmogorov-Smirnov (1933)
2. **Bayesian**: Dirichlet (1839), Metropolis-Hastings (1953)
3. **Ergodic**: Birkhoff (1931), von Neumann (1932)
4. **Galois**: Galois (1830), Legendre (1798)
5. **Fractal**: Mandelbrot (1982), Peng et al. (1994)
6. **Chaos**: Lyapunov (1892), Lorenz (1963), Takens (1981)
7. **Information**: Shannon (1948), Kolmogorov (1965)
8. **TDA**: Edelsbrunner et al. (2002), Carlsson (2009)
9. **Spectral**: Fourier (1822), Welch (1967)
10. **Network**: Lacasa et al. (2008), Newman (2006)
11. **Quantum**: Von Neumann (1927), Tononi (2004)

## Contributing

Contributions are welcome! Areas of particular interest:

- Additional frameworks (e.g., wavelet coherence, recurrence quantification)
- GPU acceleration for computationally intensive analyses
- Integration with domain-specific data formats (neurophysiology, astrophysics)
- Improved documentation and tutorials

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Author

**Francisco Molina**
ORCID: [0009-0008-6093-8267](https://orcid.org/0009-0008-6093-8267)
Email: pako.molina@gmail.com

**Research Interests**: Consciousness AI, Quantum Information Theory, Complex Systems, Stochastic Processes

## Acknowledgments

This framework builds upon decades of mathematical research. Special thanks to the Rust scientific computing community for excellent libraries including `nalgebra`, `ndarray`, `statrs`, and `rustfft`.

## Related Work

- [NIST Statistical Test Suite](https://csrc.nist.gov/projects/random-bit-generation/documentation-and-software): Standard RNG testing
- [PyTDA](https://github.com/scikit-tda/scikit-tda): Python topological data analysis
- [Information Dynamics Toolkit](https://github.com/jlizier/jidt): Java information theory
- [Integrated Information Theory](https://integratedinformationtheory.org/): Consciousness research

---

**Version**: 1.0.0
**Status**: Active Development
**Last Updated**: 2025-01-13
