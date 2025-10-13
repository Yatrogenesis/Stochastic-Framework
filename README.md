# 🎯 LOTTO Analyzer

**Advanced Lottery Analysis Framework - Multi-dimensional Mathematical Analysis Engine**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org/)

## 📋 Overview

LOTTO Analyzer is a sophisticated desktop application that performs comprehensive mathematical analysis on lottery data using **9 independent frameworks**. Based on rigorous scientific methodology, it provides data-driven insights through frequentist statistics, Bayesian inference, chaos theory, fractal analysis, and more.

**Author**: Francisco Molina (pako.molina@gmail.com)
**ORCID**: https://orcid.org/0009-0008-6093-8267
**License**: MIT
**Version**: 1.0.0

## 🔬 Mathematical Frameworks

### 1. Frequentist Probability (χ² Test)
- **Foundation**: Law of Large Numbers, Central Limit Theorem
- **Method**: Pearson's chi-squared goodness-of-fit test
- **Purpose**: Detect deviations from uniform distribution
- **Output**: χ² statistic, p-value, hypothesis test result

### 2. Bayesian Inference
- **Foundation**: Bayes' Theorem with Dirichlet conjugate prior
- **Prior**: Jeffreys non-informative (α = 0.5)
- **Posterior**: Dirichlet(α + counts)
- **Output**: Posterior probabilities, 95% credible intervals, top numbers

### 3. Ergodic Theory
- **Foundation**: Birkhoff-Khinchin Theorem
- **Tests**: Temporal mean vs ensemble mean, runs test (Z-score)
- **Analysis**: Autocorrelation function
- **Output**: Ergodicity verification, mixing properties

### 4. Range Equilibrium
- **Ranges**: Low [1-9], Medium [10-19], High [20-28]
- **Analysis**: Distribution percentages, even/odd parity
- **Test**: χ² balance test across ranges
- **Output**: Balance verification, distribution metrics

### 5. Galois Field Theory (GF(29))
- **Field**: ℤ/29ℤ (prime field)
- **Generator**: g = 2
- **Analysis**: Quadratic residues, multiplicative group structure
- **Output**: Combinatorial dimension (log₂(C(28,5)) ≈ 16.58 bits)

### 6. Multifractal Analysis
- **Method**: Detrended Fluctuation Analysis (DFA)
- **Metrics**: Hurst exponent (H), fractal dimension (D = 2 - H)
- **Interpretation**:
  - H > 0.5: Persistent (long-range correlations)
  - H = 0.5: Random walk
  - H < 0.5: Anti-persistent
- **Output**: Hurst exponent, normalized entropy

### 7. Chaos Theory
- **Methods**: Lyapunov exponent, Takens embedding reconstruction
- **Parameters**: Embedding dimension m=3, time delay τ=1
- **Tests**: λ > 0 indicates chaotic behavior
- **Output**: Lyapunov exponent, correlation dimension

### 8. Normalization & Factorization
- **Methods**: Z-score outlier detection, Shapiro-Wilk normality test
- **Transformation**: Box-Cox power transformation
- **Thresholds**: |z| > 2 for outliers
- **Output**: Outlier list, normality statistics, optimal λ

### 9. Critical Path (Multi-Objective Optimization)
- **Scoring Weights**:
  - 35% - Bayesian posterior (inverse priority)
  - 35% - Historical frequency (inverse priority)
  - 15% - Range equilibrium
  - 15% - Prime diversity
- **Output**: Recommended combination, quality score, detailed properties

## 🚀 Features

- **Multi-Format Support**: CSV, Excel (XLS/XLSX), DAT, SQLite, PostgreSQL, MySQL
- **Desktop GUI**: Modern interface built with egui/eframe
- **Parallel Processing**: Utilizes rayon for multi-core performance
- **Confidence Levels**: 80%, 90%, 95%, 99%
- **Real-Time Analysis**: Background processing with progress feedback
- **Export Results**: JSON export for further analysis
- **Comprehensive Tests**: 42+ unit tests across all modules

## 📦 Installation

### Prerequisites
- Rust 1.70 or higher
- Git

### Build from Source

```bash
# Clone the repository
git clone https://github.com/Yatrogenesis/LOTTO.git
cd LOTTO

# Build release version
cargo build --release

# Run the application
cargo run --release
```

## 📊 Data Format

### CSV Format
```csv
date,number1,number2,number3,number4,number5
2025-10-13,2,7,10,19,23
2025-10-06,1,5,12,18,26
```

### Excel Format (XLS/XLSX)
| Date       | Num1 | Num2 | Num3 | Num4 | Num5 |
|------------|------|------|------|------|------|
| 2025-10-13 | 2    | 7    | 10   | 19   | 23   |
| 2025-10-06 | 1    | 5    | 12   | 18   | 26   |

### SQLite Schema
```sql
CREATE TABLE draws (
    id INTEGER PRIMARY KEY,
    date TEXT NOT NULL,
    numbers TEXT NOT NULL  -- Format: "2,7,10,19,23"
);
```

## 🎮 Usage

1. **Launch Application**: Run `cargo run --release` or execute the binary
2. **Load Data**: Click "Load Data File" and select your lottery data
3. **Configure**: Choose confidence level (80%, 90%, 95%, 99%)
4. **Analyze**: Click "Run Complete Analysis"
5. **View Results**: See recommended combination and detailed analysis
6. **Export**: Save results as JSON for documentation

## 📈 Example Output

```
🎯 Recommended Combination: [2, 7, 10, 19, 23]
Quality Score: 0.7624

Properties:
- Sum: 61
- Mean: 12.2
- Median: 10
- Std Dev: 8.7
- Even: 2 | Odd: 3
- Primes: 4 (2, 7, 19, 23)
- Distribution: Low 40% | Medium 40% | High 20%
```

## 🧪 Testing

```bash
# Run all tests
cargo test

# Run specific module tests
cargo test --lib frequentist
cargo test --lib bayesian
cargo test --lib chaos

# Run with output
cargo test -- --nocapture
```

## 📚 Project Structure

```
LOTTO/
├── src/
│   ├── main.rs                    # Application entry point
│   ├── models.rs                  # Data structures
│   ├── data_loader.rs             # Multi-format data loading
│   ├── ui.rs                      # Desktop GUI
│   └── analysis/
│       ├── mod.rs                 # Analysis orchestrator
│       ├── frequentist.rs         # χ² test
│       ├── bayesian.rs            # Bayesian inference
│       ├── ergodic.rs             # Ergodic theory
│       ├── range_equilibrium.rs   # Range analysis
│       ├── galois.rs              # Galois field GF(29)
│       ├── multifractal.rs        # DFA, Hurst exponent
│       ├── chaos.rs               # Lyapunov, Takens
│       ├── normalization.rs       # Outliers, Box-Cox
│       └── critical_path.rs       # Multi-objective optimization
├── Cargo.toml                     # Dependencies
└── README.md                      # This file
```

## 🔬 Scientific Rigor

This framework adheres to:

- ✅ **Falsifiability** (Popper's criterion)
- ✅ **Reproducibility** (open-source, documented methods)
- ✅ **Statistical validity** (proper hypothesis testing)
- ✅ **Transparency** (all assumptions documented)
- ✅ **Peer-reviewable** (complete methodology disclosure)

## ⚠️ Important Disclaimers

1. **No Prediction Guarantee**: If the lottery uses a certified RNG (Random Number Generator), historical analysis provides NO advantage over random selection.

2. **Expected Value**: The mathematical expectation of lottery games is typically negative (house edge).

3. **Gambler's Fallacy**: "Overdue" numbers do NOT have higher probability in truly random systems.

4. **Educational Purpose**: This tool is for research, education, and mathematical exploration.

5. **Sample Size**: Small datasets (n < 500) have LOW statistical power. Results should be interpreted with extreme caution.

## 📖 Theoretical Background

### Key References

1. **Kolmogorov** (1933): Foundations of Probability Theory
2. **Birkhoff** (1931): Ergodic Theorem
3. **Shannon** (1948): Information Theory
4. **Mandelbrot** (1982): Fractal Geometry
5. **Lyapunov** (1892): Stability Theory
6. **Takens** (1981): State Space Reconstruction
7. **Galois** (1832): Field Theory
8. **Bayes-Laplace** (1763): Bayesian Inference

### Statistical Tests

- Pearson's χ² test (1900)
- Shapiro-Wilk test (1965)
- Wald-Wolfowitz runs test (1940)
- DFA (Peng et al., 1994)

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass: `cargo test`
5. Submit a pull request

## 📧 Contact

**Francisco Molina**
- Email: pako.molina@gmail.com
- ORCID: https://orcid.org/0009-0008-6093-8267
- GitHub: [@Yatrogenesis](https://github.com/Yatrogenesis)

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details

Copyright (c) 2025 Francisco Molina

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---

**⚠️ Responsible Gaming**: Gambling can be addictive. Play responsibly. Never bet more than you can afford to lose. This software is for educational and research purposes only and does not constitute financial advice.

**Last Updated**: October 13, 2025
