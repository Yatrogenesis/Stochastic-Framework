# Mathematical Foundations of STOCHASTIC Framework

This document provides the theoretical background for each of the 13 mathematical frameworks implemented in STOCHASTIC.

## Framework 1: Frequentist Statistics

### Chi-Squared (χ²) Test

**Hypothesis**: H₀: Data follows a uniform distribution

**Test Statistic**:
```
χ² = Σᵢ (Oᵢ - Eᵢ)² / Eᵢ
```
where:
- Oᵢ = observed frequency in bin i
- Eᵢ = expected frequency under H₀
- Degrees of freedom: k - 1 (k = number of bins)

**Decision Rule**: Reject H₀ if χ² > χ²₍α,k₋₁₎

**References**:
- Pearson, K. (1900). "On the criterion that a given system of deviations..."

### Kolmogorov-Smirnov Test

**Test Statistic**:
```
D = supₓ |F_n(x) - F(x)|
```
where:
- F_n(x) = empirical CDF
- F(x) = theoretical CDF

**Critical Value**: D_crit = √(-ln(α/2) / (2n))

---

## Framework 2: Bayesian Inference

### Dirichlet-Multinomial Model

**Prior**: Dirichlet(α₁, ..., αₖ)
```
p(θ|α) = [Γ(Σαᵢ) / Πᵢ Γ(αᵢ)] × Πᵢ θᵢ^(αᵢ-1)
```

**Likelihood**: Multinomial(n; θ₁, ..., θₖ)
```
p(x|θ) = (n! / Πᵢ xᵢ!) × Πᵢ θᵢ^xᵢ
```

**Posterior**: Dirichlet(α₁ + x₁, ..., αₖ + xₖ)

**Jeffreys Prior**: αᵢ = 0.5 (non-informative)

### Metropolis-Hastings MCMC

**Algorithm**:
1. Start with θ⁽⁰⁾
2. For t = 1 to T:
   - Sample θ* ~ q(·|θ⁽ᵗ⁻¹⁾)
   - Compute acceptance ratio: r = π(θ*) q(θ⁽ᵗ⁻¹⁾|θ*) / [π(θ⁽ᵗ⁻¹⁾) q(θ*|θ⁽ᵗ⁻¹⁾)]
   - Accept with probability min(1, r)

**Convergence**: Gelman-Rubin R̂ statistic

---

## Framework 3: Ergodic Theory

### Birkhoff-Khinchin Theorem

For an ergodic system:
```
lim(n→∞) (1/n) Σᵢ₌₀ⁿ⁻¹ f(Tⁱx) = ∫ f dμ   (a.e.)
```

**Time Average** = **Ensemble Average**

### Mixing Properties

**Weak Mixing**:
```
lim(n→∞) (1/n) Σₖ₌₀ⁿ⁻¹ |μ(A ∩ T⁻ᵏB) - μ(A)μ(B)| = 0
```

**Strong Mixing** (α-mixing):
```
α(k) = sup{|μ(A ∩ B) - μ(A)μ(B)| : A ∈ ℱ₀, B ∈ ℱₖ} → 0
```

### Autocorrelation Test

```
ρ(k) = Cov(Xₜ, Xₜ₊ₖ) / √(Var(Xₜ) Var(Xₜ₊ₖ))
```

For ergodic process: ρ(k) → 0 as k → ∞

---

## Framework 4: Range Equilibrium Analysis

### Partition Chi-Squared

Divide range [min, max] into m partitions.

**Test Statistic**:
```
χ²_balance = Σⱼ (nⱼ - n/m)² / (n/m)
```

**Parity Test**:
```
χ²_parity = (n_even - n_odd)² / n
```

---

## Framework 5: Galois Field Theory

### Finite Field GF(p)

For prime p, GF(p) = ℤ/pℤ with operations:
- Addition: (a + b) mod p
- Multiplication: (a × b) mod p

**Multiplicative Group**: (ℤ/pℤ)* has order p - 1

### Quadratic Residues

**Legendre Symbol**:
```
(a/p) = { 1   if a is QR mod p
        {-1   if a is NQR mod p
        { 0   if p | a
```

**Euler's Criterion**: (a/p) ≡ a^((p-1)/2) (mod p)

**Theoretical Distribution**: 50% QR, 50% NQR

---

## Framework 6: Multifractal Analysis

### Detrended Fluctuation Analysis (DFA)

**Algorithm**:
1. Integrate: Y(i) = Σₖ₌₁ⁱ [x(k) - x̄]
2. Divide into boxes of size n
3. Fit polynomial in each box
4. Compute fluctuation: F(n) = √(1/N Σᵢ[Y(i) - yₙ(i)]²)
5. Plot log F(n) vs log n

**Hurst Exponent**: H = slope

**Interpretation**:
- H > 0.5: Persistent (positive correlations)
- H = 0.5: Random walk (uncorrelated)
- H < 0.5: Anti-persistent (negative correlations)

### Fractal Dimension

```
D = 2 - H
```

### Multifractal Spectrum f(α)

**Partition Function**:
```
χ(q, ε) = Σᵢ pᵢ(ε)^q
```

**Scaling**: χ(q, ε) ~ ε^τ(q)

**Singularity Spectrum**: f(α) via Legendre transform

---

## Framework 7: Chaos Theory

### Lyapunov Exponent

**Largest Lyapunov Exponent**:
```
λ₁ = lim(t→∞) (1/t) ln(‖δx(t)‖ / ‖δx(0)‖)
```

**Interpretation**:
- λ > 0: Chaotic (exponential divergence)
- λ = 0: Marginally stable
- λ < 0: Stable (convergence)

### Takens Embedding

**Reconstruction**:
```
X(t) = [x(t), x(t+τ), x(t+2τ), ..., x(t+(m-1)τ)]
```

**Parameters**:
- m = embedding dimension (Cao method)
- τ = time delay (mutual information minimum)

### Correlation Dimension

**Correlation Sum**:
```
C(r) = (1/N²) Σᵢ≠ⱼ Θ(r - ‖Xᵢ - Xⱼ‖)
```

**Scaling**: C(r) ~ r^D₂ where D₂ = correlation dimension

---

## Framework 8: Information Theory

### Shannon Entropy

**Discrete**:
```
H(X) = -Σᵢ p(xᵢ) log₂ p(xᵢ)
```

**Differential** (continuous):
```
h(X) = -∫ f(x) log f(x) dx
```

**Properties**:
- H(X) ≥ 0
- H(X) ≤ log₂ k (uniform maximum)

### Mutual Information

```
I(X; Y) = Σₓ Σᵧ p(x,y) log[p(x,y) / (p(x)p(y))]
```

**Interpretation**: Reduction in uncertainty about X given Y

### Kolmogorov Complexity

```
K(x) = min{|p| : U(p) = x}
```

**Approximation**: Lossless compression ratio

---

## Framework 9: Normalization & Outlier Detection

### Z-Score

```
z = (x - μ) / σ
```

**Outlier Criterion**: |z| > k (typically k = 2 or 3)

### Shapiro-Wilk Test

**Test Statistic**:
```
W = (Σᵢ aᵢ x₍ᵢ₎)² / Σᵢ (xᵢ - x̄)²
```

where x₍ᵢ₎ are order statistics and aᵢ are tabulated coefficients.

**Null Hypothesis**: Data is normally distributed

### Box-Cox Transformation

```
y(λ) = { (x^λ - 1) / λ    if λ ≠ 0
       { ln(x)             if λ = 0
```

**Optimal λ**: Maximum likelihood estimate

---

## Framework 10: Topological Data Analysis

### Persistent Homology

**Vietoris-Rips Complex**: VR(X, ε) = simplicial complex where:
- Vertices: data points
- k-simplex: {x₀, ..., xₖ} if d(xᵢ, xⱼ) ≤ ε for all i,j

**Filtration**: VR(X, ε₁) ⊆ VR(X, ε₂) for ε₁ < ε₂

**Persistence**: Track birth/death of topological features

### Betti Numbers

- β₀ = number of connected components
- β₁ = number of 1-dimensional holes (loops)
- β₂ = number of 2-dimensional voids
- ...

**Persistence Diagram**: Plot (birth, death) pairs

---

## Framework 11: Spectral Analysis

### Fourier Transform

**Discrete Fourier Transform**:
```
X(k) = Σₙ₌₀ᴺ⁻¹ x(n) e^(-i2πkn/N)
```

**Power Spectral Density**:
```
S(f) = lim(T→∞) E[|X_T(f)|²] / T
```

### Welch's Method

1. Divide signal into overlapping segments
2. Window each segment (Hanning, Hamming)
3. FFT each segment
4. Average periodograms

**Variance Reduction**: √K improvement with K segments

### Coherence

**Cross-Spectral Density**:
```
Cₓᵧ(f) = ∫ Rₓᵧ(τ) e^(-i2πfτ) dτ
```

**Coherence**:
```
γ²ₓᵧ(f) = |Cₓᵧ(f)|² / [Sₓₓ(f) Sᵧᵧ(f)]
```

Range: [0, 1] (0 = uncorrelated, 1 = perfectly correlated)

---

## Framework 12: Network Theory

### Visibility Graph

**Natural Visibility**: Point i "sees" point j if:
```
∀k ∈ (i, j): y(k) < y(j) + (y(i) - y(j)) × (t(k) - t(j)) / (t(i) - t(j))
```

**Horizontal Visibility**: Simpler criterion (heights comparison)

### Centrality Measures

**Betweenness**:
```
C_B(v) = Σₛ≠v≠ₜ σₛₜ(v) / σₛₜ
```
where σₛₜ(v) = number of shortest paths through v

**Closeness**:
```
C_C(v) = 1 / Σᵤ d(v, u)
```

**Eigenvector Centrality**:
```
x_v = (1/λ) Σᵤ A_vu x_u
```
where λ = largest eigenvalue of adjacency matrix A

### Community Detection

**Modularity**:
```
Q = (1/2m) Σᵢⱼ [Aᵢⱼ - kᵢkⱼ/2m] δ(cᵢ, cⱼ)
```

**Louvain Algorithm**: Greedy modularity optimization

---

## Framework 13: Quantum Information Theory

### Density Matrix

**Pure State**: ρ = |ψ⟩⟨ψ|

**Mixed State**: ρ = Σᵢ pᵢ |ψᵢ⟩⟨ψᵢ|

**Properties**:
- Hermitian: ρ† = ρ
- Positive semidefinite: ⟨ψ|ρ|ψ⟩ ≥ 0
- Trace one: Tr(ρ) = 1

### Von Neumann Entropy

```
S(ρ) = -Tr(ρ log ρ) = -Σᵢ λᵢ log λᵢ
```

where λᵢ are eigenvalues of ρ

**Range**: [0, log d] where d = dimension of Hilbert space

### Integrated Information (Φ)

**IIT 3.0 Definition**:

1. **Partition**: Divide system into parts A and B
2. **Disconnection**: Compute minimum information partition (MIP)
3. **Φ**: Integrated information across MIP

**Effective Information**:
```
EI(C → E) = Σ p(e|c) log[p(e|c) / p(e)]
```

**Cause Information**: CI(C → E) = min partition EI

**Effect Information**: EI(C → E) = min partition EI

**Φ**: min(CI, EI)

**Interpretation**: Φ > 0 indicates irreducible integration (consciousness correlate)

### Entanglement Measures

**Negativity**:
```
N(ρ) = (‖ρ^T_A‖₁ - 1) / 2
```

where ρ^T_A is partial transpose

**Entanglement Entropy**:
```
E(ρ_AB) = S(ρ_A) = S(ρ_B)
```
for pure state |ψ⟩_AB

---

## Computational Complexity

| Framework | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| Chi-squared | O(n) | O(k) bins |
| KS test | O(n log n) | O(n) |
| Bayesian MCMC | O(T × d) | O(T) samples |
| DFA | O(n log n) | O(n) |
| Lyapunov | O(n × m²) | O(n × m) |
| FFT | O(n log n) | O(n) |
| Persistent Homology | O(n³) | O(n²) |
| Network Centrality | O(n³) | O(n²) |
| Integrated Info Φ | O(2^n) | O(2^n) |

**Legend**: n = data size, k = bins, T = MCMC iterations, d = dimensions, m = embedding dimension

---

## Statistical Power

**Minimum Sample Sizes** (approximate):

- Chi-squared: n ≥ 30 (≥ 5 per bin)
- KS test: n ≥ 20
- Bayesian: n ≥ 50 (for convergence)
- DFA: n ≥ 100 (preferably 1000+)
- Lyapunov: n ≥ 500
- FFT: n = 2^k (power of 2)
- Persistent Homology: n ≥ 100
- Φ computation: n ≤ 10 (computational limit)

---

## References

### Books

1. Cover, T. M., & Thomas, J. A. (2006). *Elements of Information Theory*
2. Kantz, H., & Schreiber, T. (2004). *Nonlinear Time Series Analysis*
3. Edelsbrunner, H., & Harer, J. (2010). *Computational Topology*
4. Newman, M. (2010). *Networks: An Introduction*
5. Nielsen, M. A., & Chuang, I. L. (2010). *Quantum Computation and Quantum Information*
6. Tononi, G. (2004). "An information integration theory of consciousness"

### Papers

1. Peng, C. K., et al. (1994). "Mosaic organization of DNA nucleotides"
2. Lacasa, L., et al. (2008). "From time series to complex networks"
3. Carlsson, G. (2009). "Topology and data"
4. Rosenstein, M. T., et al. (1993). "A practical method for calculating largest Lyapunov exponents"
5. Takens, F. (1981). "Detecting strange attractors in turbulence"

---

**Last Updated**: 2025-01-13
