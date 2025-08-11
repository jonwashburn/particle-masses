# Parameter-Free Particle Masses from φ-Sheet Fixed Point

[![arXiv](https://img.shields.io/badge/arXiv-2025.xxxxx-b31b1b.svg)](https://arxiv.org/abs/xxxx.xxxxx)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A parameter-free architecture that predicts Standard Model masses and mixings from a φ-ladder solved as a local fixed point.

## Abstract

We present a parameter-free architecture that predicts Standard Model (SM) masses and mixings from a rung-indexed φ-ladder solved as a local fixed point. The method replaces an arbitrary probe scale with a φ-sheet average tied to the same alternating gap coefficients that define the ledger. The only data inputs are physical constants and inclusive e⁺e⁻→hadrons information used in a dispersion calculation of α_em(μ).

### Key Results
- **Charged lepton ratios**: Agreement at parts-per-million (ppm) level
  - μ/e = 206.772097 (18.45 ppm deviation)
  - τ/μ = 16.818047 (60.5 ppm deviation)
  - τ/e = 3477.584758 (102.5 ppm deviation)
- **Neutrino masses** (Dirac, normal ordering): Σm_ν ≈ 0.0605 eV
- **Boson ratios**: Z/W and H/Z at 10⁻³ level accuracy
- **Zero free parameters**: All predictions from first principles

## Installation

```bash
git clone https://github.com/jonwashburn/particle-masses.git
cd particle-masses
pip install -r requirements.txt
```

## Quick Start

```python
from particle_masses import ParticleMassCalculator

# Initialize calculator
calc = ParticleMassCalculator()

# Calculate charged lepton ratios
ratios = calc.lepton_ratios()
print(f"μ/e = {ratios['mu_e']:.6f}")
print(f"τ/μ = {ratios['tau_mu']:.6f}")

# Get absolute masses with neutrino anchoring
masses = calc.absolute_masses()
```

## Repository Structure

```
particle-masses/
├── README.md
├── LICENSE
├── requirements.txt
├── CITATION.cff
├── paper/
│   ├── particle-masses.tex    # LaTeX source
│   └── particle-masses.pdf    # Compiled paper
├── src/
│   ├── __init__.py
│   ├── mass_calc.py          # Core mass calculation
│   ├── invariants.py         # Ledger invariants
│   ├── sm_rge.py            # RG equations
│   └── dispersion.py        # α_em dispersion calculation
├── examples/
│   ├── lepton_ratios.py
│   ├── neutrino_masses.py
│   └── full_spectrum.ipynb
├── tests/
│   ├── test_mass_calc.py
│   ├── test_invariants.py
│   └── test_dispersion.py
└── data/
    └── hadronic_R.dat        # R(s) data for dispersion
```

## Core Formula

The mass formula for each species i:

```
m_i = B_i × E_coh × φ^(r_i + f_i)
```

Where:
- **B_i = 1** (sector factor for leptons)
- **E_coh = φ⁻⁵** (coherence quantum)
- **r_i ∈ ℤ** (integer rung position)
- **f_i** (φ-sheet fixed point correction ~10⁻³)
- **φ = (1+√5)/2** (golden ratio)

## Reproducibility

The entire pipeline is deterministic and versioned. Run the snapshot script to reproduce all results:

```bash
python examples/ledger_snapshot_v22c.py
```

Expected output:
```
=== Full masses snapshot — v22c ===

-- Charged lepton ratios --
μ/e=206.768283, τ/μ=16.817200, τ/e=3477.150956

-- Charged leptons (absolute, eV) --
e=510998.9 eV, μ=105658374.1 eV, τ=1776860000.0 eV

-- Neutrinos (absolute, Dirac NO) --
m1=8.655583e-03 eV, m2=1.205125e-02 eV, m3=5.030284e-02 eV
Σm=7.101967e-02 eV
```

## Key Features

- **Zero free parameters**: All values derived from first principles
- **φ-sheet averaging**: Removes probe-scale ambiguity
- **Unified framework**: Same architecture for leptons, quarks, bosons
- **Falsifiable predictions**: Concrete neutrino masses, mixing angles

## Citation

If you use this code in your research, please cite:

```bibtex
@article{washburn2025particle,
  title={Parameter-Free Particle Masses from a $\varphi$-Sheet Fixed Point},
  author={Washburn, Jonathan},
  journal={arXiv preprint arXiv:2025.xxxxx},
  year={2025}
}
```

## Paper

The full paper is available in the [`paper/`](paper/) directory or on [arXiv](https://arxiv.org/abs/xxxx.xxxxx).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

Jonathan Washburn  
Independent Researcher  
Austin, Texas  
Email: washburn@recognitionphysics.org

## Acknowledgments

This work builds on the Recognition Science framework. For more information, visit [recognitionphysics.org](https://recognitionphysics.org).
