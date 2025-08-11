"""
Standard Model renormalization group equations and running couplings.

This module implements:
- QED and SM anomalous dimensions
- Dispersion calculation of α_em(μ)
- Electroweak running
"""

import math
import numpy as np
from typing import Callable, Optional

# Physical constants
M_Z = 91.1876  # GeV
ALPHA_EM_MZ = 1.0 / 127.955  # Fine structure constant at M_Z


def alpha_em_dispersion(mu: float, use_hadronic: bool = True) -> float:
    """
    Calculate running α_em(μ) using vacuum polarization.
    
    Parameters
    ----------
    mu : float
        Energy scale in GeV
    use_hadronic : bool
        Include hadronic contributions
    
    Returns
    -------
    float
        α_em at scale μ
    """
    # Simplified implementation - in production would use full dispersion integral
    alpha_0 = 1.0 / 137.035999
    
    if mu < 1.0:
        return alpha_0
    
    # Simple logarithmic running (placeholder for full dispersion)
    Q2 = mu**2
    
    # Leptonic contribution
    delta_lep = (1.0 / (3 * math.pi)) * math.log(Q2 / 0.000511**2)  # electron
    delta_lep += (1.0 / (3 * math.pi)) * math.log(Q2 / 0.105658**2)  # muon
    delta_lep += (1.0 / (3 * math.pi)) * math.log(Q2 / 1.77686**2)  # tau
    
    # Hadronic contribution (simplified)
    if use_hadronic and mu > 1.0:
        delta_had = 0.0276 * math.log(Q2 / 1.0)  # Approximate
    else:
        delta_had = 0.0
    
    alpha = alpha_0 / (1 - delta_lep - delta_had)
    
    return alpha


def gamma_qed_lepton(mu: float, lepton: str = 'e') -> float:
    """
    QED mass anomalous dimension for leptons.
    
    γ^QED_ℓ(μ) = (3α_em(μ))/(4π) [1 + (3/4)(α_em(μ)/π)]
    
    Parameters
    ----------
    mu : float
        Energy scale in GeV
    lepton : str
        Lepton type ('e', 'mu', 'tau')
    
    Returns
    -------
    float
        QED anomalous dimension
    """
    alpha = alpha_em_dispersion(mu)
    
    # One-loop + leading two-loop
    gamma = (3 * alpha) / (4 * math.pi)
    gamma *= 1 + (3.0 / 4.0) * (alpha / math.pi)
    
    return gamma


def gamma_sm_lepton(mu: float, lepton: str = 'e') -> float:
    """
    SM (electroweak + Yukawa) contributions to lepton anomalous dimension.
    
    Parameters
    ----------
    mu : float
        Energy scale in GeV
    lepton : str
        Lepton type
    
    Returns
    -------
    float
        SM anomalous dimension contribution
    """
    # Simplified - would include full 2-loop gauge and Yukawa
    if mu < M_Z:
        return 0.0
    
    # Placeholder for electroweak contributions
    g2_squared = 0.65  # SU(2) coupling squared (approximate)
    g1_squared = 0.35  # U(1) coupling squared (approximate)
    
    gamma_ew = (1.0 / (16 * math.pi**2)) * (
        (9.0 / 4.0) * g2_squared + (3.0 / 4.0) * g1_squared
    )
    
    # Yukawa contributions (mainly for tau)
    if lepton == 'tau':
        y_tau = 1.77686 / 174.0  # Approximate Yukawa coupling
        gamma_yukawa = (y_tau**2) / (16 * math.pi**2)
    else:
        gamma_yukawa = 0.0
    
    return gamma_ew + gamma_yukawa


# Specific lepton anomalous dimensions
def gamma_e(mu: float) -> float:
    """QED anomalous dimension for electron."""
    return gamma_qed_lepton(mu, 'e')


def gamma_mu(mu: float) -> float:
    """QED anomalous dimension for muon."""
    return gamma_qed_lepton(mu, 'mu')


def gamma_tau(mu: float) -> float:
    """QED anomalous dimension for tau."""
    return gamma_qed_lepton(mu, 'tau')


def gamma_e_sm(mu: float) -> float:
    """SM anomalous dimension for electron."""
    return gamma_sm_lepton(mu, 'e')


def gamma_mu_sm(mu: float) -> float:
    """SM anomalous dimension for muon."""
    return gamma_sm_lepton(mu, 'mu')


def gamma_tau_sm(mu: float) -> float:
    """SM anomalous dimension for tau."""
    return gamma_sm_lepton(mu, 'tau')


# Total anomalous dimensions
def gamma_e_total(mu: float) -> float:
    """Total (QED + SM) anomalous dimension for electron."""
    return gamma_e(mu) + gamma_e_sm(mu)


def gamma_mu_total(mu: float) -> float:
    """Total (QED + SM) anomalous dimension for muon."""
    return gamma_mu(mu) + gamma_mu_sm(mu)


def gamma_tau_total(mu: float) -> float:
    """Total (QED + SM) anomalous dimension for tau."""
    return gamma_tau(mu) + gamma_tau_sm(mu)
