"""
RS force-ladder invariant map for cos(theta_W)(mu) with no empirical inputs.

This module defines a parameter-free, monotone cosθ_W(μ) constructed solely from
the ledger's gap series g_m, the golden ratio φ, and the recognition energy E_rec.

Idea:
- Build complementary ladder proxies for the U(1)_Y and SU(2)_L sectors using a
  smoothed, φ-scaled argument x = ln(μ_eV/E_rec)/(2 ln φ), and a signed sum over
  the alternating gap coefficients g_m weighted by tanh(x/m).
- Map these proxies to positive "coupling-like" quantities by exponentiation, and
  assemble cosθ_W(μ) = g2/sqrt(g'^2 + g2^2) with g'^2 = (3/5) g1^2 (group-theoretic).

No experimental masses, splittings, or low-energy couplings are used.
"""

import math
from typing import Tuple


# Golden ratio and helpers
PHI = (1.0 + math.sqrt(5.0)) / 2.0
LN_PHI = math.log(PHI)


def _g_m(m: int) -> float:
    """Ledger alternating gap coefficient g_m = (-1)^(m+1) / (m φ^m)."""
    return ((-1.0) ** (m + 1)) / (m * (PHI ** m))


def _recognition_energy_eV() -> float:
    """E_rec = ħ c / λ_rec with λ_rec = sqrt(ħ G / (π c^3))."""
    hbar_SI = 1.054571817e-34  # J*s
    c_SI = 299792458.0  # m/s
    G_SI = 6.67430e-11  # m^3 kg^-1 s^-2
    lambda_rec = math.sqrt(hbar_SI * G_SI / (math.pi * (c_SI ** 3)))
    # ħ c in eV*m: 197.3269804 MeV*fm = 1.973269804e-7 eV*m
    hbar_c_eVm = 1.973269804e-7
    return hbar_c_eVm / lambda_rec


def _ladder_proxy_sum(x: float, max_m: int = 64, eps_tail: float = 1e-12) -> Tuple[float, float]:
    """Compute complementary proxy exponents (a_Y, a_2) using tanh smoothing.

    a_Y(x) = Σ g_m * tanh(+ x / m)
    a_2(x) = Σ g_m * tanh(- x / m)

    Truncates when tail bound from |g_m|/m is negligible.
    """
    aY = 0.0
    a2 = 0.0
    for m in range(1, max_m + 1):
        gm = _g_m(m)
        tY = math.tanh(x / m)
        t2 = math.tanh(-x / m)
        aY += gm * tY
        a2 += gm * t2
        # geometric-harmonic tail estimate ~ φ^{-m}/m
        tail = (PHI ** (-(m + 1))) / (m + 1)
        if tail < eps_tail:
            break
    return aY, a2


def cos_theta_w_rs_force(mu_GeV: float) -> float:
    """Parameter-free RS cosθ_W(μ) from force-ladder proxies.

    Steps:
    - x = ln(μ_eV / E_rec) / (2 ln φ)
    - a_Y, a_2 from alternating gap series with tanh smoothing
    - α1_hat = exp(a_Y), α2_hat = exp(a_2)
    - cosθ_W = sqrt(α2_hat) / sqrt( (3/5) α1_hat + α2_hat )
    """
    mu_eV = max(mu_GeV, 1e-12) * 1.0e9
    E_rec = _recognition_energy_eV()
    x = math.log(mu_eV / E_rec) / (2.0 * LN_PHI)
    aY, a2 = _ladder_proxy_sum(x)
    alpha1_hat = math.exp(aY)
    alpha2_hat = math.exp(a2)
    gp_sq = (3.0 / 5.0) * alpha1_hat
    g2_sq = alpha2_hat
    return math.sqrt(g2_sq) / math.sqrt(gp_sq + g2_sq)


__all__ = ["cos_theta_w_rs_force"]


