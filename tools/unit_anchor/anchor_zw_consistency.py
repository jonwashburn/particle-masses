#!/usr/bin/env python3
"""
Z/W consistency anchor for internal absolute scale.

Provides utilities to derive the global scale s (eV per ladder unit) by solving
F(mu) = (mZ_phi/mW_phi) * cos(theta_W(mu)) - 1 = 0 for mu_star, then
s = (mu_star_eV) / mW_phi.

Two tilt providers are exposed:
- cos_theta_w_sm(mu): SM-like running (1-loop gauge flow; simple, monotone)
- cos_theta_w_rs(mu): RS-internal placeholder tilt (monotone φ-based curve)

Notes:
- This module intentionally avoids using experimental masses. The SM-like mode
  uses standard 1-loop beta functions with typical electroweak inputs at M_Z.
  These inputs are couplings, not masses, and serve only to shape a monotone
  cosθ_W(μ) needed to locate μ⋆. The RS mode is a parameter-free placeholder
  until a full force-ladder invariant map is specified.
"""

from __future__ import annotations

import math
from typing import Callable, Tuple, Optional
try:
    # Prefer parameter-free force-ladder map if available
    from src.force_ladder import cos_theta_w_rs_force as _cos_rs_force
except Exception:
    _cos_rs_force = None  # fallback to simple RS placeholder below


# ------------------------------
# Physical constants and helpers
# ------------------------------

# Reference electroweak scale (GeV)
M_Z = 91.1876
M_W = 80.379

# 1-loop SM beta coefficients for gauge couplings (GUT-normalized g1)
# d(1/alpha_i) / d ln mu = - b_i / (2π)
_B1 = 41.0 / 10.0
_B2 = -19.0 / 6.0


def _evolve_alpha_1loop(alpha0: float, b: float, mu0: float, mu: float) -> float:
    """One-loop evolution for a gauge coupling in terms of α = g^2/(4π).

    1/α(μ) = 1/α(μ0) - (b / (2π)) ln(μ/μ0)
    """
    inv_alpha = (1.0 / alpha0) - (b / (2.0 * math.pi)) * math.log(mu / mu0)
    # Guard against negative/zero
    inv_alpha = max(inv_alpha, 1e-12)
    return 1.0 / inv_alpha


def cos_theta_w_sm(mu: float,
                   mu0: float = M_Z,
                   sin2_thetaW_mu0: float = 0.23122,
                   alpha_em_mu0: float = 1.0 / 127.955) -> float:
    """Compute cosθ_W(μ) from simple 1-loop SM gauge running.

    Uses GUT normalization for g1: α1 = (5/3) α_Y. Given α_em(μ0) and sin^2θ_W(μ0),
    reconstruct α1(μ0), α2(μ0) via:
      α_em = α1 α2 / (α1 + α2),   sin^2θ_W = α1 / (α1 + α2)
    then evolve (α1, α2) with 1-loop β and form cosθ_W(μ) = g2 / sqrt(g'^2 + g2^2)
    with g'^2 = (3/5) g1^2.
    """
    # Reconstruct α1, α2 at μ0
    # α1/(α1+α2) = sin^2θ_W  => α1 = s2 * (α1 + α2)
    # α_em = α1 α2 / (α1 + α2)
    s2 = sin2_thetaW_mu0
    alpha_em0 = alpha_em_mu0
    # Solve for α1 + α2 = (α1 α2) / α_em = unknown; but use relations to get α1, α2
    # From s2 = α1 / (α1+α2) => α1 = s2 S, α2 = (1 - s2) S
    # α_em = α1 α2 / S = s2 (1 - s2) S
    S = alpha_em0 / (s2 * (1.0 - s2))
    alpha1_0 = s2 * S
    alpha2_0 = (1.0 - s2) * S

    # 1-loop evolution
    alpha1 = _evolve_alpha_1loop(alpha1_0, _B1, mu0, mu)
    alpha2 = _evolve_alpha_1loop(alpha2_0, _B2, mu0, mu)

    # Convert to couplings
    g1_sq = 4.0 * math.pi * alpha1
    g2_sq = 4.0 * math.pi * alpha2

    # GUT-normalized relation: g'^2 = (3/5) g1^2
    gp_sq = (3.0 / 5.0) * g1_sq
    cos_theta = math.sqrt(g2_sq) / math.sqrt(gp_sq + g2_sq)
    return cos_theta


def cos_theta_w_rs(mu: float) -> float:
    """RS-internal placeholder for cosθ_W(μ).

    This provides a smooth, monotone curve based only on φ and logarithms,
    without external measurements. It is intended as a placeholder until the
    full force-ladder invariant map is specified.

    Form: cosθ_W(μ) = 1 / sqrt(1 + c * [ln(μ_eV / E_rec)]^2),
    where E_rec = ħ c / λ_rec, λ_rec = sqrt(ħ G / (π c^3)). The constant c is
    chosen from φ-only data: c = 1 / (2 ln φ)^2 so that the characteristic
    scale is the closed-form sheet normalizer. This avoids any experimental
    masses or low-energy coupling inputs.
    """
    PHI = (1.0 + math.sqrt(5.0)) / 2.0
    ln_phi = math.log(PHI)
    # RS bridge energy (pure constants)
    E_rec = recognition_energy_eV()
    mu_eV = max(mu, 1e-9) * 1.0e9
    x = math.log(mu_eV / E_rec)
    c = 1.0 / ((2.0 * ln_phi) ** 2)
    cos_theta = 1.0 / math.sqrt(1.0 + c * x * x)
    return cos_theta


def solve_mu_star(mZ_over_mW_phi: float,
                  cos_theta_provider: Callable[[float], float],
                  bracket: Tuple[float, float] = (50.0, 200.0),
                  tol: float = 1e-10,
                  max_iter: int = 200) -> float:
    """Solve F(μ) = (mZ_phi/mW_phi) * cosθ_W(μ) - 1 = 0 by bisection.

    Parameters
    ----------
    mZ_over_mW_phi : float
        Dimensionless ladder ratio mZ_phi / mW_phi
    cos_theta_provider : Callable[[float], float]
        Function returning cosθ_W(μ) for μ in GeV
    bracket : (float, float)
        Initial bracket [μ_lo, μ_hi] in GeV
    tol : float
        Absolute tolerance for μ
    max_iter : int
        Maximum iterations

    Returns
    -------
    float
        μ⋆ in GeV
    """
    lo, hi = bracket
    f_lo = mZ_over_mW_phi * cos_theta_provider(lo) - 1.0
    f_hi = mZ_over_mW_phi * cos_theta_provider(hi) - 1.0

    if f_lo == 0.0:
        return lo
    if f_hi == 0.0:
        return hi

    # Ensure we have a sign change; if not, expand bracket
    expand = 0
    while f_lo * f_hi > 0.0 and expand < 20:
        lo = max(1.0, lo * 0.8)
        hi = hi * 1.2
        f_lo = mZ_over_mW_phi * cos_theta_provider(lo) - 1.0
        f_hi = mZ_over_mW_phi * cos_theta_provider(hi) - 1.0
        expand += 1

    if f_lo * f_hi > 0.0:
        # Monotone scenario: choose the closer end
        return lo if abs(f_lo) < abs(f_hi) else hi

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = mZ_over_mW_phi * cos_theta_provider(mid) - 1.0
        if abs(f_mid) < 1e-14 or (hi - lo) < tol:
            return mid
        if f_lo * f_mid <= 0.0:
            hi = mid
            f_hi = f_mid
        else:
            lo = mid
            f_lo = f_mid

    return 0.5 * (lo + hi)


def recognition_energy_eV() -> float:
    """Compute E_rec = ħ c / λ_rec with λ_rec = sqrt(ħ G / (π c^3)).

    Returns energy in eV.
    """
    # Constants (SI)
    hbar_SI = 1.054571817e-34  # J*s
    c_SI = 299792458.0  # m/s
    G_SI = 6.67430e-11  # m^3 kg^-1 s^-2

    # λ_rec in meters
    lambda_rec = math.sqrt(hbar_SI * G_SI / (math.pi * c_SI**3))

    # ħ c in eV*m: 197.3269804 MeV*fm = 1.973269804e-7 eV*m
    hbar_c_eVm = 1.973269804e-7
    E_rec_eV = hbar_c_eVm / lambda_rec
    return E_rec_eV


def derive_scale_from_ZW(mW_phi: float,
                         mZ_phi: float,
                         tilt_mode: str = 'RS') -> Tuple[float, float, float]:
    """Derive (s_eV, mu_star_GeV, s_over_Erec) from Z/W consistency.

    Parameters
    ----------
    mW_phi : float
        Dimensionless W ladder output
    mZ_phi : float
        Dimensionless Z ladder output
    tilt_mode : str
        'SM' for 1-loop SM running, 'RS' for RS placeholder tilt

    Returns
    -------
    (s_eV, mu_star_GeV, s_over_Erec) : Tuple[float, float, float]
    """
    ratio = mZ_phi / mW_phi
    if tilt_mode.upper() == 'RS':
        provider = _cos_rs_force if _cos_rs_force is not None else cos_theta_w_rs
    else:
        provider = cos_theta_w_sm

    mu_star = solve_mu_star(ratio, provider)
    # s in eV per ladder unit
    s_eV = mu_star * 1.0e9 / mW_phi
    E_rec = recognition_energy_eV()
    return s_eV, mu_star, (s_eV / E_rec)


__all__ = [
    'cos_theta_w_sm',
    'cos_theta_w_rs',
    'solve_mu_star',
    'recognition_energy_eV',
    'derive_scale_from_ZW',
]


