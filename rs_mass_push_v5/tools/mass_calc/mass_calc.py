
from __future__ import annotations
import math
from typing import Callable, Dict, Optional

PHI: float = (1.0 + 5 ** 0.5) / 2.0
LN_PHI: float = math.log(PHI)

def g_m(m: int) -> float:
    """Gap coefficients for F(z)=ln(1+z/phi) → g_m = (-1)^(m+1)/(m φ^m)."""
    sign = 1 if ((m + 1) % 2 == 0) else -1
    return sign * (1.0 / (m * (PHI ** m)))

def _integrate_gamma_over_window(ln_base: float, gamma_func: Callable[[float], float], steps: int = 96) -> float:
    """∫_{ln μ}^{ln(φ μ)} γ(μ) d ln μ via composite Simpson's rule.

    ln_base is the anchor for the physical scale in GeV.
    """
    a = ln_base
    b = ln_base + LN_PHI
    if steps % 2 == 1:
        steps += 1
    h = (b - a) / steps
    total = 0.0
    for i in range(steps + 1):
        ln_mu = a + i * h
        mu = math.exp(ln_mu)
        w = 4.0 if i % 2 == 1 else 2.0
        if i == 0 or i == steps:
            w = 1.0
        total += w * gamma_func(mu)
    return (h / 3.0) * total

def compute_f_i_local_cycle(
    ln_m: float,
    gamma_func: Callable[[float], float],
    invariant_by_m: Dict[int, float],
    ln_mu_anchor: Optional[float] = None,
) -> float:
    """Single φ-window fractional residue: [∫ γ d ln μ]/ln φ + Σ_m g_m I_m.

    The γ(μ) integral is anchored at ln_mu_anchor (GeV) if provided; otherwise ln_m.
    """
    integral = _integrate_gamma_over_window(ln_mu_anchor if ln_mu_anchor is not None else ln_m, gamma_func)
    gap = sum(g_m(m) * I_m for m, I_m in invariant_by_m.items())
    return (integral / LN_PHI) + gap

def compute_f_i_phi_sheet(
    ln_m: float,
    gamma_func: Callable[[float], float],
    invariant_by_m: Dict[int, float],
    signed_weights: bool = False,
    tail_tol: float = 1e-6,
    max_K: int = 64,
    steps_per_window: int = 96,
    ln_mu_anchor: Optional[float] = None,
) -> float:
    """φ-sheet convolution with adaptive truncation of |g_{k+1}| L1-tail.

    f_i = (1/ln φ) * Σ_k w_k ∫_{ln m}^{ln φ m} γ(μ φ^k) d ln μ + Σ_m g_m I_m

    We use weights proportional to g_{k+1} (signed or absolute) normalized by L1.
    The sum stops when the remaining L1 mass of |g_{k+1}| is < tail_tol.
    """
    # Precompute L1 normalization and determine adaptive K
    weights_raw = []
    l1_total = 0.0
    for k in range(max_K):
        coeff = g_m(k + 1)
        weights_raw.append(coeff if signed_weights else abs(coeff))
        l1_total += abs(coeff)
        # early break if tail below tolerance
        # remaining tail after including up to k is:
        tail = sum(abs(g_m(j + 1)) for j in range(k + 1, max_K))
        if tail < tail_tol:
            K = k + 1
            break
    else:
        K = max_K

    l1_used = sum(abs(weights_raw[j]) for j in range(K))
    if l1_used == 0.0:
        window_weight = [0.0] * K
    else:
        window_weight = [weights_raw[j] / l1_used for j in range(K)]

    # Integrate over shifted windows anchored at physical ln μ
    integral_sum = 0.0
    for k in range(K):
        def gamma_shift(mu: float, k=k):
            return gamma_func(mu * (PHI ** k))
        ln_base = ln_mu_anchor if ln_mu_anchor is not None else ln_m
        integral_sum += window_weight[k] * _integrate_gamma_over_window(ln_base, gamma_shift, steps=steps_per_window)

    gap = sum(g_m(m) * I_m for m, I_m in invariant_by_m.items())
    return (integral_sum / LN_PHI) + gap


def _chiral_gap_closed_form(theta: float) -> float:
    """
    Closed-form resummation of the odd-harmonic chiral series contribution
    to the fractional residue, excluding m=1 (reserved for I_1=4).

    Δf_chi(r) = sum_{m odd ≥3} g_m * (4/(π m)) cos(m θ),
    where g_m = (-1)^{m+1}/(m φ^m) and θ = (π/4)(r mod 8).

    We compute this via a rapidly convergent odd-m series to numerical
    tolerance, avoiding external polylog dependencies. This is parameter-free
    and aligned with the ledger's 8-beat chiral series.
    """
    # Rapidly convergent direct sum over odd m, skipping m=1.
    # Converges like ~ φ^{-m}/m^2.
    total = 0.0
    # Choose a safe upper bound for convergence; stop on small increment.
    max_m = 801
    tol = 1e-12
    prev_total = 0.0
    for m in range(3, max_m, 2):
        gm = ((-1.0) ** (m + 1)) / (m * (PHI ** m))
        coeff = (4.0 / (math.pi * m))
        total += gm * coeff * math.cos(m * theta)
        if m > 7 and abs(total - prev_total) < tol:
            break
        prev_total = total
    return total

def compute_f_i(ln_m: float, gamma_func: Callable[[float], float], invariant_by_m: Dict[int, float], ln_mu_anchor: Optional[float] = None) -> float:
    return compute_f_i_local_cycle(ln_m, gamma_func, invariant_by_m, ln_mu_anchor=ln_mu_anchor)

def solve_mass_fixed_point_phi_sheet(
    B_i: float,
    E_coh: float,
    r_i: int,
    gamma_func: Callable[[float], float],
    invariant_by_m: Dict[int, float],
    ln_m_init: float,
    tol: float = 1e-12,
    max_iter: int = 240,
    phase_offset: int = 0,
    **phi_sheet_kwargs,
) -> float:
    """Solve ln m = ln(BE_coh) + r ln φ + f_i(ln m) ln φ, using φ-sheet f_i.

    The invariant_by_m set should contain only universal terms (I_1, I_2).
    The rung-dependent chiral odd-harmonics are resummed in closed form here
    and added to f_i to avoid truncation bias and double counting.
    """
    ln_BE = math.log(B_i * E_coh)
    ln_m = ln_m_init
    theta = (math.pi / 4.0) * ((int(r_i) + int(phase_offset)) % 8)
    delta_f_chi = _chiral_gap_closed_form(theta)
    for _ in range(max_iter):
        f_i = compute_f_i_phi_sheet(ln_m, gamma_func, invariant_by_m, **phi_sheet_kwargs)
        # Add closed-form chiral gap contribution
        ln_m_new = ln_BE + r_i * LN_PHI + (f_i + delta_f_chi) * LN_PHI
        if not math.isfinite(ln_m_new):
            ln_m_new = 0.5 * (ln_m + ln_BE + r_i * LN_PHI)
        if abs(ln_m_new - ln_m) < tol:
            return ln_m_new
        ln_m = 0.5 * ln_m + 0.5 * ln_m_new
    return ln_m

def solve_mass_fixed_point(
    B_i: float,
    E_coh: float,
    r_i: int,
    gamma_func: Callable[[float], float],
    invariant_by_m: Dict[int, float],
    ln_m_init: float,
    tol: float = 1e-12,
    max_iter: int = 240,
) -> float:
    ln_BE = math.log(B_i * E_coh)
    ln_m = ln_m_init
    for _ in range(max_iter):
        f_i = compute_f_i_local_cycle(ln_m, gamma_func, invariant_by_m)
        ln_m_new = ln_BE + r_i * LN_PHI + f_i * LN_PHI
        if not math.isfinite(ln_m_new):
            ln_m_new = 0.5 * (ln_m + ln_BE + r_i * LN_PHI)
        if abs(ln_m_new - ln_m) < tol:
            return ln_m_new
        ln_m = 0.5 * ln_m + 0.5 * ln_m_new
    return ln_m

def mass_dimensionless(B_i: float, E_coh: float, r_i: int, f_i: float) -> float:
    return B_i * E_coh * (PHI ** (r_i + f_i))
