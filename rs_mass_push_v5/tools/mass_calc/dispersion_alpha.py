
from __future__ import annotations
import math
from typing import Callable
from functools import lru_cache

# Constants
ALPHA_EM_0: float = 1.0 / 137.035999084
MZ: float = 91.1876

# Light-hadron mass scales (GeV)
M_PI: float = 0.13957
M_K: float = 0.493677
M_D_MESON: float = 1.86484
M_B_MESON: float = 5.279
M_RHO: float = 0.77526

# Hadronic thresholds
THRESH_UD: float = M_RHO
THRESH_S: float = 2.0 * M_K
THRESH_C: float = 2.0 * M_D_MESON
THRESH_B: float = 2.0 * M_B_MESON

# --- Minimal α_s(μ) (1-loop) ---
ALPHA_S_MZ: float = 0.1181
def _beta0(nf: int) -> float: return 11.0 - (2.0 / 3.0) * nf

def alpha_s_1loop(mu: float, mu0: float = MZ, alpha0: float = ALPHA_S_MZ) -> float:
    if mu <= 0.0 or mu0 <= 0.0:
        raise ValueError("Scales must be positive")
    thresholds = [1.27, 4.18]
    direction = 1 if mu > mu0 else -1
    start = mu0
    inv = 1.0 / alpha0

    def nf_at(scale: float) -> int:
        nf = 3
        if scale > thresholds[0]: nf += 1
        if scale > thresholds[1]: nf += 1
        return nf

    if direction > 0:
        points = [x for x in thresholds if start < x < mu]
        segments = [start] + points + [mu]
    else:
        points = [x for x in thresholds if mu < x < start]
        segments = [start] + points[::-1] + [mu]

    for s0, s1 in zip(segments[:-1], segments[1:]):
        ref = min(s0, s1)
        inv = inv + (_beta0(nf_at(ref)) / (2.0 * math.pi)) * math.log(s1 / s0)

    return 1.0 if inv <= 0.0 else 1.0 / inv

# --- R(s) model: resonances + continuum table (PDG-style spirit) ---
def _n_eff_quark_partons(u: bool, d: bool, s: bool, c: bool, b: bool) -> float:
    n_eff = 0.0
    if u: n_eff += 4.0 / 3.0
    if c: n_eff += 4.0 / 3.0
    if d: n_eff += 1.0 / 3.0
    if s: n_eff += 1.0 / 3.0
    if b: n_eff += 1.0 / 3.0
    return n_eff

def _R_continuum_table(E: float) -> float:
    SEG = [
        (0.81, 0.95, 1.90),
        (0.95, 1.05, 2.05),
        (1.05, 1.20, 2.12),
        (1.20, 1.40, 2.16),
        (1.40, 1.60, 2.20),
        (1.60, 1.80, 2.22),
        (1.80, 2.00, 2.24),
        (2.00, 2.20, 2.26),
        (2.20, 2.40, 2.28),
        (2.40, 2.60, 2.29),
        (2.60, 2.80, 2.30),
        (2.80, 3.00, 2.31),
        (3.00, 3.10, 2.32),
        (3.10, 3.60, 2.40),
        (3.80, 4.20, 3.45),
        (4.20, 5.00, 3.55),
        (5.00, 9.00, 3.60),
        (9.00, 10.30, 3.68),
        (10.30, 10.80, 3.70),
        (10.80, 12.00, 3.72),
        (12.00, 200.0, 3.75),
    ]
    for lo, hi, Ravg in SEG:
        if lo <= E < hi: return Ravg
    return 0.0

def _in_res(E: float, m: float, G: float, nwidth: float=2.5) -> bool:
    return abs(E - m) <= nwidth * G

def R_ratio_pdg(s: float) -> float:
    if s <= 0.0: return 0.0
    E = math.sqrt(s)
    RES = [
        (0.77526, 0.1491, 7.04e-6, 0.999),
        (0.78265, 0.00849, 0.60e-6, 0.89),
        (1.01946, 0.004249, 1.27e-6, 0.84),
        (3.0969, 9.3e-5, 5.55e-6, 0.87),
        (3.6861, 2.96e-4, 2.34e-6, 0.77),
        (3.773, 0.0272, 2.62e-7, 0.99),
        (4.040, 0.080, 8.6e-7, 0.98),
        (4.160, 0.070, 8.3e-7, 0.98),
        (4.415, 0.062, 5.8e-7, 0.98),
        (9.4603, 5.4e-5, 1.34e-6, 0.95),
        (10.0233, 3.1e-5, 0.61e-6, 0.95),
        (10.3552, 2.0e-5, 0.44e-6, 0.95),
        (10.5794, 0.0205, 2.72e-7, 0.96),
    ]
    # Continuum excluding resonance windows
    cont = 0.0 if any(_in_res(E, m, G) for (m,G,_,_) in RES) else _R_continuum_table(E)
    # Breit–Wigner resonances
    alpha0 = ALPHA_EM_0
    resonances = 0.0
    for m, Gtot, Gee, BRhad in RES:
        Ghad = BRhad * Gtot
        denom = ((s - m*m)**2) + (m*m*Gtot*Gtot)
        resonances += (9.0 * Gee * Ghad) / (alpha0 * alpha0 * denom)
    return cont + resonances

# --- Adler-function pQCD tail anchored at s0 ---
_adler_cache: dict[tuple[float,float], float] = {}
def _n_eff_quark_partons_bool(u,d,s,c,b): return _n_eff_quark_partons(u,d,s,c,b)

def _R_pqcd(s: float) -> float:
    if s <= 0.0: return 0.0
    E = math.sqrt(s)
    u = E >= THRESH_UD; d = E >= THRESH_UD; s_act = E >= THRESH_S
    c = E >= THRESH_C; b = E >= THRESH_B
    n_parton = _n_eff_quark_partons_bool(u, d, s_act, c, b)
    if n_parton == 0.0: return 0.0
    a_s = alpha_s_1loop(E); x = a_s / math.pi
    r_corr = 1.0 + x + 1.409 * (x ** 2)
    return n_parton * r_corr

def delta_alpha_had_adler(Q2: float, s0: float = (2.5 ** 2), delta_at_s0: float = 0.007541) -> float:
    if Q2 <= 0.0: return 0.0
    key = (round(Q2,6), round(s0,6))
    if key in _adler_cache: return _adler_cache[key]
    s_max = (200.0 ** 2)
    t0, t1 = math.log(s0), math.log(s_max)
    N = 2400
    h = (t1 - t0) / N
    tot_Q = 0.0; tot_s0 = 0.0
    for k in range(N + 1):
        t = t0 + k * h; s = math.exp(t)
        w = 0.5 if (k==0 or k==N) else 1.0
        R = _R_pqcd(s)
        tot_Q  += w * (R / (s * (s + Q2)))
        tot_s0 += w * (R / (s * (s + s0)))
    diff = -(ALPHA_EM_0 / (3.0 * math.pi)) * (Q2 * h * tot_Q - s0 * h * tot_s0)
    val = max(0.0, delta_at_s0 + diff)
    _adler_cache[key] = val
    return val

# --- Dispersion integral with targeted densification in 1.2–2.5 GeV ---
_cache_disp: dict[float, float] = {}

def delta_alpha_had_dispersion(Q2: float, R: Callable[[float], float] = R_ratio_pdg) -> float:
    if Q2 <= 0.0: return 0.0
    key = round(math.log(Q2), 4)
    if key in _cache_disp: return _cache_disp[key]

    s_min = 4.0 * (M_PI ** 2)
    s_max = (200.0 ** 2)
    t_min, t_max = math.log(s_min), math.log(s_max)

    # Region boundaries (s-space) for densification
    s1 = (1.20 ** 2)
    s2 = (2.50 ** 2)

    def integrate_log_trap(t0: float, t1: float, N: int) -> float:
        h = (t1 - t0) / N
        tot = 0.0
        for k in range(N + 1):
            t = t0 + k * h
            s = math.exp(t)
            w = 0.5 if (k == 0 or k == N) else 1.0
            tot += w * (R(s) / (s + Q2))
        return h * tot

    # Piecewise integration: base → dense (τ window) → base
    T0, T1, T2, T3 = t_min, math.log(s1), math.log(s2), t_max
    # Densities: increase only in the sensitive 1.2–2.5 GeV window.
    I1 = integrate_log_trap(T0, T1, 1000)    # below 1.2 GeV (slightly higher)
    I2 = integrate_log_trap(T1, T2, 3200)    # denser in 1.2–2.5 GeV
    I3 = integrate_log_trap(T2, T3, 1400)    # above 2.5 GeV (slightly higher)
    integral = I1 + I2 + I3

    delta = -(ALPHA_EM_0 * Q2 / (3.0 * math.pi)) * integral
    if delta < 0.0: delta = 0.0
    _cache_disp[key] = delta
    return delta

@lru_cache(maxsize=4096)
def alpha_em_vp_dispersion(mu: float) -> float:
    if mu <= 0.0: raise ValueError("mu must be positive")
    def delta_alpha_leptons(mass: float) -> float:
        return (ALPHA_EM_0 / (3.0 * math.pi)) * (math.log((mu * mu) / (mass * mass)) - 5.0 / 3.0) if mu > mass else 0.0
    M_E = 0.000511; M_MU = 0.10566; M_TAU = 1.77686
    da_lept = delta_alpha_leptons(M_E) + delta_alpha_leptons(M_MU) + delta_alpha_leptons(M_TAU)

    Q2 = mu * mu
    da_had = delta_alpha_had_adler(Q2) if Q2 >= (2.5 ** 2) else delta_alpha_had_dispersion(Q2)

    # Top (on-shell)
    M_T = 172.76; Q_t = 2.0 / 3.0
    da_top = 0.0
    if mu > M_T:
        da_top = (ALPHA_EM_0 * (Q_t ** 2) / (3.0 * math.pi)) * (math.log((mu * mu) / (M_T * M_T)) - 5.0 / 3.0)

    denom = 1.0 - (da_lept + da_had + da_top)
    return ALPHA_EM_0 / denom
