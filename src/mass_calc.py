"""
Core mass calculation module for particle masses.

This module implements the φ-sheet fixed point solver and mass formula:
m_i = B_i × E_coh × φ^(r_i + f_i)
"""

import math
import numpy as np
from typing import Dict, Callable, Tuple, Optional

# Golden ratio
PHI = (1 + math.sqrt(5)) / 2

# Coherence quantum
E_COH = PHI**(-5)


def mass_dimensionless(B_i: float, E_coh: float, r_i: int, f_i: float) -> float:
    """
    Calculate dimensionless mass from the fundamental formula.
    
    Parameters
    ----------
    B_i : float
        Sector factor (1 for leptons, 2 for quarks, etc.)
    E_coh : float
        Coherence quantum (φ^-5)
    r_i : int
        Integer rung position on φ-ladder
    f_i : float
        φ-sheet fixed point correction
    
    Returns
    -------
    float
        Dimensionless mass value
    """
    return B_i * E_coh * PHI**(r_i + f_i)


def compute_f_i_phi_sheet(
    ln_m: float,
    gamma_func: Callable[[float], float],
    invariants: Dict[int, float],
    max_k: int = 50,
    epsilon_sheet: float = 1e-10
) -> float:
    """
    Compute the φ-sheet correction f_i.
    
    Parameters
    ----------
    ln_m : float
        Natural log of mass at fixed point
    gamma_func : Callable
        Anomalous dimension function γ(μ)
    invariants : Dict[int, float]
        Ledger invariants {1: I_1, 2: I_2, ...}
    max_k : int
        Maximum number of sheet windows
    epsilon_sheet : float
        Truncation threshold for sheet sum
    
    Returns
    -------
    float
        φ-sheet correction f_i
    """
    ln_phi = math.log(PHI)
    
    # Sheet average of anomalous dimension
    sheet_sum = 0.0
    normalizer = 2 * ln_phi  # Closed form: Σ(1/(m*φ^m)) = 2ln(φ)
    
    for k in range(max_k):
        # Weight w_k ∝ g_{k+1} with alternating signs
        g_kp1 = ((-1)**(k+2)) / ((k+1) * PHI**(k+1))
        w_k = g_kp1 / normalizer
        
        # Window integral from ln(m) to ln(φm)
        mu_base = math.exp(ln_m) * PHI**k
        window_integral = 0.0
        
        # Simple trapezoidal integration
        n_points = 20
        for i in range(n_points):
            t = i / (n_points - 1)
            ln_mu = ln_m + t * ln_phi + k * ln_phi
            mu = math.exp(ln_mu)
            window_integral += gamma_func(mu) * ln_phi / n_points
        
        sheet_sum += w_k * window_integral
        
        # Check truncation
        tail_bound = (PHI**2) / (2 * ln_phi * k * PHI**k) if k > 0 else 1.0
        if tail_bound < epsilon_sheet:
            break
    
    # Add invariant contributions
    gap_sum = 0.0
    for m, I_m in invariants.items():
        g_m = ((-1)**(m+1)) / (m * PHI**m)
        gap_sum += g_m * I_m
    
    return sheet_sum / ln_phi + gap_sum


def solve_mass_fixed_point_phi_sheet(
    B_i: float,
    E_coh: float,
    r_i: int,
    gamma_func: Callable[[float], float],
    invariants: Dict[int, float],
    ln_m_init: float = 0.0,
    tol: float = 1e-10,
    max_iter: int = 100
) -> float:
    """
    Solve for the mass fixed point with φ-sheet averaging.
    
    The fixed point equation is:
    ln(m) = ln(B_i × E_coh) + r_i × ln(φ) + f_i(ln m) × ln(φ)
    
    Parameters
    ----------
    B_i : float
        Sector factor
    E_coh : float
        Coherence quantum
    r_i : int
        Rung number
    gamma_func : Callable
        Anomalous dimension function
    invariants : Dict[int, float]
        Ledger invariants
    ln_m_init : float
        Initial guess for ln(m)
    tol : float
        Convergence tolerance
    max_iter : int
        Maximum iterations
    
    Returns
    -------
    float
        Natural log of mass at fixed point
    """
    ln_phi = math.log(PHI)
    ln_base = math.log(B_i * E_coh) + r_i * ln_phi
    
    ln_m = ln_m_init if ln_m_init != 0.0 else ln_base
    
    for iteration in range(max_iter):
        f_i = compute_f_i_phi_sheet(ln_m, gamma_func, invariants)
        ln_m_new = ln_base + f_i * ln_phi
        
        if abs(ln_m_new - ln_m) < tol:
            return ln_m_new
        
        ln_m = ln_m_new
    
    raise ValueError(f"Fixed point did not converge after {max_iter} iterations")


class ParticleMassCalculator:
    """
    Main calculator class for particle masses.
    """
    
    def __init__(self):
        """Initialize with standard parameters."""
        self.PHI = PHI
        self.E_COH = E_COH
        
        # Rung assignments
        self.R_LEPTON = {'e': 0, 'mu': 11, 'tau': 17}
        self.R_NEUTRINO = {'nu1': 7, 'nu2': 9, 'nu3': 12}
        
        # PMNS matrix elements (NuFIT 5.2)
        self.U_e = (0.8252, 0.5482, 0.1343)
    
    def lepton_ratios(self) -> Dict[str, float]:
        """
        Calculate charged lepton mass ratios.
        
        Returns
        -------
        Dict[str, float]
            Dictionary with keys 'mu_e', 'tau_mu', 'tau_e'
        """
        from .invariants import invariants_lepton_charged
        from .sm_rge import gamma_e_total, gamma_mu_total, gamma_tau_total
        
        B = 1  # Lepton sector factor
        
        # Calculate dimensionless masses
        m_dimless = {}
        for name, r in self.R_LEPTON.items():
            invariants = invariants_lepton_charged(r)
            gamma_func = {'e': gamma_e_total, 'mu': gamma_mu_total, 'tau': gamma_tau_total}[name]
            
            ln_m = solve_mass_fixed_point_phi_sheet(B, self.E_COH, r, gamma_func, invariants)
            f_i = compute_f_i_phi_sheet(ln_m, gamma_func, invariants)
            m_dimless[name] = mass_dimensionless(B, self.E_COH, r, f_i)
        
        return {
            'mu_e': m_dimless['mu'] / m_dimless['e'],
            'tau_mu': m_dimless['tau'] / m_dimless['mu'],
            'tau_e': m_dimless['tau'] / m_dimless['e']
        }
    
    def neutrino_masses(self, dm31_squared: float = 2.44e-3) -> Dict[str, float]:
        """
        Calculate absolute neutrino masses (Dirac, normal ordering).
        
        Parameters
        ----------
        dm31_squared : float
            Atmospheric mass-squared difference in eV^2
        
        Returns
        -------
        Dict[str, float]
            Neutrino masses and derived quantities in eV
        """
        from .invariants import invariants_neutrino
        
        B = 1
        gamma_nu = lambda mu: 0.0  # Neutrinos have negligible running
        
        # Calculate dimensionless masses
        m_hat = {}
        for name, r in self.R_NEUTRINO.items():
            invariants = invariants_neutrino(r)
            ln_m = solve_mass_fixed_point_phi_sheet(B, self.E_COH, r, gamma_nu, invariants)
            f_i = compute_f_i_phi_sheet(ln_m, gamma_nu, invariants)
            m_hat[name] = mass_dimensionless(B, self.E_COH, r, f_i)
        
        # Fix absolute scale from atmospheric splitting
        dm31_hat = m_hat['nu3']**2 - m_hat['nu1']**2
        s = math.sqrt(dm31_squared / dm31_hat)
        
        # Absolute masses
        m1 = s * m_hat['nu1']
        m2 = s * m_hat['nu2']
        m3 = s * m_hat['nu3']
        
        # Derived quantities
        sum_m = m1 + m2 + m3
        dm21_squared = m2**2 - m1**2
        
        # Effective mass for beta decay
        m_beta = math.sqrt(
            self.U_e[0]**2 * m1**2 +
            self.U_e[1]**2 * m2**2 +
            self.U_e[2]**2 * m3**2
        )
        
        return {
            'm1': m1,
            'm2': m2,
            'm3': m3,
            'sum': sum_m,
            'dm21_squared': dm21_squared,
            'dm31_squared': dm31_squared,
            'm_beta': m_beta
        }
    
    def absolute_masses(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate all absolute masses with neutrino anchoring.
        
        Returns
        -------
        Dict[str, Dict[str, float]]
            Nested dictionary with particle types and their masses
        """
        # Get neutrino scale
        nu_results = self.neutrino_masses()
        
        # This would require the full implementation of charged lepton
        # absolute masses using the same scale factor
        
        return {
            'neutrinos': nu_results,
            # 'leptons': lepton_results,
            # 'quarks': quark_results,
            # 'bosons': boson_results
        }
