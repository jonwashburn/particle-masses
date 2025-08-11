"""
Parameter-Free Particle Masses Calculator
==========================================

A parameter-free architecture for calculating Standard Model particle masses
from first principles using φ-sheet fixed points.

Author: Jonathan Washburn
Email: washburn@recognitionphysics.org
"""

from .mass_calc import (
    PHI,
    compute_f_i_phi_sheet,
    solve_mass_fixed_point_phi_sheet,
    mass_dimensionless,
    ParticleMassCalculator
)

from .invariants import (
    invariants_lepton_charged,
    chiral_occupancy
)

from .sm_rge import (
    gamma_e,
    gamma_mu,
    gamma_tau,
    gamma_e_sm,
    gamma_mu_sm,
    gamma_tau_sm,
    alpha_em_dispersion
)

__version__ = "1.0.0"
__author__ = "Jonathan Washburn"

__all__ = [
    'PHI',
    'ParticleMassCalculator',
    'compute_f_i_phi_sheet',
    'solve_mass_fixed_point_phi_sheet',
    'mass_dimensionless',
    'invariants_lepton_charged',
    'chiral_occupancy',
    'gamma_e',
    'gamma_mu',
    'gamma_tau',
    'alpha_em_dispersion'
]
