"""
Unit tests for mass calculation module.
"""

import pytest
import math
from src.mass_calc import PHI, E_COH, mass_dimensionless, compute_f_i_phi_sheet


def test_golden_ratio():
    """Test that PHI is correctly defined."""
    expected = (1 + math.sqrt(5)) / 2
    assert abs(PHI - expected) < 1e-10
    assert abs(PHI**2 - PHI - 1) < 1e-10  # Golden ratio property


def test_coherence_quantum():
    """Test coherence quantum value."""
    expected = PHI**(-5)
    assert abs(E_COH - expected) < 1e-10
    assert abs(E_COH - 0.09016994) < 1e-6  # Approximate value


def test_mass_dimensionless():
    """Test basic mass formula."""
    # Ground state: r=0, f=0
    m = mass_dimensionless(B_i=1, E_coh=E_COH, r_i=0, f_i=0)
    assert abs(m - E_COH) < 1e-10
    
    # First rung: r=1, f=0
    m = mass_dimensionless(B_i=1, E_coh=E_COH, r_i=1, f_i=0)
    assert abs(m - E_COH * PHI) < 1e-10
    
    # With sector factor B=2
    m = mass_dimensionless(B_i=2, E_coh=E_COH, r_i=0, f_i=0)
    assert abs(m - 2 * E_COH) < 1e-10


def test_compute_f_i_phi_sheet():
    """Test φ-sheet correction calculation."""
    # Simple test with zero anomalous dimension
    gamma_zero = lambda mu: 0.0
    invariants = {1: 0.0, 2: 0.0}
    
    f = compute_f_i_phi_sheet(ln_m=0.0, gamma_func=gamma_zero, invariants=invariants)
    assert abs(f) < 1e-6  # Should be nearly zero
    
    # With non-zero invariants
    invariants = {1: 4.0, 2: 9/76}  # Typical lepton values
    f = compute_f_i_phi_sheet(ln_m=0.0, gamma_func=gamma_zero, invariants=invariants)
    assert f != 0.0  # Should have non-zero correction


def test_lepton_ratio_bounds():
    """Test that lepton ratios are in reasonable ranges."""
    from src.mass_calc import ParticleMassCalculator
    
    calc = ParticleMassCalculator()
    ratios = calc.lepton_ratios()
    
    # Check rough bounds (within factor of 2 of experimental values)
    assert 100 < ratios['mu_e'] < 400  # Exp: ~207
    assert 10 < ratios['tau_mu'] < 30   # Exp: ~17
    assert 2000 < ratios['tau_e'] < 5000  # Exp: ~3477


def test_neutrino_hierarchy():
    """Test that neutrino masses follow normal ordering."""
    from src.mass_calc import ParticleMassCalculator
    
    calc = ParticleMassCalculator()
    nu_masses = calc.neutrino_masses()
    
    # Normal ordering: m1 < m2 < m3
    assert nu_masses['m1'] < nu_masses['m2']
    assert nu_masses['m2'] < nu_masses['m3']
    
    # Check sum is in cosmologically allowed range
    assert 0.01 < nu_masses['sum'] < 0.3  # eV
    
    # Check mass-squared differences are positive
    assert nu_masses['dm21_squared'] > 0
    assert nu_masses['dm31_squared'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
