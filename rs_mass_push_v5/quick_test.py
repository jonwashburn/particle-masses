#!/usr/bin/env python3
"""
Quick test of the improved mass calculation with reduced computational load.
"""

import math
from tools.mass_calc.mass_calc import (
    compute_f_i_phi_sheet,
    solve_mass_fixed_point_phi_sheet,
    mass_dimensionless,
    PHI,
    _chiral_gap_closed_form,
)
from tools.mass_calc.sm_rge import (
    gamma_lepton_qed,
    MZ,
)
from tools.mass_calc.invariants import invariants_lepton_charged

# Reference lepton pole masses (GeV)
M_EXP = {
    'e': 0.00051099895,
    'mu': 0.1056583755,
    'tau': 1.77686,
}

# Rung assignments
R_ASSIGN = {
    'e': 0,
    'mu': 11,
    'tau': 17,
}

B_ASSIGN = {'e': 1, 'mu': 1, 'tau': 1}
E_COH = PHI ** (-5)

def gamma_simple(mu: float) -> float:
    """Simplified gamma function for quick testing."""
    return gamma_lepton_qed(mu)

def quick_predict(name: str) -> dict:
    """Quick prediction with reduced precision for testing."""
    m_guess = M_EXP[name]
    ln_m_guess = math.log(m_guess)
    
    # Universal invariants only (no chiral harmonics to avoid double counting)
    I_spec = invariants_lepton_charged(r_i=R_ASSIGN[name], include_left=True, include_chiral_harmonics=False)
    
    # Test the closed-form chiral contribution separately
    theta = (math.pi / 4.0) * (R_ASSIGN[name] % 8)
    delta_f_chi = _chiral_gap_closed_form(theta)
    
    # Use reduced precision for quick test
    ln_m_star = solve_mass_fixed_point_phi_sheet(
        B_i=B_ASSIGN[name],
        E_coh=E_COH,
        r_i=R_ASSIGN[name],
        gamma_func=gamma_simple,
        invariant_by_m=I_spec,
        ln_m_init=ln_m_guess,
        max_K=16,  # Reduced from 64
        tail_tol=1e-4,  # Reduced precision
        steps_per_window=24,  # Reduced from 96
    )
    
    f_phi_base = compute_f_i_phi_sheet(
        ln_m_star, 
        gamma_simple, 
        I_spec,
        max_K=16,
        tail_tol=1e-4,
        steps_per_window=24
    )
    
    m_dimless = mass_dimensionless(B_ASSIGN[name], E_COH, R_ASSIGN[name], f_phi_base)
    
    return {
        'f_i_sheet': f_phi_base, 
        'delta_f_chi': delta_f_chi,
        'm_dimless': m_dimless, 
        'ln_m_star': ln_m_star,
        'theta_deg': math.degrees(theta)
    }

def main():
    print("Quick test with closed-form chiral resummation:")
    print("=" * 60)
    
    preds = {}
    for n in ['e', 'mu', 'tau']:
        print(f"Computing {n}...")
        preds[n] = quick_predict(n)
    
    print("\nResults:")
    print("-" * 60)
    for n in ['e', 'mu', 'tau']:
        p = preds[n]
        print(f"{n}: m_dimless={p['m_dimless']:.6f}, f_sheet={p['f_i_sheet']:.6f}")
        print(f"   theta={p['theta_deg']:.1f}°, delta_f_chi={p['delta_f_chi']:.6f}")
    
    # Ratios
    ratio_mu_e = preds['mu']['m_dimless'] / preds['e']['m_dimless']
    ratio_tau_mu = preds['tau']['m_dimless'] / preds['mu']['m_dimless']
    ratio_tau_e = preds['tau']['m_dimless'] / preds['e']['m_dimless']
    
    exp_mu_e = M_EXP['mu'] / M_EXP['e']
    exp_tau_mu = M_EXP['tau'] / M_EXP['mu']
    exp_tau_e = M_EXP['tau'] / M_EXP['e']
    
    print("\nMass ratios (quick test):")
    print("-" * 60)
    print(f"mu/e:   pred={ratio_mu_e:.4f}, exp={exp_mu_e:.4f}, delta%={100*(ratio_mu_e-exp_mu_e)/exp_mu_e:+.2f}")
    print(f"tau/mu: pred={ratio_tau_mu:.4f}, exp={exp_tau_mu:.4f}, delta%={100*(ratio_tau_mu-exp_tau_mu)/exp_tau_mu:+.2f}")
    print(f"tau/e:  pred={ratio_tau_e:.4f}, exp={exp_tau_e:.4f}, delta%={100*(ratio_tau_e-exp_tau_e)/exp_tau_e:+.2f}")
    
    print("\nNote: This is a quick test with reduced precision.")
    print("The closed-form chiral resummation is now active.")

if __name__ == '__main__':
    main()
