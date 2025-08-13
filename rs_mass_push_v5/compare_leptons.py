import math
from tools.mass_calc.mass_calc import (
    compute_f_i,
    compute_f_i_local_cycle,
    compute_f_i_phi_sheet,
    solve_mass_fixed_point,
    solve_mass_fixed_point_phi_sheet,
    mass_dimensionless,
    PHI,
    _chiral_gap_closed_form,
)
from tools.mass_calc.sm_rge import (
    gamma_e,
    gamma_mu,
    gamma_tau,
    gamma_e_sm,
    gamma_mu_sm,
    gamma_tau_sm,
    MZ,
)
from tools.mass_calc.invariants import invariants_lepton_charged

# Reference lepton pole masses (GeV)
M_EXP = {
    'e': 0.00051099895,
    'mu': 0.1056583755,
    'tau': 1.77686,
}

# Candidate integer r assignments (from phi-ladder):
# Differences chosen to approximate observed mass ratios
# mu/e ~ 206.768 ~ phi^11 (≈199) with small correction
# tau/mu ~ 16.816 ~ phi^6 (≈17.94) with small correction
R_ASSIGN = {
    'e': 0,
    'mu': 11,
    'tau': 17,
}

B_ASSIGN = {'e': 1, 'mu': 1, 'tau': 1}
# Combine QED and SM electroweak/Yukawa by simple addition (1-loop-level sum)
def gamma_e_total(mu: float) -> float:
    return gamma_e(mu) + gamma_e_sm(mu)


def gamma_mu_total(mu: float) -> float:
    return gamma_mu(mu) + gamma_mu_sm(mu)


def gamma_tau_total(mu: float) -> float:
    return gamma_tau(mu) + gamma_tau_sm(mu)


GAMMA = {'e': gamma_e_total, 'mu': gamma_mu_total, 'tau': gamma_tau_total}

E_COH = PHI ** (-5)
I_series = invariants_lepton_charged()  # default; overridden per-species below
ln_mu_star = math.log(MZ)


def predict_dimensionless(name: str) -> dict:
    m_guess = M_EXP[name]
    ln_m_guess = math.log(m_guess)
    # Species-specific invariants: inject rung-sensitive I_chi via r_i
    # Use universal invariants only; chiral odd-harmonics are added in closed form
    I_spec = invariants_lepton_charged(r_i=R_ASSIGN[name], include_left=True, include_chiral_harmonics=False)
    # Solve mass via φ-sheet fixed point (amplifies species differences without parameters)
    ln_m_star = solve_mass_fixed_point_phi_sheet(
        B_i=B_ASSIGN[name],
        E_coh=E_COH,
        r_i=R_ASSIGN[name],
        gamma_func=GAMMA[name],
        invariant_by_m=I_spec,
        ln_m_init=ln_m_guess,
        phase_offset=0,
    )
    f_phi_base = compute_f_i_phi_sheet(ln_m_star, GAMMA[name], I_spec)
    # Add closed-form chiral contribution to yield the total fractional residue
    theta = (math.pi / 4.0) * (R_ASSIGN[name] % 8)
    delta_f_chi = _chiral_gap_closed_form(theta)
    f_total = f_phi_base + delta_f_chi
    m_dimless = mass_dimensionless(B_ASSIGN[name], E_COH, R_ASSIGN[name], f_total)
    return {'f_i_sheet': f_phi_base, 'f_i_total': f_total, 'm_dimless': m_dimless, 'ln_m_star': ln_m_star}


def main():
    preds = {n: predict_dimensionless(n) for n in ['e', 'mu', 'tau']}
    # Ratios (dimensionless cancel E_coh)
    ratio_mu_e = preds['mu']['m_dimless'] / preds['e']['m_dimless']
    ratio_tau_mu = preds['tau']['m_dimless'] / preds['mu']['m_dimless']
    ratio_tau_e = preds['tau']['m_dimless'] / preds['e']['m_dimless']

    exp_mu_e = M_EXP['mu'] / M_EXP['e']
    exp_tau_mu = M_EXP['tau'] / M_EXP['mu']
    exp_tau_e = M_EXP['tau'] / M_EXP['e']

    def pct(x: float) -> float:
        return 100.0 * x

    print("Mass table (dimensionless), f_i, and ratios vs. experiment:")
    for n in ['e','mu','tau']:
        print(f"{n}: m_dimless={preds[n]['m_dimless']:.9f}, f_i_total={preds[n]['f_i_total']:.9f}, f_i_sheet={preds[n]['f_i_sheet']:.9f}")
    print("ratios:")
    print(f"mu/e: pred={ratio_mu_e:.7f}, exp={exp_mu_e:.7f}, delta%={pct((ratio_mu_e-exp_mu_e)/exp_mu_e):+.4f}")
    print(f"tau/mu: pred={ratio_tau_mu:.7f}, exp={exp_tau_mu:.7f}, delta%={pct((ratio_tau_mu-exp_tau_mu)/exp_tau_mu):+.4f}")
    print(f"tau/e: pred={ratio_tau_e:.7f}, exp={exp_tau_e:.7f}, delta%={pct((ratio_tau_e-exp_tau_e)/exp_tau_e):+.4f}")


if __name__ == '__main__':
    main()

