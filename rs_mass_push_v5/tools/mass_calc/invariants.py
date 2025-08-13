
"""
Charged-lepton invariants (ledger-locked, parameter-free).

- I_1 = 4  (Y_R^2, universal)
- I_2 = 9/76  (w_L · T(T+1), universal; T=1/2, w_L=3/19)

Species separation comes from the ledger's 8-beat chiral square wave. In the
closed-form treatment we do not inject a truncated odd-harmonic set here; the
odd-harmonic contribution is resummed in the mass engine using the rung phase.

For backward-compatibility, a toggle allows inclusion of the first few odd
harmonics. By default we exclude them to avoid double counting when the
closed-form chiral term is enabled in the mass solver.
"""
from __future__ import annotations
from typing import Dict, Optional
import math

def invariants_lepton_charged(
    r_i: Optional[int] = None,
    include_left: bool = True,
    include_chiral_harmonics: bool = False,
) -> Dict[int, float]:
    """
    Return the universal invariants for charged leptons.

    - include_left: include the SU(2)_L mixing invariant I_2 (default True)
    - include_chiral_harmonics: when True, include the first odd-harmonic
      projections (m=3,5,7). For the preferred closed-form treatment of the
      chiral series, leave this False and let the mass engine add the resummed
      contribution using r_i.
    """
    inv: Dict[int, float] = {}
    inv[1] = 4.0
    if include_left:
        inv[2] = 9.0 / 76.0
    if include_chiral_harmonics and r_i is not None:
        theta = (math.pi / 4.0) * (int(r_i) % 8)
        inv[3] = (4.0 / (3.0 * math.pi)) * math.cos(3.0 * theta)
        inv[5] = (4.0 / (5.0 * math.pi)) * math.cos(5.0 * theta)
        inv[7] = (4.0 / (7.0 * math.pi)) * math.cos(7.0 * theta)
    return inv
