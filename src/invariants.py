"""
Ledger invariants module.

Implements the fixed, parameter-free invariants I_m(i) that enter
the φ-sheet fixed point calculation.
"""

from typing import Dict


def chiral_occupancy(r: int) -> float:
    """
    Calculate the 8-beat chiral occupancy Δf_χ(r).
    
    This is a closed-form expression that depends only on the
    rung class r mod 8.
    
    Parameters
    ----------
    r : int
        Rung number
    
    Returns
    -------
    float
        Chiral occupancy correction
    """
    return ((r % 8) - 4) / 8.0


def invariants_lepton_charged(r_i: int) -> Dict[int, float]:
    """
    Calculate invariants for charged leptons.
    
    Parameters
    ----------
    r_i : int
        Rung number for the lepton species
    
    Returns
    -------
    Dict[int, float]
        Dictionary of invariants {1: I_1, 2: I_2}
    """
    # Right-chiral block
    Y_R_squared = 4.0  # Hypercharge squared
    I_1 = Y_R_squared + chiral_occupancy(r_i)
    
    # Left-chiral SU(2) block
    w_L = 3.0 / 19.0  # Fixed SU(2) mixing weight
    T = 0.5  # Weak isospin
    I_2 = w_L * T * (T + 1)  # = 9/76
    
    return {1: I_1, 2: I_2}


def invariants_neutrino(r: int) -> Dict[int, float]:
    """
    Calculate invariants for neutrinos.
    
    Parameters
    ----------
    r : int
        Rung number for the neutrino species
    
    Returns
    -------
    Dict[int, float]
        Dictionary of invariants
    """
    # Neutrino-specific invariants
    I_1 = chiral_occupancy(r)
    I_2 = 9.0 / 76.0  # Same SU(2) contribution as charged leptons
    
    return {1: I_1, 2: I_2}


def invariants_quark(r: int, quark_type: str) -> Dict[int, float]:
    """
    Calculate invariants for quarks.
    
    Parameters
    ----------
    r : int
        Rung number
    quark_type : str
        'up' or 'down' type
    
    Returns
    -------
    Dict[int, float]
        Dictionary of invariants
    """
    # Quark hypercharges differ for up/down types
    if quark_type == 'up':
        Y_R_squared = (4.0/3.0)**2  # Up-type hypercharge
    else:
        Y_R_squared = (2.0/3.0)**2  # Down-type hypercharge
    
    I_1 = Y_R_squared + chiral_occupancy(r)
    I_2 = 9.0 / 76.0
    
    # Additional color factor modifications would go here
    
    return {1: I_1, 2: I_2}
