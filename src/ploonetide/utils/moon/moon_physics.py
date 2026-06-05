from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional, Literal

import numpy as np
from ploonetide.utils.constants import *
"""
ploonetide.physics  (deterministic core, prefix-updated, **heavily commented**)
==============================================================================

API QUICK REFERENCE
-------------------
Units:
  • SI for system inputs (kg, m). Densities returned in g/cc unless stated.
  • Moon masses in M⊕, moon radii in R⊕ where Fortney fits apply.

Constants:
  - G                             # [m^3 kg^-1 s^-2]
  - R_EARTH_M, M_EARTH_KG, M_SUN_KG

Fortney+07 mass–radius fits:
  - fortney_radius_ice_rock(moon_M_earth, moon_imf) -> R_earth
  - fortney_radius_rock_iron(moon_M_earth, moon_rmf) -> R_earth

Composition helpers (no RNG):
  - moon_imf_from(moon_f_ice, moon_f_rock) -> IMF in [0,1]
  - moon_rmf_from(moon_f_rock, moon_f_iron) -> RMF in [0,1]
  - moon_density_zero_porosity(moon_f_ice, moon_f_rock, moon_f_iron) -> rho0_gcc

Pressure & compaction:
  - moon_central_pressure_pa(moon_density, moon_radius) -> Pc [Pa]
  - moon_enforce_compaction(phi_guess, moon_density, moon_radius, S_c_pa) -> phi

Rigidity:
  - resolve_rigidity_profile(rigidity_profile|custom yields) -> (eta, Y_ice, Y_rock, Y_iron)
  - classify_moon_rigidity(rho, R, f_ice, f_rock, f_iron, eta, Y*) -> 'fluid'|'rigid'

Roche & outer stability:
  - density_gcc_from_mass_radius(M_kg, R_m) -> rho_gcc
  - roche_radius_from_densities(planet_radius, planet_mass, moon_density, moon_rigidity) -> a_R [m]
  - hill_radius_m(planet_orbit_a_m, planet_mass, star_mass) -> R_H [m]
  - outer_stability_limit_prograde_m(planet_orbit_a_m, planet_mass, star_mass) -> ~0.49 R_H [m]

Notes
-----
• This module is **sampling-free** and deterministic. You sample outside ploonetide.
• Prefix convention: `moon_...` for moon quantities; `planet_...` for planet ones.
• Densities are in g/cc to align with common geophysical conventions.
"""

# =============================================================================
# Physical constants and Earth/Sun reference units
# =============================================================================
# Zero-porosity (grain) densities for endmembers — typical values.
# These are used for mixture density via harmonic averaging.
RHO_ICE0 = 930  # [SI] H2O ice at low-T/low-P (approx.)
RHO_ROCK0 = 3300  # [SI] silicate rock (olivine/basalt-like, rough)
RHO_IRON0 = 7900  # [SI] Fe/Ni metal (rough)


# ---- Lane–Emden structural constants table
# (xi1, p2) where p2 ≡ -xi1^2 * theta'(xi1).
# These let us compute Pc(M,R;n) without integrating the equation at runtime.
# Values come from standard Lane–Emden tables; for n=1: analytic xi1=pi, -theta'(xi1)=1/pi.
_LE_TABLE = {
    0.0: (2.449508197, 4.89909021),
    0.5: (2.752698045, 3.788626506),
    1.0: (3.141592654, 3.141592654),
    1.5: (3.653753736, 2.714055127),
    2.0: (4.352874596, 2.411046012),
    2.5: (5.355275459, 2.187199565),
    3.0: (6.896848620, 2.018235951),
}


# =============================================================================
# Simple helpers
# =============================================================================
def _log10(x: float) -> float:
    """Safe log10 with a clearer error if mass ≤ 0 (fit domain requirement)."""
    if x <= 0:
        raise ValueError("Mass must be positive for Fortney fits.")
    return np.log10(x)

# =============================================================================
# Fortney et al. (2007) analytic mass–radius fits
# =============================================================================
# These reproduce the widely used polynomial fits for R(M) in Earth units.
# Validity is roughly 0.01–100 M⊕ for solid planets/moons.


# --- (A) Fortney docstrings: fix labels (names were already correct) ---

def fortney_radius_rock_iron(moon_mass: float, moon_rmf: float) -> float:
    """
    Radius [R⊕] for a **rock–iron** body from Fortney+07.

    Parameters
    ----------
    moon_mass : float
        Mass in Earth masses (M⊕). Valid ~ 0.01–100 M⊕.
    moon_rmf : float
        **Rock mass fraction** within the (rock+iron) sub-mixture, in [0, 1].
    """
    x = np.log10(moon_mass)
    return ((0.0592 * moon_rmf + 0.0975) * x**2.
            + (0.2337 * moon_rmf + 0.4938) * x
            + (0.3102 * moon_rmf + 0.7932))


def fortney_radius_ice_rock(moon_mass: float, moon_imf: float) -> float:
    """
    Radius [R⊕] for an **ice–rock** body from Fortney+07.

    Parameters
    ----------
    moon_mass : float
        Mass in Earth masses (M⊕). Valid ~ 0.01–100 M⊕.
    moon_imf : float
        **Ice mass fraction** within the (ice+rock) sub-mixture, in [0, 1].
    """
    x = np.log10(moon_mass)
    return ((0.0912 * moon_imf + 0.1603) * x**2.
            + (0.3330 * moon_imf + 0.7387) * x
            + (0.4639 * moon_imf + 1.1193))


# =============================================================================
# Composition helpers (no randomness)
# =============================================================================
def moon_imf_from(moon_f_ice: float, moon_f_rock: float) -> float:
    """
    Compute **IMF**: the ice fraction within (ice+rock).

    Returns 0.0 if (ice+rock) is zero to avoid division-by-zero.
    """
    denom = moon_f_ice + moon_f_rock
    return 0.0 if denom == 0.0 else (moon_f_ice / denom)


def moon_rmf_from(moon_f_rock: float, moon_f_iron: float) -> float:
    """
    Compute **RMF**: the rock fraction within (rock+iron).

    Returns 0.0 if (rock+iron) is zero to avoid division-by-zero.
    """
    denom = moon_f_rock + moon_f_iron
    return 0.0 if denom == 0.0 else (moon_f_rock / denom)


def moon_density_zero_porosity(
    moon_f_ice: float,
    moon_f_rock: float,
    moon_f_iron: float
) -> float:
    """
    Zero-porosity **mixture** density [SI] using **harmonic** averaging
    of endmember grain densities.

    Notes
    -----
    • We normalize the fractions if they do not sum to 1 (tiny drift).
    • This returns a *grain* density. Bulk density can be reduced by porosity.
    """
    s = moon_f_ice + moon_f_rock + moon_f_iron
    if not np.isclose(s, 1.0):
        moon_f_ice, moon_f_rock, moon_f_iron = moon_f_ice / s, moon_f_rock / s, moon_f_iron / s
    return 1.0 / ((moon_f_ice / RHO_ICE0) + (moon_f_rock / RHO_ROCK0) + (moon_f_iron / RHO_IRON0))


# =============================================================================
# Central pressure (toy model) & compaction
# =============================================================================
def moon_lane_emden_constants(n: float) -> Tuple[float, float, float]:
    """
    Return (xi1, p2, minus_theta_prime_at_xi1) for a polytrope of index n.
      - xi1: first zero of theta
      - p2 : -xi1^2 * theta'(xi1)
      - minus_theta_prime_at_xi1: p2 / xi1^2
    If n not in the table, returns nearest tabulated n (simple, robust).
    """
    if n not in _LE_TABLE:
        nearest = min(_LE_TABLE.keys(), key=lambda k: abs(k - n))
        xi1, p2 = _LE_TABLE[nearest]
    else:
        xi1, p2 = _LE_TABLE[n]
    minus_theta_prime = p2 / xi1**2.
    return xi1, p2, minus_theta_prime


def moon_central_pressure_uniform_pa(mass_kg: float, radius_m: float) -> float:
    """
    Uniform (incompressible) sphere central pressure (Pa):
        Pc = 3 G M^2 / (8 π R^4)
    Useful as a quick scale and for cross-checks.
    """
    if radius_m <= 0.0 or mass_kg <= 0.0:
        return float("nan")
    return 3.0 * GCONST * mass_kg**2 / (8.0 * np.pi * radius_m**4)


def moon_central_pressure_polytrope_pa(mass_kg: float, radius_m: float, n: float) -> float:
    r"""
    Central pressure for an n-index polytrope (Pa), using Lane–Emden constants:
        Pc = [ G M^2 / (4 π (n+1) R^4 ) ] * 1 / [(-θ'(ξ1))^2]
    Depends only on (M, R, n); no EOS K is required.
    """
    if radius_m <= 0.0 or mass_kg <= 0.0:
        return float("nan")
    _, _, minus_theta_prime = moon_lane_emden_constants(n)
    coeff = 1.0 / ((n + 1.0) * (minus_theta_prime ** 2))
    return (GCONST * mass_kg**2 / (4.0 * np.pi * radius_m**4)) * coeff


@dataclass
class MoonPorosityParams:
    """
    Parameters for a smooth porosity–pressure compaction law:
        φ(P) = max(φ_min, φ0_eff * exp(- κ_avg * P / P*_eff))

    Endmember anchors (order-of-magnitude, editable):
      - Ice:  φ0≈0.50, P*≈5 MPa
      - Rock: φ0≈0.30, P*≈50 MPa
      - Iron: φ0≈0.10, P*≈500 MPa
    κ_avg ~ 0.4 approximates <P> / Pc for self-gravitating bodies.

    All units:
      - Pressures in Pa
      - φ (porosity) in [0, 1)
    """
    # Endmember macroporosity at low pressure (dimensionless)
    phi0_ice: float = 0.50
    phi0_rock: float = 0.30
    phi0_iron: float = 0.10

    # Endmember compaction pressure scales (Pa)
    Pstar_ice: float = 5e6   # 5 MPa
    Pstar_rock: float = 5e7   # 50 MPa
    Pstar_iron: float = 5e8   # 500 MPa

    # Residual micro-porosity floor
    phi_min: float = 0.02

    # Volume-averaged to central-pressure scale factor
    kappa_avg: float = 0.4


def _moon_harmonic_mean_weighted(values: Tuple[float, float, float],
                                 weights: Tuple[float, float, float]) -> float:
    """
    Weighted harmonic mean. Ensures the softest phase (small P*) dominates the
    effective compaction scale (P*_eff).
    """
    v_ice, v_rock, v_iron = values
    w_ice, w_rock, w_iron = weights
    denom = (w_ice / max(v_ice, 1e-30)) + (w_rock / max(v_rock, 1e-30)) + (w_iron / max(v_iron, 1e-30))
    return (w_ice + w_rock + w_iron) / max(denom, 1e-30)


def moon_porosity_endmember_mix(f_ice: float, f_rock: float, f_iron: float,
                                pp: MoonPorosityParams) -> Tuple[float, float]:
    """
    Mix endmember porosity parameters to produce (phi0_eff, Pstar_eff).
      - φ0_eff: linear (volumetric) mixing of φ0's
      - P*_eff: weighted harmonic mean of P* scales (soft phase dominates)
    """
    w = (f_ice, f_rock, f_iron)
    phi0_eff = f_ice * pp.phi0_ice + f_rock * pp.phi0_rock + f_iron * pp.phi0_iron
    Pstar_eff = _moon_harmonic_mean_weighted((pp.Pstar_ice, pp.Pstar_rock, pp.Pstar_iron), w)
    return phi0_eff, Pstar_eff


def moon_effective_porosity_from_pressure(Pc_pa: float,
                                          f_ice: float, f_rock: float, f_iron: float,
                                          pp: Optional[MoonPorosityParams] = None) -> float:
    """
    Evaluate the smooth compaction law at a pressure scale Pc (Pa):
        φ(P) = max(φ_min, φ0_eff * exp(- κ_avg * Pc / P*_eff))
    Returns φ in [φ_min, 1).
    """
    if pp is None:
        pp = MoonPorosityParams()
    phi0_eff, Pstar_eff = moon_porosity_endmember_mix(f_ice, f_rock, f_iron, pp)
    P_eff = pp.kappa_avg * max(Pc_pa, 0.0)
    phi = phi0_eff * np.exp(- P_eff / max(Pstar_eff, 1e-30))
    return max(pp.phi_min, min(phi, 0.9999))


def moon_solve_radius_with_compaction(
    mass_kg: float,
    rho0_gcc: float,
    f_ice: float, f_rock: float, f_iron: float,
    pressure_model: str = "polytrope",
    polytrope_n: float = 1.0,
    max_iter: int = 5,
    tol_rel: float = 1e-4,
    pp: Optional[MoonPorosityParams] = None
) -> Tuple[float, float, float]:
    """
    Compute a self-consistent radius for a small/porous moon when MR fits are skipped.
      Inputs:
        mass_kg     : mass in kg
        rho0_gcc    : zero-porosity mixture density [g cm^-3]
        f_*         : mass fractions (sum ~1)
        pressure_model: 'polytrope' or 'uniform'
        polytrope_n : index if 'polytrope'
      Returns:
        (radius_m, rho_bulk_SI, Pc_pa)

    Algorithm (fixed-point):
      1) Start from R0 using zero-porosity density.
      2) Compute Pc(M,R) (chosen model).
      3) Evaluate φ(Pc) and set ρ_bulk=(1-φ)ρ0.
      4) Update R from ρ_bulk; iterate to convergence.

    Notes:
      * All internal calculations are SI; only rho0_gcc is converted at entry.
      * Keep max_iter small (5) — it converges quickly for these laws.
    """
    if pp is None:
        pp = MoonPorosityParams()

    rho0_si = rho0_gcc * 1000.0  # kg m^-3
    # Initial radius from zero-porosity density
    R = (3.0 * mass_kg / (4.0 * np.pi * rho0_si)) ** (1.0 / 3.0)

    def Pc(M, Rm):
        if pressure_model == "uniform":
            return moon_central_pressure_uniform_pa(M, Rm)
        else:
            return moon_central_pressure_polytrope_pa(M, Rm, polytrope_n)

    Pc_val = Pc(mass_kg, R)
    for _ in range(max_iter):
        phi = moon_effective_porosity_from_pressure(Pc_val, f_ice, f_rock, f_iron, pp)
        rho_bulk = (1.0 - phi) * rho0_si
        R_new = (3.0 * mass_kg / (4.0 * np.pi * max(rho_bulk, 1e-20))) ** (1.0 / 3.0)
        if abs(R_new - R) / max(R, 1e-30) < tol_rel:
            R = R_new
            break
        R = R_new
        Pc_val = Pc(mass_kg, R)

    # Final density from final Pc
    phi_final = moon_effective_porosity_from_pressure(Pc_val, f_ice, f_rock, f_iron, pp)
    rho_bulk_final = (1.0 - phi_final) * rho0_si
    return R, rho_bulk_final, Pc_val


# =============================================================================
# Rigidity profiles and classifier
# =============================================================================
def resolve_moon_rigidity_profile(
    moon_rigidity_profile: str = "baseline",
    eta_fluid: float | None = None,
    Y_ice_pa: float | None = None,
    Y_rock_pa: float | None = None,
    Y_iron_pa: float | None = None,
) -> tuple[float, float, float, float]:
    """
    Resolve a **rigidity profile** into thresholds for classifying the moon.

    Returns (eta_fluid, Y_ice, Y_rock, Y_iron) in [Pa].
    """
    # profiles = {
    #     "baseline": (10.0, 1.0e6, 5.0e7, 1.0e8),
    #     "icy_soft": (5.0, 0.5e6, 3.0e7, 0.8e8),
    #     "rocky_strong": (20.0, 2.0e6, 7.0e7, 1.5e8),
    # }
    profiles = {
        "baseline": (10.0, 1.0e6, 2.0e8, 3.0e8),   # η, Yice, Yrock, Yiron  [Pa]
        "icy_soft": (10.0, 0.5e6, 1.5e8, 2.5e8),  # slightly softer all-around
        "rocky_strong": (15.0, 1.0e6, 3.0e8, 5.0e8),
    }

    if moon_rigidity_profile == "custom":
        params = (eta_fluid, Y_ice_pa, Y_rock_pa, Y_iron_pa)
        if any(p is None for p in params):
            raise ValueError(
                "moon_rigidity_profile='custom' requires eta_fluid and all yields.")
        return eta_fluid, Y_ice_pa, Y_rock_pa, Y_iron_pa
    if moon_rigidity_profile not in profiles:
        raise ValueError(f"Unknown moon_rigidity_profile: {moon_rigidity_profile}")
    return profiles[moon_rigidity_profile]


def moon_classify_rigidity_from_pc(
    Pc_pa: float,
    f_ice: float,
    f_rock: float,
    f_iron: float,
    eta_fluid: float,
    Y_ice_pa: float,
    Y_rock_pa: float,
    Y_iron_pa: float
) -> Literal["fluid", "rigid"]:
    """
    Classify 'fluid' vs 'rigid' using a given Pc (Pa) and composition-weighted yield:
        Y_eff = f_ice*Y_ice + f_rock*Y_rock + f_iron*Y_iron
        fluid if Pc >= eta_fluid * Y_eff else rigid
    """
    Y_eff = (f_ice * Y_ice_pa) + (f_rock * Y_rock_pa) + (f_iron * Y_iron_pa)
    threshold = eta_fluid * max(Y_eff, 1.0)  # guard against zeros
    return "fluid" if Pc_pa >= threshold else "rigid"
