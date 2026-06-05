#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""Planet--moon tidal ODEs for Ploonetide.

This module keeps the original circular Barnes & O'Brien/Sasaki branch
as the default limit, but adds an eccentric/obliquity-ready branch based
on the signed-lag Darwin/Ferraz-Mello formalism.

Important implementation choices
--------------------------------
1. The evolved moon mean-motion variable is still log(nm), exactly as in
   the previous working implementation.
2. When eccentricity is active, the evolved eccentricity variable is now
   hm = em**2. The eccentricity itself is recovered internally as
   em = sqrt(max(hm, 0)). This avoids negative eccentricity steps and
   keeps the ODE system better behaved near em = 0.
3. The current circular behaviour is preserved when all new flags are off:
   em = 0, psim = 0, no moon dissipation, no moment-of-inertia term, and
   the current mass approximations are used.
4. near_synchronization(), finite_difference_jacobian(), and jacobian()
   are intentionally kept unchanged below, as requested.
"""

import numpy as np

from ploonetide.utils.functions import *
from ploonetide.utils.constants import GCONST


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------
def _flag(physics_flags, names, default=False):
    """Return True if any of the alias names is active in physics_flags."""
    if isinstance(names, str):
        names = (names,)

    for name in names:
        if name in physics_flags:
            return bool(physics_flags[name])

    return bool(default)


def _first_available(*dicts, keys, default=None):
    """Get the first available key from a list of dictionaries."""
    for dictionary in dicts:
        if not isinstance(dictionary, dict):
            continue
        for key in keys:
            if key in dictionary:
                return dictionary[key]
    return default


def _tidal_sign(omega, reference_frequency, parameters, physics_flags, *, smooth=False):
    """Signed tidal response for one tidal harmonic.

    If ``smooth=False``, this returns the discontinuous CPL sign,
    ``np.sign(omega)``.

    If ``smooth=True``, this replaces the discontinuous sign by

        tanh(omega / omega_smooth),

    which regularizes the RHS near tidal synchronization surfaces such as
    Omega_p ~= n_p. This is essential for adaptive solvers, because a hard
    sign flip can make LSODA/Radau/BDF take extremely small steps.

    The smoothing width is controlled by either

        parameters["omega_smooth"]

    or, if absent,

        parameters["tidal_sign_smoothing_fraction"] * reference_frequency.

    Important: this must use ``np.tanh`` rather than ``np.tan``.
    """
    if not smooth:
        return np.sign(omega)

    omega_smooth = parameters.get("omega_smooth", None)
    if omega_smooth is None:
        frac = parameters.get("tidal_sign_smoothing_fraction", 1.0e-4)
        omega_smooth = frac * max(abs(reference_frequency), 1.0e-30)

    if omega_smooth <= 0.0:
        return np.sign(omega)

    return np.tanh(omega / omega_smooth)


def _smooth_tidal_signs(parameters):
    """Whether to smooth signed tidal harmonics.

    Default is True because hard sign flips at synchronization surfaces
    make adaptive solvers stall. Set

        parameters["smooth_tidal_signs"] = False

    only when you deliberately want the discontinuous CPL limit.
    """
    return bool(parameters.get("smooth_tidal_signs", True))


def _planet_properties(t, integrator_args):
    """Return the planet mass and radius at time t."""
    parameters = integrator_args["parameters"]
    physics_flags = parameters["physics_flags"]
    planet_fixed_properties = parameters["planet_fixed_properties"]

    if physics_flags["planet_evolution"]:
        planet_track = integrator_args["planet_track"]
        Mp = planet_track.M(t + 10 * MYEAR)
        Rp = planet_track.R(t + 10 * MYEAR)
    else:
        Mp = planet_fixed_properties["Mp"]
        Rp = planet_fixed_properties["Rp"]

    return Mp, Rp


def _sigma_cgs_to_si(sigma_cgs):
    """Convert sigma from g^-1 cm^-2 s^-1 to kg^-1 m^-2 s^-1.

    Since

        1 g^-1 = 1e3 kg^-1
        1 cm^-2 = 1e4 m^-2

    we have

        1 g^-1 cm^-2 s^-1 = 1e7 kg^-1 m^-2 s^-1.
    """
    return 1.0e7 * sigma_cgs


def _equilibrium_tide_k2q_from_sigma(sigma_si, radius, omega):
    """Positive k2/Q amplitude for a CTL equilibrium tide.

    For the sigma-based constant-time-lag prescription,

        k2/Q ~= 3 sigma R^5 |omega| / G.

    The sign of the harmonic is applied separately by _tidal_sign().
    """
    return 3.0 * sigma_si * radius**5 * abs(omega) / GCONST


def _planet_equilibrium_sigma_si(parameters):
    """Return the planet equilibrium-tide sigma in SI units.

    User may provide either:
        planet_sigma_eq_si
    or:
        planet_sigma_eq_cgs

    Default follows the gas-giant value used by Bolmont et al. (2025):
        sigma = 2.006e-61 g^-1 cm^-2 s^-1
              = 2.006e-54 kg^-1 m^-2 s^-1.
    """
    if "planet_sigma_eq_si" in parameters:
        return parameters["planet_sigma_eq_si"]

    sigma_cgs = parameters.get("planet_sigma_eq_cgs", 2.006e-61)
    return _sigma_cgs_to_si(sigma_cgs)


def _moon_equilibrium_sigma_si(parameters, initial_conds):
    """Return the moon equilibrium-tide sigma in SI units.

    The moon should have its own sigma. The planet value should not be
    silently reused because sigma encodes the body's internal dissipation.
    """
    sigma_si = _first_available(
        parameters,
        initial_conds,
        keys=("moon_sigma_eq_si", "sigma_moon_si"),
        default=None,
    )

    if sigma_si is not None:
        return sigma_si

    sigma_cgs = _first_available(
        parameters,
        initial_conds,
        keys=("moon_sigma_eq_cgs", "sigma_moon_cgs"),
        default=None,
    )

    if sigma_cgs is not None:
        return _sigma_cgs_to_si(sigma_cgs)

    return None


def _planet_structural_k2q_components(t, op, integrator_args):
    """Return frequency-independent positive planet k2/Q components.

    These are the components that, in the current implementation, do not
    explicitly depend on the tidal harmonic frequency:

        core
        mantle
        frequency-averaged inertial-wave amplitude

    The inertial-wave amplitude returned here is only the structural,
    frequency-averaged value. Whether it is allowed for a specific harmonic
    is checked later through the inertial-range condition.
    """
    parameters = integrator_args["parameters"]
    physics_flags = parameters["physics_flags"]
    planet_fixed_properties = parameters["planet_fixed_properties"]

    rigidity = parameters["rigidity"]
    planet_envelope_interface = planet_fixed_properties["envelope_interface"]

    Mp, Rp = _planet_properties(t, integrator_args)

    Mp_core = planet_fixed_properties["Mp_core"]
    Rp_core = planet_fixed_properties["Rp_core"]

    alpha_planet = Rp_core / Rp
    beta_planet = Mp_core / Mp
    epsilon = op / omegaCritic(Mp, Rp)

    k2q_core = 0.0
    k2q_mantle = 0.0
    k2q_envelope_iw = 0.0

    if physics_flags["planet_core_dissipation"]:
        k2q_core = k2Q_planet_core(
            rigidity,
            alpha_planet,
            beta_planet,
            Mp,
            Rp,
        )

    if physics_flags["planet_mantle_dissipation"]:
        k2q_mantle = k2Q_planet_mantle(
            rigidity,
            alpha_planet,
            beta_planet,
            Mp,
            Rp,
        )

    if physics_flags["planet_envelope_dissipation"]:
        k2q_envelope_iw = k2Q_inertial_waves_convective_envelope(
            alpha_planet,
            beta_planet,
            epsilon,
            interface=planet_envelope_interface,
        )

    return {
        "core": k2q_core,
        "mantle": k2q_mantle,
        "iw": k2q_envelope_iw,
    }


def _inertial_wave_allowed(omega, op):
    """Return True if one harmonic lies inside the inertial-wave range.

    In a rotating convective region, inertial waves are excited only when

        |omega| <= 2 |Omega|.

    For the circular semi-diurnal tide omega0 = 2(Omega - n), this reduces
    to n <= 2 Omega for positive Omega and n.
    """
    return abs(omega) <= 2.0 * abs(op)


def _inertial_wave_gate_weight(omega, op, parameters):
    """Smooth activation weight for inertial-wave dissipation.

    The physical inertial-wave range is

        |omega| <= 2 |Omega_p|.

    A hard on/off switch at this boundary creates a discontinuous RHS and
    can stall adaptive integrators. This helper replaces the hard
    Heaviside switch by a narrow tanh transition.

    Returns
    -------
    weight : float
        Approximately 1 inside the inertial range and 0 outside it.
    """
    boundary = 2.0 * abs(op)

    if boundary <= 0.0:
        return 0.0

    # Width of the transition around the inertial-wave boundary.
    # Start with 1e-3 or 1e-2 for robust population runs.
    frac = parameters.get("iw_gate_smoothing_fraction", 1.0e-3)
    delta = frac * max(boundary, 1.0e-300)

    if delta <= 0.0:
        return float(abs(omega) <= boundary)

    x = (boundary - abs(omega)) / delta

    return 0.5 * (1.0 + np.tanh(x))


def _iw_body_is_active(body, parameters):
    if body == "moon":
        return bool(parameters.get("planet_envelope_dissipation_for_moon", True))

    if body == "star":
        return bool(parameters.get("planet_envelope_dissipation_for_star", True))

    return True


def _planet_positive_k2q_for_harmonic(
    t,
    op,
    omega,
    integrator_args,
    *,
    body="unknown",
):
    """Positive planet k2/Q amplitude for one tidal harmonic.

    This is the harmonic-aware dissipation layer.

    Components:
    - fixed value, if planet_energy_dissipation_fix_value=True;
    - core and mantle, currently treated as frequency-independent;
    - equilibrium tide, proportional to |omega|;
    - frequency-averaged inertial waves, included only if the harmonic lies
      inside the inertial-wave excitation range.
    """
    parameters = integrator_args["parameters"]
    physics_flags = parameters["physics_flags"]
    planet_fixed_properties = parameters["planet_fixed_properties"]

    if physics_flags["planet_energy_dissipation_fix_value"]:
        physical_channels_active = (
            _flag(
                physics_flags,
                (
                    "planet_equilibrium_tide",
                    "planet_equilibrium_dissipation",
                    "planet_eq_tide",
                ),
                default=False,
            )
            or physics_flags["planet_envelope_dissipation"]
            or physics_flags["planet_core_dissipation"]
            or physics_flags["planet_mantle_dissipation"]
        )

        if physical_channels_active and parameters.get("strict_dissipation_flags", True):
            raise ValueError(
                "Inconsistent dissipation setup: "
                "planet_energy_dissipation_fix_value=True overrides all physical "
                "dissipation channels, but at least one physical channel is also active. "
                "Set planet_energy_dissipation_fix_value=False when using "
                "planet_equilibrium_tide, planet_envelope_dissipation, "
                "planet_core_dissipation, or planet_mantle_dissipation."
            )

        return planet_fixed_properties["planet_k2q"]

    Mp, Rp = _planet_properties(t, integrator_args)

    components = _planet_structural_k2q_components(t, op, integrator_args)

    k2q_core = components["core"]
    k2q_mantle = components["mantle"]
    k2q_eq = 0.0

    # ------------------------------------------------------------
    # Equilibrium tide: CTL, harmonic-frequency dependent.
    # ------------------------------------------------------------
    if _flag(
            physics_flags,
            (
                "planet_equilibrium_tide",
                "planet_equilibrium_dissipation",
                "planet_eq_tide",
            ),
            default=False,
    ):
        sigma_p_si = _planet_equilibrium_sigma_si(parameters)
        k2q_eq = _equilibrium_tide_k2q_from_sigma(
            sigma_p_si,
            Rp,
            omega,
        )
    # ------------------------------------------------------------
    # Dynamical tide: frequency-averaged IW amplitude.
    #
    # The envelope/IW flag has already been applied inside
    # _planet_structural_k2q_components(). Therefore:
    #
    #   components["iw"] = 0
    #
    # if IWs are inactive. Here we only apply the physical
    # harmonic-by-harmonic inertial-range condition.
    # ------------------------------------------------------------
    iw_allowed = _inertial_wave_allowed(omega, op)
    iw_ratio = abs(omega) / max(2.0 * abs(op), 1.0e-300)

    iw_gate_weight = _inertial_wave_gate_weight(omega, op, parameters)
    iw_body_active = _iw_body_is_active(body, parameters)
    iw_weight = iw_gate_weight if iw_body_active else 0.0
    # ------------------------------------------------------------
    # Temporary diagnostic switch:
    # allow IWs only for selected tide-raising bodies.
    #
    # Examples:
    #   parameters["iw_apply_to_bodies"] = ("moon",)
    #   parameters["iw_apply_to_bodies"] = ("star",)
    #   parameters["iw_apply_to_bodies"] = ("moon", "star")
    #
    # If absent or None, use the normal production behaviour:
    # apply IWs to all bodies.
    # ------------------------------------------------------------
    # iw_apply_to_bodies = parameters.get("iw_apply_to_bodies", (None,))
    if not _iw_body_is_active(body, parameters):
        iw_weight = 0.0

    k2q_iw = iw_weight * components["iw"]
    k2q_total = k2q_core + k2q_mantle + k2q_eq + k2q_iw

    if parameters.get("debug_tidal_dissipation", False):
        counter = parameters.get("_tidal_diss_debug_counter", 0) + 1
        parameters["_tidal_diss_debug_counter"] = counter

        cadence = int(parameters.get("tidal_diss_debug_cadence", 500))

        if counter == 1 or counter % cadence == 0:
            print(
                "[planet k2q harmonic] "
                f"call={counter} "
                f"t={t:.6e} "
                f"body={body} "
                f"omega={omega:.6e} "
                f"op={op:.6e} "
                f"IW_allowed={iw_allowed} "
                f"IW_ratio={iw_ratio:.6e} "
                f"IW_gate_weight={iw_gate_weight:.6e} "
                f"IW_body_active={iw_body_active} "
                f"IW_weight={iw_weight:.6e} "
                f"k2q_core={k2q_core:.12e} "
                f"k2q_mantle={k2q_mantle:.12e} "
                f"k2q_eq={k2q_eq:.12e} "
                f"k2q_iw={k2q_iw:.12e} "
                f"IW_raw={components['iw']:.12e} "
                f"k2q_total={k2q_total:.12e} "
            )

    return k2q_total


def _signed_planet_k2q_for_harmonic(
    t,
    op,
    omega,
    reference_frequency,
    integrator_args,
    *,
    smooth=False,
    body='unknown',
):
    """Signed harmonic dissipation factor K_j.

    This returns

        K_j = S(omega_j) * (k2/Q)_p,j,

    where the positive amplitude is evaluated for the specific harmonic.
    """
    parameters = integrator_args["parameters"]
    physics_flags = parameters["physics_flags"]

    k2q_positive = _planet_positive_k2q_for_harmonic(
        t,
        op,
        omega,
        integrator_args,
        body=body,
    )

    sign_factor = _tidal_sign(
        omega,
        reference_frequency,
        parameters,
        physics_flags,
        smooth=smooth,
    )

    K_signed = sign_factor * k2q_positive

    if parameters.get("debug_tidal_dissipation", False):
        counter = parameters.get("_signed_harmonic_debug_counter", 0) + 1
        parameters["_signed_harmonic_debug_counter"] = counter

        cadence = int(parameters.get("signed_harmonic_debug_cadence", 500))

        if counter == 1 or counter % cadence == 0:
            print(
                "[signed harmonic] "
                f"call={counter} "
                f"body={body} "
                f"t={t:.6e} "
                f"omega={omega:.6e} "
                f"ref={reference_frequency:.6e} "
                f"sign_factor={sign_factor:.12e} "
                f"k2q_positive={k2q_positive:.12e} "
                f"K_signed={K_signed:.12e}"
            )

    return K_signed


def _planet_total_k2q(t, op, integrator_args):
    """Legacy positive planet k2/Q amplitude.

    Kept for backwards compatibility and diagnostics. The coupled RHS should
    now use _planet_positive_k2q_for_harmonic() instead, because equilibrium
    tides and inertial-wave activation are harmonic-dependent.
    """
    parameters = integrator_args["parameters"]
    physics_flags = parameters["physics_flags"]
    planet_fixed_properties = parameters["planet_fixed_properties"]

    if physics_flags["planet_energy_dissipation_fix_value"]:
        return planet_fixed_properties["planet_k2q"]

    components = _planet_structural_k2q_components(t, op, integrator_args)
    return components["core"] + components["mantle"] + components["iw"]


def _use_exact_orbital_masses(parameters):
    """Whether to use G(Mp+Mb) instead of the historical dominant-mass limit."""
    physics_flags = parameters["physics_flags"]
    return _flag(
        physics_flags,
        ("use_exact_orbital_masses", "exact_kepler_masses"),
        default=False,
    )


def _planet_moment_inertia_coeff(parameters):
    """Return the planet moment-of-inertia coefficient alpha_p.

    The main class currently provides this as
    ``planet_moment_inertia_coeff``. The fallback to the legacy
    ``planet_GR_coeff`` keeps old circular runs working.
    """
    coeff = parameters.get(
        "planet_moment_inertia_coeff",
        parameters.get("planet_GR_coeff"),
    )
    if coeff is None:
        raise KeyError(
            "Expected 'planet_moment_inertia_coeff' or legacy "
            "'planet_GR_coeff' in parameters."
        )
    return coeff


def _kepler_mass(body, Mp, Mb, parameters):
    """Mass entering Kepler's third law for the selected orbit.

    Default behaviour preserves the previous Ploonetide approximations:
    - moon orbit:  a_m^3 n_m^2 ~= G Mp
    - planet orbit: a_p^3 n_p^2 ~= G Ms

    Set use_exact_orbital_masses=True to use G(Mp + Mb).
    """
    if _use_exact_orbital_masses(parameters):
        return Mp + Mb

    if body == "star":
        return Mb

    return Mp


def _radius_factor(R_body, n_orb, M_kepler):
    """Return (R_body/a)^5 written in terms of n_orb."""
    return R_body**5 * n_orb**(10.0 / 3.0) / (GCONST * M_kepler)**(5.0 / 3.0)


def _spin_torque_factor(body, Mb, n_orb, Mp, Rp, GR, parameters):
    """Return G Mb^2 Rp^5/(Ip a^6), written in the current code's variables.

    With the default mass approximations this exactly preserves the old
    circular expressions:
    - moon:  Mm^2 Rp^3 n_m^4 / (GR G Mp^3)
    - star:  Rp^3 n_p^4 / (GR G Mp)
    """
    if _use_exact_orbital_masses(parameters):
        return Mb**2 * Rp**3 * n_orb**4 / (GR * GCONST * Mp * (Mp + Mb)**2)

    if body == "star":
        return Rp**3 * n_orb**4 / (GR * GCONST * Mp)

    return Mb**2 * Rp**3 * n_orb**4 / (GR * GCONST * Mp**3)


def _semi_major_axis_from_n(body, n_orb, Mp, Mb, parameters):
    """Recover a from n through Kepler's third law."""
    mkep = _kepler_mass(body, Mp, Mb, parameters)
    return (GCONST * mkep / n_orb**2) ** (1.0 / 3.0)


def _lambda_obliquity(body, n_orb, Mb, op, Mp, Rp, GR, parameters):
    """Angular-momentum ratio entering the Ferraz-Mello obliquity equation."""
    if abs(Mb) <= 0.0:
        return 0.0

    a_orb = _semi_major_axis_from_n(body, n_orb, Mp, Mb, parameters)
    Ip = GR * Mp * Rp**2
    return Ip * op * a_orb * n_orb / (GCONST * Mp * Mb)


def _track_derivative(track, quantity, t, parameters):
    """Best-effort derivative of a planet-track quantity.

    The preferred route is an explicit derivative method on the track. If no
    derivative is available, a small finite-difference estimate is used. This
    helper is used when planet_evolution=True and the track does not provide derivatives.
    """
    derivative_names = {
        "R": ("dRdt", "Rdot", "R_dot", "dR_dt", "dR"),
        "M": ("dMdt", "Mdot", "M_dot", "dM_dt", "dM"),
    }

    for name in derivative_names.get(quantity, ()):
        method = getattr(track, name, None)
        if callable(method):
            return method(t)

    func = getattr(track, quantity)
    step = parameters.get("planet_track_derivative_step", None)
    if step is None:
        step = max(1.0, 1.0e-6 * max(abs(t), 1.0))

    try:
        return (func(t + step) - func(t - step)) / (2.0 * step)
    except Exception:
        try:
            return (func(t + step) - func(t)) / step
        except Exception:
            return 0.0


def _planet_moment_inertia_log_derivative(t, op, integrator_args):
    """Return dln(Ip)/dt for Ip = alpha_p Mp Rp^2.

    This reuses the existing ``planet_evolution`` flag. If the planet radius
    evolves through the planetary track, the planet moment of inertia evolves
    too. For now we keep alpha_p and Mp fixed in this structural term, so

        dln(Ip)/dt ~= 2 dln(Rp)/dt.

    The track currently supplies R(t), not necessarily Rdot(t), so Rdot is
    estimated internally when no derivative method is available.
    """
    parameters = integrator_args["parameters"]
    physics_flags = parameters["physics_flags"]

    # No evolving radius track -> no structural spin-up/down term.
    if not physics_flags["planet_evolution"]:
        return 0.0

    planet_track = integrator_args["planet_track"]
    _, Rp = _planet_properties(t, integrator_args)

    track_t = t + 10 * MYEAR
    Rdot = _track_derivative(planet_track, "R", track_t, parameters)

    if Rp == 0.0:
        return 0.0

    return 2.0 * Rdot / Rp


def _moon_dissipation_is_active(parameters, initial_conds, eccm=None):
    """Return True if moon tides should contribute to the ODEs.

    Moon dissipation is dynamically relevant only if the moon has an
    eccentricity or spin-obliquity tide to damp. The enabled mechanism
    is controlled separately by flags such as moon_equilibrium_tide.
    """
    physics_flags = parameters["physics_flags"]

    # Is any moon dissipation prescription enabled?
    moon_mechanism_active = _flag(
        physics_flags,
        (
            "moon_equilibrium_tide",
            "moon_equilibrium_dissipation",
            "moon_eq_tide",
            "moon_fixed_k2q",
            "moon_energy_dissipation_fix_value",
        ),
        default=False,
    )

    if not moon_mechanism_active:
        return False

    # Current eccentricity, if supplied by the RHS.
    if eccm is None:
        eccm = float(_initial_moon_eccentricity(parameters, initial_conds) or 0.0)

    chim = float(_moon_spin_obliquity(parameters, initial_conds) or 0.0)

    e_floor = parameters.get("eccentricity_floor", 0.0)
    chi_floor = parameters.get("obliquity_floor", 0.0)

    moon_has_tide_to_damp = (
        abs(eccm) > e_floor
        or abs(chim) > chi_floor
    )

    return moon_has_tide_to_damp


def _moon_radius_and_k2q(parameters, initial_conds, nm=None):
    """Return moon radius and k2/Q amplitude for optional moon dissipation.

    If a fixed moon_k2q is provided, it is used directly.

    If moon equilibrium tides are enabled and a moon sigma is provided,
    the CTL value is added using the synchronously rotating eccentric
    annual mode, omega ~= n_m.
    """
    physics_flags = parameters["physics_flags"]
    planet_fixed_properties = parameters["planet_fixed_properties"]

    Rm = _first_available(
        parameters,
        planet_fixed_properties,
        initial_conds,
        keys=("Rm", "R_m", "moon_radius", "moon_R"),
        default=0.0,
    )

    k2q_moon = _first_available(
        parameters,
        planet_fixed_properties,
        initial_conds,
        keys=("moon_k2q", "k2q_moon", "moon_k2_over_Q", "k2Q_moon"),
        default=0.0,
    )

    k2q_moon = float(k2q_moon or 0.0)

    if _flag(
        physics_flags,
        (
            "moon_equilibrium_tide",
            "moon_equilibrium_dissipation",
            "moon_eq_tide",
        ),
        default=False,
    ):
        sigma_m_si = _moon_equilibrium_sigma_si(parameters, initial_conds)

        if sigma_m_si is not None and Rm > 0.0 and nm is not None:
            k2q_moon += _equilibrium_tide_k2q_from_sigma(
                sigma_m_si,
                Rm,
                nm,
            )

    return Rm, k2q_moon


def _moon_spin_obliquity(parameters, initial_conds):
    return _first_available(
        parameters,
        initial_conds,
        keys=("chim", "chi_m", "moon_spin_obliquity", "moon_obliquity"),
        default=0.0,
    )


def _stellar_obliquity(parameters, initial_conds):
    return _first_available(
        parameters,
        initial_conds,
        keys=("psis", "psi_s", "psi_star", "stellar_obliquity"),
        default=0.0,
    )


def _initial_obliquity(parameters, initial_conds):
    """Return initial planet--moon obliquity in radians.

    User-facing code may store the degree value elsewhere, but the ODE layer
    always works in radians.
    """
    return _first_available(
        parameters,
        initial_conds,
        keys=(
            "psim_ini",
            "psim",
            "psi_m",
            "moon_orbit_obliquity",
            "obliquity"
        ),
        default=0.0,
    )


def _initial_moon_eccentricity(parameters, initial_conds):
    """Return initial moon orbital eccentricity.

    This is the eccentricity of the moon orbit around the planet. The ODE
    layer evolves hm = em**2.
    """
    return _first_available(
        parameters,
        initial_conds,
        keys=(
            "em_ini",
            "eccm_ini",
            "moon_eccentricity_ini",
            "moon_ecc_ini",
            "moon_orbit_eccentricity",
        ),
        default=0.0,
    )


def _initial_planet_eccentricity(parameters, initial_conds):
    """Return initial planet--star orbital eccentricity.

    This is the eccentricity of the planet orbit around the star. The ODE
    layer evolves hp = ep**2.
    """
    return _first_available(
        parameters,
        initial_conds,
        keys=(
            "ep_ini",
            "eccp_ini",
            "planet_eccentricity_ini",
            "planet_ecc_ini",
            "planet_orbit_eccentricity",
        ),
        default=0.0,
    )


def _state_from_y(y, integrator_args, initial_conds):
    """Parse the planet--moon state vector.

    Base circular/coplanar state:
        y = [op, npp, log(nm)]

    Optional variables are appended in this fixed order:
        hm    = em**2    moon eccentricity squared
        psim             planet--moon obliquity
        hp    = ep**2    planet eccentricity squared

    Therefore the full state is:
        y = [op, npp, log(nm), hm, psim, hp]

    The orchestrator supplies the physical initial values em_ini, ep_ini,
    and psim_ini, and builds the ODE state using hm_ini=em_ini**2 and
    hp_ini=ep_ini**2.
    """
    y = np.asarray(y, dtype=float)
    parameters = integrator_args["parameters"]

    op = y[0]
    npp = y[1]
    log_nm = y[2]

    e_floor = parameters.get("eccentricity_floor", 0.0)
    psi_floor = parameters.get("obliquity_floor", 0.0)

    # Physical initial values. These determine which optional state
    # variables are active, while the state vector itself stores hm and hp.
    em_ini = float(_initial_moon_eccentricity(parameters, initial_conds) or 0.0)
    ep_ini = float(_initial_planet_eccentricity(parameters, initial_conds) or 0.0)
    psim_ini = float(_initial_obliquity(parameters, initial_conds) or 0.0)

    requested_ecc = abs(em_ini) > e_floor
    requested_psi = abs(psim_ini) > psi_floor
    requested_planet_ecc = abs(ep_ini) > e_floor

    expected_size = (
        3
        + int(requested_ecc)
        + int(requested_psi)
        + int(requested_planet_ecc)
    )

    if y.size != expected_size:
        raise ValueError(
            "Inconsistent planet_moon state-vector layout. Expected length "
            f"{expected_size} for requested flags "
            f"(moon eccentricity={requested_ecc}, "
            f"moon obliquity={requested_psi}, "
            f"planet eccentricity={requested_planet_ecc}), "
            f"but got length {y.size}. Expected layout is "
            "[op, npp, log_nm], optionally followed by hm, psim, hp "
            "in that order."
        )

    idx = 3

    # Moon eccentricity state: hm = em**2
    if requested_ecc:
        hm = max(y[idx], 0.0)
        eccm = np.sqrt(hm)
        idx += 1
    else:
        hm = 0.0
        eccm = 0.0

    # Planet--moon obliquity state: psim
    if requested_psi:
        psim = y[idx]
        idx += 1
    else:
        psim = 0.0

    # Planet eccentricity state: hp = ep**2
    if requested_planet_ecc:
        hp = max(y[idx], 0.0)
        eccp = np.sqrt(hp)
        idx += 1
    else:
        hp = 0.0
        eccp = 0.0

    return {
        "op": op,
        "npp": npp,
        "log_nm": log_nm,
        "nm": np.exp(log_nm),

        "eccm": eccm,
        "hm": hm,
        "psim": psim,

        "eccp": eccp,
        "hp": hp,

        "e_state_active": requested_ecc,
        "psi_state_active": requested_psi,
        "planet_e_state_active": requested_planet_ecc,

        "requested_ecc": requested_ecc,
        "requested_psi": requested_psi,
        "requested_planet_ecc": requested_planet_ecc,
    }


def _planet_tide_harmonics_and_brackets(
    t,
    op,
    n_orb,
    ecc,
    psi,
    integrator_args,
    *,
    body="unknown",
    extended_active=False,
):
    """Return signed harmonics and Ferraz-Mello brackets for a tide on the planet.

    This is the generic b-branch used for both:
    - b = moon: n_orb = nm,  ecc = em, psi = psim
    - b = star: n_orb = npp, ecc = ep, psi = psis

    The returned brackets correspond to:
        orbit_bracket -> B_n,b
        ecc_bracket   -> B_h,b
        spin_bracket  -> B_Omega,b
    """
    parameters = integrator_args["parameters"]

    ecc2 = ecc**2
    S2 = np.sin(psi) ** 2
    smooth_signs = extended_active or _smooth_tidal_signs(parameters)

    omega0 = 2.0 * op - 2.0 * n_orb
    omega1 = 2.0 * op - 3.0 * n_orb
    omega2 = 2.0 * op - 1.0 * n_orb
    omega5 = n_orb
    omega8 = op - 2.0 * n_orb
    omega9 = op

    K0 = _signed_planet_k2q_for_harmonic(
        t,
        op,
        omega0,
        n_orb,
        integrator_args,
        smooth=smooth_signs,
        body=body,
    )

    # Radial monthly tide: omega5 = n_orb > 0.
    # The sign is positive, but the amplitude is still evaluated at omega5.
    K5 = _planet_positive_k2q_for_harmonic(
        t,
        op,
        omega5,
        integrator_args,
        body=body,
    )

    K1 = K2 = K8 = K9 = 0.0

    if ecc2 > 0.0:
        K1 = _signed_planet_k2q_for_harmonic(
            t,
            op,
            omega1,
            n_orb,
            integrator_args,
            smooth=smooth_signs,
            body=body,
        )

        K2 = _signed_planet_k2q_for_harmonic(
            t,
            op,
            omega2,
            n_orb,
            integrator_args,
            smooth=smooth_signs,
            body=body,
        )

    if S2 > 0.0:
        K8 = _signed_planet_k2q_for_harmonic(
            t,
            op,
            omega8,
            n_orb,
            integrator_args,
            smooth=smooth_signs,
            body=body,
        )

        K9 = _signed_planet_k2q_for_harmonic(
            t,
            op,
            omega9,
            n_orb,
            integrator_args,
            smooth=smooth_signs,
            body=body,
        )

    orbit_bracket = (
        4.0 * K0
        - ecc2
        * (
            20.0 * K0
            - 147.0 / 2.0 * K1
            - 1.0 / 2.0 * K2
            + 3.0 * K5
        )
        - 4.0 * S2 * (K0 - K8)
    )

    ecc_bracket = (
        2.0 * K0
        - 49.0 / 2.0 * K1
        + 1.0 / 2.0 * K2
        + 3.0 * K5
    )

    spin_bracket = (
        4.0 * K0
        + ecc2 * (-20.0 * K0 + 49.0 * K1 + K2)
        + 2.0 * S2 * (-2.0 * K0 + K8 + K9)
    )

    harmonics = {
        "K0": K0,
        "K1": K1,
        "K2": K2,
        "K5": K5,
        "K8": K8,
        "K9": K9,
        "omega0": omega0,
        "omega1": omega1,
        "omega2": omega2,
        "omega5": omega5,
        "omega8": omega8,
        "omega9": omega9,
    }

    return orbit_bracket, ecc_bracket, spin_bracket, harmonics


# -----------------------------------------------------------------------------
# RHS components
# -----------------------------------------------------------------------------
def _rhs_components(
    t,
    op,
    npp,
    log_nm,
    eccm,
    psim,
    eccp,
    integrator_args,
    initial_conds
):
    """Compute the coupled RHS terms for the planet--moon problem.

    The planet tide terms follow the signed-k2/Q version of the
    Ferraz-Mello et al. (2008) second-order equations. With e=0 and
    psi=0 they reduce to the old Barnes/O'Brien/Sasaki circular branch.
    """
    parameters = integrator_args["parameters"]

    GR = _planet_moment_inertia_coeff(parameters)
    Mm = parameters["Mm"]
    # Ms is always available in the coupled solver, but keeping a safe
    # default avoids breaking standalone calls to dnmdt()/demdt().
    Ms = parameters.get("Ms", 0.0)

    nm = np.exp(log_nm)
    em2 = eccm**2
    hp = eccp**2

    Mp, Rp = _planet_properties(t, integrator_args)

    # ------------------------------------------------------------------
    # Planetary dissipation due to the moon. This is the b = moon branch
    # of the generic Ferraz-Mello signed-k2/Q equations.
    # ------------------------------------------------------------------
    extended_active = bool(parameters.get("_extended_tides_active", False))

    orbit_bracket_m, ecc_bracket_m, spin_bracket_m, K_m = (
        _planet_tide_harmonics_and_brackets(
            t,
            op,
            nm,
            eccm,
            psim,
            integrator_args,
            extended_active=extended_active,
            body='moon',
        )
    )

    K0m = K_m["K0"]
    K8m = K_m["K8"]
    K9m = K_m["K9"]

    mkep_m = _kepler_mass("moon", Mp, Mm, parameters)
    Rpm = _radius_factor(Rp, nm, mkep_m)
    Tpm = _spin_torque_factor("moon", Mm, nm, Mp, Rp, GR, parameters)

    dnm_planet = -(
        9.0 / 8.0
        * nm**2
        * (Mm / Mp)
        * Rpm
        * orbit_bracket_m
    )

    # ------------------------------------------------------------------
    # Optional diagnostics for the moon-orbit tidal bracket.
    #
    # Controlled globally by:
    #     parameters["debug_ODEs_rhs"]
    #
    # and, optionally, more specifically by:
    #     parameters["debug_moon_orbit_bracket"]
    #
    # If debug_moon_orbit_bracket is not provided, it follows debug_ODEs_rhs.
    # ------------------------------------------------------------------
    debug_rhs = bool(parameters.get("debug_ODEs_rhs", False))
    debug_moon_bracket = bool(
        parameters.get("debug_moon_orbit_bracket", debug_rhs)
    )

    if debug_rhs and debug_moon_bracket:
        counter = parameters.get("_moon_orbit_bracket_debug_counter", 0) + 1
        parameters["_moon_orbit_bracket_debug_counter"] = counter

        cadence = int(parameters.get("moon_orbit_bracket_debug_cadence", 500))

        if counter == 1 or counter % cadence == 0:
            print(
                "[moon orbit bracket] "
                f"call={counter} "
                f"t={t:.6e} "
                f"op={op:.12e} "
                f"nm={nm:.12e} "
                f"omega0m={K_m['omega0']:.12e} "
                f"K0m={K_m['K0']:.12e} "
                f"K1m={K_m['K1']:.12e} "
                f"K2m={K_m['K2']:.12e} "
                f"K5m={K_m['K5']:.12e} "
                f"eccm={eccm:.12e} "
                f"orbit_bracket_m={orbit_bracket_m:.12e} "
                f"Rpm={Rpm:.12e} "
                f"dnm_planet={dnm_planet:.12e} "
                f"dlognm_planet={dnm_planet / nm:.12e}"
            )

    dhm_planet = -(
        3.0 / 4.0
        * nm
        * em2
        * (Mm / Mp)
        * Rpm
        * ecc_bracket_m
    )

    dop_moon = -(3.0 / 8.0) * Tpm * spin_bracket_m

    # ------------------------------------------------------------------
    # Optional dissipation inside the moon. This contributes to the moon
    # orbit only; it does not directly torque the planet spin.
    # ------------------------------------------------------------------
    dnm_moon = 0.0
    dhm_moon = 0.0

    if _moon_dissipation_is_active(parameters, initial_conds, eccm=eccm):
        Rm, k2q_moon = _moon_radius_and_k2q(
            parameters,
            initial_conds,
            nm=nm,
        )
        chim = _moon_spin_obliquity(parameters, initial_conds)
        S_chi2 = np.sin(chim) ** 2

        if Rm > 0.0 and k2q_moon != 0.0:
            mkep_mp = _kepler_mass("moon", Mp, Mm, parameters)
            Rmp = _radius_factor(Rm, nm, mkep_mp)

            dnm_moon = (
                9.0 / 2.0
                * nm**2
                * (Mp / Mm)
                * Rmp
                * (7.0 * em2 + S_chi2)
                * k2q_moon
            )

            dhm_moon = -(
                21.0
                * nm
                * em2
                * (Mp / Mm)
                * Rmp
                * k2q_moon
            )

    dnm_total = dnm_planet + dnm_moon
    dhm_total = dhm_planet + dhm_moon

    if eccm > 0.0:
        dem_total = 0.5 * dhm_total / eccm
    else:
        dem_total = 0.0

    # ------------------------------------------------------------------
    # Planet-star terms.
    #
    # Important regression rule:
    # if the planet-star orbit is circular and aligned, use the original
    # circular stellar branch explicitly. This guarantees that ep=0
    # reproduces the pre-planet-eccentricity model.
    #
    # If eccp > 0 or psi_star != 0, use the generic Ferraz-Mello branch.
    # ------------------------------------------------------------------
    dnp_star = 0.0
    dhp_star = 0.0
    dep_star = 0.0
    dop_star = 0.0

    if Ms > 0.0 and npp > 0.0:
        psis = _stellar_obliquity(parameters, initial_conds)
        Ss2 = np.sin(psis) ** 2

        planet_eccentric_star_branch = eccp > 0.0
        oblique_star_branch = Ss2 > 0.0

        mkep_s = _kepler_mass("star", Mp, Ms, parameters)
        Rps = _radius_factor(Rp, npp, mkep_s)
        Tps = _spin_torque_factor("star", Ms, npp, Mp, Rp, GR, parameters)

        if not planet_eccentric_star_branch and not oblique_star_branch:
            # Exact old circular, aligned planet-star tide.
            omega0s = 2.0 * op - 2.0 * npp

            K0s = _signed_planet_k2q_for_harmonic(
                t,
                op,
                omega0s,
                npp,
                integrator_args,
                smooth=(extended_active or _smooth_tidal_signs(parameters)),
                body='star',
            )

            orbit_bracket_s = 4.0 * K0s
            spin_bracket_s = 4.0 * K0s

        else:
            # Generic eccentric/oblique planet-star tide.
            orbit_bracket_s, ecc_bracket_s, spin_bracket_s, K_s = (
                _planet_tide_harmonics_and_brackets(
                    t,
                    op,
                    npp,
                    eccp,
                    psis,
                    integrator_args,
                    extended_active=extended_active,
                    body='star',
                )
            )

            hp = eccp**2

            dhp_star = -(
                3.0 / 4.0
                * npp
                * hp
                * (Ms / Mp)
                * Rps
                * ecc_bracket_s
            )

            if eccp > 0.0:
                dep_star = 0.5 * dhp_star / eccp

        dnp_star = -(
            9.0 / 8.0
            * npp**2
            * (Ms / Mp)
            * Rps
            * orbit_bracket_s
        )

        dop_star = -(3.0 / 8.0) * Tps * spin_bracket_s
    # ------------------------------------------------------------------
    # Obliquity between the planet spin and moon orbital plane.
    # This vanishes identically if psim=0.
    # ------------------------------------------------------------------
    dpsim = 0.0
    if abs(op) > 0.0:
        Lambda_m = _lambda_obliquity("moon", nm, Mm, op, Mp, Rp, GR, parameters)
        dpsim = (
            3.0 / 4.0
            * (Tpm / op)
            * np.sin(psim)
            * ((1.0 - Lambda_m) * K0m + (1.0 + Lambda_m) * K8m - K9m)
        )

    # ------------------------------------------------------------------
    # Structural spin-up/down from Ip(t). This is tied to the existing
    # planet_evolution flag: if Rp(t) evolves, Ip changes as alpha_p Mp Rp^2.
    # For now alpha_p and Mp are held fixed, so dlnIp/dt = 2 dlnRp/dt.
    # ------------------------------------------------------------------
    dlogI_dt = _planet_moment_inertia_log_derivative(t, op, integrator_args)
    dop_inertia = -op * dlogI_dt

    dop_total = dop_moon + dop_star + dop_inertia

    # omega0s = 2.0 * (op - npp)
    # print("omega0s =", omega0s, "sign =", np.sign(omega0s), "K0s =", K0s)

    return {
        "dopdt": dop_total,
        "dnpdt": dnp_star,
        "dnmdt": dnm_total,
        "dlognmdt": dnm_total / nm,
        "demdt": dem_total,
        "dhmdt": dhm_total,
        "depdt": dep_star,
        "dhpdt": dhp_star,
        "dpsimdt": dpsim,
        "components": {
            "dop_moon": dop_moon,
            "dop_star": dop_star,
            "dop_inertia": dop_inertia,
            "dnm_planet": dnm_planet,
            "dnm_moon": dnm_moon,
            "dhm_planet": dhm_planet,
            "dhm_moon": dhm_moon,
            "dem_total": dem_total,
            "dnp_star": dnp_star,
            "dhp_star": dhp_star,
            "dep_star": dep_star,
        },
    }


#############################################################
# DIFFERENTIAL EQUATIONS
#############################################################
def dnmdt(t, y, integrator_args, initial_conds):
    """Differential equation for log(nm).

    This wrapper is kept for compatibility with the original file. The
    actual physics is assembled in _rhs_components().
    """
    y = np.asarray(y, dtype=float)
    parameters = integrator_args["parameters"]

    op = parameters["op"]
    npp = parameters.get("npp", parameters.get("np", 0.0))
    log_nm = y[0]
    eccm = parameters.get("eccm", 0.0)
    psim = parameters.get("psim", 0.0)
    eccp = parameters.get("eccp", 0.0)

    rhs = _rhs_components(
        t,
        op,
        npp,
        log_nm,
        eccm,
        psim,
        eccp,
        integrator_args,
        initial_conds,
    )
    return [rhs["dlognmdt"]]


def demdt(t, y, integrator_args, initial_conds):
    """Differential equation for the physical moon eccentricity em.

    This compatibility wrapper still returns dem/dt if called directly.
    The coupled solver below evolves hm = em**2 instead.
    """
    y = np.asarray(y, dtype=float)
    parameters = integrator_args["parameters"]

    op = parameters["op"]
    npp = parameters["npp"]
    log_nm = parameters["nm"]
    eccm = y[0]
    psim = parameters.get("psim", 0.0)
    eccp = parameters.get("eccp", 0.0)

    rhs = _rhs_components(
        t,
        op,
        npp,
        log_nm,
        eccm,
        psim,
        eccp,
        integrator_args,
        initial_conds,
    )
    return [rhs["demdt"]]


def dhmdt(t, y, integrator_args, initial_conds):
    """Differential equation for hm = em**2.

    This is the eccentricity variable used by solution_planet_moon() when
    eccentricity is active. The physical eccentricity is recovered as
    em = sqrt(max(hm, 0)).
    """
    y = np.asarray(y, dtype=float)
    parameters = integrator_args["parameters"]

    op = parameters["op"]
    npp = parameters["npp"]
    log_nm = parameters["nm"]
    hm = max(y[0], 0.0)
    eccm = np.sqrt(hm)
    psim = parameters.get("psim", 0.0)
    eccp = parameters.get("eccp", 0.0)

    rhs = _rhs_components(
        t,
        op,
        npp,
        log_nm,
        eccm,
        psim,
        eccp,
        integrator_args,
        initial_conds,
    )
    return [rhs["dhmdt"]]


def dpsimdt(t, y, integrator_args, initial_conds):
    """Differential equation for the planet-moon obliquity psim."""
    y = np.asarray(y, dtype=float)
    parameters = integrator_args["parameters"]

    op = parameters["op"]
    npp = parameters["npp"]
    log_nm = parameters["nm"]
    eccm = parameters.get("eccm", 0.0)
    psim = y[0]
    eccp = parameters.get("eccp", 0.0)

    rhs = _rhs_components(
        t,
        op,
        npp,
        log_nm,
        eccm,
        psim,
        eccp,
        integrator_args,
        initial_conds,
    )
    return [rhs["dpsimdt"]]


def dopdt(t, y, integrator_args, initial_conds):
    """Differential equation for the planet rotational rate Omega_p."""
    y = np.asarray(y, dtype=float)
    parameters = integrator_args["parameters"]

    op = y[0]
    npp = parameters["npp"]
    log_nm = parameters["nm"]
    eccm = parameters.get("eccm", 0.0)
    psim = parameters.get("psim", 0.0)
    eccp = parameters.get("eccp", 0.0)

    rhs = _rhs_components(
        t,
        op,
        npp,
        log_nm,
        eccm,
        psim,
        eccp,
        integrator_args,
        initial_conds,
    )
    return [rhs["dopdt"]]


def dnpdt(t, y, integrator_args, initial_conds):
    """Differential equation for the planet mean motion n_p."""
    y = np.asarray(y, dtype=float)
    parameters = integrator_args["parameters"]

    op = parameters["op"]
    npp = y[0]
    log_nm = parameters["nm"]
    eccm = parameters.get("eccm", 0.0)
    psim = parameters.get("psim", 0.0)
    eccp = parameters.get("eccp", 0.0)

    rhs = _rhs_components(
        t,
        op,
        npp,
        log_nm,
        eccm,
        psim,
        eccp,
        integrator_args,
        initial_conds,
    )
    return [rhs["dnpdt"]]


def finite_difference_jacobian(
    t,
    y,
    integrator_args,
    initial_conds,
    rel_step=None,
):
    """
    Central finite-difference Jacobian of solution_planet_moon.

    This is mainly useful near sign discontinuities, where the
    analytic Jacobian is not a good local linearisation.
    """
    y = np.asarray(y, dtype=float)
    n = y.size

    if rel_step is None:
        rel_step = np.sqrt(np.finfo(float).eps)

    def f(yy):
        return np.asarray(
            solution_planet_moon(t, yy, integrator_args, initial_conds),
            dtype=float,
        )

    J = np.zeros((n, n), dtype=float)

    for j in range(n):
        # State-aware perturbation.
        #
        # For op and npp, the values can be very small in SI units,
        # so do not use max(abs(y[j]), 1.0), otherwise the perturbation
        # would be absurdly large.
        #
        # For log(nm), y[j] is order ~ -10 to -5, so this is fine.
        scale = max(abs(y[j]), 1.0) if j == 2 else max(abs(y[j]), 1e-30)
        h = rel_step * scale

        yp = y.copy()
        ym = y.copy()

        yp[j] += h
        ym[j] -= h

        J[:, j] = (f(yp) - f(ym)) / (2.0 * h)

    return J


def near_synchronization(y, sync_rtol=1e-8):
    """
    Detect whether the system is close to a sign-changing tidal surface.

    State vector:
        y[0] = op
        y[1] = npp
        y[2] = log(nm)
    """
    op = y[0]
    npp = y[1]
    nm = np.exp(y[2])

    scale_pm = max(abs(op), abs(nm), 1e-30)
    scale_ps = max(abs(op), abs(npp), 1e-30)

    close_pm = abs(nm - op) <= sync_rtol * scale_pm
    close_ps = abs(npp - op) <= sync_rtol * scale_ps

    return close_pm or close_ps


def jacobian(t, y, integrator_args, initial_conds):
    """
    Jacobian for the circular planet-moon system.

    State vector:
        y[0] = op
        y[1] = npp
        y[2] = log(nm)

    RHS vector:
        f[0] = dop/dt
        f[1] = dnp/dt
        f[2] = dlog(nm)/dt
    """
    y = np.asarray(y, dtype=float)

    # ------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------
    parameters = integrator_args["parameters"]

    physics_flags = parameters["physics_flags"]
    planet_fixed_properties = parameters["planet_fixed_properties"]

    rigidity = parameters["rigidity"]
    GR = parameters['planet_GR_coeff']
    Mm = parameters["Mm"]
    # Ms is always available in the coupled solver, but keeping a safe
    # default avoids breaking standalone calls to dnmdt()/demdt().
    Ms = parameters.get("Ms", 0.0)
    planet_envelope_interface = planet_fixed_properties["envelope_interface"]

    if _smooth_tidal_signs(parameters):
        return finite_difference_jacobian(
            t,
            y,
            integrator_args,
            initial_conds,
        )

    # If eccentricity is active, the RHS is 4D but the current analytic
    # Jacobian is only 3D. Use finite differences until the full 4x4
    # analytic Jacobian is implemented.
    if len(y) != 3:
        return finite_difference_jacobian(
            t,
            y,
            integrator_args,
            initial_conds,
        )
    # Near synchronization, mimic the old solve_ivp behaviour:
    # use finite differences so the Jacobian samples the sign flip.
    if near_synchronization(y, sync_rtol=1e-8):
        return finite_difference_jacobian(
            t,
            y,
            integrator_args,
            initial_conds,
        )
    # ------------------------------------------------------------
    # Dynamic variables
    # ------------------------------------------------------------
    op = y[0]
    npp = y[1]
    nm = np.exp(y[2])

    # ------------------------------------------------------------
    # Planet properties
    # ------------------------------------------------------------
    if physics_flags["planet_evolution"]:
        planet_track = integrator_args["planet_track"]
        Mp = planet_track.M(t + 10 * MYEAR)
        Rp = planet_track.R(t + 10 * MYEAR)
    else:
        Rp = planet_fixed_properties["Rp"]
        Mp = planet_fixed_properties["Mp"]

    Mp_core = planet_fixed_properties["Mp_core"]
    Rp_core = planet_fixed_properties["Rp_core"]

    alpha_planet = Rp_core / Rp
    beta_planet = Mp_core / Mp
    epsilon = op / omegaCritic(Mp, Rp)

    # ------------------------------------------------------------
    # Dissipation
    # ------------------------------------------------------------
    if physics_flags["planet_energy_dissipation_fix_value"]:
        k2q_planet = planet_fixed_properties["planet_k2q"]
    else:
        k2q_planet_core = 0.0
        k2q_planet_mantle = 0.0
        k2q_planet_envelope = 0.0

        if physics_flags["planet_core_dissipation"]:
            k2q_planet_core = k2Q_planet_core(
                rigidity, alpha_planet, beta_planet, Mp, Rp
            )

        if physics_flags["planet_mantle_dissipation"]:
            k2q_planet_mantle = k2Q_planet_mantle(
                rigidity, alpha_planet, beta_planet, Mp, Rp
            )

        if physics_flags["planet_envelope_dissipation"]:
            k2q_planet_envelope = k2Q_inertial_waves_convective_envelope(
                alpha_planet,
                beta_planet,
                epsilon,
                interface=planet_envelope_interface,
            )

        k2q_planet = (
            k2q_planet_core
            + k2q_planet_mantle
            + k2q_planet_envelope
        )

    # ------------------------------------------------------------
    # Sign terms
    # ------------------------------------------------------------
    s_pm = np.sign(nm - op)
    s_ps = np.sign(npp - op)

    # ------------------------------------------------------------
    # Useful constants matching your RHS definitions
    # ------------------------------------------------------------
    C_spin = 3.0 / 2.0 * k2q_planet * Rp**3 / (GR * GCONST)

    C_np = (
        9.0 / 2.0
        * k2q_planet
        * Rp**5
        / (GCONST**(5.0 / 3.0) * Mp * Ms**(2.0 / 3.0))
    )

    C_lnnm = (
        9.0 / 2.0
        * k2q_planet
        * Mm
        * Rp**5
        / (GCONST**(5.0 / 3.0) * Mp**(8.0 / 3.0))
    )

    # ------------------------------------------------------------
    # Jacobian matrix
    #
    # Important:
    # derivatives of sign(...) are taken as zero away from
    # exact synchronization.
    # ------------------------------------------------------------
    J = np.zeros((3, 3), dtype=float)

    # Row 0: dop/dt
    # dop/dt = C_spin * [
    #     Mm^2 * nm^4 * sign(nm - op) / Mp^3
    #   + npp^4 * sign(npp - op) / Mp
    # ]
    J[0, 0] = 0.0

    J[0, 1] = (
        4.0
        * C_spin
        * npp**3
        * s_ps
        / Mp
    )

    # derivative wrt log(nm), not nm:
    # d(nm^4)/dlog(nm) = 4 nm^4
    J[0, 2] = (
        4.0
        * C_spin
        * Mm**2
        * nm**4
        * s_pm
        / Mp**3
    )

    # Row 1: dnp/dt
    # dnp/dt = C_np * npp^(16/3) * sign(npp - op)
    J[1, 0] = 0.0

    J[1, 1] = (
        16.0 / 3.0
        * C_np
        * npp**(13.0 / 3.0)
        * s_ps
    )

    J[1, 2] = 0.0

    # Row 2: dlog(nm)/dt
    # dlog(nm)/dt = C_lnnm * nm^(13/3) * sign(nm - op)
    J[2, 0] = 0.0
    J[2, 1] = 0.0

    # derivative wrt log(nm):
    # d(nm^(13/3))/dlog(nm) = (13/3) nm^(13/3)
    J[2, 2] = (
        13.0 / 3.0
        * C_lnnm
        * nm**(13.0 / 3.0)
        * s_pm
    )

    return J


#############################################################
# INTEGRATION OF THE WHOLE SYSTEM
#############################################################
def solution_planet_moon(t, y, integrator_args, initial_conds):
    """Coupled ODE system for the star--planet--moon tidal problem.

    Circular state vector:
        y[0] = op
        y[1] = npp
        y[2] = log(nm)

    Eccentric state vector:
        y[3] = hm = em**2

    Obliquity-only state vector:
        y[3] = psim

    Eccentric + obliquity state vector:
        y[3] = hm = em**2
        y[4] = psim

    The returned vector always matches len(y). The eccentric component
    is dhm/dt rather than dem/dt.
    """
    y = np.asarray(y, dtype=float)
    parameters = integrator_args["parameters"]

    state = _state_from_y(y, integrator_args, initial_conds)

    # Preserve the side-effect pattern used by the original implementation.
    # Some wrapper functions and external diagnostics expect these values to
    # be present in parameters during RHS evaluation.
    parameters["op"] = state["op"]
    parameters["npp"] = state["npp"]
    parameters["nm"] = state["log_nm"]
    parameters["eccm"] = state["eccm"]
    parameters["hm"] = state["hm"]
    parameters["psim"] = state["psim"]
    parameters["eccp"] = state["eccp"]
    parameters["hp"] = state["hp"]

    # Internal-only switch used by _tidal_sign(). This marks whether the
    # eccentric/oblique extension is active, but tidal-sign smoothing itself
    # is controlled separately by parameters["smooth_tidal_signs"].
    parameters["_extended_tides_active"] = (
        state["e_state_active"]
        or state["psi_state_active"]
        or state["planet_e_state_active"]
    )

    rhs = _rhs_components(
        t,
        state["op"],
        state["npp"],
        state["log_nm"],
        state["eccm"],
        state["psim"],
        state["eccp"],
        integrator_args,
        initial_conds,
    )

    solution = [rhs["dopdt"], rhs["dnpdt"], rhs["dlognmdt"]]

    if state["e_state_active"]:
        solution.append(rhs["dhmdt"])

    if state["psi_state_active"]:
        solution.append(rhs["dpsimdt"])

    if state["planet_e_state_active"]:
        solution.append(rhs["dhpdt"])

    # Fail loudly if the RHS becomes pathological. This is much easier to
    # debug than letting LSODA silently take tiny steps forever.
    if not np.all(np.isfinite(solution)):
        raise FloatingPointError(
            "Non-finite value in solution_planet_moon RHS. "
            f"t={t:g}, y={y}, rhs={solution}"
        )

    # ------------------------------------------------------------------
    # Optional diagnostics for the full ODE state and RHS.
    #
    # Controlled globally by:
    #     parameters["debug_ODEs_rhs"]
    # ------------------------------------------------------------------
    if parameters.get("debug_ODEs_rhs", False):
        counter = parameters.get("_rhs_debug_counter", 0) + 1
        parameters["_rhs_debug_counter"] = counter

        cadence = int(parameters.get("rhs_debug_cadence", 500))

        if counter == 1 or counter % cadence == 0:
            print(
                "[planet_moon RHS] "
                f"call={counter} "
                f"t={float(t):.6e} "
                f"state_size={y.size} "
                f"e_state_active={state['e_state_active']} "
                f"psi_state_active={state['psi_state_active']} "
                f"planet_e_state_active={state['planet_e_state_active']} "
                f"op={float(np.asarray(state['op'])):.6e} "
                f"np={float(np.asarray(state['npp'])):.6e} "
                f"nm={float(np.asarray(state['nm'])):.6e} "
                f"em={float(np.asarray(state['eccm'])):.6e} "
                f"hm={float(np.asarray(state['hm'])):.6e} "
                f"psim={float(np.asarray(state['psim'])):.6e} "
                f"ep={float(np.asarray(state['eccp'])):.6e} "
                f"hp={float(np.asarray(state['hp'])):.6e} "
                f"rhs={np.asarray(solution, dtype=float)}"
            )

    # Very important for LSODA/F2PY: the RHS dimension must always match
    # the state-vector dimension. Raise a normal Python error before the
    # Fortran callback can fail hard.
    if len(solution) != y.size:
        raise ValueError(
            "solution_planet_moon returned an RHS with length "
            f"{len(solution)}, but the state vector has length {y.size}. "
            "Check the initial-condition/state-vector layout: "
            "[op, npp, log_nm], optionally plus hm=em**2, psim, and hp=ep**2."
        )

    return solution
