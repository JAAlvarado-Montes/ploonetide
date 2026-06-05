#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Planet evolution track (Option 1: precompute + interpolate) with TWO closures for
the radiative-atmosphere transit-radius contribution ("quadratic" and "local").

This module is intentionally *super-commented*. Every function has a docstring,
and in-line comments annotate physics, units, numerics, and implementation details.
It is designed to be dropped into a dynamics code (e.g., ploonetide) so your
orbital/tidal ODEs can query the planet's *time-dependent* mass M(t) and radius R(t)
without adding slow planetary state variables into the ODE itself.

PHYSICS SUMMARY (with references)
---------------------------------
1) Interior cooling & contraction (Lopez & Fortney 2014, ApJ 792, 1):
   We parameterize the *interior* (structural) radius R_int(t) with a soft power law:
       R_int(t) = R_inf + dR * ( 1 + (t - t0)/tau_cool )^(-alpha)
   This compactly reflects Kelvin–Helmholtz cooling and envelope shrinkage.

2) Core-powered mass loss (CPML) (Ginzburg, Schlichting & Sari 2018, MNRAS 476, 759;
   Gupta & Schlichting 2019/2020, MNRAS 487, 24; 493, 792):
       L_cool(t) = L0 * ( 1 + (t - t0)/tau_cool )^(-gamma)
       Mdot_cp   = eps_cp * L_cool / ( G M / R )
   We subtract the dominant of CPML and (optional) energy-limited photoevaporation
   from the H/He *envelope* mass. At K2-18b’s low flux the photoevap term is off
   by default.

3) Radiative-atmosphere "lift" to the *transit* radius (Tang et al. 2025, ApJ):
   The observable (transit) radius can exceed R_int by some number of scale
   heights N*H. We implement *two closures*, selectable at runtime:
   - "quadratic" (default; Tang-style): solve
         R = R_int + A * R^2 / (G M),  where A = N * k_B * T_eq / mu.
     This captures the geometry of line-of-sight optical depth in transit.
   - "local": use the local scale height evaluated at g(R_int):
         R = R_int + min( N * H_int, f_cap * R_int ),
         H_int = k_B T_eq / (mu g_int),  g_int = G M / R_int^2.
     This is a pragmatic alternative in low-Teq regimes to avoid over-inflation.
   In both modes we apply a *smooth cap* so the radiative contribution does not
   exceed f_rad_cap * R_int; Tang report it can be a large fraction (up to ~40%).

NUMERICS
--------
- We precompute a log-spaced time grid from t0 (~10 Myr, post-disk) to t_end
  (~3 Gyr for K2-18) and update the envelope mass forward with a stable explicit
  step (slow secular evolution). We then build **C1, shape-preserving** 1D
  interpolants (Fritsch–Carlson monotone cubic) for M(t), R_int(t), R_tr(t).

- NO SciPy dependency. Everything is NumPy + pure-Python.

DEFAULTS (tailored to K2-18b @ ~3 Gyr but still general)
--------------------------------------------------------
- conservative Tang cap (f_rad_cap ~ 0.28) and modest Nscale (~2.5) in quadratic mode
- gentle CPML (eps_cp ~ 0.02, L0 ~ 1.2e21 W)
- interior cooling tuned so R lands near ~2.37 REARTH by ~3 Gyr in quadratic mode
- mass anchor: choose M_p_today so that if the 5% envelope is lost, the final mass
  is ~8.92 MEARTH (accepted value for K2-18b).

License: MIT
================================================================================
"""

from __future__ import annotations

# -------------------------
# Standard library imports
# -------------------------
import math  # for sqrt, exp, log1p, etc. (dimensionless math)
# for type annotations (not strictly required)
from typing import Optional, Dict, Any

# -------------------------
# Third-party imports
# -------------------------
from ploonetide.utils.constants import *
import numpy as np  # vectorized numerics

# ============================================================================
# Monotone, C1 cubic interpolator (Fritsch–Carlson): no new extrema introduced
# ============================================================================


class MonotoneCubic1D:
    """
    Shape-preserving, *C1-continuous* interpolator for 1D data.
    - Ensures the interpolated curve does not introduce spurious wiggles
      (i.e., no new extrema beyond what's implied by data).
    - Ideal for slow monotone series like R_int(t) and (usually) M_tot(t).

    Parameters
    ----------
    x : array-like (1D, strictly increasing) [s]
        Independent variable; here: times in seconds.
    y : array-like (1D) [units of the quantity]
        Dependent variable; e.g., mass [kg] or radius [m].

    Returns
    -------
    callable
        f(xq) returns interpolated values at query points xq (scalar or array).
    """

    def __init__(self, x: np.ndarray, y: np.ndarray):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        # Sanity checks: 1D, same length, strictly increasing x
        if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
            raise ValueError("x and y must be 1D arrays of the same length.")
        if np.any(np.diff(x) <= 0):
            raise ValueError("x must be strictly increasing.")

        self.x = x  # [s]
        self.y = y  # [units of y]

        # Pre-compute slopes and basis data for the Hermite cubic
        self._prep_slopes()

    def _prep_slopes(self) -> None:
        """Compute endpoint derivatives using the Fritsch–Carlson recipe."""
        x, y = self.x, self.y
        n = len(x)
        h = np.diff(x)          # [s] widths of each interval
        delta = np.diff(y) / h  # [units_y / s] secant slope on each interval
        m = np.zeros_like(y)    # [units_y / s] derivatives at nodes

        # If at least one interval exists
        if len(delta) > 0:
            # Interior slopes: weighted average of neighboring secants
            m[1:-1] = (delta[:-1] * h[1:] + delta[1:]
                       * h[:-1]) / (h[:-1] + h[1:])

            # Endpoints: use edge secant
            m[0] = delta[0]
            m[-1] = delta[-1]

            # Enforce monotonicity: Fritsch–Carlson limiter
            for i in range(n - 1):
                if delta[i] == 0.0:
                    # Flat segment: set both derivatives to zero to avoid overshoot
                    m[i] = 0.0
                    m[i + 1] = 0.0
                else:
                    a = m[i] / delta[i]
                    b = m[i + 1] / delta[i]
                    s = a * a + b * b
                    if s > 9.0:
                        tau = 3.0 / math.sqrt(s)
                        m[i] = tau * a * delta[i]
                        m[i + 1] = tau * b * delta[i]

        # Cache for evaluation
        self.h = h       # [s]
        self.delta = delta
        self.m = m       # [units_y / s]

    def __call__(self, xq: np.ndarray | float) -> np.ndarray | float:
        """
        Evaluate the interpolant at query points xq (scalar or array).

        Parameters
        ----------
        xq : float or array-like [s]
            Query time(s) in seconds.

        Returns
        -------
        float or ndarray
            Interpolated values in the same units as self.y.
        """
        xq_arr = np.atleast_1d(xq).astype(float)
        x, y, h, m = self.x, self.y, self.h, self.m
        out = np.empty_like(xq_arr)

        for j, xv in enumerate(xq_arr):
            # Left extrapolation: linear with endpoint slope
            if xv <= x[0]:
                out[j] = y[0] + m[0] * (xv - x[0])
                continue

            # Right extrapolation: linear with endpoint slope
            if xv >= x[-1]:
                out[j] = y[-1] + m[-1] * (xv - x[-1])
                continue

            # Locate interval i such that x[i] <= xv < x[i+1]
            i = np.searchsorted(x, xv) - 1
            hi = h[i]                               # [s] interval width
            # dimensionless local coordinate in [0, 1]
            t = (xv - x[i]) / hi

            # Hermite cubic basis functions (dimensionless)
            h00 = (2 * t**3 - 3 * t**2 + 1)
            h10 = (t**3 - 2 * t**2 + t)
            h01 = (-2 * t**3 + 3 * t**2)
            h11 = (t**3 - t**2)

            # Interpolated value (units of y)
            out[j] = (h00 * y[i] + h10 * hi * m[i]
                      + h01 * y[i + 1] + h11 * hi * m[i + 1])

        # Return scalar if input was scalar
        return out[0] if np.isscalar(xq) else out

# ================================================================
# Smooth "minimum" used for the Tang cap (keeps derivatives C1)
# ================================================================


def smooth_min(a: float, b: float, k: float = 25.0) -> float:
    """
    Differentiable approximation to min(a, b) using a softplus-like function.

    Parameters
    ----------
    a, b : float
        The two quantities to compare (same units).
    k : float, optional
        Sharpness of the transition; larger k ≈ closer to hard min. Default 25.

    Returns
    -------
    float
        Smooth approximation of min(a, b), same units as a and b.

    Notes
    -----
    - Exact min() is non-differentiable where a=b; this smooth version avoids kinks.
    - Helpful for ODE solvers when a "cap" activates/deactivates across time.
    """
    # Formula: min(a,b) ≈ b - softplus(b-a)/k, where softplus(z) = ln(1+e^{k z})/k
    return float(b - math.log1p(math.exp(k * (b - a))) / k)

# ==================================================================
# Evolution parameter container (simple class, easy to override)
# ==================================================================


class EvolutionParams:
    """
    Container for planetary evolution parameters (SI units unless stated).
    All attributes are public for clarity and can be overridden via kwargs.

    Key switch
    ----------
    closure : {"quadratic", "local"}
        - "quadratic" (default): Tang-style closure R = R_int + A R^2 / (G M).
        - "local":               R = R_int + min(N H_int, f_cap R_int), with H at g(R_int).
      Both modes apply a *smooth cap* of f_rad_cap * R_int on the radiative contribution.

    K2-18b-tailored defaults
    ------------------------
    These defaults are conservative for K2-18b (low insolation), aiming to match the
    commonly accepted present-day radius near ~2.37 REARTH by ~3 Gyr in *quadratic* mode.
    """

    def __init__(self):
        # ----------------------
        # Mass anchor and split
        # ----------------------
        # M_p_today is a "book-keeping" anchor used to define the initial core and envelope.
        # If the envelope fraction f_env0 is later lost by CPML, the final mass tends to
        # (1 - f_env0) * M_p_today. For K2-18b, a commonly quoted present-day mass is
        # ~8.92 MEARTH. If f_env0 ≈ 0.05 and the envelope is almost gone by 3 Gyr,
        # set M_p_today ≈ 8.92 / (1-0.05) ≈ 9.3895 MEARTH.
        self.M_p_today: float = 8.63 * MEARTH  # [kg]

        # Initial H/He envelope fraction at t0 (dimensionless). Typical sub-Neptunes 1–10%.
        self.f_env0: float = 0.015

        # ----------------------
        # Radiative atmosphere
        # ----------------------
        # [K]  equilibrium temperature (K2-18b ~ Earth-like)
        self.Teq: float = 280.0
        # [kg] mean molecular mass for H/He mixture
        self.mu: float = 2.3 * mH
        # [dimensionless] effective number of scale heights
        self.Nscale: float = 2.5
        # [dimensionless] cap of radiative lift relative to R_int
        self.f_rad_cap: float = 0.11627428
        # {"quadratic","local"} transit-radius closure
        self.closure: str = "quadratic"

        # ----------------------
        # Interior cooling law
        # ----------------------
        # R_int(t) = R_inf + dR * (1 + ((t - t0)/tau_cool))^(-alpha)
        # [m] asymptotic interior radius at late times
        self.R_inf: float = 2.24 * REARTH
        self.dR: float = 0.53194175 * REARTH  # [m] initial excess above R_inf at t0
        self.alpha: float = 0.35148624            # [dimensionless] softness exponent
        # [s] characteristic cooling time
        self.tau_cool: float = 0.01090396 * GYEAR

        # ----------------------
        # Core-powered mass loss (CPML) scalings
        # ----------------------
        # L_cool(t) = L0 * (1 + (t - t0)/tau_cool)^(-gamma), then
        # Mdot_cp   = eps_cp * L_cool / ( G M / R )
        # [W]     normalization for cooling luminosity
        self.L0: float = 1.2e21
        self.gamma: float = 1.0             # [–]     slope of L_cool decay
        # [–]     CPML efficiency (0..1), small for K2-18b
        self.eps_cp: float = 0.18

        # ----------------------
        # Photoevaporation (optional; OFF by default for K2-18b)
        # ----------------------
        self.enable_photoevap: bool = False  # turn on only if exploring higher XUV
        # [–]     energy-limited efficiency
        self.eta_pe: float = 0.15
        # [s]     saturation time of XUV activity
        self.t_sat: float = 1.5 * GYEAR
        # [–]     L_XUV/L_bol during saturation
        self.LXUV_over_L: float = 3e-3
        # [–]     post-saturation power-law index
        self.beta_decay: float = 1.3
        # [AU]    semi-major axis (K2-18b ~33 d)
        self.a_AU: float = 0.1429
        # [W]     host star bolometric luminosity
        self.L_star: float = 0.03 * LSUN

        # ----------------------
        # Time grid for precomputation
        # ----------------------
        # [s] start time (post-disk dispersal)
        self.t0: float = 10.0 * MYEAR
        # [s] end time (age for K2-18b runs)
        self.t_end: float = 3.0 * GYEAR
        # [–] number of log-spaced time nodes
        self.n_steps: int = 600

        # ===== Auto-calibration options (all optional) =====
        # Final (present-day) targets interpreted at t_end.

        # --- Radius target at t_end, in Earth radii ---
        self.auto_calibrate_radius: bool = False
        self.target_R_today_Re: float = 2.37     # e.g., 2.61
        self.tol_R_today_Re: float = 5e-4               # tolerance in R_earth

        # --- Mass target at t_end, in Earth masses ---
        self.auto_calibrate_mass: bool = True
        self.target_M_today_Me: float = 8.63
        self.tol_M_today_Me: float = 5e-3               # tolerance in M_earth

        # Safety / performance knobs for the 1D root finders
        self.max_calib_iter: int = 30

# ======================================
# Helper physics functions (self-contained)
# ======================================


def FXUV_of_t(t: float, pars: EvolutionParams) -> float:
    """
    Crude XUV flux history at the planet's orbit (energy-limited photoevap).

    Parameters
    ----------
    t : float [s]
        System age in seconds.
    pars : EvolutionParams
        Evolution parameter container (contains stellar and orbital values).

    Returns
    -------
    float [W m^-2]
        Stellar XUV flux at the planet's orbital distance.

    Notes
    -----
    Saturation + power-law decay model:
      If t < t_sat: L_XUV = (Lxuv/Lbol) * L_star
      else:         L_XUV = (Lxuv/Lbol) * L_star * (t/t_sat)^(-beta_decay)
      FXUV = L_XUV / (4 pi a^2)
    """
    a_m = pars.a_AU * \
        AU                              # [m] semi-major axis in meters
    if t < pars.t_sat:
        # [W] saturated XUV luminosity
        Lxuv = pars.LXUV_over_L * pars.L_star
    else:
        Lxuv = pars.LXUV_over_L * pars.L_star * \
            (t / pars.t_sat) ** (-pars.beta_decay)
    return Lxuv / (4.0 * math.pi * a_m * a_m)         # [W m^-2]


def Lcool_of_t(t: float, pars: EvolutionParams) -> float:
    """
    Kelvin–Helmholtz-like cooling luminosity used in CPML.

    L_cool(t) = L0 * (1 + (t - t0)/tau_cool)^(-gamma)

    Parameters
    ----------
    t : float [s]
        Time since system formation (absolute age).
    pars : EvolutionParams
        Parameters containing L0, tau_cool, gamma, t0.

    Returns
    -------
    float [W]
        Cooling luminosity at time t.
    """
    x = max(0.0, (t - pars.t0) / pars.tau_cool)  # [–] dimensionless time since t0
    return pars.L0 * (1.0 + x) ** (-pars.gamma)  # [W]


def Rint_of_t(t: float, pars: EvolutionParams) -> float:
    """
    Interior/structural radius (up to the radiative-convective boundary/photosphere).

    R_int(t) = R_inf + dR * (1 + (t - t0)/tau_cool)^(-alpha)

    Parameters
    ----------
    t : float [s]
        Time since system formation (absolute age).
    pars : EvolutionParams
        Parameters containing R_inf, dR, alpha, tau_cool, t0.

    Returns
    -------
    float [m]
        Interior radius at time t.
    """
    x = max(0.0, (t - pars.t0) / pars.tau_cool)  # [–]
    return pars.R_inf + pars.dR * (1.0 + x) ** (-pars.alpha)  # [m]


def transit_radius(M: float, Rint: float, pars: EvolutionParams) -> float:
    """
    Compute the *observable* (transit) radius R including a Tang-style radiative
    atmosphere contribution with a smooth cap. Two closures are supported:

    A) Quadratic closure (default): "quadratic"
       Solve for R in:   R = Rint + A * R^2 / (G M)
       where A = N * k_B * T_eq / mu  [units: m^2 s^-2].
       This is close to the analytical perspective that a few scale heights of
       low-density atmosphere can dominate the transit chord optical depth.

    B) Local scale-height closure: "local"
       Compute H at g(Rint): H_int = k_B T_eq / (mu * g_int),
                             g_int = G M / Rint^2.
       Then R = Rint + min( N * H_int, f_cap * Rint ).
       This is pragmatic for low-Teq cases to avoid over-inflation without retuning.

    In both closures we *smoothly cap* the radiative lift to <= f_rad_cap * Rint.
    This avoids kinks in the derivative when the cap turns on/off.

    Parameters
    ----------
    M : float [kg]
        Planet total mass at this instant.
    Rint : float [m]
        Interior/structural radius (up to RCB/photosphere).
    pars : EvolutionParams
        Parameter container. Uses: Nscale, Teq, mu, f_rad_cap, closure.

    Returns
    -------
    float [m]
        Transit/observable radius R at this instant.
    """
    # Precompute A = N * kB * Teq / mu  [units: (J/K * K) / kg = m^2 s^-2]
    A = pars.Nscale * kB * pars.Teq / pars.mu

    # --------------- Mode A: Tang-style quadratic closure ----------------
    if pars.closure.lower() == "quadratic":
        # We need to solve:  aQ*R^2 + bQ*R + cQ = 0, with
        # aQ = A/(G M) [1/m],  bQ = -1,  cQ = Rint [m].
        aQ = A / (GCONST * M)           # [1/m]
        bQ = -1.0                  # [–]
        cQ = Rint                  # [m]
        # Discriminant (dimensionless): b^2 - 4 a c
        disc = bQ * bQ - 4.0 * aQ * cQ

        if disc >= 0.0:
            # Positive root gives the larger radius (physical transit radius)
            R_unc = (1.0 + math.sqrt(disc)) / (2.0 * aQ)  # [m]
        else:
            # Rare fallback if rounding makes disc<0: fixed-point iteration
            R_unc = Rint  # start at interior radius
            for _ in range(50):
                R_new = Rint + (A * R_unc * R_unc) / (GCONST * M)
                if abs(R_new - R_unc) < 1e-8 * R_unc:
                    R_unc = R_new
                    break
                R_unc = R_new

        # Effective "lift" from atmosphere (before capping)
        H_eff = R_unc - Rint  # [m]
        # Smooth cap to enforce H_eff <= f_rad_cap * Rint, with C1 continuity
        H_cap = smooth_min(H_eff, pars.f_rad_cap * Rint, k=25.0)  # [m]
        return Rint + H_cap  # [m]

    # --------------- Mode B: local scale-height closure ------------------
    elif pars.closure.lower() == "local":
        # Compute local gravity at Rint
        g_int = GCONST * M / (Rint * Rint)                # [m s^-2]
        # Local scale height (ideal gas, isothermal atmosphere at Teq)
        H_int = kB * pars.Teq / (pars.mu * g_int)    # [m]
        # Candidate atmospheric "lift"
        H_eff = pars.Nscale * H_int                  # [m]
        # Smooth cap (keeps derivative continuity)
        H_cap = smooth_min(H_eff, pars.f_rad_cap * Rint, k=25.0)  # [m]
        return Rint + H_cap  # [m]

    else:
        raise ValueError(f"Unknown closure: {pars.closure} (use 'quadratic' or 'local')")


def mdot_cpml(t: float, M: float, R: float, pars: EvolutionParams) -> float:
    """
    Core-powered mass-loss rate (CPML) in kg/s.

    Mdot_cp = eps_cp * L_cool / ( G M / R )

    Parameters
    ----------
    t : float [s]
        Time at which to evaluate the rate.
    M : float [kg]
        Instantaneous *total* planet mass.
    R : float [m]
        Radius used for the escape energy scale (here we use transit radius).
    pars : EvolutionParams
        Parameters including eps_cp and Lcool_of_t configuration.

    Returns
    -------
    float [kg s^-1]
        Core-powered mass-loss rate at time t.

    References
    ----------
    - Ginzburg, Schlichting & Sari (2018), MNRAS, 476, 759
    - Gupta & Schlichting (2019, 2020), MNRAS 487, 24; 493, 792
    """
    Lcool = Lcool_of_t(t, pars)  # [W]
    # [m^2 s^-2] gravitational potential per unit mass
    Phi_g = GCONST * M / R
    return pars.eps_cp * Lcool / Phi_g  # [kg s^-1]


def mdot_photoevap(t: float, M: float, R: float, pars: EvolutionParams) -> float:
    """
    Energy-limited photoevaporation rate in kg/s (often negligible for K2-18b).

    Parameters
    ----------
    t : float [s]
        Time at which to evaluate the rate.
    M : float [kg]
        Instantaneous total planet mass.
    R : float [m]
        Effective absorption radius for XUV (we use R as an upper bound).
    pars : EvolutionParams
        Parameters including FXUV history and eta_pe.

    Returns
    -------
    float [kg s^-1]
        Energy-limited photoevaporation rate at time t (upper bound).

    Notes
    -----
    Formula used (with K_tide ~ 1 for this regime):
        Mdot_pe = eta_pe * pi * R^3 * FXUV / (G * M)
    """
    if not pars.enable_photoevap:
        return 0.0
    FXUV = FXUV_of_t(t, pars)              # [W m^-2]
    # [–] ~1 for our regime (neglect Roche effects)
    K_tide = 1.0
    return pars.eta_pe * math.pi * R**3 * FXUV / (GCONST * M * K_tide)  # [kg s^-1]

# ============================================================
# Track container: precomputed arrays + smooth interpolants
# ============================================================


class PlanetEvolutionTrack:
    """
    Container for the precomputed time series and smooth, C1 interpolants.

    Attributes
    ----------
    t     : ndarray [s]         time grid (log-spaced)
    M_tot : ndarray [kg]        total planet mass at each time
    R_int : ndarray [m]         interior radius at each time
    R_tot : ndarray [m]         transit (observable) radius at each time

    Interpolants (callables)
    ------------------------
    M(tq)    -> mass [kg] at time tq [s]
    R(tq)    -> transit radius [m] at time tq [s]
    Rint(tq) -> interior radius [m] at time tq [s]
    """

    def __init__(
        self, t: np.ndarray, M_tot: np.ndarray, M_int: np.ndarray,
        R_int: np.ndarray, R_tot: np.ndarray
    ):
        # Store arrays (as float copies to be safe)
        self.t = np.asarray(t, dtype=float)     # [s]
        self.M_tot = np.asarray(M_tot, dtype=float)  # [kg]
        self.M_int = np.asarray(M_int, dtype=float)  # [kg]
        self.R_int = np.asarray(R_int, dtype=float)  # [m]
        self.R_tot = np.asarray(R_tot, dtype=float)  # [m]

        # Build C1, shape-preserving interpolants for each series
        self._M_interp = MonotoneCubic1D(self.t, self.M_tot)  # M(t)
        self._Mint_interp = MonotoneCubic1D(self.t, self.M_int)  # M_int(t)
        self._Rint_interp = MonotoneCubic1D(self.t, self.R_int)  # R_int(t)
        self._R_interp = MonotoneCubic1D(self.t, self.R_tot)  # R_tr(t)

    # -------------- public API used inside your ODE RHS --------------
    def M(self, t_query: float | np.ndarray) -> float | np.ndarray:
        """Return total planet mass [kg] at query time(s) t_query [s]."""
        return self._M_interp(t_query)

    def R(self, t_query: float | np.ndarray) -> float | np.ndarray:
        """Return transit/observable radius [m] at query time(s) t_query [s]."""
        return self._R_interp(t_query)

    def Rint(self, t_query: float | np.ndarray) -> float | np.ndarray:
        """Return interior/structural radius [m] at query time(s) t_query [s]."""
        return self._Rint_interp(t_query)

    def Mint(self, t_query: float | np.ndarray) -> float | np.ndarray:
        """Return interior/structural mass [kg] at query time(s) t_query [s]."""
        return self._Mint_interp(t_query)

    # -------------- convenience export --------------
    def save_csv(self, path: str) -> None:
        """
        Save the track to CSV including helpful unit conversions.
        Columns:
            t_s, t_Myr, t_Gyr, M_tot_kg, M_tot_Mearth, R_int_m, R_int_Rearth, R_tr_m, R_tr_Rearth
        """
        import csv  # standard library CSV writer
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_s", "t_Myr", "t_Gyr", "M_tot_kg", "M_tot_Mearth",
                        "M_int_kg", "M_int_Mearth", "R_int_m", "R_int_Rearth",
                        "R_tr_m", "R_tr_Rearth"])
            for i in range(len(self.t)):
                t = self.t[i]
                Mt = self.M_tot[i]
                Mi = self.M_int[i]
                Ri = self.R_int[i]
                Rt = self.R_tot[i]
                w.writerow([
                    f"{t:.6e}", f"{t / MYEAR:.6f}", f"{t / GYEAR:.6f}",
                    f"{Mt:.6e}", f"{Mt / MEARTH:.6f}",
                    f"{Mi:.6e}", f"{Mi / MEARTH:.6f}",
                    f"{Ri:.6e}", f"{Ri / REARTH:.6f}",
                    f"{Rt:.6e}", f"{Rt / REARTH:.6f}"
                ])

# =============================================
# Track builder (explicit, slow-evolution step)
# =============================================


def _build_track_core(pars: EvolutionParams) -> PlanetEvolutionTrack:
    """
    DO NOT call this directly from user code.
    This is the original 'build_track' logic with NO calibration,
    used internally by the calibration wrapper.

    Build a planetary evolution track on a log-time grid.

    Algorithm per grid node t_i
    ---------------------------
    1) Compute interior radius R_int(t_i) from the cooling law.
    2) Compute transit radius R_tr(t_i) from chosen closure (+ smooth cap).
    3) Compute mass-loss rates Mdot_cp and (optionally) Mdot_pe; update M_env.
    4) Store M_tot, R_int, R_tr for the output arrays.

    Notes
    -----
    - This is an explicit forward step for the *envelope* mass only; evolution
      is secular (Myr–Gyr), so a dense log grid suffices.
    - If you enable violent regimes, you can switch to a semi-implicit update
      for M_env alone without changing the public API.
    """
    # --------------------------
    # Initialize masses at t0
    # --------------------------
    # [kg] core mass (assumed constant)
    M_core = (1.0 - pars.f_env0) * pars.M_p_today
    M_env = pars.f_env0 * pars.M_p_today           # [kg] initial envelope mass

    # --------------------------
    # Build time grid (log-spaced)
    # --------------------------
    # Use >=3 points to avoid edge cases
    # t = np.geomspace(pars.t0, pars.t_end, max(3, int(pars.n_steps)))  # [s]
    t = np.linspace(pars.t0, pars.t_end, 1000)  # [s]

    # Allocate arrays for time series
    M_tot = np.empty_like(t)  # [kg]
    M_int = np.empty_like(t)  # [kg]
    R_int = np.empty_like(t)  # [m]
    R_tr = np.empty_like(t)  # [m]

    # --------------------------
    # March forward over the grid
    # --------------------------
    for i, ti in enumerate(t):
        # (1) Interior radius from cooling law (Lopez & Fortney style)
        Rint_i = Rint_of_t(ti, pars)  # [m]

        # (2) Transit radius from chosen closure (+ cap)
        Rtr_i = transit_radius(M_core + M_env, Rint_i, pars)  # [m]

        # (3) Mass-loss rates at ti (kg/s)
        md_cp = mdot_cpml(ti, M_core + M_env, Rtr_i, pars)     # [kg s^-1]
        md_pe = mdot_photoevap(ti, M_core + M_env, Rtr_i, pars)  # [kg s^-1]
        # [kg s^-1] negative = mass loss
        md_env = -max(md_cp, md_pe)

        # Local time step [s]: neighbor spacing on log grid
        dt = (t[i + 1] - t[i]) if i == 0 else (t[i] - t[i - 1])

        # Update envelope mass; never allow negative mass
        M_env = max(0.0, M_env + md_env * dt)  # [kg]

        # Re-evaluate radii with updated total mass for consistent outputs
        Rint_i = Rint_of_t(ti, pars)                               # [m]
        Rtr_i = transit_radius(M_core + M_env, Rint_i, pars)      # [m]

        # (4) Store the series values
        M_tot[i] = M_core + M_env   # [kg]
        M_int[i] = M_core  # [kg]
        R_int[i] = Rint_i           # [m]
        R_tr[i] = Rtr_i            # [m]

    # Wrap arrays & interpolants into the track object
    return PlanetEvolutionTrack(t, M_tot, M_int, R_int, R_tr)


def calibrate_final_radius(R_target_Re, pars, max_iter=30, tol_Re=1e-3):
    """
    Adjust pars.R_inf so that the track's final transit radius at t_end equals R_target_Re (in Earth radii).
    Monotone bisection on R_inf (all other parameters held fixed).
    """
    # 1) Choose a bracket for R_inf that certainly spans the solution
    lo = 1.60 * REARTH    # small asymptote -> small final R
    hi = 2.40 * REARTH    # large asymptote  -> large final R

    def final_R_for(Rinf):
        pars.R_inf = Rinf
        tr = build_track(pars)
        R_end = float(tr.R(pars.t_end)) / REARTH  # in R_earth
        return R_end

    R_lo = final_R_for(lo)
    R_hi = final_R_for(hi)
    if not (R_lo <= R_target_Re <= R_hi):
        # If the target is not bracketed, widen the bracket or relax other knobs
        raise RuntimeError(f"Target {R_target_Re:.3f} R_earth not bracketed: "
                           f"R(lo={lo/REARTH:.2f})={R_lo:.3f}, R(hi={hi/REARTH:.2f})={R_hi:.3f}")

    # 2) Bisection
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        R_mid = final_R_for(mid)
        if abs(R_mid - R_target_Re) <= tol_Re:
            return mid  # pars.R_inf already set; done
        if R_mid < R_target_Re:
            lo, R_lo = mid, R_mid
        else:
            hi, R_hi = mid, R_mid

    # Return the last mid even if tol not met
    return mid


def _final_mass_Me_for(pars: EvolutionParams, M_p_today_trial: float) -> float:
    """
    Helper: set M_p_today to 'trial', build a *raw* track, return final total mass in M_earth.
    """
    old = pars.M_p_today
    try:
        pars.M_p_today = M_p_today_trial
        tr = _build_track_core(pars)
        return float(tr.M(pars.t_end) / MEARTH)
    finally:
        pars.M_p_today = old  # restore


def _final_radius_Re_for(pars: EvolutionParams, R_inf_trial: float) -> float:
    """
    Helper: set R_inf to 'trial', build a *raw* track, return final transit radius in R_earth.
    """
    old = pars.R_inf
    try:
        pars.R_inf = R_inf_trial
        tr = _build_track_core(pars)
        return float(tr.R(pars.t_end) / REARTH)
    finally:
        pars.R_inf = old  # restore


def _calibrate_mass_if_requested(pars: EvolutionParams) -> None:
    """
    If auto_calibrate_mass is True and a target is provided, adjust pars.M_p_today
    so that M(t_end) ~= target_M_today_Me (Earth masses). Uses a simple secant,
    robust for our smooth dependence on M_p_today.
    """
    if not (pars.auto_calibrate_mass and pars.target_M_today_Me is not None):
        return

    TOL = pars.tol_M_today_Me
    MAXI = pars.max_calib_iter
    target = float(pars.target_M_today_Me)

    # Initial guesses around current value (±10%)
    x0 = 0.9 * pars.M_p_today
    x1 = 1.1 * pars.M_p_today
    f0 = _final_mass_Me_for(pars, x0) - target
    f1 = _final_mass_Me_for(pars, x1) - target

    # If both on same side, widen gently
    widen = 0
    while (f0 * f1 > 0.0) and widen < 5:
        x0 *= 0.8
        x1 *= 1.2
        f0 = _final_mass_Me_for(pars, x0) - target
        f1 = _final_mass_Me_for(pars, x1) - target
        widen += 1

    # Secant iterations
    for _ in range(MAXI):
        if abs(f1 - f0) < 1e-14:
            break
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        f2 = _final_mass_Me_for(pars, x2) - target
        if abs(f2) <= TOL:
            pars.M_p_today = x2
            return
        x0, f0 = x1, f1
        x1, f1 = x2, f2

    # Fall back: choose the closer of {x0,x1}
    pars.M_p_today = x0 if abs(f0) < abs(f1) else x1


def _calibrate_radius_if_requested(pars: EvolutionParams) -> None:
    """
    If auto_calibrate_radius is True and a target is provided, adjust pars.R_inf
    so that R_tr(t_end) ~= target_R_today_Re (Earth radii). Uses bisection:
    R_tr(t_end) is monotone in R_inf for fixed other knobs.
    """
    if not (pars.auto_calibrate_radius and pars.target_R_today_Re is not None):
        return

    TOL = pars.tol_R_today_Re
    MAXI = pars.max_calib_iter
    target = float(pars.target_R_today_Re)

    # Start with a bracket around current R_inf (±20%), widen if needed.
    lo = 0.8 * pars.R_inf
    hi = 1.2 * pars.R_inf
    R_lo = _final_radius_Re_for(pars, lo)
    R_hi = _final_radius_Re_for(pars, hi)

    widen = 0
    while not (R_lo <= target <= R_hi) and widen < 8:
        # Expand symmetrically until it brackets the target
        span = hi - lo
        lo = max(0.5 * REARTH, lo - 0.5 * span)
        hi = hi + 0.5 * span
        R_lo = _final_radius_Re_for(pars, lo)
        R_hi = _final_radius_Re_for(pars, hi)
        widen += 1

    if not (R_lo <= target <= R_hi):
        raise RuntimeError(
            f"[radius calibration] Could not bracket target {target:.3f} R_earth "
            f"(R(lo={lo/REARTH:.2f})={R_lo:.3f}, R(hi={hi/REARTH:.2f})={R_hi:.3f}). "
            "Relax caps or Nscale, or widen the initial bracket."
        )

    # Bisection
    for _ in range(MAXI):
        mid = 0.5 * (lo + hi)
        R_mid = _final_radius_Re_for(pars, mid)
        if abs(R_mid - target) <= TOL:
            pars.R_inf = mid
            return
        if R_mid < target:
            lo, R_lo = mid, R_mid
        else:
            hi, R_hi = mid, R_mid

    # If not converged within MAXI, set best mid
    pars.R_inf = 0.5 * (lo + hi)


def build_track(pars: EvolutionParams) -> PlanetEvolutionTrack:
    """
    Public builder with optional *in-place* auto-calibration.

    Order:
      1) (optional) MASS calibration -> adjust M_p_today so M(t_end) matches target_M_today_Me
      2) (optional) RADIUS calibration -> adjust R_inf so R_tr(t_end) matches target_R_today_Re
      3) Build and return the final calibrated track

    Notes
    -----
    - Radius depends weakly on M_p_today; calibrating mass first reduces coupling.
    - Both calibrations rebuild temporary tracks internally; typical total iterations
      are small (O(10–20) builds).
    """
    # 1) mass (optional)
    _calibrate_mass_if_requested(pars)

    # 2) radius (optional)
    _calibrate_radius_if_requested(pars)

    # 3) final, calibrated track
    return _build_track_core(pars)


# ------------------------------------------------------------
# Convenience presets for "K2-18b-like" runs
# ------------------------------------------------------------


def build_default_k218_params(**overrides) -> EvolutionParams:
    """
    Create an EvolutionParams object with K2-18b-friendly defaults.
    You may override any public field via keyword arguments. Example:

        pars = build_default_k218_params(closure="local", Nscale=4.0, f_rad_cap=0.22)

    Returns
    -------
    EvolutionParams
        Parameter container ready to pass into build_track().
    """
    p = EvolutionParams()
    for k, v in overrides.items():
        if not hasattr(p, k):
            raise AttributeError(f"Unknown EvolutionParams field: {k}")
        setattr(p, k, v)
    return p


def build_default_k218_track(**overrides) -> PlanetEvolutionTrack:
    """
    Build a precomputed track using K2-18b defaults (overridable). Examples:

        # Tang-style quadratic closure (default)
        track = build_default_k218_track()

        # Local-scale-height closure
        track = build_default_k218_track(closure="local", Nscale=4.0, f_rad_cap=0.22)

    Returns
    -------
    PlanetEvolutionTrack
        Track object with M(t), R(t), Rint(t) interpolants.
    """
    return build_track(build_default_k218_params(**overrides))


# ------------------------------------------------------------
# Self-demo: run this module directly to write a CSV and print end-states
# ------------------------------------------------------------
if __name__ == "__main__":
    # Build with defaults (quadratic closure)
    pars = build_default_k218_params()
    track = build_track(pars)

    # Export a CSV with helpful unit conversions
    out_csv = "k2-18b_track_quadratic_default.csv"
    track.save_csv(out_csv)
    print("Wrote:", out_csv)

    # Show the end-point (by default, ~3 Gyr)
    t_final = pars.t_end
    print(f"At {t_final/GYEAR:.3f} Gyr: "
          f"M = {track.M(t_final)/MEARTH:.3f} MEARTH, "
          f"R_tr = {track.R(t_final)/REARTH:.3f} REARTH, "
          f"R_int = {track.Rint(t_final)/REARTH:.3f} REARTH")
