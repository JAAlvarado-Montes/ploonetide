"""Event factories for Ploonetide integrations.

The planet--moon integrator usually stores the moon mean motion as
``log(nm)``.  The ``*_log`` event factories below therefore define the
Roche/Hill surfaces directly in log-mean-motion space for circular runs.

For eccentric moon orbits, the integration state stores
``hm = em**2``.  If the event factories receive ``hm_idx``, the event
surfaces are evaluated using the instantaneous pericentre/apocentre:

    Roche:  q = a_m (1 - e_m)
    Hill:   Q = a_m (1 + e_m)

This keeps the circular behaviour backward-compatible while making the
stability boundaries eccentric-orbit aware.
"""

from __future__ import annotations

import numpy as np

from ploonetide.utils.constants import GCONST, GYEAR
from ploonetide.utils.functions import mean2axis


__all__ = [
    "make_event_roche",
    "make_event_hill",
    "make_event_roche_log",
    "make_event_hill_log",
]


_DEFAULT_LOG_NM_INDEX = 2


def _validate_positive(name: str, value: float) -> float:
    """Return ``value`` as float after checking that it is positive."""
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a finite positive number; got {value!r}.")
    return value


def _validate_buffer(name: str, buffer: float, *, upper: float | None = None) -> float:
    """Validate an event buffer."""
    buffer = float(buffer)
    if not np.isfinite(buffer) or buffer < 0.0:
        raise ValueError(f"{name} must be finite and non-negative; got {buffer!r}.")
    if upper is not None and buffer >= upper:
        raise ValueError(f"{name} must be < {upper}; got {buffer!r}.")
    return buffer


def _mean_motion_from_semimajor_axis(a: float, planet_mass: float, moon_mass: float) -> float:
    """Keplerian mean motion for a planet--moon orbit."""
    mu = GCONST * (planet_mass + moon_mass)
    return float(np.sqrt(mu / a**3))


def _semimajor_axis_from_mean_motion(nm: float, planet_mass: float, moon_mass: float) -> float:
    """Keplerian semimajor axis for a planet--moon orbit."""
    mu = GCONST * (planet_mass + moon_mass)
    return float((mu / nm**2) ** (1.0 / 3.0))


def _eccentricity_from_state(y, hm_idx: int | None) -> float:
    """Recover ``em = sqrt(hm)`` from the event state vector.

    If ``hm_idx`` is None, the event is circular and ``em=0``. Tiny negative
    values of hm can occur from numerical interpolation close to zero, so they
    are clipped before taking the square root.
    """
    if hm_idx is None:
        return 0.0

    if hm_idx >= len(y):
        raise ValueError(
            f"hm_idx={hm_idx} was provided, but the state vector has length {len(y)}. "
            "For eccentric planet--moon runs, pass hm_idx=3 when "
            "y=[op, npp, log_nm, hm]."
        )

    hm = float(y[hm_idx])
    if not np.isfinite(hm):
        return 0.0

    return float(np.sqrt(max(hm, 0.0)))


def _event_debug_print_log(
    *,
    label: str,
    t: float,
    value: float,
    log_nm: float,
    log_nm_stop: float,
    debug: bool,
    probe_tol: float,
) -> None:
    """Optionally print compact diagnostics for circular log-space events."""
    if debug and abs(value) <= probe_tol:
        print(
            f"[{label} event probe] "
            f"t={t / GYEAR:.6e} Gyr, "
            f"value={value:.3e}, "
            f"log_nm={log_nm:.6e}, "
            f"log_nm_stop={log_nm_stop:.6e}"
        )


def _event_debug_print_orbit(
    *,
    label: str,
    t: float,
    value: float,
    a: float,
    e: float,
    boundary_radius: float,
    debug: bool,
    probe_tol: float,
) -> None:
    """Optionally print compact diagnostics for eccentric orbit events."""
    if debug and abs(value) <= probe_tol * boundary_radius:
        q = a * (1.0 - e)
        Q = a * (1.0 + e)
        print(
            f"[{label} event probe] "
            f"t={t / GYEAR:.6e} Gyr, "
            f"value={value:.3e} m, "
            f"a={a:.6e} m, "
            f"e={e:.6e}, "
            f"q={q:.6e} m, "
            f"Q={Q:.6e} m, "
            f"boundary={boundary_radius:.6e} m"
        )


def make_event_roche_log(
    planet_mass: float,
    moon_mass: float,
    roche_radius: float,
    *,
    idx: int = _DEFAULT_LOG_NM_INDEX,
    hm_idx: int | None = None,
    buffer: float = 0.0,
    terminal: bool = True,
    direction: float | None = None,
    debug: bool = False,
    probe_tol: float = 1e-8,
):
    """Create a terminal Roche-limit event for ``y[idx] = log(nm)``.

    Circular case, ``hm_idx=None``
        The event reproduces the original log-space condition and stops when
        ``a_m = (1 + buffer) R_Roche``.

    Eccentric case, ``hm_idx`` supplied
        The event stops when the moon's pericentre reaches the buffered Roche
        boundary,

        ``a_m (1 - e_m) = (1 + buffer) R_Roche``.

    A positive buffer stops slightly before the formal Roche limit.
    """
    planet_mass = _validate_positive("planet_mass", planet_mass)
    moon_mass = _validate_positive("moon_mass", moon_mass)
    roche_radius = _validate_positive("roche_radius", roche_radius)
    buffer = _validate_buffer("buffer", buffer)
    probe_tol = _validate_positive("probe_tol", probe_tol)

    q_stop = (1.0 + buffer) * roche_radius
    nm_stop = _mean_motion_from_semimajor_axis(q_stop, planet_mass, moon_mass)
    log_nm_stop = float(np.log(nm_stop))

    # Circular log-space value increases through zero during inward migration.
    # Eccentric physical-space value q - q_stop decreases through zero.
    event_direction = (+1.0 if hm_idx is None else -1.0) if direction is None else direction

    def event_roche(t, y, *args):
        log_nm = float(y[idx])

        if not np.isfinite(log_nm):
            return 1.0

        # Keep the original efficient log-space event for circular runs.
        if hm_idx is None:
            value = log_nm - log_nm_stop
            _event_debug_print_log(
                label="Roche",
                t=t,
                value=value,
                log_nm=log_nm,
                log_nm_stop=log_nm_stop,
                debug=debug,
                probe_tol=probe_tol,
            )
            return float(value)

        # Eccentric orbit: Roche disruption is controlled by pericentre.
        nm = float(np.exp(log_nm))
        a = _semimajor_axis_from_mean_motion(nm, planet_mass, moon_mass)
        e = _eccentricity_from_state(y, hm_idx)
        q = a * (1.0 - e)
        value = q - q_stop

        _event_debug_print_orbit(
            label="Roche",
            t=t,
            value=value,
            a=a,
            e=e,
            boundary_radius=q_stop,
            debug=debug,
            probe_tol=probe_tol,
        )
        return float(value)

    event_roche.terminal = terminal
    event_roche.direction = event_direction
    event_roche.event_name = "roche"
    event_roche.a_stop = q_stop
    event_roche.q_stop = q_stop
    event_roche.nm_stop = nm_stop
    event_roche.log_nm_stop = log_nm_stop
    event_roche.hm_idx = hm_idx

    return event_roche


def make_event_hill_log(
    planet_mass: float,
    moon_mass: float,
    hill_critical_radius: float,
    *,
    idx: int = _DEFAULT_LOG_NM_INDEX,
    hm_idx: int | None = None,
    buffer: float = 0.0,
    terminal: bool = True,
    direction: float | None = None,
    debug: bool = False,
    probe_tol: float = 1e-8,
):
    """Create a terminal Hill/escape event for ``y[idx] = log(nm)``.

    Circular case, ``hm_idx=None``
        The event reproduces the original log-space condition and stops when
        ``a_m = (1 - buffer) R_Hill,crit``.

    Eccentric case, ``hm_idx`` supplied
        The event stops when the moon's apocentre reaches the buffered Hill
        boundary,

        ``a_m (1 + e_m) = (1 - buffer) R_Hill,crit``.

    A positive buffer stops slightly before the formal Hill boundary.
    """
    planet_mass = _validate_positive("planet_mass", planet_mass)
    moon_mass = _validate_positive("moon_mass", moon_mass)
    hill_critical_radius = _validate_positive(
        "hill_critical_radius",
        hill_critical_radius,
    )
    buffer = _validate_buffer("buffer", buffer, upper=1.0)
    probe_tol = _validate_positive("probe_tol", probe_tol)

    Q_stop = (1.0 - buffer) * hill_critical_radius
    nm_stop = _mean_motion_from_semimajor_axis(Q_stop, planet_mass, moon_mass)
    log_nm_stop = float(np.log(nm_stop))

    # Both circular log-space and eccentric physical-space values cross zero
    # with negative direction during outward migration.
    event_direction = -1.0 if direction is None else direction

    def event_hill(t, y, *args):
        log_nm = float(y[idx])

        if not np.isfinite(log_nm):
            return 1.0

        # Keep the original efficient log-space event for circular runs.
        if hm_idx is None:
            value = log_nm - log_nm_stop
            _event_debug_print_log(
                label="Hill",
                t=t,
                value=value,
                log_nm=log_nm,
                log_nm_stop=log_nm_stop,
                debug=debug,
                probe_tol=probe_tol,
            )
            return float(value)

        # Eccentric orbit: escape/stability is controlled by apocentre.
        nm = float(np.exp(log_nm))
        a = _semimajor_axis_from_mean_motion(nm, planet_mass, moon_mass)
        e = _eccentricity_from_state(y, hm_idx)
        Q = a * (1.0 + e)
        value = Q_stop - Q

        _event_debug_print_orbit(
            label="Hill",
            t=t,
            value=value,
            a=a,
            e=e,
            boundary_radius=Q_stop,
            debug=debug,
            probe_tol=probe_tol,
        )
        return float(value)

    event_hill.terminal = terminal
    event_hill.direction = event_direction
    event_hill.event_name = "hill"
    event_hill.a_stop = Q_stop
    event_hill.Q_stop = Q_stop
    event_hill.nm_stop = nm_stop
    event_hill.log_nm_stop = log_nm_stop
    event_hill.hm_idx = hm_idx

    return event_hill


def make_event_roche(
    idx: int,
    use_log: bool,
    planet_mass: float,
    moon_mass: float,
    moon_roche_radius: float,
    *,
    hm_idx: int | None = None,
    buffer: float = 0.0,
    terminal: bool = True,
    direction: float | None = None,
    debug: bool = False,
    probe_tol: float = 1e-8,
):
    """Backward-compatible Roche event factory.

    If ``use_log=True``, this delegates to :func:`make_event_roche_log`.  If
    ``use_log=False``, the event assumes ``y[idx] = nm`` and converts the
    current mean motion to semimajor axis.

    If ``hm_idx`` is supplied, Roche disruption is checked at pericentre,
    ``a_m(1-e_m)``. Otherwise, the original circular condition is used.
    """
    if use_log:
        return make_event_roche_log(
            planet_mass,
            moon_mass,
            moon_roche_radius,
            idx=idx,
            hm_idx=hm_idx,
            buffer=buffer,
            terminal=terminal,
            direction=direction,
            debug=debug,
            probe_tol=probe_tol,
        )

    planet_mass = _validate_positive("planet_mass", planet_mass)
    moon_mass = _validate_positive("moon_mass", moon_mass)
    moon_roche_radius = _validate_positive("moon_roche_radius", moon_roche_radius)
    buffer = _validate_buffer("buffer", buffer)
    probe_tol = _validate_positive("probe_tol", probe_tol)

    q_stop = (1.0 + buffer) * moon_roche_radius
    event_direction = -1.0 if direction is None else direction

    def event_roche(t, y, *args):
        nm = float(y[idx])
        if not np.isfinite(nm) or nm <= 0.0:
            return 1.0

        a = float(mean2axis(nm, planet_mass, moon_mass))
        e = _eccentricity_from_state(y, hm_idx)
        q = a * (1.0 - e)
        value = q - q_stop

        _event_debug_print_orbit(
            label="Roche",
            t=t,
            value=value,
            a=a,
            e=e,
            boundary_radius=q_stop,
            debug=debug,
            probe_tol=probe_tol,
        )
        return float(value)

    event_roche.terminal = terminal
    event_roche.direction = event_direction
    event_roche.event_name = "roche"
    event_roche.a_stop = q_stop
    event_roche.q_stop = q_stop
    event_roche.hm_idx = hm_idx

    return event_roche


def make_event_hill(
    idx: int,
    use_log: bool,
    planet_mass: float,
    moon_mass: float,
    hill_crit: float,
    *,
    hm_idx: int | None = None,
    buffer: float = 0.0,
    terminal: bool = True,
    direction: float | None = None,
    debug: bool = False,
    probe_tol: float = 1e-8,
):
    """Backward-compatible Hill/escape event factory.

    If ``use_log=True``, this delegates to :func:`make_event_hill_log`.  If
    ``use_log=False``, the event assumes ``y[idx] = nm`` and converts the
    current mean motion to semimajor axis.

    If ``hm_idx`` is supplied, escape is checked at apocentre,
    ``a_m(1+e_m)``. Otherwise, the original circular condition is used.
    """
    if use_log:
        return make_event_hill_log(
            planet_mass,
            moon_mass,
            hill_crit,
            idx=idx,
            hm_idx=hm_idx,
            buffer=buffer,
            terminal=terminal,
            direction=direction,
            debug=debug,
            probe_tol=probe_tol,
        )

    planet_mass = _validate_positive("planet_mass", planet_mass)
    moon_mass = _validate_positive("moon_mass", moon_mass)
    hill_crit = _validate_positive("hill_crit", hill_crit)
    buffer = _validate_buffer("buffer", buffer, upper=1.0)
    probe_tol = _validate_positive("probe_tol", probe_tol)

    Q_stop = (1.0 - buffer) * hill_crit

    if hm_idx is None:
        # Preserve the original non-log Hill behaviour: value=a-a_stop,
        # direction=+1 for outward migration.
        event_direction = +1.0 if direction is None else direction
    else:
        # Eccentric Hill boundary: value=Q_stop-Q, direction=-1.
        event_direction = -1.0 if direction is None else direction

    def event_hill(t, y, *args):
        nm = float(y[idx])
        if not np.isfinite(nm) or nm <= 0.0:
            return 1.0

        a = float(mean2axis(nm, planet_mass, moon_mass))
        e = _eccentricity_from_state(y, hm_idx)

        if hm_idx is None:
            value = a - Q_stop
        else:
            Q = a * (1.0 + e)
            value = Q_stop - Q

        _event_debug_print_orbit(
            label="Hill",
            t=t,
            value=value,
            a=a,
            e=e,
            boundary_radius=Q_stop,
            debug=debug,
            probe_tol=probe_tol,
        )
        return float(value)

    event_hill.terminal = terminal
    event_hill.direction = event_direction
    event_hill.event_name = "hill"
    event_hill.a_stop = Q_stop
    event_hill.Q_stop = Q_stop
    event_hill.hm_idx = hm_idx

    return event_hill
