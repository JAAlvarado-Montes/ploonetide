"""This module defines Simulation class"""
import numpy as np
import warnings

from ploonetide.utils.constants import MYEAR

from scipy.integrate import solve_ivp

# from scipy.integrate._ivp.base import OdeSolver
# from tqdm.auto import tqdm

# === Monkey-patch OdeSolver to include tqdm progress bar ===

# Save original methods
# _original_init = OdeSolver.__init__
# _original_step = OdeSolver.step


# # Define patched methods
# def _patched_init(self, fun, t0, y0, t_bound, vectorized=True, support_complex=False, **kwargs):
#     progress_total = kwargs.pop('_progress_total', None)
#     total_steps = progress_total if progress_total is not None else int(np.ceil(t_bound - t0))

#     bar_format = '{desc}{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} steps | {elapsed}<{remaining}'
#     self._pbar = tqdm(
#         desc='Computing orbital evolution: ',
#         bar_format=bar_format,
#         total=total_steps,
#         initial=0
#     )
#     self._last_t = t0

#     _original_init(self, fun, t0, y0, t_bound, vectorized, support_complex, **kwargs)


# def _patched_step(self):
#     _original_step(self)

#     delta_t = self.t - self._last_t
#     self._pbar.update(delta_t)  # One step per call to step()
#     self._last_t = self.t

#     if self.t >= self.t_bound:
#         self._pbar.close()


# # Apply patch
# OdeSolver.__init__ = _patched_init
# OdeSolver.step = _patched_step


# === Variable and Simulation classes ===

class Variable:
    """Define a new variable for integration.

    Args:
        name (str): Name of the variable
        v_ini (float): Initial value
    """

    def __init__(self, name, v_ini):
        self.name = name
        self.v_ini = v_ini

    def return_vec(self) -> np.ndarray:
        return np.array([self.v_ini])


class Simulation:
    """Build and run a simulation.

    Args:
        variables (list): List of Variable instances
    """

    supported_integration_methods = (
        "RK45",
        "RK23",
        "DOP853",
        "Radau",
        "BDF",
        "LSODA",
    )

    def __init__(self, variables):
        self.variables = list(variables)
        if not self.variables:
            raise ValueError("Simulation requires at least one variable.")

        self.N_variables = len(self.variables)
        self.Ndim = self.N_variables
        self.quant_vec = np.concatenate(
            [var.return_vec() for var in self.variables]
        )
        self.set_integration_method()

    def set_diff_eq(self, calc_diff_eqs, params, ini_conds, events):
        """
        Set the differential equation function.

        Args:
            calc_diff_eqs: Callable returning dy/dt
            **kwargs: Additional arguments passed to the function
        """
        if not callable(calc_diff_eqs):
            raise TypeError("calc_diff_eqs must be callable.")

        self.calc_diff_eqs = calc_diff_eqs
        self.diff_eq_kwargs = params
        self.diff_eq_ini_conds = ini_conds
        self.events = events

    def set_integration_method(self, method='RK45'):
        """
        Set the integration method.

        Args:
            method (str): Integration method name.
        """
        method_lookup = {
            valid_method.lower(): valid_method
            for valid_method in self.supported_integration_methods
        }
        try:
            self.integration_method = method_lookup[method.lower()]
        except (AttributeError, KeyError):
            methods = ", ".join(self.supported_integration_methods)
            raise ValueError(
                f"integration method must be one of {methods}; got {method!r}."
            ) from None

    @staticmethod
    def _as_finite_float(name, value):
        """Return a finite scalar float or raise a clear ValueError."""
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a finite scalar.") from None

        if not np.isfinite(value):
            raise ValueError(f"{name} must be a finite scalar.")

        return value

    def _validate_run_inputs(self, t, dt, t0):
        """Validate integration inputs before calling solve_ivp."""
        if not hasattr(self, "calc_diff_eqs"):
            raise RuntimeError(
                "Differential equation must be configured with set_diff_eq() "
                "before run()."
            )

        t = self._as_finite_float("t", t)
        dt = self._as_finite_float("dt", dt)
        t0 = self._as_finite_float("t0", t0)

        if dt <= 0.0:
            raise ValueError("dt must be positive.")
        if t <= t0:
            raise ValueError("t must be greater than t0.")

        y0 = np.asarray(self.quant_vec, dtype=float)
        if y0.size == 0:
            raise ValueError("initial state vector must not be empty.")
        if not np.all(np.isfinite(y0)):
            raise ValueError(
                "initial state vector must contain finite values."
            )

        return t, dt, t0, y0

    @staticmethod
    def _build_absolute_tolerance(y0):
        """Build a tolerance vector with legacy defaults where possible."""
        atol = np.full(y0.size, 1e-8, dtype=float)
        legacy_tolerances = (1e-12, 1e-13, 1e-8)

        for idx, value in enumerate(legacy_tolerances[:y0.size]):
            atol[idx] = value

        return atol

    def run(self, t, dt, t0=0.0, jacobian=None):
        """
        Run the simulation.

        Args:
            t (float): Final time
            dt (float): Timestep
            t0 (float): Initial time (default 0.0)
        """
        t, dt, t0, y0 = self._validate_run_inputs(t, dt, t0)
        self.time_step = dt
        self.total_time = t
        t_span = np.array([t0, t])  # Avoid t0=0 for stability
        t_eval = np.arange(t_span[0], t_span[1], dt)

        self.bar_fmt = '{desc}{percentage:4.0f}%|{bar}|'\
            + ' {n_fmt}/{total_fmt} steps | {elapsed}<{remaining}'

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            atol = self._build_absolute_tolerance(y0)

            total_time = t_span[1] - t_span[0]
            max_step = min(5.0 * MYEAR, 0.005 * total_time)
            # max_step = 0.1 * MYEAR

            sols = solve_ivp(
                self.calc_diff_eqs,
                t_span,
                y0,
                method=self.integration_method,
                rtol=1e-6,
                atol=atol,
                args=(self.diff_eq_kwargs, self.diff_eq_ini_conds),
                t_eval=None,
                dense_output=True,
                _progress_total=len(t_eval),
                jac=jacobian
                if self.integration_method in ("Radau", "BDF", "LSODA")
                else None,
                max_step=max_step,
                events=self.events,
            )

            self.history = sols
