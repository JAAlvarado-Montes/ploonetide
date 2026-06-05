"""This module defines Simulation class"""
import numpy as np
import warnings

from ploonetide.utils.constants import MYEAR

from scipy.integrate import solve_ivp


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

    def _make_progress_bar(self):
        """Create a tqdm progress bar for a single integration."""
        from tqdm.auto import tqdm

        return tqdm(
            desc="Computing orbital evolution: ",
            total=100.0,
            initial=0.0,
            unit="%",
            bar_format=(
                "{desc}{percentage:4.0f}%|{bar}| "
                "{elapsed}<{remaining}"
            ),
        )

    @staticmethod
    def _wrap_rhs_with_progress(rhs, t_span, progress_bar):
        """Wrap an RHS function with throttled progress updates."""
        t_start, t_end = t_span
        total_time = t_end - t_start
        min_progress_step = 0.1
        progress_state = {"last": 0.0}

        def wrapped_rhs(current_t, y, *args):
            progress = (current_t - t_start) / total_time * 100.0
            progress = min(max(progress, 0.0), 100.0)
            delta = progress - progress_state["last"]

            if delta > 0.0 and (
                delta >= min_progress_step or progress >= 100.0
            ):
                progress_bar.update(delta)
                progress_state["last"] = progress

            return rhs(current_t, y, *args)

        return wrapped_rhs

    def run(self, t, dt, t0=0.0, jacobian=None, show_progress=False):
        """
        Run the simulation.

        Args:
            t (float): Final time
            dt (float): Timestep
            t0 (float): Initial time (default 0.0)
            show_progress (bool): Show a progress bar for the integration.
        """
        t, dt, t0, y0 = self._validate_run_inputs(t, dt, t0)
        self.time_step = dt
        self.total_time = t
        t_span = np.array([t0, t])  # Avoid t0=0 for stability

        progress_bar = None
        calc_diff_eqs = self.calc_diff_eqs
        if show_progress:
            progress_bar = self._make_progress_bar()
            calc_diff_eqs = self._wrap_rhs_with_progress(
                self.calc_diff_eqs,
                t_span,
                progress_bar,
            )

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            atol = self._build_absolute_tolerance(y0)

            total_time = t_span[1] - t_span[0]
            max_step = min(5.0 * MYEAR, 0.005 * total_time)
            # max_step = 0.1 * MYEAR

            try:
                sols = solve_ivp(
                    calc_diff_eqs,
                    t_span,
                    y0,
                    method=self.integration_method,
                    rtol=1e-6,
                    atol=atol,
                    args=(self.diff_eq_kwargs, self.diff_eq_ini_conds),
                    t_eval=None,
                    dense_output=True,
                    jac=jacobian
                    if self.integration_method in ("Radau", "BDF", "LSODA")
                    else None,
                    max_step=max_step,
                    events=self.events,
                )
            finally:
                if progress_bar is not None:
                    progress_bar.close()

            self.history = sols
