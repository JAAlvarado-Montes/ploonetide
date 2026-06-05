"""This module defines Simulation class"""
import numpy as np
import warnings

from ploonetide.utils.constants import *

from scipy.integrate._ivp.base import OdeSolver
from scipy.integrate import solve_ivp
from tqdm.auto import tqdm

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

    def __init__(self, variables):
        self.variables = variables
        self.N_variables = len(self.variables)
        self.Ndim = self.N_variables
        self.quant_vec = np.concatenate([var.return_vec() for var in self.variables])

    def set_diff_eq(self, calc_diff_eqs, params, ini_conds, events):
        """
        Set the differential equation function.

        Args:
            calc_diff_eqs: Callable returning dy/dt
            **kwargs: Additional arguments passed to the function
        """
        self.calc_diff_eqs = calc_diff_eqs
        self.diff_eq_kwargs = params
        self.diff_eq_ini_conds = ini_conds
        self.events = events

    def set_integration_method(self, method='RK45'):
        """
        Set the integration method.

        Args:
            method (str): One of ['RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA']
        """
        self.integration_method = method

    def run(self, t, dt, t0=0.0, jacobian=None):
        """
        Run the simulation.

        Args:
            t (float): Final time
            dt (float): Timestep
            t0 (float): Initial time (default 0.0)
        """
        self.time_step = dt
        self.total_time = t
        t_span = np.array([t0, t])  # Avoid t0=0 for stability
        t_eval = np.arange(t_span[0], t_span[1], dt)

        self.bar_fmt = '{desc}{percentage:4.0f}%|{bar}|'\
            + ' {n_fmt}/{total_fmt} steps | {elapsed}<{remaining}'

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            y0 = np.asarray(self.quant_vec, dtype=float)

            atol = np.full(y0.size, 1e-8, dtype=float)
            atol[0] = 1e-12   # Omega_p
            atol[1] = 1e-13   # n_p
            atol[2] = 1e-8    # log(n_m)

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
                jac=jacobian if self.integration_method in ("Radau", "BDF", "LSODA") else None,
                max_step=max_step,
                events=self.events,
            )

            self.history = sols
