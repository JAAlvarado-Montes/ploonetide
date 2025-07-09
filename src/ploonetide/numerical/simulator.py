import numpy as np
import warnings

from scipy.integrate._ivp.base import OdeSolver
from scipy.integrate import solve_ivp
from tqdm.auto import tqdm

__all__ = ['Variable', 'Simulation']


# === Monkey-patch OdeSolver to include tqdm progress bar ===

# Save original methods
_original_init = OdeSolver.__init__
_original_step = OdeSolver.step


# Define patched methods
def _patched_init(self, fun, t0, y0, t_bound, vectorized=True, support_complex=False):
    bar_format = '{desc}{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} steps | {elapsed}<{remaining}'
    self._pbar = tqdm(
        desc='Computing orbital evolution: ',
        bar_format=bar_format,
        total=t_bound - t0,
        initial=t0
    )
    self._last_t = t0

    _original_init(self, fun, t0, y0, t_bound, vectorized, support_complex)


def _patched_step(self):
    _original_step(self)

    delta_t = self.t - self._last_t
    self._pbar.update(delta_t)
    self._last_t = self.t

    if self.t >= self.t_bound:
        self._pbar.close()


# Apply patch
OdeSolver.__init__ = _patched_init
OdeSolver.step = _patched_step


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
        self.N_variables = len(variables)
        self.Ndim = self.N_variables
        self.quant_vec = np.concatenate([var.return_vec() for var in variables])

    def set_diff_eq(self, calc_diff_eqs, **kwargs):
        """
        Set the differential equation function.

        Args:
            calc_diff_eqs: Callable returning dy/dt
            **kwargs: Additional arguments passed to the function
        """
        self.calc_diff_eqs = calc_diff_eqs
        self.diff_eq_kwargs = kwargs

    def set_integration_method(self, method='RK45'):
        """
        Set the integration method.

        Args:
            method (str): One of ['RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA']
        """
        self.integration_method = method

    def run(self, t, dt, t0=0.0):
        """
        Run the simulation.

        Args:
            t (float): Final time
            dt (float): Timestep
            t0 (float): Initial time (default 0.0)
        """
        t_span = np.array([0.000001, t])  # Avoid t0=0 for stability
        t_eval = np.arange(t_span[0], t_span[1], dt)

        self.bar_fmt = '{desc}{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} steps | {elapsed}<{remaining}'

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            sols = solve_ivp(
                self.calc_diff_eqs,
                t_span,
                self.quant_vec,
                method=self.integration_method,
                vectorized=True,
                rtol=1e-20,
                min_step=1e-6,
                args=(self.diff_eq_kwargs,),
                t_eval=t_eval
            )

            self.history = sols.t, sols.y
