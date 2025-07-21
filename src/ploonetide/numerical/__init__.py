"""
Numerical integrators and simulation utilities for tidal evolution.

This subpackage includes the `Simulation` class and supporting tools
used to perform orbital integrations using `scipy.integrate.solve_ivp`.
"""

from .simulator import Variable, Simulation

__all__ = [
    "Variable",
    "Simulation",
]
