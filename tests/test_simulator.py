"""Tests for the generic numerical Simulation runner."""

import numpy as np
import pytest

from ploonetide.numerical.simulator import Simulation, Variable


def _decay_rhs(_time, state, _params, _initial_conditions):
    return -state


def _configured_simulation(initial_value=1.0):
    simulation = Simulation([Variable("x", initial_value)])
    simulation.set_diff_eq(_decay_rhs, params={}, ini_conds={}, events=None)
    return simulation


def test_simulation_uses_default_integration_method():
    """A configured Simulation should run without an explicit method call."""
    simulation = _configured_simulation()

    simulation.run(0.1, 0.01)

    assert simulation.integration_method == "RK45"
    assert simulation.history.success
    assert simulation.history.y.shape[0] == 1
    assert simulation.history.y[0, -1] == pytest.approx(np.exp(-0.1), rel=1e-4)


def test_integration_method_is_validated_and_normalized():
    """Accepted method names should be case-insensitive."""
    simulation = _configured_simulation()

    simulation.set_integration_method("lsoda")

    assert simulation.integration_method == "LSODA"


def test_invalid_integration_method_raises_clear_error():
    simulation = _configured_simulation()

    with pytest.raises(ValueError, match="integration method"):
        simulation.set_integration_method("rk4")


def test_run_requires_configured_differential_equation():
    simulation = Simulation([Variable("x", 1.0)])

    with pytest.raises(RuntimeError, match="set_diff_eq"):
        simulation.run(0.1, 0.01)


@pytest.mark.parametrize(
    "t, dt, t0, message",
    [
        (0.1, 0.0, 0.0, "dt must be positive"),
        (0.1, -0.01, 0.0, "dt must be positive"),
        (0.0, 0.01, 0.0, "t must be greater than t0"),
        (0.1, 0.01, 0.1, "t must be greater than t0"),
    ],
)
def test_run_validates_time_inputs(t, dt, t0, message):
    simulation = _configured_simulation()

    with pytest.raises(ValueError, match=message):
        simulation.run(t, dt, t0=t0)


def test_run_rejects_nonfinite_initial_state():
    simulation = _configured_simulation(np.nan)

    with pytest.raises(ValueError, match="finite values"):
        simulation.run(0.1, 0.01)


def test_simulation_requires_variables():
    with pytest.raises(ValueError, match="at least one variable"):
        Simulation([])
