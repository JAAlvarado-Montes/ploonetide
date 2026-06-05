"""Regression tests for planet-moon result post-processing."""

from types import SimpleNamespace

import astropy.units as u
import numpy as np
import pytest


def _planet_moon_simulation(monkeypatch, tmp_path, **overrides):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))

    from ploonetide import TidalSimulation

    params = dict(
        system_type="planet-moon",
        planet_orbperiod=20,
        moon_fractions=(0.5, 0.5, 0.0),
        moon_semimaxis=2.0,
        planet_evolution=False,
        planet_core_dissipation=False,
        verbose=False,
    )
    params.update(overrides)

    return TidalSimulation(**params)


def test_planet_moon_results_include_terminal_event_state(
    monkeypatch,
    tmp_path,
):
    """Event fates should store root-refined terminal states."""
    simulation = _planet_moon_simulation(
        monkeypatch,
        tmp_path,
        moon_eccentricity=0.0,
        moon_obliquity=0.0,
    )
    initial = simulation.initial_conds

    states = np.array(
        [
            [initial["op_ini"], 0.99 * initial["op_ini"]],
            [initial["np_ini"], 1.01 * initial["np_ini"]],
            [initial["log_nm_ini"], 0.99 * initial["log_nm_ini"]],
            [initial["hp_ini"], initial["hp_ini"]],
        ]
    )
    terminal_state = np.array(
        [
            0.98 * initial["op_ini"],
            1.02 * initial["np_ini"],
            0.98 * initial["log_nm_ini"],
            initial["hp_ini"],
        ]
    )

    simulation.history = SimpleNamespace(
        success=True,
        message="terminal event",
        t=np.array([0.0, 1.0]),
        y=states,
        t_events=[np.array([]), np.array([1.5])],
        y_events=[
            np.empty((0, states.shape[0])),
            terminal_state[np.newaxis, :],
        ],
    )

    simulation._store_planet_moon_results(["roche", "hill"])

    assert simulation.fate == "hill"
    assert simulation.fate_time.unit == u.Gyr
    assert simulation.solutions.index.name == "Simulation Step"
    assert list(simulation.solutions.columns) == [
        "Times",
        "Planet Omega",
        "Planet Mean Motion",
        "Moon Mean Motion",
        "Moon Semimajor Axis",
    ]
    assert simulation.solutions["Times"].to_list() == [0.0, 1.0, 1.5]
    assert simulation.solutions.iloc[-1]["Moon Mean Motion"] == pytest.approx(
        np.exp(terminal_state[2])
    )
    assert "Moon Eccentricity" not in simulation.solution_units


def test_planet_moon_solution_table_maps_eccentricity_and_obliquity(
    monkeypatch,
    tmp_path,
):
    """The public table should preserve moon eccentricity/obliquity mapping."""
    simulation = _planet_moon_simulation(
        monkeypatch,
        tmp_path,
        moon_eccentricity=0.2,
        moon_obliquity=5.0,
    )
    initial = simulation.initial_conds

    solutions = np.array(
        [
            [initial["op_ini"]],
            [initial["np_ini"]],
            [initial["log_nm_ini"]],
            [initial["hm_ini"]],
            [initial["psim_ini"]],
            [initial["hp_ini"]],
        ]
    )

    table = simulation._build_planet_moon_solution_table(
        np.array([0.0]),
        solutions,
    )
    units = simulation._build_planet_moon_solution_units()

    assert list(table.columns) == [
        "Times",
        "Planet Omega",
        "Planet Mean Motion",
        "Moon Mean Motion",
        "Moon Semimajor Axis",
        "Moon Pericentre",
        "Moon Apocentre",
        "Moon Eccentricity Squared",
        "Moon Eccentricity",
        "Moon Obliquity",
    ]
    assert table.iloc[0]["Moon Eccentricity Squared"] == pytest.approx(0.04)
    assert table.iloc[0]["Moon Eccentricity"] == pytest.approx(0.2)
    assert table.iloc[0]["Moon Obliquity"] == pytest.approx(
        initial["psim_ini"]
    )
    assert units["Moon Eccentricity"] == u.Unit("")
    assert units["Moon Surface Temperature"] == u.K
    assert units["Moon Obliquity"] == u.Unit("")
