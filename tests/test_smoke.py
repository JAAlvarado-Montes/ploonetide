"""Basic public API smoke tests."""

import importlib
import math

import pytest


def test_public_import_exposes_version_and_simulation(monkeypatch, tmp_path):
    """The top-level package should expose its version and main simulator."""
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))

    ploonetide = importlib.import_module("ploonetide")

    assert ploonetide.__version__
    assert ploonetide.TidalSimulation.__name__ == "TidalSimulation"


def test_valid_planet_moon_simulation_initializes(monkeypatch, tmp_path):
    """A documented planet-moon configuration should initialize cleanly."""
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))

    from ploonetide import TidalSimulation

    simulation = TidalSimulation(
        system_type="planet-moon",
        planet_orbperiod=20,
        moon_fractions=(0.5, 0.5, 0.0),
        moon_eccentricity=0.0,
        moon_semimaxis=2.0,
        planet_evolution=False,
        planet_core_dissipation=False,
        verbose=False,
    )

    assert simulation.system_type == "planet-moon"
    assert math.isfinite(simulation.moon_density.value)
    assert math.isfinite(simulation.moon_radius.value)
    assert simulation.moon_rigidity in {"fluid", "rigid"}


def test_missing_attributes_raise_attribute_error(monkeypatch, tmp_path):
    """Missing attributes should fail normally instead of returning strings."""
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))

    from ploonetide import TidalSimulation

    simulation = TidalSimulation(
        system_type="planet-moon",
        planet_orbperiod=20,
        moon_fractions=(0.5, 0.5, 0.0),
        moon_semimaxis=2.0,
        verbose=False,
    )

    with pytest.raises(AttributeError):
        _ = simulation.does_not_exist


def test_invalid_system_type_fails_early(monkeypatch, tmp_path):
    """Unsupported system types should raise a clear validation error."""
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))

    from ploonetide import TidalSimulation

    with pytest.raises(ValueError, match="system_type"):
        TidalSimulation(
            system_type="not-a-system",
            planet_orbperiod=20,
            moon_fractions=(0.5, 0.5, 0.0),
            moon_semimaxis=2.0,
            verbose=False,
        )


def test_planet_core_mass_remains_assignable(monkeypatch, tmp_path):
    """The duplicate property definition should not remove the setter."""
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))

    from ploonetide import TidalSimulation

    simulation = TidalSimulation(
        system_type="planet-moon",
        planet_orbperiod=20,
        moon_fractions=(0.5, 0.5, 0.0),
        moon_semimaxis=2.0,
        verbose=False,
    )

    simulation.planet_core_mass = 2.0

    assert simulation.planet_core_mass.value == 2.0


def test_star_planet_construction_does_not_require_moon(monkeypatch, tmp_path):
    """Star-planet setup should not be blocked by moon-specific inputs."""
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))

    from ploonetide import TidalSimulation

    simulation = TidalSimulation(
        system_type="star-planet",
        planet_orbperiod=20,
        verbose=False,
    )

    assert simulation.system_type == "star-planet"


def test_star_planet_run_is_explicitly_blocked(monkeypatch, tmp_path):
    """The pending star-planet revision should be visible at run time."""
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))

    from ploonetide import TidalSimulation
    from ploonetide.utils.constants import YEAR

    simulation = TidalSimulation(
        system_type="star-planet",
        planet_orbperiod=20,
        verbose=False,
    )

    with pytest.raises(NotImplementedError, match="star-planet"):
        simulation.run(YEAR, 0.1 * YEAR)
