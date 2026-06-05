.. _Quick start:

Quick start
-----------

The example below creates a compact star--planet--moon system and runs a
short tidal-evolution integration. Times are passed in SI seconds.

.. code-block:: python

    from ploonetide import TidalSimulation
    from ploonetide.utils.constants import KYEAR, MYEAR

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

    integration_time = 1 * MYEAR
    timestep = 100 * KYEAR

    simulation.set_integration_method("RK45")
    simulation.run(integration_time, timestep)

    print(simulation.fate, simulation.fate_time)
