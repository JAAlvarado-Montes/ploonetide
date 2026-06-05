.. _Ploonetide Modules:

==========
Ploonetide
==========

This section provides a complete API reference for the Ploonetide package, including the main simulation interface, the ODE modules, and utility tools. Use this guide to explore all classes and functions available for scientific workflows.

Tidal Simulators
================

The core interface to run tidal interaction simulations.

Simulation Class
----------------

.. currentmodule:: ploonetide.numerical

.. autosummary::
   :toctree: api
   :nosignatures:

   Simulation
   Variable

Tidal Simulation Class
----------------------

.. currentmodule:: ploonetide

.. autosummary::
   :toctree: api
   :nosignatures:

   TidalSimulation

ODE Modules
===========

These modules define the differential equations used to simulate star-planet and planet-moon tidal systems.

Star-Planet Tidal ODEs
----------------------

.. currentmodule:: ploonetide.odes.star_planet

.. autosummary::
   :toctree: api
   :nosignatures:

   solution_star_planet

Planet-Moon Tidal ODEs
----------------------

.. currentmodule:: ploonetide.odes.planet_moon

.. autosummary::
   :toctree: api
   :nosignatures:

   solution_planet_moon

Physical and Orbital Functions
==============================

Functions to calculate physical and orbital properties

**Function Utilities**

.. currentmodule:: ploonetide.utils.functions

.. autosummary::
   :toctree: api
   :nosignatures:

   stellar_lifespan
   hill_radius
   roche_radius_masses
   omegaAngular
   luminosity

Utility Tools
=============

Helpful plotting tools, colormaps, and numerical functions.

.. currentmodule:: ploonetide.utils

.. autosummary::
   :toctree: api
   :nosignatures:

   dict2obj
   make_rgb_colormap
   colorline
   make_segments
   canonic_units
