"""This module defines TidalSimulation class"""
from __future__ import division
import os
# import logging

import astropy.units as u
import pandas as pd
import numpy as np
import pyfiglet

from pathlib import Path
from tqdm.auto import tqdm
from typing import Optional, Tuple, Literal, Dict

from . import PACKAGEDIR
from ploonetide.utils.constants import *
from ploonetide.utils.functions import *
from ploonetide.utils.events import *
from ploonetide.odes.planet_moon import solution_planet_moon, jacobian
from ploonetide.odes.star_planet import solution_star_planet
from ploonetide.forecaster.mr_forecast import Mstat2R
from ploonetide.numerical.simulator import Variable, Simulation

from ploonetide.utils.moon.moon_physics import (
    # composition & MR
    moon_imf_from, moon_rmf_from,
    moon_density_zero_porosity,
    fortney_radius_ice_rock,
    fortney_radius_rock_iron,
    resolve_moon_rigidity_profile,
    MoonPorosityParams,
    moon_central_pressure_uniform_pa,
    moon_central_pressure_polytrope_pa,
    moon_solve_radius_with_compaction,
    moon_classify_rigidity_from_pc,   # optional; or keep logic inline
)

__all__ = ['TidalSimulation']


class TidalSimulation(Simulation):
    """This class defines a tidal simulation.

    Attributes:
        args (TYPE): Description
        fate (TYPE): Description
        fate_time (TYPE): Description
        history (TYPE): Description
        history_units (TYPE): Description
        moon_radius (TYPE): Description
        simulation_type (str): Description
        system_type (TYPE): Description
    """

    simulation_type = 'Tidal'
    valid_system_types = ("star-planet", "planet-moon")

    def __init__(
        self,
        # ---- Planet properties (can be sampled externally) ----
        planet_mass: float = 1,
        planet_radius: float = 1,
        planet_core_mass: float = 1,
        planet_core_radius: float = 1,
        planet_angular_coeff: float = 0.26401,
        planet_moment_inertia_coeff: float = 0.26401,
        planet_orbperiod: float = 1,
        planet_rotperiod: float = 1,
        planet_eccentricity: float = 0.1,
        planet_rigidity: float = 4.46E10,
        planet_orbit_a_m: Optional[float] = None,
        planet_k2q: float = 1e5,

        # ---- Star properties (can be sampled externally) ----
        star_mass: Optional[float] = 1,
        star_radius: float = 1.,
        star_eff_temperature: float = 3700.,
        star_saturation_rate: float = 4.3421E-5,
        star_angular_coeff: float = 0.5,
        star_rotperiod: float = 10,
        star_alpha: float = 0.25,
        star_beta: float = 0.25,
        star_age: float = 5.,
        sun_omega: float = 2.67E-6,
        sun_mass_loss_rate: float = 1.4E-14,

        # ---- Moon inputs (sampled externally) ----
        moon_mass: float = 0.01,
        moon_k2q: float = None,
        moon_family: Literal["ice-rock", "rock-iron"] = "ice-rock",
        moon_imf: Optional[float] = None,
        moon_rmf: Optional[float] = None,
        # (f_ice, f_rock, f_iron)
        moon_fractions: Optional[Tuple[float, float, float]] = None,
        moon_albedo: float = 0.3,
        moon_eccentricity: float = 0.0,
        moon_obliquity: float = 0.0,
        moon_semimaxis: float = 10,

        # ---- Physics knobs (deterministic) ----
        moon_porous_small_mass: float = 1e-2,
        moon_S_c_pa: float = 1.0e7,
        moon_rigidity_profile: str = "baseline",
        eta_fluid: float | None = None,
        Y_ice_pa: float | None = None,
        Y_rock_pa: float | None = None,
        Y_iron_pa: float | None = None,
        # --- NEW: structure/compaction controls ---
        moon_pressure_model: str = "polytrope",   # "polytrope" or "uniform"
        moon_polytrope_n: float = 1.0,  # polytropic index if using polytrope
        moon_enable_porosity: bool = True,  # enable smooth compaction fallback
        moon_porosity_params: Optional[MoonPorosityParams] = None,
        # Properties below to calculate the moon temperature profile
        activation_energy: float = 3E5,
        heat_capacity: float = 1260,
        mantle_thickness: float = 3E6,
        melt_fraction_coeff: float = 40.,
        solidus_temperature: float = 1600.,
        breakdown_temperature: float = 1800.,
        liquidus_temperature: float = 2000.,
        surface_temperature_earth: float = 288.,
        thermal_conductivity: float = 2.,
        Rayleigh_critical: float = 1100.,
        flow_geometry: float = 1.,
        thermal_expansivity: float = 1E-4,

        # ---- Behavior ----
        system_type='star-planet',
        auto_build: bool = True,
        moon_equilibrium_tide: bool = True,
        moon_sigma_eq_cgs: float = 2.006e-61,
        planet_evolution: bool = False,
        planet_equilibrium_tide: bool = False,
        planet_sigma_eq_cgs: float = 2.006e-61,
        planet_envelope_interface: str = 'rigid-fluid',
        planet_envelope_dissipation: bool = False,
        planet_core_dissipation: bool = False,
        planet_mantle_dissipation: bool = False,
        star_internal_evolution: bool = False,
        use_exact_orbital_masses: bool = True,
        tidal_sign_smoothing_fraction: float = 1.0e-4,
        iw_gate_smoothing_fraction: float = 1.0e-5,
        event_crossing_tol: float = 1e-3,

        # ---- Debugging ----
        debug_ODEs_rhs=False,
        debug_tidal_dissipation=False,
        verbose: bool = True,
    ):
        """Construct the TidalSimulation class

        Args:
            activation_energy (float, optional): Energy of activation, default is 3e5 [J mol^-1]
            heat_capacity (int, optional): Heat capacity of moon, default is 1260 [J kg^-1 K^-1]
            mantle_thickness (float, optional): Thickness of the moon mantle, default 3000000 [m]
            melt_fraction_coeff (int, optional): Melt fraction coefficient, default 25 [No unit]
            solidus_temperature (int, optional): Temperature for solid material, default 1600 [K]
            breakdown_temperature (int, optional): Breakdown tempe, solid-liquidus, default 1800 [K]
            liquidus_temperature (int, optional): Temperature for liquid material, default 2000 [K]
            surface_temperature_earth (float, optional): veraged surface temperature of Earth [K]
            thermal_conductivity (int, optional): Description, default is 2 [W m^-1 K^-1]
            Rayleigh_critical (int, optional): Critical rayleigh number, default is 1100 [No unit]
            flow_geometry (int, optional): Constant for flow geometry [No unit]
            thermal_expansivity (float, optional): Thermal expansivity of moon, default 1E-4 [K^-1]
            planet_evolution (bool, optional): Description
            planet_envelope_dissipation (bool, optional): Description
            planet_core_dissipation (bool, optional): Description
            star_internal_evolution (bool, optional): Description
            star_mass (int, optional): Stellar mass [Msun]
            star_radius (int, optional): Stellar radius [Rsun]
            star_eff_temperature (int, optional): Stellar effective temperature [K]
            star_saturation_rate (float, optional): Star's saturation rotational rate [rad s^-1]
            star_angular_coeff (float, optional): Description
            star_rotperiod (int, optional): Stellar rotation period [d]
            star_alpha (float, optional): Description
            star_beta (float, optional): Description
            star_age (int, optional): Stellar age [Gyr]
            sun_omega (float, optional): Solar rotational rate [s^-1]
            sun_mass_loss_rate (float, optional): Solar mass loss rate [Msun yr^-1]
            planet_mass (int, optional): Planetary mass [Mjup]
            planet_radius (None, optional): Planetary radius [Rjup]
            planet_angular_coeff (float, optional): Description
            planet_orbperiod (None, optional): Planetary orbital period [d]
            planet_rotperiod (float, optional): Description
            planet_eccentricity (float, optional): Planetary eccentricity [No unit]
            planet_rigidity (float, optional): Rigidity of the planet [Pa]
            moon_radius (int, optional): Moon radius [Rearth]
            moon_density (int, optional): Moon density [kg m**-3]
            moon_albedo (float, optional): Description
            moon_eccentricity (float, optional): Eccentricity of moon's orbit [No unit]
            moon_semimaxis (int, optional): Description
            moon_type (bool, optional): Description, default is 'rigid'
            system_type (str, optional): Description
            verbose (bool, optional): Description
        """
        if system_type not in self.valid_system_types:
            allowed = ", ".join(repr(name) for name in self.valid_system_types)
            raise ValueError(
                f"system_type must be one of {allowed}; got {system_type!r}."
            )

        if verbose:
            print(pyfiglet.figlet_format(f'{self.package()}'))

        # ************************************************************
        # SET THE TYPE OF SYSTEM
        # ************************************************************
        self.system_type = system_type

        # ************************************************************
        # KEY TO INCLIDE EVOLUTION
        # ************************************************************
        self._planet_evolution = planet_evolution
        self._planet_equilibrium_tide = planet_equilibrium_tide
        self._planet_envelope_dissipation = planet_envelope_dissipation
        self._planet_envelope_interface = planet_envelope_interface
        self._planet_core_dissipation = planet_core_dissipation
        self._planet_mantle_dissipation = planet_mantle_dissipation
        self._star_internal_evolution = star_internal_evolution

        if (
            planet_envelope_dissipation
            or planet_equilibrium_tide
            or planet_core_dissipation
            or planet_mantle_dissipation
        ):
            self._planet_energy_dissipation_fix_value = False
        else:
            self._planet_energy_dissipation_fix_value = True
        # self._planet_energy_dissipation_fix_value = True

        # ************************************************************
        # GENERAL CONSTANTS IN THE SIMULATION
        # ************************************************************
        self._sun_mass_loss_rate = u.Quantity(sun_mass_loss_rate, u.Msun * u.yr**-1)
        self._sun_omega = u.Quantity(sun_omega, u.s**-1)
        self._activation_energy = u.Quantity(activation_energy, u.J * u.mol**-1)
        self._solidus_temperature = u.Quantity(solidus_temperature, u.K)
        self._breakdown_temperature = u.Quantity(breakdown_temperature, u.K)
        self._liquidus_temperature = u.Quantity(liquidus_temperature, u.K)
        self._surface_temperature_earth = u.Quantity(surface_temperature_earth, u.K)
        self._heat_capacity = u.Quantity(heat_capacity, u.J * u.kg**-1 * u.K**-1)
        self._thermal_conductivity = u.Quantity(thermal_conductivity, u.W * u.m**-1 * u.K**-1)
        self._thermal_expansivity = u.Quantity(thermal_expansivity, u.K**-1)
        self._mantle_thickness = u.Quantity(mantle_thickness, u.m)
        self._Rayleigh_critical = Rayleigh_critical
        self._flow_geometry = flow_geometry
        self._melt_fraction_coeff = melt_fraction_coeff

        # ************************************************************
        # STAR PARAMETERS
        # ************************************************************
        self._star_mass = u.Quantity(star_mass, u.Msun)
        self._star_radius = u.Quantity(star_radius, u.Rsun)
        self._star_eff_temperature = u.Quantity(star_eff_temperature, u.K)
        self._star_rotperiod = u.Quantity(star_rotperiod, u.d)
        self._star_age = u.Quantity(star_age, u.Gyr)
        self._star_saturation_rate = u.Quantity(star_saturation_rate, u.s**-1)
        self._star_angular_coeff = star_angular_coeff
        self._star_alpha = star_alpha
        self._star_beta = star_beta

        # ************************************************************
        # PLANET PARAMETERS
        # ************************************************************
        self._planet_orbperiod = u.Quantity(planet_orbperiod, u.d)
        self._planet_rotperiod = u.Quantity(planet_rotperiod, u.d)
        self._planet_mass = u.Quantity(planet_mass, u.M_jup)
        self._planet_radius = u.Quantity(planet_radius, u.R_jup)
        self._planet_core_mass = u.Quantity(planet_core_mass, u.Mearth)
        self._planet_core_radius = u.Quantity(planet_core_radius, u.Rearth)
        self._planet_rigidity = u.Quantity(planet_rigidity, u.Pa)
        self._planet_angular_coeff = planet_angular_coeff
        self._planet_moment_inertia_coeff = planet_moment_inertia_coeff
        self._planet_k2q = 1.5 / planet_k2q
        self._planet_sigma_eq_cgs = planet_sigma_eq_cgs

        self.planet_orbit_a_m = None if planet_orbit_a_m is None else float(
            planet_orbit_a_m)

        self._planet_eccentricity = planet_eccentricity
        if self.planet_eccentricity < 0 or self.planet_eccentricity > 1:
            raise ValueError("Planet eccentricity must be 0 ≤ ep ≤ 1")

        self.planet_fixed_properties = dict(
            planet_k2q=self._planet_k2q,
            Rp=self.planet_radius.to_value(u.m),
            Mp=self.planet_mass.to_value(u.kg),
            Mp_core=self.planet_core_mass.to_value(u.kg),
            Rp_core=self.planet_core_radius.to_value(u.m),
            envelope_interface=self._planet_envelope_interface,
        )
        self.star_fixed_properties = dict(
            star_k2q=self.star_k2q,
        )

        # ************************************************************
        # MOON PARAMETERS
        # ************************************************************
        # Moon inputs (external sampling)
        self._moon_mass = u.Quantity(moon_mass, u.Mearth)
        self._moon_k2q = None if moon_k2q is None else 1.5 / moon_k2q
        self.moon_family = moon_family
        self.moon_imf = None if moon_imf is None else float(moon_imf)
        self.moon_rmf = None if moon_rmf is None else float(moon_rmf)
        self.moon_fractions = None if moon_fractions is None else tuple(
            map(float, moon_fractions))
        self._moon_albedo = moon_albedo

        # Physics knobs
        self.moon_porous_small_mass = float(moon_porous_small_mass)
        self.moon_S_c_pa = float(moon_S_c_pa)
        self.moon_rigidity_profile = str(moon_rigidity_profile)
        self.eta_fluid = (
            eta_fluid if moon_rigidity_profile == "custom" else None
            if eta_fluid is None else float(eta_fluid)
        )
        self.Y_ice_pa = (
            Y_ice_pa if moon_rigidity_profile == "custom" else None
            if Y_ice_pa is None else float(Y_ice_pa)
        )
        self.Y_rock_pa = (
            Y_rock_pa if moon_rigidity_profile == "custom" else None
            if Y_rock_pa is None else float(Y_rock_pa)
        )
        self.Y_iron_pa = (
            Y_iron_pa if moon_rigidity_profile == "custom" else None
            if Y_iron_pa is None else float(Y_iron_pa)
        )

        # Store new controls
        self.moon_pressure_model = str(moon_pressure_model)
        self.moon_polytrope_n = float(moon_polytrope_n)
        self.moon_enable_porosity = bool(moon_enable_porosity)
        self.moon_porosity_params = (
            moon_porosity_params
            if moon_porosity_params is not None
            else MoonPorosityParams()
        )

        self._moon_equilibrium_tide = moon_equilibrium_tide
        if moon_sigma_eq_cgs is not None:
            self._moon_sigma_eq_cgs = moon_sigma_eq_cgs

        # Behavior
        self.auto_build = bool(auto_build)

        self._moon_eccentricity = moon_eccentricity
        if self._moon_eccentricity < 0 or self._moon_eccentricity > 1:
            raise ValueError("Moon eccentricity must be 0 ≤ em ≤ 1")

        self._moon_obliquity = moon_obliquity
        if self._moon_obliquity < 0.0 or self._moon_obliquity > 90.0:
            raise ValueError("Moon obliquity must be 0.0 ≤ psi_m ≤ 90.0")

        eccentricity_floor = 0.0
        obliquity_floor = 0.0
        self.has_eccentricity = abs(self._moon_eccentricity) > eccentricity_floor
        self.has_obliquity = abs(self._moon_obliquity) > obliquity_floor
        self.has_planet_eccentricity = abs(self._planet_eccentricity) > eccentricity_floor

        # Internal/build status & backing fields for properties
        self._built: bool = False
        self._moon_R_earth: float = float("nan")
        self._moon_density: float = float("nan")
        self._moon_f_ice: float = float("nan")
        self._moon_f_rock: float = float("nan")
        self._moon_f_iron: float = float("nan")
        self._moon_imf_used: float = float("nan")
        self._moon_rmf_used: float = float("nan")
        self._moon_rigidity: Literal["fluid", "rigid"] | None = None
        self._moon_flags: Dict[str, object] = {}

        # Auto-build now if requested. Star-planet setups should not require a moon.
        if self.system_type == "planet-moon" and self.auto_build:
            self.build_moon()

        self.verbose = verbose

        # Formalism for eccentricity and obliquity runs
        self._tidal_sign_smoothing_fraction = tidal_sign_smoothing_fraction
        self._iw_gate_smoothing_fraction = iw_gate_smoothing_fraction

        # Set a global variable for the buffer (or event-crossing tolerance)
        self.event_crossing_tol = event_crossing_tol

        if self.system_type == "planet-moon":
            # Use the property setter so Roche/Hill checks are handled in one place.
            self.moon_semimaxis = moon_semimaxis
        else:
            self._moon_semimaxis = None

        # Arguments for including/excluding different effects
        self.physics_flags = dict(
            planet_evolution=self._planet_evolution,
            planet_energy_dissipation_fix_value=self._planet_energy_dissipation_fix_value,
            planet_envelope_dissipation=self._planet_envelope_dissipation,
            planet_core_dissipation=self._planet_core_dissipation,
            planet_mantle_dissipation=self._planet_mantle_dissipation,
            star_internal_evolution=self._star_internal_evolution,
            moon_equilibrium_tide=self._moon_equilibrium_tide,
            planet_equilibrium_tide=self._planet_equilibrium_tide,
        )

        # Flasg to debug the right-hand side of the ODEs and dissipation
        self._debug_ODEs_rhs = debug_ODEs_rhs
        self._debug_tidal_dissipation = debug_tidal_dissipation

        # ************************************************************
        # INITIAL CONDITIONS FOR THE SYSTEM
        # ************************************************************
        if self.system_type == 'star-planet':
            motion_p = Variable('planet_mean_motion', self.planet_meanmo.value)
            omega_p = Variable('planet_omega', self.planet_omega.value)
            eccen_p = Variable('planet_eccentricity', self.planet_eccentricity)
            omega_s = Variable('star_omega', self.star_omega.value)
            mass_p = Variable('planet_mass', self.planet_mass.to_value(u.kg))
            initial_variables = [motion_p, omega_p, eccen_p, omega_s, mass_p]
            if self.verbose:
                print(
                    f'\nStar mass: {self.star_mass:.3f}\n',
                    f'Star radius: {self.star_radius:.3f}\n',
                    f'Star rotation period: {self.star_rotperiod:.3f}\n',
                    f'Planet orbital period: {self.planet_orbperiod:.3f}\n',
                    f'Planet mass: {self.planet_mass:.3f}\n',
                    f'Planet radius: {self.planet_radius:.3f}\n',
                    f'Planet eccentricity: {self.planet_eccentricity:.4f}\n'
                )

        elif self.system_type == 'planet-moon':
            omega_p = Variable('planet_omega', self.initial_conds['op_ini'])
            motion_p = Variable('planet_mean_motion', self.initial_conds['np_ini'])
            log_motion_m = Variable('log_moon_mean_motion', self.initial_conds['log_nm_ini'])
            eccen2_m = Variable('moon_eccentricity_squared', self.initial_conds['hm_ini'])
            obliq_m = Variable('moon_obliquity', self.initial_conds['psim_ini'])
            eccen2_p = Variable('planet_eccentricity_squared', self.initial_conds['hp_ini'])

            eccentricity_floor = self.parameters.get("eccentricity_floor", 0.0)
            obliquity_floor = self.parameters.get("obliquity_floor", 0.0)

            self.has_eccentricity = abs(self.initial_conds['em_ini']) > eccentricity_floor
            self.has_obliquity = abs(self.initial_conds['psim_ini']) > obliquity_floor
            self.has_planet_eccentricity = abs(self.initial_conds['ep_ini']) > eccentricity_floor

            # if self.has_eccentricity and self.has_obliquity:
            #     initial_variables = [omega_p, motion_p, motion_m, eccen2_m, obliq_m]
            # elif self.has_eccentricity:
            #     initial_variables = [omega_p, motion_p, motion_m, eccen2_m]
            # elif self.has_obliquity:
            #     initial_variables = [omega_p, motion_p, motion_m, obliq_m]
            # else:
            #     initial_variables = [omega_p, motion_p, motion_m]
            initial_variables = [
                omega_p,
                motion_p,
                log_motion_m,
            ]

            if self.has_eccentricity:
                initial_variables.append(eccen2_m)

            if self.has_obliquity:
                initial_variables.append(obliq_m)

            if self.has_planet_eccentricity:
                initial_variables.append(eccen2_p)

            if self.verbose:
                print(
                    f' Stellar age: {self.star_age:.3f}\n',
                    f'Stellar mass: {self.star_mass:.3f}\n',
                    f'Stellar radius: {self.star_radius:.3f}\n',
                    f'Stellar rotation period: {self.star_rotperiod:.3f}\n',
                    f'Planet semimajor axis: {self.planet_semimaxis:.3f}\n',
                    f'Planet orbital period: {self.planet_orbperiod:.3f}\n',
                    f'Planet rotational period: {self.planet_rotperiod:.3f}\n',
                    'Planet critical Hill: '
                    f'{self.planet_critical_hill_radius.to(u.au):.5f} '
                    f'({self.planet_critical_hill_radius / self.moon_roche_radius:.3f} rocheRad)\n',
                    f'Planet corotation radius: '
                    f'{self.planet_corotation_radius.to(u.au):.5f}'
                    f' ({self.planet_corotation_radius / self.moon_roche_radius:.3f} rocheRad)\n'
                    f' Planet mass: {self.planet_mass:.3f}\n',
                    f'Planet radius: {self.planet_radius:.3f}\n',
                    f'Planet core mass: {self.planet_core_mass:.3f}\n',
                    f'Planet core radius: {self.planet_core_radius:.3f}\n',
                    f'Planet eccentricity: {self.planet_eccentricity:.3f}\n',
                    f'Moon ice/rock/iron fractions: '
                    f'{moon_fractions[0]:.3f}, {moon_fractions[1]:.3f}, {moon_fractions[2]:.3f}\n',
                    f'Moon density: {self.moon_density:.3f}\n',
                    f'Moon radius: {self.moon_radius:.3f}\n',
                    f'Moon mass: {self.moon_mass.to(u.Mearth):.5f}\n',
                    f'Moon eccentricity: {self.moon_eccentricity:.3f}\n',
                    f'Moon obliquity: {self.moon_obliquity:.3f}\n',
                    f'Moon semimajor axis: {moon_semimaxis:.3f} rocheRad'
                    f' ({moon_semimaxis * self.moon_roche_radius / self.planet_radius.to(u.m):.3f}'
                    ' planetRad)\n',
                    f'Moon orbital period: {self.moon_orbperiod:.3f}\n',
                    f'Moon type: {self.moon_rigidity}'
                )

        super().__init__(variables=initial_variables)

    @property
    def parameters(self):
        """Dictionary with all the physical and orbital parameters of the class

        Returns:
            dict: Dictionary containing only the values of all the parameters of the TidalSimulation
        """
        return dict(
            Ms=self.star_mass.to_value(u.kg),
            Rs=self.star_radius.to_value(u.m),
            Ls=self.star_luminosity.value,
            os_saturation=self.star_saturation_rate.value,
            star_age=self.star_age.to_value(u.s),
            rigidity=self.planet_rigidity.value,
            E_act=self.activation_energy.value,
            T_solidus=self.solidus_temperature.value,
            T_breakdown=self.breakdown_temperature.value,
            T_liquidus=self.liquidus_temperature.value,
            Cp=self.heat_capacity.value,
            ktherm=self.thermal_conductivity.value,
            alpha_exp=self.thermal_expansivity.value,
            d_mantle=self.mantle_thickness.value,
            densm=self.moon_density.value,
            Mm=self.moon_mass.to_value(u.kg),
            Rm=self.moon_radius.to_value(u.m),
            gravm=self.moon_gravity.value,
            rigidm=self.moon_rigidity_bulk.value,
            T_surface=self.surface_temperature_earth.value,
            sun_mass_loss_rate=self.sun_mass_loss_rate.to_value(u.kg * u.s**-1),
            sun_omega=self.sun_omega.value,
            coeff_star=self._star_angular_coeff,
            coeff_planet=self._planet_angular_coeff,
            planet_GR_coeff=self._planet_moment_inertia_coeff,
            star_alpha=self._star_alpha,
            star_beta=self._star_beta,
            a2=self._flow_geometry,
            Rac=self._Rayleigh_critical,
            B=self._melt_fraction_coeff,
            physics_flags=self.physics_flags,
            planet_fixed_properties=self.planet_fixed_properties,
            star_fixed_properties=self.star_fixed_properties,
            debug_ODEs_rhs=self._debug_ODEs_rhs,
            debug_tidal_dissipation=self._debug_tidal_dissipation,
            tidal_sign_smoothing_fraction=self._tidal_sign_smoothing_fraction,
            moon_sigma_eq_cgs=self._moon_sigma_eq_cgs,
            planet_sigma_eq_cgs=self._planet_sigma_eq_cgs,
            iw_gate_smoothing_fraction=self._iw_gate_smoothing_fraction,
            planet_envelope_dissipation_for_moon=True,
            planet_envelope_dissipation_for_star=True,
        )

    @property
    def initial_conds(self):
        """Dictionary with all the initial conditions of the simulation

        Returns:
            dict: Dictionary containing only the initial values of all the integrating variables
        """
        self.hm_ini = self.moon_eccentricity**2.
        self.psim_ini = np.deg2rad(self.moon_obliquity)
        self.hp_ini = self.planet_eccentricity**2.
        return dict(
            log_nm_ini=np.log(self.moon_meanmo.value),
            em_ini=self.moon_eccentricity,
            hm_ini=self.hm_ini,
            psim_ini=self.psim_ini,
            Tm_ini=self.moon_initial_temperature.value,
            os_ini=self.star_omega.value,
            np_ini=self.planet_meanmo.value,
            op_ini=self.planet_omega.value,
            ep_ini=self.planet_eccentricity,
            hp_ini=self.hp_ini,
            mp_ini=self.planet_mass.to_value(u.kg),
        )

    # **********************************************************************************************
    # ******************************* GENERAL CONSTANTS MODIFIABLE *********************************
    # **********************************************************************************************

    @property
    def sun_mass_loss_rate(self):
        """Mass loss rate of the Sun

        Returns:
            float: Mass loss rate of the Sun [Msun yr^-1]
        """
        return self._sun_mass_loss_rate

    @sun_mass_loss_rate.setter
    def sun_mass_loss_rate(self, value):
        """Set a new value for the mass loss rate of the Sun

        Args:
            value (float): Mass loss rate of the Sun [Msun yr^-1]
        """
        self._sun_mass_loss_rate = value
        if not isinstance(self._sun_mass_loss_rate, u.Quantity):
            self._sun_mass_loss_rate = u.Quantity(value, u.Msun * u.yr**-1)

    @property
    def sun_omega(self):
        """Rotational rate (Omega) of the Sun

        Returns:
            float: Rotational rate (Omega) of the Sun [s^-1]
        """
        return self._sun_omega

    @sun_omega.setter
    def sun_omega(self, value):
        """Set a new value for the rotational rate (Omega) of the Sun

        Args:
            value (float): Rotational rate of the Sun (Omega) [s^-1]
        """
        self._sun_omega = value
        if not isinstance(self._sun_omega, u.Quantity):
            self._sun_omega = u.Quantity(value, u.s**-1)

    @property
    def activation_energy(self):
        """Activation energy of a body

        Returns:
            float: Activation energy [J mol^-1]
        """
        return self._activation_energy

    @activation_energy.setter
    def activation_energy(self, value):
        """Set new value for the activation energy of a body

        Args:
            value (float): Activation energy [J mol^-1]
        """
        self._activation_energy = value
        if not isinstance(self._activation_energy, u.Quantity):
            self._activation_energy = u.Quantity(value, u.J * u.mol**-1)

    @property
    def solidus_temperature(self):
        return self._solidus_temperature

    @solidus_temperature.setter
    def solidus_temperature(self, value):
        self._solidus_temperature = value
        if not isinstance(self._solidus_temperature, u.Quantity):
            self._solidus_temperature = u.Quantity(value, u.K)

    @property
    def liquidus_temperature(self):
        return self._liquidus_temperature

    @liquidus_temperature.setter
    def liquidus_temperature(self, value):
        self._liquidus_temperature = value
        if not isinstance(self._liquidus_temperature, u.Quantity):
            self._liquidus_temperature = u.Quantity(value, u.K)

    @property
    def surface_temperature_earth(self):
        return self._surface_temperature_earth

    @surface_temperature_earth.setter
    def surface_temperature_earth(self, value):
        self._surface_temperature_earth = value
        if not isinstance(self._surface_temperature_earth, u.Quantity):
            self._surface_temperature_earth = u.Quantity(value, u.K)

    @property
    def breakdown_temperature(self):
        return self._breakdown_temperature

    @breakdown_temperature.setter
    def breakdown_temperature(self, value):
        self._breakdown_temperature = value
        if not isinstance(self._breakdown_temperature, u.Quantity):
            self._breakdown_temperature = u.Quantity(value, u.K)

    @property
    def heat_capacity(self):
        return self._heat_capacity

    @heat_capacity.setter
    def heat_capacity(self, value):
        self._heat_capacity = value
        if not isinstance(self._heat_capacity, u.Quantity):
            self._heat_capacity = u.Quantity(value, u.J * u.kg**-1 * u.K**-1)

    @property
    def thermal_conductivity(self):
        return self._thermal_conductivity

    @thermal_conductivity.setter
    def thermal_conductivity(self, value):
        self._thermal_conductivity = value
        if not isinstance(self._thermal_conductivity, u.Quantity):
            self._thermal_conductivity = u.Quantity(value, u.W * u.m**-1 * u.K**-1)

    @property
    def thermal_expansivity(self):
        return self._thermal_expansivity

    @thermal_expansivity.setter
    def thermal_expansivity(self, value):
        self._thermal_expansivity = value
        if not isinstance(self._thermal_expansivity, u.Quantity):
            self._thermal_expansivity = u.Quantity(value, u.K**-1)

    @property
    def mantle_thickness(self):
        return self._mantle_thickness

    @mantle_thickness.setter
    def mantle_thickness(self, value):
        self._mantle_thickness = value
        if not isinstance(self._mantle_thickness, u.Quantity):
            self._mantle_thickness = u.Quantity(value, u.m)

    # **********************************************************************************************
    # ********************************* STAR DYNAMICAL PROPERTIES **********************************
    # **********************************************************************************************
    @property
    def star_mass(self):
        return self._star_mass

    @star_mass.setter
    def star_mass(self, value):
        if value <= 0:
            raise ValueError("Mass must be positive.")
        self._star_mass = value
        if not isinstance(self._star_mass, u.Quantity):
            self._star_mass = u.Quantity(value, u.Msun)

    @property
    def star_radius(self):
        return self._star_radius

    @star_radius.setter
    def star_radius(self, value):
        if value <= 0:
            raise ValueError("Radius must be positive.")
        self._star_radius = value
        if not isinstance(self._star_radius, u.Quantity):
            self._star_radius = u.Quantity(value, u.Rsun)

    @property
    def star_age(self):
        return self._star_age

    @star_age.setter
    def star_age(self, value):
        if value <= 0:
            raise ValueError("Stellar age must be positive.")
        self._star_age = value
        if not isinstance(self._star_age, u.Quantity):
            self._star_age = u.Quantity(value, u.Gyr)

    @property
    def star_saturation_rate(self):
        return self._star_saturation_rate

    @star_saturation_rate.setter
    def star_saturation_rate(self, value):
        self._star_saturation_rate = value
        if not isinstance(self._star_saturation_rate, u.Quantity):
            self._star_saturation_rate = u.Quantity(value, u.s**-1)

    @property
    def star_rotperiod(self):
        return self._star_rotperiod

    @star_rotperiod.setter
    def star_rotperiod(self, value):
        if value <= 0:
            raise ValueError("Rotational period must be positive.")
        self._star_rotperiod = value
        if not isinstance(self._star_rotperiod, u.Quantity):
            self._star_rotperiod = u.Quantity(value, u.d)

    @property
    def star_eff_temperature(self):
        return self._star_eff_temperature

    @star_eff_temperature.setter
    def star_eff_temperature(self, value):
        if value <= 0:
            raise ValueError("Stellar effective temperature must be positive.")
        self._star_eff_temperature = value
        if not isinstance(self._star_eff_temperature, u.Quantity):
            self._star_eff_temperature = u.Quantity(value, u.K)

    @property
    def star_luminosity(self):
        return u.Quantity(luminosity(self.star_radius.to_value(u.m),
                                     self.star_eff_temperature.value), u.W)

    @star_luminosity.setter
    def star_luminosity(self, value):
        if value <= 0:
            raise ValueError("Stellar luminosity must be positive.")
        self._star_luminosity = value
        if not isinstance(self._star_luminosity, u.Quantity):
            print("popito")
            self._star_luminosity = u.Quantity(value, u.W)

    @property
    def star_omega(self):
        return u.Quantity(2. * np.pi / self.star_rotperiod.to_value(u.s), u.s**-1)

    @property
    def star_epsilon(self):
        return self.star_omega.value / omegaCritic(self.star_mass.to_value(u.kg),
                                                   self.star_radius.to_value(u.m))

    @property
    def star_k2q(self):
        return k2Q_star_envelope(self._star_alpha, self._star_beta, self.star_epsilon)

    @property
    def star_lifespan(self):
        return u.Quantity(stellar_lifespan(self.star_mass.to_value(u.kg)), u.s)

    @property
    def star_saturation_period(self):
        return u.Quantity(2. * np.pi / self.star_saturation_rate.value, u.d)

    # **********************************************************************************************
    # ******************************** PLANET DYNAMICAL PROPERTIES *********************************
    # **********************************************************************************************
    @property
    def planet_orbperiod(self):
        return self._planet_orbperiod

    @planet_orbperiod.setter
    def planet_orbperiod(self, value):
        if value <= 0:
            raise ValueError("Orbital period must be positive.")
        self._planet_orbperiod = value
        if not isinstance(self._planet_orbperiod, u.Quantity):
            self._planet_orbperiod = u.Quantity(value, u.d)

    @property
    def planet_eccentricity(self):
        return self._planet_eccentricity

    @planet_eccentricity.setter
    def planet_eccentricity(self, value):
        if value < 0 or value > 1:
            raise ValueError("Planet eccentricity must be 0 ≤ ep ≤ 1")
        self._planet_eccentricity = value

    @property
    def planet_rotperiod(self):
        return self._planet_rotperiod

    @planet_rotperiod.setter
    def planet_rotperiod(self, value):
        if value <= 0:
            raise ValueError("Rotational period must be positive.")
        self._planet_rotperiod = value
        if not isinstance(self._planet_rotperiod, u.Quantity):
            self._planet_rotperiod = u.Quantity(value, u.d)

    @property
    def planet_mass(self):
        return self._planet_mass

    @planet_mass.setter
    def planet_mass(self, value):
        if value <= 0:
            raise ValueError("Mass must be positive.")
        self._planet_mass = value
        if not isinstance(self._planet_mass, u.Quantity):
            self._planet_mass = u.Quantity(value, u.M_jup)

    @property
    def planet_core_mass(self):
        return self._planet_core_mass

    @planet_core_mass.setter
    def planet_core_mass(self, value):
        if value <= 0:
            raise ValueError("Core mass must be positive.")
        self._planet_core_mass = value
        if not isinstance(self._planet_core_mass, u.Quantity):
            self._planet_core_mass = u.Quantity(value, u.Mearth)

    @property
    def planet_radius(self):
        if pd.isnull(self._planet_radius):
            planet_radius, _, _ = Mstat2R(
                mean=self.planet_mass.value, std=0.1, unit='Jupiter',
                sample_size=200, classify='Yes'
            )

            return u.Quantity(planet_radius, u.R_jup)
        else:
            return self._planet_radius

    @planet_radius.setter
    def planet_radius(self, value):
        if value <= 0:
            raise ValueError("Core radius must be positive.")
        self._planet_radius = value
        if not isinstance(self._planet_radius, u.Quantity):
            if not value:
                self._planet_radius, _, _ = Mstat2R(
                    mean=self.planet_mass.value, std=0.1, unit='Jupiter',
                    sample_size=200, classify='Yes'
                )
                self._planet_radius = u.Quantity(self._planet_radius, u.R_jup)

    @property
    def planet_core_radius(self):
        if pd.isnull(self._planet_core_radius):
            planet_core_radius, _, _ = Mstat2R(
                mean=self.planet_core_mass.value, std=0.1, unit='Earth',
                sample_size=200, classify='Yes'
            )

            return u.Quantity(planet_core_radius, u.Rearth)
        else:
            return self._planet_core_radius

    @planet_core_radius.setter
    def planet_core_radius(self, value):
        if value <= 0:
            raise ValueError("Radius must be positive.")
        self._planet_core_radius = value
        if not isinstance(self._planet_core_radius, u.Quantity):
            if not value:
                self._planet_core_radius, _, _ = Mstat2R(
                    mean=self.planet_core_mass.value, std=0.1, unit='Earth',
                    sample_size=200, classify='Yes'
                )
                self._planet_core_radius = u.Quantity(self._planet_core_radius, u.Rearth)

    @property
    def planet_moment_inertia_coeff(self):
        return self._planet_moment_inertia_coeff

    @planet_moment_inertia_coeff.setter
    def planet_moment_inertia_coeff(self, value):
        if value <= 0:
            raise ValueError("Moment of inertia coeff. must be positive.")
        self._planet_moment_inertia_coeff = value

    @property
    def planet_rigidity(self):
        return self._planet_rigidity

    @planet_rigidity.setter
    def planet_rigidity(self, value):
        if value <= 0:
            raise ValueError("Rigidity must be positive.")
        self._planet_rigidity = value
        if not isinstance(self._planet_rigidity, u.Quantity):
            self._planet_rigidity = u.Quantity(value, u.Pa)

    @property
    def planet_omega(self):
        return u.Quantity(2. * np.pi / self.planet_rotperiod.to_value(u.s), u.s**-1)

    @property
    def planet_semimaxis(self):
        return u.Quantity(semiMajorAxis(self.planet_orbperiod.to_value(u.s),
                                        self.star_mass.to_value(u.kg),
                                        self.planet_mass.to_value(u.kg)), u.m).to(u.au)

    @property
    def planet_meanmo(self):
        return u.Quantity(meanMotion(self.planet_semimaxis.to_value(u.m),
                                     self.star_mass.to_value(u.kg),
                                     self.planet_mass.to_value(u.kg)), u.s**-1)

    @property
    def planet_epsilon(self):
        return self.planet_omega.value / omegaCritic(self.planet_mass.to_value(u.kg),
                                                     self.planet_radius.to_value(u.m))

    @property
    def planet_envelope_interface(self):
        return self._planet_envelope_interface

    @property
    def planet_k2q(self):
        if not self._planet_evolution:
            if self._planet_energy_dissipation_fix_value:
                print('Planet does not evolve and dissipation is constant')
                return self.planet_fixed_properties['planet_k2q']

            else:
                print('Planet does not evolve and dissipation is fixed with models')
                k2q_planet_core = 0.0
                k2q_planet_mantle = 0.0
                k2q_planet_envelope = 0.0
                Mp = self.planet_fixed_properties['Mp']
                Mp_core = self.planet_fixed_properties['Mp_core']
                Rp = self.planet_fixed_properties['Rp']
                Rp_core = self.planet_fixed_properties['Rp_core']

                alpha_planet = Rp_core / Rp
                beta_planet = Mp_core / Mp

                if self._planet_envelope_dissipation:
                    k2q_planet_envelope = k2Q_inertial_waves_convective_envelope(
                        alpha_planet,
                        beta_planet,
                        self.planet_epsilon,
                        self._planet_envelope_interface,
                    )
                if self._planet_core_dissipation:
                    k2q_planet_core = k2Q_planet_core(
                        self.planet_rigidity.value,
                        alpha_planet,
                        beta_planet,
                        Mp, Rp
                    )
                if self._planet_mantle_dissipation:
                    k2q_planet_mantle = k2Q_planet_mantle(
                        self.planet_rigidity.value,
                        alpha_planet,
                        beta_planet,
                        Mp, Rp
                    )
                return k2q_planet_core + k2q_planet_mantle + k2q_planet_envelope
        else:
            if self._planet_energy_dissipation_fix_value:
                print('Planet evolves but dissipation is constant')
                return self.planet_fixed_properties['planet_k2q']
            else:
                print('Planet evolves and dissipation is not constant')
                return 'This is not a static property'

    @property
    def planet_roche_radius(self):
        # Roche radius of the planet (Guillochon et. al 2011)
        return u.Quantity(
            roche_radius_masses(
                self.star_mass.to_value(u.kg),
                self.planet_mass.value,
                self.planet_radius.to_value(u.m),
                rfac=2.7
            ),
            u.m
        )

    @property
    def planet_hill_radius(self):
        return u.Quantity(
            hill_radius(
                self.planet_semimaxis.to_value(u.m),
                self.planet_eccentricity,
                self.planet_mass.to_value(u.kg),
                self.star_mass.to_value(u.kg)
            ),
            u.m
        )

    @property
    def planet_critical_hill_radius(self):
        return u.Quantity(
            critical_hill_radius(
                self.planet_hill_radius,
                self.planet_eccentricity,
                self.moon_eccentricity
            ),
            u.m
        )

    @property
    def planet_critical_hill_meanMotion(self):
        return u.Quantity(
            meanMotion(
                self.planet_critical_hill_radius.value,
                self.star_mass.to_value(u.kg),
                self.planet_mass.to_value(u.kg)
            ),
            u.s**-1
        )

    @property
    def planet_corotation_radius(self):
        return u.Quantity(
            corotation_radius(
                self.planet_mass.to_value(u.kg),
                self.planet_omega.value
            ),
            u.m
        )

    # **********************************************************************************************
    # ******************************** MOON PROPERTIES *********************************************
    # **********************************************************************************************

    # ******************* Properties that could changed from outside the class *********************

    @property
    def moon_semimaxis(self):
        """Moon semimajor axis

        Returns:
            float: Moon semimajor axis [Roche radii]
        """
        return self._moon_semimaxis

    @moon_semimaxis.setter
    def moon_semimaxis(self, value):
        """Set moon semimajor axis to new value

        Args:
            value (float): Moon semimajor axis [Roche radii]
        """
        ac_r = self.planet_critical_hill_radius / self.moon_roche_radius
        if value <= 0:
            raise ValueError("Semimajor axis must be positive.")

        if value > ac_r:
            raise ValueError(
                f"Moon semimaxis/apocentre must be < a_Hill_crit (a_crit = {ac_r:.3f} R_roche)"
            )

        if value <= 1:
            raise ValueError("Moon semimajor axis must be greater than the Roche limit.")

        self._moon_semimaxis = value

        if not isinstance(self._moon_semimaxis, u.Quantity):
            self._moon_semimaxis = u.Quantity(value * self.moon_roche_radius.value, u.m)

        a_crit = self.planet_critical_hill_radius * (1.0 - self.event_crossing_tol)

        a_roche = self.moon_roche_radius * (1.0 + self.event_crossing_tol)

        if self.has_eccentricity:
            qm = self._moon_semimaxis * (1 - self._moon_eccentricity)
            Qm = self._moon_semimaxis * (1 + self._moon_eccentricity)

            if qm <= a_roche:
                qm_r = qm / self.moon_roche_radius
                a_roche_r = a_roche / self.moon_roche_radius
                raise ValueError(
                    f"Moon pericentre ({qm_r:.3f} R_roche) must be > {a_roche_r} R_roche"
                )

            if Qm >= a_crit:
                Qm_r = Qm / self.moon_roche_radius
                a_crit_r = a_crit / self.moon_roche_radius
                raise ValueError(
                    f"Moon apocentre ({Qm_r:.3f} R_roche) must be < a_crit ({a_crit_r:.3f} R_roche)"
                )

    @property
    def moon_pericentre(self):
        """Moon pericentre

        Returns:
            float: Moon pericentre [Roche radii]
        """
        if self.has_eccentricity:
            qm = u.Quantity(
                self._moon_semimaxis.value * (1 - self._moon_eccentricity),
                u.m
            )
            return qm
        else:
            return self.moon_semimaxis

    @property
    def moon_apocentre(self):
        """Moon apocentre

        Returns:
            float: Moon apocentre [Roche radii]
        """
        if self.has_eccentricity:
            Qm = u.Quantity(
                self._moon_semimaxis.value * (1 + self._moon_eccentricity),
                u.m
            )
            return Qm
        else:
            return self.moon_semimaxis

    @property
    def moon_eccentricity(self):
        return self._moon_eccentricity

    @moon_eccentricity.setter
    def moon_eccentricity(self, value):
        if value < 0 or value > 1:
            raise ValueError("Moon eccentricity must be 0 ≤ em ≤ 1")
        self._moon_eccentricity = value

    @property
    def moon_obliquity(self):
        return self._moon_obliquity

    @moon_obliquity.setter
    def moon_obliquity(self, value):
        if value < 0.0 or value > 90.:
            raise ValueError("Moon obliquity must be 0.0 ≤ psi_m ≤ 90.0")
        self._moon_obliquity = value

    #  ************************* Properties that are calculated internally *************************

    # ------------------------------------------------------------------
    # Deterministic moon build (no RNG)
    # ------------------------------------------------------------------
    def _smooth_sigmoid(self, x: float, x0: float, width: float) -> float:
        """Smooth step from 0 to 1 centered at x0."""
        width = max(width, 1e-12)
        z = np.clip((x - x0) / width, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-z))

    def _smooth_log_blend(self, y_lo: float, y_hi: float, w_hi: float) -> float:
        """Blend two positive quantities in log-space."""
        w_hi = np.clip(w_hi, 0.0, 1.0)
        if y_lo <= 0.0 or y_hi <= 0.0:
            return (1.0 - w_hi) * y_lo + w_hi * y_hi
        return np.exp((1.0 - w_hi) * np.log(y_lo) + w_hi * np.log(y_hi))

    def build_moon(self) -> None:
        """
        Build a moon from externally sampled mass + composition fractions only.

        Notes
        -----
        Production version:

        1) Require moon_fractions = (f_ice, f_rock, f_iron).
        2) Compute a zero-porosity mixture density.
        3) Compute two candidate radii:
           - a Fortney-based radius using a smooth compositional blend,
           - a fallback radius from density + optional compaction.
        4) Smoothly blend those two radii across a pragmatic mass window.
        5) Derive the final bulk density and central pressure from the final radius.
        6) Apply hard density plausibility cuts.
        7) Classify rigid vs fluid from Pc relative to the composition-weighted yield.
        """

        if self.moon_fractions is None:
            raise ValueError(
                "This production build requires moon_fractions=(f_ice, f_rock, f_iron)."
            )

        moon_f_ice, moon_f_rock, moon_f_iron = self.moon_fractions
        s = moon_f_ice + moon_f_rock + moon_f_iron
        if s <= 0:
            raise ValueError("moon_fractions must sum to > 0.")

        moon_f_ice, moon_f_rock, moon_f_iron = (
            moon_f_ice / s,
            moon_f_rock / s,
            moon_f_iron / s,
        )

        moon_imf_used = moon_imf_from(moon_f_ice, moon_f_rock)
        moon_rmf_used = moon_rmf_from(moon_f_rock, moon_f_iron)

        moon_density0 = moon_density_zero_porosity(
            moon_f_ice,
            moon_f_rock,
            moon_f_iron,
        )  # [kg m^-3]

        moon_M = float(self.moon_mass.value)   # [Mearth]
        moon_M_kg = moon_M * MEARTH            # [kg]
        if moon_M <= 0:
            raise ValueError("moon_mass must be > 0.")

        # =========================================================
        # Fortney branch: smooth compositional blend
        # Shift the low-mass Fortney onset upward so tiny moons are not
        # dominated by Fortney fits in a regime where fallback physics
        # should matter more.
        # =========================================================
        Mfort_lo = 1e-3
        Mfort_hi = 1e-2
        fortney_valid = (Mfort_lo <= moon_M <= Mfort_hi)
        moon_M_eval = np.clip(moon_M, Mfort_lo, Mfort_hi)

        R_fort_ice = fortney_radius_ice_rock(moon_M_eval, moon_imf_used)    # [Rearth]
        R_fort_rock = fortney_radius_rock_iron(moon_M_eval, moon_rmf_used)  # [Rearth]

        w_ice_branch = self._smooth_sigmoid(moon_f_ice, 0.05, 0.03)
        R_fortney = self._smooth_log_blend(R_fort_rock, R_fort_ice, w_ice_branch)

        # =========================================================
        # Fallback branch: density + optional compaction
        # =========================================================
        moon_density0_gcc = moon_density0 / 1000.0
        if self.moon_enable_porosity:
            R_fb_m, rho_fb_si, _ = moon_solve_radius_with_compaction(
                mass_kg=moon_M_kg,
                rho0_gcc=moon_density0_gcc,
                f_ice=float(moon_f_ice),
                f_rock=float(moon_f_rock),
                f_iron=float(moon_f_iron),
                pressure_model=self.moon_pressure_model,
                polytrope_n=self.moon_polytrope_n,
                pp=self.moon_porosity_params,
            )
        else:
            rho_fb_si = moon_density0
            R_fb_m = (3.0 * moon_M_kg / (4.0 * np.pi * rho_fb_si)) ** (1.0 / 3.0)

        R_fb = R_fb_m / REARTH  # [Rearth]

        # =========================================================
        # Smooth Fortney <-> fallback blend
        # =========================================================
        logM = np.log10(moon_M)
        logMlo = np.log10(Mfort_lo)
        logMhi = np.log10(Mfort_hi)
        dlogM_lo = 0.35
        dlogM_hi = 0.25

        w_low = self._smooth_sigmoid(logM, logMlo, dlogM_lo)
        w_high = 1.0 - self._smooth_sigmoid(logM, logMhi, dlogM_hi)
        w_fortney = np.clip(w_low * w_high, 0.0, 1.0)

        moon_R = self._smooth_log_blend(R_fb, R_fortney, w_fortney)

        # =========================================================
        # Final bulk density and central pressure from final radius
        # =========================================================
        moon_R_m = moon_R * REARTH
        moon_density = moon_M_kg / ((4.0 / 3.0) * np.pi * moon_R_m**3)

        Pc_val = (
            moon_central_pressure_polytrope_pa(
                moon_M_kg,
                moon_R_m,
                self.moon_polytrope_n
            )
            if (self.moon_pressure_model == "polytrope")
            else moon_central_pressure_uniform_pa(
                moon_M_kg,
                moon_R_m
            )
        )

        # Scientifically useful to know how porous surviving moons are.
        phi_bulk = max(0.0, 1.0 - moon_density / moon_density0)

        # =========================================================
        # Hard density plausibility cuts
        # =========================================================
        rho_min_admissible = 500.0
        rho_max_admissible = 6500.0

        self._moon_flags = {
            "fortney_valid": fortney_valid,
            "w_ice_branch": w_ice_branch,
            "w_fortney": w_fortney,
            "R_fortney": R_fortney,
            "R_fallback": R_fb,
            "rho_min_admissible": rho_min_admissible,
            "rho_max_admissible": rho_max_admissible,
            "phi_bulk": phi_bulk,
        }

        if moon_density < rho_min_admissible:
            self._moon_flags["moon_like_admissible"] = False
            raise ValueError(
                f"Moon rejected: bulk density too low ({moon_density:.1f} kg m^-3)."
            )

        if moon_density > rho_max_admissible:
            self._moon_flags["moon_like_admissible"] = False
            raise ValueError(
                f"Moon rejected: bulk density too high ({moon_density:.1f} kg m^-3)."
            )

        self._moon_flags["moon_like_admissible"] = True

        # =========================================================
        # Rigidity classification
        # =========================================================
        _eta, _Yice, _Yrock, _Yiron = resolve_moon_rigidity_profile(
            moon_rigidity_profile=self.moon_rigidity_profile,
            eta_fluid=self.eta_fluid,
            Y_ice_pa=self.Y_ice_pa,
            Y_rock_pa=self.Y_rock_pa,
            Y_iron_pa=self.Y_iron_pa,
        )

        moon_rigidity = moon_classify_rigidity_from_pc(
            Pc_pa=Pc_val,
            f_ice=moon_f_ice,
            f_rock=moon_f_rock,
            f_iron=moon_f_iron,
            eta_fluid=_eta,
            Y_ice_pa=_Yice,
            Y_rock_pa=_Yrock,
            Y_iron_pa=_Yiron,
        )

        # =========================================================
        # Store results
        # =========================================================
        self._moon_radius = moon_R               # [Rearth]
        self._moon_density = moon_density        # [kg m^-3]
        self._moon_central_pressure_pa = Pc_val  # [Pa]
        self._moon_f_ice = moon_f_ice
        self._moon_f_rock = moon_f_rock
        self._moon_f_iron = moon_f_iron
        self._moon_imf_used = moon_imf_used
        self._moon_rmf_used = moon_rmf_used
        self._moon_rigidity = moon_rigidity
        self._built = True

    # ------------------------------------------------------------------
    # Public alias if you update any moon_* inputs after construction
    # ------------------------------------------------------------------
    def rebuild_moon(self) -> None:
        """Recompute moon properties after changing moon_* inputs."""
        self.build_moon()

    # ------------------------------------------------------------------
    # Property accessors (read-only after build)
    # ------------------------------------------------------------------
    def _ensure_built(self) -> None:
        if not self._built:
            raise RuntimeError(
                "Moon has not been built (auto_build=False?). "
                "Set moon_* inputs then call rebuild_moon()."
            )

    @property
    def moon_mass(self):
        return self._moon_mass

    @moon_mass.setter
    def moon_mass(self, value):
        """Set moon mass to new value

        Args:
            value (float): Moon mass [Earth masses]
        """
        if value <= 0:
            raise ValueError("Mass must be positive.")
        self._moon_mass = value
        if not isinstance(self._moon_mass, u.Quantity):
            self._moon_mass = u.Quantity(value, u.Mearth)
        self.rebuild_moon()

    @property
    def moon_k2q(self):
        return self._moon_k2q

    @property
    def moon_radius(self) -> float:
        """Moon radius [R⊕] (computed).

        Returns:
            float: Moon radius [Rearth]
        """
        self._ensure_built()
        return u.Quantity(self._moon_radius, u.Rearth)

    @property
    def moon_density(self) -> float:
        """Moon bulk density [g/cc] (computed; includes porosity if any)."""
        self._ensure_built()

        return u.Quantity(self._moon_density, u.kg * u.m**-3)

    @property
    def moon_fractions_used(self) -> Tuple[float, float, float]:
        """(f_ice, f_rock, f_iron) actually used by the build (sums ≈ 1)."""
        self._ensure_built()
        return (self._moon_f_ice, self._moon_f_rock, self._moon_f_iron)

    @property
    def moon_imf_used(self) -> float:
        """IMF used for Fortney (0..1). For 'rock-iron' family this is 0 (unused)."""
        self._ensure_built()
        return self._moon_imf_used

    @property
    def moon_rmf_used(self) -> float:
        """RMF used for Fortney (0..1). For 'ice-rock' family this is 1 (unused)."""
        self._ensure_built()
        return self._moon_rmf_used

    @property
    def moon_rigidity(self) -> Literal["fluid", "rigid"]:
        """Rigidity state used to select the classical Roche coefficient."""
        self._ensure_built()
        assert self._moon_rigidity is not None
        return self._moon_rigidity

    @property
    def moon_flags(self) -> Dict[str, object]:
        """
        Diagnostic flags — current entries:
        - 'fortney_valid' : bool
        - 'phi'           : resolved porosity in small-body branch
        """
        self._ensure_built()
        return dict(self._moon_flags)

    @property
    def moon_roche_radius(self):
        """Roche radius of a solid moon

        Returns:
            float: Roche radius of a moon [m]
        """
        return u.Quantity(
            roche_radius_from_densities(
                self.planet_radius.to_value(u.m),
                self.planet_mass.to_value(u.kg),
                self.moon_density.value,
                self.moon_rigidity
            ),
            u.m
        )

    @property
    def moon_roche_meanmotion(self):
        """Roche radius of a solid moon

        Returns:
            float: Roche radius of a moon [m]
        """
        return u.Quantity(
            meanMotion(
                self.moon_roche_radius.value,
                self.planet_mass.to_value(u.kg),
                self.moon_mass.value
            ),
            u.s**-1
        )

    @property
    def moon_gravity(self):
        """Moon surface gravity

        Returns:
            float: Moon surface gravity [m s^-2]
        """
        return u.Quantity(
            gravity(
                self.moon_mass.to_value(u.kg),
                self.moon_radius.to_value(u.m)
            ),
            u.m * u.s**-2
        )

    @property
    def moon_rigidity_bulk(self):
        """Moon rigidity

        Returns:
            float: Moon rigidity [Pa]
        """
        return u.Quantity(
            self.moon_density.value * self.moon_gravity.value * self.moon_radius.to_value(u.m),
            u.Pa
        )

    @property
    def moon_meanmo(self):
        """Moon mean motion calculated using Kepler's third law

        Returns:
            float: Moon mean motion [s^-1]
        """
        return u.Quantity(
            meanMotion(
                self.moon_semimaxis.value,
                self.planet_mass.to_value(u.kg),
                self.moon_mass.to_value(u.kg)
            ),
            u.s**-1
        )

    @property
    def moon_orbperiod(self):
        """Orbital period of a moon calculated using its mean motion

        Returns:
            float: Orbital period of a moon [d]
        """
        return u.Quantity(2. * np.pi / self.moon_meanmo.value, u.s).to(u.d)

    @property
    def moon_initial_temperature(self):
        """Equilibrium temperature of a moon

        Returns:
            float: Equilibrium temperature of the moon [K]
        """

        if self.moon_eccentricity != 0.0:
            n = meanMotion(
                self.moon_semimaxis.value,
                self.planet_mass.to_value('kg'),
                self.moon_mass.value
            )

            T_stab = 0.0
            T_stab = bisection(n, self.moon_eccentricity, self.parameters)

            if T_stab > 0:
                flux, _ = tidal_heat(T_stab, n, self.moon_eccentricity, self.parameters)
                T_s = surf_temp(flux)

            elif T_stab <= 0:
                T_s = T_stab
            return u.Quantity(T_s, u.K)
        else:
            T_s = u.Quantity(
                equil_temp(
                    self.star_eff_temperature.value,
                    self.star_radius.to_value(u.m),
                    self.planet_semimaxis.to_value(u.m),
                    self._moon_albedo
                ),
                u.K
            )
            return T_s

    @classmethod
    def get_class_name(cls):
        """Get the name TidalSimulation as a string.

        Returns:
            str: Name of the class
        """
        return cls.__name__

    @classmethod
    def package(self):
        """Get the name of the package.

        Returns:
            str: Name of the package
        """
        return os.path.basename(PACKAGEDIR)

    def run(self, integration_time, timestep, t0=0):
        """Run the simulation for an integration time and time-step.

        Args:
            integration_time (float): Total integration time to run the simulation [s]
            timestep (float): Fixed time-steop of the simulation [s]
            t0 (int, optional): Initial time of the simulation. Default is 0 [s]
        """
        if self.system_type == 'star-planet':
            raise NotImplementedError(
                "star-planet integrations are pending revision and are not "
                "enabled in this release path."
            )

        if self.physics_flags['planet_evolution']:
            from ploonetide.utils.planet.evolmodels.subNeptune import build_default_k218_track
            planet_track = build_default_k218_track(
                closure='quadratic', t_end=integration_time)

            integrator_args = {
                'parameters': self.parameters,
                'planet_track': planet_track
            }
        else:
            integrator_args = {'parameters': self.parameters}

        differential_equation = solution_planet_moon

        hm_idx = 3 if self.has_eccentricity else None

        event_roche = make_event_roche_log(
            self.planet_mass.to_value(u.kg),
            self.moon_mass.to_value(u.kg),
            self.moon_roche_radius.value,
            hm_idx=hm_idx,
            buffer=self.event_crossing_tol,
        )
        event_hill = make_event_hill_log(
            self.planet_mass.to_value(u.kg),
            self.moon_mass.to_value(u.kg),
            self.planet_critical_hill_radius.value,
            hm_idx=hm_idx,
            buffer=self.event_crossing_tol,
        )

        events = [
            event_roche,
            event_hill
        ]
        event_names = [
            event_roche.event_name,
            event_hill.event_name,
        ]

        super().set_diff_eq(
            differential_equation,
            integrator_args,
            self.initial_conds,
            events
        )

        if self.verbose:
            print('\nStarting integration of moon orbital migration:\n')

        super().run(integration_time, timestep, t0=t0, jacobian=jacobian)

        if self.system_type == 'planet-moon':
            # Get the solutions from the integrator
            self.sols = self.history

            if self.verbose:
                print("success:", self.sols.success, "|", self.sols.message)
                print("t_events:", self.sols.t_events)          # list of arrays, one per event
                print("y_events shapes:", [e.shape for e in self.sols.y_events])

            # Find the fate of the moon: disrupts, escapes, or stalls
            moon_fate = find_moon_fate(self.sols, event_names)
            self.fate_time = u.Quantity(moon_fate.time / GYEAR, u.Gyr)
            self.fate = moon_fate.fate

            if self.fate == 'survives':
                times, solutions = self.sols.t, self.sols.y
            else:
                # Create a copy of times (t) and solutions (y)
                t = self.sols.t.copy()
                y = self.sols.y.copy()

                # Find the time and state of the event.
                # Use y_events rather than sol.sol(t_hit), because y_events is the
                # root-refined event state returned by solve_ivp.
                t_hit = self.sols.t_events[moon_fate.index][0]
                y_hit = self.sols.y_events[moon_fate.index][0]

                # Append the event point if it is not already present.
                if np.isclose(t[-1], t_hit, rtol=0.0, atol=1e-9 * max(abs(t_hit), 1.0)):
                    t_end = t.copy()
                    y_end = y.copy()
                    t_end[-1] = t_hit
                    y_end[:, -1] = y_hit
                else:
                    t_end = np.append(t, t_hit)
                    y_end = np.column_stack([y, y_hit])

                # Ensure strictly increasing time order.
                order = np.argsort(t_end)
                times = t_end[order]
                solutions = y_end[:, order]

            planet_omega = solutions[0]
            planet_mean_motion = solutions[1]

            # Recover the moon mean motion (nm) from inside the log(nm)
            moon_mean_motion = np.exp(solutions[2])

            # Calculate the moon semimajor axis using nm
            moon_semi_ma = mean2axis(
                moon_mean_motion,
                self.planet_mass.to_value(u.kg),
                self.moon_mass.to_value(u.kg)
            )

            if self.has_eccentricity and self.has_obliquity:
                hm = np.maximum(solutions[3], 0.0)
                moon_eccentricity = np.sqrt(hm)

                self.solutions = pd.DataFrame(
                    {
                        'Times': times,
                        'Planet Omega': solutions[0],
                        'Planet Mean Motion': solutions[1],
                        'Moon Mean Motion': moon_mean_motion,
                        'Moon Semimajor Axis': moon_semi_ma,
                        'Moon Pericentre': moon_semi_ma * (1.0 - moon_eccentricity),
                        'Moon Apocentre': moon_semi_ma * (1.0 + moon_eccentricity),
                        'Moon Eccentricity Squared': hm,
                        'Moon Eccentricity': moon_eccentricity,
                        'Moon Obliquity': solutions[4]
                    }
                )

            elif self.has_eccentricity:
                hm = np.maximum(solutions[3], 0.0)
                moon_eccentricity = np.sqrt(hm)

                self.solutions = pd.DataFrame(
                    {
                        'Times': times,
                        'Planet Omega': solutions[0],
                        'Planet Mean Motion': solutions[1],
                        'Moon Mean Motion': moon_mean_motion,
                        'Moon Semimajor Axis': moon_semi_ma,
                        'Moon Pericentre': moon_semi_ma * (1.0 - moon_eccentricity),
                        'Moon Apocentre': moon_semi_ma * (1.0 + moon_eccentricity),
                        'Moon Eccentricity Squared': hm,
                        'Moon Eccentricity': moon_eccentricity,
                    }
                )

            elif self.has_obliquity:
                self.solutions = pd.DataFrame(
                    {
                        'Times': times,
                        'Planet Omega': solutions[0],
                        'Planet Mean Motion': solutions[1],
                        'Moon Mean Motion': moon_mean_motion,
                        'Moon Semimajor Axis': moon_semi_ma,
                        'Moon Obliquity': solutions[3]
                    }
                )

            else:
                self.solutions = pd.DataFrame(
                    {
                        'Times': times,
                        'Planet Omega': planet_omega,
                        'Planet Mean Motion': planet_mean_motion,
                        'Moon Mean Motion': moon_mean_motion,
                        'Moon Semimajor Axis': moon_semi_ma
                    }
                )

            self.solutions.index.name = 'Simulation Step'

            self.solution_units = {
                'Times': u.s,
                'Planet Omega': u.s**-1,
                'Planet Mean Motion': u.s**-1,
                'Moon Mean Motion': u.s**-1,
                'Moon Semimajor Axis': u.m,
            }
            if self.initial_conds['hm_ini'] != 0.0:
                self.solution_units['Moon Eccentricity'] = u.Unit('')
                self.solution_units['Moon Surface Temperature'] = u.K
            if self.initial_conds['psim_ini'] != 0.0:
                self.solution_units['Moon Obliquity'] = u.Unit('')
            if self.verbose:
                print(f'{moon_fate.prompt}')

    def compute_moon_surface_temperature(self):
        """Compute the surface temperature of a moon for a given moon orbital position
        """
        if self.system_type == 'planet-moon':
            if self.verbose:
                print(
                    '\nStarting integration of moon surface temperature down to the Roche limit:\n'
                )

            nm = self.solutions['Moon Mean Motion']
            eccm = self.solutions['Moon Eccentricity']
            moon_surface_temperature = moon_surface_temper(
                nm,
                eccm,
                self.parameters,
                self.bar_fmt
            )

            self.solutions['Moon Surface Temperature'] = moon_surface_temperature

        else:
            if self.verbose:
                print(f'\nMethod {self.compute_heat_flux.__name__} not defined for other systems')

    def create_moon_temperature_map(
        self,
        periods: np.ndarray = np.arange(0.1, 20.11, 0.04),
        radii: np.ndarray = np.arange(250, 6551, 1000),
        min_temp: float = 0.0,
        max_temp: float = 730.0,
        output_directory: str = Path.home()
    ):
        """Create a map of moon surface temperatures for different orbital periods and moon radii

        Args:
            periods (np.array, optional): Vector of moon orbital periods [d]
            radii (np.array, optional): Vector of moon radii [km]
            min_temp (float, optional): Minimum moon surface temperature [K]
            max_temp (float, optional): Maximum moon surface temperature [K]
            output_directory (str, optional): Output path to save the moon surface temperature map
        """
        output_directory = Path(output_directory, 'Temperature_Maps')
        os.makedirs(output_directory, exist_ok=True)

        file_name = Path(output_directory, f'Temper_map_e{self.moon_eccentricity}_ploonetide.txt')

        with open(file_name, 'w') as file:
            # Vary the orbitaal period of the moon
            for i, period in enumerate(tqdm(periods, desc="Computing temperature map")):
                period = period * DAY  # orbital period [s]
                n = 2. * np.pi / period  # mean motion [1/s]

                # Vary the moon radius
                for j, radius in enumerate(radii):
                    # The rigidity and surface gravity of the moon also change automatically through
                    # the instance 'parameters' of the TidalSimulation class
                    self.moon_radius = radius * 1000. / REARTH  # Moon radius [m]

                    # Calculate the stability tmperature of the moon
                    T_stab = 0
                    T_stab = bisection(n, self.moon_eccentricity, self.parameters)

                    # Calculate the tidal heating of the moon and its surface temperature
                    if T_stab > 0:
                        flux, _ = tidal_heat(T_stab, n, self.moon_eccentricity, self.parameters)
                        T_s = surf_temp(flux)

                    elif T_stab <= 0:
                        T_s = T_stab

                    file.write('%.4e %4.i %.2f\n' % (period / DAY, radius, T_s))

        # Reset the moon radius to the original value defined at the beginning of the simulation
        self.reset_moon_radius()

        # Now, we plot the temperature mac using the plot_moon_tmperature_map of Ploonetide
        plot_moon_temperature_map(
            file_name,
            moon_eccentricity=self.moon_eccentricity,
            min_temp=min_temp,
            max_temp=max_temp
        )

    def plot_tidal_evolution(
        self,
        close_plot=True,
        polar_projection=False,
    ):
        """Method to plot the tidal evolution of a moon around the planet

        Args:
            close_plot (bool, optional): Close the generated plot
            polar_projection (bool, optional): Use a polar projection for the plot
        """
        if self.system_type == 'planet-moon':
            solutions = self.solutions

            # Times of the simulation
            times = solutions["Times"].to_numpy()

            # Quantities used below
            Mp = self.planet_mass.to_value(u.kg)
            Mm = self.moon_mass.to_value(u.kg)
            roche_radius = self.moon_roche_radius.value
            crit_hill = self.planet_critical_hill_radius.value

            fate_time = self.fate_time.to_value(u.Gyr)
            fate = self.fate

            # Interpolated solutions
            times = np.asarray(times)

            t_min = times[0]
            t_max = times[-1]

            t_plot = np.linspace(t_min, t_max, 1000)

            # Force final/event time into the plotting grid.
            if t_plot.size == 0 or not np.isclose(t_plot[-1], t_max):
                t_plot = np.append(t_plot, t_max)

            t_plot = np.unique(
                np.sort(
                    np.concatenate(
                        [
                            np.linspace(t_min, t_max, 1500),
                            times,
                        ]
                    )
                )
            )

            # Dense-output state from solve_ivp
            y_plot = self.sols.sol(t_plot)

            op_plot = y_plot[0]
            np_plot = y_plot[1]

            # y[2] = log(nm)
            nm_plot = np.exp(y_plot[2])

            # Use nm_plot, not nm
            am_plot = mean2axis(nm_plot, Mp, Mm) / roche_radius

            # Caclulate the co-rotation radius
            am_corot = corotation_radius(Mp, op_plot) / roche_radius

            if self.has_eccentricity:
                hm_plot = np.maximum(y_plot[3], 0.0)
                em_plot = np.sqrt(hm_plot)

                if fate == "roche":
                    y = am_plot * (1.0 - em_plot)
                    label_mig = r"$q_{\rm m}$"
                elif fate == "hill":
                    y = am_plot * (1.0 + em_plot)
                    label_mig = r"$Q_{\rm m}$"
                else:
                    y = am_plot
                    label_mig = r"$𝑎_{\rm m}$"
            else:
                em_plot = np.zeros_like(am_plot)
                y = am_plot
                label_mig = r"$𝑎_{\rm m}$"

            labelsize = 9.0
            ticksize = 8.0

            x = t_plot / GYEAR

            if polar_projection:
                # Polar time coordinate: the evolving tracks do not close completely,
                # avoiding an artificial seam between the first and final points.
                theta_span = t_max - t_min
                theta_track_max = 1.90 * np.pi

                if theta_span <= 0:
                    theta = np.zeros_like(t_plot)
                else:
                    theta = theta_track_max * (t_plot - t_min) / theta_span

                from matplotlib.gridspec import GridSpec

                # ============================================================
                # Figure layout: polar orbital map + frequency evolution
                # ============================================================
                fig = plt.figure(figsize=(11.5, 4.4))
                gs = GridSpec(
                    1,
                    2,
                    width_ratios=[1.15, 1.0],
                    wspace=0.32,
                    figure=fig,
                )

                ax0 = fig.add_subplot(gs[0, 0], projection="polar")
                ax1 = fig.add_subplot(gs[0, 1])
                ax = [ax0, ax1]

                # ============================================================
                # Top panel: polar tidal migration map
                # ============================================================
                # Full circle for time-independent physical boundaries.
                theta_circle = np.linspace(0.0, 2.0 * np.pi, 512)

                r_roche = 1.0
                r_hill = crit_hill / roche_radius

                # ------------------------------------------------------------
                # Radial limits: log scale, including moon track, corotation,
                # Roche limit, and critical radius.
                # ------------------------------------------------------------
                r_candidates = [
                    am_plot,
                    am_corot,
                    np.full_like(theta_circle, r_roche),
                    np.full_like(theta_circle, r_hill),
                ]

                if self.has_eccentricity:
                    r_candidates.append(y)

                r_all = np.concatenate([
                    np.asarray(r)[np.isfinite(r)]
                    for r in r_candidates
                ])

                r_all = r_all[r_all > 0.0]

                if r_all.size > 0:
                    r_min = 0.75 * np.nanmin(r_all)
                    r_max = 1.25 * np.nanmax(r_all)

                    # Always keep the Roche limit visually inside the frame.
                    r_min = min(r_min, 0.75 * r_roche)
                    r_max = max(r_max, 1.25 * r_roche)

                    ax[0].set_rlim(r_min, r_max)

                # Log radial scale is important here because corotation can move far
                # from the moon orbit before sweeping across it.
                ax[0].set_rscale("log")

                # ------------------------------------------------------------
                # Shaded regions
                # ------------------------------------------------------------
                r_lower, r_upper = ax[0].get_ylim()

                # Roche-forbidden region.
                ax[0].fill_between(
                    theta_circle,
                    r_lower,
                    r_roche,
                    alpha=0.22,
                    color="w",
                    zorder=0,
                )

                ax[0].fill_between(
                    theta_circle,
                    r_roche,
                    r_hill,
                    alpha=0.32,
                    color="lightblue",
                    zorder=0,
                )

                # Hill-unstable region.
                if r_hill < r_upper:
                    ax[0].fill_between(
                        theta_circle,
                        r_hill,
                        r_upper,
                        alpha=0.18,
                        color="w",
                        zorder=0,
                    )

                # ------------------------------------------------------------
                # Dynamical boundaries
                # ------------------------------------------------------------
                ax[0].plot(
                    theta_circle,
                    np.full_like(theta_circle, r_roche),
                    ls="--",
                    lw=0.9,
                    color="k",
                    zorder=1,
                    # label="Roche limit",
                    label=r" $𝑎_\mathrm{Roche}$",
                )

                if r_hill < r_upper:
                    ax[0].plot(
                        theta_circle,
                        np.full_like(theta_circle, r_hill),
                        ls="-.",
                        lw=0.9,
                        color="k",
                        zorder=1,
                        # label="Critical radius",
                        label=r"$𝑎_\mathrm{crit}=$" f"{r_hill:.2f}" r"$𝑎_\mathrm{Roche}$",
                    )

                # ------------------------------------------------------------
                # Corotation radius𝑎
                # ------------------------------------------------------------
                ax[0].plot(
                    theta,
                    am_corot,
                    ls="--",
                    lw=1.0,
                    color="tab:orange",
                    alpha=0.9,
                    zorder=3,
                    label="Corotation radius",
                )

                # Moon migration track, colour-coded by time.
                sc = ax[0].scatter(
                    theta,
                    y,
                    c=x,
                    s=7,
                    cmap="viridis",
                    edgecolors="none",
                    zorder=5,
                )
                # Thin guide line for the moon migration track.
                # ax[0].plot(
                #     theta,
                #     am_plot,
                #     lw=1.0,
                #     color="0.15",
                #     alpha=0.35,
                #     zorder=4,
                # )
                if (
                    self.has_eccentricity
                    and fate in ["roche", "hill"]
                ):
                    fate_radius = y[-1]
                else:
                    fate_radius = am_plot[-1]

                # ------------------------------------------------------------
                # Start and fate markers
                # ------------------------------------------------------------
                ax[0].scatter(
                    0.5,
                    0.5,
                    s=80,
                    marker="o",
                    facecolor="mediumblue",
                    alpha=0.5,
                    edgecolor="mediumblue",
                    linewidth=1.5,
                    transform=ax[0].transAxes,
                    zorder=20,
                    label="Planet",
                )

                ax[0].scatter(
                    theta[0],
                    y[0],
                    s=15,
                    marker="o",
                    facecolor="white",
                    edgecolor="k",
                    linewidth=1.1,
                    zorder=7,
                    label=f"Moon ({label_mig})",
                )

                if fate == "roche":
                    fate_marker = "x"
                    fate_label = r"$\mathbf{Roche\ disruption}$"
                elif fate == "hill":
                    fate_marker = "x"
                    fate_label = r"$\mathbf{Hill\ escape}$"
                elif fate == "survives":
                    fate_marker = ">"
                    fate_label = r"$\mathbf{Long\!-\!term\ survival}$"
                else:
                    fate_marker = "D"
                    fate_label = rf"$\mathbf{{{fate}}}$"

                ax[0].scatter(
                    theta[-1],
                    fate_radius,
                    s=30 if fate == "survives" else 20,
                    marker=fate_marker,
                    color="k",
                    zorder=8,
                    label=fate_label,
                )

                # ------------------------------------------------------------
                # Polar formatting
                # ------------------------------------------------------------
                ax[0].set_theta_zero_location("N")
                ax[0].set_theta_direction(-1)

                # Time is shown by the colour bar, so remove angular tick labels.
                ax[0].set_xticks([])
                ax[0].set_xticklabels([])

                ax[0].set_ylabel(
                    r"Distance from planet $(𝑎_{\rm Roche})$",
                    fontsize=labelsize,
                    labelpad=22,
                )

                ax[0].tick_params(axis="both", labelsize=ticksize)
                ax[0].xaxis.grid(False)          # angular/theta grid off
                ax[0].yaxis.grid(True, alpha=0.35)  # radial/r grid on

                # Colour bar for time.
                cbar = fig.colorbar(
                    sc,
                    ax=ax[0],
                    orientation="horizontal",
                    pad=0.12,
                    fraction=0.050,
                    aspect=30,
                )

                t_min_cbar = 0.0
                t_max_cbar = np.nanmax(x)

                cbar_ticks = np.linspace(t_min_cbar, t_max_cbar, 5)

                cbar.set_ticks(cbar_ticks)
                cbar.set_ticklabels([f"{tick:.2g}" for tick in cbar_ticks])

                cbar.set_label(r"Moon survival time (Gyr)", fontsize=labelsize)
                cbar.ax.tick_params(labelsize=ticksize)

                ax[0].legend(
                    ncol=6,
                    fontsize=labelsize,
                    bbox_to_anchor=(1.35, 1.02),
                    loc="lower center",
                    frameon=False,
                )

            else:
                fig, ax = plt.subplots(2, figsize=(6, 5.5), sharex=True)
                plt.subplots_adjust(hspace=0.0)
                # ax[0].set_facecolor("ghostwhite")

                ax[0].plot(x, am_plot, label=r'$𝑎_{\rm m}$')
                ax[0].plot(x, am_corot, c='orange', ls='--', lw=1.0, label=r'$𝑎_{\rm CR}$')

                if self.has_eccentricity:
                    ax[0].plot(x, y, ':k', label=label_mig)

                ax[0].set_yscale('log')

                ax[0].axhline(
                    crit_hill / roche_radius,
                    c="k", ls="-.", lw=0.9, zorder=0.0,
                    label=r"$𝑎_{\rm crit, Hill}$"
                )

                ax[0].axhline(
                    roche_radius / roche_radius,
                    c="k", ls=":", lw=0.9, zorder=0.0,
                    label=r"$𝑎_{\rm Roche}$"
                )

                # x_span = max(fate_time - x.min(), 1e-6)
                # pad = 0.015 * x_span
                pad = 0.015 * (fate_time - x.min())
                ax[0].set_xlim(x.min(), fate_time + pad)

                ax[0].tick_params(axis='both', direction='in', labelsize=ticksize)

                # ax[0].set_xlabel('Time (Gyr)', fontsize=labelsize)
                ax[0].set_ylabel(r'Circumplanetary position ($𝑎_\mathrm{Roche}$)', fontsize=labelsize)
                ax[0].legend(ncol=4, fontsize=labelsize + 1, bbox_to_anchor=(0.5, 0.99), loc='lower center')

            # ============================================================
            # Bottom panel: frequency evolution
            # ============================================================
            ax[1].plot(x, nm_plot * DAY, "b-", label=r"$n_{\rm m}$")
            ax[1].plot(x, np_plot * DAY, "k-", label=r"$n_{\rm p}$")
            ax[1].plot(x, op_plot * DAY, "r-", lw=1.0, label=r"$\Omega_{\rm p}$")
            # ax[1].set_yscale("log")

            ax[1].tick_params(axis="both", direction="in", labelsize=ticksize)
            ax[1].set_xlabel("Time (Gyr)", fontsize=labelsize)
            ax[1].set_ylabel(r"Frequency (d$^{-1}$)", fontsize=labelsize)
            ax[1].legend(
                ncol=3,
                fontsize=labelsize + 1,
                loc="upper center",
            )

            if close_plot:
                plt.close()
            return fig
