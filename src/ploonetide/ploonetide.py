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

from . import PACKAGEDIR
from ploonetide.utils.constants import PLANETS
from ploonetide.utils.functions import *
from ploonetide.odes.planet_moon import solution_planet_moon
from ploonetide.odes.star_planet import solution_star_planet
from ploonetide.forecaster.mr_forecast import Mstat2R
from ploonetide.numerical.simulator import Variable, Simulation

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

    def __init__(
        self,
        activation_energy=3E5,
        heat_capacity=1260,
        mantle_thickness=3E6,
        melt_fraction_coeff=40.,
        solidus_temperature=1600.,
        breakdown_temperature=1800.,
        liquidus_temperature=2000.,
        surface_temperature_earth=288.,
        thermal_conductivity=2.,
        Rayleigh_critical=1100.,
        flow_geometry=1.,
        thermal_expansivity=1E-4,
        planet_size_evolution=False,
        planet_envelope_dissipation=False,
        planet_core_dissipation=False,
        star_internal_evolution=False,
        star_mass=1.,
        star_radius=1.,
        star_eff_temperature=3700.,
        star_saturation_rate=4.3421E-5,
        star_angular_coeff=0.5,
        star_rotperiod=10,
        star_alpha=0.25,
        star_beta=0.25,
        star_age=5.,
        sun_omega=2.67E-6,
        sun_mass_loss_rate=1.4E-14,
        planet_mass=1.,
        planet_radius=None,
        planet_angular_coeff=0.26401,
        planet_orbperiod=None,
        planet_rotperiod=0.6,
        planet_eccentricity=0.1,
        planet_rigidity=4.46E10,
        planet_alpha=PLANETS.Saturn.alpha,
        planet_beta=PLANETS.Saturn.beta,
        moon_radius=1,
        moon_density=5515, moon_albedo=0.3,
        moon_eccentricity=0.02,
        moon_semimaxis=10,
        moon_type='rigid',
        system_type='star-planet',
        verbose=True
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
            planet_size_evolution (bool, optional): Description
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
            planet_alpha (float, optional): Description
            planet_beta (float, optional): Description
            moon_radius (int, optional): Moon radius [Rearth]
            moon_density (int, optional): Moon density [kg m**-3]
            moon_albedo (float, optional): Description
            moon_eccentricity (float, optional): Eccentricity of moon's orbit [No unit]
            moon_semimaxis (int, optional): Description
            moon_type (bool, optional): Description, default is 'rigid'
            system_type (str, optional): Description
            verbose (bool, optional): Description
        """
        if verbose:
            print(pyfiglet.figlet_format(f'{self.package()}'))

        # ************************************************************
        # SET THE TYPE OF SYSTEM
        # ************************************************************
        self.system_type = system_type

        # ************************************************************
        # KEY TO INCLIDE EVOLUTION
        # ************************************************************
        self._planet_size_evolution = planet_size_evolution
        self._planet_envelope_dissipation = planet_envelope_dissipation
        self._planet_core_dissipation = planet_core_dissipation
        self._star_internal_evolution = star_internal_evolution

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
        self._planet_rigidity = u.Quantity(planet_rigidity, u.Pa)
        self._planet_angular_coeff = planet_angular_coeff

        self._planet_eccentricity = planet_eccentricity
        if self.planet_eccentricity < 0 or self.planet_eccentricity > 1:
            raise ValueError("Planet eccentricity must be 0 ≤ ep ≤ 1")

        self._planet_alpha = planet_alpha
        self._planet_beta = planet_beta

        # ************************************************************
        # MOON PARAMETERS
        # ************************************************************
        self._moon_density = u.Quantity(moon_density, u.kg * u.m**-3)
        self._moon_radius = u.Quantity(moon_radius, u.Rearth)
        self._moon_radius_set = u.Quantity(moon_radius, u.Rearth)
        self._moon_albedo = moon_albedo
        self._moon_type = moon_type

        self.verbose = verbose

        self._moon_eccentricity = moon_eccentricity
        if self._moon_eccentricity < 0 or self._moon_eccentricity > 1:
            raise ValueError("Moon eccentricity must be 0 ≤ em ≤ 1")

        self._moon_semimaxis = u.Quantity(moon_semimaxis * self.moon_roche_radius.value, u.m)
        if self._moon_semimaxis > self.planet_critical_hill_radius:
            ah_c = self.planet_critical_hill_radius / self.moon_roche_radius
            raise ValueError(f"Moon semimajor axis must be smaller than critical Hill radius (a_crit = {ah_c:.3f} R_roche)")

        # Arguments for including/excluding different effects
        self.args = dict(
            star_internal_evolution=self._star_internal_evolution,
            star_k2q=self.star_k2q,
            planet_envelope_dissipation=self._planet_envelope_dissipation,
            planet_k2q=self.planet_k2q,
            planet_size_evolution=self._planet_size_evolution,
            Rp=self.planet_radius.to_value(u.m),
            planet_core_dissipation=self._planet_core_dissipation,
        )

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
            omega_p = Variable('omega_planet', self.initial_conds['op_ini'])
            motion_p = Variable('mean_motion_p', self.initial_conds['np_ini'])
            motion_m = Variable('mean_motion_m', self.initial_conds['nm_ini'])
            eccen_m = Variable('eccentricity', self.initial_conds['em_ini'])
            if self.initial_conds['em_ini'] != 0.0:
                initial_variables = [omega_p, motion_p, motion_m, eccen_m]
            else:
                initial_variables = [omega_p, motion_p, motion_m]
            if self.verbose:
                print(
                    f' Stellar age: {self.star_age:.3f}\n',
                    f'Stellar mass: {self.star_mass:.3f}\n',
                    f'Stellar radius: {self.star_radius:.3f}\n',
                    f'Stellar rotation period: {self.star_rotperiod:.3f}\n',
                    f'Planet orbital period: {self.planet_orbperiod:.3f}\n',
                    f'Planet mass: {self.planet_mass:.3f}\n',
                    f'Planet radius: {self.planet_radius:.3f}\n',
                    f'Planet eccentricity: {self.planet_eccentricity:.3f}\n',
                    f'Moon density: {self.moon_density:.3f}\n',
                    f'Moon radius: {self.moon_radius:.3f}\n',
                    f'Moon mass: {self.moon_mass.to(u.Mearth):.5f}\n',
                    f'Moon eccentricity: {self.moon_eccentricity:.3f}\n',
                    f'Moon semimajor axis: {moon_semimaxis:.3f} a_roche\n',
                    f'Moon orbital period: {self.moon_orbperiod:.3f}'
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
            Mp=self.planet_mass.to_value(u.kg),
            Rp=self.planet_radius.to_value(u.m),
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
            Mm=self.moon_mass.value,
            Rm=self.moon_radius.to_value(u.m),
            gravm=self.moon_gravity.value,
            rigidm=self.moon_rigidity.value,
            T_surface=self.surface_temperature_earth.value,
            sun_mass_loss_rate=self.sun_mass_loss_rate.to_value(u.kg * u.s**-1),
            sun_omega=self.sun_omega.value,
            coeff_star=self._star_angular_coeff,
            coeff_planet=self._planet_angular_coeff,
            star_alpha=self._star_alpha,
            star_beta=self._star_beta,
            planet_alpha=self._planet_alpha,
            planet_beta=self._planet_beta,
            a2=self._flow_geometry,
            Rac=self._Rayleigh_critical,
            B=self._melt_fraction_coeff,
            args=self.args
        )

    @property
    def initial_conds(self):
        """Dictionary with all the initial conditions of the simulation

        Returns:
            dict: Dictionary containing only the initial values of all the integrating variables
        """
        return dict(
            nm_ini=self.moon_meanmo.value,
            em_ini=self.moon_eccentricity,
            Tm_ini=self.moon_initial_temperature.value,
            os_ini=self.star_omega.value,
            np_ini=self.planet_meanmo.value,
            op_ini=self.planet_omega.value,
            ep_ini=self.planet_eccentricity,
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
            raise ValueError("Radius must be positive.")
        self._planet_radius = value
        if not isinstance(self._planet_radius, u.Quantity):
            if not value:
                self._planet_radius, _, _ = Mstat2R(
                    mean=self.planet_mass.value, std=0.1, unit='Jupiter',
                    sample_size=200, classify='Yes'
                )
                self._planet_radius = u.Quantity(self._planet_radius, u.R_jup)

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
    def planet_k2q(self):
        if self.__planet_core_dissipation:
            return k2Q_planet_envelope(
                self._planet_alpha,
                self._planet_beta,
                self.planet_epsilon) + k2Q_planet_core(
                self.planet_rigidity.value,
                self._planet_alpha,
                self._planet_beta,
                self.planet_mass.to_value(u.kg), self.planet_radius.to_value(u.m))
        else:
            return k2Q_planet_envelope(self._planet_alpha, self._planet_beta, self.planet_epsilon)

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

    # **********************************************************************************************
    # ******************************** MOON PROPERTIES *********************************************
    # **********************************************************************************************

    # ******************* Properties that could changed from outside the class *********************

    @property
    def moon_radius(self):
        """Moon radius

        Returns:
            float: Moon radius [Earth radii]
        """
        return self._moon_radius

    @moon_radius.setter
    def moon_radius(self, value):
        """Set moon radius to new value

        Args:
            value (float): Moon radius [Earth radii]
        """
        if value <= 0:
            raise ValueError("Radius must be positive.")
        self._moon_radius = value
        if not isinstance(self._moon_radius, u.Quantity):
            self._moon_radius = u.Quantity(value, u.Rearth)

    def reset_moon_radius(self):
        """Reset moon radius to original value in the simulation after creating temperature map
        """
        self._moon_radius = self._moon_radius_set

    @property
    def moon_density(self):
        """Moon density

        Returns:
            float: Moon density [kg m^-3]
        """
        return self._moon_density

    @moon_density.setter
    def moon_density(self, value):
        """Set moon density to new value

        Args:
            value (float): Moon density [kg m^-3]
        """
        if value <= 0:
            raise ValueError("Density must be positive.")
        self._moon_density = value
        if not isinstance(self._moon_density, u.Quantity):
            self._moon_density = u.Quantity(value, u.kg * u.m**-3)

    @property
    def moon_semimaxis(self):
        """Moon semimajor axis

        Returns:
            float: Moon semimajor axis [Roche radii]
        """
        if self._moon_semimaxis.value / self.moon_roche_radius.value <= 1:
            raise ValueError("Moon semimajor axis must be greater than the Roche limit.")
        return self._moon_semimaxis

    @moon_semimaxis.setter
    def moon_semimaxis(self, value):
        """Set moon semimajor axis to new value

        Args:
            value (float): Moon semimajor axis [Roche radii]
        """
        if value <= 0:
            raise ValueError("Semimajor axis must be positive.")

        if value > self.planet_critical_hill_radius / self.moon_roche_radius:
            ah_c = self.planet_critical_hill_radius / self.moon_roche_radius
            raise ValueError(f"Moon semimajor axis must be smaller than critical Hill radius (a_crit = {ah_c:.3f} R_roche)")

        if value <= 1:
            raise ValueError("Moon semimajor axis must be greater than the Roche limit.")
        self._moon_semimaxis = value
        if not isinstance(self._moon_semimaxis, u.Quantity):
            self._moon_semimaxis = u.Quantity(value * self.moon_roche_radius.value, u.m)

    @property
    def moon_eccentricity(self):
        return self._moon_eccentricity

    @moon_eccentricity.setter
    def moon_eccentricity(self, value):
        if value < 0 or value > 1:
            raise ValueError("Moon eccentricity must be 0 ≤ em ≤ 1")
        self._moon_eccentricity = value

    #  ************************* Properties that are calculated internally *************************

    @property
    def moon_mass(self):
        """Moon mass calculated from its radius and density

        Returns:
            float: Moon mass [kg]
        """
        return u.Quantity(
            self.moon_density.value * (4. / 3. * np.pi * self.moon_radius.to_value(u.m)**3.),
            u.kg
        )

    @property
    def moon_roche_radius(self):
        """Roche radius of a solid moon

        Returns:
            float: Roche radius of a moon [m]
        """

        density_planet = density(
            self.planet_mass.to_value(u.kg),
            self.planet_radius.to_value(u.m)
        )
        if self._moon_type == 'fluid':
            return u.Quantity(
                roche_radius_fluid(
                    self.planet_radius.to_value(u.m),
                    density_planet,
                    self.moon_density.value
                ),
                u.m
            )
        else:
            return u.Quantity(
                roche_radius_rigid(
                    self.planet_radius.to_value(u.m),
                    density_planet,
                    self.moon_density.value
                ),
                u.m
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
    def moon_rigidity(self):
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
    def __getattr__(self, name):
        return f'{self.get_class_name()} does not have "{str(name)}" attribute'

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
        differential_equation = solution_star_planet
        if self.system_type == 'planet-moon':
            differential_equation = solution_planet_moon
        super().set_diff_eq(differential_equation, self.parameters, self.initial_conds)

        if self.verbose:
            print('\nStarting integration of moon orbital migration:\n')

        super().run(integration_time, timestep, t0=0)

        if self.system_type == 'planet-moon':
            times, solutions = self.history

            moon_fate = find_moon_fate(
                times,
                self.star_mass.to_value(u.kg),
                self.planet_mass.to_value(u.kg),
                self.moon_mass.value,
                solutions[2],
                self.moon_roche_radius.value,
                self.planet_hill_radius.value,
                self.planet_eccentricity,
                self.moon_eccentricity
            )

            self.fate_time = u.Quantity(moon_fate.time, u.Gyr)
            self.fate = moon_fate.fate

            moon_semi_ma = mean2axis(
                solutions[2][:moon_fate.index],
                self.planet_mass.to_value(u.kg),
                self.moon_mass.value
            )

            if self.initial_conds['em_ini'] == 0.0:
                self.history = pd.DataFrame(
                    {'Times': times[:moon_fate.index],
                     'Planet Omega': solutions[0][:moon_fate.index],
                     'Planet Mean Motion': solutions[1][:moon_fate.index],
                     'Moon Mean Motion': solutions[2][:moon_fate.index],
                     'Moon Semimajor Axis': moon_semi_ma
                     }
                )
            elif self.initial_conds['em_ini'] != 0.0:
                self.history = pd.DataFrame(
                    {'Times': times[:moon_fate.index],
                     'Planet Omega': solutions[0][:moon_fate.index],
                     'Planet Mean Motion': solutions[1][:moon_fate.index],
                     'Moon Mean Motion': solutions[2][:moon_fate.index],
                     'Moon Semimajor Axis': moon_semi_ma,
                     'Moon Eccentricity': solutions[3][:moon_fate.index]}
                )
            self.history.index.name = 'Simulation Step'

            self.history_units = {
                'Times': u.s,
                'Planet Omega': u.s**-1,
                'Planet Mean Motion': u.s**-1,
                'Moon Mean Motion': u.s**-1,
                'Moon Semimajor Axis': u.m,
            }
            if self.initial_conds['em_ini'] != 0.0:
                self.history_units['Moon Eccentricity'] = u.Unit('')
                self.history_units['Moon Surface Temperature'] = u.K
            if self.verbose:
                print(f'{moon_fate.prompt}')

    def compute_moon_surface_temperature(self):
        """Compute the surface temperature of a moon for a given moon orbital position
        """
        if self.system_type == 'planet-moon':
            if self.verbose:
                print('\nStarting integration of moon surface temperature down to the Roche limit:\n')

            nm = self.history['Moon Mean Motion']
            eccm = self.history['Moon Eccentricity']
            moon_surface_temperature = moon_surface_temper(
                nm,
                eccm,
                self.parameters,
                self.bar_fmt
            )

            self.history['Moon Surface Temperature'] = moon_surface_temperature

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
