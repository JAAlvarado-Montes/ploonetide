"""This module contains all the general functions needed for Ploonetide calculations"""
import os
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt

from collections import namedtuple

from ploonetide.utils.constants import *
from ploonetide.utils import make_rgb_colormap


#############################################################
# SPECIFIC ROUTINES
#############################################################
def k2Q_star_envelope(alpha, beta, epsilon):
    """Calculate tidal heat function for a stellar envelope (Source: Mathis, 2015).

      Args:
          alpha (float): star's core size fraction [Rc/Rs]
          beta (float): star's core mass fraction [Mc/Ms]
          epsilon (float): star's rotational rate [Omega/Omega_crit]
          args (list, optional): contains behaviour

      Returns:
          float: tidal heat function
    """
    gamma = alpha**3. * (1 - beta) / (beta * (1 - alpha**3.))

    line1 = 100 * np.pi / 63 * epsilon**2 * (alpha**5. / (1 - alpha**5.)) * (1 - gamma)**2.
    line2 = ((1 - alpha)**4.0 * (1 + 2 * alpha + 3 * alpha**2. + 1.5 * alpha**3.)**2.0
             * (1 + (1 - gamma) / gamma * alpha**3.))
    line3 = (1 + 1.5 * gamma + 2.5 / gamma * (1 + 0.5 * gamma - 1.5 * gamma**2.)
             * alpha**3. - 9. / 4. * (1 - gamma) * alpha**5.)

    k2q1 = line1 * line2 / line3**2.0

    return k2q1


def k2Q_planet_envelope(alpha, beta, epsilon):
    """Calculate tidal heat function for the planet's envelope (Source: Mathis, 2015).

      Args:
          alpha (float): planet's core size fraction [Rc/Rp]
          beta (float): planet's core mass fraction [Mc/Mp]
          epsilon: planetary rotational rate (Omega/Omega_crit)

      Returns:
          float: tidal heat function

    """
    fac0 = alpha**3.0
    fac1 = alpha**5.0
    fac2 = fac1 / (1 - fac1)

    gamma = fac0 * (1 - beta) / (beta * (1 - fac0))
    fac3 = (1 - gamma) / gamma * fac0

    k2q = 100 * np.pi / 63 * epsilon**2 * fac2 * (1 + fac3) / (1 + 5. / 2 * fac3)**2

    return k2q


def k2Q_planet_core(G, alpha, beta, Mp, Rp):
    """Calculates the tidal heat function of a planet's rigid core (Source: Mathis, 2015).

    Parameters
    ----------
    G : `float`
        planet's core rigidity
    alpha : `float`
        planet's core size fraction [Rc/Rp]
    beta : `float`
        planet's core mass fraction [Mc/Mp]
    Mp : `float`
        planet's mass [kg]
    Rp : `float`
        planet's radius [m]

    Returns
    --------
    tidal heat function : float
        Tidal heat function of a rigid core for the planet.
    """
    gamma = alpha**3.0 * (1 - beta) / (beta * (1 - alpha**3.0))

    AA = 1.0 + 2.5 * gamma**(-1.0) * alpha**3.0 * (1.0 - gamma)
    BB = alpha**(-5.0) * (1.0 - gamma)**(-2.0)
    CC = (38.0 * np.pi * (alpha * Rp)**4.0) / (3.0 * GCONST * (beta * Mp)**2.0)
    DD = (2.0 / 3.0) * AA * BB * (1.0 - gamma) * (1.0 + 1.5 * gamma) - 1.5

    num = np.pi * G * (3.0 + 2.0 * AA)**2.0 * BB * CC
    den = DD * (6.0 * DD + 4.0 * AA * BB * CC * G)
    k2qcore = num / den
    return k2qcore


# ############RODRIGUEZ 2011########################
def S(kQ1, Mp, Ms, Rs):
    return (9 * kQ1 * Mp * Rs**5.0) / (Ms * 4.0)


def p(kQ, Mp, Ms, Rp):
    return (9 * kQ * Ms * Rp**5.0) / (Mp * 2.0)


def D(pp, SS):
    return pp / (2 * SS)
# ############RODRIGUEZ 2011########################


def Mp2Rp(Mp, t):
    if Mp >= PLANETS.Jupiter.M:
        rad = PLANETS.Jupiter.R
    else:
        rad = PLANETS.Saturn.R
    Rp = rad * A * ((t / YEAR + t0) / C)**B
    return Rp


def mloss_atmo(t, Ls, a, Mp, Rp):
    """Calculate loss of mass in the atmoshpere of the planet.

    Args:
        t (float): time
        Ls (float): stellar luminosity [W]
        a (float): planetary semi-major axis [m]
        Mp (float): mass of the planet [kg]
        Rp (float): radius of the planet [m]

    Returns:
        float: loss rate of atmospheric mass
    """
    #  Zuluaga et. al (2012)
    ti = 0.06 * GYEAR * (Ls / LSUN)**-0.65

    if t < ti:
        Lx = 6.3E-4 * Ls
    else:
        Lx = 1.8928E28 * t**(-1.55)
    # Sanz-forcada et. al (2011)
    Leuv = 10**(4.8 + 0.86 * np.log10(Lx))
    k_param = 1.0  # Sanz-forcada et. al (2011)

    lxuv = (Lx + Leuv) * 1E-7
    fxuv = lxuv / (4 * np.pi * a**2.0)

    num = np.pi * Rp**3.0 * fxuv
    deno = GCONST * Mp * k_param
    return num / deno


def mloss_dragging(a, Rp, Rs, Ms, oms, sun_mass_loss_rate, sun_omega):
    """Calculate mass loss in the planet fue to atmospheric dragging."""
    alpha_eff = 0.3  # Zendejas et. al (2010) Venus

    return (Rp / a)**2.0 * mloss_star(Rs, Ms, oms, sun_mass_loss_rate, sun_omega) * alpha_eff / 2.0


def mloss_star(Rs, Ms, oms, sun_mass_loss_rate, sun_omega):
    """Calculate the loss of mass in the star due to wind."""
    # smlr_sun = 1.4E-14 * MSUN / YEAR  # Zendejas et. al (2010) - smlr sun
    # oms_sun = 2.67E-6
    m_loss = (sun_mass_loss_rate * (Rs / RSUN)**2.0
              * (oms / sun_omega)**1.33 * (Ms / MSUN)**-3.36)

    return m_loss


def omegadt_braking(kappa, OS, OS_saturation, osini, dobbs=False):
    """Calculate the rate of magnetic braking in th star."""
    if dobbs:
        gam = 1.0
        tao = GYEAR
        odt_braking = -gam / 2 * (osini / tao) * (OS / osini)**3.0
        return odt_braking

    if isinstance(OS, np.ndarray):
        odt_braking = []
        for k, o in zip(kappa, OS):
            odtb = []
            for i in range(len(k)):
                odtb.append(-k[i] * o[i] * min(o[i], OS_saturation)**2.0)
            odt_braking.append(np.array(odtb))
        return odt_braking
    odt_braking = -kappa * OS * min(OS, OS_saturation)**2.0

    return odt_braking


def kappa_braking(OS, stellar_age, skumanich=True, alpha=0.495):
    """Calulate the kappa coefficient for mangnetic braking."""
    alpha_s = 0.5  # Skumanich (1972)
    kappa = OS**-2.0 / (2.0 * stellar_age)  # Weber-Davis

    if not skumanich:
        alpha_s = alpha  # Brown et. al (2011)
        kappa = OS**(-1.0 / alpha_s) / (stellar_age / alpha_s)  # Brown (2011)
        return kappa
    return kappa


def aRoche(Mp, densPart=3000, rfac=2.0, **args):
    """Calculate the Roche radius in term of the densities."""
    Rp = PLANETS.Saturn.R  # Since Roche radius does not depend on R this is a hypotetical one
    # Planet average density
    densP = Mp / ((4. / 3) * np.pi * Rp**3)
    # Roche radius
    ar = rfac * Rp * (densPart / densP)**(-1.0 / 3.0)
    return ar


def aRoche_solid(Mp, Mm, Rm):
    """Calculate the Roche radius using the masses.

    Args:
        Mp (float): Planet's mass [kg]
        Mm (float): Moon mass [kg]
        Rm (float): Moon radius [kg]

    Returns:
        float: Roche radius of the body with Mm.
    """
    return Rm * (2. * Mp / Mm)**(1. / 3.)


def hill_radius(a, e, m, M):
    return a * (1 - e) * (m / (3.0 * M))**(1.0 / 3.0)


def alpha2beta(Mp, alpha, **args):
    beta = KP * (Mp / PLANETS.Saturn.M)**DP * alpha**BP
    return beta


def omegaAngular(P):
    return 2 * np.pi / P


def omegaCritic(M, R):
    Oc = np.sqrt(GCONST * M / R**3)
    return Oc


def equil_temp(Ts, Rs, a, Ab):
    T_eq = Ts * (Rs / (2 * a))**0.5 * (1 - Ab)**0.25
    return T_eq


def luminosity(R, T):
    L = 4 * np.pi * R**2.0 * stefan_b_constant * T**4.0
    return u.Quantity(L, u.W)


def semiMajorAxis(P, M, m):
    a = (GCONST * (M + m) * P**2.0 / (2.0 * np.pi)**2.0)**(1.0 / 3.0)
    return a


def meanMotion(a, M, m):
    n = (GCONST * (M + m) / a**3.0)**0.5
    return n


def mean2axis(N, M, m):
    return (GCONST * (M + m) / N**2.0)**(1.0 / 3.0)


def gravity(M, R):

    return GCONST * M / R**2.


def density(M, R):

    return M / (4. / 3 * np.pi * R**3.)


def surf_temp(flux):

    return (flux / stefan_b_constant)**0.25


def stellar_lifespan(Ms):
    """Calculate lifespan of a star.

    Args:
        Ms (float): Stellar mass [kg]

    Returns:
        float: lifespan of the star [s]
    """
    return 10 * (MSUN / Ms)**2.5 * GYEAR


# ###################DOBS-DIXON 2004#######################
def f1e(ee):
    numer = (1 + 3.75 * ee**2.0 + 1.875 * ee**4.0 + 0.078125 * ee**6.0)
    deno = (1 - ee**2.0)**6.5
    return numer / deno


def f2e(ee):
    numer = (1 + 1.5 * ee**2.0 + 0.125 * ee**4.0)
    deno = (1 - ee**2.0)**5.0
    return numer / deno


def f3e(ee):
    numer = (1 + 7.5 * ee**2.0 + 5.625 * ee**4.0 + 0.3125 * ee**6.0)
    deno = (1 - ee**2.0)**6.0
    return numer / deno


def f4e(ee):
    numer = (1 + 3 * ee**2.0 + 0.375 * ee**4.0)
    deno = (1 - ee**2.0)**4.5
    return numer / deno


def factorbet(ee, OM, OS, N, KQ, KQ1, MP, MS, RP, RS):
    fac1 = f1e(ee) - 0.611 * f2e(ee) * (OM / N)
    fac2 = f1e(ee) - 0.611 * f2e(ee) * (OS / N)
    lamb = (KQ / KQ1) * (MS / MP)**2.0 * (RP / RS)**5.0
    return 18.0 / 7.0 * (fac1 + fac2 / lamb)


def power(ee, aa, KQ, Ms, Rp):
    keys = (GCONST * Ms)**1.5 * ((2 * Ms * Rp**5.0 * ee**2.0 * KQ) / 3)
    coeff = 15.75 * aa**(-7.5)
    return coeff * keys
# ###################DOBS-DIXON 2004#######################


def find_moon_fate(t, Ms, Mp, Mm, nm, am_roche, ap_hill):

    nm_roche = meanMotion(am_roche, Mp, Mm)
    np_hill = meanMotion(ap_hill, Ms, Mp)

    scale = GYEAR
    scale_label = 'Gyr'

    if scale != GYEAR:
        scale_label = 'Myr'

    try:
        pos = np.where(nm >= nm_roche)[0][0]
        rt_time = t[pos] / scale
        fate = 'crosses'
        print(f'Moon {fate} the Roche limit in {rt_time:.6f} {scale_label}')
    except IndexError:
        try:
            pos = np.where(nm <= np_hill)[0][0]
            rt_time = t[pos] / scale
            fate = 'escapes'
            print(f'Moon {fate} from the planetary Hill radius in {rt_time:.6f} {scale_label}')
        except IndexError:
            pos = -1
            rt_time = np.max(t) / scale
            fate = "stalls"
            print('Moon migrates too slow and never crosses the Hill radius or the Roche limit.')

    Outputs = namedtuple('Outputs', 'time index fate')

    return Outputs(rt_time, pos, fate)


def mu_below_T_solidus():
    # Shear modulus [Pa]
    return 50 * const.giga


def eta_o(E_act):
    # Viscosity [Pa s]
    # defining viscosity for Earth at T0 = 1000K [Pa*s] (Henning et al 2009)
    eta_set = 1e22
    # eta_set = 1e19 # defining viscosity for Mars at T0 = 1600K [Pa*s] (Shoji & Kurita 2014)
    T0 = 1000.  # defining temperature
    return eta_set / np.exp(E_act / (gas_constant * T0))


def eta_below_T_solidus(T, E_act):

    return eta_o(E_act=E_act) * np.exp(E_act / (gas_constant * T))


def mu_between_T_solidus_T_breakdown(T, mu1=8.2E4, mu2=-40.6):
    # Fischer & Spohn (1990), Eq. 16
    return 10**(mu1 / T + mu2)


def eta_between_T_solidus_T_breakdown(T, E_act, melt_fr, B):
    # Moore (2003)
    return eta_o(E_act=E_act) * np.exp(E_act / (gas_constant * T)) * np.exp(-B * melt_fr)


def mu_between_T_breakdown_T_liquidus():
    # Moore (2003)
    return 1E-7


def eta_between_T_breakdown_T_liquidus(T, melt_fr):
    # Moore (2003)
    return 1E-7 * np.exp(40000. / T) * (1.35 * melt_fr - 0.35)**(-5. / 2.)


def mu_above_T_liquidus():
    # Moore (2003)
    return 1E-7


def eta_above_T_liquidus(T):
    # Moore (2003)
    return 1E-7 * np.exp(40000. / T)


def tidal_heat(T, nm, eccm, parameters):

    # General parameters
    E_act = parameters['E_act']
    B = parameters['B']
    T_solidus = parameters['T_solidus']
    T_breakdown = parameters['T_breakdown']
    T_liquidus = parameters['T_liquidus']

    # Moon properties
    Rm = parameters['Rm']  # Moon radius [m]
    rigidm = parameters['rigidm']  # Effective rigidity of the moon [m^-1 s^-2]

    # Orbital angular frequency of the moon [1/s]
    freq = nm

    if T > T_solidus:
        # melt_fraction: Fraction of melt for ice [No unit]
        melt_fr = (T - T_solidus) / (T_liquidus - T_solidus)  # melt fraction

    if T <= T_solidus:
        mu = mu_below_T_solidus()
        eta = eta_below_T_solidus(T, E_act=E_act)

    elif T_solidus < T <= T_breakdown:
        mu = mu_between_T_solidus_T_breakdown(T)
        eta = eta_between_T_solidus_T_breakdown(T, E_act=E_act, melt_fr=melt_fr, B=B)

    elif T_breakdown < T <= T_liquidus:
        mu = mu_between_T_breakdown_T_liquidus()
        eta = eta_between_T_breakdown_T_liquidus(T, melt_fr=melt_fr)

    else:
        mu = mu_above_T_liquidus()
        eta = eta_above_T_liquidus(T)

    # Imaginary part of the second order Love number, Maxwell model (Henning et al. 2009, table 1)
    if mu == 0:
        k2_Im = 0.
    else:
        numerator = -57 * eta * freq

        denominator = 4 * rigidm * (1. + (1. + 19. * mu / (2. * rigidm))**2. * (eta * freq / mu)**2.)

        k2_Im = numerator / denominator

    # tidal surface flux of the moon [W/m^2] (Fischer & Spohn 1990)
    h_m = (-21. / 2. * k2_Im * Rm**5. * nm**5. * eccm**2. / GCONST) / (4. * np.pi * Rm**2.)

    return (h_m, eta)


def convection_heat(T, eta, parameters):

    # General parameters
    Cp = parameters['Cp']
    ktherm = parameters['ktherm']
    Rac = parameters['Rac']
    a2 = parameters['a2']
    alpha_exp = parameters['alpha_exp']
    d_mantle = parameters['d_mantle']
    T_surface = parameters['T_surface']

    # Moon properties
    rho_m = parameters['densm']  # Density of the moon [kg m^-3]
    g_m = parameters['gravm']  # Gravity of the moon [m s^-2]

    error = False

    delta = 30000.  # Initial guess for boudary layer thickness [m]
    kappa = ktherm / (rho_m * Cp)  # Thermal diffusivity [m^2/s]

    if T == 288.:
        print("___error1___: T = T_surf --> q_BL = 0")
        q_BL = 0
        error = True

    if error:
        return q_BL

    q_BL = ktherm * ((T - T_surface) / delta)  # convective heat flux [W/m^2]

    prev = q_BL + 1.

    difference = abs(q_BL - prev)

    # Iteration for calculating q_BL:
    while difference > 10e-10:
        prev = q_BL

        Ra = alpha_exp * g_m * rho_m * d_mantle**4 * q_BL / (eta * kappa * ktherm)  # Rayleigh number

        # Thickness of the conducting boundary layer [m]
        delta = d_mantle / (2. * a2) * (Ra / Rac)**(-1. / 4.)

        q_BL = ktherm * ((T - T_surface) / delta)  # convective heat flux [W/m^2]

        difference = abs(q_BL - prev)

    return q_BL


def check(difference, prev):

    if difference < 0 and prev > 0:
        T_stable = True
    elif difference == 0 and prev > 0:
        T_stable = True
    elif difference > 0 and prev < 0:
        T_stable = False
    elif difference == 0 and prev < 0:
        T_stable = False
    else:
        T_stable = -1

    return T_stable


def intersection(a, b, nm, eccm, parameters):

    done = False
    again = False
    error = False

    T_equilibrium = 0
    i = 0
    unstable = 0
    c = (a + b) / 2.

    while abs(a - c) >= 0.1:  # 0.1K error allowed

        i = i + 1

        f1a, eta = tidal_heat(a, nm, eccm, parameters)
        f2a = convection_heat(a, eta, parameters)

        f1c, eta = tidal_heat(c, nm, eccm, parameters)
        f2c = convection_heat(c, eta, parameters)

        if f2a == 0 or f2c == 0:
            error = True
            T_equilibrium = -1
            return T_equilibrium

        fa = f1a - f2a

        fc = f1c - f2c

        if (fa * fc) < 0:
            b = c
            c = (a + b) / 2.

        elif (fa * fc) > 0:
            a = c
            c = (a + b) / 2.

        elif (fa * fc) == 0:
            if fa == 0:
                f1a, eta = tidal_heat((a - 0.1), nm, eccm, parameters)
                f2a = convection_heat((a - 0.1), eta, parameters)
                stab = check((f1c - f2c), (f1a - f2a))
                if stab:
                    T_equilibrium = a
                    done = True
                elif not stab:
                    a = a + 0.01
                    unstable = 1
                    again = True
                else:
                    T_equilibrium = -3
                    error = True
            if fc == 0:
                f1c, eta = tidal_heat((c + 0.1), nm, eccm, parameters)
                f2c = convection_heat((c + 0.1), eta, parameters)
                stab = check((f1c - f2c), (f1a - f2a))
                if stab:
                    T_equilibrium = c
                    done = True
                elif not stab:
                    a = c + 0.01
                    unstable = 1
                    again = True
                else:
                    T_equilibrium = -3
                    error = True

        if done:
            break
        if again:
            break
        if error:
            break

    if error:
        if T_equilibrium == -3:
            print("___error3___: T_equilibrium not found??")

    return (T_equilibrium, a, b, again, unstable)


def bisection(nm, eccm, parameters):

    # General parameters
    T_breakdown = parameters['T_breakdown']
    T_liquidus = parameters['T_liquidus']

    a0 = 600.                  # ################ ASK ABOUT THIS OBSCURE PARAMETER #############
    b0 = T_liquidus
    a = a0
    b = T_breakdown
    error = False
    # done = False
    again = False
    T_equilibrium = 0
    i = 0

    c = (a + b) / 2.

    # Find the peak of h_m (derivative=0):
    while abs(a - c) >= 0.01:  # 0.01K error allowed

        i = i + 1

        f1a, eta = tidal_heat(a, nm, eccm, parameters)
        f1c, eta = tidal_heat(c, nm, eccm, parameters)
        f1da, eta = tidal_heat((a + 0.001), nm, eccm, parameters)
        f1dc, eta = tidal_heat((c + 0.001), nm, eccm, parameters)

        df1a = (f1da - f1a) / (a + 0.001 - a)
        df1c = (f1dc - f1c) / (c + 0.001 - c)

        if (df1a * df1c) < 0:
            b = c
            c = (a + b) / 2.

        elif (df1a * df1c) > 0:
            a = c
            c = (a + b) / 2.

        elif (df1a * df1c) == 0:
            if df1a == 0:
                peak = a
            if df1c == 0:
                peak = c

    if b == T_breakdown:
        T_equilibrium = -5
        error = True
    else:
        f1a, eta = tidal_heat(a, nm, eccm, parameters)
        f1b, eta = tidal_heat(b, nm, eccm, parameters)
        peak = (a + b) / 2.

    if error:
        if T_equilibrium == -5:
            print("___error5___: no peak of tidal heat flux??")
            return T_equilibrium

    # Find T_stab between the peak and b0:
    a = peak
    b = b0
    un1 = 0
    un2 = 0

    T_equilibrium, a, b, again, unstable = intersection(a, b, nm, eccm, parameters)

    if T_equilibrium == -3:
        return T_equilibrium

    un1 = unstable

    if T_equilibrium != 0:
        return T_equilibrium
    elif b == b0:    # T_equilibrium not found or does not exist
        b = peak
        a = a0
        again = True    # try again below the peak
    elif un1 == 0:
        f1a, eta = tidal_heat(a, nm, eccm, parameters)
        f2a = convection_heat(a, eta, parameters)

        f1b, eta = tidal_heat(b, nm, eccm, parameters)
        f2b = convection_heat(b, eta, parameters)

        if f2a == 0 or f2b == 0:
            error = True
            T_equilibrium = -1

        stab = check((f1b - f2b), (f1a - f2a))

        if stab:
            T_equilibrium = (a + b) / 2.    # stable point (T_stab) is found
            return T_equilibrium
        elif not stab:    # unstable point is found
            a = b
            b = b0
            un1 = un1 + 1
            again = True    # try again above the unstable point
        else:
            T_equilibrium = -3
            error = True

    if error:
        if T_equilibrium == -3:
            print("___error3___: T_equilibrium not found??")
        return T_equilibrium

    # Search for T_stab again (below the peak or above the unstable point)
    if again:

        T_equilibrium, a, b, again, unstable = intersection(a, b, nm, eccm, parameters)

        if T_equilibrium == -3:
            return T_equilibrium

        un2 = unstable

        if T_equilibrium != 0:
            return T_equilibrium
        elif b == b0:    # T_unstab is found, but T_stab is not found
            return T_equilibrium  # This is because the unstable and stable points are too close
        elif b == peak:    # T_equilibrium does not exist
            return T_equilibrium
        elif un2 == 0:
            f1a, eta = tidal_heat(a, nm, eccm, parameters)
            f2a = convection_heat(a, eta, parameters)

            f1b, eta = tidal_heat(b, nm, eccm, parameters)
            f2b = convection_heat(b, eta, parameters)

            if f2a == 0 or f2b == 0:
                error = True
                T_equilibrium = -1

            stab = check((f1b - f2b), (f1a - f2a))

            if stab:
                T_equilibrium = (a + b) / 2.  # stable point (T_stab) is found
                return T_equilibrium
            elif not stab:  # unstable point is found
                if un1 == 0:
                    a = b
                    b = peak
                    un2 = un2 + 1
                    again = True  # try again above the unstable point
                else:
                    T_equilibrium = -2
                    error = True
            else:
                T_equilibrium = -3
                error = True

        # T_equilibrium does not exist (no intersection)
        if un1 == 0 and un2 == 0 and T_equilibrium == 0:
            return T_equilibrium

        if error:
            if T_equilibrium == -2:
                print("___error2___: two unstable equilibrium points??")
            if T_equilibrium == -3:
                print("___error3___: T_equilibrium not found??")
            return T_equilibrium

        if again:
            T_equilibrium, a, b, again, unstable = intersection(a, b, nm, eccm, parameters)

            if T_equilibrium == -3:
                return T_equilibrium

            un3 = unstable

            if T_equilibrium != 0:
                return T_equilibrium
            # T_unstab is found, but T_stab is not found (or does not exist?)
            elif b == peak:
                T_equilibrium = -6
                error = True
            elif un3 == 0:
                f1a, eta = tidal_heat(a, nm, eccm, parameters)
                f2a = convection_heat(a, eta, parameters)

                f1b, eta = tidal_heat(b, nm, eccm, parameters)
                f2b = convection_heat(b, eta, parameters)

                if f2a == 0 or f2b == 0:
                    error = True
                    T_equilibrium = -1

                stab = check((f1b - f2b), (f1a - f2a))

                if stab:
                    T_equilibrium = (a + b) / 2.    # stable point (T_stab) is found
                    return T_equilibrium
                elif not stab:    # unstable point is found
                    T_equilibrium = -2
                    error = True    # two unstable points
                else:
                    T_equilibrium = -3
                    error = True

            if error:
                if T_equilibrium == -2:
                    print("___error2___: two unstable equilibrium points??")
                if T_equilibrium == -3:
                    print("___error3___: T_equilibrium not found??")
                if T_equilibrium == -6:
                    print("___error6___: T_unstab is found, but T_stab is not found")
                return T_equilibrium

    print(T_equilibrium)
    return T_equilibrium


def plot_moon_temperature_map(
    file,
    moon_eccentricity=None,
    min_temp=0.0,
    max_temp=730
):

    # Definition of all letter sizes
    font = {'weight': 'normal', 'size': 15}
    plt.rc('font', **font)  # A fent definialt betumeret hasznalata

    data = np.loadtxt(file)

    # Extract the periods, radii, and temperatures
    periods_vector = data[:, 0]
    radii_vector = data[:, 1]
    T_eq = data[:, 2]

    # Determine the unique values in the x and y columns
    unique_periods = np.unique(periods_vector)
    unique_radii = np.unique(radii_vector)

    # Create the meshgrid with the correct order of x and y
    X, Y = np.meshgrid(unique_radii, unique_periods)
    # Reshape the z-values to match the dimensions of the meshgrid
    temperatures = T_eq.reshape(len(unique_periods), len(unique_radii))

    fig, ax = plt.subplots(1, 1, figsize=(7.0, 5.0))

    vmin = min_temp
    vmax = max_temp
    levels = np.linspace(vmin, vmax, 5000)

    ax.set_xlabel('Moon Orbital Period (d)')
    ax.set_ylabel('Moon Radius (km)')
    ax.set_title(r'$\rho$ = $\rho_\mathrm{Earth}$'fr', $e$ = {moon_eccentricity}', fontsize=17)
    ax.axis([np.min(unique_periods), np.max(unique_periods),
             np.min(unique_radii), np.max(unique_radii)])

    wbgr = make_rgb_colormap()

    im = ax.contourf(Y, X, temperatures, levels=levels, cmap=wbgr)

    # Add a colorbar for the image
    cbar = fig.colorbar(im, ax=ax, format="%.0f")
    cbar.set_label('Surface Temperature (K)')
    cbar.set_ticks(np.linspace(vmin, vmax, 10))
    cbar.minorticks_on()

    levels = (273.0, 373.0)
    ct = ax.contour(Y, X, temperatures, levels, origin='lower', linewidths=1, colors=('w', 'w'))
    ax.clabel(ct, colors='w', inline=True, fmt='%1.f', fontsize=15, inline_spacing=12)

    # ax.plot(2.06, 6370., 'wo')
    # ax.text(1.9, 6000., r'Exo-Earth', fontsize=18, color='white')

    fig.tight_layout()

    image_name = os.path.join(os.path.dirname(file), f'temperature_map_e{moon_eccentricity}.png')
    fig.savefig(image_name, facecolor='w', dpi=300)


def f_s(e_sp, i, omega, phi_sp, phi_pm, lon, lat):
    """
    Defines the irradiation from the star [W/m**2] as a function of longitude d and latitude l [deg]
    """
    a_pm = A_pm[0] * R_p
    t = phi_sp * P_sp
    i = i * pi / 180.
    omega = omega * pi / 180.
    lon = lon * pi / 180.
    lat = lat * pi / 180.

    # [s],   orbital period of the planet-moon system
    P_pm = 2. * pi * np.sqrt(a_pm**3 / (G * (M_p + M_m)))
    M_sp = 2. * pi / P_sp * (t - tau)                      # [rad], mean anomaly

    # iterate eccentric anomaly
    delta = 10.**10
    accur = 10.**(-5.)
    E_sp = M_sp
    while np.abs(delta) > accur:
        E_tmp = M_sp + e_sp * np.sin(E_sp)
        delta = np.abs(E_tmp - E_sp)
        E_sp = E_tmp

    # Surface normal with length a_pm of the point at (lon,lat) on the moon; by neglecting the mean
    # anomaly M_sp in the cos/np.sin(phi_pm...) terms the moon always starts at the left around
    # the planet
    n_x = a_pm * (-np.sin(lat) * np.sin(i) * np.cos(omega)
                  + np.cos(lat) * (np.cos(omega) * np.cos(2 * pi * phi_pm + lon) * np.cos(i)
                                   - np.sin(omega) * np.sin(2 * pi * phi_pm + lon)))
    n_y = a_pm * (-np.sin(lat) * np.sin(i) * np.sin(omega)
                  + np.cos(lat) * (np.sin(omega) * np.cos(2 * pi * phi_pm + lon) * np.cos(i)
                                   + np.cos(omega) * np.sin(2 * pi * phi_pm + lon)))
    n_z = a_pm * (np.sin(lat) * np.cos(i) + np.cos(lat) * np.cos(2 * pi * phi_pm + lon) * np.sin(i))

    # position of the sub-planetary point on the moon (lon=0=lat)
    s_x = a_pm * (np.cos(omega) * np.cos(2. * pi * phi_pm)
                  * np.cos(i) - np.sin(omega) * np.sin(2. * pi * phi_pm))
    s_y = a_pm * (np.sin(omega) * np.cos(2. * pi * phi_pm)
                  * np.cos(i) + np.cos(omega) * np.sin(2. * pi * phi_pm))
    s_z = a_pm * np.cos(2. * pi * phi_pm) * np.sin(i)

    # vector from the planet to the star
    r_ps_x = -a_sp * (np.cos(E_sp) - e_sp)
    r_ps_y = -a_sp * np.sqrt(1. - e_sp**2) * np.sin(E_sp)
    r_ps_z = 0.
    r_ps = np.sqrt(r_ps_x**2 + r_ps_y**2 + r_ps_z**2)

    # vector from the moon to the star
    r_ms_x = (r_ps_x + s_x)
    r_ms_y = (r_ps_y + s_y)
    r_ms_z = (r_ps_z + s_z)
    r_ms = np.sqrt(r_ms_x**2 + r_ms_y**2 + r_ms_z**2)

    # r_perp is perpendicular part of the moon's vector to the star. Eclipses occur (f_s = 0)
    # while r_perp < R_p if s*r_ms > 0.
    cos_Gamma = (r_ms_x * r_ps_x + r_ms_y * r_ps_y + r_ms_z * r_ps_z) / (r_ms * r_ps)
    Gamma = np.arccos(cos_Gamma)
    r_perp = np.sin(Gamma) * r_ms

    flux_s = R_s**2. * sigma_SB * T_effs**4. / \
        (r_ms**2) * (r_ms_x * n_x + r_ms_y * n_y + r_ms_z * n_z) / (r_ms * a_pm)
    # In this case the star shines on the planet's back side.
    flux_s[where(flux_s < 0)] = 0.

    # Eclipse of the moon behind the planet
    # angular radius of the stellar disk as seen from the moon
    beta_s = 2. * arctan(R_s / (r_ps + a_pm))
    # angular radius of the planetary disk as seen from the moon
    beta_p = 2. * arctan(R_p / (a_pm))

    # In this case the planet covers the whole stellar disk during eclipse.
    if beta_p >= beta_s:
        flux_s[[x for x in range(len(flux_s)) if r_perp[x]
                < R_p and r_ms[x] > r_ps]] = 0
    # In this case the planet covers only part of the stellar disk during eclipse.
    else:
        flux_s[[x for x in range(len(flux_s)) if r_perp[x]
                < R_p and r_ms[x] > r_ps]] *= 1. - (beta_p / beta_s)**2.

    return flux_s


def f_t(e_sp, i, omega, phi_sp, phi_pm, lon, lat):
    """
    Defines the thermal irradiation from the planet [W/m**2]
    """

    a_pm = A_pm[0] * R_p
    t = phi_sp * P_sp
    # [s],   orbital period of the planet-moon system
    P_pm = 2. * pi * np.sqrt(a_pm**3 / (G * (M_p + M_m)))
    M_sp = 2. * pi / P_sp * (t - tau)                      # [rad], mean anomaly

    # iterate eccentric anomaly
    delta = 10.**10
    accur = 10.**(-5.)
    E_sp = M_sp
    while np.abs(delta) > accur:
        E_tmp = M_sp + e_sp * np.sin(E_sp)
        delta = np.abs(E_tmp - E_sp)
        E_sp = E_tmp

    # [rad], true anomaly
    nu_sp = np.arccos((np.cos(E_sp) - e_sp) / (1. - e_sp * np.cos(E_sp)))

    # by neglecting the term 2*pi*(t_tau)/P_sp in the cos/np.sin(...lon) terms the moon always
    # starts at the left around the planet
    s_x = a_pm * (np.cos(omega * pi / 180.) * np.cos(2. * pi * (phi_pm))
                  * np.cos(i * pi / 180.) - np.sin(omega * pi / 180.) * np.sin(2. * pi * (phi_pm)))
    s_y = a_pm * (np.sin(omega * pi / 180.) * np.cos(2. * pi * (phi_pm))
                  * np.cos(i * pi / 180.) + np.cos(omega * pi / 180.) * np.sin(2. * pi * (phi_pm)))
    s_z = a_pm * np.cos(2. * pi * (phi_pm)) * np.sin(i * pi / 180.)

    r_sp_x = -a_sp * (np.cos(E_sp) - e_sp)
    r_sp_y = -a_sp * (np.sqrt(1. - e_sp**2) * np.sin(E_sp))
    r_sp_z = 0.
    r_sp = np.sqrt(r_sp_x**2 + r_sp_y**2 + r_sp_z**2)

    # Compute surface temperatures on the bright (T_b) and on the dark (T_d) sides of the planet

    # Surface temperature of the planet if it was in thermal equilibrium
    T_eq = (T_effs**4 * (1 - alpha_p) * R_s**2. / (4. * r_sp**2))**(1. / 4)
    # Array of temperatures to be investigated for the true surface temperature on the bright side
    T_B = np.arange(T_eq, T_eq + dT, 1.)
    # 4th order polynomial in T_B to be investigated for the 1st zero point > T_eq. This will be T_b
    poly = T_B**4 + (T_B - dT)**4 - T_effs**4 * \
        (1 - alpha_p) * R_s**2. / (2. * r_sp**2)

    # Surface temperature on the bright side of the planet
    T_b = T_B[where(poly > 0)[0][0]]
    # Surface temperature on the dark side of the planet
    T_d = T_b - dT

    # otherwise the region is on the moon's antiplanetary hemisphere and receives no thermal flux
    # from the planet
    if np.abs(lon) < 90:
        Phi = 2. * arctan(s_y / (np.sqrt(s_x**2 + s_y**2) + s_x))
        Theta = pi / 2. - np.arccos(s_z / np.sqrt(s_x**2 + s_y**2 + s_z**2))
        l = np.arccos(np.cos(Theta) * np.cos(Phi - nu_sp))
        xi = 1. / 2 * (1. + np.cos(l))
        flux_t = np.cos(lon * pi / 180.) * np.cos(lat * pi / 180.) * R_p**2. * \
            sigma_SB / a_pm**2. * (T_b**4. * xi + T_d**4. * (1 - xi))
    else:
        flux_t = 0. * phi_pm

    return flux_t


def f_r(e_sp, i, omega, phi_sp, phi_pm, lon, lat):
    """
    Defines the stellar reflected light from the planet [W/m**2]
    """
    a_pm = A_pm[0] * R_p
    t = phi_sp * P_sp
    # [s],   orbital period of the planet-moon system
    P_pm = 2. * pi * np.sqrt(a_pm**3 / (G * (M_p + M_m)))
    M_sp = 2. * pi / P_sp * (t - tau)                      # [rad], mean anomaly

    # iterate eccentric anomaly
    delta = 10.**10
    accur = 10.**(-3.)
    E_sp = M_sp
    while np.abs(delta) > accur:
        E_tmp = M_sp + e_sp * np.sin(E_sp)
        delta = np.abs(E_tmp - E_sp)
        E_sp = E_tmp

    # [rad], true anomaly
    nu_sp = np.arccos((np.cos(E_sp) - e_sp) / (1. - e_sp * np.cos(E_sp)))

    # by neglecting the term 2*pi*(t_tau)/P_sp in the cos/np.sin(...lon) terms the moon always
    # starts at the left around the planet
    s_x = a_pm * (np.cos(omega * pi / 180.) * np.cos(2. * pi * (phi_pm))
                  * np.cos(i * pi / 180.) - np.sin(omega * pi / 180.) * np.sin(2. * pi * (phi_pm)))
    s_y = a_pm * (np.sin(omega * pi / 180.) * np.cos(2. * pi * (phi_pm))
                  * np.cos(i * pi / 180.) + np.cos(omega * pi / 180.) * np.sin(2. * pi * (phi_pm)))
    s_z = a_pm * np.cos(2. * pi * (phi_pm)) * np.sin(i * pi / 180.)

    r_sp_x = -a_sp * (np.cos(E_sp) - e_sp)
    r_sp_y = -a_sp * (np.sqrt(1. - e_sp**2) * np.sin(E_sp))
    r_sp_z = 0.
    r_sp = np.sqrt(r_sp_x**2 + r_sp_y**2 + r_sp_z**2)

    if np.abs(lon) < 90:
        # otherwise the region is on the moon's antiplanetary hemisphere and receives no
        # stellar-reflected flux from the planet
        Phi = 2. * arctan(s_y / (np.sqrt(s_x**2 + s_y**2) + s_x))
        Theta = pi / 2. - np.arccos(s_z / np.sqrt(s_x**2 + s_y**2 + s_z**2))
        l = np.arccos(np.cos(Theta) * np.cos(Phi - nu_sp))
        xi = 1. / 2 * (1. + np.cos(l))
        flux_r = np.cos(lon * pi / 180.) * np.cos(lat * pi / 180.) * R_s**2. * sigma_SB * \
            T_effs**4. / r_sp**2. * pi * R_p**2 * alpha_p / a_pm**2. * xi
    else:
        flux_r = 0. * phi_pm

    return flux_r


def f_m(e_sp, i, omega, phi_sp, phi_pm, lon, lat):
    """
    Defines the total flux on the moon.
    """
    flux_m = f_s(e_sp, i, omega, phi_sp, phi_pm, lon, lat) + f_t(e_sp, i, omega,
                                                                 phi_sp, phi_pm, lon, lat) + f_r(e_sp, i, omega, phi_sp, phi_pm, lon, lat)

    return flux_m


def f_bar(e_sp, i, omega, phi_sp, phi_pm, lon, lat):
    """
    Defines the flux at a given LONG/LAT averaged over one satellite orbit around the planet
    """
    flux_bar = f_m(e_sp, i, omega, phi_sp, phi_pm, lon, lat).sum() / len(phi_pm)
    return flux_bar
