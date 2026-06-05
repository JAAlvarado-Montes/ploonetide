"""This module provides various helper functions."""
import logging
import functools
import numpy as np

import matplotlib.collections as mcoll
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors


log = logging.getLogger(__name__)

__all__ = ['dict2obj', 'logged', 'set_xaxis_limits', 'make_rgb_colormap', 'colorline']


class dict2obj(object):

    """Converts dictionary to object
    """

    def __init__(self, dic={}):
        self.__dict__.update(dic)

    def __add__(self, other):
        for attr in other.__dict__.keys():
            exec(f'self.{attr}=other.{attr}')
        return self


def make_rgb_colormap():
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    c = mcolors.ColorConverter().to_rgb
    seq = [c('lavender'), c('lightblue'), 0.08, c('lightblue'), c('blue'), 0.2, c('blue'), c('green'),
           0.4, c('green'), c('orange'), 0.6, c('orange'), c('red'), 1.0, c('red')]
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)


def colorline(x, y, z=None, cmap='copper', linewidth=2, alpha=1.0):
    """
    Plot a colored line with coordinates x and y. Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """
    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    # to check for numerical input -- this is a hack
    if not hasattr(z, '__iter__'):
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, linewidth=linewidth, alpha=alpha,
                              norm=plt.Normalize(np.nanmin(z), np.nanmax(z)))

    ax = plt.gca()
    ax.add_collection(lc)

    return lc


def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments


def set_xaxis_limits(ax, ax1):
    lim1 = ax.get_xlim()
    lim2 = ax1.get_xlim()

    return lim2[0] + (ax.get_xticks() - lim1[0]) / (lim1[1] - lim1[0]) * (lim2[1] - lim2[0])


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def logged(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(func.__name__ + ' was called')
        return func(*args, **kwargs)
    return wrapper


#############################################################
# CANONICAL UNITS TRANSFORMATION
#############################################################
def canonic_units(*, uM=None, uL=None, uT=None, G=6.6740831e-11):
    """Convert to chosen canonical units.

    Provide exactly two among (uM, uL, uT), the third is computed.

    Args:
        uM (float, optional): Mass unit [kg]
        uL (float, optional): Length unit [m]
        uT (float, optional): Time unit [s]
        G (float, optional): Gravitational constant (default SI value)

    Returns:
        tuple: (uM, uL, uT)
    """
    # Count how many arguments are provided
    provided = [x is not None for x in (uM, uL, uT)].count(True)
    if provided != 2:
        raise ValueError("You must provide exactly two among (uM, uL, uT).")

    if uM is not None and uL is not None:
        uT = (uL**3 / (G * uM))**0.5

    elif uM is not None and uT is not None:
        uL = (G * uM * uT**2)**(1.0 / 3.0)

    elif uL is not None and uT is not None:
        uM = uL**3 / (uT**2 * G)

    # Sanity checks
    for name, val in zip(("uM", "uL", "uT"), (uM, uL, uT)):
        if not (isinstance(val, (int, float)) and val > 0):
            raise ValueError(f"{name} must be a positive number, got {val!r}")

    return uM, uL, uT


# THIS IS THE FIXED-POINT FUNCTION FOR DOING THE "k" ITERATIONS TO REFINE
# THE ROOT'S VALUE. The iteration map must be supplied explicitly; older
# versions implicitly referenced a missing global named ``g``.
def fpi(t, k, step, n):
    if not callable(step):
        raise TypeError("step must be a callable with signature step(t, n).")

    for _ in np.arange(k):
        t = step(t, n)
    return t


# ############################################################
# Util Functions
# ############################################################
def fmt(x, pos):
    """Write scientific notation formater for plots.

    Args:
        x (int): Description
        pos (TYPE): Description

    Returns:
        string: Scientific notation of input string
    """
    a, b = f'{x:.1e}'.split('e')
    b = int(b)
    return rf'${a} \times 10^{{{b}}}$'
