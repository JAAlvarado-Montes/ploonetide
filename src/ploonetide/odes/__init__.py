"""
Total ODEs for star-planet and planet-moon scenarios
"""

from .planet_moon import solution_planet_moon
from .star_planet import solution_star_planet

__all__ = [
    "solution_planet_moon",
    "solution_star_planet",
]
