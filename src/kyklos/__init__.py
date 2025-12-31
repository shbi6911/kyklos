"""
Kyklos: Orbital Mechanics and Satellite Mission Design

A Python package for spacecraft trajectory propagation, orbital mechanics,
and mission design using high-performance Taylor series integration.
"""

# Core classes
from .orbital_elements import OrbitalElements, OrbitalElements as OE
from .system import System, BodyParams, AtmoParams
from .satellite import Satellite, Satellite as Sat
from .trajectory import Trajectory, Trajectory as Traj

# Commonly-used celestial bodies
from .system import EARTH, MOON, MARS

# Standard atmosphere model
from .system import EARTH_STD_ATMO

# Package metadata
__version__ = "0.1.0"
__author__ = "Shane Billingsley"

# Define what gets imported with "from kyklos import *"
__all__ = [
    # Classes
    "OrbitalElements",
    "System",
    "BodyParams",
    "AtmoParams", 
    "Satellite",
    "Trajectory",
    # Abbreviations
    "OE",
    "Sat",
    "Traj",
    # Constants
    "EARTH",
    "MOON",
    "MARS",
    "EARTH_STD_ATMO",
]