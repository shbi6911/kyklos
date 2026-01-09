"""
Kyklos: Orbital Mechanics and Satellite Mission Design
=======================================================

A Python package for spacecraft trajectory propagation, orbital mechanics,
and mission design using high-performance Taylor series integration.

Quick Start
-----------
Create a system and propagate an orbit:

>>> from kyklos import earth_j2, OrbitalElements
>>> sys = System('2body', EARTH, perturbations=('J2',), compile=compile)
>>> orbit = OE(a=7000, e=0.01, i=0.5, omega=0, w=0, nu=0)
>>> state0 = orbit.to_cartesian()
>>> traj = sys.propagate(state0.elements, 0, 5400)

Available Modules
-----------------
Core Classes
    OrbitalElements : Coordinate transformations and orbital element handling
    System : Gravitational environment and equation of motion
    Satellite : Physical properties (mass, drag, inertia)
    Trajectory : Time-series orbital state with continuous interpolation

Default Systems (Factory Functions)
    earth_2body : Point-mass Earth
    earth_j2 : Earth with J2 oblateness
    earth_drag : Earth with J2 and atmospheric drag
    earth_moon_cr3bp : Earth-Moon circular restricted 3-body problem
    moon_2body, moon_j2 : Moon systems
    mars_2body, mars_j2 : Mars systems

Celestial Body Parameters
    EARTH, MOON, MARS : Predefined body parameters
    EARTH_STD_ATMO : Standard atmosphere model
"""

# Core classes
from .orbital_elements import OrbitalElements, OrbitalElements as OE, OEType
from .system import System, BodyParams, AtmoParams, SysType
from .satellite import Satellite, Satellite as Sat
from .trajectory import Trajectory, Trajectory as Traj
from .utils import Timer

# Commonly-used celestial bodies
from .system import EARTH, MOON, MARS

# Standard atmosphere model
from .system import EARTH_STD_ATMO

# Some commonly-used Earth orbits
from .defaults import ISS_ORBIT, GEO_ORBIT, LEO_ORBIT, SSO_ORBIT, MOLNIYA_ORBIT

# Default systems (factory functions)
from .defaults import (
    earth_2body,
    earth_j2,
    earth_drag,
    earth_moon_cr3bp,
    moon_2body,
    moon_j2,
    mars_2body,
    mars_j2,
)

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
    "OEType",
    "SysType",
    "Timer",
    # Abbreviations
    "OE",
    "Sat",
    "Traj",
    # Default Orbits
    "ISS_ORBIT",
    "GEO_ORBIT",
    "LEO_ORBIT",
    "SSO_ORBIT",
    "MOLNIYA_ORBIT",
    # Default systems
    "earth_2body",
    "earth_j2",
    "earth_drag",
    "earth_moon_cr3bp",
    "moon_2body",
    "moon_j2",
    "mars_2body",
    "mars_j2",
    # Constants
    "EARTH",
    "MOON",
    "MARS",
    "EARTH_STD_ATMO",
]