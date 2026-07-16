"""
Kyklos: Orbital Mechanics and Satellite Mission Design
=======================================================

A Python package for spacecraft trajectory propagation, orbital mechanics,
and mission design using high-performance Taylor series integration.

Quick Start
-----------
Create a system and propagate an orbit:

>>> from kyklos import earth_j2, OrbitalElements
>>> sys = System('2body', earth(), perturbations=('J2',), compile=compile)
>>> orbit = OE(a=7000, e=0.01, i=0.5, omega=0, w=0, nu=0)
>>> state0 = orbit.to_cartesian()
>>> traj = sys.propagate(state0.elements, [0 5400])

Available Modules
-----------------
Core Classes
    OrbitalElements : Coordinate transformations and orbital element handling
    System : Gravitational environment and equation of motion
    Satellite : Physical properties (mass, drag, inertia)
    Trajectory : Time-series orbital state with continuous output

Default Systems (Factory Functions)
    earth_2body : Point-mass Earth
    earth_j2 : Earth with J2 oblateness
    earth_drag : Earth with atmospheric drag
    earth_moon_cr3bp : Earth-Moon circular restricted 3-body problem
    moon_2body, moon_j2 : Moon systems
    mars_2body, mars_j2 : Mars systems

Celestial Body Parameters
    earth(), moon(), mars() : Predefined body parameters
    EARTH_STD_ATMO : Standard atmosphere model
"""

# Core classes
from .orbital_elements import OrbitalElements, OrbitalElements as OE, OEType
from .system import (
        System, TwoBodySystem, CR3BPSystem,
        BodyParams, AtmoParams, SysType, SeederResult,
    )
from .satellite import Satellite, Satellite as Sat
from .trajectory import (Trajectory, Trajectory as Traj, Node, 
    BoundaryNode, JunctionNode,
    StartBoundaryNode, EndBoundaryNode, ImpulsiveBoundaryNode,
    NullJunctionNode, ImpulsiveJunctionNode, FreeJunctionNode,
)
# Differential Corrector Classes
from .shooter import (
    DifferentialCorrector, ShooterResult,
    TerminalConstraint, TargetState, Periodicity, CallableConstraint,
)

# Classes for CR3BP Toolkit
from .periodic_orbit import PeriodicOrbit
from .registry import available_recipes
from .correction import CorrectorGuess, available_layouts, correct_as

# helper utilities and config control
from .utils import Timer
from .config import config, temp_config

# Commonly-used celestial bodies (factory functions)
from .defaults import (mercury, venus, earth, moon, mars, jupiter, saturn, uranus,
    neptune, sun)

# Standard atmosphere model
from .defaults import EARTH_STD_ATMO

# Some commonly-used Earth orbits
from .defaults import ISS_ORBIT, GEO_ORBIT, LEO_ORBIT, SSO_ORBIT, MOLNIYA_ORBIT

# Default systems and CR3BP orbits (factory functions)
from .defaults import (
    earth_2body,
    earth_j2,
    earth_drag,
    earth_moon_cr3bp,
    earth_sun_cr3bp,
    moon_2body,
    moon_j2,
    mars_2body,
    mars_j2,
    lyapunov_orbit,
    gateway_orbit,
)

# Package metadata
__version__ = "0.2.0"
__author__ = "Shane Billingsley"

# Define what gets imported with "from kyklos import *"
__all__ = [
    # Main Classes
    "OrbitalElements",
    "System",
    "Satellite",
    "Trajectory",
    # Helper Classes
    "TwoBodySystem",
    "CR3BPSystem",
    "BodyParams",
    "AtmoParams",
    "SeederResult",
    "PeriodicOrbit",
    "CorrectorGuess",
    "OEType",
    "SysType",
    "Timer",
    # Node Classes
    "Node", 
    "BoundaryNode", 
    "JunctionNode",
    "StartBoundaryNode", 
    "EndBoundaryNode", 
    "ImpulsiveBoundaryNode",
    "NullJunctionNode", 
    "ImpulsiveJunctionNode", 
    "FreeJunctionNode",
    # Shooter Classes
    "DifferentialCorrector",
    "ShooterResult",
    "TerminalConstraint",
    "TargetState",
    "Periodicity",
    "CallableConstraint",
    # Module-level Functions
    "available_recipes",
    "available_layouts",
    "correct_as",
    # Abbreviations
    "OE",
    "Sat",
    "Traj",
    # Configuration
    "config",
    "temp_config",
    # Default Orbits
    "ISS_ORBIT",
    "GEO_ORBIT",
    "LEO_ORBIT",
    "SSO_ORBIT",
    "MOLNIYA_ORBIT",
    "lyapunov_orbit",
    "gateway_orbit",
    # Default systems
    "earth_2body",
    "earth_j2",
    "earth_drag",
    "earth_moon_cr3bp",
    "earth_sun_cr3bp",
    "moon_2body",
    "moon_j2",
    "mars_2body",
    "mars_j2",
    # Predefined default bodies
    "mercury",
    "venus",
    "earth",
    "moon",
    "mars",
    "jupiter",
    "saturn",
    "uranus",
    "neptune",
    "sun",
    # Predefined atmosphere model
    "EARTH_STD_ATMO",
]