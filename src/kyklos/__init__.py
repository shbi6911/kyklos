"""
Kyklos: trajectory propagation and mission design in Python.

Kyklos is a package for spacecraft trajectory propagation, orbital mechanics,
and mission design, built on the Heyoka Taylor-series integrator. It supports
two-body dynamics (with J2, J3, and drag perturbations) and the circular
restricted three-body problem (CR3BP), with multiple-shooting differential 
correction, and pseudo-arclength continuation. The CR3BP toolkit for periodic orbits 
and their families is the most developed part of the package to date.

The package is organized around a small set of core objects:

- ``System`` : the dynamical model (equations of motion, body parameters).
  Built through the ``System`` factory or the ready-made ``earth_2body()``,
  ``earth_j2()``, ``earth_moon_cr3bp()`` and related functions.
- ``OrbitalElements`` : a state in Keplerian, Cartesian, equinoctial, or
  CR3BP form, with conversions between them.
- ``Trajectory`` : a time-contiguous segmented integration with Heyoka dense
  output objects separated by nodes, produced by ``System.propagate()``.
- ``Satellite`` : physical properties (mass, drag area, inertia) used by
  satellite-dependent perturbations.
- ``DifferentialCorrector`` : a multiple-shooting targeter driven by
  composable constraints.
- ``PeriodicOrbit`` and ``OrbitFamily`` : verified CR3BP periodic orbits and
  the families produced by continuation (``correct_as``, ``march_family``).

Examples
--------
Propagate a low Earth orbit under J2 for one period and plot (units are km, s, rad)::

    import kyklos as ky

    system = ky.earth_j2()
    orbit = ky.OrbitalElements(a=7000.0, e=0.01, i=0.5,
                               omega=0.0, w=0.0, nu=0.0,
                               system=system)
    traj = system.propagate(orbit, [0.0, orbit.orbital_period()])
    fig = traj.plot_3d()
    fig.show()

Start from the built-in Earth-Moon L1 Lyapunov orbit and march its family::

    lyap = ky.lyapunov_orbit()
    print(lyap.period, lyap.stability_index)
    family = ky.march_family(lyap, 'lyapunov', ds=0.01, n_steps=20)
    fig = family.plot_3d()
    fig.show()

See the API reference for the full list of classes and functions.
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
    Constraint, TerminalConstraint, FreeVarConstraint,
    TargetState, Periodicity, CallableConstraint, JacobiConstraint,
    PseudoArclength, FreeVarPin, PhaseConstraint, NodeSpec,
)

# Classes and functions for CR3BP Toolkit
from .periodic_orbit import PeriodicOrbit
from .orbit_family import OrbitFamily
from .registry import available_recipes
from .continuation import (CorrectorGuess, available_layouts, correct_as, 
                           march_family, available_schemes)

# custom Kyklos exceptions (more planned)
from .exceptions import ConvergenceError

# helper utilities and config control
from .utils import Timer
from .config import config, temp_config

# Specified orbit-type constructors
from .orbit_design import (circular_orbit, synchronous_orbit, molniya_orbit, 
                           sun_synchronous_orbit)

# Commonly-used celestial bodies (factory functions)
from .defaults import (mercury, venus, earth, moon, mars, jupiter, saturn, uranus,
    neptune, sun)

# Standard atmosphere model
from .defaults import EARTH_STD_ATMO

# Default systems (factory functions)
from .defaults import (earth_2body, earth_j2, earth_drag, earth_moon_cr3bp,
    earth_sun_cr3bp, moon_2body, moon_j2, mars_2body, mars_j2, 
)

# Some commonly-used Earth 2BP orbits and Earth-Moon CR3BP orbits
from .defaults import (iss_orbit, geo_orbit, leo_orbit, sso_orbit, 
                       default_molniya_orbit, lyapunov_orbit, gateway_orbit
)

# Package metadata
__version__ = "0.2.1"
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
    "OrbitFamily",
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
    "Constraint",
    "TerminalConstraint",
    "FreeVarConstraint",
    "TargetState",
    "Periodicity",
    "JacobiConstraint",
    "CallableConstraint",
    "PseudoArclength",
    "FreeVarPin",
    "PhaseConstraint",
    "NodeSpec",
    # Module-level Functions
    "available_recipes",
    "available_layouts",
    "available_schemes",
    "correct_as",
    "march_family",
    # Abbreviations
    "OE",
    "Sat",
    "Traj",
    # Configuration
    "config",
    "temp_config",
    # Exceptions
    "ConvergenceError",
    # Default Orbits and Orbit Constructors
    "iss_orbit",
    "geo_orbit",
    "leo_orbit",
    "sso_orbit",
    "default_molniya_orbit",
    "lyapunov_orbit",
    "gateway_orbit",
    "circular_orbit",
    "synchronous_orbit",
    "sun_synchronous_orbit",
    "molniya_orbit",
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