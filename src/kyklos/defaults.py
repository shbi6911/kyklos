"""
Default Orbits and System Configurations
==============================

Factory functions for commonly-used orbital systems. These functions
create System objects on demand, avoiding memory overhead until needed.

All functions accept a ``compile`` parameter (default True) to control
whether the Heyoka integrator is compiled immediately or deferred.

Examples
--------
>>> from kyklos import earth_2body, earth_moon_cr3bp
>>> sys = earth_2body()  # Standard 2-body Earth
>>> sys_lazy = earth_j2(compile=False)  # Defer compilation
"""
import numpy as np
from .orbital_elements import OrbitalElements
from .system import System, EARTH, MOON, MARS, EARTH_STD_ATMO

ISS_ORBIT = OrbitalElements(
    a=6778.0, e=0.0001, i=np.radians(51.6),
    omega=0, w=0, nu=0, mu=EARTH.mu
)

GEO_ORBIT = OrbitalElements(
    a=42164.0, e=0.0, i=0.0,
    omega=0, w=0, nu=0, mu=EARTH.mu
)

LEO_ORBIT = OrbitalElements(
    a=EARTH.radius+550, e=0.0, i=0.0,
    omega=0, w=0, nu=0, mu=EARTH.mu
)

SSO_ORBIT = OrbitalElements(
    a=EARTH.radius+500, e=0.001, i=np.radians(97.4016),
    omega=np.radians(140), w=0, nu=0, mu=EARTH.mu
)

MOLNIYA_ORBIT = OrbitalElements(
    a = 26554, e = 0.737, i = np.radians(63.4),
    omega=np.radians(100), w=np.radians(270), nu=0, mu=EARTH.mu
)

def earth_2body(compile=True):
    """
    Create a point-mass 2-body Earth system.
    
    This is the simplest orbital propagation model: spherically symmetric
    Earth gravity with no perturbations.
    
    Parameters
    ----------
    compile : bool, optional
        If True (default), compile the Heyoka integrator immediately.
        Set to False to defer compilation until first propagation.
    
    Returns
    -------
    System
        Configured 2-body Earth system
    """
    return System('2body', EARTH, compile=compile)


def earth_j2(compile=True):
    """
    Create a 2-body Earth system with J2 oblateness.
    
    Includes the dominant zonal harmonic (J2) which captures Earth's
    equatorial bulge.
    
    Parameters
    ----------
    compile : bool, optional
        If True (default), compile integrator immediately.
    
    Returns
    -------
    System
        2-body Earth system with J2 perturbation
    """
    return System('2body', EARTH, perturbations=('J2',), compile=compile)


def earth_drag(compile=True):
    """
    Create 2-body Earth with atmospheric drag.
    
    Includes exponential atmosphere model for realistic LEO propagation 
    including orbital decay.
    
    Parameters
    ----------
    compile : bool, optional
        If True (default), compile integrator immediately.
    
    Returns
    -------
    System
        2-body Earth system with J2 and drag perturbations
    
    Notes
    -----
    Drag propagation requires satellite parameters (mass, Cd*A) to be
    provided to the propagate() method. The standard atmosphere model
    uses ρ₀ = 1.225 kg/m³ at sea level with scale height H = 8.5 km.
    """
    return System(
        '2body', EARTH,
        perturbations=('drag',),
        atmosphere=EARTH_STD_ATMO,
        compile=compile
    )


def earth_moon_cr3bp(compile=True):
    """
    Create Earth-Moon circular restricted 3-body problem system.
    
    Models spacecraft motion in the rotating Earth-Moon frame.
    
    Parameters
    ----------
    compile : bool, optional
        If True (default), compile integrator immediately.
    
    Returns
    -------
    System
        Earth-Moon CR3BP system in rotating frame
    
    Notes
    -----
    The system uses nondimensional units where:
    - Characteristic length L* = 384,400 km (Earth-Moon distance)
    - Characteristic time T* = 375,700 s
    - Mass ratio μ = 0.01215 (Moon mass / total system mass)
    
    State vectors should be nondimensional. Origin is at the system
    barycenter with primaries at x = ±(μ, 1-μ).
    """
    return System(
        '3body', EARTH, MOON,
        distance=384400.0,
        compile=compile
    )


def moon_2body(compile=True):
    """
    Create a point-mass 2-body Moon system.
    
    For lunar orbit propagation. Uses Moon's gravitational parameter
    and radius.
    
    Parameters
    ----------
    compile : bool, optional
        If True (default), compile integrator immediately.
    
    Returns
    -------
    System
        Configured 2-body Moon system
    """
    return System('2body', MOON, compile=compile)


def moon_j2(compile=True):
    """
    Create a 2-body Moon system with J2 oblateness.
    
    Includes J2 zonal harmonic for Moon's equatorial bulge.
    
    Parameters
    ----------
    compile : bool, optional
        If True (default), compile integrator immediately.
    
    Returns
    -------
    System
        2-body Moon system with J2 perturbation
    """
    return System('2body', MOON, perturbations=('J2',), compile=compile)


def mars_2body(compile=True):
    """
    Create a point-mass 2-body Mars system.
    
    For Mars orbit propagation. Uses Mars's gravitational parameter
    and radius.
    
    Parameters
    ----------
    compile : bool, optional
        If True (default), compile integrator immediately.
    
    Returns
    -------
    System
        Configured 2-body Mars system
    """
    return System('2body', MARS, compile=compile)


def mars_j2(compile=True):
    """
    Create a 2-body Mars system with J2 oblateness.
    
    Includes J2 zonal harmonic for Mars's equatorial bulge.
    
    Parameters
    ----------
    compile : bool, optional
        If True (default), compile integrator immediately.
    
    Returns
    -------
    System
        2-body Mars system with J2 perturbation
    """
    return System('2body', MARS, perturbations=('J2',), compile=compile)
