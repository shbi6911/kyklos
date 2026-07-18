"""
Default Orbits and System Configurations
==============================

Default values for Solar System BodyParams, as well as Standard Atmosphere models
and some predefined Earth orbits, built using specific orbit design constructors.

BodyParams factory functions accept a 'source' parameter, which defaults to 'vallado'.
This pulls values from a local dict drawn from Vallado, Fundamentals of Astrodynamics
& Applications, Appendix D.

Factory functions for commonly-used orbital systems. These functions
create System objects on demand, avoiding memory overhead until needed.

All System functions accept a ``compile`` parameter (default True) to control
whether the Heyoka integrator is compiled immediately or deferred.

All System factory functions are cached: repeated calls with the same arguments return
the same System instance rather than constructing a new one. This keeps
Kyklos from duplicating compiled integrators, which are expensive to
build. Because System objects are immutable, sharing a single instance
across callers is safe.  Note that distinct ``compile`` values are cached separately, 
so a compiled and a deferred System can coexist.

BodyParams objects and calculated orbits are not cached, as they are cheap to construct.

Examples
--------
>>> from kyklos import earth_2body, earth_moon_cr3bp
>>> sys = earth_2body()  # Standard 2-body Earth
>>> sys_lazy = earth_j2(compile=False)  # Defer compilation
"""
import numpy as np
import functools
from typing import Literal, Final, cast, assert_never, get_args
from .orbital_elements import OrbitalElements
from .system import (
        System, TwoBodySystem, CR3BPSystem,
        BodyParams, AtmoParams,
    )
from .periodic_orbit import PeriodicOrbit
from .orbit_design import (circular_orbit, synchronous_orbit, molniya_orbit,
                           sun_synchronous_orbit)

"""
Predefined Solar System bodies
Values taken from Vallado, Fundamentals of Astrdynamics, Fifth Edition, 2022, Appendix D
Units referenced to km (i.e. mu = km^3/s^2)
"""

# Canonical list of available body data sources, not including 'user'
BodySource = Literal['vallado', 'spice']

# Dictionary of static body data drawn from Vallado, Fundamentals of Astrodynamics
# & Applications, 5th ed, Appendix D (data source for factory defaults)
_VALLADO_DATA: Final[dict[str, BodyParams]] = {'mercury' : BodyParams(
    mu=2.2032e4,
    radius=2439.0,
    J2=6.0e-5,
    rotation_rate=1.24001e-6,
    name='Mercury',
    source='vallado'
),

'venus' : BodyParams(
    mu=3.257e5,
    radius=6052.0,
    J2=2.7e-5,
    rotation_rate=-2.9926e-7,
    name='Venus',
    source='vallado'
),

'earth' : BodyParams(
    mu=3.986004415e5,
    radius=6378.1363,
    J2=1.0826269e-3,
    rotation_rate=7.2921150e-5,
    name='Earth',
    source='vallado'
),

'moon' : BodyParams(
    mu=4.902799e3,
    radius=1738.0,
    J2=2.027e-4,
    rotation_rate=2.661700e-6,
    name='Moon',
    source='vallado'
),

'mars' : BodyParams(
    mu=4.305e4,
    radius=3397.2,
    J2=1.964e-3,
    rotation_rate=7.0882181e-5,
    name='Mars',
    source='vallado'
),

'jupiter' : BodyParams(
    mu=1.268e8,
    radius=71492.0,
    J2=1.475e-2,
    rotation_rate=1.7585e-4,
    name='Jupiter',
    source='vallado'
),

'saturn' : BodyParams(
    mu=3.794e7,
    radius=60268.0,
    J2=1.645e-2,
    rotation_rate=1.662e-4,
    name='Saturn',
    source='vallado'
),

'uranus' : BodyParams(
    mu=5.794e6,
    radius=25559.0,
    J2=1.2e-2,
    rotation_rate=-1.12e-4,
    name='Uranus',
    source='vallado'
),

'neptune' : BodyParams(
    mu=6.809e6,
    radius=24764.0,
    J2=4.0e-3,
    rotation_rate=9.47e-5,
    name='Neptune',
    source='vallado'
),

'sun' : BodyParams(
    mu=1.32712428e11,
    radius=6.96e5,
    J2=None,
    rotation_rate=None,
    name='Sun',
    source='vallado'
)
}

def body(name: str, source: BodySource='vallado', ) -> BodyParams:
    """
    Look up parameters for a Solar System body from the requested source.

    The general lookup behind the named per-body factories (earth(), moon(),
    ...). Those are thin wrappers over this; call them for discoverability, or
    call body() directly when the name is dynamic.

    Parameters
    ----------
    name : str
        Body name, case- and whitespace-insensitive (e.g. 'Earth', ' earth ').
    source : {'vallado', 'spice'}, optional
        Data source. 'vallado' (default) returns values from Vallado,
        Fundamentals of Astrodynamics, 5th ed., Appendix D. 'spice' is not
        yet implemented.

    Returns
    -------
    BodyParams
        Immutable parameters for the requested body, tagged with its source.

    Raises
    ------
    ValueError
        If `source` is not a recognized source, or if `name` is not available
        from that source.
    NotImplementedError
        If `source` is 'spice' (pending SPICE integration).
    """

    if source not in get_args(BodySource):
        raise ValueError(
            f"Unknown body data source {source!r}. "
            f"Valid sources: {', '.join(get_args(BodySource))}."
        )
    
    key = name.strip().lower()
    
    if source == 'vallado':
        try:
            return _VALLADO_DATA[key]
        except KeyError:
            raise ValueError(
                f"No Vallado data for body {name!r}. "
                f"Known bodies: {', '.join(sorted(_VALLADO_DATA))}"
            ) from None
    elif source == 'spice':
        raise NotImplementedError(
            "SPICE-sourced body parameters are not yet implemented."
        )
    else:
        assert_never(source)

# ==================================================
# Default wrappers for common Solar System bodies
# ==================================================

def mercury(source: BodySource = 'vallado') -> BodyParams:
    """Mercury parameters. See module docstring for source semantics."""
    return body('mercury', source)

def venus(source: BodySource = 'vallado') -> BodyParams:
    """Venus parameters. See module docstring for source semantics."""
    return body('venus', source)

def earth(source: BodySource = 'vallado') -> BodyParams:
    """Earth parameters. See module docstring for source semantics."""
    return body('earth', source)

def moon(source: BodySource = 'vallado') -> BodyParams:
    """Lunar parameters. See module docstring for source semantics."""
    return body('moon', source)

def mars(source: BodySource = 'vallado') -> BodyParams:
    """Mars parameters. See module docstring for source semantics."""
    return body('mars', source)

def jupiter(source: BodySource = 'vallado') -> BodyParams:
    """Jupiter parameters. See module docstring for source semantics."""
    return body('jupiter', source)

def saturn(source: BodySource = 'vallado') -> BodyParams:
    """Saturn parameters. See module docstring for source semantics."""
    return body('saturn', source)

def uranus(source: BodySource = 'vallado') -> BodyParams:
    """Uranus parameters. See module docstring for source semantics."""
    return body('uranus', source)

def neptune(source: BodySource = 'vallado') -> BodyParams:
    """Neptune parameters. See module docstring for source semantics."""
    return body('neptune', source)

def sun(source: BodySource = 'vallado') -> BodyParams:
    """Solar parameters. See module docstring for source semantics."""
    return body('sun', source)

# ==================================================
# ========== Predefined atmospheric models
# ==================================================
EARTH_STD_ATMO = AtmoParams(
    rho0=1.225,
    H=8500.0,
    r0=6378136.3
)

# ==================================================
# ========== Predefined Earth orbits ==========
# ==================================================

def iss_orbit(): 
    '''Average orbit of the International Space Station, a near-circular inclined
    Earth orbit at about 420 km altitude.'''
    return OrbitalElements(
        a=earth().radius + 420, e=0.0007, i=np.radians(51.64),
        omega=0.0, w=0.0, nu=0.0, mu=earth().mu
    )

def geo_orbit():
    '''A geostationary Earth orbit'''
    return synchronous_orbit(body=earth())

def leo_orbit():
    '''A circular, equatorial Low Earth Orbit'''
    return circular_orbit(body=earth(), altitude=300)

def sso_orbit():
    '''A 500 km altitude, near-circular sun-synchronous Earth orbit. The target
    node rate is Earth's mean heliocentric rate, 2*pi per tropical year
    (2*pi / (365.2422 * 86400) ~ 1.991e-7 rad/s) -- the rate of the mean Sun,
    not Earth's inertial orbital rate (see sun_synchronous_orbit).'''
    return sun_synchronous_orbit(body=earth(), a=earth().radius + 500,
            e=0.001, node_rate=(2*np.pi / (365.2422 * 86400))
    )

def default_molniya_orbit(): 
    ''' A traditional Molniya orbit with a half-sidereal day period, at a 
    critical inclination to fix argument of periapsis at 270 degrees, over the
    northern hemisphere, with a 600 km perigee altitude and 0 RAAN.'''
    return molniya_orbit(body=earth())

# ==================================================
# ========== Predefined Systems ==========
# ==================================================

@functools.cache
def earth_2body(compile=True) -> TwoBodySystem:
    """
    Create a point-mass 2-body Earth system.
    
    This is the simplest orbital propagation model: spherically symmetric
    Earth gravity with no perturbations.

    Cached: see module docstring. Repeated calls return the same instance.
    
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
    return cast(TwoBodySystem, System('2body', earth(), compile=compile))

@functools.cache
def earth_j2(compile=True) -> TwoBodySystem:
    """
    Create a 2-body Earth system with J2 oblateness.
    
    Includes the dominant zonal harmonic (J2) which captures Earth's
    equatorial bulge.

    Cached: see module docstring. Repeated calls return the same instance.
    
    Parameters
    ----------
    compile : bool, optional
        If True (default), compile integrator immediately.
    
    Returns
    -------
    System
        2-body Earth system with J2 perturbation
    """
    return cast(TwoBodySystem, System(
        '2body', earth(),
        perturbations=('J2',),
        compile=compile
    ))

@functools.cache
def earth_drag(compile=True) -> TwoBodySystem:
    """
    Create 2-body Earth with atmospheric drag.
    
    Includes exponential atmosphere model for realistic LEO propagation 
    including orbital decay.

    Cached: see module docstring. Repeated calls return the same instance.
    
    Parameters
    ----------
    compile : bool, optional
        If True (default), compile integrator immediately.
    
    Returns
    -------
    System
        2-body Earth system with drag perturbation
    
    Notes
    -----
    Drag propagation requires satellite parameters (mass, Cd*A) to be
    provided to the propagate() method. The standard atmosphere model
    uses rho0 = 1.225 kg/m^3 at sea level with scale height H = 8.5 km.
    """
    return cast(TwoBodySystem, System(
        '2body', earth(),
        perturbations=('drag',),
        atmosphere=EARTH_STD_ATMO,
        compile=compile
    ))

@functools.cache
def earth_moon_cr3bp(compile=True) -> CR3BPSystem:
    """
    Create Earth-Moon circular restricted 3-body problem system.
    
    Models spacecraft motion in the rotating Earth-Moon frame.

    Cached: see module docstring. Repeated calls return the same instance.
    
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
    - Mass ratio mu = 0.01215 (Moon mass / total system mass)
    
    State vectors should be nondimensional. Origin is at the system
    barycenter with primaries at x = (-mu, 1 - mu).
    """
    return cast(CR3BPSystem, System(
        '3body', earth(), moon(),
        distance=384400.0,
        compile=compile
    ))

@functools.cache
def earth_sun_cr3bp(compile=True) -> CR3BPSystem:
    """
    Create Sun-Earth circular restricted 3-body problem system.
    
    Models spacecraft motion in the rotating Sun-Earth frame.

    Cached: see module docstring. Repeated calls return the same instance.
    
    Parameters
    ----------
    compile : bool, optional
        If True (default), compile integrator immediately.
    
    Returns
    -------
    System
        Sun-Earth CR3BP system in rotating frame
    
    Notes
    -----
    The system uses nondimensional units where:
    - Characteristic length L* = 149,597,870.7 km (1 AU distance)
    - Characteristic time T* = 5,022,635.6 s
    - Mass ratio mu = 3.00348e-6 (Earth mass / total system mass)
    
    State vectors should be nondimensional. Origin is at the system
    barycenter with primaries at x = (-mu, 1 - mu).
    """
    return cast(CR3BPSystem, System(
        '3body', sun(), earth(),
        distance=149597870.7,
        compile=compile
    ))

@functools.cache
def moon_2body(compile=True) -> TwoBodySystem:
    """
    Create a point-mass 2-body Moon system.
    
    For lunar orbit propagation. Uses Moon's gravitational parameter
    and radius.

    Cached: see module docstring. Repeated calls return the same instance.
    
    Parameters
    ----------
    compile : bool, optional
        If True (default), compile integrator immediately.
    
    Returns
    -------
    System
        Configured 2-body Moon system
    """
    return cast(TwoBodySystem, System('2body', moon(), compile=compile))

@functools.cache
def moon_j2(compile=True) -> TwoBodySystem:
    """
    Create a 2-body Moon system with J2 oblateness.
    
    Includes J2 zonal harmonic for Moon's equatorial bulge.

    Cached: see module docstring. Repeated calls return the same instance.
    
    Parameters
    ----------
    compile : bool, optional
        If True (default), compile integrator immediately.
    
    Returns
    -------
    System
        2-body Moon system with J2 perturbation
    """
    return cast(TwoBodySystem, System(
        '2body', moon(), 
        perturbations=('J2',), 
        compile=compile
    ))

@functools.cache
def mars_2body(compile=True) -> TwoBodySystem:
    """
    Create a point-mass 2-body Mars system.
    
    For Mars orbit propagation. Uses Mars's gravitational parameter
    and radius.

    Cached: see module docstring. Repeated calls return the same instance.
    
    Parameters
    ----------
    compile : bool, optional
        If True (default), compile integrator immediately.
    
    Returns
    -------
    System
        Configured 2-body Mars system
    """
    return cast(TwoBodySystem, System('2body', mars(), compile=compile))

@functools.cache
def mars_j2(compile=True) -> TwoBodySystem :
    """
    Create a 2-body Mars system with J2 oblateness.
    
    Includes J2 zonal harmonic for Mars's equatorial bulge.

    Cached: see module docstring. Repeated calls return the same instance.
    
    Parameters
    ----------
    compile : bool, optional
        If True (default), compile integrator immediately.
    
    Returns
    -------
    System
        2-body Mars system with J2 perturbation
    """
    return cast(TwoBodySystem, System(
        '2body', mars(), 
        perturbations=('J2',), 
        compile=compile
    ))

@functools.cache
def lyapunov_orbit() -> PeriodicOrbit:
    """
    Canonical L1 Lyapunov periodic orbit in the Earth-Moon CR3BP.

    A pre-converged planar Lyapunov orbit about the Earth-Moon L1 point,
    provided as a convenient reference orbit for examples and tests. Unstable,
    but well-behaved over a single period and far less stiff than an NRHO.

    Cached (see module docstring): repeated calls return the same instance.
    The orbit holds a reference to the cached canonical Earth-Moon CR3BP
    System (via earth_moon_cr3bp()), keeping that System alive as well; clear
    both caches if a fresh build is required.

    Returns
    -------
    PeriodicOrbit
        The L1 Lyapunov orbit, with monodromy and stability data available.
    """
    # Pre-converged initial conditions and period (nondimensional).
    initial_state = np.array([0.787904556873149, 0.0, 0.0, 
                                0.0, 0.419844679804609, 0.0])
    period = 3.744163087739812

    # Canonical Earth-Moon CR3BP system (itself cached).
    sys = earth_moon_cr3bp()

    # Propagate the pre-converged orbit WITH the STM so the PeriodicOrbit
    # constructor can read the monodromy directly, without repropagating.
    traj = sys.propagate(initial_state, [0.0, period], with_stm=True)

    # Promote to a verified PeriodicOrbit. Passing period explicitly selects
    # the 'explicit' path and skips geometric inference, which is correct here
    # since the period is known from the pre-converged solution.
    return PeriodicOrbit(traj, period, name='L1 Lyapunov (C=3.0355...)')

@functools.cache
def gateway_orbit() -> PeriodicOrbit:
    """
    Canonical Gateway L2 NRHO in the Earth-Moon CR3BP.

    The 9:2 synodic-resonant near-rectilinear halo orbit (NRHO) about the
    Earth-Moon L2 point -- the baseline orbit for the lunar Gateway station --
    provided as the realistic, stiffer reference orbit for examples and tests.
    As an NRHO it sits near the stability boundary (only weakly unstable), but
    it is numerically stiff through its fast, close perilune passage, which
    makes it the harder reference case relative to the L1 Lyapunov orbit.

    Cached (see module docstring): repeated calls return the same instance.
    The orbit holds a reference to the cached canonical Earth-Moon CR3BP
    System (via earth_moon_cr3bp()), keeping that System alive as well; clear
    both caches if a fresh build is required.

    Returns
    -------
    PeriodicOrbit
        The Gateway L2 NRHO, with monodromy and stability data available.
    """
    # Initial conditions and period (nondimensional), converged by the Kyklos
    # shooter to a constraint tolerance of 1e-14 (5 iterations). Symmetry dust
    # (y, vx, vz ~ 1e-14) zeroed explicitly; the full-period orbit still closes
    # to a residual of 3.5e-12, so the stored ICs make the mirror symmetry exact
    # without meaningfully changing closure.
    initial_state = np.array([1.021993596119699e+00, 0.0, -1.820775162891188e-01,
                                0.0, -1.031955222404240e-01, 0.0])
    period = 1.510743174597360

    # Canonical Earth-Moon CR3BP system (itself cached).
    sys = earth_moon_cr3bp()

    # Propagate the pre-converged orbit WITH the STM so the PeriodicOrbit
    # constructor can read the monodromy directly, without repropagating.
    traj = sys.propagate(initial_state, [0.0, period], with_stm=True)

    # Promote to a verified PeriodicOrbit. Passing period explicitly selects
    # the 'explicit' path and skips geometric inference, which is correct here
    # since the period is known from the pre-converged solution.
    return PeriodicOrbit(traj, period, name='Gateway L2 NRHO')
