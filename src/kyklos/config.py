"""
Global Configuration for Kyklos Package
========================================

This module provides package-wide configuration settings that users can modify
to control numerical tolerances, validation behavior, and default plotting options.

Examples
--------
View current configuration:

>>> import kyklos
>>> print(kyklos.config)

Modify settings:

>>> kyklos.config.EQUALITY_RTOL = 1e-14  # Stricter equality checks
>>> kyklos.config.DEFAULT_PLOT_POINTS = 2000  # More detailed plots

Reset to defaults:

>>> kyklos.config.reset()

Temporarily modify settings:

>>> with kyklos.temp_config(EQUALITY_RTOL=1e-6):
...     # Relaxed tolerance for this block only
...     orbit1 == orbit2

Notes
-----
These settings affect package-wide behavior. Modifying them will impact
all subsequent operations until changed again or reset.
"""

from dataclasses import dataclass
from contextlib import contextmanager
from typing import Optional
import math


@dataclass
class KyklosConfig:
    """
    Global configuration for Kyklos package.
    
    Attributes
    ----------
    EQUALITY_RTOL : float
        Relative tolerance for floating-point equality comparisons.
        Default: 1e-12 
    EQUALITY_ATOL : float
        Absolute tolerance for floating-point equality comparisons.
        Default: 1e-14
    HASH_DECIMALS : int
        Number of decimal places for rounding when computing hash values.
        Automatically computed to preserve hash contract
    SNAP_TO_ZERO_THRESHOLD : float
        Values below this threshold are treated as exactly zero.
        Useful for numerical stability with very small numbers.
        Default: 1e-10
    SNAP_TO_CIRCULAR : float
        Eccentricity below this threshold treated as circular orbit (e=0).
        Default: 1e-8
    SNAP_TO_EQUATORIAL : float
        Inclination below this threshold treated as equatorial (i=0).
        Default: 1e-8
    STRICT_VALIDATION : bool
        If True, validation failures raise exceptions.
        If False, validation failures issue warnings.
        Default: True
    DEFAULT_COMPILE : bool
        If True, System objects compile integrator immediately on construction.
        If False, compilation is deferred until first propagation.
        Default: True
    INSTANCE_WARNING_THRESHOLD:
        Default number of System instances in memory before a warning is issued
        Default: 10
    SHOOTER_TOL:
        Default tolerance which DifferentialCorrector converges to, resolved at
        instance construction.  Can be overwritten by an input parameter.
        Default: 1e-10
    SHOOTER_MAX_ITER:
        Default maximum number of iterations take by a DifferentialCorrector, 
        resolved at instance construction.  Can be overwritten by an input parameter.
        Default: 50
    SHOOTER_COND_WARN:
        Default Jacobian matrix condition number at which a warning is issued within a
        DifferentialCorrector instance, resolved at instance construction.  This
        is separate from the rank check, which is made by np.linalg.lstsq() using the
        _LSTSQ_RCOND=1e-13 constant, which is not user-adjustable.  Thus, a Jacobian
        singular value less than this constant will trigger rank-deficiency and be
        excluded from the solve.  A singular value above this trigger may still
        trigger SHOOTER_COND_WARN, depending on the other singular values.
        SHOOTER_COND_WARN can be overwritten by an input parameter.
        Default: 1e8
    SHOOTER_COND_FAIL:
        Default Jacobian matrix condition number at which a solve raises within a
        DifferentialCorrector instance, resolved at instance construction.  See 
        docstring above for SHOOTER_COND_WARN to understand how this interacts with
        a rank-deficient Jacobian.  A trigger of SHOOTER_COND_FAIL will result in
        a non-converged solve.  Can be overwritten by an input parameter. 
        Default: 1e12
    PERIODICITY_TOL : float
        Absolute tolerance for periodic-orbit closure and perpendicular-
        crossing tests [nondimensional]. A well-converged CR3BP shooter
        typically closes to ~1e-10 to 1e-12, while a non-periodic trajectory
        has a closure residual of order 0.1 or larger, so the default
        separates the two cleanly with margin. EQUALITY_ATOL (1e-14) is
        intentionally not reused: it is too tight for a freshly repropagated
        full-period arc.
        Default: 1e-9
    DEFAULT_PLOT_POINTS : int
        Default number of points for trajectory plotting.
        Default: 1000
    DEFAULT_FAMILY_PLOT_POINTS : int
        Samples per member in OrbitFamily.plot_3d. Lower than
        DEFAULT_PLOT_POINTS because a family figure holds many orbits.
        Default: 300
    DEFAULT_BODY_COLOR : str
        Default color for celestial bodies in plots.
        Default: 'lightblue'
    DEFAULT_TRAJ_COLOR : str
        Default color for trajectory lines in plots.
        Default: 'red'
    DEFAULT_TRAJ_COLOR_ADD : str
        Default color for additional trajectory lines added via add_to_plot.
        Default: 'blue'
    DEFAULT_BODY_OPACITY : float
        Default opacity for celestial body spheres (0.0 to 1.0).
        Default: 0.6
    PROXIMITY_THRESHOLD : float
        Proximity half of the automatic body test: show a body if the
        plotted trajectories come within this many body radii of it. The
        other half is bounding-box enclosure (PLOT_BBOX_MARGIN). CR3BP only;
        a 2-body primary is always shown.
        Default: 10.0
    RENDERER : str
        default renderer used by Plotly when displaying plots
        Default: 'browser'
    PLOT_BBOX_MARGIN : float
        Fractional expansion of the bounding box used by the automatic
        visibility tests for bodies and Lagrange points. 0.25 grows each
        half-width by 25 percent.
        Default: 0.25
    PLOT_MIN_EXTENT_FRAC : float
        Floor on each bounding box half-width, as a fraction of the largest,
        so a planar trajectory's box keeps some out-of-plane thickness.
        Default: 0.05
    DEFAULT_LAGRANGE_COLOR : str
        Marker color for Lagrange points.
        Default: 'black'
    DEFAULT_LAGRANGE_SYMBOL : str
        Plotly 3D marker symbol for Lagrange points.
        Default: 'x'
    DEFAULT_LAGRANGE_SIZE : int
        Marker size for Lagrange points.
        Default: 5
    """
    
    # Numerical tolerance for equality comparisons
    EQUALITY_RTOL: float = 1e-12
    EQUALITY_ATOL: float = 1e-14
    
    # Snapping behavior thresholds
    SNAP_TO_ZERO_THRESHOLD: float = 1e-10
    SNAP_TO_CIRCULAR: float = 1e-8
    SNAP_TO_EQUATORIAL: float = 1e-8
    
    # Validation behavior
    STRICT_VALIDATION: bool = True
    
    # System defaults
    DEFAULT_COMPILE: bool = True
    INSTANCE_WARNING_THRESHOLD: int = 10

    # Shooting default behavior
    SHOOTER_TOL = 1e-10        # convergence tolerance on ||F||
    SHOOTER_MAX_ITER = 50      # maximum Newton steps
    SHOOTER_COND_WARN = 1e8    # warn above this Jacobian condition number
    SHOOTER_COND_FAIL = 1e12   # abort above this condition number
    # used by PeriodicOrbit and OrbitFamily, not the shooter, but analogous to those
    PERIODICITY_TOL: float = 1e-9   
    
    # Plotting defaults
    DEFAULT_PLOT_POINTS: int = 1000
    DEFAULT_FAMILY_PLOT_POINTS: int = 300
    DEFAULT_BODY_COLOR: str = 'lightblue'
    DEFAULT_TRAJ_COLOR: str = 'red'
    DEFAULT_TRAJ_COLOR_ADD: str = 'blue'
    DEFAULT_BODY_OPACITY: float = 0.6
    PROXIMITY_THRESHOLD: float = 10.0
    RENDERER: str = 'browser'

    # Automatic visibility tests (bodies and Lagrange points)
    PLOT_BBOX_MARGIN: float = 0.25
    PLOT_MIN_EXTENT_FRAC: float = 0.05

    # Lagrange point plotting defaults
    DEFAULT_LAGRANGE_COLOR: str = 'black'
    DEFAULT_LAGRANGE_SYMBOL: str = 'x'
    DEFAULT_LAGRANGE_SIZE: int = 5

    @property
    def HASH_DECIMALS(self) -> int:
        """
        Compute hash rounding decimals from equality tolerance.
        
        The hash rounding must be coarse enough that if two values
        are equal (within EQUALITY_ATOL), they hash to the same value.
        
        Formula: HASH_DECIMALS = -floor(log10(ATOL)) - 2
        The -2 provides safety margin (2 orders of magnitude).
        
        Returns
        -------
        int
            Number of decimal places for hash rounding
        """
        magnitude = -math.floor(math.log10(self.EQUALITY_ATOL))
        return max(magnitude - 2, 0)  # At least 0 decimals
    
    def reset(self):
        """
        Reset all configuration values to package defaults.
        
        Examples
        --------
        >>> import kyklos
        >>> kyklos.config.EQUALITY_RTOL = 1e-6  # Modify
        >>> kyklos.config.reset()  # Back to defaults
        >>> kyklos.config.EQUALITY_RTOL
        1e-12
        """
        defaults = KyklosConfig()
        for key in self.__dataclass_fields__:
            setattr(self, key, getattr(defaults, key))
    
    def __repr__(self):
        """Return formatted string showing all configuration values."""
        lines = ["KyklosConfig:"]
        lines.append("  Numerical Tolerances:")
        lines.append(f"    EQUALITY_RTOL = {self.EQUALITY_RTOL}")
        lines.append(f"    EQUALITY_ATOL = {self.EQUALITY_ATOL}")
        lines.append(f"    HASH_DECIMALS = {self.HASH_DECIMALS}")
        lines.append("  Snapping Thresholds:")
        lines.append(f"    SNAP_TO_ZERO_THRESHOLD = {self.SNAP_TO_ZERO_THRESHOLD}")
        lines.append(f"    SNAP_TO_CIRCULAR = {self.SNAP_TO_CIRCULAR}")
        lines.append(f"    SNAP_TO_EQUATORIAL = {self.SNAP_TO_EQUATORIAL}")
        lines.append("  Behavior:")
        lines.append(f"    STRICT_VALIDATION = {self.STRICT_VALIDATION}")
        lines.append(f"    DEFAULT_COMPILE = {self.DEFAULT_COMPILE}")
        lines.append(f"    INSTANCE_WARNING_THRESHOLD = {self.INSTANCE_WARNING_THRESHOLD}")
        lines.append("  Differential Correction Behavior:")
        lines.append(f"    SHOOTER_TOL = {self.SHOOTER_TOL}")
        lines.append(f"    SHOOTER_MAX_ITER = {self.SHOOTER_MAX_ITER}")
        lines.append(f"    SHOOTER_COND_WARN = {self.SHOOTER_COND_WARN}")
        lines.append(f"    SHOOTER_COND_FAIL = {self.SHOOTER_COND_FAIL}")
        lines.append("  Periodic Orbit Behavior:")
        lines.append(f"    PERIODICITY_TOL = {self.PERIODICITY_TOL}")
        lines.append("  Plotting:")
        lines.append(f"    DEFAULT_PLOT_POINTS = {self.DEFAULT_PLOT_POINTS}")
        lines.append(f"    DEFAULT_FAMILY_PLOT_POINTS = {self.DEFAULT_FAMILY_PLOT_POINTS}")
        lines.append(f"    DEFAULT_BODY_COLOR = '{self.DEFAULT_BODY_COLOR}'")
        lines.append(f"    DEFAULT_TRAJ_COLOR = '{self.DEFAULT_TRAJ_COLOR}'")
        lines.append(f"    DEFAULT_TRAJ_COLOR_ADD = '{self.DEFAULT_TRAJ_COLOR_ADD}'")
        lines.append(f"    DEFAULT_BODY_OPACITY = {self.DEFAULT_BODY_OPACITY}")
        lines.append(f"    PROXIMITY_THRESHOLD = {self.PROXIMITY_THRESHOLD}")
        lines.append(f"    RENDERER = {self.RENDERER}")
        lines.append(f"    PLOT_BBOX_MARGIN = {self.PLOT_BBOX_MARGIN}")
        lines.append(f"    PLOT_MIN_EXTENT_FRAC = {self.PLOT_MIN_EXTENT_FRAC}")
        lines.append(f"    DEFAULT_LAGRANGE_COLOR = {self.DEFAULT_LAGRANGE_COLOR}")
        lines.append(f"    DEFAULT_LAGRANGE_SYMBOL = {self.DEFAULT_LAGRANGE_SYMBOL}")
        lines.append(f"    DEFAULT_LAGRANGE_SIZE = {self.DEFAULT_LAGRANGE_SIZE}")
        return "\n".join(lines)


# Global configuration instance
config = KyklosConfig()


@contextmanager
def temp_config(**kwargs):
    """
    Context manager for temporarily modifying configuration values.
    
    Configuration is automatically restored when the context exits,
    even if an exception occurs.
    
    Parameters
    ----------
    **kwargs
        Configuration attributes to temporarily modify.
    
    Examples
    --------
    >>> import kyklos
    >>> with kyklos.temp_config(EQUALITY_RTOL=1e-6, STRICT_VALIDATION=False):
    ...     # Use relaxed tolerances
    ...     orbit1 = kyklos.OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
    ...     orbit2 = kyklos.OE(a=7000.0001, e=0.01, i=0, omega=0, w=0, nu=0)
    ...     assert orbit1 == orbit2  # True with relaxed tolerance
    >>> # Original config restored here
    >>> kyklos.config.EQUALITY_RTOL
    1e-12
    
    Raises
    ------
    AttributeError
        If an invalid configuration attribute is specified.
    """
    old_values = {}
    for key, value in kwargs.items():
        if not hasattr(config, key):
            raise AttributeError(
                f"KyklosConfig has no attribute '{key}'. "
                f"Valid attributes: {list(config.__dataclass_fields__.keys())}"
            )
        old_values[key] = getattr(config, key)
        setattr(config, key, value)
    
    try:
        yield config
    finally:
        for key, value in old_values.items():
            setattr(config, key, value)