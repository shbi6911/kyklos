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
        Default: 1e-12 (approximately millimeter-level at LEO distances)
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
    DEFAULT_PLOT_POINTS : int
        Default number of points for trajectory plotting.
        Default: 1000
    DEFAULT_BODY_COLOR : str
        Default color for celestial bodies in plots.
        Default: 'lightblue'
    DEFAULT_TRAJ_COLOR : str
        Default color for trajectory lines in plots.
        Default: 'red'
    DEFAULT_BODY_OPACITY : float
        Default opacity for celestial body spheres (0.0 to 1.0).
        Default: 0.6
    PROXIMITY_THRESHOLD : float
        Show celestial body if trajectory within this many body radii.
        Only affects CR3BP plotting.
        Default: 10.0
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
    
    # Plotting defaults
    DEFAULT_PLOT_POINTS: int = 1000
    DEFAULT_BODY_COLOR: str = 'lightblue'
    DEFAULT_TRAJ_COLOR: str = 'red'
    DEFAULT_TRAJ_COLOR_ADD: str = 'blue'
    DEFAULT_BODY_OPACITY: float = 0.6
    PROXIMITY_THRESHOLD: float = 10.0

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
        import math
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
        lines.append("  Integration:")
        lines.append(f"    INTEGRATION_RTOL = {self.INTEGRATION_RTOL}")
        lines.append(f"    INTEGRATION_ATOL = {self.INTEGRATION_ATOL}")
        lines.append("  Behavior:")
        lines.append(f"    STRICT_VALIDATION = {self.STRICT_VALIDATION}")
        lines.append(f"    DEFAULT_COMPILE = {self.DEFAULT_COMPILE}")
        lines.append("  Plotting:")
        lines.append(f"    DEFAULT_PLOT_POINTS = {self.DEFAULT_PLOT_POINTS}")
        lines.append(f"    DEFAULT_BODY_COLOR = '{self.DEFAULT_BODY_COLOR}'")
        lines.append(f"    DEFAULT_TRAJ_COLOR = '{self.DEFAULT_TRAJ_COLOR}'")
        lines.append(f"    DEFAULT_BODY_OPACITY = {self.DEFAULT_BODY_OPACITY}")
        lines.append(f"    PROXIMITY_THRESHOLD = {self.PROXIMITY_THRESHOLD}")
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