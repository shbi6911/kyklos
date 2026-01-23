"""
Utility functions and classes for the Kyklos package.
"""

from time import perf_counter
import warnings
from .config import config

class Timer:
    """
    Context manager for timing code execution.
    
    Examples
    --------
    >>> from kyklos.utils import Timer
    >>> with Timer("Propagation"):
    ...     trajectory = sys.propagate(state, 0, 1000)
    Propagation: 0.123456 s
    
    >>> with Timer() as t:
    ...     # ... code ...
    >>> print(f"Took {t.elapsed:.6f} seconds")
    """
    def __init__(self, name="Operation", verbose=True):
        """
        Parameters
        ----------
        name : str, optional
            Name to display when timing completes (default: "Operation")
        verbose : bool, optional
            Whether to print timing automatically (default: True)
        """
        self.name = name
        self.verbose = verbose
        self.elapsed = None
    
    def __enter__(self):
        self.start = perf_counter()
        return self
    
    def __exit__(self, *args):
        self.end = perf_counter()
        self.elapsed = self.end - self.start
        if self.verbose:
            print(f"{self.name}: {self.elapsed:.6f} s")
    
def validation_error(message: str):
    """
    Raise error or warn based on config.STRICT_VALIDATION.
    
    This function provides consistent validation behavior across the package.
    When STRICT_VALIDATION is True (default), raises ValueError. When False,
    issues a UserWarning instead.
    
    Parameters
    ----------
    message : str
        Validation error message
    
    Raises
    ------
    ValueError
        If config.STRICT_VALIDATION is True
    
    Warns
    -----
    UserWarning
        If config.STRICT_VALIDATION is False
    
    Examples
    --------
    >>> from kyklos.utils import validation_error
    >>> from kyklos import config
    >>> config.STRICT_VALIDATION = True
    >>> validation_error("Invalid value")  # Raises ValueError
    
    >>> config.STRICT_VALIDATION = False
    >>> validation_error("Invalid value")  # Issues warning
    """
    if config.STRICT_VALIDATION:
        raise ValueError(message)
    else:
        warnings.warn(message, UserWarning)