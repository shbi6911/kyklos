"""
Utility functions and classes for the Kyklos package.
"""

from time import perf_counter
import warnings
from typing import Type
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
    
def validation_error(message: str, error_class: Type[Exception] = ValueError):
    """
    Raise error or warn based on config.STRICT_VALIDATION.
    
    This function provides consistent validation behavior across the package.
    When STRICT_VALIDATION is True (default), raises the specified exception.
    When False, issues a UserWarning instead.
    
    Parameters
    ----------
    message : str
        Validation error message
    error_class : Type[Exception], optional
        Exception class to raise if STRICT_VALIDATION is True.
        Default: ValueError
    
    Raises
    ------
    Exception (of type error_class)
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
    >>> validation_error("Computation failed", RuntimeError)  # Raises RuntimeError
    
    >>> config.STRICT_VALIDATION = False
    >>> validation_error("Invalid value")  # Issues warning
    >>> validation_error("Computation failed", RuntimeError)  # Also issues warning
    """
    if config.STRICT_VALIDATION:
        raise error_class(message)
    else:
        warnings.warn(message, UserWarning, stacklevel=2)