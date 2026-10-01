"""
Small shared helpers: a timing context manager and validation reporting.

``Timer`` measures the wall-clock time of a block of code and, by default,
prints it on exit; the elapsed seconds are also stored on the object. It is
the only public item in this module.

``validation_error`` is an internal helper and not part of the public API. It
gives the package one consistent response to invalid input: it raises an
exception when ``config.STRICT_VALIDATION`` is True (the default) and issues a
``UserWarning`` otherwise. Users control this behavior through
``config.STRICT_VALIDATION`` rather than by calling the helper.

Examples
--------
Time a propagation::

    import kyklos as ky

    with ky.Timer("Propagation") as t:
        traj = ky.earth_2body().propagate(ky.leo_orbit(), [0.0, 5400.0])
    print(t.elapsed)
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
    >>> import kyklos as ky
    >>> sys = ky.earth_2body()
    >>> with ky.Timer("Propagation"):
    ...     trajectory = sys.propagate(state, [0, 1000])
    Propagation: 0.123456 s
    
    >>> with ky.Timer() as t:
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
    
def validation_error(
    message: str,
    error_class: Type[Exception] = ValueError,
    stacklevel: int = 2          # callers can override when called from deeper frames
):
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
    stacklevel : int = 2, optional
        How far up the function call stack to point to when raising a warning
        Default: stacklevel=2
    
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
        warnings.warn(message, UserWarning, stacklevel=stacklevel)