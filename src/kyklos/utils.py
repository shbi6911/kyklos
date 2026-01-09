"""
Utility functions and classes for the Kyklos package.
"""

from time import perf_counter

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