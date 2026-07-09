"""
Kyklos exception hierarchy.

A single package-wide root, KyklosError, sits above every exception Kyklos
raises deliberately. Catching KyklosError catches "anything Kyklos raised on
purpose" while letting genuine programming bugs (KeyError, TypeError, and the
like) propagate and crash as they should.

Below the root, exceptions are grouped by the subsystem that raises them. Only
the correction subtree exists so far:

    KyklosError
    +-- CorrectionError          orbit-correction / periodic-orbit failures
        +-- ConvergenceError     the corrector did not converge
        +-- ClosureError         corrector converged, orbit fails closure

Sibling subtrees (e.g. PropagationError for the integrator, ShootingError for
the corrector internals) will join under KyklosError during the package-wide
exception sweep; they are intentionally not defined yet, to avoid speculative
empty classes for failures not actually raised.

This is a leaf module: it imports nothing and everything can import from it, so
shooter, propagator, correction, and continuation can all raise and catch these
without creating import cycles. Exception payloads are kept primitive (floats,
strings) rather than holding heavy objects (Trajectory, PeriodicOrbit) so the
module stays import-free -- a catcher needs the numbers to make a decision (how
much to tighten a tolerance), not the whole object.
"""


class KyklosError(Exception):
    """
    Base class for all deliberate Kyklos exceptions.

    Catch this to handle any error Kyklos raises intentionally, while letting
    ordinary Python bugs propagate.
    """


# ===========================================================================
# Correction subtree
# ===========================================================================
class CorrectionError(KyklosError):
    """
    Base for failures in the orbit-correction wrapper and periodic-orbit
    construction.

    Catch this to handle any correction failure regardless of specific cause;
    catch a subclass to react to one cause in particular.
    """


class ConvergenceError(CorrectionError):
    """
    The differential corrector failed to converge.

    Raised when the corrector returns no solution (e.g. the initial guess was
    too far from a periodic orbit, or the Newton iteration diverged).

    Attributes
    ----------
    recipe : str
        The recipe label the guess was being corrected against.
    """

    def __init__(self, recipe: str, message: str | None = None):
        self.recipe = recipe
        if message is None:
            message = (
                f"Corrector failed to converge for a {recipe!r} guess; the "
                f"initial guess may be too far from a periodic orbit."
            )
        super().__init__(message)


class ClosureError(CorrectionError):
    """
    The corrector converged, but the full orbit fails periodicity closure.

    Raised when the mirrored, repropagated full orbit has a periodicity defect
    exceeding the closure threshold -- typically because the corrector's own
    tolerance was not tight enough for this orbit's instability (the half-arc
    residual is amplified over the full period).

    The residual and threshold are carried as attributes so a caller (e.g. an
    adaptive-tolerance retry) can decide how much to tighten and retry, rather
    than parsing the message.

    Attributes
    ----------
    residual : float
        The achieved periodicity residual (full-orbit closure defect).
    threshold : float
        The closure threshold the residual failed to meet.
    recipe : str or None
        The recipe label, if known at the point of the failure. May be None,
        since closure is checked during periodic-orbit construction, which does
        not always know the originating recipe.
    """

    def __init__(
        self,
        residual: float,
        threshold: float,
        recipe: str | None = None,
        message: str | None = None
    ):
        self.residual = residual
        self.threshold = threshold
        self.recipe = recipe
        detail = f" for a {recipe!r} orbit" if recipe is not None else ""
        message = message if message is not None else (
            f"Orbit failed closure{detail}: periodicity residual "
            f"{residual:.2e} exceeds threshold {threshold:.2e}."
        )
        
        super().__init__(message)
