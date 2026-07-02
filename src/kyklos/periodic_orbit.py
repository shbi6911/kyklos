"""PeriodicOrbit class for verified periodic CR3BP orbits.

A PeriodicOrbit is the active, in-memory representation of a trajectory that
has been verified to close on itself over one full period. Unlike a generic
Trajectory, its existence is a certificate of periodicity: it owns a
full-period trajectory, the monodromy matrix (the STM integrated over exactly
one period), and the Floquet/stability structure that periodicity unlocks.
Invariant-manifold generation is stubbed pending implementation.

This module is a runtime leaf with respect to System and Trajectory: it imports
both only under TYPE_CHECKING and operates on live instances passed in by the
caller. That keeps the Trajectory -> PeriodicOrbit dependency (used by the
forthcoming Trajectory.to_periodic delegator) acyclic.

Construction
------------
A PeriodicOrbit is built from a converged Trajectory:

    po = PeriodicOrbit(traj)                 # period inferred from geometry
    po = PeriodicOrbit(traj, period=T)       # period supplied explicitly

When the period is omitted it is inferred from endpoint geometry:
  - full-period formulation:  start ~= end                  -> period = span
  - mirror half-orbit:        both ends are perpendicular
                              x-z plane crossings            -> period = 2*span
Closure of the resulting full-period trajectory is the universal periodicity
certificate; the perpendicular-crossing test only selects the period multiplier.
For exotic cases (other symmetry planes, multi-revolution families) supply the
period explicitly.

Created with the assistance of Claude Opus 4.8 by Anthropic.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, TYPE_CHECKING

from .orbital_elements import OrbitalElements, OEType
from .utils import validation_error
from .config import config

if TYPE_CHECKING:
    from .trajectory import Trajectory
    from .system import System


# ========== MODULE CONSTANTS ==========

# Default absolute tolerance for closure and perpendicular-crossing tests.
# A well-converged CR3BP shooter typically closes to ~1e-10 to 1e-12, while a
# non-periodic trajectory has a closure residual of order 0.1 or larger, so a
# default of 1e-9 (nondimensional units) separates the two cleanly with margin.
# config.EQUALITY_ATOL (1e-14) is intentionally not reused here: it is too tight
# for a freshly repropagated full-period arc.
_DEFAULT_PERIODICITY_TOL = 1e-9

# Enum .value sentinels, compared by string to avoid runtime imports of
# SysType / OEType / CR3BPSystem (which would reintroduce the import cycle).
_CR3BP_SYS_VALUE = '3body'   # SysType.CR3BP.value
_CR3BP_OE_VALUE = 'cr3bp'    # OEType.CR3BP.value


class PeriodicOrbit:
    """
    A trajectory verified to be periodic, with monodromy and stability data.

    A PeriodicOrbit composes a full-period Trajectory and exposes the
    structure that only a periodic orbit possesses: the monodromy matrix,
    Floquet multipliers, a stability index, and (eventually) invariant
    manifolds. It is the natural output of a converged periodic-orbit solve
    and the working object pulled from an OrbitFamily for active analysis.

    Construction repropagates the orbit with variational equations when
    necessary so that the monodromy is always available, and validates that
    the full-period trajectory closes on itself. The closure residual is
    retained and exposed via the periodicity_residual property.

    Currently restricted to CR3BP systems: the Jacobi constant and the
    perpendicular-crossing inference are CR3BP-specific.

    Parameters
    ----------
    trajectory : Trajectory
        A converged trajectory describing the orbit. May be a full-period
        arc or a mirror-symmetry half-arc (see period). Must reference a
        CR3BP System.
    period : float or None, optional
        The full period. If None (default), it is inferred from endpoint
        geometry. If supplied, it overrides inference; the trajectory is
        repropagated from its initial state over [0, period] unless period
        already matches the trajectory span and an STM is present.
    name : str, optional
        Human-readable identifier (e.g. 'L1 Lyapunov'). Default: "".
    tol : float or None, optional
        Absolute tolerance for the closure and perpendicular-crossing tests.
        If None, defaults to 1e-9 (nondimensional). Resolved at construction
        and exposed via the tol property.

    Attributes
    ----------
    trajectory : Trajectory
        The verified full-period trajectory (read-only).
    period : float
        The full period [nondimensional] (read-only).
    name : str
        Identifier (read-only).
    initial_state : OrbitalElements
        CR3BP initial conditions at the start of the period (read-only).
    monodromy : np.ndarray
        The (6, 6) monodromy matrix Phi(T, 0) (read-only copy).
    periodicity_residual : float
        Norm of (end_state - start_state) for the full-period trajectory.
    mode : str
        How the period was determined: 'full_period', 'mirror_half',
        or 'explicit'.

    Examples
    --------
    >>> po = PeriodicOrbit(traj, name='L1 Lyapunov')
    >>> po.period
    3.3897...
    >>> po.monodromy.shape
    (6, 6)
    >>> po.stability_index
    1234.5...

    See Also
    --------
    Trajectory.to_periodic : Thin delegator that constructs a PeriodicOrbit.
    OrbitFamily : Serialized / continuation-output form of a family of orbits.
    """

    # ========== CONSTRUCTION ==========
    def __init__(
            self,
            trajectory: "Trajectory",
            period: Optional[float] = None,
            *,
            name: str = "",
            tol: Optional[float] = None,
    ):
        tol = _DEFAULT_PERIODICITY_TOL if tol is None else float(tol)
        if tol < 0.0:
            raise ValueError(f"tol must be non-negative, got {tol}")
        self._tol = tol
        self._name = str(name)

        system = trajectory.system
        if getattr(system, 'base_type', None) is None \
                or system.base_type.value != _CR3BP_SYS_VALUE:
            raise ValueError(
                "PeriodicOrbit requires a CR3BP system, got "
                f"base_type={getattr(system, 'base_type', None)}."
            )

        span = float(trajectory.duration)
        if not np.isfinite(span) or span <= 0.0:
            raise ValueError(
                f"Trajectory duration must be positive and finite, got {span}."
            )

        # ----- Determine the full period and whether repropagation is needed -----
        if period is None:
            period, mode, must_reprop = self._infer_period(trajectory, span, tol)
        else:
            period = float(period)
            if period <= 0.0:
                raise ValueError(f"period must be positive, got {period}.")
            mode = 'explicit'
            # Reuse the input arc only if it already spans the full period.
            must_reprop = not np.isclose(
                period, span, rtol=config.EQUALITY_RTOL, atol=tol
            )

        # The monodromy requires an STM over the full period. If the input
        # trajectory has no STM, repropagate regardless of the period match.
        if trajectory.stm_order is None:
            must_reprop = True

        # ----- Build the verified full-period trajectory -----
        if must_reprop:
            ic_raw = trajectory.state_at_raw(trajectory.t0)
            # Single-segment repropagation from the initial state. For a
            # converged multi-segment input this collapses the (continuous)
            # junction structure into one segment.
            full = system.propagate(ic_raw, [0.0, period], with_stm=True)
        else:
            full = trajectory

        # ----- Validate closure (the universal periodicity certificate) -----
        s0 = full.state_at_raw(full.t0)
        s1 = full.state_at_raw(full.tf)
        residual = float(np.linalg.norm(s1 - s0))
        if not np.allclose(s1, s0, rtol=config.EQUALITY_RTOL, atol=tol):
            validation_error(
                f"Trajectory does not close over the period: "
                f"|end - start| = {residual:.3e} exceeds tolerance {tol:.3e}. "
                f"(period={period:.6g}, mode={mode}). If this is a partial arc, "
                f"supply the correct period explicitly."
            )

        # ----- Store verified state -----
        self._trajectory = full
        self._period = period
        self._mode = mode
        self._periodicity_residual = residual
        # state_at attaches the system, so jacobi_const() resolves mu.
        self._initial_state = full.state_at(full.t0)
        # Composite STM over one full period = monodromy. Copy to decouple
        # from any internal Heyoka buffers.
        self._monodromy = np.array(full.get_stm(full.tf), dtype=float).copy()

        # ----- Lazy analysis caches -----
        self._multipliers: Optional[np.ndarray] = None
        self._jacobi: Optional[float] = None

    # ========== INFERENCE HELPERS ==========
    @staticmethod
    def _infer_period(trajectory: "Trajectory", span: float, tol: float):
        """
        Infer the full period from endpoint geometry.

        Closure is tested first: if the arc already closes, it is a
        full-period formulation. Otherwise, if both endpoints are
        perpendicular x-z plane crossings, the arc is a mirror half-orbit
        and the full period is twice the span.

        Returns
        -------
        period : float
        mode : str
            'full_period' or 'mirror_half'.
        must_reprop : bool
            True if a full-period trajectory must be built from the arc.
        """
        s0 = trajectory.state_at_raw(trajectory.t0)
        s1 = trajectory.state_at_raw(trajectory.tf)

        if np.allclose(s1, s0, rtol=config.EQUALITY_RTOL, atol=tol):
            # Already a full period; only repropagate if the STM is absent
            # (handled by the caller via stm_order).
            return span, 'full_period', False

        if PeriodicOrbit._is_perp_crossing(s0, tol) \
                and PeriodicOrbit._is_perp_crossing(s1, tol):
            # Mirror half-orbit: build the full period by repropagating.
            return 2.0 * span, 'mirror_half', True

        raise ValueError(
            "Cannot infer periodicity from endpoint geometry: the arc neither "
            "closes (start ~= end) nor terminates in perpendicular x-z plane "
            "crossings at both ends. Supply the period explicitly via the "
            "'period' argument."
        )

    @staticmethod
    def _is_perp_crossing(state: np.ndarray, tol: float) -> bool:
        """
        Test whether a CR3BP state is a perpendicular x-z plane crossing.

        The standard CR3BP mirror symmetry is about the x-z plane (y = 0).
        A perpendicular crossing has y = 0 and velocity perpendicular to the
        plane, i.e. vx = 0 and vz = 0 (with vy != 0).

        Parameters
        ----------
        state : np.ndarray, shape (6,)
            CR3BP state [x, y, z, vx, vy, vz].
        tol : float
            Absolute tolerance.

        Returns
        -------
        bool
        """
        y, vx, vz = state[1], state[3], state[5]
        return abs(y) <= tol and abs(vx) <= tol and abs(vz) <= tol

    # ========== PROPERTIES: VERIFIED STATE ==========
    @property
    def trajectory(self) -> "Trajectory":
        """The verified full-period Trajectory."""
        return self._trajectory

    @property
    def system(self) -> "System":
        """The CR3BP System this orbit lives in."""
        return self._trajectory.system

    @property
    def period(self) -> float:
        """The full period [nondimensional]."""
        return self._period

    @property
    def name(self) -> str:
        """Human-readable identifier."""
        return self._name

    @property
    def initial_state(self) -> OrbitalElements:
        """CR3BP initial conditions at the start of the period."""
        return self._initial_state

    @property
    def monodromy(self) -> np.ndarray:
        """
        The (6, 6) monodromy matrix Phi(T, 0).

        Returns a read-only copy: the STM integrated over exactly one period,
        whose eigenvalues are the Floquet multipliers.
        """
        out = self._monodromy.copy()
        out.flags.writeable = False
        return out

    @property
    def periodicity_residual(self) -> float:
        """Norm of (end_state - start_state) for the full-period trajectory."""
        return self._periodicity_residual

    @property
    def mode(self) -> str:
        """How the period was determined: full_period, mirror_half, explicit."""
        return self._mode

    @property
    def tol(self) -> float:
        """Absolute tolerance used for closure and perpendicular-crossing tests."""
        return self._tol

    # ========== PROPERTIES: ANALYSIS (LAZY) ==========
    @property
    def jacobi(self) -> float:
        """
        Jacobi constant of the orbit.

        Delegates to OrbitalElements.jacobi_const() on the initial state,
        the single source of truth for the CR3BP Jacobi convention.
        """
        if self._jacobi is None:
            self._jacobi = float(self._initial_state.jacobi_const())
        return self._jacobi

    @property
    def floquet_multipliers(self) -> np.ndarray:
        """
        Floquet multipliers: eigenvalues of the monodromy matrix.

        Returns a length-6 complex array sorted by descending magnitude.
        For a CR3BP periodic orbit the multipliers occur in reciprocal pairs
        (lambda, 1/lambda) with a trivial pair near (1, 1) arising from the
        periodicity / energy integral.

        Returns
        -------
        np.ndarray, shape (6,), dtype complex
            Read-only copy.
        """
        if self._multipliers is None:
            vals = np.linalg.eigvals(self._monodromy)
            order = np.argsort(-np.abs(vals))
            self._multipliers = vals[order]
        out = self._multipliers.copy()
        out.flags.writeable = False
        return out

    @property
    def stability_index(self) -> float:
        """
        Maximum stability index nu = 0.5 * (|lambda_max| + 1 / |lambda_max|).

        Uses the largest-magnitude Floquet multiplier. nu ~= 1 for a linearly
        stable orbit (all multipliers on the unit circle) and grows with the
        strength of the dominant instability.

        Returns
        -------
        float
        """
        lam_max = np.abs(self.floquet_multipliers[0])
        if lam_max == 0.0:
            return np.inf
        return float(0.5 * (lam_max + 1.0 / lam_max))

    # ========== INVARIANT MANIFOLDS (STUBS) ==========
    def stable_manifold(
            self,
            step_off: float = 1e-6,
            n_trajectories: int = 50,
            integration_time: Optional[float] = None,
            direction: str = 'both',
    ):
        """
        Generate the stable invariant manifold of the orbit. (NOT IMPLEMENTED)

        Planned approach: identify the stable eigenvector(s) of the monodromy
        matrix (Floquet multipliers with |lambda| < 1), propagate the orbit
        with its STM to transport the eigenvector around the period, seed
        states at n_trajectories points along the orbit by stepping off by
        step_off along the (normalized) transported eigenvector, and integrate
        each seed backward in time over integration_time to trace the manifold.

        Parameters
        ----------
        step_off : float, optional
            Nondimensional perturbation magnitude along the eigenvector.
        n_trajectories : int, optional
            Number of manifold trajectories (seed points around the orbit).
        integration_time : float or None, optional
            Integration span per trajectory. Default: a multiple of the period.
        direction : str, optional
            'positive', 'negative', or 'both' branches of the manifold.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError(
            "stable_manifold is not yet implemented; this is a stub."
        )

    def unstable_manifold(
            self,
            step_off: float = 1e-6,
            n_trajectories: int = 50,
            integration_time: Optional[float] = None,
            direction: str = 'both',
    ):
        """
        Generate the unstable invariant manifold of the orbit. (NOT IMPLEMENTED)

        Planned approach: as for stable_manifold, but using the unstable
        eigenvector(s) (Floquet multipliers with |lambda| > 1) and integrating
        each seed forward in time.

        Parameters
        ----------
        step_off : float, optional
            Nondimensional perturbation magnitude along the eigenvector.
        n_trajectories : int, optional
            Number of manifold trajectories (seed points around the orbit).
        integration_time : float or None, optional
            Integration span per trajectory. Default: a multiple of the period.
        direction : str, optional
            'positive', 'negative', or 'both' branches of the manifold.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError(
            "unstable_manifold is not yet implemented; this is a stub."
        )

    # ========== UTILITY ==========
    def summary(self) -> None:
        """Print a human-readable summary of the periodic orbit."""
        label = self._name if self._name else "(unnamed)"
        print(f"PeriodicOrbit: {label}")
        print(f"  mode            : {self._mode}")
        print(f"  period          : {self._period:.10f}")
        print(f"  Jacobi constant : {self.jacobi:.10f}")
        print(f"  closure residual: {self._periodicity_residual:.3e}")
        print(f"  stability index : {self.stability_index:.6e}")
        mults = self.floquet_multipliers
        print("  Floquet multipliers (by |lambda|):")
        for lam in mults:
            print(f"    {lam.real:+.6e} {lam.imag:+.6e}j  |lambda|={abs(lam):.6e}")

    def __repr__(self) -> str:
        label = f"'{self._name}'" if self._name else "unnamed"
        return (
            f"PeriodicOrbit({label}, period={self._period:.6g}, "
            f"jacobi={self.jacobi:.6g}, stability_index={self.stability_index:.3g}"
        )
