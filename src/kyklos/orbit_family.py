"""OrbitFamily: a family of CR3BP periodic orbits produced by continuation.

An OrbitFamily is the output of a continuation march: an index-aligned table
of converged members (initial state, period, and solve diagnostics) plus a
header describing the system and the solve layout that produced them. It is
the serialized / bulk form of a family, against which PeriodicOrbit is the
active, single-orbit working form.

Two properties drive the whole design:

**The constructor takes data only, never a System.** Building a CR3BP System
triggers a heyoka LLVM compile, by far the most expensive operation in the
vicinity, and a family must be constructible from a file where no System
exists. A live System is rebuilt lazily and exactly once, by attach_system().

**Derived data is memoized, never stored.** Floquet multipliers and
reconstructed PeriodicOrbits are computed on first request from one shared
propagation pass, in two tiers: the multiplier cache is small and permanent,
the orbit cache is heavy (order a megabyte per member) and droppable. Two
families with identical data always behave identically, so the caches are an
implementation detail rather than logical state.

Unlike periodic_orbit, this module imports System at runtime rather than only
under TYPE_CHECKING: the family must call the System factory to rebuild its
own system. That is safe here because nothing imports orbit_family except
__init__.py and the continuation engine, so no cycle is possible.

Created with the assistance of Claude by Anthropic.
"""

from __future__ import annotations

import numpy as np
import warnings
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from dataclasses import dataclass
from typing import Optional, Sequence, TYPE_CHECKING

from .config import config
from .exceptions import ClosureError
from .orbital_elements import OrbitalElements, OEType
from .periodic_orbit import PeriodicOrbit
from .system import System, BodyParams
# Private helper, shared until plotting moves to a visualization module.
# A runtime import of trajectory is safe here for the same reason the System
# import is: system already imports trajectory, so no new edge is created.
from .trajectory import _apply_3d_layout

if TYPE_CHECKING:
    import pandas as pd


# ========== MODULE CONSTANTS ==========

# Width of a CR3BP state row. Named rather than inlined because it appears in
# the initial_states shape check, the to_frame() column list, and the
# propagation call.
_STATE_DIM = 6

# Maximum number of offending indices named in a validation message before
# the list is truncated. A bad column is usually bad in bulk; a hundred
# indices in an exception message buries the count that matters.
_MAX_REPORTED_INDICES = 10

# SysType.CR3BP.value, compared by string rather than by enum identity so a
# duck-typed System stand-in (a test fixture that never compiles a real
# heyoka integrator) can satisfy attach_system's check. Same idiom as
# periodic_orbit.
_CR3BP_SYS_VALUE = '3body'

# Coloring options for plot_3d, mapped to their colorbar titles. Stability is
# colored on a log scale: nu runs from ~1 to thousands across a typical
# family, and a linear scale would crowd nearly every member into the bottom
# of the colorscale.
_COLOR_BY = {
    'index': 'member',
    'jacobi': 'C',
    'period': 'T [nd]',
    'stability': 'log10(nu)',
}

# Column names for the six state components in to_frame(), in the canonical
# CR3BP order matching initial_states.
_STATE_COLUMNS = ('x', 'y', 'z', 'vx', 'vy', 'vz')


# ========== DIAGNOSTIC RECORDS ==========

@dataclass(frozen=True)
class ClosureFailure:
    """
    A member that converged in the corrector but failed periodicity closure.

    Carries scalars only. The originating ClosureError is deliberately not
    retained: an exception holds __traceback__, which holds frame objects,
    which hold frame locals -- including the multi-megabyte Trajectory live
    at the raise site. Caching the exception would silently pin exactly the
    objects that dropping the orbit cache exists to release.

    Attributes
    ----------
    index : int
        Positional index into *this* family. For the position in the original
        march, read member_indices[index] -- the two differ for a subfamily
        produced by slicing.
    residual : float
        Achieved full-period closure residual, |end - start|.
    threshold : float
        The closure threshold it failed to meet, i.e. the family's
        closure_tol.
    """

    index: int
    residual: float
    threshold: float

    @property
    def ratio(self) -> float:
        """
        Residual as a multiple of the threshold.

        The actionable quantity: how much tighter the march would have to be
        for this member to close. Returns inf for a zero threshold.
        """
        if self.threshold == 0.0:
            return np.inf
        return float(self.residual / self.threshold)


# ========== VALIDATION HELPERS ==========

def _format_indices(idx: np.ndarray) -> str:
    """Render an index array for an error message, truncating if long."""
    shown = idx[:_MAX_REPORTED_INDICES].tolist()
    if idx.size > _MAX_REPORTED_INDICES:
        return f"{shown} (+{idx.size - _MAX_REPORTED_INDICES} more)"
    return f"{shown}"


def _float_column(values, name: str, n: int) -> np.ndarray:
    """
    Validate and freeze a length-n float column.

    Checks one-dimensionality, length against n, and finiteness. Returns a
    read-only copy, so the caller's array can never alias family state.

    Parameters
    ----------
    values : array-like
    name : str
        Column name, used in error messages.
    n : int
        Required length.

    Returns
    -------
    np.ndarray, shape (n,), dtype float
        Read-only.
    """
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(
            f"{name} must be one-dimensional, got shape {arr.shape}."
        )
    if arr.shape[0] != n:
        raise ValueError(
            f"{name} has length {arr.shape[0]}, expected {n} to match "
            f"initial_states. Every column must be index-aligned with the "
            f"member table."
        )
    bad = np.flatnonzero(~np.isfinite(arr))
    if bad.size:
        raise ValueError(
            f"{name} contains {bad.size} non-finite value(s) at index/indices "
            f"{_format_indices(bad)}."
        )
    out = arr.copy()
    out.flags.writeable = False
    return out


def _int_column(values, name: str, n: int) -> np.ndarray:
    """
    Validate and freeze a length-n integer column.

    As _float_column, but the values must be whole numbers. A fractional
    input raises rather than being silently floored: np.asarray([1.9],
    dtype=int) yields 1 with no complaint, which would quietly corrupt an
    iteration count.

    Returns
    -------
    np.ndarray, shape (n,), dtype int
        Read-only.
    """
    as_float = np.asarray(values, dtype=float)
    if as_float.ndim != 1:
        raise ValueError(
            f"{name} must be one-dimensional, got shape {as_float.shape}."
        )
    if as_float.shape[0] != n:
        raise ValueError(
            f"{name} has length {as_float.shape[0]}, expected {n} to match "
            f"initial_states. Every column must be index-aligned with the "
            f"member table."
        )
    bad = np.flatnonzero(~np.isfinite(as_float))
    if bad.size:
        raise ValueError(
            f"{name} contains {bad.size} non-finite value(s) at index/indices "
            f"{_format_indices(bad)}."
        )
    out = as_float.astype(int)
    fractional = np.flatnonzero(out != as_float)
    if fractional.size:
        raise ValueError(
            f"{name} must contain whole numbers; got fractional value(s) at "
            f"index/indices {_format_indices(fractional)}. Refusing to "
            f"truncate silently."
        )
    out.flags.writeable = False
    return out


def _require_nonnegative(arr: np.ndarray, name: str,
                         strict: bool = False) -> None:
    """
    Reject negative (or non-positive, if strict) entries in a column.

    These quantities are non-negative by definition -- a period, an iteration
    count, a residual norm, an arclength step. Checking catches a sign error
    at the boundary where it is cheap to see rather than several plots later.
    """
    bad = np.flatnonzero(arr <= 0.0) if strict else np.flatnonzero(arr < 0.0)
    if bad.size:
        wanted = "strictly positive" if strict else "non-negative"
        raise ValueError(
            f"{name} must be {wanted}; got {arr[bad[0]]} at index "
            f"{int(bad[0])} ({bad.size} offending entry/entries at "
            f"{_format_indices(bad)})."
        )


def _normalize_body(body, label: str) -> BodyParams:
    """
    Coerce a body argument to a genuine BodyParams dataclass.

    CR3BPSystem.primary_body returns a _BodyParamsWithND wrapper rather than
    the dataclass, so an engine reading bodies off a live System hands one in
    naturally. The wrapper delegates attribute access but is not a dataclass
    instance, which breaks dataclasses.asdict() (it resolves
    __dataclass_fields__ on type(obj) and so cannot see through __getattr__)
    and any isinstance check.

    Duck-typed rather than isinstance-checked against _BodyParamsWithND,
    which is module-private to system.py; the isinstance check immediately
    after keeps the sniff safe.
    """
    if not isinstance(body, BodyParams):
        body = body.unwrap() if hasattr(body, 'unwrap') else body
    if not isinstance(body, BodyParams):
        raise TypeError(
            f"{label} must be a BodyParams (or a CR3BP System's "
            f"nondimensional wrapper around one), got "
            f"{type(body).__name__}."
        )
    return body


# ========== ACCESS HELPERS ==========

def _readonly_view(arr: np.ndarray) -> np.ndarray:
    """
    Hand out a stored array without exposing it to mutation.

    Returning the stored array directly is not sufficient. The stored arrays
    own their data, and numpy allows an owner's WRITEABLE flag to be turned
    back on, so a caller could do `out.flags.writeable = True` and then
    mutate family state in place. A view does not own its data, and numpy
    refuses to make a view of a read-only base writeable, so the realistic
    accident is blocked.

    A view rather than a copy because it costs no data movement: measured at
    n = 1000, roughly 0.4 microseconds per access against 2.0 for a copy,
    and the gap widens with n.

    Not airtight -- a caller who reaches through `out.base` and unfreezes the
    original can still get there -- but nothing short of copying on every
    access is, and copying protects only against the accident this already
    covers.

    Parameters
    ----------
    arr : np.ndarray
        A stored column, already frozen at construction.

    Returns
    -------
    np.ndarray
        Read-only view sharing memory with `arr`.
    """
    out = arr.view()
    out.flags.writeable = False
    return out


# ========== ORBIT FAMILY ==========

class OrbitFamily:
    """
    An index-aligned table of converged CR3BP periodic-orbit family members.

    Constructed once, from data, at the end of a continuation march -- there
    is no append and no finalize. Every column is length n and index-aligned;
    that alignment is the invariant every other method depends on. Arrays are
    copied and frozen at construction, so a family never aliases the caller's
    data and cannot be mutated through the property wall.

    Column names are plural throughout (periods, not period) because they are
    (n,) arrays against PeriodicOrbit's scalars. This also does guardrail
    work: indexing returns a one-member OrbitFamily rather than a
    PeriodicOrbit, so `family[3].period` raises AttributeError naming the
    exact attribute instead of silently meaning something else.

    Parameters
    ----------
    initial_states : array-like, shape (n, 6)
        Member initial states [x, y, z, vx, vy, vz], nondimensional, in the
        rotating frame, at the start of each member's period.
    periods : array-like, shape (n,)
        Full periods [nondimensional]. Must be strictly positive.
    iterations : array-like, shape (n,)
        Corrector iteration count per member. Whole numbers.
    final_residuals : array-like, shape (n,)
        Converged corrector residual ||F|| per member.
    step_sizes : array-like, shape (n,)
        Arclength step ds taken to reach each member from its predecessor.
        step_sizes[0] is 0.0 by convention: member one is the bootstrap and
        was not reached by a step. Zero is unambiguous because ContinuationRef
        enforces ds > 0.
    primary_body : BodyParams
        Larger primary. A CR3BP System's nondimensional wrapper is accepted
        and unwrapped.
    secondary_body : BodyParams
        Smaller primary. Same.
    distance : float
        Distance between primaries [km]; sets L*.
    mu : float
        Mass ratio mu_2 / (mu_1 + mu_2). Stored explicitly rather than
        re-derived, so the family cannot drift from CR3BPSystem's definition.
        The engine reads it off the live system.mass_ratio.
    recipe : str
        Recipe label the march ran, e.g. 'lyapunov'.
    scheme : str
        Continuation scheme label, e.g. 'pseudo_arclength'.
    free_vars : sequence of str
        Free start-state component names (determinacy triple).
    free_times : sequence of int
        Free boundary-time indices (determinacy triple).
    node_specs : dict or None, optional
        Multiple-shooting node specification (determinacy triple). Must be
        None; a non-None value raises NotImplementedError.
    member_indices : array-like, shape (n,), optional
        Position of each member in the original march. Defaults to
        arange(n). Slicing passes the sliced array through, so a subfamily
        still knows where its members came from.
    closure_tol : float or None, optional
        Absolute closure threshold applied uniformly to every member when
        reconstructing PeriodicOrbits. If None, resolved from
        config.PERIODICITY_TOL at construction and stored, so a later change
        to config cannot alter this family's verdicts.

    Raises
    ------
    ValueError
        On any column-length, shape, finiteness, sign, or range violation, or
        if n == 0.
    TypeError
        If a body argument is not a BodyParams or a wrapper around one.
    NotImplementedError
        If node_specs is not None.

    See Also
    --------
    PeriodicOrbit : The active, single-orbit working form.
    """

    __slots__ = (
        # Member table
        '_initial_states', '_periods', '_jacobi_constants', '_iterations',
        '_final_residuals', '_step_sizes', '_member_indices',
        # Header
        '_primary_body', '_secondary_body', '_distance', '_mu',
        '_recipe', '_scheme', '_free_vars', '_free_times', '_node_specs',
        '_closure_tol',
        # Lazy caches (never serialized)
        '_system', '_multipliers', '_orbits', '_closure_failures',
    )

    # ========== CONSTRUCTION ==========
    def __init__(
            self,
            *,
            initial_states,
            periods,
            iterations,
            final_residuals,
            step_sizes,
            primary_body,
            secondary_body,
            distance: float,
            mu: float,
            recipe: str,
            scheme: str,
            free_vars: Sequence[str],
            free_times: Sequence[int],
            node_specs: Optional[dict] = None,
            member_indices=None,
            closure_tol: Optional[float] = None,
    ):
        # ----- Multiple shooting is not representable here -----
        # The member table is structurally single-shooting: one (6,) initial
        # state per member. Storing a multiple-shooting layout would record a
        # claim the class cannot honor. Mirrors SolveSpec.corank.
        if node_specs is not None:
            raise NotImplementedError(
                "OrbitFamily does not support multiple shooting; node_specs "
                "must be None. The member table stores one initial state per "
                "member, with no junction block."
            )

        # ----- initial_states establishes n -----
        states = np.asarray(initial_states, dtype=float)
        if states.ndim != 2 or states.shape[1] != _STATE_DIM:
            hint = ""
            if states.ndim == 1 and states.shape[0] == _STATE_DIM:
                hint = (" For a single member, pass shape (1, 6) -- e.g. "
                        "state[None, :] -- not a bare (6,) state.")
            raise ValueError(
                f"initial_states must have shape (n, {_STATE_DIM}), got "
                f"{states.shape}.{hint}"
            )

        n = states.shape[0]
        if n < 1:
            # The only route to zero members is a failed bootstrap -- the
            # unclosed re-solve of member one -- which the engine must raise
            # on, since a bootstrap that will not converge means the input
            # orbit or the tolerance is wrong. So this guard catches an
            # engine bug, not a normal outcome.
            raise ValueError(
                "OrbitFamily requires at least one member. A zero-member "
                "family is unreachable by design: a march that fails after "
                "the bootstrap still has member one, and a bootstrap that "
                "fails must raise in the engine rather than produce an empty "
                "family."
            )

        bad = np.flatnonzero(~np.isfinite(states).all(axis=1))
        if bad.size:
            raise ValueError(
                f"initial_states contains non-finite values in "
                f"{bad.size} row(s) at index/indices {_format_indices(bad)}."
            )

        self._initial_states = states.copy()
        self._initial_states.flags.writeable = False

        # ----- Member table columns -----
        self._periods = _float_column(periods, 'periods', n)
        _require_nonnegative(self._periods, 'periods', strict=True)

        self._final_residuals = _float_column(
            final_residuals, 'final_residuals', n)
        _require_nonnegative(self._final_residuals, 'final_residuals')

        # step_sizes[0] is 0.0 for the bootstrap member, so the check is
        # non-strict; ContinuationRef enforces ds > 0 for every real step,
        # which is what makes zero an unambiguous "no step taken".
        self._step_sizes = _float_column(step_sizes, 'step_sizes', n)
        _require_nonnegative(self._step_sizes, 'step_sizes')

        self._iterations = _int_column(iterations, 'iterations', n)
        _require_nonnegative(self._iterations, 'iterations')

        if member_indices is None:
            self._member_indices = np.arange(n, dtype=int)
            self._member_indices.flags.writeable = False
        else:
            self._member_indices = _int_column(
                member_indices, 'member_indices', n)
            _require_nonnegative(self._member_indices, 'member_indices')

        # ----- Header: system description -----
        self._primary_body = _normalize_body(primary_body, 'primary_body')
        self._secondary_body = _normalize_body(
            secondary_body, 'secondary_body')

        self._distance = float(distance)
        if not np.isfinite(self._distance) or self._distance <= 0.0:
            raise ValueError(
                f"distance must be positive and finite, got "
                f"{self._distance!r}."
            )

        self._mu = float(mu)
        if not np.isfinite(self._mu) or not (0.0 < self._mu < 1.0):
            raise ValueError(
                f"mu must be a mass ratio in (0, 1), got {self._mu!r}. It is "
                f"mu_2 / (mu_1 + mu_2), read off CR3BPSystem.mass_ratio."
            )

        # ----- Header: solve description -----
        self._recipe = str(recipe)
        self._scheme = str(scheme)

        # A bare string is iterable, so tuple(str(v) for v in 'xy') would
        # silently yield ('x', 'y'). Reject it rather than record a layout
        # nobody specified.
        if isinstance(free_vars, str):
            raise TypeError(
                "free_vars must be a sequence of component names, not a bare "
                "string; pass ('x',) rather than 'x'."
            )
        self._free_vars = tuple(str(v) for v in free_vars)
        self._free_times = tuple(int(t) for t in free_times)
        self._node_specs = None

        # ----- Header: closure tolerance -----
        # Resolved from config at construction and stored, so a later change
        # to config.PERIODICITY_TOL cannot alter this family's verdicts, and
        # so a loaded family reproduces the verdicts it was saved with. Same
        # discipline DifferentialCorrector uses for its config defaults.
        self._closure_tol = (
            float(config.PERIODICITY_TOL) if closure_tol is None
            else float(closure_tol)
        )
        if self._closure_tol < 0.0:
            raise ValueError(
                f"closure_tol must be non-negative, got {self._closure_tol}."
            )

        # ----- Derived column: Jacobi constant -----
        # Closed form in mu alone -- no System, no integrator -- so it is
        # cheap enough to be a real column rather than a lazy value. Routed
        # through OrbitalElements to keep one source of truth for the Jacobi
        # sign/factor convention, which differs across textbooks.
        #
        # validate=False deliberately: the states are already validated
        # finite above, and OrbitalElements' CR3BP validation is a magnitude
        # heuristic (|r| > 10) routed through validation_error, which under
        # STRICT_VALIDATION would raise on a legitimate large-amplitude
        # member. The elements objects are transient -- built, read, dropped.
        jacobi = np.empty(n, dtype=float)
        for k in range(n):
            elements = OrbitalElements(
                self._initial_states[k], OEType.CR3BP,
                validate=False, mu=self._mu,
            )
            jacobi[k] = float(elements.jacobi_const())
        jacobi.flags.writeable = False
        self._jacobi_constants = jacobi

        # ----- Lazy caches -----
        # None until first requested. _system is rebuilt (or accepted and
        # validated) by attach_system; _multipliers and _orbits are filled by
        # _ensure_multipliers, the latter droppable via clear_orbit_cache.
        self._system: Optional[System] = None
        self._multipliers: Optional[np.ndarray] = None
        self._orbits: Optional[tuple] = None
        self._closure_failures: tuple[ClosureFailure, ...] = ()

    # ========== SIZE ==========
    @property
    def n(self) -> int:
        """Number of members in the family."""
        return self._initial_states.shape[0]

    def __len__(self) -> int:
        """Number of members, so len(family) works alongside indexing."""
        return self._initial_states.shape[0]

    # ========== PROPERTIES: MEMBER TABLE ==========
    # Every array property below returns a read-only *view* of its stored
    # column -- not the stored array, which a caller could unfreeze and
    # mutate, and not a fresh copy, which would cost data movement on every
    # access. See _readonly_view.
    #
    # All seven columns are length n and index-aligned: entry k of any one of
    # them describes the same member as entry k of every other, and as
    # to_orbits()[k] and floquet_multipliers()[k]. Nothing in this class
    # breaks that correspondence, including slicing.

    @property
    def initial_states(self) -> np.ndarray:
        """
        Member initial states [x, y, z, vx, vy, vz]. Read-only (n, 6).

        Nondimensional, in the rotating frame, at the start of each member's
        period.
        """
        return _readonly_view(self._initial_states)

    @property
    def periods(self) -> np.ndarray:
        """Full period of each member [nondimensional]. Read-only (n,)."""
        return _readonly_view(self._periods)

    @property
    def jacobi_constants(self) -> np.ndarray:
        """
        Jacobi constant of each member. Read-only (n,).

        Computed at construction from the stored mass ratio, via
        OrbitalElements.jacobi_const -- the single source of truth for the
        CR3BP Jacobi sign and factor convention, which differs across
        textbooks. Not lazy: the closed form needs no System and no
        integrator, so it is cheap enough to be a real column.
        """
        return _readonly_view(self._jacobi_constants)

    @property
    def iterations(self) -> np.ndarray:
        """Corrector iteration count per member. Read-only (n,), int."""
        return _readonly_view(self._iterations)

    @property
    def final_residuals(self) -> np.ndarray:
        """
        Converged corrector residual ||F|| per member. Read-only (n,).

        This is the *corrector* residual on the half arc, not the closure
        residual over the full period -- the latter is what
        closure_failures reports, and it is larger by the amplification of
        the member's instability over a full revolution.
        """
        return _readonly_view(self._final_residuals)

    @property
    def step_sizes(self) -> np.ndarray:
        """
        Arclength step ds taken to reach each member. Read-only (n,).

        Entry k is the step from member k-1 to member k, so this is the one
        column describing an edge rather than a member. Entry 0 is 0.0 by
        convention: member one is the bootstrap and was not reached by a
        step. Zero is unambiguous because ContinuationRef enforces ds > 0
        for every real step, and it keeps np.cumsum(step_sizes) meaningful
        as an arclength coordinate with the seed at s = 0.
        """
        return _readonly_view(self._step_sizes)

    @property
    def member_indices(self) -> np.ndarray:
        """
        Position of each member in the original march. Read-only (n,), int.

        arange(n) for a family straight off a march. Slicing passes the
        sliced array through unchanged, so a subfamily still reports where
        its members sat in the march that produced them: for a failure at
        positional index k, member_indices[k] is the original position.
        """
        return _readonly_view(self._member_indices)

    # ========== PROPERTIES: HEADER ==========
    # Immutable objects (frozen dataclasses, tuples, scalars), so these are
    # returned directly -- there is nothing for a caller to mutate.

    @property
    def primary_body(self) -> BodyParams:
        """
        Parameters of the larger primary.

        Always a genuine BodyParams, never the nondimensional wrapper a
        CR3BP System hands out: the constructor unwraps on the way in, so
        dataclasses.asdict() works here and the annotation is honest.
        """
        return self._primary_body

    @property
    def secondary_body(self) -> BodyParams:
        """Parameters of the smaller primary. See primary_body."""
        return self._secondary_body

    @property
    def distance(self) -> float:
        """Distance between the primaries [km]. Sets L*."""
        return self._distance

    @property
    def mu(self) -> float:
        """
        Mass ratio mu_2 / (mu_1 + mu_2) [nondimensional].

        Stored explicitly rather than re-derived from the two bodies, so the
        family cannot drift from CR3BPSystem's definition. It is also the
        only parameter entering the nondimensional equations of motion,
        which is why attach_system validates a handed-in System against it.
        """
        return self._mu

    @property
    def recipe(self) -> str:
        """Recipe label the march ran, e.g. 'lyapunov'."""
        return self._recipe

    @property
    def scheme(self) -> str:
        """Continuation scheme label, e.g. 'pseudo_arclength'."""
        return self._scheme

    @property
    def free_vars(self) -> tuple[str, ...]:
        """
        Free start-state component names.

        Part of the determinacy triple with free_times and node_specs: the
        recorded fact of what the X vector looked like, kept alongside the
        recipe and scheme labels that would reconstruct it. Comparing the
        two on load is what would catch registry drift.
        """
        return self._free_vars

    @property
    def free_times(self) -> tuple[int, ...]:
        """Free boundary-time indices. See free_vars."""
        return self._free_times

    @property
    def node_specs(self) -> None:
        """
        Multiple-shooting node specification. Always None.

        The member table stores one initial state per member with no
        junction block, so a non-None value is rejected at construction.
        Present as a header field because the determinacy triple would be
        incomplete without it and omitting it would force a format
        migration later.
        """
        return self._node_specs

    @property
    def closure_tol(self) -> float:
        """
        Closure threshold applied to every member [nondimensional].

        Resolved from config.PERIODICITY_TOL at construction if not given,
        then stored -- so a later change to config cannot alter this
        family's verdicts, and a loaded family reproduces the verdicts it
        was saved with. Uniform across the family by construction, which is
        why it is not an argument to the propagation methods.
        """
        return self._closure_tol

    # ========== SYSTEM ==========
    @property
    def has_system(self) -> bool:
        """
        Whether a live System is attached.

        Cheap and side-effect free, unlike attach_system, which builds one if
        absent. Use this to check before deciding whether a call is about to
        pay for an LLVM compile.
        """
        return self._system is not None

    def attach_system(self, system: Optional["System"] = None) -> "System":
        """
        Return the family's live System, building or accepting one as needed.

        Idempotent: every propagation-backed method calls this argument-free
        and never has to think about construction. Building costs one heyoka
        LLVM compile (plus a second, later, for the variational integrator on
        the first STM request), which is why the constructor never does it
        and why this is a method rather than a property.

        The family builds its own System -- a departure from the rest of
        Kyklos, where classes receive a preconstructed one -- because it is
        the first class designed to outlive the session that produced it.
        Once save/load lands, a family can be born from a file, where no
        System exists and none can be passed in.

        Parameters
        ----------
        system : System or None, optional
            A live CR3BP System to adopt instead of building one. Validated
            against the header (see Raises); an accepted System is
            dynamically identical to the one that would have been built, so
            the family's behavior does not depend on which it got. Passing
            the marching System here during development skips a redundant
            compile.

        Returns
        -------
        System
            The attached System. Repeated calls return the same instance.

        Raises
        ------
        ValueError
            If `system` is not a CR3BP system, if its mass ratio or
            characteristic length disagree with the header, or if a
            *different* System is already attached.
        """
        if system is not None:
            self._validate_system(system)
            if self._system is not None and self._system is not system:
                # Not a swap. Any PeriodicOrbits already cached hold
                # Trajectories bound to the old System, so swapping would
                # leave to_orbits()[k].system disagreeing with
                # attach_system(), and the permanent multiplier cache
                # cannot be dropped to force a recompute.
                raise ValueError(
                    "A different System is already attached to this family. "
                    "Cached orbits hold trajectories bound to it, so it "
                    "cannot be replaced; build a fresh family from the same "
                    "data if a different System is genuinely wanted."
                )
            self._system = system
        elif self._system is None:
            self._system = System(
                '3body', self._primary_body, self._secondary_body,
                distance=self._distance,
            )
        return self._system

    def _validate_system(self, system) -> None:
        """
        Check a handed-in System against the stored header.

        Validated rather than trusted, for two reasons. Correctness: the mass
        ratio is the only parameter entering the nondimensional equations of
        motion, so a mismatched System yields Floquet multipliers that are
        plausible and wrong, with no other symptom -- hand a Sun-Earth System
        to an Earth-Moon family and nothing looks amiss. And purity: an
        unvalidated hand-in would make two families with identical data
        behave differently depending on what was passed, which is the
        call-history dependency this class avoids everywhere else.

        The characteristic length is checked because it governs any later
        dimensionalization. The bodies' individual mu values are not checked
        beyond their ratio; past that they are metadata as far as the
        dynamics are concerned.
        """
        base_type = getattr(system, 'base_type', None)
        if base_type is None or base_type.value != _CR3BP_SYS_VALUE:
            raise ValueError(
                f"OrbitFamily requires a CR3BP system, got "
                f"base_type={base_type}."
            )

        mass_ratio = getattr(system, 'mass_ratio', None)
        if mass_ratio is None or not np.isclose(
                float(mass_ratio), self._mu,
                rtol=config.EQUALITY_RTOL, atol=config.EQUALITY_ATOL):
            raise ValueError(
                f"System mass ratio {mass_ratio!r} does not match this "
                f"family's mu ({self._mu!r}). The mass ratio is the only "
                f"parameter in the nondimensional equations of motion, so "
                f"propagating this family in that System would silently "
                f"produce wrong results."
            )

        length = getattr(system, 'L_star', None)
        if length is None:
            length = getattr(system, 'distance', None)
        if length is None or not np.isclose(
                float(length), self._distance,
                rtol=config.EQUALITY_RTOL, atol=config.EQUALITY_ATOL):
            raise ValueError(
                f"System characteristic length {length!r} does not match "
                f"this family's distance ({self._distance!r})."
            )

    # ========== PROPAGATION-BACKED PRODUCTS ==========
    # All three come from one pass, so the pass lives in one helper. Each is
    # a method, not a property: these are n integrations with the STM,
    # seconds to minutes, and a property that expensive is a trap -- debugger
    # panes, IDE inspectors and incautious reprs evaluate properties
    # unbidden. The parentheses make the cost visible at the call site.

    def _ensure_multipliers(self, retain_orbits: bool) -> None:
        """
        Propagate every member once, filling the caches this pass can serve.

        The only place that propagates, builds PeriodicOrbits, catches
        ClosureError, records failures, writes NaN rows, and warns. Building
        the System is attach_system's job, not this function's.

        The guard is on the *requested product*, not on the multiplier cache.
        A pass with retain_orbits=False leaves the orbit cache empty, so a
        later to_orbits() must propagate again even though multipliers are
        present -- guarding on multipliers alone would short-circuit and
        return an empty orbit cache, a silent wrong answer rather than a
        crash. The same applies after clear_orbit_cache().

        Parameters
        ----------
        retain_orbits : bool
            Whether to keep the constructed PeriodicOrbits. They are built
            either way -- the multipliers come off them -- but they are
            heavy (order a megabyte per member, in heyoka continuous-output
            Taylor tables), so they are dropped as they go unless asked for.
        """
        have_multipliers = self._multipliers is not None
        have_orbits = self._orbits is not None
        if have_multipliers and (have_orbits or not retain_orbits):
            return

        system = self.attach_system()
        n = self.n

        multipliers = np.full((n, _STATE_DIM), np.nan, dtype=complex)
        orbits: list = []
        failures: list[ClosureFailure] = []

        for k in range(n):
            period = float(self._periods[k])
            try:
                # Both arguments here are load-bearing for cost, not just
                # correctness. PeriodicOrbit repropagates when must_reprop is
                # set, which would integrate this identical arc a second time
                # -- silently, at 2x the cost of the whole pass. Spanning
                # exactly [0, period] makes its np.isclose(period, span)
                # check pass, and with_stm=True dodges its unconditional
                # "no STM -> repropagate" override. Propagate a half arc, or
                # leave the STM to PeriodicOrbit, and the fast path is gone.
                trajectory = system.propagate(
                    self._initial_states[k], [0.0, period], with_stm=True,
                )
                orbit = PeriodicOrbit(
                    trajectory, period, tol=self._closure_tol,
                )
            except ClosureError as exc:
                # Scalars only. The exception holds __traceback__, which
                # holds frame objects, which hold the multi-megabyte
                # Trajectory live at the raise site; retaining it would pin
                # exactly what retain_orbits=False exists to release.
                failures.append(ClosureFailure(
                    index=k,
                    residual=float(exc.residual),
                    threshold=float(exc.threshold),
                ))
                orbits.append(None)
                continue

            multipliers[k] = orbit.floquet_multipliers
            orbits.append(orbit if retain_orbits else None)

        multipliers.flags.writeable = False
        self._multipliers = multipliers
        self._closure_failures = tuple(failures)
        if retain_orbits:
            self._orbits = tuple(orbits)

        if failures:
            self._warn_closure_failures(failures, n)

    def _warn_closure_failures(self, failures: list, n: int) -> None:
        """
        Emit one summarizing warning for a pass that had closure failures.

        One warning per pass, not one per failure: dozens of per-member
        warnings bury the count and the worst case, which are the two things
        that matter. Fires on any pass that actually propagates rather than
        only the first call -- a pass re-run after clear_orbit_cache() is
        not reading a cache, and a caller who paid for n integrations again
        should see what they found again.
        """
        worst = max(failures, key=lambda f: f.residual)
        indices = np.array([f.index for f in failures], dtype=int)
        warnings.warn(
            f"{len(failures)} of {n} family members failed periodicity "
            f"closure and are None in the returned orbits (NaN in the "
            f"multiplier rows). Positional indices: "
            f"{_format_indices(indices)}. Worst residual "
            f"{worst.residual:.3e} against a threshold of "
            f"{worst.threshold:.3e}, a factor of {worst.ratio:.1f}. "
            f"This is the expected failure mode as a family marches toward "
            f"higher instability: the corrector converges on a half arc, and "
            f"that residual is amplified over the full period, so failures "
            f"cluster as a tail rather than scattering. The lever is a "
            f"tighter corrector tolerance on the march, not a looser closure "
            f"threshold. See .closure_failures for per-member residuals.",
            UserWarning,
            stacklevel=3,
        )

    def to_orbits(self) -> tuple:
        """
        Reconstruct every member as a PeriodicOrbit.

        Propagates each member over its full period with the STM. Expensive:
        n integrations, and the results are heavy enough to be worth dropping
        with clear_orbit_cache() when done.

        Members that fail closure are None, preserving index alignment with
        the member table -- entry k is member k whether or not it closed.
        Filtering them out would silently break that correspondence, which
        surfaces as a bug several plots later. A caller wanting strictness
        checks closure_failures and raises.

        Returns
        -------
        tuple of (PeriodicOrbit or None)
            Length n. The cache itself, as a tuple, so a caller cannot
            mutate the container -- distinct from mutating the immutable
            orbits inside it.
        """
        self._ensure_multipliers(retain_orbits=True)
        assert self._orbits is not None  # guaranteed by retain_orbits=True
        return self._orbits

    def floquet_multipliers(self, retain_orbits: bool = False) -> np.ndarray:
        """
        Floquet multipliers of every member.

        Full multipliers rather than only the stability index, because the
        index is a function of magnitude alone: it cannot tell a multiplier
        crossing +1 from one crossing -1, which is exactly the distinction
        bifurcation detection needs. The trivial-pair deflation fix also
        becomes post-processing over these rather than a repropagation.

        Parameters
        ----------
        retain_orbits : bool, optional
            Keep the PeriodicOrbits built along the way, as if to_orbits()
            had been called. Default False, which releases them as they go.

        Returns
        -------
        np.ndarray, shape (n, 6), dtype complex
            Read-only view. Each row is sorted by descending magnitude,
            matching PeriodicOrbit.floquet_multipliers. A member that failed
            closure has a row of NaN, which plots as a gap and so reads as
            "no answer here" rather than "stability zero".
        """
        self._ensure_multipliers(retain_orbits=retain_orbits)
        assert self._multipliers is not None
        return _readonly_view(self._multipliers)

    def stability_indices(self) -> np.ndarray:
        """
        Maximum stability index of every member.

        nu = 0.5 * (|lambda_max| + 1 / |lambda_max|), using the
        largest-magnitude multiplier: nu ~= 1 for a linearly stable orbit,
        growing with the dominant instability. Matches
        PeriodicOrbit.stability_index.

        A pure function over the multiplier cache, so it takes no
        retain_orbits flag of its own -- the retention decision belongs to
        whatever triggers propagation. If the cache is empty this propagates
        with retain_orbits=False; call to_orbits() or
        floquet_multipliers(retain_orbits=True) first to keep the orbits.

        Computed fresh on each call rather than cached (it is arithmetic over
        an array already in hand), so bind the result rather than calling it
        inside a loop.

        Returns
        -------
        np.ndarray, shape (n,), dtype float
            Read-only. NaN where the member failed closure; inf for the
            degenerate case of a zero-magnitude dominant multiplier.
        """
        if self._multipliers is None:
            self._ensure_multipliers(retain_orbits=False)
        assert self._multipliers is not None

        lam = np.abs(self._multipliers[:, 0])
        # NaN rows propagate NaN through both terms, which is what we want.
        # A zero magnitude would divide by zero; it cannot arise from a real
        # monodromy (symplectic, so multipliers come in reciprocal pairs)
        # but is guarded rather than left to emit a runtime warning.
        with np.errstate(divide='ignore', invalid='ignore'):
            nu = 0.5 * (lam + 1.0 / lam)
        nu = np.where(lam == 0.0, np.inf, nu)
        nu.flags.writeable = False
        return nu

    @property
    def closure_failures(self) -> tuple:
        """
        Per-member closure failures from the most recent propagation pass.

        The programmatic surface behind the summarizing warning: a future
        adaptive-tolerance retry reads residual / threshold (or .ratio) per
        member to size the retry. Always available between passes, so the
        detail is not a one-shot the user scrolled past.

        Empty before any pass has run, which is indistinguishable from
        "propagated, nothing failed" -- check has_multipliers to tell the two
        apart. Reading this property never triggers propagation.

        Returns
        -------
        tuple of ClosureFailure
            Ordered by positional index. `index` is into *this* family;
            member_indices[index] gives the position in the original march.
        """
        return self._closure_failures

    @property
    def has_multipliers(self) -> bool:
        """
        Whether a propagation pass has run and filled the multiplier cache.

        Distinguishes an empty closure_failures meaning "nothing has been
        computed" from one meaning "everything closed".
        """
        return self._multipliers is not None

    @property
    def has_orbits(self) -> bool:
        """Whether reconstructed PeriodicOrbits are currently cached."""
        return self._orbits is not None

    def clear_orbit_cache(self) -> None:
        """
        Release the cached PeriodicOrbits.

        The heavy tier: each orbit owns a Trajectory holding heyoka
        continuous-output objects with full Taylor coefficient tables per
        accepted step, 42 variables with the STM -- order a megabyte per
        member, hundreds of megabytes for a few hundred.

        Multipliers, closure failures, and the attached System all survive;
        that two-tier split is why dropping the orbits does not cost the
        cheap derived values. A later to_orbits() repropagates.
        """
        self._orbits = None

    # ========== SELECTION ==========
    def __getitem__(self, key) -> "OrbitFamily":
        """
        Select members, always returning an OrbitFamily.

        Type-stable by design: `family[3]` is a one-member OrbitFamily, not a
        PeriodicOrbit. A one-member family is a superset of the orbit -- the
        PeriodicOrbit is reachable through it, but the table row is not
        reachable from a PeriodicOrbit -- and a uniform return type means
        downstream code never branches on which indexing form produced its
        input. The plural column names are the guardrail: `family[3].period`
        raises AttributeError naming the exact attribute, because the family
        has `periods`. For the orbit itself, use `to_orbits()[3]`.

        Slicing is the supported way to subsample, zoom, or truncate --
        `family[::5]`, `family[40:60]`, `family[:20]` -- and it is nearly
        free: small arrays sliced, header shared by reference, and the
        parent's live System inherited if it has one, so a subfamily never
        pays a second LLVM compile. The per-member caches are not inherited,
        so every cache in the child is dense and total; that is what made
        slicing preferable to a `spacing=` argument threaded through the
        propagation methods.

        Because `__getitem__` accepts integers and raises IndexError past the
        end, iteration works through the legacy protocol: `for sub in family`
        yields n one-member families.

        Parameters
        ----------
        key : int or slice
            Integer indices are list-like, including negatives. Boolean
            masks and index arrays are deliberately unsupported (see Raises).

        Returns
        -------
        OrbitFamily
            A new family carrying the selected members, this family's header,
            and their positions in the original march via member_indices.

        Raises
        ------
        IndexError
            If an integer index is out of range.
        ValueError
            If a slice selects no members. n >= 1 is a class invariant and a
            subfamily is a full OrbitFamily, so `family[10:10]` cannot
            produce a valid object. Un-list-like and accepted: letting
            slicing build something the constructor would reject is worse.
        TypeError
            For any other key type. Boolean-mask selection is deferred -- it
            is the case that will eventually justify a separate `filter()`
            returning something other than a family, precisely because a
            threshold can legitimately match nothing.

        Notes
        -----
        step_sizes is passed through unchanged, and in a subfamily its entry
        0 is therefore a real arclength step -- the step that reached that
        member from a predecessor *outside* the subfamily -- not the 0.0 that
        marks a march's bootstrap. So np.cumsum(sub.step_sizes) measures
        arclength from the parent's preceding member rather than from zero,
        and under a reordering slice such as family[::-1] the column keeps
        its parent-relative meaning and no longer describes the child's own
        ordering. member_indices[0] != 0 is the tell that a family is a
        subfamily rather than a march.
        """
        n = self.n

        # bool is a subclass of int, so family[True] would silently mean
        # family[1]. Reject it: a boolean key reads as mask selection, which
        # is exactly the deferred feature.
        if isinstance(key, bool):
            raise TypeError(
                "OrbitFamily does not support boolean indexing; got "
                f"{key!r}. Use an integer index or a slice."
            )

        if isinstance(key, (int, np.integer)):
            index = int(key)
            if index < 0:
                index += n
            if not 0 <= index < n:
                raise IndexError(
                    f"Member index {key} is out of range for a family of "
                    f"{n} member(s)."
                )
            selection = slice(index, index + 1)

        elif isinstance(key, slice):
            start, stop, step = key.indices(n)
            if len(range(start, stop, step)) == 0:
                raise ValueError(
                    f"Slice {key} selects no members. An OrbitFamily must "
                    f"have at least one member, so an empty selection cannot "
                    f"produce a valid family."
                )
            selection = key

        else:
            raise TypeError(
                f"OrbitFamily indices must be integers or slices, got "
                f"{type(key).__name__}. Boolean masks and index arrays are "
                f"not supported."
            )

        return self._subfamily(selection)

    def _subfamily(self, selection: slice) -> "OrbitFamily":
        """
        Build a family from a slice of this one, sharing the header.

        Routed through the public constructor rather than assembling an
        instance directly, so a subfamily is validated on exactly the same
        terms as any other family and no second construction path can drift.
        The Jacobi column is recomputed rather than sliced -- a handful of
        microseconds against a second code path to keep in step.
        """
        child = OrbitFamily(
            initial_states=self._initial_states[selection],
            periods=self._periods[selection],
            iterations=self._iterations[selection],
            final_residuals=self._final_residuals[selection],
            step_sizes=self._step_sizes[selection],
            member_indices=self._member_indices[selection],
            primary_body=self._primary_body,
            secondary_body=self._secondary_body,
            distance=self._distance,
            mu=self._mu,
            recipe=self._recipe,
            scheme=self._scheme,
            free_vars=self._free_vars,
            free_times=self._free_times,
            node_specs=None,
            closure_tol=self._closure_tol,
        )

        # Inherit the live System. No validation needed: the child's header
        # is this family's header by construction, so the checks in
        # _validate_system are provably vacuous here. This is the dominant
        # duplication path -- the stated usage pattern is "slice down, then
        # analyze the slice", and without this each subfamily would compile
        # its own base and variational integrators and push
        # System._instance_count toward its warning threshold.
        #
        # Sharing a header-level object does not violate the dense-and-total
        # rule that justified slicing: that rule governs the per-member
        # caches, which would be sparse if shared. A System has no per-member
        # structure to be sparse about.
        if self._system is not None:
            child._system = self._system

        return child

    # ========== VIEWS ==========
    def to_frame(self) -> "pd.DataFrame":
        """
        Return the member table as a pandas DataFrame, for inspection.

        A view of data already in hand, not storage: cheap, and it never
        triggers propagation, so the multiplier- and orbit-backed products
        are deliberately absent. Add them yourself if wanted, e.g.
        `frame['nu'] = family.stability_indices()` -- explicit, so the cost
        is visible where it is paid.

        Deliberately decoupled from persistence, so pandas never becomes
        load-bearing for the on-disk format. DataFrame-as-storage was
        rejected: it needs a serialization dependency (pyarrow, pytables,
        version-fragile pickle, or CSV with float round-trip concerns) and
        the header is a heterogeneous nested record that does not fit a
        DataFrame at all, forcing a sidecar file -- at which point the
        DataFrame has bought nothing npz was not already giving.

        Returns
        -------
        pd.DataFrame
            Columns x, y, z, vx, vy, vz, periods, jacobi_constants,
            iterations, final_residuals, step_sizes; indexed by
            member_indices under the name 'member', so a subfamily's frame
            still reports positions in the original march. The frame owns
            its data and is freely mutable -- it is a snapshot, and editing
            it does not touch the family.

        Raises
        ------
        ImportError
            If pandas is not installed. Imported inside the method rather
            than at module scope for that reason: pandas is not yet a Kyklos
            dependency, so functions that use it import locally.
        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "OrbitFamily.to_frame() requires pandas, which is not "
                "installed. Install pandas, or read the columns directly -- "
                "they are plain numpy arrays."
            ) from exc

        data = {
            name: self._initial_states[:, i].copy()
            for i, name in enumerate(_STATE_COLUMNS)
        }
        data['periods'] = self._periods.copy()
        data['jacobi_constants'] = self._jacobi_constants.copy()
        data['iterations'] = self._iterations.copy()
        data['final_residuals'] = self._final_residuals.copy()
        data['step_sizes'] = self._step_sizes.copy()

        index = pd.Index(self._member_indices.copy(), name='member')
        return pd.DataFrame(data, index=index)

    # ========== PLOTTING ==========
    def plot_3d(
        self,
        color_by: str = 'index',
        colorscale: str = 'Viridis',
        n_points: int | None = None,
        bodies: bool | str | Sequence[str] | None = True,
        lagrange_points: bool | str | Sequence[str] | None = True,
        retain_orbits: bool = False,
        title: str | None = None,
        renderer: str | None = None) -> go.Figure:
        """
        Plot every member of the family on one 3D figure, colored by a
        family parameter.

        Expensive in the same way as to_orbits(), which it calls: n
        integrations with the STM unless the orbits are already cached.
        Slice first to plot a subset -- `family[::5].plot_3d()`,
        `family[40:60].plot_3d()` -- which propagates only the selected
        members.

        Members that failed closure are skipped; the summarizing warning from
        the propagation pass has already reported them, and the default
        title shows how many members were drawn. Members are kept out of the
        legend -- the colorbar and the per-member hover text identify them.

        Parameters
        ----------
        color_by : {'index', 'jacobi', 'period', 'stability'}, optional
            Parameter mapped onto the colorscale. 'index' is the position in
            the original march (member_indices), so a subfamily keeps its
            parent's numbering. 'stability' uses log10 of the stability
            index. Default: 'index'.
        colorscale : str, optional
            Any Plotly named colorscale. Default: 'Viridis'.
        n_points : int, optional
            Samples per member.
            If None, uses config.DEFAULT_FAMILY_PLOT_POINTS (default: None)
        bodies : bool, str, or sequence of str, optional
            Bodies to draw, as for Trajectory.plot_3d. The automatic test
            measures the whole family. (default: True)
        lagrange_points : bool, str, or sequence of str, optional
            Lagrange points to draw, as for Trajectory.plot_3d. On by
            default here: libration-point families are nearly always shown
            with their point, and automatic mode draws only what falls in
            the family's bounding box. (default: True)
        retain_orbits : bool, optional
            Keep the reconstructed PeriodicOrbits cached afterwards. If
            False (default), orbits cached by this call are released when it
            returns; orbits that were already cached before the call are
            left alone. The multipliers survive either way.
        title : str, optional
            Figure title. If None, names the recipe and the member count.
        renderer : str, optional
            Plotly renderer to display with. If None, the figure is not
            shown and config.RENDERER becomes the Plotly default.
            (default: None)

        Returns
        -------
        go.Figure
            The new figure.

        Raises
        ------
        ValueError
            If color_by is not one of the listed options. Checked before any
            propagation.
        RuntimeError
            If every member failed closure, leaving nothing to plot.
        """
        import plotly.io as pio

        # Validated before anything expensive, so a typo costs nothing.
        if color_by not in _COLOR_BY:
            raise ValueError(
                f"Unknown color_by {color_by!r}. Valid options are "
                f"{', '.join(_COLOR_BY)}."
            )
        if n_points is None:
            n_points = config.DEFAULT_FAMILY_PLOT_POINTS
        if renderer is None:
            pio.renderers.default = config.RENDERER

        # Leave the orbit cache as it was found: release it afterwards only
        # if this call is what filled it. The finally clause makes that hold
        # even if building the figure raises.
        had_orbits = self.has_orbits
        try:
            orbits = self.to_orbits()
            fig = self._build_family_figure(
                orbits, color_by, colorscale, n_points,
                bodies, lagrange_points, title,
            )
        finally:
            if not retain_orbits and not had_orbits:
                self.clear_orbit_cache()

        if renderer:
            fig.show(renderer=renderer)

        return fig

    def _build_family_figure(
        self,
        orbits: tuple,
        color_by: str,
        colorscale: str,
        n_points: int,
        bodies: bool | str | Sequence[str] | None,
        lagrange_points: bool | str | Sequence[str] | None,
        title: str | None) -> go.Figure:
        """
        Assemble the family figure from reconstructed orbits.

        Split from plot_3d so the cache bookkeeping there stays readable;
        this function never touches the caches beyond reading multipliers
        that to_orbits() has already filled.

        Returns
        -------
        go.Figure

        Raises
        ------
        RuntimeError
            If every entry of orbits is None.
        """
        members = [(k, orbit) for k, orbit in enumerate(orbits)
                   if orbit is not None]
        if not members:
            raise RuntimeError(
                f"All {self.n} family members failed periodicity closure, "
                f"so there is nothing to plot. See .closure_failures for "
                f"per-member residuals."
            )
        keep = [k for k, _ in members]

        # Free here: to_orbits() filled the multiplier cache on its way.
        nu = self.stability_indices()

        values = self._color_values(color_by, nu)[keep]
        vmin, vmax = float(values.min()), float(values.max())
        if vmax > vmin:
            fractions = (values - vmin) / (vmax - vmin)
        else:
            # One member, or a column that is constant across the plotted
            # members: color everything mid-scale and give the colorbar a
            # nonzero span to draw.
            fractions = np.full(len(keep), 0.5)
            vmin, vmax = vmin - 0.5, vmax + 0.5
        colors = sample_colorscale(colorscale, fractions.tolist())

        fig = go.Figure()

        for (k, orbit), color in zip(members, colors):
            idx = int(self._member_indices[k])
            # Adjacent string literals are joined at compile time, and only
            # the f-prefixed pieces are formatted by Python. The plain pieces
            # carry Plotly's %{x} placeholders through untouched, so no
            # brace doubling is needed.
            hover = (
                f"member {idx}<br>"
                f"C = {self._jacobi_constants[k]:.8f}<br>"
                f"T = {self._periods[k]:.8f}<br>"
                f"nu = {nu[k]:.6g}<br>"
                "x: %{x:.6g}<br>y: %{y:.6g}<br>z: %{z:.6g}"
                "<extra></extra>"
            )
            orbit.trajectory.add_to_plot(
                fig, n_points=n_points, color=color, show_nodes=False,
                traj_name=f"member {idx}", hovertemplate=hover,
                legendgroup='family', showlegend=False,
            )

        # Colorbar carrier: a two-point, zero-size marker trace whose color
        # array spans the value range. Marker-only, so _figure_line_positions
        # skips it and it cannot widen the automatic-visibility box.
        x0, y0, z0 = (float(c) for c in self._initial_states[keep[0], :3])
        fig.add_trace(go.Scatter3d(
            x=[x0, x0], y=[y0, y0], z=[z0, z0],
            mode='markers',
            marker=dict(
                size=0,
                color=[vmin, vmax],
                colorscale=colorscale,
                cmin=vmin,
                cmax=vmax,
                showscale=True,
                # Shortened and pinned to the bottom so it clears the legend
                colorbar=dict(
                    title=dict(text=_COLOR_BY[color_by]),
                    len=0.55,
                    y=0.0,
                    yanchor='bottom',
                ),
            ),
            name='colorbar',
            hoverinfo='skip',
            showlegend=False,
        ))

        # Every member shares the System, so any member's trajectory can
        # place bodies and Lagrange points; both measure the whole figure.
        anchor = members[0][1].trajectory
        if bodies is not None and bodies is not False:
            anchor.add_bodies(fig, bodies=bodies, n_points=n_points)
        if lagrange_points is not None and lagrange_points is not False:
            anchor.add_lagrange_points(fig, points=lagrange_points,
                                       n_points=n_points)

        if title is None:
            if len(members) < self.n:
                title = (f"{self._recipe} family: {len(members)} of "
                         f"{self.n} members")
            else:
                title = f"{self._recipe} family: {self.n} members"
        _apply_3d_layout(fig, True, title)

        return fig

    def _color_values(self, color_by: str, nu: np.ndarray) -> np.ndarray:
        """
        The per-member values plot_3d maps onto the colorscale.

        Parameters
        ----------
        color_by : str
            A key of _COLOR_BY, already validated.
        nu : np.ndarray, shape (n,)
            Stability indices, passed in rather than recomputed.

        Returns
        -------
        np.ndarray, shape (n,), dtype float
            Aligned with the member table. NaN where a member failed
            closure under 'stability'; the caller drops those rows before
            taking any min or max.
        """
        if color_by == 'index':
            return self._member_indices.astype(float)
        if color_by == 'jacobi':
            return np.asarray(self._jacobi_constants, dtype=float)
        if color_by == 'period':
            return np.asarray(self._periods, dtype=float)
        # 'stability'
        with np.errstate(invalid='ignore'):
            return np.log10(nu)

    def summary(self) -> None:
        """
        Print a human-readable summary of the family.

        Reports only what is already in hand: the member table, the header,
        and whatever the caches happen to hold. Never propagates, so calling
        it on a fresh family is free and stability figures appear only once
        a propagation pass has run.
        """
        print(f"OrbitFamily: {self._recipe!r} via {self._scheme!r}")
        print(f"  members         : {self.n}")
        if self.n > 1:
            first, last = int(self._member_indices[0]), \
                int(self._member_indices[-1])
            print(f"  march positions : {first} .. {last}")
        print(f"  system          : "
              f"{self._primary_body.name or 'primary'}-"
              f"{self._secondary_body.name or 'secondary'}, "
              f"mu = {self._mu:.10f}, L* = {self._distance:.6g} km")
        print(f"  period          : {self._periods.min():.6f} .. "
              f"{self._periods.max():.6f}")
        print(f"  Jacobi constant : {self._jacobi_constants.min():.6f} .. "
              f"{self._jacobi_constants.max():.6f}")
        if self.n > 1:
            # Step 0 is excluded from the range: it is the bootstrap's 0.0
            # in a march, or a parent-relative step in a subfamily, and
            # neither is a step this family took.
            steps = self._step_sizes[1:]
            print(f"  arclength span  : {float(steps.sum()):.6g} "
                  f"(steps {steps.min():.3g} .. {steps.max():.3g})")
        else:
            print("  arclength span  : n/a (single member)")
        print(f"  corrector       : {int(self._iterations.max())} iterations "
              f"worst, residual {self._final_residuals.max():.3e} worst")
        print(f"  closure_tol     : {self._closure_tol:.3e}")

        print(f"  system attached : {self.has_system}")
        if not self.has_multipliers:
            print("  stability       : not computed "
                  "(call floquet_multipliers() or to_orbits())")
        else:
            nu = self.stability_indices()
            finite = nu[np.isfinite(nu)]
            if finite.size:
                print(f"  stability index : {finite.min():.6g} .. "
                      f"{finite.max():.6g}")
            failed = len(self._closure_failures)
            if failed:
                worst = max(self._closure_failures, key=lambda f: f.residual)
                print(f"  closure failures: {failed} of {self.n}, worst "
                      f"{worst.residual:.3e} ({worst.ratio:.1f}x threshold)")
            else:
                print(f"  closure failures: none")
            print(f"  orbits cached   : {self.has_orbits}")

    def __repr__(self) -> str:
        """
        Compact representation. Cheap: never propagates, never compiles.
        """
        return (
            f"OrbitFamily({self._recipe!r}, n={self.n}, "
            f"T=[{self._periods.min():.6g}, {self._periods.max():.6g}], "
            f"C=[{self._jacobi_constants.min():.6g}, "
            f"{self._jacobi_constants.max():.6g}])"
        )
