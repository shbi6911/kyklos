'''Development code for an orbital trajectory handling package
Differential corrector (shooting) module
created with the assistance of Claude Opus by Anthropic'''

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Sequence, TYPE_CHECKING

import numpy as np

from .config import config
from .trajectory import (
    FreeJunctionNode,
    ImpulsiveJunctionNode,
    NullJunctionNode,
)

if TYPE_CHECKING:
    from .system import System
    from .trajectory import Trajectory


# Singular-value cutoff for the least-squares Newton step. Internal: any
# singular value below _LSTSQ_RCOND * s_max is treated as numerically null
# (declares rank deficiency). Conditioning policy is the cond_warn/cond_fail
# thresholds, applied to the retained subspace.
_LSTSQ_RCOND = 1e-13


# ========== STATE COMPONENT VOCABULARY ==========
# Canonical base-state ordering for the package: [x, y, z, vx, vy, vz].
_STATE_NAMES = ('x', 'y', 'z', 'vx', 'vy', 'vz')

_NAME_TO_INDEX = {name: i for i, name in enumerate(_STATE_NAMES)}

# Named free-variable categories. Each maps to the start-state component
# indices that are free (adjustable by the corrector). The complement of
# each set is held fixed at its initial-guess value.
#   'all'      -- full 6-dof initial state is free
#   'position' -- position free, velocity fixed
#   'velocity' -- velocity free, position fixed
#   'planar'   -- in-plane motion free (x, y, vx, vy); out-of-plane
#                 (z, vz) fixed. Useful for planar CR3BP work.
#   'none'     -- no start-state components free (e.g. fixed initial
#                 state with only junction states and/or times free)
_CATEGORY_INDICES = {
    'all':      [0, 1, 2, 3, 4, 5],
    'position': [0, 1, 2],
    'velocity': [3, 4, 5],
    'planar':   [0, 1, 3, 4],
    'none':     [],
}

# ========== INITIAL CONSTRAINTS ==========

def _parse_free_vars(free_vars: str | Sequence[str]) -> np.ndarray:
    """
    Resolve a free-variable specification to start-state component indices.

    The differential corrector adjusts a subset of the start-state
    components during iteration; this helper translates a user-facing
    specification into the canonical index array the corrector packs into
    the free-variable vector X. Junction post-states are handled
    separately (they are free by virtue of being FreeJunctionNodes), so
    this function concerns only the start state.

    Two input forms are accepted:

    1. A category string -- one of 'all', 'position', 'velocity',
       'planar', or 'none'. Case- and whitespace-insensitive.
    2. A sequence of individual component names drawn from
       ('x', 'y', 'z', 'vx', 'vy', 'vz'). Case- and whitespace-
       insensitive. An empty sequence is equivalent to 'none'.

    Integer indices are intentionally not accepted: the package
    convention is a naming convention, which keeps problem definitions
    self-documenting.

    Parameters
    ----------
    free_vars : str or sequence of str
        The free-variable specification. Either a single category string
        or a sequence of component-name strings.

    Returns
    -------
    np.ndarray
        Sorted 1-D integer array of the free start-state component
        indices, in canonical state order (ascending). The array is
        empty (shape (0,)) when no components are free. Sorting makes the
        packing order deterministic and independent of the order the user
        listed names in -- e.g. ['vy', 'x'] and ['x', 'vy'] both yield
        array([0, 4]).

    Raises
    ------
    ValueError
        If a category string is unrecognized, a component name is
        unrecognized, or a component name appears more than once.
    TypeError
        If free_vars is neither a string nor a list/tuple, or if a list
        entry is not a string.

    Notes
    -----
    Malformed input raises unconditionally rather than routing through
    the config STRICT_VALIDATION machinery. STRICT_VALIDATION governs
    physical near-continuity tolerances; a bad free_vars spec is an API
    usage error, which should fail loudly regardless of config state.

    Examples
    --------
    >>> _parse_free_vars('position')
    array([0, 1, 2])
    >>> _parse_free_vars('planar')
    array([0, 1, 3, 4])
    >>> _parse_free_vars(['x', 'z', 'vy'])     # xz-symmetric start
    array([0, 2, 4])
    >>> _parse_free_vars('none')
    array([], dtype=int64)
    """
    # --- Category string form ---
    if isinstance(free_vars, str):
        key = free_vars.strip().lower()
        if key not in _CATEGORY_INDICES:
            valid = ", ".join(repr(k) for k in _CATEGORY_INDICES)
            raise ValueError(
                f"Unknown free_vars category {free_vars!r}. "
                f"Valid categories: {valid}. Alternatively pass a list of "
                f"component names from {list(_STATE_NAMES)}."
            )
        return np.array(_CATEGORY_INDICES[key], dtype=int)

    # --- Sequence-of-names form ---
    if isinstance(free_vars, (list, tuple)):
        indices: list[int] = []
        seen: set[int] = set()
        for item in free_vars:
            if not isinstance(item, str):
                raise TypeError(
                    f"free_vars list entries must be component-name "
                    f"strings, got {type(item).__name__}: {item!r}."
                )
            name = item.strip().lower()
            if name not in _NAME_TO_INDEX:
                raise ValueError(
                    f"Unknown state component {item!r}. Valid components: "
                    f"{list(_STATE_NAMES)}."
                )
            idx = _NAME_TO_INDEX[name]
            if idx in seen:
                raise ValueError(
                    f"Duplicate state component {item!r} in free_vars."
                )
            seen.add(idx)
            indices.append(idx)
        indices.sort()
        return np.array(indices, dtype=int)

    raise TypeError(
        f"free_vars must be a category string or a list of component "
        f"names, got {type(free_vars).__name__}."
    )


# ========== TERMINAL CONSTRAINTS ==========

def _component_index(name: str) -> int:
    """Resolve a single state-component name to its index in [0, 6)."""
    if not isinstance(name, str):
        raise TypeError(
            f"Component name must be a string, got {type(name).__name__}: "
            f"{name!r}."
        )
    key = name.strip().lower()
    if key not in _NAME_TO_INDEX:
        raise ValueError(
            f"Unknown state component {name!r}. Valid components: "
            f"{list(_STATE_NAMES)}."
        )
    return _NAME_TO_INDEX[key]


def _resolve_component_names(names: Sequence[str]) -> np.ndarray:
    """Resolve a sequence of component names to sorted, unique indices."""
    idx: list[int] = []
    seen: set[int] = set()
    for name in names:
        i = _component_index(name)
        if i in seen:
            raise ValueError(f"Duplicate state component {name!r}.")
        seen.add(i)
        idx.append(i)
    idx.sort()
    return np.array(idx, dtype=int)


def _finite_diff(func, x: np.ndarray, eps_rel: float) -> np.ndarray:
    """
    Central-difference Jacobian of a vector function at x.

    The per-component step is eps_rel * max(1, |x_i|), so it behaves for
    both O(1) nondimensional states and large dimensional ones.

    Parameters
    ----------
    func : callable
        Maps a (6,) state to a (m_c,) residual.
    x : np.ndarray
        Point of evaluation, shape (6,).
    eps_rel : float
        Relative finite-difference step.

    Returns
    -------
    np.ndarray
        Approximate Jacobian, shape (m_c, 6).
    """
    x = np.asarray(x, dtype=float)
    f0 = np.atleast_1d(np.asarray(func(x), dtype=float))
    m_c = f0.size
    n = x.size
    J = np.zeros((m_c, n))
    for i in range(n):
        h = eps_rel * max(1.0, abs(x[i]))
        xp = x.copy(); xp[i] += h
        xm = x.copy(); xm[i] -= h
        fp = np.atleast_1d(np.asarray(func(xp), dtype=float))
        fm = np.atleast_1d(np.asarray(func(xm), dtype=float))
        J[:, i] = (fp - fm) / (2.0 * h)
    return J


class TerminalConstraint(ABC):
    """
    Base class for boundary conditions enforced at the final state.

    A terminal constraint declares how many residual rows it contributes
    (n_rows) and supplies those rows and their Jacobians through three
    methods:

    residual(state_tf, x0)
        The residual, following the package convention
        ``residual = actual - target`` so the constraint is met when the
        residual is zero. Shape (m_c,).
    jacobian_tf(state_tf, x0)
        d(residual)/d(state_tf), shape (m_c, 6). Defaults to a central
        finite difference of residual; structured subclasses override with
        analytic rows.
    jacobian_x0(state_tf, x0)
        d(residual)/d(x0), shape (m_c, 6). Defaults to zeros, since most
        constraints do not depend on the start state. Periodicity is the
        notable exception.

    Subclasses must implement residual; they may override either Jacobian
    method to supply analytic derivatives.

    Lifecycle
    ---------
    Before iteration the corrector calls bind(system) once. The default is
    a no-op returning self. Constraints needing dynamical parameters (e.g.
    a Jacobi-constant target needing mu) override bind to capture them,
    returning a new bound instance rather than mutating self -- this keeps
    the original a reusable, system-agnostic template.
    """

    # Relative step for the finite-difference Jacobian fallback.
    _fd_eps_rel: float = 1e-7

    @property
    @abstractmethod
    def n_rows(self) -> int:
        """
        Number of residual rows this constraint contributes.

        Declared up front, from the constraint's spec alone, so the corrector
        can size the constraint vector F and the Jacobian DF -- and check
        problem determinacy -- before the first propagation, rather than
        backing the count out of a propagated Jacobian's shape. The contract:
        residual(...) returns an array of exactly this length, and
        jacobian_tf / jacobian_x0 each have this many rows.
        """
        ...

    def bind(self, system) -> "TerminalConstraint":
        """
        Capture any System-dependent parameters; return the bound constraint.

        Called once by the corrector before iteration. The default needs
        nothing from the system and returns self unchanged.
        """
        return self

    @abstractmethod
    def residual(self, state_tf: np.ndarray, x0: np.ndarray) -> np.ndarray:
        """Constraint residual (actual - target), shape (m_c,)."""
        ...

    def jacobian_tf(self, state_tf: np.ndarray,
                    x0: np.ndarray) -> np.ndarray:
        """d(residual)/d(state_tf), shape (m_c, 6). Default: finite diff."""
        state_tf = np.asarray(state_tf, dtype=float)
        return _finite_diff(lambda s: self.residual(s, x0),
                            state_tf, self._fd_eps_rel)

    def jacobian_x0(self, state_tf: np.ndarray,
                    x0: np.ndarray) -> np.ndarray:
        """d(residual)/d(x0), shape (n_rows, 6). Default: zeros."""
        return np.zeros((self.n_rows, 6))


class TargetState(TerminalConstraint):
    """
    Drive specified final-state components to target values.

    Parameters
    ----------
    targets : dict
        Mapping from component name ('x','y','z','vx','vy','vz') to the
        desired final value. The xz-plane perpendicular-crossing terminal
        condition, for example, is
        TargetState({'y': 0.0, 'vx': 0.0, 'vz': 0.0}).

    Notes
    -----
    Each residual entry is state_tf[i] - target_i. The Jacobian w.r.t.
    state_tf is the corresponding rows of the identity (a selection
    matrix) and does not depend on x0.
    """

    def __init__(self, targets: dict):
        if not isinstance(targets, dict):
            raise TypeError(
                f"targets must be a dict mapping component names to values, "
                f"got {type(targets).__name__}."
            )
        if not targets:
            raise ValueError("TargetState requires at least one target.")
        pairs = []
        seen: set[int] = set()
        for name, value in targets.items():
            i = _component_index(name)
            if i in seen:
                raise ValueError(f"Duplicate state component {name!r}.")
            seen.add(i)
            pairs.append((i, float(value)))
        pairs.sort(key=lambda p: p[0])
        self._idx = np.array([p[0] for p in pairs], dtype=int)
        self._target = np.array([p[1] for p in pairs], dtype=float)

    @property
    def n_rows(self) -> int:
        """One residual row per targeted component."""
        return int(self._idx.size)

    def residual(self, state_tf, x0):
        state_tf = np.asarray(state_tf, dtype=float)
        return state_tf[self._idx] - self._target

    def jacobian_tf(self, state_tf, x0):
        J = np.zeros((self._idx.size, 6))
        J[np.arange(self._idx.size), self._idx] = 1.0
        return J


class Periodicity(TerminalConstraint):
    """
    Enforce periodicity: the final state equals the start state.

    Parameters
    ----------
    components : sequence of str, or None
        Component names to enforce equality on. None (default) enforces
        full-state periodicity (all six components).

    Notes
    -----
    Residual is (state_tf - x0) on the selected components. This is the one
    built-in constraint that depends on x0: jacobian_tf is +selection,
    jacobian_x0 is -selection.
    """

    def __init__(self, components: Sequence[str] | None = None):
        if components is None:
            self._idx = np.arange(6, dtype=int)
        else:
            self._idx = _resolve_component_names(components)
            if self._idx.size == 0:
                raise ValueError("Periodicity requires at least one component.")

    @property
    def n_rows(self) -> int:
        """One residual row per component enforced equal."""
        return int(self._idx.size)

    def residual(self, state_tf, x0):
        state_tf = np.asarray(state_tf, dtype=float)
        x0 = np.asarray(x0, dtype=float)
        return state_tf[self._idx] - x0[self._idx]

    def jacobian_tf(self, state_tf, x0):
        J = np.zeros((self._idx.size, 6))
        J[np.arange(self._idx.size), self._idx] = 1.0
        return J

    def jacobian_x0(self, state_tf, x0):
        J = np.zeros((self._idx.size, 6))
        J[np.arange(self._idx.size), self._idx] = -1.0
        return J


class CallableConstraint(TerminalConstraint):
    """
    Wrap user-supplied callables as a terminal constraint.

    The escape hatch for conditions the structured constraints cannot
    express (e.g. targeting a Jacobi constant or an angle).

    Parameters
    ----------
    g : callable
        Residual function g(state_tf, x0) -> array of shape (m_c,),
        following the actual-minus-target convention.
    n_rows : int
        The number of residual elements, which is also the row count of the
        Jacobian(s), whether derived or provided.  Must be explicitly provided 
        by the user.
    dg : callable, optional
        Analytic d(residual)/d(state_tf), signature (state_tf, x0) ->
        (m_c, 6). If None, a central finite difference of g is used.
    dg_dx0 : callable, optional
        Analytic d(residual)/d(x0), signature (state_tf, x0) -> (m_c, 6).
        If None, treated as zero (no dependence on the start state).
    """

    def __init__(self, g, n_rows, dg=None, dg_dx0=None):
        if not callable(g):
            raise TypeError("g must be callable.")
        # bool is a subclass of int; reject it so True/False are not silently
        # treated as 1/0 row counts (same guard style as _parse_free_times).
        if isinstance(n_rows, bool) or not isinstance(n_rows, (int, np.integer)):
            raise TypeError(
                f"n_rows must be an integer row count, got "
                f"{type(n_rows).__name__}."
            )
        n_rows = int(n_rows)
        if n_rows < 1:
            raise ValueError(
                f"n_rows must be a positive integer, got {n_rows}."
            )
        if dg is not None and not callable(dg):
            raise TypeError("dg must be callable or None.")
        if dg_dx0 is not None and not callable(dg_dx0):
            raise TypeError("dg_dx0 must be callable or None.")
        self._g = g
        self._n_rows = n_rows
        self._dg = dg
        self._dg_dx0 = dg_dx0

    @property
    def n_rows(self) -> int:
        """Residual row count, as declared at construction."""
        return self._n_rows

    def residual(self, state_tf, x0):
        state_tf = np.asarray(state_tf, dtype=float)
        x0 = np.asarray(x0, dtype=float)
        return np.atleast_1d(np.asarray(self._g(state_tf, x0), dtype=float))

    def jacobian_tf(self, state_tf, x0):
        if self._dg is None:
            return super().jacobian_tf(state_tf, x0)   # FD fallback
        state_tf = np.asarray(state_tf, dtype=float)
        x0 = np.asarray(x0, dtype=float)
        return np.atleast_2d(np.asarray(self._dg(state_tf, x0), dtype=float))

    def jacobian_x0(self, state_tf, x0):
        if self._dg_dx0 is None:
            return super().jacobian_x0(state_tf, x0)   # zeros
        state_tf = np.asarray(state_tf, dtype=float)
        x0 = np.asarray(x0, dtype=float)
        return np.atleast_2d(
            np.asarray(self._dg_dx0(state_tf, x0), dtype=float)
        )


# ========== INTERNAL SOLVE CONTEXT ==========

# ========== DEFECT-JACOBIAN STRUCTURE PLAN ==========
# The row plan and column plan describe the fixed layout of the constraint
# vector F and the defect Jacobian DF for a given shooting problem: how many
# rows each block occupies and in what order (row plan), and how the columns
# of X partition among the free ICs and free times (column plan). This is a
# function of the problem spec alone -- node types, free-var and free-time
# selections, and constraint row counts -- so it is computed once at context
# construction and never changes over a solve. Iterate-dependent quantities
# (segment STMs, S_tf, the vector field) are NOT here; the assembler builds
# them fresh each Newton step and reads this plan to know where to place them.

class _BlockKind(Enum):
    """Which method-family produces a row block, and how it maps into X.

    The assembler branches on this tag (not on isinstance) to fill a block:
    an INTERIOR_DEFECT block reads a junction node's state_defect / segment
    STM and places +I / -Phi directly; a TERMINAL block reads a constraint's
    residual / jacobian_tf and chains it through S_tf. A block's `index` is
    read relative to its kind: a junction index (into traj.junction_nodes and
    the segment STMs) for INTERIOR_DEFECT, a constraint index (into
    ctx.constraints) for TERMINAL.
    """
    INTERIOR_DEFECT = auto()
    TERMINAL = auto()


@dataclass(frozen=True)
class _RowBlock:
    """One contiguous block of rows in F / DF.

    Attributes
    ----------
    row_offset : int
        First row of this block in F / DF.
    row_count : int
        Number of rows (6 for a Phase 1 interior defect; a constraint's
        n_rows for a terminal block).
    kind : _BlockKind
        Method-family / fill-behavior tag; see _BlockKind.
    index : int
        Which object of the kind's family: 0-based junction index for
        INTERIOR_DEFECT, 0-based constraint index for TERMINAL.
    """
    row_offset: int
    row_count: int
    kind: _BlockKind
    index: int


@dataclass(frozen=True)
class _RowPlan:
    """Ordered row layout of F / DF: interior defects, then terminal blocks."""
    blocks: tuple[_RowBlock, ...]
    n_rows: int


@dataclass(frozen=True)
class _Selector:
    """Free-component selection and column placement for one IC in X.

    Attributes
    ----------
    components : tuple of int
        Ascending state-component indices (subset of 0..5) that are free for
        this IC. Its length is the IC's column count; it is the column
        selection applied to a 6-wide STM (the generalized free_idx). Phase 1:
        free_idx for the start state, all six for every junction post-state.
    col_start : int
        First column of this IC's block in X. Precomputed as a running sum of
        prior ICs' widths, so it stays correct when a junction contributes
        fewer than six columns (Phase 2), where col_start != n_fs + 6 * j.
    """
    components: tuple[int, ...]
    col_start: int

    @property
    def width(self) -> int:
        """Number of free columns this IC contributes to X."""
        return len(self.components)


@dataclass(frozen=True)
class _ColumnPlan:
    """Column layout of X: free start comps | junction posts | free times.

    selectors[0] is the start state x0; selectors[j + 1] is the post-state of
    junction node j (the IC x_{j+1}). The scalar boundaries are stored (though
    derivable by summing selector widths) because range cuts like [0:n_state]
    do not need per-IC structure and should not re-sum the selectors.
    """
    selectors: tuple[_Selector, ...]
    n_fs: int
    n_state_block: int
    n_X: int
    free_time_idx: tuple[int, ...]

    def start_span(self) -> tuple[int, int]:
        """Half-open column range [lo, hi) of the free start components."""
        s = self.selectors[0]
        return (s.col_start, s.col_start + s.width)

    def start_components(self) -> tuple[int, ...]:
        """Free start-state component indices (the classic free_idx)."""
        return self.selectors[0].components

    def junction_span(self, j: int) -> tuple[int, int]:
        """Half-open column range [lo, hi) of junction node j's post-state."""
        s = self.selectors[j + 1]
        return (s.col_start, s.col_start + s.width)

    def junction_components(self, j: int) -> tuple[int, ...]:
        """Free component indices of junction node j's post-state."""
        return self.selectors[j + 1].components

    def free_time_column(self, m: int) -> int:
        """Column of X holding free boundary-time index m.

        Raises ValueError if boundary time m is not free.
        """
        return self.n_state_block + self.free_time_idx.index(m)


def _build_row_plan(n_junction: int, constraints: Sequence) -> _RowPlan:
    """Assemble the row layout: interior defect blocks, then terminal blocks.

    Interior defects first (one block per junction, junction order), then
    terminal constraints (one block per constraint, constraint order) -- the
    single source of truth for the row ordering F and DF must share.
    """
    blocks: list[_RowBlock] = []
    offset = 0
    for j in range(n_junction):
        # Phase 1: every FreeJunctionNode contributes a full 6-row defect.
        # Phase 2 makes this per-node (3 for position-only continuity).
        blocks.append(_RowBlock(offset, 6, _BlockKind.INTERIOR_DEFECT, j))
        offset += 6
    for i, c in enumerate(constraints):
        n = int(c.n_rows)
        blocks.append(_RowBlock(offset, n, _BlockKind.TERMINAL, i))
        offset += n
    return _RowPlan(tuple(blocks), offset)


def _build_column_plan(
    free_idx: np.ndarray,
    n_seg: int,
    free_time_idx: np.ndarray,
) -> _ColumnPlan:
    """Assemble the column layout of X from the free-var / free-time spec.

    Column regions in order: free start components ([0, n_fs)), junction
    post-states (six columns each in Phase 1), then free-time columns.
    col_start accumulates as a running sum so the layout stays correct if a
    junction ever contributes fewer than six columns (Phase 2).
    """
    n_junction = n_seg - 1
    n_fs = int(free_idx.size)

    selectors: list[_Selector] = [_Selector(tuple(int(i) for i in free_idx), 0)]
    col = n_fs
    for _ in range(n_junction):
        # Phase 1: every junction post-state is fully free (all six comps).
        selectors.append(_Selector(tuple(range(6)), col))
        col += 6

    n_state_block = col
    n_X = n_state_block + int(free_time_idx.size)
    return _ColumnPlan(
        selectors=tuple(selectors),
        n_fs=n_fs,
        n_state_block=n_state_block,
        n_X=n_X,
        free_time_idx=tuple(int(m) for m in free_time_idx),
    )

@dataclass(frozen=True)
class _ShootingContext:
    """
    Internal, immutable bookkeeping for a single differential-correction
    solve.

    Built once from the initial-guess Trajectory and the user's problem
    specification, then shared read-only across the pack/unpack,
    constraint, and Jacobian routines so none of them re-derive structural
    data. This is solver-layer scaffolding, distinct from any future
    user-facing problem-definition object: it holds only what the corrector
    needs to map between a Trajectory and the flat free-variable vector X.

    Free-variable vector layout:

        X = [ start free comps | junction post-states | free times ]
              len n_free_start    len 6 * n_junction     len n_free_time

    Attributes
    ----------
    system : System
        Dynamical system used to re-propagate each iterate.
    n_seg : int
        Number of trajectory segments N. The trajectory has N + 1 boundary
        times and N - 1 interior junctions.
    free_idx : np.ndarray
        Sorted indices of the free start-state components, into [0, 6).
    ics_ref : np.ndarray
        Reference state for each initial condition (start, then junction post states) 
        shape (1 + n_junction,6). 
        Fixed start components are read from here during unpacking; free components 
        are overwritten from X.
    times_ref : np.ndarray
        Reference boundary times (N + 1,). Fixed times are read from here;
        free times are overwritten from X.
    free_time_idx : np.ndarray
        Sorted indices of the free boundary times, into [1, N]. Index 0
        (t0) is never free for an autonomous system; index N is the final
        time.
    constraints : tuple of TerminalConstraint
        Terminal boundary conditions, already bound to the system. May be
        empty (an interior-defect-only problem -- e.g. closing the gaps of
        a discontinuous guess with no terminal targeting).
    row_plan : _RowPlan
        A custom object laying out the row structure of the defect Jacobian (DF).
        the ordering encoded here is used by _assemble_F() and _assemble_DF() to
        coordinate the residual vector X and Jacobian DF rows.
    column_plan : _ColumnPlan
        A custom object detailing the column ordering of the defect Jacobian (DF).
        Contains the ordering of all free variables in the shooting problem,
        consisting of initial free variables, interior nodes, and free times.

    Notes
    -----
    Phase 1 scope: all interior junctions must be FreeJunctionNodes.
    ImpulsiveJunctionNode support (Parameterization A: free post-state with
    a 3-row position-continuity defect) is deferred to a later phase.
    """

    system: "System"
    n_seg: int
    free_idx: np.ndarray
    ics_ref: np.ndarray
    times_ref: np.ndarray
    free_time_idx: np.ndarray
    constraints: tuple
    row_plan: _RowPlan = field(init=False, repr=False)
    column_plan: _ColumnPlan = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # Take ownership of the array fields: store private, read-only
        # copies so the shared context is fully immutable and constructing
        # it never mutates caller-held arrays.
        for name in ('free_idx', 'ics_ref', 'times_ref', 'free_time_idx'):
            arr = np.array(getattr(self, name), copy=True)
            arr.flags.writeable = False
            object.__setattr__(self, name, arr)
        object.__setattr__(
            self, 'row_plan',
            _build_row_plan(self.n_seg - 1, self.constraints),
        )
        object.__setattr__(
            self, 'column_plan',
            _build_column_plan(self.free_idx, self.n_seg, self.free_time_idx),
        )

        # Consistency check for column plan and number of IC reference states
        n_ics = 1 + self.n_junction
        if self.ics_ref.shape != (n_ics, 6):
            raise ValueError(
                f"ics_ref must have shape ({n_ics}, 6) for {self.n_seg} "
                f"segment(s), got {self.ics_ref.shape}."
            )
        if len(self.column_plan.selectors) != n_ics:
            raise ValueError(
                f"column plan has {len(self.column_plan.selectors)} "
                f"selector(s) but there are {n_ics} IC(s)."
            )

    # --- derived sizes ---
    @property
    def n_free_start(self) -> int:
        """Number of free start-state components."""
        return int(self.column_plan.n_fs)

    @property
    def n_junction(self) -> int:
        """Number of interior junctions (patch points), N - 1."""
        return self.n_seg - 1

    @property
    def n_free_time(self) -> int:
        """Number of free boundary times."""
        return int(self.free_time_idx.size)

    @property
    def n_state_block(self) -> int:
        """Length of the state portion of X (start free comps + junctions)."""
        return self.column_plan.n_state_block

    @property
    def n_X(self) -> int:
        """Total length of the free-variable vector X."""
        return self.column_plan.n_X

    @property
    def determinacy(self) -> str:
        """Row/column shape: 'overdetermined', 'square', or 'underdetermined'.

        A report, not a gate. The least-squares / minimum-norm solver handles
        all three; this only surfaces the shape for diagnostics and (later)
        for continuation schemes to check against their intended determinacy.
        """
        m, n = self.row_plan.n_rows, self.column_plan.n_X
        if m > n:
            return 'overdetermined'
        if m < n:
            return 'underdetermined'
        return 'square'

    @classmethod
    def from_guess(
        cls,
        traj: "Trajectory",
        free_vars: str | Sequence[str],
        constraints: Sequence | None = None,
        free_times: Sequence[int | np.integer] | None = None,
    ) -> "_ShootingContext":
        """
        Build a context from an initial-guess Trajectory and problem spec.

        Performs all structural validation up front (node types, time-index
        ranges, constraint normalization/binding) so the iteration loop can
        assume a well-formed problem.

        Parameters
        ----------
        traj : Trajectory
            The initial-guess trajectory. Defines segment count, the
            reference start state, and the reference boundary times.
        free_vars : str or sequence of str
            Free start-state specification; see _parse_free_vars.
        constraints : sequence, optional
            Terminal constraints -- TerminalConstraint instances or 
            CallableConstraints. None or empty yields no terminal
            constraints.
        free_times : sequence of int, or None
            Boundary-time indices that are free, drawn from [1, n_seg].
            Index n_seg is the final time. None (default) fixes all times.

        Returns
        -------
        _ShootingContext

        Raises
        ------
        NotImplementedError
            If any interior junction is not a FreeJunctionNode (Phase 1
            scope).
        ValueError
            If a free-time index is out of range [1, n_seg], duplicated, or
            references t0 (index 0, always fixed).
        TypeError
            If free_times is neither None nor a list/tuple of ints, or a
            constraint is neither a TerminalConstraint nor callable.
        """
        free_idx = _parse_free_vars(free_vars)
        n_seg = traj.n_segments

        # Phase 1: only free junctions are supported.
        for k, node in enumerate(traj.junction_nodes, start=1):
            if isinstance(node, FreeJunctionNode):
                continue
            if isinstance(node, ImpulsiveJunctionNode):
                raise NotImplementedError(
                    f"Interior junction {k} is an ImpulsiveJunctionNode. "
                    f"Impulsive maneuver support (Parameterization A) is "
                    f"deferred to a later phase; Phase 1 handles "
                    f"FreeJunctionNode patch points only."
                )
            raise NotImplementedError(
                f"Interior junction {k} is a {type(node).__name__}. Phase 1 "
                f"supports FreeJunctionNode patch points only."
            )

        free_time_idx = cls._parse_free_times(free_times, n_seg)
        bound_constraints = cls._validate_constraints(constraints, traj.system)

        # No defensive copies here: __post_init__ takes ownership by copying
        # and freezing every array field, so passing fresh-or-not arrays is
        # safe and uniform.
        ic_states = [np.asarray(traj.start_node.post_state, dtype=float)]
        ic_states += [np.asarray(node.post_state, dtype=float)
              for node in traj.junction_nodes]
        ics_ref = np.vstack(ic_states)

        times_ref = np.asarray(traj.times, dtype=float)

        return cls(
            system=traj.system,
            n_seg=n_seg,
            free_idx=free_idx,
            ics_ref=ics_ref,
            times_ref=times_ref,
            free_time_idx=free_time_idx,
            constraints=bound_constraints,
        )

    @staticmethod
    def _parse_free_times(
        free_times: Sequence[int | np.integer] | None,
        n_seg: int,
    ) -> np.ndarray:
        """
        Validate and canonicalize a free-time specification.

        Returns a sorted int array of boundary-time indices in [1, n_seg].
        None or an empty sequence yields an empty array (all times fixed).
        Indices are validated against the segment count, and t0 (index 0)
        is rejected as it is always fixed.
        """
        if free_times is None:
            return np.array([], dtype=int)
        if not isinstance(free_times, (list, tuple)):
            raise TypeError(
                f"free_times must be None or a list/tuple of int indices, "
                f"got {type(free_times).__name__}."
            )
        seen: set[int] = set()
        for item in free_times:
            # bool is a subclass of int; reject it explicitly so True/False
            # are not silently treated as 1/0 time indices.
            if isinstance(item, bool) or not isinstance(item, (int, np.integer)):
                raise TypeError(
                    f"free_times entries must be integer indices, got "
                    f"{type(item).__name__}: {item!r}."
                )
            idx = int(item)
            if idx == 0:
                raise ValueError(
                    "free_times index 0 refers to the start time t0, which "
                    "is always fixed for an autonomous system."
                )
            if not 1 <= idx <= n_seg:
                raise ValueError(
                    f"free_times index {idx} out of range [1, {n_seg}] "
                    f"(index {n_seg} is the final time)."
                )
            if idx in seen:
                raise ValueError(f"Duplicate free_times index {idx}.")
            seen.add(idx)
        return np.array(sorted(seen), dtype=int)

    @staticmethod
    def _validate_constraints(constraints, system) -> tuple:
        """
        Normalize, validate, and bind the terminal constraints.

        Accepts TerminalConstraint instances and callables pre-wrapped 
        in CallableConstraint and treated as residual
        functions g(state_tf, x0). Each constraint is bound to the system
        once here (the single bind point), so the iteration loop only ever
        evaluates residuals and Jacobians.

        None or an empty sequence yields an empty tuple.
        """
        if constraints is None:
            return ()
        if not isinstance(constraints, (list, tuple)):
            raise TypeError(
                f"constraints must be None or a list/tuple, got "
                f"{type(constraints).__name__}."
            )
        bound = []
        for c in constraints:
            if isinstance(c, TerminalConstraint):
                constraint = c
            elif callable(c):
                raise TypeError(
                    f"A bare callable cannot be passed as a constraint: "
                    f"the corrector needs its residual row count up front. Wrap "
                    f"it as CallableConstraint(g, n_rows=<int>[, dg=..., "
                    f"dg_dx0=...]) and pass that instead. (Got a bare "
                    f"{type(c).__name__}.)"
                )
            else:
                raise TypeError(
                    f"Each constraint must be a TerminalConstraint, got "
                    f"{type(c).__name__}."
                )
            bound.append(constraint.bind(system))
        return tuple(bound)


# ========== PACK / UNPACK ==========

def _pack(traj: "Trajectory", ctx: _ShootingContext) -> np.ndarray:
    """
    Read the free-variable vector X out of a propagated Trajectory.

    Used once at the start of a solve to initialize X from the initial guess.
    The inverse mapping (X back to segment ICs and times) is _unpack; the two
    are inverses at the IC/time level, not the Trajectory level -- _pack reads
    a fully propagated trajectory, while _unpack produces only what is needed
    to propagate the next one.

    Which components of each IC are free is read from the column plan's
    per-IC selectors, not assumed to be all six: an IC contributes only its
    free components to X. In Phase 1 every junction post-state is fully free,
    so this reduces to taking the whole post-state; the selector form is what
    lets a partially-free junction (Phase 2) pack only its free components
    without changing this function.

    Parameters
    ----------
    traj : Trajectory
        A trajectory whose structure matches ctx (same segment count).
    ctx : _ShootingContext

    Returns
    -------
    np.ndarray
        The free-variable vector X, shape (ctx.n_X,), laid out as
        [start free comps, junction free comps, free times].

    Raises
    ------
    ValueError
        If traj's segment count does not match ctx.n_seg.
    """
    if traj.n_segments != ctx.n_seg:
        raise ValueError(
            f"Trajectory has {traj.n_segments} segment(s) but context "
            f"expects {ctx.n_seg}."
        )

    cp = ctx.column_plan

    # Start-state free components (selector 0).
    start_state = np.asarray(traj.start_node.post_state, dtype=float)
    start_free = start_state[list(cp.start_components())]

    # Each junction post-state contributes only its free components. The
    # list() cast is required: a tuple index on a 1D array is read as
    # multidimensional indexing, whereas a list is fancy (component) indexing.
    junction_blocks = [
        np.asarray(node.post_state, dtype=float)[list(cp.junction_components(j))]
        for j, node in enumerate(traj.junction_nodes)
    ]

    times_arr = np.asarray(traj.times, dtype=float)
    free_times = times_arr[ctx.free_time_idx]

    return np.concatenate([start_free, *junction_blocks, free_times])


def _unpack(
X: np.ndarray,
    ctx: _ShootingContext,
) -> tuple[list[np.ndarray], np.ndarray]:
    """
    Map a free-variable vector X to segment initial conditions and times.

    Integrator-free: this only reshuffles numbers into the form the
    propagator needs. It is the inverse of _pack at the IC/time level --
    _pack reads free components out of a propagated trajectory, _unpack
    scatters them back into full states ready to re-propagate.

    Every IC is reconstructed by the same operation, uniform across the start
    state and every junction post: seed the full state from the IC's
    reference row (ics_ref), then overwrite its free components with the
    corresponding slice of X. Which components are free, and where their
    columns sit in X, come from the column plan's per-IC selectors -- so a
    partially-free IC (Phase 2) scatters only its free components with no
    change here, and fixed components keep their reference value.

    Parameters
    ----------
    X : np.ndarray
        The free-variable vector, shape (ctx.n_X,).
    ctx : _ShootingContext

    Returns
    -------
    ics : list of np.ndarray
        One full (6,) initial condition per segment, in segment order.
    times : np.ndarray
        Full boundary-time vector, shape (ctx.n_seg + 1,); free times taken
        from X, fixed times from the reference.

    Raises
    ------
    ValueError
        If X does not have shape (ctx.n_X,).
    """
    X = np.asarray(X, dtype=float)
    if X.shape != (ctx.n_X,):
        raise ValueError(f"X must have shape ({ctx.n_X},), got {X.shape}.")

     # Reconstruct each IC uniformly, ics_ref rows and column-plan selectors are
    # index-aligned (guaranteed at context construction), so zip pairs each
    # reference state with the selector saying which of its components are
    # free and where they live in X. list(...) forces fancy (per-component)
    # indexing; a raw tuple would be read as multidimensional.
    ics = []
    for ref, sel in zip(ctx.ics_ref, ctx.column_plan.selectors):
        full = ref.copy()
        full[list(sel.components)] = X[sel.col_start:sel.col_start + sel.width]
        ics.append(full)

    # Free boundary times are the tail of X; fixed times keep their reference.
    times = ctx.times_ref.copy()
    times[ctx.free_time_idx] = X[ctx.n_state_block:]

    return ics, times


# ========== CONSTRAINT VECTOR ==========

def _assemble_F(traj: "Trajectory", ctx: _ShootingContext) -> np.ndarray:
    """
    Assemble the constraint vector F for a propagated iterate.

    F stacks two blocks:

    1. Interior defects -- the state discontinuity at each junction, taken
       as node.state_defect (post - pre). In Phase 1 every junction is a
       FreeJunctionNode contributing all 6 components, so this block has
       6 * (n_seg - 1) rows, ordered by junction.
    2. Terminal residuals -- each constraint's residual(state_tf, x0),
       concatenated in constraint order. state_tf is the final propagated
       state (end_node.pre_state); x0 is the current start state.

    Both blocks follow the convention residual = actual - target, so the
    problem is solved when F is (near) zero.

    Parameters
    ----------
    traj : Trajectory
        A propagated iterate, built from this context via unpack +
        propagate.
    ctx : _ShootingContext

    Returns
    -------
    np.ndarray
        Constraint vector, shape (m,), where
        m = 6 * (n_seg - 1) + sum of constraint residual lengths. Shape
        (0,) only if there are neither junctions nor constraints.
    """
    blocks: list[np.ndarray] = []

    # Interior defects: post - pre at each junction.
    for k, node in enumerate(traj.junction_nodes, start=1):
        defect = node.state_defect
        if defect is None:
            raise RuntimeError(
                f"Junction {k} has an undefined state defect; expected a "
                f"propagated FreeJunctionNode with both states present."
            )
        blocks.append(np.asarray(defect, dtype=float))

    # Terminal residuals: actual - target at the final state.
    if ctx.constraints:
        state_tf = np.asarray(traj.end_node.pre_state, dtype=float)
        x0 = np.asarray(traj.start_node.post_state, dtype=float)
        for c in ctx.constraints:
            r = np.atleast_1d(
                np.asarray(c.residual(state_tf, x0), dtype=float)
            )
            blocks.append(r)

    if not blocks:
        return np.array([], dtype=float)
    return np.concatenate(blocks)


# ========== CONSTRAINT JACOBIAN ==========

def _assemble_DF(traj: "Trajectory", ctx: _ShootingContext) -> np.ndarray:
    """
    Assemble the constraint Jacobian DF for a propagated iterate.

    DF = d(F)/d(X), with rows matching _assemble_F (interior defects then
    terminal residuals) and columns matching the X layout (start free
    components, junction post-states, then free times).

    State columns
    -------------
    Interior block (Phase 1, all free junctions). Junction i has defect
    F_i = x_{i+1} - phi_i(x_i):
        dF_i/dx_{i+1} = +I            (the junction's own post-state)
        dF_i/dx_i     = -Phi_i        (Phi_i = segment_terminal_stm(i))
    For i = 0, x_0 is the start state, so -Phi_0 is restricted to the free
    columns.

    Terminal block. Each constraint contributes
        dr/dX = J_tf @ d(state_tf)/dX + J_x0 @ d(x0)/dX,
    where the final state depends on the last segment's IC through
    Phi_{N-1}. For single-shooting Periodicity this reduces to Phi - I.

    Free-time columns
    -----------------
    Assuming autonomous dynamics, the endpoint e_i of segment i satisfies
    d(e_i)/d(t_{i+1}) = +f(e_i) and d(e_i)/d(t_i) = -f(e_i), where f is the
    vector field. A free boundary time t_m is the end of segment m-1 and
    the start of segment m, so it contributes:
        - to defect F_{m-1}:  -f(e_{m-1})         (1 <= m <= N-1)
        - to defect F_m:      +f(e_m)             (1 <= m <= N-2)
        - to the terminal rows: J_tf @ (+/- f(state_tf)) when t_m moves the
          final state, i.e. +f for the final time (m = N), -f for the last
          junction time (m = N-1).
    The vector field is evaluated at all segment endpoints in one batched
    System.vector_field call.

    Parameters
    ----------
    traj : Trajectory
        A propagated iterate built from this context.
    ctx : _ShootingContext

    Returns
    -------
    np.ndarray
        Jacobian, shape (m, n_X), m == len(F).
    """
    n_seg = ctx.n_seg
    n_fs = ctx.n_free_start
    n_state = ctx.n_state_block
    free_idx = ctx.free_idx
    m_interior = 6 * ctx.n_junction

    state_tf = np.asarray(traj.end_node.pre_state, dtype=float)
    x0 = np.asarray(traj.start_node.post_state, dtype=float)

    # Terminal Jacobian pieces (and terminal row count).
    term_Jtf, term_Jx0 = [], []
    m_terminal = 0
    for c in ctx.constraints:
        Jtf = np.atleast_2d(np.asarray(c.jacobian_tf(state_tf, x0), float))
        Jx0 = np.atleast_2d(np.asarray(c.jacobian_x0(state_tf, x0), float))
        term_Jtf.append(Jtf)
        term_Jx0.append(Jx0)
        m_terminal += Jtf.shape[0]

    DF = np.zeros((m_interior + m_terminal, ctx.n_X))

    # --- interior block (state columns): F_i = x_{i+1} - phi_i(x_i) ---
    for i in range(ctx.n_junction):
        rows = slice(6 * i, 6 * i + 6)
        cj = n_fs + 6 * i                       # x_{i+1} = junction block i
        DF[rows, cj:cj + 6] += np.eye(6)
        Phi_i = np.asarray(traj.segment_terminal_stm(i), dtype=float)
        if i == 0:
            DF[rows, 0:n_fs] += -Phi_i[:, free_idx]
        else:
            cprev = n_fs + 6 * (i - 1)
            DF[rows, cprev:cprev + 6] += -Phi_i

    # --- terminal block (state columns) ---
    if ctx.constraints:
        Phi_last = np.asarray(
            traj.segment_terminal_stm(n_seg - 1), dtype=float
        )
        S_tf = np.zeros((6, n_state))           # d(state_tf)/d(state cols)
        if n_seg == 1:
            S_tf[:, 0:n_fs] = Phi_last[:, free_idx]
        else:
            c_last = n_fs + 6 * (n_seg - 2)
            S_tf[:, c_last:c_last + 6] = Phi_last
        row = m_interior
        for Jtf, Jx0 in zip(term_Jtf, term_Jx0):
            mc = Jtf.shape[0]
            rows = slice(row, row + mc)
            DF[rows, 0:n_state] += Jtf @ S_tf
            DF[rows, 0:n_fs] += Jx0[:, free_idx]       # direct x0 dependence
            row += mc

    # --- free-time columns ---
    if ctx.n_free_time > 0:
        # Endpoint of every segment: e_i = pre of junction i (i < N-1), and
        # the final state for i = N-1. Evaluate f at all endpoints at once.
        endpoints = np.empty((6, n_seg))
        for i in range(n_seg - 1):
            endpoints[:, i] = traj.junction_nodes[i].pre_state
        endpoints[:, n_seg - 1] = state_tf
        f_vals = np.asarray(ctx.system.vector_field(endpoints), dtype=float)
        if f_vals.shape != (6, n_seg):
            f_vals = f_vals.reshape(6, n_seg)

        for p, m in enumerate(ctx.free_time_idx):
            col = n_state + p
            # interior: t_m as the end of segment m-1
            if 1 <= m <= n_seg - 1:
                r0 = 6 * (m - 1)
                DF[r0:r0 + 6, col] += -f_vals[:, m - 1]
            # interior: t_m as the start of segment m
            if 1 <= m <= n_seg - 2:
                r0 = 6 * m
                DF[r0:r0 + 6, col] += f_vals[:, m]
            # terminal: t_m moves the final state (m = N or N - 1)
            if m == n_seg or m == n_seg - 1:
                sign = 1.0 if m == n_seg else -1.0
                f_tf = f_vals[:, n_seg - 1]
                row = m_interior
                for Jtf in term_Jtf:
                    mc = Jtf.shape[0]
                    DF[row:row + mc, col] += sign * (Jtf @ f_tf)
                    row += mc

    return DF


# ========== SHOOTER RESULT ==========

@dataclass(repr=False)
class ShooterResult:
    """
    Outcome of a differential-correction solve.

    Always returned by DifferentialCorrector.solve. The converged (or last)
    trajectory is always available as `.trajectory`, so the common case is a
    one-liner: `traj = corrector.solve(...).trajectory`. Convergence status
    and the final residual are always present too, since a shooting solve can
    fail and a bare trajectory would hide that.

    Attributes
    ----------
    trajectory : Trajectory or None
        The converged trajectory on success (interior junctions converted to
        NullJunctionNodes at the solver tolerance), or the last iterate on
        non-convergence. None only if the very first propagation failed.
    converged : bool
        Whether ||F|| fell below the solver tolerance.
    iterations : int
        Number of Newton steps taken.
    final_residual : float
        The 2-norm of the constraint vector at the last evaluated iterate.
    abort_reason : str or None
        Set when the solve stopped on a failure (propagation error,
        non-finite residual, or condition number above cond_fail). None
        otherwise, including ordinary non-convergence by budget.
    diagnostics : dict or None
        Per-iteration residual and condition-number history, final rank, and
        abort reason. Populated only when solve(diagnostics=True).
    iterates : list of Trajectory or None
        The trajectory at each evaluated iterate. Populated only when
        solve(iterates=True). These are the raw propagated iterates, with
        FreeJunctionNodes intact (the Null conversion applies to the final
        `.trajectory` only).
    """

    trajectory: "Trajectory | None"
    converged: bool
    iterations: int
    final_residual: float
    abort_reason: str | None = None
    diagnostics: dict | None = None
    iterates: list | None = None

    def __repr__(self) -> str:
        status = "converged" if self.converged else "NOT converged"
        return (f"ShooterResult({status}, iterations={self.iterations}, "
                f"final_residual={self.final_residual:.3e})")


# ========== DIFFERENTIAL CORRECTOR ==========

class DifferentialCorrector:
    """
    System-agnostic differential corrector (shooting method).

    A single object handles both single and multiple shooting -- the
    distinction is just how many segments the initial-guess Trajectory has.
    It operates on a Trajectory initial guess plus a problem specification
    (free start-state components, terminal constraints, free times) and
    iterates a minimum-norm Newton scheme until the constraint vector F is
    driven below tol or a step/condition budget is hit.

    Solver configuration is set once at construction and reused across
    solves. None-valued arguments draw their default from KyklosConfig at
    construction time.

    Parameters
    ----------
    tol : float or None
        Convergence tolerance on the 2-norm of the constraint vector.
        Default config.SHOOTER_TOL.
    max_iter : int or None
        Maximum number of Newton steps. Default config.SHOOTER_MAX_ITER.
    cond_warn : float or None
        Warn when the Jacobian condition number exceeds this. Default
        config.SHOOTER_COND_WARN.
    cond_fail : float or None
        Abort when the condition number exceeds this. Default
        config.SHOOTER_COND_FAIL.
    """

    def __init__(self, tol: float | None = None,
                 max_iter: int | np.integer | None = None,
                 cond_warn: float | None = None,
                 cond_fail: float | None = None):
        self.tol = config.SHOOTER_TOL if tol is None else float(tol)
        self.max_iter = (config.SHOOTER_MAX_ITER if max_iter is None
                         else int(max_iter))
        self.cond_warn = (config.SHOOTER_COND_WARN if cond_warn is None
                          else float(cond_warn))
        self.cond_fail = (config.SHOOTER_COND_FAIL if cond_fail is None
                          else float(cond_fail))

    def solve(self, traj: "Trajectory",
              free_vars: str | Sequence[str],
              constraints: Sequence | None = None,
              free_times: Sequence[int | np.integer] | None = None,
              diagnostics: bool = False,
              iterates: bool = False) -> ShooterResult:
        """
        Correct an initial-guess trajectory to satisfy the constraints.

        Builds the internal solve context from the guess and the problem
        specification, runs the minimum-norm Newton iteration, and on success
        converts the converged interior FreeJunctionNodes into
        NullJunctionNodes at this corrector's tolerance.

        Parameters
        ----------
        traj : Trajectory
            Initial guess. Its segment count, start state, boundary times,
            and System define the problem. Interior junctions must be
            FreeJunctionNodes (Phase 1).
        free_vars : str or sequence of str
            Free start-state components: a category ('all', 'position',
            'velocity', 'planar', 'none') or a list of component names.
        constraints : sequence, optional
            Terminal constraints -- TerminalConstraint instances or 
            CallableConstraints. May be omitted for a
            pure continuity (interior-defect-only) problem.
        free_times : sequence of int, optional
            Boundary-time indices that are free, in [1, n_seg] (index n_seg
            is the final time). Default: all times fixed.
        diagnostics : bool, default False
            If True, populate result.diagnostics.
        iterates : bool, default False
            If True, populate result.iterates.

        Returns
        -------
        ShooterResult
        """
        ctx = _ShootingContext.from_guess(traj, free_vars, constraints,
                                          free_times)
        raw = self._run(ctx, traj,
                        store_diagnostics=diagnostics,
                        store_iterates=iterates)

        out_traj = raw['trajectory']
        if raw['converged'] and out_traj is not None:
            out_traj = self._finalize(out_traj)

        return ShooterResult(
            trajectory=out_traj,
            converged=raw['converged'],
            iterations=raw['iterations'],
            final_residual=raw['final_residual'],
            abort_reason=raw['abort_reason'],
            diagnostics=raw['diagnostics'],
            iterates=raw['iterates'],
        )

    def _finalize(self, traj: "Trajectory") -> "Trajectory":
        """
        Convert converged interior FreeJunctionNodes to NullJunctionNodes at
        this corrector's tolerance, reusing the converged outputs.

        A converged junction has |defect| < tol, so the NullJunctionNode
        (constructed with tol=self.tol) validates successfully and records the
        standard it was closed to. Non-Free junctions are passed through
        unchanged.
        """
        new_nodes = []
        for node in traj.junction_nodes:
            if isinstance(node, FreeJunctionNode):
                new_nodes.append(NullJunctionNode(
                    node.time, node.pre_state, node.post_state, tol=self.tol))
            else:
                new_nodes.append(node)
        return traj.with_junction_nodes(new_nodes)

    def _run(self, ctx: "_ShootingContext", guess: "Trajectory",
             store_diagnostics: bool = False,
             store_iterates: bool = False) -> dict:
        """
        Run the minimum-norm Newton iteration.

        Internal engine called by solve(). Returns a dict of raw results
        (trajectory, converged, iterations, final_residual, abort_reason,
        and optional diagnostics/iterates) which solve() maps to a
        ShooterResult.

        The loop evaluates the residual at the top of each pass, so every X
        it produces is propagated and checked before being reported, and the
        Jacobian is never assembled on the iterate where convergence is
        detected. Propagation failures, non-finite residuals, and a
        condition number above cond_fail each abort the iteration gracefully
        with a recorded reason rather than raising.
        """
        X = _pack(guess, ctx)
        converged = False
        iterations = 0
        abort_reason = None
        res = np.inf
        traj = None
        rank_seen = None
        iterates = [] if store_iterates else None
        res_history = [] if store_diagnostics else None
        cond_history = [] if store_diagnostics else None
        warned_rank = False
        warned_cond = False

        while True:
            # Propagate the current iterate (wrapped: a wild step can blow
            # up the integrator).
            try:
                ics, times = _unpack(X, ctx)
                traj = ctx.system.propagate(ics, times, with_stm=True)
            except Exception as exc:
                abort_reason = (f"propagation failed at iteration "
                                f"{iterations}: {exc}")
                break

            F = _assemble_F(traj, ctx)
            if not np.all(np.isfinite(F)):
                abort_reason = (f"non-finite constraint vector at iteration "
                                f"{iterations}")
                break
            res = float(np.linalg.norm(F))

            if store_iterates:
                iterates.append(traj)
            if store_diagnostics:
                res_history.append(res)

            if res < self.tol:
                converged = True
                break
            if iterations >= self.max_iter:
                break

            DF = _assemble_DF(traj, ctx)
            m, n = DF.shape
            if iterations == 0 and m > n:
                warnings.warn(
                    f"Shooting problem is overdetermined ({m} constraints, "
                    f"{n} free variables); a least-squares step will be used.",
                    stacklevel=2)

            dX, _, rank, svals = np.linalg.lstsq(DF, -F, rcond=_LSTSQ_RCOND)
            rank_seen = int(rank)

            full_rank = min(m, n)
            if rank < full_rank and not warned_rank:
                warnings.warn(
                    f"Jacobian is rank-deficient (rank {rank} < {full_rank}); "
                    f"constraints may not be independent. Proceeding with the "
                    f"minimum-norm step.", stacklevel=2)
                warned_rank = True

            # Condition number over the retained (non-truncated) subspace,
            # so rank-deficient directions warn rather than trip cond_fail.
            cond = float(svals[0] / svals[rank - 1]) if rank > 0 else np.inf
            if store_diagnostics:
                cond_history.append(cond)

            if cond > self.cond_fail:
                abort_reason = (
                    f"Jacobian condition number {cond:.3e} exceeded cond_fail "
                    f"{self.cond_fail:.3e} at iteration {iterations}")
                break
            if cond > self.cond_warn and not warned_cond:
                warnings.warn(
                    f"Jacobian condition number {cond:.3e} exceeded cond_warn "
                    f"{self.cond_warn:.3e}.", stacklevel=2)
                warned_cond = True

            X = X + dX
            iterations += 1

        diagnostics = None
        if store_diagnostics:
            diagnostics = {
                'residual_history': res_history,
                'condition_history': cond_history,
                'final_rank': rank_seen,
                'abort_reason': abort_reason,
            }

        return {
            'trajectory': traj,
            'converged': converged,
            'iterations': iterations,
            'final_residual': res,
            'abort_reason': abort_reason,
            'diagnostics': diagnostics,
            'iterates': iterates,
        }
