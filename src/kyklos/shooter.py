'''Development code for an orbital trajectory handling package
Differential corrector (shooting) module
created with the assistance of Claude Opus by Anthropic'''

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, InitVar
from enum import Enum, auto
from typing import Sequence, TYPE_CHECKING

import numpy as np

from .config import config
from .utils import validation_error
from .trajectory import (
    FreeJunctionNode,
    ImpulsiveJunctionNode,
    NullJunctionNode,
)

if TYPE_CHECKING:
    from .system import System
    from .trajectory import Trajectory

# ========== CONSTANTS AND ENUMS ==========

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

class _BlockKind(Enum):
    """Which method-family produces a row block, and how it maps into X.

    The assembler branches on this tag (not on isinstance) to fill a block:
    an INTERIOR_DEFECT block reads a junction node's state_defect / segment
    STM and places +I / -Phi directly; a TERMINAL block reads a constraint's
    residual / jacobian_tf and chains it through S_tf; an X_SPACE block reads
    a constraint's residual / jacobian_X, which are already functions of the
    free-variable vector X, and places the (n_rows, n_X) Jacobian directly
    across all columns with no chaining. A block's `index` is read relative to
    its kind: a junction index (into traj.junction_nodes and the segment STMs)
    for INTERIOR_DEFECT, a constraint index (into ctx.constraints) for both
    TERMINAL and X_SPACE.

    TERMINAL and X_SPACE correspond one-to-one with ConstraintSpace.TERMINAL
    and ConstraintSpace.FREEVAR: the row-plan builder reads each constraint's
    `space` once and records the matching block kind, so the assembler never
    re-inspects the constraint's type per iterate.
    """
    INTERIOR_DEFECT = auto()
    TERMINAL = auto()
    X_SPACE = auto()

class _NodeTarget(Enum):
    """Converged output node type a junction role resolves to.

    Read by DifferentialCorrector._finalize to decide which JunctionNode
    subclass a converged FreeJunctionNode is rebuilt as. This is distinct
    from _BlockKind: _BlockKind drives assembly (how a row block is filled),
    _NodeTarget drives finalization (what the junction becomes once the solve
    has converged). A junction is a FreeJunctionNode throughout the solve
    regardless of this tag; the tag only matters at the very end.

    Members
    -------
    NULL
        Continuous junction -> NullJunctionNode. All six components are driven
        continuous, so post == pre at convergence.
    IMPULSIVE
        Position-continuous velocity jump -> ImpulsiveJunctionNode. Position
        is driven continuous; velocity is free to differ, and that difference
        is the maneuver's delta_v.
    FREE
        Deliberately unlabeled / unrecognized continuity pattern ->
        FreeJunctionNode. The converged junction keeps whatever continuity was
        enforced but is not claimed to be any standard physical maneuver type.
    """
    NULL = auto()
    IMPULSIVE = auto()
    FREE = auto()

class ConstraintSpace(Enum):
    """Which variables a constraint's residual and Jacobian are functions of.

    This is the constraint author's declaration of what the residual depends
    on, and it selects how the assembler places the constraint's Jacobian into
    DF. It is fixed per family, not per instance:

    TERMINAL
        residual(state_tf, x0): a function of the final propagated state (and,
        rarely, the start state). Its state-space Jacobian is chained through
        the accumulated final-state sensitivity S_tf, and it also participates
        in the free-time pass via the vector field at the final state.
    FREEVAR
        residual(X): a function of the free-variable vector X directly. Its
        Jacobian is already (n_rows, n_X) in the full X column ordering and is
        placed straight into DF with no chaining; it does not participate in
        the free-time pass, because any free-time-column entries are already
        part of that (n_rows, n_X) block.

    The row-plan builder reads `space` once at context construction and records
    the matching _BlockKind (TERMINAL or X_SPACE); the per-iterate assembler
    dispatches on the block kind and never re-inspects the constraint.
    """
    TERMINAL = auto()
    FREEVAR = auto()

# String spellings accepted at the API boundary (NodeSpec.custom(becomes=...)),
# mapped to the internal enum. Kept as role-ish words rather than node class
# names so the spec reads in the vocabulary the rest of the package uses.
_BECOMES_FROM_STR = {
    'continuous': _NodeTarget.NULL,
    'impulsive':  _NodeTarget.IMPULSIVE,
    'free':       _NodeTarget.FREE,
}

# Canonical component tuples, named once so the roles and the inference table
# are not sprinkled with literals.
_ALL_COMPONENTS = tuple(range(6))
_POSITION_COMPONENTS = (0, 1, 2)


# ========== INITIAL CONSTRAINTS ==========

def _parse_free_vars(free_vars: str | Sequence[str]) -> np.ndarray:
    """
    Resolve a free-variable specification to start-state component indices.

    The differential corrector adjusts a subset of the start-state
    components during iteration; this helper translates a user-facing
    specification into the canonical index array the corrector packs into
    the free-variable vector X. Junction post-states are also handled by this
    function, but that specification is routed through the node_spec input.
    (See NodeSpec, esp. NodeSpec.custom())

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


# ========== NON-INTERIOR DEFECT CONSTRAINTS ==========

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


class Constraint(ABC):
    """Root of the constraint hierarchy: the family-agnostic contract.

    Everything universal to a constraint lives here -- how many residual rows
    it contributes, how it binds to a System, and which variable space its
    residual is a function of. Everything that depends on that space -- the
    residual and its Jacobian, whose *arguments* differ by space -- lives on
    the family bases TerminalConstraint and FreeVarConstraint, because their
    signatures cannot be stated identically at this level.

    Subclasses do not inherit from Constraint directly; they inherit from one
    of the two family bases, each of which fixes `space` and declares the
    space-specific residual and Jacobian methods.

    Attributes
    ----------
    space : ConstraintSpace
        Set by each family base (not here). The bare annotation below documents
        the requirement without creating an attribute: a concrete constraint
        that somehow lacks a space fails the row-plan builder's `space` read,
        which is the failure we want.

    Lifecycle
    ---------
    Before iteration the corrector calls bind(system) once, at a single bind
    point, for every constraint regardless of family. The default is a no-op
    returning self. Constraints needing dynamical parameters (e.g. a Jacobi-
    constant target needing mu) override bind to capture them, returning a new
    bound instance rather than mutating self -- this keeps the original a
    reusable, system-agnostic template.
    """

    space: ConstraintSpace

    @property
    @abstractmethod
    def n_rows(self) -> int:
        """
        Number of residual rows this constraint contributes.

        Declared up front, from the constraint's spec alone, so the corrector
        can size the constraint vector F and the Jacobian DF -- and check
        problem determinacy -- before the first propagation, rather than
        backing the count out of a propagated Jacobian's shape. The contract:
        residual(...) returns an array of exactly this length, and the family's
        Jacobian method has this many rows.
        """
        ...

    def bind(self, system) -> "Constraint":
        """
        Capture any System-dependent parameters; return the bound constraint.

        Called once by the corrector before iteration. The default needs
        nothing from the system and returns self unchanged.
        """
        return self



class TerminalConstraint(Constraint):
    """
    Base class for boundary conditions enforced at the final state.

    A terminal constraint's residual is a function of the final propagated
    state (and, for the rare case of Periodicity, the start state). It supplies
    its rows and their Jacobians through three methods:

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
    method to supply analytic derivatives. n_rows and bind are inherited from
    Constraint.

    The Jacobian is split into jacobian_tf and jacobian_x0 because a terminal
    constraint has two distinct upstream dependencies that the assembler places
    differently: the state_tf dependence is chained through S_tf, while the x0
    dependence is scattered directly into the start columns. FreeVarConstraint,
    by contrast, has a single dependence (on X) placed one way, so it exposes a
    single Jacobian method.
    """

    space = ConstraintSpace.TERMINAL

    # Relative step for the finite-difference Jacobian fallback.
    _fd_eps_rel: float = 1e-7

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


class FreeVarConstraint(Constraint):
    """
    Base class for constraints whose residual is a function of X directly.

    Where a terminal constraint sees the propagated final state, a free-
    variable constraint sees the free-variable vector X = [start free comps,
    junction free comps, free times] and constrains it directly. The pseudo-
    arclength continuation condition is the motivating case: its residual is
    (X - X_prev) . t_hat - ds, a pure function of X.

    Subclasses implement two methods:

    residual(X)
        The residual, following ``residual = actual - target``. Shape (m_c,).
    jacobian_X(X)
        d(residual)/d(X), shape (m_c, n_X), given in the full X column ordering.
        The assembler places this block directly into DF's rows across all
        columns, with no chaining -- so the constraint owns the complete column
        layout of its own rows, including any free-time columns.

    There is no finite-difference default and no jacobian split: a free-
    variable residual has a single dependence (on X), so a single analytic
    Jacobian method is both necessary and sufficient. n_rows and bind are
    inherited from Constraint. A free-variable constraint is sized by the
    reference data handed to its constructor (e.g. X_prev / t_hat / ds), so it
    never needs to be told n_X separately -- n_X is implicit in that data.
    """

    space = ConstraintSpace.FREEVAR

    @abstractmethod
    def residual(self, X: np.ndarray) -> np.ndarray:
        """Constraint residual (actual - target), shape (m_c,)."""
        ...

    @abstractmethod
    def jacobian_X(self, X: np.ndarray) -> np.ndarray:
        """d(residual)/d(X), shape (m_c, n_X), in the full X column ordering."""
        ...


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


class PseudoArclength(FreeVarConstraint):
    """
    Pseudo-arclength continuation closing condition on the free variables.

    Adds the single scalar row that squares an otherwise underdetermined-by-one
    corrector system along a solution family, and does so fold-safely. The
    residual is the affine condition

        g(X) = t_hat . (X - X_prev) - ds

    which asks the next member to sit a distance ``ds`` from the previous
    converged member ``X_prev`` along the family tangent ``t_hat``. Because the
    constraint plane is normal to the tangent (not to a coordinate axis, as in
    natural-parameter continuation), it stays transverse to the family through
    folds, where a coordinate pin would go singular.

    This is a free-variable constraint: its residual and Jacobian are functions
    of the packed unknown vector X directly, never of the propagated state, so
    the assembler places its row straight into the X columns with no STM
    chaining. The Jacobian is constant over a solve -- g is affine in X, so
    ``dg/dX = t_hat`` regardless of the iterate -- which is why it carries no
    finite-difference fallback.

    The reference data is baked in at construction: the continuation engine
    builds a fresh instance each step with that step's previous member,
    tangent, and step size. The tangent must be a unit vector (the fold-safety
    of the bordered system relies on ``t_hat . t_hat == 1``); a non-unit
    tangent is a caller error and is rejected.

    Parameters
    ----------
    X_prev : array_like
        The previous converged member, shape (n_X,). The point the step is
        measured from.
    t_hat : array_like
        The family tangent at X_prev, shape (n_X,), unit norm. Its direction
        sets the sense of travel; ``ds`` is the unsigned distance along it.
    ds : float
        The arclength step, ds > 0.

    Notes
    -----
    The residual is a raw Euclidean arclength in X, which assumes X's
    components are commensurately scaled -- true in nondimensional CR3BP
    coordinates, where states and times are O(1). A weighted inner product
    would be the remedy if X ever mixed badly-scaled variables.
    """

    def __init__(self, X_prev, t_hat, ds):
        # np.array (not asarray) forces an owned copy, never a view onto the
        # engine's buffers -- the constraint must not alias caller state.
        X_prev = np.array(X_prev, dtype=float)
        t_hat = np.array(t_hat, dtype=float)
        if X_prev.ndim != 1:
            raise ValueError(
                f"X_prev must be a 1-D vector, got shape {X_prev.shape}."
            )
        if t_hat.shape != X_prev.shape:
            raise ValueError(
                f"t_hat must match X_prev in shape; got t_hat {t_hat.shape} "
                f"vs X_prev {X_prev.shape}."
            )
        if not (np.all(np.isfinite(X_prev)) and np.all(np.isfinite(t_hat))):
            raise ValueError("X_prev and t_hat must be finite.")
        norm = float(np.linalg.norm(t_hat))
        if not np.isclose(norm, 1.0, rtol=config.EQUALITY_RTOL,
                          atol=config.EQUALITY_ATOL):
            raise ValueError(
                f"t_hat must be a unit vector; got norm {norm:.6e}. The engine "
                f"is responsible for normalizing the tangent before handing it "
                f"to the closing constraint."
            )
        ds = float(ds)
        if not np.isfinite(ds) or ds <= 0.0:
            raise ValueError(f"ds must be a positive finite step, got {ds}.")
        X_prev.flags.writeable = False
        t_hat.flags.writeable = False
        self._X_prev = X_prev
        self._t_hat = t_hat
        self._ds = ds

    @property
    def n_rows(self) -> int:
        """A single scalar closing row."""
        return 1

    def residual(self, X):
        X = np.asarray(X, dtype=float)
        g = float(self._t_hat @ (X - self._X_prev) - self._ds)
        return np.array([g], dtype=float)

    def jacobian_X(self, X):
        # dg/dX = t_hat, constant in X: a full-width, generally dense row.
        # Return a fresh writable copy, never a view onto stored t_hat.
        return self._t_hat.reshape(1, -1).copy()


class FreeVarPin(FreeVarConstraint):
    """
    Pin a single free variable to a target value (natural-parameter closing).

    The natural-parameter counterpart to PseudoArclength: instead of stepping
    along the family tangent, it holds one component of the free-variable
    vector X fixed at a target, letting the continuation march that component
    directly. The residual is the affine condition

        g(X) = X[col] - target

    and the Jacobian is a one-hot row -- a single 1 in column ``col`` -- so,
    like all free-variable constraints, it is placed straight into the X
    columns with no STM chaining and is constant over a solve.

    This pins a column of X, i.e. a free variable, whichever kind it is: a free
    boundary-time column (period sampling) and a free start-component column
    (state-amplitude sampling) are the same operation here, differing only in
    which column index is passed. The caller (the continuation engine) owns the
    column plan and resolves the semantic target -- "the period", "the start x"
    -- into the concrete column index; this class is deliberately indifferent
    to what the column means.

    Parameters
    ----------
    col : int
        The X column to pin, 0 <= col < n_X. Resolved by the engine from the
        column plan.
    target : float
        The value to hold X[col] at.
    n_X : int
        The width of the free-variable vector, needed to size the one-hot
        Jacobian row. Unlike PseudoArclength (whose width is implicit in its
        tangent), a one-hot pin's data does not carry n_X, so it is passed
        explicitly.
    """

    def __init__(self, col, target, n_X):
        # bool is a subclass of int; reject it so True/False are not silently
        # taken as 1/0 (same guard style as CallableConstraint's n_rows).
        if isinstance(col, bool) or not isinstance(col, (int, np.integer)):
            raise TypeError(
                f"col must be an integer column index, got "
                f"{type(col).__name__}."
            )
        if isinstance(n_X, bool) or not isinstance(n_X, (int, np.integer)):
            raise TypeError(
                f"n_X must be an integer width, got {type(n_X).__name__}."
            )
        col = int(col)
        n_X = int(n_X)
        if n_X < 1:
            raise ValueError(f"n_X must be a positive integer, got {n_X}.")
        if not (0 <= col < n_X):
            raise ValueError(
                f"col must satisfy 0 <= col < n_X = {n_X}, got {col}."
            )
        target = float(target)
        if not np.isfinite(target):
            raise ValueError(f"target must be finite, got {target}.")
        self._col = col
        self._target = target
        self._n_X = n_X

    @property
    def n_rows(self) -> int:
        """A single scalar pin row."""
        return 1

    def residual(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([X[self._col] - self._target], dtype=float)

    def jacobian_X(self, X):
        # dg/dX = e_col, a one-hot row, constant in X. Fresh each call.
        J = np.zeros((1, self._n_X))
        J[0, self._col] = 1.0
        return J


# ========== PER-JUNCTION ROLE SPEC ==========

@dataclass(frozen=True)
class NodeSpec:
    """
    Per-junction role for multiple shooting.

    A NodeSpec collapses the three quantities a junction's role fixes -- which
    post-state components are free (the column-axis selector), which components
    it enforces continuity on (the row-axis selector), and what node type it
    converges to -- into one immutable object. The differential corrector reads
    ``free`` and ``continuity`` when building the column and row plans, and
    ``becomes`` when finalizing a converged trajectory.

    Construct one through a named constructor rather than the raw fields:

    - ``NodeSpec.continuous()`` -- the ordinary patch point. Fully free
      post-state, full-state continuity, converges to a NullJunctionNode. This
      is the default applied to every junction with no explicit spec, so it is
      rarely written by hand.
    - ``NodeSpec.impulsive()`` -- an impulsive maneuver node. Fully free
      post-state, position-only continuity (the velocity is free to jump, and
      that jump is the maneuver), converges to an ImpulsiveJunctionNode.
    - ``NodeSpec.custom(free_vars, continuity, becomes=None)`` -- the escape
      hatch for a role that is neither of the above (e.g. a burn restricted to
      an in-plane subspace). Both component specs accept the same category
      strings and name lists as the shooter's ``free_vars`` argument.

    Fields
    ------
    free : tuple of int
        Ascending post-state component indices (subset of 0..5) that are free
        for this junction -- the column-axis selector fed to the column plan.
    continuity : tuple of int
        Ascending component indices (subset of 0..5) whose continuity defect is
        enforced -- the row-axis selector fed to the row plan. Its length is
        the junction's interior-defect row count.
    becomes : _NodeTarget
        The converged output node type. Set explicitly through the named
        constructors; inferred from ``continuity`` when ``custom`` is called
        with ``becomes=None``.

    Notes
    -----
    The two axes are independent by design: an impulsive node has three
    enforced continuity rows but six free columns (the post-velocity must be
    free -- it is the maneuver -- and the post-position must be free so the
    three position-continuity rows can pin it to the pre-position). The one
    coupling NodeSpec enforces is well-posedness: a component whose continuity
    is enforced should be among the free components, so the junction can close
    that defect by moving its own post-state. This is checked as a heuristic
    (see below), not a hard invariant, because in a multi-segment chain an
    enforced component can sometimes be closed by moving upstream ICs through
    the STM.

    Validation
    ----------
    Malformed index tuples (out of range, not ascending-unique) raise
    unconditionally -- that is an API usage error. The subset and
    ``becomes``-consistency checks route through ``validation_error`` and so
    honor ``config.STRICT_VALIDATION`` (raise when strict, warn otherwise),
    because a genuinely ill-posed global system still surfaces as the solver's
    rank-deficiency warning.

    Examples
    --------
    >>> NodeSpec.continuous()
    NodeSpec(free=(x,y,z,vx,vy,vz), continuity=(x,y,z,vx,vy,vz), becomes=continuous)
    >>> NodeSpec.impulsive()
    NodeSpec(free=(x,y,z,vx,vy,vz), continuity=(x,y,z), becomes=impulsive)
    >>> NodeSpec.custom(free_vars='all', continuity=['x', 'y', 'z', 'vz'])
    NodeSpec(free=(x,y,z,vx,vy,vz), continuity=(x,y,z,vz), becomes=free)
    """

    free: tuple[int, ...]
    continuity: tuple[int, ...]
    becomes: _NodeTarget

    # ---- construction guards ----

    def __post_init__(self) -> None:
        # Structural validity of the index tuples is a hard error: a role built
        # with an out-of-range or unsorted component tuple is malformed, not
        # merely ill-posed. The named constructors never trip these (they feed
        # sorted, in-range tuples); these guard direct/raw construction.
        self._validate_index_tuple(self.free, 'free')
        self._validate_index_tuple(self.continuity, 'continuity')

        if not isinstance(self.becomes, _NodeTarget):
            raise TypeError(
                f"becomes must be a _NodeTarget, got "
                f"{type(self.becomes).__name__}. Build NodeSpec through its "
                f"named constructors (continuous / impulsive / custom)."
            )

        # Local well-posedness: enforce continuity only on components this node
        # can move. Heuristic -- sufficient for local closure, not strictly
        # necessary globally (upstream ICs may close a component through the
        # STM), so it routes through STRICT_VALIDATION rather than raising flat.
        extra = tuple(c for c in self.continuity if c not in set(self.free))
        if extra:
            validation_error(
                f"NodeSpec enforces continuity on component(s) "
                f"{self._names(extra)} that are not free on the post-state "
                f"(free={self._names(self.free)}). This junction cannot close "
                f"those defects by adjusting its own post-state; free those "
                f"components or drop them from continuity."
            )

        # becomes-vs-continuity consistency: catch an explicit becoming that the
        # target node type's own constructor would later reject in _finalize,
        # so the contradiction fails now instead of after a full solve.
        if (self.becomes is _NodeTarget.NULL
                and self.continuity != _ALL_COMPONENTS):
            validation_error(
                f"NodeSpec becomes a continuous (NULL) junction but enforces "
                f"continuity only on {self._names(self.continuity)}; a "
                f"NullJunctionNode requires full-state continuity. Enforce all "
                f"six components, or use becomes='free'."
            )
        if (self.becomes is _NodeTarget.IMPULSIVE
                and not set(_POSITION_COMPONENTS) <= set(self.continuity)):
            validation_error(
                f"NodeSpec becomes an IMPULSIVE junction but does not enforce "
                f"position continuity (needs {self._names(_POSITION_COMPONENTS)}"
                f", has {self._names(self.continuity)}); an "
                f"ImpulsiveJunctionNode requires position continuity."
            )

    @staticmethod
    def _validate_index_tuple(idx: tuple[int, ...], label: str) -> None:
        """Hard-validate a component-index tuple: ascending, unique, in 0..5."""
        for i in idx:
            # bool is an int subclass; reject it so True/False are not read as
            # component 1/0 (same guard style as the free_times parser).
            if isinstance(i, bool) or not isinstance(i, (int, np.integer)):
                raise TypeError(
                    f"{label} entries must be integer component indices, got "
                    f"{type(i).__name__}: {i!r}."
                )
        vals = [int(i) for i in idx]
        if any(v < 0 or v > 5 for v in vals):
            raise ValueError(
                f"{label} component indices must lie in [0, 5], got {tuple(vals)}."
            )
        if vals != sorted(set(vals)):
            raise ValueError(
                f"{label} must be ascending with no duplicates, got {tuple(vals)}."
            )

    @staticmethod
    def _names(idx: tuple[int, ...]) -> str:
        """Render a component-index tuple as its state names, e.g. (x,y,z)."""
        return "(" + ",".join(_STATE_NAMES[i] for i in idx) + ")"

    # ---- named constructors (the intended API) ----

    @classmethod
    def continuous(cls) -> "NodeSpec":
        """Ordinary patch point: fully free, full continuity, becomes NULL.

        The default role for any junction the user does not name. Post == pre
        at convergence, and the junction is rebuilt as a NullJunctionNode.
        """
        return cls(free=_ALL_COMPONENTS,
                   continuity=_ALL_COMPONENTS,
                   becomes=_NodeTarget.NULL)

    @classmethod
    def impulsive(cls) -> "NodeSpec":
        """Impulsive maneuver: fully free, position continuity, becomes IMPULSIVE.

        The post-velocity is free (it carries the maneuver) and the post-
        position is free but pinned continuous to the pre-position by the three
        position-continuity rows. The converged velocity discontinuity is the
        delta_v, and the junction is rebuilt as an ImpulsiveJunctionNode.
        """
        return cls(free=_ALL_COMPONENTS,
                   continuity=_POSITION_COMPONENTS,
                   becomes=_NodeTarget.IMPULSIVE)

    @classmethod
    def custom(cls,
               free_vars: str | list[str] | tuple[str, ...],
               continuity: str | list[str] | tuple[str, ...],
               becomes: str | None = None) -> "NodeSpec":
        """Escape hatch: an arbitrary free / continuity / output combination.

        Parameters
        ----------
        free_vars : str or sequence of str
            Free post-state components. Same forms as the shooter's free_vars
            argument: a category string ('all', 'position', 'velocity',
            'planar', 'none') or a list of component names.
        continuity : str or sequence of str
            Components whose continuity is enforced. Same accepted forms as
            free_vars.
        becomes : {'continuous', 'impulsive', 'free'} or None, optional
            The converged output node type. None (default) infers it from the
            continuity pattern: full continuity -> continuous, position-only
            continuity -> impulsive, anything else -> free.

        Notes
        -----
        Both component specs are resolved through the same _parse_free_vars the
        start-state free_vars uses, so a bad category or component name fails
        here at construction with the identical error message.
        """
        free_t = tuple(int(i) for i in _parse_free_vars(free_vars))
        cont_t = tuple(int(i) for i in _parse_free_vars(continuity))
        target = cls._resolve_becomes(becomes, cont_t)
        return cls(free=free_t, continuity=cont_t, becomes=target)

    # ---- becomes resolution ----

    @staticmethod
    def _resolve_becomes(becomes: str | None,
                         continuity: tuple[int, ...]) -> _NodeTarget:
        """Resolve the user-facing becomes argument to a _NodeTarget.

        None -> inferred from the continuity pattern; a string -> looked up in
        the accepted spellings; anything else -> TypeError.
        """
        if becomes is None:
            return NodeSpec._infer_becomes(continuity)
        if isinstance(becomes, str):
            key = becomes.strip().lower()
            if key not in _BECOMES_FROM_STR:
                valid = ", ".join(repr(k) for k in _BECOMES_FROM_STR)
                raise ValueError(
                    f"Unknown becomes {becomes!r}. Valid values: {valid}, or "
                    f"None to infer from the continuity pattern."
                )
            return _BECOMES_FROM_STR[key]
        raise TypeError(
            f"becomes must be a string or None, got {type(becomes).__name__}."
        )

    @staticmethod
    def _infer_becomes(continuity: tuple[int, ...]) -> _NodeTarget:
        """Infer the output node type from the continuity pattern.

        Full continuity is a continuous (NULL) junction; position-only
        continuity is an IMPULSIVE junction; any other pattern is left as a
        FREE junction, honestly unlabeled rather than forced into a physical
        type it does not match.
        """
        if continuity == _ALL_COMPONENTS:
            return _NodeTarget.NULL
        if continuity == _POSITION_COMPONENTS:
            return _NodeTarget.IMPULSIVE
        return _NodeTarget.FREE

    # ---- display ----

    def __repr__(self) -> str:
        return (f"NodeSpec(free={self._names(self.free)}, "
                f"continuity={self._names(self.continuity)}, "
                f"becomes={self.becomes.name.lower()})")

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

@dataclass(frozen=True)
class _RowBlock:
    """One contiguous block of rows in F / DF.

    Attributes
    ----------
    row_offset : int
        First row of this block in F / DF.
    row_count : int
        Number of rows this block occupies in F / DF (varies by block type)
    continuity_components : tuple of int
        Ascending state-component indices (subset of 0..5) that are constrained to be
        continuous for this node. Its length is the block's row count, and it 
        selects which rows of the identity and segment STM this defect enforces.
        For TERMINAL rows this is None, as those are not continuity constraints and
        their row counts are independently derived.
    kind : _BlockKind
        Method-family / fill-behavior tag; see _BlockKind.
    index : int
        Which object of the kind's family: 0-based junction index for
        INTERIOR_DEFECT, 0-based constraint index for TERMINAL.
    """
    row_offset: int
    row_count : int
    continuity_components: tuple[int, ...] | None
    kind: _BlockKind
    index: int

    def __post_init__(self) -> None:
        if (self.continuity_components is not None and 
            self.row_count != len(self.continuity_components)):
            raise ValueError(
                f"row_count {self.row_count} disagrees with continuity_components "
                f"{self.continuity_components} (len {len(self.continuity_components)})."
    )

    @property
    def row_slice(self) -> slice:
        """Half-open row range [row_offset, row_offset + row_count) of this
        block in F / DF. The single home for the row-span arithmetic, so no
        call site recomputes row_offset + row_count."""
        return slice(self.row_offset, self.row_offset + self.row_count)


@dataclass(frozen=True)
class _RowPlan:
    """Ordered row layout of F / DF: interior defects, then terminal blocks."""
    blocks: tuple[_RowBlock, ...]
    n_rows: int

    @property
    def terminal_blocks(self) -> tuple["_RowBlock", ...]:
        """The terminal row blocks, in order. Used by the free-time pass,
        which places into terminal rows but does not walk the full block list.
        Filtering by kind (not position) stays correct if new block kinds are
        ever interleaved."""
        return tuple(b for b in self.blocks
                     if b.kind is _BlockKind.TERMINAL)


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

    @property
    def col_slice(self) -> slice:
        """Half-open column range [col_start, col_start + width) of this IC's
        free components in X. The single home for the column-span arithmetic,
        so no call site recomputes col_start + width."""
        return slice(self.col_start, self.col_start + self.width)


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

    def free_time_column(self, m: int) -> int:
        """Column of X holding free boundary-time index m.

        Raises ValueError if boundary time m is not free.
        """
        return self.n_state_block + self.free_time_idx.index(m)

# ========== PLAN / SPEC BUILDERS ==========

def _build_row_plan(node_specs: Sequence["NodeSpec"],
                    constraints: Sequence) -> "_RowPlan":
    """Assemble the row layout: interior defect blocks, then constraint blocks.

    Interior defects first (one block per junction, junction order), then
    constraints (one block per constraint, constraint order) -- the single
    source of truth for the row ordering F and DF must share. Each constraint's
    block kind is read from its `space`: TERMINAL constraints become TERMINAL
    blocks, FREEVAR constraints become X_SPACE blocks. Constraint order is
    preserved as passed; a mixed solve (terminal boundary conditions plus a
    free-variable closing constraint) keeps the caller's ordering, so a
    continuation engine that appends its closing constraint last gets it last.

    Each junction's defect block enforces continuity on its own
    ``spec.continuity`` components, so its row count is len(spec.continuity):
    six for a continuous patch point, three for an impulsive maneuver
    (position only).
    """
    blocks: list[_RowBlock] = []
    offset = 0
    for j, spec in enumerate(node_specs):
        cont = spec.continuity
        n = len(cont)
        blocks.append(_RowBlock(offset, n, cont, _BlockKind.INTERIOR_DEFECT, j))
        offset += n
    for i, c in enumerate(constraints):
        n = int(c.n_rows)
        kind = (_BlockKind.TERMINAL if c.space is ConstraintSpace.TERMINAL
                else _BlockKind.X_SPACE)
        blocks.append(_RowBlock(offset, n, None, kind, i))
        offset += n
    return _RowPlan(tuple(blocks), offset)


def _build_column_plan(
    free_idx: np.ndarray,
    node_specs: Sequence["NodeSpec"],
    free_time_idx: np.ndarray,
) -> "_ColumnPlan":
    """Assemble the column layout of X from the free-var / free-time spec.

    Column regions in order: free start components ([0, n_fs)), junction
    post-states, then free-time columns. Each junction contributes
    len(spec.free) columns (six for continuous and impulsive roles; fewer for
    a subspace-restricted custom role). col_start accumulates as a running sum
    so the layout stays correct when a junction contributes fewer than six
    columns.
    """
    n_fs = int(free_idx.size)

    selectors: list[_Selector] = [_Selector(tuple(int(i) for i in free_idx), 0)]
    col = n_fs
    for spec in node_specs:
        comps = spec.free
        selectors.append(_Selector(comps, col))
        col += len(comps)

    n_state_block = col
    n_X = n_state_block + int(free_time_idx.size)
    return _ColumnPlan(
        selectors=tuple(selectors),
        n_fs=n_fs,
        n_state_block=n_state_block,
        n_X=n_X,
        free_time_idx=tuple(int(m) for m in free_time_idx),
    )


def _resolve_node_specs(node_specs: dict | None,
                        n_seg: int) -> tuple[NodeSpec, ...]:
    """
    Expand a sparse per-junction node_specs mapping to one NodeSpec per junction.

    The sole home for two pieces of the node_specs convention: the mapping
    from a user-facing boundary-time key to an internal junction position, and
    the "unspecified junction defaults to continuous" rule. Both plan builders
    and _finalize go through here so a junction's role is defined in exactly
    one place -- if the key arithmetic or the default lived in two functions,
    from_guess and _finalize could silently disagree about what junction j is.

    Key convention (parallel to free_times)
    ---------------------------------------
    Keys are 1-based boundary-time indices, the same indices free_times uses:
    the interior junction between segments j and j+1 sits at boundary time
    j + 1. Valid keys are therefore [1, n_seg - 1]. Boundary indices 0 (the
    start node) and n_seg (the end node) are not interior junctions and are
    rejected. A key absent from the mapping defaults to NodeSpec.continuous(),
    so a plain patch point never needs to be written out.

    Parameters
    ----------
    node_specs : dict or None
        Mapping from a 1-based interior-junction boundary index to a NodeSpec.
        None (or an empty mapping) yields an all-continuous problem.
    n_seg : int
        Number of trajectory segments N. The trajectory has N + 1 boundary
        times and N - 1 interior junctions.

    Returns
    -------
    tuple of NodeSpec
        One NodeSpec per interior junction, length n_seg - 1, 0-indexed in
        segment order (element j is the junction between segments j and j+1).
        Empty for a single-segment (single-shooting) problem.

    Raises
    ------
    TypeError
        If node_specs is neither None nor a dict, a key is not an integer, or
        a value is not a NodeSpec.
    ValueError
        If a key is a boundary index (0 or n_seg) or otherwise outside
        [1, n_seg - 1].

    Notes
    -----
    Called at both ends of the solve, this function is relied on to give the
    same answer each time. That holds because the corrector preserves segment
    count: the converged trajectory _finalize resolves against has the same
    n_seg as the guess from_guess resolved against, so both calls see the same
    (node_specs, n_seg). If a future variant ever changed segment count during
    a solve (e.g. adaptive mesh refinement), _finalize would resolve against
    the new count and mismatch loudly here rather than silently remapping.
    """
    n_junction = n_seg - 1

    # Default every junction to a continuous patch point; override from the
    # mapping below. This is the "missing key = continuous" rule, in one place.
    resolved: list[NodeSpec] = [NodeSpec.continuous()
                                for _ in range(n_junction)]

    if node_specs is None:
        return tuple(resolved)
    if not isinstance(node_specs, dict):
        raise TypeError(
            f"node_specs must be None or a dict mapping interior-junction "
            f"indices to NodeSpec instances, got {type(node_specs).__name__}."
        )

    for key, spec in node_specs.items():
        # --- key: integer, mirroring the free_times bool/int guard ---
        # bool is an int subclass; reject it so True/False are not read as
        # boundary indices 1/0 (and so True would not collide with key 1).
        if isinstance(key, bool) or not isinstance(key, (int, np.integer)):
            raise TypeError(
                f"node_specs keys must be integer boundary-time indices, got "
                f"{type(key).__name__}: {key!r}."
            )
        k = int(key)

        # --- key: boundary nodes are not interior junctions ---
        # FUTURE (terminal/initial maneuvers): keys 0 and n_seg are exactly
        # the hook points for a start/end ImpulsiveBoundaryNode configured via
        # NodeSpec. When that feature lands, these two raises become the
        # dispatch into _finalize's boundary-node construction instead of hard
        # errors, and the messages below will need revisiting (a terminal
        # maneuver is a boundary node, not a constraint). Until then they are
        # unambiguously out of scope for node_specs, so they raise.
        if k == 0:
            raise ValueError(
                f"node_specs index 0 refers to the start time t0 (the "
                f"trajectory's start node), not an interior junction. "
                f"node_specs configures interior junctions only."
            )
        if k == n_seg:
            raise ValueError(
                f"node_specs index {k} refers to the final boundary time (the "
                f"trajectory's end node), not an interior junction. node_specs "
                f"configures interior junctions only; terminal state targeting "
                f"is done through constraints."
            )

        # --- key: otherwise in range [1, n_seg - 1] ---
        if not 1 <= k <= n_junction:
            if n_junction < 1:
                raise ValueError(
                    f"node_specs index {k} is out of range: a single-segment "
                    f"trajectory has no interior junctions."
                )
            raise ValueError(
                f"node_specs index {k} out of range [1, {n_junction}] "
                f"(interior junction indices; index {n_seg} is the final "
                f"boundary)."
            )

        # --- value: a NodeSpec, built through its named constructors ---
        if not isinstance(spec, NodeSpec):
            raise TypeError(
                f"node_specs values must be NodeSpec instances, got "
                f"{type(spec).__name__} for index {k}. Build one via "
                f"NodeSpec.continuous(), NodeSpec.impulsive(), or "
                f"NodeSpec.custom(...)."
            )

        # Boundary index k -> 0-based junction position k - 1.
        resolved[k - 1] = spec

    return tuple(resolved)

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
              len n_free_start    len per node_specs     len n_free_time

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
        coordinate the residual vector F and Jacobian DF rows.
    column_plan : _ColumnPlan
        A custom object detailing the column ordering of the defect Jacobian (DF).
        Contains the ordering of all free variables in the shooting problem,
        consisting of initial free variables, interior nodes, and free times.
        Used by _pack, _unpack and _assemble_DF() to coordinate the free variable 
        vector X and the columns of Jacobian DF. 
    """

    system: "System"
    n_seg: int
    free_idx: np.ndarray
    ics_ref: np.ndarray
    times_ref: np.ndarray
    free_time_idx: np.ndarray
    constraints: tuple
    node_specs: InitVar[tuple[NodeSpec, ...]]
    row_plan: _RowPlan = field(init=False, repr=False)
    column_plan: _ColumnPlan = field(init=False, repr=False)

    def __post_init__(self, node_specs: tuple["NodeSpec", ...]) -> None:
        # Take ownership of the array fields: store private, read-only
        # copies so the shared context is fully immutable and constructing
        # it never mutates caller-held arrays.
        for name in ('free_idx', 'ics_ref', 'times_ref', 'free_time_idx'):
            arr = np.array(getattr(self, name), copy=True)
            arr.flags.writeable = False
            object.__setattr__(self, name, arr)

        object.__setattr__(
            self, 'row_plan',
            _build_row_plan(node_specs, self.constraints),
        )
        object.__setattr__(
            self, 'column_plan',
            _build_column_plan(self.free_idx, node_specs, self.free_time_idx),
        )

        # Consistency check for column plan and number of IC reference states.
        # This also catches a node_specs / n_seg length disagreement: the
        # column plan has 1 + len(node_specs) selectors, so if the resolver
        # ever produced the wrong count it would surface here rather than as a
        # silent misalignment downstream.
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
        node_specs: dict | None = None,
    ) -> "_ShootingContext":
        """
        Build a context from an initial-guess Trajectory and problem spec.
    
        Performs all structural validation up front (node types, time-index
        ranges, constraint normalization/binding, node-spec resolution) so the
        iteration loop can assume a well-formed problem.
    
        Parameters
        ----------
        traj : Trajectory
            The initial-guess trajectory. Defines segment count, the reference
            start state, and the reference boundary times. Every interior junction
            must be a FreeJunctionNode: a shooting guess is discontinuous, and a
            junction's converged role is set through node_specs, not by its guess
            type.
        free_vars : str or sequence of str
            Free start-state specification; see _parse_free_vars.
        constraints : sequence, optional
            Terminal constraints -- TerminalConstraint instances or
            CallableConstraints. None or empty yields no terminal constraints.
        free_times : sequence of int, or None
            Boundary-time indices that are free, drawn from [1, n_seg]. Index
            n_seg is the final time. None (default) fixes all times.
        node_specs : dict, or None
            Mapping from a 1-based interior-junction boundary index (the same
            index convention as free_times) to a NodeSpec, designating that
            junction's role. Junctions absent from the mapping default to
            continuous patch points. None (default) makes every junction a
            continuous patch point. See _resolve_node_specs.
    
        Returns
        -------
        _ShootingContext
    
        Raises
        ------
        ValueError
            If any interior junction is not a FreeJunctionNode; if a free-time
            index is out of range [1, n_seg], duplicated, or references t0; or if
            a node_specs key is a boundary index or otherwise out of range.
        TypeError
            If free_times, node_specs, or a constraint has the wrong type.
        """
        free_idx = _parse_free_vars(free_vars)
        n_seg = traj.n_segments

        # Interior junctions are carried as FreeJunctionNodes in a guess. A raw
        # shooting guess is discontinuous, so an ImpulsiveJunctionNode (which
        # enforces position continuity at construction) or a NullJunctionNode
        # cannot even represent it; the *role* a junction converges to is declared
        # through node_specs, not by its type in the guess.
        for k, node in enumerate(traj.junction_nodes, start=1):
            if isinstance(node, FreeJunctionNode):
                continue
            if isinstance(node, ImpulsiveJunctionNode):
                raise ValueError(
                    f"Interior junction {k} is an ImpulsiveJunctionNode. Shooting "
                    f"guesses carry all junctions as FreeJunctionNode; designate a "
                    f"junction as an impulsive maneuver through "
                    f"node_specs={{{k}: NodeSpec.impulsive()}}, not by placing an "
                    f"ImpulsiveJunctionNode in the guess."
                )
            raise ValueError(
                f"Interior junction {k} is a {type(node).__name__}. Shooting "
                f"guesses carry all interior junctions as FreeJunctionNode; set a "
                f"junction's role through node_specs."
            )

        free_time_idx = cls._parse_free_times(free_times, n_seg)
        bound_constraints = cls._validate_constraints(constraints, traj.system)
        resolved_specs = _resolve_node_specs(node_specs, n_seg)

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
            node_specs=resolved_specs,
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
            if isinstance(c, Constraint):
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
                    f"Each constraint must be a Constraint (a TerminalConstraint "
                    f"or FreeVarConstraint), got {type(c).__name__}."
                )
            bound.append(constraint.bind(system))
        return tuple(bound)

# ========== UTILITY HELPERS ==========

def _select(matrix: np.ndarray, rows: tuple[int, ...],
           cols: tuple[int, ...]) -> np.ndarray:
    """Extract the (rows x cols) sub-block of `matrix` by component indices.

    rows and cols are ascending state-component index tuples (e.g. from a
    row block's continuity_components and a selector's components). Casts to
    lists and uses np.ix_ so the selection is the outer product (all rows x
    all cols), not pairwise. Both are required; there is no 'select all'
    default, because an empty tuple legitimately means 'select none'.
    """
    return matrix[np.ix_(list(rows), list(cols))]

def _terminal_jacobians(c, state_tf: np.ndarray, x0: np.ndarray
                        ) -> tuple[np.ndarray, np.ndarray]:
    """A terminal constraint's Jacobians, each shaped (n_rows, 6).

    Returns (Jtf, Jx0) = (d residual / d state_tf, d residual / d x0),
    atleast_2d. Recomputed wherever needed rather than cached
    across passes: a terminal Jacobian is a cheap analytic or finite-
    difference call, not a propagation, so the passes stay decoupled.
    """
    Jtf = np.atleast_2d(np.asarray(c.jacobian_tf(state_tf, x0), dtype=float))
    Jx0 = np.atleast_2d(np.asarray(c.jacobian_x0(state_tf, x0), dtype=float))
    return Jtf, Jx0

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
    free components to X. Which variables are free was input to solve() via
    node_specs and then incorporated into ctx.

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
    start_free = start_state[list(cp.selectors[0].components)]

    # Each junction post-state contributes only its free components. The selectors
    # start with the Trajectory initial state as 0 (see above), so we need j + 1
    junction_blocks = [
        np.asarray(node.post_state, dtype=float)[list(cp.selectors[j + 1].components)]
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
    partially-free IC scatters only its free components with no change here, 
    and fixed components keep their reference value.

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
        full[list(sel.components)] = X[sel.col_slice]
        ics.append(full)

    # Free boundary times are the tail of X; fixed times keep their reference.
    times = ctx.times_ref.copy()
    times[ctx.free_time_idx] = X[ctx.n_state_block:]

    return ics, times


# ========== CONSTRAINT VECTOR ==========

def _assemble_F(traj: "Trajectory", ctx: _ShootingContext,
                X: np.ndarray) -> np.ndarray:
    """
    Assemble the constraint vector F for a propagated iterate.

    F stacks up to three block families:

    1. Interior defects -- the state discontinuity at each junction, taken
       as node.state_defect (post - pre), then indexed by free variables stored
       in ctx.column_plan.  All nodes should have six free variables except for
       a custom NodeSpec.
    2. Terminal residuals -- each terminal constraint's residual(state_tf, x0),
       concatenated in constraint order. state_tf is the final propagated
       state (end_node.pre_state); x0 is the current start state.
    3. Free-variable residuals -- each free-variable constraint's residual(X),
       a function of the free-variable vector directly (no propagated state).

    Both constraint families follow the convention residual = actual - target,
    so the problem is solved when F is (near) zero.

    Parameters
    ----------
    traj : Trajectory
        A propagated iterate, built from this context via unpack +
        propagate.
    ctx : _ShootingContext
    X : np.ndarray
        The free-variable vector for this iterate, shape (ctx.n_X,). It is the
        exact vector this traj was propagated from (traj == propagate(unpack(X)))
        so the two are a matched pair. Free-variable constraints evaluate their
        residual against it directly; terminal and defect blocks ignore it.

    Returns
    -------
    np.ndarray
        Constraint vector, shape (m,), where
        sum of node free vars + sum of constraint residual lengths. Shape
        (0,) only if there are neither junctions nor constraints.
    """
    blocks: list[np.ndarray] = []

    if ctx.constraints:
        state_tf = np.asarray(traj.end_node.pre_state, dtype=float)
        x0 = np.asarray(traj.start_node.post_state, dtype=float)

    for block in ctx.row_plan.blocks:
        k = block.index                 # 0 based index for nodes and constraints
        if block.kind is _BlockKind.INTERIOR_DEFECT:
            assert block.continuity_components is not None  # true for interior defect

            defect = traj.junction_nodes[k].state_defect
            blocks.append(np.asarray(defect, dtype=float)
                            [list(block.continuity_components)
                        ])
        elif block.kind is _BlockKind.TERMINAL:
            c = ctx.constraints[k]
            r = np.atleast_1d(
                    np.asarray(c.residual(state_tf, x0), dtype=float)
                )
            blocks.append(r)
        elif block.kind is _BlockKind.X_SPACE:
            c = ctx.constraints[k]
            r = np.atleast_1d(np.asarray(c.residual(X), dtype=float))
            blocks.append(r)

    if not blocks:
        return np.array([], dtype=float)
    return np.concatenate(blocks)


# ========== CONSTRAINT JACOBIAN ==========

def _assemble_DF(traj: "Trajectory", ctx: _ShootingContext,
                 X: np.ndarray) -> np.ndarray:
    """
    Assemble the defect Jacobian DF = d(F)/dX for the current iterate.

    DF = d(F)/d(X), with rows matching _assemble_F (interior defects, then
    terminal residuals, then free-variable residuals) and columns matching the
    X layout (start free components, junction post-states, then free times).

    Two passes over the plan. Pass 1 walks the row blocks (interior defects,
    terminal constraints, and free-variable constraints -- all row-owned, every
    entry lands in the block's own rows) and dispatches on block kind. Pass 2
    handles the free-time columns, which are column-owned: a free boundary time
    scatters vector-field terms into several blocks' rows at once, cross-cutting
    the row-block structure, so it gets its own column-wise pass. All row and
    column placements are read from the plan (row_slice / col_slice /
    selectors); no offsets are computed inline.

    Free-variable (X_SPACE) blocks place their full (n_rows, n_X) Jacobian
    directly across all columns in Pass 1 and take no part in Pass 2: any
    dependence on a free boundary time is already a column of that block, so
    there is nothing to chain through the vector field. Pass 2 iterates
    row_plan.terminal_blocks, which filters by kind, so X_SPACE blocks are
    excluded automatically.

    X : np.ndarray
        The free-variable vector for this iterate, shape (ctx.n_X,); the exact
        matched partner of traj. Free-variable constraints differentiate their
        residual against it; terminal and defect blocks ignore it.

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
    Phi_{N-1}. For example, single-shooting Periodicity this reduces to Phi - I.

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
    n_state = ctx.n_state_block
    cp = ctx.column_plan

    state_tf = np.asarray(traj.end_node.pre_state, dtype=float)
    x0 = np.asarray(traj.start_node.post_state, dtype=float)

    DF = np.zeros((ctx.row_plan.n_rows, ctx.column_plan.n_X))
    eye6 = np.eye(6)

    # S_tf = d(state_tf)/d(state columns): Phi of the LAST segment, placed at
    # the last IC's columns, zero elsewhere. Constraint-independent, so built
    # once here and reused by every terminal block. selectors[-1] is the last
    # segment's IC -- the last junction post for N > 1, the start state for
    # N == 1 -- so this one expression covers single and multiple shooting.
    S_tf = None
    if ctx.constraints:
        Phi_last = np.asarray(traj.segment_terminal_stm(n_seg - 1), dtype=float)
        last = cp.selectors[-1]
        S_tf = np.zeros((6, n_state))
        S_tf[:, last.col_slice] = Phi_last[:, list(last.components)]

    # ---- PASS 1: row blocks (interior defects + terminal constraints) ----
    for block in ctx.row_plan.blocks:
        if block.kind is _BlockKind.INTERIOR_DEFECT:
            assert block.continuity_components is not None  #true for interior defects
            i = block.index                      # junction i, between seg i, i+1
            pre = cp.selectors[i]                # x_i   (pre-side IC)
            post = cp.selectors[i + 1]           # x_{i+1} (post-side IC)

            # +I : d(F_i)/d(x_{i+1}), continuity rows x free post cols
            DF[block.row_slice, post.col_slice] += _select(
                eye6, block.continuity_components, post.components)

            # -Phi_i : d(F_i)/d(x_i), continuity rows x free pre cols
            Phi_i = np.asarray(traj.segment_terminal_stm(i), dtype=float)
            DF[block.row_slice, pre.col_slice] += _select(
                -Phi_i, block.continuity_components, pre.components)

        elif block.kind is _BlockKind.TERMINAL:
            c = ctx.constraints[block.index]
            Jtf, Jx0 = _terminal_jacobians(c, state_tf, x0)
            start = cp.selectors[0]

            # implicit path: chain through the last-segment STM into state cols
            DF[block.row_slice, :n_state] += Jtf @ S_tf
            # direct path: residual's own x0 dependence into the start cols
            DF[block.row_slice, start.col_slice] += Jx0[:, list(start.components)]

        elif block.kind is _BlockKind.X_SPACE:
            c = ctx.constraints[block.index]
            # jacobian_X is already (n_rows, n_X) in the full X column ordering:
            # placed directly, no chaining. The constraint owns its column
            # layout, so free-time columns (if any) are already inside JX and
            # this block is skipped by Pass 2.
            JX = np.atleast_2d(np.asarray(c.jacobian_X(X), dtype=float))
            DF[block.row_slice, :] += JX

    # ---- PASS 2: free-time columns ----
    # Block lookup is positional -- interior block j is row_plan.blocks[j] --
    # which holds as long as interior nodes are stored in segment order.
    if ctx.n_free_time > 0:
        endpoints = np.empty((6, n_seg))
        for i in range(n_seg - 1):
            endpoints[:, i] = traj.junction_nodes[i].pre_state
        endpoints[:, n_seg - 1] = state_tf
        f_vals = np.asarray(ctx.system.vector_field(endpoints), dtype=float)
        if f_vals.shape != (6, n_seg):
            f_vals = f_vals.reshape(6, n_seg)

        for m in cp.free_time_idx:
            col = cp.free_time_column(m)

            # t_m as the END time of segment m-1 -> defect block m-1, -f
            if 1 <= m <= n_seg - 1:
                block = ctx.row_plan.blocks[m - 1]
                assert block.continuity_components is not None #true for interior defect
                vals = -f_vals[:, m - 1]
                DF[block.row_slice, col] += (vals[list(block.continuity_components)])

            # t_m as the START time of segment m -> defect block m, +f
            if 1 <= m <= n_seg - 2:
                block = ctx.row_plan.blocks[m]
                assert block.continuity_components is not None #true for interior defect
                vals = f_vals[:, m]
                DF[block.row_slice, col] += (vals[list(block.continuity_components)])

            # t_m moves the final state -> terminal rows, +/- Jtf @ f_tf
            if m == n_seg or m == n_seg - 1:
                sign = 1.0 if m == n_seg else -1.0
                f_tf = f_vals[:, n_seg - 1]
                for tblock in ctx.row_plan.terminal_blocks:
                    c = ctx.constraints[tblock.index]
                    Jtf, _ = _terminal_jacobians(c, state_tf, x0)
                    DF[tblock.row_slice, col] += sign * (Jtf @ f_tf)

    return DF

# ========== SHOOTING OUTPUTS ==========

@dataclass(frozen=True, eq=False)
class ContinuationState:
    """
    Continuation payload attached to a ShooterResult on request.

    The extra per-solve state a continuation engine needs beyond the
    trajectory, packaged as an opt-in field (solve(continuation=True)),
    parallel to diagnostics and iterates. It currently carries the final
    free-variable vector and is the growth point for any further per-step
    data a scheme needs (e.g. an unclosed corrector Jacobian for an analytic
    family tangent), addable as fields without touching the opt-in wiring.

    Invariant: when a ShooterResult carries a ContinuationState, its X is the
    free-variable vector that produced that result's .trajectory -- the two
    are a matched pair (see the population guard in solve).

    Attributes
    ----------
    X : np.ndarray
        Read-only (n_X,) free-variable vector at the final evaluated iterate,
        in the corrector's pack ordering [free start components, junction
        post-states, free boundary times]. eq is disabled because array fields
        defeat the generated __eq__.
    """
    X: np.ndarray


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
        appropriate nodes per node_specs, at the solver tolerance), or the last 
        iterate on non-convergence. None only if the very first propagation failed.
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
        FreeJunctionNodes intact (the conversion applies to the final
        `.trajectory` only).
    continuation : ContinuationState or None
        Free-variable vector (and future per-step continuation data) at the
        final iterate. Populated only when solve(continuation=True) and the
        solve exited cleanly (no abort), so its X matches `.trajectory`.
    """

    trajectory: "Trajectory | None"
    converged: bool
    iterations: int
    final_residual: float
    abort_reason: str | None = None
    diagnostics: dict | None = None
    iterates: list | None = None
    continuation: "ContinuationState | None" = None

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
    (free start-state components, terminal constraints, free times, node_specs) 
    and iterates a minimum-norm Newton scheme until the constraint vector F is
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
              node_specs: dict | None = None,
              diagnostics: bool = False,
              iterates: bool = False,
              continuation: bool = False) -> ShooterResult:
        """
        Correct an initial-guess trajectory to satisfy the constraints.

        Builds the internal solve context from the guess and the problem
        specification, runs the minimum-norm Newton iteration, and on success
        converts the converged interior FreeJunctionNodes into
        appropriate nodes per node_specs at this corrector's tolerance.

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
        node_specs : dict of NodeSpecs, int keys, optional
            NodeSpec objects specify free and constrained variables at an interior
            node, as well as the intended output node type.  Use premade constructors
            for NodeSpec (NodeSpec.continuous(), NodeSpec.impulsive()). Use
            NodeSpec.custom(...) with great caution.  Index by integer indices,
            using a 1-indexed format matching free_times (i.e. consider Node 0 to 
            be the StartNode, and begin junction node indexing at Node 1.)
            Defaults to None, which causes full state continuity and all six free
            variables at all junction nodes.
        diagnostics : bool, default False
            If True, populate result.diagnostics.
        iterates : bool, default False
            If True, populate result.iterates.
        continuation : bool, default False
            If True and the solve exits cleanly, populate
            result.continuation with the final free-variable vector.

        Returns
        -------
        ShooterResult
        """
        ctx = _ShootingContext.from_guess(traj, free_vars, constraints,
                                          free_times, node_specs)
        raw = self._run(ctx, traj,
                        store_diagnostics=diagnostics,
                        store_iterates=iterates)

        out_traj = raw['trajectory']
        if raw['converged'] and out_traj is not None:
            out_traj = self._finalize(out_traj, node_specs)

        # Continuation payload: populate only on a clean exit, where the final
        # X and out_traj are a matched pair (on an abort, traj can be stale
        # relative to X). Producer makes the read-only copy.
        cont = None
        if continuation and raw['abort_reason'] is None:
            X_final = np.array(raw['X'], dtype=float)
            X_final.flags.writeable = False
            cont = ContinuationState(X=X_final)

        return ShooterResult(
            trajectory=out_traj,
            converged=raw['converged'],
            iterations=raw['iterations'],
            final_residual=raw['final_residual'],
            abort_reason=raw['abort_reason'],
            diagnostics=raw['diagnostics'],
            iterates=raw['iterates'],
            continuation=cont,
        )

    def _finalize(self, traj: "Trajectory", node_specs: dict | None) -> "Trajectory":
        """
        Convert converged interior FreeJunctionNodes to appropriate node types at
        this corrector's tolerance, reusing the converged outputs.  The output
        node type is read from node_specs via _resolve_node_specs.

        A converged junction has |defect| < tol, so the converged nodes
        (constructed with tol=self.tol) validate successfully and record the
        standard they were closed to. Free junctions resulting from custom 
        NodeSpecs are passed through unchanged.
        """
        new_nodes = []
        resolved = _resolve_node_specs(node_specs, traj.n_segments)

        for k, node in enumerate(traj.junction_nodes):
            target = resolved[k].becomes
            if target is _NodeTarget.NULL:
                new_nodes.append(NullJunctionNode(
                    node.time, node.pre_state, node.post_state, tol=self.tol))
            elif target is _NodeTarget.IMPULSIVE:
                new_nodes.append(ImpulsiveJunctionNode(
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

            F = _assemble_F(traj, ctx, X)
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

            DF = _assemble_DF(traj, ctx, X)
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
            'X': X,
        }
