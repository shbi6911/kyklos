"""
Recipe correction wrapper: correct a guess into a verified periodic orbit.

correct_as is the public convenience entry point of the correction ecosystem.
Given a CorrectorGuess -- a state, a period estimate, a family label, and the
System it belongs to -- it looks up the family's correction recipe, runs the
differential corrector, and returns a verified PeriodicOrbit. The user says
"correct this as an L1 halo" and gets back a full family member with
repropagation and closure validation done.

CorrectorGuess lives in this module (it carries a System, so it belongs above
the dependency-free registry leaf). The registry -- recipe entries and the
label vocabulary -- exists in registry.py.

This module also contains determinacy-layout transforms.

A correction *recipe* fixes the invariant family geometry (perpendicular-
crossing free_vars and constraints). A *layout* selects which family member the
corrector converges to, by choosing which of the recipe's freedoms is pinned
and whether a node time is freed to keep the shooting system square. Recipe and
layout are orthogonal: the recipe says which family, the layout says which
member-selection coordinate.

Currently supported layouts are period_locked (the default) and x_amplitude_locked.
x_amplitude_locked is used by default when correcting from a planar seed via the 
CorrectorGuess.from_seeder_result() method.  If a period_locked correction 
from a seed is desired, the CorrectorGuess can be constructed standalone.
"""
import numpy as np
from typing import NamedTuple, Callable, TYPE_CHECKING
from dataclasses import dataclass

from .registry import _RECIPES, _RecipeEntry, available_recipes
from .shooter import (DifferentialCorrector, TargetState, ShooterResult,
                      Constraint, ConstraintSpace, FreeVarConstraint,
                      PseudoArclength)
from .periodic_orbit import PeriodicOrbit
from .system import System, SysType
from .exceptions import ConvergenceError

if TYPE_CHECKING:
    from .trajectory import Trajectory

# ===========================================================================
# Corrector guess
# ===========================================================================
def _check_guess_state(state) -> np.ndarray:
    """
    Validate and normalize a guess state to a contiguous (6,) array.

    Deliberately NOT System._check_field_state: that validator also accepts
    (6, N) batches for the batched field evaluators, but a corrector guess is
    a single orbit, so a batch is meaningless here and must be rejected at the
    funnel rather than failing later inside propagate/solve. This enforces the
    stricter single-state contract; the small duplication is intentional.

    Parameters
    ----------
    state : array-like
        Prospective state [x, y, z, vx, vy, vz].

    Returns
    -------
    np.ndarray
        Contiguous float array of shape (6,).

    Raises
    ------
    ValueError
        If the state is not a finite 1-D array of length 6.
    """
    arr = np.ascontiguousarray(state, dtype=float)
    if arr.ndim != 1 or arr.size != 6:
        raise ValueError(
            f"Guess state must be a 1-D array of length 6 "
            f"[x, y, z, vx, vy, vz], got shape {arr.shape}."
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("Guess state contains non-finite values.")
    return arr


class CorrectorGuess:
    """
    Validated input to the recipe correction wrapper.

    The single, narrow funnel through which every correction request flows,
    regardless of origin: a planar seed, a continuation step, a bifurcation
    orbit, or a hand-built state. It carries only what a corrector actually
    needs -- a starting state, a period guess, and a family label -- and
    validates all three at construction so malformed requests fail early and
    clearly rather than deep inside the corrector.

    It deliberately does NOT carry seeder diagnostics (frequency, saddle rate,
    etc.): those describe how a *linear seed* was produced and are meaningless
    for, say, a halo guess perturbed off a bifurcation orbit. A SeederResult
    can *produce* a CorrectorGuess (via ``from_seeder_result``), but a
    CorrectorGuess is not a SeederResult.

    Parameters
    ----------
    state : array-like
        Starting state [x, y, z, vx, vy, vz], shape (6,), nondimensional, in
        the rotating frame. Stored as a read-only (6,) array.
    period : float
        Period guess for the orbit, nondimensional and positive. See
        ``period_is_half`` for the half- vs full-period convention.
    system : System
        System to propagate the guess in and feed to the shooter for the solve.
        Currently restricted to CR3BP systems only.  Matching the system to the 
        guessed state is the responsibility of the caller.
    recipe : str
        Family label naming the recipe to correct against, e.g. 'lyapunov' or
        'halo'. Must be a registered recipe (see ``available_recipes()``).
    layout : str
        A label naming the layout of variables used for correction, modifies the base
        recipe.  For example, a 'lyapunov' can be converged with 'period_locked'
        layout (the default) or 'x_amplitude_locked'
    period_is_half : bool, optional
        Convention flag for ``period``. If True, ``period`` is a half period
        (the time to the next perpendicular crossing), which is what a
        symmetry recipe's free-time guess wants directly. If False (default),
        ``period`` is a full period; the wrapper halves it when a symmetry
        recipe is used. This makes the half- vs full-period distinction
        explicit rather than a silent assumption: a guess taken from a
        converged orbit's ``.period`` is a full period and should use the
        default; a guess from a seeder half-period estimate should pass True.

    Attributes
    ----------
    state : np.ndarray
        Read-only (6,) starting state.
    period : float
        Period guess as supplied.
    system : System
        System as supplied, validated to be CR3BP.
    recipe : str
        Validated recipe label.
    layout : str
        Validated layout label.
    period_is_half : bool
        The convention flag for ``period``.

    Raises
    ------
    ValueError
        If the state is not a finite (6,) array, the period is not positive and
        finite, the recipe or layout label is not registered, 
        or the System is not CR3BP.
    """

    __slots__ = ("_state", "_period", "_system", 
                 "_recipe", "_layout", "_period_is_half"
    )

    def __init__(
        self,
        state,
        period: float,
        system: System,
        recipe: str,
        layout: str,
        period_is_half: bool = False,
    ):
        # State: validate shape/finiteness, store read-only.
        arr = _check_guess_state(state)
        arr.flags.writeable = False

        # Period: must be a positive, finite scalar.
        period = float(period)
        if not np.isfinite(period) or period <= 0.0:
            raise ValueError(
                f"Period guess must be a positive, finite number, got {period}."
            )
        
        # System: must be a CR3BP system
        if system.base_type is not SysType.CR3BP:
            raise ValueError(
                f"CorrectorGuess is only valid for CR3BP Systems, "
                f"got {system.base_type}"
            )

        # Recipe label: must be registered. Single source of truth is the
        # registry, so validation and discovery cannot drift apart.
        if recipe not in _RECIPES:
            raise ValueError(
                f"Unknown recipe label {recipe!r}; "
                f"known recipes are {available_recipes()}."
            )
        
        # Layout label: must be registered. See _LAYOUTS in this file.
        if layout not in _LAYOUTS:
            raise ValueError(
                f"Unknown layout label {layout!r}; "
                f"known recipes are {available_layouts()}."
            )

        self._state = arr
        self._period = period
        self._system = system
        self._recipe = recipe
        self._layout = layout
        self._period_is_half = bool(period_is_half)

    # Read-only properties: the guess is immutable once constructed.
    @property
    def state(self) -> np.ndarray:
        """Read-only (6,) starting state."""
        return self._state

    @property
    def period(self) -> float:
        """Period guess as supplied (see period_is_half for convention)."""
        return self._period
    
    @property
    def system(self) -> System:
        """System as supplied (validated as CR3BP)."""
        return self._system

    @property
    def recipe(self) -> str:
        """Validated recipe label."""
        return self._recipe
    
    @property
    def layout(self) -> str:
        """Validated layout label."""
        return self._layout

    @property
    def period_is_half(self) -> bool:
        """True if ``period`` is a half period, False if a full period."""
        return self._period_is_half

    def half_period(self) -> float:
        """
        Return the half-period, regardless of the stored convention.

        Convenience for a symmetry recipe's free-time guess, which is always a
        half period. Resolves ``period`` / ``period_is_half`` to a half period
        without the caller having to branch on the convention.

        Returns
        -------
        float
            The half period.
        """
        return self._period if self._period_is_half else 0.5 * self._period

    @classmethod
    def from_seeder_result(cls, result, system, recipe: str) -> "CorrectorGuess":
        """
        Build a CorrectorGuess from a seeder result.

        Projects the rich seeder output down to the minimal corrector input:
        the seed state and its period, tagged with the requested family recipe.
        The seeder's diagnostic fields (frequency, saddle rate, etc.) are
        intentionally dropped -- they are not corrector input.

        This expects ``result`` to expose ``.state`` (a (6,) array) and
        ``.period`` (a full linear period, 2*pi/omega_planar). The seeder's
        period is a full period, so the resulting guess uses the full-period
        convention (period_is_half=False); the wrapper halves it for a symmetry
        recipe.

        This defaults to the x_amplitude_locked layout, which means it should converge
        an orbit a distance away from the equilibrium point corresponding to the 
        amplitude requested from planar_seeder().  If a period-locked orbit is desired,
        a CorrectorGuess can be directly constructed from SeederResult data, bypassing
        this convenience method.

        Parameters
        ----------
        result : SeederResult
            A seeder result with ``.state`` and ``.period``.
        recipe : str
            Family label to correct against, e.g. 'lyapunov'.

        Returns
        -------
        CorrectorGuess
        """
        return cls(
            state=result.state,
            period=result.period,
            system=system,
            recipe=recipe,
            layout='x_amplitude_locked',
            period_is_half=False,
        )

    def __repr__(self) -> str:
        return (
            f"CorrectorGuess(recipe={self._recipe!r}, layout={self._layout!r}, "
            f"period={self._period!r}, period_is_half={self._period_is_half!r} "
            f"state={self._state!r}), system.mass ratio={self._system.mass_ratio}"
        )
    
# ===========================================================================
# Solve layout
# ===========================================================================
class _SolveLayout(NamedTuple):
    """
    The determinacy layout of a single corrector solve.

    Carries the full determinacy triple even though standalone correction only
    populates two of the three fields: a continuation scheme edits this layout
    (pinning a variable, freeing a node time) to keep the shooting system
    square, and needs all three handles available.  Immutable; a scheme transform 
    returns a new layout via _replace rather than mutating.

    Fields
    ------
    free_vars : tuple[str, ...]
        State components the corrector solves for.
    free_times : tuple[int, ...]
        Node-time indices freed for the solve. Empty for standalone correction
        (all node times fixed); a continuation scheme populates this when it
        trades a geometric freedom for a free period.
    constraint_spec : dict[str, float]
        Terminal target conditions, {component: value}. Owned by the layout (a
        copy of the recipe's spec), so a scheme may edit it freely.
    period_convention : str
        The convention for propagation time of the guess (full or half) 
        according to the solver recipe.
    """

    free_vars: tuple[str, ...]
    free_times: tuple[int, ...]
    constraint_spec: dict[str, float]

def _base_layout(recipe: _RecipeEntry) -> _SolveLayout:
    """
    Build the standalone determinacy layout from a recipe.

    Copies the recipe's constraint_spec so the layout owns its own dict; the
    recipe entry is shared inert data and must never be mutated through the
    layout. free_times is empty (standalone fixes all node times).
    """
    return _SolveLayout(
        free_vars=recipe.free_vars,
        free_times=(),
        constraint_spec=dict(recipe.constraint_spec),
    )

# ===========================================================================
# Layout transforms
# ===========================================================================
def _period_locked(layout: _SolveLayout) -> _SolveLayout:
    """
    Pin the period: the base (fixed-time) layout, unchanged.

    x and vy stay free, all node times fixed. The corrector solves for the
    family member whose period equals the guess period. This is the identity
    transform -- the recipe's base layout is already period-locked.
    """
    return layout


def _x_amplitude_locked(layout: _SolveLayout) -> _SolveLayout:
    """
    Pin the x-amplitude, free the half-period node.

    Drops x from free_vars (the corrector then holds x at the guess trajectory's
    value, i.e. the seed amplitude) and frees the end-node time so the system
    stays square: free_vars becomes (vy,) plus the freed half-period, against
    {y: 0, vx: 0}. Solves for the family member at the guess amplitude, letting
    its period be found.

    Raises
    ------
    ValueError
        If x is not among the layout's free_vars -- the layout is incompatible
        with a recipe that does not free x, and pinning it would leave the
        system mis-determined. (This is the layout/recipe compatibility guard;
        for Lyapunov and halo, x is always free.)

    Notes
    -----
    The freed node time is the end node, assumed to be index 1: this expects a
    single-arc (two-node) guess trajectory -- a start node and an end node --
    which is what the planar seeder produces. A multi-arc guess would need the
    actual end-node index rather than a hard-coded 1.
    """
    if "x" not in layout.free_vars:
        raise ValueError(
            f"x_amplitude_locked layout pins x, but x is not in the recipe's "
            f"free_vars {layout.free_vars}; the layout is incompatible with "
            f"this recipe."
        )
    free_vars = tuple(v for v in layout.free_vars if v != "x")
    return layout._replace(free_vars=free_vars, free_times=(1,))


# ===========================================================================
# Layout registry
# ===========================================================================
# Single source of truth for the member-selection layout vocabulary. Label
# validation (in CorrectorGuess) and discovery (available_layouts) derive from
# this, so no parallel list of layout labels should exist elsewhere.
_LAYOUTS = {
    "period_locked": _period_locked,
    "x_amplitude_locked": _x_amplitude_locked,
}


def _get_layout(label: str):
    """
    Return the layout transform for a member-selection label.

    Parameters
    ----------
    label : str
        Layout label, e.g. 'x_amplitude_locked' or 'period_locked'.

    Returns
    -------
    callable
        A transform mapping a base _SolverLayout to the member-selection layout.

    Raises
    ------
    ValueError
        If the label is not a registered layout. The message enumerates the
        known layouts.
    """
    try:
        return _LAYOUTS[label]
    except KeyError:
        raise ValueError(
            f"Unknown layout label {label!r}; "
            f"known layouts are {available_layouts()}."
        )


def available_layouts() -> list[str]:
    """
    Return the sorted list of member-selection layout labels.

    Public discovery for the layout vocabulary: a user constructing a
    CorrectorGuess uses this to know which layout labels are valid, the same way
    available_recipes() exposes the recipe vocabulary.

    Returns
    -------
    list[str]
        Recognized layout labels, e.g. ['amplitude_locked', 'period_locked'].
    """
    return sorted(_LAYOUTS)


# ===========================================================================
# Shooter spec input for the continuation atomic operator
# ===========================================================================
@dataclass(frozen=True, eq=False)
class SolveSpec:
    """
    Finalized, built determinacy spec passed into solve_recipe.

    The realized form of a _SolveLayout: where _SolveLayout is the inert,
    transformed-in-a-pipeline description (a NamedTuple of free_vars /
    free_times / constraint_spec), a SolveSpec is what you get after *building*
    it -- the terminal constraints are constructed objects, ready to hand to
    the shooter. It is loop-invariant across a continuation march: constructed
    once above the loop, then passed unchanged into every step. solve_recipe
    appends the per-step continuation closer to a *fresh* list built from these
    constraints; it never mutates the spec.

    The constraints here are the TERMINAL boundary conditions only (e.g. the
    TargetState from the recipe). The continuation closer (PseudoArclength /
    FreeVarPin) is NOT in the spec -- solve_recipe appends it per step from the
    ContinuationRef. corank() counts the spec as written (closer excluded), so
    a corank-1 spec is the expected input to a closed continuation solve, and a
    corank-0 (square) spec is the isolated / bootstrap case.

    What is NOT here, and why: no System, no seed state, no propagation span --
    all of those ride on the guess Trajectory passed alongside the spec into
    solve_recipe (the shooter reads system/states/times off the trajectory via
    _pack). The spec is purely the finalized set of solve() determinacy
    arguments minus the guess and minus the run flags.

    Attributes
    ----------
    free_vars : tuple of str
        Free start-state component names (e.g. ("x", "vy")). Count of free
        start columns.
    free_times : tuple of int
        Node indices whose boundary times are free. Their *values* live on the
        guess trajectory (via _pack); the spec carries only which are free.
    constraints : tuple of Constraint
        Built terminal constraints. Stored as a tuple so the loop-invariant set
        cannot be appended to in place. eq is disabled because these objects
        (and their arrays) defeat a generated __eq__.
    node_specs : dict or None
        Per-junction role specs for multiple shooting. None for single
        shooting (the only supported mode for now); present as the MS
        extension point so MS adds values, not fields.
    """

    free_vars: tuple[str, ...]
    free_times: tuple[int, ...]
    constraints: tuple[Constraint, ...]
    node_specs: dict | None = None

    def __post_init__(self) -> None:
        # Freeze the collection fields to tuples (the layout hands free_times
        # in as a list; constraints must be un-appendable to protect the
        # loop-invariant). Frozen dataclass, so assign through object.
        object.__setattr__(self, "free_vars", tuple(self.free_vars))
        object.__setattr__(self, "free_times", tuple(self.free_times))
        object.__setattr__(self, "constraints", tuple(self.constraints))

        if self.node_specs is not None and not isinstance(self.node_specs, dict):
            raise TypeError(
                f"node_specs must be a dict or None, got "
                f"{type(self.node_specs).__name__}."
            )
        # The closer is appended by solve_recipe, not baked in; a free-variable
        # constraint here would make corank() miscount (its row would be
        # counted as terminal). Enforce terminal-only so corank() stays honest.
        for c in self.constraints:
            if c.space is not ConstraintSpace.TERMINAL:
                raise ValueError(
                    f"SolveSpec.constraints must be terminal constraints; got "
                    f"a {type(c).__name__} with space={c.space}. The "
                    f"continuation closer is appended by solve_recipe, not "
                    f"placed in the spec."
                )
        
    @property
    def n_X(self) -> int:
        """
        Length of the free variable vector, computed from spec
        """
        return len(self.free_vars) + len(self.free_times)

    @property
    def n_rows(self) -> int:
        """Number of Jacobian rows computed from spec"""
        return sum(c.n_rows for c in self.constraints)
    
    @property
    def corank(self) -> int:
        """
        Determinacy of the unclosed solve: n_unknowns - n_equations.

        0 -> square (isolated / bootstrap solve); 1 -> corank-1, underdetermined
        by exactly one, ready for a single continuation closer to square it.
        The closer is excluded here (solve_recipe appends it), so this is the
        corank *before* closing.

        Single shooting: n_X is the free start components plus the free boundary
        times, and n_rows is the terminal constraint rows. Multiple shooting
        would add junction post-state columns to n_X and interior-defect rows to
        n_rows (both driven by node_specs); until that is implemented, a non-None
        node_specs is rejected rather than silently counted wrong. The `== 1`
        determinacy invariant the loop checks is unchanged under MS -- only the
        counts grow.
        """
        if self.node_specs is not None:
            raise NotImplementedError(
                "corank() does not yet account for multiple-shooting node_specs "
                "(junction columns and interior-defect rows); MS corank is not "
                "implemented."
            )
        return self.n_X - self.n_rows


# ===========================================================================
# Input class for atomic operator containing previous solve-data
# ===========================================================================
@dataclass(frozen=True, eq=False)
class ContinuationRef:
    """
    Per-step reference state the engine hands to a continuation scheme's closer.

    The continuation engine's marching state for one step, passed into
    solve_recipe alongside the scheme selector. It plays two roles that share
    the same data: the engine *predicts* the next member with it
    (X_pred = X_prev + ds * t_hat), and the scheme's closer *references* it
    (PseudoArclength closes on t_hat . (X - X_prev) - ds). It is built fresh
    each step from the previous member and its analytic tangent, and it is the
    input counterpart to the ShooterResult.ContinuationState a solve produces:
    the engine reads a ContinuationState off step k and builds a
    ContinuationRef for step k+1.

    It is deliberately closer-agnostic. The fields are the resolved per-step
    predictor state any closer factory can read -- an arclength closer consumes
    all three directly; a pin closer would derive its (col, target) from them
    plus the layout. So validation here is limited to what is true of *any*
    predictor state (shapes, finiteness, positive step); closer-specific
    preconditions (e.g. PseudoArclength's unit-tangent requirement) are left to
    the closer, so this record never bakes in one closer's needs.

    Direction is already resolved upstream: t_hat arrives with its sign fixed
    (period-component seed at member one, tangent-continuity thereafter), so
    this record carries a finished tangent and never sees a sign question.

    Parameters
    ----------
    X_prev : array_like
        The previous converged member, shape (n_X,). Copied to an owned
        read-only array (the engine builds this from its live working vectors,
        so the record must not alias them).
    t_hat : array_like
        The family tangent at X_prev, shape (n_X,), direction already resolved.
        Copied read-only. Unit-norm is *not* checked here (that is the
        arclength closer's precondition).
    ds : float
        The arclength step, ds > 0. Direction lives in t_hat, so ds is the
        unsigned distance stepped along it.
    """

    X_prev: np.ndarray
    t_hat: np.ndarray
    ds: float

    def __post_init__(self) -> None:
        # Owned copies (np.array, not asarray) so the record never aliases the
        # engine's live arrays; frozen, so set through object.__setattr__.
        X_prev = np.array(self.X_prev, dtype=float)
        t_hat = np.array(self.t_hat, dtype=float)

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

        ds = float(self.ds)
        if not np.isfinite(ds) or ds <= 0.0:
            raise ValueError(f"ds must be a positive finite step, got {ds}.")

        X_prev.flags.writeable = False
        t_hat.flags.writeable = False
        object.__setattr__(self, "X_prev", X_prev)
        object.__setattr__(self, "t_hat", t_hat)
        object.__setattr__(self, "ds", ds)



# ===========================================================================
# Continuation schemes
# ===========================================================================
class _SchemeEntry(NamedTuple):
    """
    The two halves of a continuation scheme, bound together.

    A scheme is what turns a square, standalone correction problem into a
    marching one. It does that in two moves that must agree, so they are
    stored as one entry rather than in parallel registries:

      1. a corank-*opening* layout transform, which releases one freedom so
         the unclosed system is underdetermined by exactly one, and
      2. a closer factory, which builds the single constraint that closes it
         back up from that step's reference data.

    Open without closing and the solve is underdetermined; close without
    opening and it is overdetermined. Pairing them here means a caller
    selects one label and cannot mismatch the halves.

    This is the contrast with _LAYOUTS, and the reason for a separate
    registry: a layout transform is corank-*preserving* (it trades one
    freedom for another, keeping the system square for an isolated solve),
    while a scheme transform is corank-*opening*. Same _SolveLayout
    vocabulary, opposite contract, so the distinction is kept legible at the
    registry boundary rather than routed through one dict.

    A NamedTuple rather than a bare 2-tuple so consumers read
    ``entry.transform`` / ``entry.closer_factory`` by name; positional
    unpacking of a pair is exactly what silently swaps when a third field is
    added later.

    Fields
    ------
    transform : callable
        (_SolveLayout) -> _SolveLayout. Corank-opening; returns a new layout
        via _replace rather than mutating.
    closer_factory : callable
        (ContinuationRef, int) -> FreeVarConstraint. Called fresh each step
        with that step's reference data and the width of X.
    """

    transform: Callable[[_SolveLayout], _SolveLayout]
    closer_factory: Callable[[ContinuationRef, int], FreeVarConstraint]


def _free_period(layout: _SolveLayout) -> _SolveLayout:
    """
    Open corank 1 by freeing the end-node time.

    The corank-opening half of the pseudo-arclength scheme. It frees the
    half-period node time *without* compensating -- which is precisely how it
    differs from _x_amplitude_locked, whose freeing of the same time is paid
    for by dropping x from free_vars to stay square. Here the extra freedom is
    the point: it is what the arclength row is appended to close.

    Freeing the period is safe for a march even though it is unsafe for a
    fresh planar seed. The seed's linearized period is a *bad* period, and
    freeing it collapses the orbit toward the equilibrium point (the reason
    x_amplitude_locked exists). A continuation step starts from an
    already-converged member with a good period, so there is nothing to
    collapse toward. That makes the seed's fragility a bootstrap concern
    living upstream, not a scheme concern.

    Parameters
    ----------
    layout : _SolveLayout
        The recipe's base (square, fixed-time) layout.

    Returns
    -------
    _SolveLayout
        The same layout with the end-node time freed, corank 1.

    Raises
    ------
    ValueError
        If the layout already frees a node time. Stacking two openings would
        produce corank 2, which one closer cannot square.

    Notes
    -----
    The freed node time is the end node, assumed to be index 1: this expects a
    single-arc (two-node) guess trajectory -- a start node and an end node --
    which is what the planar seeder produces. A multi-arc guess would need the
    actual end-node index rather than a hard-coded 1.
    """
    if layout.free_times:
        raise ValueError(
            f"A corank-opening scheme transform requires a square base layout "
            f"with all node times fixed, but this layout already frees "
            f"free_times={layout.free_times}. _free_period frees the end-node "
            f"time to open corank 1; stacking it on a layout that has already "
            f"freed a time would open corank 2, which a single closing "
            f"constraint cannot square. Use a fixed-time (period_locked) base "
            f"layout."
        )
    return layout._replace(free_times=(1,))


def _arclength_closer(ref: ContinuationRef, n_X: int) -> FreeVarConstraint:
    """
    Build the pseudo-arclength closing constraint for one step.

    The closer half of the pseudo-arclength scheme. Called fresh each step by
    solve_recipe, so the step's reference data is baked into a new immutable
    constraint rather than mutated onto a persistent one.

    Parameters
    ----------
    ref : ContinuationRef
        This step's reference state. All three fields are consumed directly:
        the previous converged member, the signed unit tangent, and the step.
    n_X : int
        Width of the free-variable vector. Deliberately unused here --
        PseudoArclength sizes itself from t_hat, which is already (n_X,). It
        is in the factory signature because a one-hot pin closer cannot infer
        its width from its data (a column index and a target scalar carry no
        length) and must be told. Keeping one factory signature across schemes
        is worth one ignored argument.

    Returns
    -------
    PseudoArclength
        The single closing row t_hat . (X - X_prev) - ds.
    """
    return PseudoArclength(ref.X_prev, ref.t_hat, ref.ds)


# Single source of truth for the continuation scheme vocabulary. Labels name
# the continuation *method* the user selects, not the freedom the transform
# happens to open -- which freedom gets opened is determined by the recipe's
# determinacy arithmetic, not by user choice, and is an internal detail of the
# transform. A future natural-parameter scheme would register here as its own
# method label (e.g. opening the period and pinning x, a stepped
# x_amplitude_locked), not as a variant spelling of this one.
_SCHEMES = {
    "pseudo_arclength": _SchemeEntry(
        transform=_free_period,
        closer_factory=_arclength_closer,
    ),
}


def _get_scheme(label: str) -> _SchemeEntry:
    """
    Return the scheme entry for a continuation scheme label.

    Parameters
    ----------
    label : str
        Scheme label, e.g. 'pseudo_arclength'.

    Returns
    -------
    _SchemeEntry
        The paired corank-opening transform and closer factory.

    Raises
    ------
    ValueError
        If the label is not a registered scheme. The message enumerates the
        known schemes.
    """
    try:
        return _SCHEMES[label]
    except KeyError:
        raise ValueError(
            f"Unknown scheme label {label!r}; "
            f"known schemes are {available_schemes()}."
        )


def available_schemes() -> list[str]:
    """
    Return the sorted list of continuation scheme labels.

    Discovery for the scheme vocabulary, parallel to available_recipes() and
    available_layouts(). Not re-exported from the package yet: the only
    consumer is the continuation engine, and the vocabulary should not be
    advertised ahead of it.

    Returns
    -------
    list[str]
        Recognized scheme labels, e.g. ['pseudo_arclength'].
    """
    return sorted(_SCHEMES)



# ===========================================================================
# Correction core (ShooterResult-returning atomic operator)
# ===========================================================================
def solve_recipe(
    spec: SolveSpec,
    guess_traj: "Trajectory",
    corrector: DifferentialCorrector | None = None,
    *,
    closer_factory: Callable[[ContinuationRef, int], FreeVarConstraint] | None = None,
    ref: ContinuationRef | None = None,
    continuation: bool = False,
) -> ShooterResult:
    """
    Atomic correction operator: solve a built spec, return the raw result.

    The single step both correct_as and the continuation loop call. It takes a
    finalized SolveSpec (determinacy config with terminal constraints already
    built) and an already-propagated guess trajectory, optionally appends one
    continuation closer, runs the shooter, and returns the ShooterResult
    unwrapped -- no raise on non-convergence, no PeriodicOrbit wrap. Those are
    the wrapper's (correct_as's) concerns; keeping them out lets a continuation
    loop inspect .converged and adapt instead of catching exceptions.

    Spec construction (recipe lookup, base layout, layout/scheme transforms) and
    guess propagation both happen *above* this function now -- it receives the
    finished spec and the propagated trajectory. The trajectory carries the
    System, seed state, and times (the shooter reads them via _pack), so none of
    those live on the spec.

    The closer is appended iff BOTH closer_factory and ref are supplied -- the
    factory to build it, the ref to build it from. Supplying exactly one is a
    caller error. Supplying neither runs an unclosed solve, which is both the
    single-shot (correct_as) path and the continuation bootstrap (member one,
    where the unclosed corank-1 Jacobian's null space seeds the first tangent).

    ref and continuation are independent switches: ref toggles the *closer*,
    continuation toggles the *payload* (ShooterResult.continuation with X and
    DH). Member one needs ref=None but continuation=True -- unclosed, yet its DH
    is wanted -- so they must not be conflated.

    Determinacy (corank) is trusted, not checked here: the caller (correct_as,
    or the continuation setup above the loop) owns asserting the spec is the
    right corank for what it is doing. This function just appends and solves.

    Parameters
    ----------
    spec : SolveSpec
        Finalized, built determinacy spec (terminal constraints only; the
        closer is appended here, not in the spec).
    guess_traj : Trajectory
        Already-propagated initial guess. Carries the System and the seed
        state/times.
    corrector : DifferentialCorrector, optional
        Reused across a march when provided; a default is built otherwise.
    closer_factory : callable, optional
        (ref, n_X) -> FreeVarConstraint. The scheme's Task-2 half, resolved
        above the loop (with any static bits like a pin column bound in).
    ref : ContinuationRef, optional
        Per-step reference state (X_prev, t_hat, ds) the factory consumes.
    continuation : bool, default False
        If True and the solve converges, ShooterResult.continuation carries the
        member's X and unclosed DH (for the next analytic tangent).

    Returns
    -------
    ShooterResult
        Raw outcome; never raises on non-convergence.
    """

    # temporary guard until MS support is established
    if spec.node_specs is not None:
            raise NotImplementedError(f"Continuation does not yet support multiple "
                                        f"shooting, do not provide node_specs.")

    # set corrector as default or input
    corrector = corrector if corrector is not None else DifferentialCorrector()

    # Terminal constraints are loop-invariant; build a fresh list and append the
    # per-step closer to it, never mutating spec.constraints.
    constraints = list(spec.constraints)

    # add continuation constraint if present
    if ref is not None and closer_factory is not None:
        constraints.append(closer_factory(ref, spec.n_X))

    elif ref is not None or closer_factory is not None:
        raise ValueError(
            "closer_factory and ref must be supplied together (a closer needs "
            "both a factory and reference data) or both omitted for an unclosed "
            "solve; got exactly one."
        )
    # neither: unclosed solve, constraints stay as the input set

    solve_kwargs = {
        "free_vars": list(spec.free_vars),
        "constraints": constraints,
    }
    # Pass free_times only when non-empty, matching the corrector's
    # all-fixed-by-default convention.
    if spec.free_times:
        solve_kwargs["free_times"] = list(spec.free_times)

    return corrector.solve(guess_traj, continuation=continuation, **solve_kwargs)


# ===========================================================================
# Public wrapper
# ===========================================================================
def correct_as(
    guess, corrector: DifferentialCorrector | None = None
) -> PeriodicOrbit:
    """
    Correct a guess into a verified member of the family it names.

    Looks up the recipe for guess.recipe, propagates the guess state into a
    half-arc on its System, runs the differential corrector to enforce the
    family's perpendicular-crossing conditions, and returns a verified
    PeriodicOrbit (full arc repropagated, monodromy computed, closure checked).
    Non-symmetric solver recipes are not yet implemented, use available_recipes()
    to see implemented orbit families.

    Parameters
    ----------
    guess : CorrectorGuess
        Validated correction input: state, period estimate, recipe label, and
        System.
    corrector : DifferentialCorrector, optional
        Prebuilt corrector to use. If None, a default corrector is constructed.
        Injecting one lets a continuation loop reuse a single corrector across
        many solves.

    Returns
    -------
    PeriodicOrbit
        The corrected, verified family member.

    Raises
    ------
    NotImplementedError
        If the recipe uses a phase-pinning scheme other than 'symmetry'.
    """
    result = solve_recipe(guess, corrector)

    if result.trajectory is None:
        raise ConvergenceError(
            f"Corrector failed to converge for a {guess.recipe!r} guess; "
            f"the initial guess may be too far from a periodic orbit."
        )

    return PeriodicOrbit(result.trajectory)