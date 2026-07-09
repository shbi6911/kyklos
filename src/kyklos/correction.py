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
from typing import NamedTuple

from .registry import _RECIPES, _RecipeEntry, available_recipes
from .shooter import DifferentialCorrector, TargetState
from .periodic_orbit import PeriodicOrbit
from .system import System, SysType
from .exceptions import ConvergenceError

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
    return layout._replace(free_vars=free_vars, free_times=[1])


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
    recipe = _RECIPES.get(guess.recipe)

    # Standalone correction handles only the symmetric (perpendicular-crossing)
    # families. Fail before doing any propagation, not after.
    if recipe.phase_pinning != "symmetry":
        raise NotImplementedError(
            f"correct_as currently supports only symmetric recipes; recipe "
            f"{guess.recipe!r} uses phase_pinning={recipe.phase_pinning!r}."
        )

    layout = _base_layout(recipe)

    # transform the _SolverLayout according to the 'layout' modifier
    transform = _get_layout(guess.layout)
    layout = transform(layout)

    # --- SEAM: continuation scheme transform applies here ---------------------
    # A scheme would edit `layout` (pin a var, free a node time, append an
    # X-aware closing constraint) before assembly. This may coincide with the
    # above transform in some cases.
    # -------------------------------------------------------------------------

    corrector = corrector if corrector is not None else DifferentialCorrector()

    # Propagate the guess for appropriate period with STM: the corrector's Jacobian
    # needs the state-transition matrix along the arc.

    guess_arc = guess.system.propagate(
        guess.state, [0.0, guess.half_period()], with_stm=True
    )

    # Build corrector constraints from the (copied) spec at solve time; the
    # registry holds inert specs, not constructed constraint objects.
    constraints = [TargetState(layout.constraint_spec)]

    solve_kwargs = {
        "free_vars": list(layout.free_vars),
        "constraints": constraints,
    }
    # Pass free_times only when non-empty, matching the corrector's
    # all-fixed-by-default convention.
    if layout.free_times:
        solve_kwargs["free_times"] = list(layout.free_times)

    corrected_arc = corrector.solve(guess_arc, **solve_kwargs)

    if corrected_arc.trajectory is None:
        raise ConvergenceError(
            f"Corrector failed to converge for a {guess.recipe!r} guess; "
            f"the initial guess may be too far from a periodic orbit."
        )

    return PeriodicOrbit(corrected_arc.trajectory)