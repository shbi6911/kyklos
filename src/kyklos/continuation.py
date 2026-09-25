"""
Periodic-orbit correction and continuation over the recipe registry.

Two public entry points share one solve pipeline:

correct_as(guess)
    Isolated correction. Given a CorrectorGuess -- a state, a period
    estimate, a recipe label, and the System it belongs to -- it returns a
    verified PeriodicOrbit. The user says "correct this as an L1 halo" and
    gets back a family member with repropagation and closure validation
    done.

march_family(orbit, recipe, ds=...)
    Pseudo-arclength continuation. Given one converged PeriodicOrbit, it
    walks the family at a fixed arclength step and returns the converged
    members as an OrbitFamily.

Vocabulary
----------
Three orthogonal choices define a solve:

- A *recipe* (registry.py) fixes the invariant family geometry: which start
  components are free, which terminal conditions are targeted, and how the
  phase is pinned. It says which family.
- A *layout* (_LAYOUTS) is a corank-preserving transform used by isolated
  correction. It selects which member the corrector converges to by
  choosing which freedom to pin, keeping the system square. period_locked
  is the default; CorrectorGuess.from_seeder_result() selects
  x_amplitude_locked for a planar seed.
- A *scheme* (_SCHEMES) is a corank-opening transform paired with the
  closing constraint that squares the system back up at each step. It is
  what turns an isolated solve into a march. pseudo_arclength is the only
  scheme so far.

Organization
------------
The module reads from building blocks up to their consumers:

1. continuation helpers (_family_tangent and its direction check)
2. CorrectorGuess, the validated input to correct_as
3. layouts, then the SolveSpec and ContinuationRef input records
4. schemes
5. solve_recipe, the atomic operator both entry points call: it takes a
   built SolveSpec and an already-propagated guess trajectory, optionally
   appends one closer, and returns the raw ShooterResult
6. correct_as, then the march machinery and march_family

Spec construction and guess propagation belong to the callers, not to
solve_recipe. One consequence is that the half- vs full-period convention
is a caller obligation. Under symmetry pinning every solve integrates a
half arc to the next perpendicular crossing, so each entry point must hand
solve_recipe a half-arc guess and double the solved duration to recover the
full period. A full-period guess still converges -- the end of a full
period is also a crossing -- but silently, to the wrong multiple.

CorrectorGuess lives here rather than in registry.py because it carries a
System, and the registry stays a dependency-free leaf. This module imports
OrbitFamily; orbit_family imports nothing from here, so no cycle forms.
"""
import numpy as np
from typing import NamedTuple, Callable, Any, TYPE_CHECKING
from dataclasses import dataclass
import warnings

from .registry import (_RECIPES, _RecipeEntry, available_recipes, 
                       period_convention_for)
from .shooter import (DifferentialCorrector, TargetState, ShooterResult,
                      Constraint, ConstraintSpace, FreeVarConstraint,
                      PseudoArclength)
from .periodic_orbit import PeriodicOrbit
from .system import System, SysType, CR3BPSystem
from .orbit_family import OrbitFamily
from .exceptions import ConvergenceError

if TYPE_CHECKING:
    from .trajectory import Trajectory

# Relative tolerance on the seed-mode dot product used by _family_tangent. Below
# this, the seed member is too close to a period extremum to sign
# reliably from dT_dX alone.
_SEED_DOT_RTOL = 1e-8

# ===========================================================================
# Continuation helpers
# ===========================================================================
def _check_direction(direction) -> int:
    """
    Validate a march direction and return it as a plain int in {1, -1}.

    Shared by march_family, which checks before paying for the bootstrap
    solve, and _family_tangent, which consumes the value, so the two
    cannot drift apart.

    Raises
    ------
    TypeError
        If direction is a bool (a subclass of int, rejected so True/False
        are never silently read as 1/0) or otherwise not an integer.
    ValueError
        If direction is not 1 or -1.
    """
    if isinstance(direction, bool) or not isinstance(
        direction, (int, np.integer)
    ):
        raise TypeError(
            f"direction must be an integer, got "
            f"{type(direction).__name__}."
        )
    direction = int(direction)
    if direction not in (1, -1):
        raise ValueError(f"direction must be 1 or -1, got {direction}.")
    return direction


def _family_tangent(
    DH: np.ndarray,
    *,
    prev_t_hat: np.ndarray | None = None,
    dT_dX: np.ndarray | None = None,
    direction: int = 1,
) -> np.ndarray:
    """
    Compute the signed, unit family tangent from an unclosed corank-1
    Jacobian.

    Pure function: DH -> t_hat. Extracts the null direction DH's shape
    forces to exist (see Notes) via SVD, then resolves its sign one of
    two mutually exclusive ways, selected by which of prev_t_hat / dT_dX
    is supplied.

    Continuity mode (prev_t_hat given): flips the raw null vector, if
    needed, so its dot product with the previous step's resolved tangent
    is positive. This is every step after the first -- small ds keeps
    consecutive tangents close, so a positive dot product is the correct
    continuation of the same branch.

    Seed mode (dT_dX given): flips the raw null vector, if needed, so its
    dot product with dT_dX (the gradient of the total period with
    respect to X) has the sign of `direction`. This is member one only,
    where there is no previous tangent to be continuous with; it is safe
    because a fresh seed is not generically sitting at a period extremum,
    so the dot product is generically nonzero and well-signed. See
    Raises for what happens when that assumption fails.

    Parameters
    ----------
    DH : np.ndarray
        Unclosed corrector Jacobian at a converged corank-1 member, shape
        (n_X - 1, n_X) -- ShooterResult.continuation.DH, with the
        X_SPACE closer row already stripped.
    prev_t_hat : np.ndarray, optional
        Previous step's resolved, unit-norm tangent, shape (n_X,).
        Selects continuity mode. Mutually exclusive with dT_dX.
    dT_dX : np.ndarray, optional
        Gradient of the total period with respect to X, shape (n_X,).
        For single shooting, a one-hot vector at the free end-node
        time's column. Selects seed mode. Mutually exclusive with
        prev_t_hat.
    direction : int, default 1
        Target sign, in {1, -1}, for dT_dX . t_hat in seed mode: +1
        seeds a tangent along which the period increases, -1 along
        which it decreases. Consulted only in seed mode; harmlessly
        ignored in continuity mode, where sign is fully determined by
        prev_t_hat.

    Returns
    -------
    np.ndarray
        The signed, unit family tangent, shape (n_X,).

    Raises
    ------
    ValueError
        If neither or both of prev_t_hat / dT_dX are supplied; if DH is
        not 2-D or is not exactly one column wider than it is tall (the
        single-shooting corank-1 shape); if prev_t_hat or dT_dX is not
        shape (n_X,), not finite, or (dT_dX only) identically zero; if
        direction is not in {1, -1}; or, in seed mode, if the null
        direction is nearly orthogonal to dT_dX (the seed sits at or
        near a period extremum, so sign cannot be resolved this way).
    TypeError
        If direction is a bool (a subclass of int, rejected so True/False
        are never silently read as 1/0) or otherwise not an integer.

    Notes
    -----
    DH is always exactly one column wider than it is tall, so it has
    exactly n_X - 1 singular values -- every one of them genuine, none of
    them the tangent's. The tangent is the null direction that rank-
    nullity forces to exist purely from that shape, which is why
    np.linalg.svd is called with full_matrices=True: only the full
    (n_X, n_X) right factor gives back the n_X-th right singular vector
    (Vt's last row) spanning it. The economy SVD (full_matrices=False)
    silently returns one row short, and Vt[-1] would then be the vector
    for DH's smallest *computed* singular value instead -- a real,
    plausible-looking, wrong tangent, not an error.

    Genuine corank-1 nullity is trusted here, not checked: the caller
    (the continuation engine) has already asserted SolveSpec.corank == 1
    before this function runs. Watching the smallest computed singular
    value (S[-1], not zero -- just the smallest actually returned) for a
    drift toward zero relative to the largest is a separate, deferred
    diagnostic (the corank-2 bifurcation monitor), not this function's
    job.
    """
    DH = np.asarray(DH, dtype=float)
    if DH.ndim != 2:
        raise ValueError(f"DH must be 2-D, got shape {DH.shape}.")
    n_rows, n_X = DH.shape
    if n_rows != n_X - 1:
        raise ValueError(
            f"DH must have exactly one more column than row (the "
            f"single-shooting corank-1 shape (n_X - 1, n_X)), got "
            f"shape {DH.shape}."
        )
    if not np.all(np.isfinite(DH)):
        raise ValueError("DH must be finite.")

    if (prev_t_hat is None) == (dT_dX is None):
        raise ValueError(
            "Exactly one of prev_t_hat or dT_dX must be supplied -- "
            "prev_t_hat selects continuity mode (every step after the "
            "first), dT_dX selects seed mode (member one only)."
        )

    direction = _check_direction(direction)

    # DH is (n_X - 1, n_X): only full_matrices=True returns the n_X-th
    # right singular vector (Vt's last row) -- the forced null direction.
    # See Notes.
    _, S, Vt = np.linalg.svd(DH, full_matrices=True)
    v_raw = Vt[-1]

    if prev_t_hat is not None:
        ref = np.asarray(prev_t_hat, dtype=float)
        if ref.shape != (n_X,):
            raise ValueError(
                f"prev_t_hat must have shape ({n_X},) to match DH's "
                f"column count, got shape {ref.shape}."
            )
        if not np.all(np.isfinite(ref)):
            raise ValueError("prev_t_hat must be finite.")

        if np.dot(v_raw, ref) < 0.0:
            v_raw = -v_raw

    else:
        ref = np.asarray(dT_dX, dtype=float)
        if ref.shape != (n_X,):
            raise ValueError(
                f"dT_dX must have shape ({n_X},) to match DH's column "
                f"count, got shape {ref.shape}."
            )
        if not np.all(np.isfinite(ref)):
            raise ValueError("dT_dX must be finite.")

        ref_scale = np.linalg.norm(ref)
        if ref_scale == 0.0:
            raise ValueError("dT_dX must not be identically zero.")

        dot = float(np.dot(v_raw, ref))
        if abs(dot) < _SEED_DOT_RTOL * ref_scale:
            raise ValueError(
                f"Seed sign is degenerate: the null direction is "
                f"nearly orthogonal to dT_dX (dot = {dot:.3e} against "
                f"a scale of {ref_scale:.3e}). The seed member appears "
                f"to sit at, or very near, a period extremum, where "
                f"sign cannot be resolved this way."
            )
        if np.sign(dot) != direction:
            v_raw = -v_raw

    return v_raw

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
    scheme : str or None, optional
        Continuation scheme label, e.g. 'pseudo_arclength'. None (default)
        means an isolated solve: the layout stays square and no closer is
        appended. A non-None label selects the corank-opening transform and
        the closing constraint the continuation engine applies per step, and
        is validated against the scheme registry here so a typo fails at
        guess construction rather than deep inside a march.

        This is the single source of truth for which closer a march uses.
        correct_as rejects a scheme-bearing guess: opening corank without
        closing it would leave an underdetermined solve, and closing it needs
        per-step reference data (X_prev, t_hat, ds) that an isolated
        correction does not have.
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
    scheme : str or None
        Validated continuation scheme label, or None for an isolated solve.
    period_is_half : bool
        The convention flag for ``period``.

    Raises
    ------
    ValueError
        If the state is not a finite (6,) array, the period is not positive and
        finite, the recipe, layout, or scheme label is not registered,
        or the System is not CR3BP.
    """

    __slots__ = ("_state", "_period", "_system",
                 "_recipe", "_layout", "_scheme", "_period_is_half"
    )

    def __init__(
        self,
        state,
        period: float,
        system: System,
        recipe: str,
        layout: str,
        scheme: str | None = None,
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

        # Scheme label: optional, but must be registered when given. See
        # _SCHEMES in this file. Validated here rather than at march setup so
        # a typo fails while the guess is being built, with the guess in hand.
        if scheme is not None and scheme not in _SCHEMES:
            raise ValueError(
                f"Unknown scheme label {scheme!r}; "
                f"known schemes are {available_schemes()}."
            )

        self._state = arr
        self._period = period
        self._system = system
        self._recipe = recipe
        self._layout = layout
        self._scheme = scheme
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
    def scheme(self) -> str | None:
        """
        Validated continuation scheme label, or None for an isolated solve.
        """
        return self._scheme

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

    def replace(self, **changes) -> "CorrectorGuess":
        """
        Return a new CorrectorGuess with the given fields overridden.

        Mirrors dataclasses.replace() for the fields CorrectorGuess actually
        carries: state, period, system, recipe, layout, scheme, and
        period_is_half. CorrectorGuess is not itself a dataclass (it hand-rolls
        __slots__ and read-only properties instead), so dataclasses.replace()
        does not apply to it directly; this method reproduces the same
        functional-update pattern by hand.

        Implementation note: this funnels every call back through __init__
        rather than copying fields itself, so replace() can never drift out of
        sync with __init__'s validation rules -- any check added to __init__
        later is automatically enforced here too.

        Parameters
        ----------
        **changes
            Field names (state, period, system, recipe, layout, scheme,
            period_is_half) mapped to their replacement values. Fields not
            named here are carried over unchanged from self. Unrecognized
            field names raise TypeError rather than being silently ignored or
            forwarded to a confusing __init__ error.

        Returns
        -------
        CorrectorGuess
            A new, independently validated instance. self is untouched.

        Raises
        ------
        TypeError
            If a keyword in **changes does not name a CorrectorGuess field.
        ValueError
            If the resulting field combination fails __init__ validation (see
            CorrectorGuess.__init__).

        Notes
        -----
        replace() re-runs exactly the validation __init__ runs, which checks
        each field independently (state is finite (6,), period is positive,
        system is CR3BP, recipe/layout/scheme are each registered). It does
        NOT check the cross-field consistency that __init__ never checked
        either:

        - ``period`` and ``period_is_half`` are a coupled pair -- the numeric
          value's meaning depends on the flag. ``replace(period=new_period)``
          leaves ``period_is_half`` as it was on self, which is usually right,
          but if the new period comes from a source with the other convention
          (e.g. pulling a half-period estimate onto a guess that was built with
          a full period), pass both together: ``replace(period=...,
          period_is_half=...)``.
        - ``state`` is not re-validated against ``system`` (nor ``layout``
          against ``recipe``'s free_vars, which is checked later, at
          correct_as time). Replacing only one of a coupled set (e.g. state
          without system, or recipe without layout) can produce a guess that
          is individually valid but physically or combinatorially wrong. This
          is the same responsibility direct construction already places on the
          caller (see the state/system note in the class docstring); replace()
          does not add a new gap, but changing one field at a time makes it
          easier to forget the field that should have moved with it.

        Examples
        --------
        Retry a failed correction with a larger period guess, same everything
        else::

            try:
                orbit = correct_as(guess)
            except ConvergenceError:
                guess = guess.replace(period=1.05 * guess.period)
                orbit = correct_as(guess)

        Try the same seed state against a different family::

            halo_guess = guess.replace(recipe="halo", layout="period_locked")
        """
        current = {
            "state": self._state,
            "period": self._period,
            "system": self._system,
            "recipe": self._recipe,
            "layout": self._layout,
            "scheme": self._scheme,
            "period_is_half": self._period_is_half,
        }
        unknown = set(changes) - set(current)
        if unknown:
            raise TypeError(
                f"CorrectorGuess.replace() got unexpected field(s) "
                f"{sorted(unknown)}; valid fields are {sorted(current)}."
            )
        current.update(changes)
        return type(self)(**current)

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
            f"scheme={self._scheme!r}, "
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
        If the recipe uses a phase-pinning scheme other than 'symmetry', or if
        the guess carries a continuation scheme (see Notes).
    ValueError
        If the recipe and layout combine to a non-square solve.
    ConvergenceError
        If the corrector does not converge.
    ClosureError
        If the corrector converges but the mirrored full orbit fails
        periodicity closure. Deliberately allowed to propagate: a half arc
        that converges to the shooter tolerance can still close poorly over
        the full period, and that is a real failure the caller must see.

    Notes
    -----
    A scheme-bearing guess is rejected. A scheme opens corank by one so a
    continuation closer can square it back up, and the closer needs per-step
    reference data (X_prev, t_hat, ds) that an isolated correction has no
    source for -- honoring the scheme here would leave an underdetermined
    solve. Marching a scheme is the continuation engine's job.
    """
    entry = _RECIPES.get(guess.recipe)

    # Only the perpendicular-crossing formulation is implemented. It implies
    # the half-period convention used below; a future 'poincare' pinning would
    # propagate a full period and carry an explicit phase constraint instead.
    if entry.phase_pinning != "symmetry":
        raise NotImplementedError(
            f"correct_as implements the 'symmetry' (perpendicular-crossing) "
            f"phase pinning only; recipe {guess.recipe!r} uses "
            f"{entry.phase_pinning!r}."
        )

    if guess.scheme is not None:
        raise NotImplementedError(
            f"This guess carries the continuation scheme {guess.scheme!r}, "
            f"which opens corank 1 for a closer that correct_as cannot supply "
            f"(it has no previous member, tangent, or step size). Build the "
            f"guess without a scheme for an isolated correction, or march it "
            f"with the continuation engine."
        )

    # Spec construction: recipe -> base layout -> member-selection transform
    # -> built terminal constraints. The layout owns its own constraint_spec
    # copy, and TargetState gets a fresh dict, so the registry entry is never
    # reachable for mutation from here.
    layout = _get_layout(guess.layout)(_base_layout(entry))
    spec = SolveSpec(
        free_vars=layout.free_vars,
        free_times=layout.free_times,
        constraints=(TargetState(dict(layout.constraint_spec)),),
    )

    # solve_recipe trusts the caller on determinacy, so assert it here: an
    # isolated correction must be square. A non-zero corank means the recipe
    # and layout disagree, which the shooter would otherwise absorb into a
    # least-squares or minimum-norm step and quietly return the wrong member.
    if spec.corank != 0:
        raise ValueError(
            f"Recipe {guess.recipe!r} with layout {guess.layout!r} gives a "
            f"corank-{spec.corank} solve ({spec.n_X} free variables against "
            f"{spec.n_rows} constraint rows); an isolated correction requires "
            f"a square system."
        )

    # Symmetry pinning integrates to the next perpendicular crossing, so the
    # guess arc spans a half period. half_period() resolves the guess's own
    # full/half convention.
    guess_traj = guess.system.propagate(
        guess.state, [0.0, guess.half_period()], with_stm=True
    )

    result = solve_recipe(spec, guess_traj, corrector)

    if not result.converged or result.trajectory is None:
        raise ConvergenceError(guess.recipe)

    # Period is inferred, not supplied: both ends of the converged arc are
    # perpendicular x-z crossings, so PeriodicOrbit recognizes the mirror
    # half-orbit, repropagates the full period, and validates closure. That
    # closure check is the second gate -- the corrector tolerance governs the
    # half arc, this governs the whole orbit.
    return PeriodicOrbit(result.trajectory, name=guess.recipe)


# ===========================================================================
# Continuation march
# ===========================================================================
class _MarchSetup(NamedTuple):
    """
    Loop-invariant configuration for one continuation march.

    Built once above the loop by _build_march_setup. A NamedTuple because it
    is an inert bundle of already-built objects: no arrays, no validation
    step of its own, no methods. Consumers read fields by name.

    Fields
    ------
    entry : _RecipeEntry
        The recipe's registry entry, kept for the seed check (which reads
        phase_pinning and constraint_spec).
    spec : SolveSpec
        The corank-1 spec every solve in the march uses, bootstrap included.
    closer_factory : callable
        (ContinuationRef, int) -> FreeVarConstraint; the scheme's closing
        half.
    period_factor : float
        Full period divided by the duration of the solved arc: 2.0 under a
        'half' period convention (symmetry pinning), 1.0 under 'full'.
    """

    entry: _RecipeEntry
    spec: SolveSpec
    closer_factory: Callable[[ContinuationRef, int], FreeVarConstraint]
    period_factor: float


def _build_march_setup(recipe: str, scheme: str) -> _MarchSetup:
    """
    Run the spec pipeline for a march: recipe -> base layout -> scheme
    transform -> built SolveSpec.

    The march's counterpart to the spec construction inside correct_as,
    minus the member-selection layout (see the comment in the body) and
    with a corank-1 check in place of correct_as's corank-0 check.

    Parameters
    ----------
    recipe : str
        Registered recipe label, e.g. 'lyapunov'.
    scheme : str
        Registered continuation scheme label, e.g. 'pseudo_arclength'.

    Returns
    -------
    _MarchSetup

    Raises
    ------
    ValueError
        If recipe or scheme is not registered, or if the pair does not give
        a corank-1 spec.
    NotImplementedError
        If the recipe's phase pinning is not 'symmetry'. The half-arc guess,
        the doubled period, and the seed check all rest on it.
    """
    entry = _RECIPES.get(recipe)
    if entry.phase_pinning != "symmetry":
        raise NotImplementedError(
            f"march_family implements the 'symmetry' (perpendicular-"
            f"crossing) phase pinning only; recipe {recipe!r} uses "
            f"{entry.phase_pinning!r}."
        )
    scheme_entry = _get_scheme(scheme)

    # No member-selection layout. A layout is corank-preserving and exists
    # to choose which member an *isolated* solve lands on; a march selects
    # members by arclength instead. The scheme transform also requires the
    # square, fixed-time base layout as its input.
    layout = scheme_entry.transform(_base_layout(entry))
    spec = SolveSpec(
        free_vars=layout.free_vars,
        free_times=layout.free_times,
        constraints=(TargetState(dict(layout.constraint_spec)),),
    )

    # solve_recipe trusts its caller on determinacy; this is that caller.
    if spec.corank != 1:
        raise ValueError(
            f"Recipe {recipe!r} under scheme {scheme!r} gives a "
            f"corank-{spec.corank} solve ({spec.n_X} free variables "
            f"against {spec.n_rows} constraint rows); a march requires "
            f"corank 1, so that the scheme's single closer squares it."
        )

    convention = period_convention_for(entry.phase_pinning)
    period_factor = 2.0 if convention == "half" else 1.0
    return _MarchSetup(entry, spec, scheme_entry.closer_factory,
                       period_factor)


def _start_state(trajectory: "Trajectory") -> np.ndarray:
    """
    Return a trajectory's start state as a fresh, writeable (6,) array.

    Read through the start node. BoundaryNode.post_state is Optional in the
    base-class signature. A propagated trajectory's start node always
    carries one, but the None branch is not decoration:
    np.array(None, dtype=float) is a silent 0-d NaN, not an error.
    """
    state = trajectory.start_node.post_state
    if state is None:
        raise ValueError("Trajectory start node carries no post_state.")
    return np.array(state, dtype=float)


def _prepare_seed(state: np.ndarray, entry: _RecipeEntry,
                  tol: float) -> np.ndarray:
    """
    Check a seed state against the recipe's phase pinning, and snap it on.

    The corrector holds every start component outside free_vars at its
    guess value, and the trivial predictor hands those values from member
    to member unchanged. Whatever the seed carries in its fixed components,
    the whole family carries. So the seed's phase is a precondition of the
    march, not something the bootstrap solve will repair.

    Dispatches on phase_pinning, so a future non-symmetric recipe adds its
    own branch without touching this one.

    'symmetry': the half arc starts and ends on a perpendicular x-z
    crossing, so the seed must meet the same targets the corrector enforces
    at the end of the arc (y = 0, vx = 0, plus vz = 0 for halo). A seed
    within tol is then snapped exactly onto them. That matters: the mirror
    symmetry maps a start offset (y0, vx0) to (-y0, -vx0) at the end of the
    period, so an unsnapped offset shows up doubled in every member's
    closure residual -- enough, near tol, to fail OrbitFamily's closure
    check on members that are otherwise fine.

    Parameters
    ----------
    state : np.ndarray
        Seed start state, shape (6,).
    entry : _RecipeEntry
        The recipe being marched.
    tol : float
        Absolute tolerance on each pinning condition.

    Returns
    -------
    np.ndarray
        The snapped seed state, shape (6,). A new array.

    Raises
    ------
    ValueError
        If the seed misses a pinning condition by more than tol.
    NotImplementedError
        For a phase pinning other than 'symmetry'.
    """
    if entry.phase_pinning == "symmetry":
        crossing = TargetState(dict(entry.constraint_spec))
        residual = crossing.residual(state, state)
        worst = float(np.max(np.abs(residual)))
        if worst > tol:
            raise ValueError(
                f"Seed orbit does not start on a perpendicular x-z "
                f"crossing: the targets {entry.constraint_spec} are missed "
                f"by up to {worst:.3e}, against a tolerance of {tol:.3e}. "
                f"A symmetry-pinned march needs a seed whose trajectory "
                f"starts at the crossing."
            )
        # TargetState is a selection: its Jacobian is rows of the identity,
        # so J^T r moves exactly the targeted components by exactly their
        # residuals. One Newton step onto a linear constraint is exact.
        jac = crossing.jacobian_tf(state, state)
        return state - jac.T @ residual

    raise NotImplementedError(
        f"No seed check is implemented for phase_pinning "
        f"{entry.phase_pinning!r}."
    )


def _period_gradient(spec: SolveSpec, period_factor: float) -> np.ndarray:
    """
    Gradient of the full period with respect to X, for single shooting.

    Single-shooting X is [free start components | free boundary times].
    Under a period-opening scheme the only free time is the end node, whose
    value is the solved arc's duration; the full period is period_factor
    times that. So dT/dX is period_factor in that one column, zero
    elsewhere. Only its sign reaches the tangent (seed mode compares signs),
    but the honest magnitude costs nothing.

    Raises
    ------
    ValueError
        If the spec does not free exactly one boundary time -- the layout
        this function reads the period column from.
    """
    if len(spec.free_times) != 1:
        raise ValueError(
            f"Seeding the march direction needs exactly one free boundary "
            f"time (the end node, i.e. the period), got free_times="
            f"{spec.free_times}."
        )
    grad = np.zeros(spec.n_X)
    grad[len(spec.free_vars)] = period_factor
    return grad


def march_family(
    orbit: PeriodicOrbit,
    recipe: str,
    *,
    ds: float,
    scheme: str = "pseudo_arclength",
    n_steps: int = 100,
    direction: int = 1,
    corrector: DifferentialCorrector | None = None,
    targeter: Any | None = None,
) -> OrbitFamily:
    """
    March a family of periodic orbits by pseudo-arclength continuation.

    Takes one converged member and walks the family from it at a fixed
    arclength step, returning every converged member as an OrbitFamily.
    Single shooting, constant step, trivial predictor (each solve is seeded
    with the previous member's trajectory), analytic tangent from the null
    space of the unclosed corrector Jacobian.

    The march has a loop-and-a-half shape. The bootstrap re-solves the seed
    once, unclosed, under the march's own corank-1 spec: that yields the
    seed's X and DH in the march's layout, and gates the whole march -- a
    seed that will not re-converge raises. Every later step is a closed
    solve, identical in form.

    Parameters
    ----------
    orbit : PeriodicOrbit
        Converged seed member. Its trajectory must start on the recipe's
        pinning plane (a perpendicular x-z crossing for the symmetric
        recipes); see Raises. Its System is the one the family is marched
        and returned in.
    recipe : str
        Family recipe label, e.g. 'lyapunov' or 'halo'.
    ds : float
        Arclength step, > 0. Distance along the unit tangent in X-space,
        where X mixes state components and the half period, so it has no
        single physical unit.
    scheme : str, optional
        Continuation scheme label. Default 'pseudo_arclength'.
    n_steps : int, optional
        Number of arclength steps to take beyond the bootstrap. On complete
        success the family has n_steps + 1 members; n_steps = 0 returns the
        re-solved seed alone. Default 100.
    direction : {1, -1}, optional
        Initial marching direction: +1 toward increasing period, -1 toward
        decreasing. Sets the sign of the first tangent only; tangent
        continuity carries it after that. Default 1.
    corrector : DifferentialCorrector, optional
        Corrector reused for every solve in the march. If None, a default
        is built.
    targeter : None
        Reserved for a future stop-condition interface. Must be None.

    Returns
    -------
    OrbitFamily
        Member 0 is the re-solved seed (step_sizes[0] = 0.0), followed by
        each converged step in order. The marching System is attached, so
        propagation-backed family methods do not recompile it.

    Raises
    ------
    TypeError
        If orbit is not a PeriodicOrbit or its System is not CR3BP; if
        n_steps is not an integer (bools rejected); if direction is a bool
        or not an integer.
    ValueError
        If ds is not positive and finite; if n_steps < 0; if direction is
        not 1 or -1; if the recipe or scheme is not registered or the pair
        is not corank 1; if the seed does not start on the pinning plane
        within orbit.tol; or if the seed sits at a period extremum, where
        the first tangent's sign cannot be seeded from the period.
    NotImplementedError
        If targeter is not None, or the recipe's phase pinning is not
        'symmetry'.
    ConvergenceError
        If the bootstrap solve does not converge. A march that cannot
        re-converge its own seed has no first member, and OrbitFamily
        cannot be empty.

    Warns
    -----
    UserWarning
        If a march step fails to converge. The march stops there and the
        members converged so far are returned; the warning carries the
        step number and the corrector's abort reason.

    Notes
    -----
    Recorded periods are full periods: the solved arc is a half period
    under symmetry pinning, and its duration is doubled on the way into the
    family.

    The seed is snapped exactly onto its pinning conditions before the
    bootstrap (see _prepare_seed), so member 0 can differ from the input
    orbit by that snap as well as by the bootstrap's own minimum-norm
    correction.
    """
    # ----- Argument checks: all before the first solve -----
    if targeter is not None:
        raise NotImplementedError(
            "targeter is reserved for a future stop-condition interface "
            "and is not implemented; pass None."
        )

    ds = float(ds)
    if not np.isfinite(ds) or ds <= 0.0:
        raise ValueError(f"ds must be a positive finite step, got {ds}.")

    if isinstance(n_steps, bool) or not isinstance(
        n_steps, (int, np.integer)
    ):
        raise TypeError(
            f"n_steps must be an integer, got {type(n_steps).__name__}."
        )
    n_steps = int(n_steps)
    if n_steps < 0:
        raise ValueError(f"n_steps must be non-negative, got {n_steps}.")

    direction = _check_direction(direction)

    if not isinstance(orbit, PeriodicOrbit):
        raise TypeError(
            f"orbit must be a PeriodicOrbit, got {type(orbit).__name__}."
        )
    # A PeriodicOrbit already guarantees CR3BP (by base_type value). The
    # isinstance check is what gives the header reads at the end --
    # distance, mass_ratio, secondary_body -- their CR3BPSystem types.
    system = orbit.system
    if not isinstance(system, CR3BPSystem):
        raise TypeError(
            f"march_family requires a CR3BP system, got "
            f"{type(system).__name__}."
        )

    setup = _build_march_setup(recipe, scheme)
    seed = _prepare_seed(_start_state(orbit.trajectory), setup.entry,
                         orbit.tol)
    corrector = corrector if corrector is not None else DifferentialCorrector()

    # ----- Member table, accumulated as plain lists -----
    states: list[np.ndarray] = []
    periods: list[float] = []
    iterations: list[int] = []
    residuals: list[float] = []
    step_sizes: list[float] = []

    def record(result: ShooterResult, trajectory: "Trajectory",
               step: float) -> None:
        # One row per converged member, taken from the solve that produced
        # it, so every column in a row describes the same point.
        states.append(_start_state(trajectory))
        periods.append(setup.period_factor * trajectory.duration)
        iterations.append(result.iterations)
        residuals.append(result.final_residual)
        step_sizes.append(step)

    # ----- Bootstrap: re-solve the seed, unclosed, in the march layout ---
    guess = system.propagate(
        seed, [0.0, orbit.period / setup.period_factor], with_stm=True
    )
    result = solve_recipe(setup.spec, guess, corrector, continuation=True)
    if (not result.converged or result.trajectory is None
            or result.continuation is None):
        raise ConvergenceError(
            recipe,
            message=(
                f"march_family bootstrap failed: the seed orbit did not "
                f"re-converge under the march's corank-1 spec "
                f"(abort_reason={result.abort_reason!r}, final residual "
                f"{result.final_residual:.3e} after {result.iterations} "
                f"iterations). The seed may not be a {recipe!r} member, or "
                f"the corrector tolerance may be tighter than the seed was "
                f"converged to."
            ),
        )
    trajectory = result.trajectory
    record(result, trajectory, 0.0)
    X_prev = result.continuation.X
    t_hat = _family_tangent(
        result.continuation.DH,
        dT_dX=_period_gradient(setup.spec, setup.period_factor),
        direction=direction,
    )

    # ----- March: closed solves at constant ds -----
    for k in range(1, n_steps + 1):
        ref = ContinuationRef(X_prev=X_prev, t_hat=t_hat, ds=ds)
        # Trivial predictor: the previous member's trajectory is the guess.
        result = solve_recipe(
            setup.spec, trajectory, corrector,
            closer_factory=setup.closer_factory, ref=ref,
            continuation=True,
        )
        if (not result.converged or result.trajectory is None
                or result.continuation is None):
            warnings.warn(
                f"march_family stopped early: step {k} of {n_steps} did "
                f"not converge (abort_reason={result.abort_reason!r}, "
                f"final residual {result.final_residual:.3e} after "
                f"{result.iterations} iterations). Returning the "
                f"{len(periods)} member(s) converged so far.",
                UserWarning,
                stacklevel=2,
            )
            break

        trajectory = result.trajectory
        record(result, trajectory, ds)
        X_prev = result.continuation.X
        t_hat = _family_tangent(result.continuation.DH, prev_t_hat=t_hat)

    # ----- Assemble -----
    family = OrbitFamily(
        initial_states=np.array(states),
        periods=periods,
        iterations=iterations,
        final_residuals=residuals,
        step_sizes=step_sizes,
        primary_body=system.primary_body,
        secondary_body=system.secondary_body,
        distance=system.distance,
        mu=system.mass_ratio,
        recipe=recipe,
        scheme=scheme,
        free_vars=setup.spec.free_vars,
        free_times=setup.spec.free_times,
        node_specs=setup.spec.node_specs,
    )
    family.attach_system(system)
    return family
