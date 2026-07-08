"""
Correction recipe registry and corrector-guess input type.

This module is the lower-level leaf that both the recipe wrapper and any
guess-producing code (the planar seeder, bifurcation targeting, user-built
guesses) depend on. It holds two things:

  1. A registry of correction *recipes* -- the invariant family geometry that
     defines what it means to correct an orbit "as a Lyapunov" or "as a halo":
     which state components are free, which terminal conditions are targeted,
     and how the orbit's phase degeneracy is pinned. Recipes are inert data;
     they carry no algorithm parameters and no problem-specific context.

  2. CorrectorGuess -- the narrow, validating input funnel to the recipe
     wrapper. Many origins (seeder, continuation step, bifurcation orbit, or a
     hand-built state) converge on this one type, which validates the state
     shape, the period, and the recipe label before anything reaches the
     corrector.

Design notes
------------
Recipe entries are stored as specs (a plain constraint dict), not as
constructed corrector-constraint objects, because a continuation scheme must
be able to *modify* the constraint set at solve-assembly time (e.g. period
sampling pins time; other schemes pin or release a geometric freedom). You
cannot cleanly edit a constructed immutable constraint, but you can copy and
modify a spec dict. The registry therefore holds "data describing recipes,"
and the wrapper/scheme layer constructs the actual corrector components at
solve time.

The determinacy layout (which vars are free, which node times are free) is
split by design: the *recipe* owns the invariant family geometry below; the
continuation *scheme* owns a pure transform that edits that layout per step to
keep the shooting system square. Nothing in this module performs that
transform -- it only supplies the invariant base.

Publicness
----------
The recipe entries and the registry object are internal machinery
(underscored). The *label vocabulary* is public, because a user constructing a
CorrectorGuess must be able to discover which recipes exist: use
``available_recipes()``.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


# ===========================================================================
# Recipe entry
# ===========================================================================
class _RecipeEntry(NamedTuple):
    """
    Invariant geometry of a single correction recipe.

    Internal: users never construct or receive one of these. The wrapper
    fetches an entry from the registry and reads its fields to assemble a
    corrector solve.

    Fields
    ------
    free_vars : tuple[str, ...]
        State components the corrector solves for. A fixed sequence, hence a
        tuple (immutable, no reason it should ever be mutated).
    constraint_spec : dict[str, float]
        Terminal target conditions, as a {component: value} spec. Held as a
        plain dict, not a constructed corrector constraint, so a continuation
        scheme can copy-and-modify it at solve-assembly time. Consumers MUST
        treat it as read-only and copy before modifying; the wrapper/scheme
        layer constructs the actual constraint object from a fresh copy.
    phase_pinning : str
        How the orbit's phase (flow-direction) degeneracy is killed. Currently
        only 'symmetry' (perpendicular-crossing formulation, which implies a
        half-period convention). Reserved for a future 'poincare' scheme
        (explicit phase constraint, full-period convention) when non-symmetric
        families (e.g. L4/L5 short/long period) are added.

    Notes
    -----
    free_vars is deeply immutable (a tuple of str). constraint_spec is only
    shallowly protected -- the entry cannot be rebound, but the dict's contents
    can still be mutated in place, so the copy-before-modify contract is
    enforced by discipline (and tests), not by the type. See module docstring.
    """

    free_vars: tuple[str, ...]
    constraint_spec: dict[str, float]
    phase_pinning: str


# ===========================================================================
# Phase-pinning vocabulary
# ===========================================================================
# Single source of truth for the phase-pinning schemes and the period
# convention each one implies. 'symmetry' integrates a half arc to a
# perpendicular crossing; a future 'poincare' scheme would integrate a full
# period with an explicit phase constraint.
_PHASE_PINNING_PERIOD_CONVENTION = {
    "symmetry": "half",
    # "poincare": "full",   # reserved; see module docstring
}


def period_convention_for(phase_pinning: str) -> str:
    """
    Return the period convention ('half' or 'full') implied by a phase-pinning
    scheme.

    The period convention is not an independent choice: it is determined by how
    the phase is pinned. A symmetric (perpendicular-crossing) formulation
    solves a half arc, so its free-time guess is half the full period.

    Parameters
    ----------
    phase_pinning : str
        A phase-pinning scheme name (e.g. 'symmetry').

    Returns
    -------
    str
        'half' or 'full'.

    Raises
    ------
    ValueError
        If phase_pinning is not a recognized scheme.
    """
    try:
        return _PHASE_PINNING_PERIOD_CONVENTION[phase_pinning]
    except KeyError:
        known = sorted(_PHASE_PINNING_PERIOD_CONVENTION)
        raise ValueError(
            f"Unknown phase_pinning scheme {phase_pinning!r}; "
            f"known schemes are {known}."
        )


# ===========================================================================
# Registry
# ===========================================================================
class _RecipeRegistry:
    """
    Immutable-by-access store of correction recipes, keyed by family label.

    Internal machinery. Users do not touch the registry object; they reference
    recipes by label (validated via CorrectorGuess) and discover the label
    vocabulary via the public ``available_recipes()``.

    The registry is the single source of truth for the recipe label
    vocabulary. Label validation (in CorrectorGuess), recipe lookup (in the
    wrapper), and discovery (available_recipes) all derive from it, so no
    parallel list of labels should exist anywhere else.
    """

    def __init__(self, entries: dict[str, _RecipeEntry]):
        # Copy so the internal mapping cannot be mutated through the reference
        # that was passed in.
        self._entries: dict[str, _RecipeEntry] = dict(entries)

    def get(self, label: str) -> _RecipeEntry:
        """
        Return the recipe entry for a family label.

        Parameters
        ----------
        label : str
            Recipe label, e.g. 'lyapunov' or 'halo'.

        Returns
        -------
        _RecipeEntry

        Raises
        ------
        ValueError
            If the label is not a registered recipe. The message enumerates the
            known labels.
        """
        try:
            return self._entries[label]
        except KeyError:
            raise ValueError(
                f"Unknown recipe label {label!r}; "
                f"known recipes are {self.labels()}."
            )

    def labels(self) -> list[str]:
        """Return the sorted list of registered recipe labels."""
        return sorted(self._entries)

    def __contains__(self, label: str) -> bool:
        return label in self._entries


# ---------------------------------------------------------------------------
# The recipe definitions. This is the one place new families are registered.
#
# halo is the planar Lyapunov recipe lifted into 3D: the same perpendicular-
# crossing symmetry (cross y = 0 with velocity purely along y), with the out-
# of-plane pair (z free, vz = 0 targeted) added. Axial / vertical families,
# when added, are further variants of this same crossing template.
# ---------------------------------------------------------------------------
_RECIPES = _RecipeRegistry(
    {
        "lyapunov": _RecipeEntry(
            free_vars=("x", "vy"),
            constraint_spec={"y": 0.0, "vx": 0.0},
            phase_pinning="symmetry",
        ),
        "halo": _RecipeEntry(
            free_vars=("x", "z", "vy"),
            constraint_spec={"y": 0.0, "vx": 0.0, "vz": 0.0},
            phase_pinning="symmetry",
        ),
    }
)


def available_recipes() -> list[str]:
    """
    Return the sorted list of correction recipe labels available for use with
    CorrectorGuess and the recipe wrapper.

    This is the public way to discover which families can be corrected, e.g.
    to know what label to pass to CorrectorGuess.

    Returns
    -------
    list[str]
        Recognized recipe labels, e.g. ['halo', 'lyapunov'].
    """
    return _RECIPES.labels()
