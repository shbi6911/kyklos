"""
Test suite for the determinacy layer of the correction ecosystem.

Covers the pieces that sit between a recipe and a corrector solve, all of
which are pure data structures or pure functions on them:

- _base_layout and the _SolveLayout it produces
- the layout registry (_LAYOUTS): corank-PRESERVING member-selection
  transforms
- the scheme registry (_SCHEMES): corank-OPENING continuation transforms
  paired with their closer factories
- SolveSpec: the built, loop-invariant determinacy record and its corank
  arithmetic
- ContinuationRef: the per-step predictor record the engine hands to a
  closer factory

The organizing idea is the corank contract, which is what the two separate
registries exist to protect. A layout transform trades one freedom for
another and leaves the system square (corank 0), so it can be solved
standalone. A scheme transform releases a freedom without compensating
(corank 1), so the solve is underdetermined by exactly one and a single
closing row squares it back up. Open without closing and the solve is
underdetermined; close without opening and it is overdetermined. Both
invariants are tested registry-wide -- parametrized over every registered
entry crossed with every registered recipe -- so a future natural-parameter
scheme or a new family is covered the moment it is registered.

Nothing here propagates. There is no System, no trajectory, and no corrector
anywhere in this file; every assertion is on tuple lengths, dict contents,
array flags, and raised exceptions. That is deliberate -- these contracts are
settled and cheap to check, and checking them here means an end-to-end
continuation failure has this whole layer already excluded.

solve_recipe is NOT covered here. Its closer_factory signature is the least
settled interface in the module and is expected to move when the continuation
engine is written; testing it now would pin a design still in flux.
"""

import dataclasses

import numpy as np
import pytest

from kyklos.correction import (
    SolveSpec,
    ContinuationRef,
    _SolveLayout,
    _base_layout,
    _get_layout,
    _get_scheme,
    _period_locked,
    _x_amplitude_locked,
    _free_period,
    _arclength_closer,
    _LAYOUTS,
    _SCHEMES,
    available_layouts,
    available_schemes,
)
from kyklos.registry import _RECIPES, available_recipes
from kyklos.shooter import (
    ConstraintSpace,
    PseudoArclength,
    TargetState,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _spec_from_layout(layout: _SolveLayout) -> SolveSpec:
    """
    Build a SolveSpec from a layout the way the wrapper does.

    Mirrors correct_as: the terminal constraint is constructed from a fresh
    copy of the layout's constraint_spec, and nothing else is added. Used to
    put a layout's corank on the scale.
    """
    return SolveSpec(
        free_vars=layout.free_vars,
        free_times=layout.free_times,
        constraints=(TargetState(dict(layout.constraint_spec)),),
    )


def _corank(layout: _SolveLayout) -> int:
    """Corank of a layout once built into a spec."""
    return _spec_from_layout(layout).corank


def _ref(n: int = 3, ds: float = 0.1) -> ContinuationRef:
    """A well-formed ContinuationRef of width n."""
    t_hat = np.zeros(n)
    t_hat[0] = 1.0
    return ContinuationRef(X_prev=np.arange(n, dtype=float),
                           t_hat=t_hat, ds=ds)


ALL_RECIPES = available_recipes()


# ===========================================================================
# _base_layout / _SolveLayout
# ===========================================================================

class TestBaseLayout:
    """The inert starting point every transform is applied to."""

    @pytest.mark.parametrize("label", ALL_RECIPES)
    def test_mirrors_the_recipe_geometry(self, label):
        entry = _RECIPES.get(label)
        layout = _base_layout(entry)

        assert layout.free_vars == entry.free_vars
        assert layout.constraint_spec == entry.constraint_spec

    @pytest.mark.parametrize("label", ALL_RECIPES)
    def test_fixes_all_node_times(self, label):
        """Standalone correction frees no times; a scheme adds them."""
        assert _base_layout(_RECIPES.get(label)).free_times == ()

    @pytest.mark.parametrize("label", ALL_RECIPES)
    def test_constraint_spec_is_an_owned_copy(self, label):
        """
        The registry entry is shared inert data and must never be mutated
        through a layout. The recipe docstring makes this a discipline
        contract enforced by tests, so this is that test.
        """
        entry = _RECIPES.get(label)
        original = dict(entry.constraint_spec)
        layout = _base_layout(entry)

        layout.constraint_spec["y"] = 99.0
        assert entry.constraint_spec == original
        assert _RECIPES.get(label).constraint_spec == original

    @pytest.mark.parametrize("label", ALL_RECIPES)
    def test_base_layout_is_square(self, label):
        """
        Every recipe's base layout is a solvable standalone problem: as many
        free variables as terminal rows. This is the premise both registries
        build on, so it is asserted at the source rather than assumed.
        """
        assert _corank(_base_layout(_RECIPES.get(label))) == 0

    def test_is_immutable(self):
        """_SolveLayout is a NamedTuple; transforms use _replace."""
        layout = _base_layout(_RECIPES.get("lyapunov"))
        with pytest.raises(AttributeError):
            layout.free_vars = ("x",)          # type: ignore[misc]


# ===========================================================================
# Layout registry: corank-PRESERVING transforms
# ===========================================================================

class TestLayoutRegistry:
    """Discovery and lookup for the member-selection vocabulary."""

    def test_available_layouts_matches_the_registry(self):
        assert available_layouts() == sorted(_LAYOUTS)

    def test_available_layouts_is_sorted(self):
        assert available_layouts() == sorted(available_layouts())

    @pytest.mark.parametrize("label", sorted(_LAYOUTS))
    def test_get_layout_returns_the_registered_transform(self, label):
        assert _get_layout(label) is _LAYOUTS[label]

    def test_unknown_label_raises_and_enumerates(self):
        with pytest.raises(ValueError, match="Unknown layout label") as exc:
            _get_layout("no_such_layout")
        for known in available_layouts():
            assert known in str(exc.value)


class TestLayoutTransforms:
    """Behavior of the individual member-selection transforms."""

    def test_period_locked_is_the_identity(self):
        layout = _base_layout(_RECIPES.get("lyapunov"))
        assert _period_locked(layout) is layout

    def test_x_amplitude_locked_trades_x_for_the_end_time(self):
        layout = _base_layout(_RECIPES.get("lyapunov"))
        out = _x_amplitude_locked(layout)

        assert "x" not in out.free_vars
        assert out.free_vars == ("vy",)
        assert out.free_times == (1,)

    def test_x_amplitude_locked_preserves_remaining_order(self):
        """Only x is dropped; the rest keep the recipe's ordering."""
        layout = _base_layout(_RECIPES.get("halo"))
        assert _x_amplitude_locked(layout).free_vars == ("z", "vy")

    def test_x_amplitude_locked_does_not_mutate_its_input(self):
        layout = _base_layout(_RECIPES.get("lyapunov"))
        before = layout

        _x_amplitude_locked(layout)
        assert layout == before
        assert layout.free_times == ()

    def test_x_amplitude_locked_rejects_a_recipe_without_free_x(self):
        """The layout/recipe compatibility guard."""
        layout = _SolveLayout(free_vars=("vy",), free_times=(),
                              constraint_spec={"y": 0.0})
        with pytest.raises(ValueError, match="x is not in the recipe"):
            _x_amplitude_locked(layout)

    @pytest.mark.parametrize("recipe", ALL_RECIPES)
    @pytest.mark.parametrize("layout_label", sorted(_LAYOUTS))
    def test_every_layout_preserves_corank(self, layout_label, recipe):
        """
        The registry-wide invariant that separates _LAYOUTS from _SCHEMES: a
        member-selection transform trades one freedom for another and leaves
        the system square, so the result is solvable standalone. A new layout
        that opened corank would be caught here the moment it is registered.
        """
        base = _base_layout(_RECIPES.get(recipe))
        assert _corank(_get_layout(layout_label)(base)) == 0


# ===========================================================================
# Scheme registry: corank-OPENING transforms and their closers
# ===========================================================================

class TestSchemeRegistry:
    """Discovery and lookup for the continuation vocabulary."""

    def test_available_schemes_matches_the_registry(self):
        assert available_schemes() == sorted(_SCHEMES)

    def test_pseudo_arclength_is_registered(self):
        assert "pseudo_arclength" in available_schemes()

    @pytest.mark.parametrize("label", sorted(_SCHEMES))
    def test_get_scheme_returns_the_registered_entry(self, label):
        assert _get_scheme(label) is _SCHEMES[label]

    @pytest.mark.parametrize("label", sorted(_SCHEMES))
    def test_entry_fields_are_named_not_positional(self, label):
        """
        _SchemeEntry is a NamedTuple specifically so the two halves cannot be
        swapped by positional unpacking. Read them by name.
        """
        entry = _get_scheme(label)
        assert callable(entry.transform)
        assert callable(entry.closer_factory)

    def test_unknown_label_raises_and_enumerates(self):
        with pytest.raises(ValueError, match="Unknown scheme label") as exc:
            _get_scheme("no_such_scheme")
        for known in available_schemes():
            assert known in str(exc.value)


class TestSchemeCorankContract:
    """
    The paired invariant: every scheme opens exactly one and closes exactly
    one. Parametrized over the whole registry so a future natural-parameter
    scheme inherits the check on registration.
    """

    @pytest.mark.parametrize("recipe", ALL_RECIPES)
    @pytest.mark.parametrize("scheme_label", sorted(_SCHEMES))
    def test_transform_opens_corank_by_exactly_one(self, scheme_label,
                                                   recipe):
        base = _base_layout(_RECIPES.get(recipe))
        opened = _get_scheme(scheme_label).transform(base)

        assert _corank(base) == 0
        assert _corank(opened) == 1

    @pytest.mark.parametrize("scheme_label", sorted(_SCHEMES))
    def test_closer_contributes_exactly_one_row(self, scheme_label):
        """One opened freedom, one closing row: the halves must agree."""
        closer = _get_scheme(scheme_label).closer_factory(_ref(n=3), 3)
        assert closer.n_rows == 1

    @pytest.mark.parametrize("scheme_label", sorted(_SCHEMES))
    def test_closer_is_a_free_variable_constraint(self, scheme_label):
        """
        A closer constrains X directly, not the propagated final state. It is
        also why SolveSpec refuses it: a FREEVAR row in the spec would make
        corank miscount.
        """
        closer = _get_scheme(scheme_label).closer_factory(_ref(n=3), 3)
        assert closer.space is ConstraintSpace.FREEVAR


class TestFreePeriod:
    """The corank-opening half of pseudo-arclength."""

    def test_frees_the_end_node_time(self):
        layout = _base_layout(_RECIPES.get("lyapunov"))
        assert _free_period(layout).free_times == (1,)

    def test_leaves_free_vars_untouched(self):
        """
        This is the whole difference from _x_amplitude_locked, which frees
        the same time but pays for it by dropping x. Here the extra freedom
        is the point.
        """
        layout = _base_layout(_RECIPES.get("lyapunov"))
        assert _free_period(layout).free_vars == layout.free_vars

    def test_leaves_the_constraint_spec_untouched(self):
        layout = _base_layout(_RECIPES.get("halo"))
        assert _free_period(layout).constraint_spec == layout.constraint_spec

    def test_does_not_mutate_its_input(self):
        layout = _base_layout(_RECIPES.get("lyapunov"))
        _free_period(layout)
        assert layout.free_times == ()

    def test_assumes_a_single_arc_end_node_index(self):
        """
        The freed index is hard-coded to 1, documented as a two-node (start,
        end) assumption matching what the planar seeder produces. Pinned
        explicitly so multi-arc work trips this test rather than silently
        freeing the wrong node's time.
        """
        for label in ALL_RECIPES:
            assert _free_period(_base_layout(_RECIPES.get(label))
                                ).free_times == (1,)

    def test_refuses_to_stack_on_an_already_opened_layout(self):
        """
        Stacking two openings would give corank 2, which one closer cannot
        square. The composition below is the realistic mistake: reaching for
        a continuation scheme on top of the layout the seeder defaults to.
        """
        layout = _x_amplitude_locked(_base_layout(_RECIPES.get("lyapunov")))
        with pytest.raises(ValueError, match="already frees"):
            _free_period(layout)

    def test_error_names_the_offending_free_times(self):
        layout = _base_layout(_RECIPES.get("lyapunov"))._replace(
            free_times=(1,)
        )
        with pytest.raises(ValueError, match=r"free_times=\(1,\)"):
            _free_period(layout)

    def test_stacks_cleanly_on_the_identity_layout(self):
        """period_locked leaves times fixed, so it is a valid base."""
        base = _base_layout(_RECIPES.get("lyapunov"))
        assert _corank(_free_period(_period_locked(base))) == 1


class TestArclengthCloser:
    """
    The closer half. Deliberately thin: the factory's (ref, n_X) call shape
    is the one part of this layer that may move when the continuation engine
    is written, so these assert what the closer IS rather than how it is
    called, and survive a signature change as a one-line edit.
    """

    def test_returns_a_pseudo_arclength(self):
        assert isinstance(_arclength_closer(_ref(), 3), PseudoArclength)

    def test_carries_the_reference_data_through(self):
        ref = _ref(n=4, ds=0.25)
        closer = _arclength_closer(ref, 4)

        # Residual at X_prev is -ds by construction: t_hat . 0 - ds.
        assert closer.residual(ref.X_prev) == pytest.approx(-ref.ds)
        # And the Jacobian row is the tangent itself.
        assert closer.jacobian_X(ref.X_prev) == pytest.approx(
            ref.t_hat.reshape(1, -1)
        )

    def test_sizes_itself_from_t_hat_not_from_n_X(self):
        """
        n_X is documented as deliberately unused -- PseudoArclength sizes
        itself from t_hat, and the argument exists only so a future one-hot
        pin closer can share the signature. A wrong n_X must therefore change
        nothing, which is what stops a future scheme author from assuming the
        argument is load-bearing.
        """
        ref = _ref(n=4)
        good = _arclength_closer(ref, 4)
        wrong = _arclength_closer(ref, 999)

        assert wrong.jacobian_X(ref.X_prev).shape == (1, 4)
        assert wrong.jacobian_X(ref.X_prev) == pytest.approx(
            good.jacobian_X(ref.X_prev)
        )

    def test_builds_a_fresh_constraint_each_call(self):
        """
        Called once per step, so the step's data is baked into a new
        immutable constraint rather than mutated onto a persistent one. Two
        refs must not share state.
        """
        ref_a = _ref(n=3, ds=0.1)
        ref_b = _ref(n=3, ds=0.5)
        closer_a = _arclength_closer(ref_a, 3)
        closer_b = _arclength_closer(ref_b, 3)

        assert closer_a is not closer_b
        assert closer_a.residual(ref_a.X_prev) == pytest.approx(-0.1)
        assert closer_b.residual(ref_b.X_prev) == pytest.approx(-0.5)


# ===========================================================================
# SolveSpec
# ===========================================================================

class TestSolveSpecConstruction:
    """Normalization and validation at __post_init__."""

    def test_freezes_collection_fields_to_tuples(self):
        """
        The layout hands free_times in as a list. The annotations declare the
        STORED types, which __post_init__ coerces to, so passing the looser
        input this test exists to exercise is a deliberate annotation
        violation.
        """
        spec = SolveSpec(
            free_vars=["x", "vy"],                 # type: ignore[arg-type]
            free_times=[1],                        # type: ignore[arg-type]
            constraints=[TargetState({"y": 0.0})],  # type: ignore[arg-type]
        )

        assert isinstance(spec.free_vars, tuple)
        assert isinstance(spec.free_times, tuple)
        assert isinstance(spec.constraints, tuple)

    def test_constraints_cannot_be_appended_in_place(self):
        """
        The spec is loop-invariant across a march; solve_recipe appends the
        closer to a fresh list. A tuple makes the alternative impossible.
        """
        spec = _spec_from_layout(_base_layout(_RECIPES.get("lyapunov")))
        with pytest.raises(AttributeError):
            spec.constraints.append(TargetState({"z": 0.0}))  # type: ignore

    def test_is_frozen(self):
        spec = _spec_from_layout(_base_layout(_RECIPES.get("lyapunov")))
        with pytest.raises(dataclasses.FrozenInstanceError):
            spec.free_vars = ("x",)            # type: ignore[misc]

    def test_node_specs_defaults_to_none(self):
        assert _spec_from_layout(
            _base_layout(_RECIPES.get("lyapunov"))
        ).node_specs is None

    def test_node_specs_must_be_a_dict_or_none(self):
        with pytest.raises(TypeError, match="node_specs must be a dict"):
            SolveSpec(free_vars=("x",), free_times=(),
                      constraints=(TargetState({"y": 0.0}),),
                      node_specs=["not", "a", "dict"])  # type: ignore

    def test_rejects_a_free_variable_constraint(self):
        """
        The closer belongs to solve_recipe, not the spec. A FREEVAR row here
        would be counted as terminal and corank would silently miscount.
        """
        closer = PseudoArclength(np.zeros(2), np.array([1.0, 0.0]), 0.1)
        with pytest.raises(ValueError, match="must be terminal constraints"):
            SolveSpec(free_vars=("x", "vy"), free_times=(),
                      constraints=(TargetState({"y": 0.0}), closer))

    def test_accepts_an_empty_constraint_set(self):
        """Degenerate but well-formed: no rows, so corank is just n_X."""
        spec = SolveSpec(free_vars=("x", "vy"), free_times=(),
                         constraints=())
        assert spec.n_rows == 0
        assert spec.corank == 2


class TestSolveSpecCounts:
    """n_X, n_rows and the corank arithmetic on top of them."""

    def test_n_X_counts_free_vars_plus_free_times(self):
        spec = SolveSpec(free_vars=("x", "vy"), free_times=(1,),
                         constraints=(TargetState({"y": 0.0}),))
        assert spec.n_X == 3

    def test_n_rows_sums_across_constraints(self):
        spec = SolveSpec(
            free_vars=("x", "vy"), free_times=(),
            constraints=(TargetState({"y": 0.0}),
                         TargetState({"vx": 0.0})),
        )
        assert spec.n_rows == 2

    @pytest.mark.parametrize("label", ALL_RECIPES)
    def test_corank_zero_is_the_bootstrap_case(self, label):
        assert _corank(_base_layout(_RECIPES.get(label))) == 0

    @pytest.mark.parametrize("label", ALL_RECIPES)
    def test_corank_one_is_the_continuation_case(self, label):
        opened = _free_period(_base_layout(_RECIPES.get(label)))
        assert _corank(opened) == 1

    def test_corank_can_report_an_overdetermined_spec(self):
        """
        Not raised, just reported negative: the loop's `== 1` check is what
        rejects it, and a bare arithmetic property should not editorialize.
        """
        spec = SolveSpec(
            free_vars=("x",), free_times=(),
            constraints=(TargetState({"y": 0.0, "vx": 0.0}),),
        )
        assert spec.corank == -1

    def test_corank_refuses_multiple_shooting(self):
        """
        MS adds junction columns and interior-defect rows that this
        arithmetic does not model, so it raises rather than counting wrong.
        """
        spec = SolveSpec(free_vars=("x", "vy"), free_times=(),
                         constraints=(TargetState({"y": 0.0}),),
                         node_specs={"1": "junction"})
        with pytest.raises(NotImplementedError, match="multiple-shooting"):
            spec.corank

    def test_n_X_and_n_rows_still_work_under_node_specs(self):
        """Only corank is blocked; the raw counts remain meaningful."""
        spec = SolveSpec(free_vars=("x", "vy"), free_times=(1,),
                         constraints=(TargetState({"y": 0.0}),),
                         node_specs={"1": "junction"})
        assert spec.n_X == 3
        assert spec.n_rows == 1


# ===========================================================================
# ContinuationRef
# ===========================================================================

class TestContinuationRefValidation:
    """__post_init__ checks: true of any predictor state, closer-agnostic."""

    def test_accepts_a_well_formed_record(self):
        ref = _ref(n=6, ds=1e-3)
        assert ref.X_prev.shape == (6,)
        assert ref.ds == pytest.approx(1e-3)

    def test_rejects_a_two_dimensional_X_prev(self):
        with pytest.raises(ValueError, match="X_prev must be a 1-D vector"):
            ContinuationRef(X_prev=np.zeros((2, 3)),
                            t_hat=np.zeros((2, 3)), ds=0.1)

    def test_rejects_a_shape_mismatch(self):
        with pytest.raises(ValueError, match="t_hat must match X_prev"):
            ContinuationRef(X_prev=np.zeros(3), t_hat=np.zeros(4), ds=0.1)

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_rejects_a_nonfinite_X_prev(self, bad):
        X = np.zeros(3)
        X[1] = bad
        with pytest.raises(ValueError, match="must be finite"):
            ContinuationRef(X_prev=X, t_hat=np.ones(3), ds=0.1)

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_rejects_a_nonfinite_t_hat(self, bad):
        t = np.ones(3)
        t[2] = bad
        with pytest.raises(ValueError, match="must be finite"):
            ContinuationRef(X_prev=np.zeros(3), t_hat=t, ds=0.1)

    @pytest.mark.parametrize("bad", [0.0, -0.1, np.nan, np.inf])
    def test_rejects_a_nonpositive_or_nonfinite_ds(self, bad):
        with pytest.raises(ValueError, match="ds must be a positive finite"):
            ContinuationRef(X_prev=np.zeros(3), t_hat=np.ones(3), ds=bad)

    def test_coerces_ds_to_float(self):
        ref = ContinuationRef(X_prev=np.zeros(3), t_hat=np.ones(3), ds=1)
        assert isinstance(ref.ds, float)

    def test_does_not_require_a_unit_tangent(self):
        """
        Unit-norm is PseudoArclength's precondition, not this record's. The
        record stays closer-agnostic so it never bakes in one closer's needs;
        a pin closer has no norm requirement at all.
        """
        ref = ContinuationRef(X_prev=np.zeros(3),
                              t_hat=np.array([3.0, 4.0, 0.0]), ds=0.1)
        assert np.linalg.norm(ref.t_hat) == pytest.approx(5.0)

    def test_accepts_list_input(self):
        # Same stored-vs-input annotation split as SolveSpec above: the
        # fields are declared ndarray because that is what they hold after
        # __post_init__, while the docstring documents array_like input.
        ref = ContinuationRef(
            X_prev=[0.0, 1.0],                 # type: ignore[arg-type]
            t_hat=[1.0, 0.0],                  # type: ignore[arg-type]
            ds=0.1,
        )
        assert isinstance(ref.X_prev, np.ndarray)
        assert ref.X_prev.dtype == float


class TestContinuationRefOwnership:
    """
    The record is built from the engine's live working vectors, so it must
    copy rather than alias, and freeze rather than trust.
    """

    def test_does_not_alias_the_source_arrays(self):
        X = np.array([1.0, 2.0, 3.0])
        t = np.array([1.0, 0.0, 0.0])
        ref = ContinuationRef(X_prev=X, t_hat=t, ds=0.1)

        X[0] = 99.0
        t[0] = 99.0
        assert ref.X_prev[0] == pytest.approx(1.0)
        assert ref.t_hat[0] == pytest.approx(1.0)

    def test_copies_even_a_float_array(self):
        """
        np.asarray would pass a float64 array straight through; np.array
        copies unconditionally. A dtype-correct input is exactly the case
        asarray would fail to protect, so it is the one worth asserting.
        """
        X = np.zeros(3, dtype=float)
        ref = ContinuationRef(X_prev=X, t_hat=np.ones(3), ds=0.1)
        assert ref.X_prev is not X

    def test_stored_arrays_are_read_only(self):
        ref = _ref()
        assert not ref.X_prev.flags.writeable
        assert not ref.t_hat.flags.writeable

    def test_stored_arrays_reject_assignment(self):
        ref = _ref()
        with pytest.raises(ValueError, match="read-only"):
            ref.X_prev[0] = 1.0
        with pytest.raises(ValueError, match="read-only"):
            ref.t_hat[0] = 1.0

    def test_is_frozen(self):
        ref = _ref()
        with pytest.raises(dataclasses.FrozenInstanceError):
            ref.ds = 0.2                       # type: ignore[misc]
