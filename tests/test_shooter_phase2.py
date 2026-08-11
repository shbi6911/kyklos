"""
Tests for the Phase II version of the shooter, including row and column plans,
and per-node free vars and continuity specifications.

Every TerminalConstraint now declares its residual row count up front via the
n_rows property, so the corrector can size F and DF -- and check problem
determinacy -- before the first propagation, instead of backing the count out
of a propagated Jacobian's shape. These tests pin that contract:

  - the three built-in constraints report the correct n_rows from their spec
    alone (no propagation, no state needed to know the count);
  - n_rows is honest: it equals the actual residual length on evaluation, and
    both Jacobians carry n_rows rows;
  - CallableConstraint requires and validates an explicit n_rows;
  - a TerminalConstraint subclass that omits n_rows is abstract (cannot be
    instantiated);
  - the base-class default jacobian_x0 is sized from n_rows and never calls
    residual;
  - bare callables are no longer auto-wrapped -- they must be wrapped in a
    CallableConstraint(g, n_rows=...) explicitly, and the funnel error says so.

All tier 1 (fast): pure constraint math, no integrator, nothing propagated.
_validate_constraints is exercised directly with system=None, since the
built-in constraints' bind() ignores the system, so no trajectory fake is
needed.
"""

import numpy as np
import pytest
from dataclasses import FrozenInstanceError

from kyklos import temp_config
from kyklos.shooter import (
    TerminalConstraint, TargetState, Periodicity, CallableConstraint,
    _ShootingContext, _BlockKind, _RowBlock, _RowPlan, _Selector,
    _ColumnPlan, _build_row_plan, _build_column_plan, _select,
    NodeSpec, _NodeTarget, _resolve_node_specs,
)


# A representative final state and start state for evaluating residuals. The
# values are arbitrary but fixed; these tests exercise shapes and the
# n_rows-vs-residual-length agreement, not any dynamical meaning.
_STATE_TF = np.array([0.80, 0.10, -0.20, 0.03, 0.90, -0.05])
_X0 = np.array([0.82, 0.00, -0.20, 0.00, 0.88, 0.00])


class TestBuiltinNRows:
    """The three built-in constraints declare n_rows from their spec alone."""

    def test_targetstate_n_rows_counts_targets(self):
        assert TargetState({'y': 0.0, 'vx': 0.0}).n_rows == 2
        assert TargetState({'y': 0.0, 'vx': 0.0, 'vz': 0.0}).n_rows == 3
        assert TargetState({'x': 0.0}).n_rows == 1

    def test_periodicity_full_state_n_rows_is_six(self):
        assert Periodicity().n_rows == 6

    def test_periodicity_subset_n_rows_counts_components(self):
        assert Periodicity(['y']).n_rows == 1
        assert Periodicity(['x', 'z', 'vy']).n_rows == 3

    def test_callable_n_rows_is_declared_value(self):
        assert CallableConstraint(lambda s, x: s[:2], n_rows=2).n_rows == 2

    def test_n_rows_is_a_plain_int(self):
        # Downstream plan arithmetic and the determinacy ledger expect a plain
        # int, not a numpy integer that could surprise a comparison or a slice.
        for c in (TargetState({'y': 0.0}), Periodicity(),
                  CallableConstraint(lambda s, x: s[:3], n_rows=3)):
            assert type(c.n_rows) is int

    def test_n_rows_needs_no_propagation(self):
        # The whole point: n_rows must be knowable without ever calling
        # residual or a Jacobian. Read it straight off fresh constructions.
        for c in (TargetState({'y': 0.0}), Periodicity(),
                  CallableConstraint(lambda s, x: s[:3], n_rows=3)):
            assert c.n_rows >= 1


class TestNRowsMatchesResidual:
    """n_rows is honest: it equals the actual residual length on evaluation."""

    def test_targetstate_agreement(self):
        for spec in ({'y': 0.0}, {'y': 0.0, 'vx': 0.0},
                     {'x': 0.0, 'y': 0.0, 'vz': 0.0}):
            c = TargetState(spec)
            r = np.atleast_1d(c.residual(_STATE_TF, _X0))
            assert r.shape[0] == c.n_rows

    def test_periodicity_agreement(self):
        for comps in (None, ['y'], ['x', 'z', 'vy']):
            c = Periodicity(comps)
            r = np.atleast_1d(c.residual(_STATE_TF, _X0))
            assert r.shape[0] == c.n_rows

    def test_callable_agreement(self):
        c = CallableConstraint(
            lambda s, x: np.array([s[0] ** 2, s[1] * s[2]]), n_rows=2)
        r = np.atleast_1d(c.residual(_STATE_TF, _X0))
        assert r.shape[0] == c.n_rows

    def test_jacobian_row_counts_match_n_rows(self):
        # jacobian_tf and jacobian_x0 must both have exactly n_rows rows.
        c = TargetState({'y': 0.0, 'vx': 0.0})
        assert c.jacobian_tf(_STATE_TF, _X0).shape == (c.n_rows, 6)
        assert c.jacobian_x0(_STATE_TF, _X0).shape == (c.n_rows, 6)

        p = Periodicity(['x', 'z', 'vy'])
        assert p.jacobian_tf(_STATE_TF, _X0).shape == (p.n_rows, 6)
        assert p.jacobian_x0(_STATE_TF, _X0).shape == (p.n_rows, 6)


class TestCallableConstraintNRowsValidation:
    """CallableConstraint requires an explicit, valid n_rows."""

    def test_missing_n_rows_raises_typeerror(self):
        # n_rows is now a required positional argument.
        with pytest.raises(TypeError):
            CallableConstraint(lambda s, x: s[:2])          # type: ignore

    def test_float_n_rows_raises_typeerror(self):
        with pytest.raises(TypeError):
            CallableConstraint(lambda s, x: s[:2], n_rows=2.0)   # type: ignore

    def test_bool_n_rows_rejected(self):
        # bool is a subclass of int; reject it explicitly so True/False are
        # not silently treated as 1/0 row counts.
        with pytest.raises(TypeError):
            CallableConstraint(lambda s, x: s[:1], n_rows=True)  # type: ignore

    def test_zero_n_rows_raises_valueerror(self):
        with pytest.raises(ValueError):
            CallableConstraint(lambda s, x: np.array([]), n_rows=0)

    def test_negative_n_rows_raises_valueerror(self):
        with pytest.raises(ValueError):
            CallableConstraint(lambda s, x: s[:1], n_rows=-1)

    def test_numpy_integer_n_rows_accepted_and_coerced(self):
        c = CallableConstraint(lambda s, x: s[:2], n_rows=np.int64(2))
        assert c.n_rows == 2
        assert type(c.n_rows) is int

    def test_non_callable_g_raises_typeerror(self):
        with pytest.raises(TypeError):
            CallableConstraint(5, n_rows=1)                 # type: ignore

    def test_non_callable_dg_raises_typeerror(self):
        with pytest.raises(TypeError):
            CallableConstraint(lambda s, x: s[:1], n_rows=1,
                               dg=5)                        # type: ignore

    def test_non_callable_dg_dx0_raises_typeerror(self):
        with pytest.raises(TypeError):
            CallableConstraint(lambda s, x: s[:1], n_rows=1,
                               dg_dx0=5)                    # type: ignore

    def test_keyword_jacobians_still_supported(self):
        # dg / dg_dx0 remain optional keyword callables after the signature
        # change (n_rows inserted between g and dg).
        sentinel_tf = np.full((1, 6), 7.0)
        sentinel_x0 = np.full((1, 6), -2.0)
        c = CallableConstraint(
            lambda s, x: np.atleast_1d(s[0] - x[0]),
            n_rows=1,
            dg=lambda s, x: sentinel_tf,
            dg_dx0=lambda s, x: sentinel_x0,
        )
        np.testing.assert_array_equal(c.jacobian_tf(_STATE_TF, _X0), sentinel_tf)
        np.testing.assert_array_equal(c.jacobian_x0(_STATE_TF, _X0), sentinel_x0)


class TestBaseClassNRows:
    """The abstract base requires n_rows and sizes its defaults from it."""

    def test_subclass_without_n_rows_is_abstract(self):
        # A TerminalConstraint subclass implementing residual but not n_rows is
        # still abstract and cannot be instantiated.
        class _NoNRows(TerminalConstraint):
            def residual(self, state_tf, x0):
                return np.asarray(state_tf, dtype=float)[:2]

        with pytest.raises(TypeError):
            _NoNRows()                                       # type: ignore

    def test_subclass_with_n_rows_instantiates(self):
        class _Ok(TerminalConstraint):
            @property
            def n_rows(self):
                return 2

            def residual(self, state_tf, x0):
                return np.asarray(state_tf, dtype=float)[:2]

        assert _Ok().n_rows == 2

    def test_default_jacobian_x0_sized_from_n_rows(self):
        class _Three(TerminalConstraint):
            @property
            def n_rows(self):
                return 3

            def residual(self, state_tf, x0):
                return np.asarray(state_tf, dtype=float)[:3]

        np.testing.assert_array_equal(
            _Three().jacobian_x0(_STATE_TF, _X0), np.zeros((3, 6))
        )

    def test_default_jacobian_x0_does_not_call_residual(self):
        # Sizing from n_rows means the default jacobian_x0 must not depend on
        # residual -- prove it by making residual raise if called.
        class _Boom(TerminalConstraint):
            @property
            def n_rows(self):
                return 2

            def residual(self, state_tf, x0):
                raise AssertionError(
                    "residual must not be called by the default jacobian_x0"
                )

        np.testing.assert_array_equal(
            _Boom().jacobian_x0(_STATE_TF, _X0), np.zeros((2, 6))
        )


class TestBareCallableRejected:
    """Bare callables are no longer auto-wrapped; they must be wrapped."""

    def test_bare_callable_raises_typeerror(self):
        with pytest.raises(TypeError):
            _ShootingContext._validate_constraints([lambda s, x: s[:1]], None)

    def test_error_message_names_the_fix(self):
        # A self-curing error: the message tells the user to wrap in a
        # CallableConstraint, so the fix is discoverable from the failure.
        with pytest.raises(TypeError, match="CallableConstraint"):
            _ShootingContext._validate_constraints([lambda s, x: s[:1]], None)

    def test_wrapped_callable_passes_through(self):
        c = CallableConstraint(lambda s, x: s[:1], n_rows=1)
        out = _ShootingContext._validate_constraints([c], None)
        assert len(out) == 1 and out[0] is c

    def test_builtin_constraints_pass_through(self):
        cons = [TargetState({'y': 0.0}), Periodicity(['x'])]
        out = _ShootingContext._validate_constraints(cons, None)
        assert len(out) == 2
        assert all(isinstance(c, TerminalConstraint) for c in out)

    def test_none_yields_empty_tuple(self):
        assert _ShootingContext._validate_constraints(None, None) == ()

    def test_non_constraint_non_callable_still_raises(self):
        with pytest.raises(TypeError):
            _ShootingContext._validate_constraints([42], None)

"""
Tests for the row/column structure plan (Stage 1, step 2 of the refactor).

The plan describes the fixed layout of F and DF for a shooting problem --
row blocks (offset, count, kind, index) and column regions (per-IC free
selectors + boundaries) -- computed once at _ShootingContext construction
from the problem spec alone. These tests pin that layout against
hand-computed Phase 1 values and prove it reproduces the context's existing
property-based sizing, so the later assembler rewrite (step 3) can swap onto
the plan with no behavior change.

All tier 1 (fast): pure layout arithmetic, no integrator, nothing
propagated. Contexts are built directly (system=None is never read during
plan construction), so no trajectory fake is needed.
"""

# Canonical indexing for free variable / constraint subsets
_ALL6 = tuple(range(6))
_POS = (0, 1, 2)
_VEL = (3, 4, 5)


def _ctx(n_seg, free_idx, free_time_idx=None, constraints=(), node_specs=None):
    """Construct a _ShootingContext directly for plan tests.

    system is None (never read while building the plan); times_ref is given a
    plausible length (n_seg + 1 boundary times) though its values are unused
    by the layout. node_specs is the raw sparse mapping (boundary-index ->
    NodeSpec); it is resolved here exactly as from_guess does, and defaults to
    all-continuous when None -- so pre-Phase-II callers are unaffected.
    """
    fti = np.array([] if free_time_idx is None else free_time_idx, dtype=int)
    resolved = _resolve_node_specs(node_specs, n_seg)
    return _ShootingContext(
        system=None,                       # type: ignore  (never read here)
        n_seg=n_seg,
        free_idx=np.asarray(free_idx, dtype=int),
        ics_ref=np.zeros((n_seg, 6)),
        times_ref=np.arange(n_seg + 1, dtype=float),
        free_time_idx=fti,
        constraints=tuple(constraints),
        node_specs=resolved,
    )


def _cont(n):
    """A tuple of n all-continuous NodeSpecs (the default junction role).

    Convenience for the direct _build_row_plan / _build_column_plan tests,
    which take a resolved node_specs tuple rather than the raw dict _ctx
    accepts. `_cont(0)` is the empty tuple (single-shooting, no junctions).
    """
    return tuple(NodeSpec.continuous() for _ in range(n))


# --------------------------------------------------------------------------
# Row plan builder
# --------------------------------------------------------------------------
class TestBuildRowPlan:
    """_build_row_plan: interior defect blocks, then terminal blocks. The
    builder now takes a per-junction node_specs tuple; a continuous role
    yields the full-6 continuity block these tests pin."""

    def test_single_shooting_no_constraints_is_empty(self):
        plan = _build_row_plan(_cont(0), ())
        assert plan.blocks == ()
        assert plan.n_rows == 0

    def test_single_shooting_one_terminal_block(self):
        c = TargetState({'y': 0.0, 'vx': 0.0, 'vz': 0.0})   # n_rows == 3
        plan = _build_row_plan(_cont(0), (c,))
        assert plan.n_rows == 3
        assert plan.blocks == (
            _RowBlock(0, 3, None, _BlockKind.TERMINAL, 0),
        )

    def test_interior_defects_only(self):
        plan = _build_row_plan(_cont(2), ())
        assert plan.n_rows == 12
        assert plan.blocks == (
            _RowBlock(0, 6, _ALL6, _BlockKind.INTERIOR_DEFECT, 0),
            _RowBlock(6, 6, _ALL6, _BlockKind.INTERIOR_DEFECT, 1),
        )

    def test_interior_then_terminal_ordering_and_offsets(self):
        c = Periodicity()                                    # n_rows == 6
        plan = _build_row_plan(_cont(2), (c,))
        assert plan.n_rows == 18
        assert plan.blocks == (
            _RowBlock(0, 6, _ALL6, _BlockKind.INTERIOR_DEFECT, 0),
            _RowBlock(6, 6, _ALL6, _BlockKind.INTERIOR_DEFECT, 1),
            _RowBlock(12, 6, None, _BlockKind.TERMINAL, 0),
        )

    def test_multiple_constraints_stack_in_order(self):
        c0 = TargetState({'x': 0.0})                         # n_rows == 1
        c1 = Periodicity(['y', 'vy'])                        # n_rows == 2
        c2 = CallableConstraint(lambda s, x: s[:3], n_rows=3)
        plan = _build_row_plan(_cont(1), (c0, c1, c2))
        # one interior defect (rows 0-5), then 1 + 2 + 3 terminal rows.
        assert plan.n_rows == 6 + 1 + 2 + 3
        assert plan.blocks == (
            _RowBlock(0, 6, _ALL6, _BlockKind.INTERIOR_DEFECT, 0),
            _RowBlock(6, 1, None, _BlockKind.TERMINAL, 0),
            _RowBlock(7, 2, None, _BlockKind.TERMINAL, 1),
            _RowBlock(9, 3, None, _BlockKind.TERMINAL, 2),
        )

    def test_interior_index_is_zero_based_junction_index(self):
        plan = _build_row_plan(_cont(3), ())
        interior = [b for b in plan.blocks
                    if b.kind is _BlockKind.INTERIOR_DEFECT]
        assert [b.index for b in interior] == [0, 1, 2]

    def test_terminal_index_is_zero_based_constraint_index(self):
        cons = (TargetState({'x': 0.0}), TargetState({'y': 0.0}))
        plan = _build_row_plan(_cont(0), cons)
        terminal = [b for b in plan.blocks if b.kind is _BlockKind.TERMINAL]
        assert [b.index for b in terminal] == [0, 1]

    def test_offsets_are_contiguous_and_ordered(self):
        cons = (Periodicity(), TargetState({'x': 0.0, 'y': 0.0}))
        plan = _build_row_plan(_cont(2), cons)
        expected_offset = 0
        for b in plan.blocks:
            assert b.row_offset == expected_offset
            expected_offset += b.row_count
        assert expected_offset == plan.n_rows


# --------------------------------------------------------------------------
# Column plan builder
# --------------------------------------------------------------------------
class TestBuildColumnPlan:
    """_build_column_plan: start comps | junction posts | free times. The
    builder now takes a per-junction node_specs tuple in place of n_seg; a
    continuous role frees all six columns."""

    def test_single_shooting_all_free(self):
        cp = _build_column_plan(np.arange(6), _cont(0),
                                free_time_idx=np.array([], dtype=int))
        assert cp.n_fs == 6
        assert cp.n_state_block == 6
        assert cp.n_X == 6
        assert cp.selectors == (_Selector(_ALL6, 0),)
        assert cp.free_time_idx == ()

    def test_single_shooting_partial_free(self):
        # planar start: components x, y, vx, vy -> indices 0, 1, 3, 4.
        cp = _build_column_plan(np.array([0, 1, 3, 4]), _cont(0),
                                free_time_idx=np.array([], dtype=int))
        assert cp.n_fs == 4
        assert cp.n_state_block == 4
        assert cp.n_X == 4
        assert cp.selectors == (_Selector((0, 1, 3, 4), 0),)

    def test_three_segments_all_free_col_starts(self):
        cp = _build_column_plan(np.arange(6), _cont(2),
                                free_time_idx=np.array([], dtype=int))
        # start at 0 (width 6), junction posts at 6 and 12.
        assert cp.selectors == (
            _Selector(_ALL6, 0),
            _Selector(_ALL6, 6),
            _Selector(_ALL6, 12),
        )
        assert cp.n_fs == 6
        assert cp.n_state_block == 18
        assert cp.n_X == 18

    def test_three_segments_partial_start_shifts_col_starts(self):
        # A 4-wide free start shifts every junction col_start left by 2 vs the
        # all-free case -- the running-sum construction, not n_fs + 6*j from a
        # fixed n_fs=6. Junctions here are continuous (full-6).
        cp = _build_column_plan(np.array([0, 1, 3, 4]), _cont(2),
                                free_time_idx=np.array([], dtype=int))
        assert cp.selectors == (
            _Selector((0, 1, 3, 4), 0),
            _Selector(_ALL6, 4),
            _Selector(_ALL6, 10),
        )
        assert cp.n_fs == 4
        assert cp.n_state_block == 16
        assert cp.n_X == 16

    def test_free_times_extend_n_X_only(self):
        cp = _build_column_plan(np.arange(6), _cont(2),
                                free_time_idx=np.array([2, 3]))
        # state block unchanged; two free-time columns appended.
        assert cp.n_state_block == 18
        assert cp.n_X == 20
        assert cp.free_time_idx == (2, 3)

    def test_free_time_idx_stored_as_tuple_of_int(self):
        cp = _build_column_plan(np.arange(6), _cont(1),
                                free_time_idx=np.array([1, 2]))
        assert cp.free_time_idx == (1, 2)
        assert all(type(m) is int for m in cp.free_time_idx)


# --------------------------------------------------------------------------
# Column plan accessors
# --------------------------------------------------------------------------
class TestColumnPlanAccessors:
    """Column placement is read directly off selectors: selectors[0] is the
    start state, selectors[j+1] is junction j's post-state. col_slice gives
    the X-column range; components gives the free state-component indices."""

    def test_start_selector_slice_and_components(self):
        cp = _build_column_plan(np.array([0, 1, 3, 4]), _cont(2),
                                free_time_idx=np.array([], dtype=int))
        start = cp.selectors[0]
        assert start.col_slice == slice(0, 4)
        assert start.components == (0, 1, 3, 4)

    def test_junction_slices(self):
        cp = _build_column_plan(np.arange(6), _cont(2),
                                free_time_idx=np.array([], dtype=int))
        assert cp.selectors[1].col_slice == slice(6, 12)   # junction 0
        assert cp.selectors[2].col_slice == slice(12, 18)  # junction 1

    def test_junction_components_full_six_for_continuous(self):
        # Renamed off "phase1": a continuous junction frees all six columns.
        # (An impulsive junction also frees six; a custom role may free fewer
        # -- see TestPerNodePlans.)
        cp = _build_column_plan(np.array([0, 1, 3, 4]), _cont(2),
                                free_time_idx=np.array([], dtype=int))
        assert cp.selectors[1].components == _ALL6
        assert cp.selectors[2].components == _ALL6

    def test_col_start_uses_running_sum_not_fixed_stride(self):
        # With a 4-wide start, junction 0 begins at column 4, not at n_fs=6 as
        # it would under a fixed-6 stride -- confirms col_start is the running
        # sum of prior widths, which is what makes partial nodes work.
        cp = _build_column_plan(np.array([0, 1, 3, 4]), _cont(1),
                                free_time_idx=np.array([], dtype=int))
        j0 = cp.selectors[1]
        assert j0.col_start == 4
        assert j0.col_slice == slice(4, 10)

    def test_free_time_column_maps_position(self):
        cp = _build_column_plan(np.arange(6), _cont(2),
                                free_time_idx=np.array([2, 3]))
        assert cp.free_time_column(2) == 18   # n_state_block + position 0
        assert cp.free_time_column(3) == 19   # n_state_block + position 1

    def test_free_time_column_raises_for_non_free_time(self):
        cp = _build_column_plan(np.arange(6), _cont(2),
                                free_time_idx=np.array([2]))
        with pytest.raises(ValueError):
            cp.free_time_column(1)            # boundary time 1 is not free

    def test_selector_width(self):
        assert _Selector((0, 1, 3, 4), 0).width == 4
        assert _Selector(_ALL6, 6).width == 6


# --------------------------------------------------------------------------
# Context integration and consistency with existing sizing
# --------------------------------------------------------------------------
class TestContextPlanIntegration:
    """The context builds both plans and they agree with its property sizing."""

    def test_context_exposes_both_plans(self):
        ctx = _ctx(3, np.arange(6), constraints=(Periodicity(),))
        assert isinstance(ctx.row_plan, _RowPlan)
        assert isinstance(ctx.column_plan, _ColumnPlan)

    @pytest.mark.parametrize("n_seg, free_idx, free_times, cons", [
        (1, np.arange(6), None, ()),
        (1, [0, 1, 3, 4], None, (TargetState({'y': 0.0}),)),
        (3, np.arange(6), None, (Periodicity(),)),
        (3, [0, 1, 3, 4], None, (Periodicity(),)),
        (3, np.arange(6), [2, 3], ()),
        (4, [0, 1, 3, 4], [2, 4], (TargetState({'x': 0.0, 'y': 0.0}),)),
    ])
    def test_column_plan_matches_context_sizing(
            self, n_seg, free_idx, free_times, cons):
        # The new column plan must reproduce the battle-tested property-based
        # sizing exactly -- the green-to-green guarantee for step 3.
        ctx = _ctx(n_seg, free_idx, free_times, cons)
        assert ctx.column_plan.n_fs == ctx.n_free_start
        assert ctx.column_plan.n_state_block == ctx.n_state_block
        assert ctx.column_plan.n_X == ctx.n_X

    @pytest.mark.parametrize("n_seg, cons, expected_rows", [
        (1, (), 0),
        (1, (TargetState({'y': 0.0, 'vx': 0.0}),), 2),
        (3, (), 12),
        (3, (Periodicity(),), 18),
        (4, (TargetState({'x': 0.0}), Periodicity(['y'])), 6 * 3 + 1 + 1),
    ])
    def test_row_plan_total_matches_expected(self, n_seg, cons, expected_rows):
        ctx = _ctx(n_seg, np.arange(6), constraints=cons)
        assert ctx.row_plan.n_rows == expected_rows

    def test_selector_count_is_one_plus_n_junction(self):
        ctx = _ctx(4, np.arange(6))
        # start + 3 junction posts.
        assert len(ctx.column_plan.selectors) == 1 + ctx.n_junction


# --------------------------------------------------------------------------
# Determinacy report
# --------------------------------------------------------------------------
class TestDeterminacyReport:
    """determinacy classifies m vs n; it reports, it does not gate."""

    def test_square(self):
        # 3 segments, all free (n_X = 18), full periodicity (m = 18).
        ctx = _ctx(3, np.arange(6), constraints=(Periodicity(),))
        assert ctx.determinacy == 'square'

    def test_overdetermined(self):
        # 4-wide free start (n_X = 16) with full periodicity (m = 18).
        ctx = _ctx(3, [0, 1, 3, 4], constraints=(Periodicity(),))
        assert ctx.determinacy == 'overdetermined'

    def test_underdetermined(self):
        # all free (n_X = 18), no terminal constraints (m = 12).
        ctx = _ctx(3, np.arange(6), constraints=())
        assert ctx.determinacy == 'underdetermined'


# --------------------------------------------------------------------------
# Immutability
# --------------------------------------------------------------------------
class TestPlanImmutability:
    """Plans and their records are frozen."""

    def test_context_plan_fields_are_frozen(self):
        ctx = _ctx(2, np.arange(6))
        with pytest.raises(FrozenInstanceError):
            ctx.row_plan = _RowPlan((), 0)          # type: ignore

    def test_row_block_is_frozen(self):
        b = _RowBlock(0, 6, tuple(range(6)), _BlockKind.INTERIOR_DEFECT, 0)
        with pytest.raises(FrozenInstanceError):
            b.row_offset = 5                          # type: ignore

    def test_selector_is_frozen(self):
        s = _Selector((0, 1), 0)
        with pytest.raises(FrozenInstanceError):
            s.col_start = 3                           # type: ignore

    def test_column_plan_is_frozen(self):
        cp = _build_column_plan(np.arange(6), _cont(2),
                        np.array([], dtype=int))
        with pytest.raises(FrozenInstanceError):
            cp.n_X = 99                               # type: ignore

class TestSelectSubBlock:
    """_select extracts an outer-product sub-block by component indices.

    Phase 1 only ever calls _select with all-6 row and column tuples, so the
    np.ix_ (outer-product, not pairwise) behavior is never exercised on a
    genuine sub-block. These tests hit it on non-uniform, non-contiguous, and
    unequal-length selections -- the shapes a Phase 2 partial node will
    produce -- so a row/column axis swap or a pairwise-vs-outer-product
    regression is caught now rather than surfacing as a 'Phase 2 bug'.
    """

    def test_full_selection_is_identity_on_input(self):
        M = np.arange(36.0).reshape(6, 6)
        np.testing.assert_array_equal(
            _select(M, tuple(range(6)), tuple(range(6))), M)

    def test_square_subblock_outer_product(self):
        # rows (0,1,2) x cols (3,4,5): the top-right 3x3 quadrant. Must be the
        # outer product, NOT the pairwise diagonal [M[0,3], M[1,4], M[2,5]].
        M = np.arange(36.0).reshape(6, 6)
        out = _select(M, (0, 1, 2), (3, 4, 5))
        assert out.shape == (3, 3)
        np.testing.assert_array_equal(out, M[np.ix_([0, 1, 2], [3, 4, 5])])
        # explicit guard against the pairwise-indexing regression:
        assert out[0, 0] == M[0, 3] and out[1, 1] == M[1, 4]
        assert out[0, 1] == M[0, 4]        # off-diagonal present -> outer product

    def test_rectangular_subblock_unequal_lengths(self):
        # 3 rows x 6 cols: the shape a 3-row position-continuity defect against
        # a full-6 free post-state produces. Row count != col count, so a swap
        # of the two index axes would change the shape and fail loudly.
        M = np.arange(36.0).reshape(6, 6)
        out = _select(M, (0, 1, 2), tuple(range(6)))
        assert out.shape == (3, 6)
        np.testing.assert_array_equal(out, M[np.ix_([0, 1, 2], list(range(6)))])

    def test_non_contiguous_selection(self):
        # continuity components need not be the leading indices; (0,2,4) must
        # select exactly those rows, not range(len)=(0,1,2).
        M = np.arange(36.0).reshape(6, 6)
        out = _select(M, (0, 2, 4), (1, 3))
        assert out.shape == (3, 2)
        np.testing.assert_array_equal(out, M[np.ix_([0, 2, 4], [1, 3])])

    def test_row_and_col_axes_not_swapped(self):
        # An asymmetric M where M[i,j] != M[j,i], selecting different row and
        # col tuples: proves rows index axis 0 and cols index axis 1.
        M = np.arange(36.0).reshape(6, 6)     # M[i,j] = 6i + j, not symmetric
        out = _select(M, (1,), (4,))
        assert out.shape == (1, 1)
        assert out[0, 0] == M[1, 4]           # == 10, not M[4,1] == 25


class TestRowBlockConsistencyGuard:
    """_RowBlock.__post_init__ enforces row_count == len(continuity_components)
    for interior blocks, and permits None (terminal blocks) unchecked."""

    def test_consistent_interior_block_constructs(self):
        b = _RowBlock(0, 3, (0, 1, 2), _BlockKind.INTERIOR_DEFECT, 0)
        assert b.row_count == 3 and b.continuity_components == (0, 1, 2)

    def test_full_interior_block_constructs(self):
        b = _RowBlock(0, 6, tuple(range(6)), _BlockKind.INTERIOR_DEFECT, 0)
        assert b.row_count == len(b.continuity_components)      #type: ignore

    def test_row_count_mismatch_raises(self):
        # row_count 3 but six components -> the guard must fire.
        with pytest.raises(ValueError):
            _RowBlock(0, 3, tuple(range(6)), _BlockKind.INTERIOR_DEFECT, 0)

    def test_row_count_mismatch_other_direction_raises(self):
        # row_count 6 but three components.
        with pytest.raises(ValueError):
            _RowBlock(0, 6, (0, 1, 2), _BlockKind.INTERIOR_DEFECT, 0)

    def test_terminal_none_components_not_checked(self):
        # Terminal blocks carry None continuity_components and any row_count;
        # the guard must skip them (no ValueError).
        b = _RowBlock(6, 4, None, _BlockKind.TERMINAL, 0)
        assert b.continuity_components is None and b.row_count == 4

class TestPerNodePlans:
    """Phase II: per-junction free/continuity flow into the plans, and the two
    axes are decoupled -- an impulsive junction contributes 3 rows but 6
    columns. Pure layout arithmetic; nothing propagated."""

    # ---- row plan reads per-node continuity ----

    def test_impulsive_junction_gives_three_row_block(self):
        plan = _build_row_plan((NodeSpec.impulsive(),), ())
        assert plan.n_rows == 3
        assert plan.blocks == (
            _RowBlock(0, 3, (0, 1, 2), _BlockKind.INTERIOR_DEFECT, 0),
        )

    def test_mixed_continuous_and_impulsive_row_offsets(self):
        plan = _build_row_plan(
            (NodeSpec.continuous(), NodeSpec.impulsive()), ())
        assert plan.n_rows == 9
        assert plan.blocks == (
            _RowBlock(0, 6, _ALL6, _BlockKind.INTERIOR_DEFECT, 0),
            _RowBlock(6, 3, (0, 1, 2), _BlockKind.INTERIOR_DEFECT, 1),
        )

    def test_custom_subspace_continuity_row_count(self):
        spec = NodeSpec.custom('all', ['x', 'y', 'z', 'vz'])   # continuity 0,1,2,5
        plan = _build_row_plan((spec,), ())
        assert plan.blocks == (
            _RowBlock(0, 4, (0, 1, 2, 5), _BlockKind.INTERIOR_DEFECT, 0),
        )

    def test_impulsive_then_terminal_offsets_contiguous(self):
        plan = _build_row_plan((NodeSpec.impulsive(),), (Periodicity(),))
        assert plan.blocks == (
            _RowBlock(0, 3, (0, 1, 2), _BlockKind.INTERIOR_DEFECT, 0),
            _RowBlock(3, 6, None, _BlockKind.TERMINAL, 0),
        )
        assert plan.n_rows == 9

    # ---- column plan reads per-node free ----

    def test_impulsive_junction_keeps_six_columns(self):
        # Impulsive frees all six columns even though it enforces three rows.
        cp = _build_column_plan(np.arange(6), (NodeSpec.impulsive(),),
                                np.array([], dtype=int))
        assert cp.selectors[1].components == _ALL6
        assert cp.selectors[1].width == 6

    def test_custom_partial_free_narrows_columns(self):
        spec = NodeSpec.custom('velocity', 'velocity')   # free (3,4,5)
        cp = _build_column_plan(np.arange(6), (spec,),
                                np.array([], dtype=int))
        assert cp.selectors[1].components == (3, 4, 5)
        assert cp.selectors[1].col_slice == slice(6, 9)
        assert cp.n_state_block == 9

    def test_mixed_partial_and_full_running_sum(self):
        # start full-6, J0 velocity-only (3 cols), J1 continuous (6 cols):
        # col_starts 0, 6, 9 -- the running sum with a genuinely partial node.
        specs = (NodeSpec.custom('velocity', 'velocity'),
                 NodeSpec.continuous())
        cp = _build_column_plan(np.arange(6), specs, np.array([], dtype=int))
        assert cp.selectors == (
            _Selector(_ALL6, 0),
            _Selector((3, 4, 5), 6),
            _Selector(_ALL6, 9),
        )
        assert cp.n_state_block == 15

    # ---- the decoupling, end to end through the context ----

    def test_impulsive_decouples_rows_from_columns(self):
        # Junction 0 impulsive: its defect block is 3 rows, its post-state
        # selector is 6 columns. Rows shrink (9 vs the all-continuous 12);
        # columns do not (18). This is the core Phase II property.
        ctx = _ctx(3, np.arange(6), node_specs={1: NodeSpec.impulsive()})
        interior = [b for b in ctx.row_plan.blocks
                    if b.kind is _BlockKind.INTERIOR_DEFECT]
        j0_row = interior[0]
        assert j0_row.index == 0
        assert j0_row.continuity_components == (0, 1, 2)
        assert j0_row.row_count == 3
        assert ctx.column_plan.selectors[1].components == _ALL6
        assert ctx.row_plan.n_rows == 9          # 3 (impulsive) + 6 (continuous)
        assert ctx.column_plan.n_state_block == 18   # 6 + 6 + 6, unchanged

# ==========================================================================
# NodeSpec -- named roles
# ==========================================================================
class TestNodeSpecRoles:
    """The two named roles resolve to the correct (free, continuity, becomes)
    triple. These are the only roles the user writes by hand routinely."""

    def test_continuous_is_full_free_full_continuity_null(self):
        spec = NodeSpec.continuous()
        assert spec.free == _ALL6
        assert spec.continuity == _ALL6
        assert spec.becomes is _NodeTarget.NULL

    def test_impulsive_is_full_free_position_continuity_impulsive(self):
        spec = NodeSpec.impulsive()
        assert spec.free == _ALL6
        assert spec.continuity == _POS
        assert spec.becomes is _NodeTarget.IMPULSIVE

    def test_free_columns_are_six_for_both_roles(self):
        # An impulsive node has 3 enforced rows but 6 free columns: the
        # post-velocity carries the maneuver, the post-position is pinned by
        # the 3 continuity rows. Both roles free all six.
        assert len(NodeSpec.continuous().free) == 6
        assert len(NodeSpec.impulsive().free) == 6


# ==========================================================================
# NodeSpec -- custom escape hatch
# ==========================================================================
class TestNodeSpecCustom:
    """custom() resolves both component specs through _parse_free_vars and
    infers or accepts the output node type."""

    def test_custom_all_all_matches_continuous(self):
        spec = NodeSpec.custom(free_vars='all', continuity='all')
        assert spec.free == _ALL6
        assert spec.continuity == _ALL6
        assert spec.becomes is _NodeTarget.NULL

    def test_custom_all_position_matches_impulsive(self):
        spec = NodeSpec.custom(free_vars='all', continuity='position')
        assert spec.free == _ALL6
        assert spec.continuity == _POS
        assert spec.becomes is _NodeTarget.IMPULSIVE

    def test_custom_accepts_component_name_lists(self):
        spec = NodeSpec.custom(free_vars=['vx', 'vy', 'vz'],
                               continuity=['vx', 'vy', 'vz'])
        assert spec.free == _VEL
        assert spec.continuity == _VEL

    def test_becomes_inferred_null_from_full_continuity(self):
        assert NodeSpec.custom('all', 'all').becomes is _NodeTarget.NULL

    def test_becomes_inferred_impulsive_from_position_continuity(self):
        assert (NodeSpec.custom('all', 'position').becomes
                is _NodeTarget.IMPULSIVE)

    def test_becomes_inferred_free_from_other_continuity(self):
        # position + out-of-plane velocity: neither full nor position-only.
        spec = NodeSpec.custom('all', ['x', 'y', 'z', 'vz'])
        assert spec.continuity == (0, 1, 2, 5)
        assert spec.becomes is _NodeTarget.FREE

    def test_explicit_becomes_string_resolves(self):
        assert (NodeSpec.custom('all', 'all', becomes='free').becomes
                is _NodeTarget.FREE)

    def test_unknown_becomes_string_raises_valueerror(self):
        with pytest.raises(ValueError):
            NodeSpec.custom('all', 'all', becomes='bogus')

    def test_bad_component_name_raises_valueerror(self):
        # Routed through _parse_free_vars, which raises unconditionally.
        with pytest.raises(ValueError):
            NodeSpec.custom('xyz', 'all')


# ==========================================================================
# NodeSpec -- validation guards (heuristic vs hard)
# ==========================================================================
class TestNodeSpecGuards:
    """Subset and becomes-consistency checks route through validation_error
    (honor STRICT_VALIDATION); malformed index tuples and bad becomes types
    raise unconditionally."""

    # ---- C subset F: heuristic ----

    def test_continuity_not_subset_free_raises_under_strict(self):
        # Enforce position continuity but free only velocity: the node cannot
        # close the position defects by moving its own post-state.
        with temp_config(STRICT_VALIDATION=True):
            with pytest.raises(ValueError):
                NodeSpec.custom(free_vars='velocity', continuity='position')

    def test_continuity_not_subset_free_warns_under_nonstrict(self):
        with temp_config(STRICT_VALIDATION=False):
            with pytest.warns(UserWarning):
                spec = NodeSpec.custom(free_vars='velocity',
                                       continuity='position')
        # Still constructed with the fields intact.
        assert spec.free == _VEL
        assert spec.continuity == _POS

    # ---- becomes vs continuity consistency: heuristic ----

    def test_null_becomes_requires_full_continuity(self):
        with temp_config(STRICT_VALIDATION=True):
            with pytest.raises(ValueError):
                NodeSpec.custom('all', 'position', becomes='continuous')

    def test_impulsive_becomes_requires_position_continuity(self):
        with temp_config(STRICT_VALIDATION=True):
            with pytest.raises(ValueError):
                NodeSpec.custom('all', 'velocity', becomes='impulsive')

    # ---- malformed index tuples / becomes: hard, unconditional ----

    def test_non_ascending_free_raises_even_nonstrict(self):
        with temp_config(STRICT_VALIDATION=False):
            with pytest.raises(ValueError):
                NodeSpec((0, 2, 1), (0, 1, 2), _NodeTarget.FREE)

    def test_out_of_range_component_raises_even_nonstrict(self):
        with temp_config(STRICT_VALIDATION=False):
            with pytest.raises(ValueError):
                NodeSpec((0, 1, 6), (0, 1), _NodeTarget.FREE)

    def test_duplicate_component_raises(self):
        with pytest.raises(ValueError):
            NodeSpec((0, 0, 1), (0,), _NodeTarget.FREE)

    def test_non_target_becomes_raises_typeerror(self):
        # Raw constructor takes a _NodeTarget, not a string.
        with pytest.raises(TypeError):
            NodeSpec((0,), (0,), 'null')                            # type: ignore


# ==========================================================================
# NodeSpec -- immutability, equality, repr
# ==========================================================================
class TestNodeSpecObjectProperties:
    """Frozen value semantics: immutable, value-equal, hashable, readable."""

    def test_is_frozen(self):
        spec = NodeSpec.continuous()
        with pytest.raises(FrozenInstanceError):
            spec.free = (0,)                              # type: ignore

    def test_equal_roles_compare_equal(self):
        assert NodeSpec.continuous() == NodeSpec.continuous()
        assert NodeSpec.impulsive() == NodeSpec.impulsive()

    def test_different_roles_compare_unequal(self):
        assert NodeSpec.continuous() != NodeSpec.impulsive()

    def test_hashable(self):
        # No raise; distinct roles are distinct members.
        s = {NodeSpec.continuous(), NodeSpec.impulsive(),
             NodeSpec.continuous()}
        assert len(s) == 2

    def test_repr_names_role_and_components(self):
        r = repr(NodeSpec.impulsive())
        assert 'impulsive' in r
        assert 'vx' in r          # free components rendered as names
        assert 'x' in r           # continuity components rendered as names


# ==========================================================================
# _resolve_node_specs -- sparse dict -> dense per-junction tuple
# ==========================================================================
class TestResolveNodeSpecs:
    """The single home for the boundary-index -> junction mapping and the
    continuous default. Keys are 1-based boundary indices (parallel to
    free_times); junction j sits at boundary j+1."""

    # ---- defaults ----

    def test_none_is_all_continuous(self):
        specs = _resolve_node_specs(None, 4)
        assert len(specs) == 3
        assert all(s.becomes is _NodeTarget.NULL for s in specs)

    def test_empty_dict_is_all_continuous(self):
        specs = _resolve_node_specs({}, 4)
        assert len(specs) == 3
        assert all(s.becomes is _NodeTarget.NULL for s in specs)

    def test_length_is_n_seg_minus_one(self):
        assert len(_resolve_node_specs(None, 5)) == 4

    # ---- key convention ----

    def test_key_maps_boundary_index_to_junction(self):
        # Boundary key 2 -> junction index 1 (junction j sits at boundary j+1).
        specs = _resolve_node_specs({2: NodeSpec.impulsive()}, 4)
        assert specs[1].becomes is _NodeTarget.IMPULSIVE
        assert specs[0].becomes is _NodeTarget.NULL
        assert specs[2].becomes is _NodeTarget.NULL

    def test_multiple_keys_place_correctly(self):
        specs = _resolve_node_specs(
            {1: NodeSpec.impulsive(), 3: NodeSpec.impulsive()}, 4)
        assert specs[0].becomes is _NodeTarget.IMPULSIVE   # key 1
        assert specs[1].becomes is _NodeTarget.NULL        # default
        assert specs[2].becomes is _NodeTarget.IMPULSIVE   # key 3

    def test_numpy_int_key_accepted(self):
        specs = _resolve_node_specs({np.int64(2): NodeSpec.impulsive()}, 4)
        assert specs[1].becomes is _NodeTarget.IMPULSIVE

    # ---- boundary-node keys rejected ----

    def test_start_boundary_key_zero_raises(self):
        with pytest.raises(ValueError):
            _resolve_node_specs({0: NodeSpec.impulsive()}, 4)

    def test_final_boundary_key_raises(self):
        with pytest.raises(ValueError):
            _resolve_node_specs({4: NodeSpec.impulsive()}, 4)

    def test_out_of_range_high_key_raises(self):
        with pytest.raises(ValueError):
            _resolve_node_specs({5: NodeSpec.impulsive()}, 4)

    def test_negative_key_raises(self):
        with pytest.raises(ValueError):
            _resolve_node_specs({-1: NodeSpec.impulsive()}, 4)

    # ---- single-segment edge ----

    def test_single_segment_none_is_empty(self):
        assert _resolve_node_specs(None, 1) == ()

    def test_single_segment_final_boundary_key_raises(self):
        # key 1 == n_seg -> final boundary.
        with pytest.raises(ValueError):
            _resolve_node_specs({1: NodeSpec.impulsive()}, 1)

    def test_single_segment_out_of_range_key_raises(self):
        # key 2 is neither boundary; hits the "no interior junctions" branch.
        with pytest.raises(ValueError):
            _resolve_node_specs({2: NodeSpec.impulsive()}, 1)

    # ---- type guards ----

    def test_bool_key_rejected(self):
        with pytest.raises(TypeError):
            _resolve_node_specs({True: NodeSpec.impulsive()}, 4)

    def test_float_key_rejected(self):
        with pytest.raises(TypeError):
            _resolve_node_specs({2.0: NodeSpec.impulsive()}, 4)

    def test_non_nodespec_value_rejected(self):
        with pytest.raises(TypeError):
            _resolve_node_specs({2: 'impulsive'}, 4)

    def test_non_dict_rejected(self):
        with pytest.raises(TypeError):
            _resolve_node_specs([NodeSpec.impulsive()], 4)      #type: ignore