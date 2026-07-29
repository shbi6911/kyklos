"""
Tests for the constraint n_rows declaration (Stage 1, step 1 of the
row/column-plan refactor).

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

from kyklos.shooter import (
    TerminalConstraint, TargetState, Periodicity, CallableConstraint,
    _ShootingContext, _BlockKind, _RowBlock, _RowPlan, _Selector,
    _ColumnPlan, _build_row_plan, _build_column_plan,
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

ALL6 = tuple(range(6))


def _ctx(n_seg, free_idx, free_time_idx=None, constraints=()):
    """Construct a _ShootingContext directly for plan tests.

    system is None (never read while building the plan); times_ref is given a
    plausible length (n_seg + 1 boundary times) though its values are unused
    by the layout.
    """
    fti = np.array([] if free_time_idx is None else free_time_idx, dtype=int)
    return _ShootingContext(
        system=None,                       # type: ignore  (never read here)
        n_seg=n_seg,
        free_idx=np.asarray(free_idx, dtype=int),
        x0_ref=np.zeros(6),
        times_ref=np.arange(n_seg + 1, dtype=float),
        free_time_idx=fti,
        constraints=tuple(constraints),
    )


# --------------------------------------------------------------------------
# Row plan builder
# --------------------------------------------------------------------------
class TestBuildRowPlan:
    """_build_row_plan: interior defect blocks, then terminal blocks."""

    def test_single_shooting_no_constraints_is_empty(self):
        plan = _build_row_plan(0, ())
        assert plan.blocks == ()
        assert plan.n_rows == 0

    def test_single_shooting_one_terminal_block(self):
        c = TargetState({'y': 0.0, 'vx': 0.0, 'vz': 0.0})   # n_rows == 3
        plan = _build_row_plan(0, (c,))
        assert plan.n_rows == 3
        assert plan.blocks == (
            _RowBlock(0, 3, _BlockKind.TERMINAL, 0),
        )

    def test_interior_defects_only(self):
        plan = _build_row_plan(2, ())
        assert plan.n_rows == 12
        assert plan.blocks == (
            _RowBlock(0, 6, _BlockKind.INTERIOR_DEFECT, 0),
            _RowBlock(6, 6, _BlockKind.INTERIOR_DEFECT, 1),
        )

    def test_interior_then_terminal_ordering_and_offsets(self):
        c = Periodicity()                                    # n_rows == 6
        plan = _build_row_plan(2, (c,))
        assert plan.n_rows == 18
        assert plan.blocks == (
            _RowBlock(0, 6, _BlockKind.INTERIOR_DEFECT, 0),
            _RowBlock(6, 6, _BlockKind.INTERIOR_DEFECT, 1),
            _RowBlock(12, 6, _BlockKind.TERMINAL, 0),
        )

    def test_multiple_constraints_stack_in_order(self):
        c0 = TargetState({'x': 0.0})                         # n_rows == 1
        c1 = Periodicity(['y', 'vy'])                        # n_rows == 2
        c2 = CallableConstraint(lambda s, x: s[:3], n_rows=3)
        plan = _build_row_plan(1, (c0, c1, c2))
        # one interior defect (rows 0-5), then 1 + 2 + 3 terminal rows.
        assert plan.n_rows == 6 + 1 + 2 + 3
        assert plan.blocks == (
            _RowBlock(0, 6, _BlockKind.INTERIOR_DEFECT, 0),
            _RowBlock(6, 1, _BlockKind.TERMINAL, 0),
            _RowBlock(7, 2, _BlockKind.TERMINAL, 1),
            _RowBlock(9, 3, _BlockKind.TERMINAL, 2),
        )

    def test_interior_index_is_zero_based_junction_index(self):
        plan = _build_row_plan(3, ())
        interior = [b for b in plan.blocks
                    if b.kind is _BlockKind.INTERIOR_DEFECT]
        assert [b.index for b in interior] == [0, 1, 2]

    def test_terminal_index_is_zero_based_constraint_index(self):
        cons = (TargetState({'x': 0.0}), TargetState({'y': 0.0}))
        plan = _build_row_plan(0, cons)
        terminal = [b for b in plan.blocks if b.kind is _BlockKind.TERMINAL]
        assert [b.index for b in terminal] == [0, 1]

    def test_offsets_are_contiguous_and_ordered(self):
        cons = (Periodicity(), TargetState({'x': 0.0, 'y': 0.0}))
        plan = _build_row_plan(2, cons)
        expected_offset = 0
        for b in plan.blocks:
            assert b.row_offset == expected_offset
            expected_offset += b.row_count
        assert expected_offset == plan.n_rows


# --------------------------------------------------------------------------
# Column plan builder
# --------------------------------------------------------------------------
class TestBuildColumnPlan:
    """_build_column_plan: start comps | junction posts | free times."""

    def test_single_shooting_all_free(self):
        cp = _build_column_plan(np.arange(6), n_seg=1,
                                free_time_idx=np.array([], dtype=int))
        assert cp.n_fs == 6
        assert cp.n_state_block == 6
        assert cp.n_X == 6
        assert cp.selectors == (_Selector(ALL6, 0),)
        assert cp.free_time_idx == ()

    def test_single_shooting_partial_free(self):
        # planar start: components x, y, vx, vy -> indices 0, 1, 3, 4.
        cp = _build_column_plan(np.array([0, 1, 3, 4]), n_seg=1,
                                free_time_idx=np.array([], dtype=int))
        assert cp.n_fs == 4
        assert cp.n_state_block == 4
        assert cp.n_X == 4
        assert cp.selectors == (_Selector((0, 1, 3, 4), 0),)

    def test_three_segments_all_free_col_starts(self):
        cp = _build_column_plan(np.arange(6), n_seg=3,
                                free_time_idx=np.array([], dtype=int))
        # start at 0 (width 6), junction posts at 6 and 12.
        assert cp.selectors == (
            _Selector(ALL6, 0),
            _Selector(ALL6, 6),
            _Selector(ALL6, 12),
        )
        assert cp.n_fs == 6
        assert cp.n_state_block == 18
        assert cp.n_X == 18

    def test_three_segments_partial_start_shifts_col_starts(self):
        # A 4-wide free start shifts every junction col_start left by 2 vs the
        # all-free case -- the running-sum construction, not n_fs + 6*j from a
        # fixed n_fs=6. (In Phase 1 junctions are still full-6.)
        cp = _build_column_plan(np.array([0, 1, 3, 4]), n_seg=3,
                                free_time_idx=np.array([], dtype=int))
        assert cp.selectors == (
            _Selector((0, 1, 3, 4), 0),
            _Selector(ALL6, 4),
            _Selector(ALL6, 10),
        )
        assert cp.n_fs == 4
        assert cp.n_state_block == 16
        assert cp.n_X == 16

    def test_free_times_extend_n_X_only(self):
        cp = _build_column_plan(np.arange(6), n_seg=3,
                                free_time_idx=np.array([2, 3]))
        # state block unchanged; two free-time columns appended.
        assert cp.n_state_block == 18
        assert cp.n_X == 20
        assert cp.free_time_idx == (2, 3)

    def test_free_time_idx_stored_as_tuple_of_int(self):
        cp = _build_column_plan(np.arange(6), n_seg=2,
                                free_time_idx=np.array([1, 2]))
        assert cp.free_time_idx == (1, 2)
        assert all(type(m) is int for m in cp.free_time_idx)


# --------------------------------------------------------------------------
# Column plan accessors
# --------------------------------------------------------------------------
class TestColumnPlanAccessors:
    """Span / component / free-time-column lookups over the column plan."""

    def test_start_span_and_components(self):
        cp = _build_column_plan(np.array([0, 1, 3, 4]), n_seg=3,
                                free_time_idx=np.array([], dtype=int))
        assert cp.start_span() == (0, 4)
        assert cp.start_components() == (0, 1, 3, 4)

    def test_junction_spans(self):
        cp = _build_column_plan(np.arange(6), n_seg=3,
                                free_time_idx=np.array([], dtype=int))
        assert cp.junction_span(0) == (6, 12)
        assert cp.junction_span(1) == (12, 18)

    def test_junction_components_full_six_in_phase1(self):
        cp = _build_column_plan(np.array([0, 1, 3, 4]), n_seg=3,
                                free_time_idx=np.array([], dtype=int))
        assert cp.junction_components(0) == ALL6
        assert cp.junction_components(1) == ALL6

    def test_junction_span_uses_running_sum_not_fixed_stride(self):
        # With a 4-wide start, junction 0 begins at 4, not at n_fs=6 for a
        # full start -- confirms spans follow col_start, ready for Phase 2.
        cp = _build_column_plan(np.array([0, 1, 3, 4]), n_seg=2,
                                free_time_idx=np.array([], dtype=int))
        assert cp.junction_span(0) == (4, 10)

    def test_free_time_column_maps_position(self):
        cp = _build_column_plan(np.arange(6), n_seg=3,
                                free_time_idx=np.array([2, 3]))
        assert cp.free_time_column(2) == 18   # n_state_block + position 0
        assert cp.free_time_column(3) == 19   # n_state_block + position 1

    def test_free_time_column_raises_for_non_free_time(self):
        cp = _build_column_plan(np.arange(6), n_seg=3,
                                free_time_idx=np.array([2]))
        with pytest.raises(ValueError):
            cp.free_time_column(1)            # boundary time 1 is not free

    def test_selector_width(self):
        assert _Selector((0, 1, 3, 4), 0).width == 4
        assert _Selector(ALL6, 6).width == 6


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
        b = _RowBlock(0, 6, _BlockKind.INTERIOR_DEFECT, 0)
        with pytest.raises(FrozenInstanceError):
            b.row_offset = 5                          # type: ignore

    def test_selector_is_frozen(self):
        s = _Selector((0, 1), 0)
        with pytest.raises(FrozenInstanceError):
            s.col_start = 3                           # type: ignore

    def test_column_plan_is_frozen(self):
        cp = _build_column_plan(np.arange(6), 2,
                                np.array([], dtype=int))
        with pytest.raises(FrozenInstanceError):
            cp.n_X = 99                               # type: ignore