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

from kyklos.shooter import (
    TerminalConstraint, TargetState, Periodicity, CallableConstraint,
    _ShootingContext,
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
