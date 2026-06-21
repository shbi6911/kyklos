"""
Tests for kyklos.shooter -- the differential corrector (shooting) module.

Organized in two tiers (see conftest.py for the fixtures and fakes):

  Tier 1 (fast, deterministic)
      Pure parsing, constraint math, packing, assembly, and the Newton
      control flow, driven by the conftest fakes where a Trajectory or
      System is needed. No integrator is compiled and nothing is
      propagated.

  Tier 2 (slow, numerical)
      Real Earth-Moon CR3BP propagation, for what only real dynamics can
      validate. Marked @pytest.mark.slow; run the fast tier alone with
      `pytest -m "not slow"`.

Private helpers (leading underscore) are imported and tested directly. That
is intentional: they carry the nontrivial parsing and assembly logic the
public surface depends on, and exercising them in isolation pins that logic
down independently of the end-to-end path.
"""

import numpy as np
import pytest
import types
from dataclasses import FrozenInstanceError
from typing import cast

from kyklos.shooter import (_parse_free_vars, _component_index, 
    _resolve_component_names, _finite_diff, _pack, _unpack, _assemble_F, _assemble_DF)
from kyklos.shooter import (TerminalConstraint, TargetState, 
    Periodicity, CallableConstraint, _ShootingContext, ShooterResult, 
    DifferentialCorrector)
from kyklos import NullJunctionNode, ImpulsiveJunctionNode
from kyklos import Trajectory, System, config

class TestParseFreeVars:
    """
    _parse_free_vars: free-variable specification -> sorted component indices.

    The function accepts either a category string ('all', 'position',
    'velocity', 'planar', 'none') or a list/tuple of component names, both
    case- and whitespace-insensitive, and returns a sorted 1-D integer array
    of start-state component indices. Expected values here are written out
    explicitly rather than derived from the module's own lookup tables, so a
    wrong table in the implementation cannot make a test pass.
    """

    # ---- category strings ----

    @pytest.mark.parametrize("category, expected", [
        ("all",      [0, 1, 2, 3, 4, 5]),
        ("position", [0, 1, 2]),
        ("velocity", [3, 4, 5]),
        ("planar",   [0, 1, 3, 4]),
        ("none",     []),
    ])
    def test_category_strings(self, category, expected):
        result = _parse_free_vars(category)
        np.testing.assert_array_equal(result, np.array(expected, dtype=int))

    @pytest.mark.parametrize("spelling", ["ALL", "All", " all ", "\tall\n"])
    def test_category_case_and_whitespace_insensitive(self, spelling):
        np.testing.assert_array_equal(
            _parse_free_vars(spelling),
            np.array([0, 1, 2, 3, 4, 5], dtype=int),
        )

    # ---- component-name lists ----

    @pytest.mark.parametrize("names, expected", [
        (["x", "y", "z"],                   [0, 1, 2]),
        (["vx", "vy", "vz"],                [3, 4, 5]),
        (["x", "z", "vy"],                  [0, 2, 4]),
        (["vz"],                            [5]),
        (["x", "y", "z", "vx", "vy", "vz"], [0, 1, 2, 3, 4, 5]),
    ])
    def test_component_name_lists(self, names, expected):
        np.testing.assert_array_equal(
            _parse_free_vars(names), np.array(expected, dtype=int)
        )

    def test_name_order_does_not_matter(self):
        # Documented contract: result is sorted ascending regardless of the
        # order the caller listed names in.
        a = _parse_free_vars(["vy", "x"])
        b = _parse_free_vars(["x", "vy"])
        np.testing.assert_array_equal(a, np.array([0, 4], dtype=int))
        np.testing.assert_array_equal(a, b)

    @pytest.mark.parametrize("names, expected", [
        (["X", " vy "], [0, 4]),
        (["  Z", "VX"], [2, 3]),
    ])
    def test_name_case_and_whitespace_insensitive(self, names, expected):
        np.testing.assert_array_equal(
            _parse_free_vars(names), np.array(expected, dtype=int)
        )

    def test_tuple_accepted_like_list(self):
        np.testing.assert_array_equal(
            _parse_free_vars(("x", "vy")), np.array([0, 4], dtype=int)
        )

    @pytest.mark.parametrize("empty", [[], ()])
    def test_empty_sequence_equivalent_to_none(self, empty):
        result = _parse_free_vars(empty)
        assert result.shape == (0,)
        np.testing.assert_array_equal(result, _parse_free_vars("none"))

    # ---- return contract ----

    def test_return_type_and_dtype(self):
        result = _parse_free_vars("planar")
        assert isinstance(result, np.ndarray)
        assert result.ndim == 1
        assert np.issubdtype(result.dtype, np.integer)

    def test_result_is_sorted_ascending(self):
        result = _parse_free_vars(["vz", "x", "vy", "y"])
        assert np.all(np.diff(result) > 0)

    # ---- errors ----

    def test_unknown_category_raises_valueerror(self):
        with pytest.raises(ValueError):
            _parse_free_vars("everything")

    def test_unknown_component_name_raises_valueerror(self):
        with pytest.raises(ValueError):
            _parse_free_vars(["x", "q"])

    def test_duplicate_component_raises_valueerror(self):
        with pytest.raises(ValueError):
            _parse_free_vars(["x", "x"])

    def test_duplicate_after_normalization_raises_valueerror(self):
        # 'X' and ' x ' normalize to the same component.
        with pytest.raises(ValueError):
            _parse_free_vars(["X", " x "])

    @pytest.mark.parametrize("bad_entry", [0, 1.5, None, ["x"]])
    def test_non_string_list_entry_raises_typeerror(self, bad_entry):
        with pytest.raises(TypeError):
            _parse_free_vars(["x", bad_entry])

    @pytest.mark.parametrize("bad_input", [5, 1.5, None, {"x": 0}, np.array([0, 1])])
    def test_wrong_top_level_type_raises_typeerror(self, bad_input):
        with pytest.raises(TypeError):
            _parse_free_vars(bad_input)
    
class TestComponentIndex:
    """_component_index: single component name -> index in [0, 6)."""

    @pytest.mark.parametrize("name, idx", [
        ("x", 0), ("y", 1), ("z", 2), ("vx", 3), ("vy", 4), ("vz", 5),
    ])
    def test_names_resolve(self, name, idx):
        assert _component_index(name) == idx

    @pytest.mark.parametrize("name, idx", [("X", 0), (" vy ", 4), ("\tVZ\n", 5)])
    def test_case_and_whitespace_insensitive(self, name, idx):
        assert _component_index(name) == idx

    def test_returns_plain_python_int(self):
        assert isinstance(_component_index("z"), int)

    @pytest.mark.parametrize("bad", ["q", "a", "vw", "xx", ""])
    def test_unknown_name_raises_valueerror(self, bad):
        with pytest.raises(ValueError):
            _component_index(bad)

    @pytest.mark.parametrize("bad", [0, None, ["x"], 1.5])
    def test_non_string_raises_typeerror(self, bad):
        with pytest.raises(TypeError):
            _component_index(bad)


class TestResolveComponentNames:
    """_resolve_component_names: name sequence -> sorted, unique index array."""

    @pytest.mark.parametrize("names, expected", [
        (["x", "z", "vy"],    [0, 2, 4]),
        (["z"],               [2]),
        (["vz", "vy", "vx"],  [3, 4, 5]),
        (["X", " vy "],       [0, 4]),
    ])
    def test_sorted_unique_indices(self, names, expected):
        np.testing.assert_array_equal(
            _resolve_component_names(names), np.array(expected, dtype=int)
        )

    def test_order_independent(self):
        np.testing.assert_array_equal(
            _resolve_component_names(["vy", "x"]), np.array([0, 4], dtype=int)
        )

    def test_tuple_accepted(self):
        np.testing.assert_array_equal(
            _resolve_component_names(("x", "vy")), np.array([0, 4], dtype=int)
        )

    def test_empty_sequence_returns_empty_array(self):
        # The function itself does not reject empty input -- that rejection
        # lives in the callers (e.g. Periodicity). Here it just yields (0,).
        result = _resolve_component_names([])
        assert result.shape == (0,)
        assert np.issubdtype(result.dtype, np.integer)

    def test_return_dtype_and_ndim(self):
        result = _resolve_component_names(["x", "vy"])
        assert result.ndim == 1
        assert np.issubdtype(result.dtype, np.integer)

    @pytest.mark.parametrize("names", [["x", "x"], ["X", " x "]])
    def test_duplicate_raises_valueerror(self, names):
        with pytest.raises(ValueError):
            _resolve_component_names(names)

    def test_unknown_name_raises_valueerror(self):
        with pytest.raises(ValueError):
            _resolve_component_names(["x", "q"])

    def test_non_string_entry_raises_typeerror(self):
        with pytest.raises(TypeError):
            _resolve_component_names(["x", 0]) # type: ignore


# Module-level helpers for the finite-difference tests: a nonlinear map and
# its analytic Jacobian, so the central difference is checked against a known
# truth rather than just exercised.
def _fd_sample_func(x):
    return np.array([x[0] ** 2, x[1] * x[2], np.sin(x[3])])


def _fd_sample_jacobian(x):
    return np.array([
        [2 * x[0], 0.0,  0.0,  0.0,           0.0, 0.0],
        [0.0,      x[2], x[1], 0.0,           0.0, 0.0],
        [0.0,      0.0,  0.0,  np.cos(x[3]),  0.0, 0.0],
    ])


class TestFiniteDiff:
    """
    _finite_diff: central-difference Jacobian of a vector function.

    Central differencing is exact (to rounding) for affine maps and
    second-order accurate otherwise, so linear/affine cases are checked
    tightly and the nonlinear case against its analytic Jacobian.
    """

    def test_linear_map_is_exact(self):
        rng = np.random.default_rng(0)
        A = rng.standard_normal((4, 6))
        x = rng.standard_normal(6)
        J = _finite_diff(lambda v: A @ v, x, 1e-6)
        np.testing.assert_allclose(J, A, atol=1e-8)

    def test_affine_constant_drops_out(self):
        rng = np.random.default_rng(1)
        A = rng.standard_normal((4, 6))
        b = rng.standard_normal(4)
        x = rng.standard_normal(6)
        J = _finite_diff(lambda v: A @ v + b, x, 1e-6)
        np.testing.assert_allclose(J, A, atol=1e-8)

    def test_nonlinear_matches_analytic_jacobian(self):
        x = np.array([0.5, -1.2, 0.3, 0.8, 2.0, -0.4])
        J = _finite_diff(_fd_sample_func, x, 1e-6)
        np.testing.assert_allclose(J, _fd_sample_jacobian(x), atol=1e-7)

    def test_jacobian_shape_nonsquare(self):
        # f: R^6 -> R^3 gives a (3, 6) Jacobian.
        J = _finite_diff(_fd_sample_func, np.zeros(6), 1e-6)
        assert J.shape == (3, 6)

    def test_scalar_output_gives_row_jacobian(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        J = _finite_diff(lambda v: np.sum(v ** 2), x, 1e-6)
        assert J.shape == (1, 6)
        np.testing.assert_allclose(J[0], 2 * x, atol=1e-6)

    def test_zero_point_has_no_nan(self):
        # The step floor (max(1, |x_i|)) keeps the step nonzero at x_i = 0,
        # so no division by zero.
        J = _finite_diff(lambda v: v ** 2, np.zeros(6), 1e-6)
        assert np.all(np.isfinite(J))
        np.testing.assert_allclose(J, np.zeros((6, 6)), atol=1e-12)

    def test_relative_step_handles_large_magnitude(self):
        x = np.array([1000.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        J = _finite_diff(lambda v: v ** 2, x, 1e-6)
        np.testing.assert_allclose(np.diag(J), 2 * x, rtol=1e-5)

# Shared sample states for the constraint tests. X0 sits on the xz-plane
# perpendicularly (y, vx, vz = 0), like a real symmetric-orbit start; STATE
# is an arbitrary nearby "final" state.
_STATE_TF = np.array([0.8, 0.1, -0.2, 0.05, 0.9, -0.03])
_X0 = np.array([0.82, 0.0, -0.18, 0.0, 0.88, 0.0])

class TestTargetState:
    """TargetState: drive selected final-state components to target values."""

    def test_residual_zero_targets(self):
        c = TargetState({'y': 0.0, 'vx': 0.0, 'vz': 0.0})
        np.testing.assert_allclose(c.residual(_STATE_TF, _X0), [0.1, 0.05, -0.03])

    def test_residual_nonzero_targets(self):
        c = TargetState({'x': 1.5, 'vy': -0.3})
        np.testing.assert_allclose(c.residual(_STATE_TF, _X0), [0.8 - 1.5, 0.9 + 0.3])

    def test_target_reordered_to_match_sorted_indices(self):
        # Keys given out of order: indices sort to [0, 5], and the target
        # values must be reordered to stay aligned with them.
        c = TargetState({'vz': 1.0, 'x': 2.0})
        np.testing.assert_array_equal(c._idx, [0, 5])
        np.testing.assert_allclose(c.residual(_STATE_TF, _X0),
                                   [0.8 - 2.0, -0.03 - 1.0])

    def test_jacobian_tf_is_selection_matrix(self):
        c = TargetState({'y': 0.0, 'vx': 0.0, 'vz': 0.0})
        expected = np.zeros((3, 6))
        expected[0, 1] = expected[1, 3] = expected[2, 5] = 1.0
        np.testing.assert_array_equal(c.jacobian_tf(_STATE_TF, _X0), expected)

    def test_jacobian_x0_is_zeros(self):
        c = TargetState({'y': 0.0, 'vx': 0.0, 'vz': 0.0})
        np.testing.assert_array_equal(c.jacobian_x0(_STATE_TF, _X0), np.zeros((3, 6)))

    def test_jacobian_tf_matches_finite_difference(self):
        c = TargetState({'x': 1.5, 'vy': -0.3})
        fd = _finite_diff(lambda s: c.residual(s, _X0), _STATE_TF, 1e-6)
        np.testing.assert_allclose(c.jacobian_tf(_STATE_TF, _X0), fd, atol=1e-8)

    @pytest.mark.parametrize("bad", [[], None, 5, "x"])
    def test_non_dict_raises_typeerror(self, bad):
        with pytest.raises(TypeError):
            TargetState(bad)

    def test_empty_dict_raises_valueerror(self):
        with pytest.raises(ValueError):
            TargetState({})

    def test_duplicate_after_normalization_raises_valueerror(self):
        with pytest.raises(ValueError):
            TargetState({'x': 1.0, 'X': 2.0})

    def test_unknown_component_raises_valueerror(self):
        with pytest.raises(ValueError):
            TargetState({'q': 0.0})


class TestPeriodicity:
    """Periodicity: final state equals start state on selected components."""

    def test_full_residual(self):
        np.testing.assert_allclose(Periodicity().residual(_STATE_TF, _X0),
                                   _STATE_TF - _X0)

    def test_full_jacobian_tf_is_identity(self):
        np.testing.assert_array_equal(Periodicity().jacobian_tf(_STATE_TF, _X0),
                                      np.eye(6))

    def test_full_jacobian_x0_is_negative_identity(self):
        # The distinguishing feature: Periodicity is the one built-in that
        # depends on x0, with jacobian_x0 = -selection.
        np.testing.assert_array_equal(Periodicity().jacobian_x0(_STATE_TF, _X0),
                                      -np.eye(6))

    def test_subset_residual(self):
        c = Periodicity(['x', 'z'])
        np.testing.assert_allclose(
            c.residual(_STATE_TF, _X0),
            [_STATE_TF[0] - _X0[0], _STATE_TF[2] - _X0[2]],
        )

    def test_subset_jacobian_tf(self):
        c = Periodicity(['x', 'z'])
        expected = np.zeros((2, 6))
        expected[0, 0] = expected[1, 2] = 1.0
        np.testing.assert_array_equal(c.jacobian_tf(_STATE_TF, _X0), expected)

    def test_subset_jacobian_x0(self):
        c = Periodicity(['x', 'z'])
        expected = np.zeros((2, 6))
        expected[0, 0] = expected[1, 2] = -1.0
        np.testing.assert_array_equal(c.jacobian_x0(_STATE_TF, _X0), expected)

    def test_jacobian_tf_matches_finite_difference(self):
        c = Periodicity()
        fd = _finite_diff(lambda s: c.residual(s, _X0), _STATE_TF, 1e-6)
        np.testing.assert_allclose(c.jacobian_tf(_STATE_TF, _X0), fd, atol=1e-8)

    def test_jacobian_x0_matches_finite_difference(self):
        c = Periodicity()
        fd = _finite_diff(lambda y: c.residual(_STATE_TF, y), _X0, 1e-6)
        np.testing.assert_allclose(c.jacobian_x0(_STATE_TF, _X0), fd, atol=1e-8)

    def test_empty_components_raises_valueerror(self):
        with pytest.raises(ValueError):
            Periodicity([])

    def test_unknown_component_raises_valueerror(self):
        with pytest.raises(ValueError):
            Periodicity(['x', 'q'])

    def test_duplicate_component_raises_valueerror(self):
        with pytest.raises(ValueError):
            Periodicity(['x', 'x'])


class TestCallableConstraint:
    """CallableConstraint: wrap user callables, with analytic or FD Jacobians."""

    def test_residual_wraps_g(self):
        c = CallableConstraint(lambda s, x: s[:2] - np.array([1.0, 2.0]))
        np.testing.assert_allclose(c.residual(_STATE_TF, _X0), [0.8 - 1.0, 0.1 - 2.0])

    def test_scalar_g_promoted_to_1d(self):
        c = CallableConstraint(lambda s, x: s[0] - x[0])
        r = c.residual(_STATE_TF, _X0)
        assert r.shape == (1,)
        np.testing.assert_allclose(r, [-0.02])

    def test_fd_fallback_matches_analytic(self):
        # No dg supplied -> central-difference fallback; check against the
        # known analytic Jacobian of the nonlinear residual.
        c = CallableConstraint(lambda s, x: np.array([s[0] ** 2, s[1] * s[2]]))
        expected = np.zeros((2, 6))
        expected[0, 0] = 2 * _STATE_TF[0]
        expected[1, 1] = _STATE_TF[2]
        expected[1, 2] = _STATE_TF[1]
        np.testing.assert_allclose(c.jacobian_tf(_STATE_TF, _X0), expected, atol=1e-6)

    def test_analytic_dg_is_used_not_fd(self):
        # A sentinel dg the FD path could never produce confirms dg is used.
        sentinel = np.full((2, 6), 7.0)
        c = CallableConstraint(lambda s, x: s[:2], dg=lambda s, x: sentinel)
        np.testing.assert_array_equal(c.jacobian_tf(_STATE_TF, _X0), sentinel)

    def test_one_d_dg_promoted_to_two_d(self):
        c = CallableConstraint(lambda s, x: s[0], dg=lambda s, x: np.arange(6.0))
        assert c.jacobian_tf(_STATE_TF, _X0).shape == (1, 6)

    def test_jacobian_x0_default_zeros(self):
        c = CallableConstraint(lambda s, x: s[:3])
        np.testing.assert_array_equal(c.jacobian_x0(_STATE_TF, _X0), np.zeros((3, 6)))

    def test_analytic_dg_dx0_is_used(self):
        sentinel = np.full((1, 6), -2.0)
        c = CallableConstraint(lambda s, x: s[0] - x[0],
                               dg_dx0=lambda s, x: sentinel)
        np.testing.assert_array_equal(c.jacobian_x0(_STATE_TF, _X0), sentinel)

    @pytest.mark.parametrize("g, dg, dg_dx0", [
        (5, None, None),
        (lambda s, x: s, 5, None),
        (lambda s, x: s, None, 5),
    ])
    def test_non_callable_raises_typeerror(self, g, dg, dg_dx0):
        with pytest.raises(TypeError):
            CallableConstraint(g, dg, dg_dx0)

    def test_bind_returns_self(self):
        c = CallableConstraint(lambda s, x: s[0])
        assert c.bind(object()) is c


# Minimal concrete subclasses for exercising the abstract base's defaults.
class _QuadraticConstraint(TerminalConstraint):
    def residual(self, state_tf, x0):
        s = np.asarray(state_tf, dtype=float)
        return np.array([s[0] ** 2, s[3]])


class _ScalarConstraint(TerminalConstraint):
    def residual(self, state_tf, x0):
        return np.array([np.asarray(state_tf, dtype=float)[0]])


class TestTerminalConstraintBase:
    """TerminalConstraint base: abstract residual, FD jacobian_tf, zero jacobian_x0."""

    def test_cannot_instantiate_abstract_base(self):
        with pytest.raises(TypeError):
            TerminalConstraint()            #type: ignore

    def test_default_jacobian_tf_is_finite_difference(self):
        c = _QuadraticConstraint()
        expected = np.zeros((2, 6))
        expected[0, 0] = 2 * _STATE_TF[0]
        expected[1, 3] = 1.0
        np.testing.assert_allclose(c.jacobian_tf(_STATE_TF, _X0), expected, atol=1e-6)

    def test_default_jacobian_x0_zeros_multi_row(self):
        np.testing.assert_array_equal(
            _QuadraticConstraint().jacobian_x0(_STATE_TF, _X0), np.zeros((2, 6))
        )

    def test_default_jacobian_x0_zeros_single_row(self):
        np.testing.assert_array_equal(
            _ScalarConstraint().jacobian_x0(_STATE_TF, _X0), np.zeros((1, 6))
        )

    def test_bind_returns_self(self):
        c = _QuadraticConstraint()
        assert c.bind(object()) is c

    def test_builtin_inherits_bind_noop(self):
        c = TargetState({'x': 0.0})
        assert c.bind(None) is c

# Builders for the fake-trajectory kwargs the context tests reuse.
def _single_segment_kwargs() -> dict[str, object]:
    s = np.array([0.8, 0.0, -0.2, 0.0, 0.9, 0.0])
    return dict(start_state=s, end_state=s, times=[0.0, 3.1], stms=[np.eye(6)])


def _three_segment_kwargs()-> dict[str, object]:
    s = np.zeros(6)
    return dict(
        start_state=s, end_state=s,
        junction_pre=[np.ones(6), 2 * np.ones(6)],
        junction_post=[np.ones(6), 2 * np.ones(6)],
        times=[0.0, 1.0, 2.0, 3.0], stms=[np.eye(6)] * 3,
    )


# Minimal constraints for exercising normalization and the bind lifecycle.
class _ConstA(TerminalConstraint):
    def residual(self, state_tf, x0):
        return np.zeros(2)


class _BindSpy(TerminalConstraint):
    """bind() returns a fresh instance tagged with the system it was bound to,
    so a test can confirm bind is actually called with the right system."""
    def __init__(self, bound_to="UNBOUND"):
        self.bound_to = bound_to

    def residual(self, state_tf, x0):
        return np.zeros(1)

    def bind(self, system):
        return _BindSpy(bound_to=system)


class TestShootingContextFromGuess:
    """_ShootingContext.from_guess: build and validate the solve context."""

    def test_single_segment_sizes(self, make_fake_trajectory):
        ctx = _ShootingContext.from_guess(
            make_fake_trajectory(**_single_segment_kwargs()), 'all')
        assert (ctx.n_seg, ctx.n_junction, ctx.n_free_start, ctx.n_free_time) == (1, 0, 6, 0)
        assert (ctx.n_state_block, ctx.n_X) == (6, 6)
        assert ctx.constraints == ()
        np.testing.assert_array_equal(ctx.free_idx, [0, 1, 2, 3, 4, 5])

    def test_multi_segment_sizes(self, make_fake_trajectory):
        ctx = _ShootingContext.from_guess(
            make_fake_trajectory(**_three_segment_kwargs()), 'position', free_times=[3])
        assert (ctx.n_seg, ctx.n_junction, ctx.n_free_start, ctx.n_free_time) == (3, 2, 3, 1)
        # state block = 3 free start + 6 * 2 junctions = 15; + 1 free time = 16
        assert (ctx.n_state_block, ctx.n_X) == (15, 16)

    def test_free_idx_from_name_list(self, make_fake_trajectory):
        ctx = _ShootingContext.from_guess(
            make_fake_trajectory(**_single_segment_kwargs()), ['x', 'vy'])
        np.testing.assert_array_equal(ctx.free_idx, [0, 4])
        assert ctx.n_X == 2

    def test_x0_ref_and_times_ref_from_guess(self, make_fake_trajectory):
        kw = _single_segment_kwargs()
        ctx = _ShootingContext.from_guess(make_fake_trajectory(**kw), 'all')
        np.testing.assert_array_equal(ctx.x0_ref, kw['start_state'])
        np.testing.assert_array_equal(ctx.times_ref, [0.0, 3.1])

    def test_system_stored(self, make_fake_trajectory):
        kw = _single_segment_kwargs()
        kw['system'] = "SYS"
        ctx = _ShootingContext.from_guess(make_fake_trajectory(**kw), 'all')
        assert ctx.system == "SYS"

    def test_constraints_none_gives_empty_tuple(self, make_fake_trajectory):
        ctx = _ShootingContext.from_guess(
            make_fake_trajectory(**_single_segment_kwargs()), 'all', constraints=None)
        assert ctx.constraints == ()

    def test_terminal_constraint_passes_through(self, make_fake_trajectory):
        c = _ConstA()
        ctx = _ShootingContext.from_guess(
            make_fake_trajectory(**_single_segment_kwargs()), 'all', constraints=[c])
        assert len(ctx.constraints) == 1 and ctx.constraints[0] is c

    def test_bare_callable_is_wrapped(self, make_fake_trajectory):
        ctx = _ShootingContext.from_guess(
            make_fake_trajectory(**_single_segment_kwargs()), 'all',
            constraints=[lambda s, x: s[:1]])
        assert isinstance(ctx.constraints[0], CallableConstraint)

    def test_mixed_constraints(self, make_fake_trajectory):
        ctx = _ShootingContext.from_guess(
            make_fake_trajectory(**_single_segment_kwargs()), 'all',
            constraints=[_ConstA(), lambda s, x: s[0]])
        assert len(ctx.constraints) == 2
        assert isinstance(ctx.constraints[1], CallableConstraint)

    def test_constraints_bound_with_system(self, make_fake_trajectory):
        kw = _single_segment_kwargs()
        kw['system'] = "SYS2"
        ctx = _ShootingContext.from_guess(
            make_fake_trajectory(**kw), 'all', constraints=[_BindSpy()])
        assert ctx.constraints[0].bound_to == "SYS2"

    def test_phase1_rejects_null_junction(self):
        bad = types.SimpleNamespace(
            n_segments=2,
            junction_nodes=[NullJunctionNode(1.0, np.zeros(6), np.zeros(6))],
        )
        with pytest.raises(NotImplementedError):
            _ShootingContext.from_guess(cast(Trajectory, bad), 'all')

    def test_phase1_rejects_impulsive_junction(self):
        node = ImpulsiveJunctionNode(1.0, pre_state=np.zeros(6), delta_v=[0.0, 0.0, 0.0])
        bad = types.SimpleNamespace(n_segments=2, junction_nodes=[node])
        with pytest.raises(NotImplementedError):
            _ShootingContext.from_guess(cast(Trajectory, bad), 'all')


def _make_ctx(free_idx: "np.ndarray | None" = None):
    """Build a _ShootingContext directly for the immutability tests."""
    return _ShootingContext(
        system=cast(System, None),    # never read by these tests
        n_seg=1,
        free_idx=np.array([0, 4]) if free_idx is None else free_idx,
        x0_ref=np.zeros(6),
        times_ref=np.array([0.0, 1.0]),
        free_time_idx=np.array([], dtype=int),
        constraints=(),
    )


class TestShootingContextImmutability:
    """The context is a frozen dataclass that owns read-only array copies."""

    def test_frozen_attribute_assignment_raises(self):
        with pytest.raises(FrozenInstanceError):
            _make_ctx().n_seg = 9

    @pytest.mark.parametrize("field", ["free_idx", "x0_ref", "times_ref", "free_time_idx"])
    def test_array_fields_are_read_only(self, field):
        arr = getattr(_make_ctx(), field)
        assert not arr.flags.writeable
        with pytest.raises(ValueError):
            arr[0] = 9.0

    def test_construction_copies_input_arrays(self):
        arr = np.array([0, 4])
        ctx = _make_ctx(free_idx=arr)
        assert ctx.free_idx is not arr
        assert not ctx.free_idx.flags.writeable
        arr[0] = 99           # mutate the caller's array
        assert ctx.free_idx[0] == 0   # context is unaffected


class TestParseFreeTimes:
    """_ShootingContext._parse_free_times: validate/canonicalize free-time indices."""

    pft = staticmethod(_ShootingContext._parse_free_times)

    def test_none_is_empty(self):
        assert self.pft(None, 3).shape == (0,)

    def test_empty_list_is_empty(self):
        assert self.pft([], 3).shape == (0,)

    @pytest.mark.parametrize("free_times, n_seg, expected", [
        ([1, 2], 3, [1, 2]),
        ([3], 3, [3]),           # n_seg index = final time, allowed
        ([2, 1], 3, [1, 2]),     # sorted
        ((1, 2), 3, [1, 2]),     # tuple accepted
    ])
    def test_valid_indices(self, free_times, n_seg, expected):
        np.testing.assert_array_equal(self.pft(free_times, n_seg), expected)

    def test_numpy_integer_accepted(self):
        np.testing.assert_array_equal(self.pft([np.int64(2)], 3), [2])

    def test_returns_integer_dtype(self):
        assert np.issubdtype(self.pft([1, 2], 3).dtype, np.integer)

    @pytest.mark.parametrize("free_times, n_seg", [([0], 3), ([4], 3), ([-1], 3)])
    def test_out_of_range_or_t0_raises_valueerror(self, free_times, n_seg):
        with pytest.raises(ValueError):
            self.pft(free_times, n_seg)

    def test_duplicate_raises_valueerror(self):
        with pytest.raises(ValueError):
            self.pft([1, 1], 3)

    @pytest.mark.parametrize("free_times", [[True], [1.5], ["1"]])
    def test_bad_entry_type_raises_typeerror(self, free_times):
        # bool is explicitly rejected even though it subclasses int.
        with pytest.raises(TypeError):
            self.pft(free_times, 3)

    @pytest.mark.parametrize("free_times", [5, {1, 2}, np.array([1, 2])])
    def test_non_sequence_raises_typeerror(self, free_times):
        with pytest.raises(TypeError):
            self.pft(free_times, 3)


class TestValidateConstraints:
    """_ShootingContext._validate_constraints: normalize, wrap, and bind."""

    vc = staticmethod(_ShootingContext._validate_constraints)

    def test_none_gives_empty_tuple(self):
        assert self.vc(None, None) == ()

    def test_empty_gives_empty_tuple(self):
        assert self.vc([], None) == ()

    def test_callable_is_wrapped(self):
        out = self.vc([lambda s, x: s[:1]], None)
        assert isinstance(out[0], CallableConstraint)

    def test_terminal_constraint_passes_through(self):
        c = _ConstA()
        assert self.vc([c], None)[0] is c

    def test_mixed(self):
        out = self.vc([_ConstA(), lambda s, x: s[0]], None)
        assert len(out) == 2 and isinstance(out[1], CallableConstraint)

    @pytest.mark.parametrize("bad", [_ConstA(), 5])
    def test_non_list_raises_typeerror(self, bad):
        # A single constraint must still be wrapped in a list.
        with pytest.raises(TypeError):
            self.vc(bad, None)

    @pytest.mark.parametrize("bad", [[5], [None]])
    def test_bad_element_raises_typeerror(self, bad):
        with pytest.raises(TypeError):
            self.vc(bad, None)

    def test_bind_called_with_system(self):
        spy = _BindSpy()
        out = self.vc([spy], "SYS")
        assert out[0] is not spy
        assert out[0].bound_to == "SYS"

# Sample states for the pack/unpack scenarios. Distinct magnitudes per slot so
# a misplaced component shows up immediately in an assertion.
S0 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
JPOST = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
JPOST2 = np.array([110.0, 111.0, 112.0, 113.0, 114.0, 115.0])

@pytest.fixture
def ctx_and_traj(make_fake_trajectory):
    """Build a template trajectory and the context derived from it, for a
    given free-variable / free-time / segment-count configuration."""
    def _build(free_vars, free_times=None, n_seg=2):
        if n_seg == 1:
            kw = dict(start_state=S0, end_state=S0, times=[0.0, 1.0],
                      stms=[np.eye(6)])
        elif n_seg == 2:
            kw = dict(start_state=S0, end_state=np.zeros(6),
                      junction_pre=[JPOST], junction_post=[JPOST],
                      times=[0.0, 1.0, 2.0], stms=[np.eye(6)] * 2)
        else:  # n_seg == 3
            kw = dict(start_state=S0, end_state=np.zeros(6),
                      junction_pre=[JPOST, JPOST2], junction_post=[JPOST, JPOST2],
                      times=[0.0, 1.0, 2.0, 3.0], stms=[np.eye(6)] * 3)
        traj = make_fake_trajectory(**kw)
        return traj, _ShootingContext.from_guess(traj, free_vars, free_times=free_times)
    return _build

class TestPack:
    """_pack: trajectory -> X, laid out [start-free | junction posts | free times]."""

    def test_single_segment_all_free(self, ctx_and_traj):
        traj, ctx = ctx_and_traj('all', n_seg=1)
        np.testing.assert_array_equal(_pack(traj, ctx), S0)

    def test_single_segment_subset_free(self, ctx_and_traj):
        traj, ctx = ctx_and_traj(['x', 'vy'], n_seg=1)
        np.testing.assert_array_equal(_pack(traj, ctx), S0[[0, 4]])

    def test_multi_segment_layout(self, ctx_and_traj):
        traj, ctx = ctx_and_traj(['x', 'vy'], free_times=[2], n_seg=2)
        X = _pack(traj, ctx)
        np.testing.assert_array_equal(X, np.concatenate([S0[[0, 4]], JPOST, [2.0]]))
        assert X.shape == (9,)

    def test_three_segment_ordering(self, ctx_and_traj):
        # Junction posts appear in segment order: JPOST then JPOST2.
        traj, ctx = ctx_and_traj('position', n_seg=3)
        np.testing.assert_array_equal(
            _pack(traj, ctx), np.concatenate([S0[[0, 1, 2]], JPOST, JPOST2])
        )

    def test_none_free_vars_omits_start_block(self, ctx_and_traj):
        # No free start components -> X is just the junction posts.
        traj, ctx = ctx_and_traj('none', n_seg=2)
        np.testing.assert_array_equal(_pack(traj, ctx), JPOST)

    def test_shape_matches_n_X(self, ctx_and_traj):
        traj, ctx = ctx_and_traj(['x', 'vy'], free_times=[2], n_seg=2)
        assert _pack(traj, ctx).shape == (ctx.n_X,)

    def test_segment_count_mismatch_raises(self, ctx_and_traj, make_fake_trajectory):
        _, ctx = ctx_and_traj('all', n_seg=2)
        one_seg = make_fake_trajectory(start_state=S0, end_state=S0,
                                       times=[0.0, 1.0], stms=[np.eye(6)])
        with pytest.raises(ValueError):
            _pack(one_seg, ctx)

class TestUnpack:
    """_unpack: X -> (initial-condition list, full times array)."""

    def test_single_segment_all_free(self, ctx_and_traj):
        _, ctx = ctx_and_traj('all', n_seg=1)
        X = np.array([7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
        ics, times = _unpack(X, ctx)
        assert len(ics) == 1
        np.testing.assert_array_equal(ics[0], X)
        np.testing.assert_array_equal(times, [0.0, 1.0])

    def test_fixed_components_from_ref_free_from_X(self, ctx_and_traj):
        # free_idx = [0, 4]; X supplies those, the rest come from x0_ref (= S0).
        _, ctx = ctx_and_traj(['x', 'vy'], n_seg=1)
        ics, _ = _unpack(np.array([99.0, 88.0]), ctx)
        np.testing.assert_array_equal(ics[0], [99.0, 2.0, 3.0, 4.0, 88.0, 6.0])

    def test_times_fixed_from_ref_free_from_X(self, ctx_and_traj):
        # Only t[2] is free; t[0], t[1] stay at the reference values.
        _, ctx = ctx_and_traj('all', free_times=[2], n_seg=2)
        X = np.concatenate([np.arange(6.0), np.arange(6.0) + 20, [5.0]])
        ics, times = _unpack(X, ctx)
        np.testing.assert_array_equal(times, [0.0, 1.0, 5.0])
        np.testing.assert_array_equal(ics[0], np.arange(6.0))
        np.testing.assert_array_equal(ics[1], np.arange(6.0) + 20)

    def test_ics_length_matches_segments(self, ctx_and_traj):
        _, ctx = ctx_and_traj('all', n_seg=3)
        ics, _ = _unpack(np.zeros(ctx.n_X), ctx)
        assert len(ics) == 3

    def test_times_length_matches_segments_plus_one(self, ctx_and_traj):
        _, ctx = ctx_and_traj('all', n_seg=3)
        _, times = _unpack(np.zeros(ctx.n_X), ctx)
        assert len(times) == 4

    def test_ics_are_writable_independent_copies(self, ctx_and_traj):
        _, ctx = ctx_and_traj('all', free_times=[2], n_seg=2)
        X = np.arange(13.0)
        ics, _ = _unpack(X, ctx)
        assert ics[0].flags.writeable and ics[1].flags.writeable
        ics[0][0] = 999.0
        ics[1][0] = 999.0
        np.testing.assert_array_equal(ctx.x0_ref, S0)        # source untouched
        np.testing.assert_array_equal(X, np.arange(13.0))    # input X untouched

    def test_times_is_independent_copy(self, ctx_and_traj):
        _, ctx = ctx_and_traj('all', free_times=[2], n_seg=2)
        _, times = _unpack(np.arange(13.0), ctx)
        times[0] = 999.0
        np.testing.assert_array_equal(ctx.times_ref, [0.0, 1.0, 2.0])

    @pytest.mark.parametrize("delta", [-1, 1])
    def test_wrong_length_raises(self, ctx_and_traj, delta):
        _, ctx = ctx_and_traj(['x', 'vy'], free_times=[2], n_seg=2)
        with pytest.raises(ValueError):
            _unpack(np.zeros(ctx.n_X + delta), ctx)

    def test_wrong_ndim_raises(self, ctx_and_traj):
        _, ctx = ctx_and_traj(['x', 'vy'], free_times=[2], n_seg=2)
        with pytest.raises(ValueError):
            _unpack(np.zeros((ctx.n_X, 1)), ctx)


class TestPackUnpackRoundTrip:
    """_pack and _unpack are inverses at the free-variable / free-time level."""

    def test_unpack_then_pack_recovers_X(self, ctx_and_traj, make_fake_trajectory):
        _, ctx = ctx_and_traj(['x', 'vy'], free_times=[2], n_seg=2)
        X = np.arange(ctx.n_X, dtype=float) + 0.5
        ics, times = _unpack(X, ctx)
        rebuilt = make_fake_trajectory(
            start_state=ics[0], end_state=np.zeros(6),
            junction_pre=ics[1:], junction_post=ics[1:],
            times=times, stms=[np.eye(6)] * 2,
        )
        np.testing.assert_array_equal(_pack(rebuilt, ctx), X)

    def test_pack_then_unpack_recovers_trajectory(self, ctx_and_traj):
        traj, ctx = ctx_and_traj(['x', 'vy'], free_times=[2], n_seg=2)
        ics, times = _unpack(_pack(traj, ctx), ctx)
        np.testing.assert_array_equal(ics[0], S0)
        np.testing.assert_array_equal(ics[1], JPOST)
        np.testing.assert_array_equal(times, [0.0, 1.0, 2.0])

# States and STMs for the assembly tests. Distinct per-slot magnitudes, and
# distinctive (seeded-random) STMs, so a transposed or misplaced block shows
# up immediately. _phi(seed) is deterministic, so the same call reproduces the
# same matrix in both the trajectory and the expected value.
_P0 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
_P1 = np.array([1.1, 1.2, 1.3, 1.4, 1.5, 1.6])
_TF = np.array([7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
_ZERO6 = np.zeros(6)
_I6 = np.eye(6)


def _phi(seed):
    return np.random.default_rng(seed).standard_normal((6, 6))

class TestAssembleF:
    """_assemble_F: [interior defects (post - pre) | terminal residuals]."""

    def test_interior_defects_two_segment(self, make_fake_trajectory):
        traj = make_fake_trajectory(
            start_state=S0, end_state=_ZERO6, junction_pre=[_P0],
            junction_post=[_P1], times=[0.0, 1.0, 2.0], stms=[_I6, _I6])
        ctx = _ShootingContext.from_guess(traj, 'all')
        np.testing.assert_allclose(_assemble_F(traj, ctx), _P1 - _P0)

    def test_interior_defects_three_segment(self, make_fake_trajectory):
        traj = make_fake_trajectory(
            start_state=S0, end_state=_ZERO6, junction_pre=[_P0, _P1],
            junction_post=[_P1, _P0], times=[0.0, 1.0, 2.0, 3.0], stms=[_I6] * 3)
        ctx = _ShootingContext.from_guess(traj, 'all')
        np.testing.assert_allclose(
            _assemble_F(traj, ctx), np.concatenate([_P1 - _P0, _P0 - _P1]))

    def test_terminal_only_periodicity(self, make_fake_trajectory):
        # No junctions: F is just the periodicity residual state_tf - x0,
        # which also confirms state_tf = end_node.pre_state, x0 = start post.
        traj = make_fake_trajectory(start_state=S0, end_state=_TF,
                                    times=[0.0, 1.0], stms=[_I6])
        ctx = _ShootingContext.from_guess(traj, 'all', constraints=[Periodicity()])
        np.testing.assert_allclose(_assemble_F(traj, ctx), _TF - S0)

    def test_terminal_only_targetstate(self, make_fake_trajectory):
        # residual = actual - target, reading the targeted components of state_tf.
        traj = make_fake_trajectory(start_state=S0, end_state=_TF,
                                    times=[0.0, 1.0], stms=[_I6])
        ctx = _ShootingContext.from_guess(
            traj, 'all', constraints=[TargetState({'y': 0.0, 'vx': 0.0, 'vz': 0.0})])
        np.testing.assert_allclose(_assemble_F(traj, ctx), [_TF[1], _TF[3], _TF[5]])

    def test_both_blocks_in_order(self, make_fake_trajectory):
        # Interior defects come first, then terminal residuals.
        traj = make_fake_trajectory(
            start_state=S0, end_state=_TF, junction_pre=[_P0],
            junction_post=[_P1], times=[0.0, 1.0, 2.0], stms=[_I6, _I6])
        ctx = _ShootingContext.from_guess(traj, 'all', constraints=[Periodicity()])
        np.testing.assert_allclose(
            _assemble_F(traj, ctx), np.concatenate([_P1 - _P0, _TF - S0]))

    def test_multiple_constraints_in_order(self, make_fake_trajectory):
        traj = make_fake_trajectory(start_state=S0, end_state=_TF,
                                    times=[0.0, 1.0], stms=[_I6])
        ctx = _ShootingContext.from_guess(
            traj, 'all', constraints=[TargetState({'x': 0.0}), Periodicity(['y'])])
        np.testing.assert_allclose(_assemble_F(traj, ctx), [_TF[0], _TF[1] - S0[1]])

    def test_empty_when_no_junctions_or_constraints(self, make_fake_trajectory):
        traj = make_fake_trajectory(start_state=S0, end_state=_TF,
                                    times=[0.0, 1.0], stms=[_I6])
        ctx = _ShootingContext.from_guess(traj, 'all')
        assert _assemble_F(traj, ctx).shape == (0,)

    def test_undefined_defect_raises_runtimeerror(self):
        # The defensive guard for a junction whose defect is None.
        traj = types.SimpleNamespace(
            junction_nodes=[types.SimpleNamespace(state_defect=None)])
        ctx = types.SimpleNamespace(constraints=())
        with pytest.raises(RuntimeError):
            _assemble_F(cast(Trajectory, traj), cast(_ShootingContext, ctx))

    def test_shape(self, make_fake_trajectory):
        traj = make_fake_trajectory(
            start_state=S0, end_state=_TF, junction_pre=[_P0],
            junction_post=[_P1], times=[0.0, 1.0, 2.0], stms=[_I6, _I6])
        ctx = _ShootingContext.from_guess(traj, 'all', constraints=[Periodicity()])
        assert _assemble_F(traj, ctx).shape == (12,)   # 6 defect + 6 periodicity


class TestAssembleDFStateColumns:
    """_assemble_DF: state-column block placement, with prescribed STMs."""

    def test_interior_single_junction_all_free(self, make_fake_trajectory):
        phi0 = _phi(0)
        traj = make_fake_trajectory(
            start_state=S0, end_state=_ZERO6, junction_pre=[_P0],
            junction_post=[_P1], times=[0.0, 1.0, 2.0], stms=[phi0, _phi(9)])
        ctx = _ShootingContext.from_guess(traj, 'all')
        DF = _assemble_DF(traj, ctx)
        assert DF.shape == (6, 12)
        np.testing.assert_allclose(DF[:, 0:6], -phi0)   # -Phi_0 in start cols
        np.testing.assert_allclose(DF[:, 6:12], _I6)    # +I in junction post

    def test_interior_subset_free_restricts_stm_columns(self, make_fake_trajectory):
        phi0 = _phi(1)
        traj = make_fake_trajectory(
            start_state=S0, end_state=_ZERO6, junction_pre=[_P0],
            junction_post=[_P1], times=[0.0, 1.0, 2.0], stms=[phi0, _I6])
        ctx = _ShootingContext.from_guess(traj, ['x', 'vy'])
        DF = _assemble_DF(traj, ctx)
        assert DF.shape == (6, 8)
        np.testing.assert_allclose(DF[:, 0:2], -phi0[:, [0, 4]])  # only free cols
        np.testing.assert_allclose(DF[:, 2:8], _I6)

    def test_three_segment_predecessor_placement(self, make_fake_trajectory):
        # The key structural check: junction 1's -Phi_1 lands in junction 0's
        # post-state columns (its predecessor), not the start columns.
        phi0, phi1 = _phi(0), _phi(1)
        traj = make_fake_trajectory(
            start_state=S0, end_state=_ZERO6, junction_pre=[_P0, _P1],
            junction_post=[_P1, _P0], times=[0.0, 1.0, 2.0, 3.0],
            stms=[phi0, phi1, _phi(2)])
        ctx = _ShootingContext.from_guess(traj, 'all')
        DF = _assemble_DF(traj, ctx)
        assert DF.shape == (12, 18)
        np.testing.assert_allclose(DF[0:6, 0:6], -phi0)
        np.testing.assert_allclose(DF[0:6, 6:12], _I6)
        np.testing.assert_allclose(DF[0:6, 12:18], 0.0)
        np.testing.assert_allclose(DF[6:12, 0:6], 0.0)
        np.testing.assert_allclose(DF[6:12, 6:12], -phi1)   # predecessor
        np.testing.assert_allclose(DF[6:12, 12:18], _I6)

    def test_terminal_periodicity_single_shooting_is_phi_minus_I(self, make_fake_trajectory):
        phi0 = _phi(3)
        traj = make_fake_trajectory(start_state=S0, end_state=_TF,
                                    times=[0.0, 1.0], stms=[phi0])
        ctx = _ShootingContext.from_guess(traj, 'all', constraints=[Periodicity()])
        DF = _assemble_DF(traj, ctx)
        assert DF.shape == (6, 6)
        np.testing.assert_allclose(DF, phi0 - _I6)

    def test_terminal_targetstate_single_shooting_selects_rows(self, make_fake_trajectory):
        phi0 = _phi(4)
        traj = make_fake_trajectory(start_state=S0, end_state=_TF,
                                    times=[0.0, 1.0], stms=[phi0])
        ctx = _ShootingContext.from_guess(
            traj, 'all', constraints=[TargetState({'y': 0.0, 'vx': 0.0, 'vz': 0.0})])
        DF = _assemble_DF(traj, ctx)
        assert DF.shape == (3, 6)
        np.testing.assert_allclose(DF, phi0[[1, 3, 5], :])

    def test_interior_plus_terminal_two_segment(self, make_fake_trajectory):
        # Terminal state depends on the last junction post via Phi_last (S_tf),
        # plus the direct -I from periodicity's x0 dependence in the start cols.
        phi0, phi1 = _phi(0), _phi(1)
        traj = make_fake_trajectory(
            start_state=S0, end_state=_TF, junction_pre=[_P0],
            junction_post=[_P1], times=[0.0, 1.0, 2.0], stms=[phi0, phi1])
        ctx = _ShootingContext.from_guess(traj, 'all', constraints=[Periodicity()])
        DF = _assemble_DF(traj, ctx)
        assert DF.shape == (12, 12)
        np.testing.assert_allclose(DF[0:6, 0:6], -phi0)
        np.testing.assert_allclose(DF[0:6, 6:12], _I6)
        np.testing.assert_allclose(DF[6:12, 0:6], -_I6)     # x0 dependence
        np.testing.assert_allclose(DF[6:12, 6:12], phi1)    # final state via Phi_last


class TestAssembleDFFreeTimeColumns:
    """_assemble_DF: free-time columns, with an identity-field fake system so
    f(endpoint) equals the endpoint itself and every contribution is exact."""

    def test_interior_junction_time(self, make_fake_trajectory, make_fake_system):
        # t_1 is the end of segment 0: contributes -f(e_0) to defect F_0.
        traj = make_fake_trajectory(
            start_state=S0, end_state=_TF, junction_pre=[_P0], junction_post=[_ZERO6],
            times=[0.0, 1.0, 2.0], stms=[_I6, _I6], system=make_fake_system(_I6))
        ctx = _ShootingContext.from_guess(traj, 'all', free_times=[1])
        DF = _assemble_DF(traj, ctx)
        assert DF.shape == (6, 13)
        np.testing.assert_allclose(DF[:, 12], -_P0)

    def test_mid_segment_start_time(self, make_fake_trajectory, make_fake_system):
        # t_1 is the end of segment 0 (-f(e_0) on F_0) AND the start of segment 1
        # (+f(e_1) on F_1).
        traj = make_fake_trajectory(
            start_state=S0, end_state=_TF, junction_pre=[_P0, _P1],
            junction_post=[_ZERO6, _ZERO6], times=[0.0, 1.0, 2.0, 3.0],
            stms=[_I6] * 3, system=make_fake_system(_I6))
        ctx = _ShootingContext.from_guess(traj, 'all', free_times=[1])
        DF = _assemble_DF(traj, ctx)
        assert DF.shape == (12, 19)
        np.testing.assert_allclose(DF[0:6, 18], -_P0)
        np.testing.assert_allclose(DF[6:12, 18], _P1)

    def test_terminal_final_time(self, make_fake_trajectory, make_fake_system):
        # t_N (the final time) moves the final state: +Jtf @ f(state_tf).
        traj = make_fake_trajectory(
            start_state=S0, end_state=_TF, times=[0.0, 1.0], stms=[_I6],
            system=make_fake_system(_I6))
        ctx = _ShootingContext.from_guess(
            traj, 'all', constraints=[Periodicity()], free_times=[1])
        DF = _assemble_DF(traj, ctx)
        assert DF.shape == (6, 7)
        np.testing.assert_allclose(DF[:, 6], _TF)   # +I @ state_tf

    def test_terminal_last_junction_time(self, make_fake_trajectory, make_fake_system):
        # t_{N-1} is both an interior end-time (-f on F_0) and moves the final
        # state with the negative sign (-Jtf @ f(state_tf)).
        traj = make_fake_trajectory(
            start_state=S0, end_state=_TF, junction_pre=[_P0], junction_post=[_ZERO6],
            times=[0.0, 1.0, 2.0], stms=[_I6, _I6], system=make_fake_system(_I6))
        ctx = _ShootingContext.from_guess(
            traj, 'all', constraints=[Periodicity()], free_times=[1])
        DF = _assemble_DF(traj, ctx)
        assert DF.shape == (12, 13)
        np.testing.assert_allclose(DF[0:6, 12], -_P0)   # interior, end of seg 0
        np.testing.assert_allclose(DF[6:12, 12], -_TF)  # terminal, negative sign

class TestShooterResult:
    """ShooterResult: dataclass fields, optional defaults, and custom repr."""

    def test_required_fields(self):
        r = ShooterResult(trajectory=None, converged=True, iterations=3,
                          final_residual=1.5e-12)
        assert r.trajectory is None and r.converged is True
        assert r.iterations == 3 and r.final_residual == 1.5e-12

    def test_optional_fields_default_none(self):
        r = ShooterResult(trajectory=None, converged=True, iterations=3,
                          final_residual=1e-12)
        assert r.abort_reason is None
        assert r.diagnostics is None
        assert r.iterates is None

    def test_optional_fields_settable(self):
        r = ShooterResult(trajectory=None, converged=False, iterations=5,
                          final_residual=2.0, abort_reason="cond_fail",
                          diagnostics={'a': 1}, iterates=[1, 2])
        assert r.abort_reason == "cond_fail"
        assert r.diagnostics == {'a': 1}
        assert r.iterates == [1, 2]

    def test_repr_converged(self):
        # repr=False on the dataclass, so the custom __repr__ is what runs;
        # final_residual is formatted with three significant decimals (.3e).
        r = ShooterResult(trajectory=None, converged=True, iterations=4,
                          final_residual=1.5e-10)
        assert repr(r) == "ShooterResult(converged, iterations=4, final_residual=1.500e-10)"

    def test_repr_not_converged(self):
        r = ShooterResult(trajectory=None, converged=False, iterations=7,
                          final_residual=3.2e-3)
        assert repr(r) == "ShooterResult(NOT converged, iterations=7, final_residual=3.200e-03)"


class TestDifferentialCorrectorConfig:
    """DifferentialCorrector.__init__: config defaults, None-sentinel, coercion."""

    def test_defaults_from_config(self):
        dc = DifferentialCorrector()
        assert dc.tol == config.SHOOTER_TOL
        assert dc.max_iter == config.SHOOTER_MAX_ITER
        assert dc.cond_warn == config.SHOOTER_COND_WARN
        assert dc.cond_fail == config.SHOOTER_COND_FAIL

    def test_explicit_overrides(self):
        dc = DifferentialCorrector(tol=1e-8, max_iter=20, cond_warn=1e6, cond_fail=1e10)
        assert (dc.tol, dc.max_iter, dc.cond_warn, dc.cond_fail) == (1e-8, 20, 1e6, 1e10)

    def test_tol_coerced_to_float(self):
        dc = DifferentialCorrector(tol=1)
        assert dc.tol == 1.0 and isinstance(dc.tol, float)

    def test_max_iter_coerced_to_int(self):
        dc = DifferentialCorrector(max_iter=np.int64(30))
        assert dc.max_iter == 30 and isinstance(dc.max_iter, int)

    def test_cond_thresholds_coerced_to_float(self):
        dc = DifferentialCorrector(cond_warn=1, cond_fail=2)
        assert isinstance(dc.cond_warn, float) and isinstance(dc.cond_fail, float)

    def test_none_sentinel_resolves_each_field_independently(self):
        # Setting one field leaves the other three drawing from config.
        dc = DifferentialCorrector(tol=1e-8)
        assert dc.tol == 1e-8
        assert dc.max_iter == config.SHOOTER_MAX_ITER
        assert dc.cond_warn == config.SHOOTER_COND_WARN
        assert dc.cond_fail == config.SHOOTER_COND_FAIL

    def test_partial_mix(self):
        dc = DifferentialCorrector(max_iter=5, cond_fail=1e9)
        assert dc.max_iter == 5 and dc.cond_fail == 1e9
        assert dc.tol == config.SHOOTER_TOL and dc.cond_warn == config.SHOOTER_COND_WARN

# Affine map for the linear orchestration tests. Diagonal entries away from 1
# so (I - M) is invertible and well-conditioned, giving a clean fixed point.
_LIN_M = np.diag([0.3, 0.5, 0.7, 1.4, 1.6, 1.8])
_LIN_B = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])


def _linear_fixed_point():
    # x* solving M x* + b = x*  (the periodic point of the affine map).
    return np.linalg.solve(np.eye(6) - _LIN_M, _LIN_B)


def _linear_guess(make_fake_trajectory, system, x0, n_seg=1):
    if n_seg == 1:
        return make_fake_trajectory(start_state=x0, end_state=np.zeros(6),
                                    times=[0.0, 1.0], stms=[np.eye(6)], system=system)
    return make_fake_trajectory(
        start_state=x0, end_state=np.zeros(6),
        junction_pre=[np.ones(6)], junction_post=[np.ones(6)],
        times=[0.0, 1.0, 2.0], stms=[np.eye(6)] * 2, system=system)

def _linear_system(make_linear_system, n_seg=1, M=_LIN_M, b=_LIN_B):
    # One affine map per segment; the map count must match the guess's n_seg.
    return make_linear_system([(M, b)] * n_seg)


# Bare-system stand-ins for the two non-numerical abort paths.
class _RaisingSystem:
    def propagate(self, ics, times, with_stm=True):
        raise RuntimeError("boom")


class _NonFiniteSystem:
    """Returns a trajectory-surface with a non-finite final state, so the
    residual comes out non-finite and _run takes the graceful-abort path.
    Uses a SimpleNamespace rather than FakeTrajectory: the conftest class
    isn't in scope here, and only this minimal surface is read."""
    def propagate(self, ics, times, with_stm=True):
        x0 = np.asarray(ics[0], dtype=float)
        return types.SimpleNamespace(
            junction_nodes=[],
            start_node=types.SimpleNamespace(post_state=x0),
            end_node=types.SimpleNamespace(pre_state=np.full(6, np.nan)),
        )


class TestSolveConvergence:
    """solve on the linear fake: exact one-step Newton convergence."""

    def test_single_shoot_converges_in_one_step(self, make_fake_trajectory, 
                                                make_linear_system):
        system = _linear_system(make_linear_system)
        guess = _linear_guess(make_fake_trajectory, system, np.ones(6))
        result = DifferentialCorrector().solve(guess, 'all', constraints=[Periodicity()])
        assert result.converged and result.iterations == 1
        assert result.final_residual < DifferentialCorrector().tol
        assert result.trajectory is not None
        start = result.trajectory.start_node.post_state
        assert start is not None
        np.testing.assert_allclose(start, _linear_fixed_point(), atol=1e-8)

    def test_guess_already_converged_takes_zero_iterations(self, make_fake_trajectory, 
                                                           make_linear_system):
        system = _linear_system(make_linear_system)
        guess = _linear_guess(make_fake_trajectory, system, _linear_fixed_point())
        result = DifferentialCorrector().solve(guess, 'all', constraints=[Periodicity()])
        assert result.converged and result.iterations == 0

    def test_multi_shoot_converges_and_finalizes_to_null(self, make_fake_trajectory, 
                                                         make_linear_system):
        system = _linear_system(make_linear_system, n_seg=2)
        guess = _linear_guess(make_fake_trajectory, system, np.ones(6), n_seg=2)
        result = DifferentialCorrector().solve(guess, 'all', constraints=[Periodicity()])
        assert result.converged
        assert len(result.trajectory.junction_nodes) == 1
        assert all(isinstance(n, NullJunctionNode) for n in 
                   result.trajectory.junction_nodes)


class TestSolveAborts:
    """The three graceful aborts: a recorded reason, converged False, no raise."""

    def test_propagation_failure(self, make_fake_trajectory):
        guess = _linear_guess(make_fake_trajectory, _RaisingSystem(), np.ones(6))
        result = DifferentialCorrector().solve(guess, 'all', constraints=[Periodicity()])
        assert not result.converged
        assert result.trajectory is None          # first propagation failed
        assert result.abort_reason is not None
        assert "propagation failed" in result.abort_reason

    def test_non_finite_residual(self, make_fake_trajectory):
        guess = _linear_guess(make_fake_trajectory, _NonFiniteSystem(), np.ones(6))
        result = DifferentialCorrector().solve(guess, 'all', constraints=[Periodicity()])
        assert not result.converged
        assert result.abort_reason is not None
        assert "non-finite" in result.abort_reason

    def test_condition_number_exceeds_cond_fail(self, make_fake_trajectory, 
                                                make_linear_system):
        # M - I has one tiny-but-retained singular value -> cond ~ 1e8, full rank.
        bad_M = np.diag([2.0, 2.0, 2.0, 2.0, 2.0, 1.0 + 1e-8])
        system = _linear_system(make_linear_system, M=bad_M)
        guess = _linear_guess(make_fake_trajectory, system, np.ones(6))
        result = DifferentialCorrector(cond_fail=1e6).solve(
            guess, 'all', constraints=[Periodicity()])
        assert not result.converged
        assert result.abort_reason is not None
        assert "cond_fail" in result.abort_reason


class TestSolveBudget:
    def test_budget_exhaustion_is_not_an_abort(self, make_fake_trajectory, 
                                               make_linear_system):
        # max_iter=0 hits the budget before any step: non-convergence, but
        # abort_reason stays None (ordinary budget exhaustion, not a failure).
        system = _linear_system(make_linear_system)
        guess = _linear_guess(make_fake_trajectory, system, np.ones(6))
        result = DifferentialCorrector(max_iter=0).solve(
            guess, 'all', constraints=[Periodicity()])
        assert not result.converged
        assert result.iterations == 0
        assert result.abort_reason is None


class TestSolveWarnings:
    """Each diagnostic warning fires from its own trigger."""

    def test_overdetermined_warns(self, make_fake_trajectory, make_linear_system):
        # 1 free var vs 6 periodicity constraints -> m > n at iteration 0.
        system = _linear_system(make_linear_system)
        guess = _linear_guess(make_fake_trajectory, system, np.ones(6))
        with pytest.warns(UserWarning, match="overdetermined"):
            DifferentialCorrector(max_iter=2).solve(guess, ['x'], 
                                                    constraints=[Periodicity()])

    def test_rank_deficient_warns(self, make_fake_trajectory, make_linear_system):
        # M - I has a zero singular value -> rank 5 < 6.
        singular_M = np.diag([2.0, 2.0, 2.0, 2.0, 2.0, 1.0])
        b = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.0])   # last comp already satisfied
        system = _linear_system(make_linear_system, M=singular_M, b=b)
        guess = _linear_guess(make_fake_trajectory, system, np.ones(6))
        with pytest.warns(UserWarning, match="rank-deficient"):
            DifferentialCorrector(max_iter=2).solve(guess, 'all', 
                                                    constraints=[Periodicity()])

    def test_cond_warn_warns_without_aborting(self, make_fake_trajectory, 
                                              make_linear_system):
        warn_M = np.diag([2.0, 2.0, 2.0, 2.0, 2.0, 1.0 + 1e-6])   # cond ~ 1e6
        system = _linear_system(make_linear_system, M=warn_M)
        guess = _linear_guess(make_fake_trajectory, system, np.ones(6))
        with pytest.warns(UserWarning, match="cond_warn"):
            DifferentialCorrector(cond_warn=1e3).solve(guess, 'all', 
                                                       constraints=[Periodicity()])


class TestSolveDiagnosticsAndIterates:
    """The optional reporting toggles."""

    def test_diagnostics_none_by_default(self, make_fake_trajectory, make_linear_system):
        system = _linear_system(make_linear_system)
        guess = _linear_guess(make_fake_trajectory, system, np.ones(6))
        result = DifferentialCorrector().solve(guess, 'all', constraints=[Periodicity()])
        assert result.diagnostics is None

    def test_diagnostics_populated(self, make_fake_trajectory, make_linear_system):
        system = _linear_system(make_linear_system)
        guess = _linear_guess(make_fake_trajectory, system, np.ones(6))
        result = DifferentialCorrector().solve(
            guess, 'all', constraints=[Periodicity()], diagnostics=True)
        d = result.diagnostics
        assert d is not None
        assert set(d) == {'residual_history', 'condition_history', 'final_rank', 
                          'abort_reason'}
        # One Newton step: residual evaluated at iterate 0 and the converged
        # iterate 1; the Jacobian (hence cond) only at iterate 0.
        assert len(d['residual_history']) == 2
        assert len(d['condition_history']) == 1
        assert d['final_rank'] == 6
        assert d['abort_reason'] is None

    def test_iterates_none_by_default(self, make_fake_trajectory, make_linear_system):
        system = _linear_system(make_linear_system)
        guess = _linear_guess(make_fake_trajectory, system, np.ones(6))
        result = DifferentialCorrector().solve(guess, 'all', constraints=[Periodicity()])
        assert result.iterates is None

    def test_iterates_populated(self, make_fake_trajectory, make_linear_system):
        system = _linear_system(make_linear_system)
        guess = _linear_guess(make_fake_trajectory, system, np.ones(6))
        result = DifferentialCorrector().solve(
            guess, 'all', constraints=[Periodicity()], iterates=True)
        assert result.iterates is not None
        assert len(result.iterates) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
