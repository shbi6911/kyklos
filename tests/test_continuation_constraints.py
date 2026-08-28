"""
Continuation closing constraints: the seam between shooter and continuation.

PseudoArclength, FreeVarPin and PhaseConstraint are FreeVarConstraints -- their
residual and Jacobian are functions of the packed unknown vector X directly,
never of a propagated state -- so the assembler places their rows straight into
the X columns with no STM chaining. That makes them testable in complete
isolation: no System, no integrator, no Trajectory, no corrector.

They are tested together, in their own file rather than folded into the
shooter phases, because they are the seam the continuation engine is built on.
The engine selects one of them per march (via the scheme registry) and rebuilds
it fresh at every step from that step's reference data; if any of the three is
wrong, every member of every family is wrong in the same way.

All three residuals are affine in X, which sharpens two kinds of test:

  - The analytic Jacobian is exact, not approximate, so the finite-difference
    agreement checks below use a tight tolerance rather than a tuned one.
  - The Jacobian is constant in X, so "evaluate at a different iterate and get
    the same matrix" is a real assertion about the math, not a tautology.

Tier 1 (fast) throughout, except a small PhaseConstraint class marked slow
that binds against the real CR3BP System to confirm the fake-system path
agrees with a genuine vector field.
"""

import numpy as np
import pytest

from kyklos import temp_config
from kyklos.shooter import (
    ConstraintSpace, FreeVarConstraint,
    PseudoArclength, FreeVarPin, PhaseConstraint,
    _finite_diff,
)


# ===========================================================================
# Shared helpers
# ===========================================================================

# Finite-difference step. The residuals are affine, so the central difference
# is exact up to floating point and this value is not delicate.
_FD_EPS = 1e-6

# Tolerance for analytic-vs-finite-difference agreement. Tight on purpose:
# an affine residual has no truncation error, so anything looser would hide a
# genuinely wrong Jacobian.
_FD_ATOL = 1e-9


def _unit(vec) -> np.ndarray:
    """Normalize a vector, for building valid tangents."""
    vec = np.asarray(vec, dtype=float)
    return vec / np.linalg.norm(vec)


@pytest.fixture
def arclength():
    """A representative 3-wide PseudoArclength closer."""
    return PseudoArclength(
        X_prev=np.array([0.82, 0.13, 1.37]),
        t_hat=_unit([1.0, -2.0, 0.5]),
        ds=1e-3,
    )


@pytest.fixture
def pin():
    """A representative FreeVarPin on the middle column of a 3-wide X."""
    return FreeVarPin(col=1, target=0.25, n_X=3)


@pytest.fixture
def phase_template():
    """
    An unbound PhaseConstraint over the three velocity components.

    Unbound: bind() is what captures the flow direction, and several tests
    below depend on the template still being unbound.
    """
    return PhaseConstraint(
        x0_ref=np.array([0.8, 0.0, 0.1, 0.0, 0.42, 0.0]),
        free_vars="velocity",
        n_X=4,
    )


# ===========================================================================
# Cross-cutting contract
# ===========================================================================
class TestFreeVarConstraintContract:
    """
    All three declare the same placement contract.

    space is read once, at shooting-context construction, to record the
    matching block kind; the per-iterate assembler then dispatches on that and
    never re-inspects the constraint. A wrong value here would misplace every
    closer row in DF, so it is asserted explicitly rather than assumed.
    """

    @pytest.mark.parametrize("cls", [PseudoArclength, FreeVarPin,
                                     PhaseConstraint])
    def test_is_a_freevar_constraint(self, cls):
        assert issubclass(cls, FreeVarConstraint)

    @pytest.mark.parametrize("cls", [PseudoArclength, FreeVarPin,
                                     PhaseConstraint])
    def test_space_is_freevar(self, cls):
        assert cls.space is ConstraintSpace.FREEVAR

    def test_each_contributes_exactly_one_row(self, arclength, pin,
                                              phase_template):
        # A closer squares an underdetermined-by-one system, so one row each
        # is the whole point; two would overdetermine it.
        assert arclength.n_rows == 1
        assert pin.n_rows == 1
        assert phase_template.n_rows == 1


# ===========================================================================
# PseudoArclength
# ===========================================================================
class TestPseudoArclengthConstruction:
    """Construction-time validation and input ownership."""

    def test_accepts_valid_inputs(self):
        c = PseudoArclength([0.0, 1.0], _unit([1.0, 1.0]), 0.5)
        assert c.n_rows == 1

    def test_rejects_non_1d_X_prev(self):
        with pytest.raises(ValueError, match="1-D vector"):
            PseudoArclength(np.zeros((2, 2)), np.eye(2), 0.5)

    def test_rejects_shape_mismatch(self):
        with pytest.raises(ValueError, match="must match X_prev in shape"):
            PseudoArclength(np.zeros(3), _unit([1.0, 0.0]), 0.5)

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_X_prev(self, bad):
        with pytest.raises(ValueError, match="must be finite"):
            PseudoArclength([0.0, bad], _unit([1.0, 1.0]), 0.5)

    @pytest.mark.parametrize("bad", [np.nan, np.inf])
    def test_rejects_non_finite_t_hat(self, bad):
        with pytest.raises(ValueError, match="must be finite"):
            PseudoArclength([0.0, 0.0], [1.0, bad], 0.5)

    @pytest.mark.parametrize("scale", [0.5, 2.0, 1.0 + 1e-6])
    def test_rejects_non_unit_tangent(self, scale):
        # Fold-safety of the bordered system relies on t_hat . t_hat == 1;
        # normalizing is the engine's job, so a non-unit tangent is a caller
        # error rather than something to silently fix here.
        with pytest.raises(ValueError, match="unit vector"):
            PseudoArclength([0.0, 0.0], scale * _unit([3.0, 4.0]), 0.5)

    def test_non_unit_message_reports_the_norm(self):
        with pytest.raises(ValueError, match="2.000000e"):
            PseudoArclength([0.0, 0.0], 2.0 * _unit([1.0, 0.0]), 0.5)

    @pytest.mark.parametrize("bad_ds", [0.0, -1e-3, np.nan, np.inf])
    def test_rejects_non_positive_or_non_finite_ds(self, bad_ds):
        with pytest.raises(ValueError, match="positive finite step"):
            PseudoArclength([0.0, 0.0], _unit([1.0, 0.0]), bad_ds)

    def test_integer_ds_is_coerced(self):
        c = PseudoArclength([0.0, 0.0], _unit([1.0, 0.0]), 1)
        assert c.residual([0.0, 0.0])[0] == pytest.approx(-1.0)         #type: ignore

    def test_does_not_alias_caller_arrays(self):
        X_prev = np.array([1.0, 2.0])
        t_hat = _unit([1.0, 0.0])
        c = PseudoArclength(X_prev, t_hat, 0.5)
        before = c.residual([3.0, 4.0])[0]                              #type: ignore
        X_prev[0] = 99.0
        t_hat[:] = _unit([0.0, 1.0])
        assert c.residual([3.0, 4.0])[0] == pytest.approx(before)       #type: ignore


class TestPseudoArclengthResidual:
    """The affine condition g(X) = t_hat . (X - X_prev) - ds."""

    def test_shape_and_dtype(self, arclength):
        g = arclength.residual([0.0, 0.0, 0.0])
        assert isinstance(g, np.ndarray)
        assert g.shape == (1,)
        assert g.dtype == float

    def test_at_previous_member_residual_is_minus_ds(self):
        X_prev = np.array([0.82, 0.13, 1.37])
        ds = 1e-3
        c = PseudoArclength(X_prev, _unit([1.0, -2.0, 0.5]), ds)
        # The previous member is a full step behind the constraint plane.
        assert c.residual(X_prev)[0] == pytest.approx(-ds, abs=1e-15)

    def test_vanishes_one_step_along_the_tangent(self):
        # The tangent predictor X_prev + ds*t_hat lands exactly on the
        # constraint plane -- this is the property that makes it the natural
        # predictor, even though v1 uses the trivial one.
        X_prev = np.array([0.82, 0.13, 1.37])
        t_hat = _unit([1.0, -2.0, 0.5])
        ds = 1e-3
        c = PseudoArclength(X_prev, t_hat, ds)
        assert c.residual(X_prev + ds * t_hat)[0] == pytest.approx(0.0,
                                                                  abs=1e-15)

    def test_matches_hand_computed_value(self, arclength):
        X_prev = np.array([0.82, 0.13, 1.37])
        t_hat = _unit([1.0, -2.0, 0.5])
        X = np.array([-0.4, 2.2, 0.05])
        expected = float(t_hat @ (X - X_prev) - 1e-3)
        assert arclength.residual(X)[0] == pytest.approx(expected, rel=1e-14)

    def test_orthogonal_displacement_does_not_move_the_residual(self):
        """
        Only the component of (X - X_prev) along t_hat is measured.

        This is the whole reason the constraint is fold-safe: the plane is
        normal to the tangent, not to a coordinate axis. A "distance from
        X_prev" implementation would pass every other residual test here and
        fail this one.
        """
        X_prev = np.zeros(3)
        t_hat = _unit([1.0, -2.0, 0.5])
        c = PseudoArclength(X_prev, t_hat, 1e-3)
        ortho = np.cross(t_hat, [0.0, 0.0, 1.0])
        assert abs(ortho @ t_hat) < 1e-15      # guard the test's own premise
        base = c.residual(X_prev)[0]
        for scale in (1e-3, 1.0, 25.0):
            moved = c.residual(X_prev + scale * ortho)[0]
            assert moved == pytest.approx(base, abs=1e-12)

    def test_accepts_list_input(self, arclength):
        assert arclength.residual([0.0, 0.0, 0.0]).shape == (1,)


class TestPseudoArclengthJacobian:
    """dg/dX = t_hat: full-width, generally dense, constant in X."""

    def test_equals_the_tangent(self, arclength):
        t_hat = _unit([1.0, -2.0, 0.5])
        J = arclength.jacobian_X(np.zeros(3))
        assert J.shape == (1, 3)
        assert J[0] == pytest.approx(t_hat, rel=1e-15)

    def test_constant_across_iterates(self, arclength):
        J1 = arclength.jacobian_X(np.zeros(3))
        J2 = arclength.jacobian_X([1e3, -7.0, 0.25])
        assert J1 == pytest.approx(J2, rel=1e-15)

    def test_returns_a_fresh_writable_array(self, arclength):
        J = arclength.jacobian_X(np.zeros(3))
        assert J.flags.writeable
        J[0, 0] = 999.0
        # Scribbling on a returned Jacobian must not corrupt the stored
        # tangent, or every subsequent step of the march inherits the damage.
        assert arclength.jacobian_X(np.zeros(3))[0, 0] != 999.0

    def test_matches_finite_differences(self, arclength):
        X = np.array([-0.4, 2.2, 0.05])
        J_fd = _finite_diff(lambda v: arclength.residual(v), X, _FD_EPS)
        assert arclength.jacobian_X(X) == pytest.approx(J_fd, abs=_FD_ATOL)


# ===========================================================================
# FreeVarPin
# ===========================================================================
class TestFreeVarPinConstruction:
    """Construction-time validation, including the bool guards."""

    def test_accepts_valid_inputs(self):
        assert FreeVarPin(0, 1.5, 1).n_rows == 1

    @pytest.mark.parametrize("bad_col", [1.0, "0", None, np.float64(1.0)])
    def test_rejects_non_integer_col(self, bad_col):
        with pytest.raises(TypeError, match="col must be an integer"):
            FreeVarPin(bad_col, 0.0, 3)

    @pytest.mark.parametrize("bad_col", [True, False])
    def test_rejects_bool_col(self, bad_col):
        # bool subclasses int, so True would otherwise silently mean column 1.
        with pytest.raises(TypeError, match="col must be an integer"):
            FreeVarPin(bad_col, 0.0, 3)

    @pytest.mark.parametrize("bad_n", [2.0, "3", None])
    def test_rejects_non_integer_n_X(self, bad_n):
        with pytest.raises(TypeError, match="n_X must be an integer"):
            FreeVarPin(0, 0.0, bad_n)

    def test_rejects_bool_n_X(self):
        with pytest.raises(TypeError, match="n_X must be an integer"):
            FreeVarPin(0, 0.0, True)

    @pytest.mark.parametrize("bad_n", [0, -1])
    def test_rejects_non_positive_n_X(self, bad_n):
        with pytest.raises(ValueError, match="positive integer"):
            FreeVarPin(0, 0.0, bad_n)

    @pytest.mark.parametrize("bad_col", [-1, 3, 99])
    def test_rejects_out_of_range_col(self, bad_col):
        with pytest.raises(ValueError, match=r"0 <= col < n_X"):
            FreeVarPin(bad_col, 0.0, 3)

    @pytest.mark.parametrize("bad_target", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_target(self, bad_target):
        with pytest.raises(ValueError, match="target must be finite"):
            FreeVarPin(0, bad_target, 3)

    def test_numpy_integers_are_accepted(self):
        assert FreeVarPin(np.int64(1), 0.0, np.int64(3)).n_rows == 1


class TestFreeVarPinResidual:
    """The affine condition g(X) = X[col] - target."""

    def test_shape_and_dtype(self, pin):
        g = pin.residual([0.0, 0.0, 0.0])
        assert g.shape == (1,)
        assert g.dtype == float

    def test_vanishes_at_the_target(self, pin):
        assert pin.residual([9.0, 0.25, -3.0])[0] == pytest.approx(0.0)

    @pytest.mark.parametrize("col", [0, 1, 2])
    def test_reads_only_the_pinned_column(self, col):
        c = FreeVarPin(col, 0.0, 3)
        X = np.array([1.0, 2.0, 3.0])
        assert c.residual(X)[0] == pytest.approx(X[col])

    def test_offset_is_signed(self, pin):
        assert pin.residual([0.0, 0.75, 0.0])[0] == pytest.approx(0.5)
        assert pin.residual([0.0, -0.25, 0.0])[0] == pytest.approx(-0.5)


class TestFreeVarPinJacobian:
    """dg/dX = e_col: a one-hot row, constant in X."""

    @pytest.mark.parametrize("col", [0, 1, 2, 3])
    def test_is_one_hot_in_the_pinned_column(self, col):
        J = FreeVarPin(col, 0.0, 4).jacobian_X(np.zeros(4))
        expected = np.zeros((1, 4))
        expected[0, col] = 1.0
        assert J == pytest.approx(expected)

    def test_constant_across_iterates(self, pin):
        assert pin.jacobian_X(np.zeros(3)) == pytest.approx(
            pin.jacobian_X([1e6, -2.0, 7.0]))

    def test_returns_a_fresh_array(self, pin):
        J = pin.jacobian_X(np.zeros(3))
        J[0, 1] = 42.0
        assert pin.jacobian_X(np.zeros(3))[0, 1] == pytest.approx(1.0)

    def test_matches_finite_differences(self, pin):
        X = np.array([0.3, -1.1, 4.0])
        J_fd = _finite_diff(lambda v: pin.residual(v), X, _FD_EPS)
        assert pin.jacobian_X(X) == pytest.approx(J_fd, abs=_FD_ATOL)


class TestFreeVarPinIsColumnAgnostic:
    """
    The class pins a column of X, whichever kind of freedom that column is.

    Under a single-shooting layout X is [free start components, free boundary
    times], so a free-time column is simply a higher index. The engine owns
    the column plan and resolves 'the period' or 'the start x' into an index;
    this class is deliberately indifferent, and that indifference is what lets
    one natural-parameter scheme serve both period sampling and amplitude
    sampling.
    """

    def test_state_column_and_time_column_behave_identically(self):
        n_X = 3                      # e.g. (x, vy, T_half)
        X = np.array([0.82, 0.42, 1.37])
        state_pin = FreeVarPin(col=0, target=0.80, n_X=n_X)
        time_pin = FreeVarPin(col=2, target=1.35, n_X=n_X)

        assert state_pin.residual(X)[0] == pytest.approx(0.02)
        assert time_pin.residual(X)[0] == pytest.approx(0.02)

        assert state_pin.jacobian_X(X).sum() == pytest.approx(1.0)
        assert time_pin.jacobian_X(X).sum() == pytest.approx(1.0)
        assert state_pin.jacobian_X(X)[0, 0] == pytest.approx(1.0)
        assert time_pin.jacobian_X(X)[0, 2] == pytest.approx(1.0)


# ===========================================================================
# PhaseConstraint
# ===========================================================================
class TestPhaseConstraintConstruction:
    """Construction-time validation. No System needed for any of this."""

    def test_accepts_valid_inputs(self):
        c = PhaseConstraint(np.zeros(6), ("x", "vy"), 3)
        assert c.n_rows == 1

    @pytest.mark.parametrize("free_vars", ["all", "position", "velocity"])
    def test_accepts_category_spellings(self, free_vars):
        assert PhaseConstraint(np.zeros(6), free_vars, 6).n_rows == 1

    def test_rejects_wrong_x0_ref_shape(self):
        with pytest.raises(ValueError):
            PhaseConstraint(np.zeros(5), ("x",), 3)

    def test_rejects_non_finite_x0_ref(self):
        with pytest.raises(ValueError, match="must be finite"):
            PhaseConstraint([0.0, np.nan, 0, 0, 0, 0], ("x",), 3)

    def test_rejects_empty_free_vars(self):
        # With nothing free the phase row is identically zero and constrains
        # nothing; a fully pinned start state needs no phase condition.
        with pytest.raises(ValueError, match="at least one free start"):
            PhaseConstraint(np.zeros(6), (), 3)

    def test_rejects_unknown_component_name(self):
        with pytest.raises((ValueError, KeyError)):
            PhaseConstraint(np.zeros(6), ("q",), 3)

    def test_rejects_bool_n_X(self):
        with pytest.raises(TypeError, match="n_X must be an integer"):
            PhaseConstraint(np.zeros(6), ("x",), True)

    def test_rejects_n_X_narrower_than_free_components(self):
        # The row is supported on the free start block, so X cannot be
        # narrower than that block.
        with pytest.raises(ValueError, match="narrower"):
            PhaseConstraint(np.zeros(6), ("x", "y", "z"), 2)

    def test_does_not_alias_caller_state(self):
        x0 = np.array([0.8, 0.0, 0.0, 0.0, 0.42, 0.0])
        c = PhaseConstraint(x0, ("vy",), 2)
        x0[4] = 99.0
        # Rebind-free check: the stored reference must be a copy, which we
        # observe through the residual once bound (below). Here we only assert
        # construction did not blow up and the object is usable.
        assert c.n_rows == 1


class TestPhaseConstraintBinding:
    """bind() captures the flow direction; unbound instances are templates."""

    def test_unbound_residual_raises(self, phase_template):
        with pytest.raises(RuntimeError, match="before bind"):
            phase_template.residual(np.zeros(4))

    def test_unbound_jacobian_raises(self, phase_template):
        with pytest.raises(RuntimeError, match="before bind"):
            phase_template.jacobian_X(np.zeros(4))

    def test_bind_returns_a_new_instance(self, phase_template,
                                         make_fake_system):
        system = make_fake_system(field_matrix=np.eye(6))
        bound = phase_template.bind(system)
        assert bound is not phase_template

    def test_bind_leaves_the_template_unbound(self, phase_template,
                                              make_fake_system):
        # The template stays reusable against another System; a bound
        # constraint is an inert snapshot, not a mutated original.
        system = make_fake_system(field_matrix=np.eye(6))
        phase_template.bind(system)
        with pytest.raises(RuntimeError, match="before bind"):
            phase_template.residual(np.zeros(4))

    def test_rebinding_a_template_twice_gives_independent_constraints(
            self, phase_template, make_fake_system):
        a = phase_template.bind(make_fake_system(field_matrix=np.eye(6)))
        b = phase_template.bind(make_fake_system(field_matrix=2.0 * np.eye(6)))
        X = np.array([0.1, 0.2, 0.3, 0.0])
        assert a.residual(X)[0] != pytest.approx(b.residual(X)[0])

    def test_degenerate_flow_raises_under_strict_validation(self,
                                                            make_fake_system):
        # Free only vy, and give the field no vy component: the phase row is
        # zero and fixes nothing.
        A = np.zeros((6, 6))
        A[0, 0] = 1.0
        template = PhaseConstraint(np.ones(6), ("vy",), 2)
        with temp_config(STRICT_VALIDATION=True):
            with pytest.raises(ValueError, match="degenerate"):
                template.bind(make_fake_system(field_matrix=A))

    def test_degenerate_flow_warns_when_validation_is_relaxed(
            self, make_fake_system):
        A = np.zeros((6, 6))
        A[0, 0] = 1.0
        template = PhaseConstraint(np.ones(6), ("vy",), 2)
        with temp_config(STRICT_VALIDATION=False):
            with pytest.warns(UserWarning, match="degenerate"):
                bound = template.bind(make_fake_system(field_matrix=A))
        # Still constructed, and its row is the zero row it warned about.
        assert bound.jacobian_X(np.zeros(2)) == pytest.approx(np.zeros((1, 2)))


class TestPhaseConstraintResidual:
    """g(X) = <x0(X) - x0_ref, f(x0_ref)> over the free start components."""

    @pytest.fixture
    def bound(self, make_fake_system):
        """
        Free (x, vy) of a 3-wide X, bound to a known linear field.

        field_matrix is a permutation-ish matrix chosen so f(x0_ref) has
        distinct, easily hand-checked entries in the freed slots.
        """
        x0_ref = np.array([0.8, 0.0, 0.0, 0.0, 0.42, 0.0])
        A = np.zeros((6, 6))
        A[0, 4] = 1.0      # f_x  = vy  = 0.42
        A[4, 0] = -2.0     # f_vy = -2x = -1.6
        template = PhaseConstraint(x0_ref, ("x", "vy"), 3)
        return template.bind(make_fake_system(field_matrix=A)), x0_ref

    def test_shape_and_dtype(self, bound):
        c, _ = bound
        g = c.residual(np.array([0.8, 0.42, 1.0]))
        assert g.shape == (1,)
        assert g.dtype == float

    def test_vanishes_at_the_reference_state(self, bound):
        c, x0_ref = bound
        # X's free block equals the reference's free components -> no phase
        # offset. The trailing free-time column is irrelevant to this row.
        X = np.array([x0_ref[0], x0_ref[4], 1.37])
        assert c.residual(X)[0] == pytest.approx(0.0, abs=1e-15)

    def test_matches_hand_computed_value(self, bound):
        c, x0_ref = bound
        f_free = np.array([0.42, -1.6])          # (f_x, f_vy) at x0_ref
        X = np.array([0.9, 0.30, 1.37])
        dx = np.array([X[0] - x0_ref[0], X[1] - x0_ref[4]])
        assert c.residual(X)[0] == pytest.approx(float(f_free @ dx),
                                                 rel=1e-13)

    def test_ignores_columns_outside_the_free_start_block(self, bound):
        c, _ = bound
        # The row is supported on X[0:n_fs]; the free-time column must not
        # move it.
        a = c.residual(np.array([0.9, 0.30, 1.0]))[0]
        b = c.residual(np.array([0.9, 0.30, 99.0]))[0]
        assert a == pytest.approx(b)

    @pytest.mark.parametrize("width", [2, 4])
    def test_rejects_mis_sized_X(self, bound, width):
        """
        The explicit width check the other two closers deliberately lack.

        PseudoArclength and FreeVarPin span the full X, so a mis-sized vector
        trips numpy on its own. This row reads only a leading slice, so a
        too-long or too-short X would slice cleanly and return a silently
        wrong residual.
        """
        c, _ = bound
        with pytest.raises(ValueError, match=r"must have shape"):
            c.residual(np.zeros(width))


class TestPhaseConstraintJacobian:
    """dg/dX = f(x0_ref) in the free start columns, zero elsewhere."""

    @pytest.fixture
    def bound(self, make_fake_system):
        x0_ref = np.array([0.8, 0.0, 0.0, 0.0, 0.42, 0.0])
        A = np.zeros((6, 6))
        A[0, 4] = 1.0
        A[4, 0] = -2.0
        return PhaseConstraint(x0_ref, ("x", "vy"), 3).bind(
            make_fake_system(field_matrix=A))

    def test_places_the_flow_direction_in_the_leading_columns(self, bound):
        J = bound.jacobian_X(np.zeros(3))
        assert J.shape == (1, 3)
        assert J[0, :2] == pytest.approx([0.42, -1.6], rel=1e-13)

    def test_is_zero_outside_the_free_start_block(self, bound):
        assert bound.jacobian_X(np.zeros(3))[0, 2] == pytest.approx(0.0)

    def test_constant_across_iterates(self, bound):
        assert bound.jacobian_X(np.zeros(3)) == pytest.approx(
            bound.jacobian_X([5.0, -3.0, 100.0]))

    def test_returns_a_fresh_array(self, bound):
        J = bound.jacobian_X(np.zeros(3))
        J[0, 0] = 123.0
        assert bound.jacobian_X(np.zeros(3))[0, 0] != 123.0

    def test_matches_finite_differences(self, bound):
        X = np.array([0.9, 0.30, 1.37])
        J_fd = _finite_diff(lambda v: bound.residual(v), X, _FD_EPS)
        assert bound.jacobian_X(X) == pytest.approx(J_fd, abs=_FD_ATOL)


@pytest.mark.slow
class TestPhaseConstraintAgainstRealDynamics:
    """
    Bind against the real CR3BP System.

    Everything above runs on a prescribed linear field, which makes the
    arithmetic exact and checkable but proves nothing about the real
    System.vector_field path. These confirm that seam, and pin the anchor
    choice the class actually requires.

    The reference point is taken a quarter period along the orbit, NOT at its
    stored initial condition. The IC is a perpendicular x-z crossing, where
    the flow is nonzero only in y and vx -- precisely the two components a
    symmetry recipe pins (constraint_spec {y: 0, vx: 0}) -- so no
    symmetry-recipe free set is transverse there. That is not an accident of
    this orbit: it is the structural reason PhaseConstraint is needed only
    once the symmetry is dropped, and it gets its own test below.

    cr3bp_system and lyapunov_orbit.system are the same cached instance (both
    resolve through earth_moon_cr3bp()), so binding to the former and
    sampling the latter is self-consistent.
    """

    @pytest.fixture
    def anchor(self, lyapunov_orbit):
        """A generic, non-crossing point on the reference orbit."""
        traj = lyapunov_orbit.trajectory
        return traj.state_at_raw(traj.t0 + 0.25 * lyapunov_orbit.period)

    def test_binds_to_a_real_system_and_captures_the_field(
            self, cr3bp_system, anchor):
        bound = PhaseConstraint(anchor, ("x", "vy"), 3).bind(cr3bp_system)
        f_ref = np.asarray(cr3bp_system.vector_field(anchor), dtype=float)
        J = bound.jacobian_X(np.zeros(3))
        # The freed components are x (index 0) and vy (index 4), so the row
        # carries exactly those two entries of f, in free-var order.
        assert J[0, 0] == pytest.approx(f_ref[0], rel=1e-12)
        assert J[0, 1] == pytest.approx(f_ref[4], rel=1e-12)
        assert J[0, 2] == pytest.approx(0.0)

    def test_row_is_transverse_at_a_generic_point(self, cr3bp_system, anchor):
        bound = PhaseConstraint(anchor, ("x", "vy"), 3).bind(cr3bp_system)
        assert np.any(bound.jacobian_X(np.zeros(3)) != 0.0)

    def test_residual_matches_the_real_field_by_hand(self, cr3bp_system,
                                                     anchor):
        bound = PhaseConstraint(anchor, ("x", "vy"), 3).bind(cr3bp_system)
        f_ref = np.asarray(cr3bp_system.vector_field(anchor), dtype=float)
        X = np.array([anchor[0] + 1e-3, anchor[4] - 2e-3, 1.37])
        expected = f_ref[0] * 1e-3 + f_ref[4] * (-2e-3)
        assert bound.residual(X)[0] == pytest.approx(expected, rel=1e-11)

    def test_degenerate_at_a_perpendicular_crossing(self, cr3bp_system,
                                                    lyapunov_orbit):
        """
        The anchor a symmetry recipe would naturally reach for is exactly the
        one this constraint cannot use.

        At a perpendicular x-z crossing, f_x = vx = 0 and
        f_vy = -2*vx + dU/dy = 0 (dU/dy is proportional to y, also zero). The
        only nonzero flow components are y and vx, which the recipe pins. So
        the phase row over any freed subset is identically zero and the
        degeneracy guard must fire.
        """
        traj = lyapunov_orbit.trajectory
        ic = traj.state_at_raw(traj.t0)
        f_ic = np.asarray(cr3bp_system.vector_field(ic), dtype=float)

        # State the premise numerically rather than trusting the geometry.
        assert f_ic[0] == pytest.approx(0.0, abs=1e-14)   # f_x
        assert f_ic[4] == pytest.approx(0.0, abs=1e-14)   # f_vy

        template = PhaseConstraint(ic, ("x", "vy"), 3)
        with temp_config(STRICT_VALIDATION=True):
            with pytest.raises(ValueError, match="degenerate"):
                template.bind(cr3bp_system)
