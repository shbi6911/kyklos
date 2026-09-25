"""
Test suite for the continuation payload: ShooterResult.ContinuationState.

Verifies the DH that solve(continuation=True) hands back -- the corrector
Jacobian at the converged member with any closer (X_SPACE) rows stripped,
whose null space is the family tangent. Task 4's tangent function consumes
this matrix directly, so everything downstream of it inherits whatever error
is in it.

Why this is tested now, before a continuation loop exists
--------------------------------------------------------
A wrong DH does not raise. It produces a tangent off by a few degrees, the
predictor lands slightly off the family, the corrector obligingly pulls it
back to *a* solution, and the family drifts -- with a converged residual at
every step. That is the same silent-failure mode as a branch jump: the
solutions satisfy the recipe honestly and nothing detects the problem. Once
DH is inside a march the only observable is "does the family look right,"
which is precisely the signal that cannot distinguish these cases.

In isolation, though, an independent oracle is available: DH is a Jacobian,
so it can be finite-differenced. That oracle exists only while DH can be
requested at a single converged member, which is why these tests belong here
rather than after the engine is built.

The oracle
----------
H(X) is the terminal-only residual map -- unpack X into initial conditions
and boundary times, propagate, assemble F, delete the closer rows. This
mirrors the corrector's own inner loop (_unpack -> propagate -> _assemble_F)
but reaches the Jacobian by central differences instead of by chaining STMs,
so agreement is genuine cross-validation of two independent paths, not a
tautology.

Three levels of check, in increasing strength:

1. DH agrees with the central-difference Jacobian at a fixed step.
2. The disagreement falls as h^2 under step refinement. This is the real
   test: a DH wrong by any fixed amount would show an error that *plateaus*
   as h shrinks, while a correct DH shows error dominated by FD truncation,
   which is second order. Measured ratio is ~100x per decade until roundoff
   takes over near h = 1e-8.
3. The null vector of DH agrees in direction with a secant between two
   independently converged nearby members. This is the only check that
   confirms the null space is *the family tangent* and not merely some null
   space; measured agreement is 0.016 degrees, against a tolerance of 0.5
   set by the secant's own first-order truncation.

 The payload and Jacobian tests propagate and are marked slow class
by class. The tangent-function tests at the bottom (_check_direction
and _family_tangent) are pure linear algebra on synthetic matrices
and run in the fast tier; one slow class ties the function back to
a real DH.
"""

import numpy as np
import pytest

from kyklos.shooter import (
    DifferentialCorrector,
    TargetState,
    PseudoArclength,
    _ShootingContext,
    _assemble_F,
    _pack,
    _unpack,
    _BlockKind,
)
from kyklos.continuation import _check_direction, _family_tangent


# The recipe geometry under test: the planar Lyapunov perpendicular-crossing
# formulation, in both its determinacies. Square (free_times empty) is the
# isolated/bootstrap solve; corank 1 (end time freed) is the continuation
# solve, closed by an arclength row.
_FREE_VARS = ("x", "vy")
_FREE_TIMES = (1,)
_TARGETS = {"y": 0.0, "vx": 0.0}

# Central-difference step. Chosen from a measured sweep: the FD/analytic
# disagreement is 3.7e-8 relative here and falls cleanly as h^2 down to
# h = 1e-7, bottoming out at 1e-8 where roundoff takes over. 1e-6 sits a
# decade clear of that floor.
_FD_H = 1e-6
_FD_RTOL = 1e-6

# Natural-parameter direction and step for the closing arclength row: pure x,
# so the closer pins the amplitude and the solve is a one-step march.
_T_HAT = np.array([1.0, 0.0, 0.0])
_DS = 1e-4


# ===========================================================================
# Oracle
# ===========================================================================

def _closer_rows(ctx: _ShootingContext) -> list[int]:
    """
    Row indices of the X_SPACE (closer) blocks in the full constraint vector.

    Computed from the row plan the same way the producer does, because there
    is no other way to know which rows are closers -- but note this is only
    used to build the *oracle's* residual map. It is never used to check DH
    against itself: DH's own strip is verified by the FD agreement below,
    which would fail if the wrong rows had been removed.
    """
    return [
        r
        for block in ctx.row_plan.blocks
        if block.kind is _BlockKind.X_SPACE
        for r in range(block.row_offset, block.row_offset + block.row_count)
    ]


def _H(X: np.ndarray, ctx: _ShootingContext) -> np.ndarray:
    """
    The terminal-only residual map, evaluated by propagation.

    Mirrors the corrector's inner loop -- scatter X into initial conditions
    and boundary times, propagate, assemble F -- then deletes the closer
    rows so what remains is the unclosed map whose Jacobian is DH.
    """
    ics, times = _unpack(X, ctx)
    traj = ctx.system.propagate(ics, times, with_stm=True)
    return np.delete(_assemble_F(traj, ctx, X), _closer_rows(ctx))


def _fd_jacobian(X: np.ndarray, ctx: _ShootingContext,
                 h: float = _FD_H) -> np.ndarray:
    """Central-difference Jacobian of _H at X, shape (n_terminal, n_X)."""
    n_rows = len(_H(X, ctx))
    J = np.zeros((n_rows, ctx.n_X))
    for j in range(ctx.n_X):
        step = np.zeros(ctx.n_X)
        step[j] = h
        J[:, j] = (_H(X + step, ctx) - _H(X - step, ctx)) / (2.0 * h)
    return J


def _max_rel_error(A: np.ndarray, B: np.ndarray) -> float:
    """Largest entrywise relative difference, guarded against tiny B."""
    return float(np.max(np.abs(A - B) / np.maximum(np.abs(B), 1.0)))


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture(scope="module")
def half_arc_guess(cr3bp_system, lyapunov_orbit):
    """
    A half-period arc from the reference Lyapunov, deliberately perturbed.

    Symmetry pinning integrates to the next perpendicular crossing, so the
    guess spans a half period. The initial condition is nudged off the orbit
    so the solve takes real Newton steps -- an unperturbed reference orbit
    converges at iteration 0, which would leave DH assembled at the guess
    rather than at a genuinely converged member, and the distinction is
    exactly what these tests are for.
    """
    ic = np.asarray(lyapunov_orbit.initial_state, dtype=float).copy()
    ic[0] += 1.0e-4
    ic[4] -= 2.0e-4
    tf = 0.5 * lyapunov_orbit.period * 1.001
    return cr3bp_system.propagate(ic, [0.0, tf], with_stm=True)


@pytest.fixture(scope="module")
def clean_guess(cr3bp_system, lyapunov_orbit):
    """An unperturbed half-period arc, used as the march's member zero."""
    ic = np.asarray(lyapunov_orbit.initial_state, dtype=float)
    return cr3bp_system.propagate(
        ic, [0.0, 0.5 * lyapunov_orbit.period], with_stm=True
    )


@pytest.fixture(scope="module")
def square_solve(half_arc_guess):
    """
    Bootstrap case: square, no closer, so DH is the whole Jacobian.

    Returns (result, ctx) -- the context is what the oracle needs to
    reconstruct the residual map.
    """
    constraints = (TargetState(dict(_TARGETS)),)
    ctx = _ShootingContext.from_guess(
        half_arc_guess, _FREE_VARS, constraints, (), None
    )
    result = DifferentialCorrector().solve(
        half_arc_guess, free_vars=_FREE_VARS, constraints=constraints,
        continuation=True,
    )
    return result, ctx


@pytest.fixture(scope="module")
def closed_solve(half_arc_guess):
    """
    Continuation case: corank 1 opened by freeing the end time, closed by an
    arclength row. DH must come back with that row stripped.
    """
    base = (TargetState(dict(_TARGETS)),)
    ctx_open = _ShootingContext.from_guess(
        half_arc_guess, _FREE_VARS, base, _FREE_TIMES, None
    )
    X_prev = _pack(half_arc_guess, ctx_open)
    constraints = base + (PseudoArclength(X_prev, _T_HAT, _DS),)
    ctx = _ShootingContext.from_guess(
        half_arc_guess, _FREE_VARS, constraints, _FREE_TIMES, None
    )
    result = DifferentialCorrector().solve(
        half_arc_guess, free_vars=_FREE_VARS, free_times=_FREE_TIMES,
        constraints=constraints, continuation=True,
    )
    return result, ctx, X_prev


# ===========================================================================
# Payload provenance
# ===========================================================================

@pytest.mark.slow
class TestContinuationPayload:
    """When the payload appears, and what it carries."""

    def test_absent_unless_requested(self, half_arc_guess):
        result = DifferentialCorrector().solve(
            half_arc_guess, free_vars=_FREE_VARS,
            constraints=(TargetState(dict(_TARGETS)),),
        )
        assert result.converged
        assert result.continuation is None

    def test_absent_when_the_solve_does_not_converge(self, half_arc_guess):
        """
        Populated only on convergence, so X and DH always belong to a
        converged member. A budget of zero steps stops before the first
        Newton update, leaving a residual above tolerance.
        """
        result = DifferentialCorrector(max_iter=0).solve(
            half_arc_guess, free_vars=_FREE_VARS,
            constraints=(TargetState(dict(_TARGETS)),),
            continuation=True,
        )
        assert not result.converged
        assert result.continuation is None

    def test_present_on_a_converged_request(self, square_solve):
        result, _ = square_solve
        assert result.converged
        assert result.continuation is not None

    def test_arrays_are_read_only(self, square_solve):
        result, _ = square_solve
        assert not result.continuation.X.flags.writeable
        assert not result.continuation.DH.flags.writeable

    def test_arrays_reject_assignment(self, square_solve):
        result, _ = square_solve
        with pytest.raises(ValueError, match="read-only"):
            result.continuation.X[0] = 0.0
        with pytest.raises(ValueError, match="read-only"):
            result.continuation.DH[0, 0] = 0.0

    def test_X_is_the_vector_that_produced_the_trajectory(self, square_solve):
        """
        The docstring's provenance claim: X is not merely the last iterate,
        it is the vector the returned trajectory was propagated from. Packing
        the trajectory back down must reproduce it.
        """
        result, ctx = square_solve
        assert _pack(result.trajectory, ctx) == pytest.approx(
            result.continuation.X, abs=1e-12
        )

    def test_square_DH_keeps_every_row(self, square_solve):
        """No closer, nothing to strip: DH is (n_rows, n_X) and square."""
        result, ctx = square_solve
        assert result.continuation.DH.shape == (2, ctx.n_X)
        assert ctx.n_X == 2

    def test_closed_DH_is_corank_one(self, closed_solve):
        """
        Three free variables (x, vy, end time) against two terminal rows
        after the arclength row is stripped: the (n_X - 1, n_X) shape whose
        null space is one-dimensional.
        """
        result, ctx, _ = closed_solve
        assert ctx.n_X == 3
        assert result.continuation.DH.shape == (ctx.n_X - 1, ctx.n_X)

    def test_the_closer_actually_bound(self, closed_solve):
        """
        Guards the test itself: if the arclength row were somehow inert, the
        stripped-row tests below would pass vacuously. The converged member
        must sit ds along t_hat from the reference.
        """
        result, _, X_prev = closed_solve
        assert result.converged
        displacement = _T_HAT @ (np.asarray(result.continuation.X) - X_prev)
        assert displacement == pytest.approx(_DS, rel=1e-6)


# ===========================================================================
# The Jacobian itself
# ===========================================================================

@pytest.mark.slow
class TestUnclosedJacobian:
    """DH against an independent finite-difference oracle."""

    def test_square_case_matches_finite_differences(self, square_solve):
        result, ctx = square_solve
        fd = _fd_jacobian(np.asarray(result.continuation.X), ctx)

        assert _max_rel_error(fd, result.continuation.DH) < _FD_RTOL

    def test_closed_case_matches_the_terminal_only_map(self, closed_solve):
        """
        The strip is what is really under test. The oracle differences the
        terminal rows only, so if DH still carried the arclength row -- or
        had removed a terminal row instead -- the shapes or the values would
        disagree.
        """
        result, ctx, _ = closed_solve
        fd = _fd_jacobian(np.asarray(result.continuation.X), ctx)

        assert fd.shape == result.continuation.DH.shape
        assert _max_rel_error(fd, result.continuation.DH) < _FD_RTOL

    def test_disagreement_is_second_order_in_the_step(self, closed_solve):
        """
        The strongest statement available: refine h by ten and the
        discrepancy must fall by about a hundred.

        A DH that is wrong -- by a scale factor, a transposed entry, a
        stripped-off row -- leaves an error floor that does NOT shrink with
        h, because the disagreement is real rather than truncation. Observing
        clean h^2 decay says the only thing separating the two Jacobians is
        the difference formula. Measured ratio is ~100; the threshold is set
        at 50 to leave room for conditioning without admitting a plateau.
        """
        result, ctx, _ = closed_solve
        X = np.asarray(result.continuation.X)
        DH = result.continuation.DH

        coarse = _max_rel_error(_fd_jacobian(X, ctx, 1e-5), DH)
        fine = _max_rel_error(_fd_jacobian(X, ctx, 1e-6), DH)

        assert fine < coarse
        assert coarse / fine > 50.0

    def test_the_arclength_row_is_not_in_DH(self, closed_solve):
        """
        Direct check on the strip: the closer's Jacobian row is t_hat, which
        is a unit vector along x. If it had survived, some row of DH would
        equal it.
        """
        result, _, _ = closed_solve
        for row in result.continuation.DH:
            assert not np.allclose(row, _T_HAT, atol=1e-8)


# ===========================================================================
# The tangent DH is supposed to carry
# ===========================================================================

@pytest.mark.slow
class TestFamilyTangent:
    """DH's null space, and whether it points along the family."""

    def test_null_space_is_one_dimensional(self, closed_solve):
        """
        Corank 1 by construction, so exactly one small singular value -- and
        it must be genuinely small relative to the others, not a rank
        deficiency in disguise. Measured separation is two orders of
        magnitude between the second and third singular values.
        """
        result, _, _ = closed_solve
        svals = np.linalg.svd(result.continuation.DH, compute_uv=False)

        assert len(svals) == 2
        assert svals[-1] > 1e-8 * svals[0]

    def test_null_vector_annihilates_DH(self, closed_solve):
        result, _, _ = closed_solve
        DH = result.continuation.DH
        null = np.linalg.svd(DH)[2][-1]

        assert np.linalg.norm(DH @ null) < 1e-10 * np.linalg.norm(DH)

    def test_null_vector_points_along_the_family(
        self, clean_guess, closed_solve
    ):
        """
        The claim that makes DH useful, and the one nothing else here tests:
        its null space is *the family tangent*, not merely a null space.

        Two members are converged independently at ds and 2*ds along the same
        natural-parameter direction, and their difference is a secant
        approximation to the tangent. Agreement to a fraction of a degree
        cannot happen by accident -- a DH built from the wrong Jacobian would
        put the null direction degrees away or more. The 0.5 degree tolerance
        is set by the secant's own first-order truncation over a finite step,
        not by the analytic tangent; measured agreement is 0.016 degrees.
        """
        base = (TargetState(dict(_TARGETS)),)
        ctx_open = _ShootingContext.from_guess(
            clean_guess, _FREE_VARS, base, _FREE_TIMES, None
        )
        X0 = _pack(clean_guess, ctx_open)
        corrector = DifferentialCorrector()

        def member(ds):
            constraints = base + (PseudoArclength(X0, _T_HAT, ds),)
            out = corrector.solve(
                clean_guess, free_vars=_FREE_VARS, free_times=_FREE_TIMES,
                constraints=constraints, continuation=True,
            )
            assert out.converged, f"march member at ds={ds} did not converge"
            assert out.continuation is not None
            return out.continuation

        near, far = member(_DS), member(2.0 * _DS)

        null = np.linalg.svd(near.DH)[2][-1]
        secant = np.asarray(far.X) - np.asarray(near.X)
        secant = secant / np.linalg.norm(secant)

        # The null vector's sign is arbitrary (SVD convention); the family
        # direction is what is being compared, so align before measuring.
        if secant @ null < 0.0:
            null = -null
        angle = np.degrees(np.arccos(np.clip(secant @ null, -1.0, 1.0)))

        assert angle < 0.5

# ===========================================================================
# The tangent function: DH -> t_hat
# ===========================================================================
#
# _family_tangent is pure linear algebra, so most of it is tested on
# synthetic DH matrices whose null direction is known exactly. _dh_with_null
# builds one: an orthonormal basis for the complement of a chosen vector v,
# mixed by a random, well-conditioned matrix so the rows look nothing like
# that basis. The null space is span(v) by construction, and the sign SVD
# hands back for it is arbitrary -- which is exactly the ambiguity the
# sign-resolution logic exists to settle.

def _dh_with_null(v, seed: int = 0) -> np.ndarray:
    """
    Return a (n - 1, n) matrix whose null space is exactly span(v).

    Rows 1.. of the full right-singular basis of v^T span v's orthogonal
    complement; a random mixing matrix (identity-shifted to stay well
    conditioned) scrambles them without changing the row space.
    """
    v = np.asarray(v, dtype=float)
    v = v / np.linalg.norm(v)
    n = v.size
    complement = np.linalg.svd(v.reshape(1, -1))[2][1:]
    rng = np.random.default_rng(seed)
    mix = rng.normal(size=(n - 1, n - 1)) + 3.0 * np.eye(n - 1)
    return mix @ complement


# A unit null direction with a nonzero period component (the last entry),
# so seed mode is well-posed: (1, 2, 2) / 3.
_V = np.array([1.0, 2.0, 2.0]) / 3.0

# dT/dX for n_X = 3 with the period in the last column, the single-shooting
# layout [x, vy, T_half].
_E_T = np.array([0.0, 0.0, 1.0])


class TestCheckDirection:
    """The shared direction validator used by _family_tangent and the march."""

    @pytest.mark.parametrize("value", [1, -1, np.int64(1), np.int32(-1)])
    def test_accepts_plus_and_minus_one(self, value):
        out = _check_direction(value)
        assert out == int(value)

    def test_returns_a_plain_int(self):
        """NumPy integers come back as Python ints, not np.int64."""
        assert type(_check_direction(np.int64(-1))) is int

    @pytest.mark.parametrize("value", [True, False])
    def test_rejects_bool(self, value):
        """
        bool subclasses int, so True would otherwise pass as 1. Rejected so a
        boolean is never silently read as a direction.
        """
        with pytest.raises(TypeError, match="must be an integer"):
            _check_direction(value)

    @pytest.mark.parametrize("value", [1.0, "1", None])
    def test_rejects_non_integers(self, value):
        with pytest.raises(TypeError, match="must be an integer"):
            _check_direction(value)

    @pytest.mark.parametrize("value", [0, 2, -2])
    def test_rejects_other_integers(self, value):
        with pytest.raises(ValueError, match="must be 1 or -1"):
            _check_direction(value)


class TestFamilyTangentNullDirection:
    """Whatever the mode, the result spans DH's null space and is unit."""

    @pytest.mark.parametrize("n", [3, 4])
    def test_returns_a_unit_vector(self, n):
        v = np.arange(1.0, n + 1.0)
        t = _family_tangent(_dh_with_null(v), prev_t_hat=v / np.linalg.norm(v))
        assert np.linalg.norm(t) == pytest.approx(1.0, abs=1e-14)

    @pytest.mark.parametrize("n", [3, 4])
    def test_annihilates_DH(self, n):
        v = np.arange(1.0, n + 1.0)
        DH = _dh_with_null(v)
        t = _family_tangent(DH, prev_t_hat=v / np.linalg.norm(v))
        assert np.linalg.norm(DH @ t) < 1e-12 * np.linalg.norm(DH)

    def test_is_parallel_to_the_known_null_direction(self):
        t = _family_tangent(_dh_with_null(_V), prev_t_hat=_V)
        assert abs(t @ _V) == pytest.approx(1.0, abs=1e-12)

    def test_the_economy_svd_trap_is_real(self):
        """
        Why full_matrices=True is load-bearing. DH is one column wider than
        tall, so the economy SVD returns only n_X - 1 right singular vectors
        -- every one of them for a genuine, nonzero singular value. Its last
        row is a plausible-looking unit vector that does NOT annihilate DH.
        The first assertion proves the trap exists for this matrix, so the
        second is not passing vacuously.
        """
        DH = _dh_with_null(_V)
        economy_last = np.linalg.svd(DH, full_matrices=False)[2][-1]
        t = _family_tangent(DH, prev_t_hat=_V)

        assert np.linalg.norm(DH @ economy_last) > 1e-3 * np.linalg.norm(DH)
        assert np.linalg.norm(DH @ t) < 1e-12 * np.linalg.norm(DH)

    def test_depends_only_on_the_null_space(self):
        """
        Two DH matrices with different rows but the same null space give the
        same resolved tangent. The row mixing is how a real corrector
        Jacobian differs from any tidy basis, so this is the property that
        makes the synthetic tests stand in for real ones.
        """
        t_a = _family_tangent(_dh_with_null(_V, seed=1), dT_dX=_E_T)
        t_b = _family_tangent(_dh_with_null(_V, seed=2), dT_dX=_E_T)
        np.testing.assert_allclose(t_a, t_b, atol=1e-12)

    def test_is_invariant_to_scaling_DH(self):
        DH = _dh_with_null(_V)
        np.testing.assert_allclose(
            _family_tangent(1e6 * DH, dT_dX=_E_T),
            _family_tangent(DH, dT_dX=_E_T),
            atol=1e-12,
        )


class TestFamilyTangentContinuityMode:
    """prev_t_hat given: flip if needed to agree with the previous tangent."""

    @pytest.mark.parametrize("sign", [1.0, -1.0])
    def test_aligns_with_the_previous_tangent(self, sign):
        prev = sign * _V
        t = _family_tangent(_dh_with_null(_V), prev_t_hat=prev)
        assert t @ prev > 0.0
        np.testing.assert_allclose(t, prev, atol=1e-12)

    @pytest.mark.parametrize("seed", range(10))
    def test_sign_does_not_depend_on_the_svd_convention(self, seed):
        """
        Different row mixings leave SVD free to return either sign for the
        null vector; the resolved tangent must come out the same every time.
        """
        t = _family_tangent(_dh_with_null(_V, seed=seed), prev_t_hat=_V)
        np.testing.assert_allclose(t, _V, atol=1e-12)

    def test_aligns_with_a_rotated_previous_tangent(self):
        """
        In a march the previous tangent is not the current null vector, only
        close to it. A prev rotated a few degrees off still selects the
        nearby sign.
        """
        off = np.array([2.0, -1.0, 0.0]) / np.sqrt(5.0)   # orthogonal to _V
        prev = _V + 0.1 * off
        prev = prev / np.linalg.norm(prev)
        t = _family_tangent(_dh_with_null(_V), prev_t_hat=prev)
        np.testing.assert_allclose(t, _V, atol=1e-12)

    def test_direction_is_ignored(self):
        """Sign comes from prev_t_hat alone; direction is for seed mode."""
        DH = _dh_with_null(_V)
        np.testing.assert_allclose(
            _family_tangent(DH, prev_t_hat=_V, direction=-1),
            _family_tangent(DH, prev_t_hat=_V, direction=1),
            atol=1e-14,
        )


class TestFamilyTangentSeedMode:
    """dT_dX given: sign so the period moves the way `direction` asks."""

    def test_plus_one_points_toward_increasing_period(self):
        t = _family_tangent(_dh_with_null(_V), dT_dX=_E_T, direction=1)
        assert t @ _E_T > 0.0

    def test_minus_one_points_toward_decreasing_period(self):
        t = _family_tangent(_dh_with_null(_V), dT_dX=_E_T, direction=-1)
        assert t @ _E_T < 0.0

    def test_the_two_directions_are_exact_opposites(self):
        DH = _dh_with_null(_V)
        np.testing.assert_allclose(
            _family_tangent(DH, dT_dX=_E_T, direction=-1),
            -_family_tangent(DH, dT_dX=_E_T, direction=1),
            atol=1e-14,
        )

    @pytest.mark.parametrize("seed", range(10))
    def test_sign_does_not_depend_on_the_svd_convention(self, seed):
        t = _family_tangent(_dh_with_null(_V, seed=seed), dT_dX=_E_T)
        np.testing.assert_allclose(t, _V, atol=1e-12)

    def test_only_the_sign_of_dT_dX_matters(self):
        """
        The march passes the honest gradient (2.0 in the period column, for
        a half-arc solve), not a one-hot. Scaling must change nothing.
        """
        DH = _dh_with_null(_V)
        np.testing.assert_allclose(
            _family_tangent(DH, dT_dX=2.0 * _E_T),
            _family_tangent(DH, dT_dX=_E_T),
            atol=1e-14,
        )

    def test_raises_at_a_period_extremum(self):
        """
        A null direction with no period component is the seed-at-an-extremum
        case: the period does not change to first order along the family, so
        'increasing period' does not pick a side.
        """
        v = np.array([0.6, -0.8, 0.0])
        with pytest.raises(ValueError, match="degenerate"):
            _family_tangent(_dh_with_null(v), dT_dX=_E_T)

    def test_a_small_but_resolvable_period_component_is_accepted(self):
        """
        The degeneracy threshold is relative (1e-8 of |dT_dX|), not a demand
        for a large period component. A tangent that is mostly state-space
        but carries a clear period slope still signs.
        """
        v = np.array([1.0, 1.0, 1e-6])
        t = _family_tangent(_dh_with_null(v), dT_dX=_E_T, direction=1)
        assert t @ _E_T > 0.0


class TestFamilyTangentValidation:
    """Argument checks. Messages are matched so the right check is proven."""

    def test_requires_one_of_the_two_modes(self):
        with pytest.raises(ValueError, match="Exactly one of"):
            _family_tangent(_dh_with_null(_V))

    def test_rejects_both_modes_at_once(self):
        with pytest.raises(ValueError, match="Exactly one of"):
            _family_tangent(_dh_with_null(_V), prev_t_hat=_V, dT_dX=_E_T)

    def test_rejects_a_one_dimensional_DH(self):
        with pytest.raises(ValueError, match="must be 2-D"):
            _family_tangent(np.ones(3), prev_t_hat=_V)

    @pytest.mark.parametrize("shape", [(3, 3), (3, 2), (1, 3)])
    def test_rejects_a_DH_that_is_not_corank_one_shaped(self, shape):
        with pytest.raises(ValueError, match="exactly one more column"):
            _family_tangent(np.ones(shape), prev_t_hat=_V)

    def test_rejects_a_nonfinite_DH(self):
        DH = _dh_with_null(_V)
        DH[0, 1] = np.nan
        with pytest.raises(ValueError, match="DH must be finite"):
            _family_tangent(DH, prev_t_hat=_V)

    def test_rejects_a_prev_t_hat_of_the_wrong_width(self):
        with pytest.raises(ValueError, match="prev_t_hat must have shape"):
            _family_tangent(_dh_with_null(_V), prev_t_hat=np.ones(4))

    def test_rejects_a_nonfinite_prev_t_hat(self):
        with pytest.raises(ValueError, match="prev_t_hat must be finite"):
            _family_tangent(_dh_with_null(_V),
                            prev_t_hat=np.array([1.0, np.inf, 0.0]))

    def test_rejects_a_dT_dX_of_the_wrong_width(self):
        with pytest.raises(ValueError, match="dT_dX must have shape"):
            _family_tangent(_dh_with_null(_V), dT_dX=np.ones(2))

    def test_rejects_a_nonfinite_dT_dX(self):
        with pytest.raises(ValueError, match="dT_dX must be finite"):
            _family_tangent(_dh_with_null(_V),
                            dT_dX=np.array([0.0, 0.0, np.nan]))

    def test_rejects_an_all_zero_dT_dX(self):
        with pytest.raises(ValueError, match="identically zero"):
            _family_tangent(_dh_with_null(_V), dT_dX=np.zeros(3))

    @pytest.mark.parametrize("mode", ["continuity", "seed"])
    def test_validates_direction_in_both_modes(self, mode):
        """
        direction is only *used* in seed mode, but a bad value is still a
        caller error in continuity mode and is rejected there too.
        """
        kwargs = ({"prev_t_hat": _V} if mode == "continuity"
                  else {"dT_dX": _E_T})
        with pytest.raises(TypeError, match="must be an integer"):
            _family_tangent(_dh_with_null(_V), direction=True, **kwargs)
        with pytest.raises(ValueError, match="must be 1 or -1"):
            _family_tangent(_dh_with_null(_V), direction=0, **kwargs)


@pytest.mark.slow
class TestFamilyTangentOnRealDH:
    """
    The synthetic tests above pin the linear algebra; these tie it back to
    a DH the corrector actually produced, reusing this module's fixtures.
    """

    def test_seed_mode_signs_a_real_tangent(self, closed_solve):
        """
        On a real corank-1 DH, both directions give null vectors of DH with
        opposite period slopes. The period is the last X column for the
        (x, vy) + free end time layout.
        """
        result, _, _ = closed_solve
        DH = result.continuation.DH
        t_up = _family_tangent(DH, dT_dX=_E_T, direction=1)
        t_down = _family_tangent(DH, dT_dX=_E_T, direction=-1)

        assert t_up[-1] > 0.0
        assert t_down[-1] < 0.0
        assert np.linalg.norm(DH @ t_up) < 1e-10 * np.linalg.norm(DH)

    def test_continuity_mode_follows_the_family(self, clean_guess):
        """
        The existing null-vector test aligns SVD's sign by hand before
        comparing to the secant. Here _family_tangent does the aligning:
        handed the secant as the previous direction, it must return the null
        vector that points along the family, to the same 0.5 degree
        tolerance, with no manual sign fix.
        """
        base = (TargetState(dict(_TARGETS)),)
        ctx_open = _ShootingContext.from_guess(
            clean_guess, _FREE_VARS, base, _FREE_TIMES, None
        )
        X0 = _pack(clean_guess, ctx_open)
        corrector = DifferentialCorrector()

        def member(ds):
            constraints = base + (PseudoArclength(X0, _T_HAT, ds),)
            out = corrector.solve(
                clean_guess, free_vars=_FREE_VARS, free_times=_FREE_TIMES,
                constraints=constraints, continuation=True,
            )
            assert out.converged, f"member at ds={ds} did not converge"
            assert out.continuation is not None
            return out.continuation

        near, far = member(_DS), member(2.0 * _DS)
        secant = np.asarray(far.X) - np.asarray(near.X)
        secant = secant / np.linalg.norm(secant)

        t = _family_tangent(near.DH, prev_t_hat=secant)
        angle = np.degrees(np.arccos(np.clip(t @ secant, -1.0, 1.0)))

        assert t @ secant > 0.0
        assert angle < 0.5
