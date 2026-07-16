"""
Test suite for the PeriodicOrbit class.

Organized in three tiers:

  Tier 1 -- Mathematical invariants (the anchor tests). The monodromy of a
    CR3BP periodic orbit is symplectic, has unit determinant, and its
    eigenvalues (Floquet multipliers) occur in reciprocal pairs with a
    trivial pair near +1. These are convention-free and catch STM-path
    regressions (sign errors, index transpositions, wrong composite object)
    that would otherwise only surface later as continuation failures.

  Tier 2 -- Construction logic. The three period-determination paths
    (explicit / full-period inference / mirror-half inference) plus the
    STM-forced repropagation predicate. These exercise the branching in
    __init__ and _infer_period, including the "reuse vs. repropagate"
    optimization via object-identity checks.

  Tier 3 -- Guards and derived quantities. Validation ValueErrors, closure
    failure on a non-periodic arc, derived scalars (jacobi, stability_index,
    periodicity_residual), and the manifold stubs.

Tolerances: the monodromy invariants are checked against a loose absolute
tolerance (INVARIANT_ATOL). This is deliberate -- the Gateway NRHO closes to
~3.5e-12 and its monodromy is stiff, so symplecticity holds to ~1e-6, not
machine epsilon. Testing tighter would test integrator accuracy, not the
invariant. The well-conditioned Lyapunov will beat this bound comfortably.

Reference orbits come from the cached lyapunov_orbit / gateway_orbit defaults
(already-built PeriodicOrbit objects, used for the invariant and derived-
quantity tiers). Raw Trajectory inputs for the construction tier come from the
make_periodic_guess fixture, which returns a single-segment Trajectory.

Created with the assistance of Claude Opus 4.8 by Anthropic.
"""

import numpy as np
import pytest

import kyklos as ky
from kyklos.periodic_orbit import PeriodicOrbit
from kyklos.exceptions import ClosureError


# ========== MODULE CONSTANTS ==========

# Loose absolute tolerance for the monodromy invariants. See module docstring:
# sized for the stiff NRHO, not the well-conditioned Lyapunov.
INVARIANT_ATOL = 1e-6

# Tolerance for identifying the trivial (+1) Floquet pair. The defective Jordan
# block at +1 has sqrt-epsilon sensitivity, so the trivial multipliers can drift
# meaningfully off 1; this bound is correspondingly generous.
TRIVIAL_ATOL = 1e-4


# ========== SYMPLECTIC FORM ==========

def _symplectic_J():
    """
    Return the 6x6 symplectic form J = [[0, I], [-I, 0]] for the CR3BP state
    ordering [x, y, z, vx, vy, vz] (position block then velocity block).
    """
    I3 = np.eye(3)
    Z3 = np.zeros((3, 3))
    top = np.hstack([Z3, I3])
    bot = np.hstack([-I3, Z3])
    return np.vstack([top, bot])

def _velocity_symplectic_form():
    """
    Symplectic form for Kyklos's velocity-coordinate CR3BP state
    [x, y, z, vx, vy, vz]. Equals C^T J C, where C maps velocity to canonical
    momentum: px = vx - y, py = vy + x, pz = vz (rotating-frame Hamiltonian,
    unit rotation rate). Velocity-coordinate STMs preserve THIS form, not the
    plain J -- testing against plain J fails in the in-plane block.
    """
    C = np.array([
        [ 1,  0,  0,  0,  0,  0],   # x
        [ 0,  1,  0,  0,  0,  0],   # y
        [ 0,  0,  1,  0,  0,  0],   # z
        [ 0, -1,  0,  1,  0,  0],   # px = vx - y
        [ 1,  0,  0,  0,  1,  0],   # py = vy + x
        [ 0,  0,  0,  0,  0,  1],   # pz = vz
    ], dtype=float)
    J = _symplectic_J()
    return C.T @ J @ C


# ========== FIXTURES ==========

@pytest.fixture(params=['lyapunov', 'gateway'])
def reference_orbit(request, lyapunov_orbit, gateway_orbit):
    """
    Parametrized reference PeriodicOrbit: the L1 Lyapunov (well-conditioned)
    and the Gateway L2 NRHO (stiff). Invariant and derived-quantity tests run
    against both so the invariants are checked across the conditioning range.
    """
    return lyapunov_orbit if request.param == 'lyapunov' else gateway_orbit


# ==================================================================
# Tier 1 -- Mathematical invariants (anchor tests)
# ==================================================================
class TestMonodromyInvariants:
    """Structural invariants of the monodromy matrix and its spectrum."""

    def test_monodromy_shape_and_finiteness(self, reference_orbit):
        """Monodromy is a real, finite 6x6 matrix."""
        M = reference_orbit.monodromy
        assert M.shape == (6, 6)
        assert np.all(np.isfinite(M))

    def test_monodromy_is_symplectic(self, reference_orbit):
        """
        M^T J M = J. The central anchor test: validates the entire STM
        integration path (correct variational integration, correct composite
        object from get_stm, correct position/velocity state ordering).

        Note that Hamiltonian generalized momenta must be used in order to use the 
        canonical sypmplectic J form.  
        """
        M = reference_orbit.monodromy
        J = _velocity_symplectic_form()
        residual = M.T @ J @ M - J
        assert np.allclose(residual, 0.0, atol=INVARIANT_ATOL), (
            f"Symplecticity violated: max|M^T J M - J| = "
            f"{np.max(np.abs(residual)):.3e}"
        )

    def test_monodromy_unit_determinant(self, reference_orbit):
        """
        det(M) = 1. Corollary of symplecticity, but an independent scalar
        check: determinant drift localizes to conditioning/accuracy, whereas
        a full symplecticity failure points at structure.
        """
        det = np.linalg.det(reference_orbit.monodromy)
        assert abs(det - 1.0) < INVARIANT_ATOL, f"det(M) = {det:.6e}, expected 1"

    def test_floquet_reciprocal_pairs(self, reference_orbit):
        """
        Multipliers are closed under lambda -> 1/lambda. For each multiplier
        there is a partner equal (within tolerance) to its reciprocal. Checked
        via the multiset of reciprocals matching the multiset of multipliers.
        """
        mults = reference_orbit.floquet_multipliers
        recips = 1.0 / mults
        # Every reciprocal should match some multiplier. Greedy match so each
        # multiplier is consumed once (guards against a single value soaking up
        # multiple reciprocals).
        remaining = list(mults)
        for r in recips:
            diffs = [abs(r - m) for m in remaining]
            j = int(np.argmin(diffs))
            assert diffs[j] < 1e-3, (
                f"reciprocal {r:.4e} has no matching multiplier "
                f"(closest gap {diffs[j]:.3e})"
            )
            remaining.pop(j)

    def test_floquet_product_is_unity(self, reference_orbit):
        """Product of all six multipliers equals 1 (det via the spectrum)."""
        prod = np.prod(reference_orbit.floquet_multipliers)
        assert abs(prod - 1.0) < INVARIANT_ATOL, f"prod(lambda) = {prod:.6e}"

    def test_floquet_trivial_pair_near_unity(self, reference_orbit):
        """
        At least two multipliers lie near +1 (the trivial pair from the
        periodicity / energy integral). Tolerance is generous because the
        defective Jordan block at +1 is sqrt-epsilon sensitive.
        """
        mults = reference_orbit.floquet_multipliers
        near_one = np.abs(mults - 1.0) < TRIVIAL_ATOL
        assert np.count_nonzero(near_one) >= 2, (
            f"expected >= 2 multipliers near +1, found "
            f"{np.count_nonzero(near_one)}; multipliers = {mults}"
        )

    def test_floquet_sorted_by_descending_magnitude(self, reference_orbit):
        """floquet_multipliers is documented as sorted by descending |lambda|."""
        mags = np.abs(reference_orbit.floquet_multipliers)
        assert np.all(np.diff(mags) <= 1e-12), (
            f"multipliers not sorted by descending magnitude: {mags}"
        )


# ==================================================================
# Tier 2 -- Construction logic (period determination + reprop predicate)
# ==================================================================
class TestConstructionExplicit:
    """Explicit-period path: mode == 'explicit', reuse vs. repropagate."""

    def test_explicit_full_span_reuses_trajectory(self, make_periodic_guess,
                                                   lyapunov_orbit):
        """
        A full-period trajectory WITH an STM, given its known period explicitly,
        is reused verbatim -- not repropagated. The `is` identity check is the
        only observable signature of the "don't repropagate" optimization.
        """
        traj = make_periodic_guess(lyapunov_orbit, n_periods=1.0, with_stm=True)
        po = PeriodicOrbit(traj, period=lyapunov_orbit.period)
        assert po.mode == 'explicit'
        assert po.trajectory is traj          # reused, not rebuilt
        assert np.isclose(po.period, lyapunov_orbit.period)

    def test_explicit_period_matches_stored(self, make_periodic_guess,
                                             lyapunov_orbit):
        """The explicitly supplied period is the one stored."""
        traj = make_periodic_guess(lyapunov_orbit, n_periods=1.0, with_stm=True)
        T = lyapunov_orbit.period
        po = PeriodicOrbit(traj, period=T)
        assert po.period == float(T)


class TestConstructionInference:
    """Inference paths: full-period closure vs. mirror-half perpendicular."""

    def test_full_period_inference(self, make_periodic_guess, lyapunov_orbit):
        """
        A closed full-period arc with no explicit period infers mode
        'full_period' and period == span.
        """
        traj = make_periodic_guess(lyapunov_orbit, n_periods=1.0, with_stm=True)
        po = PeriodicOrbit(traj)
        assert po.mode == 'full_period'
        assert np.isclose(po.period, lyapunov_orbit.period,
                          atol=1e-9, rtol=1e-9)

    def test_mirror_half_inference(self, make_periodic_guess, lyapunov_orbit):
        """
        A mirror half-arc (n_periods=0.5, both endpoints perpendicular x-z
        crossings) infers mode 'mirror_half' and period == 2 * span. This is
        the subtle path -- it repropagates and doubles -- and is the preferred
        shooting formulation, so it must be correct.
        """
        half = make_periodic_guess(lyapunov_orbit, n_periods=0.5, with_stm=True)
        po = PeriodicOrbit(half)
        assert po.mode == 'mirror_half'
        assert np.isclose(po.period, lyapunov_orbit.period,
                          atol=1e-9, rtol=1e-9)

    def test_mirror_half_closes(self, make_periodic_guess, lyapunov_orbit):
        """The mirror-half-inferred full orbit actually closes on itself."""
        half = make_periodic_guess(lyapunov_orbit, n_periods=0.5, with_stm=True)
        po = PeriodicOrbit(half)
        assert po.periodicity_residual < po.tol


class TestRepropagationPredicate:
    """The must_reprop predicate: fires on missing STM, else reuses."""

    def test_no_stm_forces_repropagation(self, make_periodic_guess,
                                         lyapunov_orbit):
        """
        A full-period trajectory built WITHOUT an STM still constructs (the
        monodromy is recovered by repropagation), and the stored trajectory is
        a NEW object -- complementary to the reuse test, proving must_reprop
        both fires and doesn't-fire correctly.
        """
        traj = make_periodic_guess(lyapunov_orbit, n_periods=1.0, with_stm=False)
        assert traj.stm_order is None
        po = PeriodicOrbit(traj, period=lyapunov_orbit.period)
        assert po.trajectory is not traj                 # repropagated
        assert po.trajectory.stm_order is not None        # now has an STM
        assert po.monodromy.shape == (6, 6)


# ==================================================================
# Tier 3 -- Guards, derived quantities, stubs
# ==================================================================
class TestConstructionGuards:
    """Validation errors on malformed inputs."""

    def test_non_cr3bp_system_rejected(self, make_periodic_guess,
                                       lyapunov_orbit, earth_2bp_system):
        """
        A trajectory in a non-CR3BP system is rejected. Built by propagating a
        2-body trajectory and handing it to PeriodicOrbit.

        NOTE: adjust the fixture name / construction below to match conftest.
        """
        # A minimal 2-body trajectory; the point is only that base_type != 3body.
        oe = ky.LEO_ORBIT
        traj = earth_2bp_system.propagate(oe, [0.0, 100.0], with_stm=True)
        with pytest.raises(ValueError, match="CR3BP"):
            PeriodicOrbit(traj)

    def test_nonpositive_period_rejected(self, make_periodic_guess,
                                        lyapunov_orbit):
        """A non-positive explicit period raises ValueError."""
        traj = make_periodic_guess(lyapunov_orbit, n_periods=1.0, with_stm=True)
        with pytest.raises(ValueError, match="period must be positive"):
            PeriodicOrbit(traj, period=-1.0)

    def test_negative_tol_rejected(self, make_periodic_guess, lyapunov_orbit):
        """A negative tolerance raises ValueError."""
        traj = make_periodic_guess(lyapunov_orbit, n_periods=1.0, with_stm=True)
        with pytest.raises(ValueError, match="tol"):
            PeriodicOrbit(traj, period=lyapunov_orbit.period, tol=-1e-9)

    def test_uninferable_arc_rejected(self, make_periodic_guess,
                                     lyapunov_orbit):
        """
        A partial arc that neither closes nor terminates in double
        perpendicular crossings cannot be inferred and raises ValueError.
        A quarter-period arc of the Lyapunov is the natural example: its
        endpoints are neither coincident nor both perpendicular.
        """
        quarter = make_periodic_guess(lyapunov_orbit, n_periods=0.25,
                                      with_stm=True)
        with pytest.raises(ValueError, match="[Cc]annot infer"):
            PeriodicOrbit(quarter)

    def test_nonperiodic_arc_fails_closure(self, make_periodic_guess,
                                        lyapunov_orbit):
        """
        A wrong (non-closing) period drives the closure check to fail. A quarter-
        period span declared as a full period will not close, so validation_error
        raises. STRICT_VALIDATION is pinned True here so the test asserts the raise
        path deterministically, independent of the ambient config.
        """
        quarter = make_periodic_guess(lyapunov_orbit, n_periods=0.25,
                                    with_stm=True)
        span = quarter.duration
        with ky.temp_config(STRICT_VALIDATION=True):
            with pytest.raises(ClosureError, match="does not close"):
                PeriodicOrbit(quarter, period=span)


class TestDerivedQuantities:
    """Derived scalars and their consistency."""

    def test_jacobi_matches_initial_state(self, reference_orbit):
        """po.jacobi delegates to initial_state.jacobi_const()."""
        assert np.isclose(reference_orbit.jacobi,
                          reference_orbit.initial_state.jacobi_const())

    def test_jacobi_is_finite_scalar(self, reference_orbit):
        """Jacobi constant is a finite float."""
        C = reference_orbit.jacobi
        assert isinstance(C, float)
        assert np.isfinite(C)

    def test_stability_index_at_least_one(self, reference_orbit):
        """
        nu = 0.5(|lambda_max| + 1/|lambda_max|) >= 1 for any nonzero
        multiplier (AM-GM equality only at |lambda| = 1).
        """
        assert reference_orbit.stability_index >= 1.0 - 1e-9

    def test_lyapunov_more_stable_than_nrho_is_not_assumed(self,
                                                           lyapunov_orbit,
                                                           gateway_orbit):
        """
        Both stability indices are well-defined and finite. (We do NOT assert
        an ordering between them: the NRHO is only weakly unstable while the
        Lyapunov can be strongly unstable, so the naive 'NRHO is more stable'
        intuition does not hold as a robust invariant.)
        """
        assert np.isfinite(lyapunov_orbit.stability_index)
        assert np.isfinite(gateway_orbit.stability_index)

    def test_periodicity_residual_small(self, reference_orbit):
        """The stored closure residual is within the orbit's own tolerance."""
        assert reference_orbit.periodicity_residual < reference_orbit.tol


class TestReadOnlyOutputs:
    """Returned arrays must not alias or expose mutable internal state."""

    def test_monodromy_is_read_only(self, reference_orbit):
        """The returned monodromy is a read-only copy."""
        M = reference_orbit.monodromy
        assert M.flags.writeable is False

    def test_monodromy_returns_independent_copy(self, reference_orbit):
        """Two accesses return equal but distinct arrays (no shared buffer)."""
        M1 = reference_orbit.monodromy
        M2 = reference_orbit.monodromy
        assert M1 is not M2
        assert np.array_equal(M1, M2)

    def test_multipliers_read_only(self, reference_orbit):
        """floquet_multipliers returns a read-only array."""
        assert reference_orbit.floquet_multipliers.flags.writeable is False


class TestManifoldStubs:
    """Manifold generation is stubbed; both raise NotImplementedError."""

    def test_stable_manifold_stub(self, reference_orbit):
        with pytest.raises(NotImplementedError):
            reference_orbit.stable_manifold()

    def test_unstable_manifold_stub(self, reference_orbit):
        with pytest.raises(NotImplementedError):
            reference_orbit.unstable_manifold()


class TestReprAndSummary:
    """Smoke tests for the human-readable representations."""

    def test_repr_smoke(self, reference_orbit):
        """repr() runs and mentions the class name."""
        r = repr(reference_orbit)
        assert 'PeriodicOrbit' in r

    def test_summary_smoke(self, reference_orbit, capsys):
        """summary() runs and prints something."""
        reference_orbit.summary()
        out = capsys.readouterr().out
        assert 'PeriodicOrbit' in out
