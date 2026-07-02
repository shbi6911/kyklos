"""
Tier 2 (slow) numerical tests for the differential corrector.

These validate the shooter against real Earth-Moon CR3BP dynamics, in regimes
the fast/deterministic tier (driven by the fake systems in conftest) cannot
reach.

TestAssembleDFFiniteDifference is the gate. The analytically assembled
Jacobian DF -- built from propagated variational STMs and, for free times,
the vector field -- is checked column by column against a central finite
difference of the real shooting residual F(X). The residual is differenced as
a black box (unpack X, re-propagate every segment, assemble F), so the FD
captures every coupling F has on X without any knowledge of the block
structure. That makes this a genuine certification of _assemble_DF against the
true dynamics -- the interior +I / -Phi blocks, the terminal Jtf @ Phi + Jx0
rows, and the +/- f(endpoint) free-time columns -- rather than a restatement
of the assembly logic.

The harness mirrors the top of DifferentialCorrector._run exactly: F(X) =
_assemble_F(system.propagate(*_unpack(X, ctx)), ctx). F is independent of the
STM, so the finite-difference evaluations propagate without one (faster);
the analytic DF propagates with_stm=True, as _run does.

The whole module is slow-marked; run with `pytest -m slow` (or the full
suite), and exclude with `pytest -m "not slow"`.
"""

import numpy as np
import pytest

from kyklos.shooter import (
    _ShootingContext,
    _pack,
    _unpack,
    _assemble_F,
    _assemble_DF,
    TargetState,
    Periodicity,
    DifferentialCorrector,
)
from kyklos import NullJunctionNode

pytestmark = pytest.mark.slow


# Finite-difference step and comparison tolerances. eps_rel sits near the
# cube-root-of-machine-eps sweet spot for O(1) nondimensional states. The
# analytic columns come from variational STMs (essentially exact), so the
# discrepancy is set by the central-difference roundoff floor (~1e-10 on O(1)
# entries) -- comfortably inside these tolerances. Loosen/tighten here if a
# stiffer reference orbit or integrator tolerance needs it.
_EPS_REL = 1e-7
_RTOL = 1e-6
_ATOL = 1e-8


def _residual_at(X: np.ndarray, ctx: _ShootingContext) -> np.ndarray:
    """
    Production residual F(X): unpack -> propagate (no STM) -> assemble_F.

    Mirrors the top of DifferentialCorrector._run, minus the STM (F does not
    depend on it), so the finite difference differentiates exactly the
    residual the corrector solves.
    """
    ics, times = _unpack(X, ctx)
    traj = ctx.system.propagate(ics, times, with_stm=False)
    return _assemble_F(traj, ctx)


def _analytic_DF(X: np.ndarray, ctx: _ShootingContext) -> np.ndarray:
    """Production DF(X): unpack -> propagate (with STM) -> assemble_DF."""
    ics, times = _unpack(X, ctx)
    traj = ctx.system.propagate(ics, times, with_stm=True)
    return _assemble_DF(traj, ctx)


def _fd_DF(X: np.ndarray, ctx: _ShootingContext,
           eps_rel: float = _EPS_REL) -> np.ndarray:
    """
    Central finite-difference Jacobian of F(X).

    Column j is (F(X + h e_j) - F(X - h e_j)) / (2 h), with a per-component
    step h = eps_rel * max(1, |X_j|) so it scales sensibly for both O(1) state
    entries and any larger free-time entries. Every evaluation re-propagates
    the whole trajectory, so all couplings are captured.
    """
    X = np.asarray(X, dtype=float)
    n = X.size
    cols = []
    for j in range(n):
        h = eps_rel * max(1.0, abs(X[j]))
        Xp = X.copy()
        Xp[j] += h
        Xm = X.copy()
        Xm[j] -= h
        cols.append(
            (_residual_at(Xp, ctx) - _residual_at(Xm, ctx)) / (2.0 * h)
        )
    return np.column_stack(cols)


def _check_DF(guess, free_vars, constraints=None, free_times=None,
              perturb_scale=None, seed=0,
              eps_rel=_EPS_REL, rtol=_RTOL, atol=_ATOL):
    """
    Build the solve context from a guess and assert the analytic DF matches
    the finite-difference DF at the packed X.

    With perturb_scale set, X is displaced by perturb_scale * N(0, 1) (seeded)
    before the comparison, so the agreement is checked at a generic point
    rather than only at the structured guess.
    """
    ctx = _ShootingContext.from_guess(guess, free_vars, constraints, free_times)
    X = _pack(guess, ctx)
    if perturb_scale is not None:
        rng = np.random.default_rng(seed)
        X = X + perturb_scale * rng.standard_normal(ctx.n_X)

    DF_an = _analytic_DF(X, ctx)
    DF_fd = _fd_DF(X, ctx, eps_rel=eps_rel)

    assert DF_an.shape == DF_fd.shape, (
        f"shape mismatch: analytic {DF_an.shape} vs fd {DF_fd.shape}"
    )
    np.testing.assert_allclose(DF_an, DF_fd, rtol=rtol, atol=atol)

# Canonical component-name -> index map for the base state [x,y,z,vx,vy,vz].
_IDX = {'x': 0, 'y': 1, 'z': 2, 'vx': 3, 'vy': 4, 'vz': 5}


def _mirror_guess(system, orbit, free, perturb_rel=1e-2, t_scale=1.0,
                  with_stm=False):
    """
    Single-segment half-orbit guess for the mirror-theorem formulation.

    Starts from the orbit's stored state, forces the perpendicular-crossing
    symmetry exactly (y = vx = vz = 0 -- the stored values carry only
    numerical noise there, and the symmetry argument needs hard zeros), nudges
    the free components off the orbit by a relative perturbation so the
    corrector has something to correct, then propagates over the orbit's
    half-period. t_scale != 1 seeds an off-nominal half
    period for free-time solves so the free-time Jacobian column does real
    work; leave it at 1.0 for fixed-time solves.

    Parameters
    ----------
    system : System
        Compiled CR3BP system.
    orbit : PeriodicOrbit
        Reference orbit; orbit.initial_state is OrbitalElements, orbit.period a float.
    free : sequence of str
        Component names to perturb (the symmetric IC's nonzero entries).
    perturb_rel : float, optional
        Relative displacement applied to each free component. Default 1e-2.
    with_stm : bool, optional
        Passed through to propagate. The guess is only read for structure and
        packed values, so the default is False.

    Returns
    -------
    Trajectory
        Single-segment guess over [0, orbit.period / 2].
    """
    x = np.asarray(orbit.initial_state.elements, dtype=float).copy()
    x[[_IDX['y'], _IDX['vx'], _IDX['vz']]] = 0.0
    for name in free:
        x[_IDX[name]] *= (1.0 + perturb_rel)
    t_half = orbit.period / 2.0 * t_scale
    return system.propagate(x, [0.0, t_half], with_stm=with_stm)

class TestAssembleDFFiniteDifference:
    """Analytic _assemble_DF vs a central finite difference of the residual."""

    # ---- single shooting: start-state columns + terminal rows ----

    def test_single_shoot_targetstate(self, make_periodic_guess,
                                      lyapunov_orbit):
        """
        Free full start state, TargetState terminal. Exercises the terminal
        rows Jtf @ Phi[:, free_idx]; TargetState has no x0-dependence, so the
        Jx0 term is zero here.
        """
        guess = make_periodic_guess(lyapunov_orbit, n_periods=0.5)
        _check_DF(guess, free_vars='all',
                  constraints=[TargetState({'y': 0.0, 'vx': 0.0, 'vz': 0.0})])

    def test_single_shoot_periodicity(self, make_periodic_guess,
                                     lyapunov_orbit):
        """
        Free full start state, Periodicity terminal. Adds the x0-dependent
        terminal term: DF = (Jtf @ Phi + Jx0)[:, free_idx], i.e. Phi - I for
        the full-state periodicity residual.
        """
        guess = make_periodic_guess(lyapunov_orbit, n_periods=0.5)
        _check_DF(guess, free_vars='all', constraints=[Periodicity()])

    def test_single_shoot_free_final_time(self, make_periodic_guess,
                                          lyapunov_orbit):
        """
        Free start subset + free final time. Adds the terminal free-time
        column +Jtf @ f(state_tf). For n_seg == 1, free-time index 1 is the
        final time.
        """
        guess = make_periodic_guess(lyapunov_orbit, n_periods=0.5)
        _check_DF(guess, free_vars=['x', 'z', 'vy'],
                  constraints=[TargetState({'y': 0.0, 'vx': 0.0})],
                  free_times=[1])

    # ---- multiple shooting: interior blocks + junction columns ----

    def test_multi_shoot_fixed_times(self, make_multiseg_guess, lyapunov_orbit):
        """
        Three segments, free full start, Periodicity, fixed times. Exercises
        the interior defect rows -- +I at each junction's post columns and
        -Phi at the predecessor columns (including the junction-to-junction
        -Phi_1) -- and the terminal rows referencing the last junction's post
        columns.
        """
        guess = make_multiseg_guess(lyapunov_orbit, n_seg=3, n_periods=1.0)
        _check_DF(guess, free_vars='all', constraints=[Periodicity()])

    def test_multi_shoot_free_times(self, make_multiseg_guess, lyapunov_orbit):
        """
        Three segments with all boundary times free. Exercises every free-time
        column type at once: the interior time t1 (-f(e0) into defect F1,
        +f(e1) into defect F2), the last-junction time t2 (-f(e1) into F2 and
        -Jtf @ f(state_tf) in the terminal rows), and the final time t3
        (+Jtf @ f(state_tf)). A column-wise mismatch still localizes to the
        offending free-variable index.
        """
        guess = make_multiseg_guess(lyapunov_orbit, n_seg=3, n_periods=1.0)
        _check_DF(guess, free_vars='all',
                  constraints=[TargetState({'y': 0.0, 'vx': 0.0, 'vz': 0.0})],
                  free_times=[1, 2, 3])

    # ---- robustness: a generic point, not the structured guess ----

    def test_off_guess_point(self, make_periodic_guess, lyapunov_orbit):
        """
        The single-shooting Periodicity problem, evaluated at a small random
        displacement from the packed guess, so the agreement is not an
        artifact of the special structure at the guess itself.
        """
        guess = make_periodic_guess(lyapunov_orbit, n_periods=0.5)
        _check_DF(guess, free_vars='all', constraints=[Periodicity()],
                  perturb_scale=1e-2, seed=0)

class TestSingleShootingConvergence:
    """End-to-end convergence of a periodic orbit via the mirror half-orbit."""

    def test_lyapunov(self, cr3bp_system, lyapunov_orbit):
        """
        Planar L1 Lyapunov: free the in-plane symmetric components against the
        perpendicular recrossing at the fixed half-period. Clean 2x2 problem.
        """
        guess = _mirror_guess(cr3bp_system, lyapunov_orbit,
                              free=('x', 'vy'), perturb_rel=1e-3)
        corrector = DifferentialCorrector()
        result = corrector.solve(
            guess,
            free_vars=['x', 'vy'],
            constraints=[TargetState({'y': 0.0, 'vx': 0.0})],
        )

        assert result.converged
        assert result.final_residual < corrector.tol
        assert result.iterations <= 10            # quadratic; a few steps
        assert result.trajectory is not None

        # Genuine periodicity: the converged IC returns to itself over the
        # full (fixed) period, not merely a perpendicular recrossing at T/2.
        x0 = result.trajectory.state_at_raw(0.0)
        full = cr3bp_system.propagate(x0, [0.0, lyapunov_orbit.period])
        end = full.state_at_raw(lyapunov_orbit.period)
        np.testing.assert_allclose(end, x0, atol=1e-8)

        # Converged onto the stored orbit (same family member), not a neighbor.
        stored = np.asarray(lyapunov_orbit.initial_state.elements, dtype=float)
        np.testing.assert_allclose(x0[[0, 4]], stored[[0, 4]], atol=1e-6)

    def test_gateway_nrho(self, cr3bp_system, gateway_orbit):
        """
        3D Gateway NRHO: the full mirror formulation, free (x, z, vy) against
        the perpendicular-recrossing TargetState (y, vx, vz). Stiffer, so a
        smaller starting perturbation and looser closure tolerances.
        """
        guess = _mirror_guess(cr3bp_system, gateway_orbit,
                              free=('x', 'z', 'vy'), perturb_rel=1e-3)
        corrector = DifferentialCorrector()
        result = corrector.solve(
            guess,
            free_vars=['x', 'z', 'vy'],
            constraints=[TargetState({'y': 0.0, 'vx': 0.0, 'vz': 0.0})],
        )

        assert result.converged
        assert result.final_residual < corrector.tol
        assert result.iterations <= 15
        assert result.trajectory is not None

        x0 = result.trajectory.state_at_raw(0.0)
        full = cr3bp_system.propagate(x0, [0.0, gateway_orbit.period])
        end = full.state_at_raw(gateway_orbit.period)
        np.testing.assert_allclose(end, x0, atol=1e-7)

        stored = np.asarray(gateway_orbit.initial_state.elements, dtype=float)
        np.testing.assert_allclose(x0[[0, 2, 4]], stored[[0, 2, 4]], atol=1e-6)

class TestMultipleShootingConvergence:
    """Multiple-shooting convergence via the mirror half-orbit: interior
    defects close and the corrector lands on the periodic orbit."""

    def test_lyapunov(self, make_multiseg_guess, cr3bp_system, lyapunov_orbit):
        guess = make_multiseg_guess(lyapunov_orbit, n_seg=3, n_periods=0.5,
                                    symmetric=True)
        assert guess.n_segments == 3
        corrector = DifferentialCorrector()
        result = corrector.solve(
            guess, free_vars=['x', 'vy'],
            constraints=[TargetState({'y': 0.0, 'vx': 0.0})],
        )
        assert result.converged
        assert result.final_residual < corrector.tol
        assert result.iterations <= 15
        assert result.trajectory is not None
        # Reclassification: converged Free junctions become Null in _finalize.
        assert all(isinstance(n, NullJunctionNode)
                   for n in result.trajectory.junction_nodes)
        # Genuine periodicity over the full period.
        x0 = result.trajectory.state_at_raw(0.0)
        full = cr3bp_system.propagate(x0, [0.0, lyapunov_orbit.period])
        end = full.state_at_raw(lyapunov_orbit.period)
        np.testing.assert_allclose(end, x0, atol=1e-8)
        stored = np.asarray(lyapunov_orbit.initial_state.elements, dtype=float)
        np.testing.assert_allclose(x0[[0, 4]], stored[[0, 4]], atol=1e-6)

    def test_gateway_nrho(self, make_multiseg_guess, cr3bp_system,
                          gateway_orbit):
        guess = make_multiseg_guess(gateway_orbit, n_seg=3, n_periods=0.5,
                                    symmetric=True, perturb=1e-3)
        assert guess.n_segments == 3
        corrector = DifferentialCorrector()
        result = corrector.solve(
            guess, free_vars=['x', 'z', 'vy'],
            constraints=[TargetState({'y': 0.0, 'vx': 0.0, 'vz': 0.0})],
        )
        assert result.converged
        assert result.final_residual < corrector.tol
        assert result.iterations <= 20
        assert result.trajectory is not None
        assert all(isinstance(n, NullJunctionNode)
                   for n in result.trajectory.junction_nodes)
        x0 = result.trajectory.state_at_raw(0.0)
        full = cr3bp_system.propagate(x0, [0.0, gateway_orbit.period])
        end = full.state_at_raw(gateway_orbit.period)
        np.testing.assert_allclose(end, x0, atol=1e-7)
        stored = np.asarray(gateway_orbit.initial_state.elements, dtype=float)
        np.testing.assert_allclose(x0[[0, 2, 4]], stored[[0, 2, 4]], atol=1e-6)


class TestFreeTimeConvergence:
    """Free-period convergence via the mirror formulation -- the only real
    exercise of the free-time Jacobian columns (LinearFakeSystem.vector_field
    raises, so they never appear in a fast-tier solve)."""

    def test_lyapunov(self, cr3bp_system, lyapunov_orbit):
        # Fix amplitude (x0); free velocity and half-period.
        guess = _mirror_guess(cr3bp_system, lyapunov_orbit, free=('vy',),
                              perturb_rel=1e-4, t_scale=1.005)
        corrector = DifferentialCorrector()
        result = corrector.solve(
            guess, free_vars=['vy'],
            constraints=[TargetState({'y': 0.0, 'vx': 0.0})],
            free_times=[1],
        )
        assert result.converged
        assert result.final_residual < corrector.tol
        assert result.iterations <= 15
        assert result.trajectory is not None
        # Recovered half-period matches the orbit's.
        t_half = result.trajectory.times[-1]
        assert abs(2.0 * t_half - lyapunov_orbit.period) < 1e-6
        x0 = result.trajectory.state_at_raw(0.0)
        full = cr3bp_system.propagate(x0, [0.0, 2.0 * t_half])
        end = full.state_at_raw(2.0 * t_half)
        np.testing.assert_allclose(end, x0, atol=1e-8)
        stored = np.asarray(lyapunov_orbit.initial_state.elements, dtype=float)
        assert abs(x0[4] - stored[4]) < 1e-6

    def test_gateway_nrho(self, cr3bp_system, gateway_orbit):
        # Fix amplitude (x0); free out-of-plane amplitude, velocity, period.
        guess = _mirror_guess(cr3bp_system, gateway_orbit, free=('z', 'vy'),
                              perturb_rel=1e-3, t_scale=1.01)
        corrector = DifferentialCorrector()
        result = corrector.solve(
            guess, free_vars=['z', 'vy'],
            constraints=[TargetState({'y': 0.0, 'vx': 0.0, 'vz': 0.0})],
            free_times=[1],
        )
        assert result.converged
        assert result.final_residual < corrector.tol
        assert result.iterations <= 20
        assert result.trajectory is not None
        t_half = result.trajectory.times[-1]
        assert abs(2.0 * t_half - gateway_orbit.period) < 1e-6
        x0 = result.trajectory.state_at_raw(0.0)
        full = cr3bp_system.propagate(x0, [0.0, 2.0 * t_half])
        end = full.state_at_raw(2.0 * t_half)
        np.testing.assert_allclose(end, x0, atol=1e-7)
        stored = np.asarray(gateway_orbit.initial_state.elements, dtype=float)
        np.testing.assert_allclose(x0[[2, 4]], stored[[2, 4]], atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
