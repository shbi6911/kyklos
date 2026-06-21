"""
Shared fixtures and test doubles for the shooter / System / Trajectory suite.

The shooter tests run in two tiers:

Tier 1 (fast, deterministic)
    Driven by the fake objects defined here -- FakeTrajectory, FakeSystem,
    and LinearFakeSystem -- which expose only the surface the shooter
    touches and return values the test prescribes. No integrator is
    compiled and no propagation runs, so these tests are fast and their
    assertions are exact rather than approximate.

Tier 2 (slow, numerical)
    Driven by the real Earth-Moon CR3BP system and real propagation, for
    the things only real dynamics can validate (finite-difference checks of
    the Jacobian, end-to-end convergence). These tests are marked
    @pytest.mark.slow; run the fast tier alone with `pytest -m "not slow"`.

Fakes are exposed to tests through factory fixtures (make_fake_trajectory,
make_fake_system, make_linear_system) so test files never import from
conftest directly; they request the factory and build what they need.

The LinearFakeSystem is the keystone of Tier 1: with affine per-segment
dynamics x(t_end) = M x0 + b the shooting problem is linear, every
assembled quantity is hand-computable, and Newton converges in a single
step -- so the real _run loop is exercised deterministically with no
integrator behind it.
"""

import numpy as np
import pytest
from typing import cast, Callable

import kyklos as ky


# ===========================================================================
# Tier 2 -- real CR3BP fixtures (used by @pytest.mark.slow tests)
# ===========================================================================

@pytest.fixture(scope="session")
def cr3bp_system():
    """
    Compiled Earth-Moon CR3BP system, shared read-only across the session.

    The integrator and variational integrator compile lazily on first use
    and are cached on the System instance, so reusing one session-scoped
    system pays the (slow) compilation cost once per test run rather than
    once per test. Tests must treat it as immutable.
    """
    return ky.earth_moon_cr3bp()


@pytest.fixture(scope="session")
def lyapunov_orbit():
    """
    Default L1 Lyapunov periodic orbit -- the cheap reference for routine
    slow tests. Unstable, but well-behaved over a single period and far
    less stiff than an NRHO.
    """
    return ky.LYAPUNOV_ORBIT


@pytest.fixture(scope="session")
def gateway_orbit():
    """Default Gateway NRHO -- the realistic, stiffer reference case."""
    return ky.GATEWAY_ORBIT


@pytest.fixture
def make_periodic_guess(cr3bp_system):
    """
    Factory: propagate a single-segment periodic-orbit initial guess.

    Returns a callable (orbit, n_periods=1.0, with_stm=True) -> Trajectory
    that propagates the orbit's state from t=0 over n_periods of its period.
    Used to build initial guesses for the slow finite-difference and
    single-shooting convergence tests.

    Multi-segment (multiple-shooting) guess construction is deferred to the
    multiple-shooting chunk, where the patch points must be made
    discontinuous enough for Mode-1 propagation to infer FreeJunctionNodes
    rather than NullJunctionNodes.
    """
    def _make(orbit, n_periods=1.0, with_stm=True):
        tf = orbit.period * n_periods
        return cr3bp_system.propagate(orbit.state, [0.0, tf], with_stm=with_stm)
    return _make

@pytest.fixture
def make_multiseg_guess(cr3bp_system):
    """
    Factory: build a deliberately-discontinuous multi-segment initial guess.

    Returns a callable
        (orbit, n_seg=3, n_periods=1.0, perturb=5e-3, symmetric=False, 
        with_stm=False) -> Trajectory
    that samples the orbit at evenly spaced segment boundaries over n_periods
    of its period, then nudges every interior patch point's position off the
    orbit by `perturb` (nondimensional). The position discontinuity forces
    Mode-1 propagation to infer FreeJunctionNodes -- rather than
    NullJunctionNodes for a continuous patch, or ImpulsiveJunctionNodes for a
    velocity-only jump -- so the result is a valid multiple-shooting guess:
    interior junctions carry real, nonzero state defects for the corrector to
    close.

    The start patch point (segment 0 IC) is left on the orbit; only the
    n_seg - 1 interior patch points are perturbed. with_stm defaults to False
    because the guess is only read for structure and packed state values; the
    corrector (and the finite-difference harness) re-propagate with STMs as
    needed. With symmetric=True the start patch point is forced onto the xz-plane 
    (y = vx = vz = 0) for the mirror formulation.

    Used by the slow finite-difference Jacobian checks and the
    multiple-shooting convergence tests.
    """
    def _make(orbit, n_seg=3, n_periods=1.0, perturb=5e-3, symmetric=False,
              with_stm=False):
        if n_seg < 2:
            raise ValueError(
                f"make_multiseg_guess needs n_seg >= 2, got {n_seg}."
            )
        tf = orbit.period * n_periods
        times = np.linspace(0.0, tf, n_seg + 1)

        # Sample the orbit at the start of each segment from a single
        # continuous propagation (states only, so no STM needed).
        base = cr3bp_system.propagate(orbit.state, [0.0, tf], with_stm=False)
        nominal = np.atleast_2d(base.evaluate_raw(times[:-1]))   # (n_seg, 6)

        # Position offset: large enough to clear the continuity tolerance (so
        # junctions are inferred Free), small enough to keep each segment
        # well-behaved. Perturbing z as well takes the interior patch points
        # slightly out of plane, exercising the out-of-plane STM and
        # vector-field columns even for a planar reference orbit.
        offset = np.array([perturb, perturb, perturb, 0.0, 0.0, 0.0])
        ics = []
        for k in range(n_seg):
            x = nominal[k].copy()
            if k == 0 and symmetric:
                x[[1, 3, 5]] = 0.0       # exact xz-plane symmetry on the start
            elif k >= 1:
                x = x + offset
            ics.append(x)
        return cr3bp_system.propagate(ics, times, with_stm=with_stm)
    return _make


# ===========================================================================
# Tier 1 -- fake test doubles
# ===========================================================================

class _FakeBoundary:
    """
    Stand-in for a Start/End boundary node.

    The shooter reads only start_node.post_state and end_node.pre_state, so
    this exposes both, set to the same prescribed state.
    """

    def __init__(self, state):
        state = np.asarray(state, dtype=float)
        self.pre_state = state
        self.post_state = state


class FakeTrajectory:
    """
    Lightweight stand-in for Trajectory exposing only the surface the
    shooter touches.

    Holds prescribed start/end states, interior junction states, boundary
    times, and per-segment STMs -- no continuous output and no integrator.
    Interior junctions are constructed as *real* FreeJunctionNodes so that
    the isinstance checks in _ShootingContext.from_guess (Phase-1
    validation) and DifferentialCorrector._finalize behave exactly as they
    do in production, and so node.state_defect / pre_state / post_state are
    faithful.

    Parameters
    ----------
    start_state, end_state : array-like, shape (6,)
        Boundary states; read as start_node.post_state and
        end_node.pre_state.
    junction_pre, junction_post : sequence of array-like
        Interior junction states (incoming and outgoing), length n_seg - 1.
        Junction i sits at boundary time times[i + 1].
    times : array-like, shape (n_seg + 1,)
        Boundary times.
    stms : sequence of array-like, shape (6, 6) each
        Per-segment terminal STMs; len(stms) defines n_segments.
    system : object, optional
        The (fake) System; read as trajectory.system.
    """

    def __init__(self, *, start_state, end_state,
                 junction_pre=(), junction_post=(),
                 times, stms, system=None):
        stms = [np.asarray(S, dtype=float) for S in stms]
        n_seg = len(stms)
        times = np.asarray(times, dtype=float)

        if times.shape != (n_seg + 1,):
            raise ValueError(
                f"FakeTrajectory: times must have shape ({n_seg + 1},) for "
                f"{n_seg} segment(s), got {times.shape}."
            )
        if len(junction_pre) != n_seg - 1 or len(junction_post) != n_seg - 1:
            raise ValueError(
                f"FakeTrajectory: expected {n_seg - 1} interior junction "
                f"state(s), got {len(junction_pre)} pre / "
                f"{len(junction_post)} post."
            )

        self._n_seg = n_seg
        self._times = times
        self._stms = stms
        self._system = system
        self._start = _FakeBoundary(start_state)
        self._end = _FakeBoundary(end_state)
        self._junctions = [
            ky.FreeJunctionNode(times[i + 1],
                                np.asarray(pre, dtype=float),
                                np.asarray(post, dtype=float))
            for i, (pre, post) in enumerate(zip(junction_pre, junction_post))
        ]

    @property
    def n_segments(self):
        return self._n_seg

    @property
    def start_node(self):
        return self._start

    @property
    def end_node(self):
        return self._end

    @property
    def junction_nodes(self):
        return list(self._junctions)

    @property
    def times(self):
        return self._times.copy()

    @property
    def system(self):
        return self._system

    def segment_terminal_stm(self, i):
        return self._stms[i]

    def with_junction_nodes(self, junction_nodes):
        """
        Mimic Trajectory.with_junction_nodes: return a new FakeTrajectory
        sharing this one's states / times / STMs / system, with the
        interior junctions replaced. Lets _finalize be tested at the fast
        tier -- the returned trajectory's junction_nodes can be inspected
        for the Free->Null conversion.
        """
        new = FakeTrajectory.__new__(FakeTrajectory)
        new._n_seg = self._n_seg
        new._times = self._times
        new._stms = self._stms
        new._system = self._system
        new._start = self._start
        new._end = self._end
        new._junctions = list(junction_nodes)
        return new


class FakeSystem:
    """
    Stand-in for System exposing vector_field with a prescribed linear field
    f(x) = A x (A defaults to the identity), and an optional propagate hook.

    Used by the _assemble_DF structural tests: the test prescribes the STMs
    (via FakeTrajectory) and the vector field (via A) and checks that the
    free-time columns land in the right place with the right sign. The
    matrix multiply works for a single state (6,) and a batch (6, k) alike.
    """

    def __init__(self, field_matrix=None, propagate_fn=None):
        self._A = (np.eye(6) if field_matrix is None
                   else np.asarray(field_matrix, dtype=float))
        self._propagate_fn = propagate_fn

    def vector_field(self, state, pars=None):
        state = np.ascontiguousarray(state, dtype=float)
        return self._A @ state

    def propagate(self, ics, times, with_stm=True):
        if self._propagate_fn is None:
            raise NotImplementedError(
                "FakeSystem.propagate is not configured; pass propagate_fn "
                "or use LinearFakeSystem for iteration tests."
            )
        return self._propagate_fn(ics, times, with_stm)


class LinearFakeSystem:
    """
    Fake System with affine per-segment dynamics x(t_end) = M_i x0 + b_i.

    Because the dynamics are linear, the shooting problem is linear: every
    assembled F and DF is hand-computable, and Newton converges in exactly
    one step. propagate returns a fully self-consistent FakeTrajectory
    (segment endpoints, STMs equal to the M_i, junction defects), so the
    real DifferentialCorrector._run loop -- pack, propagate, assemble,
    least-squares step, convergence test, finalize -- runs deterministically
    with no integrator.

    For a single-segment periodicity solve with free_vars='all', the
    converged start state is x0* = -(M - I)^-1 b, which a test can compute
    in closed form and compare against.

    Parameters
    ----------
    maps : sequence of (M, b)
        One affine map per segment: M is (6, 6), b is (6,). len(maps)
        defines the number of segments.
    """

    def __init__(self, maps):
        self._maps = [(np.asarray(M, dtype=float), np.asarray(b, dtype=float))
                      for (M, b) in maps]

    @property
    def n_segments(self):
        return len(self._maps)

    def vector_field(self, state, pars=None):
        # An affine flow map has no single consistent autonomous field that
        # also reproduces the prescribed M, so refuse rather than return a
        # value that would silently corrupt a free-time Jacobian column.
        # Free-time coverage belongs in the real (slow) finite-difference
        # test, or in a structural test using FakeSystem with a prescribed A.
        raise NotImplementedError(
            "LinearFakeSystem has no consistent autonomous vector field; "
            "use fixed times with this system, or FakeSystem with a "
            "prescribed field for free-time structural tests."
        )

    def propagate(self, ics, times, with_stm=True):
        times = np.asarray(times, dtype=float)
        n = len(self._maps)
        if len(ics) != n:
            raise ValueError(f"expected {n} segment IC(s), got {len(ics)}.")
        if times.shape != (n + 1,):
            raise ValueError(
                f"expected {n + 1} boundary times, got {times.shape}."
            )

        ends = [M @ np.asarray(ics[i], dtype=float) + b
                for i, (M, b) in enumerate(self._maps)]
        junction_pre = ends[:-1]
        junction_post = [np.asarray(ics[i + 1], dtype=float)
                         for i in range(n - 1)]
        stms = [M for (M, _) in self._maps]

        return FakeTrajectory(
            start_state=ics[0],
            end_state=ends[-1],
            junction_pre=junction_pre,
            junction_post=junction_post,
            times=times,
            stms=stms,
            system=self,
        )


# ---------------------------------------------------------------------------
# Factory fixtures exposing the fakes (tests request these, never import)
# ---------------------------------------------------------------------------

@pytest.fixture
def make_fake_trajectory() -> Callable[..., "ky.Trajectory"]:
    def _make(**kwargs) -> "ky.Trajectory":
        return cast(ky.Trajectory, FakeTrajectory(**kwargs))
    return _make


@pytest.fixture
def make_fake_system():
    """Factory: build a FakeSystem with a prescribed field / propagate hook."""
    def _make(field_matrix=None, propagate_fn=None):
        return FakeSystem(field_matrix=field_matrix, propagate_fn=propagate_fn)
    return _make


@pytest.fixture
def make_linear_system():
    """Factory: build a LinearFakeSystem from per-segment (M, b) affine maps."""
    def _make(maps):
        return LinearFakeSystem(maps)
    return _make
