"""
Test suite for march_family, the pseudo-arclength continuation march.

march_family takes one converged PeriodicOrbit and walks its family at a
fixed arclength step: re-solve the seed once, unclosed, as the bootstrap;
seed the first tangent from the period gradient; then take closed steps,
each seeded with the previous member's trajectory, carrying the tangent by
continuity. It returns every converged member as an OrbitFamily.

Behavior, not internals
-----------------------
Every test here goes through march_family itself and asserts on what it
returns or raises: member counts, recorded periods and states, step sizes,
the direction of travel, closure, warnings and exceptions. None of them
reach into the private helpers (_build_march_setup, _prepare_seed,
_period_gradient, _MarchSetup). That is deliberate. A refactor of the
period-convention handling is planned, and these tests are meant to be
its safety net: run them before and after, and green means nothing
observable changed. A test pinned to a helper would have to be rewritten
alongside the code it is supposed to check.

Two tests are regression guards for bugs found in peer review:

- TestBootstrapMember: member 0's period must equal the seed's. A
  full-period guess handed to the half-arc formulation still converges --
  the end of a full period is also a perpendicular crossing -- and doubles
  every recorded period, which in turn squares every Floquet multiplier
  downstream. Nothing else would flag it.
- TestDirection / TestArclength: a march that re-seeds the tangent from the
  period at every step, instead of carrying it by continuity, reverses at a
  period turning point. Monotone periods and uniform chords along the
  march are what that bug breaks.

Why the whole module is slow
----------------------------
Every test needs a real PeriodicOrbit, and building one compiles the
Earth-Moon System. After that one-time cost the tests are quick: a march of
five small steps is a handful of STM integrations. The argument-validation
tests use a corrector that fails loudly if it is ever asked to solve, so
they also prove the checks run before any integration.

Step size
---------
ds = 2e-3 in X = [x, vy, T_half]. Small on purpose: the OrbitFamily suite
documents that natural-parameter stepping of this family hops branches at
dx = -2e-3. Pseudo-arclength is far more robust than that, but these tests
are about correctness, not about finding the largest step that works.
"""

import warnings

import numpy as np
import pytest

from kyklos.config import config
from kyklos.continuation import march_family
from kyklos.exceptions import ConvergenceError
from kyklos.orbit_family import OrbitFamily
from kyklos.periodic_orbit import PeriodicOrbit
from kyklos.shooter import DifferentialCorrector


pytestmark = pytest.mark.slow


_DS = 2e-3
_N = 5

# State-component indices, for reading columns out of initial_states.
_INDEX = {"x": 0, "y": 1, "z": 2, "vx": 3, "vy": 4, "vz": 5}


# ===========================================================================
# Helpers and doubles
# ===========================================================================

class _NoSolveCorrector:
    """
    A corrector that must never be used.

    Passed to march_family in the validation tests: if an argument check
    were ordered after the bootstrap solve, this raises AssertionError
    instead of the expected exception and the test fails.
    """

    def solve(self, *args, **kwargs):
        raise AssertionError(
            "march_family reached a solve before rejecting its arguments."
        )


def _X_rows(family: OrbitFamily) -> np.ndarray:
    """
    Rebuild each member's X = [free start components, T_half] from the
    public member table and header.

    X is the space ds is measured in (the docstring's "distance along the
    unit tangent in X-space, where X mixes state components and the half
    period"), so this is the documented meaning of ds, not a private
    layout. Column order does not matter for the distances taken from it.
    """
    cols = [_INDEX[name] for name in family.free_vars]
    return np.column_stack(
        [family.initial_states[:, cols], 0.5 * family.periods]
    )


def _march(orbit, **kwargs) -> OrbitFamily:
    """march_family on the Lyapunov recipe with this module's defaults."""
    kwargs.setdefault("ds", _DS)
    kwargs.setdefault("n_steps", _N)
    return march_family(orbit, "lyapunov", **kwargs)


# ===========================================================================
# Fixtures
# ===========================================================================
#
# Module-scoped marches: the tests only read the member table and header.
# The one test that fills the orbit cache (closure) does not affect what any
# other test reads.

@pytest.fixture(scope="module")
def forward(lyapunov_orbit):
    """Five steps toward increasing period from the reference Lyapunov."""
    return _march(lyapunov_orbit)


@pytest.fixture(scope="module")
def backward(lyapunov_orbit):
    """Five steps toward decreasing period from the same seed."""
    return _march(lyapunov_orbit, direction=-1)


@pytest.fixture(scope="module")
def phase_shifted_orbit(cr3bp_system, lyapunov_orbit):
    """
    The reference Lyapunov, re-phased to start a quarter period in.

    A perfectly valid PeriodicOrbit -- it closes -- whose trajectory does not
    start on the x-z plane. The loose tol keeps the closure check from
    being the thing under test.
    """
    T = lyapunov_orbit.period
    quarter = cr3bp_system.propagate(
        lyapunov_orbit.initial_state, [0.0, 0.25 * T]
    )
    start = quarter.state_at_raw(quarter.tf)
    full = cr3bp_system.propagate(start, [0.0, T], with_stm=True)
    return PeriodicOrbit(full, T, tol=1e-6)


@pytest.fixture(scope="module")
def near_crossing_orbit(cr3bp_system, lyapunov_orbit):
    """
    The reference Lyapunov with y0 nudged to 5e-10: on the crossing to well
    within tolerance, but not exactly.

    Built with tol = 1e-5 because the mirror symmetry doubles a start
    offset into the closure residual and the orbit's instability amplifies
    it further; the default 1e-9 would reject the orbit itself.
    """
    T = lyapunov_orbit.period
    ic = np.asarray(lyapunov_orbit.initial_state, dtype=float).copy()
    ic[_INDEX["y"]] = 5e-10
    full = cr3bp_system.propagate(ic, [0.0, T], with_stm=True)
    return PeriodicOrbit(full, T, tol=1e-5)


# ===========================================================================
# The member table
# ===========================================================================

class TestMemberTable:
    """Shape and bookkeeping of a complete march."""

    def test_has_n_steps_plus_one_members(self, forward):
        """n_steps counts steps beyond the bootstrap member."""
        assert forward.n == _N + 1

    def test_step_sizes_are_zero_then_ds(self, forward):
        """
        The bootstrap took no step, so its entry is 0.0; every later member
        was reached by exactly one step of ds.
        """
        np.testing.assert_array_equal(
            forward.step_sizes, [0.0] + [_DS] * _N
        )

    def test_member_indices_count_a_fresh_march(self, forward):
        np.testing.assert_array_equal(forward.member_indices,
                                      np.arange(_N + 1))

    def test_every_member_converged_to_the_corrector_tol(self, forward):
        assert np.all(forward.final_residuals < config.SHOOTER_TOL)

    def test_zero_steps_returns_the_bootstrap_alone(self, lyapunov_orbit):
        family = _march(lyapunov_orbit, n_steps=0)
        assert family.n == 1
        np.testing.assert_array_equal(family.step_sizes, [0.0])


class TestBootstrapMember:
    """
    Member 0 is the re-solved seed. The seed is already converged, so the
    re-solve should barely move it.
    """

    def test_period_matches_the_seed(self, forward, lyapunov_orbit):
        """
        The regression guard for the half/full period convention: a
        full-period guess would converge to a full-period arc and record
        twice the true period here.
        """
        assert forward.periods[0] == pytest.approx(lyapunov_orbit.period,
                                                   rel=1e-9)

    def test_state_matches_the_seed(self, forward, lyapunov_orbit):
        seed = np.asarray(lyapunov_orbit.initial_state, dtype=float)
        np.testing.assert_allclose(forward.initial_states[0], seed,
                                   atol=1e-9)

    def test_both_directions_share_the_bootstrap(self, forward, backward):
        """Direction only signs the first tangent; member 0 is identical."""
        np.testing.assert_array_equal(forward.initial_states[0],
                                      backward.initial_states[0])
        assert forward.periods[0] == backward.periods[0]


class TestFixedComponents:
    """What the corrector holds fixed must stay exactly on the crossing."""

    @pytest.mark.parametrize("name", ["y", "z", "vx", "vz"])
    def test_stay_exactly_zero_for_every_member(self, forward, name):
        """
        The Lyapunov recipe frees only x and vy. Everything else is held at
        the seed's value and carried member to member by the trivial
        predictor, so it must come through bit-exact, not merely small.
        """
        assert np.all(forward.initial_states[:, _INDEX[name]] == 0.0)


# ===========================================================================
# Direction and step geometry
# ===========================================================================

class TestDirection:
    """direction chooses which way along the family the march heads."""

    def test_plus_one_increases_the_period(self, forward):
        assert np.all(np.diff(forward.periods) > 0.0)

    def test_minus_one_decreases_the_period(self, backward):
        assert np.all(np.diff(backward.periods) < 0.0)


class TestArclength:
    """Consecutive members sit ds apart in X-space."""

    @pytest.mark.parametrize("which", ["forward", "backward"])
    def test_consecutive_members_are_ds_apart(self, which, request):
        """
        The closer enforces t_hat . (X_k - X_prev) = ds with |t_hat| = 1, so
        the chord |X_k - X_prev| is at least ds and exceeds it only by the
        family's curvature over one small step. Uniform chords also say the
        march never doubled back.
        """
        family = request.getfixturevalue(which)
        chords = np.linalg.norm(np.diff(_X_rows(family), axis=0), axis=1)

        assert np.all(chords >= _DS * (1.0 - 1e-6))
        np.testing.assert_allclose(chords, _DS, rtol=1e-2)


class TestFamilyContinuity:
    """
    Branch-hop guard. The OrbitFamily suite documents a natural-parameter
    march of this family landing on a different branch -- a Jacobi jump of
    0.115 against 0.001 per step, with perpendicular crossings intact so
    nothing else notices. A family that stays on its branch changes Jacobi
    smoothly and monotonically.
    """

    @pytest.mark.parametrize("which", ["forward", "backward"])
    def test_jacobi_moves_monotonically_against_the_period(self, which,
                                                            request):
        """
        Along the L1 Lyapunov family, larger orbits have longer periods and
        lower Jacobi constants, so the two must move in opposite directions
        at every step.
        """
        family = request.getfixturevalue(which)
        dT = np.diff(family.periods)
        dC = np.diff(family.jacobi_constants)
        assert np.all(dT * dC < 0.0)

    @pytest.mark.parametrize("which", ["forward", "backward"])
    def test_jacobi_changes_evenly(self, which, request):
        """
        Equal arclength steps on a smooth branch give near-equal Jacobi
        changes. A factor of 3 between the largest and smallest step is far
        looser than a smooth march needs and far tighter than a hop.
        """
        family = request.getfixturevalue(which)
        dC = np.abs(np.diff(family.jacobi_constants))
        assert dC.max() < 3.0 * dC.min()


class TestClosure:
    """Every member is a genuine periodic orbit, not just a converged arc."""

    def test_every_member_closes(self, forward):
        """
        The corrector converges each half arc; closure checks the mirrored
        full period. Failures would come back as None, be listed in
        closure_failures, and trigger a summarizing warning. Other warnings
        (the corrector's conditioning notes, say) are not this test's
        business, so only the closure warning is looked for.
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            orbits = forward.to_orbits()

        assert all(orbit is not None for orbit in orbits)
        assert forward.closure_failures == ()
        assert not any("failed periodicity closure" in str(w.message)
                       for w in caught)


# ===========================================================================
# Header and system
# ===========================================================================

class TestHeader:
    """What the family records about how it was made."""

    def test_records_the_recipe_and_scheme(self, forward):
        assert forward.recipe == "lyapunov"
        assert forward.scheme == "pseudo_arclength"

    def test_records_the_march_layout(self, forward):
        """The determinacy triple: (x, vy) free, end time free, no MS."""
        assert forward.free_vars == ("x", "vy")
        assert forward.free_times == (1,)
        assert forward.node_specs is None

    def test_records_the_seed_system_mass_ratio(self, forward,
                                                lyapunov_orbit):
        assert forward.mu == lyapunov_orbit.system.mass_ratio

    def test_the_marching_system_is_attached(self, forward, lyapunov_orbit):
        """
        The live System is handed over, so propagation-backed family methods
        reuse it instead of compiling a second one.
        """
        assert forward.has_system
        assert forward.attach_system() is lyapunov_orbit.system


# ===========================================================================
# Failure paths
# ===========================================================================

class TestFailurePaths:
    """Bootstrap failure raises; a mid-march failure returns what it has."""

    def test_a_bootstrap_that_cannot_converge_raises(self, lyapunov_orbit):
        """
        An unreachable tolerance makes the re-solve of the seed fail. A march
        with no first member has nothing to return, and OrbitFamily cannot
        be empty, so this is an error rather than a warning.
        """
        corrector = DifferentialCorrector(tol=1e-30, max_iter=2)
        with pytest.raises(ConvergenceError, match="bootstrap failed"):
            _march(lyapunov_orbit, corrector=corrector)

    def test_a_failed_step_warns_and_keeps_earlier_members(
        self, lyapunov_orbit
    ):
        """
        max_iter=1 lets the bootstrap through -- the seed is already
        converged, so it passes at iteration 0 -- but a closed step starts
        ds off its closer and cannot converge in a single Newton update. The
        march stops, warns, and returns at least the bootstrap member.
        """
        corrector = DifferentialCorrector(max_iter=1)
        with pytest.warns(UserWarning, match="stopped early"):
            family = _march(lyapunov_orbit, corrector=corrector)

        assert 1 <= family.n < _N + 1
        assert family.step_sizes[0] == 0.0


# ===========================================================================
# The seed
# ===========================================================================

class TestSeedPhase:
    """
    Symmetric recipes need the seed to start on a perpendicular x-z
    crossing: the fixed start components are held at the seed's values for
    the whole family.
    """

    def test_rejects_a_seed_that_starts_off_the_crossing(
        self, phase_shifted_orbit
    ):
        with pytest.raises(ValueError, match="perpendicular x-z crossing"):
            march_family(phase_shifted_orbit, "lyapunov", ds=_DS,
                         corrector=_NoSolveCorrector())  # type: ignore

    def test_snaps_a_seed_that_is_nearly_on_the_crossing(
        self, near_crossing_orbit
    ):
        """
        A seed within the orbit's tolerance is accepted and snapped exactly
        onto the crossing, so the small offset is not frozen into every
        member (where the mirror symmetry would double it in each member's
        closure residual).
        """
        # Guards the fixture itself: the seed really is off by the nudge.
        seed_y = near_crossing_orbit.trajectory.start_node.post_state[1]
        assert seed_y == pytest.approx(5e-10, rel=1e-6)

        family = march_family(near_crossing_orbit, "lyapunov", ds=_DS,
                              n_steps=1)
        assert np.all(family.initial_states[:, _INDEX["y"]] == 0.0)


# ===========================================================================
# Argument validation
# ===========================================================================

class TestArgumentValidation:
    """
    Every argument check runs before the first solve: the corrector here
    raises if it is ever used.
    """

    def _run(self, orbit, **kwargs):
        kwargs.setdefault("ds", _DS)
        return march_family(orbit, kwargs.pop("recipe", "lyapunov"),
                            corrector=_NoSolveCorrector(),  # type: ignore
                            **kwargs)

    def test_rejects_a_targeter(self, lyapunov_orbit):
        with pytest.raises(NotImplementedError, match="targeter"):
            self._run(lyapunov_orbit, targeter=object())

    @pytest.mark.parametrize("ds", [0.0, -1e-3, np.nan, np.inf])
    def test_rejects_a_bad_step(self, lyapunov_orbit, ds):
        with pytest.raises(ValueError, match="ds must be"):
            self._run(lyapunov_orbit, ds=ds)

    def test_ds_is_required(self, lyapunov_orbit):
        with pytest.raises(TypeError):
            march_family(lyapunov_orbit, "lyapunov")      # type: ignore

    @pytest.mark.parametrize("n_steps", [True, 2.0, "3"])
    def test_rejects_a_non_integer_step_count(self, lyapunov_orbit,
                                              n_steps):
        with pytest.raises(TypeError, match="n_steps must be an integer"):
            self._run(lyapunov_orbit, n_steps=n_steps)

    def test_rejects_a_negative_step_count(self, lyapunov_orbit):
        with pytest.raises(ValueError, match="non-negative"):
            self._run(lyapunov_orbit, n_steps=-1)

    def test_rejects_a_bool_direction(self, lyapunov_orbit):
        with pytest.raises(TypeError, match="must be an integer"):
            self._run(lyapunov_orbit, direction=True)

    @pytest.mark.parametrize("direction", [0, 2])
    def test_rejects_a_direction_other_than_plus_or_minus_one(
        self, lyapunov_orbit, direction
    ):
        with pytest.raises(ValueError, match="must be 1 or -1"):
            self._run(lyapunov_orbit, direction=direction)

    def test_rejects_something_other_than_a_periodic_orbit(
        self, lyapunov_orbit
    ):
        """A bare Trajectory is not enough; the march needs a PeriodicOrbit."""
        with pytest.raises(TypeError, match="PeriodicOrbit"):
            self._run(lyapunov_orbit.trajectory)

    def test_rejects_an_unknown_recipe(self, lyapunov_orbit):
        with pytest.raises(ValueError, match="Unknown recipe"):
            self._run(lyapunov_orbit, recipe="no_such_family")

    def test_rejects_an_unknown_scheme(self, lyapunov_orbit):
        with pytest.raises(ValueError, match="Unknown scheme"):
            self._run(lyapunov_orbit, scheme="no_such_scheme")
