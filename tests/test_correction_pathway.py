"""
Tests for the seed-to-orbit correction pathway:

    planar_seeder() -> SeederResult
                    -> CorrectorGuess.from_seeder_result()
                    -> correct_as()
                    -> PeriodicOrbit

Two tiers, following the project convention:

  Fast tier (default): pure-Python validation and projection logic -- does
      CorrectorGuess reject malformed input, does from_seeder_result project a
      seed to an amplitude-locked guess. These use an UNCOMPILED CR3BP system
      (compile=False), so they pay no JIT cost and run in milliseconds. They
      are NOT marked slow; run them alone with `pytest -m "not slow"`.

  Slow tier (@pytest.mark.slow): the full chain end-to-end on the real,
      compiled Earth-Moon system -- seed a collinear point, correct it, and
      verify the returned PeriodicOrbit (closure, period, pinned amplitude).
      These reuse the session-scoped ``cr3bp_system`` fixture so the (slow)
      compilation is paid once.

Correction-tolerance note
-------------------------
The slow success tests pass a tightened DifferentialCorrector (tol = 1e-11).
At the default corrector tolerance the amplitude-locked correction of a 1e-4
seed closes to ~5e-9, which exceeds PeriodicOrbit's 1e-9 closure threshold and
raises ClosureError; at 1e-11 it closes to ~1e-13, comfortably below.
"""

import types

import numpy as np
import pytest

import kyklos as ky
from kyklos import System, EARTH, MOON
from kyklos.system import SysType
from kyklos.periodic_orbit import PeriodicOrbit
from kyklos.exceptions import ClosureError


# ===========================================================================
# Constants
# ===========================================================================
_EM_DISTANCE = 384400.0          # Earth-Moon separation [km]
_AMPLITUDE = 1e-4                # seed x-amplitude (Earth-Moon default)
_TIGHT_TOL = 1e-11               # corrector tol giving clean closure at 1e-4
_COLLINEAR = ["L1", "L2", "L3"]

# A finite, well-shaped (6,) state for validation tests. Its physical meaning
# is irrelevant -- these tests exercise construction logic, not dynamics.
_VALID_STATE = np.array([0.8369, 0.0, 0.0, 0.0, 0.146, 0.0])
_VALID_PERIOD = 2.7


# ===========================================================================
# Fast-tier fixture: an uncompiled CR3BP system
# ===========================================================================
@pytest.fixture(scope="session")
def uncompiled_cr3bp():
    """
    A CR3BP system built with compile=False.

    CorrectorGuess only reads system.base_type (to confirm CR3BP); it never
    propagates. So the fast validation tests need a real CR3BP system for the
    base_type check but not a compiled one -- compile=False skips the JIT and
    keeps the fast tier genuinely fast.
    """
    return System("3body", EARTH, MOON, distance=_EM_DISTANCE, compile=False)


# ===========================================================================
# FAST: CorrectorGuess validation
# ===========================================================================
class TestCorrectorGuessValidation:
    """Construction-time validation of state, period, system, recipe, layout."""

    @pytest.mark.parametrize("recipe", ["lyapunov", "halo"])
    @pytest.mark.parametrize("layout", ["period_locked", "x_amplitude_locked"])
    def test_valid_construction(self, uncompiled_cr3bp, recipe, layout):
        """A well-formed guess constructs and exposes its fields."""
        g = ky.CorrectorGuess(
            _VALID_STATE, _VALID_PERIOD, uncompiled_cr3bp, recipe, layout
        )
        assert g.recipe == recipe
        assert g.layout == layout
        assert g.period == _VALID_PERIOD

    def test_bad_recipe_rejected(self, uncompiled_cr3bp):
        """An unregistered recipe label is rejected at construction."""
        with pytest.raises(ValueError, match="recipe"):
            ky.CorrectorGuess(
                _VALID_STATE, _VALID_PERIOD, uncompiled_cr3bp,
                "lyapnov", "period_locked",
            )

    def test_bad_layout_rejected(self, uncompiled_cr3bp):
        """An unregistered layout label is rejected at construction."""
        with pytest.raises(ValueError, match="layout"):
            ky.CorrectorGuess(
                _VALID_STATE, _VALID_PERIOD, uncompiled_cr3bp,
                "lyapunov", "amplitude_lockd",
            )

    def test_nonfinite_state_rejected(self, uncompiled_cr3bp):
        """A state with a non-finite component is rejected."""
        bad = np.array([0.8, 0.0, 0.0, 0.0, np.inf, 0.0])
        with pytest.raises(ValueError):
            ky.CorrectorGuess(
                bad, _VALID_PERIOD, uncompiled_cr3bp, "lyapunov", "period_locked"
            )

    def test_wrong_shape_state_rejected(self, uncompiled_cr3bp):
        """A state that is not a (6,) array is rejected (single-state contract)."""
        with pytest.raises(ValueError):
            ky.CorrectorGuess(
                np.zeros(5), _VALID_PERIOD, uncompiled_cr3bp,
                "lyapunov", "period_locked",
            )

    @pytest.mark.parametrize("bad_period", [0.0, -1.0, np.nan, np.inf])
    def test_bad_period_rejected(self, uncompiled_cr3bp, bad_period):
        """Period must be positive and finite."""
        with pytest.raises(ValueError, match="[Pp]eriod"):
            ky.CorrectorGuess(
                _VALID_STATE, bad_period, uncompiled_cr3bp,
                "lyapunov", "period_locked",
            )

    def test_non_cr3bp_system_rejected(self, earth_2bp_system):
        """A non-CR3BP system is rejected. A stand-in with a two-body base_type
        reaches the system check (state and period are valid), which raises."""
        with pytest.raises(ValueError, match="CR3BP"):
            ky.CorrectorGuess(
                _VALID_STATE, _VALID_PERIOD, earth_2bp_system,
                "lyapunov", "period_locked",
            )


# ===========================================================================
# FAST: CorrectorGuess properties and conventions
# ===========================================================================
class TestCorrectorGuessProperties:
    """Immutability and the half- vs full-period convention."""

    def test_state_is_read_only(self, uncompiled_cr3bp):
        """The stored state array is not writeable."""
        g = ky.CorrectorGuess(
            _VALID_STATE, _VALID_PERIOD, uncompiled_cr3bp,
            "lyapunov", "period_locked",
        )
        assert g.state.flags.writeable is False

    def test_half_period_from_full(self, uncompiled_cr3bp):
        """With period_is_half=False, half_period() halves the stored period."""
        g = ky.CorrectorGuess(
            _VALID_STATE, _VALID_PERIOD, uncompiled_cr3bp,
            "lyapunov", "period_locked", period_is_half=False,
        )
        assert g.half_period() == pytest.approx(0.5 * _VALID_PERIOD)

    def test_half_period_from_half(self, uncompiled_cr3bp):
        """With period_is_half=True, half_period() returns the stored period."""
        g = ky.CorrectorGuess(
            _VALID_STATE, _VALID_PERIOD, uncompiled_cr3bp,
            "lyapunov", "period_locked", period_is_half=True,
        )
        assert g.half_period() == pytest.approx(_VALID_PERIOD)


# ===========================================================================
# FAST: from_seeder_result projection
# ===========================================================================
class TestFromSeederResult:
    """Projecting a seeder result to a corrector guess."""

    @staticmethod
    def _fake_seed():
        # from_seeder_result reads only .state and .period.
        return types.SimpleNamespace(state=_VALID_STATE.copy(), period=_VALID_PERIOD)

    def test_defaults_to_amplitude_locked(self, uncompiled_cr3bp):
        """A seed-originated guess defaults to the x_amplitude_locked layout."""
        g = ky.CorrectorGuess.from_seeder_result(
            self._fake_seed(), uncompiled_cr3bp, "lyapunov"
        )
        assert g.layout == "x_amplitude_locked"

    def test_uses_full_period_convention(self, uncompiled_cr3bp):
        """The seeder period is a full period, so period_is_half is False."""
        g = ky.CorrectorGuess.from_seeder_result(
            self._fake_seed(), uncompiled_cr3bp, "lyapunov"
        )
        assert g.period_is_half is False

    def test_projects_state_period_and_recipe(self, uncompiled_cr3bp):
        """State and period pass through; the requested recipe is set."""
        seed = self._fake_seed()
        g = ky.CorrectorGuess.from_seeder_result(seed, uncompiled_cr3bp, "lyapunov")
        assert np.array_equal(g.state, seed.state)
        assert g.period == seed.period
        assert g.recipe == "lyapunov"


# ===========================================================================
# SLOW: the full seed-to-orbit chain
# ===========================================================================
@pytest.fixture(scope="module", params=_COLLINEAR)
def corrected_lyapunov(request, cr3bp_system):
    """
    Run the full chain for one collinear point and cache the result.

    Parametrized over L1/L2/L3; the (slow) correction runs once per point and
    is shared across the assertions below. Returns (point, seed, orbit).
    """
    point = request.param
    seed = cr3bp_system.planar_seeder(point, amplitude=_AMPLITUDE)
    guess = ky.CorrectorGuess.from_seeder_result(seed, cr3bp_system, "lyapunov")
    orbit = ky.correct_as(guess, ky.DifferentialCorrector(tol=_TIGHT_TOL))
    return point, seed, orbit


@pytest.mark.slow
class TestSeedToOrbitChain:
    """End-to-end: a collinear seed corrects into a verified PeriodicOrbit."""

    def test_returns_periodic_orbit(self, corrected_lyapunov):
        """correct_as returns a PeriodicOrbit."""
        _, _, orbit = corrected_lyapunov
        assert isinstance(orbit, PeriodicOrbit)

    def test_closure_well_below_threshold(self, corrected_lyapunov):
        """The verified orbit closes comfortably below the 1e-9 threshold."""
        _, _, orbit = corrected_lyapunov
        assert orbit.periodicity_residual < 1e-9

    def test_period_near_linear_estimate(self, corrected_lyapunov):
        """
        The corrected period sits close to the linear estimate 2*pi/omega_planar
        at this small amplitude. The 5% bound still catches a half/full-period
        convention slip, which would be roughly a factor of two off.
        """
        _, seed, orbit = corrected_lyapunov
        linear_period = 2.0 * np.pi / seed.omega_planar
        assert abs(orbit.period - linear_period) / linear_period < 0.05

    def test_amplitude_pinned_on_seed_side(self, corrected_lyapunov, cr3bp_system):
        """
        x_amplitude_locked pins the seed-side perpendicular crossing at
        x = L_x - amplitude (the minimum-x extremum) to corrector tolerance,
        while the far side is free and lands near +amplitude with a small
        nonlinear bulge.
        """
        point, _, orbit = corrected_lyapunov
        Lx = getattr(cr3bp_system, point)[0]
        x = orbit.trajectory.sample_raw(1000)[:, 0]

        pinned = x.min() - Lx            # seed side: solved for x is held fixed
        free = x.max() - Lx              # far side: found by the corrector
        assert abs(pinned + _AMPLITUDE) < 1e-6            # pinned ~ -amplitude
        assert abs(free - _AMPLITUDE) < 0.10 * _AMPLITUDE  # free ~ +amplitude

    def test_monodromy_available(self, corrected_lyapunov):
        """The PeriodicOrbit exposes a finite 6x6 monodromy."""
        _, _, orbit = corrected_lyapunov
        assert orbit.monodromy.shape == (6, 6)
        assert np.all(np.isfinite(orbit.monodromy))


@pytest.mark.slow
class TestClosureErrorPath:
    """A corrector too loose to close raises ClosureError through the pathway."""

    def test_loose_corrector_raises_closure_error(self, cr3bp_system):
        """
        A default corrector (tol=1e-9) converges its half-arc but
        the mirrored full orbit closes to ~5e-9, just above the 1e-9 threshold,
        so PeriodicOrbit construction raises ClosureError -- and correct_as
        lets it propagate.
        """
        seed = cr3bp_system.planar_seeder("L1", amplitude=_AMPLITUDE)
        guess = ky.CorrectorGuess.from_seeder_result(seed, cr3bp_system, "lyapunov")
        with pytest.raises(ClosureError):
            ky.correct_as(guess)


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::UserWarning")
class TestPeriodLockedLayout:
    """
    The non-default period_locked layout (manual construction) also converges.

    period_locked pins the period at the linear estimate and frees the
    amplitude, so the orbit collapses toward the libration point and the solve
    is stiff (hence the tight (1e-14) corrector tolerance and the ignored 
    conditioning warning). This exercises the manual CorrectorGuess construction path 
    and the period_locked (identity) layout end to end.
    """

    def test_period_locked_converges(self, cr3bp_system):
        seed = cr3bp_system.planar_seeder("L1", amplitude=_AMPLITUDE)
        guess = ky.CorrectorGuess(
            seed.state, seed.period, cr3bp_system, "lyapunov", "period_locked",
        )
        orbit = ky.correct_as(guess, ky.DifferentialCorrector(tol=1e-14))
        assert isinstance(orbit, PeriodicOrbit)
        assert orbit.periodicity_residual < 1e-9
