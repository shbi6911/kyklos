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
from kyklos.defaults import earth, moon
from kyklos.system import System, SysType
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
    return System("3body", earth(), moon(), distance=_EM_DISTANCE, compile=False)


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
# FAST: CorrectorGuess.replace()
# ===========================================================================
class TestCorrectorGuessReplace:
    """
    Functional-update semantics of CorrectorGuess.replace().

    replace() funnels back through __init__, so these tests check three
    things: (1) an override lands and everything else carries over from self
    unchanged, (2) self is never mutated, and (3) a bad override surfaces the
    same validation __init__ would raise on direct construction -- replace()
    adds no separate validation path to drift out of sync.
    """

    @staticmethod
    def _make(uncompiled_cr3bp, *, state=None, period=None, system=None,
              recipe="lyapunov", layout="period_locked", scheme=None,
              period_is_half=False):
        """
        Build a baseline CorrectorGuess, overriding only what a test passes.
        """
        return ky.CorrectorGuess(
            state=_VALID_STATE if state is None else state,
            period=_VALID_PERIOD if period is None else period,
            system=uncompiled_cr3bp if system is None else system,
            recipe=recipe,
            layout=layout,
            scheme=scheme,
            period_is_half=period_is_half,
        )

    def test_no_change_round_trips(self, uncompiled_cr3bp):
        """replace() with no kwargs returns an equal-valued, distinct instance."""
        g = self._make(uncompiled_cr3bp)
        g2 = g.replace()
        assert g2 is not g
        assert np.array_equal(g2.state, g.state)
        assert g2.period == g.period
        assert g2.system is g.system
        assert g2.recipe == g.recipe
        assert g2.layout == g.layout
        assert g2.scheme == g.scheme
        assert g2.period_is_half == g.period_is_half

    @pytest.mark.parametrize("field,value", [
        ("period", 5.5),
        ("recipe", "halo"),
        ("layout", "x_amplitude_locked"),
        ("scheme", "pseudo_arclength"),
        ("period_is_half", True),
    ])
    def test_single_field_override(self, uncompiled_cr3bp, field, value):
        """Overriding one field changes only that field; the rest carry over."""
        g = self._make(uncompiled_cr3bp)
        g2 = g.replace(**{field: value})
        assert getattr(g2, field) == value
        for other in ("period", "recipe", "layout", "scheme", "period_is_half"):
            if other != field:
                assert getattr(g2, other) == getattr(g, other)
        assert np.array_equal(g2.state, g.state)
        assert g2.system is g.system

    def test_state_override(self, uncompiled_cr3bp):
        """A replaced state lands, and the new copy is independently read-only."""
        g = self._make(uncompiled_cr3bp)
        new_state = np.array([0.5, 0.1, 0.0, 0.0, 0.2, 0.0])
        g2 = g.replace(state=new_state)
        assert np.array_equal(g2.state, new_state)
        assert g2.state.flags.writeable is False
        assert np.array_equal(g.state, _VALID_STATE)  # original untouched

    def test_system_override(self, uncompiled_cr3bp):
        """Overriding system with another CR3BP system swaps it, revalidated."""
        other = System("3body", earth(), moon(), distance=_EM_DISTANCE, compile=False)
        g = self._make(uncompiled_cr3bp)
        g2 = g.replace(system=other)
        assert g2.system is other
        assert g.system is uncompiled_cr3bp  # original untouched

    def test_original_is_untouched(self, uncompiled_cr3bp):
        """replace() never mutates self, even when the override is accepted."""
        g = self._make(uncompiled_cr3bp)
        original_recipe = g.recipe
        _ = g.replace(recipe="halo")
        assert g.recipe == original_recipe

    def test_unknown_field_raises_typeerror(self, uncompiled_cr3bp):
        """A keyword that does not name a CorrectorGuess field is rejected."""
        g = self._make(uncompiled_cr3bp)
        with pytest.raises(TypeError, match="unexpected"):
            g.replace(period_guess=5.5)

    def test_invalid_period_override_reraises_init_validation(self, uncompiled_cr3bp):
        """A period override that fails __init__'s check raises the same error."""
        g = self._make(uncompiled_cr3bp)
        with pytest.raises(ValueError, match="[Pp]eriod"):
            g.replace(period=-1.0)

    def test_unregistered_recipe_override_rejected(self, uncompiled_cr3bp):
        """An unregistered recipe override is rejected, same as direct construction."""
        g = self._make(uncompiled_cr3bp)
        with pytest.raises(ValueError, match="recipe"):
            g.replace(recipe="lyapnov")

    def test_non_cr3bp_system_override_rejected(self, uncompiled_cr3bp, earth_2bp_system):
        """A non-CR3BP system override is rejected, same as direct construction."""
        g = self._make(uncompiled_cr3bp)
        with pytest.raises(ValueError, match="CR3BP"):
            g.replace(system=earth_2bp_system)

    def test_coupled_period_and_flag_together(self, uncompiled_cr3bp):
        """
        Regression guard for the period / period_is_half coupling documented on
        replace(): passing both together lands half_period() where the caller
        intended, not where leaving the flag untouched would have.
        """
        g = self._make(uncompiled_cr3bp, period=2.0, period_is_half=False)
        g2 = g.replace(period=1.3, period_is_half=True)
        assert g2.period_is_half is True
        assert g2.half_period() == pytest.approx(1.3)


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
            # cond_fail is raised above the config default (1e12): this path is
            # deliberately stiff and its Jacobian condition number peaks at ~1.6e12
            # mid-solve, which aborts the solve at iteration 16 with the residual
            # still at 3e-11. Allowed past that ceiling it converges in 25
            # iterations to 1.3e-15, closing to 7e-14.
            orbit = ky.correct_as(
                guess, ky.DifferentialCorrector(tol=1e-14, cond_fail=1e14)
            )
            assert isinstance(orbit, PeriodicOrbit)
            assert orbit.periodicity_residual < 1e-9
