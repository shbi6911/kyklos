"""
Tier 2 (slow) tests for the System vector-field evaluator.

Exercises System.vector_field / compile_func / is_func_compiled against the
real, compiled Earth-Moon CR3BP dynamics. These are the batched right-hand-
side evaluations the variable-time shooting Jacobian relies on (the free-time
columns of DF), so this module pins down both the lazy-compilation contract
and the numerical correctness of the compiled cfunc.

The whole module is slow-marked: it requires Heyoka and compiles a cfunc (and,
via the session-scoped cr3bp_system fixture, an integrator). Run with
``pytest -m slow`` (or the full suite); excluded by ``pytest -m "not slow"``.

Correctness is checked against an independent analytic CR3BP RHS, _cr3bp_rhs,
which mirrors CR3BPSystem._build_eom after Heyoka differentiates the
pseudo-potential U. That helper is itself verified against a finite-difference
gradient of U; the agreement here is between two exact evaluations of the same
closed form, so the tolerances are tight (machine-precision, loosened only for
floating-point reassociation between Heyoka's compiled expression tree and the
NumPy closed form).
"""

import numpy as np
import pytest

from kyklos import System, EARTH, MOON

# Real dynamics, real compilation: Tier 2.
pytestmark = pytest.mark.slow


# Earth-Moon primary separation [km]; matches defaults.earth_moon_cr3bp and the
# conftest cr3bp_system fixture, so mass_ratio is consistent across the module.
_EM_DISTANCE = 384400.0

# Evaluation states for the correctness checks. Chosen with all six components
# nonzero (so every RHS term -- velocity copies, Coriolis, and both gravity
# gradients -- is exercised) and with comfortable clearance from both primaries
# (primary at x = -mu, secondary at x = 1 - mu), so r1 and r2 stay well away
# from zero and no symbolic singularity is touched.
_STATES = [
    np.array([0.80, 0.10, 0.05, 0.02, 0.30, -0.10]),
    np.array([0.50, -0.20, 0.10, -0.05, 0.10, 0.20]),
    np.array([1.10, 0.15, -0.08, 0.07, -0.12, 0.03]),
    np.array([-0.30, 0.40, 0.20, 0.15, -0.25, 0.05]),
    np.array([0.20, -0.35, -0.15, -0.10, 0.20, -0.18]),
]


def _cr3bp_rhs(state: np.ndarray, mu: float) -> np.ndarray:
    """
    Analytic CR3BP right-hand side in the rotating frame.

    Mirrors CR3BPSystem._build_eom after symbolic differentiation of the
    pseudo-potential U = 0.5 (x^2 + y^2) + (1 - mu)/r1 + mu/r2, with the
    primary at x = -mu and the secondary at x = 1 - mu:

        r1 = sqrt((x + mu)^2 + y^2 + z^2)
        r2 = sqrt((x - 1 + mu)^2 + y^2 + z^2)
        ax = 2 vy + x - (1 - mu)(x + mu)/r1^3 - mu (x - 1 + mu)/r2^3
        ay = -2 vx + y - (1 - mu) y/r1^3 - mu y/r2^3
        az =          - (1 - mu) z/r1^3 - mu z/r2^3

    Parameters
    ----------
    state : np.ndarray
        State [x, y, z, vx, vy, vz], shape (6,).
    mu : float
        Nondimensional mass ratio (system.mass_ratio).

    Returns
    -------
    np.ndarray
        Time derivative f(state), shape (6,).
    """
    x, y, z, vx, vy, vz = state
    r1 = np.sqrt((x + mu) ** 2 + y ** 2 + z ** 2)
    r2 = np.sqrt((x - 1.0 + mu) ** 2 + y ** 2 + z ** 2)
    ax = 2.0 * vy + x - (1.0 - mu) * (x + mu) / r1 ** 3 \
        - mu * (x - 1.0 + mu) / r2 ** 3
    ay = -2.0 * vx + y - (1.0 - mu) * y / r1 ** 3 - mu * y / r2 ** 3
    az = -(1.0 - mu) * z / r1 ** 3 - mu * z / r2 ** 3
    return np.array([vx, vy, vz, ax, ay, az])


@pytest.fixture
def fresh_cr3bp() -> System:
    """
    A freshly built, uncompiled Earth-Moon CR3BP system.

    Function-scoped and uncompiled so the lazy-compilation tests start from a
    pristine ``is_func_compiled == False`` state. The session-scoped
    cr3bp_system fixture cannot serve those tests: once any test triggers cfunc
    compilation on that shared instance, is_func_compiled stays True for the
    remainder of the session. ``compile=False`` also skips the integrator JIT,
    which the vector-field path does not need -- the cfunc is built from the
    cached symbolic EOM independently of the integrator.
    """
    return System('3body', EARTH, MOON, distance=_EM_DISTANCE, compile=False)


# ========================================================================
# Compilation contract
# ========================================================================
class TestVectorFieldCompilation:
    """Lazy compilation, idempotency, and the auto-compile path."""

    def test_not_compiled_before_use(self, fresh_cr3bp):
        """A fresh system has no cfunc until one is requested."""
        assert fresh_cr3bp.is_func_compiled is False

    def test_compile_func_compiles(self, fresh_cr3bp):
        """compile_func() builds the cfunc and flips the flag."""
        assert fresh_cr3bp.is_func_compiled is False
        fresh_cr3bp.compile_func()
        assert fresh_cr3bp.is_func_compiled is True

    def test_compile_func_returns_self(self, fresh_cr3bp):
        """compile_func() returns self for chaining."""
        assert fresh_cr3bp.compile_func() is fresh_cr3bp

    def test_vector_field_auto_compiles(self, fresh_cr3bp):
        """vector_field() compiles the cfunc on first call."""
        assert fresh_cr3bp.is_func_compiled is False
        fresh_cr3bp.vector_field(_STATES[0])
        assert fresh_cr3bp.is_func_compiled is True

    def test_compile_func_idempotent(self, fresh_cr3bp):
        """A second compile_func() is a no-op: the cached cfunc is reused."""
        fresh_cr3bp.compile_func()
        first = fresh_cr3bp._cached_func
        fresh_cr3bp.compile_func()
        # Same object -> the early-return fired; no recompilation happened.
        assert fresh_cr3bp._cached_func is first
        assert fresh_cr3bp.is_func_compiled is True

    def test_results_stable_across_recompile_attempt(self, fresh_cr3bp):
        """Repeated compile_func() leaves evaluation results unchanged."""
        mu = fresh_cr3bp.mass_ratio
        out1 = fresh_cr3bp.compile_func().vector_field(_STATES[0])
        out2 = fresh_cr3bp.compile_func().vector_field(_STATES[0])
        np.testing.assert_array_equal(out1, out2)
        np.testing.assert_allclose(
            out1, _cr3bp_rhs(_STATES[0], mu), rtol=1e-11, atol=1e-12
        )


# ========================================================================
# Single-state correctness
# ========================================================================
class TestVectorFieldValues:
    """A single (6,) state evaluates to the analytic CR3BP RHS."""

    @pytest.mark.parametrize("state", _STATES)
    def test_matches_analytic_rhs(self, cr3bp_system, state):
        """vector_field(state) equals the closed-form RHS at that state."""
        mu = cr3bp_system.mass_ratio
        out = cr3bp_system.vector_field(state)
        expected = _cr3bp_rhs(state, mu)
        np.testing.assert_allclose(out, expected, rtol=1e-11, atol=1e-12)

    def test_velocity_rows_copy_velocity(self, cr3bp_system):
        """The first three RHS components are exactly the velocity inputs."""
        state = _STATES[1]
        out = cr3bp_system.vector_field(state)
        # Kinematic rows dx/dt = v are identities, so this is exact.
        np.testing.assert_array_equal(out[:3], state[3:])

    def test_single_state_shape(self, cr3bp_system):
        """A (6,) input returns a (6,) output."""
        out = cr3bp_system.vector_field(_STATES[0])
        assert out.shape == (6,)

    def test_accepts_list_input(self, cr3bp_system):
        """A plain list is accepted (asarray inside) and matches analytic."""
        mu = cr3bp_system.mass_ratio
        state_list = [0.80, 0.10, 0.05, 0.02, 0.30, -0.10]
        out = cr3bp_system.vector_field(state_list)
        np.testing.assert_allclose(
            out, _cr3bp_rhs(np.asarray(state_list), mu),
            rtol=1e-11, atol=1e-12,
        )


# ========================================================================
# Batched correctness
# ========================================================================
class TestVectorFieldBatched:
    """A (6, k) batch evaluates column-by-column like single states."""

    def test_batch_shape(self, cr3bp_system):
        """A (6, k) input returns a (6, k) output."""
        batch = np.column_stack(_STATES)            # (6, k), C-contiguous
        out = cr3bp_system.vector_field(batch)
        assert out.shape == (6, len(_STATES))

    def test_batch_matches_analytic(self, cr3bp_system):
        """Every column of the batch matches the analytic RHS for that state."""
        mu = cr3bp_system.mass_ratio
        batch = np.column_stack(_STATES)            # (6, k)
        out = cr3bp_system.vector_field(batch)
        for j, state in enumerate(_STATES):
            np.testing.assert_allclose(
                out[:, j], _cr3bp_rhs(state, mu), rtol=1e-11, atol=1e-12
            )

    def test_batch_matches_single_calls(self, cr3bp_system):
        """Batched evaluation agrees with per-state single evaluations."""
        batch = np.column_stack(_STATES)
        out_batch = cr3bp_system.vector_field(batch)
        for j, state in enumerate(_STATES):
            out_single = cr3bp_system.vector_field(state)
            np.testing.assert_allclose(
                out_batch[:, j], out_single, rtol=1e-12, atol=1e-13
            )

    def test_non_contiguous_batch(self, cr3bp_system):
        """
        A non-C-contiguous (6, k) batch still evaluates correctly.
        """
        mu = cr3bp_system.mass_ratio
        rows = np.array(_STATES)                     # (k, 6), C-contiguous
        batch = rows.T                               # (6, k), F-contiguous view
        assert not batch.flags['C_CONTIGUOUS']       # precondition for the test
        out = cr3bp_system.vector_field(batch)
        assert out.shape == (6, len(_STATES))
        for j, state in enumerate(_STATES):
            np.testing.assert_allclose(
                out[:, j], _cr3bp_rhs(state, mu), rtol=1e-11, atol=1e-12
            )

    def test_single_column_batch(self, cr3bp_system):
        """A (6, 1) batch returns (6, 1), matching the (6,) result."""
        mu = cr3bp_system.mass_ratio
        state = _STATES[2]
        batch = state.reshape(6, 1)
        out = cr3bp_system.vector_field(batch)
        assert out.shape == (6, 1)
        np.testing.assert_allclose(
            out[:, 0], _cr3bp_rhs(state, mu), rtol=1e-11, atol=1e-12
        )


# ========================================================================
# Runtime-parameter handling (CR3BP carries none)
# ========================================================================
class TestVectorFieldParams:
    """CR3BP has n_params == 0, so pars is optional and may be omitted."""

    def test_pars_omitted_ok(self, cr3bp_system):
        """Omitting pars on a parameter-free system evaluates normally."""
        out = cr3bp_system.vector_field(_STATES[0])
        assert np.all(np.isfinite(out))

    def test_requires_satellite_false(self, cr3bp_system):
        """A CR3BP system reports no runtime-parameter requirement."""
        assert cr3bp_system.requires_satellite is False
