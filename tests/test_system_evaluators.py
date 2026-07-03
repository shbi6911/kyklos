"""
Tier 2 (slow) tests for the System vector-field evaluator.

Exercises System.vector_field / compile_func / is_func_compiled against the
real, compiled Earth-Moon CR3BP dynamics. These are the batched right-hand-
side evaluations the variable-time shooting Jacobian relies on (the free-time
columns of DF), so this module pins down both the lazy-compilation contract
and the numerical correctness of the compiled cfunc.

Companion to the vector-field tests: exercises System.field_jacobian /
compile_jacobian / is_jacobian_compiled against the real, compiled Earth-Moon
CR3BP dynamics. field_jacobian returns df/dstate -- the linearization of the
EOM right-hand side -- which the seeder (Hessian-of-U block), stability
analysis, and the continuation predictor all consume, so this module pins down
the lazy-compilation contract, the numerical correctness of the compiled cfunc,
and the CR3BP block structure.

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

For the Jacobian, correctness is checked against two independent references:

  _cr3bp_jacobian -- an analytic closed-form df/dstate assembled from the
      Hessian of the pseudo-potential U, the kinematic [0 | I] block, and the
      constant Coriolis block. Both this and the compiled cfunc are exact
      evaluations of the same closed form, so the tolerance is tight (loosened
      only for floating-point reassociation between Heyoka's compiled 1/r^5
      expression tree and the NumPy closed form).

  _fd_jacobian -- a central finite difference of System.vector_field. This is
      the independent trust anchor: it does not touch the hand-derived Hessian
      at all, so it validates the compiled Jacobian even if _cr3bp_jacobian
      itself were wrong. Its tolerance is deliberately loose (central-difference
      truncation plus roundoff); the evaluation states are clear of both
      primaries, so the field Jacobian is well-conditioned there and the FD is
      clean -- unlike a monodromy FD over a full unstable period.
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

# ===========================================================================
# Analytic reference: closed-form CR3BP field Jacobian df/dstate
# ===========================================================================
def _cr3bp_jacobian(state: np.ndarray, mu: float) -> np.ndarray:
    """
    Analytic Jacobian of the CR3BP right-hand side in the rotating frame.

    With f = [vx, vy, vz, ax, ay, az] and

        ax = 2 vy + Ux,   ay = -2 vx + Uy,   az = Uz,
        U  = 0.5 (x^2 + y^2) + (1 - mu)/r1 + mu/r2,

    the 6x6 Jacobian has the fixed block structure

        A = [ 0_3    I_3     ]
            [ G      2*Omega  ]

    where the kinematic rows d(position_dot)/dstate give [0 | I] exactly,
    2*Omega = [[0, 2, 0], [-2, 0, 0], [0, 0, 0]] is the constant Coriolis
    block, and G is the symmetric Hessian of U:

        Uxx = 1 + sum_i k_i (3 dxi^2 / ri^5 - 1/ri^3)
        Uyy = 1 + sum_i k_i (3 y^2   / ri^5 - 1/ri^3)
        Uzz =     sum_i k_i (3 z^2   / ri^5 - 1/ri^3)
        Uxy =     sum_i k_i (3 dxi y / ri^5)
        Uxz =     sum_i k_i (3 dxi z / ri^5)
        Uyz =     sum_i k_i (3 y z   / ri^5)

    with primary 1 at x = -mu (k1 = 1 - mu, dx1 = x + mu) and primary 2 at
    x = 1 - mu (k2 = mu, dx2 = x - 1 + mu). The +1 on Uxx and Uyy is the
    centrifugal contribution from 0.5 (x^2 + y^2).

    Parameters
    ----------
    state : np.ndarray
        State [x, y, z, vx, vy, vz], shape (6,).
    mu : float
        Nondimensional mass ratio (system.mass_ratio).

    Returns
    -------
    np.ndarray
        Jacobian df/dstate, shape (6, 6). Entry [i, j] is d(f_i)/d(state_j).
    """
    x, y, z = state[0], state[1], state[2]

    k1 = 1.0 - mu          # primary (larger body) coefficient
    k2 = mu                # secondary coefficient
    dx1 = x + mu           # x - (-mu), offset from primary at x = -mu
    dx2 = x - 1.0 + mu     # x - (1 - mu), offset from secondary at x = 1 - mu

    r1 = np.sqrt(dx1 ** 2 + y ** 2 + z ** 2)
    r2 = np.sqrt(dx2 ** 2 + y ** 2 + z ** 2)
    r1_3, r1_5 = r1 ** 3, r1 ** 5
    r2_3, r2_5 = r2 ** 3, r2 ** 5

    Uxx = 1.0 + k1 * (3.0 * dx1 ** 2 / r1_5 - 1.0 / r1_3) \
              + k2 * (3.0 * dx2 ** 2 / r2_5 - 1.0 / r2_3)
    Uyy = 1.0 + k1 * (3.0 * y ** 2 / r1_5 - 1.0 / r1_3) \
              + k2 * (3.0 * y ** 2 / r2_5 - 1.0 / r2_3)
    Uzz = k1 * (3.0 * z ** 2 / r1_5 - 1.0 / r1_3) \
        + k2 * (3.0 * z ** 2 / r2_5 - 1.0 / r2_3)
    Uxy = k1 * (3.0 * dx1 * y / r1_5) + k2 * (3.0 * dx2 * y / r2_5)
    Uxz = k1 * (3.0 * dx1 * z / r1_5) + k2 * (3.0 * dx2 * z / r2_5)
    Uyz = k1 * (3.0 * y * z / r1_5) + k2 * (3.0 * y * z / r2_5)

    hessian = np.array([
        [Uxx, Uxy, Uxz],
        [Uxy, Uyy, Uyz],
        [Uxz, Uyz, Uzz],
    ])
    coriolis = np.array([
        [0.0, 2.0, 0.0],
        [-2.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ])

    A = np.zeros((6, 6))
    A[:3, 3:] = np.eye(3)     # kinematic rows: d(position_dot)/d(velocity) = I
    A[3:, :3] = hessian       # d(acceleration)/d(position) = Hessian of U
    A[3:, 3:] = coriolis      # d(acceleration)/d(velocity) = Coriolis
    return A


def _fd_jacobian(system, state: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Central finite-difference Jacobian of system.vector_field at one state.

    Independent of _cr3bp_jacobian: differences the compiled RHS directly, so
    it validates field_jacobian without reference to the hand-derived Hessian.
    Column j is d f / d state_j, so entry [i, j] = d(f_i)/d(state_j), matching
    field_jacobian's convention.

    Parameters
    ----------
    system : System
        Compiled CR3BP system.
    state : np.ndarray
        State [x, y, z, vx, vy, vz], shape (6,).
    eps : float, optional
        Central-difference step. Default 1e-6, appropriate for the O(1)
        state components used here and comfortably inside the well-conditioned
        regime away from both primaries.

    Returns
    -------
    np.ndarray
        Approximate Jacobian, shape (6, 6).
    """
    state = np.asarray(state, dtype=float)
    n = state.size
    jac = np.empty((n, n))
    for j in range(n):
        step = np.zeros(n)
        step[j] = eps
        f_plus = np.asarray(system.vector_field(state + step))
        f_minus = np.asarray(system.vector_field(state - step))
        jac[:, j] = (f_plus - f_minus) / (2.0 * eps)
    return jac


# The exact structural blocks, reused by several tests.
_CORIOLIS = np.array([
    [0.0, 2.0, 0.0],
    [-2.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
])


# ===========================================================================
# Compilation contract
# ===========================================================================
class TestFieldJacobianCompilation:
    """Lazy compilation, idempotency, and the auto-compile path."""

    def test_not_compiled_before_use(self, fresh_cr3bp):
        """A fresh system has no Jacobian cfunc until one is requested."""
        assert fresh_cr3bp.is_jacobian_compiled is False

    def test_compile_jacobian_compiles(self, fresh_cr3bp):
        """compile_jacobian() builds the cfunc and flips the flag."""
        assert fresh_cr3bp.is_jacobian_compiled is False
        fresh_cr3bp.compile_jacobian()
        assert fresh_cr3bp.is_jacobian_compiled is True

    def test_compile_jacobian_returns_self(self, fresh_cr3bp):
        """compile_jacobian() returns self for chaining."""
        assert fresh_cr3bp.compile_jacobian() is fresh_cr3bp

    def test_field_jacobian_auto_compiles(self, fresh_cr3bp):
        """field_jacobian() compiles the cfunc on first call."""
        assert fresh_cr3bp.is_jacobian_compiled is False
        fresh_cr3bp.field_jacobian(_STATES[0])
        assert fresh_cr3bp.is_jacobian_compiled is True

    def test_compile_jacobian_idempotent(self, fresh_cr3bp):
        """A second compile_jacobian() is a no-op: the cached cfunc is reused."""
        fresh_cr3bp.compile_jacobian()
        first = fresh_cr3bp._cached_jacobian
        fresh_cr3bp.compile_jacobian()
        # Same object -> the early-return fired; no recompilation happened.
        assert fresh_cr3bp._cached_jacobian is first
        assert fresh_cr3bp.is_jacobian_compiled is True

    def test_jacobian_independent_of_integrator(self, fresh_cr3bp):
        """
        The Jacobian cfunc is built from the cached symbolic EOM, independent
        of the taylor_adaptive integrator: compiling it must not compile the
        integrator, and it must work on a compile=False system.
        """
        assert fresh_cr3bp.is_compiled is False          # integrator not built
        fresh_cr3bp.field_jacobian(_STATES[0])
        assert fresh_cr3bp.is_jacobian_compiled is True
        assert fresh_cr3bp.is_compiled is False          # still not built

    def test_results_stable_across_recompile_attempt(self, fresh_cr3bp):
        """Repeated compile_jacobian() leaves evaluation results unchanged."""
        mu = fresh_cr3bp.mass_ratio
        out1 = fresh_cr3bp.compile_jacobian().field_jacobian(_STATES[0])
        out2 = fresh_cr3bp.compile_jacobian().field_jacobian(_STATES[0])
        np.testing.assert_array_equal(out1, out2)
        np.testing.assert_allclose(
            out1, _cr3bp_jacobian(_STATES[0], mu), rtol=1e-9, atol=1e-11
        )


# ===========================================================================
# Single-state correctness
# ===========================================================================
class TestFieldJacobianValues:
    """A single (6,) state evaluates to the analytic CR3BP Jacobian."""

    @pytest.mark.parametrize("state", _STATES)
    def test_matches_analytic(self, cr3bp_system, state):
        """field_jacobian(state) equals the closed-form Jacobian at that state."""
        mu = cr3bp_system.mass_ratio
        out = cr3bp_system.field_jacobian(state)
        expected = _cr3bp_jacobian(state, mu)
        np.testing.assert_allclose(out, expected, rtol=1e-9, atol=1e-11)

    @pytest.mark.parametrize("state", _STATES)
    def test_matches_finite_difference(self, cr3bp_system, state):
        """
        field_jacobian(state) agrees with a central finite difference of
        vector_field. Independent of the analytic reference, so this catches a
        wrong compiled Jacobian even if _cr3bp_jacobian were mis-derived. Loose
        tolerance: central-difference truncation and roundoff dominate.
        """
        out = cr3bp_system.field_jacobian(state)
        fd = _fd_jacobian(cr3bp_system, state)
        np.testing.assert_allclose(out, fd, rtol=1e-6, atol=1e-7)

    def test_single_state_shape(self, cr3bp_system):
        """A (6,) input returns a (6, 6) matrix, not (6, 6, 1)."""
        out = cr3bp_system.field_jacobian(_STATES[0])
        assert out.shape == (6, 6)

    def test_accepts_list_input(self, cr3bp_system):
        """A plain list is accepted (converted inside) and matches analytic."""
        mu = cr3bp_system.mass_ratio
        state_list = [0.80, 0.10, 0.05, 0.02, 0.30, -0.10]
        out = cr3bp_system.field_jacobian(state_list)
        np.testing.assert_allclose(
            out, _cr3bp_jacobian(np.asarray(state_list), mu),
            rtol=1e-9, atol=1e-11,
        )


# ===========================================================================
# Batched correctness
# ===========================================================================
class TestFieldJacobianBatched:
    """A (6, k) batch evaluates column-by-column into a (6, 6, k) stack."""

    def test_batch_shape(self, cr3bp_system):
        """A (6, k) input returns a (6, 6, k) output."""
        batch = np.column_stack(_STATES)            # (6, k), C-contiguous
        out = cr3bp_system.field_jacobian(batch)
        assert out.shape == (6, 6, len(_STATES))

    def test_batch_matches_analytic(self, cr3bp_system):
        """Each (6, 6) slice matches the analytic Jacobian for that state."""
        mu = cr3bp_system.mass_ratio
        batch = np.column_stack(_STATES)            # (6, k)
        out = cr3bp_system.field_jacobian(batch)
        for j, state in enumerate(_STATES):
            np.testing.assert_allclose(
                out[:, :, j], _cr3bp_jacobian(state, mu),
                rtol=1e-9, atol=1e-11,
            )

    def test_batch_matches_single_calls(self, cr3bp_system):
        """Batched evaluation agrees with per-state single evaluations."""
        batch = np.column_stack(_STATES)
        out_batch = cr3bp_system.field_jacobian(batch)
        for j, state in enumerate(_STATES):
            out_single = cr3bp_system.field_jacobian(state)
            np.testing.assert_allclose(
                out_batch[:, :, j], out_single, rtol=1e-12, atol=1e-13
            )

    def test_single_column_batch(self, cr3bp_system):
        """A (6, 1) batch returns (6, 6, 1), matching the (6,) result."""
        mu = cr3bp_system.mass_ratio
        state = _STATES[2]
        batch = state.reshape(6, 1)
        out = cr3bp_system.field_jacobian(batch)
        assert out.shape == (6, 6, 1)
        np.testing.assert_allclose(
            out[:, :, 0], _cr3bp_jacobian(state, mu), rtol=1e-9, atol=1e-11
        )

    def test_non_contiguous_batch(self, cr3bp_system):
        """A non-C-contiguous (6, k) batch still evaluates correctly."""
        mu = cr3bp_system.mass_ratio
        rows = np.array(_STATES)                     # (k, 6), C-contiguous
        batch = rows.T                               # (6, k), F-contiguous view
        assert not batch.flags['C_CONTIGUOUS']       # precondition for the test
        out = cr3bp_system.field_jacobian(batch)
        assert out.shape == (6, 6, len(_STATES))
        for j, state in enumerate(_STATES):
            np.testing.assert_allclose(
                out[:, :, j], _cr3bp_jacobian(state, mu),
                rtol=1e-9, atol=1e-11,
            )


# ===========================================================================
# CR3BP block structure
# ===========================================================================
class TestCR3BPBlockStructure:
    """
    The CR3BP field Jacobian has a rigid block form,

        A = [ 0_3   I_3     ]
            [ G     2*Omega  ],

    that is independent of the state (only G varies). These are the cheapest
    correctness checks: a transpose, a Coriolis sign flip, or an indexing slip
    in the ravel/reshape breaks one of them immediately.
    """

    def test_kinematic_rows_zero_and_identity(self, cr3bp_system):
        """Top-left 3x3 is exactly zero and top-right 3x3 is exactly I."""
        A = cr3bp_system.field_jacobian(_STATES[0])
        # Structural (d(position_dot)/dstate), so exact -- no tolerance.
        np.testing.assert_array_equal(A[:3, :3], np.zeros((3, 3)))
        np.testing.assert_array_equal(A[:3, 3:], np.eye(3))

    def test_coriolis_block_exact(self, cr3bp_system):
        """Bottom-right 3x3 is exactly the constant Coriolis block 2*Omega."""
        A = cr3bp_system.field_jacobian(_STATES[0])
        # d(acceleration)/d(velocity) is the constant Coriolis term: exact.
        np.testing.assert_array_equal(A[3:, 3:], _CORIOLIS)

    @pytest.mark.parametrize("state", _STATES)
    def test_coriolis_block_state_independent(self, cr3bp_system, state):
        """The Coriolis block is the same 2*Omega at every state."""
        A = cr3bp_system.field_jacobian(state)
        np.testing.assert_array_equal(A[3:, 3:], _CORIOLIS)

    @pytest.mark.parametrize("state", _STATES)
    def test_hessian_block_symmetric(self, cr3bp_system, state):
        """
        The bottom-left 3x3 (Hessian of U) is symmetric. The compiled tree
        derives Uij and Uji along separate paths, so they agree only to
        reassociation rounding, not bit-for-bit -- hence a tight allclose
        rather than assert_array_equal.
        """
        A = cr3bp_system.field_jacobian(state)
        hess = A[3:, :3]
        np.testing.assert_allclose(hess, hess.T, rtol=1e-12, atol=1e-13)

    def test_hessian_block_matches_analytic(self, cr3bp_system):
        """The bottom-left block equals the analytic Hessian of U."""
        mu = cr3bp_system.mass_ratio
        A = cr3bp_system.field_jacobian(_STATES[0])
        expected = _cr3bp_jacobian(_STATES[0], mu)[3:, :3]
        np.testing.assert_allclose(A[3:, :3], expected, rtol=1e-9, atol=1e-11)


# ===========================================================================
# Input validation (the (6,) / (6, k) contract, via _check_field_state)
# ===========================================================================
class TestFieldJacobianValidation:
    """The shared shape guard rejects malformed input with clear messages."""

    def test_transposed_batch_rejected(self, cr3bp_system):
        """A (k, 6) row-major batch (the MATLAB-habit transpose) is rejected."""
        rows = np.array(_STATES)                     # (5, 6): shape[0] != 6
        with pytest.raises(ValueError, match="transposed"):
            cr3bp_system.field_jacobian(rows)

    def test_wrong_1d_size_rejected(self, cr3bp_system):
        """A 1D input that is not length 6 is rejected."""
        with pytest.raises(ValueError, match="Single state"):
            cr3bp_system.field_jacobian(np.zeros(5))

    def test_3d_input_rejected(self, cr3bp_system):
        """A 3D input is rejected."""
        with pytest.raises(ValueError, match="1D"):
            cr3bp_system.field_jacobian(np.zeros((6, 6, 2)))


# ===========================================================================
# Singularity guard (non-finite output at a primary)
# ===========================================================================
class TestFieldJacobianSingularity:
    """
    Evaluated exactly on a primary the Hessian diverges (1/r^3, 1/r^5 -> inf),
    and the finite-output guard converts the resulting non-finite values into a
    clear ValueError rather than propagating inf/nan into a downstream solve.
    """

    def test_singular_at_primary_raises(self, cr3bp_system):
        """field_jacobian at the primary (r1 = 0) raises on non-finite output."""
        mu = cr3bp_system.mass_ratio
        primary = np.array([-mu, 0.0, 0.0, 0.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="non-finite"):
            cr3bp_system.field_jacobian(primary)
    
    def test_exact_secondary_is_finite_but_huge(self, cr3bp_system):
        """
        The exact secondary does NOT trigger the non-finite guard, unlike the
        primary. Root cause: -mu is an exactly representable double (the stored
        parameter), so at the primary dx1 = -mu + mu = 0 exactly and r1 = 0,
        giving 0/0 -> nan. But 1 - mu is not representable, so at the secondary
        dx2 = (x - 1) + mu carries a ~1e-18 cancellation residual; r2 is floored
        at |dx2|, never reaching zero, so 1/r2^3 stays finite (~1e51). No state
        near the secondary can overflow, because r2 >= |dx2|.
        """
        mu = cr3bp_system.mass_ratio
        secondary = np.array([1.0 - mu, 0.0, 0.0, 0.0, 0.0, 0.0])
        out = cr3bp_system.field_jacobian(secondary)
        assert np.all(np.isfinite(out))          # guard does not fire here
        assert np.abs(out[3, 0]) > 1e40          # but the entry is absurdly large


# ===========================================================================
# Runtime-parameter handling (CR3BP carries none)
# ===========================================================================
class TestFieldJacobianParams:
    """CR3BP has n_params == 0, so pars is optional and may be omitted."""

    def test_pars_omitted_ok(self, cr3bp_system):
        """Omitting pars on a parameter-free system evaluates normally."""
        out = cr3bp_system.field_jacobian(_STATES[0])
        assert np.all(np.isfinite(out))
