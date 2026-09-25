"""
Test suite for solve_recipe, the atomic correction operator.

solve_recipe is the one step both public entry points call -- correct_as
for an isolated correction, march_family for every solve in a march. It
takes a built SolveSpec and an already-propagated guess trajectory,
optionally appends one continuation closer, hands everything to the
corrector, and returns the ShooterResult untouched.

Why it is tested now
--------------------
It was deliberately left untested while its closer_factory interface was
unsettled: pinning a design still in flux makes the tests an argument for
keeping it. march_family is now its second consumer and has fixed that
shape, so the contract is pinned here.

The contract is almost entirely about what gets forwarded, which a spy
corrector observes exactly:

  Tier 1 (fast) -- the closer gate (both, neither, or exactly one of ref and
  closer_factory), the kwargs handed to corrector.solve, the guarantee that
  spec.constraints is never mutated, the multiple-shooting guard, and the
  no-wrap, no-raise return. No System, no propagation.

  Tier 2 (slow) -- that the forwarded arguments produce real solves in each
  of the three call patterns (isolated, bootstrap, closed march step), and
  that the default corrector is built when none is passed.
"""

import numpy as np
import pytest

from kyklos.continuation import (
    SolveSpec,
    ContinuationRef,
    solve_recipe,
    _arclength_closer,
)
from kyklos.shooter import (
    DifferentialCorrector,
    ShooterResult,
    TargetState,
    PseudoArclength,
)


# ===========================================================================
# Test doubles and helpers
# ===========================================================================

class _SpyCorrector:
    """
    Records every solve() call and returns a fixed result.

    solve_recipe never inspects the corrector beyond calling .solve, so a
    duck-typed spy observes the full forwarding contract with nothing
    integrated.
    """

    def __init__(self, result=None):
        self.calls = []
        self.result = result if result is not None else ShooterResult(
            trajectory=None, converged=False, iterations=0,
            final_residual=1.0,
        )

    def solve(self, guess_traj, **kwargs):
        self.calls.append((guess_traj, kwargs))
        return self.result


class _SpyFactory:
    """A closer factory that records its (ref, n_X) calls."""

    def __init__(self):
        self.calls = []
        self.built = []

    def __call__(self, ref, n_X):
        self.calls.append((ref, n_X))
        closer = object()          # a fresh, identifiable marker per call
        self.built.append(closer)
        return closer


# A stand-in guess. solve_recipe passes it straight to corrector.solve and
# reads nothing off it, so identity is the whole test.
_GUESS = object()

_TARGET = TargetState({"y": 0.0, "vx": 0.0})


def _spec(free_times=(1,), node_specs=None) -> SolveSpec:
    """The Lyapunov continuation spec: (x, vy) + free end time, n_X = 3."""
    return SolveSpec(free_vars=("x", "vy"), free_times=free_times,
                     constraints=(_TARGET,), node_specs=node_specs)


def _ref(n: int = 3, ds: float = 0.1) -> ContinuationRef:
    t_hat = np.zeros(n)
    t_hat[0] = 1.0
    return ContinuationRef(X_prev=np.zeros(n), t_hat=t_hat, ds=ds)


def _only_call(spy: _SpyCorrector):
    """The single (guess, kwargs) pair a spy saw, asserting there was one."""
    assert len(spy.calls) == 1
    return spy.calls[0]


# ===========================================================================
# Tier 1: forwarding
# ===========================================================================

class TestForwarding:
    """What reaches corrector.solve, and what comes back."""

    def test_passes_the_guess_through_untouched(self):
        spy = _SpyCorrector()
        solve_recipe(_spec(), _GUESS, spy)                 # type: ignore
        guess, _ = _only_call(spy)
        assert guess is _GUESS

    def test_forwards_free_vars_as_a_list_in_order(self):
        spy = _SpyCorrector()
        solve_recipe(_spec(), _GUESS, spy)                 # type: ignore
        _, kwargs = _only_call(spy)
        assert kwargs["free_vars"] == ["x", "vy"]
        assert isinstance(kwargs["free_vars"], list)

    @pytest.mark.parametrize("flag", [True, False])
    def test_forwards_the_continuation_flag(self, flag):
        spy = _SpyCorrector()
        solve_recipe(_spec(), _GUESS, spy,                 # type: ignore
                     continuation=flag)
        _, kwargs = _only_call(spy)
        assert kwargs["continuation"] is flag

    def test_continuation_defaults_to_false(self):
        spy = _SpyCorrector()
        solve_recipe(_spec(), _GUESS, spy)                 # type: ignore
        _, kwargs = _only_call(spy)
        assert kwargs["continuation"] is False

    def test_forwards_free_times_as_a_list_when_present(self):
        spy = _SpyCorrector()
        solve_recipe(_spec(free_times=(1,)), _GUESS, spy)  # type: ignore
        _, kwargs = _only_call(spy)
        assert kwargs["free_times"] == [1]

    def test_omits_free_times_when_empty(self):
        """
        An isolated correction fixes every node time. The key is left out
        entirely rather than passed as [], matching the corrector's own
        all-fixed default instead of relying on it treating [] the same.
        """
        spy = _SpyCorrector()
        solve_recipe(_spec(free_times=()), _GUESS, spy)    # type: ignore
        _, kwargs = _only_call(spy)
        assert "free_times" not in kwargs

    def test_forwards_nothing_else(self):
        """
        Pins the full keyword set, so a new pass-through (and the coupling it
        brings) shows up here as a deliberate test edit.
        """
        spy = _SpyCorrector()
        solve_recipe(_spec(), _GUESS, spy)                 # type: ignore
        _, kwargs = _only_call(spy)
        assert set(kwargs) == {"free_vars", "constraints", "free_times",
                               "continuation"}

    def test_returns_the_corrector_result_unwrapped(self):
        """No PeriodicOrbit wrap: the raw ShooterResult comes straight back."""
        spy = _SpyCorrector()
        out = solve_recipe(_spec(), _GUESS, spy)           # type: ignore
        assert out is spy.result

    def test_does_not_raise_on_non_convergence(self):
        """
        Non-convergence is reported, not raised, so a march can inspect
        .converged and stop cleanly. Raising is correct_as's job.
        """
        failed = ShooterResult(trajectory=None, converged=False,
                               iterations=50, final_residual=1e-3,
                               abort_reason="cond_fail")
        spy = _SpyCorrector(result=failed)
        out = solve_recipe(_spec(), _GUESS, spy)           # type: ignore
        assert out is failed
        assert not out.converged


# ===========================================================================
# Tier 1: the closer gate
# ===========================================================================

class TestCloserGate:
    """The closer is appended iff both ref and closer_factory are given."""

    def test_neither_runs_an_unclosed_solve(self):
        spy = _SpyCorrector()
        solve_recipe(_spec(), _GUESS, spy)                 # type: ignore
        _, kwargs = _only_call(spy)
        assert kwargs["constraints"] == [_TARGET]

    def test_both_appends_exactly_one_closer_last(self):
        spy, factory, ref = _SpyCorrector(), _SpyFactory(), _ref()
        solve_recipe(_spec(), _GUESS, spy,                 # type: ignore
                     closer_factory=factory, ref=ref)
        _, kwargs = _only_call(spy)

        assert len(factory.built) == 1
        assert kwargs["constraints"] == [_TARGET, factory.built[0]]

    def test_the_factory_receives_the_ref_and_the_spec_width(self):
        spy, factory, ref = _SpyCorrector(), _SpyFactory(), _ref()
        spec = _spec()
        solve_recipe(spec, _GUESS, spy,                    # type: ignore
                     closer_factory=factory, ref=ref)

        assert len(factory.calls) == 1
        got_ref, got_n = factory.calls[0]
        assert got_ref is ref
        assert got_n == spec.n_X == 3

    def test_the_real_arclength_closer_lands_last(self):
        """The registered pseudo-arclength factory, not just the spy."""
        spy = _SpyCorrector()
        solve_recipe(_spec(), _GUESS, spy,                 # type: ignore
                     closer_factory=_arclength_closer, ref=_ref())
        _, kwargs = _only_call(spy)
        assert isinstance(kwargs["constraints"][-1], PseudoArclength)
        assert kwargs["constraints"][:-1] == [_TARGET]

    def test_ref_alone_raises_before_solving(self):
        spy = _SpyCorrector()
        with pytest.raises(ValueError, match="must be supplied together"):
            solve_recipe(_spec(), _GUESS, spy,             # type: ignore
                         ref=_ref())
        assert spy.calls == []

    def test_factory_alone_raises_before_solving_or_building(self):
        spy, factory = _SpyCorrector(), _SpyFactory()
        with pytest.raises(ValueError, match="must be supplied together"):
            solve_recipe(_spec(), _GUESS, spy,             # type: ignore
                         closer_factory=factory)
        assert spy.calls == []
        assert factory.calls == []

    def test_ref_does_not_imply_continuation(self):
        """
        ref toggles the closer, continuation toggles the payload -- two
        independent switches. The march's closed steps set both; the
        bootstrap sets only continuation.
        """
        spy = _SpyCorrector()
        solve_recipe(_spec(), _GUESS, spy,                 # type: ignore
                     closer_factory=_SpyFactory(), ref=_ref())
        _, kwargs = _only_call(spy)
        assert kwargs["continuation"] is False


# ===========================================================================
# Tier 1: the spec is loop-invariant
# ===========================================================================

class TestSpecIsNotMutated:
    """
    The spec is built once above a march and reused every step. Each call
    must build a fresh constraint list and leave the spec as it found it.
    """

    def test_constraints_are_unchanged_after_a_closed_solve(self):
        spec = _spec()
        before = spec.constraints
        solve_recipe(spec, _GUESS, _SpyCorrector(),        # type: ignore
                     closer_factory=_SpyFactory(), ref=_ref())

        assert spec.constraints is before
        assert spec.constraints == (_TARGET,)

    def test_each_call_gets_its_own_list(self):
        """
        Two consecutive march steps with the same spec: the second closer
        must not appear in the first call's list, and the lists must be
        distinct objects. A shared, appended-to list would accumulate one
        closer per step -- an overdetermined solve by step two.
        """
        spec, spy, factory = _spec(), _SpyCorrector(), _SpyFactory()
        for ds in (0.1, 0.2):
            solve_recipe(spec, _GUESS, spy,                # type: ignore
                         closer_factory=factory, ref=_ref(ds=ds))

        first = spy.calls[0][1]["constraints"]
        second = spy.calls[1][1]["constraints"]
        assert first is not second
        assert first == [_TARGET, factory.built[0]]
        assert second == [_TARGET, factory.built[1]]

    def test_the_forwarded_list_is_not_the_spec_tuple(self):
        spec, spy = _spec(), _SpyCorrector()
        solve_recipe(spec, _GUESS, spy)                    # type: ignore
        _, kwargs = _only_call(spy)
        assert isinstance(kwargs["constraints"], list)
        assert kwargs["constraints"] is not spec.constraints


# ===========================================================================
# Tier 1: multiple-shooting guard
# ===========================================================================

class TestMultipleShootingGuard:
    """node_specs is the MS extension point; solve_recipe refuses it."""

    def test_rejects_node_specs_before_solving(self):
        spy = _SpyCorrector()
        spec = _spec(node_specs={1: "junction"})
        with pytest.raises(NotImplementedError, match="multiple"):
            solve_recipe(spec, _GUESS, spy)                # type: ignore
        assert spy.calls == []

    def test_rejects_node_specs_even_with_a_closer(self):
        spy, factory = _SpyCorrector(), _SpyFactory()
        spec = _spec(node_specs={1: "junction"})
        with pytest.raises(NotImplementedError, match="multiple"):
            solve_recipe(spec, _GUESS, spy,                # type: ignore
                         closer_factory=factory, ref=_ref())
        assert factory.calls == []


# ===========================================================================
# Tier 2: real solves in each call pattern
# ===========================================================================

@pytest.fixture(scope="module")
def half_arc(cr3bp_system, lyapunov_orbit):
    """
    The reference Lyapunov's half arc: start crossing to the next crossing.
    Unperturbed, so it is already a converged member.
    """
    ic = np.asarray(lyapunov_orbit.initial_state, dtype=float)
    return cr3bp_system.propagate(
        ic, [0.0, 0.5 * lyapunov_orbit.period], with_stm=True
    )


@pytest.mark.slow
class TestRealSolves:
    """
    The three call patterns the two entry points use, each against real
    Earth-Moon dynamics.
    """

    def test_isolated_square_solve_converges(self, half_arc):
        """correct_as's pattern: square spec, no closer, no payload."""
        result = solve_recipe(_spec(free_times=()), half_arc,
                              DifferentialCorrector())
        assert result.converged
        assert result.continuation is None

    def test_default_corrector_is_built(self, half_arc):
        """corrector=None builds a default rather than failing."""
        result = solve_recipe(_spec(free_times=()), half_arc)
        assert result.converged

    def test_bootstrap_pattern_returns_the_payload(self, half_arc):
        """
        march_family's bootstrap: corank-1 spec, no closer, payload on. The
        unclosed DH is (n_X - 1, n_X): two terminal rows, three unknowns.
        """
        result = solve_recipe(_spec(), half_arc, DifferentialCorrector(),
                              continuation=True)
        assert result.converged
        assert result.continuation is not None
        assert result.continuation.DH.shape == (2, 3)

    def test_closed_step_lands_ds_along_the_tangent(self, half_arc):
        """
        march_family's steady-state step: corank-1 spec closed by the real
        arclength closer. The converged member must sit ds along t_hat from
        X_prev -- proof the closer was forwarded and actually bound. The
        direction is pure x, as in the tangent suite's closed solve.
        """
        corrector = DifferentialCorrector()
        boot = solve_recipe(_spec(), half_arc, corrector, continuation=True)
        assert boot.continuation is not None

        X_prev = np.asarray(boot.continuation.X)
        t_hat = np.array([1.0, 0.0, 0.0])
        ds = 1e-4
        ref = ContinuationRef(X_prev=X_prev, t_hat=t_hat, ds=ds)

        step = solve_recipe(_spec(), half_arc, corrector,
                            closer_factory=_arclength_closer, ref=ref,
                            continuation=True)
        assert step.converged
        assert step.continuation is not None
        displacement = t_hat @ (np.asarray(step.continuation.X) - X_prev)
        assert displacement == pytest.approx(ds, rel=1e-6)
