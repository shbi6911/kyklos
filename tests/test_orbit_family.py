"""
OrbitFamily: the continuation engine's output container.

An OrbitFamily is an index-aligned table of converged family members plus a
header describing the system and solve layout that produced them. It is
constructed once, from data, and is immutable thereafter; everything derived
-- reconstructed PeriodicOrbits, Floquet multipliers, stability indices -- is
memoized on first request from a single propagation pass.

The tests split along the one line that matters for cost:

  Tier 1 -- construction, validation, properties, view semantics, selection,
  to_frame, repr, and attach_system's hand-in validation. None of these touch
  a System, so they run on synthetic column data in milliseconds. The family's
  constructor is deliberately System-free precisely so this is possible.

  Tier 2 (slow) -- everything propagation-backed. These use five genuinely
  converged L1 Lyapunov orbits and the real Earth-Moon CR3BP system. Real
  propagation turned out to be cheap enough (five full-period STM integrations
  in ~0.02 s) that faking it would have bought nothing but a coupling to a
  module-level import name.

Fixture scoping is load-bearing. The five corrections run once, session-scoped,
because they are the expensive part; but every test gets a freshly constructed
OrbitFamily over that shared data. A session-scoped *family* would carry its
memoized caches from test to test, and the cache-state assertions -- which are
the whole point of several tests here -- would silently stop meaning anything.

The member set is five independently seeded and corrected Lyapunov orbits
rather than a continuation march. Marching this family by stepping x under the
x_amplitude_locked layout works only for very small steps: at dx = -2e-3 the
corrector converges to a different branch entirely (Jacobi discontinuity of
0.115 against 0.001 per step, stability index collapsing from 230 to 1.0), and
at dx = -5e-3 it lands on an unrelated orbit of period 18.8 that passes near
the Earth. Those solutions satisfy the recipe honestly -- perpendicular
crossings at both ends -- so nothing detects the hop. Independent seeding has
no path dependence and therefore no cliff to sit next to.
"""

import dataclasses
import warnings
from typing import Any

import numpy as np
import pytest
import plotly.graph_objects as go
from kyklos.trajectory import _figure_line_positions

import kyklos as ky
from kyklos import temp_config
from kyklos.orbit_family import OrbitFamily, ClosureFailure
from kyklos.system import BodyParams, SysType, _BodyParamsWithND


# ===========================================================================
# Synthetic data (Tier 1): valid columns with no System anywhere
# ===========================================================================

_EARTH = BodyParams(mu=3.986004415e5, radius=6378.1363, name="Earth",
                    source="vallado")
_MOON = BodyParams(mu=4.902799e3, radius=1738.0, name="Moon",
                   source="vallado")
_MU_EM = _MOON.mu / (_EARTH.mu + _MOON.mu)
_DIST = 384400.0

# The stored L1 Lyapunov initial condition. Used as the base row for synthetic
# families so their Jacobi constants are physically meaningful, which lets the
# construction tests cross-check against a published value.
_LYAP_IC = np.array([0.787904556873149, 0.0, 0.0, 0.0, 0.419844679804609, 0.0])
_LYAP_PERIOD = 3.744163087739812

# defaults.lyapunov_orbit() names itself 'L1 Lyapunov (C=3.0355...)'.
_LYAP_C = 3.0355167021


def _columns(n: int = 3) -> dict[str, Any]:
    """Valid, mutually consistent synthetic columns for n members."""
    states = np.tile(_LYAP_IC, (n, 1))
    states[:, 0] += np.linspace(0.0, -0.01, n)
    return dict(
        initial_states=states,
        periods=np.linspace(_LYAP_PERIOD, _LYAP_PERIOD + 0.2, n),
        iterations=np.arange(3, 3 + n),
        final_residuals=np.linspace(1e-13, 9e-12, n),
        step_sizes=np.array([0.0] + [0.01] * (n - 1)),
        primary_body=_EARTH,
        secondary_body=_MOON,
        distance=_DIST,
        mu=_MU_EM,
        recipe="lyapunov",
        scheme="pseudo_arclength",
        free_vars=("x", "vy"),
        free_times=(1,),
    )


@pytest.fixture
def make_family():
    """Factory: a valid synthetic family, with any column overridable."""
    def _make(n=3, **overrides):
        kwargs = _columns(n)
        kwargs.update(overrides)
        return OrbitFamily(**kwargs)
    return _make


@pytest.fixture
def family(make_family):
    """A representative three-member synthetic family."""
    return make_family()


class _FakeCR3BP:
    """
    Duck-typed System stand-in for attach_system's hand-in path.

    attach_system reads exactly these three attributes, and checks base_type
    by .value rather than enum identity, so no real System (and no LLVM
    compile) is needed to exercise the validation.
    """

    base_type = SysType.CR3BP

    def __init__(self, mass_ratio=_MU_EM, L_star=_DIST):
        self.mass_ratio = mass_ratio
        self.L_star = L_star


class _FakeTwoBody:
    base_type = SysType.TWO_BODY
    mass_ratio = _MU_EM
    L_star = _DIST


# ===========================================================================
# Real data (Tier 2): genuinely converged Lyapunov members
# ===========================================================================

@pytest.fixture(scope="session")
def lyapunov_members(cr3bp_system):
    """
    Five real L1 Lyapunov orbits, independently seeded and corrected.

    Session-scoped: the corrections are the expensive part and are pure, so
    they run once. Ordered by decreasing x (increasing amplitude), which is
    the order a march would produce them in.
    """
    orbits = []
    for amplitude in (1e-4, 3e-4, 1e-3, 3e-3):
        seed = cr3bp_system.planar_seeder("L1", amplitude=amplitude)
        guess = ky.CorrectorGuess.from_seeder_result(
            seed, cr3bp_system, "lyapunov")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            orbits.append(ky.correct_as(
                guess, ky.DifferentialCorrector(tol=1e-14)))
    orbits.append(ky.lyapunov_orbit())
    return orbits


@pytest.fixture(scope="session")
def real_columns(lyapunov_members, cr3bp_system):
    """Header and member table for the real family, built once."""
    return dict(
        initial_states=np.array(
            [np.asarray(m.initial_state.elements) for m in lyapunov_members]),
        periods=np.array([m.period for m in lyapunov_members]),
        iterations=np.full(len(lyapunov_members), 4),
        final_residuals=np.full(len(lyapunov_members), 1e-14),
        step_sizes=np.array([0.0] + [0.01] * (len(lyapunov_members) - 1)),
        primary_body=cr3bp_system.primary_body,
        secondary_body=cr3bp_system.secondary_body,
        distance=cr3bp_system.distance,
        mu=cr3bp_system.mass_ratio,
        recipe="lyapunov",
        scheme="pseudo_arclength",
        free_vars=("x", "vy"),
        free_times=(1,),
    )


@pytest.fixture
def make_real_family(real_columns, cr3bp_system):
    """
    Factory: a FRESH real family per call, with the shared System attached.

    Function-scoped by design. The family memoizes, so handing the same
    instance to several tests would let one test's propagation satisfy
    another's cache assertions. Constructing a new one is cheap -- it is
    array copying and a Jacobi loop, no System and no integrator.

    The System is attached rather than built, so no test in this file pays a
    second LLVM compile; the argument-free build path has its own test.
    """
    def _make(**overrides):
        kwargs = dict(real_columns)
        kwargs.update(overrides)
        fam = OrbitFamily(**kwargs)
        fam.attach_system(cr3bp_system)
        return fam
    return _make


# ===========================================================================
# Construction and validation
# ===========================================================================
class TestConstruction:
    """The happy path, and what lands where."""

    def test_builds_and_reports_size(self, family):
        assert family.n == 3
        assert len(family) == 3

    def test_columns_have_expected_dtypes(self, family):
        assert family.initial_states.dtype == float
        assert family.periods.dtype == float
        assert family.jacobi_constants.dtype == float
        assert family.final_residuals.dtype == float
        assert family.step_sizes.dtype == float
        assert np.issubdtype(family.iterations.dtype, np.integer)
        assert np.issubdtype(family.member_indices.dtype, np.integer)

    def test_member_indices_default_to_arange(self, family):
        assert family.member_indices == pytest.approx(np.arange(3))

    def test_member_indices_passed_through_when_given(self, make_family):
        fam = make_family(member_indices=[40, 41, 42])
        assert fam.member_indices == pytest.approx([40, 41, 42])

    def test_caches_start_empty(self, family):
        assert not family.has_system
        assert not family.has_multipliers
        assert not family.has_orbits
        assert family.closure_failures == ()

    def test_keyword_only(self, family):
        # Seven-plus columns is too many to get right positionally; an
        # omission must be a TypeError at the call site, not a silent shift.
        with pytest.raises(TypeError):
            OrbitFamily(np.zeros((1, 6)))          # type: ignore[misc]

    def test_omitted_column_names_itself(self, make_family):
        kwargs = _columns(3)
        del kwargs["step_sizes"]
        with pytest.raises(TypeError, match="step_sizes"):
            OrbitFamily(**kwargs)

    def test_slots_are_live(self, family):
        assert not hasattr(family, "__dict__")
        with pytest.raises(AttributeError):
            family.typo = 1                        # type: ignore[attr-defined]

    def test_does_not_alias_caller_arrays(self, make_family):
        states = np.tile(_LYAP_IC, (2, 1))
        periods = np.array([3.7, 3.8])
        fam = make_family(n=2, initial_states=states, periods=periods,
                          iterations=[4, 3], final_residuals=[1e-12, 2e-12],
                          step_sizes=[0.0, 0.01])
        states[0, 0] = 99.0
        periods[0] = 99.0
        assert fam.initial_states[0, 0] == pytest.approx(_LYAP_IC[0])
        assert fam.periods[0] == pytest.approx(3.7)


class TestJacobiColumn:
    """Computed at construction, from mu alone."""

    def test_matches_the_published_value(self, make_family):
        """
        The strongest single assertion available in this file.

        A one-member family over the stored L1 Lyapunov IC must reproduce the
        Jacobi constant named in defaults.lyapunov_orbit()'s own label. This
        validates the column, the mu plumbing, and the routing through
        OrbitalElements in one shot.
        """
        fam = make_family(n=1, initial_states=_LYAP_IC[None, :],
                          periods=[_LYAP_PERIOD], iterations=[4],
                          final_residuals=[1e-12], step_sizes=[0.0])
        assert fam.jacobi_constants[0] == pytest.approx(_LYAP_C, abs=1e-9)

    def test_uses_the_stored_mass_ratio(self, make_family):
        a = make_family(mu=0.01)
        b = make_family(mu=0.30)
        assert a.jacobi_constants[0] != pytest.approx(b.jacobi_constants[0])

    def test_is_computed_per_member(self, family):
        # Distinct states must give distinct constants; a broadcast bug would
        # give one value repeated.
        assert len(set(family.jacobi_constants.tolist())) == 3


class TestValidationRejections:
    """Every guard, with the message that makes the failure legible."""

    def test_zero_members(self, make_family):
        # The only route to zero members is a failed bootstrap, which must
        # raise in the engine; this guard catches an engine bug.
        with pytest.raises(ValueError, match="at least one member"):
            make_family(n=0, initial_states=np.zeros((0, 6)), periods=[],
                        iterations=[], final_residuals=[], step_sizes=[])

    def test_bare_six_vector_gets_a_reshape_hint(self, make_family):
        with pytest.raises(ValueError, match=r"state\[None, :\]"):
            make_family(n=1, initial_states=_LYAP_IC, periods=[3.7],
                        iterations=[4], final_residuals=[1e-12],
                        step_sizes=[0.0])

    def test_wrong_state_width(self, make_family):
        with pytest.raises(ValueError, match=r"shape \(n, 6\)"):
            make_family(initial_states=np.zeros((3, 5)))

    @pytest.mark.parametrize("column", ["periods", "iterations",
                                        "final_residuals", "step_sizes",
                                        "member_indices"])
    def test_wrong_column_length(self, make_family, column):
        with pytest.raises(ValueError, match="index-aligned"):
            make_family(**{column: [1.0, 2.0]})

    @pytest.mark.parametrize("column", ["periods", "final_residuals",
                                        "step_sizes"])
    def test_non_finite_float_column(self, make_family, column):
        with pytest.raises(ValueError, match="non-finite"):
            make_family(**{column: [np.nan, 1.0, 1.0]})

    def test_non_finite_state(self, make_family):
        states = np.tile(_LYAP_IC, (3, 1))
        states[1, 2] = np.inf
        with pytest.raises(ValueError, match="non-finite"):
            make_family(initial_states=states)

    def test_fractional_iterations_refused_not_truncated(self, make_family):
        # np.asarray([1.9], dtype=int) silently yields 1; refusing is the
        # difference between a corrupted diagnostic and a loud failure.
        with pytest.raises(ValueError, match="whole numbers"):
            make_family(iterations=[4.5, 3, 3])

    def test_non_positive_period(self, make_family):
        with pytest.raises(ValueError, match="strictly positive"):
            make_family(periods=[0.0, 3.7, 3.8])

    @pytest.mark.parametrize("column,value", [
        ("iterations", [-1, 3, 3]),
        ("final_residuals", [-1e-12, 1e-12, 1e-12]),
        ("step_sizes", [-0.01, 0.01, 0.01]),
        ("member_indices", [-1, 1, 2]),
    ])
    def test_negative_where_impossible(self, make_family, column, value):
        with pytest.raises(ValueError, match="non-negative"):
            make_family(**{column: value})

    def test_step_sizes_allows_the_bootstrap_zero(self, make_family):
        # 0.0 is the bootstrap convention, so the check is non-strict.
        assert make_family(step_sizes=[0.0, 0.01, 0.01]).n == 3

    @pytest.mark.parametrize("bad_mu", [0.0, 1.0, 1.5, -0.1, np.nan])
    def test_mu_outside_the_unit_interval(self, make_family, bad_mu):
        with pytest.raises(ValueError, match="mass ratio"):
            make_family(mu=bad_mu)

    @pytest.mark.parametrize("bad", [0.0, -1.0, np.inf])
    def test_non_positive_distance(self, make_family, bad):
        with pytest.raises(ValueError, match="distance must be positive"):
            make_family(distance=bad)

    def test_node_specs_rejected(self, make_family):
        # The member table is structurally single-shooting.
        with pytest.raises(NotImplementedError, match="multiple shooting"):
            make_family(node_specs={1: "x"})

    def test_free_vars_as_bare_string(self, make_family):
        # A string is iterable, so 'xy' would silently record ('x', 'y').
        with pytest.raises(TypeError, match="not a bare string"):
            make_family(free_vars="xy")

    def test_negative_closure_tol(self, make_family):
        with pytest.raises(ValueError, match="closure_tol"):
            make_family(closure_tol=-1e-9)

    def test_bad_body_type(self, make_family):
        with pytest.raises(TypeError, match="primary_body must be"):
            make_family(primary_body="earth")


class TestBodyNormalization:
    """CR3BPSystem hands out a wrapper; the family stores the dataclass."""

    def test_accepts_a_raw_bodyparams(self, family):
        assert isinstance(family.primary_body, BodyParams)

    def test_unwraps_the_nondimensional_wrapper(self, make_family):
        fam = make_family(primary_body=_BodyParamsWithND(_EARTH, _DIST))
        assert isinstance(fam.primary_body, BodyParams)
        assert fam.primary_body == _EARTH

    def test_nesting_the_wrapper_is_refused(self):
        # The wrapper enforces its own flatness, and CR3BPSystem unwraps
        # before storing, so a nested wrapper can never reach OrbitFamily
        # and _normalize_body needs no unnesting loop. The inner argument
        # deliberately violates the annotation, hence the ignore.
        with pytest.raises(TypeError, match="wraps a BodyParams"):
            _BodyParamsWithND(
                _BodyParamsWithND(_EARTH, _DIST),  # type: ignore[arg-type]
                _DIST,
            )

    def test_stored_body_survives_an_asdict_round_trip(self, make_family):
        # The wrapper is not a dataclass instance, so asdict() would raise on
        # it; this is why normalization happens at the boundary rather than
        # being left to the deferred save path.
        fam = make_family(primary_body=_BodyParamsWithND(_EARTH, _DIST))
        assert BodyParams(**dataclasses.asdict(fam.primary_body)) == _EARTH


class TestClosureTolResolution:
    """Resolved from config at construction, then frozen."""

    def test_defaults_from_config(self, family):
        assert family.closure_tol == pytest.approx(ky.config.PERIODICITY_TOL)

    def test_explicit_value_is_kept(self, make_family):
        assert make_family(closure_tol=1e-11).closure_tol == 1e-11

    def test_zero_is_allowed(self, make_family):
        # Mirrors PeriodicOrbit, which rejects negative but permits zero.
        assert make_family(closure_tol=0.0).closure_tol == 0.0

    def test_immune_to_a_later_config_change(self, family):
        """
        The purity property: a family's verdicts cannot shift under it.

        Resolving lazily would mean two families with identical data behaving
        differently depending on when their first propagation happened to
        run, which is the call-history dependence this class avoids.
        """
        before = family.closure_tol
        with temp_config(PERIODICITY_TOL=1e-3):
            assert family.closure_tol == pytest.approx(before)
        assert family.closure_tol == pytest.approx(before)


# ===========================================================================
# Properties
# ===========================================================================
_TABLE = ("initial_states", "periods", "jacobi_constants", "iterations",
          "final_residuals", "step_sizes", "member_indices")


class TestTableProperties:
    """Shapes, and the read-only view contract."""

    @pytest.mark.parametrize("name", _TABLE)
    def test_is_read_only(self, family, name):
        assert not getattr(family, name).flags.writeable

    @pytest.mark.parametrize("name", _TABLE)
    def test_is_a_view_not_a_copy(self, family, name):
        # A copy per access would make the natural
        # `for k in range(n): family.initial_states[k]` idiom O(n^2) in bytes
        # moved. Views cost nothing and share memory.
        assert np.shares_memory(getattr(family, name),
                                getattr(family, "_" + name))

    @pytest.mark.parametrize("name", _TABLE)
    def test_never_hands_out_the_stored_array(self, family, name):
        # Handing out the owner would let a caller re-enable WRITEABLE and
        # mutate family state; a view of a read-only base cannot be unfrozen.
        assert getattr(family, name) is not getattr(family, "_" + name)

    def test_direct_write_is_refused(self, family):
        with pytest.raises(ValueError, match="read-only"):
            family.periods[0] = 99.0

    def test_unfreezing_a_returned_view_is_refused(self, family):
        view = family.periods
        with pytest.raises(ValueError, match="WRITEABLE"):
            view.flags.writeable = True
        assert family.periods[0] == pytest.approx(_LYAP_PERIOD)

    def test_shapes(self, family):
        assert family.initial_states.shape == (3, 6)
        for name in _TABLE[1:]:
            assert getattr(family, name).shape == (3,)

    def test_no_setters(self, family):
        with pytest.raises(AttributeError):
            family.periods = np.zeros(3)           # type: ignore[misc]


class TestHeaderProperties:
    """Immutable objects, returned directly."""

    def test_values(self, family):
        assert family.distance == _DIST
        assert family.mu == pytest.approx(_MU_EM)
        assert family.recipe == "lyapunov"
        assert family.scheme == "pseudo_arclength"
        assert family.free_vars == ("x", "vy")
        assert family.free_times == (1,)
        assert family.node_specs is None

    def test_bodies_are_frozen_dataclasses(self, family):
        with pytest.raises(dataclasses.FrozenInstanceError):
            family.primary_body.mu = 1.0           # type: ignore[misc]

    def test_free_vars_are_coerced_to_tuples(self, make_family):
        fam = make_family(free_vars=["x", "vy"], free_times=[1])
        assert isinstance(fam.free_vars, tuple)
        assert isinstance(fam.free_times, tuple)


# ===========================================================================
# Selection
# ===========================================================================
class TestIntegerIndexing:
    """Type-stable: an integer index returns a one-member OrbitFamily."""

    def test_returns_a_family_not_an_orbit(self, family):
        assert isinstance(family[1], OrbitFamily)
        assert family[1].n == 1

    def test_values_match_the_parent_row(self, family):
        one = family[1]
        assert one.initial_states[0] == pytest.approx(family.initial_states[1])
        assert one.periods[0] == pytest.approx(family.periods[1])
        assert one.jacobi_constants[0] == pytest.approx(
            family.jacobi_constants[1])

    def test_plural_naming_is_the_guardrail(self, family):
        # The payoff of the type-stability decision: a user reaching for the
        # scalar gets an immediate AttributeError naming the attribute.
        with pytest.raises(AttributeError, match="period"):
            family[1].period                       # type: ignore[attr-defined]

    def test_negative_indices(self, family):
        assert family[-1].member_indices[0] == 2

    @pytest.mark.parametrize("bad", [3, 99, -4])
    def test_out_of_range(self, family, bad):
        with pytest.raises(IndexError, match="out of range"):
            family[bad]

    def test_numpy_integer_accepted(self, family):
        assert family[np.int64(1)].n == 1

    def test_bool_rejected(self, family):
        # bool subclasses int, so True would silently mean family[1]; a
        # boolean key reads as mask selection, the deferred feature.
        with pytest.raises(TypeError, match="boolean indexing"):
            family[True]

    @pytest.mark.parametrize("bad", ["x", 1.0, None, np.array([0, 1])])
    def test_other_key_types_rejected(self, family, bad):
        with pytest.raises(TypeError, match="integers or slices"):
            family[bad]


class TestSlicing:
    """Slices return families, and carry their provenance."""

    @pytest.mark.parametrize("key,expected", [
        (slice(None, 2), [0, 1]),
        (slice(1, 3), [1, 2]),
        (slice(None, None, 2), [0, 2]),
        (slice(None, None, -1), [2, 1, 0]),
    ])
    def test_member_indices_pass_through(self, family, key, expected):
        assert family[key].member_indices == pytest.approx(expected)

    def test_subfamily_is_a_full_family(self, family):
        sub = family[:2]
        assert isinstance(sub, OrbitFamily)
        assert sub.recipe == family.recipe
        assert sub.closure_tol == family.closure_tol

    def test_empty_slice_raises(self, family):
        # n >= 1 is a class invariant; slicing must not produce something the
        # constructor would reject.
        with pytest.raises(ValueError, match="at least one member"):
            family[3:3]

    def test_iteration_yields_one_member_families(self, family):
        subs = list(family)
        assert len(subs) == 3
        assert all(isinstance(s, OrbitFamily) and s.n == 1 for s in subs)

    def test_step_sizes_stay_parent_relative(self, family):
        """
        A subfamily's entry 0 is a real step, not the bootstrap 0.0.

        Documented rather than fixed: zeroing it would destroy a true fact.
        member_indices[0] != 0 is the tell that a family is a subfamily, and
        np.cumsum on its step_sizes measures from the parent's preceding
        member rather than from zero.
        """
        sub = family[1:]
        assert sub.step_sizes[0] == pytest.approx(family.step_sizes[1])
        assert sub.step_sizes[0] != 0.0
        assert sub.member_indices[0] != 0


# ===========================================================================
# System attachment
# ===========================================================================
class TestAttachSystemValidation:
    """A handed-in System is validated, never trusted. No compile needed."""

    def test_adopts_a_valid_system(self, family):
        fake = _FakeCR3BP()
        assert family.attach_system(fake) is fake
        assert family.has_system

    def test_is_idempotent(self, family):
        fake = _FakeCR3BP()
        family.attach_system(fake)
        assert family.attach_system() is fake
        assert family.attach_system(fake) is fake

    def test_rejects_a_non_cr3bp_system(self, family):
        with pytest.raises(ValueError, match="requires a CR3BP system"):
            family.attach_system(_FakeTwoBody())

    def test_rejects_a_mismatched_mass_ratio(self, family):
        # The load-bearing check: mass_ratio is the only parameter entering
        # the nondimensional EOM, so a mismatch yields plausible, wrong
        # multipliers with no other symptom.
        with pytest.raises(ValueError, match="mass ratio"):
            family.attach_system(_FakeCR3BP(mass_ratio=0.5))

    def test_rejects_a_mismatched_length_scale(self, family):
        with pytest.raises(ValueError, match="characteristic length"):
            family.attach_system(_FakeCR3BP(L_star=1.0))

    def test_refuses_to_swap(self, family):
        family.attach_system(_FakeCR3BP())
        with pytest.raises(ValueError, match="already attached"):
            family.attach_system(_FakeCR3BP())

    def test_has_system_is_side_effect_free(self, family):
        assert not family.has_system
        assert not family.has_system
        assert family._system is None


@pytest.mark.slow
class TestAttachSystemBuild:
    """The argument-free path, which really does compile."""

    def test_builds_a_working_cr3bp_system(self, make_family):
        fam = make_family()
        built = fam.attach_system()
        assert built.base_type is SysType.CR3BP
        assert built.mass_ratio == pytest.approx(fam.mu)
        assert built.L_star == pytest.approx(fam.distance)

    def test_builds_only_once(self, make_family):
        fam = make_family()
        assert fam.attach_system() is fam.attach_system()


# ===========================================================================
# Propagation-backed products
# ===========================================================================
@pytest.mark.slow
class TestRealFamilyFixture:
    """Guard the fixture's own premises before relying on them."""

    def test_members_are_distinct_and_ordered(self, real_columns):
        x = real_columns["initial_states"][:, 0]
        assert np.all(np.diff(x) < 0)              # decreasing x
        assert len(set(x.tolist())) == len(x)

    def test_members_are_genuinely_converged(self, lyapunov_members):
        for m in lyapunov_members:
            assert m.periodicity_residual < 1e-11

    def test_family_reports_them(self, make_real_family):
        fam = make_real_family()
        assert fam.n == 5
        assert repr(fam).startswith("OrbitFamily('lyapunov', n=5")


@pytest.mark.slow
class TestPropagationProducts:
    """to_orbits, floquet_multipliers, stability_indices over real dynamics."""

    def test_to_orbits_returns_periodic_orbits(self, make_real_family):
        orbits = make_real_family().to_orbits()
        assert isinstance(orbits, tuple)
        assert len(orbits) == 5
        assert all(isinstance(o, ky.PeriodicOrbit) for o in orbits)

    def test_reconstructed_orbits_match_the_table(self, make_real_family):
        fam = make_real_family()
        for k, orbit in enumerate(fam.to_orbits()):
            assert orbit.period == pytest.approx(fam.periods[k])
            assert orbit.jacobi == pytest.approx(fam.jacobi_constants[k],
                                                 rel=1e-12)

    def test_multipliers_shape_and_ordering(self, make_real_family):
        mult = make_real_family().floquet_multipliers()
        assert mult.shape == (5, 6)
        assert mult.dtype == complex
        for row in mult:
            mag = np.abs(row)
            assert np.all(np.diff(mag) <= 1e-12)   # descending magnitude

    def test_multipliers_agree_with_periodic_orbit(self, make_real_family):
        fam = make_real_family()
        mult = fam.floquet_multipliers(retain_orbits=True)
        for k, orbit in enumerate(fam.to_orbits()):
            assert mult[k] == pytest.approx(orbit.floquet_multipliers)

    def test_stability_indices_agree_with_periodic_orbit(self,
                                                         make_real_family):
        fam = make_real_family()
        nu = fam.stability_indices()
        for k, orbit in enumerate(fam.to_orbits()):
            assert nu[k] == pytest.approx(orbit.stability_index, rel=1e-12)

    def test_stability_indices_are_read_only(self, make_real_family):
        assert not make_real_family().stability_indices().flags.writeable

    def test_multipliers_are_a_read_only_view(self, make_real_family):
        fam = make_real_family()
        mult = fam.floquet_multipliers()
        assert not mult.flags.writeable
        assert np.shares_memory(mult, fam._multipliers)


@pytest.mark.slow
class TestCacheStateMachine:
    """
    The guard is on the requested product, not on the multiplier cache.

    This is the subtlest logic in the class and the one place a plausible
    simplification produces a silent wrong answer rather than a crash.
    """

    def test_multipliers_pass_leaves_the_orbit_cache_empty(self,
                                                           make_real_family):
        fam = make_real_family()
        fam.floquet_multipliers()
        assert fam.has_multipliers
        assert not fam.has_orbits

    def test_to_orbits_repropagates_after_a_multipliers_only_pass(
            self, make_real_family):
        # A guard reading "are multipliers cached" would short-circuit here
        # and hand back an empty orbit cache.
        fam = make_real_family()
        fam.floquet_multipliers()
        orbits = fam.to_orbits()
        assert all(o is not None for o in orbits)
        assert fam.has_orbits

    def test_retain_orbits_keeps_them(self, make_real_family):
        fam = make_real_family()
        fam.floquet_multipliers(retain_orbits=True)
        assert fam.has_orbits

    def test_second_call_reuses_the_cache(self, make_real_family):
        fam = make_real_family()
        first = fam.to_orbits()
        assert fam.to_orbits() is first

    def test_clear_orbit_cache_keeps_the_cheap_tier(self, make_real_family):
        fam = make_real_family()
        fam.to_orbits()
        mult_before = np.array(fam.floquet_multipliers())
        fam.clear_orbit_cache()
        assert not fam.has_orbits
        assert fam.has_multipliers
        assert fam.has_system
        assert fam.floquet_multipliers() == pytest.approx(mult_before)

    def test_to_orbits_after_clear_repropagates(self, make_real_family):
        fam = make_real_family()
        fam.to_orbits()
        fam.clear_orbit_cache()
        assert all(o is not None for o in fam.to_orbits())

    def test_stability_indices_populate_the_cache_without_orbits(
            self, make_real_family):
        fam = make_real_family()
        fam.stability_indices()
        assert fam.has_multipliers
        assert not fam.has_orbits

    def test_subfamily_caches_are_independent(self, make_real_family):
        fam = make_real_family()
        fam.to_orbits()
        sub = fam[:2]
        # Dense-and-total: the child owns its own per-member caches.
        assert not sub.has_multipliers
        assert not sub.has_orbits

    def test_subfamily_inherits_the_system(self, make_real_family):
        fam = make_real_family()
        sub = fam[::2]
        assert sub.has_system
        assert sub.attach_system() is fam.attach_system()


@pytest.mark.slow
class TestClosureFailures:
    """
    Forced by an impossible closure_tol, so the failures are real.

    Nothing is staged: the orbits genuinely fail the threshold, PeriodicOrbit
    genuinely raises ClosureError, and the family records what it caught.
    """

    @pytest.fixture
    def failing(self, make_real_family):
        return make_real_family(closure_tol=1e-16)

    def test_all_members_fail(self, failing):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            orbits = failing.to_orbits()
        assert all(o is None for o in orbits)

    def test_index_alignment_is_preserved(self, failing):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            orbits = failing.to_orbits()
        # Filtering failures out would silently break the correspondence with
        # the member table.
        assert len(orbits) == failing.n

    def test_multiplier_rows_are_nan(self, failing):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mult = failing.floquet_multipliers()
        assert np.all(np.isnan(mult))

    def test_stability_indices_are_nan(self, failing):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nu = failing.stability_indices()
        assert np.all(np.isnan(nu))

    def test_failures_are_recorded_with_scalars(self, failing):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            failing.to_orbits()
        failures = failing.closure_failures
        assert len(failures) == failing.n
        assert all(isinstance(f, ClosureFailure) for f in failures)
        assert [f.index for f in failures] == list(range(failing.n))
        assert all(f.threshold == 1e-16 for f in failures)
        assert all(f.residual > f.threshold for f in failures)
        assert all(f.ratio > 1.0 for f in failures)

    def test_warns_once_per_propagating_pass(self, failing):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            failing.to_orbits()
        assert len(caught) == 1
        assert issubclass(caught[0].category, UserWarning)

    def test_cached_call_does_not_rewarn(self, failing):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            failing.to_orbits()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            failing.to_orbits()
        assert len(caught) == 0

    def test_rewarns_after_clear_orbit_cache(self, failing):
        # A repropagating pass is not reading a cache; a caller who paid for
        # n integrations again should see what they found again.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            failing.to_orbits()
        failing.clear_orbit_cache()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            failing.to_orbits()
        assert len(caught) == 1

    def test_warning_carries_the_actionable_numbers(self, failing):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            failing.to_orbits()
        msg = str(caught[0].message)
        assert "5 of 5" in msg
        assert "half arc" in msg
        assert "corrector tolerance" in msg
        assert ".closure_failures" in msg

    def test_closure_failures_empty_before_any_pass(self, make_real_family):
        # Indistinguishable from "everything closed" -- has_multipliers is
        # what tells the two apart.
        fam = make_real_family()
        assert fam.closure_failures == ()
        assert not fam.has_multipliers


# ===========================================================================
# Views
# ===========================================================================
class TestToFrame:
    """A cheap snapshot of data already in hand."""

    def test_columns_and_index(self, family):
        frame = family.to_frame()
        assert list(frame.columns) == [
            "x", "y", "z", "vx", "vy", "vz", "periods", "jacobi_constants",
            "iterations", "final_residuals", "step_sizes"]
        assert frame.index.name == "member"
        assert list(frame.index) == [0, 1, 2]

    def test_values_match_the_table(self, family):
        frame = family.to_frame()
        assert frame["periods"].to_numpy() == pytest.approx(family.periods)
        assert frame["x"].to_numpy() == pytest.approx(
            family.initial_states[:, 0])

    def test_frame_is_mutable_and_independent(self, family):
        frame = family.to_frame()
        frame.loc[0, "periods"] = 99.0
        assert family.periods[0] != pytest.approx(99.0)

    def test_subfamily_frame_reports_march_positions(self, family):
        assert list(family[1:].to_frame().index) == [1, 2]

    def test_does_not_propagate(self, family):
        family.attach_system(_FakeCR3BP())
        family.to_frame()
        assert not family.has_multipliers


class TestSummaryAndRepr:
    """Both must be cheap: no propagation, no compile."""

    def test_summary_does_not_propagate(self, family, capsys):
        family.to_frame()
        family.summary()
        out = capsys.readouterr().out
        assert "not computed" in out
        assert not family.has_multipliers
        assert not family.has_system

    def test_summary_reports_the_header(self, family, capsys):
        family.summary()
        out = capsys.readouterr().out
        assert "lyapunov" in out
        assert "Earth-Moon" in out

    def test_single_member_summary(self, family, capsys):
        family[1].summary()
        assert "single member" in capsys.readouterr().out

    def test_repr_is_compact_and_cheap(self, family):
        text = repr(family)
        assert text.startswith("OrbitFamily('lyapunov', n=3")
        assert "T=[" in text and "C=[" in text
        assert not family.has_multipliers


@pytest.mark.slow
class TestSummaryAfterPropagation:
    def test_reports_stability_once_computed(self, make_real_family, capsys):
        fam = make_real_family()
        fam.to_orbits()
        fam.summary()
        out = capsys.readouterr().out
        assert "stability index" in out
        assert "closure failures: none" in out
        assert "orbits cached   : True" in out


# ===========================================================================
# Package surface
# ===========================================================================
class TestExports:
    def test_orbit_family_is_exported(self):
        assert ky.OrbitFamily is OrbitFamily
        assert "OrbitFamily" in ky.__all__

    def test_closure_failure_is_not_exported(self):
        # Package-internal: reading .residual off an instance does not
        # require the name.
        assert "ClosureFailure" not in ky.__all__


class TestClosureFailureRecord:
    def test_is_frozen(self):
        record = ClosureFailure(index=3, residual=4.7e-8, threshold=1e-9)
        with pytest.raises(dataclasses.FrozenInstanceError):
            record.residual = 1.0                  # type: ignore[misc]

    def test_ratio(self):
        assert ClosureFailure(0, 4.7e-8, 1e-9).ratio == pytest.approx(47.0)

    def test_ratio_with_zero_threshold_is_inf(self):
        assert ClosureFailure(0, 1e-9, 0.0).ratio == np.inf

# ===========================================================================
# Plotting tests
# ===========================================================================

def _family_lines(fig):
    """The member line traces, identified by their legend group."""
    return [t for t in fig.data
            if getattr(t, 'legendgroup', None) == 'family']
 
 
def _colorbar(fig):
    (bar,) = [t for t in fig.data if t.name == 'colorbar']
    return bar
 
 
class TestPlot3dArgumentChecks:
    """Checks that must fail before any propagation."""
 
    def test_bad_color_by_raises_without_propagating(self, family):
        # The fake System has no propagate(). If validation came after
        # to_orbits(), this would die with AttributeError instead -- a
        # tripwire on the ordering, not just on the message.
        family.attach_system(_FakeCR3BP())
        with pytest.raises(ValueError, match="Unknown color_by"):
            family.plot_3d(color_by='amplitude')
        assert not family.has_multipliers
 
 
@pytest.mark.slow
class TestPlot3d:
    """OrbitFamily.plot_3d over the five real Lyapunov members."""
 
    N_POINTS = 50
 
    # ---------- structure ----------
 
    def test_one_line_per_member(self, make_real_family):
        fam = make_real_family()
        lines = _family_lines(fam.plot_3d(n_points=self.N_POINTS))
        assert len(lines) == fam.n
        assert all(len(t.x) == self.N_POINTS for t in lines)  # type: ignore
 
    def test_default_points_come_from_config(self, make_real_family):
        fam = make_real_family()
        with temp_config(DEFAULT_FAMILY_PLOT_POINTS=40):
            lines = _family_lines(fam.plot_3d())
        assert all(len(t.x) == 40 for t in lines)  # type: ignore
 
    def test_members_are_kept_out_of_the_legend(self, make_real_family):
        fig = make_real_family().plot_3d(n_points=self.N_POINTS)
        assert all(t.showlegend is False for t in _family_lines(fig))
 
    def test_default_title_names_recipe_and_count(self, make_real_family):
        fig = make_real_family().plot_3d(n_points=self.N_POINTS)
        assert fig.layout.title.text == "lyapunov family: 5 members"
 
    def test_colorbar_stays_out_of_the_figure_extent(self, make_real_family):
        fam = make_real_family()
        fig = fam.plot_3d(n_points=self.N_POINTS)
        positions = _figure_line_positions(fig)
        assert positions is not None
        assert positions.shape == (fam.n * self.N_POINTS, 3)
 
    # ---------- cache handling ----------
 
    def test_releases_orbits_it_cached(self, make_real_family):
        fam = make_real_family()
        fam.plot_3d(n_points=self.N_POINTS)
        assert not fam.has_orbits
        assert fam.has_multipliers          # the cheap tier survives
 
    def test_retain_orbits_keeps_them(self, make_real_family):
        fam = make_real_family()
        fam.plot_3d(n_points=self.N_POINTS, retain_orbits=True)
        assert fam.has_orbits
 
    def test_leaves_an_existing_cache_alone(self, make_real_family):
        fam = make_real_family()
        orbits = fam.to_orbits()
        fam.plot_3d(n_points=self.N_POINTS)          # retain_orbits=False
        assert fam.to_orbits() is orbits             # same cache, no repropagation
 
    # ---------- coloring ----------
 
    @pytest.mark.parametrize("color_by,title", [
        ('index', 'member'),
        ('jacobi', 'C'),
        ('period', 'T [nd]'),
        ('stability', 'log10(nu)'),
    ])
    def test_every_option_colors_and_labels(self, make_real_family,
                                            color_by, title):
        fig = make_real_family().plot_3d(n_points=self.N_POINTS,
                                         color_by=color_by)
        bar = _colorbar(fig)
        assert bar.marker.showscale is True
        assert bar.marker.colorbar.title.text == title
        lines = _family_lines(fig)
        # Five members span the parameter, so the end colors must differ.
        assert lines[0].line.color != lines[-1].line.color
 
    def test_stability_is_colored_on_a_log_scale(self, make_real_family):
        fam = make_real_family()
        fig = fam.plot_3d(n_points=self.N_POINTS, color_by='stability')
        nu = fam.stability_indices()
        bar = _colorbar(fig)
        assert bar.marker.cmin == pytest.approx(np.log10(np.nanmin(nu)))
        assert bar.marker.cmax == pytest.approx(np.log10(np.nanmax(nu)))
 
    # ---------- hover ----------
 
    def test_hover_reports_march_position_for_a_subfamily(
            self, make_real_family):
        sub = make_real_family()[2:]
        lines = _family_lines(sub.plot_3d(n_points=self.N_POINTS))
        assert lines[0].hovertemplate.startswith("member 2<br>")
 
    def test_hover_carries_the_member_values(self, make_real_family):
        fam = make_real_family()
        lines = _family_lines(fam.plot_3d(n_points=self.N_POINTS))
        text = lines[0].hovertemplate
        assert f"C = {fam.jacobi_constants[0]:.8f}" in text
        assert f"T = {fam.periods[0]:.8f}" in text
        assert "%{x:.6g}" in text           # Plotly placeholder survived
 
    # ---------- bodies and Lagrange points ----------
 
    def test_lagrange_points_on_by_default(self, make_real_family):
        fig = make_real_family().plot_3d(n_points=self.N_POINTS)
        drawn = {t.name for t in fig.data
                 if getattr(t, 'legendgroup', None) == 'lagrange_points'}
        assert 'L1' in drawn
 
    def test_bodies_false_passes_through(self, make_real_family):
        fig = make_real_family().plot_3d(n_points=self.N_POINTS,
                                         bodies=False)
        assert not any(isinstance(t, go.Surface) for t in fig.data)
 
    # ---------- closure failures ----------
 
    def test_all_failed_raises_and_still_releases(self, make_real_family):
        failing = make_real_family(closure_tol=1e-16)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(RuntimeError, match="nothing to plot"):
                failing.plot_3d(n_points=self.N_POINTS)
        # The finally clause ran even though the figure build raised.
        assert not failing.has_orbits
