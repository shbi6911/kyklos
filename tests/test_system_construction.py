"""
Test suite for System class construction and validation.

Tests cover:
- Valid construction patterns (2-body, 3-body, with perturbations)
- Invalid construction (missing params, incompatible combinations,
  direct subclass instantiation)
- Property access under the TwoBodySystem / CR3BPSystem split
- Instance counting
- Immutability
- String representations

Note on architecture: System('2body', ...) returns a TwoBodySystem instance
and System('3body', ...) returns a CR3BPSystem instance.  Both are subclasses
of System, so isinstance(sys, System) is True for either.  Properties that
only apply to one system type (e.g. L_star, perturbations) exist only on the
relevant subclass -- they are not present as None on the other.
"""

import pytest
import numpy as np
from kyklos import (
    System, TwoBodySystem, CR3BPSystem,
    BodyParams, AtmoParams, EARTH, MOON, MARS, EARTH_STD_ATMO
)


class TestValidConstruction:
    """Test valid System construction patterns."""

    def test_2body_point_mass_minimal(self):
        """Minimal 2-body system with no perturbations."""
        sys = System('2body', EARTH)

        assert isinstance(sys, TwoBodySystem)
        assert isinstance(sys, System)
        assert sys.base_type.value == '2body'
        assert sys.primary_body == EARTH
        assert sys.perturbations == ()
        assert sys.atmosphere is None

    def test_2body_with_j2(self):
        """2-body system with J2 perturbation."""
        sys = System('2body', EARTH, perturbations=('J2',))

        assert isinstance(sys, TwoBodySystem)
        assert sys.perturbations == ('J2',)
        assert sys.primary_body.J2 is not None

    def test_2body_with_drag(self):
        """2-body system with atmospheric drag."""
        sys = System('2body', EARTH,
                     perturbations=('drag',),
                     atmosphere=EARTH_STD_ATMO)

        assert sys.perturbations == ('drag',)
        assert sys.atmosphere == EARTH_STD_ATMO

    def test_2body_with_j2_and_drag(self):
        """2-body system with both J2 and drag."""
        sys = System('2body', EARTH,
                     perturbations=('J2', 'drag'),
                     atmosphere=EARTH_STD_ATMO)

        assert sys.perturbations == ('J2', 'drag')
        assert sys.atmosphere == EARTH_STD_ATMO

    def test_cr3bp_earth_moon(self):
        """CR3BP system with Earth-Moon."""
        distance = 384400.0
        sys = System('3body', EARTH,
                     secondary_body=MOON,
                     distance=distance)

        assert isinstance(sys, CR3BPSystem)
        assert isinstance(sys, System)
        assert sys.base_type.value == '3body'
        assert sys.primary_body == EARTH
        assert sys.secondary_body == MOON
        assert sys.distance == distance

    def test_cr3bp_positional_secondary(self):
        """CR3BP accepts secondary_body as positional argument."""
        sys = System('3body', EARTH, MOON, distance=384400.0)

        assert isinstance(sys, CR3BPSystem)
        assert sys.secondary_body == MOON

    def test_construction_with_compile_false(self):
        """System can be constructed without immediate compilation."""
        sys = System('2body', EARTH, compile=False)

        assert not sys.is_compiled

    def test_construction_with_different_bodies(self):
        """System works with different celestial bodies."""
        sys_moon = System('2body', MOON)
        sys_mars = System('2body', MARS)

        assert sys_moon.primary_body == MOON
        assert sys_mars.primary_body == MARS

    def test_custom_body_params(self):
        """System works with custom BodyParams."""
        custom_body = BodyParams(
            mu=1000.0,
            radius=100.0,
            J2=1e-3,
            rotation_rate=1e-4
        )
        sys = System('2body', custom_body)

        assert sys.primary_body == custom_body

    def test_alternate_string_aliases(self):
        """All documented string aliases for base_type are accepted."""
        for alias in ('2body', '2BODY', 'Two_Body'):
            sys = System(alias, EARTH, compile=False)
            assert isinstance(sys, TwoBodySystem)

        for alias in ('3body', '3BODY', 'Three_Body', 'CR3BP'):
            sys = System(alias, EARTH, MOON, distance=384400.0, compile=False)
            assert isinstance(sys, CR3BPSystem)


class TestInvalidConstruction:
    """Test that invalid construction patterns fail appropriately."""

    # --- 2-body validation ---
    def test_j2_perturbation_without_j2_coefficient(self):
        """J2 perturbation requires J2 coefficient in body."""
        body_no_j2 = BodyParams(mu=1000.0, radius=100.0)

        with pytest.raises(ValueError, match="J2 perturbation requested"):
            System('2body', body_no_j2, perturbations=('J2',))

    def test_drag_without_atmosphere(self):
        """Drag perturbation requires atmosphere."""
        with pytest.raises(ValueError, match="drag perturbation requested"):
            System('2body', EARTH, perturbations=('drag',))

    def test_drag_without_rotation_rate(self):
        """Drag perturbation requires rotation rate."""
        body_no_rotation = BodyParams(mu=1000.0, radius=100.0)
        atmo = AtmoParams(rho0=1.0, H=8500.0, r0=6378137.0)

        with pytest.raises(ValueError, match="rotation_rate"):
            System('2body', body_no_rotation,
                   perturbations=('drag',),
                   atmosphere=atmo)

    def test_unknown_perturbation(self):
        """Unknown perturbation name raises error."""
        with pytest.raises(ValueError, match="Unknown perturbation"):
            System('2body', EARTH, perturbations=('gravity_gradient',))

    def test_duplicate_perturbations(self):
        """Duplicate perturbations not allowed."""
        with pytest.raises(ValueError, match="Duplicate perturbations"):
            System('2body', EARTH, perturbations=('J2', 'J2'))

    def test_2body_rejects_secondary_body(self):
        """TwoBodySystem raises TypeError if secondary_body is passed."""
        with pytest.raises(TypeError, match="does not accept secondary_body"):
            System('2body', EARTH, secondary_body=MOON)

    def test_2body_rejects_distance(self):
        """TwoBodySystem raises TypeError if distance is passed."""
        with pytest.raises(TypeError, match="does not accept distance"):
            System('2body', EARTH, distance=384400.0)

    # --- CR3BP validation ---
    def test_cr3bp_without_secondary(self):
        """CR3BP requires secondary body."""
        with pytest.raises(ValueError, match="requires secondary_body"):
            System('3body', EARTH)

    def test_cr3bp_without_distance(self):
        """CR3BP requires distance between bodies."""
        with pytest.raises(ValueError, match="requires distance"):
            System('3body', EARTH, secondary_body=MOON)

    def test_cr3bp_with_negative_distance(self):
        """CR3BP distance must be positive."""
        with pytest.raises(ValueError, match="Distance must be positive"):
            System('3body', EARTH,
                   secondary_body=MOON,
                   distance=-100.0)

    def test_cr3bp_rejects_perturbations(self):
        """CR3BP raises ValueError for perturbations argument."""
        with pytest.raises(ValueError,
                           match="do not currently support perturbations"):
            System('3body', EARTH,
                   secondary_body=MOON,
                   distance=384400.0,
                   perturbations=('J2',))

    def test_cr3bp_rejects_atmosphere(self):
        """CR3BP raises ValueError for atmosphere argument."""
        with pytest.raises(ValueError, match="do not support atmosphere"):
            System('3body', EARTH,
                   secondary_body=MOON,
                   distance=384400.0,
                   atmosphere=EARTH_STD_ATMO)

    def test_invalid_base_type(self):
        """Invalid base_type string raises error."""
        with pytest.raises(ValueError, match="Unknown base type"):
            System('4body', EARTH)

    # --- Direct subclass instantiation ---
    def test_cannot_instantiate_twobodysystem_directly(self):
        """Direct TwoBodySystem instantiation raises TypeError."""
        with pytest.raises(TypeError, match="cannot be instantiated directly"):
            TwoBodySystem('2body', EARTH)

    def test_cannot_instantiate_cr3bpsystem_directly(self):
        """Direct CR3BPSystem instantiation raises TypeError."""
        with pytest.raises(TypeError, match="cannot be instantiated directly"):
            CR3BPSystem('3body', EARTH, MOON, distance=384400.0)


class TestTypeIdentity:
    """Test isinstance relationships under the subclass architecture."""

    def test_2body_is_twobodysystem(self):
        """2-body System is a TwoBodySystem."""
        sys = System('2body', EARTH, compile=False)
        assert isinstance(sys, TwoBodySystem)

    def test_2body_is_system(self):
        """2-body System is also a System (base class)."""
        sys = System('2body', EARTH, compile=False)
        assert isinstance(sys, System)

    def test_2body_is_not_cr3bp(self):
        """2-body System is not a CR3BPSystem."""
        sys = System('2body', EARTH, compile=False)
        assert not isinstance(sys, CR3BPSystem)

    def test_cr3bp_is_cr3bpsystem(self):
        """CR3BP System is a CR3BPSystem."""
        sys = System('3body', EARTH, MOON, distance=384400.0, compile=False)
        assert isinstance(sys, CR3BPSystem)

    def test_cr3bp_is_system(self):
        """CR3BP System is also a System (base class)."""
        sys = System('3body', EARTH, MOON, distance=384400.0, compile=False)
        assert isinstance(sys, System)

    def test_cr3bp_is_not_twobody(self):
        """CR3BP System is not a TwoBodySystem."""
        sys = System('3body', EARTH, MOON, distance=384400.0, compile=False)
        assert not isinstance(sys, TwoBodySystem)


class TestPropertyAccess:
    """Test that System properties return correct values and types."""

    def test_2body_properties(self):
        """TwoBodySystem has exactly its own properties."""
        sys = System('2body', EARTH, perturbations=('J2',), compile=False)

        assert sys.primary_body == EARTH
        assert sys.perturbations == ('J2',)
        assert sys.atmosphere is None

        # CR3BP-only attributes must not exist on TwoBodySystem
        assert not hasattr(sys, 'secondary_body')
        assert not hasattr(sys, 'distance')
        assert not hasattr(sys, 'L_star')
        assert not hasattr(sys, 'T_star')
        assert not hasattr(sys, 'mass_ratio')
        assert not hasattr(sys, 'n_mean')
        assert not hasattr(sys, 'L1')
        assert not hasattr(sys, 'lagrange_points')

    def test_cr3bp_properties(self):
        """CR3BPSystem has computed nondimensional parameters."""
        distance = 384400.0
        sys = System('3body', EARTH,
                     secondary_body=MOON,
                     distance=distance,
                     compile=False)

        assert sys.primary_body == EARTH
        assert sys.secondary_body == MOON
        assert sys.distance == distance

        # All nondimensional params are plain values (not Optional)
        assert isinstance(sys.L_star, float)
        assert isinstance(sys.T_star, float)
        assert isinstance(sys.mass_ratio, float)
        assert isinstance(sys.n_mean, float)

        # Sanity checks
        assert sys.L_star == distance
        assert sys.T_star > 0
        assert 0 < sys.mass_ratio < 1
        assert sys.n_mean > 0

        # 2-body-only attributes must not exist on CR3BPSystem
        assert not hasattr(sys, 'perturbations')
        assert not hasattr(sys, 'atmosphere')

    def test_requires_satellite_false_no_perturbations(self):
        """System without perturbations doesn't require satellite."""
        sys = System('2body', EARTH, compile=False)
        assert not sys.requires_satellite

    def test_requires_satellite_false_j2_only(self):
        """System with only J2 doesn't require satellite."""
        sys = System('2body', EARTH, perturbations=('J2',), compile=False)
        assert not sys.requires_satellite

    def test_requires_satellite_true_with_drag(self):
        """System with drag requires satellite."""
        sys = System('2body', EARTH,
                     perturbations=('drag',),
                     atmosphere=EARTH_STD_ATMO)
        assert sys.requires_satellite

    def test_requires_satellite_cr3bp(self):
        """CR3BP system never requires satellite."""
        sys = System('3body', EARTH, MOON, distance=384400.0, compile=False)
        assert not sys.requires_satellite

    def test_is_compiled_property(self):
        """is_compiled property reflects compilation state."""
        sys_compiled     = System('2body', EARTH, compile=True)
        sys_not_compiled = System('2body', EARTH, compile=False)

        assert sys_compiled.is_compiled
        assert not sys_not_compiled.is_compiled

    def test_cached_eom_property(self):
        """cached_eom property returns list of tuples."""
        sys = System('2body', EARTH, compile=False)
        eom = sys.cached_eom

        assert eom is not None
        assert isinstance(eom, list)
        assert len(eom) == 6

        for item in eom:
            assert isinstance(item, tuple)
            assert len(item) == 2


class TestLagrangePoints:
    """Test Lagrange point computation on CR3BPSystem."""

    @pytest.fixture
    def earth_moon(self):
        return System('3body', EARTH, MOON, distance=384400.0, compile=False)

    def test_individual_lagrange_points(self, earth_moon):
        """L1 through L5 are accessible and have correct shape."""
        for Lk in (earth_moon.L1, earth_moon.L2, earth_moon.L3,
                   earth_moon.L4, earth_moon.L5):
            assert Lk.shape == (6,)
            assert not Lk.flags.writeable

    def test_lagrange_points_array(self, earth_moon):
        """lagrange_points property returns (5, 3) array."""
        pts = earth_moon.lagrange_points
        assert pts.shape == (5, 3)
        assert not pts.flags.writeable

    def test_l1_between_primaries(self, earth_moon):
        """L1 x-coordinate is between -mu and 1-mu."""
        mu = earth_moon.mass_ratio
        assert -mu < earth_moon.L1[0] < 1 - mu

    def test_l2_beyond_secondary(self, earth_moon):
        """L2 x-coordinate is beyond the secondary."""
        mu = earth_moon.mass_ratio
        assert earth_moon.L2[0] > 1 - mu

    def test_l4_l5_equilateral(self, earth_moon):
        """L4 and L5 form equilateral triangles with primaries."""
        mu = earth_moon.mass_ratio
        assert np.isclose(earth_moon.L4[1],  np.sqrt(3)/2, atol=1e-10)
        assert np.isclose(earth_moon.L5[1], -np.sqrt(3)/2, atol=1e-10)


class TestInstanceCounting:
    """Test System instance counting mechanism."""

    def setup_method(self):
        System.reset_instance_count()

    def test_instance_count_increments(self):
        """Instance count increments with each System creation."""
        initial = System.get_instance_count()

        s1 = System('2body', EARTH)
        assert System.get_instance_count() == initial + 1

        s2 = System('2body', MARS)
        assert System.get_instance_count() == initial + 2

    def test_instance_count_decrements_on_deletion(self):
        """Instance count decrements when System is deleted."""
        initial = System.get_instance_count()

        sys = System('2body', EARTH)
        assert System.get_instance_count() == initial + 1

        del sys
        assert System.get_instance_count() == initial

    def test_warning_at_threshold(self):
        """Warning issued when instance count exceeds threshold."""
        System.reset_instance_count()

        systems = [System('2body', EARTH) for _ in range(10)]

        with pytest.warns(ResourceWarning, match="Created .* System instances"):
            sys_over = System('2body', EARTH)


class TestImmutability:
    """Test that System properties are read-only."""

    def test_cannot_modify_primary_body(self):
        """Cannot assign to primary_body property."""
        sys = System('2body', EARTH)
        with pytest.raises(AttributeError):
            sys.primary_body = MARS

    def test_cannot_modify_perturbations(self):
        """Cannot assign to perturbations property."""
        sys = System('2body', EARTH, perturbations=('J2',))
        with pytest.raises(AttributeError):
            sys.perturbations = ('drag',)

    def test_cannot_modify_l_star(self):
        """Cannot assign to L_star property."""
        sys = System('3body', EARTH, MOON, distance=384400.0)
        with pytest.raises(AttributeError):
            sys.L_star = 999.0

    def test_bodyparams_frozen(self):
        """BodyParams is a frozen dataclass."""
        with pytest.raises(Exception):
            EARTH.mu = 999.0

    def test_atmoparams_frozen(self):
        """AtmoParams is a frozen dataclass."""
        with pytest.raises(Exception):
            EARTH_STD_ATMO.rho0 = 999.0

    def test_lagrange_points_readonly(self):
        """Lagrange point arrays are read-only."""
        sys = System('3body', EARTH, MOON, distance=384400.0)
        with pytest.raises(ValueError):
            sys.L1[0] = 999.0


class TestStringRepresentations:
    """Test __repr__ and summary output."""

    def test_repr_2body(self):
        """__repr__ works for 2-body system."""
        sys = System('2body', EARTH)
        r = repr(sys)
        assert '2body' in r
        assert 'km^3/s^2' in r

    def test_repr_cr3bp(self):
        """__repr__ works for CR3BP system."""
        sys = System('3body', EARTH,
                     secondary_body=MOON,
                     distance=384400.0)
        r = repr(sys)
        assert 'mu_1' in r
        assert 'mu_2' in r

    def test_summary_2body_runs(self):
        """TwoBodySystem.summary() executes without error."""
        sys = System('2body', EARTH,
                     perturbations=('J2', 'drag'),
                     atmosphere=EARTH_STD_ATMO)
        sys.summary()  # must not raise

    def test_summary_cr3bp_runs(self):
        """CR3BPSystem.summary() executes without error."""
        sys = System('3body', EARTH, MOON, distance=384400.0)
        sys.summary()  # must not raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
