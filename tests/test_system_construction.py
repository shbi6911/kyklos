"""
Test suite for System class construction and validation.

Tests cover:
- Valid construction patterns (2-body, 3-body, with perturbations)
- Invalid construction (missing params, incompatible combinations)
- Property access
- Instance counting
"""

import pytest
import numpy as np
from kyklos import System, BodyParams, AtmoParams, EARTH, MOON, MARS, EARTH_STD_ATMO


class TestValidConstruction:
    """Test valid System construction patterns."""
    
    def test_2body_point_mass_minimal(self):
        """Minimal 2-body system with no perturbations."""
        sys = System('2body', EARTH)
        
        assert sys.base_type.value == '2body'
        assert sys.primary_body == EARTH
        assert sys.perturbations == ()
        assert sys.secondary_body is None
        assert sys.atmosphere is None
    
    def test_2body_with_j2(self):
        """2-body system with J2 perturbation."""
        sys = System('2body', EARTH, perturbations=('J2',))
        
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
        distance = 384400.0  # km
        sys = System('3body', EARTH, 
                    secondary_body=MOON,
                    distance=distance)
        
        assert sys.base_type.value == '3body'
        assert sys.primary_body == EARTH
        assert sys.secondary_body == MOON
        assert sys.distance == distance
        assert sys.perturbations == ()
    
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


class TestInvalidConstruction:
    """Test that invalid construction patterns fail appropriately."""
    
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
    
    def test_cr3bp_with_perturbations(self):
        """CR3BP does not support perturbations."""
        with pytest.raises(ValueError, match="do not currently support perturbations"):
            System('3body', EARTH,
                  secondary_body=MOON,
                  distance=384400.0,
                  perturbations=('J2',))
    
    def test_invalid_base_type(self):
        """Invalid base_type string raises error."""
        with pytest.raises(ValueError, match="Unknown base type"):
            System('4body', EARTH)


class TestPropertyAccess:
    """Test that System properties return correct values and types."""
    
    def test_2body_properties(self):
        """2-body system has correct properties."""
        sys = System('2body', EARTH, perturbations=('J2',))
        
        # Basic properties
        assert sys.primary_body == EARTH
        assert sys.perturbations == ('J2',)
        
        # 3-body properties should be None
        assert sys.secondary_body is None
        assert sys.distance is None
        assert sys.L_star is None
        assert sys.T_star is None
        assert sys.mass_ratio is None
        assert sys.n_mean is None
    
    def test_cr3bp_properties(self):
        """CR3BP system has computed nondimensional parameters."""
        distance = 384400.0
        sys = System('3body', EARTH,
                    secondary_body=MOON,
                    distance=distance)
        
        # Basic properties
        assert sys.primary_body == EARTH
        assert sys.secondary_body == MOON
        assert sys.distance == distance
        
        # Computed properties should not be None
        assert sys.L_star is not None
        assert sys.T_star is not None
        assert sys.mass_ratio is not None
        assert sys.n_mean is not None
        
        # Sanity checks on computed values
        assert sys.L_star == distance
        assert sys.T_star > 0
        assert 0 < sys.mass_ratio < 1
        assert sys.n_mean > 0
    
    def test_param_info_no_perturbations(self):
        """System without perturbations has empty param_info."""
        sys = System('2body', EARTH)
        
        assert sys.param_info is not None
        assert sys.param_info['param_map'] == []
        assert sys.param_info['description'] == {}
    
    def test_param_info_with_drag(self):
        """System with drag has satellite parameters in param_info."""
        sys = System('2body', EARTH,
                    perturbations=('drag',),
                    atmosphere=EARTH_STD_ATMO)
        
        param_map = sys.param_info['param_map']
        
        # Should have Cd_A and mass
        assert len(param_map) == 2
        param_names = [name for name, idx in param_map]
        assert 'Cd_A' in param_names
        assert 'mass' in param_names
    
    def test_is_compiled_property(self):
        """is_compiled property reflects compilation state."""
        sys_compiled = System('2body', EARTH, compile=True)
        sys_not_compiled = System('2body', EARTH, compile=False)
        
        assert sys_compiled.is_compiled
        assert not sys_not_compiled.is_compiled
    
    def test_cached_eom_property(self):
        """cached_eom property returns list of tuples."""
        sys = System('2body', EARTH)
        
        eom = sys.cached_eom
        assert eom is not None
        assert isinstance(eom, list)
        assert len(eom) == 6  # 6 state variables
        
        # Each should be a tuple (variable, rhs)
        for item in eom:
            assert isinstance(item, tuple)
            assert len(item) == 2


class TestInstanceCounting:
    """Test System instance counting mechanism."""
    
    def setup_method(self):
        """Reset instance count before each test."""
        System.reset_instance_count()
    
    def test_instance_count_increments(self):
        """Instance count increments with each System creation."""
        initial_count = System.get_instance_count()
        
        sys1 = System('2body', EARTH)
        assert System.get_instance_count() == initial_count + 1
        
        sys2 = System('2body', MARS)
        assert System.get_instance_count() == initial_count + 2
    
    def test_instance_count_decrements_on_deletion(self):
        """Instance count decrements when System is deleted."""
        initial_count = System.get_instance_count()
        
        sys = System('2body', EARTH)
        assert System.get_instance_count() == initial_count + 1
        
        del sys
        assert System.get_instance_count() == initial_count
    
    def test_warning_at_threshold(self):
        """Warning issued when instance count exceeds threshold."""
        System.reset_instance_count()
        
        # Create systems up to threshold (should not warn)
        systems = []
        for i in range(10):
            systems.append(System('2body', EARTH))
        
        # Next one should warn
        with pytest.warns(ResourceWarning, match="Created .* System instances"):
            sys_over = System('2body', EARTH)


class TestImmutability:
    """Test that System properties are immutable."""
    
    def test_cannot_modify_primary_body(self):
        """Cannot modify primary_body property."""
        sys = System('2body', EARTH)
        
        with pytest.raises(AttributeError):
            sys.primary_body = MARS
    
    def test_cannot_modify_perturbations(self):
        """Cannot modify perturbations property."""
        sys = System('2body', EARTH, perturbations=('J2',))
        
        with pytest.raises(AttributeError):
            sys.perturbations = ('drag',)
    
    def test_bodyparams_frozen(self):
        """BodyParams is a frozen dataclass."""
        with pytest.raises(Exception):  # dataclass raises FrozenInstanceError
            EARTH.mu = 999.0
    
    def test_atmoparams_frozen(self):
        """AtmoParams is a frozen dataclass."""
        with pytest.raises(Exception):
            EARTH_STD_ATMO.rho0 = 999.0


class TestStringRepresentations:
    """Test __repr__ and summary output."""
    
    def test_repr_2body(self):
        """__repr__ works for 2-body system."""
        sys = System('2body', EARTH)
        repr_str = repr(sys)
        
        assert '2body' in repr_str
        assert 'km³/s²' in repr_str
    
    def test_repr_cr3bp(self):
        """__repr__ works for CR3BP system."""
        sys = System('3body', EARTH,
                    secondary_body=MOON,
                    distance=384400.0)
        repr_str = repr(sys)
        
        assert 'μ₁' in repr_str or 'mu1' in repr_str.lower()
        assert 'μ₂' in repr_str or 'mu2' in repr_str.lower()
    
    def test_summary_runs_without_error(self):
        """summary() method executes without error."""
        sys = System('2body', EARTH, perturbations=('J2', 'drag'),
                    atmosphere=EARTH_STD_ATMO)
        
        # Should not raise
        sys.summary()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
