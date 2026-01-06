"""
Test suite for Satellite class.

Tests cover:
- Valid construction patterns
- Parameter validation
- Property access
- Immutability
- Special methods (__repr__, __eq__, __hash__)
- Edge cases
"""

import pytest
import numpy as np
from kyklos import Satellite


class TestConstruction:
    """Test valid Satellite construction patterns."""
    
    def test_basic_construction(self):
        """Satellite can be constructed with valid parameters."""
        I = np.diag([100, 150, 200])
        
        sat = Satellite(
            mass=500.0,
            drag_coeff=2.2,
            cross_section=5.0,
            inertia=I
        )
        
        assert sat.mass == 500.0
        assert sat.drag_coeff == 2.2
        assert sat.cross_section == 5.0
        assert np.array_equal(sat.inertia, I)
    
    def test_construction_with_name(self):
        """Satellite can be constructed with optional name."""
        I = np.diag([100, 150, 200])
        
        sat = Satellite(
            mass=500.0,
            drag_coeff=2.2,
            cross_section=5.0,
            inertia=I,
            name="TestSat"
        )
        
        assert sat.name == "TestSat"
    
    def test_construction_without_name(self):
        """Satellite without name has None."""
        I = np.diag([100, 150, 200])
        
        sat = Satellite(
            mass=500.0,
            drag_coeff=2.2,
            cross_section=5.0,
            inertia=I
        )
        
        assert sat.name is None
    
    def test_construction_with_symmetric_inertia(self):
        """Satellite accepts symmetric non-diagonal inertia."""
        I = np.array([
            [100, 10, 5],
            [10, 150, 8],
            [5, 8, 200]
        ])
        
        sat = Satellite(
            mass=500.0,
            drag_coeff=2.2,
            cross_section=5.0,
            inertia=I
        )
        
        assert np.allclose(sat.inertia, I)


class TestValidation:
    """Test that invalid parameters are caught."""
    
    def test_negative_mass_rejected(self):
        """Negative mass raises ValueError."""
        I = np.diag([100, 150, 200])
        
        with pytest.raises(ValueError, match="Mass must be positive"):
            Satellite(mass=-500.0, drag_coeff=2.2, cross_section=5.0, inertia=I)
    
    def test_zero_mass_rejected(self):
        """Zero mass raises ValueError."""
        I = np.diag([100, 150, 200])
        
        with pytest.raises(ValueError, match="Mass must be positive"):
            Satellite(mass=0.0, drag_coeff=2.2, cross_section=5.0, inertia=I)
    
    def test_negative_drag_coeff_rejected(self):
        """Negative drag coefficient raises ValueError."""
        I = np.diag([100, 150, 200])
        
        with pytest.raises(ValueError, match="Drag coefficient must be positive"):
            Satellite(mass=500.0, drag_coeff=-2.2, cross_section=5.0, inertia=I)
    
    def test_zero_drag_coeff_rejected(self):
        """Zero drag coefficient raises ValueError."""
        I = np.diag([100, 150, 200])
        
        with pytest.raises(ValueError, match="Drag coefficient must be positive"):
            Satellite(mass=500.0, drag_coeff=0.0, cross_section=5.0, inertia=I)
    
    def test_negative_cross_section_rejected(self):
        """Negative cross-sectional area raises ValueError."""
        I = np.diag([100, 150, 200])
        
        with pytest.raises(ValueError, match="Cross-sectional area must be positive"):
            Satellite(mass=500.0, drag_coeff=2.2, cross_section=-5.0, inertia=I)
    
    def test_zero_cross_section_rejected(self):
        """Zero cross-sectional area raises ValueError."""
        I = np.diag([100, 150, 200])
        
        with pytest.raises(ValueError, match="Cross-sectional area must be positive"):
            Satellite(mass=500.0, drag_coeff=2.2, cross_section=0.0, inertia=I)
    
    def test_wrong_inertia_shape_rejected(self):
        """Non-3x3 inertia tensor raises ValueError."""
        I_wrong = np.diag([100, 150])  # 2x2, not 3x3
        
        with pytest.raises(ValueError, match="Inertia tensor must be 3x3"):
            Satellite(mass=500.0, drag_coeff=2.2, cross_section=5.0, inertia=I_wrong)
    
    def test_non_square_inertia_rejected(self):
        """Non-square inertia raises ValueError."""
        I_wrong = np.array([[100, 150, 200]])  # 1x3, not 3x3
        
        with pytest.raises(ValueError, match="Inertia tensor must be 3x3"):
            Satellite(mass=500.0, drag_coeff=2.2, cross_section=5.0, inertia=I_wrong)
    
    def test_non_symmetric_inertia_rejected(self):
        """Non-symmetric inertia tensor raises ValueError."""
        I_asymmetric = np.array([
            [100, 10, 5],
            [20, 150, 8],  # Note: I[1,0] != I[0,1]
            [5, 8, 200]
        ])
        
        with pytest.raises(ValueError, match="Inertia tensor must be symmetric"):
            Satellite(mass=500.0, drag_coeff=2.2, cross_section=5.0, inertia=I_asymmetric)
    
    def test_negative_definite_inertia_rejected(self):
        """Inertia with negative eigenvalues raises ValueError."""
        I_negative = np.array([
            [100, 0, 0],
            [0, -150, 0],  # Negative eigenvalue
            [0, 0, 200]
        ])
        
        with pytest.raises(ValueError, match="Inertia tensor must be positive definite"):
            Satellite(mass=500.0, drag_coeff=2.2, cross_section=5.0, inertia=I_negative)
    
    def test_zero_eigenvalue_rejected(self):
        """Inertia with zero eigenvalue raises ValueError."""
        I_singular = np.array([
            [100, 0, 0],
            [0, 0, 0],  # Zero eigenvalue (singular)
            [0, 0, 200]
        ])
        
        with pytest.raises(ValueError, match="Inertia tensor must be positive definite"):
            Satellite(mass=500.0, drag_coeff=2.2, cross_section=5.0, inertia=I_singular)


class TestPropertyAccess:
    """Test that properties return correct values."""
    
    def test_all_properties_accessible(self):
        """All properties can be accessed."""
        I = np.diag([100, 150, 200])
        sat = Satellite(
            mass=500.0,
            drag_coeff=2.2,
            cross_section=5.0,
            inertia=I,
            name="TestSat"
        )
        
        # Should not raise
        _ = sat.mass
        _ = sat.drag_coeff
        _ = sat.cross_section
        _ = sat.inertia
        _ = sat.inv_inertia
        _ = sat.name
    
    def test_properties_return_correct_types(self):
        """Properties return expected types."""
        I = np.diag([100, 150, 200])
        sat = Satellite(
            mass=500.0,
            drag_coeff=2.2,
            cross_section=5.0,
            inertia=I,
            name="TestSat"
        )
        
        assert isinstance(sat.mass, float)
        assert isinstance(sat.drag_coeff, float)
        assert isinstance(sat.cross_section, float)
        assert isinstance(sat.inertia, np.ndarray)
        assert isinstance(sat.inv_inertia, np.ndarray)
        assert isinstance(sat.name, str)
    
    def test_inverse_inertia_correct(self):
        """Inverse inertia is correctly computed."""
        I = np.diag([100, 150, 200])
        sat = Satellite(
            mass=500.0,
            drag_coeff=2.2,
            cross_section=5.0,
            inertia=I
        )
        
        # For diagonal matrix, inverse is easy to verify
        expected_inv = np.diag([1/100, 1/150, 1/200])
        
        assert np.allclose(sat.inv_inertia, expected_inv)
    
    def test_inverse_inertia_identity_product(self):
        """Inertia times inverse equals identity."""
        I = np.array([
            [100, 10, 5],
            [10, 150, 8],
            [5, 8, 200]
        ])
        sat = Satellite(
            mass=500.0,
            drag_coeff=2.2,
            cross_section=5.0,
            inertia=I
        )
        
        product = sat.inertia @ sat.inv_inertia
        identity = np.eye(3)
        
        assert np.allclose(product, identity)


class TestImmutability:
    """Test that Satellite is immutable after construction."""
    
    def test_properties_are_read_only(self):
        """Cannot assign to properties."""
        I = np.diag([100, 150, 200])
        sat = Satellite(
            mass=500.0,
            drag_coeff=2.2,
            cross_section=5.0,
            inertia=I
        )
        
        with pytest.raises(AttributeError):
            sat.mass = 600.0
        
        with pytest.raises(AttributeError):
            sat.drag_coeff = 2.5
        
        with pytest.raises(AttributeError):
            sat.cross_section = 6.0
        
        with pytest.raises(AttributeError):
            sat.name = "NewName"
    
    def test_inertia_array_immutable(self):
        """Inertia array elements cannot be modified."""
        I = np.diag([100, 150, 200])
        sat = Satellite(
            mass=500.0,
            drag_coeff=2.2,
            cross_section=5.0,
            inertia=I
        )
        
        with pytest.raises(ValueError, match="read-only"):
            sat.inertia[0, 0] = 999
    
    def test_inv_inertia_array_immutable(self):
        """Inverse inertia array elements cannot be modified."""
        I = np.diag([100, 150, 200])
        sat = Satellite(
            mass=500.0,
            drag_coeff=2.2,
            cross_section=5.0,
            inertia=I
        )
        
        with pytest.raises(ValueError, match="read-only"):
            sat.inv_inertia[0, 0] = 999
    
    def test_modifying_input_doesnt_affect_satellite(self):
        """Modifying input inertia after construction doesn't affect Satellite."""
        I = np.diag([100, 150, 200])
        sat = Satellite(
            mass=500.0,
            drag_coeff=2.2,
            cross_section=5.0,
            inertia=I
        )
        
        original_value = sat.inertia[0, 0]
        I[0, 0] = 999  # Modify the input array
        
        assert sat.inertia[0, 0] == original_value  # Satellite unchanged


class TestSpecialMethods:
    """Test special methods (__repr__, __eq__, __hash__)."""
    
    def test_repr_with_name(self):
        """__repr__() works for named satellite."""
        I = np.diag([100, 150, 200])
        sat = Satellite(
            mass=500.0,
            drag_coeff=2.2,
            cross_section=5.0,
            inertia=I,
            name="TestSat"
        )
        
        repr_str = repr(sat)
        
        assert isinstance(repr_str, str)
        assert "TestSat" in repr_str
        assert "500" in repr_str  # Mass
        assert "2.2" in repr_str or "2.20" in repr_str  # Drag coeff
    
    def test_repr_without_name(self):
        """__repr__() works for unnamed satellite."""
        I = np.diag([100, 150, 200])
        sat = Satellite(
            mass=500.0,
            drag_coeff=2.2,
            cross_section=5.0,
            inertia=I
        )
        
        repr_str = repr(sat)
        
        assert isinstance(repr_str, str)
        assert "unnamed" in repr_str.lower()
    
    def test_equal_satellites_compare_equal(self):
        """Identical satellites compare as equal."""
        I = np.diag([100, 150, 200])
        
        sat1 = Satellite(mass=500.0, drag_coeff=2.2, cross_section=5.0, inertia=I)
        sat2 = Satellite(mass=500.0, drag_coeff=2.2, cross_section=5.0, inertia=I)
        
        assert sat1 == sat2
    
    def test_different_satellites_not_equal(self):
        """Satellites with different parameters are not equal."""
        I = np.diag([100, 150, 200])
        
        sat1 = Satellite(mass=500.0, drag_coeff=2.2, cross_section=5.0, inertia=I)
        sat2 = Satellite(mass=600.0, drag_coeff=2.2, cross_section=5.0, inertia=I)
        
        assert sat1 != sat2
    
    def test_equality_uses_tolerances(self):
        """Nearly equal floats are considered equal."""
        I = np.diag([100, 150, 200])
        
        sat1 = Satellite(mass=500.0, drag_coeff=2.2, cross_section=5.0, inertia=I)
        sat2 = Satellite(mass=500.0 + 1e-13, drag_coeff=2.2, 
                        cross_section=5.0, inertia=I)
        
        assert sat1 == sat2
    
    def test_different_names_not_equal(self):
        """Satellites with different names are not equal."""
        I = np.diag([100, 150, 200])
        
        sat1 = Satellite(mass=500.0, drag_coeff=2.2, cross_section=5.0, 
                        inertia=I, name="Sat1")
        sat2 = Satellite(mass=500.0, drag_coeff=2.2, cross_section=5.0, 
                        inertia=I, name="Sat2")
        
        assert sat1 != sat2
    
    def test_equality_with_non_satellite(self):
        """Comparing with non-Satellite returns NotImplemented."""
        I = np.diag([100, 150, 200])
        sat = Satellite(mass=500.0, drag_coeff=2.2, cross_section=5.0, inertia=I)
        
        result = sat.__eq__("not a satellite")
        
        assert result is NotImplemented
    
    def test_equal_satellites_same_hash(self):
        """Equal satellites have the same hash."""
        I = np.diag([100, 150, 200])
        
        sat1 = Satellite(mass=500.0, drag_coeff=2.2, cross_section=5.0, inertia=I)
        sat2 = Satellite(mass=500.0, drag_coeff=2.2, cross_section=5.0, inertia=I)
        
        assert hash(sat1) == hash(sat2)
    
    def test_different_satellites_likely_different_hash(self):
        """Different satellites likely have different hashes."""
        I1 = np.diag([100, 150, 200])
        I2 = np.diag([110, 160, 210])
        
        sat1 = Satellite(mass=500.0, drag_coeff=2.2, cross_section=5.0, inertia=I1)
        sat2 = Satellite(mass=600.0, drag_coeff=2.3, cross_section=6.0, inertia=I2)
        
        # Not guaranteed but highly likely
        assert hash(sat1) != hash(sat2)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_very_small_mass(self):
        """Very small but positive mass is accepted."""
        I = np.diag([1e-6, 1e-6, 1e-6])
        
        sat = Satellite(mass=1e-3, drag_coeff=2.2, cross_section=0.01, inertia=I)
        
        assert sat.mass == 1e-3
    
    def test_very_large_mass(self):
        """Very large mass is accepted."""
        I = np.diag([1e8, 1e8, 1e8])
        
        sat = Satellite(mass=1e6, drag_coeff=2.2, cross_section=100.0, inertia=I)
        
        assert sat.mass == 1e6
    
    def test_identity_inertia(self):
        """Identity matrix as inertia is valid."""
        I = np.eye(3)
        
        sat = Satellite(mass=500.0, drag_coeff=2.2, cross_section=5.0, inertia=I)
        
        assert np.allclose(sat.inertia, I)
        assert np.allclose(sat.inv_inertia, I)  # Inverse of identity is identity
    
    def test_diagonal_inertia(self):
        """Diagonal inertia tensor works correctly."""
        I = np.diag([50, 75, 100])
        
        sat = Satellite(mass=500.0, drag_coeff=2.2, cross_section=5.0, inertia=I)
        
        assert np.allclose(sat.inertia, I)
        # Check diagonal elements of inverse
        assert np.allclose(np.diag(sat.inv_inertia), [1/50, 1/75, 1/100])
    
    def test_nearly_singular_inertia(self):
        """Inertia with small but positive eigenvalues works."""
        # Create matrix with one very small eigenvalue
        I = np.array([
            [100, 0, 0],
            [0, 100, 0],
            [0, 0, 1e-6]  # Very small but positive
        ])
        
        sat = Satellite(mass=500.0, drag_coeff=2.2, cross_section=5.0, inertia=I)
        
        # Should work, though inverse will have large values
        assert sat.inv_inertia[2, 2] > 1e5  # 1/1e-6 = 1e6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
