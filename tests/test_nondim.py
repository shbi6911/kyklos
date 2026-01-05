"""
Test nondimensionalization methods for CR3BP Systems
Tests r2nd, r2d, v2nd, v2d, t2nd, t2d methods
"""

import pytest
import numpy as np

from kyklos import System, EARTH, MOON

class TestNondimensionalization:
    """Test suite for CR3BP nondimensionalization methods"""
    
    @pytest.fixture
    def earth_moon_system(self):
        """Create Earth-Moon CR3BP system for testing"""
        # Earth-Moon distance
        distance = 384400.0  # km
        sys = System('3body', EARTH, MOON, distance=distance)
        return sys
    
    @pytest.fixture
    def earth_2body_system(self):
        """Create 2-body system to test error handling"""
        return System('2body', EARTH)
    
    def test_r2nd_scalar(self, earth_moon_system):
        """Test position nondimensionalization with scalar input"""
        r_dim = 384400.0  # km (one L_star)
        r_nd = earth_moon_system.r2nd(r_dim)
        # Scalar input should return scalar output (np.float64)
        assert np.isscalar(r_nd)
        assert np.isclose(r_nd, 1.0, rtol=1e-12)
    
    def test_r2nd_array(self, earth_moon_system):
        """Test position nondimensionalization with array input"""
        r_dim = np.array([384400.0, 192200.0, 768800.0])  # km
        r_nd = earth_moon_system.r2nd(r_dim)
        expected = np.array([1.0, 0.5, 2.0])
        assert np.allclose(r_nd, expected, rtol=1e-12)
    
    def test_r2d_scalar(self, earth_moon_system):
        """Test position dimensionalization with scalar input"""
        r_nd = 1.0
        r_dim = earth_moon_system.r2d(r_nd)
        # Scalar input should return scalar output
        assert np.isscalar(r_dim)
        assert np.isclose(r_dim, 384400.0, rtol=1e-12)
    
    def test_r2d_array(self, earth_moon_system):
        """Test position dimensionalization with array input"""
        r_nd = np.array([1.0, 0.5, 2.0])
        r_dim = earth_moon_system.r2d(r_nd)
        expected = np.array([384400.0, 192200.0, 768800.0])
        assert np.allclose(r_dim, expected, rtol=1e-12)
    
    def test_v2nd_scalar(self, earth_moon_system):
        """Test velocity nondimensionalization with scalar input"""
        # Velocity characteristic scale: L_star / T_star
        v_char = earth_moon_system.L_star / earth_moon_system.T_star
        v_dim = v_char  # km/s
        v_nd = earth_moon_system.v2nd(v_dim)
        # Scalar input should return scalar output
        assert np.isscalar(v_nd)
        assert np.isclose(v_nd, 1.0, rtol=1e-12)
    
    def test_v2nd_array(self, earth_moon_system):
        """Test velocity nondimensionalization with array input"""
        v_char = earth_moon_system.L_star / earth_moon_system.T_star
        v_dim = np.array([v_char, 0.5 * v_char, 2.0 * v_char])
        v_nd = earth_moon_system.v2nd(v_dim)
        expected = np.array([1.0, 0.5, 2.0])
        assert np.allclose(v_nd, expected, rtol=1e-12)
    
    def test_v2d_scalar(self, earth_moon_system):
        """Test velocity dimensionalization with scalar input"""
        v_nd = 1.0
        v_dim = earth_moon_system.v2d(v_nd)
        v_char = earth_moon_system.L_star / earth_moon_system.T_star
        # Scalar input should return scalar output
        assert np.isscalar(v_dim)
        assert np.isclose(v_dim, v_char, rtol=1e-12)
    
    def test_v2d_array(self, earth_moon_system):
        """Test velocity dimensionalization with array input"""
        v_nd = np.array([1.0, 0.5, 2.0])
        v_dim = earth_moon_system.v2d(v_nd)
        v_char = earth_moon_system.L_star / earth_moon_system.T_star
        expected = np.array([v_char, 0.5 * v_char, 2.0 * v_char])
        assert np.allclose(v_dim, expected, rtol=1e-12)
    
    def test_t2nd_scalar(self, earth_moon_system):
        """Test time nondimensionalization with scalar input"""
        t_dim = earth_moon_system.T_star  # s (one T_star)
        t_nd = earth_moon_system.t2nd(t_dim)
        # Scalar input should return scalar output
        assert np.isscalar(t_nd)
        assert np.isclose(t_nd, 1.0, rtol=1e-12)
    
    def test_t2nd_array(self, earth_moon_system):
        """Test time nondimensionalization with array input"""
        T = earth_moon_system.T_star
        t_dim = np.array([T, 0.5 * T, 2.0 * T])
        t_nd = earth_moon_system.t2nd(t_dim)
        expected = np.array([1.0, 0.5, 2.0])
        assert np.allclose(t_nd, expected, rtol=1e-12)
    
    def test_t2d_scalar(self, earth_moon_system):
        """Test time dimensionalization with scalar input"""
        t_nd = 1.0
        t_dim = earth_moon_system.t2d(t_nd)
        # Scalar input should return scalar output
        assert np.isscalar(t_dim)
        assert np.isclose(t_dim, earth_moon_system.T_star, rtol=1e-12)
    
    def test_t2d_array(self, earth_moon_system):
        """Test time dimensionalization with array input"""
        t_nd = np.array([1.0, 0.5, 2.0])
        t_dim = earth_moon_system.t2d(t_nd)
        T = earth_moon_system.T_star
        expected = np.array([T, 0.5 * T, 2.0 * T])
        assert np.allclose(t_dim, expected, rtol=1e-12)
    
    def test_roundtrip_position(self, earth_moon_system):
        """Test round-trip conversion for position"""
        r_dim_orig = np.array([100000.0, 200000.0, 300000.0])
        r_nd = earth_moon_system.r2nd(r_dim_orig)
        r_dim_final = earth_moon_system.r2d(r_nd)
        assert np.allclose(r_dim_orig, r_dim_final, rtol=1e-12)
    
    def test_roundtrip_velocity(self, earth_moon_system):
        """Test round-trip conversion for velocity"""
        v_dim_orig = np.array([0.5, 1.0, 1.5])
        v_nd = earth_moon_system.v2nd(v_dim_orig)
        v_dim_final = earth_moon_system.v2d(v_nd)
        assert np.allclose(v_dim_orig, v_dim_final, rtol=1e-12)
    
    def test_roundtrip_time(self, earth_moon_system):
        """Test round-trip conversion for time"""
        t_dim_orig = np.array([100000.0, 200000.0, 300000.0])
        t_nd = earth_moon_system.t2nd(t_dim_orig)
        t_dim_final = earth_moon_system.t2d(t_nd)
        assert np.allclose(t_dim_orig, t_dim_final, rtol=1e-12)
    
    def test_error_on_2body_r2nd(self, earth_2body_system):
        """Test that r2nd raises error for 2-body system"""
        with pytest.raises(ValueError, match="Nondimensionalization only available for CR3BP"):
            earth_2body_system.r2nd(1000.0)
    
    def test_error_on_2body_r2d(self, earth_2body_system):
        """Test that r2d raises error for 2-body system"""
        with pytest.raises(ValueError, match="Redimensionalization only available for CR3BP"):
            earth_2body_system.r2d(1.0)
    
    def test_error_on_2body_v2nd(self, earth_2body_system):
        """Test that v2nd raises error for 2-body system"""
        with pytest.raises(ValueError, match="Nondimensionalization only available for CR3BP"):
            earth_2body_system.v2nd(1.0)
    
    def test_error_on_2body_v2d(self, earth_2body_system):
        """Test that v2d raises error for 2-body system"""
        with pytest.raises(ValueError, match="Redimensionalization only available for CR3BP"):
            earth_2body_system.v2d(1.0)
    
    def test_error_on_2body_t2nd(self, earth_2body_system):
        """Test that t2nd raises error for 2-body system"""
        with pytest.raises(ValueError, match="Nondimensionalization only available for CR3BP"):
            earth_2body_system.t2nd(1000.0)
    
    def test_error_on_2body_t2d(self, earth_2body_system):
        """Test that t2d raises error for 2-body system"""
        with pytest.raises(ValueError, match="Redimensionalization only available for CR3BP"):
            earth_2body_system.t2d(1.0)
    
    def test_full_state_conversion(self, earth_moon_system):
        """Test converting a full state vector"""
        # Dimensional state [x, y, z, vx, vy, vz]
        state_dim = np.array([100000.0, 200000.0, 50000.0, 0.5, 1.0, 0.2])
        
        # Convert position and velocity separately
        r_nd = earth_moon_system.r2nd(state_dim[:3])
        v_nd = earth_moon_system.v2nd(state_dim[3:])
        state_nd = np.concatenate([r_nd, v_nd])
        
        # Convert back
        r_dim = earth_moon_system.r2d(state_nd[:3])
        v_dim = earth_moon_system.v2d(state_nd[3:])
        state_dim_final = np.concatenate([r_dim, v_dim])
        
        assert np.allclose(state_dim, state_dim_final, rtol=1e-12)
    
    def test_characteristic_scales(self, earth_moon_system):
        """Verify characteristic scales are computed correctly"""
        # L_star should equal distance
        assert np.isclose(earth_moon_system.L_star, 384400.0, rtol=1e-12)
        
        # T_star = sqrt(L_star^3 / mu_total)
        mu_total = EARTH.mu + MOON.mu
        T_expected = np.sqrt(earth_moon_system.L_star**3 / mu_total)
        assert np.isclose(earth_moon_system.T_star, T_expected, rtol=1e-12)
        
        # Verify velocity scale consistency
        v_char = earth_moon_system.L_star / earth_moon_system.T_star
        v_test = 1.0  # km/s
        v_nd = earth_moon_system.v2nd(v_test)
        v_back = earth_moon_system.v2d(v_nd)
        assert np.isclose(v_test, v_back, rtol=1e-12)
    
    def test_primary_body_radius_nd(self, earth_moon_system):
        """Test nondimensional radius property for primary body"""
        r_nd = earth_moon_system.primary_body.radius_nd
        expected = EARTH.radius / earth_moon_system.L_star
        assert np.isclose(r_nd, expected, rtol=1e-12)

    def test_secondary_body_radius_nd(self, earth_moon_system):
        """Test nondimensional radius property for secondary body"""
        r_nd = earth_moon_system.secondary_body.radius_nd
        expected = MOON.radius / earth_moon_system.L_star
        assert np.isclose(r_nd, expected, rtol=1e-12)

    def test_body_params_delegation(self, earth_moon_system):
        """Test that other BodyParams attributes still work with wrapper"""
        # All normal BodyParams attributes should still be accessible
        assert earth_moon_system.primary_body.mu == EARTH.mu
        assert earth_moon_system.primary_body.radius == EARTH.radius
        assert earth_moon_system.primary_body.J2 == EARTH.J2
        assert earth_moon_system.secondary_body.mu == MOON.mu
        assert earth_moon_system.secondary_body.radius == MOON.radius

    def test_radius_nd_consistency_with_r2nd(self, earth_moon_system):
        """Test that radius_nd matches using r2nd method"""
        # Using property
        r1_nd = earth_moon_system.primary_body.radius_nd
        r2_nd = earth_moon_system.secondary_body.radius_nd
        
        # Using r2nd method
        r1_nd_method = earth_moon_system.r2nd(EARTH.radius)
        r2_nd_method = earth_moon_system.r2nd(MOON.radius)
        
        assert np.isclose(r1_nd, r1_nd_method, rtol=1e-12)
        assert np.isclose(r2_nd, r2_nd_method, rtol=1e-12)

    def test_2body_no_radius_nd(self, earth_2body_system):
        """Test that 2-body systems don't have radius_nd wrapper"""
        # For 2-body, should get regular BodyParams without radius_nd
        assert not hasattr(earth_2body_system.primary_body, 'radius_nd')


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
