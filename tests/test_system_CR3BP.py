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

class TestLagrangePoints:
    """Test computation and access of Lagrange points for CR3BP systems."""
    
    @pytest.fixture
    def earth_moon_system(self):
        """Create Earth-Moon CR3BP system for testing."""
        return System('3body', EARTH, MOON, distance=384400.0)
    
    @pytest.fixture
    def earth_2body_system(self):
        """Create 2-body system to test error handling."""
        return System('2body', EARTH)
    
    def test_all_points_computed(self, earth_moon_system):
        """All five Lagrange points are computed and accessible."""
        sys = earth_moon_system
        
        # All points should be non-None
        assert sys.L1 is not None
        assert sys.L2 is not None
        assert sys.L3 is not None
        assert sys.L4 is not None
        assert sys.L5 is not None
    
    def test_point_shapes(self, earth_moon_system):
        """Each Lagrange point is a 3-element array."""
        sys = earth_moon_system
        
        for i in range(1, 6):
            point = getattr(sys, f'L{i}')
            assert isinstance(point, np.ndarray)
            assert point.shape == (3,)
    
    def test_lagrange_points_array_shape(self, earth_moon_system):
        """lagrange_points property returns (5, 3) array."""
        sys = earth_moon_system
        
        points = sys.lagrange_points
        assert isinstance(points, np.ndarray)
        assert points.shape == (5, 3)
    
    def test_lagrange_points_array_order(self, earth_moon_system):
        """lagrange_points array has correct row order."""
        sys = earth_moon_system
        
        points = sys.lagrange_points
        assert np.array_equal(points[0], sys.L1)
        assert np.array_equal(points[1], sys.L2)
        assert np.array_equal(points[2], sys.L3)
        assert np.array_equal(points[3], sys.L4)
        assert np.array_equal(points[4], sys.L5)
    
    def test_collinear_points_on_x_axis(self, earth_moon_system):
        """L1, L2, L3 should have y=0, z=0."""
        sys = earth_moon_system
        
        assert np.allclose(sys.L1[1:], [0.0, 0.0], atol=1e-14)
        assert np.allclose(sys.L2[1:], [0.0, 0.0], atol=1e-14)
        assert np.allclose(sys.L3[1:], [0.0, 0.0], atol=1e-14)
    
    def test_triangular_points_symmetric(self, earth_moon_system):
        """L4 and L5 should be symmetric about x-axis."""
        sys = earth_moon_system
        
        # Same x and z coordinates
        assert np.isclose(sys.L4[0], sys.L5[0], atol=1e-14)
        assert np.isclose(sys.L4[2], sys.L5[2], atol=1e-14)
        
        # Opposite y coordinates
        assert np.isclose(sys.L4[1], -sys.L5[1], atol=1e-14)
        
        # y coordinate should be sqrt(3)/2
        assert np.isclose(abs(sys.L4[1]), np.sqrt(3)/2, atol=1e-14)
    
    def test_earth_moon_L1_known_value(self, earth_moon_system):
        """L1 for Earth-Moon system matches known value."""
        sys = earth_moon_system
        
        # Earth-Moon L1 is approximately 0.8369 nondimensional units
        # (about 326,000 km from Earth)
        assert 0.83 < sys.L1[0] < 0.84
    
    def test_earth_moon_L2_known_value(self, earth_moon_system):
        """L2 for Earth-Moon system matches known value."""
        sys = earth_moon_system
        
        # Earth-Moon L2 is approximately 1.1557 nondimensional units
        # (about 448,000 km from Earth)
        assert 1.15 < sys.L2[0] < 1.17
    
    def test_collinear_points_at_equilibrium(self, earth_moon_system):
        """Collinear points satisfy equilibrium condition dU/dx = 0."""
        sys = earth_moon_system
        mu = sys.mass_ratio
        
        def check_equilibrium(x):
            """Check if x satisfies dU/dx = 0."""
            r1 = abs(x + mu)
            r2 = abs(x - 1 + mu)
            dU_dx = x - (1-mu)*(x+mu)/r1**3 - mu*(x-1+mu)/r2**3
            return abs(dU_dx)
        
        # All collinear points should have dU/dx ~ 0
        assert check_equilibrium(sys.L1[0]) < 1e-12
        assert check_equilibrium(sys.L2[0]) < 1e-12
        assert check_equilibrium(sys.L3[0]) < 1e-12
    
    def test_L1_between_primaries(self, earth_moon_system):
        """L1 is between primary and secondary."""
        sys = earth_moon_system
        mu = sys.mass_ratio
        
        # Primary at -mu, secondary at 1-mu
        assert -mu < sys.L1[0] < 1 - mu
    
    def test_L2_beyond_secondary(self, earth_moon_system):
        """L2 is beyond the secondary body."""
        sys = earth_moon_system
        mu = sys.mass_ratio
        
        assert sys.L2[0] > 1 - mu
    
    def test_L3_beyond_primary(self, earth_moon_system):
        """L3 is beyond the primary body (opposite side)."""
        sys = earth_moon_system
        mu = sys.mass_ratio
        
        assert sys.L3[0] < -mu
    
    def test_arrays_are_readonly(self, earth_moon_system):
        """Lagrange point arrays cannot be modified."""
        sys = earth_moon_system
        
        with pytest.raises(ValueError, match="read-only"):
            sys.L1[0] = 999
        
        with pytest.raises(ValueError, match="read-only"):
            sys.lagrange_points[0, 0] = 999
    
    def test_2body_system_raises_error(self, earth_2body_system):
        """Accessing Lagrange points on 2-body system raises error."""
        sys = earth_2body_system
        
        with pytest.raises(ValueError, match="only available for CR3BP"):
            _ = sys.L1
        
        with pytest.raises(ValueError, match="only available for CR3BP"):
            _ = sys.lagrange_points
    
    def test_triangular_point_locations(self, earth_moon_system):
        """L4 and L5 form equilateral triangles with primaries."""
        sys = earth_moon_system
        mu = sys.mass_ratio
        
        # Primary at (-mu, 0, 0), Secondary at (1-mu, 0, 0)
        primary_pos = np.array([-mu, 0.0, 0.0])
        secondary_pos = np.array([1-mu, 0.0, 0.0])
        
        # Distance between primaries
        d_primaries = np.linalg.norm(secondary_pos - primary_pos)
        
        # Distances from L4 to each primary should equal distance between primaries
        d_L4_to_primary = np.linalg.norm(sys.L4 - primary_pos)
        d_L4_to_secondary = np.linalg.norm(sys.L4 - secondary_pos)
        
        assert np.isclose(d_L4_to_primary, d_primaries, rtol=1e-12)
        assert np.isclose(d_L4_to_secondary, d_primaries, rtol=1e-12)
        
        # Same for L5
        d_L5_to_primary = np.linalg.norm(sys.L5 - primary_pos)
        d_L5_to_secondary = np.linalg.norm(sys.L5 - secondary_pos)
        
        assert np.isclose(d_L5_to_primary, d_primaries, rtol=1e-12)
        assert np.isclose(d_L5_to_secondary, d_primaries, rtol=1e-12)


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
