"""
Test suite for System propagation interface.

Tests cover:
- Basic propagation interface (input/output types)
- Parameter handling (systems with/without satellite params)
- Time handling (forward/backward propagation)
- Smoke tests for each system configuration
"""

import pytest
import numpy as np
from kyklos import (
    System, EARTH, MOON, MARS, EARTH_STD_ATMO,
    OE, OrbitalElements, OEType, Trajectory
)


class TestPropagationInterface:
    """Test basic propagation interface contracts."""
    
    def test_accepts_orbital_elements_input(self):
        """propagate() accepts OrbitalElements."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        
        traj = sys.propagate(orbit, t_start=0, t_end=100)
        
        assert isinstance(traj, Trajectory)
    
    def test_accepts_numpy_array_input(self):
        """propagate() accepts numpy array."""
        sys = System('2body', EARTH)
        state = np.array([-6045, -3490, 2500, -3.457, 6.618, 2.533])
        
        traj = sys.propagate(state, t_start=0, t_end=100)
        
        assert isinstance(traj, Trajectory)
    
    def test_accepts_keplerian_elements(self):
        """propagate() works with Keplerian elements."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=np.radians(45), 
                  omega=0, w=0, nu=0)
        
        traj = sys.propagate(orbit, t_start=0, t_end=100)
        
        assert isinstance(traj, Trajectory)
    
    def test_accepts_equinoctial_elements(self):
        """propagate() works with Equinoctial elements."""
        sys = System('2body', EARTH)
        # Create via conversion to ensure valid equinoctial
        kep = OE(a=7000, e=0.01, i=np.radians(45), omega=0, w=0, nu=0)
        equi = kep.to_equinoctial()
        
        traj = sys.propagate(equi, t_start=0, t_end=100)
        
        assert isinstance(traj, Trajectory)
    
    def test_returns_trajectory_object(self):
        """propagate() returns Trajectory instance."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        
        traj = sys.propagate(orbit, t_start=0, t_end=100)
        
        assert isinstance(traj, Trajectory)
        assert hasattr(traj, 't0')
        assert hasattr(traj, 'tf')
        assert hasattr(traj, 'system')
    
    def test_trajectory_has_correct_times(self):
        """Returned Trajectory has correct t0 and tf."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        
        traj = sys.propagate(orbit, t_start=10, t_end=500)
        
        assert traj.t0 == 10
        assert traj.tf == 500
    
    def test_trajectory_references_system(self):
        """Returned Trajectory references the System."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        
        traj = sys.propagate(orbit, t_start=0, t_end=100)
        
        assert traj.system is sys


class TestParameterHandling:
    """Test satellite parameter handling in propagation."""
    
    def test_no_params_system_propagates_without_params(self):
        """System without perturbations doesn't need satellite_params."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        
        # Should work without satellite_params
        traj = sys.propagate(orbit, t_start=0, t_end=100)
        
        assert traj is not None
    
    def test_drag_system_requires_params(self):
        """System with drag requires satellite_params."""
        sys = System('2body', EARTH,
                    perturbations=('drag',),
                    atmosphere=EARTH_STD_ATMO)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        
        with pytest.raises(ValueError, match="requires satellite parameters"):
            sys.propagate(orbit, t_start=0, t_end=100)
    
    def test_drag_system_accepts_dict_params(self):
        """System with drag accepts dict satellite_params."""
        sys = System('2body', EARTH,
                    perturbations=('drag',),
                    atmosphere=EARTH_STD_ATMO)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        
        params = {'Cd_A': 5.0, 'mass': 500.0}
        traj = sys.propagate(orbit, t_start=0, t_end=100,
                           satellite_params=params)
        
        assert traj is not None
    
    def test_drag_system_accepts_array_params(self):
        """System with drag accepts array satellite_params."""
        sys = System('2body', EARTH,
                    perturbations=('drag',),
                    atmosphere=EARTH_STD_ATMO)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        
        params = np.array([5.0, 500.0])  # [Cd_A, mass]
        traj = sys.propagate(orbit, t_start=0, t_end=100,
                           satellite_params=params)
        
        assert traj is not None
    
    def test_missing_required_param_raises_error(self):
        """Missing required parameter raises clear error."""
        sys = System('2body', EARTH,
                    perturbations=('drag',),
                    atmosphere=EARTH_STD_ATMO)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        
        params = {'Cd_A': 5.0}  # Missing 'mass'
        
        with pytest.raises(ValueError, match="Missing required parameter"):
            sys.propagate(orbit, t_start=0, t_end=100,
                        satellite_params=params)
    
    def test_wrong_number_params_array_raises_error(self):
        """Array with wrong number of parameters raises error."""
        sys = System('2body', EARTH,
                    perturbations=('drag',),
                    atmosphere=EARTH_STD_ATMO)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        
        params = np.array([5.0])  # Need 2 params
        
        with pytest.raises(ValueError, match="Expected .* parameters"):
            sys.propagate(orbit, t_start=0, t_end=100,
                        satellite_params=params)


class TestTimeHandling:
    """Test propagation with different time configurations."""
    
    def test_forward_propagation(self):
        """Forward propagation (t_end > t_start) works."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        
        traj = sys.propagate(orbit, t_start=0, t_end=1000)
        
        assert traj.t0 == 0
        assert traj.tf == 1000
    
    def test_backward_propagation(self):
        """Backward propagation (t_end < t_start) works."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        
        traj = sys.propagate(orbit, t_start=1000, t_end=0)
        
        assert traj.t0 == 1000
        assert traj.tf == 0
    
    def test_negative_times(self):
        """Propagation works with negative times."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        
        traj = sys.propagate(orbit, t_start=-500, t_end=500)
        
        assert traj.t0 == -500
        assert traj.tf == 500
    
    def test_large_time_values(self):
        """Propagation works with large time values."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        
        traj = sys.propagate(orbit, t_start=1e6, t_end=1e6 + 1000)
        
        assert traj.t0 == 1e6
        assert traj.tf == 1e6 + 1000


class TestSmokeTests:
    """Smoke tests: verify each system type can propagate."""
    
    def test_2body_point_mass_smoke(self):
        """2-body point mass propagates successfully."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=np.radians(45), 
                  omega=0, w=0, nu=0)
        
        traj = sys.propagate(orbit, t_start=0, t_end=5400)
        
        # Basic sanity checks
        assert isinstance(traj, Trajectory)
        assert traj.t0 == 0
        assert traj.tf == 5400
        assert traj.system is sys
        
        # Can evaluate at midpoint
        state_mid = traj(2700)
        assert isinstance(state_mid, OrbitalElements)
        assert state_mid.element_type == OEType.CARTESIAN
    
    def test_2body_j2_smoke(self):
        """2-body with J2 propagates successfully."""
        sys = System('2body', EARTH, perturbations=('J2',))
        orbit = OE(a=7000, e=0.01, i=np.radians(45),
                  omega=0, w=0, nu=0)
        
        traj = sys.propagate(orbit, t_start=0, t_end=5400)
        
        assert isinstance(traj, Trajectory)
        state_mid = traj(2700)
        assert isinstance(state_mid, OrbitalElements)
    
    def test_2body_drag_smoke(self):
        """2-body with drag propagates successfully."""
        sys = System('2body', EARTH,
                    perturbations=('drag',),
                    atmosphere=EARTH_STD_ATMO)
        orbit = OE(a=6800, e=0.001, i=np.radians(45),
                  omega=0, w=0, nu=0)
        
        params = {'Cd_A': 5.0, 'mass': 500.0}
        traj = sys.propagate(orbit, t_start=0, t_end=5400,
                           satellite_params=params)
        
        assert isinstance(traj, Trajectory)
        state_mid = traj(2700)
        assert isinstance(state_mid, OrbitalElements)
    
    def test_2body_j2_drag_smoke(self):
        """2-body with J2 + drag propagates successfully."""
        sys = System('2body', EARTH,
                    perturbations=('J2', 'drag'),
                    atmosphere=EARTH_STD_ATMO)
        orbit = OE(a=6800, e=0.001, i=np.radians(98),
                  omega=0, w=0, nu=0)
        
        params = {'Cd_A': 5.0, 'mass': 500.0}
        traj = sys.propagate(orbit, t_start=0, t_end=5400,
                           satellite_params=params)
        
        assert isinstance(traj, Trajectory)
        state_mid = traj(2700)
        assert isinstance(state_mid, OrbitalElements)
    
    def test_cr3bp_smoke(self):
        """CR3BP propagates successfully."""
        sys = System('3body', EARTH,
                    secondary_body=MOON,
                    distance=384400.0)
        
        # Use nondimensional CR3BP state
        state = np.array([0.8, 0.0, 0.0, 0.0, 0.1, 0.0])
        
        traj = sys.propagate(state, t_start=0, t_end=10)  # Nondim time
        
        assert isinstance(traj, Trajectory)
        state_mid = traj(5)
        assert isinstance(state_mid, OrbitalElements)
        assert state_mid.element_type == OEType.CR3BP  # should be CR3BP elements


class TestMultiplePropagations:
    """Test that same System can be reused for multiple propagations."""
    
    def test_same_system_multiple_propagations(self):
        """Can propagate multiple times with same System."""
        sys = System('2body', EARTH)
        orbit1 = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        orbit2 = OE(a=8000, e=0.02, i=np.radians(30), 
                   omega=0, w=0, nu=0)
        
        traj1 = sys.propagate(orbit1, t_start=0, t_end=100)
        traj2 = sys.propagate(orbit2, t_start=0, t_end=100)
        
        assert traj1 is not traj2
        assert traj1.system is sys
        assert traj2.system is sys
    
    def test_multiple_propagations_different_times(self):
        """Can propagate same orbit over different time spans."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        
        traj_short = sys.propagate(orbit, t_start=0, t_end=100)
        traj_long = sys.propagate(orbit, t_start=0, t_end=10000)
        
        assert traj_short.tf == 100
        assert traj_long.tf == 10000
    
    def test_propagations_independent(self):
        """Multiple propagations don't interfere with each other."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        
        traj1 = sys.propagate(orbit, t_start=0, t_end=100)
        state1 = traj1(50)
        
        traj2 = sys.propagate(orbit, t_start=0, t_end=200)
        state2 = traj2(50)
        
        # Should give same state at t=50
        assert np.allclose(state1.elements, state2.elements, rtol=1e-10)


class TestDifferentBodies:
    """Test propagation with different celestial bodies."""
    
    def test_moon_propagation(self):
        """Can propagate around Moon."""
        sys = System('2body', MOON)
        orbit = OE(a=2000, e=0.01, i=np.radians(45),
                  omega=0, w=0, nu=0, system=sys)
        
        traj = sys.propagate(orbit, t_start=0, t_end=1000)
        
        assert isinstance(traj, Trajectory)
    
    def test_mars_propagation(self):
        """Can propagate around Mars."""
        sys = System('2body', MARS)
        orbit = OE(a=5000, e=0.01, i=np.radians(45),
                  omega=0, w=0, nu=0, system=sys)
        
        traj = sys.propagate(orbit, t_start=0, t_end=1000)
        
        assert isinstance(traj, Trajectory)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
