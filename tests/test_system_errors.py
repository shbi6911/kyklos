"""
Test suite for System edge cases and error handling.

Tests cover:
- Invalid initial states (NaN, Inf, unphysical)
- Extreme propagation times
- Trajectory evaluation outside bounds
- Reusing System for multiple propagations
- Error message clarity
"""

import pytest
import numpy as np
from kyklos import (
    System, EARTH, MOON, EARTH_STD_ATMO,
    OE, OrbitalElements, Trajectory
)


class TestInvalidInitialStates:
    """Test handling of invalid initial conditions."""
    
    def test_nan_in_state_array(self):
        """NaN in state array should be handled gracefully."""
        sys = System('2body', EARTH)
        state_with_nan = np.array([7000, 0, 0, np.nan, 0, 7.5])
        
        # Should either raise during validation or propagation
        with pytest.raises((ValueError, RuntimeError)):
            traj = sys.propagate(state_with_nan, t_start=0, t_end=100)
    
    def test_inf_in_state_array(self):
        """Inf in state array should be handled gracefully."""
        sys = System('2body', EARTH)
        state_with_inf = np.array([7000, np.inf, 0, 0, 0, 7.5])
        
        with pytest.raises((ValueError, RuntimeError)):
            traj = sys.propagate(state_with_inf, t_start=0, t_end=100)
    
    def test_zero_position_vector(self):
        """Zero position vector is unphysical."""
        sys = System('2body', EARTH)
        state = np.array([0, 0, 0, 1, 1, 1])
        
        # Should fail during propagation (singularity)
        with pytest.raises((ValueError, RuntimeError)):
            traj = sys.propagate(state, t_start=0, t_end=100)
    
    def test_extremely_large_position(self):
        """Extremely large position should work (hyperbolic)."""
        sys = System('2body', EARTH)
        # Far away position (escape trajectory)
        state = np.array([1e6, 0, 0, 0, 1, 0])
        
        # Should work (hyperbolic orbit)
        traj = sys.propagate(state, t_start=0, t_end=1000)
        assert isinstance(traj, Trajectory)
    
    def test_extremely_small_position(self):
        """Position inside body should fail or warn."""
        sys = System('2body', EARTH)
        # Position at 100 km (inside Earth)
        state = np.array([100, 0, 0, 0, 7, 0])
        
        # This might work numerically but is unphysical
        # At minimum should not crash
        try:
            traj = sys.propagate(state, t_start=0, t_end=10)
            # If it works, trajectory should exist
            assert isinstance(traj, Trajectory)
        except (ValueError, RuntimeError):
            # Or it might reasonably reject the state
            pass
    
    def test_invalid_orbital_elements_validation(self):
        """Invalid OrbitalElements caught during validation."""
        sys = System('2body', EARTH)
        
        # This should raise during OE construction
        with pytest.raises(ValueError):
            orbit = OE(a=-7000, e=0.01, i=0, omega=0, w=0, nu=0)


class TestExtremePropagationTimes:
    """Test propagation with unusual time configurations."""
    
    def test_zero_duration(self):
        """Propagation with t_start == t_end."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        
        traj = sys.propagate(orbit, t_start=100, t_end=100)
        
        assert traj.t0 == 100
        assert traj.tf == 100
        assert traj.duration == 0
        
        # Should be able to evaluate at the single time point
        state = traj.state_at(100)
        assert isinstance(state, OrbitalElements)
    
    def test_very_short_duration(self):
        """Propagation for very short time (microseconds)."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        
        traj = sys.propagate(orbit, t_start=0, t_end=1e-6)
        
        assert traj.duration == 1e-6
        state = traj.state_at(5e-7)
        assert isinstance(state, OrbitalElements)
    
    def test_very_long_duration(self):
        """Propagation for very long time (years)."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        
        # 1 year in seconds
        one_year = 365.25 * 24 * 3600
        
        traj = sys.propagate(orbit, t_start=0, t_end=one_year)
        
        assert traj.tf == one_year
        # Should be evaluable
        state = traj.state_at(one_year / 2)
        assert isinstance(state, OrbitalElements)
    
    def test_extremely_negative_times(self):
        """Propagation with large negative times."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        
        traj = sys.propagate(orbit, t_start=-1e6, t_end=-1e6 + 100)
        
        assert traj.t0 == -1e6
        assert traj.duration == 100


class TestTrajectoryEvaluationErrors:
    """Test error handling for trajectory evaluation."""
    
    def test_evaluate_before_t0(self):
        """Evaluating before t0 raises error."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        traj = sys.propagate(orbit, t_start=0, t_end=100)
        
        with pytest.raises(ValueError, match="outside trajectory bounds"):
            traj.state_at(-10)
    
    def test_evaluate_after_tf(self):
        """Evaluating after tf raises error."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        traj = sys.propagate(orbit, t_start=0, t_end=100)
        
        with pytest.raises(ValueError, match="outside trajectory bounds"):
            traj.state_at(150)
    
    def test_evaluate_at_boundaries(self):
        """Evaluating exactly at t0 and tf works."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        traj = sys.propagate(orbit, t_start=0, t_end=100)
        
        state_0 = traj.state_at(0)
        state_100 = traj.state_at(100)
        
        assert isinstance(state_0, OrbitalElements)
        assert isinstance(state_100, OrbitalElements)
    
    def test_slice_invalid_bounds(self):
        """Slice with invalid bounds raises error."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        traj = sys.propagate(orbit, t_start=0, t_end=100)
        
        # t_start >= t_end
        with pytest.raises(ValueError, match="must be <"):
            traj.slice(50, 50)
        
        # Outside trajectory bounds
        with pytest.raises(ValueError, match="outside trajectory"):
            traj.slice(-10, 50)
        
        with pytest.raises(ValueError, match="outside trajectory"):
            traj.slice(50, 150)
    
    def test_sample_with_invalid_n_points(self):
        """Sample with n_points < 2 raises error."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        traj = sys.propagate(orbit, t_start=0, t_end=100)
        
        with pytest.raises(ValueError, match="must be at least 2"):
            traj.sample(n_points=1)


class TestSystemReuse:
    """Test that System can be safely reused."""
    
    def test_sequential_propagations_same_orbit(self):
        """Can propagate same orbit multiple times."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        
        traj1 = sys.propagate(orbit, t_start=0, t_end=100)
        traj2 = sys.propagate(orbit, t_start=0, t_end=100)
        
        # Different trajectory objects
        assert traj1 is not traj2
        
        # But should give same results
        state1 = traj1.state_at(50)
        state2 = traj2.state_at(50)
        assert np.allclose(state1.elements, state2.elements)
    
    def test_interleaved_evaluations(self):
        """Can evaluate multiple trajectories in any order."""
        sys = System('2body', EARTH)
        orbit1 = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        orbit2 = OE(a=8000, e=0.02, i=np.radians(30), 
                   omega=0, w=0, nu=0)
        
        traj1 = sys.propagate(orbit1, t_start=0, t_end=100)
        traj2 = sys.propagate(orbit2, t_start=0, t_end=100)
        
        # Interleave evaluations
        s1_a = traj1.state_at(20)
        s2_a = traj2.state_at(30)
        s1_b = traj1.state_at(40)
        s2_b = traj2.state_at(60)
        
        # All should work
        assert all(isinstance(s, OrbitalElements) 
                  for s in [s1_a, s2_a, s1_b, s2_b])
    
    def test_propagate_after_long_idle(self):
        """Can propagate after System has been idle."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        
        # First propagation
        traj1 = sys.propagate(orbit, t_start=0, t_end=100)
        
        # Simulate idle time (in real usage, this might be minutes/hours)
        # Here we just propagate again
        traj2 = sys.propagate(orbit, t_start=0, t_end=100)
        
        assert isinstance(traj2, Trajectory)


class TestErrorMessageClarity:
    """Test that error messages are clear and helpful."""
    
    def test_missing_satellite_params_error_message(self):
        """Missing satellite params has clear error."""
        sys = System('2body', EARTH,
                    perturbations=('drag',),
                    atmosphere=EARTH_STD_ATMO)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        
        with pytest.raises(ValueError) as exc_info:
            sys.propagate(orbit, t_start=0, t_end=100)
        
        # Should mention which parameters are needed
        error_msg = str(exc_info.value)
        assert 'requires satellite parameters' in error_msg.lower() or \
               'Cd_A' in error_msg or 'mass' in error_msg
    
    def test_trajectory_bounds_error_message(self):
        """Out of bounds error has clear message."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        traj = sys.propagate(orbit, t_start=0, t_end=100)
        
        with pytest.raises(ValueError) as exc_info:
            traj.state_at(150)
        
        error_msg = str(exc_info.value)
        assert 'outside trajectory bounds' in error_msg.lower()
        # Should show the bounds
        assert '0' in error_msg and '100' in error_msg
    
    def test_invalid_perturbation_error_message(self):
        """Invalid perturbation has clear error."""
        with pytest.raises(ValueError) as exc_info:
            sys = System('2body', EARTH, 
                        perturbations=('invalid_pert',))
        
        error_msg = str(exc_info.value)
        assert 'unknown perturbation' in error_msg.lower()
        # Should suggest valid options
        assert 'J2' in error_msg or 'drag' in error_msg


class TestNumericalEdgeCases:
    """Test numerical edge cases in orbital mechanics."""
    
    def test_circular_orbit(self):
        """Perfectly circular orbit (e=0)."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.0, i=0, omega=0, w=0, nu=0)
        
        traj = sys.propagate(orbit, t_start=0, t_end=5400)
        state = traj.state_at(2700)
        
        assert isinstance(state, OrbitalElements)
    
    def test_equatorial_orbit(self):
        """Equatorial orbit (i=0)."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        
        traj = sys.propagate(orbit, t_start=0, t_end=5400)
        state = traj.state_at(2700)
        
        assert isinstance(state, OrbitalElements)
    
    def test_polar_orbit(self):
        """Polar orbit (i=90°)."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=np.radians(90), 
                  omega=0, w=0, nu=0)
        
        traj = sys.propagate(orbit, t_start=0, t_end=5400)
        state = traj.state_at(2700)
        
        assert isinstance(state, OrbitalElements)
    
    def test_high_eccentricity(self):
        """High eccentricity orbit (e=0.9)."""
        sys = System('2body', EARTH)
        orbit = OE(a=20000, e=0.9, i=np.radians(30),
                  omega=0, w=0, nu=0)
        
        traj = sys.propagate(orbit, t_start=0, t_end=10000)
        state = traj.state_at(5000)
        
        assert isinstance(state, OrbitalElements)
    
    def test_retrograde_orbit(self):
        """Retrograde orbit (i>90°)."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=np.radians(120),
                  omega=0, w=0, nu=0)
        
        traj = sys.propagate(orbit, t_start=0, t_end=5400)
        state = traj.state_at(2700)
        
        assert isinstance(state, OrbitalElements)


class TestCR3BPEdgeCases:
    """Test CR3BP-specific edge cases."""
    
    def test_near_primary(self):
        """State very close to primary body."""
        sys = System('3body', EARTH,
                    secondary_body=MOON,
                    distance=384400.0)
        
        # Near Earth (mu is mass ratio, Earth is at -mu)
        state = np.array([-sys.mass_ratio + 0.01, 0, 0, 0, 0.1, 0])
        
        try:
            traj = sys.propagate(state, t_start=0, t_end=1)
            assert isinstance(traj, Trajectory)
        except (ValueError, RuntimeError):
            # Might reasonably fail due to singularity
            pass
    
    def test_near_secondary(self):
        """State very close to secondary body."""
        sys = System('3body', EARTH,
                    secondary_body=MOON,
                    distance=384400.0)
        
        # Near Moon (Moon is at 1-mu)
        state = np.array([1 - sys.mass_ratio - 0.01, 0, 0, 0, 0.1, 0]) # type: ignore
        
        try:
            traj = sys.propagate(state, t_start=0, t_end=1)
            assert isinstance(traj, Trajectory)
        except (ValueError, RuntimeError):
            pass
    
    def test_far_from_both_bodies(self):
        """State far from both primary and secondary."""
        sys = System('3body', EARTH,
                    secondary_body=MOON,
                    distance=384400.0)
        
        # Far in z-direction
        state = np.array([0.5, 0, 2.0, 0, 0, 0.1])
        
        traj = sys.propagate(state, t_start=0, t_end=5)
        assert isinstance(traj, Trajectory)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
