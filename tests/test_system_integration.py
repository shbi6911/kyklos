"""
Test suite for System integration with other classes.

Tests cover:
- Integration with OrbitalElements (all element types)
- Integration with Trajectory (evaluation, properties)
- Round-trip workflows (OE → System → Trajectory → OE)
"""

import pytest
import numpy as np
from kyklos import (
    System, EARTH, MOON, EARTH_STD_ATMO,
    OE, OrbitalElements, OEType, Trajectory
)


class TestOrbitalElementsIntegration:
    """Test System integration with OrbitalElements."""
    
    def test_propagate_from_keplerian(self):
        """Can propagate from Keplerian elements."""
        sys = System('2body', EARTH)
        kep = OE(a=7000, e=0.01, i=np.radians(45),
                omega=0, w=0, nu=0)
        
        traj = sys.propagate(kep, t_start=0, t_end=100)
        
        assert isinstance(traj, Trajectory)
    
    def test_propagate_from_cartesian(self):
        """Can propagate from Cartesian elements."""
        sys = System('2body', EARTH)
        cart = OE(x=-6045, y=-3490, z=2500,
                 vx=-3.457, vy=6.618, vz=2.533)
        
        traj = sys.propagate(cart, t_start=0, t_end=100)
        
        assert isinstance(traj, Trajectory)
    
    def test_propagate_from_equinoctial(self):
        """Can propagate from Equinoctial elements."""
        sys = System('2body', EARTH)
        # Create via conversion
        kep = OE(a=7000, e=0.01, i=np.radians(45),
                omega=0, w=0, nu=0)
        equi = kep.to_equinoctial()
        
        traj = sys.propagate(equi, t_start=0, t_end=100)
        
        assert isinstance(traj, Trajectory)
    
    def test_orbital_elements_with_system_reference(self):
        """OrbitalElements can reference System for mu."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0, 
                  system=sys)
        
        assert orbit.system is sys
        assert orbit.mu == EARTH.mu
    
    def test_propagate_preserves_element_validity(self):
        """Propagation from valid elements produces valid trajectory."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=np.radians(45),
                  omega=0, w=0, nu=0)
        
        # Should not raise during propagation
        traj = sys.propagate(orbit, t_start=0, t_end=100)
        
        # Evaluate at arbitrary point
        state = traj(50)
        # Should be valid OrbitalElements
        assert isinstance(state, OrbitalElements)


class TestTrajectoryIntegration:
    """Test System integration with Trajectory."""
    
    def test_trajectory_state_at_returns_orbital_elements(self):
        """Trajectory.state_at() returns OrbitalElements."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        traj = sys.propagate(orbit, t_start=0, t_end=100)
        
        state = traj.state_at(50)
        
        assert isinstance(state, OrbitalElements)
    
    def test_trajectory_returns_cartesian_elements(self):
        """Trajectory returns Cartesian elements by default."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        traj = sys.propagate(orbit, t_start=0, t_end=100)
        
        state = traj.state_at(50)
        
        assert state.element_type == OEType.CARTESIAN
    
    def test_trajectory_callable_syntax(self):
        """Trajectory supports traj(t) syntax."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        traj = sys.propagate(orbit, t_start=0, t_end=100)
        
        state = traj(50)
        
        assert isinstance(state, OrbitalElements)
    
    def test_trajectory_evaluate_multiple_times(self):
        """Trajectory.evaluate() works with array of times."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        traj = sys.propagate(orbit, t_start=0, t_end=100)
        
        times = np.array([10, 30, 50, 70, 90])
        states = traj.evaluate(times)
        
        assert len(states) == 5
        assert all(isinstance(s, OrbitalElements) for s in states)
    
    def test_trajectory_sample(self):
        """Trajectory.sample() returns list of OrbitalElements."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        traj = sys.propagate(orbit, t_start=0, t_end=100)
        
        states = traj.sample(n_points=50)
        
        assert len(states) == 50
        assert all(isinstance(s, OrbitalElements) for s in states)
    
    def test_trajectory_has_system_reference(self):
        """Trajectory maintains reference to System."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        traj = sys.propagate(orbit, t_start=0, t_end=100)
        
        assert traj.system is sys


class TestRoundTripWorkflows:
    """Test complete workflows: OE → System → Trajectory → OE."""
    
    def test_keplerian_round_trip(self):
        """Can convert Keplerian → propagate → Keplerian."""
        sys = System('2body', EARTH)
        kep_initial = OE(a=7000, e=0.01, i=np.radians(45),
                        omega=0, w=0, nu=0)
        
        # Propagate
        traj = sys.propagate(kep_initial, t_start=0, t_end=100)
        
        # Get state at final time
        state_final = traj.state_at(100)
        print(f"DEBUG: state_final = {state_final}")
        # Convert back to Keplerian
        kep_final = state_final.to_keplerian()
        
        assert kep_final.element_type == OEType.KEPLERIAN
        assert isinstance(kep_final.a, float)
    
    def test_cartesian_round_trip(self):
        """Can work entirely in Cartesian coordinates."""
        sys = System('2body', EARTH)
        cart_initial = OE(x=-6045, y=-3490, z=2500,
                         vx=-3.457, vy=6.618, vz=2.533)
        
        traj = sys.propagate(cart_initial, t_start=0, t_end=100)
        state_final = traj.state_at(100)
        
        assert state_final.element_type == OEType.CARTESIAN
    
    def test_equinoctial_round_trip(self):
        """Can convert Equinoctial → propagate → Equinoctial."""
        sys = System('2body', EARTH)
        kep = OE(a=7000, e=0.01, i=np.radians(45),
                omega=0, w=0, nu=0)
        equi_initial = kep.to_equinoctial()
        
        traj = sys.propagate(equi_initial, t_start=0, t_end=100)
        state_final = traj.state_at(100)
        equi_final = state_final.to_equinoctial()
        
        assert equi_final.element_type == OEType.EQUINOCTIAL
    
    def test_multi_conversion_workflow(self):
        """Can chain conversions: Kep → propagate → Cart → Kep."""
        sys = System('2body', EARTH)
        kep_initial = OE(a=7000, e=0.01, i=np.radians(45),
                        omega=0, w=0, nu=0)
        
        # Propagate (returns Cartesian)
        traj = sys.propagate(kep_initial, t_start=0, t_end=100)
        cart_state = traj.state_at(50)
        
        # Convert to Keplerian
        kep_propagated = cart_state.to_keplerian()
        
        assert kep_propagated.element_type == OEType.KEPLERIAN
        # Semi-major axis should be approximately conserved (point mass)
        assert abs(kep_propagated.a - kep_initial.a) < 1.0  # Within 1 km


class TestElementTypePreservation:
    """Test that appropriate element types are used throughout."""
    
    def test_propagation_works_regardless_of_input_type(self):
        """Propagation works for any input element type."""
        sys = System('2body', EARTH)
        
        kep = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        cart = kep.to_cartesian()
        equi = kep.to_equinoctial()
        
        traj_kep = sys.propagate(kep, t_start=0, t_end=100)
        traj_cart = sys.propagate(cart, t_start=0, t_end=100)
        traj_equi = sys.propagate(equi, t_start=0, t_end=100)
        
        # All should produce valid trajectories
        assert isinstance(traj_kep, Trajectory)
        assert isinstance(traj_cart, Trajectory)
        assert isinstance(traj_equi, Trajectory)
    
    def test_trajectory_always_returns_cartesian(self):
        """Trajectory.state_at() always returns Cartesian."""
        sys = System('2body', EARTH)
        
        kep = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        cart = kep.to_cartesian()
        equi = kep.to_equinoctial()
        
        traj_kep = sys.propagate(kep, t_start=0, t_end=100)
        traj_cart = sys.propagate(cart, t_start=0, t_end=100)
        traj_equi = sys.propagate(equi, t_start=0, t_end=100)
        
        # All should return Cartesian
        assert traj_kep.state_at(50).element_type == OEType.CARTESIAN
        assert traj_cart.state_at(50).element_type == OEType.CARTESIAN
        assert traj_equi.state_at(50).element_type == OEType.CARTESIAN


class TestCR3BPIntegration:
    """Test CR3BP-specific integration patterns."""
    
    def test_cr3bp_propagation_workflow(self):
        """CR3BP propagation workflow."""
        sys = System('3body', EARTH,
                    secondary_body=MOON,
                    distance=384400.0)
        
        # Nondimensional state
        state = np.array([0.8, 0.0, 0.0, 0.0, 0.1, 0.0])
        
        traj = sys.propagate(state, t_start=0, t_end=10)
        state_final = traj.state_at(10)
        
        assert isinstance(state_final, OrbitalElements)
        # Trajectory should recognize System type and output accordingly
        assert state_final.element_type == OEType.CR3BP

    def test_cr3bp_with_cr3bp_orbital_elements(self):
        """CR3BP system accepts CR3BP OrbitalElements."""
        sys = System('3body', EARTH,
                    secondary_body=MOON,
                    distance=384400.0)
        
        # Create CR3BP OrbitalElements
        state = OE(x_nd=0.8, y_nd=0.0, z_nd=0.0,
                vx_nd=0.0, vy_nd=0.1, vz_nd=0.0,
                system=sys)
        
        # Should propagate successfully
        traj = sys.propagate(state, t_start=0, t_end=10)
        state_final = traj.state_at(10)
        
        assert isinstance(state_final, OrbitalElements)
        assert state_final.element_type == OEType.CR3BP

    def test_cr3bp_rejects_dimensional_orbital_elements(self):
        """CR3BP system rejects Cartesian/Keplerian OrbitalElements."""
        sys = System('3body', EARTH,
                    secondary_body=MOON,
                    distance=384400.0)
        
        # Try to use Cartesian elements (dimensional, wrong for CR3BP)
        state_cart = OE(x=0.8, y=0.8, z=0.0,
                    vx=0.0, vy=0.1, vz=0.0)
        
        with pytest.raises(ValueError, match="CR3BP.*Cartesian|nondimensional"):
            sys.propagate(state_cart, t_start=0, t_end=10)
        
        # Try to use Keplerian elements (also wrong for CR3BP)
        state_kep = OE(a=384400, e=0.01, i=0.1, 
                    omega=0, w=0, nu=0)
        
        with pytest.raises(ValueError, match="CR3BP.*Keplerian|nondimensional"):
            sys.propagate(state_kep, t_start=0, t_end=10)
    
    def test_cr3bp_jacobi_constant_accessible(self):
        """Can compute Jacobi constant for CR3BP states."""
        sys = System('3body', EARTH,
                    secondary_body=MOON,
                    distance=384400.0)
        
        # Create OrbitalElements with CR3BP type
        state = OE(x_nd=0.8, y_nd=0.0, z_nd=0.0,
                  vx_nd=0.0, vy_nd=0.1, vz_nd=0.0,
                  system=sys)
        
        # Should be able to compute Jacobi constant
        C = state.jacobi_const()
        assert isinstance(C, float)


class TestTrajectoryMethods:
    """Test Trajectory methods work correctly with System."""
    
    def test_trajectory_to_dataframe(self):
        """Trajectory.to_dataframe() produces valid DataFrame."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        traj = sys.propagate(orbit, t_start=0, t_end=100)
        
        df = traj.to_dataframe(n_points=50)
        
        assert len(df) == 50
        assert 'time' in df.columns
        assert 'x' in df.columns
        assert 'vx' in df.columns
    
    def test_trajectory_slice(self):
        """Trajectory.slice() creates valid sub-trajectory."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        traj = sys.propagate(orbit, t_start=0, t_end=100)
        
        traj_slice = traj.slice(25, 75)
        
        assert traj_slice.t0 == 25
        assert traj_slice.tf == 75
        assert traj_slice.system is sys
        
        # Can evaluate within slice bounds
        state = traj_slice(50)
        assert isinstance(state, OrbitalElements)
    
    def test_trajectory_contains_time(self):
        """Trajectory.contains_time() works correctly."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        traj = sys.propagate(orbit, t_start=0, t_end=100)
        
        assert traj.contains_time(50)
        assert not traj.contains_time(150)
        assert traj.contains_time(0)
        assert traj.contains_time(100)


class TestBatchOperations:
    """Test workflows with multiple orbits/trajectories."""
    
    def test_propagate_multiple_orbits_sequentially(self):
        """Can propagate multiple different orbits."""
        sys = System('2body', EARTH)
        
        orbits = [
            OE(a=7000, e=0.01, i=np.radians(i), omega=0, w=0, nu=0)
            for i in [0, 30, 60, 90]
        ]
        
        trajs = [sys.propagate(o, t_start=0, t_end=100) for o in orbits]
        
        assert len(trajs) == 4
        assert all(isinstance(t, Trajectory) for t in trajs)
        assert all(t.system is sys for t in trajs)
    
    def test_evaluate_all_trajectories_at_same_time(self):
        """Can evaluate multiple trajectories at same time."""
        sys = System('2body', EARTH)
        
        orbits = [
            OE(a=7000, e=0.01, i=np.radians(i), omega=0, w=0, nu=0)
            for i in [0, 30, 60]
        ]
        
        trajs = [sys.propagate(o, t_start=0, t_end=100) for o in orbits]
        
        # Evaluate all at t=50
        states = [t.state_at(50) for t in trajs]
        
        assert len(states) == 3
        assert all(isinstance(s, OrbitalElements) for s in states)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
