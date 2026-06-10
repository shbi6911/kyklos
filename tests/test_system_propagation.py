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
    OE, OrbitalElements, OEType, Trajectory, Satellite
)
from kyklos.trajectory import (
    StartBoundaryNode,
    EndBoundaryNode,
    NullJunctionNode,
)

class TestPropagationInterface:
    """Test basic propagation interface contracts."""
    
    def test_accepts_orbital_elements_input(self):
        """propagate() accepts OrbitalElements."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        
        traj = sys.propagate(orbit, times=[0, 100])
        
        assert isinstance(traj, Trajectory)
    
    def test_accepts_numpy_array_input(self):
        """propagate() accepts numpy array."""
        sys = System('2body', EARTH)
        state = np.array([-6045, -3490, 2500, -3.457, 6.618, 2.533])
        
        traj = sys.propagate(state, times=[0, 100])
        
        assert isinstance(traj, Trajectory)
    
    def test_accepts_keplerian_elements(self):
        """propagate() works with Keplerian elements."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=np.radians(45), 
                  omega=0, w=0, nu=0)
        
        traj = sys.propagate(orbit, times=[0, 100])
        
        assert isinstance(traj, Trajectory)
    
    def test_accepts_equinoctial_elements(self):
        """propagate() works with Equinoctial elements."""
        sys = System('2body', EARTH)
        # Create via conversion to ensure valid equinoctial
        kep = OE(a=7000, e=0.01, i=np.radians(45), omega=0, w=0, nu=0)
        equi = kep.to_equinoctial()
        
        traj = sys.propagate(equi, times=[0, 100])
        
        assert isinstance(traj, Trajectory)
    
    def test_returns_trajectory_object(self):
        """propagate() returns Trajectory instance."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        
        traj = sys.propagate(orbit, times=[0, 100])
        
        assert isinstance(traj, Trajectory)
        assert hasattr(traj, 't0')
        assert hasattr(traj, 'tf')
        assert hasattr(traj, 'system')
    
    def test_trajectory_has_correct_times(self):
        """Returned Trajectory has correct t0 and tf."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        
        traj = sys.propagate(orbit, times=[10, 500])
        
        assert traj.t0 == 10.0
        assert traj.tf == 500.0
    
    def test_trajectory_references_system(self):
        """Returned Trajectory references the System."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        
        traj = sys.propagate(orbit, times=[0, 100])
        
        assert traj.system is sys


class TestParameterHandling:
    """Test satellite parameter handling in propagation."""
    
    def test_no_params_system_propagates_without_params(self):
        """System without perturbations doesn't need satellite_params."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        
        # Should work without satellite_params
        traj = sys.propagate(orbit, times=[0, 100])
        
        assert traj is not None
    
    def test_drag_system_requires_params(self):
        """System with drag requires satellite_params."""
        sys = System('2body', EARTH,
                    perturbations=('drag',),
                    atmosphere=EARTH_STD_ATMO)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        
        with pytest.raises(ValueError, match="requires a Satellite object"):
            sys.propagate(orbit, times=[0, 100])
    
    def test_drag_system_accepts_satellite(self):
        """System with drag accepts dict satellite_params."""
        sys = System('2body', EARTH,
                    perturbations=('drag',),
                    atmosphere=EARTH_STD_ATMO)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        
        sat = Satellite.for_drag_only(100,11)
        traj = sys.propagate(orbit, times=[0, 100],
                           satellite=sat)
        
        assert traj is not None

class TestTimeHandling:
    """Test propagation with different time configurations."""
    
    def test_forward_propagation(self):
        """Forward propagation (t_end > t_start) works."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        
        traj = sys.propagate(orbit, times=[0, 1000])
        
        assert traj.t0 == 0.0
        assert traj.tf == 1000.0
    
    def test_backward_propagation_raises(self):
        """propagate() no longer accepts backward time arrays."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        with pytest.raises(ValueError, match="strictly increasing"):
            sys.propagate(orbit, [1000, 0])
    
    def test_negative_times(self):
        """Propagation works with negative times."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        
        traj = sys.propagate(orbit, times=[-500, 500])
        
        assert traj.t0 == -500
        assert traj.tf == 500
    
    def test_large_time_values(self):
        """Propagation works with large time values."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        
        traj = sys.propagate(orbit, times=[1e6, 1e6 + 1000])
        
        assert traj.t0 == 1e6
        assert traj.tf == 1e6 + 1000


class TestSmokeTests:
    """Smoke tests: verify each system type can propagate."""
    
    def test_2body_point_mass_smoke(self):
        """2-body point mass propagates successfully."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=np.radians(45), 
                  omega=0, w=0, nu=0)
        
        traj = sys.propagate(orbit, times=[0, 5400])
        
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
        
        traj = sys.propagate(orbit, times=[0, 5400])
        
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
        
        sat = Satellite.for_drag_only(100,11)
        traj = sys.propagate(orbit, times=[0, 5400],
                           satellite=sat)
        
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
        
        sat = Satellite.for_drag_only(100,11)
        traj = sys.propagate(orbit, times=[0, 5400],
                           satellite=sat)
        
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
        
        traj = sys.propagate(state, times=[0, 10])  # Nondim time
        
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
        
        traj1 = sys.propagate(orbit1, times=[0, 100])
        traj2 = sys.propagate(orbit2, times=[0, 100])
        
        assert traj1 is not traj2
        assert traj1.system is sys
        assert traj2.system is sys
    
    def test_multiple_propagations_different_times(self):
        """Can propagate same orbit over different time spans."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        
        traj_short = sys.propagate(orbit, times=[0, 100])
        traj_long = sys.propagate(orbit, times=[0, 100000])
        
        assert traj_short.tf == 100.0
        assert traj_long.tf == 100000.0
    
    def test_propagations_independent(self):
        """Multiple propagations don't interfere with each other."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        
        traj1 = sys.propagate(orbit, times=[0, 100])
        state1 = traj1(50)
        
        traj2 = sys.propagate(orbit, times=[0, 200])
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
        
        traj = sys.propagate(orbit, times=[0, 1000])
        
        assert isinstance(traj, Trajectory)
    
    def test_mars_propagation(self):
        """Can propagate around Mars."""
        sys = System('2body', MARS)
        orbit = OE(a=5000, e=0.01, i=np.radians(45),
                  omega=0, w=0, nu=0, system=sys)
        
        traj = sys.propagate(orbit, times=[0, 1000])
        
        assert isinstance(traj, Trajectory)

class TestNewPropagationModes:
    """Test multi-segment and node-based propagation interfaces."""

    # --- Mutual exclusivity ---

    def test_nodes_and_initial_state_raises(self):
        """Providing both nodes and initial_state raises ValueError."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        state = np.array([7000.0, 0.0, 0.0, 0.0, 7.546, 0.0])
        start = StartBoundaryNode(0.0, state)
        end   = EndBoundaryNode(100.0, state)

        with pytest.raises(ValueError, match="Cannot provide both"):
            sys.propagate(orbit, nodes=[start, end])

    def test_nodes_and_times_raises(self):
        """Providing both nodes and times raises ValueError."""
        sys = System('2body', EARTH)
        state = np.array([7000.0, 0.0, 0.0, 0.0, 7.546, 0.0])
        start = StartBoundaryNode(0.0, state)
        end   = EndBoundaryNode(100.0, state)

        with pytest.raises(ValueError, match="[Cc]annot provide times"):
            sys.propagate(nodes=[start, end], times=[0, 100])

    def test_neither_nodes_nor_initial_state_raises(self):
        """Omitting both nodes and initial_state raises ValueError."""
        sys = System('2body', EARTH)

        with pytest.raises(ValueError, match="[Mm]ust provide"):
            sys.propagate(times=[0, 100])

    # --- times array validation ---

    def test_times_length_one_raises(self):
        """times of length 1 raises ValueError."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)

        with pytest.raises(ValueError, match="length >= 2"):
            sys.propagate(orbit, [0.0])

    def test_non_increasing_times_raises(self):
        """Non-monotonic times raises ValueError."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)

        with pytest.raises(ValueError, match="strictly increasing"):
            sys.propagate(orbit, [100.0, 0.0])

    # --- Mode 1: multi-segment ---

    def test_mode1_list_of_states(self):
        """Mode 1: list of states + times produces multi-segment Trajectory."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)

        ref = sys.propagate(orbit, [0, 200])
        s0  = ref.state_at_raw(0)
        s1  = ref.state_at_raw(100)

        traj = sys.propagate([s0, s1], [0, 100, 200])

        assert isinstance(traj, Trajectory)
        assert traj.n_segments == 2
        assert traj.t0 == 0
        assert traj.tf == 200

    def test_mode1_2d_numpy_array(self):
        """Mode 1: 2D numpy array of shape (n_seg, 6) works."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)

        ref    = sys.propagate(orbit, [0, 200])
        states = np.vstack([ref.state_at_raw(0), ref.state_at_raw(100)])

        traj = sys.propagate(states, [0, 100, 200])

        assert isinstance(traj, Trajectory)
        assert traj.n_segments == 2

    def test_mode1_length_mismatch_raises(self):
        """Mode 1: len(initial_states) != len(times) - 1 raises ValueError."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        s = sys.propagate(orbit, [0, 100]).state_at_raw(0)

        with pytest.raises(ValueError, match="[Ss]egment"):
            sys.propagate([s, s], [0, 100, 200, 300])

    def test_mode1_single_segment_via_list(self):
        """Mode 1: single-element list with two times is valid."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        s = sys.propagate(orbit, [0, 100]).state_at_raw(0)

        traj = sys.propagate([s], [0, 100])

        assert isinstance(traj, Trajectory)
        assert traj.n_segments == 1

    # --- Mode 2: node-based ---

    def test_mode2_basic_smoke(self):
        """Mode 2: two-node list produces a valid single-segment Trajectory."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        ref   = sys.propagate(orbit, [0, 100])

        start = StartBoundaryNode(0.0,   ref.state_at_raw(0))
        end   = EndBoundaryNode(100.0,   ref.state_at_raw(100))

        traj = sys.propagate(nodes=[start, end])

        assert isinstance(traj, Trajectory)
        assert traj.t0 == 0.0
        assert traj.tf == 100.0

    def test_mode2_preserves_start_node(self):
        """Mode 2: input start_node is preserved on the returned Trajectory."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        ref   = sys.propagate(orbit, [0, 100])

        start = StartBoundaryNode(0.0,   ref.state_at_raw(0))
        end   = EndBoundaryNode(100.0,   ref.state_at_raw(100))

        traj = sys.propagate(nodes=[start, end])

        assert traj.start_node is start

    def test_mode2_two_segments(self):
        """Mode 2: three-node list produces a two-segment Trajectory."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        ref   = sys.propagate(orbit, [0, 200])

        s0  = ref.state_at_raw(0)
        s1  = ref.state_at_raw(100)
        s2  = ref.state_at_raw(200)

        start   = StartBoundaryNode(0.0,   s0)
        junc    = NullJunctionNode(100.0,  s1, s1.copy())
        end     = EndBoundaryNode(200.0,   s2)

        traj = sys.propagate(nodes=[start, junc, end])

        assert traj.n_segments == 2
        assert traj.t0 == 0.0
        assert traj.tf == 200.0

    def test_mode2_wrong_first_node_raises(self):
        """Mode 2: non-BoundaryNode first element raises ValueError."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        s     = sys.propagate(orbit, [0, 100]).state_at_raw(0)

        junc = NullJunctionNode(0.0, s, s.copy())
        end  = EndBoundaryNode(100.0, s)

        with pytest.raises(ValueError, match="[Ss]tart"):
            sys.propagate(nodes=[junc, end])

    def test_mode2_too_few_nodes_raises(self):
        """Mode 2: single-element nodes list raises ValueError."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        s     = sys.propagate(orbit, [0, 100]).state_at_raw(0)

        with pytest.raises(ValueError, match="[Aa]t least 2"):
            sys.propagate(nodes=[StartBoundaryNode(0.0, s)])

    # --- BoundaryNode as initial_state ---

    def test_boundary_node_as_initial_state(self):
        """StartBoundaryNode accepted as initial_state."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        s     = sys.propagate(orbit, [0, 100]).state_at_raw(0)
        start = StartBoundaryNode(0.0, s)

        traj = sys.propagate(start, [0.0, 100.0])

        assert isinstance(traj, Trajectory)
        assert traj.start_node is start

    def test_boundary_node_time_mismatch_raises(self):
        """BoundaryNode time not matching times[0] raises ValueError."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        s     = sys.propagate(orbit, [0, 100]).state_at_raw(0)
        start = StartBoundaryNode(50.0, s)   # node says t=50, times[0]=0

        with pytest.raises(ValueError, match="[Tt]ime"):
            sys.propagate(start, [0.0, 100.0])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
