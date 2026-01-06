"""
Test suite for Trajectory class.

Tests cover:
- Element type parameter (new feature)
- Raw array output methods
- Trajectory-specific methods (extend, get_times, __call__)
- Edge cases (backward propagation, zero duration)
- String representations
- Plotting functions (smoke tests)
"""

import pytest
import numpy as np
import plotly.graph_objects as go
from kyklos import (
    System, EARTH, MOON, EARTH_STD_ATMO,
    OE, OEType, Trajectory
)


class TestElementTypeParameter:
    """Test element_type parameter for state output methods."""
    
    def test_state_at_with_keplerian(self):
        """state_at() can return Keplerian elements."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        traj = sys.propagate(orbit, t_start=0, t_end=100)
        
        state = traj.state_at(50, element_type='kep')
        
        assert state.element_type == OEType.KEPLERIAN
        assert hasattr(state, 'a')
    
    def test_state_at_with_equinoctial(self):
        """state_at() can return Equinoctial elements."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        traj = sys.propagate(orbit, t_start=0, t_end=100)
        
        state = traj.state_at(50, element_type='equi')
        
        assert state.element_type == OEType.EQUINOCTIAL
    
    def test_state_at_with_enum(self):
        """state_at() accepts OEType enum."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        traj = sys.propagate(orbit, t_start=0, t_end=100)
        
        state = traj.state_at(50, element_type=OEType.KEPLERIAN)
        
        assert state.element_type == OEType.KEPLERIAN
    
    def test_state_at_auto_detect_2body(self):
        """state_at() auto-detects Cartesian for 2-body."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        traj = sys.propagate(orbit, t_start=0, t_end=100)
        
        state = traj.state_at(50)  # No element_type
        
        assert state.element_type == OEType.CARTESIAN
    
    def test_state_at_auto_detect_cr3bp(self):
        """state_at() auto-detects CR3BP for 3-body systems."""
        sys = System('3body', EARTH, secondary_body=MOON, distance=384400.0)
        state = np.array([0.8, 0.0, 0.0, 0.0, 0.1, 0.0])
        traj = sys.propagate(state, t_start=0, t_end=5)
        
        state = traj.state_at(2.5)  # No element_type
        
        assert state.element_type == OEType.CR3BP
    
    def test_evaluate_with_keplerian(self):
        """evaluate() can return Keplerian elements."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        traj = sys.propagate(orbit, t_start=0, t_end=100)
        
        times = np.array([25, 50, 75])
        states = traj.evaluate(times, element_type='kep')
        
        assert len(states) == 3
        assert all(s.element_type == OEType.KEPLERIAN for s in states)
    
    def test_evaluate_scalar_with_element_type(self):
        """evaluate() with scalar respects element_type."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        traj = sys.propagate(orbit, t_start=0, t_end=100)
        
        state = traj.evaluate(50, element_type='kep')
        
        assert state.element_type == OEType.KEPLERIAN
    
    def test_sample_with_equinoctial(self):
        """sample() can return Equinoctial elements."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        traj = sys.propagate(orbit, t_start=0, t_end=100)
        
        states = traj.sample(n_points=10, element_type='equi')
        
        assert len(states) == 10
        assert all(s.element_type == OEType.EQUINOCTIAL for s in states)
    
    def test_invalid_element_type_string(self):
        """Invalid element type string raises error."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        traj = sys.propagate(orbit, t_start=0, t_end=100)
        
        with pytest.raises(ValueError, match="Unknown element type"):
            traj.state_at(50, element_type='invalid')


class TestRawArrayMethods:
    """Test raw array output methods."""
    
    def test_state_at_raw_returns_1d_array(self):
        """state_at_raw() returns 1D array of shape (6,)."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        traj = sys.propagate(orbit, t_start=0, t_end=100)
        
        state = traj.state_at_raw(50)
        
        assert isinstance(state, np.ndarray)
        assert state.shape == (6,)
    
    def test_evaluate_raw_scalar_returns_1d(self):
        """evaluate_raw() with scalar returns 1D array."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        traj = sys.propagate(orbit, t_start=0, t_end=100)
        
        state = traj.evaluate_raw(50)
        
        assert isinstance(state, np.ndarray)
        assert state.shape == (6,)
    
    def test_evaluate_raw_array_returns_2d(self):
        """evaluate_raw() with array returns 2D array."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        traj = sys.propagate(orbit, t_start=0, t_end=100)
        
        times = np.array([25, 50, 75])
        states = traj.evaluate_raw(times)
        
        assert isinstance(states, np.ndarray)
        assert states.shape == (3, 6)
    
    def test_sample_raw_returns_2d_array(self):
        """sample_raw() returns 2D array."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        traj = sys.propagate(orbit, t_start=0, t_end=100)
        
        states = traj.sample_raw(n_points=20)
        
        assert isinstance(states, np.ndarray)
        assert states.shape == (20, 6)
    
    def test_raw_methods_return_finite_values(self):
        """Raw methods return finite (non-NaN, non-Inf) values."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        traj = sys.propagate(orbit, t_start=0, t_end=100)
        
        state_single = traj.state_at_raw(50)
        states_array = traj.evaluate_raw(np.array([25, 50, 75]))
        states_sample = traj.sample_raw(n_points=10)
        
        assert np.all(np.isfinite(state_single))
        assert np.all(np.isfinite(states_array))
        assert np.all(np.isfinite(states_sample))


class TestTrajectorySpecificMethods:
    """Test methods specific to Trajectory class."""
    
    def test_extend_creates_new_trajectory(self):
        """extend() creates a new Trajectory with correct times."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        traj1 = sys.propagate(orbit, t_start=0, t_end=100)
        
        traj2 = traj1.extend(200)
        
        assert isinstance(traj2, Trajectory)
        assert traj2.t0 == 100
        assert traj2.tf == 200
        assert traj2.duration == 100
    
    def test_extend_original_unchanged(self):
        """extend() doesn't modify original trajectory."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        traj1 = sys.propagate(orbit, t_start=0, t_end=100)
        
        traj2 = traj1.extend(200)
        
        assert traj1.tf == 100  # Original unchanged
        assert traj2.tf == 200
    
    def test_extend_validates_new_tf(self):
        """extend() requires new_tf > current tf."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        traj = sys.propagate(orbit, t_start=0, t_end=100)
        
        with pytest.raises(ValueError, match="must be >"):
            traj.extend(50)  # Backward
        
        with pytest.raises(ValueError, match="must be >"):
            traj.extend(100)  # Same time
    
    def test_extend_with_drag_system(self):
        """extend() works with drag systems requiring satellite_params."""
        sys = System('2body', EARTH,
                    perturbations=('drag',),
                    atmosphere=EARTH_STD_ATMO)
        orbit = OE(a=6800, e=0.001, i=0, omega=0, w=0, nu=0)
        params = {'Cd_A': 5.0, 'mass': 500.0}
        
        traj1 = sys.propagate(orbit, t_start=0, t_end=100,
                            satellite_params=params)
        traj2 = traj1.extend(200, satellite_params=params)
        
        assert traj2.tf == 200
    
    def test_get_times_returns_correct_array(self):
        """get_times() returns uniform time array."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        traj = sys.propagate(orbit, t_start=0, t_end=100)
        
        times = traj.get_times(n_points=11)
        
        assert isinstance(times, np.ndarray)
        assert len(times) == 11
        assert times[0] == 0
        assert times[-1] == 100
        assert np.allclose(np.diff(times), 10.0)  # Uniform spacing
    
    def test_callable_syntax(self):
        """__call__() syntax works as alias for state_at()."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        traj = sys.propagate(orbit, t_start=0, t_end=100)
        
        state1 = traj.state_at(50)
        state2 = traj(50)
        
        assert np.allclose(state1.elements, state2.elements)
    
    def test_callable_with_element_type(self):
        """__call__() accepts element_type parameter."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        traj = sys.propagate(orbit, t_start=0, t_end=100)
        
        state = traj(50, element_type='kep')
        
        assert state.element_type == OEType.KEPLERIAN


class TestEdgeCases:
    """Test edge cases in trajectory behavior."""
    
    def test_backward_propagation(self):
        """Trajectories with t_end < t_start work correctly."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        
        traj = sys.propagate(orbit, t_start=100, t_end=0)
        
        assert traj.t0 == 100
        assert traj.tf == 0
        assert traj.duration == -100
        
        # Can evaluate in backward time
        state = traj.state_at(50)
        assert isinstance(state, OE)
    
    def test_zero_duration_trajectory(self):
        """Trajectory with t_start == t_end."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        
        traj = sys.propagate(orbit, t_start=100, t_end=100)
        
        assert traj.t0 == 100
        assert traj.tf == 100
        assert traj.duration == 0
        
        # Can evaluate at the single time point
        state = traj.state_at(100)
        assert isinstance(state, OE)
    
    def test_negative_times(self):
        """Trajectory with negative times."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        
        traj = sys.propagate(orbit, t_start=-100, t_end=100)
        
        assert traj.t0 == -100
        assert traj.tf == 100
        
        state_neg = traj.state_at(-50)
        state_pos = traj.state_at(50)
        assert isinstance(state_neg, OE)
        assert isinstance(state_pos, OE)


class TestStringRepresentations:
    """Test string representations don't crash."""
    
    def test_repr_doesnt_crash(self):
        """__repr__() executes without error."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        traj = sys.propagate(orbit, t_start=0, t_end=100)
        
        repr_str = repr(traj)
        
        assert isinstance(repr_str, str)
        assert len(repr_str) > 0
    
    def test_str_doesnt_crash(self):
        """__str__() executes without error."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        traj = sys.propagate(orbit, t_start=0, t_end=100)
        
        str_repr = str(traj)
        
        assert isinstance(str_repr, str)
        assert len(str_repr) > 0
    
    def test_repr_contains_time_info(self):
        """__repr__() contains trajectory time information."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        traj = sys.propagate(orbit, t_start=0, t_end=100)
        
        repr_str = repr(traj)
        
        assert '0' in repr_str or 't0' in repr_str.lower()
        assert '100' in repr_str or 'tf' in repr_str.lower()


class TestPlotting:
    """Smoke tests for plotting functions."""
    
    def test_plot_3d_2body_returns_figure(self):
        """plot_3d() returns a valid Figure for 2-body."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=np.radians(45), 
                  omega=0, w=0, nu=0)
        traj = sys.propagate(orbit, t_start=0, t_end=5400)
        
        fig = traj.plot_3d(n_points=100)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0  # Has traces # type: ignore
    
    def test_plot_3d_2body_with_show_body_false(self):
        """plot_3d() with show_body=False."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        traj = sys.propagate(orbit, t_start=0, t_end=5400)
        
        fig = traj.plot_3d(n_points=100, show_body=False)
        
        assert isinstance(fig, go.Figure)
    
    def test_plot_3d_cr3bp_returns_figure(self):
        """plot_3d() returns valid Figure for CR3BP."""
        sys = System('3body', EARTH, secondary_body=MOON, distance=384400.0)
        state = np.array([0.8, 0.0, 0.0, 0.0, 0.1, 0.0])
        traj = sys.propagate(state, t_start=0, t_end=10)
        
        fig = traj.plot_3d(n_points=100)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0 # type: ignore
    
    def test_plot_3d_cr3bp_with_show_body_false(self):
        """plot_3d() for CR3BP with show_body=False."""
        sys = System('3body', EARTH, secondary_body=MOON, distance=384400.0)
        state = np.array([0.8, 0.0, 0.0, 0.0, 0.1, 0.0])
        traj = sys.propagate(state, t_start=0, t_end=10)
        
        fig = traj.plot_3d(n_points=100, show_body=False)
        
        assert isinstance(fig, go.Figure)
    
    def test_add_to_plot_doesnt_crash(self):
        """add_to_plot() executes without error."""
        sys = System('2body', EARTH)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        traj = sys.propagate(orbit, t_start=0, t_end=5400)
        
        # Create base figure
        fig = go.Figure()
        
        # Add trajectory
        result = traj.add_to_plot(fig, n_points=100, color='blue', name='Test')
        
        assert result is fig  # Returns same figure
        assert len(fig.data) > 0  # Added trace # type: ignore
    
    def test_add_to_plot_with_multiple_trajectories(self):
        """add_to_plot() can add multiple trajectories to same figure."""
        sys = System('2body', EARTH)
        orbit1 = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        orbit2 = OE(a=8000, e=0.02, i=np.radians(30), 
                   omega=0, w=0, nu=0)
        
        traj1 = sys.propagate(orbit1, t_start=0, t_end=5400)
        traj2 = sys.propagate(orbit2, t_start=0, t_end=5400)
        
        fig = go.Figure()
        traj1.add_to_plot(fig, color='red', name='Orbit 1')
        traj2.add_to_plot(fig, color='blue', name='Orbit 2')
        
        assert len(fig.data) == 2  # Two trajectories # type: ignore


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
