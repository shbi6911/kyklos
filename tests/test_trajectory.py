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
    OE, OEType, Trajectory, Satellite, earth_2body
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
        """extend() works with drag systems requiring satellite input."""
        sys = System('2body', EARTH,
                    perturbations=('drag',),
                    atmosphere=EARTH_STD_ATMO)
        sat = Satellite.for_drag_only(100,11)
        orbit = OE(a=6800, e=0.001, i=0, omega=0, w=0, nu=0)
        
        traj1 = sys.propagate(orbit, t_start=0, t_end=100,
                            satellite = sat)
        traj2 = traj1.extend(200, satellite=sat)
        
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

class TestTrajectoryOutputIndependence:
    """
    Test that all Trajectory output methods return independent copies.
    
    This tests for the Heyoka buffer aliasing bug where consecutive
    scalar queries could overwrite each other if .copy() wasn't used.
    """
    
    @pytest.fixture
    def system(self):
        """Create a simple 2-body Earth system for testing."""
        return earth_2body()
    
    @pytest.fixture
    def initial_state(self):
        """Create initial conditions for a circular LEO orbit."""
        return OE(
            a=6378.0 + 400.0,  # 400 km altitude
            e=0.001,
            i=np.radians(51.6),
            omega=0.0,
            w=0.0,
            nu=0.0
        )
    
    @pytest.fixture
    def traj_with_stm(self, system, initial_state):
        """Trajectory with STM enabled for testing."""
        return system.propagate(initial_state, 0.0, 5400.0, with_stm=True)
    
    @pytest.fixture
    def traj_no_stm(self, system, initial_state):
        """Trajectory without STM for testing."""
        return system.propagate(initial_state, 0.0, 5400.0, with_stm=False)
    
    # ========== Scalar State Queries ==========
    
    def test_state_at_raw_independence(self, traj_no_stm):
        """Test state_at_raw returns independent copies."""
        t1, t2 = 0.0, 2700.0
        
        # Query at t1
        state1 = traj_no_stm.state_at_raw(t1)
        state1_original = state1.copy()  # Save for comparison
        
        # Query at t2 (should not modify state1)
        state2 = traj_no_stm.state_at_raw(t2)
        
        # Verify state1 unchanged
        np.testing.assert_array_equal(
            state1, state1_original,
            err_msg="state_at_raw(t1) was modified by subsequent query"
        )
        
        # Verify states are different (orbit has evolved)
        assert not np.allclose(state1, state2), \
            "States at different times should differ"
    
    def test_state_at_independence(self, traj_no_stm):
        """Test state_at returns independent OrbitalElements."""
        t1, t2 = 0.0, 2700.0
        
        # Query at t1
        oe1 = traj_no_stm.state_at(t1)
        oe1_elements_original = oe1.elements.copy()
        
        # Query at t2
        oe2 = traj_no_stm.state_at(t2)
        
        # Verify oe1.elements unchanged
        np.testing.assert_array_equal(
            oe1.elements, oe1_elements_original,
            err_msg="state_at(t1).elements was modified by subsequent query"
        )
        
        # Verify states are different
        assert not np.allclose(oe1.elements, oe2.elements)
    
    def test_state_full_independence(self, traj_with_stm):
        """Test state_full returns independent copies."""
        t1, t2 = 0.0, 2700.0
        
        # Query at t1
        full1 = traj_with_stm.state_full(t1)
        full1_original = full1.copy()
        
        # Query at t2
        full2 = traj_with_stm.state_full(t2)
        
        # Verify full1 unchanged
        np.testing.assert_array_equal(
            full1, full1_original,
            err_msg="state_full(t1) was modified by subsequent query"
        )
        
        # Verify states are different
        assert not np.allclose(full1, full2)
    
    def test_callable_interface_independence(self, traj_no_stm):
        """Test __call__ returns independent OrbitalElements."""
        t1, t2 = 0.0, 2700.0
        
        # Query at t1 using callable
        oe1 = traj_no_stm(t1)
        oe1_elements_original = oe1.elements.copy()
        
        # Query at t2
        oe2 = traj_no_stm(t2)
        
        # Verify oe1.elements unchanged
        np.testing.assert_array_equal(
            oe1.elements, oe1_elements_original,
            err_msg="traj(t1).elements was modified by subsequent query"
        )
        
        # Verify states are different
        assert not np.allclose(oe1.elements, oe2.elements)
    
    # ========== Array State Queries ==========
    
    def test_evaluate_raw_scalar_independence(self, traj_no_stm):
        """Test evaluate_raw with scalar input returns independent copy."""
        t1, t2 = 0.0, 2700.0
        
        # Scalar queries
        state1 = traj_no_stm.evaluate_raw(t1)
        state1_original = state1.copy()
        
        state2 = traj_no_stm.evaluate_raw(t2)
        
        # Verify state1 unchanged
        np.testing.assert_array_equal(
            state1, state1_original,
            err_msg="evaluate_raw(scalar) result was modified"
        )
        
        assert not np.allclose(state1, state2)
    
    def test_evaluate_raw_array_independence(self, traj_no_stm):
        """Test evaluate_raw with array input returns independent copy."""
        times1 = np.array([0.0, 1000.0])
        times2 = np.array([2000.0, 3000.0])
        
        # Array queries
        states1 = traj_no_stm.evaluate_raw(times1)
        states1_original = states1.copy()
        
        states2 = traj_no_stm.evaluate_raw(times2)
        
        # Verify states1 unchanged
        np.testing.assert_array_equal(
            states1, states1_original,
            err_msg="evaluate_raw(array) result was modified"
        )
        
        # Verify they're different
        assert not np.allclose(states1, states2)
    
    def test_evaluate_independence(self, traj_no_stm):
        """Test evaluate returns independent OrbitalElements list."""
        times = [0.0, 2700.0]
        
        # Get list of OrbitalElements
        oe_list = traj_no_stm.evaluate(times)
        oe0_elements_original = oe_list[0].elements.copy()
        
        # Subsequent query
        _ = traj_no_stm.evaluate([1000.0, 4000.0])
        
        # Verify first element unchanged
        np.testing.assert_array_equal(
            oe_list[0].elements, oe0_elements_original,
            err_msg="evaluate() OrbitalElements was modified"
        )
    
    def test_sample_raw_independence(self, traj_no_stm):
        """Test sample_raw returns independent copy."""
        # First sample
        states1 = traj_no_stm.sample_raw(n_points=5)
        states1_original = states1.copy()
        
        # Second sample (different n_points to get different data)
        states2 = traj_no_stm.sample_raw(n_points=10)
        
        # Verify states1 unchanged
        np.testing.assert_array_equal(
            states1, states1_original,
            err_msg="sample_raw() result was modified"
        )
    
    def test_sample_independence(self, traj_no_stm):
        """Test sample returns independent OrbitalElements list."""
        # First sample
        oe_list1 = traj_no_stm.sample(n_points=5)
        oe0_elements_original = oe_list1[0].elements.copy()
        
        # Second sample
        _ = traj_no_stm.sample(n_points=10)
        
        # Verify first element unchanged
        np.testing.assert_array_equal(
            oe_list1[0].elements, oe0_elements_original,
            err_msg="sample() OrbitalElements was modified"
        )
    
    def test_evaluate_full_scalar_independence(self, traj_with_stm):
        """Test evaluate_full with scalar returns independent copy."""
        t1, t2 = 0.0, 2700.0
        
        full1 = traj_with_stm.evaluate_full(t1)
        full1_original = full1.copy()
        
        full2 = traj_with_stm.evaluate_full(t2)
        
        np.testing.assert_array_equal(
            full1, full1_original,
            err_msg="evaluate_full(scalar) was modified"
        )
        
        assert not np.allclose(full1, full2)
    
    def test_evaluate_full_array_independence(self, traj_with_stm):
        """Test evaluate_full with array returns independent copy."""
        times1 = np.array([0.0, 1000.0])
        times2 = np.array([2000.0, 3000.0])
        
        full1 = traj_with_stm.evaluate_full(times1)
        full1_original = full1.copy()
        
        full2 = traj_with_stm.evaluate_full(times2)
        
        np.testing.assert_array_equal(
            full1, full1_original,
            err_msg="evaluate_full(array) was modified"
        )
    
    def test_sample_full_independence(self, traj_with_stm):
        """Test sample_full returns independent copy."""
        full1 = traj_with_stm.sample_full(n_points=5)
        full1_original = full1.copy()
        
        full2 = traj_with_stm.sample_full(n_points=10)
        
        np.testing.assert_array_equal(
            full1, full1_original,
            err_msg="sample_full() was modified"
        )
    
    # ========== STM-Specific Queries ==========
    
    def test_get_stm_independence(self, traj_with_stm):
        """Test get_stm returns independent copies (original bug)."""
        t1, t2 = 0.0, 2700.0
        
        # This is the original failing pattern
        stm1 = traj_with_stm.get_stm(t1)
        stm1_original = stm1.copy()
        
        stm2 = traj_with_stm.get_stm(t2)
        
        # Verify stm1 unchanged (this was failing before)
        np.testing.assert_array_equal(
            stm1, stm1_original,
            err_msg="get_stm(t1) was modified by get_stm(t2) - BUFFER ALIASING!"
        )
        
        # Verify STMs are different (STM evolves)
        assert not np.allclose(stm1, stm2)
        
        # Verify stm1 is still identity at t0
        np.testing.assert_allclose(stm1, np.eye(6), rtol=1e-14)
    
    def test_evaluate_stm_scalar_independence(self, traj_with_stm):
        """Test evaluate_stm with scalar returns independent copy."""
        t1, t2 = 0.0, 2700.0
        
        stm1 = traj_with_stm.evaluate_stm(t1)
        stm1_original = stm1.copy()
        
        stm2 = traj_with_stm.evaluate_stm(t2)
        
        np.testing.assert_array_equal(
            stm1, stm1_original,
            err_msg="evaluate_stm(scalar) was modified"
        )
        
        assert not np.allclose(stm1, stm2)
    
    def test_evaluate_stm_array_independence(self, traj_with_stm):
        """Test evaluate_stm with array returns independent copy."""
        times1 = np.array([0.0, 1000.0])
        times2 = np.array([2000.0, 3000.0])
        
        stms1 = traj_with_stm.evaluate_stm(times1)
        stms1_original = stms1.copy()
        
        stms2 = traj_with_stm.evaluate_stm(times2)
        
        np.testing.assert_array_equal(
            stms1, stms1_original,
            err_msg="evaluate_stm(array) was modified"
        )
    
    def test_sample_stm_independence(self, traj_with_stm):
        """Test sample_stm returns independent copy."""
        stms1 = traj_with_stm.sample_stm(n_points=5)
        stms1_original = stms1.copy()
        
        stms2 = traj_with_stm.sample_stm(n_points=10)
        
        np.testing.assert_array_equal(
            stms1, stms1_original,
            err_msg="sample_stm() was modified"
        )
    
    def test_mixed_query_pattern_independence(self, traj_with_stm):
        """Test mixing different query types doesn't cause aliasing."""
        # This tests the exact pattern that was failing
        stms_sampled = traj_with_stm.sample_stm(n_points=10)
        stms_sampled_original = stms_sampled.copy()
        
        # Mix of queries
        _ = traj_with_stm.get_stm(1000.0)
        _ = traj_with_stm.state_at_raw(2000.0)
        stm_scalar = traj_with_stm.get_stm(0.0)
        
        # Verify nothing was corrupted
        np.testing.assert_array_equal(
            stms_sampled, stms_sampled_original,
            err_msg="Mixed queries corrupted sample_stm result"
        )
        
        np.testing.assert_allclose(
            stm_scalar, np.eye(6), rtol=1e-14,
            err_msg="get_stm(0.0) corrupted after mixed queries"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
