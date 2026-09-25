"""
Test suite for Trajectory class.

Tests cover:
- Element type parameter
- Raw array output methods
- Trajectory-specific methods (extend, get_times, __call__)
- Edge cases
- String representations
- Plotting functions (smoke tests)
"""

import pytest
import numpy as np
import plotly.graph_objects as go
from kyklos import (
    System, earth, moon, EARTH_STD_ATMO, config,
    OE, OEType, Trajectory, Satellite, earth_2body
)
from kyklos.trajectory import (FreeJunctionNode, _DEFAULT_HOVER, 
                               _figure_line_positions)


# ========== FIXTURES AND HELPERS ==========
 
@pytest.fixture(scope="module")
def em_system():
    return System('3body', earth(), moon(), distance=384400.0)
 
 
@pytest.fixture(scope="module")
def tb_system():
    return System('2body', earth())
 
 
@pytest.fixture(scope="module")
def cr3bp_traj(em_system):
    """Very short arc near x = 0.8: far from both bodies on its own."""
    state = np.array([0.8, 0.0, 0.0, 0.0, 0.1, 0.0])
    return em_system.propagate(state, times=[0, 0.2])
 
 
@pytest.fixture(scope="module")
def tb_traj(tb_system):
    orbit = OE(a=7000, e=0.01, i=np.radians(30), omega=0, w=0, nu=0)
    return tb_system.propagate(orbit, times=[0, 5400])
 
 
def _ring(center_x, radius, n=400):
    """Planar circle in the x-y plane about (center_x, 0, 0)."""
    theta = np.linspace(0.0, 2.0 * np.pi, n)
    return np.column_stack([
        center_x + radius * np.cos(theta),
        radius * np.sin(theta),
        np.zeros(n),
    ])
 
 
def _line_figure(positions):
    """Figure holding one line trace, as a plotted trajectory would."""
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=positions[:, 0], y=positions[:, 1], z=positions[:, 2],
        mode='lines',
    ))
    return fig
 
 
def _surfaces(fig):
    return [t for t in fig.data if isinstance(t, go.Surface)]
 
 
# Sized against Earth-Moon: Moon radius ~0.00452 nd, so the default
# 10-radius proximity band is ~0.045 nd (~17,000 km).
L1_X = 0.8369
 
 
def _moon_x(system):
    return 1.0 - system.mass_ratio


class TestElementTypeParameter:
    """Test element_type parameter for state output methods."""
    
    def test_state_at_with_keplerian(self):
        """state_at() can return Keplerian elements."""
        sys = System('2body', earth())
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        traj = sys.propagate(orbit, times=[0,100])
        
        state = traj.state_at(50, element_type='kep')
        
        assert state.element_type == OEType.KEPLERIAN
        assert hasattr(state, 'a')
    
    def test_state_at_with_equinoctial(self):
        """state_at() can return Equinoctial elements."""
        sys = System('2body', earth())
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        traj = sys.propagate(orbit, times=[0,100])
        
        state = traj.state_at(50, element_type='equi')
        
        assert state.element_type == OEType.EQUINOCTIAL
    
    def test_state_at_with_enum(self):
        """state_at() accepts OEType enum."""
        sys = System('2body', earth())
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        traj = sys.propagate(orbit, times=[0,100])
        
        state = traj.state_at(50, element_type=OEType.KEPLERIAN)
        
        assert state.element_type == OEType.KEPLERIAN
    
    def test_state_at_auto_detect_2body(self):
        """state_at() auto-detects Cartesian for 2-body."""
        sys = System('2body', earth())
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        traj = sys.propagate(orbit, times=[0,100])
        
        state = traj.state_at(50)  # No element_type
        
        assert state.element_type == OEType.CARTESIAN
    
    def test_state_at_auto_detect_cr3bp(self):
        """state_at() auto-detects CR3BP for 3-body systems."""
        sys = System('3body', earth(), secondary_body=moon(), distance=384400.0)
        state = np.array([0.8, 0.0, 0.0, 0.0, 0.1, 0.0])
        traj = sys.propagate(state, times=[0,5])
        
        state = traj.state_at(2.5)  # No element_type
        
        assert state.element_type == OEType.CR3BP
    
    def test_evaluate_with_keplerian(self):
        """evaluate() can return Keplerian elements."""
        sys = System('2body', earth())
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        traj = sys.propagate(orbit, times=[0,100])
        
        times = np.array([25, 50, 75])
        states = traj.evaluate(times, element_type='kep')
        
        assert len(states) == 3
        assert all(s.element_type == OEType.KEPLERIAN for s in states)
    
    def test_evaluate_scalar_with_element_type(self):
        """evaluate() with scalar respects element_type."""
        sys = System('2body', earth())
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        traj = sys.propagate(orbit, times=[0,100])
        
        state = traj.evaluate(50, element_type='kep')
        
        assert state.element_type == OEType.KEPLERIAN
    
    def test_sample_with_equinoctial(self):
        """sample() can return Equinoctial elements."""
        sys = System('2body', earth())
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        traj = sys.propagate(orbit, times=[0,100])
        
        states = traj.sample(n_points=10, element_type='equi')
        
        assert len(states) == 10
        assert all(s.element_type == OEType.EQUINOCTIAL for s in states)
    
    def test_invalid_element_type_string(self):
        """Invalid element type string raises error."""
        sys = System('2body', earth())
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        traj = sys.propagate(orbit, times=[0,100])
        
        with pytest.raises(ValueError, match="Unknown element type"):
            traj.state_at(50, element_type='invalid')


class TestRawArrayMethods:
    """Test raw array output methods."""
    
    def test_state_at_raw_returns_1d_array(self):
        """state_at_raw() returns 1D array of shape (6,)."""
        sys = System('2body', earth())
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        traj = sys.propagate(orbit, times=[0,100])
        
        state = traj.state_at_raw(50)
        
        assert isinstance(state, np.ndarray)
        assert state.shape == (6,)
    
    def test_evaluate_raw_scalar_returns_1d(self):
        """evaluate_raw() with scalar returns 1D array."""
        sys = System('2body', earth())
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        traj = sys.propagate(orbit, times=[0,100])
        
        state = traj.evaluate_raw(50)
        
        assert isinstance(state, np.ndarray)
        assert state.shape == (6,)
    
    def test_evaluate_raw_array_returns_2d(self):
        """evaluate_raw() with array returns 2D array."""
        sys = System('2body', earth())
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        traj = sys.propagate(orbit, times=[0,100])
        
        times = np.array([25, 50, 75])
        states = traj.evaluate_raw(times)
        
        assert isinstance(states, np.ndarray)
        assert states.shape == (3, 6)
    
    def test_sample_raw_returns_2d_array(self):
        """sample_raw() returns 2D array."""
        sys = System('2body', earth())
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        traj = sys.propagate(orbit, times=[0,100])
        
        states = traj.sample_raw(n_points=20)
        
        assert isinstance(states, np.ndarray)
        assert states.shape == (20, 6)
    
    def test_raw_methods_return_finite_values(self):
        """Raw methods return finite (non-NaN, non-Inf) values."""
        sys = System('2body', earth())
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        traj = sys.propagate(orbit,times=[0,100])
        
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
        sys = System('2body', earth())
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        traj1 = sys.propagate(orbit, times=[0,100])
        
        traj2 = traj1.extend(200)
        
        assert isinstance(traj2, Trajectory)
        assert traj2.t0 == 0
        assert traj2.tf == 200
        assert traj2.n_segments == 2
        assert traj2.duration == 200
    
    def test_extend_original_unchanged(self):
        """extend() doesn't modify original trajectory."""
        sys = System('2body', earth())
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        traj1 = sys.propagate(orbit, times=[0,100])
        
        traj2 = traj1.extend(200)
        
        assert traj1.tf == 100  # Original unchanged
        assert traj2.tf == 200
    
    def test_extend_validates_new_tf(self):
        """extend() requires new_tf > current tf."""
        sys = System('2body', earth())
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        traj = sys.propagate(orbit, times=[0,100])
        
        with pytest.raises(ValueError, match="greater than"):
            traj.extend(50)  # Backward
        
        with pytest.raises(ValueError, match="greater than"):
            traj.extend(100)  # Same time
    
    def test_extend_with_drag_system(self):
        """extend() works with drag systems requiring satellite input."""
        sys = System('2body', earth(),
                    perturbations=('drag',),
                    atmosphere=EARTH_STD_ATMO)
        sat = Satellite.for_drag_only(100,11)
        orbit = OE(a=6800, e=0.001, i=0, omega=0, w=0, nu=0)
        
        traj1 = sys.propagate(orbit, times=[0,100],
                            satellite = sat)
        traj2 = traj1.extend(200, satellite=sat)
        
        assert traj2.tf == 200
    
    def test_get_times_returns_correct_array(self):
        """get_times() returns uniform time array."""
        sys = System('2body', earth())
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        traj = sys.propagate(orbit, times=[0,100])
        
        times = traj.get_times(n_points=11)
        
        assert isinstance(times, np.ndarray)
        assert len(times) == 11
        assert times[0] == 0
        assert times[-1] == 100
        assert np.allclose(np.diff(times), 10.0)  # Uniform spacing
    
    def test_callable_syntax(self):
        """__call__() syntax works as alias for state_at()."""
        sys = System('2body', earth())
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        traj = sys.propagate(orbit, times=[0,100])
        
        state1 = traj.state_at(50)
        state2 = traj(50)
        
        assert np.allclose(state1.elements, state2.elements)
    
    def test_callable_with_element_type(self):
        """__call__() accepts element_type parameter."""
        sys = System('2body', earth())
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        traj = sys.propagate(orbit, times=[0,100])
        
        state = traj(50, element_type='kep')
        
        assert state.element_type == OEType.KEPLERIAN


class TestEdgeCases:
    """Test edge cases in trajectory behavior."""
    
    def test_backward_propagation_raises(self):
        """Backward propagation via propagate() now raises ValueError.
        Use Trajectory.extend_back() for backward extension."""
        sys = System('2body', earth())
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)

        with pytest.raises(ValueError, match="strictly increasing"):
            sys.propagate(orbit, [1000, 0])
    
    def test_zero_duration_raises(self):
        """Trajectory with t_start == t_end."""
        sys = System('2body', earth())
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        
        with pytest.raises(ValueError, match="strictly increasing"):
            traj = sys.propagate(orbit, times=[100,100])
    
    def test_negative_times(self):
        """Trajectory with negative times."""
        sys = System('2body', earth())
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        
        traj = sys.propagate(orbit, times=[-100,100])
        
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
        sys = System('2body', earth())
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        traj = sys.propagate(orbit, times=[0,100])
        
        repr_str = repr(traj)
        
        assert isinstance(repr_str, str)
        assert len(repr_str) > 0
    
    def test_str_doesnt_crash(self):
        """__str__() executes without error."""
        sys = System('2body', earth())
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        traj = sys.propagate(orbit, times=[0,100])
        
        str_repr = str(traj)
        
        assert isinstance(str_repr, str)
        assert len(str_repr) > 0
    
    def test_repr_contains_time_info(self):
        """__repr__() contains trajectory time information."""
        sys = System('2body', earth())
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        traj = sys.propagate(orbit, times=[0,100])
        
        repr_str = repr(traj)
        
        assert '0' in repr_str or 't0' in repr_str.lower()
        assert '100' in repr_str or 'tf' in repr_str.lower()


class TestPlotting:
    """Smoke tests for plotting functions."""
    
    def test_plot_3d_2body_returns_figure(self):
        """plot_3d() returns a valid Figure for 2-body."""
        sys = System('2body', earth())
        orbit = OE(a=7000, e=0.01, i=np.radians(45), 
                  omega=0, w=0, nu=0)
        traj = sys.propagate(orbit, times=[0,5400])
        
        fig = traj.plot_3d(n_points=100)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0  # Has traces # type: ignore
    
    def test_plot_3d_2body_with_show_body_false(self):
        """plot_3d() with show_body=False."""
        sys = System('2body', earth())
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        traj = sys.propagate(orbit, times=[0,5400])
        
        fig = traj.plot_3d(n_points=100, bodies=False)
        
        assert isinstance(fig, go.Figure)
    
    def test_plot_3d_cr3bp_returns_figure(self):
        """plot_3d() returns valid Figure for CR3BP."""
        sys = System('3body', earth(), secondary_body=moon(), distance=384400.0)
        state = np.array([0.8, 0.0, 0.0, 0.0, 0.1, 0.0])
        traj = sys.propagate(state, times=[0,10])
        
        fig = traj.plot_3d(n_points=100)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0 # type: ignore
    
    def test_plot_3d_cr3bp_with_show_body_false(self):
        """plot_3d() for CR3BP with show_body=False."""
        sys = System('3body', earth(), secondary_body=moon(), distance=384400.0)
        state = np.array([0.8, 0.0, 0.0, 0.0, 0.1, 0.0])
        traj = sys.propagate(state, times=[0,10])
        
        fig = traj.plot_3d(n_points=100, bodies=False)
        
        assert isinstance(fig, go.Figure)
    
    def test_add_to_plot_doesnt_crash(self):
        """add_to_plot() executes without error."""
        sys = System('2body', earth())
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        traj = sys.propagate(orbit, times=[0,5400])
        
        # Create base figure
        fig = go.Figure()
        
        # Add trajectory
        result = traj.add_to_plot(fig, n_points=100, color='blue', traj_name='Test')
        
        assert result is fig  # Returns same figure
        assert len(fig.data) > 0  # Added trace # type: ignore
    
    def test_add_to_plot_with_multiple_trajectories(self):
        """add_to_plot() can add multiple trajectories to same figure."""
        sys = System('2body', earth())
        orbit1 = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        orbit2 = OE(a=8000, e=0.02, i=np.radians(30), 
                   omega=0, w=0, nu=0)
        
        traj1 = sys.propagate(orbit1, times=[0,5400])
        traj2 = sys.propagate(orbit2, times=[0,5400])
        
        fig = go.Figure()
        traj1.add_to_plot(fig, color='red', traj_name='Orbit 1')
        traj2.add_to_plot(fig, color='blue', traj_name='Orbit 2')
        
        assert len(fig.data) >= 2  # Two trajectories # type: ignore

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
        return system.propagate(initial_state, times=[0,5400], with_stm=True)
    
    @pytest.fixture
    def traj_no_stm(self, system, initial_state):
        """Trajectory without STM for testing."""
        return system.propagate(initial_state, times=[0,5400], with_stm=False)
    
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

@pytest.mark.slow
class TestWithJunctionNodes:
    """Trajectory.with_junction_nodes: relabel interior junctions, reusing
    outputs and boundary nodes (no re-propagation), with count/time
    validation. The Free->Null reclassification semantics are covered
    end-to-end by the multiple-shooting convergence tests; this pins the
    plumbing contract of the method itself."""

    @pytest.fixture
    def multiseg_traj(self, cr3bp_system, lyapunov_orbit):
        tf = lyapunov_orbit.period / 2.0
        times = np.linspace(0.0, tf, 4)
        base = cr3bp_system.propagate(lyapunov_orbit.initial_state, [0.0, tf],
                                      with_stm=False)
        ics = [base.state_at_raw(t) for t in times[:-1]]
        return cr3bp_system.propagate(ics, times, with_stm=False)

    def _markers(self, traj):
        # Replacement Free nodes at the existing boundary times; states are
        # arbitrary markers (with_junction_nodes validates count and time, not
        # state consistency).
        out = []
        for k, j in enumerate(traj.junction_nodes):
            m = np.full(6, float(k + 1))
            out.append(FreeJunctionNode(j.time, m, m))
        return out

    def test_replaces_junctions(self, multiseg_traj):
        repl = self._markers(multiseg_traj)
        new = multiseg_traj.with_junction_nodes(repl)
        for n, r in zip(new.junction_nodes, repl):
            np.testing.assert_array_equal(n.post_state, r.post_state)

    def test_reuses_outputs_no_repropagation(self, multiseg_traj):
        new = multiseg_traj.with_junction_nodes(self._markers(multiseg_traj))
        assert new.start_node is multiseg_traj.start_node
        assert new.end_node is multiseg_traj.end_node
        for i in range(multiseg_traj.n_segments):
            assert new._outputs[i] is multiseg_traj._outputs[i]

    def test_rejects_wrong_count(self, multiseg_traj):
        with pytest.raises(ValueError):
            multiseg_traj.with_junction_nodes(self._markers(multiseg_traj)[:-1])

    def test_rejects_time_mismatch(self, multiseg_traj):
        bad = self._markers(multiseg_traj)
        b0 = bad[0]
        bad[0] = FreeJunctionNode(b0.time + 0.5, b0.pre_state, b0.post_state)
        with pytest.raises(ValueError):
            multiseg_traj.with_junction_nodes(bad)

    def test_original_unchanged(self, multiseg_traj):
        before = [np.array(j.post_state, dtype=float)
                  for j in multiseg_traj.junction_nodes]
        new = multiseg_traj.with_junction_nodes(self._markers(multiseg_traj))
        assert new is not multiseg_traj
        for j, snap in zip(multiseg_traj.junction_nodes, before):
            np.testing.assert_array_equal(j.post_state, snap)

# ====================================
# plotting tests
# ====================================

# ========== _body_geometry ==========
 
class TestBodyGeometry:
 
    def test_cr3bp_primary(self, cr3bp_traj, em_system):
        center, radius = cr3bp_traj._body_geometry('primary')
        assert center == pytest.approx([-em_system.mass_ratio, 0.0, 0.0])
        assert radius == pytest.approx(em_system.primary_body.radius_nd)
 
    def test_cr3bp_secondary(self, cr3bp_traj, em_system):
        center, radius = cr3bp_traj._body_geometry('secondary')
        assert center == pytest.approx([_moon_x(em_system), 0.0, 0.0])
        assert radius == pytest.approx(em_system.secondary_body.radius_nd)
 
    def test_two_body_primary_is_at_origin_in_km(self, tb_traj, tb_system):
        center, radius = tb_traj._body_geometry('primary')
        assert center == pytest.approx([0.0, 0.0, 0.0])
        assert radius == pytest.approx(tb_system.primary_body.radius)
 
 
# ========== _resolve_body_names: selection modes ==========
 
class TestResolveSelectionModes:
 
    @pytest.mark.parametrize("spec", [None, False])
    def test_no_selection(self, cr3bp_traj, spec):
        assert cr3bp_traj._resolve_body_names(spec, _ring(L1_X, 0.02)) == []
 
    def test_two_body_automatic_always_primary(self, tb_traj):
        # Positions deliberately irrelevant: every 2-body orbit is about
        # its primary.
        far = _ring(1.0e6, 10.0)
        assert tb_traj._resolve_body_names(True, far) == ['primary']
 
    def test_explicit_overrides_automatic(self, cr3bp_traj):
        # This ring would select nothing automatically.
        names = cr3bp_traj._resolve_body_names(
            'primary', _ring(L1_X, 0.02))
        assert names == ['primary']
 
    def test_explicit_is_case_insensitive_and_trimmed(self, cr3bp_traj):
        assert cr3bp_traj._resolve_body_names(
            ' Secondary ', None) == ['secondary']
 
    def test_explicit_is_canonical_order_and_deduplicated(self, cr3bp_traj):
        names = cr3bp_traj._resolve_body_names(
            ['secondary', 'primary', 'PRIMARY'], None)
        assert names == ['primary', 'secondary']
 
    def test_unknown_designator_raises(self, cr3bp_traj):
        with pytest.raises(ValueError, match="Unknown body designator"):
            cr3bp_traj._resolve_body_names('moon', None)
 
    def test_non_string_designator_raises(self, cr3bp_traj):
        with pytest.raises(ValueError, match="must be strings"):
            cr3bp_traj._resolve_body_names([2], None)
 
    def test_secondary_on_two_body_raises(self, tb_traj):
        with pytest.raises(ValueError, match="only a primary"):
            tb_traj._resolve_body_names('secondary', None)
 
 
# ========== _resolve_body_names: the automatic rule ==========
 
class TestResolveAutomaticRule:
    """
    The rule is proximity OR enclosure. Each case below is built so that
    exactly one condition (or neither) holds, which pins down that both
    halves are live and that neither alone is doing all the work.
    """
 
    def test_neither_condition_selects_nothing(self, cr3bp_traj):
        # Small L1 ring: ~0.15 nd from the Moon, box nowhere near either.
        names = cr3bp_traj._resolve_body_names(True, _ring(L1_X, 0.02))
        assert names == []
 
    def test_enclosure_alone_selects_body(self, cr3bp_traj, em_system):
        """
        A DRO-like ring about the Moon at ~0.18 nd (~70,000 km): never
        within the ~0.045 nd proximity band, but the Moon sits in the box.
        This is the case the proximity-only rule got wrong.
        """
        ring = _ring(_moon_x(em_system), 0.18)
        closest = np.min(np.linalg.norm(
            ring - [_moon_x(em_system), 0.0, 0.0], axis=1))
        r_moon = em_system.secondary_body.radius_nd
        assert closest > config.PROXIMITY_THRESHOLD * r_moon  # precondition
 
        assert cr3bp_traj._resolve_body_names(True, ring) == ['secondary']
 
    def test_proximity_alone_selects_body(self, cr3bp_traj, em_system):
        """
        A straight pass at y = 0.03 nd (~6.6 Moon radii): inside the
        proximity band, but the box is a thin slab at y ~ 0.03 that does
        not contain the Moon's center at y = 0.
        """
        x = np.linspace(0.9, 1.08, 300)
        line = np.column_stack([x, np.full_like(x, 0.03), np.zeros_like(x)])
        assert cr3bp_traj._resolve_body_names(True, line) == ['secondary']
 
    def test_proximity_threshold_is_respected(self, cr3bp_traj):
        # Same pass, threshold shrunk to one radius: now neither holds.
        x = np.linspace(0.9, 1.08, 300)
        line = np.column_stack([x, np.full_like(x, 0.03), np.zeros_like(x)])
        names = cr3bp_traj._resolve_body_names(
            True, line, proximity_threshold=1.0)
        assert names == []
 
 
# ========== add_bodies ==========
 
class TestAddBodies:
 
    def test_returns_same_figure(self, cr3bp_traj):
        fig = go.Figure()
        assert cr3bp_traj.add_bodies(fig) is fig
 
    def test_measures_the_figure_not_the_trajectory(self, cr3bp_traj,
                                                   em_system):
        # cr3bp_traj alone selects nothing; the ring on the figure encloses
        # the Moon, and the figure is what the automatic test measures.
        fig = _line_figure(_ring(_moon_x(em_system), 0.18))
        cr3bp_traj.add_bodies(fig)
        assert [s.meta for s in _surfaces(fig)] == ['secondary']
 
    def test_falls_back_to_own_samples_on_empty_figure(self, cr3bp_traj):
        fig = go.Figure()
        cr3bp_traj.add_bodies(fig)          # must not raise
        assert _surfaces(fig) == []         # tiny arc near 0.8: nothing
 
    def test_repeated_calls_do_not_duplicate(self, cr3bp_traj):
        fig = go.Figure()
        cr3bp_traj.add_bodies(fig, bodies='primary')
        cr3bp_traj.add_bodies(fig, bodies=['primary', 'secondary'])
        metas = [str(s.meta) for s in _surfaces(fig)]
        assert sorted(metas) == ['primary', 'secondary']
 
    def test_false_draws_nothing(self, cr3bp_traj):
        fig = go.Figure()
        cr3bp_traj.add_bodies(fig, bodies=False)
        assert len(fig.data) == 0                       # type: ignore
 
    def test_trace_identity_and_label(self, cr3bp_traj, em_system):
        fig = go.Figure()
        cr3bp_traj.add_bodies(fig, bodies='secondary')
        (surface,) = _surfaces(fig)
        expected_label = em_system.secondary_body.name or 'Secondary'
        assert surface.meta == 'secondary'
        assert surface.name == expected_label
        assert surface.legendgroup == 'bodies'
 
    def test_styling_arguments_applied(self, cr3bp_traj):
        fig = go.Figure()
        cr3bp_traj.add_bodies(fig, bodies='primary', color='gray',
                              opacity=0.3)
        (surface,) = _surfaces(fig)
        assert surface.opacity == pytest.approx(0.3)
        assert surface.colorscale[0][1] == 'gray'
 
    def test_spheres_do_not_enter_the_figure_extent(self, cr3bp_traj):
        """
        Body surfaces must not feed the automatic tests, or drawing Earth
        would drag the Lagrange bounding box across the whole system.
        """
        ring = _ring(L1_X, 0.02)
        fig = _line_figure(ring)
        cr3bp_traj.add_bodies(fig, bodies=['primary', 'secondary'])
        positions = _figure_line_positions(fig)
        assert positions is not None
        assert positions.shape == ring.shape
 
 
# ========== plot_3d ==========
 
class TestPlot3dBodies:
 
    def test_bodies_false_draws_no_surfaces(self, tb_traj):
        fig = tb_traj.plot_3d(n_points=50, bodies=False)
        assert _surfaces(fig) == []
 
    def test_two_body_default_draws_primary(self, tb_traj):
        fig = tb_traj.plot_3d(n_points=50)
        assert [s.meta for s in _surfaces(fig)] == ['primary']
 
    def test_bodies_drawn_after_trajectory_line(self, tb_traj):
        # Order matters: the automatic tests measure line traces already on
        # the figure, so the line must be added first.
        fig = tb_traj.plot_3d(n_points=50, show_nodes=False)
        kinds = [type(t).__name__ for t in fig.data]
        assert kinds.index('Scatter3d') < kinds.index('Surface')
 
    def test_show_body_keyword_is_gone(self, tb_traj):
        # Documents the intentional signature break.
        with pytest.raises(TypeError):
            tb_traj.plot_3d(n_points=50, show_body=False)  # type: ignore
 
    def test_line_uses_default_hover(self, tb_traj):
        fig = tb_traj.plot_3d(n_points=50, bodies=False, show_nodes=False)
        assert fig.data[0].hovertemplate == _DEFAULT_HOVER
 
 
# ========== add_to_plot hover ==========
 
class TestAddToPlotHover:
 
    def test_default_hover_uses_significant_figures(self, cr3bp_traj):
        fig = go.Figure()
        cr3bp_traj.add_to_plot(fig, n_points=20, show_nodes=False)
        assert fig.data[0].hovertemplate == _DEFAULT_HOVER
        assert '.6g' in _DEFAULT_HOVER
        assert '.1f' not in _DEFAULT_HOVER
 
    def test_hovertemplate_override_does_not_collide(self, cr3bp_traj):
        # Before the kwargs.pop fix this raised TypeError: multiple values
        # for keyword argument 'hovertemplate'.
        fig = go.Figure()
        cr3bp_traj.add_to_plot(fig, n_points=20, show_nodes=False,
                               hovertemplate='custom<extra></extra>')
        assert fig.data[0].hovertemplate == 'custom<extra></extra>'
 
    def test_other_kwargs_still_pass_through(self, cr3bp_traj):
        fig = go.Figure()
        cr3bp_traj.add_to_plot(fig, n_points=20, show_nodes=False,
                               legendgroup='family', showlegend=False)
        assert fig.data[0].legendgroup == 'family'
        assert fig.data[0].showlegend is False
 
 
# ========== config rename ==========
 
class TestPlotConfigRename:
 
    def test_new_names_exist_with_old_defaults(self):
        assert config.PLOT_BBOX_MARGIN == pytest.approx(0.25)
        assert config.PLOT_MIN_EXTENT_FRAC == pytest.approx(0.05)
 
    @pytest.mark.parametrize("old", ["LAGRANGE_BBOX_MARGIN",
                                     "LAGRANGE_MIN_EXTENT_FRAC"])
    def test_old_names_are_gone(self, old):
        assert not hasattr(config, old)
 
    def test_repr_lists_plotting_settings(self):
        text = repr(config)
        for key in ("PLOT_BBOX_MARGIN", "PLOT_MIN_EXTENT_FRAC",
                    "DEFAULT_LAGRANGE_SIZE"):
            assert key in text

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
