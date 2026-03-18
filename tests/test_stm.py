"""
Test suite for State Transition Matrix (STM) propagation.

Section 1: Basic Functionality Tests
Tests basic STM propagation features and API.
"""

import pytest
import numpy as np
from kyklos import System, OrbitalElements, earth_2body, ISS_ORBIT


class TestSTMBasicFunctionality:
    """Basic STM propagation functionality tests."""
    
    @pytest.fixture
    def system(self):
        """Create a simple 2-body Earth system for testing."""
        return earth_2body()
    
    @pytest.fixture
    def initial_state(self):
        """Create initial conditions for a circular LEO orbit."""
        return OrbitalElements(
            a=6378.0 + 400.0,  # 400 km altitude
            e=0.001,
            i=np.radians(51.6),
            omega=0.0,
            w=0.0,
            nu=0.0
        )
    
    def test_stm_propagation_enabled(self, system, initial_state):
        """Test that trajectory can be created with STM enabled."""
        traj = system.propagate(
            initial_state,
            t_start=0.0,
            t_end=5400.0,  # 1.5 hours
            with_stm=True
        )
        
        # Verify trajectory has STM enabled
        assert traj._stm_order == 1, "Trajectory should have STM order 1"
    
    def test_get_stm_returns_6x6_matrix(self, system, initial_state):
        """Test that get_stm() returns a 6x6 numpy array."""
        traj = system.propagate(
            initial_state,
            t_start=0.0,
            t_end=5400.0,
            with_stm=True
        )
        
        # Query STM at arbitrary time
        t_query = 2700.0  # Halfway through
        stm = traj.get_stm(t_query)
        
        # Check type and shape
        assert isinstance(stm, np.ndarray), "STM should be numpy array"
        assert stm.shape == (6, 6), f"STM should be 6x6, got {stm.shape}"
        assert stm.dtype == np.float64, "STM should be float64"
    
    def test_stm_identity_at_t0(self, system, initial_state):
        """Test that STM is identity matrix at initial time."""
        t0 = 0.0
        tf = 5400.0
        
        traj = system.propagate(
            initial_state,
            t_start=t0,
            t_end=tf,
            with_stm=True
        )
        
        # Get STM at initial time
        stm_t0 = traj.get_stm(t0)
        
        # Should be identity
        identity = np.eye(6)
        np.testing.assert_allclose(
            stm_t0, 
            identity,
            rtol=1e-12,
            atol=1e-14,
            err_msg="STM at t0 should be identity matrix"
        )
    
    def test_stm_not_identity_after_propagation(self, system, initial_state):
        """Test that STM evolves away from identity during propagation."""
        traj = system.propagate(
            initial_state,
            t_start=0.0,
            t_end=5400.0,
            with_stm=True
        )
        
        # Get STM at final time
        stm_tf = traj.get_stm(5400.0)
        identity = np.eye(6)
        
        # Should NOT be identity anymore
        difference = np.linalg.norm(stm_tf - identity)
        assert difference > 1e-6, \
            "STM should evolve away from identity during propagation"
    
    def test_evaluate_stm_scalar_input(self, system, initial_state):
        """Test that evaluate_stm() works with scalar time input."""
        traj = system.propagate(
            initial_state,
            t_start=0.0,
            t_end=5400.0,
            with_stm=True
        )
        
        # Scalar time
        t = 2700.0
        stm = traj.evaluate_stm(t)
        
        # Should return single 6x6 matrix
        assert stm.shape == (6, 6), \
            f"evaluate_stm(scalar) should return (6,6), got {stm.shape}"
        
        # Should match get_stm() result
        stm_direct = traj.get_stm(t)
        np.testing.assert_array_equal(
            stm,
            stm_direct,
            err_msg="evaluate_stm(scalar) should match get_stm()"
        )
    
    def test_evaluate_stm_array_input(self, system, initial_state):
        """Test that evaluate_stm() works with array of times."""
        traj = system.propagate(
            initial_state,
            t_start=0.0,
            t_end=5400.0,
            with_stm=True
        )
        
        # Array of times
        times = np.array([1000.0, 2000.0, 3000.0, 4000.0])
        stms = traj.evaluate_stm(times)
        
        # Should return (n_times, 6, 6) array
        assert stms.shape == (4, 6, 6), \
            f"evaluate_stm(array) should return (n,6,6), got {stms.shape}"
        
        # Each entry should match individual get_stm() calls
        for i, t in enumerate(times):
            stm_direct = traj.get_stm(t)
            np.testing.assert_allclose(
                stms[i],
                stm_direct,
                rtol=1e-14,
                err_msg=f"evaluate_stm()[{i}] should match get_stm({t})"
            )
    
    def test_sample_stm_returns_correct_shape(self, system, initial_state):
        """Test that sample_stm() returns correct array shape."""
        traj = system.propagate(
            initial_state,
            t_start=0.0,
            t_end=5400.0,
            with_stm=True
        )
        
        # Sample 50 points
        n_points = 50
        stms = traj.sample_stm(n_points=n_points)
        
        # Should return (n_points, 6, 6)
        assert stms.shape == (n_points, 6, 6), \
            f"sample_stm({n_points}) should return ({n_points},6,6), got {stms.shape}"
    
    def test_sample_stm_uniform_spacing(self, system, initial_state):
        """Test that sample_stm() produces uniformly spaced samples."""
        t0 = 0.0
        tf = 5400.0
        n_points = 10

        traj = system.propagate(
            initial_state,
            t_start=t0,
            t_end=tf,
            with_stm=True
        )

        # DIAGNOSTIC: Check continuous output object
        print(f"\n=== Continuous output type: {type(traj._output)}")
        print(f"Trajectory t0: {traj.t0}, tf: {traj.tf}")
        
        stms = traj.sample_stm(n_points=n_points)
        print(f"After sample_stm, stms[0] is identity? {np.allclose(stms[0], np.eye(6))}")

        # Generate expected times
        expected_times = np.linspace(t0, tf, n_points)
        print(f"expected_times[0]: {expected_times[0]}, type: {type(expected_times[0])}")
        
        # CRITICAL: Query raw continuous output directly
        print(f"\n=== Querying continuous output directly ===")
        raw_output_t0 = traj._output(0.0)
        print(f"raw_output(0.0) shape: {raw_output_t0.shape}")
        print(f"raw_output(0.0)[6:42] (STM flattened):")
        print(raw_output_t0[6:42].reshape(6,6))
        
        # Verify first and last match
        stm_first = traj.get_stm(expected_times[0])
        print(f"\nget_stm(expected_times[0]) result:")
        print(stm_first)
        
        # Don't assert, just report
        print(f"\nDo they match? {np.allclose(stms[0], stm_first)}")

    def test_heyoka_continuous_output_array_then_scalar(self, system, initial_state):
        """Minimal test: array query followed by scalar query."""
        traj = system.propagate(initial_state, 0.0, 5400.0, with_stm=True)
        
        # Array query
        times = np.array([0.0, 1000.0, 2000.0])
        array_result = traj._output(times)  # Returns (3, 42)
        
        # Scalar query immediately after
        scalar_result = traj._output(0.0)   # Returns (42,)
        
        # Should both give identity at t=0
        stm_from_array = array_result[0, 6:42].reshape(6, 6)
        stm_from_scalar = scalar_result[6:42].reshape(6, 6)
        
        print(f"STM from array query: identity? {np.allclose(stm_from_array, np.eye(6))}")
        print(f"STM from scalar query: identity? {np.allclose(stm_from_scalar, np.eye(6))}")
        
        np.testing.assert_allclose(stm_from_array, np.eye(6), rtol=1e-14)
        np.testing.assert_allclose(stm_from_scalar, np.eye(6), rtol=1e-14)
    
    def test_linspace_output_type(self, system, initial_state):
        """Test if np.linspace output causes issues."""
        traj = system.propagate(initial_state, 0.0, 5400.0, with_stm=True)
        
        # Create times array with linspace (like the failing test)
        expected_times = np.linspace(0.0, 5400.0, 10)
        
        print(f"expected_times dtype: {expected_times.dtype}")
        print(f"expected_times[0]: {expected_times[0]}, type: {type(expected_times[0])}")
        
        # Array query with linspace array
        array_result = traj._output(expected_times)
        
        # Scalar query with first element from linspace
        scalar_result_from_linspace = traj._output(expected_times[0])
        
        # Scalar query with plain 0.0
        scalar_result_plain = traj._output(0.0)
        
        # Extract STMs
        stm_from_array = array_result[0, 6:42].reshape(6, 6)
        stm_from_linspace_elem = scalar_result_from_linspace[6:42].reshape(6, 6)
        stm_from_plain = scalar_result_plain[6:42].reshape(6, 6)
        
        print(f"\nSTM from array[0]: identity? {np.allclose(stm_from_array, np.eye(6))}")
        print(f"STM from expected_times[0]: identity? {np.allclose(stm_from_linspace_elem, np.eye(6))}")
        print(f"STM from plain 0.0: identity? {np.allclose(stm_from_plain, np.eye(6))}")
        
        # All should be identity
        np.testing.assert_allclose(stm_from_array, np.eye(6), rtol=1e-14)
        np.testing.assert_allclose(stm_from_linspace_elem, np.eye(6), rtol=1e-14)
        np.testing.assert_allclose(stm_from_plain, np.eye(6), rtol=1e-14)
    
    def test_exact_sample_stm_then_get_stm_sequence(self, system, initial_state):
        """Replicate exact operations of sample_stm() followed by get_stm()."""
        traj = system.propagate(initial_state, 0.0, 5400.0, with_stm=True)
        
        # === Exactly what sample_stm() does ===
        n_points = 10
        times = np.linspace(traj.t0, traj.tf, n_points)
        full_states = traj._output(times)  # Shape: (n, 42)
        stm_flat = full_states[:, 6:42]     # Shape: (n, 36)
        stms = stm_flat.reshape(n_points, 6, 6)
        
        print(f"After array query, stms[0] is identity? {np.allclose(stms[0], np.eye(6))}")
        
        # === Exactly what get_stm() does ===
        t = times[0]
        traj._validate_time(t)
        full_state = traj._output(float(t))
        stm_flat_scalar = full_state[6:42]
        stm_from_get = stm_flat_scalar.reshape(6, 6)
        
        print(f"After scalar query, stm_from_get is identity? {np.allclose(stm_from_get, np.eye(6))}")
    
        # Should match
        np.testing.assert_allclose(stms[0], np.eye(6), rtol=1e-14)
        np.testing.assert_allclose(stm_from_get, np.eye(6), rtol=1e-14)
        np.testing.assert_allclose(stm_from_get, stms[0], rtol=1e-14)
    
    def test_sample_stm_uniform_spacing_original(self, system, initial_state):
        """Test that sample_stm() produces uniformly spaced samples."""
        t0 = 0.0
        tf = 5400.0
        n_points = 10

        traj = system.propagate(
            initial_state,
            t_start=t0,
            t_end=tf,
            with_stm=True
        )

        stms = traj.sample_stm(n_points=n_points)

        # Generate expected times
        expected_times = np.linspace(t0, tf, n_points)

        # Verify first and last match
        stm_first = traj.get_stm(expected_times[0])
        stm_last = traj.get_stm(expected_times[-1])

        np.testing.assert_allclose(
            stms[0],
            stm_first,
            rtol=1e-14,
            err_msg="First sampled STM should match t0"
        )

        np.testing.assert_allclose(
            stms[-1],
            stm_last,
            rtol=1e-14,
            err_msg="Last sampled STM should match tf"
        )
    
    def test_sample_stm_get_stm_order_matters(self, system, initial_state):
        """Test whether calling sample_stm affects subsequent get_stm calls."""
        traj = system.propagate(
            initial_state,
            t_start=0.0,
            t_end=5400.0,
            with_stm=True
        )
        
        # Call get_stm BEFORE sample_stm
        stm_before = traj.get_stm(0.0)
        print(f"get_stm(0) BEFORE sample_stm: identity? {np.allclose(stm_before, np.eye(6))}")
        
        # Now call sample_stm
        stms = traj.sample_stm(n_points=10)
        
        # Call get_stm AFTER sample_stm
        stm_after = traj.get_stm(0.0)
        print(f"get_stm(0) AFTER sample_stm: identity? {np.allclose(stm_after, np.eye(6))}")
        
        # They should match
        np.testing.assert_allclose(stm_before, stm_after, rtol=1e-14)
    
    def test_stm_error_when_not_enabled(self, system, initial_state):
        """Test that accessing STM raises error when not enabled."""
        # Propagate WITHOUT STM
        traj = system.propagate(
            initial_state,
            t_start=0.0,
            t_end=5400.0,
            with_stm=False  # Explicit
        )
        
        # Should raise ValueError when trying to access STM
        with pytest.raises(ValueError, match="not propagated with STM"):
            traj.get_stm(1000.0)
        
        with pytest.raises(ValueError, match="not propagated with STM"):
            traj.evaluate_stm([1000.0, 2000.0])
        
        with pytest.raises(ValueError, match="not propagated with STM"):
            traj.sample_stm(n_points=10)

    def test_sample_stm_first_call_bug(self, system, initial_state):
        """Test calling sample_stm() as the FIRST query to continuous output."""
        traj = system.propagate(
            initial_state,
            t_start=0.0,
            t_end=5400.0,
            with_stm=True
        )
        
        # Call sample_stm FIRST (no prior get_stm calls)
        stms = traj.sample_stm(n_points=10)
        
        # Now try get_stm - does it return garbage?
        stm_at_t0 = traj.get_stm(0.0)
        
        print(f"After sample_stm, get_stm(0) is identity? {np.allclose(stm_at_t0, np.eye(6))}")
        print(f"stm_at_t0[0,0] = {stm_at_t0[0, 0]}")
        
        # Should be identity
        np.testing.assert_allclose(
            stm_at_t0,
            np.eye(6),
            rtol=1e-14,
            err_msg="get_stm(0) should be identity even after sample_stm"
        )
    
    def test_sample_stm_minimum_points_error(self, system, initial_state):
        """Test that sample_stm() raises error for n_points < 2."""
        traj = system.propagate(
            initial_state,
            t_start=0.0,
            t_end=5400.0,
            with_stm=True
        )
        
        with pytest.raises(ValueError, match="at least 2"):
            traj.sample_stm(n_points=1)
        
        with pytest.raises(ValueError, match="at least 2"):
            traj.sample_stm(n_points=0)
    
    def test_stm_with_different_initial_conditions(self, system):
        """Test STM propagation with various orbit types."""
        # Circular orbit
        ic_circular = OrbitalElements(
            a=7000.0,
            e=0.0,
            i=0.0,
            omega=0.0,
            w=0.0,
            nu=0.0
        )
        
        traj_circ = system.propagate(
            ic_circular,
            t_start=0.0,
            t_end=3600.0,
            with_stm=True
        )
        
        stm_circ = traj_circ.get_stm(1800.0)
        assert stm_circ.shape == (6, 6), "Circular orbit STM should be 6x6"
        
        # Eccentric orbit
        ic_eccentric = OrbitalElements(
            a=10000.0,
            e=0.3,
            i=np.radians(45.0),
            omega=np.radians(30.0),
            w=np.radians(60.0),
            nu=np.radians(90.0)
        )
        
        traj_ecc = system.propagate(
            ic_eccentric,
            t_start=0.0,
            t_end=3600.0,
            with_stm=True
        )
        
        stm_ecc = traj_ecc.get_stm(1800.0)
        assert stm_ecc.shape == (6, 6), "Eccentric orbit STM should be 6x6"


if __name__ == "__main__":
    # Allow running tests directly with: python test_stm.py
    pytest.main([__file__, "-v"])