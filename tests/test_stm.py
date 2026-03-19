"""
Test suite for State Transition Matrix (STM) propagation.

Section 1: Basic Functionality Tests
Tests basic STM propagation features and API.
"""

import pytest
import numpy as np
from kyklos import System, OrbitalElements, earth_2body, ISS_ORBIT

# ========== MODULE-LEVEL FIXTURES  ==========

@pytest.fixture
def system():
    """Create a simple 2-body Earth system for testing."""
    return earth_2body()

@pytest.fixture
def initial_state():
    """Create initial conditions for a circular LEO orbit."""
    return OrbitalElements(
        a=6378.0 + 400.0,  # 400 km altitude
        e=0.001,
        i=np.radians(51.6),
        omega=0.0,
        w=0.0,
        nu=0.0
    )

class TestSTMBasicFunctionality:
    """Basic STM propagation functionality tests."""
    
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

class TestSTMIntegration:
    """Integration tests for STM propagation properties."""
    
    def test_stm_determinant_unity(self, system, initial_state):
        """Test that STM determinant ≈ 1 (symplectic property for Hamiltonian systems).
        """
        traj = system.propagate(
            initial_state,
            t_start=0.0,
            t_end=5400.0,  # 1.5 hours
            with_stm=True
        )
        
        # Sample STM at multiple points
        n_samples = 20
        stms = traj.sample_stm(n_points=n_samples)
        
        # Check determinant at each point
        for i, stm in enumerate(stms):
            det = np.linalg.det(stm)
            np.testing.assert_allclose(
                det,
                1.0,
                rtol=1e-10,
                atol=1e-12,
                err_msg=f"STM determinant at sample {i} should be ≈ 1 "
            )
    
    def test_stm_symplectic_property(self, system, initial_state):
        """Test that STM satisfies symplectic property: Φ^T J Φ = J."""
        traj = system.propagate(
            initial_state,
            t_start=0.0,
            t_end=5400.0,
            with_stm=True
        )
        
        # Symplectic matrix J for 6-DOF system
        # J = [0  I]
        #     [-I 0]
        J = np.block([
            [np.zeros((3, 3)), np.eye(3)],
            [-np.eye(3), np.zeros((3, 3))]
        ])
        
        # Test at multiple times
        times = np.linspace(0.0, 5400.0, 10)
        
        for t in times:
            stm = traj.get_stm(t)
            
            # Compute Φ^T J Φ
            result = stm.T @ J @ stm
            
            # Should equal J
            np.testing.assert_allclose(
                result,
                J,
                rtol=1e-10,
                atol=1e-10,
                err_msg=f"STM at t={t} should satisfy symplectic property Φ^T J Φ = J"
            )
    
    def test_stm_forward_backward_inverse(self, system, initial_state):
        """Test that Φ(t0, t1) = Φ^(-1)(t1, t0)."""
        t0 = 0.0
        t_mid = 2700.0
        t1 = 5400.0
        
        # Forward propagation: t0 -> t1
        traj_forward = system.propagate(
            initial_state,
            t_start=t0,
            t_end=t1,
            with_stm=True
        )
        stm_forward = traj_forward.get_stm(t1)  # Φ(t1, t0)
        
        # Backward propagation: t1 -> t0
        final_state = traj_forward.state_at(t1)
        traj_backward = system.propagate(
            final_state,
            t_start=t1,
            t_end=t0,
            with_stm=True
        )
        stm_backward = traj_backward.get_stm(t0)  # Φ(t0, t1)
        
        # Φ(t1, t0) @ Φ(t0, t1) should equal identity
        product = stm_forward @ stm_backward
        identity = np.eye(6)
        
        np.testing.assert_allclose(
            product,
            identity,
            rtol=1e-09,
            atol=1e-09,
            err_msg="Forward and backward STMs should be inverses"
        )
        
        # Also test that Φ_backward ≈ Φ_forward^(-1)
        stm_forward_inv = np.linalg.inv(stm_forward)
        np.testing.assert_allclose(
            stm_backward,
            stm_forward_inv,
            rtol=1e-09,
            atol=1e-09,
            err_msg="Backward STM should equal inverse of forward STM"
        )
    
    def test_extend_preserves_stm(self, system, initial_state):
        """Test that extend() preserves STM settings."""
        # Original trajectory with STM
        traj1 = system.propagate(
            initial_state,
            t_start=0.0,
            t_end=2700.0,
            with_stm=True
        )
        
        # Extend it
        traj2 = traj1.extend(new_tf=5400.0)
        
        # Extended trajectory should have STM enabled
        assert traj2._stm_order == 1, "Extended trajectory should have STM enabled"
        
        # STM should be identity at the extension point (new t0 for traj2)
        stm_at_boundary = traj2.get_stm(2700.0)
        np.testing.assert_allclose(
            stm_at_boundary,
            np.eye(6),
            rtol=1e-14,
            err_msg="STM should be identity at extend start (new t0)"
        )
        
        # Can query STM in extended region (should evolve from identity)
        stm_extended = traj2.get_stm(4000.0)
        assert stm_extended.shape == (6, 6)
        
        # Should NOT be identity anymore
        assert np.linalg.norm(stm_extended - np.eye(6)) > 1e-6, \
            "STM should evolve from identity in extended region"
    
    def test_extend_without_stm_stays_without_stm(self, system, initial_state):
        """Test that extending a non-STM trajectory doesn't add STM."""
        # Original trajectory WITHOUT STM
        traj1 = system.propagate(
            initial_state,
            t_start=0.0,
            t_end=2700.0,
            with_stm=False
        )
        
        # Extend it
        traj2 = traj1.extend(new_tf=5400.0)
        
        # Extended trajectory should NOT have STM
        assert traj2._stm_order is None, "Extended trajectory should not have STM if original didn't"
        
        # Accessing STM should raise error
        with pytest.raises(ValueError, match="not propagated with STM"):
            traj2.get_stm(4000.0)
    
    def test_slice_preserves_stm(self, system, initial_state):
        """Test that slice() preserves STM settings."""
        # Original trajectory with STM
        traj = system.propagate(
            initial_state,
            t_start=0.0,
            t_end=5400.0,
            with_stm=True
        )
        
        # Slice a middle portion
        traj_sliced = traj.slice(t_start=1000.0, t_end=4000.0)
        
        # Sliced trajectory should have STM enabled
        assert traj_sliced._stm_order == 1, "Sliced trajectory should have STM enabled"
        
        # STM should be identity at slice start (new t0 for sliced trajectory)
        stm_at_start = traj_sliced.get_stm(1000.0)
        np.testing.assert_allclose(
            stm_at_start,
            np.eye(6),
            rtol=1e-14,
            err_msg="STM should be identity at slice start (new t0)"
        )
        
        # STM should evolve from identity within slice
        stm_mid = traj_sliced.get_stm(2500.0)
        assert np.linalg.norm(stm_mid - np.eye(6)) > 1e-6, \
            "STM should evolve from identity within slice"
    
    def test_slice_without_stm_stays_without_stm(self, system, initial_state):
        """Test that slicing a non-STM trajectory doesn't add STM."""
        # Original trajectory WITHOUT STM
        traj = system.propagate(
            initial_state,
            t_start=0.0,
            t_end=5400.0,
            with_stm=False
        )
        
        # Slice it
        traj_sliced = traj.slice(t_start=1000.0, t_end=4000.0)
        
        # Sliced trajectory should NOT have STM
        assert traj_sliced._stm_order is None, "Sliced trajectory should not have STM if original didn't"
        
        # Accessing STM should raise error
        with pytest.raises(ValueError, match="not propagated with STM"):
            traj_sliced.get_stm(2000.0)
    
    def test_stm_continuity_across_multiple_orbits(self, system, initial_state):
        """Test STM behavior over multiple orbital periods."""
        # Get orbital period
        period = initial_state.orbital_period()
        
        # Propagate for 3 periods
        traj = system.propagate(
            initial_state,
            t_start=0.0,
            t_end=3 * period,
            with_stm=True
        )
        
        # Sample STM at end of each period
        stm_1period = traj.get_stm(period)
        stm_2period = traj.get_stm(2 * period)
        stm_3period = traj.get_stm(3 * period)
        
        # All should have determinant ≈ 1
        for i, stm in enumerate([stm_1period, stm_2period, stm_3period], 1):
            det = np.linalg.det(stm)
            np.testing.assert_allclose(
                det,
                1.0,
                rtol=1e-10,
                err_msg=f"STM determinant after {i} period(s) should be ≈ 1"
            )
        
        # STM should be evolving (getting further from identity)
        norm_1 = np.linalg.norm(stm_1period - np.eye(6))
        norm_2 = np.linalg.norm(stm_2period - np.eye(6))
        norm_3 = np.linalg.norm(stm_3period - np.eye(6))
        
        assert norm_2 > norm_1, "STM should evolve further from identity over time"
        assert norm_3 > norm_2, "STM should continue evolving further from identity"
    
    def test_stm_with_state_queries(self, system, initial_state):
        """Test that STM and state queries can be mixed without interference."""
        traj = system.propagate(
            initial_state,
            t_start=0.0,
            t_end=5400.0,
            with_stm=True
        )
        
        t = 2700.0
        
        # Mix different query types
        state = traj.state_at(t)
        stm1 = traj.get_stm(t)
        state_raw = traj.state_at_raw(t)
        stm2 = traj.get_stm(t)
        states = traj.sample(n_points=10)
        stm3 = traj.get_stm(t)
        
        # All STM queries should return same result
        np.testing.assert_array_equal(
            stm1, stm2,
            err_msg="STM queries should be consistent"
        )
        np.testing.assert_array_equal(
            stm2, stm3,
            err_msg="STM queries should be consistent after state queries"
        )
        
        # State queries should be unaffected by STM queries
        assert isinstance(state, OrbitalElements), "state_at should return OrbitalElements"
        assert state_raw.shape == (6,), "state_at_raw should return 6-element array"
        assert len(states) == 10, "sample should return list of OrbitalElements"

class TestSTMMATLABValidation:
    """Validate STM propagation against MATLAB reference data."""
    
    @pytest.fixture
    def matlab_reference(self):
        """Load MATLAB reference data from file."""
        import pandas as pd
        from pathlib import Path
        
        # Construct path to reference file
        test_dir = Path(__file__).parent
        ref_file = test_dir / "data" / "stm_ref.txt"
        
        # Load reference data (comma-delimited)
        df = pd.read_csv(ref_file)  # Default is comma-separated
        
        return df
    
    def test_iss_orbit_matlab_validation(self, matlab_reference):
        """Validate ISS orbit propagation with J2 against MATLAB."""
        from kyklos import earth_j2, ISS_ORBIT
        
        # Get reference data
        ref_col = matlab_reference['iss']
        t_final = ref_col.iloc[0]
        ref_state = ref_col.iloc[1:7].values
        ref_stm = ref_col.iloc[7:43].values.reshape(6, 6, order='F')
        
        # Propagate with Kyklos
        system = earth_j2()
        traj = system.propagate(
            ISS_ORBIT,
            t_start=0.0,
            t_end=t_final,
            with_stm=True
        )
        
        # Get final state and STM
        final_state = traj.state_at_raw(t_final)
        final_stm = traj.get_stm(t_final)
        
        # Compare state
        np.testing.assert_allclose(
            final_state,
            ref_state,
            rtol=1e-10,
            atol=1e-10,
            err_msg="ISS final state should match MATLAB reference"
        )
        
        # Compare STM
        np.testing.assert_allclose(
            final_stm,
            ref_stm,
            rtol=1e-10,
            atol=1e-10,
            err_msg="ISS final STM should match MATLAB reference"
        )
    
    def test_geo_orbit_matlab_validation(self, matlab_reference):
        """Validate GEO orbit propagation with J2 against MATLAB."""
        from kyklos import earth_j2, GEO_ORBIT
        
        ref_col = matlab_reference['geo']
        t_final = ref_col.iloc[0]
        ref_state = ref_col.iloc[1:7].values
        ref_stm = ref_col.iloc[7:43].values.reshape(6, 6, order='F')
        
        system = earth_j2()
        traj = system.propagate(
            GEO_ORBIT,
            t_start=0.0,
            t_end=t_final,
            with_stm=True
        )
        
        final_state = traj.state_at_raw(t_final)
        final_stm = traj.get_stm(t_final)
        
        np.testing.assert_allclose(
            final_state,
            ref_state,
            rtol=1e-10,
            atol=1e-10,
            err_msg="GEO final state should match MATLAB reference"
        )
        
        np.testing.assert_allclose(
            final_stm,
            ref_stm,
            rtol=1e-09,
            atol=1e-09,
            err_msg="GEO final STM should match MATLAB reference"
        )
    
    def test_leo_orbit_matlab_validation(self, matlab_reference):
        """Validate LEO orbit propagation with J2 against MATLAB."""
        from kyklos import earth_j2, LEO_ORBIT
        
        ref_col = matlab_reference['leo']
        t_final = ref_col.iloc[0]
        ref_state = ref_col.iloc[1:7].values
        ref_stm = ref_col.iloc[7:43].values.reshape(6, 6, order='F')
        
        system = earth_j2()
        traj = system.propagate(
            LEO_ORBIT,
            t_start=0.0,
            t_end=t_final,
            with_stm=True
        )
        
        final_state = traj.state_at_raw(t_final)
        final_stm = traj.get_stm(t_final)
        
        np.testing.assert_allclose(
            final_state,
            ref_state,
            rtol=1e-10,
            atol=1e-10,
            err_msg="LEO final state should match MATLAB reference"
        )
        
        np.testing.assert_allclose(
            final_stm,
            ref_stm,
            rtol=1e-10,
            atol=1e-10,
            err_msg="LEO final STM should match MATLAB reference"
        )
    
    def test_sso_orbit_matlab_validation(self, matlab_reference):
        """Validate SSO orbit propagation with J2 against MATLAB."""
        from kyklos import earth_j2, SSO_ORBIT
        
        ref_col = matlab_reference['sso']
        t_final = ref_col.iloc[0]
        ref_state = ref_col.iloc[1:7].values
        ref_stm = ref_col.iloc[7:43].values.reshape(6, 6, order='F')
        
        system = earth_j2()
        traj = system.propagate(
            SSO_ORBIT,
            t_start=0.0,
            t_end=t_final,
            with_stm=True
        )
        
        final_state = traj.state_at_raw(t_final)
        final_stm = traj.get_stm(t_final)
        
        np.testing.assert_allclose(
            final_state,
            ref_state,
            rtol=1e-10,
            atol=1e-10,
            err_msg="SSO final state should match MATLAB reference"
        )
        
        np.testing.assert_allclose(
            final_stm,
            ref_stm,
            rtol=1e-10,
            atol=1e-10,
            err_msg="SSO final STM should match MATLAB reference"
        )
    
    def test_molniya_orbit_matlab_validation(self, matlab_reference):
        """Validate Molniya orbit propagation with J2 against MATLAB."""
        from kyklos import earth_j2, MOLNIYA_ORBIT
        
        ref_col = matlab_reference['mol']
        t_final = ref_col.iloc[0]
        ref_state = ref_col.iloc[1:7].values
        ref_stm = ref_col.iloc[7:43].values.reshape(6, 6, order='F')
        
        system = earth_j2()
        traj = system.propagate(
            MOLNIYA_ORBIT,
            t_start=0.0,
            t_end=t_final,
            with_stm=True
        )
        
        final_state = traj.state_at_raw(t_final)
        final_stm = traj.get_stm(t_final)
        
        np.testing.assert_allclose(
            final_state,
            ref_state,
            rtol=1e-10,
            atol=1e-10,
            err_msg="Molniya final state should match MATLAB reference"
        )
        
        np.testing.assert_allclose(
            final_stm,
            ref_stm,
            rtol=1e-10,
            atol=1e-10,
            err_msg="Molniya final STM should match MATLAB reference"
        )
    
    def test_lyapunov_orbit_matlab_validation(self, matlab_reference):
        """Validate Lyapunov orbit (CR3BP) propagation against MATLAB."""
        from kyklos import earth_moon_cr3bp, LYAPUNOV_ORBIT
        
        ref_col = matlab_reference['lyap']
        t_final = ref_col.iloc[0]  # Nondimensional time
        ref_state = ref_col.iloc[1:7].values
        ref_stm = ref_col.iloc[7:43].values.reshape(6, 6, order='F')
        
        system = earth_moon_cr3bp()
        traj = system.propagate(
            LYAPUNOV_ORBIT,
            t_start=0.0,
            t_end=t_final,
            with_stm=True
        )
        
        final_state = traj.state_at_raw(t_final)
        final_stm = traj.get_stm(t_final)
        
        np.testing.assert_allclose(
            final_state,
            ref_state,
            rtol=1e-10,
            atol=1e-10,
            err_msg="Lyapunov final state should match MATLAB reference"
        )
        
        np.testing.assert_allclose(
            final_stm,
            ref_stm,
            rtol=1e-10,
            atol=1e-10,
            err_msg="Lyapunov final STM should match MATLAB reference"
        )
    
    def test_gateway_orbit_matlab_validation(self, matlab_reference):
        """Validate Gateway/halo orbit (CR3BP) propagation against MATLAB."""
        from kyklos import earth_moon_cr3bp, GATEWAY_ORBIT
        
        ref_col = matlab_reference['halo']
        t_final = ref_col.iloc[0]  # Nondimensional time
        ref_state = ref_col.iloc[1:7].values
        ref_stm = ref_col.iloc[7:43].values.reshape(6, 6, order='F')
        
        system = earth_moon_cr3bp()
        traj = system.propagate(
            GATEWAY_ORBIT,
            t_start=0.0,
            t_end=t_final,
            with_stm=True
        )
        
        final_state = traj.state_at_raw(t_final)
        final_stm = traj.get_stm(t_final)
        
        np.testing.assert_allclose(
            final_state,
            ref_state,
            rtol=1e-10,
            atol=1e-10,
            err_msg="Gateway final state should match MATLAB reference"
        )
        
        np.testing.assert_allclose(
            final_stm,
            ref_stm,
            rtol=1e-10,
            atol=1e-10,
            err_msg="Gateway final STM should match MATLAB reference"
        )

class TestSTMEdgeCases:
    """Edge case tests for STM propagation."""
    
    @pytest.fixture
    def system_j2(self):
        """Create Earth system with J2."""
        from kyklos import earth_j2
        return earth_j2()
    
    @pytest.fixture
    def system_drag(self):
        """Create Earth system with drag."""
        from kyklos import earth_drag
        return earth_drag()
    
    @pytest.fixture
    def system_j2_drag(self):
        """Create Earth system with J2 and drag."""
        from kyklos import System, EARTH, EARTH_STD_ATMO
        return System(
            '2body',
            EARTH,
            perturbations=('J2', 'drag'),
            atmosphere=EARTH_STD_ATMO,
            compile=True
        )
    
    @pytest.fixture
    def test_satellite(self):
        """Create satellite for drag tests."""
        from kyklos import Satellite
        return Satellite.for_drag_only(
            mass=500.0,      # kg
            Cd_A=2.2 * 10.0  # Cd * Area [m^2]
        )
    
    # Lower altitude orbit for stronger drag effect
    @pytest.fixture
    def low_orbit(self): 
        return OrbitalElements(
            a=6378.0 + 250.0,  # 250 km altitude
            e=0.001,
            i=np.radians(51.6),
            omega=0.0,
            w=0.0,
            nu=0.0
        )
    
    def test_very_short_propagation(self, system, initial_state):
        """Test STM propagation over very short time intervals."""
        # Test multiple short intervals
        short_times = [0.1, 1.0, 10.0]
        
        for dt in short_times:
            traj = system.propagate(
                initial_state,
                t_start=0.0,
                t_end=dt,
                with_stm=True
            )
            
            # STM should be close to identity for very short times
            stm_final = traj.get_stm(dt)
            identity = np.eye(6)
            
            # Should be close to identity (but not exactly)
            deviation = np.linalg.norm(stm_final - identity)
            
            # For dt=0.1s, should be very close to identity
            # For dt=10s, should have evolved somewhat
            if dt < 1.0:
                assert deviation < 1, \
                    f"STM should be close to identity for dt={dt}s"
            
            # Determinant should still be 1
            det = np.linalg.det(stm_final)
            np.testing.assert_allclose(
                det, 1.0, rtol=1e-10,
                err_msg=f"STM determinant should be 1 even for dt={dt}s"
            )
    
    def test_backward_propagation(self, system, initial_state):
        """Test STM with backward time propagation (t_end < t_start)."""
        t0 = 5400.0
        tf = 0.0  # Backward in time
        
        # Get initial state at t=5400
        traj_forward = system.propagate(
            initial_state,
            t_start=0.0,
            t_end=5400.0,
            with_stm=False  # Don't need STM for setup
        )
        state_at_5400 = traj_forward.state_at(5400.0)
        
        # Propagate backward with STM
        traj_backward = system.propagate(
            state_at_5400,
            t_start=t0,
            t_end=tf,
            with_stm=True
        )
        
        # Should be able to query STM
        stm_at_start = traj_backward.get_stm(t0)
        stm_at_end = traj_backward.get_stm(tf)
        
        # STM at t0 should be identity
        np.testing.assert_allclose(
            stm_at_start,
            np.eye(6),
            rtol=1e-14,
            err_msg="STM at t0 should be identity for backward propagation"
        )
        
        # STM at tf should have evolved
        assert np.linalg.norm(stm_at_end - np.eye(6)) > 1e-6, \
            "STM should evolve during backward propagation"
        
        # Determinant should be 1
        det = np.linalg.det(stm_at_end)
        np.testing.assert_allclose(
            det, 1.0, rtol=1e-10,
            err_msg="STM determinant should be 1 for backward propagation"
        )
    
    def test_backward_forward_round_trip(self, system, initial_state):
        """Test that forward then backward propagation returns to identity."""
        # Forward propagation
        traj_fwd = system.propagate(
            initial_state,
            t_start=0.0,
            t_end=2700.0,
            with_stm=True
        )
        stm_forward = traj_fwd.get_stm(2700.0)
        
        # Get state at t=2700
        state_mid = traj_fwd.state_at(2700.0)
        
        # Backward propagation
        traj_back = system.propagate(
            state_mid,
            t_start=2700.0,
            t_end=0.0,
            with_stm=True
        )
        stm_backward = traj_back.get_stm(0.0)
        
        # Composition: Φ_back(0, 2700) @ Φ_fwd(2700, 0) should equal identity
        product = stm_backward @ stm_forward
        identity = np.eye(6)
        
        np.testing.assert_allclose(
            product,
            identity,
            rtol=1e-9,
            atol=1e-9,
            err_msg="Round-trip forward-backward should return to identity"
        )
    
    def test_drag_with_stm(self, system_drag, low_orbit, test_satellite):
        """Test STM propagation with atmospheric drag perturbation."""
        traj = system_drag.propagate(
            low_orbit,
            t_start=0.0,
            t_end=10800.0,
            with_stm=True,
            satellite=test_satellite
        )
        
        # Should be able to query STM
        stm_initial = traj.get_stm(0.0)
        stm_final = traj.get_stm(5400.0)
        
        # Initial STM should be identity
        np.testing.assert_allclose(
            stm_initial,
            np.eye(6),
            rtol=1e-14,
            err_msg="STM at t0 should be identity for drag system"
        )
        
        # Final STM should have evolved
        assert np.linalg.norm(stm_final - np.eye(6)) > 1e-6, \
            "STM should evolve during drag propagation"
        
        # Determinant should be  < 1  (drag is dissipative)
        det = np.linalg.det(stm_final)
        assert det < 1.0, \
            "STM determinant should be < 1 for dissipative drag system"
    
    def test_j2_drag_combined_with_stm(self, system_j2_drag, low_orbit, test_satellite):
        """Test STM propagation with combined J2 and drag perturbations."""
        traj = system_j2_drag.propagate(
            low_orbit,
            t_start=0.0,
            t_end=10800.0,
            with_stm=True,
            satellite=test_satellite
        )
        
        # Should be able to query STM
        stm_initial = traj.get_stm(0.0)
        stm_final = traj.get_stm(5400.0)
        
        # Initial STM should be identity
        np.testing.assert_allclose(
            stm_initial,
            np.eye(6),
            rtol=1e-14,
            err_msg="STM at t0 should be identity for J2+drag system"
        )
        
        # Final STM should have evolved
        assert np.linalg.norm(stm_final - np.eye(6)) > 1e-6, \
            "STM should evolve during J2+drag propagation"
        
        # Determinant should be < 1 (drag is dissipative)
        det = np.linalg.det(stm_final)
        assert det < 1.0, \
            "STM determinant should be < 1 for dissipative drag system"
    
    def test_drag_without_satellite_raises_error(self, system_drag, initial_state):
        """Test that drag system requires satellite parameter."""
        with pytest.raises(ValueError, match="requires a Satellite"):
            system_drag.propagate(
                initial_state,
                t_start=0.0,
                t_end=100.0,
                with_stm=True
                # Missing satellite parameter
            )
    
    def test_stm_with_different_satellite_masses(self, system_drag, low_orbit):
        """Test STM sensitivity to satellite mass parameter."""
        from kyklos import Satellite, OrbitalElements
        
        # Two satellites with different masses
        sat_light = Satellite.for_drag_only(mass=100.0, Cd_A=2.2 * 5.0)
        sat_heavy = Satellite.for_drag_only(mass=1000.0, Cd_A=2.2 * 5.0)
        
        # Propagate both with STM
        traj_light = system_drag.propagate(
            low_orbit,  # Use lower orbit
            t_start=0.0,
            t_end=10800.0,  # 3 hours instead of 1.5
            with_stm=True,
            satellite=sat_light
        )
        
        traj_heavy = system_drag.propagate(
            low_orbit,
            t_start=0.0,
            t_end=10800.0,
            with_stm=True,
            satellite=sat_heavy
        )
        
        # STMs should be different (drag affects them differently)
        stm_light = traj_light.get_stm(10800.0)
        stm_heavy = traj_heavy.get_stm(10800.0)
        
        # Should NOT be equal
        assert not np.allclose(stm_light, stm_heavy, rtol=1e-4), \
            "STMs should differ for different satellite masses"
        
        # Determinants should be < 1 (dissipative system)
        det_light = np.linalg.det(stm_light)
        det_heavy = np.linalg.det(stm_heavy)
        
        assert det_light < 1.0, \
            "STM determinant should be < 1 for dissipative drag system"
        assert det_heavy < 1.0, \
            "STM determinant should be < 1 for dissipative drag system"
        
        # Light satellite experiences more drag → more volume contraction
        assert det_light < det_heavy, \
            "Lighter satellite should have smaller determinant (more drag effect)"

if __name__ == "__main__":
    # Allow running tests directly with: python test_stm.py
    pytest.main([__file__, "-v"])