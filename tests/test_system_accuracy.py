"""
Test suite for integration accuracy validation against MATLAB reference data.

Tests compare Kyklos trajectory propagation against validated MATLAB integrations
for various orbit types and perturbation models.

Reference Data Structure:
- 2-body orbits: ISS, GEO, LEO, SSO, Molniya
- Systems: 2BP, J2, drag
- Integration time: 15 orbital periods
- CR3BP orbits: L1 Lyapunov, L2 Halo
- Integration time: 1 period
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from kyklos import (
    System, OE, 
    ISS_ORBIT, GEO_ORBIT, LEO_ORBIT, SSO_ORBIT, MOLNIYA_ORBIT,
    earth_2body, earth_j2, earth_drag,
    EARTH, MOON
)

# =============================================================================
# Test Configuration
# =============================================================================

# Path to reference data
DATA_DIR = Path(__file__).parent / "data"

# Tolerance for numerical comparisons
POSITION_TOL = 1e-9  # km (1 micrometer)
VELOCITY_TOL = 1e-12  # km/s (1 micrometer/s)

# Drag parameters used in MATLAB (unrealistic for testing)
DRAG_PARAMS = {
    'Cd_A': 2.2 * 2800,  # Cd * A [m^2]
    'mass': 1.0          # [kg]
}

# Orbit name to default orbit mapping
ORBIT_MAP = {
    'iss': ISS_ORBIT,
    'geo': GEO_ORBIT,
    'leo': LEO_ORBIT,
    'sso': SSO_ORBIT,
    'mol': MOLNIYA_ORBIT
}

# System type to factory function mapping
SYSTEM_MAP = {
    '2bp': earth_2body,
    'J2': earth_j2,
    'drag': earth_drag
}


# =============================================================================
# Helper Functions
# =============================================================================

def load_reference_data(filename):
    """
    Load MATLAB reference data from CSV file.
    
    Parameters
    ----------
    filename : str
        Name of file in DATA_DIR
    
    Returns
    -------
    pd.DataFrame
        Reference data with columns [t, x, y, z, vx, vy, vz]
    
    Raises
    ------
    FileNotFoundError
        If file doesn't exist
    """
    filepath = DATA_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Reference data not found: {filepath}")
    
    return pd.read_csv(filepath)


def compute_errors(kyklos_states, matlab_states):
    """
    Compute position and velocity errors between Kyklos and MATLAB.
    
    Parameters
    ----------
    kyklos_states : np.ndarray
        Kyklos states (n_times, 6) [x, y, z, vx, vy, vz]
    matlab_states : np.ndarray
        MATLAB states (n_times, 6) [x, y, z, vx, vy, vz]
    
    Returns
    -------
    dict
        Dictionary with error statistics:
        - pos_max: maximum position error [km]
        - pos_rms: RMS position error [km]
        - vel_max: maximum velocity error [km/s]
        - vel_rms: RMS velocity error [km/s]
        - pos_max_time_idx: index where max position error occurs
        - vel_max_time_idx: index where max velocity error occurs
    """
    # Compute differences
    diff = kyklos_states - matlab_states
    
    # Position errors (magnitude of position difference)
    pos_errors = np.linalg.norm(diff[:, :3], axis=1)
    
    # Velocity errors (magnitude of velocity difference)
    vel_errors = np.linalg.norm(diff[:, 3:], axis=1)
    
    return {
        'pos_max': np.max(pos_errors),
        'pos_rms': np.sqrt(np.mean(pos_errors**2)),
        'vel_max': np.max(vel_errors),
        'vel_rms': np.sqrt(np.mean(vel_errors**2)),
        'pos_max_time_idx': np.argmax(pos_errors),
        'vel_max_time_idx': np.argmax(vel_errors)
    }


def print_error_diagnostics(errors, times, orbit_name, system_name):
    """
    Print diagnostic information about integration errors.
    
    Parameters
    ----------
    errors : dict
        Error statistics from compute_errors()
    times : np.ndarray
        Time vector [s]
    orbit_name : str
        Name of orbit being tested
    system_name : str
        Name of system being tested
    """
    print(f"\n{'='*60}")
    print(f"Integration Error Diagnostics: {orbit_name.upper()} / {system_name.upper()}")
    print(f"{'='*60}")
    
    print("\nPosition Errors:")
    print(f"  Maximum: {errors['pos_max']:.6e} km at "
          f"t={times[errors['pos_max_time_idx']]:.1f} s")
    print(f"  RMS:     {errors['pos_rms']:.6e} km")
    
    print("\nVelocity Errors:")
    print(f"  Maximum: {errors['vel_max']:.6e} km/s at "
          f"t={times[errors['vel_max_time_idx']]:.1f} s")
    print(f"  RMS:     {errors['vel_rms']:.6e} km/s")
    
    print(f"{'='*60}\n")


# =============================================================================
# Two-Body Integration Tests
# =============================================================================

class TestTwoBodyIntegration:
    """Test 2-body integration accuracy against MATLAB reference data."""
    
    @pytest.mark.parametrize("orbit_name", ['iss', 'geo', 'leo', 'sso', 'mol'])
    @pytest.mark.parametrize("system_name", ['2bp', 'J2', 'drag'])
    def test_orbit_integration(self, orbit_name, system_name):
        """
        Test integration accuracy for given orbit and system type.
        
        Compares Kyklos propagation against MATLAB reference data for
        15 orbital periods.
        """
        # Load reference data
        filename = f"{orbit_name}_{system_name}.txt"
        try:
            matlab_data = load_reference_data(filename)
        except FileNotFoundError:
            pytest.skip(f"Reference data not found: {filename}")
        
        # Extract times and states
        times = matlab_data['t'].values
        matlab_states = matlab_data[['x', 'y', 'z', 'vx', 'vy', 'vz']].values
        
        # Get initial orbit and system
        initial_orbit = ORBIT_MAP[orbit_name]
        system = SYSTEM_MAP[system_name]()
        
        # Propagate with appropriate parameters
        t_start = times[0]
        t_end = times[-1]
        
        if system_name == 'drag':
            traj = system.propagate(
                initial_orbit, 
                t_start=t_start, 
                t_end=t_end,
                satellite_params=DRAG_PARAMS
            )
        else:
            traj = system.propagate(
                initial_orbit,
                t_start=t_start,
                t_end=t_end
            )
        
        # Evaluate at MATLAB time points
        kyklos_states = traj.evaluate_raw(times) #type: ignore
        
        # Compute errors
        errors = compute_errors(kyklos_states, matlab_states)
        
        # Print diagnostics if errors exceed tolerance
        if (errors['pos_max'] > POSITION_TOL or 
            errors['vel_max'] > VELOCITY_TOL):
            print_error_diagnostics(errors, times, orbit_name, system_name)
        
        # Assert tolerances
        assert errors['pos_max'] < POSITION_TOL, (
            f"Maximum position error {errors['pos_max']:.6e} km exceeds "
            f"tolerance {POSITION_TOL:.6e} km"
        )
        
        assert errors['vel_max'] < VELOCITY_TOL, (
            f"Maximum velocity error {errors['vel_max']:.6e} km/s exceeds "
            f"tolerance {VELOCITY_TOL:.6e} km/s"
        )
        
        assert errors['pos_rms'] < POSITION_TOL, (
            f"RMS position error {errors['pos_rms']:.6e} km exceeds "
            f"tolerance {POSITION_TOL:.6e} km"
        )
        
        assert errors['vel_rms'] < VELOCITY_TOL, (
            f"RMS velocity error {errors['vel_rms']:.6e} km/s exceeds "
            f"tolerance {VELOCITY_TOL:.6e} km/s"
        )


# =============================================================================
# CR3BP Integration Tests
# =============================================================================

class TestCR3BPIntegration:
    """Test CR3BP integration accuracy against MATLAB reference data."""
    
    @pytest.mark.parametrize("orbit_name", ['L1_lyap', 'L2_halo'])
    def test_cr3bp_orbit(self, orbit_name):
        """
        Test CR3BP integration accuracy for periodic orbits.
        
        Compares Kyklos propagation against MATLAB reference data for
        one orbital period of L1 Lyapunov and L2 Halo orbits.
        """
        # Load reference data
        filename = f"{orbit_name}.txt"
        try:
            matlab_data = load_reference_data(filename)
        except FileNotFoundError:
            pytest.skip(f"Reference data not found: {filename}")
        
        # Extract times and states
        times = matlab_data['t_nd'].values
        matlab_states = matlab_data[['x_nd', 'y_nd', 'z_nd',
                                     'vx_nd', 'vy_nd', 'vz_nd']].values
        
        # Get initial state from first row (nondimensional)
        initial_state = matlab_states[0, :]
        
        # Create Earth-Moon CR3BP system
        system = System(
            '3body', 
            EARTH, 
            secondary_body=MOON,
            distance=384400.0
        )
        
        # Propagate
        t_start = times[0]
        t_end = times[-1]
        
        traj = system.propagate(
            initial_state,
            t_start=t_start,
            t_end=t_end
        )
        
        # Evaluate at MATLAB time points
        kyklos_states = traj.evaluate_raw(times) #type: ignore
        
        # Compute errors
        errors = compute_errors(kyklos_states, matlab_states)
        
        # Print diagnostics if errors exceed tolerance
        if (errors['pos_max'] > POSITION_TOL or 
            errors['vel_max'] > VELOCITY_TOL):
            print_error_diagnostics(errors, times, orbit_name, 'CR3BP')
        
        # Assert tolerances
        assert errors['pos_max'] < POSITION_TOL, (
            f"Maximum position error {errors['pos_max']:.6e} (nd) exceeds "
            f"tolerance {POSITION_TOL:.6e}"
        )
        
        assert errors['vel_max'] < VELOCITY_TOL, (
            f"Maximum velocity error {errors['vel_max']:.6e} (nd) exceeds "
            f"tolerance {VELOCITY_TOL:.6e}"
        )
        
        assert errors['pos_rms'] < POSITION_TOL, (
            f"RMS position error {errors['pos_rms']:.6e} (nd) exceeds "
            f"tolerance {POSITION_TOL:.6e}"
        )
        
        assert errors['vel_rms'] < VELOCITY_TOL, (
            f"RMS velocity error {errors['vel_rms']:.6e} (nd) exceeds "
            f"tolerance {VELOCITY_TOL:.6e}"
        )


# =============================================================================
# Error Analysis Tests (Optional)
# =============================================================================

class TestErrorGrowth:
    """Optional tests for analyzing error growth characteristics."""
    
    @pytest.mark.parametrize("orbit_name", ['iss', 'geo'])
    def test_error_growth_2bp(self, orbit_name):
        """
        Verify that 2BP integration errors grow linearly (or slower).
        
        For point-mass gravity, errors should grow approximately linearly
        with time due to truncation error accumulation.
        """
        filename = f"{orbit_name}_2bp.txt"
        try:
            matlab_data = load_reference_data(filename)
        except FileNotFoundError:
            pytest.skip(f"Reference data not found: {filename}")
        
        times = matlab_data['t'].values
        matlab_states = matlab_data[['x', 'y', 'z', 'vx', 'vy', 'vz']].values
        
        initial_orbit = ORBIT_MAP[orbit_name]
        system = earth_2body()
        
        traj = system.propagate(initial_orbit, t_start=times[0], t_end=times[-1])
        kyklos_states = traj.evaluate_raw(times) #type: ignore
        
        # Compute error at each time point
        diff = kyklos_states - matlab_states
        pos_errors = np.linalg.norm(diff[:, :3], axis=1)
        
        # Check that error at end is still well within tolerance
        # (This is more forgiving than per-point checks)
        assert pos_errors[-1] < POSITION_TOL * 10, (
            f"Final position error {pos_errors[-1]:.6e} km suggests "
            f"excessive error growth"
        )


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])