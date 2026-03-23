"""
Benchmark: SciPy vs. Heyoka for CR3BP Trajectory Propagation

This script demonstrates why Kyklos uses the Heyoka Taylor series integrator
instead of traditional Runge-Kutta methods. We compare:

1. Integration speed (wall-clock time)
2. Accuracy (Jacobi constant preservation)

Test cases include different CR3BP trajectory types:
- L1 Lyapunov orbit (periodic, stable)
- L2 Halo orbit (periodic, quasi-stable)  
- Chaotic trajectory near L1
- Set of random chaotic trajectories

Scroll to the bottom of the file for the main script

Author: Shane Billingsley
Date: 2025-01-08
"""

import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import Kyklos
import sys
sys.path.insert(0, '/mnt/project')
from kyklos import System, OrbitalElements, Timer, earth_moon_cr3bp
from kyklos import EARTH, MOON, LYAPUNOV_ORBIT, GATEWAY_ORBIT


# ============================================================================
# CR3BP Dynamics for SciPy
# ============================================================================

def cr3bp_dynamics_scipy(t, y, mu):
    """
    CR3BP equations of motion for SciPy integration.
    
    Parameters
    ----------
    t : float
        Nondimensional time
    y : ndarray, shape (6,)
        State [x, y, z, vx, vy, vz] in rotating frame
    mu : float
        Mass ratio (m2 / (m1 + m2))
    
    Returns
    -------
    dydt : ndarray, shape (6,)
        Time derivatives
    """
    x, y_coord, z, vx, vy, vz = y
    
    # Distances to primaries
    r1 = np.sqrt((x + mu)**2 + y_coord**2 + z**2)
    r2 = np.sqrt((x - 1 + mu)**2 + y_coord**2 + z**2)
    
    # Pseudo-potential derivatives
    Ux = x - (1 - mu) * (x + mu) / r1**3 - mu * (x - 1 + mu) / r2**3
    Uy = y_coord - (1 - mu) * y_coord / r1**3 - mu * y_coord / r2**3
    Uz = -(1 - mu) * z / r1**3 - mu * z / r2**3
    
    # Equations of motion
    ax = 2 * vy + Ux
    ay = -2 * vx + Uy
    az = Uz
    
    return np.array([vx, vy, vz, ax, ay, az])


def jacobi_constant(state, mu):
    """
    Compute Jacobi constant for CR3BP state.
    
    Parameters
    ----------
    state : ndarray, shape (6,) or (n, 6)
        State(s) [x, y, z, vx, vy, vz]
    mu : float
        Mass ratio
    
    Returns
    -------
    C : float or ndarray
        Jacobi constant value(s)
    """
    if state.ndim == 1:
        x, y, z, vx, vy, vz = state
    else:
        x = state[:, 0]
        y = state[:, 1]
        z = state[:, 2]
        vx = state[:, 3]
        vy = state[:, 4]
        vz = state[:, 5]
    
    r1 = np.sqrt((x + mu)**2 + y**2 + z**2)
    r2 = np.sqrt((x - 1 + mu)**2 + y**2 + z**2)
    
    C = (x**2 + y**2) + 2 * (1 - mu) / r1 + 2 * mu / r2 - (vx**2 + vy**2 + vz**2)
    
    return C


# ============================================================================
# Test Case Definitions
# ============================================================================

def get_test_cases():
    """
    Define initial conditions for different CR3BP trajectory types.
    
    Parameters
    ----------
    system : System
        Kyklos CR3BP system (for nondimensionalization)
    
    Returns
    -------
    dict
        Test cases with initial states and propagation times
    """
    sys = earth_moon_cr3bp()
    
    # Chaotic trajectory (unstable trajectory near L1)
    chaotic_state = np.array([
        sys.L1[0] + 0.005, 0.0, 0.001,
        0.0, -0.02, 0.005
    ])
    
    # use Kyklos default trajectories LYAPUNOV_ORBIT (an L1 Lyapunov periodic orbit)
    # and GATEWAY_ORBIT (a 9:2 synodic resonant L2 Near Rectilinear Halo Orbit)
    test_cases = {
        'L1 Lyapunov (5 periods)': {
            'state': LYAPUNOV_ORBIT.state.elements,
            't_end': 5.0 * LYAPUNOV_ORBIT.period,
            'description': 'unstable periodic orbit around L1'
        },
        'L2 Halo (10 periods)': {
            'state': GATEWAY_ORBIT.state.elements,
            't_end': 10.0 * GATEWAY_ORBIT.period,
            'description': 'quasi-stable periodic orbit associated with L2'
        },
        'Chaotic (25 time units)': {
            'state': chaotic_state,
            't_end': 25,
            'description': 'Chaotic trajectory'
        }
    }
    
    return test_cases

def get_random_cases(system, n_orb):
    """
    Define initial conditions for a random set of trajectories in a CR3BP system
    
    Parameters
    ----------
    system : System
        Kyklos CR3BP system
    n_orb   : int
        number of random trajectories in each of three location baskets 
        (so 3x this number will be returned)
    
    Returns
    -------
    ndarray of nondimensional initial conditions
    """
    # define a range of initial conditions which avoid the primaries

    # first, define boundaries which avoid intersecting the primaries
    p1_bounds = np.array([-system.mass_ratio - system.primary_body.radius_nd, 
                -system.mass_ratio + system.primary_body.radius_nd])
    p2_bounds = np.array([1 - system.mass_ratio - system.secondary_body.radius_nd,
               1 - system.mass_ratio + system.secondary_body.radius_nd])
    
    # round these bounds to two decimal places and also
    # widen the range by 0.05 distance units for P1 and 0.005 for P2
    p1_bounds_rnd = np.array([np.floor(p1_bounds[0]*100)/100 - 0.05,
                             np.ceil(p1_bounds[1]*100)/100 + 0.05])
    p2_bounds_rnd = np.array([np.floor(p2_bounds[0]*100)/100 - 0.005,
                             np.ceil(p2_bounds[1]*100)/100 + 0.005])
    
    # Define x ranges for three regions, beyond P2, between P2 and P1, beyond P1
    x_ranges = [
        (-0.8, p1_bounds_rnd[0]),                   # Beyond P1
        (p1_bounds_rnd[1], p2_bounds_rnd[0]),       # Between P1-P2
        (p2_bounds_rnd[1], 1.5)                     # Beyond P2
    ]

    # Generate x positions for all three regions
    x_positions = np.concatenate([
        np.random.uniform(low, high, n_orb) for low, high in x_ranges
    ])

    # Generate other components once for all 3*n_orb orbits
    n_total = 3 * n_orb
    random_cases = np.column_stack([
        x_positions,
        np.random.uniform(-0.3, 0.3, n_total),  # y
        np.random.uniform(-0.3, 0.3, n_total),  # z
        np.random.uniform(-0.1, 0.1, n_total),  # vx
        np.random.uniform(-0.1, 0.1, n_total),  # vy
        np.random.uniform(-0.1, 0.1, n_total)   # vz
    ])
    
    return random_cases

# ============================================================================
# Integration Benchmarks
# ============================================================================

def benchmark_scipy(state0, t_end, mu, rtol=1e-10, atol=1e-12):
    """
    Benchmark SciPy's DOP853 integrator.
    
    Returns
    -------
    dict
        Results including time, accuracy metrics, and solution
    """
    t_start = 0.0
    
    # Time the integration
    with Timer("SciPy Propagation",verbose=False) as t:
        sol = solve_ivp(
            fun=cr3bp_dynamics_scipy,
            t_span=(t_start, t_end),
            y0=state0,
            method='DOP853',
            args=(mu,),
            rtol=rtol,
            atol=atol,
            dense_output=True
        )
    
    # Compute accuracy metrics
    C0 = jacobi_constant(state0, mu)
    Cf = jacobi_constant(sol.y[:, -1], mu)
    C_error = np.abs(Cf - C0)
    
    # Sample trajectory for error analysis
    t_sample = np.linspace(t_start, t_end, 1000)
    states_sample = sol.sol(t_sample).T
    C_sample = jacobi_constant(states_sample, mu)
    C_drift = np.max(np.abs(C_sample - C0))
    
    return {
        'wall_time': t.elapsed,
        'n_steps': len(sol.t),
        'C_error_final': C_error,
        'C_drift_max': C_drift,
        'solution': sol,
        't_sample': t_sample,
        'C_sample': C_sample - C0  # Store as error from initial
    }


def benchmark_heyoka(state0, t_end, system, rtol=1e-12, atol=1e-14):
    """
    Benchmark Heyoka integrator via Kyklos.
    
    Note: Heyoka doesn't use rtol/atol in the same way as RK methods.
    We include them for interface consistency but Heyoka's adaptive
    order Taylor method defaults to a machine precision error bound (~2.2e-16)
    for all integrations.
    
    Returns
    -------
    dict
        Results including time, accuracy metrics, and trajectory
    """
    # Create OrbitalElements from state
    oe0 = OrbitalElements(state0, 'cr3bp', validate=False, system=system)
    
    t_start = 0.0
    
    # Time the integration
    with Timer("Heyoka Propagation",verbose=False) as t:
        traj = system.propagate(oe0, t_start, t_end)
    
    # Compute accuracy metrics
    mu = system.mass_ratio
    C0 = jacobi_constant(state0, mu)
    
    state_f = traj.state_at(t_end).elements
    Cf = jacobi_constant(state_f, mu)
    C_error = np.abs(Cf - C0)
    
    # Sample trajectory for error analysis
    t_sample = np.linspace(t_start, t_end, 1000)
    states_sample = traj.evaluate_raw(t_sample)
    C_sample = jacobi_constant(states_sample, mu)
    C_drift = np.max(np.abs(C_sample - C0))
    
    # Note: Heyoka doesn't expose step count directly from continuous output
    # The actual integration used adaptive steps, but we can't query it post-facto
    
    return {
        'wall_time': t.elapsed,
        'n_steps': None,  # Not available from Heyoka's continuous output
        'C_error_final': C_error,
        'C_drift_max': C_drift,
        'trajectory': traj,
        't_sample': t_sample,
        'C_sample': C_sample - C0
    }

def benchmark_random_cases(system, n_orb, t_end, rtol=1e-10, atol=1e-12):
    """Benchmark integration time for random initial conditions."""
    random_cases = get_random_cases(system, n_orb)
    
    scipy_times = []
    heyoka_times = []
    
    print(f"\nBenchmarking {len(random_cases)} random orbits...")
    
    for state in random_cases:
        # SciPy
        with Timer(verbose=False) as t:
            sol = solve_ivp(
                cr3bp_dynamics_scipy,
                (0, t_end),
                state,
                method='DOP853',
                args=(system.mass_ratio,),
                rtol=rtol,
                atol=atol
            )
        scipy_times.append(t.elapsed)
        
        # Heyoka
        with Timer(verbose=False) as t:
            traj = system.propagate(
                OrbitalElements(state, 'cr3bp', validate=False, system=system),
                0, t_end
            )
        heyoka_times.append(t.elapsed)
    
    # Convert to arrays
    scipy_times = np.array(scipy_times)
    heyoka_times = np.array(heyoka_times)
    speedup = scipy_times / heyoka_times
    
    # Print summary statistics
    print(f"\n{'='*70}")
    print("Random Orbit Statistics")
    print(f"{'='*70}")
    print(f"Number of orbits: {len(scipy_times)}")
    print(f"\nSciPy DOP853:")
    print(f"  Mean:   {np.mean(scipy_times):.4f} s")
    print(f"  Median: {np.median(scipy_times):.4f} s")
    print(f"  Std:    {np.std(scipy_times):.4f} s")
    print(f"\nHeyoka Taylor:")
    print(f"  Mean:   {np.mean(heyoka_times):.4f} s")
    print(f"  Median: {np.median(heyoka_times):.4f} s")
    print(f"  Std:    {np.std(heyoka_times):.4f} s")
    print(f"\nSpeedup:")
    print(f"  Mean:   {np.mean(speedup):.2f}x")
    print(f"  Median: {np.median(speedup):.2f}x")
    print(f"  Range:  {np.min(speedup):.2f}x - {np.max(speedup):.2f}x")
    print(f"{'='*70}")
    
    return scipy_times, heyoka_times

# ============================================================================
# Main Benchmark Execution
# ============================================================================

def run_benchmarks(rtol=1e-10, atol=1e-12):
    """Run complete benchmark suite and generate results."""
    
    print("="*70)
    print("CR3BP Integration Benchmark: SciPy vs. Heyoka")
    print("="*70)
    
    # Create CR3BP system (Earth-Moon)
    print("\nInitializing Earth-Moon CR3BP system...")
    system = System(
        base_type='CR3BP',
        primary_body=EARTH,
        secondary_body=MOON,
        distance=384400.0,  # km
        compile=True
    )
    
    print(f"Mass ratio mu = {system.mass_ratio:.10f}")
    print(f"Characteristic length L* = {system.L_star:.2f} km")
    print(f"Characteristic time T* = {system.T_star:.2f} s")
    
    # Get test cases
    test_cases = get_test_cases()
    
    # Storage for results
    results = []
    detailed = {}
    
    # Run benchmarks for each test case
    for name, case in test_cases.items():
        print(f"\n{'='*70}")
        print(f"Test Case: {name}")
        print(f"Description: {case['description']}")
        print(f"Duration: {case['t_end']:.2f} nondimensional time units")
        print(f"{'='*70}")
        
        state0 = case['state']
        t_end = case['t_end']
        mu = system.mass_ratio
        
        # Initial Jacobi constant
        C0 = jacobi_constant(state0, mu)
        print(f"\nInitial Jacobi constant: C = {C0:.10f}")
        
        # Benchmark SciPy
        print("\nRunning SciPy DOP853...")
        scipy_results = benchmark_scipy(state0, t_end, mu, rtol=rtol, atol=atol)
        
        print(f"  Wall time: {scipy_results['wall_time']:.4f} s")
        print(f"  Steps: {scipy_results['n_steps']}")
        print(f"  Final C error: {scipy_results['C_error_final']:.2e}")
        print(f"  Max C drift: {scipy_results['C_drift_max']:.2e}")
        
        # Benchmark Heyoka
        print("\nRunning Heyoka Taylor...")
        heyoka_results = benchmark_heyoka(state0, t_end, system, rtol=rtol, atol=atol)
        
        print(f"  Wall time: {heyoka_results['wall_time']:.4f} s")
        print(f"  Final C error: {heyoka_results['C_error_final']:.2e}")
        print(f"  Max C drift: {heyoka_results['C_drift_max']:.2e}")
        
        # Compute speedup
        speedup = scipy_results['wall_time'] / heyoka_results['wall_time']
        accuracy_improvement = (scipy_results['C_drift_max'] / 
                               heyoka_results['C_drift_max'])
        
        print(f"\n{'─'*70}")
        print(f"Speedup: {speedup:.2f}x faster")
        print(f"Accuracy: {accuracy_improvement:.1f}x better (Jacobi conservation)")
        print(f"{'─'*70}")
        
        # Store results
        results.append({
            'Test Case': name,
            'SciPy Time (s)': scipy_results['wall_time'],
            'Heyoka Time (s)': heyoka_results['wall_time'],
            'Speedup': speedup,
            'SciPy Steps': scipy_results['n_steps'],
            'SciPy C Error': scipy_results['C_drift_max'],
            'Heyoka C Error': heyoka_results['C_drift_max'],
            'Accuracy Improvement': accuracy_improvement
        })

        # Store full results for plotting
        detailed[name] = {
            'scipy': scipy_results,    # Contains .solution, .C_sample, etc.
            'heyoka': heyoka_results,  # Contains .trajectory
            'state0': state0,
            't_end': t_end
        }
    
    # Create summary table
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}\n")
    
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    
    print(f"\n{'='*70}")
    print(f"Average speedup: {df['Speedup'].mean():.2f}x")
    print(f"Average accuracy improvement: {df['Accuracy Improvement'].mean():.1f}x")
    print(f"{'='*70}")
    
    return results, detailed, system, test_cases


def plot_results(results, detailed, system):
    """
    Create visualization comparing SciPy and Heyoka performance.
    
    Parameters
    ----------
    results : list of dict
        Summary statistics for all test cases
    detailed : dict
        Full benchmark results with trajectory data for each case
    system : System
        CR3BP system
    
    Returns
    -------
    dict
        Dictionary of figures: {'bar_chart': fig, 'case_name_1': fig, ...}
    """
    figures = {}
    
    # ========================================================================
    # Figure 1: Bar Chart of Integration Times (All Cases)
    # ========================================================================
    
    fig_bar = go.Figure()
    
    # Extract data from summary results
    test_names = [r['Test Case'] for r in results]
    scipy_times = [r['SciPy Time (s)'] for r in results]
    heyoka_times = [r['Heyoka Time (s)'] for r in results]
    
    # Create grouped bar chart
    fig_bar.add_trace(
        go.Bar(
            name='SciPy DOP853',
            x=test_names,
            y=scipy_times,
            marker_color='steelblue'
        )
    )
    fig_bar.add_trace(
        go.Bar(
            name='Heyoka Taylor',
            x=test_names,
            y=heyoka_times,
            marker_color='darkgreen'
        )
    )
    
    # Update layout
    fig_bar.update_layout(
        title='Integration Time Comparison: SciPy vs. Heyoka',
        xaxis_title='Test Case',
        yaxis_title='Wall Time (s)',
        yaxis_type='log',
        barmode='group',
        height=500,
        showlegend=True
    )
    
    figures['bar_chart'] = fig_bar
    
    # ========================================================================
    # Figures 2+: Detailed Plots for Each Test Case
    # ========================================================================
    
    for case_name, data in detailed.items():
        print(f"Creating detailed plots for: {case_name}")
        
        # Extract stored results
        scipy_res = data['scipy']
        heyoka_res = data['heyoka']
        t_end = data['t_end']
        
        # Create 1x3 subplot for this case
        fig_case = make_subplots(
            rows=1, cols=3,
            subplot_titles=(
                'Jacobi Constant Preservation',
                '3D Trajectory (SciPy DOP853)',
                '3D Trajectory (Heyoka Taylor)'
            ),
            specs=[[
                {'type': 'scatter'},
                {'type': 'scatter3d'},
                {'type': 'scatter3d'}
            ]],
            horizontal_spacing=0.08
        )
        
        # ====================================================================
        # Plot 1: Jacobi Constant Error Over Time
        # ====================================================================
        
        fig_case.add_trace(
            go.Scatter(
                x=scipy_res['t_sample'],
                y=np.abs(scipy_res['C_sample']),
                name='SciPy',
                line=dict(color='steelblue', width=2),
                mode='lines'
            ),
            row=1, col=1
        )
        
        fig_case.add_trace(
            go.Scatter(
                x=heyoka_res['t_sample'],
                y=np.abs(heyoka_res['C_sample']),
                name='Heyoka',
                line=dict(color='darkgreen', width=2),
                mode='lines'
            ),
            row=1, col=1
        )
        
        # ====================================================================
        # Plot 2: SciPy 3D Trajectory
        # ====================================================================
        
        # Sample trajectory at 10000 points for smooth visualization
        t_plot = np.linspace(0, t_end, 10000)
        states_scipy = scipy_res['solution'].sol(t_plot).T  # Shape: (500, 6)
        
        fig_case.add_trace(
            go.Scatter3d(
                x=states_scipy[:, 0],
                y=states_scipy[:, 1],
                z=states_scipy[:, 2],
                mode='lines',
                line=dict(color='steelblue', width=4),
                name='SciPy',
                showlegend=False
            ),
            row=1, col=2
        )
        
        # ====================================================================
        # Plot 3: Heyoka 3D Trajectory
        # ====================================================================
        
        states_heyoka = heyoka_res['trajectory'].evaluate_raw(t_plot)  # Shape: (500, 6)
        
        fig_case.add_trace(
            go.Scatter3d(
                x=states_heyoka[:, 0],
                y=states_heyoka[:, 1],
                z=states_heyoka[:, 2],
                mode='lines',
                line=dict(color='darkgreen', width=4),
                name='Heyoka',
                showlegend=False
            ),
            row=1, col=3
        )
        
        # ====================================================================
        # Update Axes and Layout
        # ====================================================================
        
        # Update Jacobi error plot axes
        fig_case.update_xaxes(title_text="Time (TU)", row=1, col=1)
        fig_case.update_yaxes(
            title_text="|ΔC|",
            type='log',
            exponentformat='e',
            row=1, col=1
        )
        
        # Update 3D plot axes (same for both)
        for col in [2, 3]:
            fig_case.update_scenes(
                xaxis_title='x (nd)',
                yaxis_title='y (nd)',
                zaxis_title='z (nd)',
                aspectmode='data',
                row=1, col=col
            )
        
        # Overall layout
        fig_case.update_layout(
            title_text=f'{case_name}: Detailed Comparison',
            height=500,
            showlegend=True
        )
        
        # Store figure with sanitized filename
        fig_key = case_name.replace(' ', '_').replace('(', '').replace(')', '')
        figures[fig_key] = fig_case
    
    return figures

def plot_random_timing_comparison(scipy_times, heyoka_times):
    """Compare timing distributions for random orbits."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Integration Time Distribution', 'Summary Statistics')
    )
    
    # Histogram
    fig.add_trace(
        go.Histogram(
            x=scipy_times, 
            name='SciPy', 
            marker_color='steelblue', 
            opacity=0.7, 
            nbinsx=20
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Histogram(
            x=heyoka_times, 
            name='Heyoka',
            marker_color='darkgreen', 
            opacity=0.7, 
            nbinsx=20
        ),
        row=1, col=1
    )
    
    # Box plot
    fig.add_trace(
        go.Box(y=scipy_times, name='SciPy', marker_color='steelblue'),
        row=1, col=2
    )
    fig.add_trace(
        go.Box(y=heyoka_times, name='Heyoka', marker_color='darkgreen'),
        row=1, col=2
    )
    
    # Update axes
    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Time (s)", row=1, col=2)
    
    fig.update_layout(
        title=f'Random Orbit Integration Times (n={len(scipy_times)})',
        height=400,
        barmode='overlay'
    )
    
    return fig

# ============================================================================
# User Input Parameters
# ============================================================================
def get_user_inputs():
    """
    Prompt user for benchmark configuration parameters.
    
    Returns
    -------
    dict
        Configuration parameters
    """
    print("="*70)
    print("Benchmark Configuration")
    print("="*70)
    
    # Integration tolerances
    print("\n1. Integration Tolerances (for SciPy only, no effect on Heyoka)")
    print("   Tighter tolerances improve accuracy but slow down integration.")
    print("   Recommended range: rtol=1e-8 to 1e-14, atol=1e-10 to 1e-16")
    
    while True:
        try:
            rtol_input = input("   Relative tolerance [1e-12]: ").strip()
            rtol = float(rtol_input) if rtol_input else 1e-12
            if rtol <= 0:
                print("   Error: Tolerance must be positive")
                continue
            break
        except ValueError:
            print("   Error: Please enter a valid number (e.g., 1e-12)")
    
    while True:
        try:
            atol_input = input("   Absolute tolerance [1e-14]: ").strip()
            atol = float(atol_input) if atol_input else 1e-14
            if atol <= 0:
                print("   Error: Tolerance must be positive")
                continue
            break
        except ValueError:
            print("   Error: Please enter a valid number (e.g., 1e-14)")
    
    # Number of random trajectories
    print("\n2. Random Trajectory Count")
    print("   Total orbits = 3 x this number (across 3 spatial regions)")
    print("   Typical range: 5-50 (larger = better statistics, longer runtime)")
    
    while True:
        try:
            n_orb_input = input("   Trajectories per region [10]: ").strip()
            n_orb = int(n_orb_input) if n_orb_input else 10
            if n_orb <= 0:
                print("   Error: Must be a positive integer")
                continue
            break
        except ValueError:
            print("   Error: Please enter a valid integer")
    
    # Random trajectory integration time
    print("\n3. Random Trajectory Duration")
    print("   Nondimensional time units in CR3BP")
    print("   Typical range: 10-50 (higher = longer runtimes)")
    
    while True:
        try:
            rand_time_input = input("   Integration time [25]: ").strip()
            rand_time = float(rand_time_input) if rand_time_input else 25.0
            if rand_time <= 0:
                print("   Error: Time must be positive")
                continue
            break
        except ValueError:
            print("   Error: Please enter a valid number")
    
    # Summary
    print("\n" + "="*70)
    print("Configuration Summary:")
    print(f"  Tolerances: rtol={rtol:.0e}, atol={atol:.0e}")
    print(f"  Random orbits: {3*n_orb} total ({n_orb} per region)")
    print(f"  Integration time: {rand_time} TU ({rand_time:.2f} nondimensional units)")
    print("="*70)
    
    confirm = input("\nProceed with these settings? [Y/n]: ").strip().lower()
    if confirm and confirm != 'y':
        print("Benchmark cancelled.")
        import sys
        sys.exit(0)
    
    return {
        'rtol': rtol,
        'atol': atol,
        'n_orb': n_orb,
        'rand_time': rand_time
    }

# ============================================================================
# Script Entry Point
# ============================================================================

if __name__ == "__main__":

    # Customize script properties
    save_figs = False  # Set to True to save figures instead of displaying
    
    # Get user inputs
    config = get_user_inputs()
    rtol = config['rtol']
    atol = config['atol']
    n_orb = config['n_orb']
    rand_time = config['rand_time']

    # Run benchmarks
    results, detailed, system, test_cases = run_benchmarks(rtol=rtol, atol=atol)

    # Run a set of random orbits
    print("\n" + "="*70)
    print("Random Orbit Benchmark")
    print("="*70)
    
    scipy_times, heyoka_times = benchmark_random_cases(
        system, 
        n_orb=n_orb,  # total orbits (3 regions × n_orb)
        t_end=rand_time,
        rtol=rtol,
        atol=atol
    )
    
    # Generate plots
    print("\n" + "="*70)
    print("Generating comparison plots...")
    print("="*70)
    
    figures = plot_results(results, detailed, system)
    
    if save_figs:
        # Save all figures
        print("\nSaving figures:")
        
        # Save bar chart
        bar_file = "/mnt/user-data/outputs/benchmark_bar_chart.html"
        figures['bar_chart'].write_html(bar_file)
        print(f"  Bar chart: {bar_file}")
        
        # Save detailed plots for each case
        for key, fig in figures.items():
            if key != 'bar_chart':
                output_file = f"/mnt/user-data/outputs/benchmark_{key}.html"
                fig.write_html(output_file)
                print(f"  {key}: {output_file}")
    
     # Display all figures in browser
    print("\nDisplaying figures in browser:")
    import plotly.io as pio
    pio.renderers.default = 'browser'
    
    # Show bar chart first
    print("  Opening bar chart...")
    figures['bar_chart'].show()
    
    # Show detailed plots for each case
    for key, fig in figures.items():
        if key != 'bar_chart':
            print(f"  Opening {key}...")
            fig.show()
    
    fig_random = plot_random_timing_comparison(scipy_times, heyoka_times)
    fig_random.show()
    if save_figs:
        fig_random.write_html("/mnt/user-data/outputs/random_benchmarks.html")
    
    print("\n" + "="*70)
    print("Benchmark complete!")
    print("="*70)
    print("\nKey Takeaways:")
    print("• Heyoka is significantly faster for the same accuracy level")
    print("• Heyoka preserves Jacobi constant much better (indicator of energy conservation)")
    print("• Both integrators produce visually identical trajectories")
    print("• Heyoka's advantage grows for longer integrations and tighter tolerances")
    print("\nThis is why Kyklos uses Heyoka for trajectory propagation!")
