# Kyklos

**Trajectory propagation and mission design in Python**

Kyklos is a Python package for spacecraft trajectory propagation, orbital mechanics, and mission design, built on the Heyoka Taylor-series integrator. It supports two-body dynamics (with J2, J3, and drag perturbations) and the circular restricted three-body problem (CR3BP), with multiple-shooting differential correction and pseudo-arclength continuation. The CR3BP toolkit for periodic orbits and their families is the most developed part of the package to date. Kyklos is designed for astrodynamics researchers and engineers who want reliable, efficient orbital mechanics tools with a clean, consistent API.

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-alpha-orange.svg)](https://github.com/shbi6911/kyklos.git)
[![Documentation](https://readthedocs.org/projects/kyklos/badge/?version=latest)](https://kyklos.readthedocs.io/en/latest/)

## Key Features

- **Dynamics**: Two-body dynamics with J2, J3, and drag perturbations, and the CR3BP with automatic nondimensionalization
- **High-Performance Integration**: Heyoka Taylor-series propagation with LLVM compilation, including automatic variational equations (state transition matrices)
- **Continuous-Time Trajectories**: Dense-output trajectories built from segments and nodes (impulsive maneuvers and junctions), evaluated at any time without re-integration
- **Differential Correction**: A multiple-shooting corrector driven by composable constraints
- **CR3BP Periodic Orbits**: A planar seeder, Lyapunov and halo recipes, `PeriodicOrbit` with monodromy and stability index, and pseudo-arclength continuation into an `OrbitFamily`
- **Orbital Elements**: Conversions between Keplerian, Cartesian, equinoctial, and CR3BP forms, plus constructors for circular, synchronous, Molniya, and sun-synchronous orbits
- **Satellite Model, Plotting, and Export**: A `Satellite` class for mass and drag properties, Plotly 3D visualization, and pandas export
- **Type-Safe Design**: Immutable objects with comprehensive validation for reliable simulations, and pre-configured Earth, Moon, and Mars systems for quick setup

## Installation

### Prerequisites

Kyklos requires Python 3.10 or later and the [Heyoka](https://github.com/bluescarni/heyoka) integrator, which is best installed via conda:

```bash
# Create a new environment (recommended)
conda create -n kyklos python=3.11
conda activate kyklos

# Install Heyoka from conda-forge
conda install -c conda-forge heyoka.py
```

Note that Heyoka consists of a C++ library (heyoka) and a Python wrapper (heyoka.py).  Kyklos uses Heyoka 
through the heyoka.py Python API, and so it is necessary to install heyoka.py specifically.

### Install Kyklos (Alpha Testing)

**Note:** Kyklos is currently in alpha testing and not yet published to PyPI.

**Option 1: Install from GitHub (recommended for testers)**
```bash
pip install git+https://github.com/shbi6911/kyklos.git
```

**Option 2: Install from source**
```bash
git clone https://github.com/shbi6911/kyklos.git
cd kyklos
pip install .
```

**For development:**
```bash
git clone https://github.com/shbi6911/kyklos.git
cd kyklos
pip install -e .
```

## Quick Start

All examples use the usual import convention, `import kyklos as ky`. Units are km, s, and rad for two-body systems, and nondimensional for the CR3BP.

### Basic Two-Body Propagation

```python
import kyklos as ky

# Create Earth system with J2 perturbation
system = ky.earth_j2()

# Define initial orbit (LEO with modest eccentricity)
orbit = ky.OrbitalElements(
    a=7000,      # Semi-major axis [km]
    e=0.01,      # Eccentricity
    i=0.5,       # Inclination [rad]
    omega=0,     # RAAN [rad]
    w=0,         # Argument of periapsis [rad]
    nu=0,        # True anomaly [rad]
    system=system
)

# Propagate for one orbit (~97 minutes)
period = orbit.orbital_period()
traj = system.propagate(orbit, [0, period])
# The propagate method converts the OrbitalElements
# to a Cartesian state vector automatically.

# Evaluate state at any time - no re-integration needed!
state_at_half_orbit = traj(period / 2)
print(state_at_half_orbit)

# Convert back to Keplerian elements (osculating)
kep_state = state_at_half_orbit.to_keplerian()
print(kep_state)
```

### Earth-Moon CR3BP Trajectory

```python
import kyklos as ky

# Create Earth-Moon system (nondimensionalized)
system = ky.earth_moon_cr3bp()

# Initial state in rotating frame (near L1)
state = ky.OrbitalElements(
    x_nd=0.8,  y_nd=0,  z_nd=0,
    vx_nd=0.1, vy_nd=0.1, vz_nd=0.1,
    system=system
)

# Propagate in nondimensional time
traj = system.propagate(state, [0, 5])

# Sample trajectory at evenly spaced points
states = traj.sample(n_points=5)
print(states)

# Sample with numpy array output
states_raw = traj.sample_raw(n_points=5)
print(states_raw)

# Visualize in 3D
fig = traj.plot_3d()
fig.show()
```

### Working with Multiple Orbits

```python
import numpy as np
import kyklos as ky

system = ky.earth_2body()

# Create a constellation of satellites
altitudes = np.linspace(400, 1000, 25)        # 400-1000 km altitude
ecc = np.linspace(0.01, 0.1, 25)              # range of eccentricities
inc = np.radians(np.linspace(0, 55, 25))      # range of inclinations
orbits = [
    ky.OrbitalElements(a=6378.137 + h, e=e, i=i,
                       omega=0, w=0, nu=0, system=system)
    for h, e, i in zip(altitudes, ecc, inc)
]

# Propagate each orbit
trajectories = [
    system.propagate(orb, [0, 7000])
    for orb in orbits
]

# Export orbits to pandas DataFrame
# (pandas import is handled by the method)
traj_df = [
    traj.to_dataframe(n_points=1000)
    for traj in trajectories
]
print(traj_df[10])

# Visualize in 3D
fig = trajectories[0].plot_3d()
for traj in trajectories[1:]:
    fig = traj.add_to_plot(fig)
fig.show()
```

### CR3BP Periodic Orbits and Continuation

Seed a planar Lyapunov orbit about the Earth-Moon L1 point, correct it into a verified periodic orbit, and march its family with pseudo-arclength continuation:

```python
import kyklos as ky

cr3bp = ky.earth_moon_cr3bp()

# Linear seed for a small planar orbit about L1
seed = cr3bp.planar_seeder('L1')

# Correct the seed into a verified periodic orbit.
# Orbits this close to a Lagrange point are sensitive, so a tighter
# corrector tolerance than the default is often needed to close.
guess = ky.CorrectorGuess.from_seeder_result(seed, cr3bp, 'lyapunov')
dc = ky.DifferentialCorrector(tol=1e-12)
orbit = ky.correct_as(guess, dc)
print(orbit.period, orbit.stability_index)

# March the Lyapunov family outward from the corrected orbit
family = ky.march_family(orbit, 'lyapunov', ds=0.01, n_steps=20, corrector=dc)

# Visualize the family in 3D
fig = family.plot_3d()
fig.show()
```

A converged orbit is also available ready-made from `ky.lyapunov_orbit()` (an Earth-Moon L1 Lyapunov orbit), and `ky.gateway_orbit()` provides the Gateway NRHO.

## Documentation

**[Full Documentation](https://kyklos.readthedocs.io)**

- **[Installation Guide](https://kyklos.readthedocs.io/en/latest/installation.html)** - Detailed setup instructions
- **[Quick Start Tutorial](https://kyklos.readthedocs.io/en/latest/quickstart.html)** - First examples
- **[Module Guide](https://kyklos.readthedocs.io/en/latest/modules.html)** - An overview of each source module
- **[API Reference](https://kyklos.readthedocs.io/en/latest/api.html)** - Complete class and method documentation

## Project Status

**Version:** 0.2.1 (Alpha Release)

Kyklos is in active development and currently suitable for testing and academic use. The core functionality is stable, but API changes
should be expected to occur.  Testing has been minimal, so bugs are likely.  

### Current Capabilities ✓

- Two-body dynamics with J2, J3, and drag perturbations
- Circular Restricted 3-Body Problem (CR3BP)
- Orbital element conversions (Keplerian, Cartesian, Equinoctial)
- Continuous-time trajectory evaluation, with state transition matrices
- Multiple-shooting differential correction
- CR3BP periodic orbits and pseudo-arclength continuation of orbit families
- Basic visualization with Plotly

### Roadmap

Planned work, in no particular order:

- Bifurcation targeting
- Family switching at bifurcations
- Saving and loading orbit families
- JPL Horizons API queries
- Flexible state vector configurations via `StateLayout` objects
- Attitude dynamics
- Orbit determination models

### Known Issues and Limitations

- OrbitalElements not yet fully robust to edge cases (circular, equatorial, etc.)
  Unexpected behavior may occur, especially with round-trip conversions, where
  angles can return on a different branch than they were given
- The `SNAP_TO_*` thresholds in `kyklos.config` are reserved for snapping
  near-degenerate elements, but are not yet implemented and have no effect
- Limited atmosphere models (exponential only)
- No coordinate frame transformations (implied frames only) and no ephemeris support
- Continuation covers perpendicular-crossing (symmetric) families with
  pseudo-arclength only. Switching families at a bifurcation is not yet automated
- Orbits very close to a libration point, such as planar seeds with a small
  amplitude, can converge in the corrector but fail the periodicity closure check
  at the default tolerance. A tighter tolerance, e.g.
  `ky.DifferentialCorrector(tol=1e-12)`, usually resolves this
- If a step fails to converge mid-march, `march_family` issues a warning and
  returns the members computed so far, so check the length of the returned
  family. A failure on the seed itself raises `ConvergenceError`

## Requirements

### Python Dependencies

- Python ≥ 3.10
- NumPy ≥ 1.20
- SciPy ≥ 1.7
- pandas ≥ 2.0 (DataFrame export)
- Plotly ≥ 6.0 (visualization)
- heyoka.py ≥ 5.0 (via conda-forge)

### System Requirements

- Heyoka requires LLVM for JIT compilation (installed with the conda package)

## Contributing

Kyklos is currently in alpha testing. Feedback from early users is welcome!

### For Testers

- Try the examples and report any issues
- Share your use cases and desired features
- Help identify API pain points

### Reporting Issues

Please include:
- Kyklos version (`import kyklos; print(kyklos.__version__)`)
- Python version
- Heyoka version
- Minimal code to reproduce
- Expected vs. actual behavior

**Issue Tracker:** [GitHub Issues](https://github.com/shbi6911/kyklos/issues)

## Citation

If you use Kyklos in academic work, please cite:

```bibtex
@software{kyklos2026,
  author = {Billingsley, Shane},
  title = {Kyklos: Trajectory Propagation and Mission Design in Python},
  year = {2026},
  version = {0.2.1},
  url = {https://github.com/shbi6911/kyklos.git}
}
```

## License

Kyklos is released under the BSD 3-clause License, aligning with the scientific Python ecosystem (NumPy, pandas, etc.) See [LICENSE](LICENSE) for details.

## Acknowledgments

- Built with [Heyoka](https://github.com/bluescarni/heyoka) by Francesco Biscani and Dario Izzo
- Sincere thanks to Matt Bolliger and Galen Savidge of Advanced Space.  Kyklos' structural features
  are heavily inspired by design features of their internal company software packages.
- Part of MS research at CU Boulder Ann & H.J. Smead Department of Aerospace Engineering Sciences

## AI Assistance

Kyklos was developed with the assistance of Claude (Anthropic), used for drafting code, tests, and documentation and for design discussion. The architecture and numerical approach are the author's, and all AI-assisted changes were reviewed by the author.

## Learn More

- **Heyoka Documentation**: https://bluescarni.github.io/heyoka.py/
- **Orbital Mechanics Primer**: [Vallado, "Fundamentals of Astrodynamics and Applications"](https://www.celestrak.com/software/vallado-sw.php)
- **CR3BP Introduction**: [Koon et al., "Dynamical Systems, the Three-Body Problem and Space Mission Design"](http://www.cds.caltech.edu/~marsden/volume/missionDesign/)

**Questions?** Open an issue or contact: shane.billingsley@colorado.edu
