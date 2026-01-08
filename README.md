# Kyklos

**High-performance orbital mechanics and trajectory propagation for Python**

Kyklos is a Python package for spacecraft trajectory analysis and mission design, built on the Heyoka Taylor series integrator for exceptional accuracy and speed. Designed for astrodynamics researchers and engineers who need reliable, efficient orbital mechanics tools with a clean, intuitive API.

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-alpha-orange.svg)](https://github.com/shbi6911/kyklos.git)

---

## Key Features

- **Orbital Element Conversions**: Seamless transformations between Keplerian, Cartesian, and Equinoctial element sets
- **High-Performance Integration**: Taylor series propagation via Heyoka with automatic differentiation and LLVM compilation
- **Perturbation Models**: J2 oblateness and atmospheric drag with exponential atmosphere
- **CR3BP Support**: Circular Restricted 3-Body Problem with automatic nondimensionalization
- **Continuous-Time Trajectories**: Evaluate orbital state at any time without re-integration
- **Type-Safe Design**: Immutable objects with comprehensive validation for reliable simulations
- **Factory Patterns**: Pre-configured Earth, Moon, and Mars systems for quick setup

---

## Installation

### Prerequisites

Kyklos requires the [Heyoka](https://github.com/bluescarni/heyoka) integrator, which is best installed via conda:

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

### Basic Two-Body Propagation

```python
from kyklos import earth_j2, OrbitalElements

# Create Earth system with J2 perturbation
sys = earth_j2()

# Define initial orbit (LEO with modest eccentricity)
orbit = OrbitalElements(
    a=7000,      # Semi-major axis [km]
    e=0.01,      # Eccentricity
    i=0.5,       # Inclination [rad]
    omega=0,     # RAAN [rad]
    w=0,         # Argument of periapsis [rad]
    nu=0,        # True anomaly [rad]
    system=sys	 # System reference
)

# Propagate for 1 orbit (~97 minutes)
period = orbit.orbital_period()
traj = sys.propagate(orbit, t_start=0, t_end=period)
# The sys.propagate method converts the OrbitalElements
# to a Cartesian state vector automatically.

# Evaluate state at any time - no re-integration needed!
state_at_half_orbit = traj(2700)
print(state_at_half_orbit)

# Convert back to Keplerian elements (should be osculating)
kep_state = state_at_half_orbit.to_keplerian()
print(kep_state)
```

### Earth-Moon CR3BP Trajectory

```python
from kyklos import earth_moon_cr3bp, OrbitalElements

# Create Earth-Moon system (nondimensionalized)
sys = earth_moon_cr3bp()

# Initial state in rotating frame (near L1)
state = OrbitalElements(
    x_nd=0.8,  y_nd=0,  z_nd=0,
    vx_nd=0.1, vy_nd=0.1, vz_nd=0.1,
    system=sys
)

# Propagate in nondimensional time
traj = sys.propagate(state, t_start=0, t_end=5)

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
from kyklos import earth_2body, OrbitalElements

sys = earth_2body()

# Create constellation of satellites
altitudes = np.linspace(400, 1000, 25)  # 400-1000 km altitude
ecc = np.linspace(0.01, 0.1, 25) # range of eccentricities
inc = np.radians(np.linspace(0, 55, 25)) #range of inclinations
orbits = [
    OrbitalElements(a=6378.137 + h, e=e, i=i,
                   omega=0, w=0, nu=0, system=sys)
    for h, e, i in zip(altitudes, ecc, inc)
]

# Propagate each orbit
trajectories = [
    sys.propagate(orb, 0, 7000)
    for orb in orbits
]

# Export orbits to pandas DataFrame
# pandas import is handled by the method
traj_df = [
	traj.to_dataframe(n_points=1000) 
	for traj in trajectories
]
print(traj_df[10])

# Visualize in 3D
fig = trajectories[0].plot_3d()
for i in trajectories[1:]:
    fig = i.add_to_plot(fig)
fig.show()
```

---

## Documentation

  **[Full Documentation](https://kyklos.readthedocs.io)** (Coming Soon)

- **[Installation Guide](docs/installation.md)** - Detailed setup instructions
- **[Quick Start Tutorial](docs/quickstart.md)** - Comprehensive first examples
- **[API Reference](docs/api/)** - Complete class and method documentation
- **[Examples Gallery](docs/examples/)** - Real-world use cases

Documentation is a work in progress, and not all of the above may be available.

---

## Project Status

**Version:** 0.1.0 (Alpha Release)

Kyklos is in active development and currently suitable for testing and academic use. The core functionality is stable, but API changes
should be expected to occur.  Testing has been minimal, so bugs are likely.  

### Current Capabilities ✓

- Two-body dynamics with J2 and drag perturbations
- Circular Restricted 3-Body Problem (CR3BP)
- Orbital element conversions (Keplerian, Cartesian, Equinoctial)
- Continuous-time trajectory evaluation
- Basic visualization with Plotly

### Roadmap

- **v0.2.0**: Attitude dynamics (quaternion integration)
- **v0.3.0**: basic targeting algorithm for 2BP and CR3BP
- **v0.4.0**: continuation for CR3BP periodic orbit families
- **v0.5.0**: Monte Carlo support and Kalman filter simulation
- **v0.6.0**: Relative motion propagation
- **v0.7.0**: comprehensive visualization module via Plotly
- **v0.8.0**: n-body simulation & additional perturbations

### Known Issues and Limitations

- OrbitalElements not yet fully robust to edge cases (circular, equatorial, etc.)
  Unexpected behavior may occur, especially with round-trip conversions
- Drag model runs error-free but does not produce reliable output, actively debugging

- Limited atmosphere models (exponential only)
- No coordinate frame transformations (implied frames only)

---

## Requirements

### Python Dependencies

- Python ≥ 3.9
- NumPy ≥ 1.20
- Heyoka ≥ 4.0 (via conda-forge)
- pandas ≥ 1.3 (optional, for DataFrame export)
- Plotly ≥ 5.0 (optional, for visualization)

### System Requirements

- Heyoka requires LLVM for JIT compilation
- ~2-5 seconds for initial System compilation
- Recommended: 4+ GB RAM for complex propagations

---

## Contributing

Kyklos is currently in alpha testing. We welcome feedback from early users!

### For Testers

- Try the examples and report any issues
- Share your use cases and desired features
- Help us identify API pain points

### Reporting Issues

Please include:
- Kyklos version (`import kyklos; print(kyklos.__version__)`)
- Python version
- Heyoka version
- Minimal code to reproduce
- Expected vs. actual behavior

**Issue Tracker:** [GitHub Issues](https://github.com/shbi6911/kyklos/issues)

---

## Citation

If you use Kyklos in academic work, please cite:

```bibtex
@software{kyklos2025,
  author = {Billingsley, Shane},
  title = {Kyklos: High-Performance Orbital Mechanics for Python},
  year = {2025},
  version = {0.1.0},
  url = {https://github.com/shbi6911/kyklos.git}
}
```

---

## License

Kyklos is released under the BSD 3-clause License, aligning with the scientific Python ecosystem (NumPy, pandas, etc.) See [LICENSE](LICENSE) for details.

---

## Acknowledgments

- Built with [Heyoka](https://github.com/bluescarni/heyoka) by Francesco Biscani and Dario Izzo
- Development assisted by Claude Sonnet 4.5 (Anthropic)
- Part of MS research at CU Boulder Ann & H.J. Smead Department of Aerospace Engineering Sciences

---

## Learn More

- **Heyoka Documentation**: https://bluescarni.github.io/heyoka.py/
- **Orbital Mechanics Primer**: [Vallado, "Fundamentals of Astrodynamics and Applications"](https://www.celestrak.com/software/vallado-sw.php)
- **CR3BP Introduction**: [Koon et al., "Dynamical Systems, the Three-Body Problem and Space Mission Design"](http://www.cds.caltech.edu/~marsden/volume/missionDesign/)

---

**Questions?** Open an issue or contact: shane.billingsley@colorado.edu
