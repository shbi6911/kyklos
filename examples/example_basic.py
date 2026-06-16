from kyklos import earth_j2, earth_moon_cr3bp, earth_2body, OrbitalElements
import numpy as np

# Create Earth system with J2 perturbation (using factory default function)
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
print(f"Orbital Elements of 2BP orbit:")
print(orbit)
print(f"Convert to Cartesian:")
print(orbit.to_cartesian())

# Propagate for 1 orbit (~97 minutes)
period = orbit.orbital_period()
traj = sys.propagate(orbit, times=[0,period])
# The sys.propagate method converts the OrbitalElements
# to a Cartesian state vector automatically.

print(f"Evaluate state of propagated orbit at half-period:")
# Evaluate state at any time - no re-integration needed!
state_at_half_orbit = traj(2700)
print(state_at_half_orbit)

print(f"Keplerian elements should be osculating because of J2 perturbation:")
kep_state = state_at_half_orbit.to_keplerian()
print(kep_state)

print(f"Earth-Moon CR3BP Trajectory:")

# Create Earth-Moon system (nondimensionalized)
sys = earth_moon_cr3bp()

# Initial state in rotating frame (near L1)
state = OrbitalElements(
    x_nd=0.8,  y_nd=0,  z_nd=0,
    vx_nd=0.1, vy_nd=0.1, vz_nd=0.1,
    system=sys
)

# Propagate in nondimensional time
traj = sys.propagate(state, times=[0,5])

print(f"Nondimensional states evenly spaced along the trajectory:")
# Sample trajectory at evenly spaced points
states = traj.sample(n_points=5)
print(states)

print(f"The same states as raw numpy arrays:")
# Sample with numpy array output
states_raw = traj.sample_raw(n_points=5)
print(states_raw)

# Visualize in 3D (uses default number of points, see KyklosConfig)
fig = traj.plot_3d()

print(f"Propagating and Plotting Multiple Orbits:")

sys = earth_2body()

print(f"Create constellation with varying altitude, eccentricity and inclination")
altitudes = np.linspace(400, 1000, 25)  # 400-1000 km altitude
ecc = np.linspace(0.01, 0.1, 25) # range of eccentricities
inc = np.radians(np.linspace(0, 55, 25)) #range of inclinations
# Assemble a list of OrbitalElements objects
orbits = [
    OrbitalElements(a=6378.137 + h, e=e, i=i,
                   omega=0, w=0, nu=0, system=sys)
    for h, e, i in zip(altitudes, ecc, inc)
]

print(f"Propagate all orbits as a Python list comprehension:")
trajectories = [
    sys.propagate(orb, [0, 7000])
    for orb in orbits
]

# Export orbits to pandas DataFrame
# pandas import is handled by the method
traj_df = [
	traj.to_dataframe(n_points=1000) 
	for traj in trajectories
]
print(f"Exporting all trajectories to DataFrames and printing one:")
print(traj_df[10])

# Visualize in 3D
fig = trajectories[0].plot_3d()
for i in trajectories[1:]:
    fig = i.add_to_plot(fig)
fig.show()