import kyklos as ky
import numpy as np
# This script demonstrates some uses of the DifferentialCorrector (shooting method)

# establish an Earth/Moon CR3BP system
sys = ky.earth_moon_cr3bp()

# use the default 9:2 resonant L2 Halo orbit as an initial guess
elements = ky.GATEWAY_ORBIT.state

# we will target a member of the L2 Halo family with a period of +1 hour
# propagate a guess from the stored initial state (at apolune) for the stored period
# plus an hour (nondimensionalized)
orbit_guess = sys.propagate(elements,[0, (ky.GATEWAY_ORBIT.period + sys.t2nd(3600))])

# propagate for half the new period for the symmetric formulation
guess_half = sys.propagate(elements,[0, (ky.GATEWAY_ORBIT.period + sys.t2nd(3600))/2])

#instantiate a DifferentialCorrector instance with all settings as package default
dc = ky.DifferentialCorrector()

print(f"Correcting Orbit 1 (Overconstrained):")
# set up an overconstrained problem with all final states required to be periodic 
# and also three initial states fixed (the ones not set as free_vars)
result1 = dc.solve(orbit_guess, free_vars=['x', 'z', 'vy'], 
                           constraints = [ky.Periodicity()])
# Extract the final iterated trajectory from the ShooterResult object
orbit1 = result1.trajectory
# Print some other available information:
print(f"Convergence status: {result1.converged}")
print(f"Number of iterations: {result1.iterations}")

print(f"Correcting Orbit 2 (General Formulation):")
# general formulation for any periodic orbit, 
# constraints = free_vars but doesn't fix initial state at all
result2 = dc.solve(orbit_guess, free_vars='all', constraints = [ky.Periodicity()])
orbit2 = result2.trajectory
print(f"Convergence status: {result2.converged}")
print(f"Number of iterations: {result2.iterations}")

print(f"Correcting Orbit 3 (Symmetric Formulation):")
# Use a formulation that exploits symmetry about the xz plane, by fixing
# [y0, vx0, vz0], and targeting a perpendicular xz-plane crossing
# Note that this formulation uses only half the orbit
result3 = dc.solve(guess_half, free_vars=['x', 'z', 'vy'], 
                  constraints = [ky.TargetState({'y': 0.0, 'vx': 0.0, 'vz': 0.0})])
orbit3_half = result3.trajectory
# repropagate from the initial state for double duration to recover the full orbit
orbit3 = sys.propagate(orbit3_half.start_node.post_state, [0,orbit3_half.duration*2])
print(f"Convergence status: {result3.converged}")
print(f"Number of iterations: {result3.iterations}")

print(f"All orbits are periodic to within good tolerance:")
print(f"orbit1 start - end = {orbit1.start_node.post_state - orbit1.end_node.pre_state}")
print(f"orbit2 start - end = {orbit2.start_node.post_state - orbit2.end_node.pre_state}")
print(f"orbit3 start - end = {orbit3.start_node.post_state - orbit3.end_node.pre_state}")

print(f"All orbits have the same period:")
print(f"duration diff 1 - 3 = {orbit1.duration - orbit3.duration}")
print(f"duration diff 2 - 3 = {orbit2.duration - orbit3.duration}")

print(f"All orbits have the same Jacobi constant to within tolerance:")
print(f"orbit 3 - 1 C = {(orbit3.state_at(orbit3.t0).jacobi_const() - 
                         orbit1.state_at(orbit1.t0).jacobi_const())}")
print(f"orbit 3 - 2 C = {(orbit3.state_at(orbit3.t0).jacobi_const() - 
                         orbit1.state_at(orbit1.t0).jacobi_const())}")
print(f"We can conclude that all three of these are the same halo orbit.")

# sample states as numpy arrays
orbit1_states = orbit1.sample_raw(10)
orbit2_states = orbit2.sample_raw(10)
orbit3_states = orbit3.sample_raw(10)
print(f"However, we see dissimilar state differences:")
print(f"Norm state diff 1 - 3 = {np.linalg.norm((orbit1_states - orbit3_states), 
                                                axis=0)}")
print(f"Norm state diff 2 - 3 = {np.linalg.norm((orbit2_states - orbit3_states), 
                                                axis=0)}")
print(f"We look at initial states:")
print(f"orbit 1: {orbit1.start_node.post_state}")
print(f"orbit 2: {orbit2.start_node.post_state}")
print(f"orbit 3: {orbit3.start_node.post_state}")
print(f"The general formulation did not constrain the phase, so the initial \n"
      f"state is off of the xz-plane by a small amount, explaining the state \n"
      f"discrepancy.  Use symmetry whenever possible.")

# all orbits should looks visually identical on the plot
fig2 = orbit1.plot_3d()
orbit2.add_to_plot(fig2, color='blue')
orbit3.add_to_plot(fig2, color='green')
fig2.show()
