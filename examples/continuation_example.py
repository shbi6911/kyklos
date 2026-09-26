
import kyklos as ky

print(f"We first establish an Earth-Moon CR3BP System.")
sys = ky.earth_moon_cr3bp()

print(f"We get a planar seed from linearizing about L1, which requires "
      f"evaluating the system's Jacobian at the point.")
seed = sys.planar_seeder('L1')
print(f"We correct this initial guess into a PeriodicOrbit.  This requires the "
      f"variational equations for the corrector STM, and the vector field "
      f"evaluator, because we free the period of the solve.")

guess = ky.CorrectorGuess.from_seeder_result(seed,sys,'lyapunov')
dc = ky.DifferentialCorrector(tol=1e-12)
initial_orbit = ky.correct_as(guess, dc)

lyapunov_kwargs = {'orbit':initial_orbit, 
                   'recipe':'lyapunov',
                   'ds': 0.01,
                   'n_steps': 400,
                   'corrector': dc}
print(f"Marching L1 Lyapunov family with step size {lyapunov_kwargs['ds']}, "
      f"targeting {lyapunov_kwargs['n_steps']} steps, "
      f"with corrector tolerance {lyapunov_kwargs['corrector'].tol}")
with ky.Timer(verbose=False) as t1:
    family = ky.march_family(**lyapunov_kwargs)
print(f"Marching L1 Lyapunovs took {t1.elapsed:.4g} seconds.")

fig = family.plot_3d(color_by = 'stability')

fig.show()
initial_halo = ky.gateway_orbit()
halo_kwargs = {'orbit':initial_halo, 
                'recipe':'halo',
                'ds': 0.005,
                'n_steps': 425,
                'corrector': dc}

print(f"Marching L2 Halo family with step size {halo_kwargs['ds']}, "
      f"targeting {halo_kwargs['n_steps']} steps, "
      f"with corrector tolerance {halo_kwargs['corrector'].tol}")
with ky.Timer(verbose=False) as t2:
    halo_family = ky.march_family(**halo_kwargs)
print(f"Marching L2 Halos took {t2.elapsed:.4g} seconds.")
halo_fig = halo_family.plot_3d(color_by = 'period')
halo_fig.show()

print(f"We need a small step size to prevent the halos from diverting onto the "
      f"Lyapunovs, but this can interfere with visualization.")
print(f"We can slice out some elements and only plot those.")

new_halos = halo_family[::5]
new_halo_fig = new_halos.plot_3d(color_by='jacobi')
print(f"Note that we have several properties of the family which can control "
      f"the colorbar grading, according to the color_by input.")
new_halo_fig.show()