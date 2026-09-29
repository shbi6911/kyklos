import numpy as np
from kyklos import *

system = earth_moon_cr3bp()

orbit = gateway_orbit()
initial_state = orbit.initial_state.elements
period = orbit.period

constraints_base = [Periodicity(['x','y','z','vx','vz']),TargetState({'y':0.0})]
free_vars = ['x','y','z','vx','vy','vz']
period_mult = 1

# constraints_base: list[Constraint] = [TargetState({'y': 0.0, 'vx': 0.0, 'vz': 0.0})]
# free_vars = ['x','z','vy']
# period_mult = 0.5

free_times = [1]

dc = DifferentialCorrector(tol=1e-12)

guess = system.propagate(initial_state, [0, period*period_mult], with_stm=True)

result = dc.solve(guess, free_vars=free_vars, constraints=constraints_base,
                free_times = free_times, continuation=True)
orbit = result.trajectory

fig1 = orbit.plot_3d()
fig1.show()

_, S, Vt = np.linalg.svd(result.continuation.DH, full_matrices=True)
v_raw = Vt[-1]

# increasing period for first step
if v_raw[-1] < 0:
    v_raw = -v_raw

t_hat_new = v_raw
X_prev = result.continuation.X

n_steps = 400
ds = 0.01


# ----- Prepare storage for eventual OrbitFamily constructor -----
states: list[np.ndarray] = []
periods: list[float] = []
iterations: list[int] = []
residuals: list[float] = []
step_sizes: list[float] = []

# ----- Store information for bootstrap solve -----
states.append(orbit.initial_state_raw)
periods.append(orbit.duration*(1 / period_mult))
iterations.append(result.iterations)
residuals.append(result.final_residual)
# OrbitFamily stores 0.0 as the "step size" for the bootstrap solve by convention
# It is not required to store the bootstrap solve, but if this is done this
# convention is recommended, but the OrbitFamily constructor will not require it.
step_sizes.append(0.0)

for k in range(1, n_steps + 1):

    print(f"Converging orbit {k}...     ", end="\r", flush=True)
    constraints_loop = constraints_base.copy()
    constraints_loop.append(PseudoArclength(X_prev=X_prev,t_hat=t_hat_new,ds=ds))

    result = dc.solve(orbit, free_vars=free_vars, 
            constraints=constraints_loop, free_times = free_times, continuation=True)
    
    if not result.converged:
        print(f"Continuation loop terminated at iteration {k}, "
              f"members 1 - {k-1} stored. \n"
              f"Final solve took {result.iterations} iterations, with final residual "
              f"{result.final_residual} and abort reason \"{result.abort_reason}\".")
        break

    orbit = result.trajectory

    t_hat_old = t_hat_new

    states.append(orbit.initial_state_raw)
    periods.append(orbit.duration*(1 / period_mult))
    iterations.append(result.iterations)
    residuals.append(result.final_residual)
    step_sizes.append(ds)

    _, S, Vt = np.linalg.svd(result.continuation.DH, full_matrices=True)
    v_raw = Vt[-1]

    # increasing period for first step
    if np.dot(v_raw, t_hat_old) < 0.0:
        v_raw = -v_raw
    
    t_hat_new = v_raw
    X_prev = result.continuation.X

# ----- Store final result as an OrbitFamily -----
family = OrbitFamily(
    initial_states=np.array(states),
    periods=periods,
    iterations=iterations,
    final_residuals=residuals,
    step_sizes=step_sizes,
    primary_body=system.primary_body,
    secondary_body=system.secondary_body,
    distance=system.distance,
    mu=system.mass_ratio,
    recipe='user',
    scheme='pseudo_arclength',
    free_vars=free_vars,
    free_times=free_times,
    node_specs=None,
    )
family.attach_system(system)

fig2 = family.plot_3d(color_by = 'index')
fig2.show()


    
    


