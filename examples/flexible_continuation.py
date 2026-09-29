"""
The Kyklos continuation ecosystem is still very limited, but continuation in the
CR3BP can be done manually with user-facing Kyklos tools.  A rewrite of the Kyklos 
continuation engine which will increase flexibility is planned, but this script 
will demonstrate methods for a user to implement continuation themselves.
"""


import numpy as np
from kyklos import *

"""We will use a default Earth-Moon CR3BP system."""
system = earth_moon_cr3bp()

"""
We can get an initial state and period for an Earth-Moon L1 Axial orbit from
JPL's Horizons web resource.  This orbit is #1250 of the L1 Axials.  Horizons API
queries in Kyklos are planned but not yet implemented, so this orbit has been
retrieved and hardcoded.  Note that Horizons uses an Earth-Moon CR3BP with slightly
different mass ratio and distance than Kyklos.  This will be handled by the bootstrap
solve below.

We use an axial orbit because this is a family which Kyklos' continuation engine
does not yet have a recipe for.
"""
initial_state = np.array([7.8238338906632610E-1,-9.6450643837285686E-29,
                          1.1277601721618780E-15,-2.8274501611883817E-15,
                          4.4001109799932431E-1,-5.0629676308870608E-2])
period = 3.9520274666159665

"""
To allow some experimentation without code changes, we will define some parameters
via user input.
"""
defaults = {'solve':'symmetry',
            'n_steps':165,
            'ds':0.01,
            'tol': 1e-12
            }
def get_user_inputs(defaults):
    """
    Prompt user for configuration parameters.
    
    Returns
    -------
    dict
        Configuration parameters
    """
    valid = ('symmetry', 'general')
    raw = defaults['solve']
    solve = raw.strip().lower() if isinstance(raw, str) else None

    if solve not in valid:
        print("Solve must be 'symmetry' or 'general'")
        while True:
            solve = input("Choose 'symmetry' or 'general' : ").strip().lower()
            if solve in valid:
                break
            print("Invalid choice. Please enter 'symmetry' or 'general'")
    else:
        print(f"Default solve set to {defaults['solve']}. ")
        confirm = input("\nChange? [Y/n]: ").strip().lower()
        if confirm and confirm == 'y':
            solve = 'general' if solve == 'symmetry' else 'symmetry'

    # Step Size and Number of orbits
    print("\n1. Number of Orbits and Pseudo-Arc Length step size")
    print(f"   Defaults: {defaults['n_steps']} orbits at step size {defaults['ds']} ")
    
    while True:
        try:
            n_steps_input = input(f"   Number of orbits "
                                    f"[{defaults['n_steps']}]: ").strip()
            n_steps = int(n_steps_input) if n_steps_input else defaults['n_steps']
            if n_steps <= 1:
                print("   Error: Must have positive number of steps")
                continue
            break
        except ValueError:
            print("   Error: Please enter an integer number of steps.")
    
    while True:
        try:
            ds_input = input(f"   Step size [{defaults['ds']}]: ").strip()
            ds = float(ds_input) if ds_input else defaults['ds']
            if ds <= 0:
                print("   Error: Step size must be positive")
                continue
            break
        except ValueError:
            print("   Error: Please enter a valid number (e.g., 0.01).")

    print(f"\n2. Corrector Tolerance")
    print(f"   Default: {defaults['tol']}")
    
    while True:
        try:
            tol_input = input(f"   Corrector tolerance: [{defaults['tol']}]").strip()
            tol = float(tol_input) if tol_input else defaults['tol']
            if tol <= 0:
                print("   Error: Must be positive")
                continue
            break
        except ValueError:
            print("   Error: Please enter a valid number (e.g. 1e-12)")

    confirm = input("\nProceed with these settings? [Y/n]: ").strip().lower()
    if confirm and confirm != 'y':
        print("Continuation cancelled, rerun.")
        import sys
        sys.exit(0)

    return {
            'solve': solve,
            'n_steps': n_steps,
            'ds': ds,
            'tol': tol
        }

params = get_user_inputs(defaults)

"""
We can set up a more general formulation for the base solve than the symmetry
solves used by the continuation engine.  Here we enforce periodicity on five state
variables (trusting the Jacobi constant to implicitly constrain vy).  We set a zero
target value for y to pin the phase to the xz plane.  This formulation works for any
xz or x-axis symmetric periodic orbit, but it can be less well conditioned than a 
pure symmetry solve customized to a particular family.

This demonstrates implementation of a continuation framework beyond the canned recipes
currently available in the continuation engine.
"""
if params['solve'] == 'general':
    constraints_base = [Periodicity(['x','y','z','vx','vz']),TargetState({'y':0.0})]
    free_vars = ['x','y','z','vx','vy','vz']
    period_mult = 1

"""
This is a symmetry solve for the Axial periodic orbit family, provided for
contrast to the general solve above.  Solve structures which rely on symmetry often
result in better convergence, but they have to be customized to the specific orbit
family.  The axials, for example, are three-dimensional and symmetric about the
x-axis, unlike the Lyapunovs which are planar and x-axis symmetric, and the Halos
which are symmetric about the xz-plane.  When the axial family is included in the
Kyklos continuation engine, this will be its basic solve recipe.
"""
if params['solve'] == 'symmetry':
    constraints_base: list[Constraint] = [TargetState({'y': 0.0, 'z': 0.0, 'vx': 0.0})]
    free_vars = ['x','vy','vz']
    period_mult = 0.5

"""Freeing the integration time on an otherwise square solve induces the structural
corank-1 state required by pseudo arclength continuation.  We need a structural rank
deficiency in order to have a tangent vector to parameterize the family.
"""
free_times = [1]

"""
This defines the shooter algorithmic class which converges an orbit using Newton's
method.  Its stored data is minimal, currently only tolerance and failure
thresholds.  Default tolerance is 1e-10.  When using a symmetry solve, only the 
*half-orbit* is converged to corrector tolerance.  Thus, when the orbit is
repropagated to full period and periodicity closure is checked (tf - t0 = 0), 
this can still fail.  Increasing corrector tolerance usually solves this issue.
Note that closure tolerance is checked by the PeriodicOrbit class constructor, so
it follows the default periodicity_tol of that class, *not* corrector tolerance.
"""
dc = DifferentialCorrector(tol=params['tol'])

"""
Pseudo arclength continuation requires a previous free-variable vector (X_prev)
and a corank-1 Jacobian DH, from whose nullspace we get the tangent vector.  Thus
to bootstrap continuation we re-solve the existing orbit with a corank-1 solve and
no continuation closer.  This procedure solves despite the rank deficiency if our
initial guess is sufficiently close to a member of the target family.
"""
guess = system.propagate(initial_state, [0, period*period_mult], with_stm=True)

result = dc.solve(guess, free_vars=free_vars, constraints=constraints_base,
                free_times = free_times, continuation=True)
orbit = result.trajectory

"""The Trajectory class has a plot_3d() method for visualization. Note that if the
symmetry solve is used this will show only a half-orbit, as that is what is required
for the initial guess.
"""
fig1 = orbit.plot_3d()
fig1.show()

"""
Now we get the tangent vector as the nullspace vector of the DH matrix.  We perform
a singular value decomposition wherein the last row of the left matrix factor (V^T),
(i.e. the rightmost column of V), corresponds to the right-singular vector associated
with the structural singular value of zero induced by the corank-1.
"""
_, S, Vt = np.linalg.svd(result.continuation.DH, full_matrices=True)
v_raw = Vt[-1]

"""
We need to set a direction for the first continuation step.  The element of the
nullspace vector corresponding to the time variable represents the direction of
the family's evolution in integration time (i.e. period for a periodic orbit), so
we force this element to be positive (i.e. period is increasing).
"""
if v_raw[-1] < 0:
    v_raw = -v_raw

t_hat_new = v_raw
X_prev = result.continuation.X

"""
We set the loop variables of step size (applied as a multiplier to the tangent vector,
thus without physical units) and number of orbits according to user input.
"""
n_steps = params['n_steps']
ds = params['ds']

"""
We will store results as a Kyklos OrbitFamily object.  This is a minimalistic object
which only stores data to recover the trajectories, not the trajectories themselves.
If desired, in a user-constructed continuation loop a list of Kyklos Trajectory or
PeriodicOrbit objects could be stored instead, but the user should be mindful of 
memory use.  Trajectories can be heavyweight objects.
"""
states: list[np.ndarray] = []
periods: list[float] = []
iterations: list[int] = []
residuals: list[float] = []
step_sizes: list[float] = []

# ----- Store information for bootstrap solve -----
states.append(orbit.initial_state_raw)
"""
OrbitFamily expects to store the full period of the orbit, so if a symmetry formulation
using a half-period is used, this should be accounted for.
"""
periods.append(orbit.duration*(1 / period_mult))
iterations.append(result.iterations)
residuals.append(result.final_residual)
"""
OrbitFamily stores 0.0 as the "step size" for the bootstrap solve by convention.
It is not required to store the bootstrap solve at all.  If this is done, this
convention is recommended, but the OrbitFamily constructor will not require it.
The Kyklos continuation engine follows this convention when it constructs an
OrbitFamily.
"""
step_sizes.append(0.0)

"""
We now loop through the assigned number of steps, converging each orbit as we go.
We append a PseudoArclength constraint to the constraint list, using data from
the previous solve.  This closes the system and should result in a square solve.
Note that because this constraint uses data (X_prev, t_hat) from the previous solve,
we must construct a new PseudoArclength instance each loop.
"""
for k in range(1, n_steps + 1):

    w = len(str(n_steps))
    print(f"Converging orbit {k:>{w}d} of {n_steps:>{w}d}...", end="\r", flush=True)

    constraints_loop = constraints_base.copy()
    constraints_loop.append(PseudoArclength(X_prev=X_prev,t_hat=t_hat_new,ds=ds))

    result = dc.solve(orbit, free_vars=free_vars, 
            constraints=constraints_loop, free_times = free_times, continuation=True)

    """
    For numerical algorithms fundamentally based on Newton's method, there are many
    circumstances which can cause failure to converge.  The Kyklos shooter attempts 
    to handle these as gracefully as possible, but failures should always be expected
    and accounted for.
    """
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

    """
    For subsequent steps past the first, instead of checking a specific direction,
    we ensure that we continue in the *same* direction by checking the sign of the dot
    product between the current and previous tangent vectors.  This should work well 
    in most circumstances, but if the family is changing very rapidly and the vectors
    are close to orthogonal, it could fail.
    """
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
"""
OrbitFamily is designed for serialization, and thus does not have an attached System
object by default.  Instead it stores the minimal data (two bodies and a distance)
to reconstruct a CR3BP System.  The attach_system() method either reconstructs the
System or attaches an input System after checking that its mass ratio corresponds
to the one stored.  An attached System is necessary to repropagate the orbits for
plotting.
"""
family.attach_system(system)

"""
OrbitFamily has its own plot_3d() method paralleling Trajectory.plot_3d().  Because
OrbitFamily only stores initial states and periods, the orbits must be repropagated
in order to plot them.  By default these orbits are not stored, but retain_orbits=True
can be input to the plot_3d() method to retain them.  Previous warnings about memory
use apply.  The repropagation takes some time, but when using Heyoka this is rarely
a significant problem!
"""
fig2 = family.plot_3d(color_by = 'index')
fig2.show()


    
    


