# Shane Billingsley
# 8-21-25
'''
testing the heyoka.py package
'''
import time
import heyoka as hy

def pendulum(vrb=False):
    # Create the symbolic variables x and v.
    x, v = hy.make_vars("x", "v")

    # Create the integrator object.
    ta = hy.taylor_adaptive(
        # Definition of the ODE system:
        # x' = v
        # v' = -9.8 * sin(x)
        sys=[(x, v), (v, -9.8 * hy.sin(x))],
        # Initial conditions for x and v.
        state=[0.05, 0.025],
    )
    
    if vrb:
        print("Initial taylor adaptive system")
        print(ta)
    
    # Perform a single step.
    oc, h = ta.step()

    # Print the outcome flag and the timestep used.
    if vrb:
        print("Step Forward")
        print("Outcome : {}".format(oc))
        print("Timestep: {}".format(h))
        print(ta)
    # Perform a step backward.
    oc, h = ta.step_backward()

    # Print the outcome flag and the timestep used.
    if vrb:
        print("Step Backward")
        print("Outcome : {}".format(oc))
        print("Timestep: {}".format(h))    
        print(ta)
    
    # Print the current time.
    if vrb:
        print("Current time        : {}".format(ta.time))

        # Print out the current state vector.
        print("Current state vector: {}\n".format(ta.state))

    # Reset the time and state to the initial values.
    ta.time = 0.0
    ta.state[:] = [0.05, 0.025]

    # Print them again.
    if vrb:
        print("Current time        : {}".format(ta.time))
        print("Current state vector: {}".format(ta.state))
    
    # Propagate for 5 time units.
    status, min_h, max_h, nsteps, _, _ = ta.propagate_for(delta_t=5.0)
    if vrb:
        print("Propagate for 5 time units")
        print("Outcome      : {}".format(status))
        print("Min. timestep: {}".format(min_h))
        print("Max. timestep: {}".format(max_h))
        print("Num. of steps: {}".format(nsteps))
        print("Current time : {}\n".format(ta.time))

    # Propagate until t = 20.
    status, min_h, max_h, nsteps, _, _ = ta.propagate_until(t=20.0)
    if vrb:
        print("Propagate until t=20")
        print("Outcome      : {}".format(status))
        print("Min. timestep: {}".format(min_h))
        print("Max. timestep: {}".format(max_h))
        print("Num. of steps: {}".format(nsteps))
        print("Current time : {}".format(ta.time))

# main calls
if __name__ == "__main__":
    t0 = time.time()

    pendulum(vrb=True)

    t1 = time.time()
    print(f"Elapsed time is {t1 - t0} seconds")