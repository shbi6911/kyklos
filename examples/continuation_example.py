
import kyklos as ky

# ============================================================================
# User Input Parameters
# ============================================================================
def get_user_inputs(defaults):
    """
    Prompt user for benchmark configuration parameters.
    
    Returns
    -------
    dict
        Configuration parameters
    """

    valid = ('lyapunov', 'halo')
    raw = defaults['family']
    family = raw.strip().lower() if isinstance(raw, str) else None

    if family not in valid:
        print("Family must be 'lyapunov' or 'halo'")
        while True:
            family = input("Choose 'lyapunov' or 'halo' : ").strip().lower()
            if family in valid:
                break
            print("Invalid choice. Please enter 'lyapunov' or 'halo'")

    if family == 'lyapunov':
        lagrange = 'L1'
    if family == 'halo':
        lagrange = 'L2'

    print("="*70)
    print(f"Earth-Moon Periodic Orbit {family.capitalize()} Family Parameters")
    print("="*70)
    
    # Step Size and Number of orbits
    print("\n1. Number of Orbits and Pseudo-Arc Length step size")
    print(f"   More orbits will take longer.  Larger step size is more likely to\n" 
          f"   jump to the wrong family.")
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
    
    # Corrector tolerance
    print(f"\n2. Corrector Tolerance")
    print(f"   Lower tolerance improves accuracy but takes longer, and some orbits")
    print(f"   may be incapable of meeting a low tolerance.")
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

    # Colorbar
    print(f"\n3. Colorbar Key Parameter")
    print(f"   Select one of the following parameters for the plot colorbar:")
    print(f"   Default: {defaults['cbar']} ")

    while True:
        msg = "Choose 'index', 'jacobi', 'period', 'stability' : "
        cbar_input = input(msg).strip().lower()
        cbar = (cbar_input.replace('"', '').replace("'", "") 
                if cbar_input else defaults['cbar'])
        
        if cbar in ['index', 'jacobi', 'period', 'stability']:
            break  # Valid input received, exit the loop
            
        print("Invalid choice. Please enter 'index', 'jacobi', 'period', 'stability'")

    print(f"\n4. Verbosity")
    print(f"   If verbosity is True a progress counter will display while marching.")
    print(f"   Default: {defaults['verbose']}. ")
    change = input("\nChange [Y/n]: ").strip().lower()
    verbose = defaults['verbose']
    if change and change == 'y':
        verbose = True if verbose == False else False

    
    # Summary
    print("\n" + "="*70)
    print("Configuration Summary:")
    print(f"  Family: {family.capitalize()}")
    print(f"  Orbits and Step Size: {n_steps} total orbits at {ds} step size")
    print(f"  Corrector tolerance: {tol}")
    print(f"  Colorbar displays: {cbar}")
    print(f"  Verbosity: {verbose}")
    print("="*70)
    
    confirm = input("\nProceed with these settings? [Y/n]: ").strip().lower()
    if confirm and confirm != 'y':
        print("Continuation cancelled, rerun.")
        import sys
        sys.exit(0)
    
    return {
        'family': family,
        'lagrange': lagrange,
        'n_steps': n_steps,
        'ds': ds,
        'tol': tol,
        'cbar': cbar,
        'verbose': verbose
    }

def initialize():

    print(f"We first establish an Earth-Moon CR3BP System, using a factory default.")
    sys = ky.earth_moon_cr3bp()

    print(f"We get a planar seed from linearizing about L1, which requires "
          f"evaluating the system's Jacobian at the point.")
    seed = sys.planar_seeder('L1')

    print(f"We correct this initial guess into a PeriodicOrbit.  This requires the "
      f"variational equations for the corrector STM, and the vector field "
      f"evaluator, because we free the period of the solve.")

    guess = ky.CorrectorGuess.from_seeder_result(seed,sys,'lyapunov')
    dc = ky.DifferentialCorrector(tol=1e-12)
    initial_lyapunov = ky.correct_as(guess, dc)

    print(f"Finally, we create an initial Near Rectilinear Halo Orbit using "
          f"a premade factory default.")
    initial_halo = ky.gateway_orbit()

    return ({'lyapunov':initial_lyapunov, 'halo':initial_halo})

def plot_family(inputs):

    input_kwargs = {'orbit':inputs['orbit'], 
                    'recipe':inputs['family'],
                    'ds': inputs['ds'],
                    'n_steps': inputs['n_steps'],
                    'corrector': ky.DifferentialCorrector(tol=inputs['tol']),
                    'verbose': inputs['verbose']
                    }

    print(f"Marching {inputs['lagrange']} {inputs['family'].capitalize()} family "
          f"with step size {inputs['ds']}, targeting {inputs['n_steps']} steps, "
          f"with corrector tolerance {inputs['tol']}")
    
    with ky.Timer(verbose=False) as t:
        family = ky.march_family(**input_kwargs)
    print(f"Marching {inputs['lagrange']} {inputs['family'].capitalize()} family "
          f"took {t.elapsed:.4g} seconds.")

    fig1 = family.plot_3d(color_by = inputs['cbar'])

    print(f"We can slice out every fifth element and only plot those.")
    fig2 = family[::5].plot_3d(color_by=inputs['cbar'], 
        title= f"{inputs['lagrange']} {inputs['family'].capitalize()} Family "
               f"Sliced Every 5 Orbits")

    fig1.show()
    fig2.show()


# ============================================================================
# Script Entry Point
# ============================================================================

if __name__ == "__main__":

    initial_orbits = initialize()

    lyapunov_defaults = {'family':'lyapunov',
                         'n_steps': 300,
                         'ds': 0.01,
                         'tol': 1e-12,
                         'cbar': 'stability',
                         'verbose': True,
                        }

    halo_defaults = {'family':'halo',
                             'n_steps': 425,
                             'ds': 0.005,
                             'tol': 1e-12,
                             'cbar': 'period',
                             'verbose': True,
                            }

    lyapunov_inputs = get_user_inputs(lyapunov_defaults)
    lyapunov_inputs['orbit'] = initial_orbits['lyapunov']

    plot_family(lyapunov_inputs)

    halo_inputs = get_user_inputs(halo_defaults)
    halo_inputs['orbit'] = initial_orbits['halo']

    plot_family(halo_inputs)