import numpy as np
from kyklos import *

sys = earth_moon_cr3bp()

orbit = gateway_orbit()

constraints_base = [Periodicity(['x','y','z','vx','vz']),TargetState({'y':0.0})]

dc = DifferentialCorrector(tol=1e-12)
bootstrap = dc.solve(orbit.trajectory, free_vars='all', constraints=constraints_base,
                     free_times = [1], continuation=True)

_, S, Vt = np.linalg.svd(bootstrap.continuation.DH, full_matrices=True)
v_raw = Vt[-1]

# increasing period for first step
if v_raw[6] < 0:
    v_raw = -v_raw

X_prev = bootstrap.continuation.X

n_steps = 100
ds = 0.005

for k in range(1, n_steps + 1):


