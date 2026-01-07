from kyklos import earth_drag, Satellite, OrbitalElements
import plotly.io as pio
import numpy as np
pio.renderers.default = 'browser'

# Create Earth system with J2 and drag
sys = earth_drag()

# Define satellite properties
sat = Satellite(
    mass=100,              # kg
    drag_coeff=2.2,        # Dimensionless
    cross_section=100,      # m²
    inertia=np.eye(3)*100  # kg⋅m²
)

# Low Earth orbit subject to drag
orbit = OrbitalElements(a=6578, e=0.001, i=0.9, 
			omega=0, w=0, nu=0, system=sys)

# Propagate with satellite parameters
traj = sys.propagate(
    orbit,
    t_start=0,
    t_end=86400,  # 1 day
    satellite_params={'Cd_A': sat.drag_coeff * sat.cross_section, 
                      'mass': sat.mass}
)

# Observe altitude decay
times = np.linspace(0, 86400, 1000)
states = traj.evaluate(times, element_type='kep')
altitudes = [s.a - 6378.137 for s in states]

import plotly.graph_objects as go
fig = go.Figure(data=go.Scatter(x=times/3600, y=altitudes))
fig.update_layout(xaxis_title='Time [hours]', yaxis_title='Altitude [km]')
fig.show()