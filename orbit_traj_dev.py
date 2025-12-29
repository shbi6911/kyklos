'''Development code for an orbital trajectory handling package
Trajectory class definition
created with the assistance of Claude Sonnet 4.5 by Anthropic'''

import numpy as np
import pandas as pd
#import warnings
#from dataclasses import dataclass
from typing import Union, Optional, Dict, List, Tuple, Any
#from enum import Enum
import heyoka as hy
import plotly.graph_objects as go

class Trajectory:
    """
    A trajectory segment with continuous-time state access.
    
    Attributes:
        system: Reference to parent System (immutable)
        integrator: taylor_adaptive object with dense output
        t0: Start time
        tf: End time
    """
    # ========== CONSTRUCTION ==========
    def __init__(self, system: System, integrator, t0: float, tf: float):
        self._system = system  # Immutable reference
        self._integrator = integrator  # Must have dense output enabled
        self._t0 = t0
        self._tf = tf
        
        # Validation
        if not integrator.with_c_out:
            raise ValueError("Integrator must have continuous output enabled")
    
    # ========== PROPERTY ACCESS ==========
    @property
    def system(self):
        return self._system
    
    @property
    def t0(self):
        return self._t0
    
    @property
    def tf(self):
        return self._tf
    
    @property
    def duration(self):
        """Trajectory duration."""
        return self.tf - self.t0
    
    # ========== UTILITY METHODS ==========
    def state_at(self, t: float) -> OrbitalElements:
        """
        Get orbital state at specified time.
        
        Parameters:
            t: Time to query (must be in [t0, tf])
            
        Returns:
            OrbitalElements at time t
            
        Raises:
            ValueError: If t is outside trajectory bounds
        """
        self._validate_time(t)
        state_array = self._integrator.c_output(t)
        return cartesian(state_array)
    
    def evaluate(self, 
                 times: Union[float, 
                 np.ndarray, list]) -> Union[OrbitalElements, 
                                       list[OrbitalElements]]:
        """
        Evaluate trajectory at one or more times.
        
        Parameters:
            times: Single time or array of times
            
        Returns:
            Single OrbitalElements if times is scalar,
            list of OrbitalElements if times is array-like
        """
        # Handle scalar input
        if isinstance(times, (int, float)):
            return self.state_at(times)
        
        # Handle array input
        times = np.asarray(times)
        return [self.state_at(t) for t in times]
    
    def sample(self, n_points: int = 100) -> list[OrbitalElements]:
        """
        Uniformly sample trajectory in time.
        
        Parameters:
            n_points: Number of points to sample (default: 100)
            
        Returns:
            List of OrbitalElements uniformly spaced in time
        """
        if n_points < 2:
            raise ValueError("n_points must be at least 2, use .state_at")
        
        times = np.linspace(self.t0, self.tf, n_points)
        return self.evaluate(times)
    
    def state_at_raw(self, t: float) -> np.ndarray:
        """Get raw state array at time t """
        self._validate_time(t)
        return self._integrator.c_output(t)
    
    def evaluate_raw(self, times: np.ndarray) -> np.ndarray:
        """
        Evaluate at multiple times, returning raw arrays.
        
        Returns:
            Array of shape (n_times, n_states)
        """
        times = np.asarray(times)
        return np.array([self.state_at_raw(t) for t in times])
    
    def _validate_time(self, t: float):
        """Validate that time is within trajectory bounds."""
        if not (self.t0 <= t <= self.tf):
            raise ValueError(
                f"Time {t} outside trajectory bounds [{self.t0}, {self.tf}]"
            )
    
    def contains_time(self, t: float) -> bool:
        """Check if time is within trajectory bounds."""
        return self.t0 <= t <= self.tf
    
    def get_times(self, n_points: int = 100) -> np.ndarray:
        """Generate uniform time array spanning trajectory."""
        return np.linspace(self.t0, self.tf, n_points)
    
    def to_dataframe(self, 
                    times: Optional[np.ndarray] = None,
                    n_points: int = 1000) -> pd.DataFrame:
        """
        Export trajectory to pandas DataFrame.
        
        Parameters:
            times: Specific times to evaluate. If None, uses uniform sampling.
            n_points: Number of uniform samples if times not provided (default: 1000)
            
        Returns:
            DataFrame with columns for time and state components
        """
        # Get evaluation times
        if times is None:
            times = np.linspace(self.t0, self.tf, n_points)
        else:
            times = np.asarray(times)
        
        # Evaluate states
        states = self.evaluate(times)
        
        # Build data dictionary
        data = {
            'time': times,
            'x': [s.cartesian[0] for s in states],
            'y': [s.cartesian[1] for s in states],
            'z': [s.cartesian[2] for s in states],
            'vx': [s.cartesian[3] for s in states],
            'vy': [s.cartesian[4] for s in states],
            'vz': [s.cartesian[5] for s in states],
        }
        
        return pd.DataFrame(data)
    
    def extend(self, new_tf: float, **propagation_kwargs) -> 'Trajectory':
        """
        Extend trajectory by continuing propagation to a new final time.
        
        This creates a NEW Trajectory object that continues from the current
        end state. The original trajectory is unchanged.
        
        Parameters:
            new_tf: New final time (must be > current tf)
        
        Returns:
            New Trajectory object spanning [self.tf, new_tf]
            
        Raises:
            ValueError: If new_tf <= self.tf
        """
        if new_tf <= self.tf:
            raise ValueError(f"new_tf ({new_tf}) must be > current tf ({self.tf})")
        
        # Get the final state of current trajectory
        final_state = self.state_at(self.tf)
        
        # Continue propagation from final state
        #TODO: establish parameter pass-thru for propagation
        raise NotImplementedError(
            "extend() requires System.propagate() method. "
            "This will be implemented when propagation interface is finalized."
    )
    
    def slice(self, t_start: float, t_end: float) -> 'Trajectory':
        """
        Extract a time window as a new Trajectory.
        
        Creates a new Trajectory object representing the same orbital motion
        but over a smaller time interval.
        
        Parameters:
            t_start: Start time of slice (must be >= self.t0)
            t_end: End time of slice (must be <= self.tf)
            
        Returns:
            New Trajectory object spanning [t_start, t_end]
            
        Raises:
            ValueError: If slice bounds are invalid or outside trajectory bounds
        """
        # Validate bounds
        if t_start >= t_end:
            raise ValueError(f"t_start ({t_start}) must be < t_end ({t_end})")
        
        if t_start < self.t0 or t_end > self.tf:
            raise ValueError(
                f"Slice bounds [{t_start}, {t_end}] outside trajectory "
                f"bounds [{self.t0}, {self.tf}]"
            )
    
        # Create new Trajectory with same integrator but different time bounds
        return Trajectory(
            system=self.system,
            integrator=self._integrator,  # Reuse same integrator
            t0=t_start,
            tf=t_end
        )

    # ========== SPECIAL METHODS ==========
    def __repr__(self):
        return (f"Trajectory(system={self.system.body_name}, "
                f"t0={self.t0}, tf={self.tf}, duration={self.duration})")
    
    def __str__(self):
        return f"Trajectory around {self.system.body_name}: t ∈ [{self.t0}, {self.tf}]"
    
    def __call__(self, t: float) -> OrbitalElements:
        """
        Evaluate trajectory at time t.
        Syntactic sugar for .state_at(t). Allows traj(t) syntax.
        
        Parameters:
            t: Time to query
            
        Returns:
            OrbitalElements at time t
        """
        return self.state_at(t)
    # ========== PLOTTING ==========
    # these methods are temporary until a Visualization module is established
    def plot_3d(self, n_points: int = 1000, show_body: bool = True, 
                body_color: str = 'lightblue', traj_color: str = 'red',
                body_opacity: float = 0.6) -> 'plotly.graph_objects.Figure':
        """
        Create 3D plot of trajectory with optional central body.
        
        Parameters:
            n_points: Number of points to sample trajectory (default: 1000)
            show_body: Whether to show central body sphere (default: True)
            body_color: Color of central body (default: 'lightblue')
            traj_color: Color of trajectory line (default: 'red')
            body_opacity: Opacity of central body (default: 0.6)
            
        Returns:
            Plotly Figure object
        """
        
        # Sample trajectory
        states = self.sample(n_points=n_points)
        positions = np.array(states)[:, 0:3]  # Extract position components
        
        # Create figure
        fig = go.Figure()
        
        # Add central body sphere if requested
        if show_body:
            # Get body radius from system (fallback to estimate if not available)
            try:
                radius = self.system.primary_body.radius
            except AttributeError:
                # Estimate: 1% of max trajectory extent
                max_extent = np.max(np.abs(positions))
                radius = max_extent * 0.01
            
            # Generate sphere mesh
            u = np.linspace(0, 2 * np.pi, 30)
            v = np.linspace(0, np.pi, 20)
            x = radius * np.outer(np.cos(u), np.sin(v))
            y = radius * np.outer(np.sin(u), np.sin(v))
            z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
            
            fig.add_trace(go.Surface(
                x=x, y=y, z=z,
                colorscale=[[0, body_color], [1, body_color]],
                showscale=False,
                opacity=body_opacity,
                name=self.system.body_name if hasattr(self.system, 'body_name') else 'Central Body',
                hoverinfo='name'
            ))
        
        # Add trajectory
        fig.add_trace(go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode='lines',
            line=dict(color=traj_color, width=3),
            name='Trajectory',
            hovertemplate='x: %{x:.1f}<br>y: %{y:.1f}<br>z: %{z:.1f}<extra></extra>'
        ))
        
        # Set layout
        fig.update_layout(
            scene=dict(
                xaxis_title='X [km]',
                yaxis_title='Y [km]',
                zaxis_title='Z [km]',
                aspectmode='data'  # Equal aspect ratio
            ),
            title=f'Trajectory: {self.system.body_name if hasattr(self.system, "body_name") else "Unknown Body"}',
            showlegend=True
        )
        
        return fig


    def add_to_plot(self, fig: 'plotly.graph_objects.Figure', 
                    n_points: int = 1000, color: str = 'blue',
                    name: Optional[str] = None, **kwargs) -> 'plotly.graph_objects.Figure':
        """
        Add this trajectory to an existing Plotly figure.
        
        Parameters:
            fig: Existing Plotly Figure object
            n_points: Number of points to sample trajectory (default: 1000)
            color: Color of trajectory line (default: 'blue')
            name: Legend name for this trajectory (default: 'Trajectory N')
            **kwargs: Additional arguments passed to Scatter3d
            
        Returns:
            Updated Plotly Figure object (same object, modified in place)
        """
        # Sample trajectory
        states = self.sample(n_points=n_points)
        positions = np.array(states)[:, 0:3]  # Extract position components
        
        # Default name if not provided
        if name is None:
            # Count existing scatter3d traces
            n_existing = sum(1 for trace in fig.data if isinstance(trace, go.Scatter3d))
            name = f'Trajectory {n_existing + 1}'
        
        # Add trajectory to figure
        fig.add_trace(go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode='lines',
            line=dict(color=color, width=3),
            name=name,
            hovertemplate='x: %{x:.1f}<br>y: %{y:.1f}<br>z: %{z:.1f}<extra></extra>',
            **kwargs
        ))
        
        return fig
