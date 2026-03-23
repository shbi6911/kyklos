'''Development code for an orbital trajectory handling package
Trajectory class definition
created with the assistance of Claude Sonnet 4.5 by Anthropic'''

import numpy as np
import pandas as pd
import warnings
from typing import Union, Optional, Dict, List, Tuple, Any, TYPE_CHECKING
import heyoka as hy
import plotly.graph_objects as go
from .orbital_elements import OrbitalElements, OEType
if TYPE_CHECKING:
    from .system import System
from .config import config

class Trajectory:
    """
    A trajectory segment with continuous-time state access via dense output.

    Trajectory wraps the continuous output object returned by Heyoka's
    Taylor series integrator. Rather than storing a fixed grid of states, 
    it stores the underlying Taylor polynomial coefficients, so state evaluation at any
    time is a cheap evaluation -- not an interpolation.

    Trajectory objects are not constructed directly. They are returned by
    System.propagate() and by the Trajectory.extend() and Trajectory.slice()
    methods. The associated System is immutable and stored by reference.

    State Vector Convention
    -----------------------
    The base state is always a 6-element vector [x, y, z, vx, vy, vz].
    For 2-body systems this is in km and km/s (ECI Cartesian).
    For CR3BP systems this is nondimensional (position in L_star, velocity
    in L_star/T_star). When an STM is present, the integrator state is
    extended to 42 elements (6 base + 36 flattened STM) for order=1.

    State Transition Matrix (STM)
    ------------------------------
    If the Trajectory was propagated with with_stm=True, the STM
    Phi(t, t0) is stored alongside the state in the continuous output.
    Phi is initialized to the identity at t0 and integrated forward via
    the variational equations.

    Sampling API Summary
    --------------------
    All sampling methods come in three variants based on return type:

    OrbitalElements variants (wrap result in OrbitalElements objects):
        state_at(t)           -- single time, returns OrbitalElements
        evaluate(times)       -- scalar or array, returns OrbitalElements
                                 or list of OrbitalElements
        sample(n_points)      -- uniform grid, returns list of OrbitalElements

    Raw array variants (return plain NumPy arrays, for plotting or analysis):
        state_at_raw(t)       -- single time, returns shape (6,)
        evaluate_raw(times)   -- scalar or array, returns (6,) or (n, 6)
        sample_raw(n_points)  -- uniform grid, returns (n, 6)

    STM variants (only valid if Trajectory has STM):
        get_stm(t)            -- single time, returns (6, 6)
        evaluate_stm(times)   -- scalar or array, returns (6, 6) or (n, 6, 6)
        sample_stm(n_points)  -- uniform grid, returns (n, 6, 6)

    Full state variants (base state + STM if present, otherwise warns):
        state_full(t)         -- single time, returns full integrator state
        evaluate_full(times)  -- scalar or array, returns full integrator state
        sample_full(n_points) -- uniform grid, returns full integrator state

    Trajectory Manipulation
    -----------------------
    extend(new_tf)            -- continue propagation beyond current tf,
                                 returns new Trajectory from [tf, new_tf]
    slice(t_start, t_end)     -- re-propagate a sub-interval, returns new
                                 Trajectory from [t_start, t_end]

    Both methods preserve STM settings, but the STM is reinitialized.  With or without 
    an STM, a new Trajectory object is returned. The original is unchanged.

    Parameters
    ----------
    system : System
        The dynamical environment used to generate this trajectory.
        Stored by reference; System is intended to be immutable so that associated 
        Trajectories remain valid.
    output : heyoka continuous output object
        Dense output returned by taylor_adaptive.propagate_until() with
        c_output=True. Stores Taylor polynomial coefficients over the
        integration interval.
    t0 : float
        Start time of the propagated interval [s for 2-body, nd for CR3BP]
    tf : float
        End time of the propagated interval [s for 2-body, nd for CR3BP]
    stm_order : int or None, optional
        Order of the variational equations used during propagation.
        1 means first-order STM is present (6x6 Phi). None means no STM.
        Default: None

    Attributes
    ----------
    system : System
        Reference to parent System (read-only)
    t0 : float
        Trajectory start time (read-only)
    tf : float
        Trajectory end time (read-only)
    duration : float
        Total propagated time span tf - t0 (read-only)

    Examples
    --------
    Basic propagation and state access:

    >>> sys = earth_2body()
    >>> orbit = OE(a=7000, e=0.01, i=0.5, omega=0, w=0, nu=0)
    >>> traj = sys.propagate(orbit, 0, 5400)
    >>> state = traj.state_at(2700)          # midpoint state as OrbitalElements
    >>> state = traj(2700)                   # callable syntax, equivalent
    >>> arr = traj.state_at_raw(2700)        # midpoint as raw (6,) array

    Uniform sampling:

    >>> states = traj.sample(n_points=500)   # list of OrbitalElements
    >>> arr = traj.sample_raw(n_points=500)  # (500, 6) NumPy array

    STM propagation:

    >>> traj_stm = sys.propagate(orbit, 0, 5400, with_stm=True)
    >>> phi = traj_stm.get_stm(2700)         # (6, 6) STM at t=2700 s
    >>> phis = traj_stm.sample_stm(100)      # (100, 6, 6) STM history

    Export and visualization:

    >>> df = traj.to_dataframe(n_points=1000)
    >>> fig = traj.plot_3d()

    See Also
    --------
    System.propagate : Primary method for creating Trajectory objects
    Trajectory.extend : Extend propagation beyond current tf
    Trajectory.slice : Extract a sub-interval as a new Trajectory
    """
    # ========== CONSTRUCTION ==========
    def __init__(self, system, output, t0, tf, stm_order=None):
        self._system = system
        self._output = output
        self._t0 = t0
        self._tf = tf
        self._stm_order = stm_order  # None or 1 or 2
        
    
    # ========== PROPERTY ACCESS ==========
    @property
    def system(self) -> "System":
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
    
    # ========== SAMPLING METHODS ==========
    def state_at(self, t: float, 
                 element_type: Union[OEType, str, None] = None) -> OrbitalElements:
        """
        Get orbital state at specified time.
        
        Parameters:
            t: Time to query (must be in [t0, tf])
            element_type: Desired element type (defaults to system-appropriate type)
                        Can be OEType enum or string ('cart', 'kep', 'equi', 'cr3bp')
        """
        self._validate_time(t)
        state_array = self._output(float(t)) # Heyoka requires float input
        
        # Auto-detect type from system if not specified
        if element_type is None:
            from .system import SysType
            if self.system.base_type == SysType.CR3BP:
                element_type = OEType.CR3BP
            else:
                element_type = OEType.CARTESIAN
        else:
            # Parse string to enum if needed
            element_type = OrbitalElements._parse_element_type(element_type)
        
        return OrbitalElements(state_array[:6], element_type, 
                            validate=False, system=self.system)
    
    def evaluate(self, 
                 times: Union[float, np.ndarray, list],
                 element_type: Union[OEType, str, None] 
                 = None) -> Union[OrbitalElements, list[OrbitalElements]]:
        """
        Evaluate trajectory at one or more times.
        
        Parameters:
            times: Single time or array of times
            element_type: Desired element type (defaults to system-appropriate type)
                        Can be OEType enum or string ('cart', 'kep', 'equi', 'cr3bp')
            
        Returns:
            Single OrbitalElements if times is scalar,
            list of OrbitalElements if times is array-like
        """
        # Auto-detect type from system if not specified
        if element_type is None:
            from .system import SysType
            if self.system.base_type == SysType.CR3BP:
                element_type = OEType.CR3BP
            else:
                element_type = OEType.CARTESIAN
        else:
            # Parse string to enum if needed
            element_type = OrbitalElements._parse_element_type(element_type)
        # Handle scalar input
        if isinstance(times, (int, float)):
            return self.state_at(times, element_type=element_type)
        
        # Handle array input
        times = np.asarray(times, dtype=float)
        states_array = self._output(times)  # One vectorized call to Heyoka
        return [OrbitalElements(row[:6], element_type, validate=False, 
                                system=self.system) 
                for row in states_array]
    
    def sample(self, element_type: Union[OEType, str, None] = None, 
               n_points: int = 100) -> list[OrbitalElements]:
        """
        Uniformly sample trajectory in time.
        
        Parameters:
            n_points: Number of points to sample (default: 100)
            element_type: Desired element type (defaults to system-appropriate type)
                        Can be OEType enum or string ('cart', 'kep', 'equi', 'cr3bp')
            
        Returns:
            List of OrbitalElements uniformly spaced in time
        """
        if n_points < 2:
            raise ValueError("n_points must be at least 2, use .state_at()")
        # generate evenly spaced times array
        times = np.linspace(self.t0, self.tf, n_points)
         # Auto-detect type from system if not specified
        if element_type is None:
            from .system import SysType
            if self.system.base_type == SysType.CR3BP:
                element_type = OEType.CR3BP
            else:
                element_type = OEType.CARTESIAN
        else:
            # Parse string to enum if needed
            element_type = OrbitalElements._parse_element_type(element_type)
        
        states_array = self._output(times)  # One vectorized call to Heyoka
        return [OrbitalElements(row[:6], element_type, validate=False, 
                                system=self.system) 
                for row in states_array]
    
    def state_at_raw(self, t: float) -> np.ndarray:
        """Get raw state array at time t """
        self._validate_time(t)
        # return output at time t and copy to avoid aliasing
        return self._output(float(t))[:6].copy()  # Heyoka needs float input
    
    def evaluate_raw(self, times: Union[float, np.ndarray, list]) -> np.ndarray:
        """
        Evaluate at one or more times, returning raw arrays.
        
        Parameters:
            times: Single time or array of times
            
        Returns:
            State array of shape (6,) if times is scalar,
            Array of shape (n_times, 6) if times is array-like
        """
        # Handle scalar input
        if isinstance(times, (int, float)):
            return self.state_at_raw(times)
        
        # Handle array input
        times = np.asarray(times, dtype=float)
        return self._output(times)[:, :6]
    
    def sample_raw(self, n_points: int = 100) -> np.ndarray:
        """
        Uniformly sample trajectory in time, returning raw arrays.
        
        Parameters:
            n_points: Number of points to sample (default: 100)
            
        Returns:
            Array of shape (n_points, 6) with uniformly spaced states
        """
        if n_points < 2:
            raise ValueError("n_points must be at least 2, use .state_at_raw()")
        
        times = np.linspace(self.t0, self.tf, n_points)
        return self._output(times)[:, :6]
    
    def get_stm(self, t: float) -> np.ndarray:
        """
        Get State Transition Matrix at time t.
        
        Parameters
        ----------
        t : float
            Time at which to evaluate STM
            
        Returns
        -------
        np.ndarray
            6x6 State Transition Matrix Phi(t, t0)
            
        Raises
        ------
        ValueError
            If trajectory was not propagated with STM
        """
        if self._stm_order is None:
            raise ValueError(
                "Trajectory not propagated with STM. "
                "Use with_stm=True in System.propagate()"
            )
        
        self._validate_time(t)
        
        # Query continuous output for full state
        full_state = self._output(float(t))
        
        # Extract STM (indices 6:42 for 6-state system)
        stm_flat = full_state[6:42]
        
        # Reshape to 6x6 matrix and copy to avoid aliasing
        return stm_flat.reshape(6, 6).copy()
    
    def state_full(self, t: float) -> np.ndarray:
        """Get raw state array at time t, including STM if present"""
        if self._stm_order is None:
            warnings.warn(
                "Trajectory does not have STM enabled. "
                "Use state_at_raw() instead for non-STM trajectories.",
                UserWarning,
                stacklevel=2
        )
        self._validate_time(t)
        # return a copy of output at time t to avoid aliasing
        return self._output(float(t)).copy()  # Heyoka needs float input
    
    def evaluate_stm(self, times: Union[float, np.ndarray, list]) -> np.ndarray:
        """
        Evaluate STM at one or more times.
        
        Parameters
        ----------
        times : float or array-like
            Single time or array of times
            
        Returns
        -------
        np.ndarray
            STM array of shape (6, 6) if times is scalar,
            Array of shape (n_times, 6, 6) if times is array-like
            
        Raises
        ------
        ValueError
            If trajectory was not propagated with STM
        """
        if self._stm_order is None:
            raise ValueError(
                "Trajectory not propagated with STM. "
                "Use with_stm=True in System.propagate()"
            )
        
        # Handle scalar input
        if isinstance(times, (int, float)):
            return self.get_stm(times)
        
        # Handle array input
        times = np.asarray(times, dtype=float)
        full_states = self._output(times)  # Shape: (n, 42)
        stm_flat = full_states[:, 6:42]     # Shape: (n, 36)
        
        # Reshape to (n, 6, 6)
        n_times = len(times)
        return stm_flat.reshape(n_times, 6, 6)
    
    def evaluate_full(self, times: Union[float, np.ndarray, list]) -> np.ndarray:
        """
        Evaluate at one or more times, returning raw arrays, including STM if present.
        
        Parameters:
            times: Single time or array of times
            
        Returns:
            State array of shape (6,) if times is scalar,
            Array of shape (n_times, 6) if times is array-like
        """
        if self._stm_order is None:
            warnings.warn(
                "Trajectory does not have STM enabled. "
                "Use evaluate_raw() instead for non-STM trajectories.",
                UserWarning,
                stacklevel=2
        )
        # Handle scalar input
        if isinstance(times, (int, float)):
            return self.state_full(times)
        
        # Handle array input
        times = np.asarray(times, dtype=float)
        return self._output(times)
    
    def sample_stm(self, n_points: int = 100) -> np.ndarray:
        """
        Uniformly sample STM in time.
        
        Parameters
        ----------
        n_points : int, optional
            Number of points to sample (default: 100)
            
        Returns
        -------
        np.ndarray
            Array of shape (n_points, 6, 6) with uniformly spaced STMs
            
        Raises
        ------
        ValueError
            If trajectory was not propagated with STM or if n_points < 2
        """
        if self._stm_order is None:
            raise ValueError(
                "Trajectory not propagated with STM. "
                "Use with_stm=True in System.propagate()"
            )
        
        if n_points < 2:
            raise ValueError("n_points must be at least 2, use .get_stm()")
        
        times = np.linspace(self.t0, self.tf, n_points)
        full_states = self._output(times)  # Shape: (n, 42)
        stm_flat = full_states[:, 6:42]     # Shape: (n, 36)
        
        # Reshape to (n, 6, 6)
        return stm_flat.reshape(n_points, 6, 6)
    
    def sample_full(self, n_points: int = 100) -> np.ndarray:
        """
        Uniformly sample trajectory in time, returning raw arrays, including STM.
        
        Parameters:
            n_points: Number of points to sample (default: 100)
            
        Returns:
            Array of shape (n_points, 6) with uniformly spaced states
        """
        if self._stm_order is None:
            warnings.warn(
                "Trajectory does not have STM enabled. "
                "Use evaluate_raw() instead for non-STM trajectories.",
                UserWarning,
                stacklevel=2
        )
        
        if n_points < 2:
            raise ValueError("n_points must be at least 2, use .state_at_raw()")
        
        times = np.linspace(self.t0, self.tf, n_points)
        return self._output(times)
    
    # ========== UTILITY METHODS ==========
    def _validate_time(self, t: float):
        """Validate that time is within trajectory bounds."""
        t_min = min(self.t0, self.tf)
        t_max = max(self.t0, self.tf)
        
        if not (t_min <= t <= t_max):
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
            times = np.asarray(times, dtype=float)
        
        # Evaluate states to a numpy array
        states = self.evaluate_raw(times)
        
        # Build data dictionary using array slicing
        data = {
            'time': times,
            'x': states[:, 0],
            'y': states[:, 1],
            'z': states[:, 2],
            'vx': states[:, 3],
            'vy': states[:, 4],
            'vz': states[:, 5],
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
        
        # Ensure proper type for new tf
        new_tf = float(new_tf)

        # Get the final state of current trajectory
        final_state = self.state_at(self.tf)

        # Preserve STM settings from original trajectory
        with_stm = self._stm_order is not None
        stm_order = self._stm_order if with_stm else 1

        # Extract satellite from kwargs if provided
        satellite = propagation_kwargs.get('satellite', None)
        
        # Continue propagation from final state
        return self._system.propagate(
            final_state, 
            self.tf, 
            new_tf,
            with_stm=with_stm,
            stm_order=stm_order,
            satellite=satellite
        )
    
    def slice(self, t_start: float, t_end: float, **propagation_kwargs) -> 'Trajectory':
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
        # Ensure proper type for input times
        t_start = float(t_start)
        t_end = float(t_end)

        # Preserve STM settings from original trajectory
        with_stm = self._stm_order is not None
        stm_order = self._stm_order if with_stm else 1

        # Extract satellite_params from kwargs if provided
        satellite = propagation_kwargs.get('satellite', None)
        
        # set initial state 
        new_initial_state = self.state_at(t_start)
        # Create new Trajectory with same integrator but different time bounds
        return self._system.propagate(
            new_initial_state, 
            t_start, 
            t_end,
            with_stm=with_stm,
            stm_order=stm_order,
            satellite=satellite
        )

    # ========== SPECIAL METHODS ==========
    def __repr__(self):
        return (f"Trajectory(system={self.system.primary_body.name}, "
                f"t0={self.t0}, tf={self.tf}, duration={self.duration})")
    
    def __str__(self):
        return (f"Trajectory around {self.system.primary_body.name}: "
                f"t ∈ [{self.t0}, {self.tf}]")
    
    def __call__(self, t: float, element_type: Union[OEType, str, None] = None
                 ) -> OrbitalElements:
        """
        Evaluate trajectory at time t.
        Syntactic sugar for .state_at(t). Allows traj(t) syntax.
        
        Parameters:
            t: Time to query
            
        Returns:
            OrbitalElements at time t
        """
        return self.state_at(t, element_type)
    
    # ========== PLOTTING ==========
    # these methods are temporary until a Visualization module is established
    def plot_3d(self, n_points: int | None = None, show_body: bool = True, 
            body_color: str | None = None, traj_color: str | None = None,
            body_opacity: float | None = None, 
            proximity_threshold: float | None = None,
            renderer: str | None = None) -> go.Figure:
        """
        Create 3D plot of trajectory with optional central body.
        
        Parameters:
            n_points : int, optional
                Number of points to sample trajectory.
                If None, uses config.DEFAULT_PLOT_POINTS (default: None)
            show_body: Whether to show central body sphere (default: True)
            body_color : str, optional
                Color of central body.
                If None, uses config.DEFAULT_BODY_COLOR (default: None)
            traj_color: str, optional
                Color of trajectory line.
                If None, uses config.DEFAULT_TRAJ_COLOR (default: None)
            body_opacity: float, optional
                Opacity of central body.
                If None, uses config.DEFAULT_BODY_OPACITY (default: None)
            proximity_threshold: float, optional
                Show body if trajectory within this many radii.
                If None, uses config.PROXIMITY_THRESHOLD (default: 10)
            renderer: str, optional
                controls the Plotly renderer used to display plots
                if None, uses config.RENDERER (default: 'browser')
            
        Returns:
            Plotly Figure object
        """
        from .system import SysType
        import plotly.io as pio

        # Apply config defaults where user didn't specify
        if n_points is None:
            n_points = config.DEFAULT_PLOT_POINTS
        if body_color is None:
            body_color = config.DEFAULT_BODY_COLOR
        if traj_color is None:
            traj_color = config.DEFAULT_TRAJ_COLOR
        if body_opacity is None:
            body_opacity = config.DEFAULT_BODY_OPACITY
        if proximity_threshold is None:
            proximity_threshold = config.PROXIMITY_THRESHOLD
        if renderer is None:
            pio.renderers.default = config.RENDERER
        
        # Sample trajectory
        states = self.sample_raw(n_points=n_points)
        positions = states[:, 0:3]
        
        # Create figure
        fig = go.Figure()
        
        # Determine which bodies to show
        is_cr3bp = self.system.base_type == SysType.CR3BP
        show_primary = False
        show_secondary = False
        
        if show_body:
            if is_cr3bp:
                mu = self.system.mass_ratio
                assert mu is not None  # Always true for CR3BP
                
                # Body positions in rotating frame
                p1_pos = np.array([-mu, 0, 0])
                p2_pos = np.array([1 - mu, 0, 0])
                
                # Body radii
                r1 = self.system.primary_body.radius_nd
                r2 = self.system.secondary_body.radius_nd
                
                # Calculate minimum distance from trajectory to each body
                dist_to_p1 = np.linalg.norm(positions - p1_pos, axis=1)
                dist_to_p2 = np.linalg.norm(positions - p2_pos, axis=1)
                
                min_dist_p1 = np.min(dist_to_p1)
                min_dist_p2 = np.min(dist_to_p2)
                
                # Show body if trajectory gets within threshold
                show_primary = (min_dist_p1 < proximity_threshold * r1)
                show_secondary = (min_dist_p2 < proximity_threshold * r2)
                
                # Inform user if no bodies shown
                if not show_primary and not show_secondary:
                    print(f"Trajectory does not approach either body within "
                        f"{proximity_threshold} radii.")
                    print(f"  Closest approach to primary: {min_dist_p1/r1:.1f} radii")
                    print(f"  Closest approach to secondary: {min_dist_p2/r2:.1f} radii")
                    print(f"  No bodies shown in plot.")
            else:
                # For 2-body systems, always show primary if show_body=True
                show_primary = True
        
        # Add primary body sphere if needed
        if show_primary:
            if is_cr3bp:
                mu = self.system.mass_ratio
                self._add_sphere_to_plot(
                    fig, 
                    center=(-mu, 0, 0),
                    radius=self.system.primary_body.radius_nd,
                    color=body_color,
                    opacity=body_opacity,
                    name="Primary"
                )
            else:
                # 2-body: central body at origin
                self._add_sphere_to_plot(
                    fig,
                    center=(0, 0, 0),
                    radius=self.system.primary_body.radius,
                    color=body_color,
                    opacity=body_opacity,
                    name="Central Body"
                )
        
        # Add secondary body sphere if needed
        if show_secondary and is_cr3bp:
            mu = self.system.mass_ratio
            assert mu is not None   # always true if CR3BP
            self._add_sphere_to_plot(
                fig,
                center=(1 - mu, 0, 0),
                radius=self.system.secondary_body.radius_nd,
                color=body_color,
                opacity=body_opacity,
                name="Secondary"
            )
        
        # Add trajectory
        fig.add_trace(go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode='lines',
            line=dict(color=traj_color, width=3),
            name='Trajectory',
            hovertemplate='x: %{x:.6f}<br>y: %{y:.6f}<br>z: %{z:.6f}<extra></extra>'
        ))
        
        # Set layout
        units = 'nd' if is_cr3bp else 'km'
        fig.update_layout(
            scene=dict(
                xaxis_title=f'X [{units}]',
                yaxis_title=f'Y [{units}]',
                zaxis_title=f'Z [{units}]',
                aspectmode='data'
            ),
            title='CR3BP Trajectory' if is_cr3bp else 'Orbital Trajectory',
            showlegend=True
        )
        
        return fig

    def _add_sphere_to_plot(self, fig, center, radius, color, opacity, name):
        """Helper to add a sphere to the plot at specified center."""
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 20)
        
        x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
        y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
        z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
        
        fig.add_trace(go.Surface(
            x=x, y=y, z=z,
            colorscale=[[0, color], [1, color]],
            showscale=False,
            opacity=opacity,
            name=name,
            hoverinfo='name'
        ))


    def add_to_plot(self, fig: go.Figure, 
                    n_points: int | None = None, color: str | None = None,
                    name: Optional[str] = None, **kwargs) -> go.Figure:
        """
        Add this trajectory to an existing Plotly figure.
        
        Parameters:
            fig: Existing Plotly Figure object
            n_points : int, optional
                Number of points to sample trajectory.
                If None, uses config.DEFAULT_PLOT_POINTS (default: None)
            color: str, optional
                Color of trajectory line.
                If None, uses config.DEFAULT_TRAJ_COLOR_ADD (default: None)
            name: Legend name for this trajectory (default: 'Trajectory N')
            **kwargs: Additional arguments passed to Scatter3d
            
        Returns:
            Updated Plotly Figure object (same object, modified in place)
        """
        # Apply config defaults where user didn't specify
        if n_points is None:
            n_points = config.DEFAULT_PLOT_POINTS
        if color is None:
            color = config.DEFAULT_TRAJ_COLOR

        # Sample trajectory
        states = self.sample_raw(n_points=n_points)
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
