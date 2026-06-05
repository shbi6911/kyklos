'''Development code for an orbital trajectory handling package
Trajectory class definition
created with the assistance of Claude Sonnet by Anthropic'''

import numpy as np
import pandas as pd
import warnings
import heyoka as hy
import plotly.graph_objects as go
import bisect
from typing import Union, Optional, Dict, List, Tuple, Any, TYPE_CHECKING
from abc import ABC, abstractmethod
from .orbital_elements import OrbitalElements, OEType
from __future__ import annotations
if TYPE_CHECKING:
    from .system import System
from .config import config
from .utils import validation_error

class Trajectory:
    """
    A trajectory with continuous-time state access via dense output.

    A Trajectory consists of one or more propagated segments connected
    by JunctionNodes, bracketed by a StartBoundaryNode and EndBoundaryNode.
    A single-segment trajectory is the degenerate case with no internal
    junction nodes.

    Trajectory objects are not constructed directly. They are returned by
    System.propagate() and by the Trajectory.extend() and Trajectory.slice()
    methods.

    State Vector Convention
    -----------------------
    The base state is always a 6-element vector [x, y, z, vx, vy, vz].
    For 2-body systems this is in km and km/s (ECI Cartesian).
    For CR3BP systems this is nondimensional (position in L_star, velocity
    in L_star/T_star). When an STM is present, the integrator state is
    extended to 42 elements (6 base + 36 flattened STM) for order=1.

    State Transition Matrix (STM)
    ------------------------------
    If the Trajectory was propagated with with_stm=True, the STM is
    stored in each segment's continuous output, initialized to identity
    at the segment's own t0. Two STM interfaces are provided:

    get_stm(t) returns the composite Phi(t, t0) across all segments and
    junction maneuver Jacobians up to time t -- the full sensitivity of
    the state at t to the initial state.

    get_stm_seg(t) returns the segment-local Phi_k(t, t_k) referenced
    to the start of whichever segment contains t -- used by the shooting
    corrector, which works with individual segment STMs.

    Sampling API Summary
    --------------------
    OrbitalElements variants:
        state_at(t)           -- single time, returns OrbitalElements
        evaluate(times)       -- scalar or array of times
        sample(n_points)      -- uniform grid

    Raw array variants:
        state_at_raw(t)       -- single time, returns (6,)
        evaluate_raw(times)   -- scalar or array, returns (6,) or (n, 6)
        sample_raw(n_points)  -- uniform grid, returns (n, 6)

    Composite STM variants (Phi referenced to t0):
        get_stm(t)            -- single time, returns (6, 6)
        evaluate_stm(times)   -- scalar or array
        sample_stm(n_points)  -- uniform grid

    Segment-local STM variants (Phi referenced to segment t0):
        get_stm_seg(t)            -- single time, returns (6, 6)
        evaluate_stm_seg(times)   -- scalar or array
        sample_stm_seg(n_points)  -- uniform grid

    Full integrator state variants:
        state_full(t)         -- single time, full Heyoka state vector
        evaluate_full(times)  -- scalar or array
        sample_full(n_points) -- uniform grid

    Parameters
    ----------
    system : System
        The dynamical environment used to generate this trajectory.
    outputs : list
        List of Heyoka continuous output objects, one per segment.
    start_node : BoundaryNode
        Node at t0 of the trajectory.
    end_node : BoundaryNode
        Node at tf of the trajectory.
    junction_nodes : list of JunctionNode, optional
        Internal nodes between segments. len must equal len(outputs) - 1.
        Default: empty list (single-segment trajectory).
    stm_order : int or None, optional
        Order of variational equations. 1 = first-order STM. None = no STM.
        Must be consistent across all segments. Default: None.

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
    def __init__(self, system, outputs, junction_nodes=None, stm_order=None,
             start_node=None, end_node=None):
        """
        Parameters
        ----------
        system : System
            The dynamical environment for this trajectory.
        outputs : list or single Heyoka continuous output
            Heyoka continuous output object(s), one per segment. A single
            output is wrapped in a list automatically.
        junction_nodes : list of JunctionNode, optional
            Internal nodes between segments. Must contain len(outputs) - 1
            elements. Default: empty list (single-segment trajectory).
        stm_order : int or None, optional
            Order of variational equations. 1 = first-order STM present.
            None = no STM. Must be consistent across all segments.
            Default: None.
        start_node : BoundaryNode, optional
            Node at trajectory start. If None, a StartBoundaryNode is
            constructed from the first output. Default: None.
        end_node : BoundaryNode, optional
            Node at trajectory end. If None, an EndBoundaryNode is
            constructed from the last output. Default: None.
        """
        n_segments = len(outputs)

        if junction_nodes is None:
            junction_nodes = [
                Trajectory._infer_junction(outputs[k], outputs[k + 1])
                for k in range(n_segments - 1)
            ]
        else:
            # Explicitly provided — validate count and times
            if len(junction_nodes) != n_segments - 1:
                raise ValueError(
                    f"Expected {n_segments - 1} junction node(s) for "
                    f"{n_segments} segment(s), got {len(junction_nodes)}."
                )
            for k, node in enumerate(junction_nodes):
                expected = float(outputs[k].bounds[-1])
                if not np.isclose(node.time, expected,
                                rtol=config.EQUALITY_RTOL,
                                atol=config.EQUALITY_ATOL):
                    raise ValueError(
                        f"Junction node {k} time ({node.time:.6g}) does not match "
                        f"segment {k} end time ({expected:.6g})."
                    )

        # Extract boundary times from Heyoka output objects.
        # Use float() immediately to avoid holding references to Heyoka's
        # internal numpy arrays, which may be overwritten on subsequent calls.
        t0 = float(outputs[0].bounds[0])
        tf = float(outputs[-1].bounds[-1])

        # Validate time adjacency of consecutive segments
        for k in range(n_segments - 1):
            tf_k  = float(outputs[k].bounds[-1])
            t0_k1 = float(outputs[k + 1].bounds[0])
            if not np.isclose(tf_k, t0_k1,
                            rtol=config.EQUALITY_RTOL,
                            atol=config.EQUALITY_ATOL):
                raise ValueError(
                    f"Segment {k} end time ({tf_k:.6g}) does not match "
                    f"segment {k + 1} start time ({t0_k1:.6g}). "
                    f"Segments must be time-adjacent."
                )

        # Validate junction node times against segment boundaries
        for k, node in enumerate(junction_nodes):
            expected = float(outputs[k].bounds[-1])
            if not np.isclose(node.time, expected,
                            rtol=config.EQUALITY_RTOL,
                            atol=config.EQUALITY_ATOL):
                raise ValueError(
                    f"Junction node {k} time ({node.time:.6g}) does not match "
                    f"segment {k} end time ({expected:.6g})."
                )

        self._system = system
        self._outputs = outputs
        self._junction_nodes = list(junction_nodes)
        self._stm_order = stm_order

        # Construct default boundary nodes from outputs if not provided.
        # Copy output arrays immediately to avoid Heyoka buffer aliasing.
        if start_node is None:
            initial_state = outputs[0](t0)[:6].copy()
            self._start_node = StartBoundaryNode(t0, initial_state)
        else:
            if not np.isclose(start_node.time, t0,
                              rtol=config.EQUALITY_RTOL,
                              atol=config.EQUALITY_ATOL):
                raise ValueError(
                    f"start_node time ({start_node.time:.6g}) does not match "
                    f"segment start time ({t0:.6g})."
                )
            self._start_node = start_node

        if end_node is None:
            final_state = outputs[-1](tf)[:6].copy()
            self._end_node = EndBoundaryNode(tf, final_state)
        else:
            if not np.isclose(end_node.time, tf,
                              rtol=config.EQUALITY_RTOL,
                              atol=config.EQUALITY_ATOL):
                raise ValueError(
                    f"end_node time ({end_node.time:.6g}) does not match "
                    f"segment end time ({tf:.6g})."
                )
            self._end_node = end_node

        # Cache junction times for O(log n) segment dispatch
        self._junction_times = [node.time for node in self._junction_nodes]
        
    # ========== PROPERTY ACCESS ==========
    @property
    def system(self) -> System:
        """The dynamical environment for this trajectory."""
        return self._system
    
    @property
    def t0(self) -> float:
        """Trajectory start time."""
        return self._start_node.time

    @property
    def tf(self) -> float:
        """Trajectory end time."""
        return self._end_node.time

    @property
    def duration(self) -> float:
        """Total time span tf - t0."""
        return self.tf - self.t0
    
    @property
    def start_node(self) -> BoundaryNode:
        """Node at the start of this trajectory."""
        return self._start_node

    @property
    def end_node(self) -> BoundaryNode:
        """Node at the end of this trajectory."""
        return self._end_node

    @property
    def junction_nodes(self) -> list:
        """
        Internal junction nodes between segments.

        Returns a defensive copy. Empty for single-segment trajectories.
        """
        return list(self._junction_nodes)

    @property
    def n_segments(self) -> int:
        """Number of propagated segments."""
        return len(self._outputs)

    @property
    def is_multisegment(self) -> bool:
        """True if this trajectory contains more than one segment."""
        return len(self._outputs) > 1

    @property
    def stm_order(self) -> Optional[int]:
        """
        Order of the variational equations used during propagation.
        1 indicates a first-order STM is present. None means no STM.
        """
        return self._stm_order

    @property
    def has_stm(self) -> bool:
        """True if this trajectory was propagated with an STM."""
        return self._stm_order is not None
    
    @property
    def times(self) -> list:
        """
        Structural node times for this trajectory.

        Returns [t0, t_j1, ..., t_jN, tf] where t_j1...t_jN are junction
        times. For a single-segment trajectory, returns [t0, tf].
        """
        return (
            [self._start_node.time]
            + [node.time for node in self._junction_nodes]
            + [self._end_node.time]
        )

    @property
    def n_steps(self) -> int | list[int]:
        """
        Number of integration steps per segment.

        Returns a scalar int for single-segment trajectories, or a list
        of ints for multi-segment trajectories.
        """
        if self.n_segments == 1:
            return self._outputs[0].n_steps
        return [output.n_steps for output in self._outputs]

    @property
    def step_times(self) -> np.ndarray:
        """
        Integration step times across all segments as a single sorted array.

        Junction times that appear at the boundary of two adjacent segments
        are deduplicated. Useful for inspecting integrator step distribution
        and diagnosing integration quality across a multi-segment trajectory.
        """
        all_times = np.concatenate(
            [output.times.copy() for output in self._outputs]
        )
        return np.unique(all_times)
    
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
    
    # ========== STATIC METHODS ==========
    @staticmethod
    def _infer_junction(output_pre, output_post) -> JunctionNode:
        """
        Infer the appropriate JunctionNode type from adjacent segment endpoints.

        Examines the state continuity at the junction time and returns the
        most specific node type consistent with the observed discontinuity.

        Parameters
        ----------
        output_pre : Heyoka continuous output
            Output object for the incoming segment.
        output_post : Heyoka continuous output
            Output object for the outgoing segment.

        Returns
        -------
        JunctionNode
            NullJunctionNode if full state is continuous within tolerance.
            ImpulsiveJunctionNode if position is continuous but velocity is not.
            FreeJunctionNode if position is discontinuous.
        """
        t_junc = float(output_pre.bounds[-1])
        pre_state  = output_pre(t_junc)[:6].copy()
        post_state = output_post(t_junc)[:6].copy()

        # Full state continuous
        if np.allclose(pre_state, post_state,
                    rtol=config.EQUALITY_RTOL,
                    atol=config.EQUALITY_ATOL):
            return NullJunctionNode(t_junc, pre_state, post_state)

        # Position continuous, velocity discontinuous
        if np.allclose(pre_state[:3], post_state[:3],
                    rtol=config.EQUALITY_RTOL,
                    atol=config.EQUALITY_ATOL):
            return ImpulsiveJunctionNode(t_junc,
                                        pre_state=pre_state,
                                        post_state=post_state)

        # Full state discontinuous
        return FreeJunctionNode(t_junc, pre_state, post_state)
    
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

 # Node class, used by Trajectory to define state mappings between segments

class Node(ABC):
    """
    Abstract base class for trajectory nodes.
    
    A Node represents a point in a trajectory at which a state mapping
    may occur. The mapping may be trivial (identity, for null nodes) or
    represent a physical event such as an impulsive maneuver.
    
    Subclasses
    ----------
    BoundaryNode : start or end of a Trajectory
    JunctionNode : internal node in a Trajectory, shared between
                   two adjacent segments
    
    Notes
    -----
    Node is abstract and cannot be instantiated directly. Subclasses must
    implement pre_state and post_state. Derived quantities (delta_v,
    state_defect) are implemented here in terms of those abstract
    properties and are inherited for free.
    """
    
    # ========== CONSTRUCTION ==========
    def __init__(self, time: float):
        """
        Parameters
        ----------
        time : float
            Time at which this node occurs [s or nondimensional]
        """
        self._time = float(time)
    
    # ========== PROPERTY ACCESS ==========
    @property
    def time(self) -> float:
        """Time at which this node occurs."""
        return self._time
    
    @property
    @abstractmethod
    def pre_state(self) -> Optional[np.ndarray]:
        """
        State on the incoming side of this node [km, km/s].
        None if this node has no incoming segment (e.g. a start boundary).
        """
        ...
    
    @property
    @abstractmethod
    def post_state(self) -> Optional[np.ndarray]:
        """
        State on the outgoing side of this node [km, km/s].
        None if this node has no outgoing segment (e.g. an end boundary).
        """
        ...
    
    @property
    def delta_v(self) -> Optional[np.ndarray]:
        """
        Velocity change at this node [km/s].
        None if either pre_state or post_state is unavailable.
        """
        if self.pre_state is None or self.post_state is None:
            return None
        return self.post_state[3:6] - self.pre_state[3:6]
    
    @property
    def state_defect(self) -> Optional[np.ndarray]:
        """
        Full 6-element state discontinuity at this node.
        Zero vector for a continuous node. None if either side unavailable.
        """
        if self.pre_state is None or self.post_state is None:
            return None
        return self.post_state - self.pre_state
    
    # ========== VALIDATION ==========
    @staticmethod
    def _validate_state(state: np.ndarray, name: str = 'state') -> np.ndarray:
        """
        Validate and condition a state vector.
        
        Parameters
        ----------
        state : array-like
            Candidate state vector
        name : str, optional
            Name for error messages (default: 'state')
            
        Returns
        -------
        np.ndarray
            Validated 6-element float array
        """
        state = np.asarray(state, dtype=float)
        if state.shape != (6,):
            raise ValueError(
                f"{name} must be a 6-element vector, got shape {state.shape}"
            )
        if not np.all(np.isfinite(state)):
            raise ValueError(f"{name} contains NaN or Inf values")
        return state

# ========== BOUNDARY NODES ==========
class BoundaryNode(Node):
    """
    Abstract base for nodes at the start or end of a Trajectory segment.
    
    A BoundaryNode has physical state on exactly one side. The other side
    returns None, reflecting the absence of a connected segment.
    
    Subclasses
    ----------
    StartBoundaryNode : node at t0 of a Trajectory, post_state populated
    EndBoundaryNode : node at tf of a Trajectory, pre_state populated
    ImpulsiveBoundaryNode : node carrying maneuver context on both sides
    """
    pass


class StartBoundaryNode(BoundaryNode):
    """
    Node at the start of a Trajectory segment.
    
    Carries the initial state of the segment as post_state. pre_state
    is None, reflecting the absence of an incoming segment. Used as
    the default start node when no maneuver context is needed.
    
    Parameters
    ----------
    time : float
        Start time of the trajectory segment [s or nondimensional]
    post_state : array-like
        Initial state vector [x, y, z, vx, vy, vz] [km, km/s]
    """
    
    def __init__(self, time: float, post_state: np.ndarray):
        super().__init__(time)
        self._post_state = self._validate_state(post_state, 'post_state')
        self._post_state = self._post_state.copy()
        self._post_state.flags.writeable = False
    
    @property
    def pre_state(self) -> None:
        """No incoming segment. Always None."""
        return None
    
    @property
    def post_state(self) -> np.ndarray:
        """Initial state of the trajectory segment [km, km/s]."""
        return self._post_state
    
    def __repr__(self) -> str:
        return f"StartBoundaryNode(t={self.time:.6g})"


class EndBoundaryNode(BoundaryNode):
    """
    Node at the end of a Trajectory segment.
    
    Carries the final state of the segment as pre_state. post_state
    is None, reflecting the absence of an outgoing segment. Used as
    the default end node when no maneuver context is needed.
    
    Parameters
    ----------
    time : float
        End time of the trajectory segment [s or nondimensional]
    pre_state : array-like
        Final state vector [x, y, z, vx, vy, vz] [km, km/s]
    """
    
    def __init__(self, time: float, pre_state: np.ndarray):
        super().__init__(time)
        self._pre_state = self._validate_state(pre_state, 'pre_state')
        self._pre_state = self._pre_state.copy()
        self._pre_state.flags.writeable = False
    
    @property
    def pre_state(self) -> np.ndarray:
        """Final state of the trajectory segment [km, km/s]."""
        return self._pre_state
    
    @property
    def post_state(self) -> None:
        """No outgoing segment. Always None."""
        return None
    
    def __repr__(self) -> str:
        return f"EndBoundaryNode(t={self.time:.6g})"
    
class ImpulsiveBoundaryNode(BoundaryNode):
    """
    Boundary node carrying maneuver context for trajectory expansion.
    
    Used when a trajectory segment begins or ends with a known impulsive
    burn and the pre-burn state should be preserved to allow future
    expansion into a multi-segment Trajectory.
    
    Accepts any two of pre_state, post_state, and delta_v. The third
    is derived. Position continuity is always enforced: an impulsive
    maneuver changes only velocity, not position.
    
    Parameters
    ----------
    time : float
        Time at which the maneuver occurs [s or nondimensional]
    pre_state : array-like, optional
        State before the maneuver [x, y, z, vx, vy, vz] [km, km/s]
    post_state : array-like, optional
        State after the maneuver [x, y, z, vx, vy, vz] [km, km/s]
    delta_v : array-like, optional
        Velocity change vector [dvx, dvy, dvz] [km/s]
    
    Notes
    -----
    Exactly two of pre_state, post_state, delta_v must be provided.
    Derivation rules:
      pre_state  + delta_v   -> post_state = pre_state + [0, 0, 0, delta_v]
      post_state + delta_v   -> pre_state  = post_state - [0, 0, 0, delta_v]
      pre_state  + post_state -> delta_v   = post_state[3:6] - pre_state[3:6]
                                (position continuity enforced)
    
    For nodes with full state discontinuity, use FreeJunctionNode instead.
    """
    
    def __init__(self, time: float,
                 pre_state: Optional[np.ndarray] = None,
                 post_state: Optional[np.ndarray] = None,
                 delta_v: Optional[np.ndarray] = None):
        super().__init__(time)
        
        # Count provided arguments
        n_provided = sum(x is not None for x in [pre_state, post_state, delta_v])
        if n_provided != 2:
            raise ValueError(
                f"Exactly two of pre_state, post_state, delta_v must be "
                f"provided. Got {n_provided}."
            )
        
        # Validate delta_v if provided
        if delta_v is not None:
            delta_v = np.asarray(delta_v, dtype=float)
            if delta_v.shape != (3,):
                raise ValueError(
                    f"delta_v must be a 3-element vector, "
                    f"got shape {delta_v.shape}"
                )
            if not np.all(np.isfinite(delta_v)):
                raise ValueError("delta_v contains NaN or Inf values")
        
        # Derive missing quantity
        if pre_state is not None and delta_v is not None:
            # pre_state + delta_v -> post_state
            # Position continuity enforced by construction
            pre_state = self._validate_state(pre_state, 'pre_state')
            dv_full = np.concatenate([np.zeros(3), delta_v])
            post_state = pre_state + dv_full
        
        elif post_state is not None and delta_v is not None:
            # post_state + delta_v -> pre_state
            # Position continuity enforced by construction
            post_state = self._validate_state(post_state, 'post_state')
            dv_full = np.concatenate([np.zeros(3), delta_v])
            pre_state = post_state - dv_full
        
        else:
            # pre_state + post_state -> delta_v
            # Position continuity must be explicitly checked
            assert pre_state is not None and post_state is not None
            pre_state = self._validate_state(pre_state, 'pre_state')
            post_state = self._validate_state(post_state, 'post_state')
            pos_discont = np.linalg.norm(post_state[:3] - pre_state[:3])
            if not np.allclose(pre_state[:3], post_state[:3],
                               rtol=config.EQUALITY_RTOL,
                               atol=config.EQUALITY_ATOL):
                raise ValueError(
                    f"Position must be continuous for an impulsive maneuver. "
                    f"Position discontinuity: {pos_discont:.6e} km. "
                    f"Use FreeJunctionNode for nodes with position discontinuity."
                )
        
        # Store as immutable arrays
        self._pre_state = pre_state.copy()
        self._pre_state.flags.writeable = False
        self._post_state = post_state.copy()
        self._post_state.flags.writeable = False
    
    @property
    def pre_state(self) -> np.ndarray:
        """State before the maneuver [km, km/s]."""
        return self._pre_state
    
    @property
    def post_state(self) -> np.ndarray:
        """State after the maneuver [km, km/s]."""
        return self._post_state
    
    @property
    def delta_v(self) -> np.ndarray:  # never None in this subclass
        """Velocity change at this node [km/s]."""
        return self._post_state[3:6] - self._pre_state[3:6]
    
    def __repr__(self) -> str:
        dv_mag = np.linalg.norm(self.delta_v)
        return (f"ImpulsiveBoundaryNode(t={self.time:.6g}, "
                f"|dv|={dv_mag:.6g} km/s)")

class JunctionNode(Node):
    """
    Abstract base for nodes internal to a multi-segment Trajectory.
    
    A JunctionNode sits between two adjacent trajectory segments and
    always has physical state on both sides. Unlike BoundaryNode,
    neither pre_state nor post_state is ever None.
    
    Adds maneuver_jacobian() as a required abstract method, since the
    Jacobian is only meaningful when segments exist on both sides.
    Narrows return types of pre_state, post_state, delta_v, and
    state_defect to np.ndarray, removing Optional from the base class
    signatures.
    
    Subclasses
    ----------
    NullJunctionNode : continuous junction, no maneuver
    ImpulsiveJunctionNode : velocity discontinuity, position continuous
    FreeJunctionNode : full state discontinuity, for shooting iterates
    """
    
    @property
    @abstractmethod
    def pre_state(self) -> np.ndarray:
        """State on the incoming side [km, km/s]. Never None."""
        ...
    
    @property
    @abstractmethod
    def post_state(self) -> np.ndarray:
        """State on the outgoing side [km, km/s]. Never None."""
        ...
    
    @property
    def delta_v(self) -> np.ndarray:
        """Velocity change at this node [km/s]. Never None."""
        return self.post_state[3:6] - self.pre_state[3:6]
    
    @property
    def state_defect(self) -> np.ndarray:
        """Full 6-element state discontinuity. Never None."""
        return self.post_state - self.pre_state
    
    @abstractmethod
    def maneuver_jacobian(self) -> np.ndarray:
        """
        Sensitivity of post_state to pre_state.
        
        Used for STM composition across this junction in a
        multi-segment Trajectory.
        
        Returns
        -------
        np.ndarray
            6x6 matrix M = d(post_state)/d(pre_state)
        """
        ...


class NullJunctionNode(JunctionNode):
    """
    Continuous junction with no maneuver.
    
    Used when two segments connect smoothly — for example, when a
    single trajectory is split at an interior time, or when segments
    connect across a model boundary with matching states. pre_state
    and post_state are conceptually identical, and mathematically close 
    within a small tolerance, set by package config.
    
    Parameters
    ----------
    time : float
        Junction time [s or nondimensional]
    state : array-like
        Continuous state at the junction [x, y, z, vx, vy, vz] [km, km/s]
    """
    
    def __init__(self, time: float,
             pre_state: np.ndarray,
             post_state: np.ndarray):
        super().__init__(time)
        self._pre_state = self._validate_state(pre_state, 'pre_state')
        self._post_state = self._validate_state(post_state, 'post_state')
        if not np.allclose(pre_state, post_state,
                   rtol=config.EQUALITY_RTOL,
                   atol=config.EQUALITY_ATOL):
            defect_mag = np.linalg.norm(post_state - pre_state)
            validation_error(
                f"NullJunctionNode states differ by {defect_mag:.3e}. "
                f"For intentional discontinuities use ImpulsiveJunctionNode "
                f"or FreeJunctionNode."
            )
        self._pre_state = pre_state.copy()
        self._pre_state.flags.writeable = False
        self._post_state = post_state.copy()
        self._post_state.flags.writeable = False
    
    @property
    def pre_state(self) -> np.ndarray:
        """State on the incoming side [km, km/s]."""
        return self._pre_state
    
    @property
    def post_state(self) -> np.ndarray:
        """State on the outgoing side [km, km/s]."""
        return self._post_state
    
    def maneuver_jacobian(self) -> np.ndarray:
        """Identity matrix — no maneuver, no state sensitivity."""
        return np.eye(6)
    
    def __repr__(self) -> str:
        return f"NullJunctionNode(t={self.time:.6g})"


class ImpulsiveJunctionNode(JunctionNode):
    """
    Junction node representing a fixed impulsive maneuver.
    
    Accepts any two of pre_state, post_state, and delta_v. The third
    is derived. Position continuity is enforced: an impulsive maneuver
    changes only velocity, not position.
    
    Parameters
    ----------
    time : float
        Time of the maneuver [s or nondimensional]
    pre_state : array-like, optional
        State before the maneuver [x, y, z, vx, vy, vz] [km, km/s]
    post_state : array-like, optional
        State after the maneuver [x, y, z, vx, vy, vz] [km, km/s]
    delta_v : array-like, optional
        Velocity change vector [dvx, dvy, dvz] [km/s]
    
    Notes
    -----
    Exactly two of pre_state, post_state, delta_v must be provided.
    Derivation rules:
      pre_state  + delta_v    -> post_state = pre_state + [0, 0, 0, delta_v]
      post_state + delta_v    -> pre_state  = post_state - [0, 0, 0, delta_v]
      pre_state  + post_state -> delta_v   = post_state[3:6] - pre_state[3:6]
                                 (position continuity enforced)
    
    The maneuver Jacobian is identity for a fixed burn — the delta_v
    does not depend on the incoming state. For state-dependent maneuvers,
    a future subclass will provide the appropriate Jacobian.
    """
    
    def __init__(self, time: float,
                 pre_state: Optional[np.ndarray] = None,
                 post_state: Optional[np.ndarray] = None,
                 delta_v: Optional[np.ndarray] = None):
        super().__init__(time)
        
        n_provided = sum(x is not None for x in [pre_state, post_state, delta_v])
        if n_provided != 2:
            raise ValueError(
                f"Exactly two of pre_state, post_state, delta_v must be "
                f"provided. Got {n_provided}."
            )
        
        if delta_v is not None:
            delta_v = np.asarray(delta_v, dtype=float)
            if delta_v.shape != (3,):
                raise ValueError(
                    f"delta_v must be a 3-element vector, "
                    f"got shape {delta_v.shape}"
                )
            if not np.all(np.isfinite(delta_v)):
                raise ValueError("delta_v contains NaN or Inf values")
        
        if pre_state is not None and delta_v is not None:
            pre_state = self._validate_state(pre_state, 'pre_state')
            dv_full = np.concatenate([np.zeros(3), delta_v])
            post_state = pre_state + dv_full
        
        elif post_state is not None and delta_v is not None:
            post_state = self._validate_state(post_state, 'post_state')
            dv_full = np.concatenate([np.zeros(3), delta_v])
            pre_state = post_state - dv_full
        
        else:
            assert pre_state is not None and post_state is not None
            pre_state = self._validate_state(pre_state, 'pre_state')
            post_state = self._validate_state(post_state, 'post_state')
            pos_discont = np.linalg.norm(post_state[:3] - pre_state[:3])
            if not np.allclose(pre_state[:3], post_state[:3],
                               rtol=config.EQUALITY_RTOL,
                               atol=config.EQUALITY_ATOL):
                raise ValueError(
                    f"Position must be continuous for an impulsive maneuver. "
                    f"Position discontinuity: {pos_discont:.6e} km. "
                    f"Use FreeJunctionNode for nodes with position discontinuity."
                )
        
        self._pre_state = pre_state.copy()
        self._pre_state.flags.writeable = False
        self._post_state = post_state.copy()
        self._post_state.flags.writeable = False
    
    @property
    def pre_state(self) -> np.ndarray:
        """State before the maneuver [km, km/s]."""
        return self._pre_state
    
    @property
    def post_state(self) -> np.ndarray:
        """State after the maneuver [km, km/s]."""
        return self._post_state
    
    @property
    def delta_v(self) -> np.ndarray:
        """Velocity change at this node [km/s]."""
        return self._post_state[3:6] - self._pre_state[3:6]
    
    def maneuver_jacobian(self) -> np.ndarray:
        """
        Identity matrix for a fixed impulsive burn.
        
        A fixed delta_v does not depend on the incoming state, so
        d(post_state)/d(pre_state) = I.
        """
        return np.eye(6)
    
    def __repr__(self) -> str:
        dv_mag = np.linalg.norm(self.delta_v)
        return (f"ImpulsiveJunctionNode(t={self.time:.6g}, "
                f"|dv|={dv_mag:.6g} km/s)")


class FreeJunctionNode(JunctionNode):
    """
    Junction node with unconstrained state discontinuity.
    
    Both pre_state and post_state are specified independently, allowing
    full state discontinuity including position. Used to represent
    initial guesses for multiple shooting patch points, where the defect
    at each junction is driven to zero by the corrector.
    
    Parameters
    ----------
    time : float
        Junction time [s or nondimensional]
    pre_state : array-like
        State on the incoming side [x, y, z, vx, vy, vz] [km, km/s]
    post_state : array-like
        State on the outgoing side [x, y, z, vx, vy, vz] [km, km/s]
    
    Notes
    -----
    This is the only node type that permits position discontinuity.
    A converged multiple shooting solution will have state_defect
    near zero at every FreeJunctionNode, at which point the node
    is equivalent to a NullJunctionNode or ImpulsiveJunctionNode
    depending on whether a maneuver was intended.
    """
    
    def __init__(self, time: float,
                 pre_state: np.ndarray,
                 post_state: np.ndarray):
        super().__init__(time)
        self._pre_state = self._validate_state(pre_state, 'pre_state')
        self._pre_state = self._pre_state.copy()
        self._pre_state.flags.writeable = False
        self._post_state = self._validate_state(post_state, 'post_state')
        self._post_state = self._post_state.copy()
        self._post_state.flags.writeable = False
    
    @property
    def pre_state(self) -> np.ndarray:
        """State on the incoming side [km, km/s]."""
        return self._pre_state
    
    @property
    def post_state(self) -> np.ndarray:
        """State on the outgoing side [km, km/s]."""
        return self._post_state
    
    def maneuver_jacobian(self) -> np.ndarray:
        """
        Identity matrix.
        
        A FreeJunctionNode carries no defined maneuver, so the nominal
        sensitivity of post_state to pre_state is identity. The shooting
        corrector treats the defect as a constraint to eliminate rather
        than a physical state change to differentiate through.
        """
        return np.eye(6)
    
    def __repr__(self) -> str:
        defect_mag = np.linalg.norm(self.state_defect)
        return (f"FreeJunctionNode(t={self.time:.6g}, "
                f"|defect|={defect_mag:.6g})")