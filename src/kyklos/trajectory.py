'''Development code for an orbital trajectory handling package
Trajectory class definition
created with the assistance of Claude Sonnet by Anthropic'''

from __future__ import annotations

import numpy as np
import pandas as pd
import warnings
import heyoka as hy
import plotly.graph_objects as go
import bisect
from typing import Union, Optional, cast, Any, TYPE_CHECKING
from numpy.typing import ArrayLike
from abc import ABC, abstractmethod
from .orbital_elements import OrbitalElements, OEType
if TYPE_CHECKING:
    from .system import System
from .config import config
from .utils import validation_error

# ========== CONSTANTS ==========
_NODE_COLORS = {
    'StartBoundaryNode':     '#2ecc71',  # green
    'EndBoundaryNode':       '#e74c3c',  # red
    'ImpulsiveBoundaryNode': '#f39c12',  # orange
    'NullJunctionNode':      '#bdc3c7',  # light gray
    'ImpulsiveJunctionNode': '#e67e22',  # dark orange
    'FreeJunctionNode':      '#9b59b6',  # purple
}

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

    # ========== DISPATCH INFRASTRUCTURE ==========
    def _find_segment(self, t: float | np.ndarray) -> int | np.ndarray:
        """
        Return index of segment(s) containing time(s) t.

        At exact junction times, the later segment is preferred.
        Accepts scalar or array input, returning the corresponding type.

        Parameters
        ----------
        t : float or np.ndarray
            Query time(s). Assumed within [t0, tf].

        Returns
        -------
        int
            Segment index, if t is scalar.
        np.ndarray
            Array of segment indices, if t is array.
        """
        if isinstance(t, (int, float)):
            return bisect.bisect_right(self._junction_times, t)
        return np.searchsorted(self._junction_times, t, side='right')

    def _validate_time(self, t: float):
        """Raise ValueError if t is outside [t0, tf]."""
        t_min = min(self.t0, self.tf)
        t_max = max(self.t0, self.tf)
        if not (t_min <= t <= t_max):
            raise ValueError(
                f"Time {t} outside trajectory bounds [{self.t0}, {self.tf}]"
            )

    def contains_time(self, t: float) -> bool:
        """Return True if t is within trajectory bounds."""
        return self.t0 <= t <= self.tf

    def get_times(self, n_points: int = 100) -> np.ndarray:
        """Generate uniform time array spanning trajectory."""
        return np.linspace(self.t0, self.tf, n_points)

    def _resolve_element_type(self, element_type: OEType | str | None) -> OEType:
        """
        Resolve element type, defaulting to the system-appropriate type.

        Parameters
        ----------
        element_type : OEType, str, or None
            Desired element type, or None to use system default.

        Returns
        -------
        OEType
            Resolved element type enum value.
        """
        if element_type is None:
            from .system import SysType
            if self.system.base_type == SysType.CR3BP:
                return OEType.CR3BP
            return OEType.CARTESIAN
        return OrbitalElements._parse_element_type(element_type)
    
    # ========== RAW ARRAY SAMPLING ==========
    def state_at_raw(self, t: float) -> np.ndarray:
        """
        Get raw state array at time t.

        Parameters
        ----------
        t : float
            Time to query. Must be within [t0, tf].

        Returns
        -------
        np.ndarray
            State vector of shape (6,) [km, km/s].
        """
        self._validate_time(t)
        seg_idx = self._find_segment(t)
        return self._outputs[seg_idx](float(t))[:6].copy()

    def evaluate_raw(self, times: float | np.ndarray | list) -> np.ndarray:
        """
        Evaluate state at one or more times, returning raw arrays.

        Parameters
        ----------
        times : float or array-like
            Single time or array of times. All must be within [t0, tf].

        Returns
        -------
        np.ndarray
            Shape (6,) for scalar input, (n, 6) for array input [km, km/s].
        """
        if isinstance(times, (int, float)):
            return self.state_at_raw(times)

        times = np.asarray(times, dtype=float)

        # Fast path for single segment
        if self.n_segments == 1:
            return self._outputs[0](times)[:, :6]

        # Multi-segment: group times by segment for batch evaluation
        n_times = len(times)
        result = np.empty((n_times, 6))
        seg_indices = self._find_segment(times)

        for seg_idx in range(self.n_segments):
            mask = seg_indices == seg_idx
            if not np.any(mask):
                continue
            result[mask] = self._outputs[seg_idx](times[mask])[:, :6]

        return result

    def sample_raw(self, n_points: int = 100) -> np.ndarray:
        """
        Uniformly sample trajectory in time, returning raw arrays.

        Parameters
        ----------
        n_points : int, optional
            Number of sample points. Must be >= 2. Default: 100.

        Returns
        -------
        np.ndarray
            Shape (n_points, 6) [km, km/s].
        """
        if n_points < 2:
            raise ValueError(
                "n_points must be at least 2. Use state_at_raw() for a "
                "single time."
            )
        times = np.linspace(self.t0, self.tf, n_points)
        return self.evaluate_raw(times)

        # ========== ORBITAL ELEMENTS SAMPLING ==========
    def state_at(self, t: float,
                element_type: OEType | str | None = None) -> OrbitalElements:
        """
        Get orbital state at time t as OrbitalElements.

        Parameters
        ----------
        t : float
            Time to query. Must be within [t0, tf].
        element_type : OEType, str, or None, optional
            Desired element type. Defaults to system-appropriate type.

        Returns
        -------
        OrbitalElements
        """
        raw = self.state_at_raw(t)
        element_type = self._resolve_element_type(element_type)
        return OrbitalElements(raw, element_type, validate=False,
                            system=self.system)

    def evaluate(self,
                times: float | np.ndarray | list,
                element_type: OEType | str | None = None
                ) -> OrbitalElements | list:
        """
        Evaluate state at one or more times as OrbitalElements.

        Parameters
        ----------
        times : float or array-like
            Single time or array of times. All must be within [t0, tf].
        element_type : OEType, str, or None, optional
            Desired element type. Defaults to system-appropriate type.

        Returns
        -------
        OrbitalElements or list of OrbitalElements
        """
        if isinstance(times, (int, float)):
            return self.state_at(times, element_type)

        element_type = self._resolve_element_type(element_type)
        raw = self.evaluate_raw(times)
        return [OrbitalElements(row, element_type, validate=False,
                                system=self.system)
                for row in raw]

    def sample(self, element_type: OEType | str | None = None,
            n_points: int = 100) -> list:
        """
        Uniformly sample trajectory in time as OrbitalElements.

        Parameters
        ----------
        element_type : OEType, str, or None, optional
            Desired element type. Defaults to system-appropriate type.
        n_points : int, optional
            Number of sample points. Must be >= 2. Default: 100.

        Returns
        -------
        list of OrbitalElements
        """
        if n_points < 2:
            raise ValueError(
                "n_points must be at least 2. Use state_at() for a "
                "single time."
            )
        times = np.linspace(self.t0, self.tf, n_points)
        return cast(list, self.evaluate(times, element_type))
    
    # ========== VARIATIONAL ODE (STM) SAMPLING ==========
    def _require_stm(self):
        """Raise ValueError if trajectory was not propagated with STM."""
        if self._stm_order is None:
            raise ValueError(
                "Trajectory not propagated with STM. "
                "Use with_stm=True in System.propagate()."
            )
    
    # ========== FULL STATE METHODS ==========
    def state_full(self, t: float) -> np.ndarray:
        """
        Get raw integrator state at time t.

        Returns the full Heyoka state vector, including the segment-local
        STM if the trajectory was propagated with with_stm=True. The STM
        portion is Phi_k(t, t_k), referenced to the start of the segment
        containing t. For the composite Phi(t, t0), use get_stm().

        Parameters
        ----------
        t : float
            Time to query. Must be within [t0, tf].

        Returns
        -------
        np.ndarray
            Shape (6,) without STM, (42,) with first-order STM.
        """
        if not self.has_stm:
            if config.STRICT_VALIDATION:
                warnings.warn(
                    "Trajectory does not have STM enabled. "
                    "Use state_at_raw() for non-STM trajectories.",
                    UserWarning,
                    stacklevel=2
                )
        self._validate_time(t)
        seg_idx = self._find_segment(t)
        return self._outputs[seg_idx](float(t)).copy()

    def evaluate_full(self, times: float | np.ndarray | list) -> np.ndarray:
        """
        Evaluate raw integrator state at one or more times.

        Returns the full Heyoka state vector at each time, including the
        segment-local STM if present. For a multi-segment trajectory, the
        STM portion at time t is Phi_k(t, t_k), referenced to the start of
        the segment containing t. For the composite Phi(t, t0), use
        evaluate_stm().

        Parameters
        ----------
        times : float or array-like
            Single time or array of times. All must be within [t0, tf].

        Returns
        -------
        np.ndarray
            Shape (6,) or (42,) for scalar input.
            Shape (n, 6) or (n, 42) for array input.
        """
        if not self.has_stm:
            if config.STRICT_VALIDATION:
                warnings.warn(
                    "Trajectory does not have STM enabled. "
                    "Use state_at_raw() for non-STM trajectories.",
                    UserWarning,
                    stacklevel=2
                )
        # handle scalar input by delegation
        if isinstance(times, (int, float)):
            return self.state_full(times)

        times = np.asarray(times, dtype=float)

        # Fast path for single segment
        if self.n_segments == 1:
            return self._outputs[0](times).copy()

        # Multi-segment: dispatch by segment
        # Determine full state width from stm_order
        full_width = 42 if self._stm_order == 1 else 6
        n_times = len(times)
        result = np.empty((n_times, full_width))
        seg_indices = self._find_segment(times)

        for seg_idx in range(self.n_segments):
            mask = seg_indices == seg_idx
            if not np.any(mask):
                continue
            result[mask] = self._outputs[seg_idx](times[mask])

        return result

    def sample_full(self, n_points: int = 100) -> np.ndarray:
        """
        Uniformly sample raw integrator state in time.

        Returns the full Heyoka state vector at each sample time, including
        the segment-local STM if present. See evaluate_full() for details
        on STM referencing.

        Parameters
        ----------
        n_points : int, optional
            Number of sample points. Must be >= 2. Default: 100.

        Returns
        -------
        np.ndarray
            Shape (n_points, 6) or (n_points, 42).
        """
        if not self.has_stm:
            if config.STRICT_VALIDATION:
                warnings.warn(
                    "Trajectory does not have STM enabled. "
                    "Use state_at_raw() for non-STM trajectories.",
                    UserWarning,
                    stacklevel=2
                )
        if n_points < 2:
            raise ValueError(
                "n_points must be at least 2. Use state_full() for a "
                "single time."
            )
        times = np.linspace(self.t0, self.tf, n_points)
        return self.evaluate_full(times)
    
    # ========== COMPOSITE STM METHODS ==========
    def get_stm(self, t: float) -> np.ndarray:
        """
        Get composite STM at time t, referenced to trajectory t0.

        Returns Phi(t, t0), the full sensitivity of the state at t to the
        initial state. For multi-segment trajectories this is assembled by
        chaining segment STMs through junction maneuver Jacobians:

            Phi(t, t0) = Phi_k(t, t_k) @ M_{k-1} @ ... @ M_0 @ Phi_0(t_1, t_0)

        where M_i = junction_nodes[i].maneuver_jacobian().

        Parameters
        ----------
        t : float
            Time at which to evaluate STM. Must be within [t0, tf].

        Returns
        -------
        np.ndarray
            6x6 composite STM Phi(t, t0).

        Raises
        ------
        ValueError
            If trajectory was not propagated with STM.
        """
        self._require_stm()
        self._validate_time(t)
        seg_idx = self._find_segment(t)

        # Segment-local STM: Phi_k(t, t_k)
        full_state = self._outputs[seg_idx](float(t))
        stm_local = full_state[6:42].reshape(6, 6).copy()

        if seg_idx == 0:
            return stm_local

        # Compose backwards through junctions to t0
        composite = stm_local
        for k in range(seg_idx - 1, -1, -1):
            t_junc = self._junction_nodes[k].time
            full_state_k = self._outputs[k](float(t_junc))
            stm_k_terminal = full_state_k[6:42].reshape(6, 6).copy()
            M_k = self._junction_nodes[k].maneuver_jacobian()
            composite = composite @ M_k @ stm_k_terminal

        return composite

    def evaluate_stm(self, times: float | np.ndarray | list) -> np.ndarray:
        """
        Evaluate composite STM at one or more times.

        Returns Phi(t, t0) at each time. See get_stm() for details on
        the composition across junctions.

        Parameters
        ----------
        times : float or array-like
            Single time or array of times. All must be within [t0, tf].

        Returns
        -------
        np.ndarray
            Shape (6, 6) for scalar input.
            Shape (n, 6, 6) for array input.

        Raises
        ------
        ValueError
            If trajectory was not propagated with STM.
        """
        self._require_stm()

        # Handle scalar input by delegation
        if isinstance(times, (int, float)):
            return self.get_stm(times)

        times = np.asarray(times, dtype=float)
        n_times = len(times)

        if self.n_segments == 1:
            # Fast path: vectorized Heyoka call, no composition needed
            full_states = self._outputs[0](times)
            return full_states[:, 6:42].reshape(n_times, 6, 6)

        # Multi-segment: composition requires per-time computation
        return np.array([self.get_stm(t) for t in times])

    def sample_stm(self, n_points: int = 100) -> np.ndarray:
        """
        Uniformly sample composite STM in time.

        Returns Phi(t, t0) at uniformly spaced times. See get_stm() for
        details on the composition across junctions.

        Parameters
        ----------
        n_points : int, optional
            Number of sample points. Must be >= 2. Default: 100.

        Returns
        -------
        np.ndarray
            Shape (n_points, 6, 6).

        Raises
        ------
        ValueError
            If trajectory was not propagated with STM.
        """
        self._require_stm()
        if n_points < 2:
            raise ValueError(
                "n_points must be at least 2. Use get_stm() for a "
                "single time."
            )
        times = np.linspace(self.t0, self.tf, n_points)
        return self.evaluate_stm(times)
    
    # ========== SEGMENT-LOCAL STM METHODS ==========
    def get_stm_seg(self, t: float) -> np.ndarray:
        """
        Get segment-local STM at time t, referenced to segment start.

        Returns Phi_k(t, t_k) where t_k is the start time of the segment
        containing t. For the composite Phi(t, t0), use get_stm().

        Parameters
        ----------
        t : float
            Time at which to evaluate STM. Must be within [t0, tf].

        Returns
        -------
        np.ndarray
            6x6 segment-local STM Phi_k(t, t_k).

        Raises
        ------
        ValueError
            If trajectory was not propagated with STM.
        """
        self._require_stm()
        self._validate_time(t)
        seg_idx = self._find_segment(t)
        full_state = self._outputs[seg_idx](float(t))
        return full_state[6:42].reshape(6, 6).copy()

    def evaluate_stm_seg(self, times: float | np.ndarray | list) -> np.ndarray:
        """
        Evaluate segment-local STM at one or more times.

        Returns Phi_k(t, t_k) at each time. For multi-segment trajectories,
        times in different segments will have STMs referenced to different
        t_k values. For the composite Phi(t, t0), use evaluate_stm().

        Parameters
        ----------
        times : float or array-like
            Single time or array of times. All must be within [t0, tf].

        Returns
        -------
        np.ndarray
            Shape (6, 6) for scalar input.
            Shape (n, 6, 6) for array input.

        Raises
        ------
        ValueError
            If trajectory was not propagated with STM.
        """
        self._require_stm()

        # Handle scalar input by delegation
        if isinstance(times, (int, float)):
            return self.get_stm_seg(times)

        times = np.asarray(times, dtype=float)
        n_times = len(times)

        # Fast path for single segment
        if self.n_segments == 1:
            full_states = self._outputs[0](times)
            return full_states[:, 6:42].reshape(n_times, 6, 6)

        # Multi-segment: batch by segment, no composition needed
        result = np.empty((n_times, 6, 6))
        seg_indices = self._find_segment(times)

        for seg_idx in range(self.n_segments):
            mask = seg_indices == seg_idx
            if not np.any(mask):
                continue
            n_seg = int(np.sum(mask))
            full_states = self._outputs[seg_idx](times[mask])
            result[mask] = full_states[:, 6:42].reshape(n_seg, 6, 6)

        return result

    def sample_stm_seg(self, n_points: int = 100) -> np.ndarray:
        """
        Uniformly sample segment-local STM in time.

        Returns Phi_k(t, t_k) at uniformly spaced times. See
        evaluate_stm_seg() for details on referencing across segments.

        Parameters
        ----------
        n_points : int, optional
            Number of sample points. Must be >= 2. Default: 100.

        Returns
        -------
        np.ndarray
            Shape (n_points, 6, 6).

        Raises
        ------
        ValueError
            If trajectory was not propagated with STM.
        """
        self._require_stm()
        if n_points < 2:
            raise ValueError(
                "n_points must be at least 2. Use get_stm_seg() for a "
                "single time."
            )
        times = np.linspace(self.t0, self.tf, n_points)
        return self.evaluate_stm_seg(times)

    def segment_terminal_stm(self, seg_idx: int) -> np.ndarray:
        """
        Get STM of segment seg_idx at its terminal time.

        Returns Phi_k(t_{k+1}, t_k) -- the STM of segment k evaluated at
        the junction or end time bounding that segment. Used by the shooting
        corrector to assemble the defect Jacobian without querying by time.

        Parameters
        ----------
        seg_idx : int
            Segment index in [0, n_segments - 1].

        Returns
        -------
        np.ndarray
            6x6 terminal STM of the specified segment.

        Raises
        ------
        ValueError
            If trajectory was not propagated with STM, or seg_idx is
            out of range.
        """
        self._require_stm()
        if not 0 <= seg_idx < self.n_segments:
            raise ValueError(
                f"seg_idx {seg_idx} out of range "
                f"[0, {self.n_segments - 1}]."
            )
        if seg_idx < self.n_segments - 1:
            terminal_time = self._junction_nodes[seg_idx].time
        else:
            terminal_time = self._end_node.time

        full_state = self._outputs[seg_idx](float(terminal_time))
        return full_state[6:42].reshape(6, 6).copy()
    
    # ========== UTILITY METHODS ==========
    def to_dataframe(self,
                 times: np.ndarray | None = None,
                 n_points: int = 1000) -> pd.DataFrame:
        """
        Export trajectory to a pandas DataFrame.

        Parameters
        ----------
        times : np.ndarray, optional
            Specific times to evaluate. If None, uses uniform sampling
            over [t0, tf]. Default: None.
        n_points : int, optional
            Number of uniform samples if times not provided. Default: 1000.

        Returns
        -------
        pd.DataFrame
            Columns: time, segment, x, y, z, vx, vy, vz.
            segment is the zero-indexed segment containing each time point,
            consistent with the post-junction convention used throughout.
        """
        if times is None:
            times = np.linspace(self.t0, self.tf, n_points)
        else:
            times = np.asarray(times, dtype=float)

        states = self.evaluate_raw(times)
        seg_indices = self._find_segment(times)

        return pd.DataFrame({
            'time':    times,
            'segment': seg_indices,
            'x':       states[:, 0],
            'y':       states[:, 1],
            'z':       states[:, 2],
            'vx':      states[:, 3],
            'vy':      states[:, 4],
            'vz':      states[:, 5],
        })
    
    # ========== TRAJECTORY MANIPULATION ==========
    
    # ========== HELPER METHODS ==========
    @staticmethod
    def _boundary_from_junction(node: JunctionNode, role: str) -> BoundaryNode:
        """
        Create a BoundaryNode from a JunctionNode.

        Used when a slice or segment extraction begins or ends at a junction.
        ImpulsiveJunctionNode produces an ImpulsiveBoundaryNode preserving
        full maneuver context. All other node types produce a plain boundary
        using the appropriate side: post_state for 'start', pre_state for 'end'.

        Parameters
        ----------
        node : JunctionNode
            Source junction node.
        role : str
            'start' for a StartBoundaryNode, 'end' for an EndBoundaryNode.
        """
        if isinstance(node, ImpulsiveJunctionNode):
            return ImpulsiveBoundaryNode(node.time,
                                        pre_state=node.pre_state,
                                        post_state=node.post_state)
        if role == 'start':
            return StartBoundaryNode(node.time, node.post_state)
        return EndBoundaryNode(node.time, node.pre_state)
    
    @staticmethod
    def _recreate_junction(node: JunctionNode) -> JunctionNode:
        """
        Create a new JunctionNode with identical data to the given node.

        Used to produce fresh node objects during slice and segment_slice
        rather than sharing references with the source trajectory.
        """
        if isinstance(node, NullJunctionNode):
            return NullJunctionNode(node.time, node.pre_state, node.post_state)
        if isinstance(node, ImpulsiveJunctionNode):
            return ImpulsiveJunctionNode(node.time,
                                        pre_state=node.pre_state,
                                        post_state=node.post_state)
        if isinstance(node, FreeJunctionNode):
            return FreeJunctionNode(node.time, node.pre_state, node.post_state)
        raise TypeError(f"Unrecognised JunctionNode type: {type(node).__name__}")
    
    def with_junction_nodes(self, junction_nodes: list) -> "Trajectory":
        """
        Return a new Trajectory with the same segments but replaced junction
        nodes.

        Reuses this trajectory's continuous outputs and boundary nodes -- no
        re-propagation -- so it is the cheap way to relabel junctions, for
        example a shooting corrector converting converged FreeJunctionNodes
        into NullJunctionNodes. The replacement list must have length
        n_segments - 1, and each node's time must match its segment boundary;
        the constructor validates both.

        Parameters
        ----------
        junction_nodes : list of JunctionNode
            Replacement interior nodes, one per interior boundary.

        Returns
        -------
        Trajectory
            A new trajectory sharing this one's outputs and boundary nodes.
        """
        return Trajectory(
            self._system,
            self._outputs,
            junction_nodes=junction_nodes,
            stm_order=self._stm_order,
            start_node=self._start_node,
            end_node=self._end_node,
        )
    
    def _back_and_forward_propagate(
        self, new_t0: float, back_state: np.ndarray,
        with_stm: bool, stm_order: int, satellite) -> object:
        """
        Produce a forward Heyoka output spanning [new_t0, self.t0].

        Back-propagates from (self.t0, back_state) to find the state at
        new_t0, then re-propagates forward. The backward probe propagation
        uses no STM regardless of trajectory settings, since only the
        terminal state is needed from it. The forward re-propagation
        respects with_stm and stm_order.

        Parameters
        ----------
        new_t0 : float
            New start time. Must be less than self.t0.
        back_state : np.ndarray
            State to propagate backward from self.t0 [km, km/s].
        with_stm : bool
            Whether to propagate with STM on the forward pass.
        stm_order : int
            STM order for the forward pass.
        satellite : Satellite or None
            Satellite model for perturbations.

        Returns
        -------
        Heyoka continuous output object
            Forward-propagated output for [new_t0, self.t0].

        Notes
        -----
        This method performs two full propagations and is therefore more
        expensive than a single forward propagation. The backward probe
        result is discarded after state extraction.
        """
        # Step 1: Back-propagate to find state at new_t0.
        # Bypasses system.propagate() to avoid _validate_times() rejecting
        # backward time input. _propagate_single_output() calls Heyoka directly.
        c_out_back = self._system._propagate_single_output(
            back_state, self.t0, new_t0,
            with_stm=False, stm_order=1, satellite=satellite
        )
        state_at_new_t0 = c_out_back(float(new_t0))[:6].copy()

        # Step 2: Re-propagate forward [new_t0, self.t0] — valid forward propagation.
        fwd_seg = self._system.propagate(
            state_at_new_t0, [new_t0, self.t0],
            with_stm=with_stm, stm_order=stm_order, satellite=satellite
        )
        return fwd_seg._outputs[0]
    
    # ========== SLICING METHODS ==========
    def segment_slice(self, start_idx: int, end_idx: int) -> Trajectory:
        """
        Extract a contiguous range of segments as a new Trajectory.

        Segments are selected by index. Both start_idx and end_idx are
        inclusive. The extracted trajectory reuses the underlying Heyoka
        output objects from the original for efficiency; all node objects
        are recreated.

        Parameters
        ----------
        start_idx : int
            Index of the first segment to include. Must be >= 0.
        end_idx : int
            Index of the last segment to include (inclusive).
            Must be >= start_idx and <= n_segments - 1.

        Returns
        -------
        Trajectory
            New Trajectory containing segments [start_idx, ..., end_idx].

        Raises
        ------
        ValueError
            If indices are out of range or start_idx > end_idx.
        """
        if not isinstance(start_idx, int) or not isinstance(end_idx, int):
            raise TypeError("start_idx and end_idx must be integers.")
        if not 0 <= start_idx <= end_idx <= self.n_segments - 1:
            raise ValueError(
                f"Segment indices [{start_idx}, {end_idx}] out of valid range "
                f"[0, {self.n_segments - 1}]. Both endpoints are inclusive."
            )

        # Reuse Heyoka output objects for selected segments
        outputs = self._outputs[start_idx:end_idx + 1]

        # Recreate internal junction nodes as new objects.
        # junction_nodes[k] sits between segment k and segment k+1, so
        # internal junctions for segments [start_idx..end_idx] are
        # junction_nodes[start_idx..end_idx-1].
        junction_nodes = [
            self._recreate_junction(self._junction_nodes[k])
            for k in range(start_idx, end_idx)
        ]

        # Determine start boundary node
        if start_idx == 0:
            # Slice starts at the original trajectory boundary
            start_node = self._start_node
        else:
            # Slice starts at what was an internal junction
            start_node = self._boundary_from_junction(
                self._junction_nodes[start_idx - 1], 'start'
            )

        # Determine end boundary node
        if end_idx == self.n_segments - 1:
            # Slice ends at the original trajectory boundary
            end_node = self._end_node
        else:
            # Slice ends at what was an internal junction
            end_node = self._boundary_from_junction(
                self._junction_nodes[end_idx], 'end'
            )

        return Trajectory(
            self._system,
            outputs,
            junction_nodes=junction_nodes,
            stm_order=self._stm_order,
            start_node=start_node,
            end_node=end_node
        )
    
    def slice(self, t_start: float, t_end: float,
          **propagation_kwargs) -> Trajectory:
        """
        Extract a time interval as a new Trajectory by re-propagation.

        The interval [t_start, t_end] may span multiple segments. Internal
        junction nodes within the interval are preserved: their type and
        maneuver data are carried through to the new trajectory, but new
        node objects are constructed from the re-propagated states. This
        ensures physical consistency even when t_start differs from the
        original segment boundary.

        If t_start or t_end coincides (within tolerance) with a junction
        time, the junction's maneuver context is preserved in the resulting
        boundary node. For ImpulsiveJunctionNode, this produces an
        ImpulsiveBoundaryNode.

        Parameters
        ----------
        t_start : float
            Start of the extracted interval. Must be >= t0.
        t_end : float
            End of the extracted interval. Must be <= tf and > t_start.

        Returns
        -------
        Trajectory
            New re-propagated Trajectory spanning [t_start, t_end].

        Raises
        ------
        ValueError
            If bounds are invalid or outside the trajectory time range.
        """
        t_start, t_end = float(t_start), float(t_end)

        if t_start >= t_end:
            raise ValueError(
                f"t_start ({t_start:.6g}) must be less than t_end ({t_end:.6g})."
            )
        if t_start < self.t0 or t_end > self.tf:
            raise ValueError(
                f"Slice bounds [{t_start:.6g}, {t_end:.6g}] fall outside "
                f"trajectory bounds [{self.t0:.6g}, {self.tf:.6g}]."
            )

        with_stm  = self._stm_order is not None
        stm_order = self._stm_order if self._stm_order is not None else 1
        satellite = propagation_kwargs.get('satellite', None)

        # Detect if t_start or t_end coincide with any junction time
        def _find_coincident_junction(t: float) -> int | None:
            return next(
                (k for k, t_j in enumerate(self._junction_times)
                if np.isclose(t, t_j,
                            rtol=config.EQUALITY_RTOL,
                            atol=config.EQUALITY_ATOL)),
                None
            )

        start_junc_idx = _find_coincident_junction(t_start)
        end_junc_idx   = _find_coincident_junction(t_end)

        # Find internal junction indices strictly within (t_start, t_end).
        # If t_start/t_end land on a junction, exclude that junction from
        # internal nodes — it becomes a boundary node instead.
        lo = (start_junc_idx + 1 if start_junc_idx is not None
            else bisect.bisect_right(self._junction_times, t_start))
        hi = (end_junc_idx if end_junc_idx is not None
            else bisect.bisect_left(self._junction_times, t_end))
        internal_indices = list(range(lo, hi))

        # Build propagation intervals separated by internal junctions
        t_splits = (
            [t_start]
            + [self._junction_times[k] for k in internal_indices]
            + [t_end]
        )

        # Propagate each interval, applying junction maneuvers between them
        outputs = []
        current_state = self.state_at_raw(t_start)

        for i, (t_s, t_e) in enumerate(zip(t_splits[:-1], t_splits[1:])):
            seg = self._system.propagate(
                current_state, [t_s, t_e],
                with_stm=with_stm,
                stm_order=stm_order,
                satellite=satellite
            )
            outputs.append(seg._outputs[0])

            if i < len(internal_indices):
                # Apply the maneuver from the original junction
                orig_node = self._junction_nodes[internal_indices[i]]
                pre_state = seg.state_at_raw(t_e)

                if isinstance(orig_node, ImpulsiveJunctionNode):
                    current_state = pre_state.copy()
                    current_state[3:6] += orig_node.delta_v
                elif isinstance(orig_node, FreeJunctionNode):
                    current_state = pre_state + orig_node.state_defect
                else:
                    # NullJunctionNode: no state change
                    current_state = pre_state.copy()

        # Build new internal junction nodes from the re-propagated states
        new_junction_nodes = []
        for i, k in enumerate(internal_indices):
            orig_node = self._junction_nodes[k]
            t_j = self._junction_times[k]
            pre_state = outputs[i](float(t_j))[:6].copy()

            if isinstance(orig_node, ImpulsiveJunctionNode):
                post_state = pre_state.copy()
                post_state[3:6] += orig_node.delta_v
                new_junction_nodes.append(
                    ImpulsiveJunctionNode(t_j,
                                        pre_state=pre_state,
                                        post_state=post_state)
                )
            elif isinstance(orig_node, FreeJunctionNode):
                post_state = pre_state + orig_node.state_defect
                new_junction_nodes.append(
                    FreeJunctionNode(t_j, pre_state, post_state)
                )
            else:
                new_junction_nodes.append(
                    NullJunctionNode(t_j, pre_state, pre_state.copy())
                )

        # Determine boundary nodes
        start_node = (self._boundary_from_junction(
                        self._junction_nodes[start_junc_idx], 'start')
                    if start_junc_idx is not None
                    else None)

        end_node = (self._boundary_from_junction(
                        self._junction_nodes[end_junc_idx], 'end')
                    if end_junc_idx is not None
                    else None)

        return Trajectory(
            self._system,
            outputs,
            junction_nodes=new_junction_nodes,
            stm_order=self._stm_order,
            start_node=start_node,
            end_node=end_node
        )

    # ========== EXTENSION METHODS ==========
    def extend(self, new_tf: float, junction=None,
           **propagation_kwargs) -> Trajectory:
        """
        Extend trajectory forward in time by appending a new segment at the end.

        The junction parameter controls how the new segment connects to the
        current trajectory at tf. Behavior depends on junction type:

        - None + EndBoundaryNode  : continuous propagation from end state.
                                    NullJunctionNode inferred automatically.
        - None + ImpulsiveBoundaryNode : stored maneuver data used directly.
                                        ImpulsiveJunctionNode created from
                                        the node's pre/post states.
                                        Propagation starts from post_state.
        - 3-element np.ndarray (delta_v) : impulsive burn applied at tf.
                                            ImpulsiveJunctionNode constructed.
                                            Propagation from post-burn state.
        - JunctionNode instance   : explicit junction. Propagation from
                                    junction.post_state.

        Parameters
        ----------
        new_tf : float
            New end time. Must be greater than current tf.
        junction : None, np.ndarray, or JunctionNode, optional
            Junction specification at the connection point. Default: None.

        Returns
        -------
        Trajectory
            New Trajectory with an additional segment appended.

        Raises
        ------
        ValueError
            If new_tf <= tf, or if a JunctionNode time does not match tf.
        TypeError
            If junction is not None, np.ndarray, or JunctionNode.
        """
        if new_tf <= self.tf:
            raise ValueError(
                f"new_tf ({new_tf:.6g}) must be greater than "
                f"current tf ({self.tf:.6g})."
            )
        new_tf = float(new_tf)  # Heyoka requires float input

        with_stm  = self._stm_order is not None
        stm_order = self._stm_order if self._stm_order is not None else 1
        satellite = propagation_kwargs.get('satellite', None)

        # Determine initial state and junction node for the new segment
        new_junction_node = None  # None triggers _infer_junction below

        if junction is None:
            if isinstance(self._end_node, ImpulsiveBoundaryNode):
                initial_state    = self._end_node.post_state
                new_junction_node = ImpulsiveJunctionNode(
                    self.tf,
                    pre_state=self._end_node.pre_state,
                    post_state=self._end_node.post_state
                )
            else:
                initial_state = self._end_node.pre_state

        elif isinstance(junction, np.ndarray):
            delta_v = np.asarray(junction, dtype=float)
            if delta_v.shape != (3,):
                raise ValueError(
                    f"delta_v must be a 3-element vector, "
                    f"got shape {delta_v.shape}."
                )
            pre_state  = self._end_node.pre_state
            post_state = pre_state.copy()
            post_state[3:6] += delta_v
            initial_state    = post_state
            new_junction_node = ImpulsiveJunctionNode(
                self.tf,
                pre_state=pre_state,
                post_state=post_state
            )

        elif isinstance(junction, JunctionNode):
            if not np.isclose(junction.time, self.tf,
                            rtol=config.EQUALITY_RTOL,
                            atol=config.EQUALITY_ATOL):
                raise ValueError(
                    f"JunctionNode time ({junction.time:.6g}) does not match "
                    f"trajectory end time ({self.tf:.6g})."
                )
            initial_state    = junction.post_state
            new_junction_node = junction

        else:
            raise TypeError(
                f"junction must be None, a 3-element delta_v array, or a "
                f"JunctionNode instance. Got {type(junction).__name__}."
            )

        # Propagate the new segment
        new_seg = self._system.propagate(
            initial_state, [self.tf, new_tf],
            with_stm=with_stm, stm_order=stm_order, satellite=satellite
        )
        new_output = new_seg._outputs[0]

        # Build junction between current end and new segment
        if new_junction_node is None:
            new_junction_node = Trajectory._infer_junction(
                self._outputs[-1], new_output
            )

        return Trajectory(
            self._system,
            self._outputs + [new_output],
            junction_nodes=self._junction_nodes + [new_junction_node],
            stm_order=self._stm_order,
            start_node=self._start_node,
            end_node=None   # auto-construct from new output
        )


    def extend_back(self, new_t0: float, junction=None,
                    **propagation_kwargs) -> Trajectory:
        """
        Extend trajectory backward in time by prepending a new segment at the start.

        Mirrors extend() but operates at the trajectory start. Because the
        new segment must be a standard forward Heyoka output, this method
        performs two propagations internally: a backward probe to find the
        state at new_t0, then a forward re-propagation. This makes
        extend_back() more expensive than extend().

        Note: existing segments and nodes retain their original time stamps.
        If self.t0 == 0.0, this forces new_t0 to be negative. A future fix
        will re-propagate existing segments with adjusted node times once
        System.propagate() accepts Node input, eliminating this constraint.

        The junction parameter controls how the new segment connects at t0:

        - None + StartBoundaryNode      : continuous back-propagation from
                                        current start state.
                                        NullJunctionNode inferred.
        - None + ImpulsiveBoundaryNode  : back-propagation from pre_state.
                                        ImpulsiveJunctionNode created from
                                        the node's pre/post states.
        - 3-element np.ndarray (delta_v): impulsive burn at t0.
                                        pre_state = post_state - [0,0,0,dv].
                                        Back-propagation from pre_state.
        - JunctionNode instance         : explicit junction. Back-propagation
                                        from junction.pre_state.

        Parameters
        ----------
        new_t0 : float
            New start time. Must be less than current t0.
        junction : None, np.ndarray, or JunctionNode, optional
            Junction specification at the connection point. Default: None.

        Returns
        -------
        Trajectory
            New Trajectory with an additional segment prepended.

        Raises
        ------
        ValueError
            If new_t0 >= t0, or if a JunctionNode time does not match t0.
        TypeError
            If junction is not None, np.ndarray, or JunctionNode.
        """
        if new_t0 >= self.t0:
            raise ValueError(
                f"new_t0 ({new_t0:.6g}) must be less than "
                f"current t0 ({self.t0:.6g})."
            )
        new_t0 = float(new_t0) # Heyoka requires float input

        with_stm  = self._stm_order is not None
        stm_order = self._stm_order if self._stm_order is not None else 1
        satellite = propagation_kwargs.get('satellite', None)

        new_junction_node = None

        if junction is None:
            if isinstance(self._start_node, ImpulsiveBoundaryNode):
                back_state       = self._start_node.pre_state
                new_junction_node = ImpulsiveJunctionNode(
                    self.t0,
                    pre_state=self._start_node.pre_state,
                    post_state=self._start_node.post_state
                )
            else:
                back_state = self._start_node.post_state

        elif isinstance(junction, np.ndarray):
            delta_v = np.asarray(junction, dtype=float)
            if delta_v.shape != (3,):
                raise ValueError(
                    f"delta_v must be a 3-element vector, "
                    f"got shape {delta_v.shape}."
                )
            post_state = self._start_node.post_state
            pre_state  = post_state.copy()
            pre_state[3:6] -= delta_v
            back_state       = pre_state
            new_junction_node = ImpulsiveJunctionNode(
                self.t0,
                pre_state=pre_state,
                post_state=post_state
            )

        elif isinstance(junction, JunctionNode):
            if not np.isclose(junction.time, self.t0,
                            rtol=config.EQUALITY_RTOL,
                            atol=config.EQUALITY_ATOL):
                raise ValueError(
                    f"JunctionNode time ({junction.time:.6g}) does not match "
                    f"trajectory start time ({self.t0:.6g})."
                )
            back_state       = junction.pre_state
            new_junction_node = junction

        else:
            raise TypeError(
                f"junction must be None, a 3-element delta_v array, or a "
                f"JunctionNode instance. Got {type(junction).__name__}."
            )

        # Produce the new forward segment via back-and-forward propagation
        new_output = self._back_and_forward_propagate(
            new_t0, back_state, with_stm, stm_order, satellite
        )

        # Build junction between new segment and current start
        if new_junction_node is None:
            new_junction_node = Trajectory._infer_junction(
                new_output, self._outputs[0]
            )

        return Trajectory(
            self._system,
            [new_output] + self._outputs,
            junction_nodes=[new_junction_node] + self._junction_nodes,
            stm_order=self._stm_order,
            start_node=None,    # auto-construct from new output
            end_node=self._end_node
        )

    # ========== SPECIAL METHODS ==========
    def __repr__(self) -> str:
        seg_str = f"{self.n_segments} segment{'s' if self.n_segments > 1 else ''}"
        stm_str = f", stm_order={self._stm_order}" if self.has_stm else ""
        return (
            f"Trajectory(system={self._system.primary_body.name}, "
            f"t0={self.t0:.6g}, tf={self.tf:.6g}, "
            f"{seg_str}{stm_str})"
        )

    def __str__(self) -> str:
        seg_str = f"{self.n_segments} segment{'s' if self.n_segments > 1 else ''}"
        return (
            f"Trajectory around {self._system.primary_body.name}: "
            f"t in [{self.t0:.6g}, {self.tf:.6g}], {seg_str}"
        )

    def __call__(self, t: float,
                element_type: OEType | str | None = None) -> OrbitalElements:
        """
        Evaluate trajectory at time t.
        Syntactic sugar for state_at(t). Allows traj(t) syntax.

        Parameters
        ----------
        t : float
            Time to query. Must be within [t0, tf].
        element_type : OEType, str, or None, optional
            Desired element type. Defaults to system-appropriate type.

        Returns
        -------
        OrbitalElements
        """
        return self.state_at(t, element_type)

    def __len__(self) -> int:
        """
        Number of segments in this trajectory.

        Allows len(traj) as a natural way to inspect multi-segment structure.
        Single-segment trajectories return 1.
        """
        return self.n_segments

    def __getitem__(self, seg_idx: int) -> Trajectory:
        """
        Access a single segment as a new Trajectory.

        Equivalent to segment_slice(k, k). Supports negative indexing:
        traj[-1] returns the last segment.

        Parameters
        ----------
        seg_idx : int
            Segment index. Negative values count from the end.

        Returns
        -------
        Trajectory
            Single-segment Trajectory for the requested segment.

        Raises
        ------
        TypeError
            If seg_idx is not an integer.
        IndexError
            If seg_idx is out of range.
        """
        if not isinstance(seg_idx, int):
            raise TypeError(
                f"Segment index must be an integer, "
                f"got {type(seg_idx).__name__}."
            )
        # Support negative indexing
        if seg_idx < 0:
            seg_idx = self.n_segments + seg_idx
        if not 0 <= seg_idx < self.n_segments:
            raise IndexError(
                f"Segment index {seg_idx} out of range for trajectory "
                f"with {self.n_segments} segment(s)."
            )
        return self.segment_slice(seg_idx, seg_idx)
    
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
    def plot_3d(self, n_points: int | None = None, 
                      show_body: bool = True,
                      show_nodes: bool = True,
                      node_symbol: str = 'circle',
                      traj_name: str | None = None, 
                      body_color: str | None = None, 
                      traj_color: str | None = None,
                      body_opacity: float | None = None, 
                      proximity_threshold: float | None = None,
                      renderer: str | None = None) -> go.Figure:
        """
        Create 3D plot of trajectory with optional central body.
        
        Parameters:
            n_points : int, optional
                Number of points to sample trajectory.
                If None, uses config.DEFAULT_PLOT_POINTS (default: None)
            show_body : bool, optional
                Whether to show central body sphere (default: True)
            show_nodes : bool, optional
                Whether to plot trajectory Node locations (default: True)
            node_symbol : str, optional
                Plotly symbol to use for this Trajectory's Nodes (default: 'circle')
            traj_name : str, optional
                Label for this trajectory in the legend and hover text 
                (default: Trajectory)
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
            name=traj_name or 'Trajectory',
            legendgroup='trajectory',
            legendgrouptitle=dict(text='Trajectory'),
            showlegend=True
        ))

        # Add nodes if specified
        if show_nodes:
            for trace in self._build_node_traces(node_symbol, traj_name):
                fig.add_trace(trace)
        
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
            showlegend=True,
            legend=dict(groupclick='toggleitem')
        )

        if renderer:
            fig.show(renderer=renderer)
        
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
                    show_nodes: bool = True, node_symbol: str = 'diamond',
                    traj_name: Optional[str] = None, **kwargs) -> go.Figure:
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
            show_nodes : bool, optional
                Whether to plot trajectory Node locations (default: True)
            node_symbol : str, optional
                Plotly symbol to use for this Trajectory's Nodes (default: 'diamond')
            traj_name: Legend name for this trajectory (default: 'Trajectory N')
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
        if traj_name is None:
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
            name=traj_name,
            hovertemplate='x: %{x:.1f}<br>y: %{y:.1f}<br>z: %{z:.1f}<extra></extra>',
            **kwargs
        ))

        # Add nodes if specified
        if show_nodes:
            existing_node_types = {
                trace.name for trace in fig.data
                if getattr(trace, 'legendgroup', None) == 'node_types'
            }
            for trace in self._build_node_traces(node_symbol, traj_name,
                                                skip_legend_types=existing_node_types):
                fig.add_trace(trace)
        
        return fig
    
    def _build_node_traces(
        self,
        node_symbol: str,
        traj_name: str | None,
        skip_legend_types: set | None = None) -> list:
        """
        Build Plotly Scatter3d traces for all nodes on this trajectory.

        Node types are color-coded using the constant _NODE_COLORS scheme.
        Each node type appears once in the legend regardless of how many
        nodes of that type are present. All node traces are assigned to the
        'node_types' legend group, producing a titled section in the legend
        separate from the trajectory line.

        Parameters
        ----------
        node_symbol : str
            Plotly 3D marker symbol for this trajectory's nodes.
            Different symbols allow visual distinction of nodes from
            multiple trajectories on the same figure.
        traj_name : str or None
            Optional label prefix for node hover text.

        Returns
        -------
        list
            List of go.Scatter3d traces ready to add to a figure.
        """

        if skip_legend_types is None:
            skip_legend_types = set()

        all_nodes = (
            [self._start_node]
            + self._junction_nodes
            + [self._end_node]
        )

        traces = []
        plotted_types: set[str] = set()
        first_node_overall = True

        for node in all_nodes:
            node_type = type(node).__name__
            color = _NODE_COLORS.get(node_type, '#000000')

            # Position from pre_state where available, else post_state
            if node.pre_state is not None:
                pos = node.pre_state[:3]
            else:
                pos = node.post_state[:3]

            prefix    = f"{traj_name}: " if traj_name else ""
            hover     = f"{prefix}{node_type}<br>t = {node.time:.6g}"
            show_leg = (node_type not in plotted_types and
                        node_type not in skip_legend_types)
            plotted_types.add(node_type)

            traces.append(go.Scatter3d(
                x=[pos[0]], y=[pos[1]], z=[pos[2]],
                mode='markers',
                marker=dict(
                    color=color,
                    size=8,
                    symbol=node_symbol,
                    line=dict(color='white', width=1)
                ),
                name=node_type,
                text=[hover],
                hovertemplate='%{text}<extra></extra>',
                legendgroup='node_types',
                legendgrouptitle=dict(text='Node Types')
                            if first_node_overall else None,
                showlegend=show_leg
            ))
            first_node_overall = False

        return traces

# ========== NODE CLASSES ==========
# These classes are used by Trajectory to bound and connect Trajectory segments

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
    def __init__(self, time: float | int):
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
    def _validate_state(state: ArrayLike, name: str = 'state') -> np.ndarray:
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
    
    def __init__(self, time: float | int, post_state: ArrayLike):
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
    
    def __init__(self, time: float | int, pre_state: ArrayLike):
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

    No node-level tol here (unlike ImpulsiveJunctionNode): this class' tolerance
    check is a user-error guard, not a solver-convergence path, so position
    continuity is checked against the config tolerance directly. Add tol
    only if a tool ever constructs this node from approximately-
    continuous explicit states (e.g. a transfer/Lambert solver).
    """
    
    def __init__(self, time: float | int,
                 pre_state: Optional[ArrayLike] = None,
                 post_state: Optional[ArrayLike] = None,
                 delta_v: Optional[ArrayLike] = None):
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

    Used when two segments connect smoothly -- for example, when a single
    trajectory is split at an interior time, when segments connect across
    a model boundary with matching states, or when a shooting corrector
    closes a FreeJunctionNode by driving its state defect to zero.
    pre_state and post_state are conceptually identical and mathematically
    close to within a continuity tolerance.

    The continuity tolerance is stored on the node (see tol) so the node
    carries its own validation contract. This matters when a differential
    corrector converges to a tolerance looser than the global config
    default: the converged junction is still a legitimate NullJunctionNode
    relative to the standard it was built to satisfy, and that standard
    travels with the node (e.g. across serialization) rather than being
    re-checked against a possibly-tighter global config.

    Parameters
    ----------
    time : float
        Junction time [s or nondimensional].
    pre_state : array-like
        State on the incoming side [x, y, z, vx, vy, vz] [km, km/s].
    post_state : array-like
        State on the outgoing side [x, y, z, vx, vy, vz] [km, km/s].
    tol : float or None, optional
        Absolute continuity tolerance the states must satisfy. The check
        is allclose(pre, post, rtol=config.EQUALITY_RTOL, atol=tol), so tol
        sets the absolute floor while the config relative tolerance still
        provides magnitude scaling. If None, defaults to config.EQUALITY_ATOL
        at construction time. The resolved value is stored and exposed via
        the tol property.

    Notes
    -----
    tol is provenance metadata: it does not participate in node equality
    or hashing, which compare physical state only. Two NullJunctionNodes
    with identical states but different tol values compare equal.
    """

    def __init__(self, time: float | int,
                 pre_state: ArrayLike,
                 post_state: ArrayLike,
                 tol: float | None = None):
        super().__init__(time)

        pre_state = self._validate_state(pre_state, 'pre_state')
        post_state = self._validate_state(post_state, 'post_state')

        # Resolve and store the concrete tolerance value now, so the node's
        # validation contract survives later changes to the global config.
        resolved_tol = config.EQUALITY_ATOL if tol is None else float(tol)
        if resolved_tol < 0.0:
            validation_error(
                f"NullJunctionNode tol must be non-negative, "
                f"got {resolved_tol:.3e}."
            )
        self._tol = resolved_tol

        if not np.allclose(pre_state, post_state,
                           rtol=config.EQUALITY_RTOL,
                           atol=resolved_tol):
            defect_mag = np.linalg.norm(post_state - pre_state)
            validation_error(
                f"NullJunctionNode states differ by {defect_mag:.3e}, "
                f"exceeding continuity tolerance {resolved_tol:.3e}. "
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

    @property
    def tol(self) -> float:
        """Absolute continuity tolerance this node was validated against."""
        return self._tol

    def maneuver_jacobian(self) -> np.ndarray:
        """Identity matrix -- no maneuver, no state sensitivity."""
        return np.eye(6)

    def __repr__(self) -> str:
        defect_mag = np.linalg.norm(self._post_state - self._pre_state)
        return (f"NullJunctionNode(t={self.time:.6g}, "
                f"|defect|={defect_mag:.6g}, tol={self._tol:.3g})")


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
    tol : float or None, optional
        Absolute continuity tolerance the states must satisfy. The check
        is allclose(pre, post, rtol=config.EQUALITY_RTOL, atol=tol), so tol
        sets the absolute floor while the config relative tolerance still
        provides magnitude scaling. If None, defaults to config.EQUALITY_ATOL
        at construction time. The resolved value is stored and exposed via
        the tol property.
    
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

    tol is provenance metadata: it does not participate in node equality
    or hashing, which compare physical state only. Two ImpulsiveJunctionNodes
    with identical states but different tol values compare equal.
    """
    
    def __init__(self, time: float | int,
                 pre_state: Optional[ArrayLike] = None,
                 post_state: Optional[ArrayLike] = None,
                 delta_v: Optional[ArrayLike] = None,
                 tol: float | None = None):
        super().__init__(time)

        # Resolve and store the concrete position-continuity tolerance now,
        # so the node's validation contract survives later config changes.
        resolved_tol = config.EQUALITY_ATOL if tol is None else float(tol)
        if resolved_tol < 0.0:
            raise ValueError(
                f"ImpulsiveJunctionNode tol must be non-negative, "
                f"got {resolved_tol:.3e}."
            )
        self._tol = resolved_tol
        
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
                               atol=resolved_tol):
                raise ValueError(
                    f"ImpulsiveJunctionNode positions differ by {pos_discont:.3e}, "
                    f"exceeding continuity tolerance {resolved_tol:.3e}. "
                    f"For intentional discontinuities use FreeJunctionNode."
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
    
    @property
    def tol(self) -> float:
        """Absolute position-continuity tolerance this node was validated against."""
        return self._tol
    
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
                f"|dv|={dv_mag:.6g} km/s, tol={self._tol:.3g})")


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
    A converged multiple shooting solution will convert FreeJunctionNodes
    to either a NullJunctionNode or an ImpulsiveJunctionNode as appropriate.
    """
    
    def __init__(self, time: float | int,
                 pre_state: ArrayLike,
                 post_state: ArrayLike):
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