"""System class definition for orbital propagation.

Defines the gravitational environment (equations of motion, system parameters)
for trajectory propagation. Two concrete system types are provided:

    TwoBodySystem  -- 2-body point-mass gravity with optional J2, J3, drag
    CR3BPSystem    -- Circular Restricted 3-Body Problem in rotating frame

Both are constructed through the unified System factory:

    sys = System('2body', EARTH)
    sys = System('3body', EARTH, MOON, distance=384400.0)

Direct instantiation of TwoBodySystem or CR3BPSystem raises TypeError.
Use isinstance() checks where type-specific branching is needed:

    if isinstance(sys, CR3BPSystem):
        print(sys.L_star)

Created with the assistance of Claude Sonnet 4.6 by Anthropic.
"""

import numpy as np
import warnings
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Any
from enum import Enum
import heyoka as hy
from .orbital_elements import OrbitalElements, OEType
from .trajectory import (
    Trajectory,
    Node,
    BoundaryNode,
    StartBoundaryNode,
    EndBoundaryNode,
    ImpulsiveBoundaryNode,
    JunctionNode,
    NullJunctionNode,
    ImpulsiveJunctionNode,
    FreeJunctionNode
)
from .satellite import Satellite
from .utils import validation_error
from .config import config


# ========== ENUMERATIONS ==========

class SysType(Enum):
    TWO_BODY = '2body'
    CR3BP = '3body'
    N_BODY = 'Nbody'


# ========== PARAMETER DATACLASSES ==========

@dataclass(frozen=True)
class BodyParams:
    """
    Immutable parameters for a celestial body.

    Attributes
    ----------
    mu : float
        Gravitational parameter [km^3/s^2]
    radius : float
        Equatorial radius [km]
    J2 : float, optional
        J2 zonal harmonic coefficient [dimensionless].
        Required if J2 perturbations are enabled.
    J3 : float, optional
        J3 zonal harmonic coefficient [dimensionless].
        Requires J2 to also be set.
    rotation_rate : float, optional
        Angular rotation rate [rad/s].
        Required if atmospheric drag is enabled.
    name : str, optional
        Human-readable body name.
    """
    mu: float
    radius: float
    J2: Optional[float] = None
    J3: Optional[float] = None
    rotation_rate: Optional[float] = None
    name: Optional[str] = None

    def __post_init__(self):
        if self.mu <= 0:
            raise ValueError(
                f"Gravitational parameter must be positive, got {self.mu}"
            )
        if self.radius <= 0:
            raise ValueError(
                f"Radius must be positive, got {self.radius}"
            )
        if self.J2 is not None and abs(self.J2) > 1:
            validation_error(
                f"J2 coefficient seems unrealistic: {self.J2}. "
                "Expected |J2| << 1 for physical bodies."
            )
        if self.J3 is not None and self.J2 is None:
            raise ValueError(
                "J2 must be specified when J3 is used. "
                "Set J2=0.0 explicitly if you intend to model J3 without J2."
            )
        if self.J3 is not None:
            if abs(self.J3) > 1:
                validation_error(
                    f"J3 coefficient seems unrealistic: {self.J3}. "
                    "Expected |J3| << 1 for physical bodies."
                )
            assert self.J2 is not None  # guaranteed by the dependency check above
            if abs(self.J3) > abs(self.J2):
                validation_error(
                    f"|J3| ({abs(self.J3):.3e}) exceeds |J2| ({abs(self.J2):.3e}). "
                    "Physical gravity models have J2 dominant. "
                    "If J2=0.0 was set intentionally, consider whether this "
                    "model represents your intended gravity field."
                )


@dataclass(frozen=True)
class AtmoParams:
    """
    Immutable parameters for an exponential atmosphere model.

    The density profile follows: rho(r) = rho0 * exp(-(r - r0)/H)

    Attributes
    ----------
    rho0 : float
        Reference density at reference altitude [kg/m^3]
    H : float
        Scale height [m]
    r0 : float
        Reference radius at which rho0 is defined [m]

    Notes
    -----
    AtmosphereParams uses SI units (m, kg/m^3), unlike the rest of the
    package which uses km. Conversions are applied internally when
    building the symbolic EOM.
    """
    rho0: float
    H: float
    r0: float

    def __post_init__(self):
        if self.rho0 <= 0:
            raise ValueError(f"Reference density must be positive, got {self.rho0}")
        if self.H <= 0:
            raise ValueError(f"Scale height must be positive, got {self.H}")
        if self.r0 <= 0:
            raise ValueError(f"Reference radius must be positive, got {self.r0}")


# ========== PERIODIC ORBIT DATACLASS ==========

@dataclass(frozen=True)
class PeriodicOrbit:
    """
    Initial conditions and period for a periodic CR3BP orbit.

    The period is a dynamical property that cannot be computed from the
    state vector alone -- it must be determined numerically and stored here.

    Parameters
    ----------
    state : OrbitalElements
        Initial conditions (CR3BP nondimensional elements)
    period : float
        Nondimensional time for one complete period
    name : str, optional
        Human-readable identifier (e.g. 'L1 Lyapunov')
    jacobi : float, optional
        Jacobi constant (useful for orbit family identification)
    """
    state: OrbitalElements
    period: float
    name: str = ""
    jacobi: Optional[float] = None

    def __post_init__(self):
        if self.period <= 0:
            raise ValueError(f"Period must be positive, got {self.period}")
        if self.state.element_type.value != 'cr3bp':
            raise ValueError("PeriodicOrbit requires CR3BP elements")


# ========== INTERNAL HELPERS ==========

class _BodyParamsWithND:
    """
    Wrapper for BodyParams that adds a nondimensional radius property.

    Used by CR3BPSystem to expose radius_nd without modifying the
    immutable BodyParams dataclass.
    """
    def __init__(self, body_params: BodyParams, L_star: float):
        self._body_params = body_params
        self._L_star = L_star

    # Delegate all other attributes to the underlying BodyParams
    @property
    def radius_nd(self) -> float:
        """Nondimensional radius [L_star units]."""
        return self._body_params.radius / self._L_star

    def __getattr__(self, name):
        return getattr(self._body_params, name)

    def __repr__(self):
        return repr(self._body_params)

     # need to preserve equality through the wrapper
    def __eq__(self, other):
        if isinstance(other, _BodyParamsWithND):
            return self._body_params == other._body_params
        elif isinstance(other, BodyParams):
            return self._body_params == other
        return NotImplemented

    # Hash based on wrapped BodyParams.
    def __hash__(self):
        return hash(self._body_params)


# ========== SYSTEM BASE CLASS ==========

class System:
    """
    Factory and base class for orbital propagation systems.

    Construct via the factory interface -- do not instantiate directly:

        sys = System('2body', primary_body, ...)
        sys = System('3body', primary_body, secondary_body, distance=...)

    The factory returns a TwoBodySystem or CR3BPSystem instance depending
    on the base_type argument. Both are subclasses of System, so
    isinstance(sys, System) is True for either.

    Parameters
    ----------
    base_type : str or SysType
        Type of dynamics. Accepted strings:
        '2body', '2BODY', 'Two_Body' -> TwoBodySystem
        '3body', '3BODY', 'Three_Body', 'CR3BP' -> CR3BPSystem
    primary_body : BodyParams
        Parameters for the primary celestial body.
    *args, **kwargs
        Forwarded to the appropriate subclass __init__.
        See TwoBodySystem and CR3BPSystem for their specific parameters.

    Notes
    -----
    - System instances are immutable after construction.
    - Heyoka integrators are compiled on construction by default.
      Use compile=False to defer compilation.
    - Instance counting: a ResourceWarning is issued when more than
      INSTANCE_WARNING_THRESHOLD (see KyklosConfig) System objects exist 
      simultaneously, since each compiled integrator consumes significant memory.
    """

    # ========== CLASS VARIABLES ==========
    _instance_count = 0
    _VALID_PERTURBATIONS = frozenset(("J2", "J3", "drag"))

    # ========== FACTORY DISPATCH ==========
    def __new__(cls, base_type, primary_body, *args, **kwargs):
        """
        Allocate a TwoBodySystem or CR3BPSystem instance.

        When called as System(...), routes to the appropriate subclass.
        Direct instantiation of TwoBodySystem or CR3BPSystem is blocked.
        """
        if cls is System:
            sys_type = System._parse_base_type(base_type)
            if sys_type == SysType.TWO_BODY:
                return object.__new__(TwoBodySystem)
            elif sys_type == SysType.CR3BP:
                return object.__new__(CR3BPSystem)
            else:
                raise ValueError(
                    f"No System subclass is implemented for type "
                    f"'{sys_type.value}'. Currently supported: '2body', '3body'."
                )
        else:
            # Prevent direct subclass instantiation.
            # Subclasses are public for isinstance() checks but cannot be
            # constructed without going through the System factory.
            raise TypeError(
                f"{cls.__name__} cannot be instantiated directly. "
                f"Use System('2body', ...) or System('3body', ...) instead."
            )

    # ========== STATIC / CLASS METHODS ==========
    @staticmethod
    def _parse_base_type(base_type):
        """Convert string or SysType enum to SysType."""
        if isinstance(base_type, SysType):
            return base_type
        elif isinstance(base_type, str):
            type_map = {
                '2body':      SysType.TWO_BODY,
                '2BODY':      SysType.TWO_BODY,
                'Two_Body':   SysType.TWO_BODY,
                '3body':      SysType.CR3BP,
                '3BODY':      SysType.CR3BP,
                'Three_Body': SysType.CR3BP,
                'CR3BP':      SysType.CR3BP,
            }
            if base_type in type_map:
                return type_map[base_type]
            raise ValueError(
                f"Unknown base type '{base_type}'. "
                f"Use: {list(type_map.keys())}"
            )
        raise TypeError(
            f"base_type must be SysType or str, got {type(base_type)}"
        )

    @classmethod
    def get_instance_count(cls):
        """Return the current number of live System instances."""
        return cls._instance_count

    @classmethod
    def reset_instance_count(cls):
        """Reset instance counter to zero. Intended for use in tests."""
        cls._instance_count = 0

    # ========== SHARED PROPERTIES ==========
    @property
    def base_type(self) -> SysType:
        """Type of base dynamics (SysType enum)."""
        return self._base_type

    @property
    def is_compiled(self) -> bool:
        """True if the Heyoka integrator has been compiled."""
        return self._cached_integrator is not None
    
    @property
    def is_func_compiled(self) -> bool:
        """True if the vector-field evaluator (cfunc) has been compiled."""
        return self._cached_func is not None

    @property
    def cached_eom(self) -> Optional[List[Tuple]]:
        """Cached symbolic equations of motion as list of (var, rhs) tuples."""
        return self._cached_eom

    @property
    def requires_satellite(self) -> bool:
        """True if this system requires a Satellite object for propagation."""
        return len(self._param_info['param_map']) > 0

    # ========== PUBLIC COMPILE METHODS ==========
    def compile(self):
        """
        Compile the Heyoka integrator if not already compiled.

        Called automatically on construction unless compile=False was passed.

        Returns
        -------
        self
            Returns self for method chaining.
        """
        self._compile_integrator()
        return self

    def compile_var(self, order=1):
        """
        Compile the variational ODE integrator if not already compiled.

        Not compiled by default. Required if STM output is needed.
        The propagator auto-compiles this if with_stm=True is requested.

        Parameters
        ----------
        order : int, optional
            Order of variational equations. Default 1 (first-order STM).

        Returns
        -------
        self
            Returns self for method chaining.
        """
        self._compile_var_integrator(order=order)
        return self
    
    def compile_func(self):
        """
        Compile the vector-field evaluator (cfunc) if not already compiled.

        Not compiled by default. Required to evaluate the equations-of-motion
        right-hand side numerically -- for example, the time-derivative
        columns of a variable-time shooting Jacobian. vector_field()
        auto-compiles this on first use, mirroring how the propagator
        auto-compiles the variational integrator on first STM request.

        Returns
        -------
        self
            Returns self for method chaining.
        """
        self._compile_func_evaluator()
        return self

    def vector_field(self, state, pars=None):
        """
        Evaluate the equations-of-motion right-hand side f(state).

        Compiles the cfunc evaluator on first call and caches it, so the
        compilation cost is paid once. Supports a single state, shape (6,),
        or a batch of states stacked as columns, shape (6, k), evaluated in
        one call.

        Parameters
        ----------
        state : array-like
            State [x, y, z, vx, vy, vz], shape (6,), or a batch (6, k).
        pars : array-like, optional
            Runtime parameter values for the hy.par[] slots, length
            n_params. Required only for systems that carry runtime
            parameters (e.g. drag or SRP supplied via a Satellite). CR3BP
            and unperturbed two-body have none, so this may be omitted.

        Returns
        -------
        np.ndarray
            Time derivative f(state), the same shape as the input.

        Notes
        -----
        Assumes autonomous dynamics: f depends on the state only, with no
        explicit time dependence. This holds for the rotating-frame CR3BP
        and for inertial two-body with position/velocity-dependent
        perturbations. Non-autonomous systems would require a time argument
        and are out of scope here.
        """
        self._compile_func_evaluator()

        state = np.ascontiguousarray(state, dtype=float)
        n_params = len(self._param_info['param_map'])
        if pars is None:
            if n_params != 0:
                raise ValueError(
                    f"This system has {n_params} runtime parameter(s); supply "
                    f"pars (e.g. from a Satellite) to evaluate the vector "
                    f"field."
                )
        else:
            pars = np.asarray(pars, dtype=float)

        if n_params == 0:
            out = self._cached_func(state)
        else:
            out = self._cached_func(state, pars=pars)
        return np.asarray(out)

    # ========== INTERNAL COMPILE METHODS ==========
    def _compile_message(self) -> str:
        """
        Return the compilation status message.

        Overridden in subclasses to include type-specific detail
        (e.g. active perturbations for TwoBodySystem).
        """
        return f"Compiling {self._base_type.value} integrator"

    def _compile_integrator(self):
        """
        Compile the Heyoka taylor_adaptive integrator.

        This performs automatic differentiation and LLVM JIT compilation.
        Typically takes 1-5 seconds depending on EOM complexity.
        """
        if self._cached_integrator is not None:
            return  # Already compiled

        n_params = len(self._param_info['param_map'])
        dummy_params = [0.0] * n_params

        print(self._compile_message() + "...")
        self._cached_integrator = hy.taylor_adaptive(
            sys=self._cached_eom,
            state=[0.0] * 6,
            pars=dummy_params,
        )
        print("Compilation complete")

    def _compile_var_integrator(self, order=1):
        """
        Compile the variational Heyoka integrator.

        Constructs a var_ode_sys from the cached symbolic EOM and compiles
        with compact_mode=True (required for variational systems).
        """
        if self._cached_var_integrator is not None:
            return  # Already compiled

        if order > 1:
            raise ValueError(
                "Higher order State Transition Tensors not supported yet."
            )

        vsys = hy.var_ode_sys(
            self._cached_eom,
            hy.var_args.vars,
            order=order
        )

        n_params = len(self._param_info['param_map'])
        dummy_params = [0.0] * n_params

        print(f"Compiling variational integrator (order={order})...")
        self._cached_var_integrator = hy.taylor_adaptive(
            sys=vsys,
            state=[0.0] * 6,
            pars=dummy_params,
            compact_mode=True  # CRITICAL for variational systems
        )
        self._var_order = order
        print("Variational compilation complete")

    def _compile_func_evaluator(self):
        """
        Compile the cfunc vector-field evaluator from the cached EOM.

        Extracts the right-hand-side expressions from the cached symbolic
        EOM and compiles them as a numerical function of the six state
        variables, in canonical [x, y, z, vx, vy, vz] order. Cheap relative
        to the integrator compile, since there are no variational equations.
        """
        if self._cached_func is not None:
            return  # Already compiled

        rhs = [expr for (_, expr) in self._cached_eom]
        x, y, z, vx, vy, vz = hy.make_vars("x", "y", "z", "vx", "vy", "vz")

        print("Compiling vector-field evaluator (cfunc)...")
        self._cached_func = hy.cfunc(rhs, vars=[x, y, z, vx, vy, vz])
        print("Vector-field compilation complete")

    # ========== PROPAGATION HELPERS ==========
    def _process_satellite(self, satellite: Satellite) -> np.ndarray:
        """
        Extract runtime parameters from a Satellite object.

        Maps satellite properties to the hy.par[] array in the order
        defined by self._param_info['param_map'].

        Parameters
        ----------
        satellite : Satellite
            Satellite object containing required physical properties.

        Returns
        -------
        np.ndarray
            Parameter values in the order expected by the integrator.

        Raises
        ------
        ValueError
            If satellite is missing a required property.
        """
        params = []
        for name, idx in self._param_info['param_map']:
            if name == 'Cd_A':
                params.append(satellite.Cd_A)
            elif name == 'mass':
                params.append(satellite.mass)
            else:
                # This shouldn't happen until we add new perturbations
                raise ValueError(
                    f"Unknown satellite parameter '{name}' required by system. "
                    "This may indicate a bug in the System implementation."
                )
        return np.array(params)

    def _validate_times(self, times) -> np.ndarray:
        """
        Validate and convert a times sequence to a sorted float array.

        Parameters
        ----------
        times : array-like
            Sequence of boundary times. Must be 1D, length >= 2, and
            strictly increasing (forward propagation only).

        Returns
        -------
        np.ndarray
            Validated float array of times.
        """
        times = np.asarray(times, dtype=float)
        if times.ndim != 1 or len(times) < 2:
            raise ValueError(
                f"times must be a 1D sequence of length >= 2, "
                f"got shape {times.shape}."
            )
        if not np.all(np.diff(times) > 0):
            raise ValueError(
                "Times must be strictly increasing. "
                "Backward propagation is not supported via propagate(). "
                "Use Trajectory.extend_back() for backward extension."
            )
        return times

    def _propagate_single_output(
            self,
            state_array: np.ndarray,
            t_start: float,
            t_end: float,
            with_stm: bool,
            stm_order: int,
            satellite: Satellite | None
    ) -> Any:
        """
        Propagate one segment and return the Heyoka continuous output object.

        This is the atomic integration operation used by all propagation
        modes. It does not construct a Trajectory.

        Parameters
        ----------
        state_array : np.ndarray
            6-element initial state [km, km/s].
        t_start : float
            Segment start time.
        t_end : float
            Segment end time. Must be > t_start.
        with_stm : bool
            If True, propagate with variational equations.
        stm_order : int
            Order of variational equations.
        satellite : Satellite or None
            Satellite model for parameterised perturbations.

        Returns
        -------
        Heyoka continuous output object
        """
        # Select integrator
        if with_stm:
            if (self._cached_var_integrator is None
                    or self._var_order != stm_order):
                self._compile_var_integrator(order=stm_order)
            ta = self._cached_var_integrator
        else:
            if not self.is_compiled:
                self._compile_integrator()
            ta = self._cached_integrator
        assert ta is not None

        # Handle satellite parameters
        if satellite is not None:
            params_array = self._process_satellite(satellite)
            ta.pars[:] = params_array
        elif len(self._param_info['param_map']) > 0:
            raise ValueError(
                "This system requires a Satellite object. "
                "The following properties are needed: "
                f"{[name for name, _ in self._param_info['param_map']]}"
            )

        # Set initial conditions
        ta.time = float(t_start)
        ta.state[:6] = state_array
        if with_stm:
            ta.state[6:42] = np.eye(6).flatten()
            if stm_order == 2:
                ta.state[42:] = 0.0

        # Propagate
        c_out = ta.propagate_until(float(t_end), c_output=True)[4]

        # Validate integration result
        if not np.all(np.isfinite(ta.state)):
            validation_error(
                f"Integration failed: state became invalid during propagation.\n"
                f"Initial state: {state_array}\n"
                f"Final time: {ta.time}\n"
                f"Final state: {ta.state}\n"
                f"Likely causes:\n"
                f"  - Initial position too close to central body\n"
                f"  - Collision with central body during propagation\n"
                f"  - Numerical instability in perturbation models"
            )
        if c_out is None:
            validation_error(
                "Integration produced no continuous output. "
                "This may indicate a severe integration failure."
            )

        return c_out

    def _propagate_single(
            self,
            initial_state: OrbitalElements | np.ndarray | BoundaryNode,
            times: np.ndarray,
            with_stm: bool,
            stm_order: int,
            satellite: Satellite | None
    ) -> Trajectory:
        """Single-segment propagation. Handles BoundaryNode input."""
        t_start, t_end = float(times[0]), float(times[1])
        provided_start_node = None

        if isinstance(initial_state, BoundaryNode):
            if not np.isclose(initial_state.time, t_start,
                              rtol=config.EQUALITY_RTOL,
                              atol=config.EQUALITY_ATOL):
                raise ValueError(
                    f"BoundaryNode time ({initial_state.time:.6g}) does not "
                    f"match times[0] ({t_start:.6g})."
                )
            if initial_state.post_state is None:
                raise ValueError(
                    "BoundaryNode provided as initial_state must have a "
                    "post_state. Use StartBoundaryNode or ImpulsiveBoundaryNode."
                )
            state_array = self._state_to_array(initial_state.post_state)
            provided_start_node = initial_state
        else:
            state_array = self._state_to_array(initial_state)

        c_out = self._propagate_single_output(
            state_array, t_start, t_end, with_stm, stm_order, satellite
        )

        return Trajectory(
            self, [c_out],
            stm_order=stm_order if with_stm else None,
            start_node=provided_start_node
        )

    def _propagate_multi(
            self,
            initial_states: list[OrbitalElements | np.ndarray] | np.ndarray,
            times: np.ndarray,
            with_stm: bool,
            stm_order: int,
            satellite: Satellite | None
    ) -> Trajectory:
        """
        Multi-segment propagation from arrays (Mode 1).

        Each segment is propagated independently from its given initial
        state. Junction nodes are inferred automatically from actual
        propagated states via Trajectory._infer_junction.
        """
        if isinstance(initial_states, np.ndarray):
            if initial_states.ndim != 2 or initial_states.shape[1] != 6:
                raise ValueError(
                    f"2D initial_state must have shape (n_seg, 6), "
                    f"got {initial_states.shape}."
                )
            states_list = [initial_states[k] for k in range(len(initial_states))]
        else:
            states_list = initial_states

        n_seg = len(times) - 1
        if len(states_list) != n_seg:
            raise ValueError(
                f"initial_state has {len(states_list)} element(s) but "
                f"times implies {n_seg} segment(s) (len(times) - 1 = {n_seg})."
            )

        outputs = []
        for k in range(n_seg):
            state_array = self._state_to_array(states_list[k])
            t_s, t_e = float(times[k]), float(times[k + 1])
            c_out = self._propagate_single_output(
                state_array, t_s, t_e, with_stm, stm_order, satellite
            )
            outputs.append(c_out)

        return Trajectory(
            self, outputs,
            stm_order=stm_order if with_stm else None
        )

    def _propagate_from_nodes(
            self,
            nodes: list,
            with_stm: bool,
            stm_order: int,
            satellite: Satellite | None
    ) -> Trajectory:
        """
        Node-based multi-segment propagation (Mode 2).

        Times and initial states are extracted from the node list. Junction
        nodes are recreated from actual integration results, preserving the
        design intent (delta-v, state jump) from the input nodes while
        updating pre-states to reflect true dynamics.
        """
        if len(nodes) < 2:
            raise ValueError(
                f"nodes must contain at least 2 elements, got {len(nodes)}."
            )

        if (not isinstance(nodes[0], BoundaryNode)
                or nodes[0].post_state is None):
            raise ValueError(
                "nodes[0] must be a StartBoundaryNode or ImpulsiveBoundaryNode "
                "(must have a post_state defining the trajectory start)."
            )

        if (not isinstance(nodes[-1], BoundaryNode)
                or nodes[-1].pre_state is None):
            raise ValueError(
                "nodes[-1] must be an EndBoundaryNode or ImpulsiveBoundaryNode "
                "(must have a pre_state defining the trajectory end)."
            )

        for k, node in enumerate(nodes[1:-1], start=1):
            if not isinstance(node, JunctionNode):
                raise ValueError(
                    f"nodes[{k}] must be a JunctionNode, "
                    f"got {type(node).__name__}."
                )

        n_seg = len(nodes) - 1
        outputs = []
        new_junction_nodes = []
        current_post_state = nodes[0].post_state

        for k in range(n_seg):
            state_array = self._state_to_array(current_post_state)
            t_start = float(nodes[k].time)
            t_end   = float(nodes[k + 1].time)

            if t_end <= t_start:
                raise ValueError(
                    f"Segment {k} has t_end ({t_end:.6g}) <= t_start "
                    f"({t_start:.6g}). All segments must propagate forward."
                )

            c_out = self._propagate_single_output(
                state_array, t_start, t_end, with_stm, stm_order, satellite
            )
            outputs.append(c_out)

            if k < n_seg - 1:
                orig = nodes[k + 1]
                actual_pre = c_out(float(t_end))[:6].copy()

                if isinstance(orig, NullJunctionNode):
                    new_node = NullJunctionNode(
                        t_end, actual_pre, actual_pre.copy()
                    )
                    current_post_state = actual_pre

                elif isinstance(orig, ImpulsiveJunctionNode):
                    post = actual_pre.copy()
                    post[3:6] += orig.delta_v
                    new_node = ImpulsiveJunctionNode(
                        t_end, pre_state=actual_pre, post_state=post
                    )
                    current_post_state = post

                elif isinstance(orig, FreeJunctionNode):
                    discrepancy = np.linalg.norm(actual_pre - orig.pre_state)
                    if (config.STRICT_VALIDATION
                            and discrepancy > config.EQUALITY_ATOL):
                        warnings.warn(
                            f"FreeJunctionNode at nodes[{k + 1}]: propagated "
                            f"pre_state differs from nominal by "
                            f"{discrepancy:.3e}. Node updated with actual "
                            f"integration result. Set STRICT_VALIDATION=False "
                            f"to silence this warning.",
                            UserWarning,
                            stacklevel=3
                        )
                    new_node = FreeJunctionNode(
                        t_end, actual_pre, orig.post_state
                    )
                    current_post_state = orig.post_state

                else:
                    raise TypeError(
                        f"Unrecognised JunctionNode type at nodes[{k + 1}]: "
                        f"{type(orig).__name__}"
                    )

                new_junction_nodes.append(new_node)

        return Trajectory(
            self, outputs,
            junction_nodes=new_junction_nodes,
            stm_order=stm_order if with_stm else None,
            start_node=nodes[0]
        )

    # ========== PROPAGATION (PUBLIC) ==========
    def propagate(
            self,
            initial_state: (OrbitalElements | np.ndarray
                            | BoundaryNode | list | None) = None,
            times: list | np.ndarray | None = None,
            *,
            nodes: list | None = None,
            with_stm: bool = False,
            stm_order: int = 1,
            satellite: Satellite | None = None
    ) -> Trajectory:
        """
        Propagate trajectories with dense output.

        Three input modes are supported:

        **Single segment**:
        Provide a single initial_state and times=[t_start, t_end].
        initial_state may be OrbitalElements, np.ndarray, or a BoundaryNode.
        If a BoundaryNode, times[0] must match the node time within tolerance.

        **Multi-segment (Mode 1)**:
        Provide a list of initial states and times of length n_seg + 1.
        Each adjacent pair in times defines one segment.
        len(initial_state) must equal len(times) - 1.
        Nodes are inferred automatically from actual propagated states.

        **Multi-segment node-based (Mode 2)**:
        Provide nodes=[StartBoundaryNode, Junction1, ..., EndBoundaryNode].
        times and initial_state must be None.
        Junction nodes are recreated from actual integration results,
        preserving design intent (delta-v, state jump) from the input nodes.

        Parameters
        ----------
        initial_state : OrbitalElements, np.ndarray, BoundaryNode, list, or None
            Initial state(s). Single state for one segment, list for multiple.
            Must be None when nodes is provided.
        times : array-like or None
            Boundary times of length n_seg + 1.  Must be strictly increasing.
            Must be None when nodes is provided.
        nodes : list of Node, optional
            [StartBoundaryNode, JunctionNode, ..., EndBoundaryNode] for
            node-based propagation. Cannot be combined with initial_state
            or times.
        with_stm : bool, optional
            If True, propagate the State Transition Matrix via variational
            equations. Default: False.
        stm_order : int, optional
            Order of variational equations (1 = first-order STM). Default: 1.
        satellite : Satellite or None, optional
            Satellite object providing physical properties (mass, Cd*A).
            Required when the system includes satellite-dependent perturbations
            such as drag.

        Returns
        -------
        Trajectory

        Raises
        ------
        ValueError
            If both nodes and initial_state are provided, if inputs are
            structurally inconsistent, or if integration fails.

        Notes
        -----
        STM propagation increases computational cost:
        - Compilation: 2-5x longer due to symbolic differentiation
        - Memory: state vector grows from 6 to 42 (order=1) or 258 (order=2)
          elements per integration step
        The STM is automatically initialised to the identity matrix at each
        segment boundary.
        """
        # Mutual exclusivity
        if nodes is not None and initial_state is not None:
            raise ValueError(
                "Cannot provide both nodes and initial_state. "
                "Use nodes alone for node-based propagation, or "
                "initial_state with times for state-based propagation."
            )
        if nodes is not None and times is not None:
            raise ValueError(
                "Cannot provide times when using node-based propagation. "
                "Times are determined from node timestamps."
            )

        # Mode 2: node-based
        if nodes is not None:
            return self._propagate_from_nodes(
                nodes, with_stm, stm_order, satellite
            )

        # Modes 1 and single: require initial_state and times
        if initial_state is None:
            raise ValueError(
                "Must provide either initial_state (with times) or nodes."
            )
        if times is None:
            raise ValueError(
                "times must be provided when using initial_state."
            )

        times_arr = self._validate_times(times)

        if (isinstance(initial_state, list)
                or (isinstance(initial_state, np.ndarray)
                    and initial_state.ndim == 2)):
            return self._propagate_multi(
                initial_state, times_arr, with_stm, stm_order, satellite
            )
        else:
            if len(times_arr) != 2:
                raise ValueError(
                    f"Single-segment propagation requires times of length 2 "
                    f"[t_start, t_end], got length {len(times_arr)}. "
                    "For multi-segment propagation, provide initial_state "
                    "as a list."
                )
            return self._propagate_single(
                initial_state, times_arr, with_stm, stm_order, satellite
            )

    # ========== LIFECYCLE ==========
    def __del__(self):
        """Decrement instance count when System is garbage collected."""
        System._instance_count -= 1


# ========== TWO-BODY SYSTEM ==========

class TwoBodySystem(System):
    """
    Two-body gravitational system with optional perturbations.

    Models spacecraft motion under point-mass central body gravity, with
    optional zonal harmonic (J2, J3) and atmospheric drag perturbations.

    Construct via the System factory -- do not instantiate directly:

        sys = System('2body', primary_body)
        sys = System('2body', primary_body, perturbations=('J2',))
        sys = System('2body', primary_body,
                     perturbations=('drag',), atmosphere=atmo_params)

    Parameters
    ----------
    base_type : str or SysType
        Must be '2body' or equivalent string. Provided by System factory.
    primary_body : BodyParams
        Parameters for the central body.
    perturbations : tuple of str, optional
        Perturbation models to include. Options: 'J2', 'J3', 'drag'.
        Default is empty tuple (point mass dynamics only).
        Use trailing comma for single perturbation: ('J2',)
    atmosphere : AtmoParams, optional
        Atmospheric model parameters. Required if 'drag' is in perturbations.
    compile : bool, optional
        If True (default), compile the Heyoka integrator immediately.

    Attributes
    ----------
    base_type : SysType
        Always SysType.TWO_BODY.
    primary_body : BodyParams
        Central body parameters.
    perturbations : tuple of str
        Active perturbation models.
    atmosphere : AtmoParams or None
        Atmosphere model if drag is active, else None.
    is_compiled : bool
        True if the Heyoka integrator has been compiled.
    requires_satellite : bool
        True if drag perturbation is active (requires Satellite at propagation).
    """

    def __init__(
            self,
            base_type,
            primary_body: BodyParams,
            secondary_body=None,
            *,
            perturbations: tuple = (),
            atmosphere: Optional[AtmoParams] = None,
            distance=None,
            compile: bool | None = None
    ):
        # Reject 2-body-irrelevant arguments
        if secondary_body is not None:
            raise TypeError(
                "TwoBodySystem does not accept secondary_body. "
                "For 3-body dynamics use System('3body', ...)."
            )
        if distance is not None:
            raise TypeError(
                "TwoBodySystem does not accept distance. "
                "For 3-body dynamics use System('3body', ...)."
            )

        if compile is None:
            compile = config.DEFAULT_COMPILE

        self._base_type = SysType.TWO_BODY

        # --- Validate perturbations ---
        for pert in perturbations:
            if pert not in System._VALID_PERTURBATIONS:
                raise ValueError(
                    f"Unknown perturbation '{pert}'. "
                    f"Valid options: {System._VALID_PERTURBATIONS}"
                )
        if len(perturbations) != len(set(perturbations)):
            raise ValueError(f"Duplicate perturbations found: {perturbations}")

        if "J2" in perturbations and primary_body.J2 is None:
            raise ValueError(
                "J2 perturbation requested but primary_body.J2 is None"
            )
        if "J3" in perturbations and primary_body.J3 is None:
            raise ValueError(
                "J3 perturbation requested but primary_body.J3 is None"
            )
        if "drag" in perturbations:
            if atmosphere is None:
                raise ValueError(
                    "drag perturbation requested but atmosphere is None"
                )
            if primary_body.rotation_rate is None:
                raise ValueError(
                    "drag perturbation requires primary_body.rotation_rate"
                )

        # --- Store parameters ---
        self._primary_body = primary_body
        self._perturbations = tuple(perturbations)
        self._atmosphere = atmosphere

        # --- Integrator cache ---
        self._cached_eom = None
        self._cached_integrator = None
        self._param_info = None
        self._cached_var_integrator = None
        self._var_order = None
        self._cached_func = None

        # --- Build EOM and compile ---
        self._cached_eom, self._param_info = self._build_eom()
        if compile:
            self._compile_integrator()

        # --- Instance counting ---
        System._instance_count += 1
        if System._instance_count > config.INSTANCE_WARNING_THRESHOLD:
            warnings.warn(
                f"Created {System._instance_count} System instances. "
                "Each System caches compiled Heyoka integrators, "
                "which can consume significant memory. Consider reusing "
                "System objects when possible.",
                ResourceWarning,
                stacklevel=2
            )

    # ========== COMPILE MESSAGE ==========
    def _compile_message(self) -> str:
        msg = "Compiling 2-body integrator"
        if self._perturbations:
            msg += f" with {', '.join(self._perturbations)}"
        return msg

    # ========== EOM CONSTRUCTION ==========
    def _build_eom(self):
        """
        Build symbolic 2-body equations of motion with active perturbations.

        Constructs Heyoka symbolic expressions for the acceleration vector,
        starting from point-mass gravity and adding requested perturbations.

        Returns
        -------
        sys : list of (var, rhs) tuples
            Heyoka ODE system definition ready for taylor_adaptive().
        param_info : dict
            Runtime parameter mapping:
            - 'param_map': list of (name, index) tuples for hy.par[] array
            - 'description': dict with human-readable parameter descriptions
        """
        x, y, z, vx, vy, vz = hy.make_vars("x", "y", "z", "vx", "vy", "vz")
        r = hy.sqrt(x**2 + y**2 + z**2)
        mu = self._primary_body.mu

        # Point-mass gravitational acceleration
        a_total_x = -mu * x / r**3
        a_total_y = -mu * y / r**3
        a_total_z = -mu * z / r**3

        param_map = []
        param_desc = {}
        next_param_idx = 0

        if "J2" in self._perturbations:
            a_J2_x, a_J2_y, a_J2_z = self._build_J2_perturbation(
                x, y, z, r, mu
            )
            a_total_x = a_total_x + a_J2_x
            a_total_y = a_total_y + a_J2_y
            a_total_z = a_total_z + a_J2_z

        if "J3" in self._perturbations:
            a_J3_x, a_J3_y, a_J3_z = self._build_J3_perturbation(
                x, y, z, r, mu
            )
            a_total_x = a_total_x + a_J3_x
            a_total_y = a_total_y + a_J3_y
            a_total_z = a_total_z + a_J3_z

        if "drag" in self._perturbations:
            a_drag_x, a_drag_y, a_drag_z, drag_params = \
                self._build_drag_perturbation(
                    x, y, z, vx, vy, vz, r, next_param_idx
                )
            a_total_x = a_total_x + a_drag_x
            a_total_y = a_total_y + a_drag_y
            a_total_z = a_total_z + a_drag_z
            param_map.extend(drag_params['param_map'])
            param_desc.update(drag_params['description'])
            next_param_idx += len(drag_params['param_map'])

        eom = [
            (x, vx),
            (y, vy),
            (z, vz),
            (vx, a_total_x),
            (vy, a_total_y),
            (vz, a_total_z)
        ]
        param_info = {'param_map': param_map, 'description': param_desc}
        return eom, param_info

    def _build_J2_perturbation(self, x, y, z, r, mu):
        """Build symbolic J2 perturbation acceleration terms."""
        J2 = self._primary_body.J2
        R = self._primary_body.radius
        assert J2 is not None  # validated in __init__
        # Common factor: (3/2) * J2 * mu * R^2 / r^5
        factor = 1.5 * J2 * mu * R**2 / r**5
        # a_J2 = factor * [x(5z^2/r^2 - 1), y(5z^2/r^2 - 1), z(5z^2/r^2 - 3)]
        z2_r2 = z**2 / r**2
        a_J2_x = factor * x * (5.0 * z2_r2 - 1.0)
        a_J2_y = factor * y * (5.0 * z2_r2 - 1.0)
        a_J2_z = factor * z * (5.0 * z2_r2 - 3.0)
        return a_J2_x, a_J2_y, a_J2_z

    def _build_J3_perturbation(self, x, y, z, r, mu):
        """Build symbolic J3 perturbation acceleration terms."""
        J3 = self._primary_body.J3
        R = self._primary_body.radius
        assert J3 is not None  # validated in __init__
        z2_r2 = z**2 / r**2
        # a_J3_x = (5/2) * mu * J3 * R^3 * x * z / r^7 * (7*z^2/r^2 - 3)
        factor_xy = 2.5 * J3 * mu * R**3 * z / r**7
        a_J3_x = factor_xy * x * (7.0 * z2_r2 - 3.0)
        a_J3_y = factor_xy * y * (7.0 * z2_r2 - 3.0)
        # a_J3_z = (mu*J3*R^3/r^5) * (1.5 - 15*z2_r2 + 17.5*z2_r2^2)
        factor_z = mu * J3 * R**3 / r**5
        a_J3_z = factor_z * (1.5 - 15.0 * z2_r2 + 17.5 * z2_r2**2)
        return a_J3_x, a_J3_y, a_J3_z

    def _build_drag_perturbation(self, x, y, z, vx, vy, vz, r, param_start_idx):
        """
        Build symbolic atmospheric drag perturbation.

        Uses exponential atmosphere model and accounts for body rotation.
        Drag parameters (Cd*A, mass) are runtime values extracted from
        a Satellite object and passed via hy.par[] at propagation time.

        Parameters
        ----------
        param_start_idx : int
            Starting index in the hy.par[] array for this perturbation.

        Returns
        -------
        a_drag_x, a_drag_y, a_drag_z : symbolic expressions
            Drag acceleration components.
        param_info : dict
            Parameter mapping with keys 'param_map' and 'description'.
        """
        assert self._atmosphere is not None  # validated in __init__

        rho0    = self._atmosphere.rho0         # kg/m^3
        H       = self._atmosphere.H            # m
        r0      = self._atmosphere.r0           # m
        omega   = self._primary_body.rotation_rate  # rad/s

        # Convert atmospheric params to km (package standard)
        rho0_km = rho0 * 1e9   # kg/m^3 -> kg/km^3
        H_km    = H / 1000.0   # m -> km
        r0_km   = r0 / 1000.0  # m -> km

        # Altitude-dependent density: rho(r) = rho0 * exp(-(r - r0)/H)
        rho = rho0_km * hy.exp(-(r - r0_km) / H_km)

        # Velocity relative to rotating atmosphere
        # v_rel = v_inertial - omega x r
        # For body rotation about z-axis: omega x r = [-omega*y, omega*x, 0]
        vx_rel = vx + omega * y
        vy_rel = vy - omega * x
        vz_rel = vz

        v_rel = hy.sqrt(vx_rel**2 + vy_rel**2 + vz_rel**2)

        # Satellite parameters via runtime hy.par[] array
        Cd_A  = hy.par[param_start_idx]        # Drag coeff * area [m^2]
        mass  = hy.par[param_start_idx + 1]    # Satellite mass [kg]
        Cd_A_km = Cd_A / 1e6                   # m^2 -> km^2

        # a_drag = -(1/2) * rho * (Cd*A/m) * |v_rel| * v_rel_vec
        drag_factor = -0.5 * rho * Cd_A_km / mass * v_rel

        a_drag_x = drag_factor * vx_rel
        a_drag_y = drag_factor * vy_rel
        a_drag_z = drag_factor * vz_rel

        param_info = {
            'param_map': [
                ('Cd_A', param_start_idx),
                ('mass', param_start_idx + 1)
            ],
            'description': {
                'Cd_A': 'Drag coefficient times reference area [m^2]',
                'mass': 'Satellite mass [kg]'
            }
        }
        return a_drag_x, a_drag_y, a_drag_z, param_info

    # ========== STATE CONVERSION ==========
    def _state_to_array(
            self,
            state: OrbitalElements | np.ndarray
    ) -> np.ndarray:
        """
        Convert initial state to a validated 6-element Cartesian float array.

        Parameters
        ----------
        state : OrbitalElements or array-like
            Input state. OrbitalElements are converted to Cartesian.
            CR3BP elements are rejected.

        Returns
        -------
        np.ndarray
            Shape (6,) float array [km, km/s].
        """
        if isinstance(state, OrbitalElements):
            if state.element_type == OEType.CR3BP:
                raise ValueError(
                    "Cannot use CR3BP (nondimensional) elements with a 2-body "
                    "system. Use Cartesian or Keplerian elements instead."
                )
            return state.to_cartesian().elements.astype(float)
        else:
            arr = np.asarray(state, dtype=float)
            if arr.shape != (6,):
                raise ValueError(
                    f"State array must have shape (6,), got {arr.shape}."
                )
            if not np.all(np.isfinite(arr)):
                raise ValueError(
                    f"Initial state contains NaN or Inf values: {arr}"
                )
            return arr

    # ========== PROPERTIES ==========
    @property
    def primary_body(self) -> BodyParams:
        """Central body parameters."""
        return self._primary_body

    @property
    def perturbations(self) -> tuple:
        """Tuple of active perturbation model names."""
        return self._perturbations

    @property
    def atmosphere(self) -> Optional[AtmoParams]:
        """Atmospheric model parameters, or None if drag is not active."""
        return self._atmosphere

    # ========== UTILITY ==========
    def summary(self):
        """Print a human-readable summary of system parameters."""
        print(f"System Type: {self._base_type.value}")
        print(
            f"Primary Body: mu = {self._primary_body.mu:.6e} km^3/s^2, "
            f"R = {self._primary_body.radius:.3f} km"
        )
        if self._perturbations:
            print(f"Perturbations: {', '.join(self._perturbations)}")
            if "J2" in self._perturbations:
                print(f"  J2 = {self._primary_body.J2:.6e}")
            if "J3" in self._perturbations:
                print(f"  J3 = {self._primary_body.J3:.6e}")
            if "drag" in self._perturbations:
                assert self._atmosphere is not None
                print(
                    f"  Atmosphere: rho0 = {self._atmosphere.rho0} kg/m^3, "
                    f"H = {self._atmosphere.H} m"
                )
                print(
                    f"  Rotation: omega = "
                    f"{self._primary_body.rotation_rate:.6e} rad/s"
                )
        else:
            print("Perturbations: None (point mass)")

    def __repr__(self):
        parts = [
            f"System(base_type='2body'",
            f"primary={self._primary_body.mu:.3e} km^3/s^2"
        ]
        if self._perturbations:
            parts.append(f"perturbations={self._perturbations}")
        return ", ".join(parts) + ")"


# ========== CR3BP SYSTEM ==========

class CR3BPSystem(System):
    """
    Circular Restricted 3-Body Problem system in the rotating frame.

    Models spacecraft motion in the co-rotating frame of two massive bodies
    (primaries) using nondimensional units. The primaries move on circular
    orbits about their common barycenter.

    Construct via the System factory -- do not instantiate directly:

        sys = System('3body', primary_body, secondary_body, distance=d)

    Parameters
    ----------
    base_type : str or SysType
        Must be '3body', 'CR3BP', or equivalent. Provided by System factory.
    primary_body : BodyParams
        Parameters for the larger primary (e.g. Earth in Earth-Moon).
    secondary_body : BodyParams
        Parameters for the smaller primary (e.g. Moon in Earth-Moon).
    distance : float
        Distance between primaries [km]. Sets the characteristic length L*.
    compile : bool, optional
        If True (default), compile the Heyoka integrator immediately.

    Attributes
    ----------
    base_type : SysType
        Always SysType.CR3BP.
    primary_body : _BodyParamsWithND
        Primary body parameters, wrapped to expose radius_nd.
    secondary_body : _BodyParamsWithND
        Secondary body parameters, wrapped to expose radius_nd.
    distance : float
        Dimensional distance between primaries [km].
    L_star : float
        Characteristic length [km]. Equal to distance.
    T_star : float
        Characteristic time [s]. T* = sqrt(L*^3 / mu_total).
    mass_ratio : float
        Nondimensional mass ratio mu = mu_2 / (mu_1 + mu_2).
    n_mean : float
        Mean motion [rad/s]. n = sqrt(mu_total / L*^3).
    L1 ... L5 : np.ndarray
        Nondimensional Lagrange point states, shape (6,), read-only.
    lagrange_points : np.ndarray
        All five Lagrange points, shape (5, 3), read-only.

    Notes
    -----
    - All state vectors are nondimensional. Use s2nd/r2nd/v2nd/t2nd to convert
      dimensional inputs before propagating.
    - Perturbations are not supported for CR3BP systems.
    - The rotating frame places the barycenter at the origin with primaries
      on the x-axis: primary at x = -mu, secondary at x = 1 - mu.
    """

    def __init__(
            self,
            base_type,
            primary_body: BodyParams,
            secondary_body: Optional[BodyParams] = None,
            *,
            distance: Optional[float] = None,
            compile: bool | None = None,
            perturbations=None,
            atmosphere=None
    ):
        # Reject 2-body-specific arguments with informative errors
        if perturbations is not None and len(perturbations) > 0:
            raise ValueError(
                "CR3BP systems do not currently support perturbations. "
                f"Got: {perturbations}"
            )
        if atmosphere is not None:
            raise ValueError(
                "CR3BP systems do not support atmosphere models."
            )

        # Validate required arguments
        if secondary_body is None:
            raise ValueError("3-body system requires secondary_body")
        if distance is None:
            raise ValueError("3-body system requires distance between bodies")
        if distance <= 0:
            raise ValueError(f"Distance must be positive, got {distance}")

        if compile is None:
            compile = config.DEFAULT_COMPILE

        self._base_type = SysType.CR3BP
        self._primary_body   = primary_body
        self._secondary_body = secondary_body
        self._distance       = distance

        # Compute nondimensional parameters and Lagrange points
        self._compute_CR3BP_params()
        self._compute_lagrange_points()

        # Integrator cache
        self._cached_eom = None
        self._cached_integrator = None
        self._param_info = None
        self._cached_var_integrator = None
        self._var_order = None
        self._cached_func = None

        # Build EOM and compile
        self._cached_eom, self._param_info = self._build_eom()
        if compile:
            self._compile_integrator()

        # Instance counting
        System._instance_count += 1
        if System._instance_count > config.INSTANCE_WARNING_THRESHOLD:
            warnings.warn(
                f"Created {System._instance_count} System instances. "
                "Each System caches compiled Heyoka integrators, "
                "which can consume significant memory. Consider reusing "
                "System objects when possible.",
                ResourceWarning,
                stacklevel=2
            )

    # ========== COMPILE MESSAGE ==========
    def _compile_message(self) -> str:
        return "Compiling CR3BP integrator"

    # ========== CR3BP PARAMETER COMPUTATION ==========
    def _compute_CR3BP_params(self):
        """Compute nondimensionalization parameters from primary/secondary bodies."""
        self._L_star = self._distance
        mu_total = self._primary_body.mu + self._secondary_body.mu
        self._mass_ratio = self._secondary_body.mu / mu_total
        self._T_star = np.sqrt(self._L_star**3 / mu_total)
        self._n_mean = np.sqrt(mu_total / self._L_star**3)

    def _compute_lagrange_points(self):
        """
        Compute nondimensional Lagrange point locations.

        Uses Brent's method for collinear points (L1, L2, L3) with
        series-expansion initial guesses. Triangular points (L4, L5) are
        computed analytically.

        Notes
        -----
        Series expansions accurate to O(mu^3) from Szebehely (1967).
        All stored arrays are read-only.
        """
        import scipy.optimize as opt

        mu = self._mass_ratio

        def eq_func(x):
            """Equilibrium condition dU/dx = 0 on the x-axis."""
            r1 = abs(x + mu)
            r2 = abs(x - 1 + mu)
            return x - (1 - mu) * (x + mu) / r1**3 - mu * (x - 1 + mu) / r2**3

        # L1: between primaries
        alpha1 = (mu / 3)**(1/3) * (1 - (1/3)*(mu/3)**(1/3) + (1/3)*(mu/3)**(2/3))
        x_L1_expected = 1 - mu - alpha1
        x_L1 = opt.brentq(eq_func, -mu + 1e-6, 1 - mu - 1e-6, xtol=1e-14)
        if abs(x_L1 - x_L1_expected) > 0.1:
            validation_error(
                f"L1 location {x_L1} differs significantly from expected "
                f"value {x_L1_expected}. Possible rootfinding error."
            )

        # L2: beyond secondary
        alpha2 = (mu / 3)**(1/3) * (1 + (1/3)*(mu/3)**(1/3) + (1/3)*(mu/3)**(2/3))
        x_L2_expected = 1 - mu + alpha2
        x_L2 = opt.brentq(eq_func, 1 - mu + 1e-6, 2.0, xtol=1e-14)
        if abs(x_L2 - x_L2_expected) > 0.1:
            validation_error(
                f"L2 location {x_L2} differs significantly from expected "
                f"value {x_L2_expected}. Possible rootfinding error."
            )

        # L3: beyond primary
        x_L3 = opt.brentq(eq_func, -2.0, -mu - 1e-6, xtol=1e-14)

        # L4, L5: equilateral triangles (analytic)
        L4 = np.array([0.5 - mu, np.sqrt(3)/2, 0.0, 0.0, 0.0, 0.0])
        L5 = np.array([0.5 - mu, -np.sqrt(3)/2, 0.0, 0.0, 0.0, 0.0])

        self._L1 = np.array([x_L1, 0.0, 0.0, 0.0, 0.0, 0.0])
        self._L2 = np.array([x_L2, 0.0, 0.0, 0.0, 0.0, 0.0])
        self._L3 = np.array([x_L3, 0.0, 0.0, 0.0, 0.0, 0.0])
        self._L4 = L4
        self._L5 = L5

        for arr in (self._L1, self._L2, self._L3, self._L4, self._L5):
            arr.flags.writeable = False

    # ========== EOM CONSTRUCTION ==========
    def _build_eom(self):
        """
        Build symbolic CR3BP equations of motion in the rotating frame.

        Uses the pseudo-potential U and Heyoka's symbolic differentiation
        to derive acceleration components. All quantities are nondimensional.

        Returns
        -------
        eom : list of (var, rhs) tuples
            Heyoka ODE system definition ready for taylor_adaptive().
        param_info : dict
            Always empty for CR3BP (no runtime parameters).
        """
        x, y, z, vx, vy, vz = hy.make_vars("x", "y", "z", "vx", "vy", "vz")
        mu = self._mass_ratio

        # Distances to primaries
        r1 = hy.sqrt((x + mu)**2 + y**2 + z**2)
        r2 = hy.sqrt((x - 1.0 + mu)**2 + y**2 + z**2)

        # Pseudo-potential (includes centrifugal term)
        U = 0.5 * (x**2 + y**2) + (1.0 - mu) / r1 + mu / r2

        # Equations of motion in rotating frame
        eom = [
            (x, vx),
            (y, vy),
            (z, vz),
            (vx, 2.0 * vy + hy.diff(U, x)),
            (vy, -2.0 * vx + hy.diff(U, y)),
            (vz, hy.diff(U, z))
        ]

        # CR3BP has no runtime parameters
        param_info = {'param_map': [], 'description': {}}
        return eom, param_info

    # ========== STATE CONVERSION ==========
    def _state_to_array(
            self,
            state: OrbitalElements | np.ndarray
    ) -> np.ndarray:
        """
        Convert initial state to a validated 6-element nondimensional array.

        Parameters
        ----------
        state : OrbitalElements or array-like
            Input state. Must use OEType.CR3BP if OrbitalElements.

        Returns
        -------
        np.ndarray
            Shape (6,) float array [nondimensional].
        """
        if isinstance(state, OrbitalElements):
            if state.element_type != OEType.CR3BP:
                raise ValueError(
                    "CR3BP system requires CR3BP (nondimensional) elements. "
                    f"Got {state.element_type.value}. "
                    "Convert to nondimensional coordinates first."
                )
            return state.elements.astype(float)
        else:
            arr = np.asarray(state, dtype=float)
            if arr.shape != (6,):
                raise ValueError(
                    f"State array must have shape (6,), got {arr.shape}."
                )
            if not np.all(np.isfinite(arr)):
                raise ValueError(
                    f"Initial state contains NaN or Inf values: {arr}"
                )
            return arr

    # ========== PROPERTIES ==========
    @property
    def primary_body(self) -> _BodyParamsWithND:
        """Primary body parameters with nondimensional radius support."""
        return _BodyParamsWithND(self._primary_body, self._L_star)

    @property
    def secondary_body(self) -> _BodyParamsWithND:
        """Secondary body parameters with nondimensional radius support."""
        return _BodyParamsWithND(self._secondary_body, self._L_star)

    @property
    def distance(self) -> float:
        """Dimensional distance between primaries [km]."""
        return self._distance

    @property
    def L_star(self) -> float:
        """Characteristic length [km]. Equal to distance."""
        return self._L_star

    @property
    def T_star(self) -> float:
        """Characteristic time [s]. T* = sqrt(L*^3 / mu_total)."""
        return self._T_star

    @property
    def mass_ratio(self) -> float:
        """Nondimensional mass ratio mu = mu_2 / (mu_1 + mu_2)."""
        return self._mass_ratio

    @property
    def n_mean(self) -> float:
        """Mean motion [rad/s]. n = sqrt(mu_total / L*^3)."""
        return self._n_mean

    @property
    def L1(self) -> np.ndarray:
        """Nondimensional L1 Lagrange point state [x, y, z, vx, vy, vz]."""
        return self._L1

    @property
    def L2(self) -> np.ndarray:
        """Nondimensional L2 Lagrange point state [x, y, z, vx, vy, vz]."""
        return self._L2

    @property
    def L3(self) -> np.ndarray:
        """Nondimensional L3 Lagrange point state [x, y, z, vx, vy, vz]."""
        return self._L3

    @property
    def L4(self) -> np.ndarray:
        """Nondimensional L4 Lagrange point state [x, y, z, vx, vy, vz]."""
        return self._L4

    @property
    def L5(self) -> np.ndarray:
        """Nondimensional L5 Lagrange point state [x, y, z, vx, vy, vz]."""
        return self._L5

    @property
    def lagrange_points(self) -> np.ndarray:
        """
        All five Lagrange points as a (5, 3) position array [nondimensional].

        Returns
        -------
        np.ndarray
            Shape (5, 3) with rows [L1, L2, L3, L4, L5].
            Each row is [x, y, z, vx, vy, vz] in nondimensional coordinates. Read-only.
        """
        pts = np.vstack([
            self._L1[:3], self._L2[:3], self._L3[:3],
            self._L4[:3], self._L5[:3]
        ])
        pts.flags.writeable = False
        return pts

    # ========== NONDIMENSIONALIZATION ==========
    def r2nd(self, r):
        """
        Convert dimensional position to nondimensional.

        Parameters
        ----------
        r : float or array-like
            Dimensional position [km].

        Returns
        -------
        np.float64 or np.ndarray
            Nondimensional position.
        """
        return np.asarray(r) / self._L_star

    def r2d(self, r_nd):
        """
        Convert nondimensional position to dimensional.

        Parameters
        ----------
        r_nd : float or array-like
            Nondimensional position.

        Returns
        -------
        np.float64 or np.ndarray
            Dimensional position [km].
        """
        return np.asarray(r_nd) * self._L_star

    def v2nd(self, v):
        """
        Convert dimensional velocity to nondimensional.

        Parameters
        ----------
        v : float or array-like
            Dimensional velocity [km/s].

        Returns
        -------
        np.float64 or np.ndarray
            Nondimensional velocity. v_nd = v * T* / L*
        """
        return np.asarray(v) * self._T_star / self._L_star

    def v2d(self, v_nd):
        """
        Convert nondimensional velocity to dimensional.

        Parameters
        ----------
        v_nd : float or array-like
            Nondimensional velocity.

        Returns
        -------
        np.float64 or np.ndarray
            Dimensional velocity [km/s]. v = v_nd * L* / T*
        """
        return np.asarray(v_nd) * self._L_star / self._T_star
    
    def s2nd(self, s):
        """
        Convert a dimensional 6-state to nondimensional.

        Applies r2nd to the position components and v2nd to the velocity
        components. Input may be a single state (6,) or a batch (N, 6).

        Parameters
        ----------
        s : array-like, shape (6,) or (N, 6)
            Dimensional state [km, km/s].

        Returns
        -------
        np.ndarray
            Nondimensional state, same shape as input.
        """
        s = np.asarray(s, dtype=float)
        out = s.copy()
        out[..., :3] = s[..., :3] / self._L_star
        out[..., 3:] = s[..., 3:] * self._T_star / self._L_star
        return out

    def s2d(self, s_nd):
        """
        Convert a nondimensional 6-state to dimensional.

        Applies r2d to the position components and v2d to the velocity
        components. Input may be a single state (6,) or a batch (N, 6).

        Parameters
        ----------
        s_nd : array-like, shape (6,) or (N, 6)
            Nondimensional state.

        Returns
        -------
        np.ndarray
            Dimensional state [km, km/s], same shape as input.
        """
        s_nd = np.asarray(s_nd, dtype=float)
        out = s_nd.copy()
        out[..., :3] = s_nd[..., :3] * self._L_star
        out[..., 3:] = s_nd[..., 3:] * self._L_star / self._T_star
        return out

    def t2nd(self, t):
        """
        Convert dimensional time to nondimensional.

        Parameters
        ----------
        t : float or array-like
            Dimensional time [s].

        Returns
        -------
        np.float64 or np.ndarray
            Nondimensional time.
        """
        return np.asarray(t) / self._T_star

    def t2d(self, t_nd):
        """
        Convert nondimensional time to dimensional.

        Parameters
        ----------
        t_nd : float or array-like
            Nondimensional time.

        Returns
        -------
        np.float64 or np.ndarray
            Dimensional time [s].
        """
        return np.asarray(t_nd) * self._T_star

    # ========== UTILITY ==========
    def summary(self):
        """Print a human-readable summary of system parameters."""
        print(f"System Type: {self._base_type.value}")
        print(
            f"Primary Body:   mu = {self._primary_body.mu:.6e} km^3/s^2, "
            f"R = {self._primary_body.radius:.3f} km"
        )
        print(
            f"Secondary Body: mu = {self._secondary_body.mu:.6e} km^3/s^2, "
            f"R = {self._secondary_body.radius:.3f} km"
        )
        print(f"Distance: {self._distance:.3f} km")
        print(f"\nNondimensional Parameters:")
        print(f"  L* = {self._L_star:.6e} km")
        print(f"  T* = {self._T_star:.6e} s")
        print(f"  mu = {self._mass_ratio:.10f}")
        print(f"  n  = {self._n_mean:.6e} rad/s")

    def __repr__(self):
        return (
            f"System(base_type='3body', "
            f"mu_1 ={self._primary_body.mu:.3e}, "
            f"mu_2 ={self._secondary_body.mu:.3e}, "
            f"L* ={self._L_star:.3e} km, "
            f"mu ={self._mass_ratio:.6f})"
        )
