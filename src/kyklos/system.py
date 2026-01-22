'''Development code for an orbital trajectory handling package
System class definition
created with the assistance of Claude Sonnet 4.5 by Anthropic'''

import numpy as np
import warnings
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Any, Union
from enum import Enum
import heyoka as hy
from .orbital_elements import OrbitalElements, OEType
from .trajectory import Trajectory
from .satellite import Satellite

# define an enumerated list of system types
class SysType(Enum):
    TWO_BODY = '2body'
    CR3BP = '3body'
    N_BODY = 'Nbody'
"""
Core dataclasses for System class components.
This module defines immutable dataclasses for celestial body parameters
and atmospheric models.  Other models may be added later.
"""
@dataclass(frozen=True)
class BodyParams:
    """
    Immutable parameters for a celestial body.
    
    Attributes
    ----------
    mu : float
        Gravitational parameter [kmÂ³/sÂ²]
    radius : float
        Equatorial radius [km]
    J2 : float, optional
        J2 zonal harmonic coefficient [dimensionless]
        Required if J2 perturbations are enabled
    rotation_rate : float, optional
        Angular rotation rate [rad/s]
        Required if atmospheric drag is enabled
    """
    mu: float
    radius: float
    J2: Optional[float] = None
    rotation_rate: Optional[float] = None
    name: Optional[str] = None
    
    def __post_init__(self):
        #Validate parameters
        if self.mu <= 0:
            raise ValueError(f"Gravitational parameter must be positive, got {self.mu}")
        if self.radius <= 0:
            raise ValueError(f"Radius must be positive, got {self.radius}")
        if self.J2 is not None and abs(self.J2) > 1:
            raise ValueError(f"J2 coefficient seems unrealistic: {self.J2}")

@dataclass(frozen=True)
class AtmoParams:
    """
    Immutable parameters for exponential atmosphere model.
    
    The density profile follows: Ï(r) = Ïâ‚€ * exp(-(r - râ‚€)/H)
    
    Attributes
    ----------
    rho0 : float
        Reference density at reference altitude [kg/mÂ³]
    H : float
        Scale height [m]
    r0 : float
        Reference radius (altitude where Ïâ‚€ is defined) [m]
    
    Notes
    -----
    This is a simple exponential model suitable for preliminary analysis.
    More sophisticated models can be added later as separate classes 
    or via callback functions.
    """
    rho0: float
    H: float
    r0: float
    
    def __post_init__(self):
        """Validate parameters."""
        if self.rho0 <= 0:
            raise ValueError(f"Reference density must be positive, got {self.rho0}")
        if self.H <= 0:
            raise ValueError(f"Scale height must be positive, got {self.H}")
        if self.r0 <= 0:
            raise ValueError(f"Reference radius must be positive, got {self.r0}")

class _BodyParamsWithND:
        """
        Wrapper for BodyParams that adds nondimensional radius property.
        Used internally by System to provide .radius_nd access for CR3BP systems.
        """
        def __init__(self, body_params: BodyParams, L_star: float):
            self._body_params = body_params
            self._L_star = L_star
        
        @property
        def radius_nd(self) -> float:
            """Nondimensional radius [L_star units]"""
            return self._body_params.radius / self._L_star
        
        # Delegate all other attributes to the underlying BodyParams
        def __getattr__(self, name):
            return getattr(self._body_params, name)
        
        def __repr__(self):
            return repr(self._body_params)
        
        # need to preserve equality through the wrapper
        def __eq__(self, other):
            """Compare based on wrapped BodyParams."""
            if isinstance(other, _BodyParamsWithND):
                return self._body_params == other._body_params
            elif isinstance(other, BodyParams):
                return self._body_params == other
            return NotImplemented
        
        def __hash__(self):
            """Hash based on wrapped BodyParams."""
            return hash(self._body_params)

class System:
    """
    Immutable system definition for orbital propagation.
    
    Represents a gravitational environment with a primary body and optional
    secondary body (for 3-body problems), along with perturbation models.
    
    Parameters
    ----------
    base_type : str
        Type of base dynamics: "2body" or "3body"
    primary_body : BodyParams
        Parameters for the primary celestial body
    secondary_body : BodyParams, optional
        Parameters for secondary body (required for 3-body systems)
    perturbations : tuple of str, optional
        Perturbation models to include. Options: "J2", "drag"
        Default is empty tuple (point mass dynamics)
        Note: Use trailing comma for single perturbation: ('J2',)
    atmosphere : AtmosphereParams, optional
        Atmospheric model parameters (required if "drag" in perturbations)
    distance : float, optional
        Distance between primary and secondary bodies [km]
        Required for 3-body systems
        
    Attributes (computed for 3-body)
    ---------------------------------
    L_star : float
        Characteristic length [km]
    T_star : float  
        Characteristic time [s]
    mass_ratio : float
        Nondimensional mass ratio mu = mu_2/(mu_1 + mu_2)
    n_mean : float
        Mean motion [rad/s]

    Notes
    -----
    - System is immutable - create a new instance to change parameters
    - 3-body systems currently do not support perturbations
    - Default length unit is km (except AtmosphereParams uses m)
    - Instance counting: Warning issued when more than 10 Systems exist
      simultaneously (due to Heyoka compilation overhead)
    """
    # ========== CLASS CONSTANTS ==========
    # Class variable for instance counting
    _instance_count = 0
    _instance_warning_threshold = 10
    # currently implemented perturbations
    _VALID_PERTURBATIONS = frozenset(("J2", "drag"))

    # ========== CONSTRUCTION ==========
    def __init__(
        self,
        base_type,
        primary_body: BodyParams,
        secondary_body: Optional[BodyParams] = None,
        perturbations: tuple = (),
        atmosphere: Optional[AtmoParams] = None,
        distance: Optional[float] = None,
        compile: bool = True  # Default: compile integrator immediately
    ):
        """
        Initialize System with validation and computed parameters.
        
        Raises
        ------
        ValueError
            If parameters are invalid or incompatible
        """
        # Validate before storing
        self._validate_params(
            base_type, primary_body, secondary_body,
            perturbations, atmosphere, distance
        )
        
        # Store parameters in private attributes for immutability
        self._primary_body = primary_body
        self._secondary_body = secondary_body
        self._perturbations = tuple(perturbations)  # Ensure tuple (immutable)
        self._atmosphere = atmosphere
        self._distance = distance
        
        # Compute 3-body parameters if needed
        if self._base_type == SysType.CR3BP:
            self._compute_CR3BP_params()
            self._compute_lagrange_points()
        else:
            self._L_star = None
            self._T_star = None
            self._mass_ratio = None
            self._n_mean = None
            self._L1 = None
            self._L2 = None
            self._L3 = None
            self._L4 = None
            self._L5 = None
        
        # Initialize cached EOMs and heyoka integrator
        self._cached_eom = None
        self._cached_integrator = None
        self._param_info = None

        # Build symbolic EOM
        self._cached_eom, self._param_info = self._build_eom()
        
        # Compile if requested
        if compile:
            self._compile_integrator()
        
        # Instance counting
        System._instance_count += 1
        if System._instance_count > System._instance_warning_threshold:
            warnings.warn(
                f"Created {System._instance_count} System instances. "
                f"Each System will cache compiled Heyoka integrators, "
                f"which can consume significant memory. Consider reusing "
                f"System objects when possible.",
                ResourceWarning,
                stacklevel=2
            )
    
    # ========== VALIDATION ==========
    def _validate_params(
        self,
        base_type,
        primary_body: BodyParams,
        secondary_body: Optional[BodyParams],
        perturbations: tuple,
        atmosphere: Optional[AtmoParams],
        distance: Optional[float]
    ):
        """
        Validate system parameters for consistency.
        
        Raises
        ------
        ValueError
            If parameters are invalid or incompatible
        """
        # Validate base_type and mark immutable
        self._base_type = self._parse_base_type(base_type)
        
        # Validate perturbations
        for pert in perturbations:
            if pert not in System._VALID_PERTURBATIONS:
                raise ValueError(
                    f"Unknown perturbation '{pert}'. "
                    f"Valid options: {System._VALID_PERTURBATIONS}"
                )
        
        # Check for duplicate perturbations
        if len(perturbations) != len(set(perturbations)):
            raise ValueError(f"Duplicate perturbations found: {perturbations}")
        
        # 3-body specific validation
        if self._base_type == SysType.CR3BP:
            if secondary_body is None:
                raise ValueError("3-body system requires secondary_body")
            if distance is None:
                raise ValueError("3-body system requires distance between bodies")
            if distance <= 0:
                raise ValueError(f"Distance must be positive, got {distance}")
            if perturbations:
                raise ValueError(
                    f"3-body systems do not currently support perturbations. "
                    f"Got: {perturbations}"
                )
            if atmosphere is not None:
                warnings.warn(
                    "Atmosphere specified for 3-body system but will be ignored"
                )
        
        # 2-body specific validation
        if self._base_type == SysType.TWO_BODY:
            if secondary_body is not None:
                warnings.warn(
                    "secondary_body specified for 2-body system but will be ignored"
                )
            if distance is not None:
                warnings.warn(
                    "distance specified for 2-body system but will be ignored"
                )
        
        # Perturbation-specific validation
        if "J2" in perturbations:
            if primary_body.J2 is None:
                raise ValueError(
                    "J2 perturbation requested but primary_body.J2 is None"
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
    
    # ========== PROPAGATION ==========
    def propagate(
        self, 
        initial_state: "OrbitalElements | np.ndarray", 
        t_start: float, 
        t_end: float, 
        satellite: Optional[Satellite] = None
    ) -> "Trajectory":
        """
        Propagate from t_start to t_end with dense output.
        
        Uses Heyoka's continuous output to store Taylor series coefficients,
        enabling high-accuracy state evaluation at any time in [t_start, t_end]
        without re-integration.
        
        Parameters
        ----------
        initial_state : OrbitalElements or array_like
            Initial state vector [x, y, z, vx, vy, vz] in km, km/s
        t_start : float
            Start time [s]
        t_end : float
            End time [s]
        satellite : Satellite, optional
            Satellite object containing physical properties (mass, drag 
            coefficient, cross-sectional area, inertia tensor). Required 
            if system has perturbations that depend on satellite properties
            (e.g., drag). For systems without perturbations, can be None.
        
        Returns
        -------
        Trajectory
            Trajectory object containing dense output
        """
        # Ensure compiled
        if not self.is_compiled:
            self._compile_integrator()
        # Get integrator
        ta = self._cached_integrator
        # Assert for type checker (should always be true after compile)
        assert ta is not None, "Integrator should be compiled"

        # cast input times explicitly to floats (required by heyoka)
        t_start = float(t_start)
        t_end = float(t_end)

        # validate and condition input state to produce initial state vector
        if isinstance(initial_state, OrbitalElements):
            # Check for type mismatch between system and elements
            if self._base_type == SysType.CR3BP:
                # CR3BP systems require CR3BP elements
                if initial_state.element_type != OEType.CR3BP:
                    raise ValueError(
                        f"CR3BP system requires CR3BP (nondimensional) elements. "
                        f"Got {initial_state.element_type.value}. "
                        f"Convert to nondimensional coordinates first."
                    )
                state_array = initial_state.elements
            else:
                # 2-body systems should not get CR3BP elements
                if initial_state.element_type == OEType.CR3BP:
                    raise ValueError(
                        f"Cannot use CR3BP (nondimensional) elements with "
                        f"{self._base_type.value} system. "
                        f"Use Cartesian or Keplerian elements instead."
                    )
                # Convert to Cartesian for propagation
                cart_state = initial_state.to_cartesian()
                state_array = cart_state.elements
        else:
            # Raw array input
            state_array = np.asarray(initial_state)
            # Validate array input (OrbitalElements already validated)
            if not np.all(np.isfinite(state_array)):
                raise ValueError(
                    f"Initial state contains NaN or Inf values: {state_array}"
                )
        
        # Handle satellite parameters
        if satellite is not None:
            params_array = self._process_satellite(satellite)
            ta.pars[:] = params_array
        elif len(self._param_info['param_map']) > 0:
            raise ValueError(
                f"This system requires a Satellite object. "
                f"The following properties are needed: "
                f"{[name for name, _ in self._param_info['param_map']]}"
            )
        
        # Set initial conditions
        ta.time = t_start
        ta.state[:] = state_array
        
        # Propagate until ending time
        traj = ta.propagate_until(t_end, c_output=True)[4]

        # Check for integration failure
        if not np.all(np.isfinite(ta.state)):
            raise ValueError(
                f"Integration failed: state became invalid during propagation.\n"
                f"Initial state: {state_array}\n"
                f"Final time: {ta.time}\n"
                f"Final state: {ta.state}\n"
                f"Likely causes:\n"
                f"  - Initial position too close to central body\n"
                f"  - Collision with central body during propagation\n"
                f"  - Numerical instability in perturbation models"
            )
        
        # Check that continuous output object produces a valid trajectory
        if traj is None:
            raise ValueError(
                f"Integration produced no continuous output (c_output is None).\n"
                f"This may indicate a severe integration failure."
            )

        return Trajectory(self,traj,t_start,t_end)

    def _process_satellite(self, satellite: Satellite) -> np.ndarray:
        """
        Extract parameters from Satellite object for Heyoka integrator.
        
        Parameters
        ----------
        satellite : Satellite
            Satellite object containing required physical properties
            
        Returns
        -------
        np.ndarray
            Array of parameter values in order expected by integrator,
            as defined by self._param_info['param_map']
            
        Raises
        ------
        ValueError
            If satellite is missing a required property
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
                    f"This may indicate a bug in the System implementation."
                )
        return np.array(params)

    # ========== NONDIMENSIONALIZATION METHODS ==========
    def r2nd(self, r):
        """
        Convert dimensional position to nondimensional (CR3BP only).
        
        Parameters
        ----------
        r : float or array-like
            Dimensional position [km]
            
        Returns
        -------
        np.float64 or np.ndarray
            Nondimensional position (scalar if input is scalar, array otherwise)
            
        Raises
        ------
        ValueError
            If system is not CR3BP type
        """
        if self._base_type != SysType.CR3BP:
            raise ValueError(
                "Nondimensionalization only available for CR3BP systems. "
                f"Current system type: {self._base_type.value}"
            )
        assert self._L_star is not None  # Type checker hint
        return np.asarray(r) / self._L_star
    
    def r2d(self, r_nd):
        """
        Convert nondimensional position to dimensional (CR3BP only).
        
        Parameters
        ----------
        r_nd : float or array-like
            Nondimensional position
            
        Returns
        -------
        np.float64 or np.ndarray
            Dimensional position [km] (scalar if input is scalar, array otherwise)
            
        Raises
        ------
        ValueError
            If system is not CR3BP type
        """
        if self._base_type != SysType.CR3BP:
            raise ValueError(
                "Redimensionalization only available for CR3BP systems. "
                f"Current system type: {self._base_type.value}"
            )
        assert self._L_star is not None  # Type checker hint
        return np.asarray(r_nd) * self._L_star
    
    def v2nd(self, v):
        """
        Convert dimensional velocity to nondimensional (CR3BP only).
        
        Parameters
        ----------
        v : float or array-like
            Dimensional velocity [km/s]
            
        Returns
        -------
        np.float64 or np.ndarray
            Nondimensional velocity (scalar if input is scalar, array otherwise)
            
        Raises
        ------
        ValueError
            If system is not CR3BP type
        """
        if self._base_type != SysType.CR3BP:
            raise ValueError(
                "Nondimensionalization only available for CR3BP systems. "
                f"Current system type: {self._base_type.value}"
            )
        # Type checker hint: these are guaranteed non-None for CR3BP
        assert self._L_star is not None and self._T_star is not None
        # v_nd = v * T_star / L_star
        return np.asarray(v) * self._T_star / self._L_star
    
    def v2d(self, v_nd):
        """
        Convert nondimensional velocity to dimensional (CR3BP only).
        
        Parameters
        ----------
        v_nd : float or array-like
            Nondimensional velocity
            
        Returns
        -------
        np.float64 or np.ndarray
            Dimensional velocity [km/s] (scalar if input is scalar, array otherwise)
            
        Raises
        ------
        ValueError
            If system is not CR3BP type
        """
        if self._base_type != SysType.CR3BP:
            raise ValueError(
                "Redimensionalization only available for CR3BP systems. "
                f"Current system type: {self._base_type.value}"
            )
        # Type checker hint: these are guaranteed non-None for CR3BP
        assert self._L_star is not None and self._T_star is not None
        # v = v_nd * L_star / T_star
        return np.asarray(v_nd) * self._L_star / self._T_star
    
    def t2nd(self, t):
        """
        Convert dimensional time to nondimensional (CR3BP only).
        
        Parameters
        ----------
        t : float or array-like
            Dimensional time [s]
            
        Returns
        -------
        np.float64 or np.ndarray
            Nondimensional time (scalar if input is scalar, array otherwise)
            
        Raises
        ------
        ValueError
            If system is not CR3BP type
        """
        if self._base_type != SysType.CR3BP:
            raise ValueError(
                "Nondimensionalization only available for CR3BP systems. "
                f"Current system type: {self._base_type.value}"
            )
        assert self._T_star is not None  # Type checker hint
        return np.asarray(t) / self._T_star
    
    def t2d(self, t_nd):
        """
        Convert nondimensional time to dimensional (CR3BP only).
        
        Parameters
        ----------
        t_nd : float or array-like
            Nondimensional time
            
        Returns
        -------
        np.float64 or np.ndarray
            Dimensional time [s] (scalar if input is scalar, array otherwise)
            
        Raises
        ------
        ValueError
            If system is not CR3BP type
        """
        if self._base_type != SysType.CR3BP:
            raise ValueError(
                "Redimensionalization only available for CR3BP systems. "
                f"Current system type: {self._base_type.value}"
            )
        assert self._T_star is not None  # Type checker hint
        return np.asarray(t_nd) * self._T_star

    # ========== PROPERTY ACCESS ==========
    def summary(self):
        """Print detailed summary of system parameters."""
        print(f"System Type: {self._base_type}")
        print(f"Primary Body: Î¼ = {self._primary_body.mu:.6e} kmÂ³/sÂ², "
              f"R = {self._primary_body.radius:.3f} km")
        
        if self._base_type == SysType.CR3BP:
            print(f"Secondary Body: Î¼ = {self._secondary_body.mu:.6e} kmÂ³/sÂ², "
                  f"R = {self._secondary_body.radius:.3f} km")
            print(f"Distance: {self._distance:.3f} km")
            print(f"\nNondimensional Parameters:")
            print(f"  L* = {self._L_star:.6e} km")
            print(f"  T* = {self._T_star:.6e} s")
            print(f"  Î¼  = {self._mass_ratio:.10f}")
            print(f"  n  = {self._n_mean:.6e} rad/s")
        
        if self._perturbations:
            print(f"\nPerturbations: {', '.join(self._perturbations)}")
            
            if "J2" in self._perturbations:
                print(f"  Jâ‚‚ = {self._primary_body.J2:.6e}")
            
            if "drag" in self._perturbations:
                print(f"  Atmosphere: Ïâ‚€ = {self._atmosphere.rho0} kg/mÂ³, "
                      f"H = {self._atmosphere.H} m")
                print(f"  Rotation: Ï‰ = {self._primary_body.rotation_rate:.6e} rad/s")
        else:
            print("\nPerturbations: None (point mass)")
    
    # Interface with System instance counting
    @classmethod
    def get_instance_count(cls):
        """Get current number of System instances."""
        return cls._instance_count
    
    @classmethod
    def reset_instance_count(cls):
        """Reset instance counter (useful for testing)."""
        cls._instance_count = 0
    
    # Immutability via read-only properties
    @property
    def base_type(self) -> SysType:
        """Type of base dynamics."""
        return self._base_type
    
    @property
    def primary_body(self) -> BodyParams | _BodyParamsWithND:
        """Primary celestial body parameters."""
        # For CR3BP, wrap with nondimensional radius support
        if self._base_type == SysType.CR3BP:
            assert self._L_star is not None #always true for CR3BP system
            return _BodyParamsWithND(self._primary_body, self._L_star)
        return self._primary_body

    @property
    def secondary_body(self) -> BodyParams | _BodyParamsWithND | None:
        """Secondary celestial body parameters."""
        # For CR3BP, wrap with nondimensional radius support
        if self._base_type == SysType.CR3BP and self._secondary_body is not None:
            assert self._L_star is not None #always true for CR3BP system
            return _BodyParamsWithND(self._secondary_body, self._L_star)
        return self._secondary_body
    
    @property
    def perturbations(self) -> tuple:
        """Tuple of perturbation model names."""
        return self._perturbations
    
    @property
    def atmosphere(self) -> Optional[AtmoParams]:
        """Atmospheric model parameters (if drag enabled)."""
        return self._atmosphere
       
    @property
    def distance(self) -> Optional[float]:
        """Distance between primary and secondary [km]."""
        return self._distance
    
    @property
    def L_star(self) -> Optional[float]:
        """Characteristic length [km]."""
        return self._L_star
    
    @property
    def T_star(self) -> Optional[float]:
        """Characteristic time [s]."""
        return self._T_star
    
    @property
    def mass_ratio(self) -> Optional[float]:
        """Nondimensional mass ratio mu= mu_2/(mu_1 + mu_2)."""
        return self._mass_ratio
    
    @property
    def n_mean(self) -> Optional[float]:
        """Mean motion [rad/s]."""
        return self._n_mean
    
    @property
    def is_compiled(self) -> bool:
        """Check if integrator has been compiled."""
        return self._cached_integrator is not None
    
    @property
    def cached_eom(self) -> Optional[List[Tuple]]:
        """Cached set of symbolic equations of motion"""
        return self._cached_eom
    
    @property
    def requires_satellite(self) -> bool:
        """Check if this system requires satellite properties."""
        return len(self._param_info['param_map']) > 0
    
    @property
    def L1(self) -> np.ndarray:
        """Nondimensional L1 Lagrange point location [x, y, z]."""
        if self._base_type != SysType.CR3BP:
            raise ValueError(
                "Lagrange points only available for CR3BP systems. "
                f"Current system type: {self._base_type.value}"
            )
        return self._L1 #type: ignore

    @property
    def L2(self) -> np.ndarray:
        """Nondimensional L2 Lagrange point location [x, y, z]."""
        if self._base_type != SysType.CR3BP:
            raise ValueError(
                "Lagrange points only available for CR3BP systems. "
                f"Current system type: {self._base_type.value}"
            )
        return self._L2 #type: ignore

    @property
    def L3(self) -> np.ndarray:
        """Nondimensional L3 Lagrange point location [x, y, z]."""
        if self._base_type != SysType.CR3BP:
            raise ValueError(
                "Lagrange points only available for CR3BP systems. "
                f"Current system type: {self._base_type.value}"
            )
        return self._L3 #type: ignore

    @property
    def L4(self) -> np.ndarray:
        """Nondimensional L4 Lagrange point location [x, y, z]."""
        if self._base_type != SysType.CR3BP:
            raise ValueError(
                "Lagrange points only available for CR3BP systems. "
                f"Current system type: {self._base_type.value}"
            )
        return self._L4 #type: ignore

    @property
    def L5(self) -> np.ndarray:
        """Nondimensional L5 Lagrange point location [x, y, z]."""
        if self._base_type != SysType.CR3BP:
            raise ValueError(
                "Lagrange points only available for CR3BP systems. "
                f"Current system type: {self._base_type.value}"
            )
        return self._L5 #type: ignore

    @property
    def lagrange_points(self) -> np.ndarray:
        """
        All five Lagrange points as (5, 3) array [nondimensional].
        
        Returns
        -------
        np.ndarray
            Shape (5, 3) with rows [L1, L2, L3, L4, L5]
            Each row is [x, y, z] in nondimensional coordinates
        
        Raises
        ------
        ValueError
            If system is not CR3BP type
        """
        if self._base_type != SysType.CR3BP:
            raise ValueError(
                "Lagrange points only available for CR3BP systems. "
                f"Current system type: {self._base_type.value}"
            )
        
        # Stack into (5, 3) array
        points = np.vstack([self._L1, self._L2, self._L3, self._L4, self._L5]) #type: ignore
        points.flags.writeable = False  # Read-only
        return points

    # ========== UTILITY METHODS ==========
    def _compute_CR3BP_params(self):
        """Compute nondimensionalization parameters for CR3BP."""
        # Characteristic length (distance between primaries)
        self._L_star = self._distance
        
        # Total gravitational parameter [kmÂ³/sÂ²]
        mu_total = self._primary_body.mu + self._secondary_body.mu
        
        # Mass ratio: Î¼ = mâ‚‚/(mâ‚+mâ‚‚) = Î¼â‚‚/(Î¼â‚+Î¼â‚‚)
        self._mass_ratio = self._secondary_body.mu / mu_total
        
        # Characteristic time: T* = sqrt(L*Â³/Î¼_total)
        self._T_star = np.sqrt(self._L_star**3 / mu_total)
        
        # Mean motion: n = 2Ï€/T* = sqrt(Î¼_total/L*Â³)
        self._n_mean = np.sqrt(mu_total / self._L_star**3)

    def _compute_lagrange_points(self):
        """
        Compute nondimensional Lagrange point locations for CR3BP.
        
        Uses Brent's method for collinear points (L1, L2, L3) with series
        expansion initial guesses. Triangular points (L4, L5) computed analytically.
        
        Notes
        -----
        Series expansions accurate to O(mu^3) from Szebehely (1967).
        """
        import scipy.optimize as opt
        
        mu = self._mass_ratio
        assert mu is not None   #this is only called for CR3BP systems
        
        # Equilibrium condition on x-axis: dU/dx = 0
        def eq_func(x):
            """Equilibrium function for collinear points."""
            r1 = abs(x + mu)
            r2 = abs(x - 1 + mu)
            return x - (1-mu)*(x+mu)/r1**3 - mu*(x-1+mu)/r2**3
        
        # L1: between primaries
        # Series expansion for initial guess
        alpha1 = (mu/3)**(1/3) * (1 - (1/3)*(mu/3)**(1/3) + (1/3)*(mu/3)**(2/3))
        L1_a, L1_b = -mu + 1e-6, 1 - mu - 1e-6
        x_L1 = opt.brentq(eq_func, L1_a, L1_b, xtol=1e-14)
        
        # L2: beyond secondary
        alpha2 = (mu/3)**(1/3) * (1 + (1/3)*(mu/3)**(1/3) + (1/3)*(mu/3)**(2/3))
        L2_a, L2_b = 1 - mu + 1e-6, 2.0
        x_L2 = opt.brentq(eq_func, L2_a, L2_b, xtol=1e-14)
        
        # L3: beyond primary
        L3_a, L3_b = -2.0, -mu - 1e-6
        x_L3 = opt.brentq(eq_func, L3_a, L3_b, xtol=1e-14)
        
        # L4, L5: equilateral triangles (analytic)
        L4 = np.array([0.5 - mu, np.sqrt(3)/2, 0.0])
        L5 = np.array([0.5 - mu, -np.sqrt(3)/2, 0.0])
        
        # Store as nondimensional arrays (read-only)
        self._L1 = np.array([x_L1, 0.0, 0.0])
        self._L2 = np.array([x_L2, 0.0, 0.0])
        self._L3 = np.array([x_L3, 0.0, 0.0])
        self._L4 = L4
        self._L5 = L5
        
        # Make arrays read-only
        self._L1.flags.writeable = False
        self._L2.flags.writeable = False
        self._L3.flags.writeable = False
        self._L4.flags.writeable = False
        self._L5.flags.writeable = False
    
    def _build_eom(self):
        """
        Build symbolic Heyoka equations of motion for this system.
        
        Creates symbolic expressions for the equations of motion based on
        the system's base dynamics (2-body or CR3BP) and any perturbations
        (J2, drag).
        
        Returns
        -------
        sys : list of (var, rhs) tuples
            Heyoka ODE system definition ready for taylor_adaptive()
        param_info : dict
            Information about runtime parameters:
            - 'param_map': list of (name, index) tuples for hy.par[] array
            - 'description': dict with human-readable parameter descriptions
        
        Notes
        -----
        - System parameters (Î¼, J2, atmospheric params) are hardcoded into
        the symbolic expressions since they're part of the immutable System
        - Satellite parameters (Cd*A, mass) use hy.par[] for runtime binding,
        allowing one System to propagate multiple satellites
        - State vector order: [x, y, z, vx, vy, vz] (km, km/s)
        """
        # Create symbolic state variables
        x, y, z, vx, vy, vz = hy.make_vars("x", "y", "z", "vx", "vy", "vz")
        
        # Build equations based on system type
        if self._base_type == SysType.TWO_BODY:
            sys, param_info = self._build_2body_eom(x, y, z, vx, vy, vz)
        elif self._base_type == SysType.CR3BP:
            sys, param_info = self._build_cr3bp_eom(x, y, z, vx, vy, vz)
        else:
            raise NotImplementedError(
                f"build_eom() not yet implemented for {self._base_type}"
            )
        
        return sys, param_info

    def _build_2body_eom(self, x, y, z, vx, vy, vz):
        """Build 2-body equations with optional perturbations."""
        # Position magnitude
        r = hy.sqrt(x**2 + y**2 + z**2)
        # Gravitational parameter (hardcoded from System)
        mu = self._primary_body.mu
        
        # Base gravitational acceleration
        a_grav_x = -mu * x / r**3
        a_grav_y = -mu * y / r**3
        a_grav_z = -mu * z / r**3
        
        # Start with gravitational acceleration then add perturbations
        a_total_x = a_grav_x
        a_total_y = a_grav_y
        a_total_z = a_grav_z
        
        # Parameter tracking
        param_map = []
        param_desc = {}
        next_param_idx = 0
        
        # Add perturbations
        if "J2" in self._perturbations:
            a_J2_x, a_J2_y, a_J2_z = self._build_J2_perturbation(
                x, y, z, r, mu
            )
            a_total_x = a_total_x + a_J2_x
            a_total_y = a_total_y + a_J2_y
            a_total_z = a_total_z + a_J2_z
        
        if "drag" in self._perturbations:
            a_drag_x, a_drag_y, a_drag_z, drag_params = self._build_drag_perturbation(
                x, y, z, vx, vy, vz, r, next_param_idx
            )
            a_total_x = a_total_x + a_drag_x
            a_total_y = a_total_y + a_drag_y
            a_total_z = a_total_z + a_drag_z
            
            # Add drag parameters to map
            param_map.extend(drag_params['param_map'])
            param_desc.update(drag_params['description'])
            next_param_idx += len(drag_params['param_map'])
        
        # Assemble system
        sys = [
            (x, vx),
            (y, vy),
            (z, vz),
            (vx, a_total_x),
            (vy, a_total_y),
            (vz, a_total_z)
        ]
        
        param_info = {
            'param_map': param_map,
            'description': param_desc
        }
        
        return sys, param_info

    def _build_J2_perturbation(self, x, y, z, r, mu):
        """Build J2 perturbation acceleration terms."""
        J2 = self._primary_body.J2
        R = self._primary_body.radius
        assert J2 is not None # validated in _compute_params
        # J2 perturbation (zonal harmonic)
        # Common factor: (3/2) * J2 * mu * R^2 / r^5
        factor = 1.5 * J2 * mu * R**2 / r**5
        # Acceleration components
        # a_J2 = factor * [x(5z^2/r^2 - 1), y(5z^2/r^2 - 1), z(5z^2/r^2 - 3)]
        z2_r2 = z**2 / r**2
        
        a_J2_x = factor * x * (5.0 * z2_r2 - 1.0)
        a_J2_y = factor * y * (5.0 * z2_r2 - 1.0)
        a_J2_z = factor * z * (5.0 * z2_r2 - 3.0)
        
        return a_J2_x, a_J2_y, a_J2_z

    def _build_drag_perturbation(self, x, y, z, vx, vy, vz, r, param_start_idx):
        """
        Build atmospheric drag perturbation.

        Uses exponential atmosphere model and accounts for Earth rotation.
        Drag parameters (Cd*A, mass) are extracted from Satellite object
        during propagation and passed via hy.par[] array.

        Parameters
        ----------
        param_start_idx : int
            Starting index for hy.par[] array

        Returns
        -------
        a_drag_x, a_drag_y, a_drag_z : symbolic expressions
            Drag acceleration components
        param_info : dict
            Parameter mapping information with keys:
            - 'param_map': [('Cd_A', idx), ('mass', idx)]
            - 'description': Human-readable parameter descriptions
        """
        # Atmospheric parameters (hardcoded from System)
        rho0 = self._atmosphere.rho0  # kg/mÂ³
        H = self._atmosphere.H  # m
        r0 = self._atmosphere.r0  # m
        omega = self._primary_body.rotation_rate  # rad/s
        
        # Convert atmospheric params to km (package standard)
        # Density will be kg/kmÂ³, scale height in km
        rho0_km = rho0 * 1e9  # kg/mÂ³ -> kg/kmÂ³
        H_km = H / 1000.0  # m -> km
        r0_km = r0 / 1000.0  # m -> km
        
        # Altitude-dependent density (exponential model)
        # Ï(r) = Ïâ‚€ * exp(-(r - râ‚€)/H)
        rho = rho0_km * hy.exp(-(r - r0_km) / H_km)
        
        # Velocity relative to rotating atmosphere
        # v_rel = v_inertial - Ï‰ Ã— r
        # For Earth rotation about z-axis: Ï‰ Ã— r = [-Ï‰*y, Ï‰*x, 0]
        vx_rel = vx + omega * y
        vy_rel = vy - omega * x
        vz_rel = vz
        
        # Relative velocity magnitude
        v_rel = hy.sqrt(vx_rel**2 + vy_rel**2 + vz_rel**2)
        
        # Satellite parameters (runtime via hy.par[])
        Cd_A = hy.par[param_start_idx]      # Drag coefficient * area [mÂ²]
        mass = hy.par[param_start_idx + 1]  # Satellite mass [kg]
        
        # Convert Cd*A to kmÂ² for unit consistency
        Cd_A_km = Cd_A / 1e6  # mÂ² -> kmÂ²
        
        # Drag acceleration: a_drag = -(1/2) * Ï * (Cd*A/m) * v_rel * vâƒ—_rel
        # Factor: -(1/2) * Ï * (Cd*A/m) * v_rel
        drag_factor = -0.5 * rho * Cd_A_km / mass * v_rel
        
        a_drag_x = drag_factor * vx_rel
        a_drag_y = drag_factor * vy_rel
        a_drag_z = drag_factor * vz_rel
        
        # Parameter information
        param_info = {
            'param_map': [
                ('Cd_A', param_start_idx),
                ('mass', param_start_idx + 1)
            ],
            'description': {
                'Cd_A': 'Drag coefficient times reference area [mÂ²]',
                'mass': 'Satellite mass [kg]'
            }
        }
        
        return a_drag_x, a_drag_y, a_drag_z, param_info

    def _build_cr3bp_eom(self, x, y, z, vx, vy, vz):
        """Build CR3BP equations in rotating frame."""
        # Mass ratio (nondimensional)
        mu = self._mass_ratio
        assert mu is not None # validated in _compute_CR3BP_params
        # Distances to primaries (in rotating frame)
        r1 = hy.sqrt((x + mu)**2 + y**2 + z**2)
        r2 = hy.sqrt((x - 1.0 + mu)**2 + y**2 + z**2)
        # Pseudo-potential U (includes centrifugal effect)
        U = 0.5 * (x**2 + y**2) + (1.0 - mu) / r1 + mu / r2
        
        # Equations of motion in rotating frame
        sys = [
            (x, vx),
            (y, vy),
            (z, vz),
            (vx, 2.0 * vy + hy.diff(U, x)),   
            (vy, -2.0 * vx + hy.diff(U, y)), 
            (vz, hy.diff(U, z))
        ]
        
        # CR3BP has no runtime parameters (everything is nondimensional)
        param_info = {
            'param_map': [],
            'description': {}
        }
        
        return sys, param_info
    
    def _compile_integrator(self):
        """
        Compile Heyoka integrator (expensive operation).
        
        This performs automatic differentiation and LLVM compilation,
        which takes 1-5 seconds depending on system complexity.
        """
        if self._cached_integrator is not None:
            return  # Already compiled
        
        # Create dummy parameters for compilation
        n_params = len(self._param_info['param_map'])
        dummy_params = [0.0] * n_params
        
        # Print message for long compilation
        msg = f"Compiling {self._base_type.value} integrator"
        if self._perturbations:
            msg += f" with {', '.join(self._perturbations)}"
        print(msg + "...")
        
        # EXPENSIVE: Compile integrator
        self._cached_integrator = hy.taylor_adaptive(
            sys=self._cached_eom,
            state=[0.0] * 6,  # Dummy state
            pars=dummy_params,
        )
        print(f"Compilation complete")
    
    def compile(self):
        """
        Explicitly compile integrator if not already compiled.
        
        Compilation occurs immediately on instance creation by default
        Call this method to compile if initializing with compile=False
        
        Returns
        -------
        self
            Returns self for method chaining
        """
        self._compile_integrator()
        return self

    # ========== SPECIAL METHODS ==========
    def __del__(self):
        """Decrement instance count when System is garbage collected."""
        System._instance_count -= 1

    def __repr__(self):
        """Readable string representation."""
        parts = [f"System(base_type='{self._base_type.value}'"]
        
        if self._base_type == SysType.TWO_BODY:
            parts.append(f"primary={self._primary_body.mu:.3e} km^3/s^2")
            if self._perturbations:
                parts.append(f"perturbations={self._perturbations}")
        else:  # 3body
            parts.append(f"mu_1‚={self._primary_body.mu:.3e}")
            parts.append(f"mu_2‚={self._secondary_body.mu:.3e}")
            parts.append(f"L*={self._L_star:.3e} km")
            parts.append(f"mu={self._mass_ratio:.6f}")
        
        return ", ".join(parts) + ")"

    # ========== STATIC METHODS ==========
    @staticmethod
    def _parse_base_type(base_type):
        """Convert string or enum to SysType enum"""
        if isinstance(base_type, SysType):
            return base_type
        elif isinstance(base_type, str):
            # Map string to enum
            type_map = {
                '2body' : SysType.TWO_BODY,
                '2BODY' : SysType.TWO_BODY,
                'Two_Body': SysType.TWO_BODY,
                '3body' : SysType.CR3BP,
                '3BODY' : SysType.CR3BP,
                'Three_Body': SysType.CR3BP,
                'CR3BP' : SysType.CR3BP
            }
            if base_type in type_map:
                return type_map[base_type]
            else:
                raise ValueError(f"Unknown base type '{base_type}'. "
                           f"Use: {list(type_map.keys())}")
        else:
            raise TypeError(f"base_type must be SysType or str, got {type(base_type)}")