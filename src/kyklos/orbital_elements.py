'''Development code for an orbital trajectory handling package
OrbitalElements class definition
created with the assistance of Claude Sonnet 4.5 by Anthropic'''

import numpy as np
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .system import SysType

# define an enumerated list of element types
class OEType(Enum):
    CARTESIAN = 'cart'      # [x;y;z;vx;vy;vz]
    KEPLERIAN = 'kep'       # [a;e;i;Omega;w;nu]
    EQUINOCTIAL = 'equi'    # [p;f;g;h;k;L]
    CR3BP = 'cr3bp'         # [x_nd;y_nd;z_nd;vx_nd;vy_nd;vz_nd]

#define basic orbital element class
class OrbitalElements:
    """
    Represents orbital elements as a set of six phase space invariants
    Currently does not reference a specific coordinate frame
    Cartesian representations assume an equatorial frame centered on primary system body
    OrbitalElements is immutable, extract elements using numpy methods and create a
    new instance to change
    """
    # ========== CLASS CONSTANTS ==========
    # Tolerance for floating-point equality comparisons
    _EQUALITY_RTOL = 1e-12  # Relative tolerance (~mm at LEO)
    _EQUALITY_ATOL = 1e-14  # Absolute tolerance
    _HASH_DECIMALS = 10     # Rounding for consistent hashing
    
    # Default gravitational parameter (Earth)
    DEFAULT_MU = 398600.435507  # km³/s²
    
    # ========== CONSTRUCTION ==========
    def __init__(self, elements=None, element_type=None, validate=True, 
             system=None, mu=None, **kwargs):
        """
        Create orbital elements.
        
        Can be called in two ways:
        
        1. Array-based (fast for propagation):
        OrbitalElements([7000, 0.01, 0.5, 0, 0, 0], 'kep', mu=398600.4418)
        
        2. Named parameters (readable for setup):
        OrbitalElements(a=7000, e=0.01, i=0.5, omega=0, w=0, nu=0, mu=398600.4418)
        OrbitalElements(x=-6045, y=-3490, z=2500, vx=-3.457, vy=6.618, vz=2.533, 
                        system=EARTH)
        
        Parameters
        ----------
        elements : array-like, optional
            6-element array of orbital elements
        element_type : OEType or str, optional
            Type of elements ('cart', 'kep') - required if using elements array
        validate : bool, optional
            Whether to validate elements (default True)
        system : System, optional
            System object containing gravitational parameters
        mu : float, optional
            Gravitational parameter (km³/s²) if system not provided
            Defaults to Earth's GM if neither system nor mu provided
        **kwargs : dict
            Named parameters for appropriate orbital element set
            Keplerian (a, e, i, omega, w, nu)
            Cartesian (x, y, z, vx, vy, vz)
            Equinoctial (p, f, g, h, k, L)
            CR3BP (x_nd, y_nd, z_nd, vx_nd, vy_nd, vz_nd)
        """
       # Store system or mu
        self._system = system
        if system is not None:
            # Extract mu from system
            from .system import SysType
            if system.base_type == SysType.CR3BP:
                # For CR3BP, use mass ratio instead of gravitational parameter
                self._mu = system.mass_ratio
            elif hasattr(system, 'primary_body'):
                self._mu = system.primary_body.mu
            else:
                raise ValueError("System object must have primary_body attribute")
        elif mu is not None:
            self._mu = mu
        else:
            # Use default (Earth)
            self._mu = self.DEFAULT_MU

        # Determine construction method
        if elements is not None:
            # Array-based construction
            self.elements = np.array(elements)
            self.element_type = self._parse_element_type(element_type)
        
        elif kwargs:
            # Named parameter construction - auto-detect type
            self.elements, self.element_type = self._from_named_params(kwargs)
        
        else:
            raise ValueError(
                "Must provide either:\n"
                "  - elements array and element_type, or named parameters: \n"
                "  - (a, e, i, omega, w, nu) for Keplerian, or\n"
                "  - (x, y, z, vx, vy, vz) for Cartesian, or\n"
                "  - (p, f, g, h, k, L) for Equinoctial, or\n"
                "  - (x_nd, y_nd, z_nd, vx_nd, vy_nd, vz_nd) for CR3BP"
            )
        # Ensure immutability of elements array
        self.elements.flags.writeable = False
        # run validation checks on input parameters (if not flagged otherwise)
        if validate:
            self._validate()
    
    # define alternate constructors to bypass validation and automatically input type
    @classmethod
    def cartesian(cls, elements, system=None, mu=None):
        """
        Create Cartesian orbital elements without validation (for automated processes)
        
        Args:
            elements: 6-element array [x, y, z, vx, vy, vz]
            system: System object (optional)
            mu: Gravitational parameter (optional, defaults to Earth)
        
        Returns:
            OrbitalElements instance
        """
        return cls(elements, OEType.CARTESIAN, validate=False, system=system, mu=mu)

    @classmethod
    def keplerian(cls, elements, system=None, mu=None):
        """
        Create Keplerian orbital elements without validation (for automated processes)
        
        Args:
            elements: 6-element array [a, e, i, Ω, ω, ν]
            system: System object (optional)
            mu: Gravitational parameter (optional, defaults to Earth) 
        
        Returns:
            OrbitalElements instance
        """
        return cls(elements, OEType.KEPLERIAN, validate=False, system=system, mu=mu)

    @classmethod
    def equinoctial(cls, elements, system=None, mu=None):
        """
        Create Equinoctial orbital elements without validation (for automated processes)
        
        Args:
            elements: 6-element array [p, f, g, h, k, L]
            system: System object (optional)
            mu: Gravitational parameter (optional, defaults to Earth)
        
        Returns:
            OrbitalElements instance
        """
        return cls(elements, OEType.EQUINOCTIAL, validate=False, system=system, mu=mu)
    
    # ========== VALIDATION ==========
    def _validate(self):
        """Check if elements conform to their claimed type
        If validation fails inappropriately, set validate=False for constructor
        """
        # check some properties common to all element sets
        if len(self.elements) != 6:
            raise ValueError("Orbital elements must be 6-element vector")
        if not np.all(np.isfinite(self.elements)):
            raise ValueError("Elements contain NaN or Inf")
        if np.iscomplexobj(self.elements):
            raise ValueError("Elements cannot contain complex values")
        
        if self.element_type == OEType.KEPLERIAN:
            self._validate_keplerian()
        elif self.element_type == OEType.CARTESIAN:
            self._validate_cartesian()
        elif self.element_type == OEType.EQUINOCTIAL:
            self._validate_equinoctial()
        elif self.element_type == OEType.CR3BP:
            self._validate_cr3bp()
    
    def _validate_keplerian(self):
        a, e, i, omega, w, nu = self.elements
        # Validate a-e combination for physical consistency
        if e < 1 and a <= 0:
            raise ValueError(f"Elliptic orbit (e={e}) "
                         f"requires positive semi-major axis, got a={a}")
        if e >= 1 and a >= 0:
            raise ValueError(f"Hyperbolic orbit (e={e}) "
                             f"requires negative semi-major axis, got a={a}")
        # check appropriate ranges
        if e < 0 or e > 10:
            raise ValueError("Eccentricity out of range")
        if i > np.pi or i < 0:
            raise ValueError("Inclination out of range")
        # check range for Euler angles
        if omega < -np.pi or omega > 2*np.pi:
            raise ValueError("RAAN out of range")
        if w < -np.pi or w > 2*np.pi:
            raise ValueError("Arg of Periapsis out of range")
        if nu < -np.pi or nu > 2*np.pi:
            raise ValueError("True Anomaly out of range")
    
    def _validate_equinoctial(self):
        p, f, g, h, k, L = self.elements
        if f < -10 or f > 10:
            raise ValueError("f out of range")
        if g < -10 or g > 10:
            raise ValueError("g out of range")
        if L < 0:
            raise ValueError("L out of range")
    
    def _validate_cartesian(self):
        # Basic sanity checks for position/velocity
        pos = np.linalg.norm(self.elements[:3])
        vel = np.linalg.norm(self.elements[3:])
        if pos < 10 * vel:
            raise ValueError(
                f"Position magnitude ({pos:.2f}) should be at least "
                f"an order of magnitude greater than velocity magnitude ({vel:.2f})"
            )
    
    def _validate_cr3bp(self):
        # Basic sanity checks for position/velocity
        pos = np.linalg.norm(self.elements[:3])
        vel = np.linalg.norm(self.elements[3:])
        if pos > 10:
            raise ValueError(
                f"Position magnitude is > 10 nondimensional units, check units.")
        if vel > 5:
            raise ValueError(
                f"Velocity magnitude is > 5 nondimensional units, check units.")
        
    
    # ========== FACTORY METHODS ==========
    @classmethod
    def from_numpy(cls, array, element_type, validate=True, system=None, mu=None):
        """
        Create list of OrbitalElements from NumPy array.
        
        Parameters
        ----------
        array : np.ndarray
            Array of shape (n_orbits, 6)
        element_type : OEType or str
        validate: bool, optional, defaults to True
        system : System, optional
        mu : float, optional
        
        Returns
        -------
        list of OrbitalElements
        """
        if array.ndim != 2 or array.shape[1] != 6:
            raise ValueError(f"Array must have shape (n, 6), got {array.shape}")
        
        return [cls(row, element_type, validate=validate, 
                   system=system, mu=mu) for row in array]
    
    @classmethod
    def from_dataframe(cls, df, element_type=None, validate=True, system=None, mu=None):
        """
        Create list of OrbitalElements from pandas DataFrame.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with 6 columns of orbital elements
        element_type : OEType or str, optional
            If None, inferred from column names
        validate : bool, optional, defaults to True
        system : System, optional
        mu : float, optional
        
        Returns
        -------
        list of OrbitalElements
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas required for from_dataframe()")
        
        if element_type is None:
            # Infer from column names
            cols = set(df.columns)
            if cols == {'a', 'e', 'i', 'RAAN', 'omega', 'nu'}:
                element_type = OEType.KEPLERIAN
            elif cols == {'x', 'y', 'z', 'vx', 'vy', 'vz'}:
                element_type = OEType.CARTESIAN
            elif cols == {'p', 'f', 'g', 'h', 'k', 'L'}:
                element_type = OEType.EQUINOCTIAL
            elif cols == {'x_nd', 'y_nd', 'z_nd', 'vx_nd', 'vy_nd', 'vz_nd'}:
                element_type = OEType.CR3BP
            else:
                raise ValueError(
                    "Could not infer element type from columns. "
                    "Provide element_type explicitly."
                )
        
        return [cls(row.values, element_type, validate=validate,
                   system=system, mu=mu) for _, row in df.iterrows()]
    
    # ========== ELEMENT TYPE CONVERSIONS ==========
    def convert_to(self, target_type):
        """
        Convert orbital elements to a different representation.
        
        Parameters:
        -----------
        target_type : OEType or str
            The desired orbital element type to convert to
            Can be OEType enum or string ('cart', 'kep', 'equi')
        mu : float, optional
            Gravitational parameter (km³/s²), default is Earth's GM
            
        Returns:
        --------
        OrbitalElements
            New OrbitalElements object with elements in target type
        """
        # Convert string to enum if necessary
        target_type = self._parse_element_type(target_type)
        
        if target_type == self.element_type:
            # No conversion needed, return copy
            return OrbitalElements.copy(self)
        
        if self.element_type == OEType.KEPLERIAN and target_type == OEType.CARTESIAN:
            # Convert Keplerian to Cartesian
            converted_elements = self._keplerian_to_cartesian()
        
        elif self.element_type == OEType.KEPLERIAN and target_type == OEType.EQUINOCTIAL:
            # Convert Keplerian to Modified Equinoctial
            converted_elements = self._keplerian_to_equinoctial()
            
        elif self.element_type == OEType.CARTESIAN and target_type == OEType.KEPLERIAN:
            # Convert Cartesian to Keplerian
            converted_elements = self._cartesian_to_keplerian()
        
        elif self.element_type == OEType.CARTESIAN and target_type == OEType.EQUINOCTIAL:
            # Convert Cartesian to Modified Equinoctial
            converted_elements = self._cartesian_to_equinoctial()
        
        elif self.element_type == OEType.EQUINOCTIAL and target_type == OEType.KEPLERIAN:
            # Convert Modified Equinoctial to Keplerian
            converted_elements = self._equinoctial_to_keplerian()
        
        elif self.element_type == OEType.EQUINOCTIAL and target_type == OEType.CARTESIAN:
            # Convert Modified Equinoctial to Cartesian
            converted_elements = self._equinoctial_to_cartesian()
        else:
            raise ValueError(f"Conversion from {self.element_type.value} "
                             f"to {target_type.value} not implemented")
        
        return OrbitalElements(converted_elements, target_type, 
                         validate=False, system=self._system, mu=self._mu)
    
    def _keplerian_to_cartesian(self):
        """Convert Keplerian elements to Cartesian state vector."""
        a, e, i, omega, w, nu = self.elements
        # find semi-latus rectum
        p = a*(1 - e**2)
        # find position in perifocal frame
        r_mag = p / (1 + e*np.cos(nu))
        rvec = np.array([r_mag*np.cos(nu), r_mag*np.sin(nu), 0])
        # find velocity in perifocal frame
        vvec = np.array([-np.sqrt(self._mu/p) * np.sin(nu),
                     np.sqrt(self._mu/p) * (e + np.cos(nu)),0])
        # rotate from perifocal frame to inertial frame using DCM
        # rotation about z-axis by RAAN
        R3_omega = np.array([
            [np.cos(omega), -np.sin(omega), 0],
            [np.sin(omega),  np.cos(omega), 0],
            [0,              0,              1]
        ])
        # rotation about x-axis by inclination
        R1_i = np.array([
            [1,  0,             0            ],
            [0,  np.cos(i),    -np.sin(i)    ],
            [0,  np.sin(i),     np.cos(i)    ]
        ])
        # rotation about z-axis by argument of periapsis
        R3_w = np.array([
            [np.cos(w), -np.sin(w), 0],
            [np.sin(w),  np.cos(w), 0],
            [0,          0,          1]
        ])
        # Combined rotation
        DCM = R3_omega @ R1_i @ R3_w
        R = DCM @ rvec
        V = DCM @ vvec
        return np.concatenate([R,V])
    
    def _keplerian_to_equinoctial(self):
        """Convert Keplerian elements to Modified Equinoctial elements"""
        a, e, i, omega, w, nu = self.elements
        # define elements according to Walker et al, Celestial Mechanics, v.36,pp.409
        p = a*(1-e**2)
        f = e*np.cos(omega + w)
        g = e*np.sin(omega + w)
        h = np.tan(i/2)*np.cos(omega)
        k = np.tan(i/2)*np.sin(omega)
        L = w + omega + nu
        return np.array([p,f,g,h,k,L])
    
    def _cartesian_to_keplerian(self):
        """Convert Cartesian state vector to Keplerian elements.
        Uses algorithm from Flores & Fantino, Advances in Space Research, v.75,pp.4910
        """
        # Extract position and velocity
        rvec = self.elements[:3]
        vvec = self.elements[3:]
        # calculate angular momentum vector h = r × v
        hvec = np.cross(rvec,vvec)
        # calculate inclination
        i = np.arctan2(np.sqrt(hvec[0]**2 + hvec[1]**2),hvec[2])
        # find longitude of ascending node
        omega = np.arctan2(hvec[0],-hvec[1])
        # define line of nodes vector
        nhat = np.array([np.cos(omega),np.sin(omega),0])
        # define an intermediate vector b in the orbit plane
        bhat = np.cross(hvec/np.linalg.norm(hvec),nhat)
        # find semimajor axis from energy equation
        a = ((2/np.linalg.norm(rvec)) - (np.dot(vvec,vvec)/self._mu))**(-1)
        # find eccentricity vector
        evec = np.cross(vvec,hvec)/self._mu - rvec/np.linalg.norm(rvec)
        # find argument of periapsis and true anomaly
        w = np.arctan2(np.dot(evec,bhat),np.dot(evec,nhat))
        nu = np.arctan2(np.dot(rvec,bhat),np.dot(rvec,nhat)) - w
        # find eccentricity
        e = np.linalg.norm(evec)
        return np.array([a,e,i,omega,w,nu])
    
    def _cartesian_to_equinoctial(self):
        """
        Convert Cartesian state vector to Modified Equinoctial elements.
        Uses the prograde formulation (singularity at i = 180°).
        """
        # Extract position and velocity
        rvec = self.elements[:3]
        vvec = self.elements[3:]
        # Compute normalized position
        r_mag = np.linalg.norm(rvec)
        r_hat = rvec / r_mag
        # Compute normalized angular momentum vector
        h_vec = np.cross(rvec, vvec)
        h_mag = np.linalg.norm(h_vec)
        h_hat = h_vec / h_mag
        # Semi-latus rectum
        p = h_mag**2 / self._mu
        # Node vector components (h, k)
        # h = -h_y / (1 + h_z), k = h_x / (1 + h_z)
        h_elem = -h_hat[1] / (1 + h_hat[2])
        k_elem = h_hat[0] / (1 + h_hat[2])
        # Eccentricity vector: e_vec = (v × h)/mu - r_hat
        evec = np.cross(vvec, h_vec) / self._mu - r_hat
        # Construct f_hat and g_hat basis vectors
        h_sq = h_elem**2
        k_sq = k_elem**2
        s2 = 1 + h_sq + k_sq
        tkh = 2 * k_elem * h_elem
        f_hat = np.array([1 - k_sq + h_sq,tkh,-2 * k_elem]) / s2
        g_hat = np.array([tkh,1 + k_sq - h_sq,2 * h_elem]) / s2
        # Eccentricity components (f, g)
        f_elem = np.dot(evec, f_hat)
        g_elem = np.dot(evec, g_hat)
        # For true longitude L, we need position projected onto f_hat and g_hat
        # r·v gives us information about where we are in the orbit
        rdv = np.dot(rvec, vvec)
        # Velocity perpendicular to radial direction
        v_hat = (r_mag * vvec - rdv * r_hat) / h_mag
        # Position and velocity components in f-g frame
        x = np.dot(rvec, f_hat)
        y = np.dot(rvec, g_hat)
        x_dot = np.dot(v_hat, f_hat)
        y_dot = np.dot(v_hat, g_hat)
        # True longitude from atan2, in range [0,2pi]
        L = np.arctan2(y, x) % (2 * np.pi)
        return np.array([p, f_elem, g_elem, h_elem, k_elem, L])
    
    def _equinoctial_to_keplerian(self):
        """
        Convert Modified Equinoctial elements to Keplerian elements.
        Uses the prograde formulation (singularity at i = 180°).
        """
        p, f, g, h, k, L = self.elements
        a = p/(1 - f**2 - g**2)
        e = np.sqrt(f**2 + g**2)
        i = np.atan2(2*np.sqrt(h**2 + k**2),1-h**2-k**2)
        w = np.atan2(g*h - f*k,f*h + g*k)
        RAAN = np.atan2(k,h)
        nu = L - (w+RAAN)
        return np.array([a,e,i,RAAN,w,nu])
    
    def _equinoctial_to_cartesian(self):
        """
        Convert Modified Equinoctial elements to Cartesian state vector.
        Uses the prograde formulation (singularity at i = 180°).
        """
        p, f, g, h, k, L = self.elements
        # define intermediate quantities
        asqr = h**2 - k**2
        ssqr = 1 + h**2 + k**2
        w = 1 + f*np.cos(L) + g*np.sin(L)
        r = p/w
        #map to Cartesian state
        r1 = (r/ssqr)*(np.cos(L) + asqr*np.cos(L) + 2*h*k*np.sin(L))
        r2 = (r/ssqr)*(np.sin(L) - asqr*np.sin(L) + 2*h*k*np.cos(L))
        r3 = ((2*r)/ssqr)*(h*np.sin(L) - k*np.cos(L))
        v1 = (-(1/ssqr)*np.sqrt(self._mu/p)*(np.sin(L) + asqr*np.sin(L) - 
                    2*h*k*np.cos(L) + g - 2*f*h*k + asqr*g))
        v2 = (-(1/ssqr)*np.sqrt(self._mu/p)*(-np.cos(L) + asqr*np.cos(L) + 
                    2*h*k*np.sin(L) - f + 2*g*h*k + asqr*f))
        v3 = (2/ssqr)*np.sqrt(self._mu/p)*(h*np.cos(L) + k*np.sin(L) + f*h + g*k)
        return np.array([r1,r2,r3,v1,v2,v3])
    
    # conversion shortcuts for convenience
    def to_cartesian(self):
        """Shortcut for convert_to('cart')"""
        return self.convert_to(OEType.CARTESIAN)

    def to_keplerian(self):
        """Shortcut for convert_to('kep')"""
        return self.convert_to(OEType.KEPLERIAN)

    def to_equinoctial(self):
        """Shortcut for convert_to('equi')"""
        return self.convert_to(OEType.EQUINOCTIAL)

    # ========== PROPERTY ACCESS ==========
    @property
    def system(self):
        """System object (if provided)"""
        return self._system

    @property
    def mu(self):
        """Gravitational parameter [km³/s²]"""
        return self._mu
    
    @property
    def a(self):
        """Semi-major axis (only for Keplerian elements)"""
        if self.element_type != OEType.KEPLERIAN:
            raise AttributeError(
                "Semi-major axis only available for Keplerian elements")
        return self.elements[0]

    @property
    def e(self):
        """Eccentricity"""
        if self.element_type == OEType.KEPLERIAN:
            return self.elements[1]
        elif self.element_type == OEType.EQUINOCTIAL:
            f, g = self.elements[1], self.elements[2]
            return np.sqrt(f**2 + g**2)
        elif self.element_type == OEType.CARTESIAN:
            raise NotImplementedError(
                "Computing eccentricity from Cartesian requires conversion")

    @property
    def position(self):
        """Position vector (only for Cartesian and CR3BP)"""
        if self.element_type == OEType.CARTESIAN:
            return self.elements[:3]
        elif self.element_type == OEType.CR3BP:
            return self.elements[:3]
        else:
            raise AttributeError(
                "Position only directly available for Cartesian & CR3BP elements")

    @property
    def velocity(self):
        """Velocity vector (only for Cartesian and CR3BP)"""
        if self.element_type == OEType.CARTESIAN:
            return self.elements[3:]
        elif self.element_type == OEType.CR3BP:
            return self.elements[3:]
        else:
            raise AttributeError(
                "Velocity only directly available for Cartesian & CR3BP elements")
    
    # ========== ORBITAL PROPERTIES ==========
    def orbital_period(self):
        """
        Calculate orbital period
        
        Returns period in seconds (only for elliptic orbits)
        """
        if self.element_type == OEType.CR3BP:
            raise ValueError(
                "Orbital Period not available for CR3BP elements, use a Trajectory")
        elif self.element_type == OEType.KEPLERIAN:
            a = self.elements[0]
            e = self.elements[1]
        elif self.element_type == OEType.EQUINOCTIAL:
            p = self.elements[0]
            f, g = self.elements[1], self.elements[2]
            e = np.sqrt(f**2 + g**2)
            a = p / (1 - e**2)
        else:
            # Convert to Keplerian first
            kep = self.to_keplerian()
            a = kep.elements[0]
            e = kep.elements[1]
        
        if e >= 1:
            raise ValueError("Orbital period undefined for parabolic/hyperbolic orbits")
        
        return 2 * np.pi * np.sqrt(a**3 / self._mu)

    def specific_energy(self):
        """Calculate specific orbital energy (energy per unit mass)"""
        if self.element_type == OEType.CR3BP:
            raise ValueError("Specific Energy not applicable for CR3BP trajectories")
        elif self.element_type == OEType.CARTESIAN:
            r = self.elements[:3]
            v = self.elements[3:]
            r_mag = np.linalg.norm(r)
            v_mag = np.linalg.norm(v)
            return v_mag**2 / 2 - self._mu / r_mag
        elif self.element_type == OEType.KEPLERIAN:
            a = self.elements[0]
            return -self._mu / (2 * a)
        else:
            # Convert to Cartesian
            cart = self.to_cartesian()
            return cart.specific_energy()

    def specific_angular_momentum(self):
        """
        Calculate specific angular momentum magnitude
        
        Returns h = |cross(position,velocity)|
        """
        if self.element_type == OEType.CR3BP:
            raise ValueError(
                "Specific Angular Momentum not relevant for CR3BP trajectories")
        elif self.element_type == OEType.CARTESIAN:
            r = self.elements[:3]
            v = self.elements[3:]
            return np.linalg.norm(np.cross(r, v))
        elif self.element_type == OEType.KEPLERIAN:
            a = self.elements[0]
            e = self.elements[1]
            p = a * (1 - e**2)
            return np.sqrt(self._mu * p)
        elif self.element_type == OEType.EQUINOCTIAL:
            p = self.elements[0]
            return np.sqrt(self._mu * p)
    
    def mean_motion(self):
        """
        Calculate mean motion (n = √(μ/a³))
        
        Returns
        -------
        float
            Mean motion [rad/s]
        
        Raises
        ------
        ValueError
            If called on CR3BP elements or parabolic/hyperbolic orbit
        """
        if self.element_type == OEType.CR3BP:
            raise ValueError(
                "Mean motion not applicable for CR3BP elements (use nondim time)")
        
        # Get semi-major axis and eccentricity
        if self.element_type == OEType.KEPLERIAN:
            a = self.elements[0]
            e = self.elements[1]
        elif self.element_type == OEType.EQUINOCTIAL:
            p = self.elements[0]
            f, g = self.elements[1], self.elements[2]
            e = np.sqrt(f**2 + g**2)
            a = p / (1 - e**2)
        else:  # CARTESIAN
            # Convert to Keplerian first
            kep = self.to_keplerian()
            a = kep.elements[0]
            e = kep.elements[1]
    
        # Check for elliptic orbit
        if e >= 1:
            raise ValueError("Mean motion undefined for parabolic/hyperbolic orbits")
        
        return np.sqrt(self._mu / a**3)
        
    def jacobi_const(self):
        """Calculate Jacobi constant (only for CR3BP elements)"""
        if self.element_type != OEType.CR3BP:
            raise ValueError("Jacobi constant not defined except for CR3BP systems")
        else:
            x, y, z, vx, vy, vz = self.elements
            r1 = np.sqrt((x + self._mu)**2 + y**2 + z**2)
            r2 = np.sqrt((x - 1 + self._mu)**2 + y**2 + z**2)
            return ((x**2 + y**2) + ((2 * (1 - self._mu)) / r1) + 
                    ((2 * self._mu) / r2) - (vx**2 + vy**2 + vz**2))

    # ========== UTILITY METHODS ==========

    def copy(self):
        """Create a deep copy of the orbital elements"""
        return OrbitalElements(self.elements.copy(), self.element_type, 
                               validate=False, system=self._system, mu=self._mu)
    
    # ========== BATCH OPERATIONS ==========
    class Batch:
        """
        Batch operations on collections of OrbitalElements.
        
        All methods accept a list of OrbitalElements and return
        a list of OrbitalElements or computed values.
        """
        @staticmethod
        def copy(orbits, target_type):
            """Convert multiple orbits to target type"""
            return [o.copy() for o in orbits]
        
        @staticmethod
        def convert_to(orbits, target_type):
            """Convert multiple orbits to target type"""
            return [o.convert_to(target_type) for o in orbits]
        
        @staticmethod
        def to_cartesian(orbits):
            """Convert multiple orbits to Cartesian"""
            return [o.to_cartesian() for o in orbits]
        
        @staticmethod
        def to_keplerian(orbits):
            """Convert multiple orbits to Keplerian"""
            return [o.to_keplerian() for o in orbits]
        
        @staticmethod
        def to_equinoctial(orbits):
            """Convert multiple orbits to Equinoctial"""
            return [o.to_equinoctial() for o in orbits]
        
        @staticmethod
        def a(orbits):
            """Get semi-major axis for multiple orbits"""
            return np.array([o.a() for o in orbits])
        
        @staticmethod
        def e(orbits):
            """Get eccentricity for multiple orbits"""
            return np.array([o.e() for o in orbits])

        @staticmethod
        def pos(orbits):
            """Get position for multiple orbits"""
            return np.array([o.pos() for o in orbits])
        
        @staticmethod
        def vel(orbits):
            """Get velocity for multiple orbits"""
            return np.array([o.vel() for o in orbits])
        
        @staticmethod
        def orbital_period(orbits):
            """Get orbital periods for multiple orbits"""
            return np.array([o.orbital_period() for o in orbits])
        
        @staticmethod
        def mean_motion(orbits):
            """Get mean motions for multiple orbits"""
            return np.array([o.mean_motion() for o in orbits])
        
        @staticmethod
        def specific_energy(orbits):
            """Get specific energy for multiple orbits"""
            return np.array([o.specific_energy() for o in orbits])
        
        @staticmethod
        def specific_angular_momentum(orbits):
            """Get specific angular momentum for multiple orbits"""
            return np.array([o.specific_angular_momentum() for o in orbits])
        
        @staticmethod
        def to_numpy(orbits):
            """
            Convert list of OrbitalElements to NumPy array.
            
            Parameters
            ----------
            orbits : list of OrbitalElements
            
            Returns
            -------
            np.ndarray
                Array of shape (n_orbits, 6) containing orbital elements
            
            Notes
            -----
            All orbits must be the same element type. The returned array
            contains the raw element values without type information.
            """
            # Check all same type
            elem_type = orbits[0].element_type
            if not all(o.element_type == elem_type for o in orbits):
                raise ValueError("All orbits must have the same element type")
            
            return np.array([o.elements for o in orbits])
        
        @staticmethod
        def to_dataframe(orbits, index=None):
            """
            Convert list of OrbitalElements to pandas DataFrame.
            
            Parameters
            ----------
            orbits : list of OrbitalElements
                List of orbital elements
            index : array-like, optional
                Index for the DataFrame (e.g., time values).
                If None, uses integer index.
            
            Returns
            -------
            pd.DataFrame
                DataFrame with columns named according to element type
            
            Raises
            ------
            ValueError
                If orbits have different element types or if index length
                doesn't match number of orbits

            Notes
            -----
            Column names depend on element type:
            - Keplerian: ['a', 'e', 'i', 'RAAN', 'omega', 'nu']
            - Cartesian: ['x', 'y', 'z', 'vx', 'vy', 'vz']
            - Equinoctial: ['p', 'f', 'g', 'h', 'k', 'L']
            - CR3BP: ['x_nd', 'y_nd', 'z_nd', 'vx_nd', 'vy_nd', 'vz_nd']
            """
            # pandas isn't needed unless this function is used
            try:
                import pandas as pd
            except ImportError:
                raise ImportError("pandas required for from_dataframe()")
            # check for empty list input and return empty DataFrame
            if not orbits:
                return pd.DataFrame()

            # Check all same type
            elem_type = orbits[0].element_type
            if not all(o.element_type == elem_type for o in orbits):
                raise ValueError("All orbits must have the same element type")
            
            # Validate index length
            if index is not None:
                if len(index) != len(orbits):
                    raise ValueError(
                        f"Index length ({len(index)}) must match "
                        f"number of orbits ({len(orbits)})"
                    )
            
            # Get column names based on element type
            if elem_type == OEType.KEPLERIAN:
                columns = ['a', 'e', 'i', 'RAAN', 'omega', 'nu']
            elif elem_type == OEType.CARTESIAN:
                columns = ['x', 'y', 'z', 'vx', 'vy', 'vz']
            elif elem_type == OEType.EQUINOCTIAL:
                columns = ['p', 'f', 'g', 'h', 'k', 'L']
            elif elem_type == OEType.CR3BP:
                columns = ['x_nd', 'y_nd', 'z_nd', 'vx_nd', 'vy_nd', 'vz_nd']
            else:
                raise ValueError("Orbital Elements must have defined type")
            
            # Create DataFrame
            data = np.array([o.elements for o in orbits])
            df = pd.DataFrame(data, columns=columns, index=index)
            
            return df

    # ========== SPECIAL METHODS ==========
    def __len__(self):
        #Length of element vector (always 6)
        return 6

    def __getitem__(self, key):
        #Allow indexing like orbit[0]
        return self.elements[key]

    def __iter__(self):
        #Allow iteration over elements
        return iter(self.elements)
    
    def __repr__(self):
        #Machine-readable representation
        return f"OrbitalElements({self.elements.tolist()}, {self.element_type})"

    def __str__(self):
        #Human-readable representation
        if self.element_type == OEType.KEPLERIAN:
            a, e, i, omega, w, nu = self.elements
            return (f"Keplerian Elements:\n"
                    f"  a     = {a:12.4f} km\n"
                    f"  e     = {e:12.6f}\n"
                    f"  i     = {np.degrees(i):12.4f}°\n"
                    f"  RAAN  = {np.degrees(omega):12.4f}°\n"
                    f"  ω     = {np.degrees(w):12.4f}°\n"
                    f"  ν     = {np.degrees(nu):12.4f}°")
        
        elif self.element_type == OEType.CARTESIAN:
            r = self.elements[:3]
            v = self.elements[3:]
            return (f"Cartesian Elements:\n"
                    f"  r = [{r[0]:12.4f}, {r[1]:12.4f}, {r[2]:12.4f}] km\n"
                    f"  v = [{v[0]:12.4f}, {v[1]:12.4f}, {v[2]:12.4f}] km/s")
        
        elif self.element_type == OEType.EQUINOCTIAL:
            p, f, g, h, k, L = self.elements
            return (f"Modified Equinoctial Elements:\n"
                    f"  p = {p:12.4f} km\n"
                    f"  f = {f:12.6f}\n"
                    f"  g = {g:12.6f}\n"
                    f"  h = {h:12.6f}\n"
                    f"  k = {k:12.6f}\n"
                    f"  L = {np.degrees(L):12.4f}°")
        
        elif self.element_type == OEType.CR3BP:
            r = self.elements[:3]
            v = self.elements[3:]
            return (f"CR3BP Nondimensional Elements:\n"
                    f"  r = [{r[0]:4.12f}, {r[1]:4.12f}, {r[2]:4.12f}] nd\n"
                    f"  v = [{v[0]:4.12f}, {v[1]:4.12f}, {v[2]:4.12f}] nd")
    
    def __eq__(self, other):
        #Check equality with tolerance
        if not isinstance(other, OrbitalElements):
            return False
        return (self.element_type == other.element_type and 
                np.allclose(self.elements, other.elements,
                           rtol=self._EQUALITY_RTOL,
                           atol=self._EQUALITY_ATOL))
    
    def __hash__(self):
        #Hash with rounding to match equality
        rounded = tuple(round(x, self._HASH_DECIMALS) for x in self.elements)
        return hash((self.element_type, rounded))
    
    def __sizeof__(self):
        #Return memory footprint in bytes
        size = object.__sizeof__(self)  # Base object overhead
        size += self.elements.nbytes    # NumPy array data
        # element_type is just an enum reference, negligible
        return size
    
    # ========== STATIC METHODS ==========
    @staticmethod
    def _parse_element_type(element_type):
        """Convert string or enum to OEType enum"""
        if isinstance(element_type, OEType):
            return element_type
        elif isinstance(element_type, str):
            # Map string to enum
            type_map = {
                'cart': OEType.CARTESIAN,
                'cartesian' : OEType.CARTESIAN,
                'kep': OEType.KEPLERIAN,
                'kepler' : OEType.KEPLERIAN,
                'keplerian' : OEType.KEPLERIAN,
                'eq': OEType.EQUINOCTIAL,
                'equi': OEType.EQUINOCTIAL,
                'equinoctial' : OEType.EQUINOCTIAL,
                'cr3bp' : OEType.CR3BP,
                'CR3BP' : OEType.CR3BP,
                '3body' : OEType.CR3BP
            }
            if element_type in type_map:
                return type_map[element_type]
            else:
                raise ValueError(f"Unknown element type '{element_type}'. "
                           f"Use: {list(type_map.keys())}")
        else:
            raise TypeError(f"element_type must be OEType or str, "
                            f"got {type(element_type)}")
        
    @staticmethod
    def _from_named_params(kwargs):
        """
        Convert named parameters to elements array and detect type.
        
        Returns
        -------
        elements : np.ndarray
            6-element array
        element_type : OEType
            Detected element type
        """
        # Check for Keplerian parameters
        kep_params = ['a', 'e', 'i', 'omega', 'w', 'nu']
        if all(k in kwargs for k in kep_params):
            elements = np.array([kwargs[k] for k in kep_params])
            return elements, OEType.KEPLERIAN
        
        # Check for Cartesian parameters
        cart_params = ['x', 'y', 'z', 'vx', 'vy', 'vz']
        if all(k in kwargs for k in cart_params):
            elements = np.array([kwargs[k] for k in cart_params])
            return elements, OEType.CARTESIAN
        
        # Check for Equinoctial parameters
        equi_params = ['p', 'f', 'g', 'h', 'k', 'L']
        if all(k in kwargs for k in equi_params):
            elements = np.array([kwargs[k] for k in equi_params])
            return elements, OEType.EQUINOCTIAL
        
        # Check for CR3BP parameters
        cr3bp_params = ['x_nd', 'y_nd', 'z_nd', 'vx_nd', 'vy_nd', 'vz_nd']
        if all(k in kwargs for k in cr3bp_params):
            elements = np.array([kwargs[k] for k in cr3bp_params])
            return elements, OEType.CR3BP
        
        # Error: couldn't determine type
        provided = list(kwargs.keys())
        raise ValueError(
            f"Could not determine element type from parameters: {provided}\n"
            f"Keplerian requires: {kep_params}\n"
            f"Cartesian requires: {cart_params}\n"
            f"Equinoctial requires: {equi_params}\n"
            f"CR3BP requires: {cr3bp_params}"
        )