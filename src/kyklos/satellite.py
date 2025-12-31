'''Development code for an orbital trajectory handling package
Satellite class definition
created with the assistance of Claude Sonnet 4.5 by Anthropic'''

import numpy as np
from typing import Optional

class Satellite:
    """
    Represents a satellite's physical properties for dynamics modeling.
    
    Currently supports drag and attitude dynamics. Designed for future
    expansion to include reflectivity models, complex geometries, etc.
    
    Parameters
    ----------
    mass : float
        Satellite mass [kg]
    drag_coeff : float
        Dimensionless drag coefficient (typically 2.0-2.5 for satellites)
    cross_section : float
        Reference cross-sectional area for drag [m^2]
    inertia : np.ndarray
        3x3 inertia tensor [kg⋅m^2] in body frame
    name : str, optional
        Satellite identifier
    """
    # ========== CLASS CONSTANTS ==========
    # Tolerance for floating-point equality comparisons
    _EQUALITY_RTOL = 1e-12  # Relative tolerance (~mm at LEO)
    _EQUALITY_ATOL = 1e-14  # Absolute tolerance
    _HASH_DECIMALS = 10     # Rounding for consistent hashing
    
    def __init__(
        self,
        mass: float,
        drag_coeff: float,
        cross_section: float,
        inertia: np.ndarray,
        name: Optional[str] = None
    ):
        # Validate inputs
        if mass <= 0:
            raise ValueError(f"Mass must be positive, got {mass}")
        if drag_coeff <= 0:
            raise ValueError(f"Drag coefficient must be positive, got {drag_coeff}")
        if cross_section <= 0:
            raise ValueError(f"Cross-sectional area must be positive, "
                             f"got {cross_section}")
        
        inertia = np.asarray(inertia, dtype=float)
        if inertia.shape != (3, 3):
            raise ValueError(f"Inertia tensor must be 3x3, got shape {inertia.shape}")
        
        # Check symmetry
        if not np.allclose(inertia, inertia.T, rtol=self._EQUALITY_RTOL, 
                                               atol=self._EQUALITY_ATOL):
            raise ValueError("Inertia tensor must be symmetric")
        
        # Check positive definiteness (all eigenvalues > 0)
        eigvals = np.linalg.eigvalsh(inertia)  # Hermitian eigenvalues
        if np.any(eigvals <= 0):
            raise ValueError("Inertia tensor must be positive definite")
        
        # Store as immutable (following System pattern)
        self._mass = float(mass)
        self._drag_coeff = float(drag_coeff)
        self._cross_section = float(cross_section)
        self._inertia = inertia.copy()
        self._inertia.flags.writeable = False  # Make array immutable
        self._name = name
        
        # Cache commonly used quantities
        self._inv_inertia = np.linalg.inv(inertia)
        self._inv_inertia.flags.writeable = False
    
    @property
    def mass(self) -> float:
        """Satellite mass [kg]"""
        return self._mass
    
    @property
    def drag_coeff(self) -> float:
        """Drag coefficient (dimensionless)"""
        return self._drag_coeff
    
    @property
    def cross_section(self) -> float:
        """Reference cross-sectional area [m^2]"""
        return self._cross_section
    
    @property
    def inertia(self) -> np.ndarray:
        """Inertia tensor [kg⋅m^2] (read-only)"""
        return self._inertia
    
    @property
    def inv_inertia(self) -> np.ndarray:
        """Inverse inertia tensor [kg^-1⋅m^-2] (read-only)"""
        return self._inv_inertia
    
    @property
    def name(self) -> Optional[str]:
        """Satellite identifier"""
        return self._name
    
    def __repr__(self) -> str:
        name_str = f"'{self.name}'" if self.name else "unnamed"
        return (f"Satellite({name_str}, mass={self.mass:.2f} kg, "
                f"Cd={self.drag_coeff:.2f}, A={self.cross_section:.2f} m²)")
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Satellite):
            return NotImplemented
        
        # Use appropriate tolerances for orbital mechanics
        return (
            np.isclose(self.mass, other.mass, 
                       rtol=self._EQUALITY_RTOL, atol=self._EQUALITY_ATOL) and
            np.isclose(self.drag_coeff, other.drag_coeff, 
                       rtol=self._EQUALITY_RTOL, atol=self._EQUALITY_ATOL) and
            np.isclose(self.cross_section, other.cross_section, 
                       rtol=self._EQUALITY_RTOL, atol=self._EQUALITY_ATOL) and
            np.allclose(self.inertia, other.inertia, 
                        rtol=self._EQUALITY_RTOL, atol=self._EQUALITY_ATOL) and
            self.name == other.name
        )
    
    def __hash__(self) -> int:
        # Round to tolerance for hashing (similar to OrbitalElements)
        mass_rounded = round(self.mass / self._HASH_DECIMALS)
        cd_rounded = round(self.drag_coeff / self._HASH_DECIMALS)
        area_rounded = round(self.cross_section / self._HASH_DECIMALS)
        inertia_rounded = tuple(round(x/self._HASH_DECIMALS) for x in self.inertia.flat)
        
        return hash((mass_rounded, cd_rounded, area_rounded, 
                     inertia_rounded, self.name))