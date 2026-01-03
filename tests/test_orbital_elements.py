"""
Test suite for OrbitalElements class conversion functions.

Tests include:
1. Roundtrip conversions (kep->cart->kep, etc.)
2. Validation against MATLAB reference data
3. Edge cases and special orbits

Created for Kyklos orbital mechanics package.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

# Import orbital_elements (will use our local version with mock system module)
from kyklos import OrbitalElements, OEType


# =============================================================================
# Test Configuration
# =============================================================================

# Path to test data
DATA_DIR = Path(__file__).parent / "data"

# Tolerance for numerical comparisons
# These are based on the class constants in OrbitalElements
RTOL = 1e-12  # Relative tolerance (~mm at LEO)
ATOL = 1e-14  # Absolute tolerance

# For angular comparisons (degrees)
ANGLE_RTOL = 1e-10
ANGLE_ATOL = 1e-12

# Earth's gravitational parameter (matching MATLAB test data)
MU_EARTH = 398600.435507  # km³/s²


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def matlab_kep_to_cart():
    """Load MATLAB reference data for Keplerian to Cartesian conversion."""
    csv_path = DATA_DIR / "kep_to_cart.txt"
    if not csv_path.exists():
        pytest.skip(f"Test data not found: {csv_path}")
    return pd.read_csv(csv_path)


@pytest.fixture
def matlab_kep_to_equi():
    """Load MATLAB reference data for Keplerian to Equinoctial conversion."""
    csv_path = DATA_DIR / "kep_to_equi.txt"
    if not csv_path.exists():
        pytest.skip(f"Test data not found: {csv_path}")
    return pd.read_csv(csv_path)


@pytest.fixture
def matlab_cart_to_kep():
    """Load MATLAB reference data for Cartesian to Keplerian conversion."""
    csv_path = DATA_DIR / "cart_to_kep.txt"
    if not csv_path.exists():
        pytest.skip(f"Test data not found: {csv_path}")
    return pd.read_csv(csv_path)


@pytest.fixture
def matlab_cart_to_equi():
    """Load MATLAB reference data for Cartesian to Equinoctial conversion."""
    csv_path = DATA_DIR / "cart_to_equi.txt"
    if not csv_path.exists():
        pytest.skip(f"Test data not found: {csv_path}")
    return pd.read_csv(csv_path)


@pytest.fixture
def matlab_equi_to_kep():
    """Load MATLAB reference data for Equinoctial to Keplerian conversion."""
    csv_path = DATA_DIR / "equi_to_kep.txt"
    if not csv_path.exists():
        pytest.skip(f"Test data not found: {csv_path}")
    return pd.read_csv(csv_path)


@pytest.fixture
def matlab_equi_to_cart():
    """Load MATLAB reference data for Equinoctial to Cartesian conversion."""
    csv_path = DATA_DIR / "equi_to_cart.txt"
    if not csv_path.exists():
        pytest.skip(f"Test data not found: {csv_path}")
    return pd.read_csv(csv_path)


# =============================================================================
# Test Roundtrip Conversions
# =============================================================================

class TestRoundtripConversions:
    """Test that conversions are self-consistent (A->B->A should equal A)."""
    
    @pytest.mark.parametrize("orbit_idx", range(18))
    def test_kep_to_cart_to_kep(self, matlab_kep_to_cart, orbit_idx):
        """Test Keplerian -> Cartesian -> Keplerian roundtrip."""
        # Get original Keplerian elements from MATLAB data
        row = matlab_kep_to_cart.iloc[orbit_idx]
        kep_orig = np.array([row['a'], row['e'], row['i'], 
                             row['omega'], row['w'], row['nu']])
        
        # Create OrbitalElements object
        oe_kep = OrbitalElements(kep_orig, OEType.KEPLERIAN, 
                                 validate=False, mu=MU_EARTH)
        
        # Convert to Cartesian and back
        oe_cart = oe_kep.to_cartesian()
        oe_kep_final = oe_cart.to_keplerian()
        
        # Check roundtrip accuracy
        # a and e should always match
        assert np.allclose(oe_kep_final.elements[0], kep_orig[0], 
                          rtol=RTOL, atol=ATOL), \
            f"Semi-major axis mismatch: {oe_kep_final.elements[0]} vs {kep_orig[0]}"
        
        assert np.allclose(oe_kep_final.elements[1], kep_orig[1], 
                          rtol=RTOL, atol=ATOL), \
            f"Eccentricity mismatch: {oe_kep_final.elements[1]} vs {kep_orig[1]}"
        
        # Detect singular cases
        e = kep_orig[1]
        i = kep_orig[2]
        
        is_circular = e < 1e-6
        is_equatorial = np.abs(i) < 1e-6 or np.abs(i - np.pi) < 1e-6
        
        # Inclination should match unless equatorial
        if not is_equatorial:
            angle_diff = np.abs(oe_kep_final.elements[2] - kep_orig[2])
            assert (angle_diff < ANGLE_ATOL or 
                   np.abs(angle_diff - 2*np.pi) < ANGLE_ATOL), \
                f"Orbit {orbit_idx}: i mismatch: {oe_kep_final.elements[2]} vs {kep_orig[2]}"
        
        # RAAN is undefined for equatorial orbits
        if not is_equatorial:
            angle_diff = np.abs(oe_kep_final.elements[3] - kep_orig[3])
            assert (angle_diff < ANGLE_ATOL or 
                   np.abs(angle_diff - 2*np.pi) < ANGLE_ATOL), \
                f"Orbit {orbit_idx}: Omega mismatch: {oe_kep_final.elements[3]} vs {kep_orig[3]}"
        
        # Arg of periapsis is undefined for circular orbits
        if not is_circular:
            angle_diff = np.abs(oe_kep_final.elements[4] - kep_orig[4])
            assert (angle_diff < ANGLE_ATOL or 
                   np.abs(angle_diff - 2*np.pi) < ANGLE_ATOL), \
                f"Orbit {orbit_idx}: w mismatch: {oe_kep_final.elements[4]} vs {kep_orig[4]}"
        
        # True anomaly is undefined for circular orbits (use true longitude instead)
        # But for roundtrip, the Cartesian position should be preserved
        # So we check via Cartesian elements instead
        cart_roundtrip = oe_kep_final.to_cartesian()
        
        # Use looser tolerance for circular equatorial orbits due to numerical noise
        # in undefined angles
        if is_circular and is_equatorial:
            # For circular equatorial, use position tolerance of 1mm and velocity of 1mm/s
            assert np.allclose(cart_roundtrip.elements[:3], oe_cart.elements[:3],
                              rtol=RTOL, atol=1e-6), \
                f"Orbit {orbit_idx}: Position not preserved (circular equatorial)"
            assert np.allclose(cart_roundtrip.elements[3:], oe_cart.elements[3:],
                              rtol=RTOL, atol=1e-9), \
                f"Orbit {orbit_idx}: Velocity not preserved (circular equatorial)"
        else:
            assert np.allclose(cart_roundtrip.elements, oe_cart.elements,
                              rtol=RTOL, atol=ATOL), \
                f"Orbit {orbit_idx}: Cartesian state not preserved in roundtrip"
    
    @pytest.mark.parametrize("orbit_idx", range(18))
    def test_kep_to_equi_to_kep(self, matlab_kep_to_equi, orbit_idx):
        """Test Keplerian -> Equinoctial -> Keplerian roundtrip."""
        row = matlab_kep_to_equi.iloc[orbit_idx]
        kep_orig = np.array([row['a'], row['e'], row['i'], 
                             row['omega'], row['w'], row['nu']])
        
        oe_kep = OrbitalElements(kep_orig, OEType.KEPLERIAN, 
                                validate=False, mu=MU_EARTH)
        oe_equi = oe_kep.to_equinoctial()
        oe_kep_final = oe_equi.to_keplerian()
        
        # Detect singular cases
        e = kep_orig[1]
        i = kep_orig[2]
        is_circular = e < 1e-6
        is_equatorial = np.abs(i) < 1e-6 or np.abs(i - np.pi) < 1e-6
        
        # Check roundtrip (a, e should always match)
        assert np.allclose(oe_kep_final.elements[0], kep_orig[0], 
                          rtol=RTOL, atol=ATOL)
        assert np.allclose(oe_kep_final.elements[1], kep_orig[1], 
                          rtol=RTOL, atol=ATOL)
        
        # Inclination (unless equatorial)
        if not is_equatorial:
            angle_diff = np.abs(oe_kep_final.elements[2] - kep_orig[2])
            angle_diff_mod = angle_diff % (2*np.pi)
            assert (angle_diff_mod < ANGLE_ATOL or 
                   np.abs(angle_diff_mod - 2*np.pi) < ANGLE_ATOL)
        
        # RAAN (undefined for equatorial)
        if not is_equatorial:
            angle_diff = np.abs(oe_kep_final.elements[3] - kep_orig[3])
            angle_diff_mod = angle_diff % (2*np.pi)
            assert (angle_diff_mod < ANGLE_ATOL or 
                   np.abs(angle_diff_mod - 2*np.pi) < ANGLE_ATOL)
        
        # Arg of periapsis (undefined for circular)
        if not is_circular:
            angle_diff = np.abs(oe_kep_final.elements[4] - kep_orig[4])
            angle_diff_mod = angle_diff % (2*np.pi)
            assert (angle_diff_mod < ANGLE_ATOL or 
                   np.abs(angle_diff_mod - 2*np.pi) < ANGLE_ATOL)
        
        # For circular/equatorial orbits, verify position via Cartesian
        if is_circular or is_equatorial:
            cart_orig = oe_kep.to_cartesian()
            cart_final = oe_kep_final.to_cartesian()
            if is_circular and is_equatorial:
                assert np.allclose(cart_final.elements[:3], cart_orig.elements[:3],
                                  rtol=RTOL, atol=1e-6)
                assert np.allclose(cart_final.elements[3:], cart_orig.elements[3:],
                                  rtol=RTOL, atol=1e-9)
            else:
                assert np.allclose(cart_final.elements, cart_orig.elements,
                                  rtol=RTOL, atol=ATOL)
    
    @pytest.mark.parametrize("orbit_idx", range(18))
    def test_cart_to_kep_to_cart(self, matlab_cart_to_kep, orbit_idx):
        """Test Cartesian -> Keplerian -> Cartesian roundtrip."""
        row = matlab_cart_to_kep.iloc[orbit_idx]
        cart_orig = np.array([row['x'], row['y'], row['z'],
                             row['vx'], row['vy'], row['vz']])
        
        oe_cart = OrbitalElements(cart_orig, OEType.CARTESIAN, 
                                  validate=False, mu=MU_EARTH)
        oe_kep = oe_cart.to_keplerian()
        oe_cart_final = oe_kep.to_cartesian()
        
        # Check if circular equatorial (angles undefined)
        e = oe_kep.elements[1]
        i = oe_kep.elements[2]
        is_circular = e < 1e-6
        is_equatorial = np.abs(i) < 1e-6 or np.abs(i - np.pi) < 1e-6
        
        # For circular equatorial, use looser tolerance
        if is_circular and is_equatorial:
            assert np.allclose(oe_cart_final.elements[:3], cart_orig[:3],
                              rtol=RTOL, atol=1e-6), \
                f"Orbit {orbit_idx}: Position not preserved (circular equatorial)"
            assert np.allclose(oe_cart_final.elements[3:], cart_orig[3:],
                              rtol=RTOL, atol=1e-9), \
                f"Orbit {orbit_idx}: Velocity not preserved (circular equatorial)"
        else:
            # Cartesian elements should match exactly
            assert np.allclose(oe_cart_final.elements, cart_orig, 
                              rtol=RTOL, atol=ATOL), \
                f"Roundtrip failed:\nOriginal: {cart_orig}\nFinal: {oe_cart_final.elements}"
    
    @pytest.mark.parametrize("orbit_idx", range(18))
    def test_cart_to_equi_to_cart(self, matlab_cart_to_equi, orbit_idx):
        """Test Cartesian -> Equinoctial -> Cartesian roundtrip."""
        row = matlab_cart_to_equi.iloc[orbit_idx]
        cart_orig = np.array([row['x'], row['y'], row['z'],
                             row['vx'], row['vy'], row['vz']])
        
        oe_cart = OrbitalElements(cart_orig, OEType.CARTESIAN, 
                                  validate=False, mu=MU_EARTH)
        oe_equi = oe_cart.to_equinoctial()
        oe_cart_final = oe_equi.to_cartesian()
        
        # Check if circular equatorial
        # For equinoctial: e = sqrt(f^2 + g^2), i from h,k
        f, g, h, k = oe_equi.elements[1:5]
        e = np.sqrt(f**2 + g**2)
        i = 2 * np.arctan(np.sqrt(h**2 + k**2))
        
        is_circular = e < 1e-6
        is_equatorial = np.abs(i) < 1e-6 or np.abs(i - np.pi) < 1e-6
        
        # For circular equatorial, use looser tolerance
        if is_circular and is_equatorial:
            assert np.allclose(oe_cart_final.elements[:3], cart_orig[:3],
                              rtol=RTOL, atol=1e-6), \
                f"Orbit {orbit_idx}: Position not preserved (circular equatorial)"
            assert np.allclose(oe_cart_final.elements[3:], cart_orig[3:],
                              rtol=RTOL, atol=1e-9), \
                f"Orbit {orbit_idx}: Velocity not preserved (circular equatorial)"
        else:
            assert np.allclose(oe_cart_final.elements, cart_orig, 
                              rtol=RTOL, atol=ATOL)
    
    @pytest.mark.parametrize("orbit_idx", range(18))
    def test_equi_to_kep_to_equi(self, matlab_equi_to_kep, orbit_idx):
        """Test Equinoctial -> Keplerian -> Equinoctial roundtrip."""
        row = matlab_equi_to_kep.iloc[orbit_idx]
        equi_orig = np.array([row['p'], row['f'], row['g'],
                             row['h'], row['k'], row['L']])
        
        oe_equi = OrbitalElements(equi_orig, OEType.EQUINOCTIAL, 
                                  validate=False, mu=MU_EARTH)
        oe_kep = oe_equi.to_keplerian()
        oe_equi_final = oe_kep.to_equinoctial()
        
        # Equinoctial elements should match exactly (with L angle wrapping)
        assert np.allclose(oe_equi_final.elements[:5], equi_orig[:5], 
                          rtol=RTOL, atol=ATOL)
        
        # Handle L (true longitude) wrapping - can wrap by multiple of 2π
        L_diff = np.abs(oe_equi_final.elements[5] - equi_orig[5])
        L_diff_mod = L_diff % (2*np.pi)
        assert (L_diff_mod < ANGLE_ATOL or 
               np.abs(L_diff_mod - 2*np.pi) < ANGLE_ATOL)
    
    @pytest.mark.parametrize("orbit_idx", range(18))
    def test_equi_to_cart_to_equi(self, matlab_equi_to_cart, orbit_idx):
        """Test Equinoctial -> Cartesian -> Equinoctial roundtrip."""
        row = matlab_equi_to_cart.iloc[orbit_idx]
        equi_orig = np.array([row['p'], row['f'], row['g'],
                             row['h'], row['k'], row['L']])
        
        oe_equi = OrbitalElements(equi_orig, OEType.EQUINOCTIAL, 
                                  validate=False, mu=MU_EARTH)
        oe_cart = oe_equi.to_cartesian()
        oe_equi_final = oe_cart.to_equinoctial()
        
        assert np.allclose(oe_equi_final.elements[:5], equi_orig[:5], 
                          rtol=RTOL, atol=ATOL)
        
        # L can wrap by any multiple of 2π
        L_diff = np.abs(oe_equi_final.elements[5] - equi_orig[5])
        L_diff_mod = L_diff % (2*np.pi)  # Reduce to [0, 2π]
        assert (L_diff_mod < ANGLE_ATOL or 
               np.abs(L_diff_mod - 2*np.pi) < ANGLE_ATOL), \
            f"L mismatch: {oe_equi_final.elements[5]} vs {equi_orig[5]} (diff={L_diff})"


# =============================================================================
# Test Against MATLAB Reference Data
# =============================================================================

class TestMATLABReference:
    """Test conversions against validated MATLAB implementation."""
    
    @pytest.mark.parametrize("orbit_idx", range(18))
    def test_kep_to_cart_matlab(self, matlab_kep_to_cart, orbit_idx):
        """Compare Keplerian to Cartesian conversion with MATLAB."""
        row = matlab_kep_to_cart.iloc[orbit_idx]
        
        # Input: Keplerian
        kep = np.array([row['a'], row['e'], row['i'], 
                       row['omega'], row['w'], row['nu']])
        
        # Expected output: Cartesian (from MATLAB)
        cart_matlab = np.array([row['x'], row['y'], row['z'],
                               row['vx'], row['vy'], row['vz']])
        
        # Python conversion
        oe = OrbitalElements(kep, OEType.KEPLERIAN, validate=False, mu=MU_EARTH)
        cart_python = oe.to_cartesian().elements
        
        # Compare
        assert np.allclose(cart_python, cart_matlab, rtol=RTOL, atol=ATOL), \
            f"Orbit {orbit_idx} Kep->Cart mismatch:\nMATLAB: {cart_matlab}\nPython: {cart_python}\nDiff: {cart_python - cart_matlab}"
    
    @pytest.mark.parametrize("orbit_idx", range(18))
    def test_kep_to_equi_matlab(self, matlab_kep_to_equi, orbit_idx):
        """Compare Keplerian to Equinoctial conversion with MATLAB."""
        row = matlab_kep_to_equi.iloc[orbit_idx]
        
        kep = np.array([row['a'], row['e'], row['i'], 
                       row['omega'], row['w'], row['nu']])
        equi_matlab = np.array([row['p'], row['f'], row['g'],
                               row['h'], row['k'], row['L']])
        
        oe = OrbitalElements(kep, OEType.KEPLERIAN, validate=False, mu=MU_EARTH)
        equi_python = oe.to_equinoctial().elements
        
        # For equinoctial, L might have 2π ambiguity
        assert np.allclose(equi_python[:5], equi_matlab[:5], 
                          rtol=RTOL, atol=ATOL), \
            f"Orbit {orbit_idx} Kep->Equi mismatch (p,f,g,h,k)"
        
        L_diff = np.abs(equi_python[5] - equi_matlab[5])
        assert (L_diff < ANGLE_ATOL or np.abs(L_diff - 2*np.pi) < ANGLE_ATOL), \
            f"Orbit {orbit_idx} Kep->Equi L mismatch: {equi_python[5]} vs {equi_matlab[5]}"
    
    @pytest.mark.parametrize("orbit_idx", range(18))
    def test_cart_to_kep_matlab(self, matlab_cart_to_kep, orbit_idx):
        """Compare Cartesian to Keplerian conversion with MATLAB."""
        row = matlab_cart_to_kep.iloc[orbit_idx]
        
        cart = np.array([row['x'], row['y'], row['z'],
                        row['vx'], row['vy'], row['vz']])
        kep_matlab = np.array([row['a'], row['e'], row['i'],
                              row['omega'], row['w'], row['nu']])
        
        oe = OrbitalElements(cart, OEType.CARTESIAN, validate=False, mu=MU_EARTH)
        kep_python = oe.to_keplerian().elements
        
        # a, e should match exactly
        assert np.allclose(kep_python[0], kep_matlab[0], 
                          rtol=RTOL, atol=ATOL), \
            f"Orbit {orbit_idx} Cart->Kep 'a' mismatch"
        assert np.allclose(kep_python[1], kep_matlab[1], 
                          rtol=RTOL, atol=ATOL), \
            f"Orbit {orbit_idx} Cart->Kep 'e' mismatch"
        
        # Angles with 2π wrapping
        for idx, name in [(2, 'i'), (3, 'Omega'), (4, 'w'), (5, 'nu')]:
            angle_diff = np.abs(kep_python[idx] - kep_matlab[idx])
            assert (angle_diff < ANGLE_ATOL or 
                   np.abs(angle_diff - 2*np.pi) < ANGLE_ATOL), \
                f"Orbit {orbit_idx} Cart->Kep '{name}' mismatch: {kep_python[idx]} vs {kep_matlab[idx]}"
    
    @pytest.mark.parametrize("orbit_idx", range(18))
    def test_cart_to_equi_matlab(self, matlab_cart_to_equi, orbit_idx):
        """Compare Cartesian to Equinoctial conversion with MATLAB."""
        row = matlab_cart_to_equi.iloc[orbit_idx]
        
        cart = np.array([row['x'], row['y'], row['z'],
                        row['vx'], row['vy'], row['vz']])
        equi_matlab = np.array([row['p'], row['f'], row['g'],
                               row['h'], row['k'], row['L']])
        
        oe = OrbitalElements(cart, OEType.CARTESIAN, validate=False, mu=MU_EARTH)
        equi_python = oe.to_equinoctial().elements
        
        assert np.allclose(equi_python[:5], equi_matlab[:5], 
                          rtol=RTOL, atol=ATOL)
        
        L_diff = np.abs(equi_python[5] - equi_matlab[5])
        assert (L_diff < ANGLE_ATOL or np.abs(L_diff - 2*np.pi) < ANGLE_ATOL)
    
    @pytest.mark.parametrize("orbit_idx", range(18))
    def test_equi_to_kep_matlab(self, matlab_equi_to_kep, orbit_idx):
        """Compare Equinoctial to Keplerian conversion with MATLAB."""
        row = matlab_equi_to_kep.iloc[orbit_idx]
        
        equi = np.array([row['p'], row['f'], row['g'],
                        row['h'], row['k'], row['L']])
        kep_matlab = np.array([row['a'], row['e'], row['i'],
                              row['omega'], row['w'], row['nu']])
        
        oe = OrbitalElements(equi, OEType.EQUINOCTIAL, validate=False, mu=MU_EARTH)
        kep_python = oe.to_keplerian().elements
        
        assert np.allclose(kep_python[0], kep_matlab[0], rtol=RTOL, atol=ATOL)
        assert np.allclose(kep_python[1], kep_matlab[1], rtol=RTOL, atol=ATOL)
        
        for idx in [2, 3, 4, 5]:
            angle_diff = np.abs(kep_python[idx] - kep_matlab[idx])
            assert (angle_diff < ANGLE_ATOL or 
                   np.abs(angle_diff - 2*np.pi) < ANGLE_ATOL)
    
    @pytest.mark.parametrize("orbit_idx", range(18))
    def test_equi_to_cart_matlab(self, matlab_equi_to_cart, orbit_idx):
        """Compare Equinoctial to Cartesian conversion with MATLAB."""
        row = matlab_equi_to_cart.iloc[orbit_idx]
        
        equi = np.array([row['p'], row['f'], row['g'],
                        row['h'], row['k'], row['L']])
        cart_matlab = np.array([row['x'], row['y'], row['z'],
                               row['vx'], row['vy'], row['vz']])
        
        oe = OrbitalElements(equi, OEType.EQUINOCTIAL, validate=False, mu=MU_EARTH)
        cart_python = oe.to_cartesian().elements
        
        assert np.allclose(cart_python, cart_matlab, rtol=RTOL, atol=ATOL), \
            f"Orbit {orbit_idx} Equi->Cart mismatch:\nMATLAB: {cart_matlab}\nPython: {cart_python}"


# =============================================================================
# Test Edge Cases and Special Orbits
# =============================================================================

class TestEdgeCases:
    """Test special cases that might cause numerical issues."""
    
    def test_circular_equatorial(self):
        """Circular equatorial orbit (e=0, i=0) has angle singularities."""
        # RAAN and arg of periapsis are undefined for circular equatorial
        kep = np.array([7000.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        oe = OrbitalElements(kep, OEType.KEPLERIAN, validate=False, mu=MU_EARTH)
        cart = oe.to_cartesian()
        kep_back = cart.to_keplerian()
        
        # Semi-major axis and eccentricity should be preserved
        assert np.allclose(kep_back.elements[0], 7000.0, rtol=RTOL)
        assert np.allclose(kep_back.elements[1], 0.0, atol=ATOL)
        assert np.allclose(kep_back.elements[2], 0.0, atol=ATOL)
    
    def test_circular_inclined(self):
        """Circular inclined orbit (e=0, i≠0) - arg of periapsis undefined."""
        kep = np.array([8500.0, 0.0, np.deg2rad(60), 
                       np.deg2rad(45), 0.0, np.deg2rad(30)])
        
        oe = OrbitalElements(kep, OEType.KEPLERIAN, validate=False, mu=MU_EARTH)
        cart = oe.to_cartesian()
        kep_back = cart.to_keplerian()
        
        assert np.allclose(kep_back.elements[0], 8500.0, rtol=RTOL)
        assert np.allclose(kep_back.elements[1], 0.0, atol=ATOL)
        assert np.allclose(kep_back.elements[2], np.deg2rad(60), rtol=ANGLE_RTOL)
    
    def test_near_circular_polar(self):
        """Near-circular polar orbit - tests numerical stability."""
        kep = np.array([8500.0, 0.001, np.deg2rad(90), 
                       np.deg2rad(60), np.deg2rad(30), np.deg2rad(120)])
        
        oe = OrbitalElements(kep, OEType.KEPLERIAN, validate=False, mu=MU_EARTH)
        equi = oe.to_equinoctial()
        kep_back = equi.to_keplerian()
        
        # Should preserve all elements within tolerance
        assert np.allclose(kep_back.elements[:2], kep[:2], rtol=RTOL, atol=ATOL)
    
    def test_highly_eccentric(self):
        """Molniya-type highly eccentric orbit."""
        kep = np.array([26553.0, 0.737, np.deg2rad(63.4), 
                       np.deg2rad(0), np.deg2rad(0), np.deg2rad(0)])
        
        oe = OrbitalElements(kep, OEType.KEPLERIAN, validate=False, mu=MU_EARTH)
        cart = oe.to_cartesian()
        equi = oe.to_equinoctial()
        
        # Roundtrip through both conversions
        kep_from_cart = cart.to_keplerian()
        kep_from_equi = equi.to_keplerian()
        
        assert np.allclose(kep_from_cart.elements[:2], kep[:2], rtol=RTOL)
        assert np.allclose(kep_from_equi.elements[:2], kep[:2], rtol=RTOL)

# =============================================================================
# Test CR3BP Element Construction
# =============================================================================

class TestCR3BP:
    """Test construction and validation of CR3BP elements."""
    def test_cr3bp_construction(self):
        """Test construction of CR3BP orbital elements."""
        # Representative nondimensional CR3BP state
        # (e.g., near L1 point in Earth-Moon system)
        cr3bp_elements = np.array([0.8, 0.0, 0.0, 0.0, 0.1, 0.0])
        
        # Create CR3BP elements
        oe = OrbitalElements(cr3bp_elements, OEType.CR3BP, validate=False, mu=0.012)
        
        # Verify construction
        assert oe.element_type == OEType.CR3BP
        assert len(oe.elements) == 6
        assert np.allclose(oe.elements, cr3bp_elements)
        
        # Verify properties work
        assert np.allclose(oe.position, cr3bp_elements[:3])
        assert np.allclose(oe.velocity, cr3bp_elements[3:])
        
        # Verify mu is stored (as mass ratio for CR3BP)
        assert oe.mu == 0.012


    def test_cr3bp_construction_with_named_params(self):
        """Test CR3BP construction using named parameters."""
        oe = OrbitalElements(
            x_nd=0.8, y_nd=0.0, z_nd=0.0,
            vx_nd=0.0, vy_nd=0.1, vz_nd=0.0,
            mu=0.012,
            validate=False
        )
        
        assert oe.element_type == OEType.CR3BP
        assert np.allclose(oe.elements, [0.8, 0.0, 0.0, 0.0, 0.1, 0.0])


    def test_cr3bp_validation(self):
        """Test that CR3BP validation catches unreasonable values."""
        # Position magnitude > 10 (nondimensional units) should fail
        with pytest.raises(ValueError, match="Position magnitude is > 10"):
            OrbitalElements(
                [100.0, 0.0, 0.0, 0.0, 0.1, 0.0],
                OEType.CR3BP,
                validate=True,
                mu=0.012
            )


# =============================================================================
# Test Utilities
# =============================================================================

class TestUtilities:
    """Test utility functions and class methods."""
    
    def test_batch_conversion(self):
        """Test batch conversion of multiple orbits."""
        # Create several test orbits
        keps = [
            np.array([7000.0, 0.001, 0.0, 0.0, 0.0, 0.0]),
            np.array([8500.0, 0.01, np.deg2rad(60), 0.0, 0.0, 0.0]),
            np.array([26560.0, 0.015, np.deg2rad(55), 0.0, 0.0, 0.0])
        ]
        
        orbits = [OrbitalElements(k, OEType.KEPLERIAN, validate=False, mu=MU_EARTH) 
                 for k in keps]
        
        # Batch convert to Cartesian
        carts = OrbitalElements.Batch.to_cartesian(orbits)
        
        assert len(carts) == len(orbits)
        assert all(c.element_type == OEType.CARTESIAN for c in carts)
    
    def test_numpy_export(self):
        """Test conversion to NumPy array."""
        keps = [
            np.array([7000.0, 0.001, 0.0, 0.0, 0.0, 0.0]),
            np.array([8500.0, 0.01, np.deg2rad(60), 0.0, 0.0, 0.0])
        ]
        
        orbits = [OrbitalElements(k, OEType.KEPLERIAN, validate=False, mu=MU_EARTH) 
                 for k in keps]
        
        array = OrbitalElements.Batch.to_numpy(orbits)
        
        assert array.shape == (2, 6)
        assert np.allclose(array[0], keps[0])
        assert np.allclose(array[1], keps[1])

    #Tests for orbital property calculation methods.
    
    # Test orbit: LEO circular at 7000 km altitude
    @pytest.fixture
    def leo_circular_kep(self):
        """LEO circular orbit in Keplerian elements."""
        return OrbitalElements(
            [7000.0, 0.0, np.deg2rad(28.5), 0.0, 0.0, 0.0],
            OEType.KEPLERIAN,
            validate=False,
            mu=MU_EARTH
        )
    
    @pytest.fixture
    def leo_circular_cart(self, leo_circular_kep):
        """Same orbit in Cartesian elements."""
        return leo_circular_kep.to_cartesian()
    
    @pytest.fixture
    def leo_circular_equi(self, leo_circular_kep):
        """Same orbit in Equinoctial elements."""
        return leo_circular_kep.to_equinoctial()
    
    @pytest.fixture
    def molniya_kep(self):
        """Molniya orbit (highly eccentric)."""
        return OrbitalElements(
            [26553.0, 0.737, np.deg2rad(63.4), 0.0, 0.0, 0.0],
            OEType.KEPLERIAN,
            validate=False,
            mu=MU_EARTH
        )
    
    @pytest.fixture
    def cr3bp_orbit(self):
        """CR3BP orbit (Earth-Moon system)."""
        return OrbitalElements(
            [0.8, 0.0, 0.0, 0.0, 0.1, 0.0],
            OEType.CR3BP,
            validate=False,
            mu=0.012  # Earth-Moon mass ratio
        )
    
    # ========== ORBITAL PERIOD TESTS ==========
    
    def test_orbital_period_keplerian(self, leo_circular_kep):
        """Test orbital period calculation from Keplerian elements."""
        T = leo_circular_kep.orbital_period()
        
        # Expected: T = 2π√(a³/μ)
        a = 7000.0
        T_expected = 2 * np.pi * np.sqrt(a**3 / MU_EARTH)
        
        assert np.isclose(T, T_expected, rtol=1e-10)
        assert T > 0  # Period should be positive
    
    def test_orbital_period_cartesian(self, leo_circular_cart, leo_circular_kep):
        """Test orbital period from Cartesian elements."""
        T_cart = leo_circular_cart.orbital_period()
        T_kep = leo_circular_kep.orbital_period()
        
        assert np.isclose(T_cart, T_kep, rtol=1e-10)
    
    def test_orbital_period_equinoctial(self, leo_circular_equi, leo_circular_kep):
        """Test orbital period from Equinoctial elements."""
        T_equi = leo_circular_equi.orbital_period()
        T_kep = leo_circular_kep.orbital_period()
        
        assert np.isclose(T_equi, T_kep, rtol=1e-10)
    
    def test_orbital_period_eccentric(self, molniya_kep):
        """Test orbital period for eccentric orbit."""
        T = molniya_kep.orbital_period()
        
        # Molniya: a ≈ 26553 km → T ≈ 12 hours
        T_hours = T / 3600
        assert 11.5 < T_hours < 12.5
    
    def test_orbital_period_hyperbolic_raises(self):
        """Test that hyperbolic orbit raises ValueError."""
        hyperbolic = OrbitalElements(
            [10000.0, 1.5, 0.0, 0.0, 0.0, 0.0],  # e > 1
            OEType.KEPLERIAN,
            validate=False,
            mu=MU_EARTH
        )
        
        with pytest.raises(ValueError, match="parabolic/hyperbolic"):
            hyperbolic.orbital_period()
    
    def test_orbital_period_cr3bp_raises(self, cr3bp_orbit):
        """Test that CR3BP orbit raises ValueError."""
        with pytest.raises(ValueError, match="not available for CR3BP"):
            cr3bp_orbit.orbital_period()
    
    # ========== SPECIFIC ENERGY TESTS ==========
    
    def test_specific_energy_keplerian(self, leo_circular_kep):
        """Test specific energy from Keplerian elements."""
        energy = leo_circular_kep.specific_energy()
        
        # Expected: ε = -μ/(2a)
        a = 7000.0
        energy_expected = -MU_EARTH / (2 * a)
        
        assert np.isclose(energy, energy_expected, rtol=1e-10)
        assert energy < 0  # Elliptic orbit
    
    def test_specific_energy_cartesian(self, leo_circular_cart):
        """Test specific energy from Cartesian elements."""
        energy = leo_circular_cart.specific_energy()
        
        # Calculate manually
        r = leo_circular_cart.elements[:3]
        v = leo_circular_cart.elements[3:]
        r_mag = np.linalg.norm(r)
        v_mag = np.linalg.norm(v)
        energy_expected = v_mag**2 / 2 - MU_EARTH / r_mag
        
        assert np.isclose(energy, energy_expected, rtol=1e-10)
    
    def test_specific_energy_equinoctial(self, leo_circular_equi, leo_circular_kep):
        """Test specific energy from Equinoctial elements."""
        energy_equi = leo_circular_equi.specific_energy()
        energy_kep = leo_circular_kep.specific_energy()
        
        assert np.isclose(energy_equi, energy_kep, rtol=1e-10)
    
    def test_specific_energy_consistency(self, molniya_kep):
        """Test energy is consistent across element types."""
        energy_kep = molniya_kep.specific_energy()
        energy_cart = molniya_kep.to_cartesian().specific_energy()
        energy_equi = molniya_kep.to_equinoctial().specific_energy()
        
        assert np.isclose(energy_kep, energy_cart, rtol=1e-10)
        assert np.isclose(energy_kep, energy_equi, rtol=1e-10)
    
    def test_specific_energy_cr3bp_raises(self, cr3bp_orbit):
        """Test that CR3BP orbit raises ValueError."""
        with pytest.raises(ValueError, match="not applicable for CR3BP"):
            cr3bp_orbit.specific_energy()
    
    # ========== SPECIFIC ANGULAR MOMENTUM TESTS ==========
    
    def test_specific_angular_momentum_keplerian(self, leo_circular_kep):
        """Test angular momentum from Keplerian elements."""
        h = leo_circular_kep.specific_angular_momentum()
        
        # Expected: h = √(μp) where p = a(1-e²)
        a = 7000.0
        e = 0.0
        p = a * (1 - e**2)
        h_expected = np.sqrt(MU_EARTH * p)
        
        assert np.isclose(h, h_expected, rtol=1e-10)
        assert h > 0
    
    def test_specific_angular_momentum_cartesian(self, leo_circular_cart):
        """Test angular momentum from Cartesian elements."""
        h = leo_circular_cart.specific_angular_momentum()
        
        # Calculate manually: h = |r × v|
        r = leo_circular_cart.elements[:3]
        v = leo_circular_cart.elements[3:]
        h_expected = np.linalg.norm(np.cross(r, v))
        
        assert np.isclose(h, h_expected, rtol=1e-10)
    
    def test_specific_angular_momentum_equinoctial(self, leo_circular_equi):
        """Test angular momentum from Equinoctial elements."""
        h = leo_circular_equi.specific_angular_momentum()
        
        # From equinoctial: h = √(μp)
        p = leo_circular_equi.elements[0]
        h_expected = np.sqrt(MU_EARTH * p)
        
        assert np.isclose(h, h_expected, rtol=1e-10)
    
    def test_specific_angular_momentum_consistency(self, molniya_kep):
        """Test angular momentum is consistent across element types."""
        h_kep = molniya_kep.specific_angular_momentum()
        h_cart = molniya_kep.to_cartesian().specific_angular_momentum()
        h_equi = molniya_kep.to_equinoctial().specific_angular_momentum()
        
        assert np.isclose(h_kep, h_cart, rtol=1e-10)
        assert np.isclose(h_kep, h_equi, rtol=1e-10)
    
    def test_specific_angular_momentum_cr3bp_raises(self, cr3bp_orbit):
        """Test that CR3BP orbit raises ValueError."""
        with pytest.raises(ValueError, match="not relevant for CR3BP"):
            cr3bp_orbit.specific_angular_momentum()
    
    # ========== MEAN MOTION TESTS ==========
    
    def test_mean_motion_keplerian(self, leo_circular_kep):
        """Test mean motion from Keplerian elements."""
        n = leo_circular_kep.mean_motion()
        
        # Expected: n = √(μ/a³)
        a = 7000.0
        n_expected = np.sqrt(MU_EARTH / a**3)
        
        assert np.isclose(n, n_expected, rtol=1e-10)
        assert n > 0
    
    def test_mean_motion_cartesian(self, leo_circular_cart, leo_circular_kep):
        """Test mean motion from Cartesian elements."""
        n_cart = leo_circular_cart.mean_motion()
        n_kep = leo_circular_kep.mean_motion()
        
        assert np.isclose(n_cart, n_kep, rtol=1e-10)
    
    def test_mean_motion_equinoctial(self, leo_circular_equi, leo_circular_kep):
        """Test mean motion from Equinoctial elements."""
        n_equi = leo_circular_equi.mean_motion()
        n_kep = leo_circular_kep.mean_motion()
        
        assert np.isclose(n_equi, n_kep, rtol=1e-10)
    
    def test_mean_motion_period_relationship(self, leo_circular_kep):
        """Test that mean motion and period are related by n = 2π/T."""
        n = leo_circular_kep.mean_motion()
        T = leo_circular_kep.orbital_period()
        
        assert np.isclose(n * T, 2 * np.pi, rtol=1e-10)
    
    def test_mean_motion_hyperbolic_raises(self):
        """Test that hyperbolic orbit raises ValueError."""
        hyperbolic = OrbitalElements(
            [10000.0, 1.5, 0.0, 0.0, 0.0, 0.0],
            OEType.KEPLERIAN,
            validate=False,
            mu=MU_EARTH
        )
        
        with pytest.raises(ValueError, match="parabolic/hyperbolic"):
            hyperbolic.mean_motion()
    
    def test_mean_motion_cr3bp_raises(self, cr3bp_orbit):
        """Test that CR3BP orbit raises ValueError."""
        with pytest.raises(ValueError, match="not applicable for CR3BP"):
            cr3bp_orbit.mean_motion()
    
    # ========== JACOBI CONSTANT TESTS ==========
    
    def test_jacobi_const_cr3bp(self, cr3bp_orbit):
        """Test Jacobi constant calculation for CR3BP orbit."""
        C = cr3bp_orbit.jacobi_const()
        
        # Jacobi constant should be a real number
        assert isinstance(C, (int, float, np.number))
        assert np.isfinite(C)
        
        # Calculate manually to verify
        x, y, z, vx, vy, vz = cr3bp_orbit.elements
        mu = cr3bp_orbit.mu
        r1 = np.sqrt((x + mu)**2 + y**2 + z**2)
        r2 = np.sqrt((x - 1 + mu)**2 + y**2 + z**2)
        
        C_expected = (x**2 + y**2) + 2*(1-mu)/r1 + 2*mu/r2 - (vx**2 + vy**2 + vz**2)
        
        assert np.isclose(C, C_expected, rtol=1e-10)
    
    def test_jacobi_const_non_cr3bp_raises(self, leo_circular_kep):
        """Test that non-CR3BP orbit raises ValueError."""
        with pytest.raises(ValueError, match="not defined except for CR3BP"):
            leo_circular_kep.jacobi_const()

# =============================================================================
# Test Special Methods
# =============================================================================

class TestSpecialMethods:
    """Tests for special methods (__eq__, __hash__, __getitem__, __iter__, etc.)."""
    
    @pytest.fixture
    def orbit_a(self):
        """Reference orbit."""
        return OrbitalElements(
            [7000.0, 0.01, np.deg2rad(28.5), 0.0, 0.0, 0.0],
            OEType.KEPLERIAN,
            validate=False,
            mu=MU_EARTH
        )
    
    @pytest.fixture
    def orbit_b(self):
        """Identical orbit (different instance)."""
        return OrbitalElements(
            [7000.0, 0.01, np.deg2rad(28.5), 0.0, 0.0, 0.0],
            OEType.KEPLERIAN,
            validate=False,
            mu=MU_EARTH
        )
    
    @pytest.fixture
    def orbit_different(self):
        """Different orbit."""
        return OrbitalElements(
            [8000.0, 0.02, np.deg2rad(45.0), 0.0, 0.0, 0.0],
            OEType.KEPLERIAN,
            validate=False,
            mu=MU_EARTH
        )
    
    # ========== EQUALITY TESTS ==========
    
    def test_eq_identical_orbits(self, orbit_a, orbit_b):
        """Test that identical orbits are equal."""
        assert orbit_a == orbit_b
        assert orbit_b == orbit_a  # Symmetry
    
    def test_eq_different_orbits(self, orbit_a, orbit_different):
        """Test that different orbits are not equal."""
        assert orbit_a != orbit_different
        assert not (orbit_a == orbit_different)
    
    def test_eq_nearly_equal_within_tolerance(self):
        """Test that nearly equal orbits (within tolerance) are equal."""
        orbit1 = OrbitalElements(
            [7000.0, 0.01, 0.0, 0.0, 0.0, 0.0],
            OEType.KEPLERIAN,
            validate=False,
            mu=MU_EARTH
        )
        # Perturb by amount within tolerance (RTOL = 1e-12)
        orbit2 = OrbitalElements(
            [7000.0 + 1e-9, 0.01 + 1e-14, 0.0, 0.0, 0.0, 0.0],  # ~1e-13 relative
            OEType.KEPLERIAN,
            validate=False,
            mu=MU_EARTH
        )
        
        assert orbit1 == orbit2
    
    def test_eq_nearly_equal_outside_tolerance(self):
        """Test that nearly equal orbits (outside tolerance) are not equal."""
        orbit1 = OrbitalElements(
            [7000.0, 0.01, 0.0, 0.0, 0.0, 0.0],
            OEType.KEPLERIAN,
            validate=False,
            mu=MU_EARTH
        )
        # Perturb by amount outside tolerance
        orbit2 = OrbitalElements(
            [7000.1, 0.01, 0.0, 0.0, 0.0, 0.0],  # 100 m difference
            OEType.KEPLERIAN,
            validate=False,
            mu=MU_EARTH
        )
        
        assert orbit1 != orbit2
    
    def test_eq_different_element_types(self):
        """Test that same orbit in different representations are not equal."""
        kep = OrbitalElements(
            [7000.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            OEType.KEPLERIAN,
            validate=False,
            mu=MU_EARTH
        )
        cart = kep.to_cartesian()
        
        # Different element types should not be equal even if same orbit
        assert kep != cart
    
    def test_eq_with_non_orbital_elements(self, orbit_a):
        """Test equality comparison with non-OrbitalElements object."""
        assert orbit_a != "not an orbit"
        assert orbit_a != [7000.0, 0.01, 0.0, 0.0, 0.0, 0.0]
        assert orbit_a != None
    
    # ========== HASH TESTS ==========
    
    def test_hash_equal_objects_equal_hashes(self, orbit_a, orbit_b):
        """Test that equal objects have equal hashes (critical contract)."""
        assert orbit_a == orbit_b
        assert hash(orbit_a) == hash(orbit_b)
    
    def test_hash_different_objects_different_hashes(self, orbit_a, orbit_different):
        """Test that different objects (likely) have different hashes."""
        # Note: Hash collisions are possible but unlikely
        assert hash(orbit_a) != hash(orbit_different)
    
    def test_hash_usable_in_set(self, orbit_a, orbit_b, orbit_different):
        """Test that OrbitalElements can be used in sets."""
        orbit_set = {orbit_a, orbit_b, orbit_different}
        
        # orbit_a and orbit_b are equal, so set should have 2 elements
        assert len(orbit_set) == 2
        assert orbit_a in orbit_set
        assert orbit_different in orbit_set
    
    def test_hash_usable_as_dict_key(self, orbit_a, orbit_b, orbit_different):
        """Test that OrbitalElements can be used as dict keys."""
        orbit_dict = {
            orbit_a: "LEO 1",
            orbit_different: "LEO 2"
        }
        
        # orbit_b is equal to orbit_a, so should access same value
        assert orbit_dict[orbit_b] == "LEO 1"
        assert orbit_dict[orbit_different] == "LEO 2"
        assert len(orbit_dict) == 2
    
    def test_hash_stable(self, orbit_a):
        """Test that hash is stable across multiple calls."""
        hash1 = hash(orbit_a)
        hash2 = hash(orbit_a)
        hash3 = hash(orbit_a)
        
        assert hash1 == hash2 == hash3
    
    # ========== GETITEM TESTS ==========
    
    def test_getitem_indexing(self, orbit_a):
        """Test that indexing returns correct elements."""
        assert orbit_a[0] == 7000.0  # a
        assert orbit_a[1] == 0.01     # e
        assert np.isclose(orbit_a[2], np.deg2rad(28.5))  # i
        assert orbit_a[5] == 0.0      # nu
    
    def test_getitem_negative_indexing(self, orbit_a):
        """Test negative indexing."""
        assert orbit_a[-1] == orbit_a[5]  # Last element
        assert orbit_a[-2] == orbit_a[4]  # Second to last
        assert orbit_a[-6] == orbit_a[0]  # First element
    
    def test_getitem_slicing(self, orbit_a):
        """Test slicing."""
        # First three elements (position in Cartesian)
        first_three = orbit_a[0:3]
        assert len(first_three) == 3
        assert first_three[0] == 7000.0
        
        # Last three elements
        last_three = orbit_a[3:6]
        assert len(last_three) == 3
        
        # Every other element
        every_other = orbit_a[::2]
        assert len(every_other) == 3
    
    def test_getitem_out_of_bounds(self, orbit_a):
        """Test that out of bounds indexing raises IndexError."""
        with pytest.raises(IndexError):
            _ = orbit_a[6]
        
        with pytest.raises(IndexError):
            _ = orbit_a[100]
        
        with pytest.raises(IndexError):
            _ = orbit_a[-7]
    
    # ========== ITER TESTS ==========
    
    def test_iter_iteration(self, orbit_a):
        """Test that iteration yields all elements."""
        elements = list(orbit_a)
        
        assert len(elements) == 6
        assert elements[0] == 7000.0
        assert elements[1] == 0.01
    
    def test_iter_unpacking(self, orbit_a):
        """Test that unpacking works."""
        a, e, i, omega, w, nu = orbit_a
        
        assert a == 7000.0
        assert e == 0.01
        assert np.isclose(i, np.deg2rad(28.5))
        assert omega == 0.0
        assert w == 0.0
        assert nu == 0.0
    
    def test_iter_for_loop(self, orbit_a):
        """Test iteration in for loop."""
        count = 0
        for elem in orbit_a:
            assert isinstance(elem, (int, float, np.number))
            count += 1
        
        assert count == 6
    
    def test_iter_conversion_to_list(self, orbit_a):
        """Test conversion to list."""
        as_list = list(orbit_a)
        
        assert isinstance(as_list, list)
        assert len(as_list) == 6
        assert np.allclose(as_list, orbit_a.elements)
    
    def test_iter_conversion_to_tuple(self, orbit_a):
        """Test conversion to tuple."""
        as_tuple = tuple(orbit_a)
        
        assert isinstance(as_tuple, tuple)
        assert len(as_tuple) == 6
    
    # ========== REPR AND STR SMOKE TESTS ==========
    
    def test_repr_does_not_crash_keplerian(self):
        """Test that __repr__ doesn't crash for Keplerian."""
        orbit = OrbitalElements(
            [7000.0, 0.01, 0.5, 0.1, 0.2, 0.3],
            OEType.KEPLERIAN,
            validate=False,
            mu=MU_EARTH
        )
        
        r = repr(orbit)
        assert isinstance(r, str)
        assert len(r) > 0
        assert 'OrbitalElements' in r
    
    def test_repr_does_not_crash_cartesian(self):
        """Test that __repr__ doesn't crash for Cartesian."""
        orbit = OrbitalElements(
            [7000.0, 0.0, 0.0, 0.0, 7.5, 0.0],
            OEType.CARTESIAN,
            validate=False,
            mu=MU_EARTH
        )
        
        r = repr(orbit)
        assert isinstance(r, str)
        assert len(r) > 0
    
    def test_repr_does_not_crash_equinoctial(self):
        """Test that __repr__ doesn't crash for Equinoctial."""
        orbit = OrbitalElements(
            [7000.0, 0.01, 0.02, 0.0, 0.0, 1.0],
            OEType.EQUINOCTIAL,
            validate=False,
            mu=MU_EARTH
        )
        
        r = repr(orbit)
        assert isinstance(r, str)
        assert len(r) > 0
    
    def test_str_does_not_crash_keplerian(self):
        """Test that __str__ doesn't crash for Keplerian."""
        orbit = OrbitalElements(
            [7000.0, 0.01, 0.5, 0.1, 0.2, 0.3],
            OEType.KEPLERIAN,
            validate=False,
            mu=MU_EARTH
        )
        
        s = str(orbit)
        assert isinstance(s, str)
        assert len(s) > 0
        assert 'Keplerian' in s
    
    def test_str_does_not_crash_cartesian(self):
        """Test that __str__ doesn't crash for Cartesian."""
        orbit = OrbitalElements(
            [7000.0, 0.0, 0.0, 0.0, 7.5, 0.0],
            OEType.CARTESIAN,
            validate=False,
            mu=MU_EARTH
        )
        
        s = str(orbit)
        assert isinstance(s, str)
        assert len(s) > 0
        assert 'Cartesian' in s
    
    def test_str_does_not_crash_equinoctial(self):
        """Test that __str__ doesn't crash for Equinoctial."""
        orbit = OrbitalElements(
            [7000.0, 0.01, 0.02, 0.0, 0.0, 1.0],
            OEType.EQUINOCTIAL,
            validate=False,
            mu=MU_EARTH
        )
        
        s = str(orbit)
        assert isinstance(s, str)
        assert len(s) > 0
        assert 'Equinoctial' in s
    
    def test_str_does_not_crash_cr3bp(self):
        """Test that __str__ doesn't crash for CR3BP."""
        orbit = OrbitalElements(
            [0.8, 0.0, 0.0, 0.0, 0.1, 0.0],
            OEType.CR3BP,
            validate=False,
            mu=0.012
        )
        
        s = str(orbit)
        assert isinstance(s, str)
        assert len(s) > 0
        assert 'CR3BP' in s