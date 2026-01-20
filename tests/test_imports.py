"""Smoke tests to verify package imports work."""

def test_package_imports():
    """Test that all main classes can be imported."""
    from kyklos import OrbitalElements, System, Satellite, Trajectory
    assert OrbitalElements is not None
    assert System is not None
    assert Satellite is not None
    assert Trajectory is not None

def test_version_exists():
    """Test that version is defined."""
    import kyklos
    assert hasattr(kyklos, '__version__')
    assert kyklos.__version__ == "0.1.0"

def test_can_create_orbital_elements():
    """Test basic OrbitalElements creation."""
    from kyklos import OrbitalElements
    oe = OrbitalElements([7000,0.01,0.1,0,0,0],'kep')
    assert oe.a == 7000

def test_can_create_system():
    """Test basic System creation."""
    from kyklos import System, EARTH
    sys = System('2body',EARTH)
    assert sys.primary_body.mu == 3.986004415e5

def test_can_create_satellite():
    """Test basic Satellite creation."""
    from kyklos import Satellite
    import numpy as np
    sat = Satellite(mass=1000, 
                    drag_coeff=2.2, 
                    cross_section=10.0, 
                    inertia=np.array([[100,0,0],[0,50,0],[0,0,25]]))
    assert sat.mass == 1000