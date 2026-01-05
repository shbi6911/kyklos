"""
Test suite for System compilation behavior.

Tests cover:
- Lazy vs eager compilation
- Explicit compilation via .compile() method
- is_compiled property
- Multiple systems with independent integrators
"""

import pytest
import numpy as np
from kyklos import System, EARTH, MOON, EARTH_STD_ATMO


class TestEagerCompilation:
    """Test default eager compilation behavior."""
    
    def test_default_compiles_immediately(self):
        """System compiles by default during construction."""
        sys = System('2body', EARTH)
        
        assert sys.is_compiled
    
    def test_compiled_has_cached_integrator(self):
        """Compiled system has non-None integrator."""
        sys = System('2body', EARTH)
        
        # This is testing internal state, but important for the pattern
        assert sys._cached_integrator is not None
    
    def test_compiles_with_perturbations(self):
        """System with perturbations compiles successfully."""
        sys = System('2body', EARTH, perturbations=('J2',))
        
        assert sys.is_compiled
    
    def test_compiles_with_drag(self):
        """System with drag compiles successfully."""
        sys = System('2body', EARTH,
                    perturbations=('drag',),
                    atmosphere=EARTH_STD_ATMO)
        
        assert sys.is_compiled
    
    def test_cr3bp_compiles(self):
        """CR3BP system compiles successfully."""
        sys = System('3body', EARTH,
                    secondary_body=MOON,
                    distance=384400.0)
        
        assert sys.is_compiled


class TestLazyCompilation:
    """Test lazy compilation (compile=False)."""
    
    def test_compile_false_defers_compilation(self):
        """compile=False prevents immediate compilation."""
        sys = System('2body', EARTH, compile=False)
        
        assert not sys.is_compiled
    
    def test_explicit_compile_method(self):
        """Can explicitly compile via .compile() method."""
        sys = System('2body', EARTH, compile=False)
        assert not sys.is_compiled
        
        result = sys.compile()
        
        assert sys.is_compiled
        assert result is sys  # Returns self for chaining
    
    def test_compile_is_idempotent(self):
        """Calling .compile() multiple times is safe."""
        sys = System('2body', EARTH, compile=False)
        
        sys.compile()
        assert sys.is_compiled
        
        # Should not raise or recompile
        sys.compile()
        assert sys.is_compiled
    
    def test_already_compiled_compile_does_nothing(self):
        """Calling .compile() on already-compiled system is no-op."""
        sys = System('2body', EARTH, compile=True)
        integrator_before = sys._cached_integrator
        
        sys.compile()
        
        # Should be same integrator object
        assert sys._cached_integrator is integrator_before


class TestAutoCompilation:
    """Test automatic compilation during propagation."""
    
    def test_propagate_triggers_compilation(self):
        """propagate() automatically compiles if needed."""
        from kyklos import OE
        
        sys = System('2body', EARTH, compile=False)
        assert not sys.is_compiled
        
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        
        # Should auto-compile during propagate
        traj = sys.propagate(orbit, 0, 100)
        
        assert sys.is_compiled
    
    def test_already_compiled_propagate_works(self):
        """propagate() works on already-compiled system."""
        from kyklos import OE
        
        sys = System('2body', EARTH, compile=True)
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        
        # Should work without recompiling
        traj = sys.propagate(orbit, 0, 100)
        
        assert traj is not None


class TestMultipleSystemsIndependence:
    """Test that multiple System instances have independent integrators."""
    
    def test_different_systems_independent_integrators(self):
        """Different systems have different integrator objects."""
        sys1 = System('2body', EARTH)
        sys2 = System('2body', MOON)
        
        assert sys1._cached_integrator is not sys2._cached_integrator
    
    def test_same_config_different_integrators(self):
        """Even identically-configured systems have separate integrators."""
        sys1 = System('2body', EARTH, perturbations=('J2',))
        sys2 = System('2body', EARTH, perturbations=('J2',))
        
        assert sys1._cached_integrator is not sys2._cached_integrator
    
    def test_propagating_one_doesnt_affect_other(self):
        """Propagating one system doesn't affect another."""
        from kyklos import OE
        
        sys1 = System('2body', EARTH)
        sys2 = System('2body', EARTH)
        
        orbit = OE(a=7000, e=0.01, i=0, omega=0, w=0, nu=0)
        
        # Propagate with sys1
        traj1 = sys1.propagate(orbit, 0, 100)
        
        # sys2 should still be in initial state
        # (This is mostly checking internal state doesn't leak)
        traj2 = sys2.propagate(orbit, 0, 200)
        
        assert traj1.tf == 100
        assert traj2.tf == 200


class TestCompilationWithDifferentConfigurations:
    """Test compilation works for various system configurations."""
    
    def test_point_mass_compiles(self):
        """Point mass (no perturbations) compiles."""
        sys = System('2body', EARTH)
        assert sys.is_compiled
    
    def test_j2_only_compiles(self):
        """J2 perturbation compiles."""
        sys = System('2body', EARTH, perturbations=('J2',))
        assert sys.is_compiled
    
    def test_drag_only_compiles(self):
        """Drag perturbation compiles."""
        sys = System('2body', EARTH,
                    perturbations=('drag',),
                    atmosphere=EARTH_STD_ATMO)
        assert sys.is_compiled
    
    def test_j2_plus_drag_compiles(self):
        """Combined J2 + drag compiles."""
        sys = System('2body', EARTH,
                    perturbations=('J2', 'drag'),
                    atmosphere=EARTH_STD_ATMO)
        assert sys.is_compiled
    
    def test_cr3bp_compiles(self):
        """CR3BP compiles."""
        sys = System('3body', EARTH,
                    secondary_body=MOON,
                    distance=384400.0)
        assert sys.is_compiled


class TestCompilationOrder:
    """Test that compilation order doesn't matter."""
    
    def test_compile_before_accessing_eom(self):
        """Can compile before accessing cached_eom."""
        sys = System('2body', EARTH, compile=False)
        
        sys.compile()
        eom = sys.cached_eom
        
        assert eom is not None
        assert sys.is_compiled
    
    def test_access_eom_before_compile(self):
        """Can access cached_eom before compiling."""
        sys = System('2body', EARTH, compile=False)
        
        eom = sys.cached_eom
        assert eom is not None
        assert not sys.is_compiled  # Still not compiled
        
        sys.compile()
        assert sys.is_compiled


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
