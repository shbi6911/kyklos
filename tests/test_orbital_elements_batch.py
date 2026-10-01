"""
Test suite for OrbitalElements.Batch operations.

Every Batch method is a thin wrapper that maps a per-object operation over a
list of OrbitalElements. The tests therefore use one oracle throughout: the
batch result must equal the per-object result, element by element. This keeps
the suite independent of the underlying orbital mechanics (which are covered
in test_orbital_elements.py) and focused on the wrappers themselves.

Tests include:
1. Conversion wrappers (convert_to, to_cartesian, to_keplerian,
   to_equinoctial) and copy
2. Element accessors (a, e, position, velocity), including the element-type
   restrictions inherited from the single-object properties
3. Derived-quantity wrappers (orbital_period, mean_motion, specific_energy,
   specific_angular_momentum)
4. Export wrappers (to_numpy, to_dataframe)
5. Empty-list and mixed-type error paths

Regression note: Batch.a, Batch.e, Batch.pos and Batch.vel originally called
the single-object properties as if they were methods (o.a(), o.e()) or called
methods that do not exist (o.pos(), o.vel()), so they raised on any input.
Batch.pos and Batch.vel were renamed to Batch.position and Batch.velocity to
match the properties they wrap. The accessor tests below guard that fix.

Created for Kyklos orbital mechanics package.
"""

import numpy as np
import pandas as pd
import pytest

from kyklos import OrbitalElements, OEType

Batch = OrbitalElements.Batch

MU_EARTH = 398600.435507  # km^3/s^2
MU_CR3BP = 0.012150585609624  # Earth-Moon mass ratio (nondimensional)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def kep_orbits():
    """Three valid, distinct elliptic Keplerian orbits."""
    return [
        OrbitalElements(a=7000.0, e=0.001, i=np.deg2rad(28.5),
                        omega=np.deg2rad(10.0), w=np.deg2rad(20.0),
                        nu=np.deg2rad(30.0), mu=MU_EARTH),
        OrbitalElements(a=8500.0, e=0.01, i=np.deg2rad(60.0),
                        omega=np.deg2rad(45.0), w=np.deg2rad(90.0),
                        nu=np.deg2rad(120.0), mu=MU_EARTH),
        OrbitalElements(a=26560.0, e=0.015, i=np.deg2rad(55.0),
                        omega=np.deg2rad(200.0), w=np.deg2rad(270.0),
                        nu=np.deg2rad(300.0), mu=MU_EARTH),
    ]


@pytest.fixture
def cart_orbits(kep_orbits):
    """The Keplerian fixtures expressed in Cartesian elements."""
    return [o.to_cartesian() for o in kep_orbits]


@pytest.fixture
def equi_orbits(kep_orbits):
    """The Keplerian fixtures expressed in equinoctial elements."""
    return [o.to_equinoctial() for o in kep_orbits]


@pytest.fixture
def cr3bp_orbits():
    """Two CR3BP (nondimensional) states. Built without validation."""
    states = [
        np.array([0.8369, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.9, 0.05, 0.01, -0.02, 0.1, 0.005]),
    ]
    return [OrbitalElements(s, OEType.CR3BP, validate=False, mu=MU_CR3BP)
            for s in states]


# =============================================================================
# Conversion wrappers
# =============================================================================

class TestBatchConversion:
    """Batch conversions must match per-object conversions."""

    @pytest.mark.parametrize("target", [OEType.CARTESIAN, OEType.KEPLERIAN,
                                        OEType.EQUINOCTIAL])
    def test_convert_to_matches_single(self, kep_orbits, target):
        batch = Batch.convert_to(kep_orbits, target)
        assert len(batch) == len(kep_orbits)
        for b, o in zip(batch, kep_orbits):
            expected = o.convert_to(target)
            assert b.element_type == target
            assert np.allclose(b.elements, expected.elements,
                               rtol=1e-12, atol=1e-12)

    def test_convert_to_accepts_string_type(self, kep_orbits):
        batch = Batch.convert_to(kep_orbits, 'cart')
        assert all(b.element_type == OEType.CARTESIAN for b in batch)

    def test_to_cartesian(self, kep_orbits):
        batch = Batch.to_cartesian(kep_orbits)
        for b, o in zip(batch, kep_orbits):
            assert b.element_type == OEType.CARTESIAN
            assert np.allclose(b.elements, o.to_cartesian().elements,
                               rtol=1e-12, atol=1e-12)

    def test_to_keplerian(self, cart_orbits):
        batch = Batch.to_keplerian(cart_orbits)
        for b, o in zip(batch, cart_orbits):
            assert b.element_type == OEType.KEPLERIAN
            assert np.allclose(b.elements, o.to_keplerian().elements,
                               rtol=1e-12, atol=1e-12)

    def test_to_equinoctial(self, kep_orbits):
        batch = Batch.to_equinoctial(kep_orbits)
        for b, o in zip(batch, kep_orbits):
            assert b.element_type == OEType.EQUINOCTIAL
            assert np.allclose(b.elements, o.to_equinoctial().elements,
                               rtol=1e-12, atol=1e-12)

    def test_roundtrip_through_batch(self, kep_orbits):
        """kep -> cart -> kep through Batch recovers the original elements."""
        carts = Batch.to_cartesian(kep_orbits)
        keps = Batch.to_keplerian(carts)
        for original, recovered in zip(kep_orbits, keps):
            # a, e, i are unambiguous
            assert np.allclose(original.elements[:3], recovered.elements[:3],
                               rtol=1e-9, atol=1e-9)
            # Angles are compared modulo 2 pi: the conversion returns
            # equivalent angles that may lie on a different 2 pi branch.
            dang = original.elements[3:] - recovered.elements[3:]
            wrapped = (dang + np.pi) % (2 * np.pi) - np.pi
            assert np.allclose(wrapped, 0.0, atol=1e-9)

    def test_conversion_preserves_mu(self, kep_orbits):
        for b in Batch.to_cartesian(kep_orbits):
            assert b.mu == MU_EARTH

    def test_empty_list_returns_empty_list(self):
        assert Batch.to_cartesian([]) == []
        assert Batch.to_keplerian([]) == []
        assert Batch.to_equinoctial([]) == []
        assert Batch.convert_to([], 'cart') == []


class TestBatchCopy:
    """Batch.copy returns equal but independent objects."""

    def test_copy_equal_but_distinct(self, kep_orbits):
        copies = Batch.copy(kep_orbits)
        assert len(copies) == len(kep_orbits)
        for c, o in zip(copies, kep_orbits):
            assert c is not o
            assert c == o

    def test_copy_does_not_share_element_buffer(self, kep_orbits):
        copies = Batch.copy(kep_orbits)
        for c, o in zip(copies, kep_orbits):
            assert not np.shares_memory(c.elements, o.elements)

    def test_copy_empty(self):
        assert Batch.copy([]) == []


# =============================================================================
# Element accessors (regression tests for the o.a() / o.pos() bug)
# =============================================================================

class TestBatchAccessors:
    """Batch.a, e, position, velocity."""

    def test_a_keplerian(self, kep_orbits):
        result = Batch.a(kep_orbits)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)
        assert np.allclose(result, [o.a for o in kep_orbits])
        assert np.allclose(result, [7000.0, 8500.0, 26560.0])

    def test_e_keplerian(self, kep_orbits):
        result = Batch.e(kep_orbits)
        assert result.shape == (3,)
        assert np.allclose(result, [0.001, 0.01, 0.015])

    def test_e_equinoctial(self, equi_orbits, kep_orbits):
        """Eccentricity is defined for equinoctial elements (sqrt(f^2+g^2))."""
        result = Batch.e(equi_orbits)
        assert np.allclose(result, [o.e for o in kep_orbits], atol=1e-12)

    def test_position_cartesian(self, cart_orbits):
        result = Batch.position(cart_orbits)
        assert result.shape == (3, 3)
        for row, o in zip(result, cart_orbits):
            assert np.allclose(row, o.position)

    def test_velocity_cartesian(self, cart_orbits):
        result = Batch.velocity(cart_orbits)
        assert result.shape == (3, 3)
        for row, o in zip(result, cart_orbits):
            assert np.allclose(row, o.velocity)

    def test_position_velocity_cr3bp(self, cr3bp_orbits):
        pos = Batch.position(cr3bp_orbits)
        vel = Batch.velocity(cr3bp_orbits)
        assert pos.shape == (2, 3)
        assert vel.shape == (2, 3)
        assert np.allclose(pos[1], [0.9, 0.05, 0.01])
        assert np.allclose(vel[1], [-0.02, 0.1, 0.005])

    def test_position_velocity_partition_state(self, cart_orbits):
        """position and velocity together reproduce the full state."""
        pos = Batch.position(cart_orbits)
        vel = Batch.velocity(cart_orbits)
        assert np.allclose(np.hstack([pos, vel]), Batch.to_numpy(cart_orbits))

    def test_old_names_removed(self):
        """pos and vel were never functional and are replaced, not aliased."""
        assert not hasattr(Batch, 'pos')
        assert not hasattr(Batch, 'vel')

    # ---- element-type restrictions inherited from the single-object API ----

    def test_a_raises_for_cartesian(self, cart_orbits):
        with pytest.raises(AttributeError):
            Batch.a(cart_orbits)

    def test_e_raises_for_cartesian(self, cart_orbits):
        with pytest.raises(NotImplementedError):
            Batch.e(cart_orbits)

    def test_position_raises_for_keplerian(self, kep_orbits):
        with pytest.raises(AttributeError):
            Batch.position(kep_orbits)

    def test_velocity_raises_for_keplerian(self, kep_orbits):
        with pytest.raises(AttributeError):
            Batch.velocity(kep_orbits)

    def test_accessors_empty_list(self):
        for fn in (Batch.a, Batch.e, Batch.position, Batch.velocity):
            result = fn([])
            assert isinstance(result, np.ndarray)
            assert result.size == 0


# =============================================================================
# Derived-quantity wrappers
# =============================================================================

class TestBatchDerivedQuantities:
    """Batch orbital_period, mean_motion, specific_energy, angular momentum."""

    def test_orbital_period(self, kep_orbits):
        result = Batch.orbital_period(kep_orbits)
        assert result.shape == (3,)
        assert np.allclose(result, [o.orbital_period() for o in kep_orbits])
        # Kepler's third law, independent check on the first orbit
        a = 7000.0
        assert result[0] == pytest.approx(
            2 * np.pi * np.sqrt(a**3 / MU_EARTH), rel=1e-12)

    def test_mean_motion(self, kep_orbits):
        result = Batch.mean_motion(kep_orbits)
        assert np.allclose(result, [o.mean_motion() for o in kep_orbits])

    def test_period_and_mean_motion_consistent(self, kep_orbits):
        """n * T = 2 pi for every member of the batch."""
        n = Batch.mean_motion(kep_orbits)
        T = Batch.orbital_period(kep_orbits)
        assert np.allclose(n * T, 2 * np.pi, rtol=1e-12)

    def test_specific_energy(self, kep_orbits):
        result = Batch.specific_energy(kep_orbits)
        assert np.allclose(result,
                           [o.specific_energy() for o in kep_orbits])
        # Elliptic orbits have negative specific energy, -mu / (2a)
        assert np.all(result < 0)
        assert np.allclose(result, [-MU_EARTH / (2 * o.a)
                                    for o in kep_orbits], rtol=1e-12)

    def test_specific_angular_momentum(self, kep_orbits):
        result = Batch.specific_angular_momentum(kep_orbits)
        assert np.allclose(
            result, [o.specific_angular_momentum() for o in kep_orbits])
        # h = sqrt(mu * a * (1 - e^2)) for a Keplerian orbit
        expected = [np.sqrt(MU_EARTH * o.a * (1 - o.e**2))
                    for o in kep_orbits]
        assert np.allclose(result, expected, rtol=1e-10)

    def test_derived_quantities_same_for_cartesian_input(self, kep_orbits,
                                                        cart_orbits):
        """Results do not depend on which representation the batch is in."""
        assert np.allclose(Batch.specific_energy(cart_orbits),
                           Batch.specific_energy(kep_orbits), rtol=1e-10)
        assert np.allclose(Batch.orbital_period(cart_orbits),
                           Batch.orbital_period(kep_orbits), rtol=1e-10)

    def test_orbital_period_raises_for_cr3bp(self, cr3bp_orbits):
        with pytest.raises(ValueError):
            Batch.orbital_period(cr3bp_orbits)

    def test_derived_quantities_empty_list(self):
        for fn in (Batch.orbital_period, Batch.mean_motion,
                   Batch.specific_energy, Batch.specific_angular_momentum):
            result = fn([])
            assert isinstance(result, np.ndarray)
            assert result.size == 0


# =============================================================================
# Export wrappers
# =============================================================================

class TestBatchExport:
    """Batch.to_numpy and Batch.to_dataframe."""

    def test_to_numpy_shape_and_values(self, kep_orbits):
        array = Batch.to_numpy(kep_orbits)
        assert array.shape == (3, 6)
        for row, o in zip(array, kep_orbits):
            assert np.array_equal(row, o.elements)

    def test_to_numpy_single_orbit(self, kep_orbits):
        assert Batch.to_numpy(kep_orbits[:1]).shape == (1, 6)

    def test_to_numpy_mixed_types_raises(self, kep_orbits, cart_orbits):
        with pytest.raises(ValueError, match="same element type"):
            Batch.to_numpy([kep_orbits[0], cart_orbits[1]])

    def test_to_numpy_returns_independent_array(self, kep_orbits):
        """Mutating the exported array must not alter the orbits."""
        before = kep_orbits[0].elements.copy()
        array = Batch.to_numpy(kep_orbits)
        array[0, 0] = -1.0
        assert np.array_equal(kep_orbits[0].elements, before)

    @pytest.mark.parametrize("fixture_name, columns", [
        ("kep_orbits",    ['a', 'e', 'i', 'omega', 'w', 'nu']),
        ("cart_orbits",   ['x', 'y', 'z', 'vx', 'vy', 'vz']),
        ("equi_orbits",   ['p', 'f', 'g', 'h', 'k', 'L']),
        ("cr3bp_orbits",  ['x_nd', 'y_nd', 'z_nd',
                           'vx_nd', 'vy_nd', 'vz_nd']),
    ])
    def test_to_dataframe_columns(self, request, fixture_name, columns):
        orbits = request.getfixturevalue(fixture_name)
        df = Batch.to_dataframe(orbits)
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == columns
        assert len(df) == len(orbits)
        assert np.allclose(df.to_numpy(), Batch.to_numpy(orbits))

    def test_to_dataframe_custom_index(self, kep_orbits):
        times = [0.0, 60.0, 120.0]
        df = Batch.to_dataframe(kep_orbits, index=times)
        assert list(df.index) == times

    def test_to_dataframe_index_length_mismatch(self, kep_orbits):
        with pytest.raises(ValueError, match="Index length"):
            Batch.to_dataframe(kep_orbits, index=[0.0, 1.0])

    def test_to_dataframe_mixed_types_raises(self, kep_orbits, cart_orbits):
        with pytest.raises(ValueError, match="same element type"):
            Batch.to_dataframe([kep_orbits[0], cart_orbits[1]])

    def test_to_dataframe_empty_list(self):
        df = Batch.to_dataframe([])
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_dataframe_roundtrip(self, kep_orbits):
        """to_dataframe -> from_dataframe recovers the original elements."""
        df = Batch.to_dataframe(kep_orbits)
        recovered = OrbitalElements.from_dataframe(
            df, element_type=OEType.KEPLERIAN, mu=MU_EARTH)
        assert len(recovered) == len(kep_orbits)
        for r, o in zip(recovered, kep_orbits):
            assert np.allclose(r.elements, o.elements)


class TestDataframeTypeInference:
    """from_dataframe infers the element type from to_dataframe's columns."""

    @pytest.mark.parametrize("fixture_name", [
        "cart_orbits", "equi_orbits", "cr3bp_orbits",
    ])
    def test_inference_roundtrip(self, request, fixture_name):
        orbits = request.getfixturevalue(fixture_name)
        df = Batch.to_dataframe(orbits)
        recovered = OrbitalElements.from_dataframe(
            df, validate=False, mu=MU_EARTH)
        assert [r.element_type for r in recovered] == \
               [o.element_type for o in orbits]

    def test_inference_roundtrip_keplerian(self, kep_orbits):
        df = Batch.to_dataframe(kep_orbits)
        recovered = OrbitalElements.from_dataframe(df, mu=MU_EARTH)
        assert all(r.element_type == OEType.KEPLERIAN for r in recovered)


# =============================================================================
# Immutability of inputs
# =============================================================================

class TestBatchDoesNotMutateInputs:
    """Batch operations return new objects and leave their inputs alone."""

    def test_inputs_unchanged_after_batch_operations(self, kep_orbits):
        snapshot = [o.elements.copy() for o in kep_orbits]
        Batch.to_cartesian(kep_orbits)
        Batch.to_equinoctial(kep_orbits)
        Batch.copy(kep_orbits)
        Batch.to_numpy(kep_orbits)
        Batch.a(kep_orbits)
        Batch.orbital_period(kep_orbits)
        for o, before in zip(kep_orbits, snapshot):
            assert o.element_type == OEType.KEPLERIAN
            assert np.array_equal(o.elements, before)
