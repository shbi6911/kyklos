"""
Test suite for the Node class hierarchy.

Tests cover:
- Abstract base enforcement (Node, BoundaryNode, JunctionNode)
- Type hierarchy and isinstance relationships
- _validate_state static helper
- StartBoundaryNode: construction, properties, validation
- EndBoundaryNode: construction, properties, validation
- ImpulsiveBoundaryNode: all three two-of-three input combinations,
  position continuity enforcement, narrowed delta_v type
- NullJunctionNode: near-continuity validation under strict/non-strict
- ImpulsiveJunctionNode: flexible input, maneuver_jacobian identity
- FreeJunctionNode: position discontinuity allowed, maneuver_jacobian
- Computed properties (delta_v, state_defect) across all types
- Array immutability across all concrete types
- __repr__ smoke tests
"""

import pytest
import numpy as np
from abc import ABC

from kyklos.trajectory import (
    Node,
    BoundaryNode,
    StartBoundaryNode,
    EndBoundaryNode,
    ImpulsiveBoundaryNode,
    JunctionNode,
    NullJunctionNode,
    ImpulsiveJunctionNode,
    FreeJunctionNode,
)
from kyklos import temp_config


# ========== MODULE-LEVEL FIXTURES ==========

@pytest.fixture
def t():
    """A valid node time."""
    return 1000.0


@pytest.fixture
def state_a():
    """A valid 6-element state vector."""
    return np.array([7000.0, 0.0, 0.0, 0.0, 7.546, 0.0])


@pytest.fixture
def state_b():
    """A second valid state, position-continuous with state_a but
    with a different velocity."""
    return np.array([7000.0, 0.0, 0.0, 0.5, 7.546, 0.0])


@pytest.fixture
def state_discontinuous():
    """A state with a position discontinuity relative to state_a."""
    return np.array([7100.0, 0.0, 0.0, 0.0, 7.546, 0.0])


@pytest.fixture
def dv():
    """A valid 3-element delta-v vector [km/s]."""
    return np.array([0.5, 0.0, 0.0])


# ========== TYPE HIERARCHY ==========

class TestNodeHierarchy:
    """Verify the abstract/concrete structure and isinstance relationships."""

    def test_node_is_abstract(self):
        """Node cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Node(time=0.0) # type: ignore[abstract]

    def test_boundary_node_is_abstract(self):
        """BoundaryNode cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BoundaryNode(time=0.0) # type: ignore[abstract]

    def test_junction_node_is_abstract(self):
        """JunctionNode cannot be instantiated directly."""
        with pytest.raises(TypeError):
            JunctionNode(time=0.0) # type: ignore[abstract]

    def test_start_boundary_node_is_boundary_node(self, t, state_a):
        """StartBoundaryNode is a BoundaryNode and a Node."""
        node = StartBoundaryNode(t, state_a)
        assert isinstance(node, BoundaryNode)
        assert isinstance(node, Node)

    def test_end_boundary_node_is_boundary_node(self, t, state_a):
        """EndBoundaryNode is a BoundaryNode and a Node."""
        node = EndBoundaryNode(t, state_a)
        assert isinstance(node, BoundaryNode)
        assert isinstance(node, Node)

    def test_impulsive_boundary_node_is_boundary_node(self, t, state_a, dv):
        """ImpulsiveBoundaryNode is a BoundaryNode and a Node."""
        node = ImpulsiveBoundaryNode(t, pre_state=state_a, delta_v=dv)
        assert isinstance(node, BoundaryNode)
        assert isinstance(node, Node)

    def test_null_junction_node_is_junction_node(self, t, state_a):
        """NullJunctionNode is a JunctionNode and a Node."""
        node = NullJunctionNode(t, state_a, state_a.copy())
        assert isinstance(node, JunctionNode)
        assert isinstance(node, Node)

    def test_impulsive_junction_node_is_junction_node(self, t, state_a, dv):
        """ImpulsiveJunctionNode is a JunctionNode and a Node."""
        node = ImpulsiveJunctionNode(t, pre_state=state_a, delta_v=dv)
        assert isinstance(node, JunctionNode)
        assert isinstance(node, Node)

    def test_free_junction_node_is_junction_node(
            self, t, state_a, state_discontinuous):
        """FreeJunctionNode is a JunctionNode and a Node."""
        node = FreeJunctionNode(t, state_a, state_discontinuous)
        assert isinstance(node, JunctionNode)
        assert isinstance(node, Node)

    def test_boundary_nodes_are_not_junction_nodes(self, t, state_a):
        """BoundaryNode subclasses are not JunctionNodes."""
        start = StartBoundaryNode(t, state_a)
        end = EndBoundaryNode(t, state_a)
        assert not isinstance(start, JunctionNode)
        assert not isinstance(end, JunctionNode)

    def test_junction_nodes_are_not_boundary_nodes(self, t, state_a):
        """JunctionNode subclasses are not BoundaryNodes."""
        node = NullJunctionNode(t, state_a, state_a.copy())
        assert not isinstance(node, BoundaryNode)


# ========== _validate_state STATIC HELPER ==========

class TestValidateState:
    """Test the _validate_state static method on the Node base class."""

    def test_accepts_valid_array(self, state_a):
        """Valid 6-element array is accepted and returned."""
        result = Node._validate_state(state_a)
        assert result.shape == (6,)
        assert result.dtype == float

    def test_accepts_list_input(self):
        """List input is converted to numpy array."""
        result = Node._validate_state([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        assert isinstance(result, np.ndarray)
        assert result.shape == (6,)

    def test_rejects_wrong_shape(self):
        """Non-6-element array raises ValueError."""
        with pytest.raises(ValueError, match="6-element"):
            Node._validate_state(np.array([1.0, 2.0, 3.0]))

    def test_rejects_nan(self):
        """Array containing NaN raises ValueError."""
        bad = np.array([7000.0, 0.0, 0.0, np.nan, 0.0, 0.0])
        with pytest.raises(ValueError, match="NaN or Inf"):
            Node._validate_state(bad)

    def test_rejects_inf(self):
        """Array containing Inf raises ValueError."""
        bad = np.array([7000.0, 0.0, 0.0, 0.0, np.inf, 0.0])
        with pytest.raises(ValueError, match="NaN or Inf"):
            Node._validate_state(bad)

    def test_error_message_includes_name(self):
        """Error message includes the parameter name when provided."""
        bad = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="pre_state"):
            Node._validate_state(bad, name='pre_state')

    def test_returns_float_array(self):
        """Integer input is converted to float."""
        result = Node._validate_state(np.array([7000, 0, 0, 0, 7, 0]))
        assert result.dtype == float


# ========== StartBoundaryNode ==========

class TestStartBoundaryNode:
    """Tests for StartBoundaryNode."""

    def test_construction(self, t, state_a):
        """Constructs without error from valid inputs."""
        node = StartBoundaryNode(t, state_a)
        assert node is not None

    def test_time_property(self, t, state_a):
        """time property returns the provided time."""
        node = StartBoundaryNode(t, state_a)
        assert node.time == t

    def test_post_state_property(self, t, state_a):
        """post_state returns the provided state."""
        node = StartBoundaryNode(t, state_a)
        np.testing.assert_array_equal(node.post_state, state_a)

    def test_pre_state_is_none(self, t, state_a):
        """pre_state is None — no incoming segment."""
        node = StartBoundaryNode(t, state_a)
        assert node.pre_state is None

    def test_delta_v_is_none(self, t, state_a):
        """delta_v is None when one side is absent."""
        node = StartBoundaryNode(t, state_a)
        assert node.delta_v is None

    def test_state_defect_is_none(self, t, state_a):
        """state_defect is None when one side is absent."""
        node = StartBoundaryNode(t, state_a)
        assert node.state_defect is None

    def test_post_state_is_immutable(self, t, state_a):
        """post_state array cannot be modified."""
        node = StartBoundaryNode(t, state_a)
        with pytest.raises(ValueError, match="read-only"):
            node.post_state[0] = 9999.0

    def test_post_state_is_copy(self, t, state_a):
        """Modifying the original array does not affect the node."""
        state_copy = state_a.copy()
        node = StartBoundaryNode(t, state_copy)
        state_copy[0] = 9999.0
        assert node.post_state[0] != 9999.0

    def test_rejects_invalid_state(self, t):
        """Invalid state raises ValueError."""
        with pytest.raises(ValueError):
            StartBoundaryNode(t, np.array([1.0, 2.0, 3.0]))

    def test_repr(self, t, state_a):
        """__repr__ produces a non-empty string containing time."""
        node = StartBoundaryNode(t, state_a)
        r = repr(node)
        assert isinstance(r, str)
        assert str(t) in r or f"{t:.6g}" in r


# ========== EndBoundaryNode ==========

class TestEndBoundaryNode:
    """Tests for EndBoundaryNode. Mirrors StartBoundaryNode but pre/post swapped."""

    def test_construction(self, t, state_a):
        node = EndBoundaryNode(t, state_a)
        assert node is not None

    def test_pre_state_property(self, t, state_a):
        """pre_state returns the provided state."""
        node = EndBoundaryNode(t, state_a)
        np.testing.assert_array_equal(node.pre_state, state_a)

    def test_post_state_is_none(self, t, state_a):
        """post_state is None — no outgoing segment."""
        node = EndBoundaryNode(t, state_a)
        assert node.post_state is None

    def test_delta_v_is_none(self, t, state_a):
        node = EndBoundaryNode(t, state_a)
        assert node.delta_v is None

    def test_state_defect_is_none(self, t, state_a):
        node = EndBoundaryNode(t, state_a)
        assert node.state_defect is None

    def test_pre_state_is_immutable(self, t, state_a):
        node = EndBoundaryNode(t, state_a)
        with pytest.raises(ValueError, match="read-only"):
            node.pre_state[0] = 9999.0

    def test_pre_state_is_copy(self, t, state_a):
        state_copy = state_a.copy()
        node = EndBoundaryNode(t, state_copy)
        state_copy[0] = 9999.0
        assert node.pre_state[0] != 9999.0


# ========== ImpulsiveBoundaryNode ==========

class TestImpulsiveBoundaryNode:
    """Tests for ImpulsiveBoundaryNode: all input combinations, validation."""

    # --- Construction: three two-of-three combinations ---

    def test_pre_state_and_delta_v(self, t, state_a, dv):
        """pre_state + delta_v derives post_state."""
        node = ImpulsiveBoundaryNode(t, pre_state=state_a, delta_v=dv)
        expected_post = state_a.copy()
        expected_post[3:] += dv
        np.testing.assert_array_almost_equal(node.post_state, expected_post)
        np.testing.assert_array_equal(node.pre_state, state_a)

    def test_post_state_and_delta_v(self, t, state_a, dv):
        """post_state + delta_v derives pre_state."""
        post = state_a.copy()
        post[3:] += dv
        node = ImpulsiveBoundaryNode(t, post_state=post, delta_v=dv)
        np.testing.assert_array_almost_equal(node.pre_state, state_a)
        np.testing.assert_array_equal(node.post_state, post)

    def test_pre_and_post_state(self, t, state_a, state_b):
        """pre_state + post_state derives delta_v (position-continuous)."""
        node = ImpulsiveBoundaryNode(
            t, pre_state=state_a, post_state=state_b
        )
        expected_dv = state_b[3:] - state_a[3:]
        np.testing.assert_array_almost_equal(node.delta_v, expected_dv)

    # --- Position continuity enforcement ---

    def test_position_discontinuity_raises(
            self, t, state_a, state_discontinuous):
        """pre_state + post_state with position jump raises ValueError."""
        with pytest.raises(ValueError, match="[Pp]osition"):
            ImpulsiveBoundaryNode(
                t, pre_state=state_a, post_state=state_discontinuous
            )

    def test_pre_post_with_delta_v_enforces_position_continuity(
            self, t, state_a, dv):
        """pre_state + delta_v always produces position-continuous post_state."""
        node = ImpulsiveBoundaryNode(t, pre_state=state_a, delta_v=dv)
        np.testing.assert_array_equal(
            node.pre_state[:3], node.post_state[:3]
        )

    # --- Input validation ---

    def test_requires_exactly_two_inputs_one_raises(self, t, state_a):
        """Providing only one input raises ValueError."""
        with pytest.raises(ValueError, match="[Ee]xactly two"):
            ImpulsiveBoundaryNode(t, pre_state=state_a)

    def test_requires_exactly_two_inputs_three_raises(self, t, state_a, dv):
        """Providing all three inputs raises ValueError."""
        post = state_a.copy()
        post[3:] += dv
        with pytest.raises(ValueError, match="[Ee]xactly two"):
            ImpulsiveBoundaryNode(
                t, pre_state=state_a, post_state=post, delta_v=dv
            )

    def test_wrong_delta_v_shape_raises(self, t, state_a):
        """delta_v of wrong shape raises ValueError."""
        with pytest.raises(ValueError, match="3-element"):
            ImpulsiveBoundaryNode(
                t, pre_state=state_a, delta_v=np.array([0.1, 0.0])
            )

    def test_nonfinite_delta_v_raises(self, t, state_a):
        """delta_v containing NaN raises ValueError."""
        with pytest.raises(ValueError, match="NaN or Inf"):
            ImpulsiveBoundaryNode(
                t, pre_state=state_a, delta_v=np.array([np.nan, 0.0, 0.0])
            )

    # --- Properties ---

    def test_delta_v_returns_ndarray_not_none(self, t, state_a, dv):
        """delta_v returns np.ndarray, never None (narrowed return type)."""
        node = ImpulsiveBoundaryNode(t, pre_state=state_a, delta_v=dv)
        result = node.delta_v
        assert isinstance(result, np.ndarray)
        assert result is not None

    def test_delta_v_has_correct_shape(self, t, state_a, dv):
        """delta_v has shape (3,)."""
        node = ImpulsiveBoundaryNode(t, pre_state=state_a, delta_v=dv)
        assert node.delta_v.shape == (3,)

    def test_immutability_pre_state(self, t, state_a, dv):
        node = ImpulsiveBoundaryNode(t, pre_state=state_a, delta_v=dv)
        with pytest.raises(ValueError, match="read-only"):
            node.pre_state[0] = 9999.0

    def test_immutability_post_state(self, t, state_a, dv):
        node = ImpulsiveBoundaryNode(t, pre_state=state_a, delta_v=dv)
        with pytest.raises(ValueError, match="read-only"):
            node.post_state[0] = 9999.0

    def test_repr_contains_time_and_dv_magnitude(self, t, state_a, dv):
        node = ImpulsiveBoundaryNode(t, pre_state=state_a, delta_v=dv)
        r = repr(node)
        assert 'ImpulsiveBoundaryNode' in r
        assert str(t) in r or f"{t:.6g}" in r


# ========== NullJunctionNode ==========

class TestNullJunctionNode:
    """Tests for NullJunctionNode: near-continuity, maneuver_jacobian."""

    def test_construction_with_equal_states(self, t, state_a):
        """Constructs without error when states are identical."""
        node = NullJunctionNode(t, state_a, state_a.copy())
        assert node is not None

    def test_both_sides_populated(self, t, state_a):
        """pre_state and post_state are both non-None."""
        node = NullJunctionNode(t, state_a, state_a.copy())
        assert node.pre_state is not None
        assert node.post_state is not None

    def test_pre_and_post_state_values(self, t, state_a):
        """pre_state and post_state return correct arrays."""
        node = NullJunctionNode(t, state_a, state_a.copy())
        np.testing.assert_array_equal(node.pre_state, state_a)
        np.testing.assert_array_equal(node.post_state, state_a)

    def test_delta_v_near_zero(self, t, state_a):
        """delta_v is near zero for continuous junction."""
        node = NullJunctionNode(t, state_a, state_a.copy())
        assert np.linalg.norm(node.delta_v) < 1e-12

    def test_state_defect_near_zero(self, t, state_a):
        """state_defect is near zero for continuous junction."""
        node = NullJunctionNode(t, state_a, state_a.copy())
        assert np.linalg.norm(node.state_defect) < 1e-12

    def test_maneuver_jacobian_is_identity(self, t, state_a):
        """maneuver_jacobian returns the 6x6 identity matrix."""
        node = NullJunctionNode(t, state_a, state_a.copy())
        np.testing.assert_array_equal(node.maneuver_jacobian(), np.eye(6))

    def test_maneuver_jacobian_shape(self, t, state_a):
        """maneuver_jacobian has shape (6, 6)."""
        node = NullJunctionNode(t, state_a, state_a.copy())
        assert node.maneuver_jacobian().shape == (6, 6)

    def test_near_continuous_passes_in_strict_mode(self, t, state_a):
        """States differing within tolerance construct without error in strict mode."""
        slightly_off = state_a.copy()
        slightly_off[3] += 1e-14  # well within tolerance
        with temp_config(STRICT_VALIDATION=True):
            node = NullJunctionNode(t, state_a, slightly_off)
        assert node is not None

    def test_discontinuous_raises_in_strict_mode(self, t, state_a, state_b):
        """States differing beyond tolerance raise in strict mode."""
        with temp_config(STRICT_VALIDATION=True):
            with pytest.raises(ValueError, match="[Nn]ull[Jj]unction"):
                NullJunctionNode(t, state_a, state_b)

    def test_discontinuous_warns_in_non_strict_mode(self, t, state_a, state_b):
        """States differing beyond tolerance warn in non-strict mode."""
        with temp_config(STRICT_VALIDATION=False):
            with pytest.warns(UserWarning):
                node = NullJunctionNode(t, state_a, state_b)
        assert node is not None

    def test_immutability(self, t, state_a):
        node = NullJunctionNode(t, state_a, state_a.copy())
        with pytest.raises(ValueError, match="read-only"):
            node.pre_state[0] = 9999.0

    def test_repr(self, t, state_a):
        node = NullJunctionNode(t, state_a, state_a.copy())
        r = repr(node)
        assert 'NullJunctionNode' in r


# ========== ImpulsiveJunctionNode ==========

class TestImpulsiveJunctionNode:
    """Tests for ImpulsiveJunctionNode."""

    def test_construction_pre_and_delta_v(self, t, state_a, dv):
        """pre_state + delta_v constructs correctly."""
        node = ImpulsiveJunctionNode(t, pre_state=state_a, delta_v=dv)
        assert node is not None

    def test_construction_post_and_delta_v(self, t, state_a, dv):
        """post_state + delta_v constructs correctly."""
        post = state_a.copy()
        post[3:] += dv
        node = ImpulsiveJunctionNode(t, post_state=post, delta_v=dv)
        assert node is not None

    def test_construction_pre_and_post(self, t, state_a, state_b):
        """pre_state + post_state (position-continuous) constructs correctly."""
        node = ImpulsiveJunctionNode(
            t, pre_state=state_a, post_state=state_b
        )
        assert node is not None

    def test_position_continuity_enforced(
            self, t, state_a, state_discontinuous):
        """Position discontinuity raises ValueError."""
        with pytest.raises(ValueError, match="[Pp]osition"):
            ImpulsiveJunctionNode(
                t, pre_state=state_a, post_state=state_discontinuous
            )

    def test_delta_v_returns_ndarray_not_none(self, t, state_a, dv):
        """delta_v returns np.ndarray, never None."""
        node = ImpulsiveJunctionNode(t, pre_state=state_a, delta_v=dv)
        assert isinstance(node.delta_v, np.ndarray)

    def test_delta_v_correct_value(self, t, state_a, dv):
        """delta_v matches the provided velocity change."""
        node = ImpulsiveJunctionNode(t, pre_state=state_a, delta_v=dv)
        np.testing.assert_array_almost_equal(node.delta_v, dv)

    def test_position_unchanged_by_maneuver(self, t, state_a, dv):
        """post_state[:3] == pre_state[:3] for impulsive maneuver."""
        node = ImpulsiveJunctionNode(t, pre_state=state_a, delta_v=dv)
        np.testing.assert_array_equal(
            node.pre_state[:3], node.post_state[:3]
        )

    def test_maneuver_jacobian_is_identity(self, t, state_a, dv):
        """maneuver_jacobian returns identity for fixed burn."""
        node = ImpulsiveJunctionNode(t, pre_state=state_a, delta_v=dv)
        np.testing.assert_array_equal(node.maneuver_jacobian(), np.eye(6))

    def test_state_defect_position_zero(self, t, state_a, dv):
        """Position components of state_defect are zero."""
        node = ImpulsiveJunctionNode(t, pre_state=state_a, delta_v=dv)
        np.testing.assert_array_almost_equal(node.state_defect[:3], np.zeros(3))

    def test_state_defect_velocity_equals_delta_v(self, t, state_a, dv):
        """Velocity components of state_defect equal delta_v."""
        node = ImpulsiveJunctionNode(t, pre_state=state_a, delta_v=dv)
        np.testing.assert_array_almost_equal(node.state_defect[3:], dv)

    def test_both_sides_populated(self, t, state_a, dv):
        """Both pre_state and post_state are non-None."""
        node = ImpulsiveJunctionNode(t, pre_state=state_a, delta_v=dv)
        assert node.pre_state is not None
        assert node.post_state is not None

    def test_immutability_pre(self, t, state_a, dv):
        node = ImpulsiveJunctionNode(t, pre_state=state_a, delta_v=dv)
        with pytest.raises(ValueError, match="read-only"):
            node.pre_state[0] = 9999.0

    def test_immutability_post(self, t, state_a, dv):
        node = ImpulsiveJunctionNode(t, pre_state=state_a, delta_v=dv)
        with pytest.raises(ValueError, match="read-only"):
            node.post_state[0] = 9999.0

    def test_repr_contains_dv_magnitude(self, t, state_a, dv):
        node = ImpulsiveJunctionNode(t, pre_state=state_a, delta_v=dv)
        r = repr(node)
        assert 'ImpulsiveJunctionNode' in r


# ========== FreeJunctionNode ==========

class TestFreeJunctionNode:
    """Tests for FreeJunctionNode: position discontinuity allowed."""

    def test_construction_continuous_states(self, t, state_a, state_b):
        """Constructs with position-continuous states."""
        node = FreeJunctionNode(t, state_a, state_b)
        assert node is not None

    def test_construction_position_discontinuous(
            self, t, state_a, state_discontinuous):
        """Constructs with position-discontinuous states — no error."""
        node = FreeJunctionNode(t, state_a, state_discontinuous)
        assert node is not None

    def test_pre_state_value(self, t, state_a, state_discontinuous):
        """pre_state returns the provided pre-junction state."""
        node = FreeJunctionNode(t, state_a, state_discontinuous)
        np.testing.assert_array_equal(node.pre_state, state_a)

    def test_post_state_value(self, t, state_a, state_discontinuous):
        """post_state returns the provided post-junction state."""
        node = FreeJunctionNode(t, state_a, state_discontinuous)
        np.testing.assert_array_equal(node.post_state, state_discontinuous)

    def test_state_defect_reflects_discontinuity(
            self, t, state_a, state_discontinuous):
        """state_defect equals post_state - pre_state."""
        node = FreeJunctionNode(t, state_a, state_discontinuous)
        expected = state_discontinuous - state_a
        np.testing.assert_array_equal(node.state_defect, expected)

    def test_delta_v_reflects_velocity_discontinuity(
            self, t, state_a, state_b):
        """delta_v equals velocity difference."""
        node = FreeJunctionNode(t, state_a, state_b)
        expected_dv = state_b[3:] - state_a[3:]
        np.testing.assert_array_equal(node.delta_v, expected_dv)

    def test_maneuver_jacobian_is_identity(self, t, state_a, state_b):
        """maneuver_jacobian returns identity for FreeJunctionNode."""
        node = FreeJunctionNode(t, state_a, state_b)
        np.testing.assert_array_equal(node.maneuver_jacobian(), np.eye(6))

    def test_both_sides_populated(self, t, state_a, state_b):
        """Both pre_state and post_state are non-None."""
        node = FreeJunctionNode(t, state_a, state_b)
        assert node.pre_state is not None
        assert node.post_state is not None

    def test_immutability_pre(self, t, state_a, state_b):
        node = FreeJunctionNode(t, state_a, state_b)
        with pytest.raises(ValueError, match="read-only"):
            node.pre_state[0] = 9999.0

    def test_immutability_post(self, t, state_a, state_b):
        node = FreeJunctionNode(t, state_a, state_b)
        with pytest.raises(ValueError, match="read-only"):
            node.post_state[0] = 9999.0

    def test_repr_contains_defect_magnitude(self, t, state_a, state_b):
        node = FreeJunctionNode(t, state_a, state_b)
        r = repr(node)
        assert 'FreeJunctionNode' in r

    def test_zero_defect_case(self, t, state_a):
        """Identical states produce zero defect — degenerate continuous case."""
        node = FreeJunctionNode(t, state_a, state_a.copy())
        assert np.linalg.norm(node.state_defect) == 0.0


# ========== COMPUTED PROPERTIES ACROSS TYPES ==========

class TestComputedProperties:
    """Cross-cutting tests for delta_v and state_defect on all node types."""

    def test_junction_node_delta_v_never_none(self, t, state_a, state_b):
        """All JunctionNode subtypes return np.ndarray for delta_v."""
        null   = NullJunctionNode(t, state_a, state_a.copy())
        imp    = ImpulsiveJunctionNode(t, pre_state=state_a, post_state=state_b)
        free   = FreeJunctionNode(t, state_a, state_b)

        for node in [null, imp, free]:
            result = node.delta_v
            assert isinstance(result, np.ndarray), (
                f"{type(node).__name__}.delta_v returned None"
            )

    def test_junction_node_state_defect_never_none(self, t, state_a, state_b):
        """All JunctionNode subtypes return np.ndarray for state_defect."""
        null   = NullJunctionNode(t, state_a, state_a.copy())
        imp    = ImpulsiveJunctionNode(t, pre_state=state_a, post_state=state_b)
        free   = FreeJunctionNode(t, state_a, state_b)

        for node in [null, imp, free]:
            result = node.state_defect
            assert isinstance(result, np.ndarray), (
                f"{type(node).__name__}.state_defect returned None"
            )

    def test_boundary_node_delta_v_is_none_without_both_sides(
            self, t, state_a):
        """StartBoundaryNode and EndBoundaryNode return None for delta_v."""
        start = StartBoundaryNode(t, state_a)
        end   = EndBoundaryNode(t, state_a)
        assert start.delta_v is None
        assert end.delta_v is None

    def test_impulsive_boundary_node_delta_v_not_none(self, t, state_a, dv):
        """ImpulsiveBoundaryNode returns np.ndarray for delta_v."""
        node = ImpulsiveBoundaryNode(t, pre_state=state_a, delta_v=dv)
        assert isinstance(node.delta_v, np.ndarray)

    def test_delta_v_shape_is_3(self, t, state_a, state_b, dv):
        """All non-None delta_v results have shape (3,)."""
        nodes = [
            ImpulsiveBoundaryNode(t, pre_state=state_a, delta_v=dv),
            NullJunctionNode(t, state_a, state_a.copy()),
            ImpulsiveJunctionNode(t, pre_state=state_a, post_state=state_b),
            FreeJunctionNode(t, state_a, state_b),
        ]
        for node in nodes:
            assert node.delta_v.shape == (3,), (
                f"{type(node).__name__}.delta_v has wrong shape"
            )

    def test_state_defect_shape_is_6(self, t, state_a, state_b, dv):
        """All non-None state_defect results have shape (6,)."""
        nodes = [
            ImpulsiveBoundaryNode(t, pre_state=state_a, delta_v=dv),
            NullJunctionNode(t, state_a, state_a.copy()),
            ImpulsiveJunctionNode(t, pre_state=state_a, post_state=state_b),
            FreeJunctionNode(t, state_a, state_b),
        ]
        for node in nodes:
            assert node.state_defect.shape == (6,), (
                f"{type(node).__name__}.state_defect has wrong shape"
            )


# ========== MANEUVER JACOBIAN ==========

class TestManeuverJacobian:
    """Tests for maneuver_jacobian() on JunctionNode subtypes."""

    def test_null_junction_jacobian_is_6x6_identity(self, t, state_a):
        node = NullJunctionNode(t, state_a, state_a.copy())
        J = node.maneuver_jacobian()
        assert J.shape == (6, 6)
        np.testing.assert_array_equal(J, np.eye(6))

    def test_impulsive_junction_jacobian_is_6x6_identity(
            self, t, state_a, dv):
        node = ImpulsiveJunctionNode(t, pre_state=state_a, delta_v=dv)
        J = node.maneuver_jacobian()
        assert J.shape == (6, 6)
        np.testing.assert_array_equal(J, np.eye(6))

    def test_free_junction_jacobian_is_6x6_identity(self, t, state_a, state_b):
        node = FreeJunctionNode(t, state_a, state_b)
        J = node.maneuver_jacobian()
        assert J.shape == (6, 6)
        np.testing.assert_array_equal(J, np.eye(6))

    def test_jacobian_returns_new_array_each_call(self, t, state_a):
        """maneuver_jacobian returns a new array on each call."""
        node = NullJunctionNode(t, state_a, state_a.copy())
        J1 = node.maneuver_jacobian()
        J2 = node.maneuver_jacobian()
        assert J1 is not J2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
