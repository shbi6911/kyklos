"""
Test suite for multi-segment Trajectory functionality.

Tests cover:
- Multi-segment properties (n_segments, times, n_steps, step_times, nodes)
- Segment dispatch (_find_segment, junction time convention)
- Sampling across segment boundaries (raw arrays, OrbitalElements, DataFrame)
- Composite and segment-local STM methods
- Trajectory manipulation (extend, extend_back, slice, segment_slice)
- Special methods (__len__, __getitem__, __iter__)
"""

import pytest
import numpy as np

from kyklos import System, EARTH, OE, OrbitalElements, earth_2body
from kyklos.trajectory import (
    Trajectory,
    StartBoundaryNode,
    EndBoundaryNode,
    ImpulsiveBoundaryNode,
    NullJunctionNode,
    ImpulsiveJunctionNode,
    FreeJunctionNode,
    JunctionNode,
    BoundaryNode,
)


# ========== TIME CONSTANTS ==========

T0         = 0.0
T_MID      = 2700.0
TF         = 5400.0
T_THIRD    = 1800.0
T_TWOTHIRD = 3600.0
DV_TEST    = np.array([0.1, 0.0, 0.0])   # Small impulsive burn [km/s]


# ========== MODULE-LEVEL FIXTURES ==========

@pytest.fixture(scope='module')
def system():
    """Simple 2-body Earth system."""
    return earth_2body()


@pytest.fixture(scope='module')
def initial_state():
    """Initial conditions for a circular LEO orbit."""
    return OE(
        a=6778.0, e=0.001, i=np.radians(51.6),
        omega=0.0, w=0.0, nu=0.0
    )


@pytest.fixture(scope='module')
def traj_single(system, initial_state):
    """Single-segment trajectory [T0, TF]."""
    return system.propagate(initial_state, [T0, TF])


@pytest.fixture(scope='module')
def traj_single_stm(system, initial_state):
    """Single-segment trajectory with STM [T0, TF]."""
    return system.propagate(initial_state, [T0, TF], with_stm=True)


@pytest.fixture(scope='module')
def traj_two_seg(system, initial_state):
    """Two-segment trajectory via extend(): [T0, T_MID, TF]."""
    traj = system.propagate(initial_state, [T0, T_MID])
    return traj.extend(TF)


@pytest.fixture(scope='module')
def traj_two_seg_stm(system, initial_state):
    """Two-segment trajectory with STM via extend()."""
    traj = system.propagate(initial_state, [T0, T_MID], with_stm=True)
    return traj.extend(TF)


@pytest.fixture(scope='module')
def traj_three_seg(system, initial_state):
    """Three-segment trajectory [T0, T_THIRD, T_TWOTHIRD, TF]."""
    traj = system.propagate(initial_state, [T0, T_THIRD])
    traj = traj.extend(T_TWOTHIRD)
    return traj.extend(TF)


@pytest.fixture(scope='module')
def traj_impulsive_end(system, initial_state):
    """
    Single-segment trajectory whose end_node is an ImpulsiveBoundaryNode.
    Used to test ImpulsiveBoundaryNode auto-expansion in extend().
    Constructed directly since there is no public API to set a custom
    end_node on a propagated trajectory.
    """
    ref = system.propagate(initial_state, [T0, T_MID])
    state_at_end = ref.state_at_raw(T_MID)
    imp_end = ImpulsiveBoundaryNode(T_MID, pre_state=state_at_end, delta_v=DV_TEST)
    return Trajectory(system, ref._outputs, end_node=imp_end)


# ========== MULTI-SEGMENT PROPERTIES ==========

class TestMultiSegmentProperties:
    """Test properties specific to the multi-segment architecture."""

    def test_n_segments_single(self, traj_single):
        assert traj_single.n_segments == 1

    def test_n_segments_two(self, traj_two_seg):
        assert traj_two_seg.n_segments == 2

    def test_n_segments_three(self, traj_three_seg):
        assert traj_three_seg.n_segments == 3

    def test_is_multisegment_false_for_single(self, traj_single):
        assert traj_single.is_multisegment is False

    def test_is_multisegment_true_for_two(self, traj_two_seg):
        assert traj_two_seg.is_multisegment is True

    def test_times_single_has_two_values(self, traj_single):
        """Single-segment times: [t0, tf]."""
        times = traj_single.times
        assert len(times) == 2
        assert times[0] == T0
        assert times[-1] == TF

    def test_times_two_seg_has_three_values(self, traj_two_seg):
        """Two-segment times: [t0, t_junction, tf]."""
        times = traj_two_seg.times
        assert len(times) == 3
        assert times[0] == T0
        assert np.isclose(times[1], T_MID)
        assert times[2] == TF

    def test_times_three_seg_has_four_values(self, traj_three_seg):
        assert len(traj_three_seg.times) == 4

    def test_n_steps_single_returns_scalar_int(self, traj_single):
        """n_steps for single segment is a scalar int."""
        result = traj_single.n_steps
        assert isinstance(result, int)
        assert result > 0

    def test_n_steps_multi_returns_list(self, traj_two_seg):
        """n_steps for multi-segment is a list of ints."""
        result = traj_two_seg.n_steps
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(n, int) and n > 0 for n in result)

    def test_step_times_is_sorted(self, traj_two_seg):
        """step_times is monotonically increasing."""
        st = traj_two_seg.step_times
        assert np.all(np.diff(st) > 0)

    def test_step_times_has_no_duplicates(self, traj_two_seg):
        """Junction time appears exactly once in step_times."""
        st = traj_two_seg.step_times
        assert len(st) == len(np.unique(st))

    def test_step_times_spans_full_range(self, traj_two_seg):
        """step_times covers [t0, tf]."""
        st = traj_two_seg.step_times
        assert np.isclose(st[0],  T0, atol=1e-10)
        assert np.isclose(st[-1], TF, atol=1e-10)

    def test_start_node_is_start_boundary(self, traj_two_seg):
        assert isinstance(traj_two_seg.start_node, StartBoundaryNode)

    def test_end_node_is_end_boundary(self, traj_two_seg):
        assert isinstance(traj_two_seg.end_node, EndBoundaryNode)

    def test_junction_nodes_count(self, traj_two_seg):
        """junction_nodes has n_segments - 1 elements."""
        assert len(traj_two_seg.junction_nodes) == traj_two_seg.n_segments - 1

    def test_junction_nodes_are_junction_type(self, traj_two_seg):
        for node in traj_two_seg.junction_nodes:
            assert isinstance(node, JunctionNode)

    def test_junction_nodes_returns_defensive_copy(self, traj_two_seg):
        """Mutating the returned list does not affect the trajectory."""
        junctions = traj_two_seg.junction_nodes
        original_len = len(junctions)
        junctions.clear()
        assert len(traj_two_seg.junction_nodes) == original_len

    def test_t0_matches_start_node_time(self, traj_two_seg):
        assert traj_two_seg.t0 == traj_two_seg.start_node.time

    def test_tf_matches_end_node_time(self, traj_two_seg):
        assert traj_two_seg.tf == traj_two_seg.end_node.time

    def test_has_stm_false_without_stm(self, traj_two_seg):
        assert traj_two_seg.has_stm is False

    def test_has_stm_true_with_stm(self, traj_two_seg_stm):
        assert traj_two_seg_stm.has_stm is True


# ========== SEGMENT DISPATCH ==========

class TestSegmentDispatch:
    """Test _find_segment dispatch including junction time convention."""

    def test_time_before_junction_is_segment_0(self, traj_two_seg):
        assert traj_two_seg._find_segment(T_MID * 0.5) == 0

    def test_time_after_junction_is_segment_1(self, traj_two_seg):
        assert traj_two_seg._find_segment(T_MID + 100.0) == 1

    def test_exact_junction_time_prefers_later_segment(self, traj_two_seg):
        """At exact junction time, bisect_right convention gives later segment."""
        assert traj_two_seg._find_segment(T_MID) == 1

    def test_t0_maps_to_segment_0(self, traj_two_seg):
        assert traj_two_seg._find_segment(T0) == 0

    def test_tf_maps_to_last_segment(self, traj_two_seg):
        last = traj_two_seg.n_segments - 1
        assert traj_two_seg._find_segment(TF) == last

    def test_vectorized_input_returns_array(self, traj_two_seg):
        """Array input returns numpy array of indices."""
        times = np.array([500.0, T_MID, T_MID + 500.0])
        indices = traj_two_seg._find_segment(times)
        assert isinstance(indices, np.ndarray)
        assert indices[0] == 0
        assert indices[1] == 1   # junction convention
        assert indices[2] == 1

    def test_three_segment_dispatch(self, traj_three_seg):
        """Three-segment dispatch returns correct index for each segment."""
        assert traj_three_seg._find_segment(T_THIRD * 0.5)     == 0
        assert traj_three_seg._find_segment(T_THIRD + 100.0)   == 1
        assert traj_three_seg._find_segment(T_TWOTHIRD + 100.0)== 2

    def test_single_segment_always_returns_0(self, traj_single):
        """Single segment always maps to index 0."""
        for t in [T0, T_MID, TF]:
            assert traj_single._find_segment(t) == 0


# ========== SAMPLING ACROSS SEGMENTS ==========

class TestSamplingAcrossSegments:
    """Test state sampling methods work correctly across segment boundaries."""

    def test_state_at_raw_in_first_segment(self, traj_two_seg):
        state = traj_two_seg.state_at_raw(T_MID * 0.5)
        assert state.shape == (6,)
        assert np.all(np.isfinite(state))

    def test_state_at_raw_in_second_segment(self, traj_two_seg):
        state = traj_two_seg.state_at_raw(T_MID + 500.0)
        assert state.shape == (6,)
        assert np.all(np.isfinite(state))

    def test_state_at_raw_at_junction(self, traj_two_seg):
        """Querying at junction time returns a valid state."""
        state = traj_two_seg.state_at_raw(T_MID)
        assert state.shape == (6,)
        assert np.all(np.isfinite(state))

    def test_state_continuity_at_junction(self, traj_two_seg):
        """For continuous junction, state just before and after are close."""
        state_pre  = traj_two_seg.state_at_raw(T_MID - 0.00001)
        state_post = traj_two_seg.state_at_raw(T_MID + 0.00001)
        np.testing.assert_allclose(state_pre, state_post, atol=1e-3)

    def test_evaluate_raw_spans_both_segments(self, traj_two_seg):
        """evaluate_raw with array spanning both segments returns (n, 6)."""
        times = np.linspace(T0, TF, 50)
        states = traj_two_seg.evaluate_raw(times)
        assert states.shape == (50, 6)
        assert np.all(np.isfinite(states))

    def test_evaluate_raw_consistent_with_state_at_raw(self, traj_two_seg):
        """Single-element evaluate_raw matches state_at_raw."""
        t = T_MID + 300.0
        via_eval = traj_two_seg.evaluate_raw(np.array([t]))[0]
        via_state = traj_two_seg.state_at_raw(t)
        np.testing.assert_array_equal(via_eval, via_state)

    def test_sample_raw_correct_shape(self, traj_two_seg):
        states = traj_two_seg.sample_raw(n_points=100)
        assert states.shape == (100, 6)
        assert np.all(np.isfinite(states))

    def test_state_at_returns_orbital_elements(self, traj_two_seg):
        state = traj_two_seg.state_at(T_MID + 300.0)
        assert isinstance(state, OrbitalElements)

    def test_evaluate_array_spans_both_segments(self, traj_two_seg):
        times = np.array([500.0, T_MID - 100.0, T_MID + 100.0, TF - 100.0])
        states = traj_two_seg.evaluate(times)
        assert len(states) == 4
        assert all(isinstance(s, OrbitalElements) for s in states)

    def test_sample_returns_list(self, traj_two_seg):
        states = traj_two_seg.sample(n_points=20)
        assert isinstance(states, list)
        assert len(states) == 20

    def test_multi_matches_single_in_first_segment(self, system, initial_state):
        """Multi-segment trajectory gives same states as single-segment
        for times within the first segment."""
        traj_s = system.propagate(initial_state, [T0, TF])
        traj_m = system.propagate(
            initial_state, [T0, T_MID]
        ).extend(TF)

        t_query = T_MID * 0.5
        state_s = traj_s.state_at_raw(t_query)
        state_m = traj_m.state_at_raw(t_query)
        np.testing.assert_allclose(state_s, state_m, rtol=1e-10)

    def test_to_dataframe_segment_column_present(self, traj_two_seg):
        df = traj_two_seg.to_dataframe(n_points=50)
        assert 'segment' in df.columns

    def test_to_dataframe_segment_column_position(self, traj_two_seg):
        """segment column is the second column (after time)."""
        df = traj_two_seg.to_dataframe(n_points=10)
        cols = list(df.columns)
        assert cols[0] == 'time'
        assert cols[1] == 'segment'

    def test_to_dataframe_segment_values_correct(self, traj_two_seg):
        """Times before junction belong to segment 0, after to segment 1."""
        df = traj_two_seg.to_dataframe(n_points=100)
        pre  = df[df['time'] < T_MID - 1]['segment']
        post = df[df['time'] > T_MID + 1]['segment']
        assert (pre  == 0).all()
        assert (post == 1).all()


# ========== COMPOSITE AND SEGMENT-LOCAL STM ==========

class TestCompositeSTM:
    """Test STM methods on multi-segment trajectories."""

    def test_get_stm_identity_at_t0(self, traj_two_seg_stm):
        """Composite STM is identity at t0."""
        stm = traj_two_seg_stm.get_stm(T0)
        np.testing.assert_allclose(stm, np.eye(6), rtol=1e-12)

    def test_get_stm_evolves_in_first_segment(self, traj_two_seg_stm):
        """Composite STM evolves from identity in first segment."""
        stm = traj_two_seg_stm.get_stm(T_MID * 0.5)
        assert np.linalg.norm(stm - np.eye(6)) > 1e-6

    def test_get_stm_not_identity_after_junction(self, traj_two_seg_stm):
        """Composite STM is not identity at time in second segment."""
        stm = traj_two_seg_stm.get_stm(T_MID + 300.0)
        assert np.linalg.norm(stm - np.eye(6)) > 1e-6

    def test_get_stm_shape(self, traj_two_seg_stm):
        assert traj_two_seg_stm.get_stm(T_MID + 300.0).shape == (6, 6)

    def test_get_stm_seg_identity_at_junction(self, traj_two_seg_stm):
        """Segment-local STM is identity at start of second segment."""
        stm_seg = traj_two_seg_stm.get_stm_seg(T_MID)
        np.testing.assert_allclose(stm_seg, np.eye(6), rtol=1e-12)

    def test_get_stm_seg_evolves_within_segment(self, traj_two_seg_stm):
        """Segment-local STM evolves from identity within a segment."""
        stm_seg = traj_two_seg_stm.get_stm_seg(T_MID + 500.0)
        assert np.linalg.norm(stm_seg - np.eye(6)) > 1e-6

    def test_composite_and_seg_differ_after_junction(self, traj_two_seg_stm):
        """Composite and segment-local STMs differ at time in second segment."""
        t = T_MID + 500.0
        assert not np.allclose(
            traj_two_seg_stm.get_stm(t),
            traj_two_seg_stm.get_stm_seg(t)
        )

    def test_composite_and_seg_agree_in_first_segment(self, traj_two_seg_stm):
        """In first segment, composite and segment-local STMs are identical."""
        t = T_MID * 0.5
        np.testing.assert_array_equal(
            traj_two_seg_stm.get_stm(t),
            traj_two_seg_stm.get_stm_seg(t)
        )

    def test_segment_terminal_stm_shape(self, traj_two_seg_stm):
        assert traj_two_seg_stm.segment_terminal_stm(0).shape == (6, 6)

    def test_segment_terminal_stm_matches_stm_seg_at_junction(
            self, traj_two_seg_stm):
        """segment_terminal_stm(0) matches get_stm_seg at the junction time
        approached from within segment 0."""
        terminal   = traj_two_seg_stm.segment_terminal_stm(0)
        at_junc    = traj_two_seg_stm.get_stm_seg(T_MID - 1e-9)
        np.testing.assert_allclose(terminal, at_junc, rtol=1e-10)

    def test_segment_terminal_stm_out_of_range_raises(self, traj_two_seg_stm):
        with pytest.raises(ValueError):
            traj_two_seg_stm.segment_terminal_stm(10)

    def test_evaluate_stm_array_shape(self, traj_two_seg_stm):
        """evaluate_stm with array across both segments returns (n, 6, 6)."""
        times = np.linspace(T0, TF, 20)
        stms = traj_two_seg_stm.evaluate_stm(times)
        assert stms.shape == (20, 6, 6)

    def test_evaluate_stm_seg_identity_at_segment_starts(
            self, traj_two_seg_stm):
        """evaluate_stm_seg at both segment start times gives identity."""
        times = np.array([T0, T_MID])
        stms = traj_two_seg_stm.evaluate_stm_seg(times)
        assert stms.shape == (2, 6, 6)
        np.testing.assert_allclose(stms[0], np.eye(6), rtol=1e-12)
        np.testing.assert_allclose(stms[1], np.eye(6), rtol=1e-12)

    def test_stm_requires_stm_trajectory(self, traj_two_seg):
        """get_stm raises if trajectory was not propagated with STM."""
        with pytest.raises(ValueError, match="[Nn]ot propagated with STM"):
            traj_two_seg.get_stm(T_MID + 300.0)


# ========== EXTEND ==========

class TestExtend:
    """Test extend() and extend_back() on single and multi-segment trajectories."""

    def test_extend_creates_two_segments(self, system, initial_state):
        traj = system.propagate(initial_state, [T0, T_MID])
        assert traj.extend(TF).n_segments == 2

    def test_extend_preserves_t0(self, system, initial_state):
        traj = system.propagate(initial_state, [T0, T_MID])
        assert traj.extend(TF).t0 == T0

    def test_extend_has_correct_tf(self, system, initial_state):
        traj = system.propagate(initial_state, [T0, T_MID])
        assert traj.extend(TF).tf == TF

    def test_extend_infers_null_junction_for_continuous(
            self, system, initial_state):
        """Continuous extension infers NullJunctionNode at junction."""
        traj = system.propagate(initial_state, [T0, T_MID])
        junction = traj.extend(TF).junction_nodes[0]
        assert isinstance(junction, NullJunctionNode)

    def test_extend_with_delta_v_creates_impulsive_junction(
            self, system, initial_state):
        """delta_v argument creates ImpulsiveJunctionNode."""
        traj = system.propagate(initial_state, [T0, T_MID])
        junction = traj.extend(TF, junction=DV_TEST).junction_nodes[0]
        assert isinstance(junction, ImpulsiveJunctionNode)

    def test_extend_delta_v_correct_magnitude(self, system, initial_state):
        """ImpulsiveJunctionNode carries the correct delta_v."""
        traj = system.propagate(initial_state, [T0, T_MID])
        junction = traj.extend(TF, junction=DV_TEST).junction_nodes[0]
        np.testing.assert_allclose(junction.delta_v, DV_TEST, rtol=1e-12)

    def test_extend_explicit_junction_node_used_directly(
            self, system, initial_state):
        """Explicit JunctionNode is used as-is."""
        traj = system.propagate(initial_state, [T0, T_MID])
        pre = traj.state_at_raw(T_MID)
        post = pre.copy(); post[3:] += DV_TEST
        explicit = ImpulsiveJunctionNode(T_MID, pre_state=pre, post_state=post)

        extended = traj.extend(TF, junction=explicit)
        assert extended.junction_nodes[0] is explicit

    def test_extend_impulsive_boundary_node_auto_expansion(
            self, traj_impulsive_end):
        """End ImpulsiveBoundaryNode is auto-expanded into ImpulsiveJunctionNode."""
        extended = traj_impulsive_end.extend(TF)
        assert extended.n_segments == 2
        assert isinstance(extended.junction_nodes[0], ImpulsiveJunctionNode)

    def test_extend_impulsive_boundary_node_preserves_delta_v(
            self, traj_impulsive_end):
        """Expanded ImpulsiveJunctionNode preserves the original delta_v."""
        extended = traj_impulsive_end.extend(TF)
        np.testing.assert_allclose(
            extended.junction_nodes[0].delta_v, DV_TEST, rtol=1e-10
        )

    def test_extend_chain_gives_three_segments(self, system, initial_state):
        """Two chained extend() calls produce three segments."""
        traj = system.propagate(initial_state, [T0, T_THIRD])
        traj = traj.extend(T_TWOTHIRD)
        traj = traj.extend(TF)
        assert traj.n_segments == 3

    def test_extend_does_not_modify_original(self, system, initial_state):
        """extend() is non-destructive — original trajectory unchanged."""
        traj = system.propagate(initial_state, [T0, T_MID])
        _ = traj.extend(TF)
        assert traj.n_segments == 1
        assert traj.tf == T_MID

    def test_extend_invalid_new_tf_raises(self, system, initial_state):
        """new_tf <= current tf raises ValueError."""
        traj = system.propagate(initial_state, [T0, T_MID])
        with pytest.raises(ValueError, match="greater than"):
            traj.extend(T_MID - 100.0)

    def test_extend_back_creates_two_segments(self, system, initial_state):
        """extend_back() produces a two-segment trajectory."""
        traj = system.propagate(initial_state, [-T_MID, T0])
        extended = traj.extend_back(-TF)
        assert extended.n_segments == 2

    def test_extend_back_new_t0_less_than_t0(self, system, initial_state):
        """extend_back() raises if new_t0 >= current t0."""
        traj = system.propagate(initial_state, [T_MID, TF])
        with pytest.raises(ValueError, match="less than"):
            traj.extend_back(T_MID + 100.0)

    def test_extend_preserves_stm(self, system, initial_state):
        """extend() preserves stm_order from original trajectory."""
        traj = system.propagate(initial_state, [T0, T_MID], with_stm=True)
        extended = traj.extend(TF)
        assert extended._stm_order == 1


# ========== SLICE ==========

class TestSlice:
    """Test slice() and segment_slice() on multi-segment trajectories."""

    def test_slice_within_one_segment_stays_single(self, traj_two_seg):
        """Slice entirely within first segment gives single-segment result."""
        sliced = traj_two_seg.slice(100.0, T_MID - 100.0)
        assert sliced.n_segments == 1

    def test_slice_spanning_junction_gives_two_segments(self, traj_two_seg):
        """Slice spanning the junction gives two-segment result."""
        sliced = traj_two_seg.slice(T_MID - 500.0, T_MID + 500.0)
        assert sliced.n_segments == 2

    def test_slice_correct_t0_and_tf(self, traj_two_seg):
        t_s, t_e = 300.0, T_MID + 300.0
        sliced = traj_two_seg.slice(t_s, t_e)
        assert np.isclose(sliced.t0, t_s)
        assert np.isclose(sliced.tf, t_e)

    def test_slice_preserves_stm_order(self, traj_two_seg_stm):
        sliced = traj_two_seg_stm.slice(500.0, T_MID + 500.0)
        assert sliced._stm_order == 1

    def test_slice_out_of_bounds_raises(self, traj_two_seg):
        with pytest.raises(ValueError):
            traj_two_seg.slice(T0 - 100.0, T_MID)

    def test_slice_preserves_junction_type(self, traj_two_seg):
        """Junction nodes within slice are recreated with correct type."""
        sliced = traj_two_seg.slice(T_MID - 500.0, T_MID + 500.0)
        assert isinstance(sliced.junction_nodes[0], NullJunctionNode)

    # --- segment_slice ---

    def test_segment_slice_first_segment(self, traj_two_seg):
        """segment_slice(0, 0) returns the first segment."""
        seg = traj_two_seg.segment_slice(0, 0)
        assert seg.n_segments == 1
        assert np.isclose(seg.t0, T0)
        assert np.isclose(seg.tf, T_MID)

    def test_segment_slice_last_segment(self, traj_two_seg):
        """segment_slice(1, 1) returns the last segment."""
        seg = traj_two_seg.segment_slice(1, 1)
        assert seg.n_segments == 1
        assert np.isclose(seg.t0, T_MID)
        assert np.isclose(seg.tf, TF)

    def test_segment_slice_all_segments(self, traj_three_seg):
        """segment_slice(0, n-1) returns all segments."""
        n = traj_three_seg.n_segments
        seg = traj_three_seg.segment_slice(0, n - 1)
        assert seg.n_segments == n
        assert np.isclose(seg.t0, T0)
        assert np.isclose(seg.tf, TF)

    def test_segment_slice_middle_segment(self, traj_three_seg):
        """segment_slice(1, 1) extracts the middle segment."""
        seg = traj_three_seg.segment_slice(1, 1)
        assert seg.n_segments == 1
        assert np.isclose(seg.t0, T_THIRD)
        assert np.isclose(seg.tf, T_TWOTHIRD)

    def test_segment_slice_inclusive_end_convention(self, traj_three_seg):
        """segment_slice(0, 1) includes segments 0 AND 1 (inclusive)."""
        seg = traj_three_seg.segment_slice(0, 1)
        assert seg.n_segments == 2

    def test_segment_slice_out_of_range_raises(self, traj_two_seg):
        with pytest.raises(ValueError):
            traj_two_seg.segment_slice(0, 5)

    def test_segment_slice_preserves_stm_order(self, traj_two_seg_stm):
        seg = traj_two_seg_stm.segment_slice(0, 0)
        assert seg._stm_order == 1

    def test_segment_slice_reuses_heyoka_outputs(self, traj_two_seg):
        """segment_slice shares the underlying Heyoka output objects."""
        seg = traj_two_seg.segment_slice(0, 0)
        assert seg._outputs[0] is traj_two_seg._outputs[0]

    def test_segment_slice_junction_nodes_are_new_objects(
            self, traj_three_seg):
        """segment_slice creates new junction node objects, not shared refs."""
        seg = traj_three_seg.segment_slice(0, 1)
        assert seg._junction_nodes[0] is not traj_three_seg._junction_nodes[0]


# ========== SPECIAL METHODS ==========

class TestSpecialMethods:
    """Test __len__, __getitem__, and iteration on multi-segment trajectories."""

    def test_len_single(self, traj_single):
        assert len(traj_single) == 1

    def test_len_two(self, traj_two_seg):
        assert len(traj_two_seg) == 2

    def test_len_three(self, traj_three_seg):
        assert len(traj_three_seg) == 3

    def test_getitem_first_segment(self, traj_two_seg):
        """traj[0] returns single-segment Trajectory for first segment."""
        seg = traj_two_seg[0]
        assert isinstance(seg, Trajectory)
        assert seg.n_segments == 1
        assert np.isclose(seg.t0, T0)
        assert np.isclose(seg.tf, T_MID)

    def test_getitem_second_segment(self, traj_two_seg):
        """traj[1] returns single-segment Trajectory for second segment."""
        seg = traj_two_seg[1]
        assert seg.n_segments == 1
        assert np.isclose(seg.t0, T_MID)
        assert np.isclose(seg.tf, TF)

    def test_getitem_negative_last(self, traj_two_seg):
        """traj[-1] returns the last segment."""
        seg = traj_two_seg[-1]
        assert np.isclose(seg.t0, T_MID)
        assert np.isclose(seg.tf, TF)

    def test_getitem_negative_first(self, traj_two_seg):
        """traj[-2] returns the first segment for two-segment trajectory."""
        seg = traj_two_seg[-2]
        assert np.isclose(seg.t0, T0)
        assert np.isclose(seg.tf, T_MID)

    def test_getitem_out_of_range_raises_index_error(self, traj_two_seg):
        with pytest.raises(IndexError):
            _ = traj_two_seg[5]

    def test_getitem_non_integer_raises_type_error(self, traj_two_seg):
        with pytest.raises(TypeError):
            _ = traj_two_seg[0.5]  # type: ignore[index]

    def test_getitem_negative_out_of_range_raises(self, traj_two_seg):
        with pytest.raises(IndexError):
            _ = traj_two_seg[-5]

    def test_getitem_matches_segment_slice(self, traj_three_seg):
        """traj[k] gives the same result as segment_slice(k, k)."""
        for k in range(traj_three_seg.n_segments):
            by_index = traj_three_seg[k]
            by_slice = traj_three_seg.segment_slice(k, k)
            assert np.isclose(by_index.t0, by_slice.t0)
            assert np.isclose(by_index.tf, by_slice.tf)

    def test_iter_yields_correct_count(self, traj_three_seg):
        """Iterating yields exactly n_segments items."""
        items = list(traj_three_seg)
        assert len(items) == 3

    def test_iter_all_items_are_trajectories(self, traj_two_seg):
        for seg in traj_two_seg:
            assert isinstance(seg, Trajectory)

    def test_iter_all_items_are_single_segment(self, traj_two_seg):
        for seg in traj_two_seg:
            assert seg.n_segments == 1

    def test_iter_segments_are_time_adjacent(self, traj_three_seg):
        """Consecutive iterated segments share a boundary time."""
        segs = list(traj_three_seg)
        for k in range(len(segs) - 1):
            assert np.isclose(segs[k].tf, segs[k + 1].t0)

    def test_iter_covers_full_time_range(self, traj_two_seg):
        """First and last iterated segments cover [t0, tf]."""
        segs = list(traj_two_seg)
        assert np.isclose(segs[0].t0,  T0)
        assert np.isclose(segs[-1].tf, TF)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
