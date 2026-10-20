"""Phase 3 — Lethe -> design-map bridge (Plan.md Section 8, 43, 48, 49, 50, 63.1, 90).

Fixes the semantic-to-computational bridge:
    - LetheTraceBuilder turns a MemoryRecord recall stream into a RequestTrace.
    - the retention signal is CALIBRATED to horizon reuse (Section 50), so
      predictive retention actually functions on the Lethe trace.
    - measure_correlation_distance recovers the workload's reuse distance and
      classifies it short/mid/long (the design-map axis, Section 63.1).
    - place_on_design_map recommends a mechanism from regime x capacity.
    - run_lethe_design_map verifies end to end that the recommended mechanism
      beats the LRU baseline (Section 90 gate).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from semantic.lethe_trace import (
    LetheTraceBuilder, synthesize_lethe_stream, measure_correlation_distance,
    place_on_design_map, p_sem_list,
)
from compare.lethe_design_map import run_lethe_design_map, run_lethe_sweep
from simulator.discrete import TraceRequest


# --- Trace building ----------------------------------------------------------

def test_lethe_stream_builds_trace_with_matching_keys():
    events = synthesize_lethe_stream(n_objects=10, n_recalls=60, reuse_gap=4, seed=1)
    trace = LetheTraceBuilder().build(events, horizon=4)
    assert len(trace) == len(events)
    assert [t.key for t in trace] == [e.memory.chunk_id for e in events]


def test_retention_signal_is_calibrated_to_horizon():
    """future_use_probability is 1.0 iff the chunk recurs within horizon, not the
    raw position-independent Lethe utility."""
    events = synthesize_lethe_stream(n_objects=8, n_recalls=80, reuse_gap=4, seed=2)
    trace = LetheTraceBuilder().build(events, horizon=4)
    keys = [t.key for t in trace]
    for i, t in enumerate(trace):
        recurs = keys[i] in keys[i + 1: i + 1 + 4]
        assert t.future_use_probability == (1.0 if recurs else 0.0)


def test_raw_semantic_signal_kept_for_calibration_metrics():
    events = synthesize_lethe_stream(n_objects=8, n_recalls=40, reuse_gap=4, seed=3)
    raw = p_sem_list(events)
    assert len(raw) == len(events)
    assert all(0.0 <= p <= 0.99 for p in raw)


# --- Correlation-distance measurement ----------------------------------------

def test_measured_distance_matches_requested_gap():
    """A round-robin stream with period `gap` measures back as ~gap."""
    for gap in (2, 4, 8, 12):
        events = synthesize_lethe_stream(n_objects=20, n_recalls=200, reuse_gap=gap, seed=5)
        trace = LetheTraceBuilder().build(events, horizon=gap)
        prof = measure_correlation_distance(trace)
        assert prof.median_reuse_distance == float(gap), (gap, prof.median_reuse_distance)



# --- Design-map placement ----------------------------------------------------

def test_place_on_design_map_recommends_by_regime():
    for gap, expected_regime in [(2, "short"), (4, "mid"), (8, "long"), (12, "long")]:
        events = synthesize_lethe_stream(n_objects=20, n_recalls=200, reuse_gap=gap, seed=5)
        trace = LetheTraceBuilder().build(events, horizon=gap)
        placement = place_on_design_map(trace, capacity=6)
        assert placement.profile.regime == expected_regime
        assert placement.recommended_mechanism in ("lru", "retention", "prefetch", "adaptive")
        # feed_retention is on for mid/long, off for short (Section 76 #5).
        assert placement.feed_retention == (expected_regime in ("mid", "long"))


def test_capacity_band_selection():
    events = synthesize_lethe_stream(n_objects=20, n_recalls=200, reuse_gap=8, seed=5)
    trace = LetheTraceBuilder().build(events, horizon=8)
    assert place_on_design_map(trace, capacity=4).capacity_band == "tight"
    assert place_on_design_map(trace, capacity=12).capacity_band == "roomy"


# --- End-to-end bridge (Section 90 gate) -------------------------------------

def test_end_to_end_recommendation_beats_lru():
    """Section 90: for every regime, the recommended mechanism beats the LRU
    baseline on the same Lethe trace, and recommendation == measured winner."""
    from compare.lethe_design_map import run_lethe_sweep
    results = run_lethe_sweep(capacity=6)
    assert len(results) == 4
    for r in results:
        assert r.semantic_beats_lru, (r.regime, r.recommended, r.lru_net)
        assert r.recommended == r.winner


def test_long_range_retention_and_adaptive_win():
    r = run_lethe_design_map(reuse_gap=12, capacity=6)
    assert r.regime == "long"
    assert r.retention_net > r.lru_net
    assert r.adaptive_net >= r.retention_net


def test_measured_median_monotone_in_gap():
    medians = []
    for gap in (2, 4, 8, 12):
        events = synthesize_lethe_stream(n_objects=20, n_recalls=200, reuse_gap=gap, seed=5)
        trace = LetheTraceBuilder().build(events, horizon=gap)
        medians.append(measure_correlation_distance(trace).median_reuse_distance)
    assert medians == sorted(medians)


def test_reproducible_under_fixed_seed():
    a = run_lethe_design_map(reuse_gap=8, capacity=6, seed=11)
    b = run_lethe_design_map(reuse_gap=8, capacity=6, seed=11)
    assert a.recommended == b.recommended
    assert a.adaptive_net == b.adaptive_net


def _regime_for_constant_distance(dist):
    """Build a synthetic trace where each key recurs exactly `dist` steps later."""
    n = int(dist)
    keys = [f"k{i % n}" for i in range(n * 5)]
    trace = [TraceRequest(request_id=str(i), key=k, arrival_time=float(i))
             for i, k in enumerate(keys)]
    return measure_correlation_distance(trace).regime


def test_regime_classification_thresholds():
    assert _regime_for_constant_distance(2) == "short"
    assert _regime_for_constant_distance(4) == "mid"
    assert _regime_for_constant_distance(7) == "mid"
    assert _regime_for_constant_distance(8) == "long"
    assert _regime_for_constant_distance(20) == "long"
