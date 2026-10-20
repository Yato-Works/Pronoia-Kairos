"""Phase 0b: discrete-event simulator + latency model (Plan.md Section 36, 40-43)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.discrete import DiscreteEventSimulator, TraceRequest, EventKind
from simulator.latency import LatencyModel
from runtime.planner import AdaptivePlanner


def make_trace(keys):
    return [
        TraceRequest(request_id=f"r{i}", key=k, arrival_time=float(i),
                     input_tokens=256, output_tokens=32)
        for i, k in enumerate(keys)
    ]


def test_latency_decomposition_hit_faster_than_miss():
    """Section 36: T_reuse < T_miss when the prefix is cached."""
    lm = LatencyModel()
    hit = lm.ttft_hit(queue_wait=0.0, cached_tokens=1024)
    miss = lm.ttft_miss(queue_wait=0.0, input_tokens=1024)
    assert hit < miss


def test_latency_e2e_includes_decode():
    lm = LatencyModel()
    ttft = 10.0
    e2e = lm.e2e(ttft, output_tokens=11)
    assert e2e == ttft + lm.t_decode_per_token * 10


def test_simulator_counts_match_requests():
    keys = ["A", "B", "A", "C", "A", "B"]
    sim = DiscreteEventSimulator(capacity=3)
    m = sim.run(make_trace(keys))
    assert m.requests == len(keys)
    assert m.hits + m.misses == m.requests
    assert m.hit_rate() == m.hits / len(keys)


def test_simulator_lru_semantics():
    """Capacity 1 with alternating keys: every access misses."""
    keys = ["A", "B", "A", "B", "A", "B"]
    sim = DiscreteEventSimulator(capacity=1)
    m = sim.run(make_trace(keys))
    assert m.hits == 0
    assert m.misses == len(keys)


def test_planner_integration_produces_same_hit_count_as_baseline():
    """Planner present must not corrupt cache semantics (same LRU core)."""
    keys = ["A", "B", "C", "A", "D", "A"]
    base = DiscreteEventSimulator(capacity=3)
    planned = DiscreteEventSimulator(capacity=3, planner=AdaptivePlanner())
    m_base = base.run(make_trace(keys))
    m_plan = planned.run(make_trace(keys))
    assert m_base.hits == m_plan.hits
    assert m_base.evictions == m_plan.evictions


def test_high_confidence_trace_is_recorded():
    sim = DiscreteEventSimulator(capacity=2, planner=AdaptivePlanner())
    trace = [TraceRequest("r1", "A", 0.0, 256, 16,
                          future_use_probability=0.95, semantic_confidence=0.99)]
    m = sim.run(trace)
    assert m.requests == 1
    assert m.records[0].ttft > 0


def test_ttft_reflects_queue_wait():
    """Sequential arrivals on a busy server add queue_wait to TTFT."""
    sim = DiscreteEventSimulator(capacity=4)
    m = sim.run(make_trace(["A", "B", "C"]))
    # All arrive near t=0..2 but service is serial: TTFT must grow for later requests.
    assert m.records[-1].ttft >= m.records[0].ttft
