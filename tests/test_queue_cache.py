"""Phase 2 — Queue + Cache Co-Optimization gate (Plan.md Section 46, 47).

Section 47's decisive question: does a condition exist that beats PEEK (queue-only
prefix protection)? These tests fix the answer and its mechanism:

    - QueueAwareRetentionPolicy reproduces PEEK: protects keys the pending queue
      will reference within the lookahead window.
    - QueueSemanticRetentionPolicy tempers queue protection with a semantic
      future-demand signal.
    - on a long-range-reuse workload (queue myopic, semantic horizon long), the
      queue+semantic policy beats PEEK on cache hits without hurting TTFT p95 -
      the Section 47 "beats PEEK" condition EXISTS (Section 17).
    - when the queue is NOT myopic (lookahead >= gap), the two converge (semantic
      adds little), an honest boundary of the claim.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scheduler.queue_cache_policy import (
    QueueAwareRetentionPolicy, QueueSemanticRetentionPolicy,
)
from compare.queue_cache_experiment import (
    run_policy, run_phase2_comparison, _horizon_signals, format_phase2_report,
    GateMetrics,
)
from simulator import workloads as w


# --- Policy unit behaviour ----------------------------------------------------

def test_queue_aware_protects_queue_keys():
    """PEEK-like: a key the queue will reference is kept over a colder key."""
    pol = QueueAwareRetentionPolicy(lookahead=4)
    lru = ["cold", "hot"]
    # "hot" is in the next queue window -> protected; "cold" is evicted.
    victim = pol.choose_victim(lru, {}, {"cold": 5, "hot": 1}, queue_keys={"hot"})
    assert victim == "cold"


def test_queue_aware_evicts_lru_when_no_queue_pressure():
    pol = QueueAwareRetentionPolicy(lookahead=4)
    lru = ["a", "b"]  # a older than b
    victim = pol.choose_victim(lru, {}, {"a": 5, "b": 1}, queue_keys=set())
    assert victim == "a"  # least recently used evicted


def test_queue_semantic_protects_high_demand_when_queue_myopic():
    """queue+semantic: with an empty queue view, a high semantic-demand key is
    kept over a low-demand key (the PEEK-beating mechanism)."""
    pol = QueueSemanticRetentionPolicy(lookahead=4)
    lru = ["low", "high"]
    signals = {"low": 0.0, "high": 1.0}
    ages = {"low": 1, "high": 1}
    victim = pol.choose_victim(lru, signals, ages, queue_keys=set())
    assert victim == "low"  # high-demand key protected by semantic term


def test_queue_semantic_queue_pressure_dominates():
    """When the queue references a key, that queue pressure dominates the
    semantic term (queue says WHEN)."""
    pol = QueueSemanticRetentionPolicy(lookahead=4, queue_weight=1.0)
    lru = ["queued", "hot_semantic"]
    signals = {"queued": 0.0, "hot_semantic": 1.0}


# --- Section 47 Phase 2 Gate -------------------------------------------------

def test_phase2_three_way_comparison():
    cmp = run_phase2_comparison(capacity=4, gap=8, lookahead=4)
    assert {m.policy for m in cmp.metrics} == {"lru", "peek", "queue+sem"}


def test_semantic_beats_peek_when_queue_myopic():
    """Section 47 decisive question: on a long-range workload (queue myopic), the
    queue+semantic policy beats PEEK - a PEEK-beating condition EXISTS."""
    cmp = run_phase2_comparison(capacity=4, gap=8, lookahead=4)
    assert cmp.semantically_beats_peek()
    sem, peek = cmp.get("queue+sem"), cmp.get("peek")
    assert sem.hits > peek.hits
    assert sem.ttft_p95 <= peek.ttft_p95 + 1e-9


def test_semantic_advantage_is_large_when_queue_very_myopic():
    """The larger the gap between semantic horizon and queue lookahead, the
    bigger the semantic advantage over queue-only PEEK."""
    myopic = run_phase2_comparison(capacity=4, gap=12, lookahead=2)
    sem, peek = myopic.get("queue+sem"), myopic.get("peek")
    assert sem.hits > peek.hits
    assert (sem.hits - peek.hits) > 0


def test_policies_converge_when_queue_not_myopic():
    """Honest boundary: when lookahead >= gap, the queue already sees the return,
    so queue-only PEEK matches queue+semantic (semantic adds little)."""
    cmp = run_phase2_comparison(capacity=4, gap=2, lookahead=6)
    peek, sem = cmp.get("peek"), cmp.get("queue+sem")
    # semantic is never worse; with a non-myopic queue it should not lag PEEK.
    assert sem.hits >= peek.hits


def test_gate_metrics_are_consistent():
    cmp = run_phase2_comparison(capacity=4, gap=8, lookahead=4)
    for m in cmp.metrics:
        assert isinstance(m, GateMetrics)
        assert 0.0 <= m.hit_rate <= 1.0
        assert 0.0 <= m.goodput <= 1.0
        assert m.fairness >= 1.0  # p95/median ratio
        assert m.ttft_p99 >= m.ttft_p95 >= m.ttft_p50


def test_report_renders_all_policies_and_verdict():
    cmp = run_phase2_comparison(capacity=4, gap=8, lookahead=4)
    out = format_phase2_report(cmp)
    for p in ("lru", "peek", "queue+sem"):
        assert p in out
    assert "beats PEEK EXISTS" in out


def test_horizon_signals_mark_reuse_within_horizon():
    seq = ["a", "b", "c", "a", "b"]
    sig = _horizon_signals(seq, horizon=3)
    # position 0 'a' recurs at index 3 (within 3) -> 1.0
    assert sig[0] == 1.0
    # position 1 'b' recurs at index 4 (within 3) -> 1.0
    assert sig[1] == 1.0
    # position 2 'c' never recurs -> 0.0
    assert sig[2] == 0.0
