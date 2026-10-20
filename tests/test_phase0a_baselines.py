"""Phase 0a: baseline policies + harness + kill gate (Plan.md Section 15, 37-39)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compare.baselines import (
    belady_optimal, lru_hit_count, lfu_hit_count,
    gds_like_hit_count, gdsf_like_hit_count, peek_like_hit_count,
    POLICIES, regret_vs_belady,
)
from compare.harness import evaluate_workload, run_phase0a, phase0a_kill_gate
from simulator import workloads as w


SEQ = w.zipf_like(30, 300)
CAP = 4


def test_belady_is_upper_bound():
    """No correct policy may exceed Belady (regret definition)."""
    belady = belady_optimal(SEQ, CAP)
    for name, fn in POLICIES.items():
        if name == "Belady":
            continue
        hits = fn(SEQ, CAP)
        assert hits <= belady, f"{name} exceeded Belady"
        assert regret_vs_belady(hits, belady) <= 0


def test_all_policies_registered():
    assert set(POLICIES) == {"LRU", "LFU", "GDS-like", "GDSF-like", "PEEK-like", "Belady"}


def test_session_workload_saturates():
    """Tight multi-turn sessions fit in cache: all policies hit ~100%."""
    seq = w.multi_turn_session(4, 6)
    for name, fn in POLICIES.items():
        rate = fn(seq, CAP) / len(seq)
        assert rate > 0.8, f"{name} unexpectedly low on session workload"


def test_kill_gate_continues_when_headroom_exists():
    wl = {"zipf": SEQ, "agent": w.agent_workflow(50, 5)}
    results = run_phase0a(wl, capacity=CAP)
    gate = phase0a_kill_gate(results)
    assert gate["kill_triggered"] is False
    assert "gaps_pp" in gate


def test_kill_gate_triggers_on_no_headroom():
    """Session workload alone: Belady == LRU -> gap ~0 -> kill flag."""
    results = run_phase0a({"session": w.multi_turn_session(4, 6)}, capacity=CAP)
    gate = phase0a_kill_gate(results)
    assert gate["kill_triggered"] is True


def test_harness_result_shape():
    r = evaluate_workload("zipf", SEQ, CAP)
    assert r.n_requests == len(SEQ)
    assert set(r.hit_rates) == set(POLICIES)
    assert set(r.regrets_vs_belady) == set(POLICIES)
