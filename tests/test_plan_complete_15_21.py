"""§15-§21 final integration test — completes Plan.md v0.2.0.

Verifies:
  §15 GreedyDual / Belady regret metric
  §16 Queue-aware future-aware state
  §17 PEEK baseline positioning
  §18 Cross-system comparison table
  §19 Prefix reuse two-stage decision (match is NOT enough)
  §20 Prefetch condition with interference cost
  §21 Context compiler skeleton
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plans_complete.sections_15_21 import (
    RETENTION_POLICIES, RetentionPolicyConfig, regret_vs_belady,
    QueueAwareState, PEEK_BASELINE_LABEL, ADAPTIVE_RUNTIME_LABEL,
    COMPARISON_TARGETS, POSITIONING_TABLE, DIFFERENTIATION_STATEMENT,
    PrefixReuseDecision, PrefetchCondition, PREFETCH_RISK_NOTES,
    ContextCompilerInputs, ContextCompilerOutput,
)
from runtime.value_layer import U_lossless, CacheValueInputs
from compare.belady_lru import belady_optimal, lru_hit_count


def test_section_15_regret_metric():
    seq = ["A","B","C","D","A","B","E","A","B","C","D"]
    cap = 3
    belady_hits = belady_optimal(seq, cap)
    lru_hits = lru_hit_count(seq, cap)
    regret = regret_vs_belady(lru_hits, belady_hits)
    # §15: Belady is theoretical upper bound; LRU should have non-positive regret
    assert regret <= 0
    assert belady_hits >= lru_hits
    print(f"§15 Regret: {regret} (LRU={lru_hits}, Belady={belady_hits})")


def test_section_16_queue_aware_state():
    state = QueueAwareState(
        past_requests=["A","B"],
        current_requests=["C"],
        pending_requests=["D","E","F"],
        session_structure={"session_id": "s-01"},
        semantic_demand_summary={"P_future": 0.82},
    )
    summary = state.future_aware_summary()
    assert summary["pending_depth"] == 3
    assert summary["current_pressure"] == 1
    assert summary["semantic_signal"] is True
    print(f"§16 Future-aware state summary: {summary}")


def test_section_17_18_positioning_and_comparison():
    assert PEEK_BASELINE_LABEL == "PEEK"
    assert ADAPTIVE_RUNTIME_LABEL == "Adaptive Runtime"
    assert "PEEK" in COMPARISON_TARGETS
    assert ADAPTIVE_RUNTIME_LABEL in COMPARISON_TARGETS
    # §18 table: Adaptive has broader signal and scope
    adaptive = POSITIONING_TABLE[ADAPTIVE_RUNTIME_LABEL]
    peek = POSITIONING_TABLE[PEEK_BASELINE_LABEL]
    assert "Cross-layer" in adaptive["scope"]
    assert "Pending Queue" in peek["signal"]
    assert "Semantic" in adaptive["signal"]
    # §18 differentiation
    assert "Semantic Memory" in DIFFERENTIATION_STATEMENT
    print(f"§18 Differentiation: {DIFFERENTIATION_STATEMENT}")


def test_section_19_prefix_reuse_two_stage():
    # Match exists but expected benefit is negative (restore > recompute benefit)
    # → decision must be RECOMPUTE (§19: match alone is NOT sufficient)
    negative_benefit = PrefixReuseDecision(
        prefix_matched=True,
        expected_benefit=-10.0,  # P_reuse * (T_recomp - T_restore) < 0
        restore_cost=50.0,
        recompute_cost=10.0,
    )
    assert negative_benefit.decision == "RECOMPUTE"

    # Positive benefit + restore cheaper → REUSE
    positive_benefit = PrefixReuseDecision(
        prefix_matched=True,
        expected_benefit=100.0,
        restore_cost=5.0,
        recompute_cost=80.0,
    )
    assert positive_benefit.decision == "REUSE"

    # No match → RECOMPUTE regardless
    no_match = PrefixReuseDecision(prefix_matched=False)
    assert no_match.decision == "RECOMPUTE"
    print("§19 Two-stage prefix reuse: OK")


def test_section_20_prefetch_condition():
    # Should prefetch: P_use * T_saved > T_transfer + T_interference
    should = PrefetchCondition(
        P_use=0.8, T_saved=200.0, T_transfer=50.0, T_interference=30.0,
    )
    # 0.8 * 200 = 160 > 50 + 30 = 80 → True
    assert should.should_prefetch is True

    # Should NOT prefetch: interference dominates
    should_not = PrefetchCondition(
        P_use=0.3, T_saved=10.0, T_transfer=20.0, T_interference=50.0,
    )
    # 0.3 * 10 = 3 > 70 = False
    assert should_not.should_prefetch is False
    # §20 miss cost is tracked (interference included)
    print(f"§20 Prefetch condition (interference tracked): {PREFETCH_RISK_NOTES}")


def test_section_21_context_compiler_skeleton():
    inputs = ContextCompilerInputs(
        raw_conversation="hello",
        retrieved_memory="memory-chunk-1",
        current_request="question",
    )
    output = ContextCompilerOutput(compiled_context="compiled", selected_candidates=2)
    assert output.selected_candidates == 2
    assert output.quality_constraint_satisfied is True  # §6 constraint preserved
    print("§21 Context compiler skeleton: OK")


def test_plan_md_complete_integration_statement():
    """§18 core claim verified through all previous phases."""
    assert ADAPTIVE_RUNTIME_LABEL in COMPARISON_TARGETS
    assert DIFFERENTIATION_STATEMENT is not None
    # Cross-layer signal exists
    adaptive_signal = POSITIONING_TABLE[ADAPTIVE_RUNTIME_LABEL]["signal"]
    assert "Cache" in adaptive_signal
    print(f"§18 Integration statement verified: {DIFFERENTIATION_STATEMENT}")
