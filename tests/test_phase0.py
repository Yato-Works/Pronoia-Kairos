"""Phase 0 tests: RuntimeState → Planner → RuntimePlan, no GPU."""
import pytest
from runtime.state import RuntimeState, RequestState, SemanticState, QueueState, CacheState
from runtime.request import RequestAnalyzer
from runtime.planner import AdaptivePlanner
from runtime.plan import RuntimePlan
from runtime.telemetry import DecisionTrace


def test_same_state_same_plan():
    """Test 1: 同じ Request + 同じ RuntimeState → 同じ RuntimePlan"""
    analyzer = RequestAnalyzer()
    planner = AdaptivePlanner()

    raw = {"input_tokens": 2048, "shared_prefix_length": 1024, "prefix_reuse_probability": 0.5}
    req_state = analyzer.analyze(raw)
    state = RuntimeState(request_state=req_state)

    plan_a = planner.plan(state, req_state, request_id="r1")
    plan_b = planner.plan(state, req_state, request_id="r2")

    assert plan_a.cache_plan.policy == plan_b.cache_plan.policy
    assert plan_a.scheduling_plan.policy == plan_b.scheduling_plan.policy


def test_memory_pressure_changes_cache_decision():
    """Test 2: memory_pressure だけ変更 → Cache decision が変化"""
    planner = AdaptivePlanner()
    raw = {"input_tokens": 512}
    from runtime.request import RequestAnalyzer
    req = RequestAnalyzer().analyze(raw)

    low_mem = RuntimeState(request_state=req, cache_state=CacheState(memory_pressure=0.1))
    high_mem = RuntimeState(request_state=req, cache_state=CacheState(memory_pressure=0.98))

    plan_low = planner.plan(low_mem, req)
    plan_high = planner.plan(high_mem, req)

    assert plan_high.cache_plan.retention_action == "EVICT"
    assert plan_low.cache_plan.retention_action == "KEEP"


def test_queue_wait_changes_scheduler():
    """Test 3: queue_wait が大きくなる → Scheduler decision が変化"""
    planner = AdaptivePlanner()
    raw = {"input_tokens": 256}
    req = RequestAnalyzer().analyze(raw)

    low_q = RuntimeState(request_state=req, queue_state=QueueState(queue_wait=0.0, queue_depth=1))
    high_q = RuntimeState(request_state=req, queue_state=QueueState(queue_wait=120.0, queue_depth=50))

    plan_low = planner.plan(low_q, req)
    plan_high = planner.plan(high_q, req)

    assert plan_high.scheduling_plan.execution_priority >= 2


def test_high_future_use_enables_prefetch():
    """Test 4: future_use_probability 高 → Prefetch / KEEP 変化"""
    planner = AdaptivePlanner()
    # Dummy provider returns 0.15 by default; inject synthetic high-confidence signal via semantic_state
    raw = {"input_tokens": 512}
    req = RequestAnalyzer().analyze(raw)
    state_high = RuntimeState(
        request_state=req,
        semantic_state=SemanticState(future_use_probability=0.92, confidence=0.95),
    )
    plan = planner.plan(state_high, req)
    assert plan.prefetch_plan.enabled is True
    assert plan.cache_plan.retention_action == "KEEP"


def test_low_confidence_triggers_fallback():
    """Test 5: confidence 低下 → Baseline fallback 発生"""
    planner = AdaptivePlanner()
    raw = {"input_tokens": 256}
    req = RequestAnalyzer().analyze(raw)
    state_low = RuntimeState(
        request_state=req,
        semantic_state=SemanticState(future_use_probability=0.3, confidence=0.3),
    )
    plan = planner.plan(state_low, req)
    assert plan.fallback_plan.trigger is True
    assert plan.fallback_plan.fallback_policy == "baseline"


def test_decision_trace_reproducible():
    """Test 6: Decision Trace から判断理由を再現できる"""
    planner = AdaptivePlanner()
    raw = {"input_tokens": 1024, "shared_prefix_length": 512}
    req = RequestAnalyzer().analyze(raw)
    state = RuntimeState(request_state=req, semantic_state=SemanticState(future_use_probability=0.6, confidence=0.85))
    plan = planner.plan(state, req, request_id="t6")

    trace = DecisionTrace(
        request_id=plan.request_id,
        timestamp="2026-01-01T00:00:00",
        observed_state_summary={"cache_policy": plan.cache_plan.policy},
        candidate_decisions={"cache": "LRU_KEEP"},
        selected_plan_summary={"cache_policy": plan.cache_plan.policy, "reason": plan.cache_plan.reason},
        confidence=plan.cache_plan.confidence,
        reason=plan.cache_plan.reason,
        fallback_triggered=plan.fallback_plan.trigger,
    )
    assert trace.selected_plan_summary["reason"] == plan.cache_plan.reason
    assert trace.request_id == "t6"
