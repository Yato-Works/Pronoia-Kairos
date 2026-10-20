"""Decision Trace (Plan.md Section 78, 98-99): plan_with_trace + JSONL recorder."""
import sys, os, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runtime.planner import AdaptivePlanner
from runtime.state import RuntimeState, SemanticState, CacheState
from runtime.request import RequestAnalyzer
from runtime.telemetry import DecisionTraceRecorder


def make_plan_trace(planner, semantic=None, cache=None, request_id="req-1"):
    req = RequestAnalyzer().analyze({"input_tokens": 512})
    state = RuntimeState(
        request_state=req,
        semantic_state=semantic or SemanticState(),
        cache_state=cache or CacheState(),
    )
    return planner.plan_with_trace(state, req, request_id=request_id, timestamp="2026-01-01T00:00:00")


def test_plan_with_trace_returns_plan_and_trace():
    planner = AdaptivePlanner()
    plan, trace = make_plan_trace(planner)
    assert plan.request_id == "req-1"
    assert trace.request_id == "req-1"
    assert trace.timestamp == "2026-01-01T00:00:00"
    assert "cache" in trace.candidate_decisions
    assert "retention_action" in trace.selected_plan_summary


def test_trace_is_deterministic_for_same_state():
    """Section 99: trace lets us replay why a decision was made."""
    planner = AdaptivePlanner()
    _, t1 = make_plan_trace(planner)
    _, t2 = make_plan_trace(planner)
    assert t1.candidate_decisions == t2.candidate_decisions
    assert t1.selected_plan_summary == t2.selected_plan_summary
    assert t1.fallback_triggered == t2.fallback_triggered


def test_fallback_is_tagged_in_trace():
    planner = AdaptivePlanner()
    _, trace = make_plan_trace(
        planner,
        semantic=SemanticState(future_use_probability=0.1, confidence=0.2),
    )
    assert trace.fallback_triggered is True
    assert "fallback_baseline" in trace.explanation_tags


def test_prefetch_is_tagged_in_trace():
    planner = AdaptivePlanner()
    _, trace = make_plan_trace(
        planner,
        semantic=SemanticState(future_use_probability=0.95, confidence=0.97),
    )
    assert "prefetch_enabled" in trace.explanation_tags


def test_recorder_writes_and_reads_jsonl():
    planner = AdaptivePlanner()
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "planner_decisions.jsonl")
        rec = DecisionTraceRecorder(path)
        for i in range(3):
            _, trace = make_plan_trace(planner, request_id=f"req-{i}")
            rec.record(trace)
        loaded = rec.load()
        assert len(loaded) == 3
        assert loaded[0]["request_id"] == "req-0"
        assert len(rec) == 3
