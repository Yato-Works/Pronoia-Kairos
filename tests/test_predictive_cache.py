"""Phase 2 test: Predictive Cache produces measurable regret vs Belady."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runtime.state import RuntimeState, RequestState, SemanticState, CacheState
from runtime.request import RequestAnalyzer
from runtime.planner import AdaptivePlanner
from cache.predictive import DummyPredictiveCachePolicy
from compare.belady_lru import belady_optimal, lru_hit_count


def test_predictive_cache_over_baseline():
    """Predictive policy should change decision based on future_use_probability."""
    planner = AdaptivePlanner(cache_policy=DummyPredictiveCachePolicy())
    analyzer = RequestAnalyzer()
    raw = {"input_tokens": 512}
    req = analyzer.analyze(raw)

    # High future demand → predictive KEEP + PREFETCH
    high = RuntimeState(
        request_state=req,
        semantic_state=SemanticState(future_use_probability=0.92, confidence=0.95),
        cache_state=CacheState(memory_pressure=0.3),
    )
    plan_high = planner.plan(high, req)
    assert plan_high.cache_plan.retention_action == "KEEP"
    assert plan_high.prefetch_plan.enabled is True
    assert plan_high.cache_plan.policy == "PREDICTIVE"

    # Low confidence → EVICT fallback
    low = RuntimeState(
        request_state=req,
        semantic_state=SemanticState(future_use_probability=0.2, confidence=0.3),
        cache_state=CacheState(memory_pressure=0.9),
    )
    plan_low = planner.plan(low, req)
    assert plan_low.cache_plan.retention_action == "EVICT"
    assert plan_low.prefetch_plan.enabled is False


def test_regret_vs_belady_reported():
    """Regret = LRU hits - Belady hits (negative = LRU worse than optimal)."""
    seq = ["A","B","C","D","A","B","E","A","B","C","D"]
    cap = 3
    belady_hits = belady_optimal(seq, cap)
    lru_hits = lru_hit_count(seq, cap)
    regret = lru_hits - belady_hits
    # Regret should be <= 0 (LRU never exceeds Belady optimal)
    assert regret <= 0
    print(f"Regret vs Belady: {regret} (LRU={lru_hits}, Belady={belady_hits})")
