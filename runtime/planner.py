"""Adaptive Planner — Phase 0 core: RuntimeState → RuntimePlan."""
from __future__ import annotations
from typing import Dict, Any

from runtime.state import RuntimeState, RequestState, SemanticState, QueueState, CacheState, HardwareState
from runtime.plan import RuntimePlan, ContextPlan, CachePlan, SchedulingPlan, PrefetchPlan, ExecutionPlan, FallbackPlan
from semantic.interface import SemanticDemandProvider, DummySemanticDemandProvider
from cache.interface import CachePolicy, DummyCachePolicy
from scheduler.interface import SchedulerPolicy, DummySchedulerPolicy


class AdaptivePlanner:
    def __init__(
        self,
        semantic_provider: SemanticDemandProvider = None,
        cache_policy: CachePolicy = None,
        scheduler_policy: SchedulerPolicy = None,
    ):
        self.semantic = semantic_provider or DummySemanticDemandProvider()
        self.cache_policy = cache_policy or DummyCachePolicy()
        self.scheduler_policy = scheduler_policy or DummySchedulerPolicy()

    def _semantic_signals(self, state: RuntimeState) -> Dict[str, Any]:
        """§7/§9: semantic provider signals, with state-side override.

        state.semantic_state を優先（テストで直接注入された値を尊重）。
        """
        provider_signals = self.semantic.predict_future_demand(state)
        future_prob = state.semantic_state.future_use_probability if state.semantic_state.future_use_probability > 0 else provider_signals.get("future_use_probability", 0.0)
        conf = state.semantic_state.confidence if state.semantic_state.confidence > 0 else provider_signals.get("confidence", 0.0)
        # state 側が 0.0 でない限り、テストで注入された値を尊重
        if state.semantic_state.confidence > 0:
            conf = state.semantic_state.confidence
            future_prob = state.semantic_state.future_use_probability
        return {
            "future_use_probability": future_prob,
            "confidence": conf,
            "predicted_demand_items": provider_signals.get("predicted_demand_items", []),
        }

    def plan(self, state: RuntimeState, request: RequestState, request_id: str = "") -> RuntimePlan:
        # 1. Semantic signals
        semantic_signals = self._semantic_signals(state)

        # 2. Cache decision
        cache_decision = self.cache_policy.decide(state, request, semantic_signals)

        # 3. Scheduling decision
        schedule_decision = self.scheduler_policy.decide(state, request)

        # 4. Prefetch: only when future demand is high AND confidence is high
        future_prob = semantic_signals.get("future_use_probability", 0.0)
        conf = semantic_signals.get("confidence", 0.0)
        prefetch_enabled = (future_prob > 0.7) and (conf > 0.8) and not cache_decision.get("fallback_triggered", False)

        # 5. Fallback check: low confidence → baseline fallback
        low_confidence = conf < 0.5
        fallback = FallbackPlan(
            trigger=low_confidence or cache_decision.get("fallback_triggered", False),
            fallback_policy="baseline",
            reason="low_semantic_confidence_or_memory_pressure",
        )
        if fallback.trigger:
            # Override to safe baseline
            prefetch_enabled = False
            cache_decision = dict(cache_decision)
            cache_decision["retention_action"] = "EVICT"
            cache_decision["reason"] = "fallback_baseline_eviction"

        # 6. Build RuntimePlan
        plan = RuntimePlan(
            request_id=request_id,
            context_plan=ContextPlan(
                strategy="full_candidate" if conf > 0.6 else "minimal",
                selected_candidates=3 if conf > 0.6 else 1,
                confidence=conf,
            ),
            cache_plan=CachePlan(
                policy=cache_decision.get("policy", "LRU"),
                retention_action=cache_decision.get("retention_action", "KEEP"),
                prefetch_enabled=prefetch_enabled,
                confidence=cache_decision.get("confidence", 0.9),
                reason=cache_decision.get("reason", ""),
            ),
            scheduling_plan=SchedulingPlan(
                policy=schedule_decision.get("policy", "FIFO"),
                starvation_protection=schedule_decision.get("starvation_protection", True),
                execution_priority=schedule_decision.get("execution_priority", 1),
                confidence=schedule_decision.get("confidence", 0.9),
            ),
            prefetch_plan=PrefetchPlan(
                enabled=prefetch_enabled,
                targets=["predicted_high_utility_chunk"] if prefetch_enabled else [],
                confidence=conf,
            ),
            execution_plan=ExecutionPlan(
                mode="normal",
                speculative_decoding=False,
                batch_size=1,
                confidence=conf,
            ),
            fallback_plan=fallback,
        )
        return plan

    def plan_with_trace(
        self, state: RuntimeState, request: RequestState, request_id: str = "", timestamp: str = ""
    ):
        """§32/§99: produce RuntimePlan + DecisionTrace in one call.

        The trace captures observed state, candidate decisions and the
        selected plan so that any decision can be explained/replayed.
        """
        from runtime.telemetry import DecisionTrace, now_iso

        plan = self.plan(state, request, request_id=request_id)
        signals = self._semantic_signals(state)

        obs = {
            "cache_pressure": state.cache_state.memory_pressure,
            "queue_depth": state.queue_state.queue_depth,
            "queue_wait": state.queue_state.queue_wait,
            "future_use_probability": signals["future_use_probability"],
            "semantic_confidence": signals["confidence"],
            "fallback_triggered": plan.fallback_plan.trigger,
        }
        tags = []
        if plan.fallback_plan.trigger:
            tags.append("fallback_baseline")
        if plan.prefetch_plan.enabled:
            tags.append("prefetch_enabled")
        if plan.cache_plan.retention_action == "EVICT":
            tags.append("evict")
        elif plan.cache_plan.retention_action == "KEEP":
            tags.append("keep")

        trace = DecisionTrace(
            request_id=request_id,
            timestamp=timestamp or now_iso(),
            observed_state_summary=obs,
            candidate_decisions={
                "cache": f"{plan.cache_plan.policy}_{plan.cache_plan.retention_action}",
                "schedule": plan.scheduling_plan.policy,
                "prefetch": plan.prefetch_plan.enabled,
            },
            selected_plan_summary={
                "cache_policy": plan.cache_plan.policy,
                "retention_action": plan.cache_plan.retention_action,
                "execution_priority": plan.scheduling_plan.execution_priority,
                "reason": plan.cache_plan.reason,
            },
            confidence=plan.cache_plan.confidence,
            reason=plan.cache_plan.reason,
            fallback_triggered=plan.fallback_plan.trigger,
            explanation_tags=tags,
        )
        return plan, trace
