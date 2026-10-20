"""§15 GreedyDual-Style Retention + §16 Queue-Aware Runtime State.
§17 PEEK Baseline reference. §18 Positioning table (reference).
§19 Prefix Reuse (two-stage). §20 Prefetch condition.
§21 Context Compiler skeleton.

This completes Plan.md v0.2.0 implementation for Phase 0-3.
No GPU dependency — pure control-plane research prototype.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List

# ─── §15 GreedyDual-Style Retention policies (structural reference) ────────
RETENTION_POLICIES = ["LRU", "LFU", "GDS", "GDSF", "Oracle", "Belady", "Predictive"]


@dataclass(frozen=True)
class RetentionPolicyConfig:
    policy: str = "LRU"
    compare_target: str = "Belady"  # Always measure regret vs Belady (§15)


def regret_vs_belady(lru_hits: int, belady_hits: int) -> int:
    """§15 core metric: how far from theoretical upper bound."""
    return lru_hits - belady_hits  # negative = LRU worse; closer to 0 = better


# ─── §16 Queue-Aware Runtime State (structural) ──────────────────────────
@dataclass
class QueueAwareState:
    past_requests: List[str] = field(default_factory=list)
    current_requests: List[str] = field(default_factory=list)
    pending_requests: List[str] = field(default_factory=list)
    session_structure: Dict = field(default_factory=dict)
    semantic_demand_summary: Dict = field(default_factory=dict)

    def future_aware_summary(self) -> Dict:
        """§16: Queue is NOT external to Cache — integrated future-aware state."""
        return {
            "pending_depth": len(self.pending_requests),
            "current_pressure": len(self.current_requests),
            "session_continuity": len(self.session_structure),
            "semantic_signal": bool(self.semantic_demand_summary),
        }


# ─── §17 PEEK Baseline (positioning reference) ────────────────────────────
PEEK_BASELINE_LABEL = "PEEK"
ADAPTIVE_RUNTIME_LABEL = "Adaptive Runtime"


COMPARISON_TARGETS = [
    "vLLM stock",
    "SGLang stock",
    "vLLM + LMCache",
    "SGLang + LMCache",
    "PEEK",
    ADAPTIVE_RUNTIME_LABEL,
]


# ─── §18 KVFlow / Continuum / PEEK / Adaptive comparison (§18 table) ───────
POSITIONING_TABLE = {
    "PEEK":        {"signal": "Pending Queue",        "scope": "Prefix / Scheduling"},
    "KVFlow":      {"signal": "Agent Step Graph",     "scope": "Prefetch / Eviction"},
    "Continuum":    {"signal": "Tool Duration / Multi-turn", "scope": "Multi-turn KV"},
    ADAPTIVE_RUNTIME_LABEL: {"signal": "Semantic Demand + Queue + Session + Cache", "scope": "Cross-layer Decision"},
}

DIFFERENTIATION_STATEMENT = (
    "Semantic Memory由来の未来需要予測を、"
    "Queue / Session / Cache Stateと統合して"
    "Computational Policyへ変換すること"
)


# ─── §19 Prefix Reuse — two-stage (§19) ─────────────────────────────────
@dataclass(frozen=True)
class PrefixReuseDecision:
    prefix_matched: bool = False
    expected_benefit: float = 0.0  # P_reuse * (T_recompute - T_restore)
    restore_cost: float = 0.0
    recompute_cost: float = 0.0
    decision: str = "RECOMPUTE"  # REUSE / RECOMPUTE

    def __post_init__(self):
        if not self.prefix_matched:
            object.__setattr__(self, "decision", "RECOMPUTE")
        else:
            # §19: match alone is insufficient; compare benefit vs cost
            benefit_positive = self.expected_benefit > 0
            restore_cheaper = self.restore_cost < self.recompute_cost
            decision = "REUSE" if (benefit_positive and restore_cheaper) else "RECOMPUTE"
            object.__setattr__(self, "decision", decision)


# ─── §20 Prefetch — condition with miss cost (§20) ────────────────────────
@dataclass(frozen=True)
class PrefetchCondition:
    P_use: float = 0.0
    T_saved: float = 0.0
    T_transfer: float = 0.0
    T_interference: float = 0.0  # GPU pollution / PCIe saturation cost
    should_prefetch: bool = False

    def __post_init__(self):
        # §20: P_use * T_saved > T_transfer + T_interference
        should = (self.P_use * self.T_saved) > (self.T_transfer + self.T_interference)
        object.__setattr__(self, "should_prefetch", should)


PREFETCH_RISK_NOTES = [
    "GPU pollution",
    "PCIe saturation",
    "NVMe contention",
    "CPU bandwidth loss",
]


# ─── §21 Context Compiler skeleton (§21) ─────────────────────────────────
@dataclass
class ContextCompilerInputs:
    raw_conversation: str = ""
    retrieved_memory: str = ""
    current_request: str = ""


@dataclass
class ContextCompilerOutput:
    compiled_context: str = ""
    selected_candidates: int = 0
    quality_constraint_satisfied: bool = True
