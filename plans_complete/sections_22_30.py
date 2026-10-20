"""§22 Context Utility Model — pure formula structure.
§23 Joint Budget concept — resource allocation problem skeleton.
§24-§26 Scheduling / Fairness — lexicographic structure.
§27 Adaptive Execution — mode interface only (v0.x scope per Plan.md).
§28 Speculation Controller — skeleton (out of scope for v0.x per Plan.md).
§29 PD Disaggregation — architecture interface (v0.x interface only).
§30 Cognitive IR — partial skeleton (future).

These sections complete Plan.md §22-§30 as defined.
No GPU dependency kept; pure structural/prototype implementation.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional

# ─── §22 Context Utility Model ────────────────────────────────────────────
@dataclass(frozen=True)
class MemoryUtilityInputs:
    relevance: float = 0.0
    confidence: float = 0.0
    provenance: str = ""
    token_cost: int = 0
    prefill_cost: float = 0.0
    future_reuse: float = 0.0
    utility: float = 0.0  # computed externally by U(m)


def U_context(m: MemoryUtilityInputs) -> float:
    """§22: U(m) = Benefit(m) - TokenCost(m) - PrefillCost(m)"""
    benefit = m.relevance * m.confidence * (1 + m.future_reuse)
    return benefit - (m.token_cost * 0.001) - m.prefill_cost


# Quality Safety Constraint (§22, §23)
class QualitySafetyConstraint:
    def __init__(self, epsilon: float = 0.05):
        self.epsilon = epsilon  # max allowed quality loss

    def satisfied(self, loss_estimate: float) -> bool:
        return loss_estimate <= self.epsilon


# ─── §23 Context / KV Joint Budget — concept equation ───────────────────────
@dataclass
class JointBudgetProblem:
    """§23: minimize total cost (prefill + restore + memory + transfer)
    subject to quality <= epsilon, VRAM <= B_GPU, RAM <= B_CPU."""
    C_prefill: float = 0.0
    C_restore: float = 0.0
    C_memory: float = 0.0
    C_transfer: float = 0.0
    quality_loss_estimate: float = 0.0
    vram_usage: float = 0.0
    ram_usage: float = 0.0
    B_GPU: float = float('inf')
    B_CPU: float = float('inf')

    def feasible(self) -> bool:
        return (self.quality_loss_estimate <= 0.05 and
                self.vram_usage <= self.B_GPU and
                self.ram_usage <= self.B_CPU)

    def total_cost(self) -> float:
        return self.C_prefill + self.C_restore + self.C_memory + self.C_transfer


# ─── §24 Scheduler — lexicographic priority (§25 detail) ───────────────────
# Plan.md removes fixed linear score model; uses lexicographic priority
SCHEDULER_LEXICOGRAPHIC_ORDER = [
    "Hard_SLO_protection",
    "Starvation_bound",
    "Reuse_gain",
    "Batch_compatibility",
    "Cache_locality",
    "Transfer_cost",
]


# ─── §25 Constraint-Based Scheduling — structural definition ───────────────
@dataclass
class SchedulingConstraints:
    slo_constraints: Dict = None  # type: ignore
    fairness_constraints: Dict = None  # type: ignore
    memory_constraints: Dict = None  # type: ignore
    batch_constraints: Dict = None  # type: ignore


# ─── §26 Fairness — structural reference ────────────────────────────────────
FAIRNESS_METHODS = ["Maximum_Queue_Wait", "Age_based_priority"]


# ─── §27 Adaptive Execution — interface only (§27 Plan.md note) ────────────
EXECUTION_MODES = [
    "NORMAL_DECODE", "BATCHED_DECODE", "PREFIX_REUSE",
    "SPECULATIVE", "MTP", "CHUNKED_PREFILL", "PD_DISAGGREGATION",
]


@dataclass
class ExecutionModeSelector:
    """§27: interface for selecting execution mode — not core research for v0.x."""
    available_modes: List[str] = None  # type: ignore
    selected_mode: str = "NORMAL_DECODE"

    def __post_init__(self):
        if self.available_modes is None:
            object.__setattr__(self, "available_modes", EXECUTION_MODES)


# ─── §28 Speculation Controller — skeleton (§28 out of scope for v0.x) ────
@dataclass
class SpeculationControllerState:
    draft_cost: float = 0.0
    acceptance_rate: float = 0.0
    verification_cost: float = 0.0
    memory_pressure: float = 0.0
    queue_state: Dict = None  # type: ignore


def expected_speculation_gain(
    P_accept: float, saved_cost: float,
    draft_cost: float, verification_overhead: float,
) -> float:
    """§28 conceptual equation — out of scope for v0.x; skeleton only."""
    return P_accept * saved_cost - draft_cost - verification_overhead


# ─── §29 PD Disaggregation — architecture interface (§29 v0.x interface) ──
@dataclass
class PDDisaggregationFlow:
    """§29: Prefill planner → KV Transfer → Decode planner interface."""
    prefill_plan: Optional[str] = None
    decode_plan: Optional[str] = None
    kv_transfer_plan: Optional[str] = None


# ─── §30 Cognitive IR — partial skeleton (§30 future) ──────────────────────
class CognitiveIRSkeleton:
    """§30: future execution representation skeleton."""
    def __init__(self):
        self.representation_type = "future"

    def describe(self) -> str:
        return "Cognitive IR — future execution representation (Plan.md §30)"
