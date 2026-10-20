"""Runtime Plan — explicit execution plan produced by Adaptive Planner."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass(frozen=True)
class ContextPlan:
    strategy: str = "full_candidate"  # full_candidate / selective / minimal
    selected_candidates: int = 0
    confidence: float = 1.0


@dataclass(frozen=True)
class CachePlan:
    policy: str = "LRU"
    retention_action: str = "KEEP"  # KEEP / EVICT / OFFLOAD / PREFETCH
    prefetch_enabled: bool = False
    confidence: float = 1.0
    reason: str = ""


@dataclass(frozen=True)
class SchedulingPlan:
    policy: str = "FIFO"
    starvation_protection: bool = True
    execution_priority: int = 1
    confidence: float = 1.0


@dataclass(frozen=True)
class PrefetchPlan:
    enabled: bool = False
    targets: list = field(default_factory=list)
    confidence: float = 0.0


@dataclass(frozen=True)
class ExecutionPlan:
    mode: str = "normal"
    speculative_decoding: bool = False
    batch_size: int = 1
    confidence: float = 1.0


@dataclass(frozen=True)
class FallbackPlan:
    trigger: bool = False
    fallback_policy: str = "baseline"
    reason: str = ""


@dataclass(frozen=True)
class RuntimePlan:
    request_id: str = ""
    context_plan: ContextPlan = field(default_factory=ContextPlan)
    cache_plan: CachePlan = field(default_factory=CachePlan)
    scheduling_plan: SchedulingPlan = field(default_factory=SchedulingPlan)
    prefetch_plan: PrefetchPlan = field(default_factory=PrefetchPlan)
    execution_plan: ExecutionPlan = field(default_factory=ExecutionPlan)
    fallback_plan: FallbackPlan = field(default_factory=FallbackPlan)
