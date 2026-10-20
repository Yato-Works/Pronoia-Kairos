"""Shared Runtime State — immutable snapshot for Planner reference."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass(frozen=True)
class RequestState:
    input_tokens: int = 0
    expected_output_tokens: int = 0
    shared_prefix_length: int = 0
    prefix_reuse_probability: float = 0.0
    session_id: str = ""
    session_age: float = 0.0
    quality_sensitivity: float = 1.0  # 0..1, higher = stricter quality


@dataclass(frozen=True)
class SemanticState:
    future_use_probability: float = 0.0  # P_future_demand
    confidence: float = 0.0


@dataclass(frozen=True)
class QueueState:
    queue_wait: float = 0.0
    queue_depth: int = 0
    batch_pressure: float = 0.0


@dataclass(frozen=True)
class CacheState:
    memory_pressure: float = 0.0  # 0..1
    available_vram_gb: float = 0.0


@dataclass(frozen=True)
class HardwareState:
    gpu_util: float = 0.0
    memory_util: float = 0.0


@dataclass(frozen=True)
class TelemetryState:
    # empty initially; filled by Planner after execution planning
    observations: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionState:
    # placeholder until real execution
    mode: str = "normal"


@dataclass(frozen=True)
class TransferState:
    # placeholder for KV offload / prefetch movement
    transfer_cost_estimate: float = 0.0


@dataclass(frozen=True)
class RuntimeState:
    request_state: RequestState = field(default_factory=RequestState)
    session_state: Dict[str, Any] = field(default_factory=dict)
    semantic_state: SemanticState = field(default_factory=SemanticState)
    context_state: Dict[str, Any] = field(default_factory=dict)
    queue_state: QueueState = field(default_factory=QueueState)
    cache_state: CacheState = field(default_factory=CacheState)
    transfer_state: TransferState = field(default_factory=TransferState)
    execution_state: ExecutionState = field(default_factory=ExecutionState)
    hardware_state: HardwareState = field(default_factory=HardwareState)
    telemetry_state: TelemetryState = field(default_factory=TelemetryState)
