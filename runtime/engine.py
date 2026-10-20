"""Phase 6 — Execution Engine (Plan.md §59, §36, §80 Hardware Abstraction).

§59's architecture ends at "Execution Engine -> CUDA / CPU / ...". This module
fixes that boundary: the Adaptive Runtime hands the engine a planned request
and receives TTFT / TPOT / E2E priced by an explicit latency model — never a
single magic number (§36 decomposes; nothing is a fixed constant).

SimulatedExecutionEngine prices every stage with simulator/latency.py:

    TTFT = queue_wait + t_context + t_cache_lookup + t_sync
           + stage(cached, uncached) + prefetch_spill
    stage = max(restore, prefill)        # §36 prefetch/restore overlap
          = restore + prefill            # when overlap is disabled
    restore = t_transfer_per_token * cached_tokens + t_restore
    prefill = t_prefill_per_token * uncached_tokens
    E2E  = TTFT + t_decode_per_token * (output_tokens - 1)

Honest boundary (§43): this is the control-plane prototype's engine — abstract
time units, no GPU. The interface (execute + the pricing helpers the runtime
needs) is the deliverable; a native engine (§93 / §116 "Native Runtime")
implements the same surface behind §80's hardware abstraction without touching
the runtime. Every configuration in the Phase 6 ladder shares ONE engine and
ONE latency model, so measured differences isolate DECISIONS, not pricing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from simulator.latency import LatencyModel


@dataclass(frozen=True)
class EngineRequest:
    """One planned request as handed to the engine by the runtime (§59)."""
    request_id: str
    input_tokens: int            # scheduled context length (prefix + kept chunks)
    output_tokens: int
    cached_tokens: int           # part of input already resident in KV
    queue_wait: float = 0.0
    prefetch_spill: float = 0.0  # §36: prefetch transfer that failed to overlap


@dataclass(frozen=True)
class EngineResult:
    ttft: float
    tpot: float
    e2e: float
    restore_time: float
    prefill_time: float


class SimulatedExecutionEngine:
    """§36 decomposed latency model behind the §59/§80 engine interface."""

    def __init__(self, model: Optional[LatencyModel] = None,
                 overlap: bool = True):
        self.model = model or LatencyModel()
        self.overlap = overlap

    def restore_time(self, cached_tokens: int) -> float:
        """T_transfer + T_restore for KV-resident tokens (§36 T_reuse)."""
        if cached_tokens <= 0:
            return 0.0
        return self.model.transfer_cost(cached_tokens)

    def prefill_time(self, uncached_tokens: int) -> float:
        """T_prefill (T_recompute) for tokens that must be computed (§36)."""
        return self.model.t_prefill_per_token * max(0, uncached_tokens)

    def stage_time(self, cached_tokens: int, uncached_tokens: int) -> float:
        """The compute/transfer window a prefetch can race into (§36:
        T_effective = max(T_compute, T_transfer)). The runtime uses this to
        price prefetch overlap without duplicating the formula."""
        restore = self.restore_time(cached_tokens)
        prefill = self.prefill_time(uncached_tokens)
        return max(restore, prefill) if self.overlap else restore + prefill

    def execute(self, req: EngineRequest) -> EngineResult:
        m = self.model
        cached = max(0, min(req.cached_tokens, req.input_tokens))
        uncached = req.input_tokens - cached
        restore = self.restore_time(cached)
        prefill = self.prefill_time(uncached)
        stage = max(restore, prefill) if self.overlap else restore + prefill
        ttft = (req.queue_wait + m.t_context + m.t_cache_lookup
                + stage + m.t_sync + req.prefetch_spill)
        tpot = m.t_decode_per_token
        e2e = ttft if req.output_tokens <= 1 else ttft + tpot * (req.output_tokens - 1)
        return EngineResult(ttft=ttft, tpot=tpot, e2e=e2e,
                            restore_time=restore, prefill_time=prefill)
