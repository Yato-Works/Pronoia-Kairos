"""Phase 3 — Semantic Demand Predictor (Plan.md §7, §8, §9).

Two-stage probability model (§9.1):

    P_semantic(x | S_t)  ->  P_reuse(c | x, S_t, C_t)

Lethe provides: memory_id, source_id, chunk_id, relevance, confidence,
provenance, temporal_relation, session_relation, access_history, future_utility.

The predictor converts these into P_future_demand(x | S_t) without
directly controlling KV Cache — Adaptive Runtime combines this with
request history + session + queue + cache state (§8).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List
from abc import ABC, abstractmethod


@dataclass
class MemoryRecord:
    memory_id: str = ""
    source_id: str = ""
    chunk_id: str = ""
    relevance: float = 0.0
    confidence: float = 0.0
    provenance: str = ""
    temporal_relation: str = ""
    session_relation: str = ""
    access_history: List[float] = field(default_factory=list)
    future_utility: float = 0.0  # Lethe's estimated future value


class SemanticDemandPredictor(ABC):
    """§7.2: P_future_demand(x | S_t)"""
    @abstractmethod
    def predict(self, session_state: Dict[str, Any],
                memories: List[MemoryRecord]) -> Dict[str, Any]:
        ...


class DummySemanticDemandPredictor(SemanticDemandPredictor):
    # Adapter for interface compatibility (§8 Integration)
    def predict_future_demand(self, state=None) -> dict:
        return self.predict(
            getattr(state, "session_state", {}) if state else {},
            [],
        )

    """Placeholder predictor: uses Lethe's future_utility directly
    as P_semantic proxy (§9.1 first stage)."""
    def predict(self, session_state, memories) -> Dict[str, Any]:
        if not memories:
            return {
                "P_semantic_max": 0.0,
                "top_memory_id": None,
                "future_demand_items": [],
                "future_use_probability": 0.0,
                "confidence": 0.0,
            }
        # Aggregate future_utility from Lethe records
        top = max(memories, key=lambda m: m.future_utility)
        avg_utility = sum(m.future_utility for m in memories) / len(memories)
        return {
            "P_semantic_max": top.relevance * top.confidence,
            "top_memory_id": top.memory_id,
            "top_chunk_id": top.chunk_id,
            "top_future_utility": top.future_utility,
            "avg_future_utility": avg_utility,
            "future_demand_items": [
                {"memory_id": m.memory_id, "chunk_id": m.chunk_id,
                 "P_semantic": m.relevance * m.confidence * m.future_utility}
                for m in memories
            ],
            "future_use_probability": min(0.99, top.relevance * top.confidence * (1 + top.future_utility)),
            "confidence": top.confidence,
        }
