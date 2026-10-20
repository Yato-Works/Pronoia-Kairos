"""Semantic Demand Provider — interface only; dummy for Phase 0."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict
from semantic.demand_predictor import DummySemanticDemandPredictor

# Neutral prior used when no semantic evidence (Lethe memories) is available.
# A runtime with zero evidence must NOT be permanently in "low confidence
# fallback" mode (§105 Conservative Default) — that would disable the control
# plane by default. Values follow Plan.md Phase 0 test expectations:
# low future demand (no prefetch trigger) + high confidence (no fallback).
BASELINE_FUTURE_USE_PROBABILITY = 0.15
BASELINE_CONFIDENCE = 0.9


class SemanticDemandProvider(ABC):
    @abstractmethod
    def predict_future_demand(self, state: Any) -> Dict[str, Any]:
        ...


class DummySemanticDemandProvider(SemanticDemandProvider):
    def predict_future_demand(self, state) -> Dict[str, Any]:
        # Delegate to DummySemanticDemandPredictor behavior
        predictor = DummySemanticDemandPredictor()
        signals = predictor.predict(
            getattr(state, "session_state", {}),
            [],  # No Lethe memories injected at interface layer
        )
        # Prior: no semantic evidence → neutral baseline, not zero-confidence.
        if signals.get("confidence", 0.0) <= 0.0:
            signals = dict(signals)
            signals["future_use_probability"] = signals.get("future_use_probability", 0.0) or BASELINE_FUTURE_USE_PROBABILITY
            signals["confidence"] = BASELINE_CONFIDENCE
            signals["source"] = "neutral_prior_no_evidence"
        return signals
