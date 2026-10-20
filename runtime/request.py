"""Request Analyzer — low-cost feature extraction (synthetic first)."""
from __future__ import annotations
from .state import RequestState, RuntimeState


class RequestAnalyzer:
    def analyze(self, raw: dict) -> RequestState:
        # Synthetic extraction — no real tokenizer yet
        return RequestState(
            input_tokens=int(raw.get("input_tokens", 0)),
            expected_output_tokens=int(raw.get("expected_output_tokens", 100)),
            shared_prefix_length=int(raw.get("shared_prefix_length", 0)),
            prefix_reuse_probability=float(raw.get("prefix_reuse_probability", 0.0)),
            session_id=str(raw.get("session_id", "")),
            session_age=float(raw.get("session_age", 0.0)),
            quality_sensitivity=float(raw.get("quality_sensitivity", 1.0)),
        )
