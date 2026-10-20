"""Telemetry / Decision Trace — why the Planner decided this plan."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List
from datetime import datetime
import json
import os


@dataclass(frozen=True)
class DecisionTrace:
    request_id: str
    timestamp: str
    observed_state_summary: Dict[str, Any]
    candidate_decisions: Dict[str, Any] = field(default_factory=dict)
    selected_plan_summary: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    reason: str = ""
    fallback_triggered: bool = False
    explanation_tags: list = field(default_factory=list)
    # §92: when the plan was lowered to the Cognitive IR (§30/§57), the graph
    # summary (op counts, passes applied) rides along with the decision.
    ir_summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


class DecisionTraceRecorder:
    """§98/§99: append-only JSONL recorder for planner decisions.

    Every plan the Adaptive Planner emits can be persisted so that a
    decision can be replayed and explained later (research artifact:
    planner_decisions/).
    """

    def __init__(self, path: str):
        self.path = path
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)

    def record(self, trace: DecisionTrace) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(trace.to_dict(), ensure_ascii=False) + "\n")

    def load(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.path):
            return []
        with open(self.path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def __len__(self) -> int:
        return len(self.load())
