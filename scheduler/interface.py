"""Scheduler Policy — interface only; dummy FIFO + starvation guard."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict


class SchedulerPolicy(ABC):
    @abstractmethod
    def decide(self, state: Any, request: Any) -> Dict:
        ...


class DummySchedulerPolicy(SchedulerPolicy):
    def decide(self, state, request) -> Dict:
        q = state.queue_state if hasattr(state, "queue_state") else None
        wait_time = q.queue_wait if q else 0.0
        depth = q.queue_depth if q else 0
        return {
            "policy": "FIFO",
            "starvation_protection": True,
            "execution_priority": 2 if depth > 10 else 1,
            "confidence": 0.9,
            "wait_time": wait_time,
            "queue_depth": depth,
        }
