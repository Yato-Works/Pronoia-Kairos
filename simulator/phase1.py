"""Simulator — Phase 1: Request stream + Cache state + Memory capacity.

This is NOT a GPU inference simulator. It is the Pronoia Kairos Runtime
experimental ground:

    Request stream (access sequence)
        ↓
    Simulator feeds RuntimeState → AdaptivePlanner
        ↓
    RuntimePlan (with DecisionTrace record)
        ↓
    Metrics: hit_rate, miss_rate, evictions, prefetch_count, wasted_prefetch,
             memory_utilization, regret_vs_belady
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class AccessEvent:
    key: str  # e.g. "A", "B", "C"
    size: int = 1  # abstract memory units


@dataclass
class CacheMetrics:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    prefetches: int = 0
    wasted_prefetches: int = 0  # prefetches that were never accessed
    memory_utilization: float = 0.0
    total_requests: int = 0

    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def miss_rate(self) -> float:
        total = self.hits + self.misses
        return self.misses / total if total > 0 else 0.0


@dataclass
class SimulationConfig:
    memory_capacity: int = 4  # abstract units
    access_sequence: List[str] = field(default_factory=list)


class Simulator:
    """Phase 1 core: run access sequence through Runtime + Planner + baseline policies."""

    def __init__(self, planner=None):
        self.planner = planner
        self.metrics = CacheMetrics()
        self.cache_memory: List[str] = []  # simulated memory slots
        self.prefetch_set: set = set()

    def run_baseline_sequence(
        self,
        sequence: List[str],
        config: SimulationConfig,
    ) -> CacheMetrics:
        """Basic LRU-style simulation for baseline comparison.
        Planner is invoked but execution is synthetic.
        """
        self.cache_memory = []
        self.prefetch_set = set()
        self.metrics = CacheMetrics()

        for i, key in enumerate(sequence):
            self.metrics.total_requests += 1
            # Synthetic Planner invocation: create dummy RuntimeState
            # (Actual state construction deferred to full integration)
            if key in self.cache_memory:
                self.metrics.hits += 1
                # LRU update: move to end
                self.cache_memory.remove(key)
                self.cache_memory.append(key)
            else:
                self.metrics.misses += 1
                # Eviction if at capacity
                if len(self.cache_memory) >= config.memory_capacity:
                    evicted = self.cache_memory.pop(0)
                    self.metrics.evictions += 1
                self.cache_memory.append(key)
        return self.metrics
