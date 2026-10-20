"""§11 Computational Memory Architecture — L0..L4 tier model.

Not a GPU/CPU/SSD real transfer layer yet — pure structural model
that Planner can reference when deciding placement / movement / prefetch.
LMCache-style hierarchy used as reference baseline (§11 note).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Tier:
    name: str          # "L0", "L1", ... "L4"
    label: str         # "Active GPU KV", "GPU Resident", ...
    capacity_abstract: int  # abstract units (not bytes yet)
    latency_estimate: float  # relative time units
    cost_factor: float  # retention cost coefficient (shadow price proxy)


TIER_ARCHITECTURE = [
    Tier("L0", "Active GPU KV", capacity_abstract=8, latency_estimate=1.0, cost_factor=1.5),
    Tier("L1", "GPU Resident Cache", capacity_abstract=32, latency_estimate=3.0, cost_factor=1.0),
    Tier("L2", "Host RAM Cache", capacity_abstract=256, latency_estimate=20.0, cost_factor=0.3),
    Tier("L3", "Local SSD / NVMe", capacity_abstract=1024, latency_estimate=200.0, cost_factor=0.05),
    Tier("L4", "Remote KV Store", capacity_abstract=9999, latency_estimate=2000.0, cost_factor=0.01),
]


class MemoryTierModel:
    """§11: which KV, which tier, when to move / prefetch —
    decided by Semantic Demand + Runtime State (§11 core thesis)."""
    def __init__(self, tiers=None):
        self.tiers = tiers or TIER_ARCHITECTURE

    def tier_by_name(self, name: str) -> Optional[Tier]:
        for t in self.tiers:
            if t.name == name:
                return t
        return None

    def best_tier_for_reuse(self, future_use_prob: float, confidence: float) -> str:
        # Simplified heuristic: high future demand + high conf → L0/L1
        # Low conf / low demand → L3/L4 (cheap retention)
        # This connects to §14 Memory Shadow Price and §13 Value Function later
        if future_use_prob > 0.7 and confidence > 0.8:
            return "L1"
        elif future_use_prob > 0.3:
            return "L2"
        else:
            return "L3"
