"""§12 Lossless KV Layer — operations.
§13 Lossless Cache Value Function — U_lossless(c).
§14 Memory Shadow Price — λ_tier.

All in one file: pure mathematical structures, no GPU dependence.
These are the three core formulas from Plan.md v0.2.0.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


# ─── §12 Lossless Operations ──────────────────────────────────────────
LOSSLESS_OPERATIONS = [
    "KEEP", "EVICT", "OFFLOAD", "PREFETCH",
    "RESTORE", "REUSE", "PIN", "UNPIN",
]


@dataclass(frozen=True)
class CacheOperation:
    op: str  # one of LOSSLESS_OPERATIONS
    target_key: str = ""
    source_tier: str = ""
    dest_tier: str = ""


# ─── §13 Lossless Cache Value Function ──────────────────────────────────
@dataclass(frozen=True)
class CacheValueInputs:
    P_reuse: float             # probability of future reuse (§9)
    T_recompute: float         # expected recomputation time
    T_restore: float           # expected restore time
    C_retain: float           # retention cost (§14)
    # Separated retention costs (§13 note)
    gpu_cost: float = 0.0
    cpu_cost: float = 0.0
    ssd_cost: float = 0.0
    transfer_cost: float = 0.0
    prefetch_bandwidth_cost: float = 0.0


def U_lossless(inputs: CacheValueInputs) -> float:
    """§13: U_lossless(c) = P_reuse * (T_recompute - T_restore) - C_retain"""
    avoided_cost = inputs.P_reuse * (inputs.T_recompute - inputs.T_restore)
    total_retain = (
        inputs.gpu_cost + inputs.cpu_cost + inputs.ssd_cost
        + inputs.transfer_cost + inputs.prefetch_bandwidth_cost
    )
    # If inputs.C_retain is explicitly set, use it; otherwise sum separated
    retain = inputs.C_retain if inputs.C_retain > 0 else total_retain
    return avoided_cost - retain


# ─── §14 Memory Shadow Price ────────────────────────────────────────────
@dataclass(frozen=True)
class ShadowPrices:
    lambda_gpu: float = 1.0
    lambda_cpu: float = 0.5
    lambda_ssd: float = 0.1
    lambda_network: float = 0.05


def utility_tier(c_value: float, shadow: ShadowPrices, tier: str) -> float:
    """§14: Utility(c, tier) = ExpectedAvoidedCost - ResourcePrice(tier)"""
    price_map = {
        "L0": shadow.lambda_gpu,
        "L1": shadow.lambda_gpu,
        "L2": shadow.lambda_cpu,
        "L3": shadow.lambda_ssd,
        "L4": shadow.lambda_network,
    }
    price = price_map.get(tier, shadow.lambda_network)
    return c_value - price
