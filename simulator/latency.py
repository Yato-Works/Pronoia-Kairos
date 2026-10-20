"""§36 Latency Model — decomposed, not a single fixed number.

T_total = T_queue + T_context + T_cache_lookup + T_transfer
        + T_prefill + T_decode + T_sync

Cache hit:   T_reuse = T_lookup + T_transfer + T_restore
Cache miss:  T_miss  = T_lookup + T_recompute (= T_prefill on uncached prefix)
Prefetch:    T_effective = max(T_compute, T_transfer) when transfer overlaps
             compute; otherwise transfer is added on the critical path.

All values are abstract time units until calibrated against real hardware
(§43 Simulator Calibration Gate). No GPU dependency.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class LatencyModel:
    # Per-stage base costs (abstract time units)
    t_context: float = 2.0          # T_context: context assembly
    t_cache_lookup: float = 0.5     # T_cache_lookup: prefix/KV index probe
    t_transfer_per_token: float = 0.01   # PCIe/NVMe proxy per cached token
    t_restore: float = 1.0          # T_restore: KV de-serialize into device mem
    t_prefill_per_token: float = 0.05    # prefill compute per token
    t_decode_per_token: float = 0.2      # TPOT proxy per output token
    t_sync: float = 0.3             # T_sync: barrier / scheduling sync

    def ttft_hit(self, queue_wait: float, cached_tokens: int) -> float:
        """Time to first token when prefix KV is reused."""
        transfer = self.t_transfer_per_token * cached_tokens
        # Reused prefix: no prefill for cached tokens
        return (queue_wait + self.t_context + self.t_cache_lookup
                + transfer + self.t_restore + self.t_sync)

    def ttft_miss(self, queue_wait: float, input_tokens: int) -> float:
        """Time to first token when prefix must be recomputed (prefill)."""
        return (queue_wait + self.t_context + self.t_cache_lookup
                + self.t_prefill_per_token * input_tokens + self.t_sync)

    def tpot(self) -> float:
        return self.t_decode_per_token

    def e2e(self, ttft: float, output_tokens: int) -> float:
        """End-to-end = TTFT + TPOT * (output_tokens - 1)."""
        if output_tokens <= 1:
            return ttft
        return ttft + self.t_decode_per_token * (output_tokens - 1)

    def prefetch_overlap(self, compute_time: float, transfer_time: float) -> float:
        """§36 Prefetch: T_effective = max(T_compute, T_transfer)."""
        return max(compute_time, transfer_time)

    def transfer_cost(self, tokens: int) -> float:
        """§20 T_transfer: cost to fetch an object into device memory.

        Models PCIe/NVMe transfer of the object's KV: per-token transfer plus
        the fixed restore/de-serialize step. This is what a prefetch must pay
        up front, before it knows whether the object will actually be used.
        """
        return self.t_transfer_per_token * tokens + self.t_restore

    def saved_recompute(self, tokens: int) -> float:
        """§20 T_saved: recompute (prefill) cost avoided if the object is reused.

        A prefetch hit saves the full prefill of the object's prefix; a miss
        would have had to recompute it. This is the benefit side of the §20
        prefetch condition.
        """
        return self.t_prefill_per_token * tokens

    def interference_cost(self, tokens: int) -> float:
        """§20 T_interference: collateral cost of a (possibly wrong) prefetch.

        GPU pollution / PCIe saturation proxy. A prefetched object occupies
        device memory and bandwidth even if unused, so its transfer and
        residency impose a fixed interference penalty on the critical path.
        Scaled mildly by size to reflect larger objects polluting more.
        """
        return self.t_restore + self.t_transfer_per_token * tokens

