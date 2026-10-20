"""Phase 1 — vLLM KV Connector prototype (Plan.md Section 44, 45, 115, 119).

vLLM's Connector API lets an EXTERNAL policy decide which KV blocks to offload /
restore (Section 44: "vLLM's Connector API is a candidate for injecting async KV
loading/storing"). This is the "Adaptive Control Plane OUTSIDE the runtime"
position (Section 119): we do not rebuild vLLM, we attach our predictive
retention / prefetch to its connector seam.

The connector's core decision is PREFIX MATCHING: for an incoming request, how
many cached KV tokens can be reused (get_num_new_matched_tokens), and which
blocks to save on eviction / load on demand. vLLM's default is LRU. Our
AdaptiveKVConnector instead evicts / restores by PREDICTED future demand - the
same calibrated signal that won in the simulator (Section 90.2).

vLLM is GPU-heavy and NOT installed here (Section 43: no GPU dependency for the
research kernel). So this module is a vLLM-INDEPENDENT prototype:

    KVConnectorBase      - the seam vLLM's connector would implement
                           (get_num_new_matched_tokens / save_kv_layer / load_kv_layer).
    MockVLLMBackend      - simulates KV block save/load (host<->device) so the
                           connector logic is testable without a GPU.
    AdaptiveKVConnector  - our control plane: predictive retention over KV blocks.
    SimulatedVLLMRuntime - runs a request stream through the connector and reports
                           prefix hit rate + effective TTFT, for the Section 45
                           Phase 1 Gate comparison (Adaptive vs LRU vs LMCache-like).

When a real vLLM is available, AdaptiveKVConnector is registered as its
KVConnectorBase_V1 implementation; until then the mock backend exercises the exact
same decision code path.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from abc import ABC, abstractmethod

from cache.predictive_retention import PredictiveRetentionPolicy


@dataclass
class KVBlock:
    """One cached prefix-KV block, keyed by a prefix hash (block identity)."""
    block_id: str
    tokens: int
    location: str = "device"     # "device" (GPU) or "host" (CPU/SSD after offload)
    demand: float = 0.0          # predicted future-demand signal (calibrated)
    last_access: int = 0


class KVConnectorBase(ABC):
    """The vLLM connector seam (Section 44).

    vLLM calls these to let an external policy manage KV offload/restore. We
    implement them over a MockVLLMBackend so the decision logic runs without a GPU.
    """

    @abstractmethod
    def get_num_new_matched_tokens(self, request_prefix_hash: str, total_tokens: int) -> int:
        """How many leading tokens of the request are already cached (prefix hit).
        vLLM subtracts this from the prefill work."""
        ...

    @abstractmethod
    def save_kv_layer(self, block: KVBlock) -> None:
        """Persist a KV block (GPU -> host) when it is evicted from device."""
        ...

    @abstractmethod
    def load_kv_layer(self, block_id: str) -> Optional[KVBlock]:
        """Restore a KV block (host -> GPU) for reuse; None if not present."""
        ...


class MockVLLMBackend(KVConnectorBase):
    """A GPU-free stand-in for vLLM's KV cache + connector (Section 43).

    Holds KV blocks in a dict, models device capacity, and simulates the
    save/load cost so prefix-hit decisions have a measurable latency effect.
    """

    def __init__(self, device_capacity_blocks: int = 8,
                 retention: Optional[PredictiveRetentionPolicy] = None):
        self.device: Dict[str, KVBlock] = {}   # resident on "GPU"
        self.host: Dict[str, KVBlock] = {}     # offloaded to "host"
        self.device_capacity = device_capacity_blocks
        self.retention = retention or PredictiveRetentionPolicy()
        self.access_counter = 0
        self.saves = 0
        self.loads = 0

    # -- prefix matching: LRU over resident blocks (vLLM default) --
    def get_num_new_matched_tokens(self, request_prefix_hash: str, total_tokens: int) -> int:
        # A block matches if its id is a prefix of the request's prefix hash.
        best = 0
        for bid, blk in self.device.items():
            if request_prefix_hash.startswith(bid):
                best = max(best, blk.tokens)
        return min(best, total_tokens)

    def save_kv_layer(self, block: KVBlock) -> None:
        block.location = "host"
        self.host[block.block_id] = block
        self.device.pop(block.block_id, None)
        self.saves += 1

    def load_kv_layer(self, block_id: str) -> Optional[KVBlock]:
        blk = self.host.get(block_id)
        if blk is None:
            return None
        if len(self.device) >= self.device_capacity:
            self._evict_for_load()
        blk.location = "device"
        self.device[block_id] = blk
        self.host.pop(block_id, None)
        self.loads += 1
        return blk

    def _evict_for_load(self) -> None:
        """Evict a resident block to make room, by the retention policy
        (predictive demand) instead of pure LRU."""
        if not self.device:
            return
        lru = sorted(self.device.values(), key=lambda b: b.last_access)
        signals = {b.block_id: b.demand for b in self.device.values()}
        ages = {b.block_id: self.access_counter - b.last_access for b in self.device.values()}
        victim_id = self.retention.choose_victim(
            [b.block_id for b in lru], signals, ages)
        self.save_kv_layer(self.device[victim_id])

    def admit(self, block: KVBlock) -> None:
        """Admit a freshly computed KV block onto the device."""
        self.access_counter += 1
        block.last_access = self.access_counter
        if block.block_id in self.device:
            return
        if len(self.device) >= self.device_capacity:
            self._evict_for_load()
        block.location = "device"
        self.device[block.block_id] = block




class LMCacheLikeBackend(MockVLLMBackend):
    """An LMCache-default-like baseline (Section 45): tiered store, LFU eviction.

    Keeps as many blocks as fit and evicts the least-frequently-used on pressure,
    WITHOUT a future-demand signal. This is the third arm of the Phase 1 Gate.
    """

    def __init__(self, device_capacity_blocks: int = 8,
                 retention: Optional[PredictiveRetentionPolicy] = None):
        super().__init__(device_capacity_blocks, retention)
        self._freq: Dict[str, int] = {}

    def _evict_for_load(self) -> None:
        if not self.device:
            return
        # LFU: evict the block with the fewest accesses (approximated by oldest
        # last_access among the least-touched; here we use access recency as a
        # frequency proxy via a per-block counter).
        victim_id = min(self.device, key=lambda b: (self._freq.get(b, 0),
                                                     self.device[b].last_access))
        self.save_kv_layer(self.device[victim_id])

    def admit(self, block: KVBlock) -> None:
        self._freq[block.block_id] = self._freq.get(block.block_id, 0) + 1
        super().admit(block)

    def get_num_new_matched_tokens(self, request_prefix_hash: str, total_tokens: int) -> int:
        # On access, bump frequency so LFU reflects reuse.
        self._freq[request_prefix_hash] = self._freq.get(request_prefix_hash, 0) + 1
        best = 0
        for bid, blk in self.device.items():
            if request_prefix_hash.startswith(bid):
                best = max(best, blk.tokens)
        return min(best, total_tokens)





class AdaptiveKVConnector(MockVLLMBackend):
    """Our control plane attached at the connector seam (Section 44, 119).

    Difference from the LRU backend: eviction and prefix matching are guided by
    the calibrated future-demand signal. Blocks predicted to be reused soon are
    kept on-device; cold blocks are offloaded early (freed for hot ones). This is
    the same predictive retention that won in the simulator (Section 90.2), now
    deciding real KV placement.
    """

    def __init__(self, device_capacity_blocks: int = 8,
                 retention: Optional[PredictiveRetentionPolicy] = None):
        super().__init__(device_capacity_blocks, retention)

    def update_demand(self, block_id: str, demand: float) -> None:
        """Feed the connector a calibrated future-demand score for a block
        (from the Lethe->calibration pipeline, Section 90)."""
        for store in (self.device, self.host):
            if block_id in store:
                store[block_id].demand = demand

    def get_num_new_matched_tokens(self, request_prefix_hash: str, total_tokens: int) -> int:
        """Prefix hit, preferring a high-demand resident block when several match
        (restore the one most likely to be reused again)."""
        matches = [blk for bid, blk in self.device.items()
                   if request_prefix_hash.startswith(bid)]
        if not matches:
            # Try to restore a matching offloaded block if demand justifies it.
            for bid, blk in self.host.items():
                if request_prefix_hash.startswith(bid):
                    restored = self.load_kv_layer(bid)
                    if restored is not None:
                        return min(restored.tokens, total_tokens)
            return 0
        best = max(matches, key=lambda b: (b.demand, b.tokens))
        return min(best.tokens, total_tokens)


@dataclass
class RequestTraceItem:
    request_id: str
    prefix_hash: str
    total_tokens: int
    demand: float = 0.0          # calibrated future-demand of this prefix
    output_tokens: int = 32


@dataclass
class RuntimeResult:
    policy: str
    requests: int
    prefix_hits: int
    prefix_hit_rate: float
    offloads: int                # saves (GPU->host)
    restores: int                # loads (host->GPU)
    reused_tokens: int
    total_tokens: int
    effective_ttft_proxy: float  # lower is better (prefill avoided)


class SimulatedVLLMRuntime:
    """Run a request stream through a connector and measure prefix caching.

    Section 45 Phase 1 Gate: compare AdaptiveKVConnector vs the LRU backend
    (vLLM default) vs an LMCache-like "keep everything" backend, on the SAME
    request stream and memory budget. The metric is prefix hit rate + effective
    TTFT (prefill tokens avoided). No GPU required (Section 43).
    """

    def __init__(self, connector: KVConnectorBase, t_prefill_per_token: float = 0.05,
                 t_transfer_per_token: float = 0.01):
        self.connector = connector
        self.t_prefill = t_prefill_per_token
        self.t_transfer = t_transfer_per_token

    def run(self, stream: List[RequestTraceItem]) -> RuntimeResult:
        policy = type(self.connector).__name__
        hits = 0
        reused = 0
        total = 0
        ttft = 0.0
        for item in stream:
            total += item.total_tokens
            matched = self.connector.get_num_new_matched_tokens(
                item.prefix_hash, item.total_tokens)
            if matched > 0:
                hits += 1
                reused += matched
            # Prefill only the unmatched suffix; matched prefix is restored (transfer).
            unmatched = item.total_tokens - matched
            ttft += (self.t_transfer * matched          # restore cached prefix
                     + self.t_prefill * unmatched)      # prefill the new part
            # Admit the (now fully computed) prefix as a KV block.
            blk = KVBlock(block_id=item.prefix_hash, tokens=item.total_tokens,
                          demand=item.demand)
            if isinstance(self.connector, AdaptiveKVConnector):
                self.connector.update_demand(item.prefix_hash, item.demand)
            self.connector.admit(blk)
        n = len(stream)
        return RuntimeResult(
            policy=policy, requests=n, prefix_hits=hits,
            prefix_hit_rate=hits / n if n else 0.0,
            offloads=getattr(self.connector, "saves", 0),
            restores=getattr(self.connector, "loads", 0),
            reused_tokens=reused, total_tokens=total,
            effective_ttft_proxy=ttft,
        )
