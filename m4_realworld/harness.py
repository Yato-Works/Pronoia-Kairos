"""M4 harness (PREREG_M4_REALWORLD §4/§8-8).

Replays AdaptiveRuntime._serve_one's event order over a real (or fake)
executor. Every judgement call goes straight to the AdaptiveRuntime instance
(mentor verdict F: no decision logic is duplicated here); the harness adds
only (a) an execution adapter (real prefill/decode instead of engine.execute),
(b) wall-clock instrumentation (T_control / T_gpu / E2E, §6), and (c) the
real KV store kept in 1:1 sync with the mirror's evictions (§8-4).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from m4_realworld.gpu_store import SegmentTrieStore
from m4_realworld.workloads import segment_token_ids
from runtime.adaptive_runtime import (AdaptiveRuntime, RuntimeConfig,
                                      WorkloadRequest)


class RuntimeAdapter:
    """Timed proxy: measures T_control around AdaptiveRuntime calls (§6)."""

    def __init__(self, config: RuntimeConfig):
        self.rt = AdaptiveRuntime(config)
        self.control_ns = 0

    def call(self, fn, *args):
        t0 = time.perf_counter_ns()
        out = fn(*args)
        self.control_ns += time.perf_counter_ns() - t0
        return out


@dataclass
class RequestTrace:
    request_id: str
    arrival: float
    virtual_start: float
    queue_depth_at_decision: int
    selected_chunks: int
    quality_loss: float
    mirror_prefix_hit: bool
    mirror_chunk_hits: int
    reused_tokens: int
    new_tokens: int
    ttft_s: float
    e2e_s: float
    prefill_gpu_ms: float
    decode_gpu_ms: float
    control_ns: int
    output_tokens: int = 0
    evicted_keys: List[str] = field(default_factory=list)
    stored_tokens_after: int = 0
    slo_met: bool = False

    def decision_columns(self) -> Dict:
        """Determinism-checked columns for MC5 (timings excluded)."""
        return {
            "request_id": self.request_id,
            "selected_chunks": self.selected_chunks,
            "mirror_prefix_hit": self.mirror_prefix_hit,
            "mirror_chunk_hits": self.mirror_chunk_hits,
            "reused_tokens": self.reused_tokens,
            "new_tokens": self.new_tokens,
            "evicted_keys": list(self.evicted_keys),
            "quality_loss": round(self.quality_loss, 12),
        }


def run_stream(stream, spec, config, executor, *, collect_logits: bool = False):
    """Serve the reduced stream with virtual arrival pacing (§8-1/§8-2)."""
    ad = RuntimeAdapter(config)
    rt = ad.rt
    store = SegmentTrieStore(spec.kv_capacity_tokens)
    pending = sorted(stream, key=lambda r: (r.arrival_time, r.request_id))
    n = len(pending)
    traces: List[RequestTrace] = []
    if n == 0:
        return {"traces": traces, "e2e_wall_s": 0.0, "t_control_s": 0.0,
                "t_gpu_s": 0.0, "max_queue_depth": 0, "store": store}
    queue: List[WorkloadRequest] = []
    idx = 0
    vtime = pending[0].arrival_time
    t_run0: Optional[float] = None
    max_queue = 1
    while idx < n or queue:
        while idx < n and pending[idx].arrival_time <= vtime:
            queue.append(pending[idx])
            idx += 1
        if not queue:
            vtime = pending[idx].arrival_time
            continue
        req = ad.call(rt._pick_next, queue)
        queue.remove(req)
        if t_run0 is None:
            t_run0 = time.perf_counter()
        trace = _serve_one_real(req, queue, ad, store, executor, vtime)
        traces.append(trace)
        vtime = trace.virtual_start + trace.e2e_s
        max_queue = max(max_queue, len(queue) + 1)
    e2e_wall = time.perf_counter() - t_run0
    return {
        "traces": traces,
        "e2e_wall_s": e2e_wall,
        "t_control_s": ad.control_ns / 1e9,
        "t_gpu_s": sum(t.prefill_gpu_ms + t.decode_gpu_ms for t in traces) / 1e3,
        "max_queue_depth": max_queue,
        "store": store,
        "kv_final": sorted(rt.kv),
        "mirror_evictions": int(rt.evictions),
        "mirror_prefix_hits": sum(1 for t in traces if t.mirror_prefix_hit),
    }


def _serve_one_real(req, queue, ad: RuntimeAdapter, store: SegmentTrieStore,
                    executor, vtime: float) -> RequestTrace:
    """One request: mirror decisions + real execution (§8-8 boundary).

    Event order mirrors AdaptiveRuntime._serve_one: head_context -> residency
    lookups -> engine (real GPU here) -> touch -> enforce_capacity -> record.
    """
    rt = ad.rt
    t0 = time.perf_counter()
    start = max(vtime, req.arrival_time)
    control0 = ad.control_ns
    # 1. context decision (A/D configs use head truncation §47; planning
    #    configs go through the optimizer — branch reads config, the judgement
    #    itself is always a direct AdaptiveRuntime call, §8-8)
    if rt.config.context_planning:
        selected_ids, qloss, _ir = ad.call(rt._plan_context, req, start)
    else:
        selected_ids, qloss, _ir = ad.call(rt._head_context, req)
    by_id = {c.chunk_id: c for c in req.chunks}
    # 2. mirror residency at lookup time (ex-ante, same as _serve_one)
    prefix_resident = ad.call(rt._resident, req.prefix_key, start)
    chunk_resident = {cid: ad.call(rt._resident, cid, start)
                      for cid in selected_ids}
    # 3. exact token path over segments: prefix then selected chunks
    segs: List[Tuple[str, int]] = [(req.prefix_key, req.prefix_tokens)]
    segs += [(cid, by_id[cid].tokens) for cid in selected_ids]
    tokens: List[int] = []
    boundaries: List[Tuple[int, str]] = []
    for key, size in segs:
        tokens.extend(segment_token_ids(key, size, executor.vocab_size))
        boundaries.append((len(tokens), key))
    # 4. deepest resident exact path (MC0-exact reuse, §8-4)
    hit_end, _node = store.walk(tokens, boundaries)
    cached = store.cache_up_to(tokens, hit_end)
    # 5. real prefill of the suffix only
    cache_handle, layers_full, prefill_gpu_ms, first_logits = executor.prefill(
        tokens, hit_end, cached)
    ttft_s = time.perf_counter() - t0
    # 6. real greedy decode
    out_ids, decode_gpu_ms, _logits = executor.decode(
        cache_handle, req.output_tokens, first_logits)
    e2e_s = time.perf_counter() - t0
    # 7. materialize new segments, gated by pre-decision mirror residency
    gates: Dict[str, bool] = {req.prefix_key: not prefix_resident}
    for cid in selected_ids:
        gates[cid] = not chunk_resident[cid]
    store.materialize(tokens, layers_full, boundaries, gates, hit_end)
    # 8. mirror updates in _serve_one order: touch -> enforce -> record
    ad.call(rt._touch, req.prefix_key, req.prefix_tokens)
    rt.key_tokens[req.prefix_key] = req.prefix_tokens
    for cid in selected_ids:
        ad.call(rt._touch, cid, by_id[cid].tokens)
        rt.key_tokens[cid] = by_id[cid].tokens
    before = set(rt.kv)
    ad.call(rt._enforce_capacity, queue)
    evicted = sorted(before - set(rt.kv))
    store.evict_keys(evicted)
    ad.call(rt._record_access, [req.prefix_key] + list(selected_ids))
    slo_met = (ttft_s <= req.slo_ttft
               and qloss <= rt.config.quality_epsilon)
    return RequestTrace(
        request_id=req.request_id, arrival=req.arrival_time, virtual_start=start,
        queue_depth_at_decision=len(queue) + 1,
        selected_chunks=len(selected_ids), quality_loss=qloss,
        mirror_prefix_hit=prefix_resident,
        mirror_chunk_hits=sum(1 for v in chunk_resident.values() if v),
        reused_tokens=hit_end, new_tokens=len(tokens) - hit_end,
        ttft_s=ttft_s, e2e_s=e2e_s,
        prefill_gpu_ms=prefill_gpu_ms, decode_gpu_ms=decode_gpu_ms,
        control_ns=ad.control_ns - control0, output_tokens=len(out_ids),
        evicted_keys=evicted, stored_tokens_after=store.usage, slo_met=slo_met)
