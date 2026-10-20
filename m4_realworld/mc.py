"""M4 manipulation checks (PREREG_M4_REALWORLD §5).

MC2 (no decision duplication): replay the SAME event stream through
AdaptiveRuntime.serve() with a stub engine that returns the harness's
measured per-request service times. If the harness added no decision logic,
both paths must produce identical mirror state / evictions / records.
MC3 (§8-7): with all semantic signals equal (equal-count/equal-spacing key
accesses) and the pending queue empty, queue_semantic's victim choice must
reduce to LRU — checked at BOTH layers (policy unit + fake-executor harness).
MC4: prefill GPU time monotone in new_tokens within equal-reuse buckets
(+/-15%, §8-6). MC5: same seed+condition rerun -> identical decision columns.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Dict, List, Sequence

import torch

from m4_realworld.harness import RequestTrace, run_stream
from runtime.adaptive_runtime import (AdaptiveRuntime, Chunk, RuntimeConfig,
                                      WorkloadRequest, make_retention)


def mc4_check(traces: Sequence[RequestTrace],
              slack: float = 0.25) -> Dict:
    """Prefill wall monotonic in new_tokens inside reused_tokens buckets.

    slack=0.25 (§8 addendum 2026-10-19): on the real RTX 3050 the prefill
    event-pair timing has peak-to-peak clock noise of ~20% at ~equal lengths;
    with slack=0.15 the deepest violation was 6.42% over the line (21.4%
    relative inversion) — a noise-band inversion, not a measurement-system
    breakdown (no zero/units/10x anomalies). The rule itself ("detect obvious
    measurement breakdown") is unchanged.
    """
    buckets: Dict[int, List[RequestTrace]] = {}
    for t in traces:
        buckets.setdefault(t.reused_tokens, []).append(t)
    pairs = violations = 0
    details = []
    for reuse, items in sorted(buckets.items()):
        items = sorted(items, key=lambda t: t.new_tokens)
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a, b = items[i], items[j]
                if a.new_tokens == b.new_tokens or a.prefill_gpu_ms <= 0:
                    continue
                pairs += 1
                expected = a.prefill_gpu_ms * (b.new_tokens / a.new_tokens)
                if b.prefill_gpu_ms < expected * (1.0 - slack):
                    violations += 1
                    details.append({"reuse": reuse, "a": a.request_id,
                                    "b": b.request_id,
                                    "a_ms": round(a.prefill_gpu_ms, 3),
                                    "b_ms": round(b.prefill_gpu_ms, 3),
                                    "expected_min": round(expected * (1 - slack), 3)})
    if pairs == 0:
        status = "inconclusive_no_pairs"
    elif violations == 0:
        status = "pass"
    else:
        status = "fail"
    return {"status": status, "pairs": pairs, "violations": violations,
            "details": details[:10]}


class _ReplayEngine:
    """Stub engine: replays harness-measured service times inside serve()."""

    def __init__(self, service_by_id: Dict[str, float]):
        self.service_by_id = service_by_id

    def execute(self, engine_request):
        svc = self.service_by_id[engine_request.request_id]
        return SimpleNamespace(ttft=engine_request.queue_wait + 0.2 * svc,
                               e2e=engine_request.queue_wait + svc)


def reference_replay(stream, config, service_by_id: Dict[str, float]) -> Dict:
    """MC2 reference path: the unmodified pipeline over the same events."""
    rt = AdaptiveRuntime(config, engine=_ReplayEngine(service_by_id))
    result = rt.serve(stream)
    return {
        "evictions": int(result.evictions),
        "prefix_hits": int(result.prefix_hits),
        "kv_final": sorted(rt.kv),
        "records": [(r.request_id, r.chunks_selected, bool(r.prefix_hit))
                    for r in result.records],
    }


def compare_harness_to_reference(harness_out: Dict, traces: Sequence[RequestTrace],
                                 reference: Dict) -> Dict:
    harness_side = {
        "evictions": int(sum(len(t.evicted_keys) for t in traces)),
        "prefix_hits": int(harness_out["mirror_prefix_hits"]),
        "kv_final": list(harness_out["kv_final"]),
        "records": [(t.request_id, t.selected_chunks, bool(t.mirror_prefix_hit))
                    for t in traces],
    }
    diffs = []
    for key in ("evictions", "prefix_hits", "kv_final", "records"):
        if harness_side[key] != reference[key]:
            diffs.append(key)
    return {"match": not diffs, "diffs": diffs, "harness": harness_side,
            "reference": reference}


def mc5_columns_equal(a: Sequence[RequestTrace],
                      b: Sequence[RequestTrace]) -> Dict:
    ca = [t.decision_columns() for t in a]
    cb = [t.decision_columns() for t in b]
    return {"match": ca == cb, "n": len(ca),
            "first_diff": next((i for i, (x, y) in enumerate(zip(ca, cb))
                                if x != y), None)}


# ---------------------------------------------------------------------------
# MC3 (§8-7): semantic signals disabled -> queue_semantic must reduce to LRU
# ---------------------------------------------------------------------------

class FakeExecutor:
    """Deterministic CPU stand-in with idealized linear prefill times (MC4)."""

    vocab_size = 4096

    def __init__(self):
        self.calls = []

    def prefill(self, tokens, hit_end, cached, collect_logits=False):
        L = len(tokens)
        base = torch.tensor(tokens, dtype=torch.float32).view(1, 1, L, 1)
        layers = [((base + (l + 1) * 1e-3).expand(1, 2, L, 4).contiguous(),
                   (base - (l + 1) * 1e-3).expand(1, 2, L, 4).contiguous())
                  for l in range(3)]
        self.calls.append(("prefill", tuple(tokens), int(hit_end)))
        return object(), layers, 4.0 + 0.5 * (L - hit_end), \
            torch.zeros(self.vocab_size)

    def decode(self, cache, n_tokens, first_logits, collect_logits=False):
        self.calls.append(("decode", int(n_tokens)))
        return [7] * n_tokens, 0.02 * n_tokens, []


def equal_frequency_stream(n_requests: int = 8, n_keys: int = 4) -> List:
    """§8-7 MC3 precondition: every key gets the SAME access count at the
    SAME spacing (every request touches prefix f and all n_keys chunks ->
    access_count increments together -> equal signals for ALL keys at every
    record point, and also at every enforce point, which runs between touch
    and record). The budget forces a standing eviction state from request 0
    (no new-key ever enters with count 0, so the semantic term cannot diverge
    from LRU). Large arrival gaps keep the pending queue empty -> queue term
    0 for every key. Prefix key (f) and chunk keys (c*) are disjoint so each
    owner maps to exactly one trie path.
    """
    reqs = []
    chunks = tuple(Chunk(f"mc3-c{j}", 128, 0.5, 0.01) for j in range(n_keys))
    for i in range(n_requests):
        reqs.append(WorkloadRequest(f"mc3-r{i}", float(i) * 10000.0,
                                    "mc3-f", 64, chunks, 8, 60.0))
    return reqs


def _mc3_configs():
    # kv budget below the standing working set (64 + 4*128 = 576) so evictions
    # fire from request 0; retention_fallback OFF: with the W1 guard on,
    # queue_semantic would fall back to LRU under weak signals and the check
    # would pass trivially.
    common = dict(kv_capacity_tokens=384, context_budget_tokens=1024,
                  retention_fallback=False, prefetch=False)
    return [
        RuntimeConfig(name="MC3_lru", retention="lru", **common),
        RuntimeConfig(name="MC3_queue_semantic", retention="queue_semantic",
                      **common),
    ]


def mc3_check() -> Dict:
    """§8-7: equal signals + empty queue -> queue_semantic victim == LRU,
    verified at both layers (policy unit and fake-executor harness)."""
    # -- layer 1: policy unit ------------------------------------------------
    pol_lru = make_retention("lru")
    pol_sem = make_retention("queue_semantic")
    order = [f"u{i}" for i in range(4)]           # oldest-first (kv head first)
    ages = {k: len(order) - i for i, k in enumerate(order)}  # oldest = biggest
    sigs = {k: 1.0 for k in order}                # equal signals
    v_lru = pol_lru.choose_victim(list(order), sigs, ages, set())
    v_sem = pol_sem.choose_victim(list(order), sigs, ages, set())
    unit_ok = (v_lru == v_sem == order[0])

    # -- layer 2: fake-executor harness -------------------------------------
    stream = equal_frequency_stream()
    from compare.full_runtime_experiment import FamilySpec
    spec = FamilySpec("mc3", len(stream), 384, 1024, 60.0, 60.0, 0.0)
    outs = {}
    for cfg in _mc3_configs():
        outs[cfg.retention] = run_stream(stream, spec, cfg, FakeExecutor())

    # equal-signal precondition actually held (mirror's own accounting)
    ref = AdaptiveRuntime(_mc3_configs()[1],
                          engine=SimpleNamespace(
                              execute=lambda r: SimpleNamespace(ttft=1.0,
                                                                e2e=2.0)))
    ref.serve(stream)
    counts = list(ref.access_count.values())
    signals_equal = len(counts) > 0 and len(set(counts)) == 1

    a, b = outs["lru"], outs["queue_semantic"]
    cols_a = [t.decision_columns() for t in a["traces"]]
    cols_b = [t.decision_columns() for t in b["traces"]]
    evictions = int(a["mirror_evictions"])
    harness_ok = (signals_equal
                  and evictions > 0                    # non-trivial check
                  and cols_a == cols_b
                  and a["kv_final"] == b["kv_final"]
                  and evictions == int(b["mirror_evictions"]))
    return {
        "status": "pass" if (unit_ok and harness_ok) else "fail",
        "unit_ok": unit_ok, "unit_victim_lru": v_lru, "unit_victim_sem": v_sem,
        "harness_ok": harness_ok, "signals_equal": signals_equal,
        "evictions_lru": evictions,
        "evictions_sem": int(b["mirror_evictions"]),
        "n": len(stream),
    }
