"""CPU tests for M4 real-world harness (FakeExecutor); MC0 needs GPU.

python -m pytest tests/test_m4_realworld.py -q -k "not mc0_gpu"
"""
import pytest
import torch

from compare.full_runtime_experiment import ladder
from m4_realworld.gpu_store import SegmentTrieStore
from m4_realworld.harness import run_stream
from m4_realworld.mc import (FakeExecutor, compare_harness_to_reference,
                             mc3_check, mc4_check, mc5_columns_equal,
                             reference_replay)
from m4_realworld.workloads import (build_reduced_w6, canonical_chunk_tokens,
                                    segment_token_ids, stream_hash)


def test_workloads_deterministic_and_bounded():
    stream, spec, stats = build_reduced_w6(1)
    stream2, _s2, _stats2 = build_reduced_w6(1)
    assert len(stream) == 40
    assert stream_hash(stream) == stream_hash(stream2)
    assert stream_hash(stream) != stream_hash(build_reduced_w6(2)[0])
    assert stats["input_tokens_max"] <= 2048
    assert stats["chunks_per_request_max"] == 8
    assert spec.kv_capacity_tokens >= 1024
    first = stream[0]
    assert 128 <= canonical_chunk_tokens(first.chunks[0].chunk_id) <= 168
    ids = segment_token_ids("seg-a", 128, 151643)
    assert ids == segment_token_ids("seg-a", 128, 151643) and len(ids) == 128
    assert all(0 <= t < 151643 for t in ids)


def test_gpu_store_exact_path_and_eviction():
    store = SegmentTrieStore(1024)
    assert store.check_accounting()
    toks_a = list(range(16))
    toks_b = list(range(16, 32))
    tokens = toks_a + toks_b
    boundaries = [(16, "a"), (32, "b")]
    layers_a = [(torch.zeros(1, 2, 16, 4), torch.zeros(1, 2, 16, 4))]
    layers_full = [(torch.zeros(1, 2, 32, 4), torch.zeros(1, 2, 32, 4))]
    gates = {"a": True, "b": True}
    store.materialize(tokens, layers_a, [(16, "a")], gates, 0)
    assert store.walk(tokens, boundaries)[0] == 16
    store.materialize(tokens, layers_full, boundaries, gates, 16)
    hit, node = store.walk(tokens, boundaries)
    assert hit == 32 and node.owner_key == "b"
    cached = store.cache_up_to(tokens, hit)
    assert cached is not None and cached[0][0].shape[2] == 32
    # §8-4: evicting the ancestor ("a") drops the unreachable descendant
    # ("b") too — cache_up_to must then return None, not stale layers.
    freed = store.evict_keys(["a"])
    assert freed == 32 and store.usage == 0
    assert store.walk(tokens, boundaries)[0] == 0
    assert store.cache_up_to(tokens, 32) is None
    assert store.check_accounting()


def test_harness_fake_executor_mc2_mc4_mc5():
    from dataclasses import replace as _rep
    stream, spec, _stats = build_reduced_w6(1)
    used = stream[:8]
    # §7 ladder rows: D_semantic (True condition), A_naive (LRU baseline)
    base = next(c for c in ladder(spec) if c.name == "D_semantic")
    cfg = _rep(base, name="T_sem")
    out = run_stream(used, spec, cfg, FakeExecutor())
    traces = out["traces"]
    assert len(traces) == len(used)
    assert all(t.output_tokens == used[0].output_tokens for t in traces)
    assert all(t.new_tokens >= 1 for t in traces)
    # MC2: harness == reference replay of AdaptiveRuntime.serve (no dup logic)
    service = {t.request_id: t.e2e_s for t in traces}
    ref = reference_replay(used, cfg, service)
    cmp2 = compare_harness_to_reference(out, traces, ref)
    assert cmp2["match"], cmp2["diffs"]
    # MC4: fake prefill linear in new_tokens -> monotone per reuse bucket
    mc4 = mc4_check(traces)
    assert mc4["status"] in ("pass", "inconclusive_no_pairs"), mc4
    # MC5: rerun same condition -> identical decision columns
    out2 = run_stream(used, spec, cfg, FakeExecutor())
    mc5 = mc5_columns_equal(traces, out2["traces"])
    assert mc5["match"], mc5
    # paired arm config: LRU baseline from the same ladder point
    cfg_lru = _rep(next(c for c in ladder(spec) if c.name == "A_naive"),
                   name="T_lru")
    out3 = run_stream(used, spec, cfg_lru, FakeExecutor())
    assert len(out3["traces"]) == len(used)


def test_mc3_equal_signal_victim_lru_equivalence():
    res = mc3_check()
    assert res["status"] == "pass", res


def test_mc0_gpu_optional():
    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    from m4_realworld.gpu_executor import load_executor
    from m4_realworld.run_m4 import mc0_check
    executor = load_executor(max_batch=4, max_prefill=256, max_decode=256)
    res = mc0_check(executor, [500, 123, 77, 900, 31])
    assert res["status"] in ("exact", "tolerance_ok"), res
