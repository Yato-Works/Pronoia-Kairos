"""Phase 1 — vLLM KV connector prototype (Plan.md Section 44, 45, 115, 119).

The connector is the seam that lets our Adaptive Control Plane run OUTSIDE vLLM
(Section 119), deciding which KV prefixes to keep / restore. These tests fix:

    - the connector seam (prefix matching, save/load, admit) works on a
      GPU-free MockVLLMBackend (Section 43).
    - the AdaptiveKVConnector, guided by a future-demand signal, keeps hot
      prefixes resident and restores them -> higher prefix hit rate than the
      LRU / LMCache-like backends on a shared-prefix workload.
    - the Section 45 Phase 1 Gate: Adaptive improves the Pareto frontier
      (best prefix hit rate AND best effective TTFT).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runtime.vllm_connector import (
    KVConnectorBase, MockVLLMBackend, AdaptiveKVConnector, LMCacheLikeBackend,
    SimulatedVLLMRuntime, RequestTraceItem, KVBlock, RuntimeResult,
)
from compare.vllm_connector_experiment import (
    make_prefix_reuse_stream, run_phase1_gate, format_gate_report,
)


# --- Connector seam (Section 44) ---------------------------------------------

def test_backends_implement_the_connector_seam():
    for cls in (MockVLLMBackend, AdaptiveKVConnector, LMCacheLikeBackend):
        assert issubclass(cls, KVConnectorBase)
        b = cls(device_capacity_blocks=2)
        assert hasattr(b, "get_num_new_matched_tokens")
        assert hasattr(b, "save_kv_layer") and hasattr(b, "load_kv_layer")


def test_prefix_matching_returns_cached_tokens():
    b = MockVLLMBackend(device_capacity_blocks=4)
    b.admit(KVBlock(block_id="pfx-a", tokens=256))
    # request whose prefix hash starts with the cached block id -> hit.
    assert b.get_num_new_matched_tokens("pfx-a/turn2", 512) == 256
    # a different prefix -> no match.
    assert b.get_num_new_matched_tokens("pfx-b", 512) == 0


def test_save_and_load_roundtrip():
    b = MockVLLMBackend(device_capacity_blocks=1)
    b.admit(KVBlock(block_id="x", tokens=100))
    b.admit(KVBlock(block_id="y", tokens=100))  # evicts x to host
    assert "y" in b.device and "x" in b.host
    restored = b.load_kv_layer("x")
    assert restored is not None and restored.location == "device"
    assert b.loads == 1


def test_device_capacity_is_respected():
    b = MockVLLMBackend(device_capacity_blocks=2)
    for i in range(5):
        b.admit(KVBlock(block_id=f"b{i}", tokens=64))
    assert len(b.device) <= 2
    assert b.saves >= 3  # the overflow was offloaded


# --- Adaptive advantage (Section 45) -----------------------------------------



# --- Section 45 Phase 1 Gate -------------------------------------------------

def test_phase1_gate_three_way_comparison():
    cmp = run_phase1_gate(device_capacity_blocks=4, n_sessions=6,
                          turns_per_session=10, n_distinct_prefixes=8)
    assert len(cmp.results) == 3
    policies = {r.policy for r in cmp.results}
    assert policies == {"AdaptiveKVConnector", "MockVLLMBackend", "LMCacheLikeBackend"}


def test_adaptive_improves_pareto_frontier():
    """Section 45: Adaptive has the best prefix hit rate AND best effective TTFT."""
    cmp = run_phase1_gate(device_capacity_blocks=4, n_sessions=6,
                          turns_per_session=10, n_distinct_prefixes=8)
    assert cmp.adaptive_wins()
    assert cmp.best_hit_rate().policy == "AdaptiveKVConnector"
    assert cmp.best_ttft().policy == "AdaptiveKVConnector"


def test_adaptive_hit_rate_positive_others_zero_on_growing_prefix():
    """On a growing-prefix workload, only the demand-guided Adaptive connector
    achieves prefix reuse (partial match + restore); exact-match baselines do not."""
    cmp = run_phase1_gate(device_capacity_blocks=4)
    adp = next(r for r in cmp.results if r.policy == "AdaptiveKVConnector")
    assert adp.prefix_hit_rate > 0
    assert adp.effective_ttft_proxy < min(
        r.effective_ttft_proxy for r in cmp.results if r.policy != "AdaptiveKVConnector")


def test_runtime_result_metrics_are_consistent():
    cmp = run_phase1_gate(device_capacity_blocks=4)
    for r in cmp.results:
        assert isinstance(r, RuntimeResult)
        assert 0.0 <= r.prefix_hit_rate <= 1.0
        assert r.reused_tokens <= r.total_tokens
        assert r.prefix_hits <= r.requests


def test_gate_report_renders_all_policies():
    cmp = run_phase1_gate(device_capacity_blocks=4)
    out = format_gate_report(cmp)
    for p in ("AdaptiveKVConnector", "MockVLLMBackend", "LMCacheLikeBackend"):
        assert p in out
    assert "Pareto frontier" in out

def test_adaptive_beats_lru_on_shared_prefix_workload():
    stream = make_prefix_reuse_stream(n_sessions=6, turns_per_session=10,
                                      n_distinct_prefixes=8, seed=7)
    adp = SimulatedVLLMRuntime(AdaptiveKVConnector(4)).run(stream)
    lru = SimulatedVLLMRuntime(MockVLLMBackend(4)).run(stream)
    assert adp.prefix_hit_rate > lru.prefix_hit_rate
    assert adp.effective_ttft_proxy < lru.effective_ttft_proxy


def test_adaptive_restores_from_host_guided_by_demand():
    """The Adaptive connector pulls a hot prefix back from host (restore), which
    the LRU backend never does - that is the demand-guided difference."""
    stream = make_prefix_reuse_stream(n_sessions=6, turns_per_session=10, seed=7)
    adp = SimulatedVLLMRuntime(AdaptiveKVConnector(4)).run(stream)
    lru = SimulatedVLLMRuntime(MockVLLMBackend(4)).run(stream)
    assert adp.restores > lru.restores
    assert adp.reused_tokens > lru.reused_tokens
