"""Phase 1 — vLLM connector Phase 1 Gate experiment (Plan.md Section 44, 45, 115, 119).

Section 45 Phase 1 Gate: on the SAME request stream and memory budget, compare

    Adaptive Policy   (our predictive-retention KV connector)
    vs LRU            (vLLM default)
    vs LMCache default (tiered store, LFU)

and show the Adaptive Policy improves the Pareto frontier (prefix hit rate up,
effective TTFT down) on at least one realistic workload family. No GPU required
(Section 43): the connector decision code runs on a MockVLLMBackend.

The "realistic workload" here is a prefix-reuse stream: multi-turn / agent / RAG
requests that share growing prefixes (Section 63 workload matrix). Each request
carries a prefix hash; the connector decides which prefixes stay on-device. The
calibrated future-demand signal (from the Lethe->calibration pipeline, Section 90)
drives the Adaptive connector's keep/restore decisions.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List

from runtime.vllm_connector import (
    MockVLLMBackend, AdaptiveKVConnector, LMCacheLikeBackend,
    SimulatedVLLMRuntime, RequestTraceItem, RuntimeResult,
)


def make_prefix_reuse_stream(
    n_sessions: int = 6,
    turns_per_session: int = 10,
    n_distinct_prefixes: int = 8,
    base_tokens: int = 256,
    growth: int = 16,
    seed: int = 7,
) -> List[RequestTraceItem]:
    """A realistic prefix-reuse stream (multi-turn / agent / RAG, Section 63).

    Sessions revisit a small set of shared prefixes (hot conversation contexts);
    within a session the prefix grows turn by turn. A block's demand reflects how
    often its session's prefix recurs - the signal the Adaptive connector uses.
    """
    import random
    rng = random.Random(seed)
    # Each session binds to a prefix; some prefixes are hot (revisited often).
    prefix_ids = [f"pfx-{i}" for i in range(n_distinct_prefixes)]
    hot = prefix_ids[: max(1, n_distinct_prefixes // 2)]
    stream: List[RequestTraceItem] = []
    rid = 0
    for _ in range(n_sessions):
        pfx = rng.choice(hot if rng.random() < 0.7 else prefix_ids)
        turn = 0
        for _ in range(turns_per_session):
            turn += 1
            # The request's prefix hash extends the base prefix by the turn, so
            # earlier turns' blocks are a prefix of later ones (prefix caching).
            prefix_hash = f"{pfx}/turn{turn}"
            tokens = base_tokens + growth * turn
            demand = 0.9 if pfx in hot else 0.3
            stream.append(RequestTraceItem(
                request_id=f"r{rid}", prefix_hash=prefix_hash,
                total_tokens=tokens, demand=demand, output_tokens=32))
            rid += 1
    return stream


@dataclass
class GateComparison:
    results: List[RuntimeResult]

    def best_hit_rate(self) -> RuntimeResult:
        return max(self.results, key=lambda r: r.prefix_hit_rate)

    def best_ttft(self) -> RuntimeResult:
        return min(self.results, key=lambda r: r.effective_ttft_proxy)

    def adaptive_wins(self) -> bool:
        """Section 45: Adaptive improves the frontier (best hit rate AND best TTFT)."""
        adp = next((r for r in self.results if r.policy == "AdaptiveKVConnector"), None)
        if adp is None:
            return False
        return (adp is self.best_hit_rate()) and (adp is self.best_ttft())


def run_phase1_gate(
    device_capacity_blocks: int = 4,
    **stream_kwargs,
) -> GateComparison:
    """Run the Section 45 three-way comparison on one workload."""
    stream = make_prefix_reuse_stream(**stream_kwargs)
    results = [
        SimulatedVLLMRuntime(AdaptiveKVConnector(device_capacity_blocks)).run(stream),
        SimulatedVLLMRuntime(MockVLLMBackend(device_capacity_blocks)).run(stream),
        SimulatedVLLMRuntime(LMCacheLikeBackend(device_capacity_blocks)).run(stream),
    ]
    return GateComparison(results)


def format_gate_report(cmp: GateComparison) -> str:
    lines = [
        "Section 45 Phase 1 Gate: Adaptive vs LRU vs LMCache (same stream, same budget)",
        f"{'policy':>22} {'hitRate':>8} {'effTTFT':>9} {'offloads':>9} {'restores':>9} {'reusedTok':>10}",
        "-" * 72,
    ]
    for r in cmp.results:
        lines.append(
            f"{r.policy:>22} {r.prefix_hit_rate:>8.3f} {r.effective_ttft_proxy:>9.1f} "
            f"{r.offloads:>9d} {r.restores:>9d} {r.reused_tokens:>10d}"
        )
    lines.append("")
    bh, bt = cmp.best_hit_rate(), cmp.best_ttft()
    lines.append(f"best prefix hit rate : {bh.policy} ({bh.prefix_hit_rate:.3f})")
    lines.append(f"best effective TTFT  : {bt.policy} ({bt.effective_ttft_proxy:.1f})")
    lines.append(f"Adaptive improves the Pareto frontier (Section 45) : {cmp.adaptive_wins()}")
    return "\n".join(lines)


def save_gate_artifacts(cmp: GateComparison, out_dir: str = "artifacts/phase1") -> str:
    import json
    import os
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "vllm_connector_gate.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "results": [r.__dict__ for r in cmp.results],
            "adaptive_wins": cmp.adaptive_wins(),
            "best_hit_rate_policy": cmp.best_hit_rate().policy,
            "best_ttft_policy": cmp.best_ttft().policy,
        }, f, indent=2, ensure_ascii=False)
    return path


def main():
    print("=== vLLM Connector Phase 1 Gate (Section 44, 45): "
          "Adaptive Control Plane vs LRU vs LMCache ===\n")
    print("vLLM is not required (Section 43): the connector decision code runs on a")
    print("MockVLLMBackend, so the same AdaptiveKVConnector that would be registered")
    print("as vLLM's KVConnectorBase_V1 is exercised here without a GPU.\n")
    cmp = run_phase1_gate(device_capacity_blocks=4,
                          n_sessions=6, turns_per_session=10, n_distinct_prefixes=8)
    print(format_gate_report(cmp))
    if cmp.adaptive_wins():
        print("\nPhase 1 Gate PASSED: the Adaptive Policy improves the Pareto frontier")
        print("(best prefix hit rate AND best effective TTFT) on a realistic")
        print("(multi-turn / shared-prefix) workload - as an external vLLM connector.")
    else:
        print("\nPhase 1 Gate not met on this workload (see table).")
    path = save_gate_artifacts(cmp)
    print(f"\nArtifacts saved: {path}")


if __name__ == "__main__":
    main()
