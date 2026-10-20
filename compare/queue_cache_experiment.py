"""Phase 2 — Queue + Cache Co-Optimization gate (Plan.md Section 46, 47).

Section 46: reproduce PEEK-like (queue-aware) behaviour, then extend it with
Queue-only + Semantic demand prediction. Section 47's decisive question:

    > "Does a condition exist that beats PEEK?"

We run three retention policies through the DiscreteEventSimulator on the SAME
request stream and compare the Section 47 metrics (TTFT p50/p95/p99, E2E p95,
goodput, cache hit rate, fairness):

    lru        - plain LRU baseline.
    peek       - QueueAwareRetentionPolicy: protect keys the pending queue will
                 reference within `lookahead` (queue-ONLY signal).
    queue+sem  - QueueSemanticRetentionPolicy: queue protection TEMPERED by a
                 calibrated semantic future-demand signal.

The hypothesized PEEK-beating condition (Section 17): a workload whose reuse is
LONG-range (semantic horizon >> queue lookahead). PEEK only sees the near queue,
so it evicts a hot long-range prefix as if cold; queue+semantic protects it via
its demand signal. We use long_range_reuse (anchors revisit after `gap` accesses,
uncorrelated with recency) with a SHORT queue lookahead to create exactly that
regime.

Fairness (Section 26) is measured as the p95/median ratio of per-request E2E
latency (lower = fairer); a policy must not win goodput by starving some requests.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

from scheduler.queue_cache_policy import (
    QueueAwareRetentionPolicy, QueueSemanticRetentionPolicy,
)
from simulator.discrete import DiscreteEventSimulator, TraceRequest
from simulator.latency import LatencyModel
from simulator import workloads as w


@dataclass
class GateMetrics:
    policy: str
    hits: int
    hit_rate: float
    ttft_p50: float
    ttft_p95: float
    ttft_p99: float
    e2e_p95: float
    goodput: float
    fairness: float            # p95/median E2E (lower = fairer)


def _fairness(records) -> float:
    """p95/median of per-request E2E latency (lower = fairer, Section 26)."""
    e2es = sorted(r.e2e for r in records if r.e2e > 0)
    if not e2es:
        return 1.0

    def pct(xs, p):
        idx = min(len(xs) - 1, max(0, int(round(p / 100.0 * (len(xs) - 1)))))
        return xs[idx]
    median = pct(e2es, 50)
    p95 = pct(e2es, 95)
    return p95 / median if median > 0 else 1.0


def _ttft_p99(records) -> float:
    ttfts = sorted(r.ttft for r in records if r.ttft > 0)
    if not ttfts:
        return 0.0
    idx = min(len(ttfts) - 1, max(0, int(round(0.99 * (len(ttfts) - 1)))))
    return ttfts[idx]


def run_policy(sequence: List[str], capacity: int, policy: str,
               lookahead: int = 4, slo_ttft: float = 60.0,
               signals: Optional[List[float]] = None,
               tokens: int = 512) -> GateMetrics:
    """Run one retention policy on a key sequence; return Section 47 metrics."""
    trace = []
    for i, key in enumerate(sequence):
        p_use = signals[i] if signals is not None else 0.0
        trace.append(TraceRequest(
            request_id=f"r{i}", key=key, arrival_time=float(i) * 3.0,
            input_tokens=tokens, output_tokens=32,
            future_use_probability=p_use, semantic_confidence=0.9 if p_use > 0 else 0.0))
    if policy == "lru":
        retention, q_look = None, 0
    elif policy == "peek":
        retention, q_look = QueueAwareRetentionPolicy(lookahead=lookahead), lookahead
    elif policy == "queue+sem":
        retention, q_look = QueueSemanticRetentionPolicy(lookahead=lookahead), lookahead
    else:
        raise ValueError(f"unknown policy: {policy}")
    sim = DiscreteEventSimulator(
        capacity=capacity, latency=LatencyModel(), retention_policy=retention,
        slo_ttft=slo_ttft, queue_lookahead=q_look)
    m = sim.run(trace)
    n = m.hits + m.misses
    return GateMetrics(
        policy=policy, hits=m.hits, hit_rate=m.hit_rate(),
        ttft_p50=m.ttft_p50, ttft_p95=m.ttft_p95, ttft_p99=_ttft_p99(m.records),
        e2e_p95=m.e2e_p95, goodput=m.goodput_ratio(), fairness=_fairness(m.records))


def _horizon_signals(sequence: List[str], horizon: int) -> List[float]:
    """Calibrated semantic demand for predictive retention: 1.0 if the key at
    position i recurs within `horizon` (Section 50), else 0.0. This is the signal
    the semantic-aware policy protects on; horizon > queue lookahead is the
    PEEK-beating condition (Section 17)."""
    n = len(sequence)
    out = [0.0] * n
    for i in range(n):
        out[i] = 1.0 if sequence[i] in sequence[i + 1: i + 1 + horizon] else 0.0
    return out


@dataclass
class Phase2Comparison:
    metrics: List[GateMetrics]
    regime: str

    def get(self, policy: str) -> GateMetrics:
        return next(m for m in self.metrics if m.policy == policy)

    def semantically_beats_peek(self) -> bool:
        """Section 47 decisive question: does queue+semantic beat PEEK?
        Beat = strictly more cache hits AND not worse TTFT p95 (both matter)."""
        sem = self.get("queue+sem")
        peek = self.get("peek")
        return sem.hits > peek.hits and sem.ttft_p95 <= peek.ttft_p95 + 1e-9


def run_phase2_comparison(
    capacity: int = 4,
    gap: int = 8,
    lookahead: int = 4,
    n_anchors: int = 6,
    repeats: int = 6,
    slo_ttft: float = 60.0,
) -> Phase2Comparison:
    """Run the Section 47 three-way comparison on a long-range-reuse workload.

    gap >> lookahead creates the regime where the queue is myopic: it cannot see
    an anchor returning after `gap` accesses, but the semantic horizon (gap+2)
    can. That is the hypothesized PEEK-beating condition.
    """
    sequence = w.long_range_reuse(n_anchors, gap, repeats)
    horizon = gap + 2  # semantic horizon exceeds the revisit distance
    signals = _horizon_signals(sequence, horizon)
    metrics = [
        run_policy(sequence, capacity, "lru", lookahead, slo_ttft, signals),
        run_policy(sequence, capacity, "peek", lookahead, slo_ttft, signals),
        run_policy(sequence, capacity, "queue+sem", lookahead, slo_ttft, signals),
    ]
    regime = f"gap={gap} >> lookahead={lookahead}"
    return Phase2Comparison(metrics=metrics, regime=regime)


def format_phase2_report(cmp: Phase2Comparison) -> str:
    lines = [
        f"Section 47 Phase 2 Gate: queue x cache co-optimization ({cmp.regime})",
        f"{'policy':>10} {'hits':>6} {'hitRate':>8} {'t50':>7} {'t95':>7} {'t99':>7} "
        f"{'e2e95':>8} {'goodput':>9} {'fair':>6}",
        "-" * 78,
    ]
    for m in cmp.metrics:
        lines.append(
            f"{m.policy:>10} {m.hits:>6} {m.hit_rate:>8.3f} {m.ttft_p50:>7.1f} "
            f"{m.ttft_p95:>7.1f} {m.ttft_p99:>7.1f} {m.e2e_p95:>8.1f} "
            f"{m.goodput:>9.3f} {m.fairness:>6.2f}"
        )
    lines.append("")
    sem, peek = cmp.get("queue+sem"), cmp.get("peek")
    lines.append(f"queue+semantic hits {sem.hits} vs PEEK hits {peek.hits} "
                 f"(+{sem.hits - peek.hits}), TTFT p95 {sem.ttft_p95:.1f} vs {peek.ttft_p95:.1f}")
    lines.append(f"Section 47 verdict: a condition that beats PEEK EXISTS = "
                 f"{cmp.semantically_beats_peek()}")
    return "\n".join(lines)


def save_phase2_artifacts(cmp: Phase2Comparison, out_dir: str = "artifacts/phase2") -> str:
    import json
    import os
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "queue_cache_gate.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "regime": cmp.regime,
            "metrics": [m.__dict__ for m in cmp.metrics],
            "semantically_beats_peek": cmp.semantically_beats_peek(),
        }, f, indent=2, ensure_ascii=False)
    return path


def main():
    print("=== Phase 2 Queue + Cache Co-Optimization (Section 46, 47) ===\n")
    print("Workload: long_range_reuse - anchors revisit after `gap` accesses, so the")
    print("queue lookahead is myopic and the semantic horizon (gap+2) sees the return.")
    print("This is the hypothesized regime where semantic demand beats queue-only (PEEK).\n")
    cmp = run_phase2_comparison(capacity=4, gap=8, lookahead=4)
    print(format_phase2_report(cmp))
    if cmp.semantically_beats_peek():
        print("\nPhase 2 Gate: a condition that beats PEEK EXISTS - adding semantic")
        print("demand to the queue-aware policy raises cache hits without hurting")
        print("TTFT, exactly where the queue alone is myopic (Section 17).")
    else:
        print("\nPhase 2 Gate: queue+semantic did NOT beat PEEK in this regime.")
    path = save_phase2_artifacts(cmp)
    print(f"\nArtifacts saved: {path}")


if __name__ == "__main__":
    main()
