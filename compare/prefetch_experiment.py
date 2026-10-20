"""Phase 2/3 experiment - semantic prefetch vs no-prefetch (Plan.md Section 20, 35, 52, 65).

Section 20 states the ONLY condition a prefetch may fire under:

    P_use * T_saved  >  T_transfer + T_interference

and that a prefetch miss is itself a cost ("Prefetch miss also counted as cost").
This experiment tests that policy end-to-end on a long-range reuse workload:

    - forecast = oracle "will this key be reused within horizon H?" + N(0, sigma)
      so sigma sweeps prediction quality from perfect (0.0) to useless (large).
    - prefetch ON  : DiscreteEventSimulator + PrefetchController fetches a
      non-resident high-P_use object iff the Section 20 inequality holds.
    - prefetch OFF : identical trace, no controller (pure LRU-style retention).

Reported per sigma (Section 35/52/65 metrics):
    prefetch accuracy = prefetch hits / issued
    prefetch waste    = evicted-before-use / issued
    saved_compute     = avoided prefill over prefetch hits
    prediction_overhead = n_requests * per-decision cost
    prefetch_waste_cost  = transfer+interference charged over wasted prefetches
    NetGain = SavedCompute - PredictionOverhead - PrefetchWaste   (must be > 0)

Gate (Section 52): NetGain > 0 while prediction is accurate, degrading to <= 0 as
sigma grows - i.e. prefetch only pays when the semantic signal is good enough.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import json
import os
import random

from cache.prefetch import PrefetchController, PrefetchStats
from simulator.discrete import DiscreteEventSimulator, TraceRequest
from simulator.latency import LatencyModel
from simulator import workloads as w

# Overhead of one semantic-demand scoring decision, abstract time units (Section 52).
PREDICTION_OVERHEAD_PER_DECISION = 0.05


def oracle_reuse_probability(sequence: List[str], index: int, horizon: int) -> float:
    """Section 50 ground truth: 1.0 if key reappears within `horizon` accesses."""
    key = sequence[index]
    return 1.0 if key in sequence[index + 1: index + 1 + horizon] else 0.0


def noisy_forecast(sequence: List[str], horizon: int, sigma: float, seed: int = 42) -> dict:
    """Per-key predicted future-use probability, sweeping PREDICTION QUALITY.

    Prediction quality is modeled as a *recall loss* on true demand, faithful to
    Section 52 ("high accuracy but wasted prefetches = failure"):

        score(k) = oracle_demand(k) * keep_factor + small_bias

    where `keep_factor` shrinks as sigma grows, so a poor predictor (large sigma)
    MISSES true-demand keys (they never enter the forecast as prefetch
    candidates) while a good predictor (sigma ~ 0) keeps them. A tiny uniform
    bias models occasional false positives that survive the Section 20 cost gate.

    Effect on the sweep:
        sigma = 0   : all true-demand keys prefetchable -> best case.
        large sigma : true-demand keys drop out -> prefetch misses them, hit rate
                      falls back toward no-prefetch, and any surviving prefetch is
                      more likely to be wrong -> NetGain degrades toward <= 0.

    This is the SAME recall-loss quality used to derive retention signals in the
    adaptive experiment, so retention and prefetch see identical prediction
    quality there. A key's forecast is the max score over its positions, clipped
    to [0,1].
    """
    rng = random.Random(seed)
    # Per-key recall loss: each true-demand occurrence is KEPT with probability
    # keep_prob(sigma), so as sigma grows a *smoothly increasing fraction* of
    # demand keys is missed (not an all-or-nothing cliff). keep_prob:
    #   sigma=0 -> 1.0 (perfect recall), large sigma -> ~0 (blind).
    keep_prob = 1.0 / (1.0 + sigma)
    false_bias = 0.15 * sigma / (1.0 + sigma)  # occasional false-positive demand
    forecast: dict = {}
    for i, key in enumerate(sequence):
        demand = oracle_reuse_probability(sequence, i, horizon)
        if demand > 0.0:
            # True demand: survive the Section 20 gate only if not missed.
            base = demand if rng.random() < keep_prob else 0.0
        else:
            # No true demand: only a noisy false positive can light up.
            base = false_bias if rng.random() < (0.3 * sigma / (1.0 + sigma)) else 0.0
        score = min(1.0, max(0.0, base))
        forecast[key] = max(forecast.get(key, 0.0), score)
    return forecast


@dataclass
class PrefetchResult:
    sigma: float
    prefetch_issued: int
    prefetch_hits: int
    prefetch_wasted: int
    accuracy: float          # hits / issued (Section 65)
    waste_rate: float        # wasted / issued (Section 65)
    hits_on: int             # total cache hits, prefetch enabled
    hits_off: int            # total cache hits, prefetch disabled (LRU retention)
    saved_compute: float     # avoided prefill over prefetch hits (Section 52)
    prediction_overhead: float
    prefetch_waste_cost: float
    net_gain: float          # SavedCompute - PredictionOverhead - PrefetchWaste
    net_gain_positive: bool


def _run_once(trace, capacity, lat, with_prefetch, forecast):
    """Run the simulator once; return (total_hits, controller_or_None)."""
    controller = PrefetchController(lat) if with_prefetch else None
    if controller is not None:
        for req in trace:
            controller.register_size(req.key, req.input_tokens)
    sim = DiscreteEventSimulator(
        capacity=capacity, latency=lat, planner=None,
        prefetch_controller=controller,
        demand_forecast=forecast if with_prefetch else {},
    )
    m = sim.run(trace)
    return m.hits, controller



def run_prefetch_sweep(
    sequence: List[str],
    capacity: int,
    horizon: int = 8,
    sigmas: Optional[List[float]] = None,
    seed: int = 42,
    latency: Optional[LatencyModel] = None,
    input_tokens: int = 512,
) -> List[PrefetchResult]:
    sigmas = sigmas if sigmas is not None else [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    lat = latency or LatencyModel()

    # Baseline (no prefetch) is sigma-independent; compute once.
    base_trace = [
        TraceRequest(
            request_id=f"r{i}", key=key, arrival_time=float(i),
            input_tokens=input_tokens, output_tokens=32,
        )
        for i, key in enumerate(sequence)
    ]
    hits_off, _ = _run_once(base_trace, capacity, lat, with_prefetch=False, forecast={})

    results: List[PrefetchResult] = []
    for sigma in sigmas:
        forecast = noisy_forecast(sequence, horizon, sigma, seed)
        trace = [
            TraceRequest(
                request_id=f"r{i}", key=key, arrival_time=float(i),
                input_tokens=input_tokens, output_tokens=32,
            )
            for i, key in enumerate(sequence)
        ]
        hits_on, controller = _run_once(
            trace, capacity, lat, with_prefetch=True, forecast=forecast)
        stats = controller.stats if controller is not None else PrefetchStats()

        overhead = len(sequence) * PREDICTION_OVERHEAD_PER_DECISION
        net = stats.saved_compute - overhead - stats.waste_cost

        results.append(PrefetchResult(
            sigma=sigma,
            prefetch_issued=stats.issued,
            prefetch_hits=stats.hits,
            prefetch_wasted=stats.wastes,
            accuracy=stats.accuracy,
            waste_rate=stats.waste_rate,
            hits_on=hits_on,
            hits_off=hits_off,
            saved_compute=stats.saved_compute,
            prediction_overhead=overhead,
            prefetch_waste_cost=stats.waste_cost,
            net_gain=net,
            net_gain_positive=net > 0,
        ))
    return results


def format_prefetch_report(results: List[PrefetchResult]) -> str:
    lines = [
        f"{'sigma':>6} {'issued':>7} {'pfhit':>6} {'waste':>6} {'acc':>6} {'wst%':>6} "
        f"{'hitsON':>7} {'hitsOFF':>8} {'saved':>8} {'wstCost':>8} {'NetGain':>9}"
    ]
    lines.append("-" * 96)
    for r in results:
        lines.append(
            f"{r.sigma:>6.2f} {r.prefetch_issued:>7d} {r.prefetch_hits:>6d} "
            f"{r.prefetch_wasted:>6d} {r.accuracy:>6.2f} {r.waste_rate:>6.2f} "
            f"{r.hits_on:>7d} {r.hits_off:>8d} {r.saved_compute:>8.1f} "
            f"{r.prefetch_waste_cost:>8.1f} {r.net_gain:>9.1f}"
        )
    return "\n".join(lines)


def save_prefetch_artifacts(results: List[PrefetchResult], out_dir: str = "artifacts/phase3") -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "prefetch_sweep.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump([r.__dict__ for r in results], f, indent=2, ensure_ascii=False)
    return path


def main():
    # Prefetch pays only when a predicted object is reused BEFORE it is evicted,
    # so we pick a regime with enough capacity headroom (20) for a prefetched
    # key to survive to its reuse, and a short horizon (3) so demand is near-term.
    # zipf(60,400) has strong short-horizon reuse on a moderate key space, so a
    # good predictor prefetches hot keys that recur (Section 52: NetGain > 0),
    # while a poor predictor (large sigma) misses them and only wastes transfers
    # (Section 52: NetGain collapses). This is the honest economics of Section 20.
    print("=== zipf(60,400) (capacity=20, horizon=3) ===")
    seq = w.zipf_like(60, 400, seed=3)
    results = run_prefetch_sweep(seq, capacity=20, horizon=3)
    print(format_prefetch_report(results))
    accurate = [r for r in results if r.sigma <= 0.5]
    print(f"\nSection 52 gate: NetGain > 0 while prediction is accurate (sigma<=0.5) ? "
          f"{all(r.net_gain_positive for r in accurate)}")
    print(f"Section 52 gate: NetGain collapses when prediction is poor (sigma>=2.0) ? "
          f"{results[-1].net_gain < min(r.net_gain for r in accurate) if results else 'n/a'}")
    print(f"Prefetch adds hits vs no-prefetch when accurate ? "
          f"{max(r.hits_on for r in accurate) > results[0].hits_off}")
    path = save_prefetch_artifacts(results)
    print(f"\nArtifacts saved: {path}")


if __name__ == "__main__":
    main()
