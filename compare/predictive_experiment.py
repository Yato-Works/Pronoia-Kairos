"""Phase 2/3 experiment — predictive cache regret vs Belady (Plan.md §15, §35, §52).

Signal quality sweep:
    ground truth = "will this key be reused within the next W accesses?"
    predicted signal = truth + Gaussian noise(sigma)
    sigma=0.0  -> perfect prediction (upper bound for any predictor)
    sigma=inf  -> useless prediction (retention degenerates to ~LRU)

For each sigma, run the DiscreteEventSimulator with
PredictiveRetentionPolicy on the SAME key sequence, then compare:

    predictive_hits vs lru_hits vs belady_hits
    regret = predictive_hits - belady_hits      (must be <= 0)
    improvement = predictive_hits - lru_hits    (the research question)

§52 NetGain proxy (abstract time units, calibrated later §43):
    NetGain = SavedCompute - PredictionOverhead - PrefetchWaste
    SavedCompute       = (lru_misses - pred_misses) * prefill_cost
    PredictionOverhead = n_requests * overhead_per_decision
Gate: NetGain > 0 AND regret improves as sigma decreases.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import json
import os
import random

from compare.belady_lru import belady_optimal, lru_hit_count
from cache.predictive_retention import PredictiveRetentionPolicy
from simulator.discrete import DiscreteEventSimulator, TraceRequest
from simulator.latency import LatencyModel
from semantic.prediction_metrics import precision_recall, roc_auc, expected_calibration_error

# Overhead of one semantic-demand scoring decision, abstract time units.
PREDICTION_OVERHEAD_PER_DECISION = 0.05


def oracle_reuse_probability(sequence: List[str], index: int, window: int) -> float:
    """§50 ground truth: P(reuse within horizon H) = 1 if key reappears
    within `window` accesses after `index`, else 0."""
    key = sequence[index]
    return 1.0 if key in sequence[index + 1: index + 1 + window] else 0.0


def noisy_signals(sequence: List[str], window: int, sigma: float, seed: int = 42) -> List[float]:
    """Predicted future-use probability = oracle + N(0, sigma), clipped to [0,1]."""
    rng = random.Random(seed)
    out = []
    for i in range(len(sequence)):
        p = oracle_reuse_probability(sequence, i, window)
        out.append(min(1.0, max(0.0, p + rng.gauss(0.0, sigma))))
    return out


@dataclass
class SweepResult:
    sigma: float
    predictive_hits: int
    predictive_misses: int
    lru_hits: int
    belady_hits: int
    regret_vs_belady: int      # <= 0 (correctness: Belady upper bound)
    improvement_over_lru: int  # > 0 = semantic prediction helped
    precision: float
    recall: float
    auc: float
    ece: float
    net_gain: float            # §52 proxy
    net_gain_positive: bool


def run_predictive_sweep(
    sequence: List[str],
    capacity: int,
    sigmas: Optional[List[float]] = None,
    window: int = 4,
    seed: int = 42,
    latency: Optional[LatencyModel] = None,
    input_tokens: int = 512,
) -> List[SweepResult]:
    sigmas = sigmas if sigmas is not None else [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]
    lat = latency or LatencyModel()
    prefill_cost = lat.t_prefill_per_token * input_tokens

    belady = belady_optimal(sequence, capacity)
    lru = lru_hit_count(sequence, capacity)
    results: List[SweepResult] = []

    for sigma in sigmas:
        probs = noisy_signals(sequence, window, sigma, seed)
        trace = [
            TraceRequest(
                request_id=f"r{i}", key=key, arrival_time=float(i),
                input_tokens=input_tokens, output_tokens=32,
                future_use_probability=probs[i], semantic_confidence=0.95,
            )
            for i, key in enumerate(sequence)
        ]
        sim = DiscreteEventSimulator(
            capacity=capacity, latency=lat, planner=None,
            retention_policy=PredictiveRetentionPolicy(),
        )
        m = sim.run(trace)

        labels = [oracle_reuse_probability(sequence, i, window) for i in range(len(sequence))]
        preds_bin = [1 if p >= 0.5 else 0 for p in probs]
        prec, rec = precision_recall(labels, preds_bin)
        auc = roc_auc(probs, labels)
        ece = expected_calibration_error(probs, labels)

        saved = (m.hits - lru) * prefill_cost  # positive when prediction beats LRU
        overhead = len(sequence) * PREDICTION_OVERHEAD_PER_DECISION
        net = saved - overhead  # prefetch waste tracked separately by sim metrics

        results.append(SweepResult(
            sigma=sigma, predictive_hits=m.hits, predictive_misses=m.misses,
            lru_hits=lru, belady_hits=belady,
            regret_vs_belady=m.hits - belady,
            improvement_over_lru=m.hits - lru,
            precision=prec, recall=rec, auc=auc, ece=ece,
            net_gain=net, net_gain_positive=net > 0,
        ))
    return results


def format_sweep_report(results: List[SweepResult]) -> str:
    lines = [
        f"{'sigma':>6} {'pred':>6} {'LRU':>6} {'Belady':>7} {'regret':>7} {'vs LRU':>7} "
        f"{'prec':>6} {'rec':>6} {'AUC':>6} {'ECE':>6} {'NetGain':>9}"
    ]
    lines.append("-" * 92)
    for r in results:
        lines.append(
            f"{r.sigma:>6.2f} {r.predictive_hits:>6d} {r.lru_hits:>6d} {r.belady_hits:>7d} "
            f"{r.regret_vs_belady:>7d} {r.improvement_over_lru:>7d} "
            f"{r.precision:>6.2f} {r.recall:>6.2f} {r.auc:>6.2f} {r.ece:>6.2f} "
            f"{r.net_gain:>9.1f}"
        )
    return "\n".join(lines)


def save_sweep_artifacts(results: List[SweepResult], out_dir: str = "artifacts/phase3") -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "predictive_sweep.json")
    payload = [r.__dict__ for r in results]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


def main():
    from simulator import workloads as w
    wl = {"zipf(30,400)": w.zipf_like(30, 400), "agent(60,6)": w.agent_workflow(60, 6)}
    last = None
    for name, seq in wl.items():
        print(f"\n=== {name} (capacity=4, horizon=4) ===")
        results = run_predictive_sweep(seq, capacity=4, window=4)
        last = results
        print(format_sweep_report(results))
        print(f"52 gate: all NetGain > 0 ? {all(r.net_gain_positive for r in results)}")
        print(f"Regret <= 0 everywhere ? {all(r.regret_vs_belady <= 0 for r in results)}")
    if last is not None:
        path = save_sweep_artifacts(last)
        print(f"\nArtifacts saved: {path}")


if __name__ == "__main__":
    main()
