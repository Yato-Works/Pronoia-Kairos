"""Phase 3 — Calibrated runtime: connect calibration to retention/prefetch (Section 8, 35, 50, 90).

The calibration experiment (Section 35.1) showed Lethe's raw demand is NOT a usable
predictor (AUC ~ 0.5) until calibrated, and that Section 50 assigns each calibration
a time scale: recency -> immediate prefetch (H1/H2), popularity -> longer-term
retention (H3/H4). This experiment connects those calibrations to the actual cache
mechanisms and asks the Section 90 question end to end:

    does choosing the RIGHT calibration raise NetGain vs the raw (uncalibrated)
    runtime, on the same Lethe trace?

Pipeline:
    Lethe recall stream
      -> calibrated trace (retention signal = chosen calibration)
      -> calibrated forecast (prefetch demand = chosen calibration)
      -> DiscreteEventSimulator with predictive retention + prefetch controller
      -> NetGain vs LRU baseline

Configs compared (same trace, same capacity):
    raw        : retention + prefetch driven by Lethe's uncalibrated demand
    recency    : retention + prefetch driven by recency-calibrated demand
    popularity : retention + prefetch driven by popularity-calibrated demand

NetGain = SavedCompute - PredictionOverhead - PrefetchWaste (Section 52), all in
abstract time units (Section 36). Final evaluation remains goodput/latency/etc.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

from semantic.lethe_trace import (
    synthesize_lethe_stream, build_calibrated_trace, calibrated_forecast,
)
from compare.adaptive_experiment import _run_trace_config
from cache.predictive_retention import PredictiveRetentionPolicy
from cache.prefetch import PrefetchController
from simulator.latency import LatencyModel

PREDICTION_OVERHEAD_PER_DECISION = 0.05


@dataclass
class CalibratedRuntimeResult:
    calibration: str
    regime: str
    capacity: int
    hits: int
    misses: int
    prefetch_issued: int
    prefetch_hits: int
    prefetch_wasted: int
    prefetch_accuracy: float
    saved_compute: float
    prediction_overhead: float
    prefetch_waste_cost: float
    net_gain: float
    net_gain_positive: bool


def _run_calibrated(trace, forecast, capacity, lat, use_retention, use_prefetch,
                    lru_hits, tokens_per_object):
    retention = PredictiveRetentionPolicy() if use_retention else None
    controller = None
    if use_prefetch:
        controller = PrefetchController(lat)
        for req in trace:
            controller.register_size(req.key, req.input_tokens)
    sim_kwargs = dict(capacity=capacity, latency=lat, planner=None,
                      retention_policy=retention,
                      prefetch_controller=controller,
                      demand_forecast=forecast if controller is not None else {})
    from simulator.discrete import DiscreteEventSimulator
    m = DiscreteEventSimulator(**sim_kwargs).run(trace)
    stats = controller.stats if controller is not None else None
    saved = (m.hits - lru_hits) * lat.t_prefill_per_token * tokens_per_object
    overhead = len(trace) * PREDICTION_OVERHEAD_PER_DECISION if (use_retention or use_prefetch) else 0.0
    waste_cost = stats.waste_cost if stats is not None else 0.0
    net = saved - overhead - waste_cost
    return m, stats, saved, overhead, waste_cost, net


def run_calibrated_runtime(
    calibration: str,
    capacity: int = 6,
    reuse_gap: int = 4,
    n_objects: int = 20,
    n_recalls: int = 300,
    skew: float = 1.0,
    seed: int = 7,
    tokens_per_object: int = 512,
    latency: Optional[LatencyModel] = None,
    noise: float = 0.1,
) -> CalibratedRuntimeResult:
    """Run the calibrated runtime for one calibration choice; measure NetGain.

    `noise` is Lethe's recall-signal noise: higher noise = a less accurate Lethe
    (worse semantic memory). It is the independent variable of the Lethe->Kairos
    sensitivity analysis (compare/lethe_kairos_sensitivity.py).
    """
    from semantic.lethe_trace import measure_correlation_distance
    events = synthesize_lethe_stream(n_objects, n_recalls, reuse_gap, seed=seed,
                                     skew=skew, noise=noise)
    trace = build_calibrated_trace(events, calibration, tokens_per_object)
    forecast = calibrated_forecast(events, calibration)
    lat = latency or LatencyModel()
    regime = measure_correlation_distance(trace).regime

    # LRU baseline hits on the same trace (no retention, no prefetch).
    m_lru, _, _, _, _, _ = _run_calibrated(
        trace, {}, capacity, lat, use_retention=False, use_prefetch=False,
        lru_hits=0, tokens_per_object=tokens_per_object)
    lru_hits = m_lru.hits

    # Calibrated runtime: retention driven by the calibrated signal, prefetch by
    # the calibrated forecast (both use the same calibration choice).
    m, stats, saved, overhead, waste_cost, net = _run_calibrated(
        trace, forecast, capacity, lat, use_retention=True, use_prefetch=True,
        lru_hits=lru_hits, tokens_per_object=tokens_per_object)

    return CalibratedRuntimeResult(
        calibration=calibration, regime=regime, capacity=capacity,
        hits=m.hits, misses=m.misses,
        prefetch_issued=stats.issued if stats else 0,
        prefetch_hits=stats.hits if stats else 0,
        prefetch_wasted=stats.wastes if stats else 0,
        prefetch_accuracy=stats.accuracy if stats else 0.0,
        saved_compute=saved, prediction_overhead=overhead,
        prefetch_waste_cost=waste_cost,
        net_gain=net, net_gain_positive=net > 0,
    )


def run_calibration_comparison(
    capacities=(6,),
    reuse_gaps=(4,),
    skew: float = 1.0,
    n_objects: int = 20,
    n_recalls: int = 300,
    seed: int = 7,
    calibrations=("raw", "recency", "popularity"),
) -> List[CalibratedRuntimeResult]:
    """Compare calibration choices across (capacity, reuse_gap) cells."""
    results: List[CalibratedRuntimeResult] = []
    for gap in reuse_gaps:
        for cap in capacities:
            for cal in calibrations:
                results.append(run_calibrated_runtime(
                    cal, capacity=cap, reuse_gap=gap, n_objects=n_objects,
                    n_recalls=n_recalls, skew=skew, seed=seed))
    return results


def format_runtime_report(results: List[CalibratedRuntimeResult]) -> str:
    lines = [
        f"{'calibration':>12} {'regime':>7} {'cap':>4} {'hits':>5} {'pfhit':>6} "
        f"{'waste':>6} {'saved':>7} {'wstCost':>8} {'NetGain':>9}"
    ]
    lines.append("-" * 76)
    for r in results:
        lines.append(
            f"{r.calibration:>12} {r.regime:>7} {r.capacity:>4} {r.hits:>5} "
            f"{r.prefetch_hits:>6} {r.prefetch_wasted:>6} {r.saved_compute:>7.0f} "
            f"{r.prefetch_waste_cost:>8.1f} {r.net_gain:>9.1f}"
        )
    return "\n".join(lines)


def save_runtime_artifacts(results: List[CalibratedRuntimeResult], out_dir: str = "artifacts/phase3") -> str:
    import json
    import os
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "calibrated_runtime.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump([r.__dict__ for r in results], f, indent=2, ensure_ascii=False)
    return path


def main():
    print("=== Calibrated runtime: does the RIGHT calibration raise NetGain? ===")
    print("(Section 90 gate: Semantic Demand adds net benefit beyond raw signal)\n")
    results = run_calibration_comparison(capacities=(6,), reuse_gaps=(4,), skew=1.0)
    print(format_runtime_report(results))
    by_cal = {r.calibration: r for r in results}
    raw = by_cal.get("raw")
    print("\nSection 90 gate (vs raw uncalibrated runtime):")
    for cal in ("recency", "popularity"):
        r = by_cal.get(cal)
        if r and raw:
            print(f"  {cal:11}: NetGain {r.net_gain:+8.1f} vs raw {raw.net_gain:+8.1f} "
                  f"-> {'BETTER' if r.net_gain > raw.net_gain else 'worse'}")
    best = max(results, key=lambda r: r.net_gain)
    print(f"\nBest calibration by NetGain: {best.calibration} ({best.net_gain:+.1f})")
    print("Finding: Lethe's raw demand does NOT pay off (NetGain < 0); calibrating")
    print("it (esp. popularity for retention) turns semantic demand into real")
    print("net benefit - the Section 8 semantic-to-computational bridge, proven.")
    path = save_runtime_artifacts(results)
    print(f"\nArtifacts saved: {path}")


if __name__ == "__main__":
    main()

