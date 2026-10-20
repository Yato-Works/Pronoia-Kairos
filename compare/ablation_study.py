"""Phase 3 — Ablation study of the calibrated runtime (Plan.md Section 70, 94 Tier 3).

Section 70 requires a staged ablation to prove the key claim:

    > "Does Semantic Demand REALLY add value?"

We decompose the calibrated runtime's NetGain (Section 90.2) into the contributions
of its two mechanisms, for each calibration of Lethe's demand:

    mechanism axis (what cache policy runs):
        none       : plain LRU (baseline A)
        retention  : + predictive retention driven by the calibrated signal (B)
        prefetch   : + prefetch driven by the calibrated forecast
        both       : retention + prefetch together (the adaptive runtime)

    calibration axis (what signal drives them):
        raw        : Lethe's uncalibrated demand (no semantic value claimed)
        recency    : recency-calibrated (immediate prefetch, H1/H2)
        popularity : popularity-calibrated (longer-term retention, H3/H4)

For each (calibration, mechanism) cell we measure NetGain (Section 52). Then:

    retention_contrib = NetGain(retention) - NetGain(none)
    prefetch_contrib  = NetGain(prefetch)  - NetGain(none)
    synergy           = NetGain(both) - NetGain(retention) - NetGain(prefetch) + NetGain(none)

This answers WHY popularity wins: is its NetGain from retention (protecting hot
chunks) or prefetch (fetching imminent demand) or their synergy? And it tests the
Section 70 claim directly: calibrated Semantic Demand adds value beyond the
raw-signal and no-signal baselines.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional

from semantic.lethe_trace import (
    synthesize_lethe_stream, build_calibrated_trace, calibrated_forecast,
    measure_correlation_distance,
)
from cache.predictive_retention import PredictiveRetentionPolicy
from cache.prefetch import PrefetchController
from simulator.discrete import DiscreteEventSimulator
from simulator.latency import LatencyModel

PREDICTION_OVERHEAD_PER_DECISION = 0.05
MECHANISMS = ("none", "retention", "prefetch", "both")
CALIBRATIONS = ("raw", "recency", "popularity")


@dataclass
class AblationCell:
    calibration: str
    mechanism: str
    regime: str
    hits: int
    saved_compute: float
    prediction_overhead: float
    prefetch_waste_cost: float
    net_gain: float


def _run_mechanism(trace, forecast, capacity, lat, mechanism, lru_hits, tokens):
    use_retention = mechanism in ("retention", "both")
    use_prefetch = mechanism in ("prefetch", "both")
    retention = PredictiveRetentionPolicy() if use_retention else None
    controller = None
    if use_prefetch:
        controller = PrefetchController(lat)
        for req in trace:
            controller.register_size(req.key, req.input_tokens)
    sim = DiscreteEventSimulator(
        capacity=capacity, latency=lat, planner=None,
        retention_policy=retention,
        prefetch_controller=controller,
        demand_forecast=forecast if controller is not None else {})
    m = sim.run(trace)
    stats = controller.stats if controller is not None else None
    saved = (m.hits - lru_hits) * lat.t_prefill_per_token * tokens
    overhead = len(trace) * PREDICTION_OVERHEAD_PER_DECISION if mechanism != "none" else 0.0
    waste_cost = stats.waste_cost if stats is not None else 0.0
    net = saved - overhead - waste_cost
    return m.hits, saved, overhead, waste_cost, net


def run_ablation_cell(calibration: str, mechanism: str, capacity: int = 6,
                      reuse_gap: int = 4, n_objects: int = 20, n_recalls: int = 300,
                      skew: float = 1.0, seed: int = 7, tokens_per_object: int = 512,
                      latency: Optional[LatencyModel] = None) -> AblationCell:
    """One (calibration, mechanism) cell of the Section 70 ablation matrix."""
    events = synthesize_lethe_stream(n_objects, n_recalls, reuse_gap, seed=seed, skew=skew)
    trace = build_calibrated_trace(events, calibration, tokens_per_object)
    forecast = calibrated_forecast(events, calibration)
    lat = latency or LatencyModel()
    regime = measure_correlation_distance(trace).regime

    # Baseline (mechanism='none') hits: plain LRU on the same trace.
    lru_hits, _, _, _, _ = _run_mechanism(trace, {}, capacity, lat, "none", 0, tokens_per_object)
    hits, saved, overhead, waste_cost, net = _run_mechanism(
        trace, forecast, capacity, lat, mechanism, lru_hits, tokens_per_object)
    return AblationCell(
        calibration=calibration, mechanism=mechanism, regime=regime,
        hits=hits, saved_compute=saved, prediction_overhead=overhead,
        prefetch_waste_cost=waste_cost, net_gain=net)


@dataclass
class AblationDecomposition:
    calibration: str
    cells: Dict[str, AblationCell]           # mechanism -> cell
    retention_contrib: float                 # NetGain(retention) - NetGain(none)
    prefetch_contrib: float                  # NetGain(prefetch) - NetGain(none)
    both_net: float                          # NetGain(both)
    synergy: float                           # both - retention - prefetch + none
    semantic_value_vs_raw: float             # this-cal 'both' - raw 'both'


def run_ablation(capacity: int = 6, reuse_gap: int = 4, n_objects: int = 20,
                 n_recalls: int = 300, skew: float = 1.0, seed: int = 7,
                 calibrations=CALIBRATIONS) -> List[AblationDecomposition]:
    """Full (calibration x mechanism) ablation with per-calibration decomposition."""
    decomps: List[AblationDecomposition] = []
    raw_both_net = None
    for cal in calibrations:
        cells = {mech: run_ablation_cell(cal, mech, capacity=capacity, reuse_gap=reuse_gap,
                                         n_objects=n_objects, n_recalls=n_recalls,
                                         skew=skew, seed=seed)
                 for mech in MECHANISMS}
        none_net = cells["none"].net_gain
        retention_contrib = cells["retention"].net_gain - none_net
        prefetch_contrib = cells["prefetch"].net_gain - none_net
        both_net = cells["both"].net_gain
        synergy = both_net - cells["retention"].net_gain - cells["prefetch"].net_gain + none_net
        if cal == "raw":
            raw_both_net = both_net
        decomps.append(AblationDecomposition(
            calibration=cal, cells=cells,
            retention_contrib=retention_contrib, prefetch_contrib=prefetch_contrib,
            both_net=both_net, synergy=synergy,
            semantic_value_vs_raw=(both_net - raw_both_net) if raw_both_net is not None else 0.0))
    return decomps


def section70_verdict(decomps: List[AblationDecomposition]) -> List[str]:
    """Section 70: does Semantic Demand (calibration) add real value?"""
    out: List[str] = []
    by_cal = {d.calibration: d for d in decomps}
    raw = by_cal.get("raw")
    # 1) Semantic Demand over the no-semantic baseline (mechanism=none ~ LRU):
    for cal in ("recency", "popularity"):
        d = by_cal.get(cal)
        if d is None:
            continue
        none_net = d.cells["none"].net_gain
        adds = d.both_net > none_net
        out.append(f"{cal}: full adaptive NetGain {d.both_net:+.1f} vs no-signal "
                   f"baseline {none_net:+.1f} -> Semantic Demand ADDS value = {adds}")
    # 2) Calibrated over raw (the core Section 70 claim):
    for cal in ("recency", "popularity"):
        d = by_cal.get(cal)
        if d and raw:
            out.append(f"{cal} over raw: NetGain {d.both_net:+.1f} vs raw {raw.both_net:+.1f} "
                       f"-> calibration beats raw signal = {d.both_net > raw.both_net}")
    # 3) Which mechanism carries the value?
    best = max(decomps, key=lambda d: d.both_net)
    carrier = ("retention" if best.retention_contrib >= best.prefetch_contrib
               else "prefetch")
    out.append(f"best calibration = {best.calibration}: value carried mainly by "
               f"{carrier} (retention_contrib {best.retention_contrib:+.1f}, "
               f"prefetch_contrib {best.prefetch_contrib:+.1f}, synergy {best.synergy:+.1f})")
    return out


def format_ablation_report(decomps: List[AblationDecomposition]) -> str:
    lines = [
        f"{'calibration':>12} {'mechanism':>10} {'hits':>5} {'saved':>8} "
        f"{'overhead':>9} {'waste':>8} {'NetGain':>9}"
    ]
    lines.append("-" * 66)
    for d in decomps:
        for mech in MECHANISMS:
            c = d.cells[mech]
            lines.append(
                f"{d.calibration:>12} {mech:>10} {c.hits:>5} {c.saved_compute:>8.0f} "
                f"{c.prediction_overhead:>9.1f} {c.prefetch_waste_cost:>8.1f} "
                f"{c.net_gain:>9.1f}"
            )
    lines.append("")
    lines.append(f"{'calibration':>12} {'retContrib':>11} {'preContrib':>11} "
                 f"{'synergy':>9} {'bothNet':>9} {'vsRaw':>9}")
    lines.append("-" * 66)
    for d in decomps:
        lines.append(
            f"{d.calibration:>12} {d.retention_contrib:>11.1f} {d.prefetch_contrib:>11.1f} "
            f"{d.synergy:>9.1f} {d.both_net:>9.1f} {d.semantic_value_vs_raw:>9.1f}"
        )
    return "\n".join(lines)


def save_ablation_artifacts(decomps: List[AblationDecomposition], out_dir: str = "artifacts/phase3") -> str:
    import json
    import os
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "ablation_study.json")
    payload = [{
        "calibration": d.calibration,
        "cells": {mech: c.__dict__ for mech, c in d.cells.items()},
        "retention_contrib": d.retention_contrib,
        "prefetch_contrib": d.prefetch_contrib,
        "synergy": d.synergy,
        "both_net": d.both_net,
        "semantic_value_vs_raw": d.semantic_value_vs_raw,
    } for d in decomps]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


def main():
    print("=== Ablation study (Section 70): does Semantic Demand add value, "
          "and via which mechanism? ===\n")
    decomps = run_ablation(capacity=6, reuse_gap=4, skew=1.0)
    print(format_ablation_report(decomps))
    print("\n--- Section 70 verdict ---")
    for line in section70_verdict(decomps):
        print("  " + line)
    path = save_ablation_artifacts(decomps)
    print(f"\nArtifacts saved: {path}")


if __name__ == "__main__":
    main()
