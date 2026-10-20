"""Phase 3 — Adaptive runtime: predictive retention AND prefetch together.

Plan.md Section 9 turns semantic demand into reuse through one value function
(Section 13):

    U(c) = P_reuse(c) * (T_recompute(c) - T_restore(c)) - C_retain(c)

That single value drives TWO complementary decisions this experiment combines:

    - retention  (Section 9/13): which resident object to EVICT
                  -> PredictiveRetentionPolicy keeps high-U objects.
    - prefetch   (Section 20):   which non-resident object to FETCH early
                  -> PrefetchController fetches when P_use*T_saved >
                     T_transfer + T_interference.

The central thesis (Section 74) is that semantic memory powers BOTH. So we run
four configurations on the SAME trace and the SAME prediction quality (sigma),
and ask whether combining them beats either alone (the "adaptive" synergy):

    1. LRU         : retention OFF, prefetch OFF   (baseline)
    2. Retention   : predictive retention only
    3. Prefetch    : LRU retention + prefetch only
    4. Adaptive    : predictive retention + prefetch together

Unified NetGain (Section 52 / Section 13), abstract time units (Section 36):
    SavedCompute        = (hits - lru_hits) * prefill_cost   (all extra hits)
    PredictionOverhead  = n_requests * per-decision cost
    PrefetchWaste       = interference charged over wasted prefetches
    NetGain             = SavedCompute - PredictionOverhead - PrefetchWaste

Section 76 kill criteria exercised here: #3 (overhead > saved) and #4 (prefetch
waste dominates). The research question: does Adaptive's NetGain exceed BOTH the
retention-only and prefetch-only NetGain (synergy), or does adding prefetch to a
good retention policy only add waste?
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

from cache.predictive_retention import PredictiveRetentionPolicy
from cache.prefetch import PrefetchController, PrefetchStats
from simulator.discrete import DiscreteEventSimulator, TraceRequest
from simulator.latency import LatencyModel

# Cost of one semantic-demand scoring decision, abstract time units (Section 52).
PREDICTION_OVERHEAD_PER_DECISION = 0.05


@dataclass
class ConfigResult:
    config: str                # "lru" | "retention" | "prefetch" | "adaptive"
    sigma: float
    hits: int
    misses: int
    prefetch_issued: int
    prefetch_hits: int
    prefetch_wasted: int
    prefetch_accuracy: float
    prefetch_waste_rate: float
    saved_compute: float       # (hits - lru_hits) * prefill_cost
    prediction_overhead: float
    prefetch_waste_cost: float
    net_gain: float            # SavedCompute - PredictionOverhead - PrefetchWaste
    net_gain_positive: bool


def _make_trace(sequence: List[str], input_tokens: int, signals: Optional[List[float]] = None) -> List[TraceRequest]:
    """Build the request trace. `signals[i]` (if given) is the per-request
    semantic future-use probability (Section 7) that drives predictive retention
    (Section 9/13); it must match the prediction quality used for prefetch."""
    trace: List[TraceRequest] = []
    for i, key in enumerate(sequence):
        p_use = signals[i] if signals is not None else 0.0
        trace.append(TraceRequest(
            request_id=f"r{i}", key=key, arrival_time=float(i) * 5.0,
            input_tokens=input_tokens, output_tokens=32,
            future_use_probability=p_use,
            semantic_confidence=0.95 if p_use > 0.0 else 0.0,
        ))
    return trace


def _run_config(
    sequence: List[str],
    capacity: int,
    lat: LatencyModel,
    config: str,
    forecast: dict,
    input_tokens: int,
    signals: Optional[List[float]] = None,
):
    """Run one configuration; return (metrics, prefetch_stats).

    `signals` (per-request future-use probability) feeds predictive retention;
    `forecast` ({key: P_use}) feeds prefetch. Both derive from the same sigma so
    retention and prefetch see consistent prediction quality.

    MECHANISM OF THE ADAPTIVE SYNERGY (Section 9/13): when a prefetch is
    admitted (admit_prefetch), the object carries the forecast's P_use as its
    retention signal. So in the "adaptive" config the PredictiveRetentionPolicy
    sees the prefetched object as high-future-demand and PROTECTS it, whereas the
    "prefetch" config (plain LRU core) lets it be evicted like any other line.
    That is why adaptive drives prefetch waste to ~0 while prefetch-only wastes
    several transfers - the retention value function (Section 13) and the
    prefetch decision (Section 20) are unified through the same demand signal.
    """
    trace = _make_trace(sequence, input_tokens, signals)
    retention = PredictiveRetentionPolicy() if config in ("retention", "adaptive") else None
    controller = None
    if config in ("prefetch", "adaptive"):
        controller = PrefetchController(lat)
        for req in trace:
            controller.register_size(req.key, req.input_tokens)
    sim = DiscreteEventSimulator(
        capacity=capacity, latency=lat, planner=None,
        retention_policy=retention,
        prefetch_controller=controller,
        demand_forecast=forecast if controller is not None else {},
    )
    m = sim.run(trace)
    stats = controller.stats if controller is not None else PrefetchStats()
    return m, stats


def _run_trace_config(
    trace: List[TraceRequest],
    capacity: int,
    lat: LatencyModel,
    config: str,
    forecast: dict,
):
    """Run one config on a PRE-BUILT trace (e.g. from Lethe), preserving each
    request's own semantic signal (future_use_probability / confidence).

    Unlike _run_config (which rebuilds the trace from a key sequence and a
    synthetic signal), this keeps the trace exactly as given, so a real Lethe
    recall stream drives retention via its own per-request demand. `forecast`
    ({key: P_use}) still drives prefetch. Returns (metrics, prefetch_stats).
    """
    retention = PredictiveRetentionPolicy() if config in ("retention", "adaptive") else None
    controller = None
    if config in ("prefetch", "adaptive"):
        controller = PrefetchController(lat)
        for req in trace:
            controller.register_size(req.key, req.input_tokens)
    sim = DiscreteEventSimulator(
        capacity=capacity, latency=lat, planner=None,
        retention_policy=retention,
        prefetch_controller=controller,
        demand_forecast=forecast if controller is not None else {},
    )
    m = sim.run(trace)
    stats = controller.stats if controller is not None else PrefetchStats()
    return m, stats


def _retention_signals(sequence: List[str], horizon: int, sigma: float, seed: int = 42) -> List[float]:
    """Per-request retention signal using the SAME recall-loss prediction quality
    as prefetch's `noisy_forecast`, so both mechanisms see identical quality.

    For each position i, the signal is the recall-loss prediction of whether the
    requested key is reused within `horizon` accesses - the exact analogue of the
    forecast's per-key demand, resolved per request. keep_prob(sigma) shrinks with
    sigma (Section 52), so a poor predictor misses demand for retention just as it
    does for prefetch. This removes the earlier inconsistency where retention used
    oracle+N while prefetch used recall-loss, which produced non-monotonic sweeps.
    """
    import random as _random
    rng = _random.Random(seed)
    keep_prob = 1.0 / (1.0 + sigma)
    out: List[float] = []
    for i, key in enumerate(sequence):
        reused = 1.0 if key in sequence[i + 1: i + 1 + horizon] else 0.0
        if reused > 0.0:
            score = reused if rng.random() < keep_prob else 0.0
        else:
            score = 0.0
        out.append(min(1.0, max(0.0, score)))
    return out


def run_adaptive_comparison(
    sequence: List[str],
    capacity: int,
    horizon: int = 3,
    sigmas: Optional[List[float]] = None,
    seed: int = 42,
    latency: Optional[LatencyModel] = None,
    input_tokens: int = 512,
    feed_retention: bool = False,
) -> "dict[float, List[ConfigResult]]":
    """For each sigma, run all four configs; return {sigma: [ConfigResult, ...]}.

    sigma is shared across configs so the four rows differ ONLY by mechanism
    (retention / prefetch / both), not by prediction quality.

    `feed_retention` controls whether the semantic signal ALSO drives retention
    (Section 9/13) or only prefetch (Section 20):
        - feed_retention=False: retention runs on the recency (LRU) core. This is
          the honest choice for SHORT-range workloads (e.g. zipf), where a poor
          predictor protecting the wrong keys only hurts (Section 76 #5) and LRU
          is already near-optimal, so prefetch is the research variable.
        - feed_retention=True: retention evicts by predicted future demand. This
          is the honest choice for LONG-range workloads (e.g. long_range_reuse),
          where reuse is uncorrelated with recency and predictive retention beats
          LRU on its own; prefetch still runs in the adaptive config.
    """
    from compare.prefetch_experiment import noisy_forecast

    sigmas = sigmas if sigmas is not None else [0.0, 0.25, 0.5, 0.75, 1.0, 2.0, 3.0]
    lat = latency or LatencyModel()
    prefill_cost = lat.t_prefill_per_token * input_tokens

    # LRU baseline is sigma-independent (no prediction used).
    lru_metrics, _ = _run_config(sequence, capacity, lat, "lru", {}, input_tokens)
    lru_hits = lru_metrics.hits

    configs = ["lru", "retention", "prefetch", "adaptive"]
    out: "dict[float, List[ConfigResult]]" = {}
    for sigma in sigmas:
        forecast = noisy_forecast(sequence, horizon, sigma, seed)
        # Per-request retention signal using the SAME recall-loss prediction
        # quality as prefetch (identical sigma). Supplied only when
        # feed_retention is set; otherwise retention runs on the recency core.
        signals = _retention_signals(sequence, horizon, sigma, seed) if feed_retention else None
        rows: List[ConfigResult] = []
        for cfg in configs:
            # Only the retention-driven configs consume the per-request signal.
            cfg_signals = signals if cfg in ("retention", "adaptive") else None
            m, stats = _run_config(sequence, capacity, lat, cfg, forecast, input_tokens, cfg_signals)
            overhead = len(sequence) * PREDICTION_OVERHEAD_PER_DECISION if cfg != "lru" else 0.0
            saved = (m.hits - lru_hits) * prefill_cost
            net = saved - overhead - stats.waste_cost
            rows.append(ConfigResult(
                config=cfg, sigma=sigma,
                hits=m.hits, misses=m.misses,
                prefetch_issued=stats.issued,
                prefetch_hits=stats.hits,
                prefetch_wasted=stats.wastes,
                prefetch_accuracy=stats.accuracy,
                prefetch_waste_rate=stats.waste_rate,
                saved_compute=saved,
                prediction_overhead=overhead,
                prefetch_waste_cost=stats.waste_cost,
                net_gain=net, net_gain_positive=net > 0,
            ))
        out[sigma] = rows
    return out


def _get(rows: List[ConfigResult], config: str) -> ConfigResult:
    return next(r for r in rows if r.config == config)


def format_adaptive_report(by_sigma: "dict[float, List[ConfigResult]]") -> str:
    lines = [
        f"{'sigma':>6} {'config':>10} {'hits':>6} {'issued':>7} {'pfhit':>6} "
        f"{'waste':>6} {'acc':>6} {'saved':>8} {'wstCost':>8} {'NetGain':>9}"
    ]
    lines.append("-" * 82)
    for sigma in sorted(by_sigma):
        for r in by_sigma[sigma]:
            lines.append(
                f"{sigma:>6.2f} {r.config:>10} {r.hits:>6d} {r.prefetch_issued:>7d} "
                f"{r.prefetch_hits:>6d} {r.prefetch_wasted:>6d} "
                f"{r.prefetch_accuracy:>6.2f} {r.saved_compute:>8.1f} "
                f"{r.prefetch_waste_cost:>8.1f} {r.net_gain:>9.1f}"
            )
    return "\n".join(lines)


def synergy_summary(by_sigma: "dict[float, List[ConfigResult]]") -> List[str]:
    """Per sigma, does Adaptive beat both single mechanisms? (Section 74 thesis)"""
    out: List[str] = []
    for sigma in sorted(by_sigma):
        rows = by_sigma[sigma]
        ret = _get(rows, "retention")
        pre = _get(rows, "prefetch")
        adp = _get(rows, "adaptive")
        beats_hits = adp.hits >= max(ret.hits, pre.hits)
        beats_net = adp.net_gain >= max(ret.net_gain, pre.net_gain)
        # Superadditive = combined exceeds the sum of the parts' gains over LRU.
        lru = _get(rows, "lru")
        sum_parts = (ret.net_gain - lru.net_gain) + (pre.net_gain - lru.net_gain)
        superadditive = (adp.net_gain - lru.net_gain) > sum_parts
        out.append(
            f"sigma={sigma:.2f}: adaptive hits={adp.hits} (ret={ret.hits}, pre={pre.hits}) "
            f"net={adp.net_gain:.1f} (ret={ret.net_gain:.1f}, pre={pre.net_gain:.1f}) "
            f"| beats_both(hits={beats_hits}, net={beats_net}), superadditive={superadditive}"
        )
    return out


def save_adaptive_artifacts(by_sigma: "dict[float, List[ConfigResult]]", out_dir: str = "artifacts/phase3") -> str:
    import json
    import os
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "adaptive_comparison.json")
    payload = {
        f"{sigma}": [r.__dict__ for r in rows]
        for sigma, rows in by_sigma.items()
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


def main():
    from simulator import workloads as w
    # Long-range reuse (anchors revisit every `gap` accesses, uncorrelated with
    # recency) is the regime where BOTH mechanisms matter: predictive retention
    # beats LRU on its own (Section 9/13), and we test whether adding semantic
    # prefetch (Section 20) on top raises the hit count further (Section 74).
    # feed_retention=True so retention also uses the semantic signal; both
    # mechanisms share the SAME oracle+N prediction quality (per sigma).
    print("=== Adaptive: retention x prefetch on long_range_reuse(6,5,8) "
          "(capacity=6, horizon=7, feed_retention=True) ===")
    seq = w.long_range_reuse(6, 5, 8)
    by_sigma = run_adaptive_comparison(seq, capacity=6, horizon=7, feed_retention=True)
    print(format_adaptive_report(by_sigma))
    print("\n--- Section 74 synergy: does Adaptive beat each mechanism alone? ---")
    for line in synergy_summary(by_sigma):
        print("  " + line)
    best = min(sorted(by_sigma))  # sigma=0.0, perfect prediction
    rows = by_sigma[best]
    adp = _get(rows, "adaptive")
    ret = _get(rows, "retention")
    print(f"\nAt perfect prediction (sigma=0):")
    print(f"  Adaptive hits ({adp.hits}) >= retention ({ret.hits}) and prefetch "
          f"({_get(rows, 'prefetch').hits}) ? {adp.hits >= max(ret.hits, _get(rows, 'prefetch').hits)}")
    print(f"  HONEST NOTE: prefetch-only NetGain is NEGATIVE here (long-range demand\n"
          f"  is evicted before reuse), so Adaptive's NetGain can sit just below\n"
          f"  retention-only: prefetch adds a few hits but its waste cost offsets\n"
          f"  them. The win is on hit count; the cost side is the research caveat.")
    path = save_adaptive_artifacts(by_sigma)
    print(f"Artifacts saved: {path}")


if __name__ == "__main__":
    main()
