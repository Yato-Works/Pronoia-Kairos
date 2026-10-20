"""Phase 3 — Signal calibration experiment (Plan.md Section 35, 48, 50, 90).

The Lethe bridge showed that Lethe's RAW semantic demand
(relevance * confidence * (1 + future_utility)) does NOT directly drive predictive
retention, because it is position-INDEPENDENT "general future value" - it does not
say "needed again within horizon H from HERE". Calibrating the signal to the
horizon reuse structure (Section 50 prediction target) makes retention function.

This experiment QUANTIFIES that effect with the Section 35 prediction metrics.

Prediction task (per position i, Section 50): will the chunk recalled at position
i be reused within horizon H later recalls? Ground truth = realized reuse.

We compare two signals on the SAME task:
    raw        = Lethe's position-independent utility  (what Lethe emits)
    calibrated = 1.0 if reused within horizon else 0    (calibrated to Section 50)

Metrics (Section 35):
    AUC        - threshold-free ranking quality (does the signal rank future-used
                 chunks above not-reused ones?)
    ECE        - calibration: is a 0.8-signal right 80% of the time? (lower better)
    precision/recall @ 0.5 - binary future-use accuracy

The research question: does calibration improve AUC / lower ECE, and is the gain
regime-dependent (short/mid/long correlation distance)? Final evaluation remains
goodput/latency/memory/quality (Section 35 note), measured in the Lethe bridge.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

from semantic.lethe_trace import (
    synthesize_lethe_stream, p_sem_list, measure_correlation_distance,
    LetheTraceBuilder,
)
from semantic.prediction_metrics import (
    precision_recall, roc_auc, expected_calibration_error,
)


@dataclass
class SignalQuality:
    regime: str
    horizon: int
    skew: float
    n: int
    base_rate: float                 # fraction of positives (reused within horizon)
    # raw Lethe signal
    raw_auc: float
    raw_ece: float
    raw_precision: float
    raw_recall: float
    # recency-calibrated signal (immediate prefetch, H1/H2)
    rec_auc: float
    rec_ece: float
    rec_precision: float
    rec_recall: float
    # popularity-calibrated signal (longer-term retention, H3/H4)
    pop_auc: float
    pop_ece: float
    pop_precision: float
    pop_recall: float

    @property
    def rec_auc_gain(self) -> float:
        return self.rec_auc - self.raw_auc

    @property
    def pop_auc_gain(self) -> float:
        return self.pop_auc - self.raw_auc

    @property
    def rec_ece_reduction(self) -> float:
        return self.raw_ece - self.rec_ece  # positive = calibration improved it

    @property
    def pop_ece_reduction(self) -> float:
        return self.raw_ece - self.pop_ece

    @property
    def best_calibration(self) -> str:
        """Which calibration wins at THIS horizon (Section 50 separation)."""
        gains = {"recency": self.rec_auc_gain, "popularity": self.pop_auc_gain}
        return max(gains, key=gains.get)


def _labels(keys: List[str], horizon: int) -> List[int]:
    """Ground truth: 1 if chunk at position i recurs within `horizon` later."""
    n = len(keys)
    labels = [0] * n
    for i in range(n):
        if keys[i] in keys[i + 1: i + 1 + horizon]:
            labels[i] = 1
    return labels


def evaluate_signal_quality(
    reuse_gap: int,
    horizon: Optional[int] = None,
    n_objects: int = 20,
    n_recalls: int = 300,
    seed: int = 7,
    skew: float = 0.0,
    decouple: bool = False,
    noise: float = 0.1,
) -> SignalQuality:
    """Compare raw vs recency vs popularity signal on the horizon-reuse task."""
    events = synthesize_lethe_stream(n_objects, n_recalls, reuse_gap, seed=seed,
                                     skew=skew, decouple=decouple, noise=noise)
    keys = [ev.memory.chunk_id for ev in events]
    horizon = horizon if horizon is not None else max(1, int(reuse_gap) + 2)

    labels = _labels(keys, horizon)
    raw = p_sem_list(events)
    # CALIBRATION 1 - recency (immediate prefetch, H1/H2, Section 50):
    # raw utility * recency, NO future info. recency = decay^age, age = accesses
    # since last use. Turns "general importance" into "likely needed soon HERE".
    last_seen: dict = {}
    decay = 0.8
    recency_sig = []
    for i, k in enumerate(keys):
        age = (i - last_seen[k]) if k in last_seen else 0
        recency_sig.append(min(1.0, max(0.0, raw[i] * (decay ** max(0, age)))))
        last_seen[k] = i
    # CALIBRATION 2 - popularity (longer-term retention, H3/H4, Section 50):
    # raw utility * running access frequency, NO future info. A chunk accessed
    # often so far is "important" over a longer window even if not used recently,
    # which is exactly the retention decision (keep what will matter later).
    counts: dict = {}
    popularity_sig = []
    for i, k in enumerate(keys):
        counts[k] = counts.get(k, 0) + 1
        freq = counts[k] / (i + 1)  # running share of this chunk so far
        popularity_sig.append(min(1.0, max(0.0, raw[i] * freq)))

    raw_bin = [1 if p >= 0.5 else 0 for p in raw]
    rec_bin = [1 if p >= 0.5 else 0 for p in recency_sig]
    pop_bin = [1 if p >= 0.5 else 0 for p in popularity_sig]

    rp, rr = precision_recall(labels, raw_bin)
    rcp, rcr = precision_recall(labels, rec_bin)
    pcp, pcr = precision_recall(labels, pop_bin)
    # Regime from the key sequence (reuse-distance analysis, Section 63.1 axis).
    trace = LetheTraceBuilder().build(events, horizon=horizon)
    profile = measure_correlation_distance(trace)

    return SignalQuality(
        regime=profile.regime,
        horizon=horizon,
        skew=skew,
        n=len(keys),
        base_rate=sum(labels) / len(labels) if labels else 0.0,
        raw_auc=roc_auc(raw, labels),
        raw_ece=expected_calibration_error(raw, labels),
        raw_precision=rp, raw_recall=rr,
        rec_auc=roc_auc(recency_sig, labels),
        rec_ece=expected_calibration_error(recency_sig, labels),
        rec_precision=rcp, rec_recall=rcr,
        pop_auc=roc_auc(popularity_sig, labels),
        pop_ece=expected_calibration_error(popularity_sig, labels),
        pop_precision=pcp, pop_recall=pcr,
    )


def run_calibration_sweep(
    horizons=(2, 4, 8),
    skews=(0.0, 1.0),
    n_objects: int = 30,
    n_recalls: int = 400,
    seed: int = 7,
    decouple: bool = False,
) -> List[SignalQuality]:
    """Sweep horizon x popularity-skew; return per-cell signal quality.

    decouple=False: popularity and recency are correlated (realistic skewed
    traffic). decouple=True: popularity and recency are anti-correlated, the
    regime that separates the Section 50 immediate (recency) vs long-term
    (popularity) signals.
    """
    results: List[SignalQuality] = []
    for skew in skews:
        for hz in horizons:
            results.append(evaluate_signal_quality(
                reuse_gap=hz, horizon=hz, n_objects=n_objects,
                n_recalls=n_recalls, seed=seed, skew=skew, decouple=decouple))
    return results


def format_calibration_report(results: List[SignalQuality]) -> str:
    lines = [
        f"{'regime':>7} {'hz':>3} {'skew':>5} {'base':>5} "
        f"{'rawAUC':>7} {'recAUC':>7} {'popAUC':>7} "
        f"{'recGain':>8} {'popGain':>8} {'best':>11}"
    ]
    lines.append("-" * 84)
    for r in results:
        lines.append(
            f"{r.regime:>7} {r.horizon:>3} {r.skew:>5.1f} {r.base_rate:>5.2f} "
            f"{r.raw_auc:>7.3f} {r.rec_auc:>7.3f} {r.pop_auc:>7.3f} "
            f"{r.rec_auc_gain:>+8.3f} {r.pop_auc_gain:>+8.3f} {r.best_calibration:>11}"
        )
    return "\n".join(lines)


def save_calibration_artifacts(results: List[SignalQuality], out_dir: str = "artifacts/phase3") -> str:
    import json
    import os
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "calibration_sweep.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump([r.__dict__ for r in results], f, indent=2, ensure_ascii=False)
    return path


def main():
    print("=== Signal calibration: raw vs recency vs popularity ===")
    print("(Section 35 metrics; Section 50 separates immediate prefetch [recency]\n"
          " from longer-term retention [popularity])\n")

    print("--- COUPLED traffic (popularity ~ recency, realistic skewed) ---")
    coupled = run_calibration_sweep(horizons=(2, 4, 8), skews=(1.0,), decouple=False)
    print(format_calibration_report(coupled))

    print("\n--- DECOUPLED traffic (popularity anti-correlated with recency) ---")
    decoupled = run_calibration_sweep(horizons=(2, 4, 8), skews=(1.0,), decouple=True)
    print(format_calibration_report(decoupled))

    print("\nSection 50 separation (decoupled regime):")
    short = next(r for r in decoupled if r.horizon == 2)
    long_ = next(r for r in decoupled if r.horizon == 8)
    print(f"  immediate (H1/H2, hz=2): best = {short.best_calibration} "
          f"(rec_auc={short.rec_auc:.3f}, pop_auc={short.pop_auc:.3f})")
    print(f"  long-term (H3/H4, hz=8): best = {long_.best_calibration} "
          f"(rec_auc={long_.rec_auc:.3f}, pop_auc={long_.pop_auc:.3f})")
    print(f"  both calibrations beat raw at every horizon: "
          f"{all(r.rec_auc_gain > 0 and r.pop_auc_gain > 0 for r in decoupled)}")

    path = save_calibration_artifacts(coupled + decoupled)
    print(f"\nArtifacts saved: {path}")


if __name__ == "__main__":
    main()

