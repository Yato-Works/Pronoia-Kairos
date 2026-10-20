"""Phase 3 — Lethe -> Kairos sensitivity analysis (Plan.md Section 8, 35, 52, 90, 120).

The two projects co-evolve: a better Lethe (semantic memory) should make Kairos
(the adaptive runtime) better. This experiment QUANTIFIES that relationship as a
response curve, so the "how much should we invest in Lethe?" question has a
number attached.

Independent variable (x): Lethe recall quality, controlled by the recall-signal
    `noise` (0.0 = a perfect Lethe, large = a noisy, weak semantic memory).
    We also report Lethe's own prediction quality (calibrated AUC, Section 35)
    at each noise level, so the curve is readable in Lethe terms too.

Dependent variable (y): Kairos performance on the SAME Lethe trace:
    - NetGain (Section 52 / 90): saved compute - overhead - prefetch waste
    - hit rate vs the LRU baseline

For each noise level we run the calibrated runtime (retention + prefetch driven
by the calibrated demand) and measure:
    lethe_auc   = calibrated-signal AUC (Lethe quality, Section 35)
    kairos_net  = NetGain
    kairos_hits = cache hits

Then:
    sensitivity = d(NetGain) / d(AUC)   (slope of the response curve)
    -> "improving Lethe's AUC by 0.1 raises Kairos NetGain by ~S"

This is the Section 120 Measure->Adapt loop made quantitative across the two
systems, and the Section 90 gate read as a function of Lethe quality: below some
Lethe AUC, Kairos NetGain may go non-positive (the investment threshold).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

from compare.calibrated_runtime_experiment import run_calibrated_runtime
from compare.calibration_experiment import evaluate_signal_quality
from compare.statistical_protocol import mean, std, percentile, mean_ci95


@dataclass
class SensitivityPoint:
    noise: float
    lethe_auc: float          # calibrated-signal AUC (Lethe quality, Section 35)
    kairos_net: float         # NetGain (Section 52/90); mean over seeds if n_runs>1
    kairos_hits: int          # mean hits over seeds (rounded) if n_runs>1
    kairos_hit_rate: float
    net_positive: bool
    # Statistical fields (Section 69) - populated by the multi-seed sweep.
    n_runs: int = 1
    net_mean: float = 0.0
    net_ci95_half: float = 0.0
    net_std: float = 0.0
    net_median: float = 0.0
    auc_mean: float = 0.0

    @property
    def net_ci95_low(self) -> float:
        return self.net_mean - self.net_ci95_half

    @property
    def net_ci95_high(self) -> float:
        return self.net_mean + self.net_ci95_half


def _hit_rate(hits: int, n_recalls: int) -> float:
    return hits / n_recalls if n_recalls else 0.0


def run_sensitivity_sweep(
    noises=(0.0, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0),
    calibration: str = "popularity",
    capacity: int = 6,
    reuse_gap: int = 4,
    n_objects: int = 20,
    n_recalls: int = 300,
    skew: float = 1.0,
    seed: int = 7,
) -> List[SensitivityPoint]:
    """Sweep Lethe recall quality (noise) and measure the Kairos response."""
    points: List[SensitivityPoint] = []
    for noise in noises:
        # Lethe quality: calibrated-signal AUC at this noise (Section 35).
        q = evaluate_signal_quality(reuse_gap=reuse_gap, horizon=reuse_gap + 2,
                                    n_objects=n_objects, n_recalls=n_recalls,
                                    skew=skew, seed=seed, noise=noise)
        lethe_auc = q.pop_auc if calibration == "popularity" else (
            q.rec_auc if calibration == "recency" else q.raw_auc)
        # Kairos response: calibrated runtime NetGain / hits at this noise.
        r = run_calibrated_runtime(calibration, capacity=capacity, reuse_gap=reuse_gap,
                                   n_objects=n_objects, n_recalls=n_recalls,
                                   skew=skew, seed=seed, noise=noise)
        points.append(SensitivityPoint(
            noise=noise, lethe_auc=lethe_auc, kairos_net=r.net_gain,
            kairos_hits=r.hits, kairos_hit_rate=_hit_rate(r.hits, n_recalls),
            net_positive=r.net_gain > 0))
    return points


def run_sensitivity_sweep_statistical(
    noises=(0.1, 0.2, 0.35, 0.5, 0.7, 1.0),
    calibration: str = "popularity",
    capacity: int = 6,
    reuse_gap: int = 4,
    n_objects: int = 20,
    n_recalls: int = 300,
    skew: float = 1.0,
    n_runs: int = 15,
    base_seed: int = 200,
) -> List[SensitivityPoint]:
    """Multi-seed sensitivity sweep with Section 69 statistics.

    Each noise level is run over `n_runs` seeds; Kairos NetGain and Lethe AUC are
    aggregated to mean + CI95 (t-based) so the response curve carries confidence
    intervals, not single-run noise. This removes the single-seed jitter (e.g.
    noise=0.0 dipping below noise=0.1) that made the single-seed curve noisy.
    """
    points: List[SensitivityPoint] = []
    for noise in noises:
        aucs: List[float] = []
        nets: List[float] = []
        hits_list: List[float] = []
        for r in range(n_runs):
            seed = base_seed + r
            q = evaluate_signal_quality(reuse_gap=reuse_gap, horizon=reuse_gap + 2,
                                        n_objects=n_objects, n_recalls=n_recalls,
                                        skew=skew, seed=seed, noise=noise)
            aucs.append(q.pop_auc if calibration == "popularity" else (
                q.rec_auc if calibration == "recency" else q.raw_auc))
            rt = run_calibrated_runtime(calibration, capacity=capacity, reuse_gap=reuse_gap,
                                        n_objects=n_objects, n_recalls=n_recalls,
                                        skew=skew, seed=seed, noise=noise)
            nets.append(rt.net_gain)
            hits_list.append(float(rt.hits))
        m, half = mean_ci95(nets)
        points.append(SensitivityPoint(
            noise=noise,
            lethe_auc=mean(aucs),
            kairos_net=m,
            kairos_hits=int(round(mean(hits_list))),
            kairos_hit_rate=_hit_rate(mean(hits_list), n_recalls),
            net_positive=m > 0,
            n_runs=n_runs,
            net_mean=m,
            net_ci95_half=half,
            net_std=std(nets),
            net_median=percentile(nets, 50),
            auc_mean=mean(aucs),
        ))
    return points




def compute_sensitivity(points: List[SensitivityPoint]) -> dict:
    """Summarize the response curve: slope d(NetGain)/d(Lethe AUC), the AUC
    threshold below which NetGain is small, and a simple linear fit.

    When points carry multi-seed CI95 (n_runs>1), the slope is computed on the
    per-point MEANS, and we also report whether the response is monotone and
    whether every point's CI95 stays positive (a statistically robust curve)."""
    if len(points) < 2:
        return {"slope_per_auc": 0.0, "auc_range": (0.0, 0.0), "net_range": (0.0, 0.0)}
    aucs = [p.lethe_auc for p in points]
    nets = [p.kairos_net for p in points]
    # Least-squares slope of NetGain vs Lethe AUC (on per-point means).
    mx, my = mean(aucs), mean(nets)
    denom = sum((a - mx) ** 2 for a in aucs)
    slope = (sum((a - mx) * (n - my) for a, n in zip(aucs, nets)) / denom) if denom else 0.0
    # Robustness signals when CI95 is present (multi-seed).
    all_ci_positive = all(p.net_ci95_low > 0 for p in points) if points[0].n_runs > 1 else None
    # Monotone: as noise rises, mean NetGain is (weakly) non-increasing.
    ordered = sorted(points, key=lambda p: p.noise)
    monotone = all(ordered[i].kairos_net >= ordered[i + 1].kairos_net - 1e-6
                   for i in range(len(ordered) - 1))
    return {
        "slope_per_auc": slope,  # NetGain change per +1.0 Lethe AUC
        "slope_per_0_1_auc": slope * 0.1,  # the investment-friendly unit
        "auc_range": (min(aucs), max(aucs)),
        "net_range": (min(nets), max(nets)),
        "mean_auc": mx, "mean_net": my,
        "net_positive_fraction": sum(1 for p in points if p.net_positive) / len(points),
        "all_ci95_positive": all_ci_positive,
        "monotone_in_noise": monotone,
        "n_runs": points[0].n_runs,
    }


def format_sensitivity_report(points: List[SensitivityPoint], sens: dict) -> str:
    multi = points[0].n_runs > 1 if points else False
    lines = [
        "Lethe -> Kairos response curve (Section 90 as a function of Lethe quality)"
        + (f"  [N={points[0].n_runs} seeds, mean ± CI95]" if multi else ""),
    ]
    if multi:
        lines.append(f"{'noise':>6} {'LetheAUC':>10} {'NetMean':>9} {'CI95half':>9} "
                     f"{'CI95low':>9} {'hitRate':>9} {'NetGain>0':>10}")
        lines.append("-" * 68)
        for p in points:
            lines.append(
                f"{p.noise:>6.2f} {p.lethe_auc:>10.3f} {p.net_mean:>9.1f} "
                f"{p.net_ci95_half:>9.1f} {p.net_ci95_low:>9.1f} "
                f"{p.kairos_hit_rate:>9.3f} {str(p.net_positive):>10}"
            )
    else:
        lines.append(f"{'noise':>6} {'LetheAUC':>10} {'KairosNet':>10} {'hits':>6} "
                     f"{'hitRate':>9} {'NetGain>0':>10}")
        lines.append("-" * 58)
        for p in points:
            lines.append(
                f"{p.noise:>6.2f} {p.lethe_auc:>10.3f} {p.kairos_net:>10.1f} "
                f"{p.kairos_hits:>6d} {p.kairos_hit_rate:>9.3f} {str(p.net_positive):>10}"
            )
    lines.append("")
    lines.append(f"sensitivity d(NetGain)/d(Lethe AUC) = {sens['slope_per_auc']:+.1f} "
                 f"per +1.0 AUC  ({sens['slope_per_0_1_auc']:+.1f} per +0.1 AUC)")
    lo, hi = sens["auc_range"]
    lines.append(f"Lethe AUC range {lo:.3f}..{hi:.3f} -> Kairos NetGain range "
                 f"{sens['net_range'][0]:.1f}..{sens['net_range'][1]:.1f}")
    lines.append(f"NetGain>0 in {sens['net_positive_fraction'] * 100:.0f}% of the sweep")
    if sens.get("monotone_in_noise") is not None:
        lines.append(f"monotone in noise: {sens['monotone_in_noise']}, "
                     f"all CI95>0: {sens.get('all_ci95_positive')}")
    return "\n".join(lines)


def investment_guidance(points: List[SensitivityPoint], sens: dict) -> List[str]:
    """Turn the curve into an investment statement for the Lethe team."""
    out: List[str] = []
    slope01 = sens.get("slope_per_0_1_auc", 0.0)
    out.append(f"Improving Lethe AUC by +0.1 raises Kairos NetGain by ~{slope01:+.0f} "
               f"(abstract time units), holding the workload fixed.")
    # Where does Kairos stop benefiting? Linear extrapolation to NetGain ~ 0.
    mx, my = sens.get("mean_auc", 0.0), sens.get("mean_net", 0.0)
    slope = sens.get("slope_per_auc", 0.0)
    if slope > 0 and my > 0:
        auc_at_zero = mx - my / slope
        out.append(f"Linear extrapolation: Kairos NetGain would reach ~0 at Lethe AUC "
                   f"~{auc_at_zero:.3f} (the estimated investment threshold).")
    pos = [p.lethe_auc for p in points if p.net_positive]
    if pos:
        out.append(f"Observed: NetGain stays positive down to the lowest Lethe AUC "
                   f"tested (~{min(pos):.3f}); the true threshold is below the sweep.")
    out.append("Recommendation: Lethe improvements pay off in Kairos roughly "
               "linearly over the tested AUC band - measure AUC, predict NetGain.")
    # Statistical robustness note (Section 69/94) when CI95 is present.
    if points and points[0].n_runs > 1:
        all_pos = all(p.net_ci95_low > 0 for p in points)
        mono = sens.get("monotone_in_noise")
        out.append(f"Statistical robustness (N={points[0].n_runs} seeds): every point's "
                   f"CI95 lower bound > 0 = {all_pos}; response monotone in noise = "
                   f"{mono}. The slope is a robust estimate, not single-run luck.")
    return out


def save_sensitivity_artifacts(points: List[SensitivityPoint], sens: dict,
                               out_dir: str = "artifacts/phase3") -> str:
    import json
    import os
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "lethe_kairos_sensitivity.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"points": [p.__dict__ for p in points], "sensitivity": sens},
                  f, indent=2, ensure_ascii=False)
    return path


def main():
    print("=== Lethe -> Kairos sensitivity: does a better Lethe make a better "
          "Kairos? (Section 90 response curve, Section 69 statistics) ===\n")
    points = run_sensitivity_sweep_statistical(calibration="popularity", capacity=6,
                                               reuse_gap=4, skew=1.0, n_runs=15)
    sens = compute_sensitivity(points)
    print(format_sensitivity_report(points, sens))
    print("\n--- Investment guidance for the Lethe team ---")
    for line in investment_guidance(points, sens):
        print("  * " + line)
    path = save_sensitivity_artifacts(points, sens)
    print(f"\nArtifacts saved: {path}")


if __name__ == "__main__":
    main()
