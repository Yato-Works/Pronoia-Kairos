"""Phase 3 — Statistical protocol for the calibrated runtime (Plan.md Section 69, 94).

Section 69 is explicit: never call a win from a single run. Every benchmark is
warmup + N repeated runs, and we must record median / p95 / p99 / confidence
interval. This module applies that protocol to the calibrated-runtime result
(Section 90.2): it re-runs each calibration choice over N seeds (and optional
capacity / correlation-distance sweeps), aggregates NetGain into a distribution,
and tests whether a calibration's NetGain is STATISTICALLY better than the raw
uncalibrated runtime - not just better on one lucky seed.

Statistics (no numpy/scipy dependency; pure Python, Section 43 offline):
    mean, std, and a t-based confidence interval on the mean
    percentiles: median (p50), p95, p99
    paired sign test on (calibration_net - raw_net) across seeds: how many seeds
    favor the calibration, and is that a majority beyond chance? (binomial p)

Tier 0 (Section 94 "Statistically sound") = the winning calibration beats raw with
a confidence interval that excludes 0 improvement, across the seed sweep.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import math

from compare.calibrated_runtime_experiment import run_calibrated_runtime, CalibratedRuntimeResult


# --- Descriptive statistics ---------------------------------------------------

def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def std(xs: List[float]) -> float:
    """Sample standard deviation (n-1)."""
    n = len(xs)
    if n < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def percentile(xs: List[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    idx = min(len(s) - 1, max(0, int(round(p / 100.0 * (len(s) - 1)))))
    return s[idx]


# t critical values (two-sided 95%) for small samples; adequate for n<=30 runs.
_T95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365,
        9: 2.306, 10: 2.262, 11: 2.228, 12: 2.201, 13: 2.179, 14: 2.160,
        15: 2.145, 16: 2.131, 17: 2.120, 18: 2.110, 19: 2.101, 20: 2.093,
        21: 2.086, 22: 2.080, 23: 2.074, 24: 2.069, 25: 2.064, 26: 2.060,
        27: 2.056, 28: 2.052, 29: 2.045, 30: 2.042}


def mean_ci95(xs: List[float]) -> tuple:
    """(mean, half_width) of the 95% t-CI on the mean."""
    n = len(xs)
    if n < 2:
        return (mean(xs), 0.0)
    t = _T95.get(n, 1.96)
    half = t * std(xs) / math.sqrt(n)
    return (mean(xs), half)


def paired_sign_test(diffs: List[float]) -> dict:
    """Sign test on paired differences (calibration - raw) across seeds.

    Returns wins/losses/ties and a two-sided binomial p-value for wins vs the
    null that the calibration is no better than raw (p=0.5 per seed, ties drop).
    """
    wins = sum(1 for d in diffs if d > 0)
    losses = sum(1 for d in diffs if d < 0)
    ties = sum(1 for d in diffs if d == 0)
    n = wins + losses
    # Two-sided binomial p for observing >= wins successes out of n at p=0.5.
    if n == 0:
        p = 1.0
    else:
        k = max(wins, losses)
        tail = sum(math.comb(n, i) for i in range(k, n + 1)) / (2 ** n)
        p = min(1.0, 2.0 * tail)
    return {"wins": wins, "losses": losses, "ties": ties, "n": n, "p_value": p}


@dataclass
class CalibrationStats:
    calibration: str
    n_runs: int
    net_gains: List[float]
    hits: List[float]
    mean_net: float = 0.0
    std_net: float = 0.0
    ci95_half: float = 0.0
    median_net: float = 0.0
    p95_net: float = 0.0
    p99_net: float = 0.0
    mean_hits: float = 0.0

    def finalize(self) -> "CalibrationStats":
        m, half = mean_ci95(self.net_gains)
        self.mean_net, self.ci95_half = m, half
        self.std_net = std(self.net_gains)
        self.median_net = percentile(self.net_gains, 50)
        self.p95_net = percentile(self.net_gains, 95)
        self.p99_net = percentile(self.net_gains, 99)
        self.mean_hits = mean(self.hits)
        return self

    @property
    def ci95_low(self) -> float:
        return self.mean_net - self.ci95_half

    @property
    def ci95_high(self) -> float:
        return self.mean_net + self.ci95_half


# --- N repeated runs (Section 69) --------------------------------------------

def run_statistical_validation(
    n_runs: int = 20,
    capacity: int = 6,
    reuse_gap: int = 4,
    n_objects: int = 20,
    n_recalls: int = 300,
    skew: float = 1.0,
    base_seed: int = 100,
    calibrations=("raw", "recency", "popularity"),
) -> "tuple[Dict[str, CalibrationStats], Dict[str, dict]]":
    """Warmup + N repeated runs per calibration; aggregate NetGain distributions.

    Returns (per-calibration CalibrationStats, per-calibration sign-test result vs
    raw). Each run uses a distinct seed so the NetGain sample reflects run-to-run
    (data) variance, exactly the Section 69 protocol.
    """
    # Warmup run (discarded): primes any caches / import side effects.
    for cal in calibrations:
        run_calibrated_runtime(cal, capacity=capacity, reuse_gap=reuse_gap,
                               n_objects=n_objects, n_recalls=n_recalls, skew=skew,
                               seed=base_seed - 1)

    nets: Dict[str, List[float]] = {cal: [] for cal in calibrations}
    hits: Dict[str, List[float]] = {cal: [] for cal in calibrations}
    for r in range(n_runs):
        seed = base_seed + r
        for cal in calibrations:
            res = run_calibrated_runtime(cal, capacity=capacity, reuse_gap=reuse_gap,
                                         n_objects=n_objects, n_recalls=n_recalls,
                                         skew=skew, seed=seed)
            nets[cal].append(res.net_gain)
            hits[cal].append(float(res.hits))

    stats = {cal: CalibrationStats(cal, n_runs, nets[cal], hits[cal]).finalize()
             for cal in calibrations}

    # Paired sign test of each calibration vs raw on the SAME seeds.
    sign_tests: Dict[str, dict] = {}
    raw_nets = nets.get("raw", [])
    for cal in calibrations:
        if cal == "raw":
            continue
        diffs = [c - r for c, r in zip(nets[cal], raw_nets)]
        sign_tests[cal] = paired_sign_test(diffs)
    return stats, sign_tests


def robustness_summary(
    stats: Dict[str, CalibrationStats], sign_tests: Dict[str, dict]
) -> List[str]:
    """Tier 0 (Section 94): is the winning calibration's improvement statistically
    sound? A calibration 'beats raw soundly' if its mean NetGain CI95 excludes the
    raw mean AND the paired sign test favors it (p < 0.05)."""
    raw = stats.get("raw")
    out: List[str] = []
    for cal, st in stats.items():
        if cal == "raw" or raw is None:
            continue
        ci_excludes_raw = st.ci95_low > raw.mean_net
        sig = sign_tests.get(cal, {})
        significant = sig.get("p_value", 1.0) < 0.05 and sig.get("wins", 0) > sig.get("losses", 0)
        sound = ci_excludes_raw and significant
        out.append(
            f"{cal}: mean NetGain {st.mean_net:+.1f} "
            f"[CI95 {st.ci95_low:+.1f}, {st.ci95_high:+.1f}] vs raw {raw.mean_net:+.1f} "
            f"| sign test wins={sig.get('wins', 0)}/{sig.get('n', 0)} p={sig.get('p_value', 1.0):.3f} "
            f"| STATISTICALLY SOUND = {sound}"
        )
    return out


def format_stats_report(stats: Dict[str, CalibrationStats]) -> str:
    lines = [
        f"{'calibration':>12} {'n':>4} {'meanNet':>9} {'std':>8} "
        f"{'CI95low':>9} {'CI95high':>9} {'median':>9} {'p95':>9} {'p99':>9} {'meanHits':>9}"
    ]
    lines.append("-" * 92)
    for cal in ("raw", "recency", "popularity"):
        st = stats.get(cal)
        if st is None:
            continue
        lines.append(
            f"{cal:>12} {st.n_runs:>4} {st.mean_net:>9.1f} {st.std_net:>8.1f} "
            f"{st.ci95_low:>9.1f} {st.ci95_high:>9.1f} {st.median_net:>9.1f} "
            f"{st.p95_net:>9.1f} {st.p99_net:>9.1f} {st.mean_hits:>9.1f}"
        )
    return "\n".join(lines)


def save_statistical_artifacts(
    stats: Dict[str, CalibrationStats], sign_tests: Dict[str, dict],
    out_dir: str = "artifacts/phase3",
) -> str:
    import json
    import os
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "statistical_protocol.json")
    payload = {
        "per_calibration": {
            cal: {
                "n_runs": st.n_runs,
                "net_gains": st.net_gains,
                "mean_net": st.mean_net, "std_net": st.std_net,
                "ci95_low": st.ci95_low, "ci95_high": st.ci95_high,
                "median_net": st.median_net, "p95_net": st.p95_net,
                "p99_net": st.p99_net, "mean_hits": st.mean_hits,
            } for cal, st in stats.items()
        },
        "sign_tests_vs_raw": sign_tests,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


def main():
    print("=== Statistical protocol (Section 69): N repeated runs, "
          "median/p95/p99/CI95, paired sign test ===\n")
    stats, sign_tests = run_statistical_validation(n_runs=20, capacity=6, reuse_gap=4, skew=1.0)
    print(format_stats_report(stats))
    print("\n--- Robustness vs raw (Section 94 Tier 0: statistically sound) ---")
    for line in robustness_summary(stats, sign_tests):
        print("  " + line)
    path = save_statistical_artifacts(stats, sign_tests)
    print(f"\nArtifacts saved: {path}")


if __name__ == "__main__":
    main()
