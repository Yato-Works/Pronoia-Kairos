"""Phase 3 — Statistical protocol for calibrated runtime (Plan.md Section 69, 94).

Section 69 forbids calling a win from a single run: warmup + N repeated runs,
record median/p95/p99/CI, and never trust one lucky seed. These tests fix the
statistical machinery (pure-Python, no scipy) and the robustness verdict:

    - descriptive stats (mean/std/percentile/CI95) are correct
    - the paired sign test detects a real advantage and is not fooled by noise
    - over N seeds, popularity calibration is STATISTICALLY SOUND vs raw
      (CI95 excludes raw, sign test significant) - the Section 94 Tier 0 bar
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compare.statistical_protocol import (
    mean, std, percentile, mean_ci95, paired_sign_test, CalibrationStats,
    run_statistical_validation, robustness_summary, format_stats_report,
)


# --- Descriptive statistics ---------------------------------------------------

def test_mean_and_std():
    xs = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    assert abs(mean(xs) - 5.0) < 1e-9
    # population std is 2.0; sample std (n-1) is ~2.138
    assert abs(std(xs) - 2.13809) < 1e-3


def test_percentiles():
    xs = list(range(1, 101))  # 1..100
    # nearest-rank: idx = round(p/100 * (n-1)); value = sorted[idx]
    assert percentile(xs, 50) == 51   # idx=round(49.5)=50 -> xs[50]=51
    assert percentile(xs, 95) == 95   # idx=round(94.05)=94 -> xs[94]=95
    assert percentile(xs, 99) == 99   # idx=round(98.01)=98 -> xs[98]=99
    assert percentile([], 50) == 0.0


def test_mean_ci95_width_shrinks_with_n():
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    _, half5 = mean_ci95(xs)
    xs10 = xs + [1.5, 2.5, 3.5, 4.5, 5.5]
    _, half10 = mean_ci95(xs10)
    assert half10 < half5  # more samples -> tighter CI
    assert mean_ci95([1.0]) == (1.0, 0.0)  # n<2 -> zero width


# --- Paired sign test ---------------------------------------------------------

def test_sign_test_detects_real_advantage():
    diffs = [5.0] * 18 + [-1.0] * 2  # calibration better in 18/20
    r = paired_sign_test(diffs)
    assert r["wins"] == 18 and r["losses"] == 2 and r["n"] == 20
    assert r["p_value"] < 0.05  # significant advantage


def test_sign_test_not_significant_when_balanced():
    diffs = [1.0] * 10 + [-1.0] * 10  # 50/50
    r = paired_sign_test(diffs)
    assert r["wins"] == 10 and r["losses"] == 10
    assert r["p_value"] > 0.05  # not significant


def test_sign_test_handles_ties():
    diffs = [0.0] * 20
    r = paired_sign_test(diffs)
    assert r["ties"] == 20 and r["n"] == 0 and r["p_value"] == 1.0


def test_calibration_stats_finalize():
    st = CalibrationStats("x", 3, [1.0, 2.0, 3.0], [10.0, 20.0, 30.0]).finalize()
    assert st.mean_net == 2.0 and st.median_net == 2.0
    assert st.mean_hits == 20.0
    assert st.ci95_low <= st.mean_net <= st.ci95_high
    assert st.ci95_half > 0


# --- Robustness (Section 94 Tier 0) ------------------------------------------

def test_popularity_is_statistically_sound_vs_raw():
    """The headline robustness result: over N seeds, popularity calibration's
    NetGain CI95 excludes raw's mean and the sign test is significant."""
    stats, sign_tests = run_statistical_validation(n_runs=20, capacity=6, reuse_gap=4, skew=1.0)
    pop = stats["popularity"]
    raw = stats["raw"]
    # CI95 excludes raw mean (improvement is bounded away from 0)
    assert pop.ci95_low > raw.mean_net
    # sign test: popularity wins in (essentially) all seeds
    assert sign_tests["popularity"]["wins"] >= 18
    assert sign_tests["popularity"]["p_value"] < 0.05


def test_robustness_summary_reports_sound():
    stats, sign_tests = run_statistical_validation(n_runs=20, capacity=6, reuse_gap=4, skew=1.0)
    summary = robustness_summary(stats, sign_tests)
    text = "\n".join(summary)
    assert "popularity" in text and "STATISTICALLY SOUND = True" in text
    # raw is never in the robustness comparison (it is the reference)
    assert not any(line.startswith("raw:") for line in summary)


def test_report_renders_all_calibrations():
    stats, _ = run_statistical_validation(n_runs=8, capacity=6, reuse_gap=4, skew=1.0)
    out = format_stats_report(stats)
    for col in ("meanNet", "std", "CI95low", "CI95high", "median", "p95", "p99"):
        assert col in out
    assert "raw" in out and "recency" in out and "popularity" in out
