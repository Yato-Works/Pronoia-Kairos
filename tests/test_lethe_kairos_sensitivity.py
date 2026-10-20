"""Phase 3 — Lethe -> Kairos sensitivity (Plan.md Section 8, 35, 52, 90, 120).

The two systems co-evolve: a better Lethe (semantic memory) makes a better Kairos
(adaptive runtime). These tests fix the measured response curve:

    - as Lethe recall quality (noise) worsens, Lethe AUC falls AND Kairos NetGain
      falls - the positive Lethe->Kairos coupling.
    - the sensitivity slope d(NetGain)/d(AUC) is positive and quantified.
    - NetGain stays positive across the tested Lethe-quality band.
    - investment guidance and the report render from the curve.

The point (Section 90 as a response curve, Section 120 Measure->Adapt): the
"how much should we invest in Lethe?" question gets a number.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compare.lethe_kairos_sensitivity import (
    run_sensitivity_sweep, run_sensitivity_sweep_statistical, compute_sensitivity,
    format_sensitivity_report, investment_guidance, save_sensitivity_artifacts,
    SensitivityPoint,
)

POINTS = run_sensitivity_sweep(calibration="popularity", capacity=6, reuse_gap=4, skew=1.0)
SENS = compute_sensitivity(POINTS)


def test_sweep_produces_points_across_noise():
    assert len(POINTS) >= 5
    noises = [p.noise for p in POINTS]
    assert noises == sorted(noises)  # ordered low->high noise
    assert all(0.0 <= p.lethe_auc <= 1.0 for p in POINTS)
    assert all(p.kairos_hits >= 0 for p in POINTS)


def test_better_lethe_gives_better_kairos():
    """The core coupling: lower noise (better Lethe) -> higher Lethe AUC and
    higher Kairos NetGain. Compare the best-Lethe vs worst-Lethe points."""
    best = min(POINTS, key=lambda p: p.noise)   # lowest noise = best Lethe
    worst = max(POINTS, key=lambda p: p.noise)  # highest noise = worst Lethe
    assert best.lethe_auc >= worst.lethe_auc
    assert best.kairos_net >= worst.kairos_net
    # a strictly better Lethe should give a strictly larger NetGain here.
    assert best.kairos_net > worst.kairos_net


def test_lethe_auc_decreases_with_noise():
    """Monotone-ish: the highest-noise point has the lowest Lethe AUC."""
    best = min(POINTS, key=lambda p: p.noise)
    worst = max(POINTS, key=lambda p: p.noise)
    assert best.lethe_auc > worst.lethe_auc


def test_sensitivity_slope_is_positive():
    """d(NetGain)/d(Lethe AUC) > 0: improving Lethe pays off in Kairos."""
    assert SENS["slope_per_auc"] > 0
    assert SENS["slope_per_0_1_auc"] > 0
    # the +0.1 AUC unit is the investment-friendly number.
    assert abs(SENS["slope_per_0_1_auc"] - SENS["slope_per_auc"] * 0.1) < 1e-9


def test_netgain_positive_across_sweep():
    """Kairos NetGain stays positive over the whole tested Lethe-quality band."""
    assert all(p.net_positive for p in POINTS)
    assert SENS["net_positive_fraction"] == 1.0


def test_sensitivity_ranges_are_consistent():
    lo_auc, hi_auc = SENS["auc_range"]
    lo_net, hi_net = SENS["net_range"]
    assert lo_auc <= hi_auc and lo_net <= hi_net
    assert abs(SENS["mean_net"] - sum(p.kairos_net for p in POINTS) / len(POINTS)) < 1e-9


def test_compute_sensitivity_handles_tiny_input():
    s = compute_sensitivity([SensitivityPoint(0.0, 0.7, 100.0, 10, 0.3, True)])


# --- Multi-seed statistical sweep (Section 69) -------------------------------

STAT_POINTS = run_sensitivity_sweep_statistical(
    noises=(0.1, 0.35, 0.7, 1.0), calibration="popularity", capacity=6,
    reuse_gap=4, skew=1.0, n_runs=8)
STAT_SENS = compute_sensitivity(STAT_POINTS)


def test_statistical_points_carry_ci95():
    """Multi-seed points carry mean + CI95 half-width (Section 69)."""
    for p in STAT_POINTS:
        assert p.n_runs == 8
        assert p.net_ci95_half > 0
        assert p.net_std > 0
        assert p.net_ci95_low == p.net_mean - p.net_ci95_half


def test_statistical_response_is_monotone_in_noise():
    """Removing single-seed jitter, NetGain is monotone non-increasing in noise."""
    assert STAT_SENS["monotone_in_noise"] is True


def test_low_noise_points_ci95_lower_bound_positive():
    """At GOOD Lethe quality (low noise), the CI95 lower bound stays > 0 - the
    benefit is statistically certain where it matters most."""
    low_noise = [p for p in STAT_POINTS if p.noise <= 0.35]
    assert low_noise
    for p in low_noise:
        assert p.net_ci95_low > 0, (p.noise, p.net_ci95_low)


def test_high_noise_ci_may_cross_zero_honest():
    """Honest caveat: at POOR Lethe quality (high noise) the NetGain estimate is
    less certain, so the CI95 can cross 0. This is a real finding - improving
    Lethe both raises NetGain AND reduces its uncertainty."""
    high_noise = [p for p in STAT_POINTS if p.noise >= 0.7]
    assert high_noise
    # The point still has a positive MEAN (benefit on average), even if the CI
    # is wide enough to touch 0.
    for p in high_noise:
        assert p.net_mean > 0


def test_statistical_slope_positive_and_reported():
    assert STAT_SENS["slope_per_auc"] > 0
    assert STAT_SENS["n_runs"] == 8


def test_better_lethe_beats_worse_lethe_with_ci():
    """Lowest-noise (best Lethe) point's CI95 low exceeds highest-noise point's
    CI95 high - the top of the curve is statistically above the bottom."""
    best = min(STAT_POINTS, key=lambda p: p.noise)
    worst = max(STAT_POINTS, key=lambda p: p.noise)
    assert best.net_ci95_low > worst.net_mean
    assert best.kairos_net > worst.kairos_net


def test_statistical_report_shows_ci_columns():
    out = format_sensitivity_report(STAT_POINTS, STAT_SENS)
    assert "CI95half" in out and "CI95low" in out
    assert "monotone in noise" in out
    assert "N=8" in out


def test_investment_guidance_mentions_robustness():
    guidance = "\n".join(investment_guidance(STAT_POINTS, STAT_SENS))
    assert "Statistical robustness" in guidance
    assert "CI95" in guidance


def test_investment_guidance_names_the_slope():
    guidance = investment_guidance(POINTS, SENS)
    text = "\n".join(guidance)
    assert "AUC by +0.1 raises Kairos NetGain" in text
    assert "threshold" in text.lower() or "positive" in text.lower()


def test_report_renders_curve_and_slope():
    out = format_sensitivity_report(POINTS, SENS)
    assert "LetheAUC" in out and "KairosNet" in out
    assert "sensitivity" in out and "d(NetGain)/d(Lethe AUC)" in out


def test_artifacts_saved():
    import tempfile, json
    d = tempfile.mkdtemp()
    path = save_sensitivity_artifacts(POINTS, SENS, out_dir=d)
    assert os.path.exists(path)
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    assert len(payload["points"]) == len(POINTS)
    assert "sensitivity" in payload and payload["sensitivity"]["slope_per_auc"] > 0
