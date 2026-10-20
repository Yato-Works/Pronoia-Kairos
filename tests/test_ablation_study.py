"""Phase 3 — Ablation study of the calibrated runtime (Plan.md Section 70, 94).

Section 70's key question: does Semantic Demand REALLY add value, and via which
mechanism? These tests fix the decomposition of the calibrated runtime's NetGain
into retention / prefetch / synergy contributions, per calibration:

    - the (calibration x mechanism) matrix is complete
    - calibrated Semantic Demand beats the no-signal baseline (Section 70 claim)
    - popularity's value comes mainly from RETENTION (protecting hot chunks),
      consistent with Section 50 (popularity = long-term retention signal)
    - raw uncalibrated signal makes prefetch HARMFUL (waste dominates)
    - the decomposition arithmetic is exact (contribs = NetGain differences)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compare.ablation_study import (
    run_ablation, run_ablation_cell, section70_verdict, format_ablation_report,
    MECHANISMS, CALIBRATIONS,
)

DECOMPS = run_ablation(capacity=6, reuse_gap=4, skew=1.0)


def _d(cal):
    return next(d for d in DECOMPS if d.calibration == cal)


def test_ablation_matrix_is_complete():
    for d in DECOMPS:
        assert set(d.cells.keys()) == set(MECHANISMS)
    assert {d.calibration for d in DECOMPS} == set(CALIBRATIONS)


def test_none_mechanism_is_zero_baseline():
    """mechanism='none' is plain LRU: NetGain is exactly 0 (the ablation zero)."""
    for d in DECOMPS:
        assert d.cells["none"].net_gain == 0.0
        assert d.cells["none"].prediction_overhead == 0.0


def test_decomposition_arithmetic_is_exact():
    """retention/prefetch contributions are NetGain differences vs none; synergy
    = both - retention - prefetch + none (interaction term)."""
    for d in DECOMPS:
        none_net = d.cells["none"].net_gain
        ret_net = d.cells["retention"].net_gain
        pre_net = d.cells["prefetch"].net_gain
        both_net = d.cells["both"].net_gain
        assert abs(d.retention_contrib - (ret_net - none_net)) < 1e-9
        assert abs(d.prefetch_contrib - (pre_net - none_net)) < 1e-9
        assert abs(d.synergy - (both_net - ret_net - pre_net + none_net)) < 1e-9


def test_semantic_demand_adds_value_over_no_signal():
    """Section 70: calibrated Semantic Demand's adaptive NetGain exceeds the
    no-signal (mechanism=none) baseline for both calibrations."""
    for cal in ("recency", "popularity"):
        d = _d(cal)
        assert d.both_net > d.cells["none"].net_gain, cal


def test_calibration_beats_raw_signal():
    """The core Section 70 claim: calibrated demand beats the raw uncalibrated
    signal at the full adaptive runtime."""
    raw = _d("raw")
    for cal in ("recency", "popularity"):
        assert _d(cal).both_net > raw.both_net, cal


def test_popularity_value_carried_by_retention():
    """Why popularity wins: its NetGain comes mainly from predictive retention
    (protecting hot chunks), not prefetch - Section 50 long-term signal."""
    d = _d("popularity")
    assert d.retention_contrib > d.prefetch_contrib
    assert d.retention_contrib > 0
    # retention alone captures almost the whole adaptive NetGain.
    assert d.retention_contrib >= 0.9 * d.both_net


def test_raw_prefetch_is_harmful():
    """Uncalibrated signal drives prefetch into waste: NetGain < 0."""
    d = _d("raw")
    assert d.cells["prefetch"].net_gain < 0
    assert d.cells["prefetch"].prefetch_waste_cost > 0
    # popular-ish retention alone is fine; the harm is specifically prefetch.
    assert d.cells["retention"].net_gain > d.cells["prefetch"].net_gain


def test_verdict_names_retention_as_carrier():
    verdict = "\n".join(section70_verdict(DECOMPS))
    assert "Semantic Demand ADDS value = True" in verdict
    assert "calibration beats raw signal = True" in verdict
    # best calibration's value is attributed to a mechanism.
    assert "value carried mainly by retention" in verdict


def test_report_renders_matrix_and_decomposition():
    out = format_ablation_report(DECOMPS)
    assert "mechanism" in out and "retContrib" in out and "synergy" in out
    for cal in CALIBRATIONS:
        assert cal in out
