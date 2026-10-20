"""Phase 3 — Calibrated runtime: connect calibration to retention/prefetch (Section 8, 35, 50, 90).

Fixes the end-to-end Section 90 result: choosing the RIGHT calibration of Lethe's
raw demand turns semantic demand from a net LOSS (raw, uncalibrated) into real
net benefit, because calibrated signals drive predictive retention / prefetch
correctly.

    - raw (uncalibrated) runtime: NetGain < 0 (retention cannot rank, prefetch wastes)
    - recency-calibrated: NetGain improves over raw (immediate prefetch, H1/H2)
    - popularity-calibrated: best NetGain (retention protects hot chunks, H3/H4)

Calibrations are ex-ante (no future info); NetGain = Saved - Overhead - Waste (52).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compare.calibrated_runtime_experiment import (
    run_calibrated_runtime, run_calibration_comparison, format_runtime_report,
)
from semantic.lethe_trace import (
    build_calibrated_trace, calibrated_forecast, recency_signals,
    popularity_signals, p_sem_list, synthesize_lethe_stream,
)

CAP = 6
GAP = 4


def _results():
    return {r.calibration: r for r in run_calibration_comparison(
        capacities=(CAP,), reuse_gaps=(GAP,), skew=1.0)}


def test_raw_uncalibrated_runtime_does_not_pay_off():
    """Section 8/90: Lethe's raw, uncalibrated demand does NOT add net benefit -
    retention cannot rank the (all-high) signal and prefetch wastes transfers."""
    r = run_calibrated_runtime("raw", capacity=CAP, reuse_gap=GAP, skew=1.0)
    assert r.net_gain < 0


def test_calibration_improves_netgain_over_raw():
    """Both calibrations raise NetGain above the raw runtime (calibration pays)."""
    res = _results()
    raw = res["raw"]
    for cal in ("recency", "popularity"):
        assert res[cal].net_gain > raw.net_gain, (cal, res[cal].net_gain, raw.net_gain)


def test_popularity_is_best_calibration_by_netgain():
    """On skewed traffic popularity calibration wins (retention protects hot
    chunks -> most extra hits, zero prefetch waste)."""
    res = _results()
    best = max(res.values(), key=lambda r: r.net_gain)
    assert best.calibration == "popularity"
    assert best.net_gain > 0
    assert best.hits > res["raw"].hits


def test_popularity_retention_adds_hits_zero_waste():
    """Popularity retention converts extra hits with no wasted prefetch."""
    r = run_calibrated_runtime("popularity", capacity=CAP, reuse_gap=GAP, skew=1.0)
    assert r.hits > _lru_hits()
    assert r.prefetch_wasted == 0


def _lru_hits():
    from compare.adaptive_experiment import _run_trace_config
    from simulator.latency import LatencyModel
    events = synthesize_lethe_stream(20, 300, GAP, seed=7, skew=1.0)
    trace = build_calibrated_trace(events, "raw")
    m, _ = _run_trace_config(trace, CAP, LatencyModel(), "lru", {})
    return m.hits


def test_netgain_formula_matches_components():
    """NetGain = SavedCompute - PredictionOverhead - PrefetchWaste (Section 52)."""
    for cal in ("raw", "recency", "popularity"):
        r = run_calibrated_runtime(cal, capacity=CAP, reuse_gap=GAP, skew=1.0)
        expected = r.saved_compute - r.prediction_overhead - r.prefetch_waste_cost
        assert abs(r.net_gain - expected) < 1e-9


def test_calibrated_signals_are_ex_ante():
    """The calibrated signals use only past info (recency of last use, running
    access frequency), never whether reuse happens after the current position."""
    events = synthesize_lethe_stream(15, 120, GAP, seed=3, skew=1.0)
    rec = recency_signals(events)
    pop = popularity_signals(events)
    raw = p_sem_list(events)
    assert len(rec) == len(pop) == len(raw) == len(events)
    assert all(0.0 <= v <= 1.0 for v in rec + pop)
    # Calibration only ever shrinks the raw signal (a weighting in [0,1]).
    assert all(rec[i] <= raw[i] + 1e-9 for i in range(len(raw)))
    assert all(pop[i] <= raw[i] + 1e-9 for i in range(len(raw)))


def test_report_renders_all_calibrations():
    results = run_calibration_comparison(capacities=(CAP,), reuse_gaps=(GAP,), skew=1.0)
    out = format_runtime_report(results)
    assert "raw" in out and "recency" in out and "popularity" in out
    assert "NetGain" in out
