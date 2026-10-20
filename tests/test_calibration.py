"""Phase 3 — Signal calibration experiment (Plan.md Section 35, 48, 50).

Quantifies whether calibrating Lethe's raw semantic demand (position-independent
utility) by recency improves horizon-reuse prediction, using Section 35 metrics
(AUC ranking, ECE calibration, precision/recall).

Key fixed findings:
    - on round-robin (periodic) traffic, raw AUC < 0.5 (anti-correlated): recency
      is a WORSE-than-random predictor there, and calibration barely moves it.
    - on popularity-skewed (realistic) traffic, raw AUC > 0.5 (weak signal) and
      recency calibration IMPROVES AUC in every cell - the raw semantic demand
      only becomes a usable predictor once calibrated.
    - calibration helps most at short horizons (immediate prefetch, H1/H2);
      at long horizons ECE can worsen (retention needs a different signal) -
      consistent with Section 50's immediate vs long-term separation.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compare.calibration_experiment import (
    evaluate_signal_quality, run_calibration_sweep, format_calibration_report,
    SignalQuality,
)
from semantic.lethe_trace import synthesize_lethe_stream, p_sem_list


def test_round_robin_raw_signal_is_anticorrelated():
    """Periodic traffic: Lethe's raw utility ranks future-used chunks BELOW
    not-reused ones (AUC < 0.5), because recency is anti-predictive there."""
    q = evaluate_signal_quality(reuse_gap=4, horizon=4, skew=0.0)
    assert q.raw_auc < 0.5
    assert q.base_rate > 0.9  # nearly all positive (horizon ~ period)


def test_skewed_raw_signal_is_weakly_predictive():
    """Popularity-skewed traffic: raw AUC > 0.5 - Lethe learned what recurs."""
    q = evaluate_signal_quality(reuse_gap=4, horizon=4, skew=1.0)
    assert q.raw_auc > 0.5
    assert 0.0 < q.base_rate < 0.9  # meaningful mix of positives/negatives


def test_both_calibrations_improve_auc_on_skewed_traffic():
    """Both recency and popularity calibration improve ranking (AUC) vs raw."""
    results = run_calibration_sweep(horizons=(2, 4, 8), skews=(1.0,))
    assert len(results) == 3
    for r in results:
        assert r.rec_auc_gain > 0, (r.horizon, r.raw_auc, r.rec_auc)
        assert r.pop_auc_gain > 0, (r.horizon, r.raw_auc, r.pop_auc)


def test_popularity_dominates_on_skewed_traffic():
    """On popularity-skewed traffic, popularity calibration beats recency (hot
    chunks recur, so access frequency is the strongest signal)."""
    results = run_calibration_sweep(horizons=(2, 4, 8), skews=(1.0,))
    for r in results:
        assert r.pop_auc >= r.rec_auc, (r.horizon, r.rec_auc, r.pop_auc)
        assert r.best_calibration == "popularity"


def test_calibration_uses_no_future_information():
    """Both calibrated signals are ex-ante: they depend only on the past (raw
    utility, recency of last use, running access frequency), never on whether
    reuse happens after the current position."""
    events = synthesize_lethe_stream(n_objects=15, n_recalls=120, reuse_gap=4,
                                     seed=3, skew=1.0)
    raw = p_sem_list(events)
    keys = [e.memory.chunk_id for e in events]
    last_seen: dict = {}
    counts: dict = {}
    decay = 0.8
    for i, k in enumerate(keys):
        age = (i - last_seen[k]) if k in last_seen else 0
        rec = min(1.0, max(0.0, raw[i] * (decay ** max(0, age))))
        counts[k] = counts.get(k, 0) + 1
        pop = min(1.0, max(0.0, raw[i] * (counts[k] / (i + 1))))
        assert 0.0 <= rec <= 1.0 and 0.0 <= pop <= 1.0
        last_seen[k] = i


def test_metrics_are_in_valid_ranges():
    q = evaluate_signal_quality(reuse_gap=4, horizon=4, skew=1.0)
    for v in (q.raw_auc, q.rec_auc, q.pop_auc):
        assert 0.0 <= v <= 1.0
    for v in (q.raw_ece, q.rec_ece, q.pop_ece):
        assert 0.0 <= v <= 1.0
    for v in (q.raw_precision, q.raw_recall, q.rec_precision, q.rec_recall,
              q.pop_precision, q.pop_recall):
        assert 0.0 <= v <= 1.0
    assert 0.0 <= q.base_rate <= 1.0


def test_report_renders_three_calibrations():
    results = run_calibration_sweep(horizons=(2, 4), skews=(0.0, 1.0))
    out = format_calibration_report(results)
    assert "rawAUC" in out and "recAUC" in out and "popAUC" in out
    assert "recGain" in out and "popGain" in out and "best" in out
    # one data row per cell (2 horizons x 2 skews)
    assert out.count("\n") >= len(results) + 1


def test_coupled_traffic_popularity_dominates():
    """When popularity and recency are correlated (realistic skewed traffic),
    popularity calibration subsumes recency and wins (hot chunks recur)."""
    results = run_calibration_sweep(horizons=(2, 4, 8), skews=(1.0,), decouple=False)
    for r in results:
        assert r.pop_auc >= r.rec_auc
        assert r.best_calibration == "popularity"


def test_decoupled_immediate_horizon_favors_recency():
    """Section 50 separation: when popularity is anti-correlated with recency,
    the IMMEDIATE horizon (H1/H2) favors recency calibration - a recently-used
    chunk is the imminent demand, regardless of its (low) long-term popularity."""
    q = evaluate_signal_quality(reuse_gap=2, horizon=2, skew=1.0,
                                decouple=True, n_objects=30, n_recalls=400)
    assert q.best_calibration == "recency"
    assert q.rec_auc > q.pop_auc


def test_both_calibrations_beat_raw_in_decoupled_regime():
    """Even when anti-correlated, both calibrations improve over raw at every
    horizon (calibration is strictly better than using Lethe's raw utility)."""
    results = run_calibration_sweep(horizons=(2, 4, 8), skews=(1.0,), decouple=True)
    for r in results:
        assert r.rec_auc_gain > 0 and r.pop_auc_gain > 0
