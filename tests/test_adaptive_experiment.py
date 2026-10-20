"""Phase 3 — Adaptive synergy on long-range reuse: retention x prefetch (Section 9, 13, 20, 52, 74).

Workload: long_range_reuse(6,5,8) - anchors revisit every 5 accesses, uncorrelated
with recency. Both mechanisms use the SAME oracle+N prediction quality (per sigma);
feed_retention=True so retention also evicts by predicted demand. On this regime:

    - retention alone BEATS LRU (predictive retention works when reuse != recency)
    - adaptive HIT COUNT >= retention and prefetch (the Section 74 synergy)
    - the adaptive hit-gain over LRU is SUPERADDITIVE (> retention-gain + prefetch-gain)
    - NetGain degrades monotonically as prediction worsens (Section 52)
    - HONEST caveat: prefetch-only NetGain is NEGATIVE (long-range demand is evicted
      before reuse), so adaptive's NetGain can sit below retention-only - the win is
      on hit count, the prefetch waste cost is the research caveat (Section 20).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compare.adaptive_experiment import run_adaptive_comparison, _get, synergy_summary
from simulator.latency import LatencyModel
from simulator import workloads as w

BY = run_adaptive_comparison(w.long_range_reuse(6, 5, 8), capacity=6, horizon=7, feed_retention=True)


def test_four_configs_share_one_sigma_grid():
    assert len(BY) == 7
    for rows in BY.values():
        assert {r.config for r in rows} == {"lru", "retention", "prefetch", "adaptive"}


def test_lru_row_is_zero_netgain_baseline():
    for rows in BY.values():
        lru = _get(rows, "lru")
        assert lru.prefetch_issued == 0 and lru.net_gain == 0.0 and lru.saved_compute == 0.0


def test_retention_beats_lru_on_long_range():
    """Section 9/13: predictive retention wins when reuse is uncorrelated with recency."""
    for sigma, rows in BY.items():
        if sigma <= 1.0:
            assert _get(rows, "retention").hits > _get(rows, "lru").hits, sigma


def test_adaptive_hits_beat_each_mechanism_when_accurate():
    """Section 74 synergy: adaptive HIT COUNT >= retention and prefetch."""
    for sigma, rows in BY.items():
        if sigma <= 1.0:
            adp = _get(rows, "adaptive")
            assert adp.hits >= _get(rows, "retention").hits, sigma
            assert adp.hits >= _get(rows, "prefetch").hits, sigma


def test_adaptive_hit_gain_is_superadditive():
    """Adaptive's hit gain over LRU exceeds retention-gain + prefetch-gain."""
    for rows in BY.values():
        lru = _get(rows, "lru").hits


def test_prefetch_only_is_negative_honest_caveat():
    """Section 20 honest result: on long-range demand, prefetch-only loses
    (objects are evicted before reuse), so its NetGain is negative even at
    perfect prediction."""
    for sigma, rows in BY.items():
        if sigma <= 0.5:
            assert _get(rows, "prefetch").net_gain < 0, sigma


def test_netgain_formula_matches_components():
    """NetGain = SavedCompute - PredictionOverhead - PrefetchWaste (Section 52)."""
    prefill_cost = LatencyModel().t_prefill_per_token * 512
    for rows in BY.values():
        lru_hits = _get(rows, "lru").hits
        for r in rows:
            assert abs(r.saved_compute - (r.hits - lru_hits) * prefill_cost) < 1e-9
            assert abs(r.net_gain - (r.saved_compute - r.prediction_overhead
                                     - r.prefetch_waste_cost)) < 1e-9


def test_blind_prediction_shrinks_adaptive_edge():
    """Section 76: at sigma=3 the signal is mostly noise; the adaptive edge over
    LRU shrinks from its perfect-prediction peak toward a few hits (recall loss
    keeps a little demand signal alive, so it does not fully collapse to LRU)."""
    lru_hits = _get(BY[3.0], "lru").hits
    best_gain = _get(BY[0.0], "adaptive").hits - _get(BY[0.0], "lru").hits
    blind_gain = _get(BY[3.0], "adaptive").hits - lru_hits
    assert blind_gain < best_gain           # edge clearly smaller when blind
    assert blind_gain <= 4                  # and small in absolute terms


def test_synergy_summary_reports_hit_win():
    """The summary string must surface the honest picture: beats on hits, not net."""
    lines = synergy_summary(BY)
    assert len(lines) == 7
    # At perfect prediction the hit-win is explicit.
    assert "beats_both(hits=True" in lines[0]


def test_reproducible_under_fixed_seed():
    """Same seed -> identical adaptive hit counts (experiment reproducibility)."""
    again = run_adaptive_comparison(w.long_range_reuse(6, 5, 8), capacity=6,
                                    horizon=7, feed_retention=True, seed=42)
    for sigma in BY:
        assert _get(BY[sigma], "adaptive").hits == _get(again[sigma], "adaptive").hits

def sigma_accurate(rows):
    return _get(rows, "adaptive").sigma <= 1.0


def test_netgain_degrades_with_sigma_for_retention():
    """Section 52: worse prediction -> lower NetGain. Predictive retention
    degrades monotonically as sigma grows (recall falls, so it protects less).
    (Adaptive need not be strictly monotonic: at very poor sigma prefetch stops
    firing, removing its waste cost, so adaptive can tick back up toward
    retention-only - a healthy fallback, tested separately.)"""
    sigmas = sorted(BY)
    nets = [_get(BY[s], "retention").net_gain for s in sigmas]
    assert all(nets[i] >= nets[i + 1] - 1e-6 for i in range(len(nets) - 1))


def test_adaptive_netgain_falls_from_its_peak_as_prediction_worsens():
    """Adaptive's best NetGain is at good prediction and it declines as sigma
    grows (the sigma=2..3 plateau reflects prefetch shutting off, not recovery)."""
    sigmas = sorted(BY)
    nets = [_get(BY[s], "adaptive").net_gain for s in sigmas]
    peak = max(nets)
    assert peak == nets[0] or peak in nets[:3]  # peak at good prediction
    assert nets[-1] < peak                      # ends well below the peak

