"""Phase 2/3: predictive retention + signal-quality sweep (Section 9, 13, 51, 52)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cache.predictive_retention import PredictiveRetentionPolicy
from compare.predictive_experiment import (
    oracle_reuse_probability, noisy_signals,
    run_predictive_sweep, format_sweep_report,
)
from simulator import workloads as w


def test_policy_evicts_lowest_decayed_signal():
    p = PredictiveRetentionPolicy(decay=0.5)
    lru = ["a", "b", "c"]
    signals = {"a": 0.9, "b": 0.1, "c": 0.5}
    ages = {"a": 0, "b": 0, "c": 0}
    assert p.choose_victim(lru, signals, ages) == "b"


def test_stale_signal_loses_protection():
    """A once-high prediction must decay so it can be evicted."""
    p = PredictiveRetentionPolicy(decay=0.5)
    lru = ["stale", "fresh"]
    signals = {"stale": 0.9, "fresh": 0.6}
    ages = {"stale": 3, "fresh": 0}   # stale: 0.9*0.125=0.1125 < 0.6
    assert p.choose_victim(lru, signals, ages) == "stale"


def test_unsignaled_key_evicted_first():
    p = PredictiveRetentionPolicy()
    lru = ["known", "unknown"]
    signals = {"known": 0.8}
    ages = {"known": 0}
    assert p.choose_victim(lru, signals, ages) == "unknown"


def test_oracle_signal_is_binary():
    seq = ["A", "B", "A", "C"]
    assert oracle_reuse_probability(seq, 0, window=2) == 1.0   # A reused at dist 2
    assert oracle_reuse_probability(seq, 1, window=2) == 0.0   # B never reused in window
    assert oracle_reuse_probability(seq, 2, window=2) == 0.0   # A last occurrence


def test_noisy_signals_clipped_and_seeded():
    seq = w.zipf_like(10, 50)
    s1 = noisy_signals(seq, 4, 0.3, seed=1)
    s2 = noisy_signals(seq, 4, 0.3, seed=1)
    assert s1 == s2                       # reproducible (Section 100)
    assert all(0.0 <= x <= 1.0 for x in s1)


def test_sweep_regret_always_non_positive():
    """Belady upper bound must hold at every prediction quality (Section 15)."""
    seq = w.zipf_like(20, 200)
    results = run_predictive_sweep(seq, capacity=4, sigmas=[0.0, 0.5, 1.0])
    for r in results:
        assert r.regret_vs_belady <= 0


def test_better_prediction_shrinks_regret_on_zipf():
    """Core thesis: as prediction quality improves, regret vs Belady shrinks."""
    seq = w.zipf_like(30, 400)
    results = {r.sigma: r for r in run_predictive_sweep(
        seq, capacity=4, sigmas=[0.1, 2.0])}
    assert results[0.1].regret_vs_belady > results[2.0].regret_vs_belady
    # and the good band must actually beat LRU
    assert results[0.1].improvement_over_lru > 0


def test_net_gain_sign_tracks_beating_lru():
    seq = w.zipf_like(30, 400)
    for r in run_predictive_sweep(seq, capacity=4, sigmas=[0.1, 2.0]):
        if r.improvement_over_lru > 0:
            assert r.net_gain_positive
        else:
            assert not r.net_gain_positive


def test_metrics_track_quality():
    seq = w.zipf_like(20, 200)
    res = {r.sigma: r for r in run_predictive_sweep(
        seq, capacity=4, sigmas=[0.0, 2.0])}
    assert res[0.0].auc >= res[2.0].auc
    assert res[0.0].ece <= res[2.0].ece


def test_report_formatting():
    seq = w.zipf_like(10, 100)
    report = format_sweep_report(run_predictive_sweep(seq, capacity=3, sigmas=[0.1]))
    assert "sigma" in report and "NetGain" in report
