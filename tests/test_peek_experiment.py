"""Phase 2 §17 — PEEK vs Adaptive Runtime win-condition experiment (Section 17, 18, 62).

Verifies:
  §17 The Adaptive Runtime (semantic-aware retention) beats the strongest
      queue-aware PEEK baseline under an explicit, reported condition.
  §62 The PEEK baseline is tuned to its best lookahead before comparison
      (a weak baseline would not be a fair research result).
  §50 The win requires the semantic horizon to match the reuse distance:
      W too small -> cannot see the reuse; W too large -> over-protects.
  §15 regret vs Belady stays <= 0 for the semantic policy at every cell.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compare.peek_experiment import (
    predictive_hits, best_peek, run_peek_comparison, win_condition,
    format_peek_report, save_artifacts,
)
from compare.baselines import peek_like_hit_count
from compare.belady_lru import lru_hit_count, belady_optimal
from simulator import workloads as w


def test_best_peek_is_strong_baseline():
    """§62: best_peek must return the max over the swept lookaheads."""
    seq = w.long_range_reuse()
    best_hits, best_la = best_peek(seq, 4, lookaheads=(2, 4, 8, 16))
    per_la = {la: peek_like_hit_count(seq, 4, la) for la in (2, 4, 8, 16)}
    assert best_hits == max(per_la.values())
    assert best_la == max(per_la, key=lambda k: per_la[k])


def test_lookahead_saturates_for_long_range():
    """PEEK's advantage saturates: bigger lookahead past a point stops helping,
    which is why a semantic horizon can overtake it (Section 17 mechanism)."""
    seq = w.long_range_reuse()
    assert best_peek(seq, 4, (2, 4, 8, 16))[0] == best_peek(seq, 4, (2, 4, 8, 16, 32, 64))[0]


def test_adaptive_beats_peek_on_long_range():
    """§17 core result: on long-horizon reuse the semantic policy beats best PEEK."""
    seq = w.long_range_reuse()
    peek_hits, _ = best_peek(seq, 4, (2, 4, 8, 16))
    ph = predictive_hits(seq, capacity=4, window=16, sigma=0.0)
    assert ph > peek_hits


def test_win_condition_reports_horizon_and_quality():
    """§17: the reported condition must name horizon-exceeds-lookahead + accuracy."""
    seq = w.long_range_reuse()
    rows = run_peek_comparison(seq, "longrange", capacity=4,
                               sem_windows=(16,), sigmas=(0.0, 0.5))
    summary = win_condition(rows)
    assert summary["beats_peek_anywhere"] is True
    assert "horizon" in summary["interpretation"].lower()
    assert "accurate" in summary["interpretation"].lower()


def test_horizon_must_match_reuse_distance():
    """§50: W too small cannot see reuse; W too large over-protects and decays.
    The winning W band sits between (peek_lookahead, reuse_distance-ish)."""
    seq = w.long_range_reuse()
    peek_hits, _ = best_peek(seq, 4, (2, 4, 8, 16))
    small = predictive_hits(seq, 4, window=4, sigma=0.0)     # too small
    good = predictive_hits(seq, 4, window=16, sigma=0.0)      # matched
    huge = predictive_hits(seq, 4, window=64, sigma=0.0)      # too large
    assert small < good
    assert huge < good
    assert good > peek_hits


def test_regret_non_positive_everywhere():
    """§15: semantic policy must never exceed Belady at any (W, sigma) cell."""
    seq = w.long_range_reuse()
    belady = belady_optimal(seq, 4)
    rows = run_peek_comparison(seq, "longrange", capacity=4,
                               sem_windows=(4, 8, 16, 32), sigmas=(0.0, 0.5, 1.0))
    for r in rows:
        assert r.regret_vs_belady <= 0
        assert r.predictive_hits <= belady


def test_queue_dominant_on_short_range_workloads():
    """Honest negative result: on short-horizon reuse the pending queue is
    already a sufficient signal, so semantic prediction does not beat best PEEK.
    This bounds the §17 claim (it is not universal)."""
    seq = w.zipf_like(30, 400)
    peek_hits, _ = best_peek(seq, 4, (2, 4, 8, 16))
    best_pred = max(predictive_hits(seq, 4, W, 0.0) for W in (4, 8, 16))
    assert best_pred <= peek_hits


def test_margin_and_beats_flag_consistent():
    seq = w.long_range_reuse()
    for r in run_peek_comparison(seq, "lr", capacity=4, sem_windows=(8, 16), sigmas=(0.0,)):
        assert r.margin_over_peek == r.predictive_hits - r.peek_hits
        assert r.beats_peek == (r.margin_over_peek > 0)
        assert r.horizon_exceeds_peek == (r.sem_window > r.peek_lookahead)


def test_report_and_artifacts(tmp_path):
    seq = w.long_range_reuse()
    rows = run_peek_comparison(seq, "lr", capacity=4, sem_windows=(16,), sigmas=(0.0,))
    report = format_peek_report(rows)
    assert "W" in report and "peek" in report and "margin" in report
    out_dir = str(tmp_path / "artifacts_test")
    path = save_artifacts(rows, win_condition(rows), out_dir=out_dir)
    assert os.path.exists(path)
    import json
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    assert "win_condition" in payload and "grid" in payload
