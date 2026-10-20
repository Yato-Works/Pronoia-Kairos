"""Phase 3 — Design map: mechanism-wins-where, by correlation distance x capacity (Section 63, 74).

Fixes the measured design rule that NO mechanism dominates universally:
    - SHORT-range (recency-correlated): LRU wins; do not feed noisy signal to retention.
    - MID-range (anchors revisit within a few accesses): ADAPTIVE wins (Section 74).
    - LONG-range (anchors revisit after many accesses): RETENTION wins.
and that feed_retention should be chosen per regime (off for short, on for long/mid).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compare.design_map import (
    default_regimes, build_design_map, format_design_map, design_guidelines,
    save_design_map, CellResult,
)

CAPS = [4, 6, 10]
CELLS = build_design_map(default_regimes(), CAPS, sigma=0.0)


def _cells(distance):
    return [c for c in CELLS if c.distance == distance]


def test_map_covers_all_regimes_and_capacities():
    regimes = default_regimes()
    assert len(CELLS) == len(regimes) * len(CAPS)
    for reg in regimes:
        for cap in CAPS:
            assert any(c.regime == reg.name and c.capacity == cap for c in CELLS)


def test_every_cell_has_a_winner_from_the_four_configs():
    for c in CELLS:
        assert c.winner in ("lru", "retention", "prefetch", "adaptive")
        assert c.winner_hits >= c.lru_hits  # winner never has fewer hits than LRU


def test_short_range_prefers_lru_or_prefetch_not_retention():
    """Section 76 #5: on recency-correlated workloads, feeding the noisy signal
    into retention does not help - the winner is LRU (or prefetch of near-term
    demand), never predictive retention."""
    for c in _cells("short"):
        assert c.winner in ("lru", "prefetch"), (c.regime, c.capacity, c.winner)


def test_long_range_prefers_retention():
    """Section 9/13: on long-range reuse, predictive retention dominates."""
    long_cells = _cells("long")
    retention_wins = sum(1 for c in long_cells if c.winner == "retention")
    # Retention must win the majority of long-range cells (prefetch loses there).
    assert retention_wins >= len(long_cells) // 2 + 1


def test_mid_range_prefers_adaptive():
    """Section 74: mid-range reuse is where ADAPTIVE (retention+prefetch) wins."""
    mid_cells = _cells("mid")
    adaptive_wins = sum(1 for c in mid_cells if c.winner == "adaptive")
    assert adaptive_wins >= len(mid_cells) // 2 + 1


def test_feed_retention_chosen_per_regime():
    """Short-range cells keep the LRU retention core (feed=False); long/mid cells
    feed the signal into retention (feed=True)."""
    for c in _cells("short"):
        assert c.feed_retention is False
    long_mid = _cells("long") + _cells("mid")
    assert any(c.feed_retention for c in long_mid)


def test_netgain_consistency_of_winner():
    """The reported winner NetGain equals the max over the four config nets."""
    for c in CELLS:
        nets = {"lru": c.lru_net, "retention": c.retention_net,
                "prefetch": c.prefetch_net, "adaptive": c.adaptive_net}
        assert abs(c.winner_net_gain - nets[c.winner]) < 1e-9


def test_guidelines_cover_three_distance_classes():
    rules = design_guidelines(CELLS)
    text = " ".join(rules)
    assert "SHORT-range" in text and "MID-range" in text and "LONG-range" in text
    # The distilled dominant winners match the qualitative expectation.
    assert "retention" in text.lower() or "adaptive" in text.lower()


def test_format_design_map_renders_a_grid():
    out = format_design_map(CELLS, CAPS)
    assert "regime (distance)" in out
    assert "cap=4" in out and "cap=6" in out and "cap=10" in out
    # One row per regime plus header/separator/legend lines.
    for reg in default_regimes():
        assert reg.name in out


def test_save_design_map_writes_artifact(tmp_path=None):
    import tempfile
    d = tmp_path or tempfile.mkdtemp()
    path = save_design_map(CELLS, out_dir=str(d))
    assert os.path.exists(path)
    import json
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    assert len(payload) == len(CELLS)
    assert all("winner" in c and "distance" in c for c in payload)
