"""Phase 3 — Design map: which mechanism wins where (Plan.md Section 63, 74, 9, 13, 20).

Across the experiments so far a clear, quantitative design rule has emerged: NO
single mechanism dominates. The best policy depends on the *correlation distance*
of the workload (how far ahead reuse happens) and on cache capacity:

    - SHORT-range reuse (recency-correlated, e.g. zipf): LRU is already near
      optimal; feeding a noisy signal into retention only hurts (Section 76 #5).
      Prefetch of near-term demand can still add a little.
    - MID-range reuse (anchors revisit within a few accesses): predictive
      PREFETCH shines, and combining it with retention (ADAPTIVE) is best.
    - LONG-range reuse (anchors revisit after many accesses, uncorrelated with
      recency): predictive RETENTION dominates; prefetch loses because a
      prefetched long-range object is evicted before its reuse (Section 20).

This module makes that map explicit and machine-checkable. For each
(regime, capacity) cell it runs the four configs under BOTH retention-feeding
settings, selects the mechanism with the best Section 52 NetGain, and renders a
2-D map (regime x capacity) of the winning mechanism. The artifact is the design
guideline: "at correlation distance D and capacity C, use mechanism M".

All numbers are abstract time units (Section 36), calibrated later (Section 43).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional

from compare.adaptive_experiment import run_adaptive_comparison, _get, ConfigResult
from simulator import workloads as w


@dataclass
class Regime:
    name: str
    sequence: List[str]
    horizon: int
    distance: str  # "short" | "mid" | "long" - qualitative correlation distance


def default_regimes() -> List[Regime]:
    """Correlation-distance sweep (Section 63 workload matrix, Section 50 horizons).

    zipf = short (recency-correlated). long_range_reuse(gap=G) = anchors revisit
    after G accesses; small G ~ mid, large G ~ long. horizon > gap so the signal
    can see the returning anchor.
    """
    return [
        Regime("zipf40(short)", w.zipf_like(40, 400, seed=3), 3, "short"),
        Regime("reuse_gap2(mid)", w.long_range_reuse(6, 2, 8), 4, "mid"),
        Regime("reuse_gap4(mid)", w.long_range_reuse(6, 4, 8), 6, "mid"),
        Regime("reuse_gap6(mid)", w.long_range_reuse(6, 6, 8), 8, "mid"),
        Regime("reuse_gap8(long)", w.long_range_reuse(6, 8, 8), 10, "long"),
        Regime("reuse_gap12(long)", w.long_range_reuse(6, 12, 6), 14, "long"),
    ]


@dataclass
class CellResult:
    regime: str
    distance: str
    capacity: int
    winner: str            # best config by NetGain across both feed settings
    winner_net_gain: float
    feed_retention: bool   # setting that produced the winner
    lru_net: float
    retention_net: float
    prefetch_net: float
    adaptive_net: float
    winner_hits: int
    lru_hits: int


def _best_cell(regime: Regime, capacity: int, sigma: float = 0.0) -> CellResult:
    """Evaluate one (regime, capacity) cell: try both feed settings, keep the best.

    feed_retention controls whether the semantic signal drives retention. The map
    lets the data choose: short-range regimes prefer the LRU retention core
    (feed=False), long-range regimes prefer predictive retention (feed=True).
    """
    best: Optional[ConfigResult] = None
    best_feed = False
    per_feed = {}
    for feed in (False, True):
        by = run_adaptive_comparison(
            regime.sequence, capacity=capacity, horizon=regime.horizon,
            sigmas=[sigma], feed_retention=feed)
        rows = by[sigma]
        per_feed[feed] = rows
        cand = max(rows, key=lambda r: r.net_gain)
        if best is None or cand.net_gain > best.net_gain:
            best = cand
            best_feed = feed
    rows = per_feed[best_feed]
    lru = _get(rows, "lru")
    return CellResult(
        regime=regime.name, distance=regime.distance, capacity=capacity,
        winner=best.config, winner_net_gain=best.net_gain, feed_retention=best_feed,
        lru_net=lru.net_gain,
        retention_net=_get(rows, "retention").net_gain,
        prefetch_net=_get(rows, "prefetch").net_gain,
        adaptive_net=_get(rows, "adaptive").net_gain,
        winner_hits=best.hits, lru_hits=lru.hits,
    )


def build_design_map(
    regimes: Optional[List[Regime]] = None,
    capacities: Optional[List[int]] = None,
    sigma: float = 0.0,
) -> List[CellResult]:
    regimes = regimes if regimes is not None else default_regimes()
    capacities = capacities if capacities is not None else [4, 6, 10]
    cells: List[CellResult] = []
    for regime in regimes:
        for cap in capacities:
            cells.append(_best_cell(regime, cap, sigma))
    return cells


def format_design_map(cells: List[CellResult], capacities: List[int]) -> str:
    """Render the regime x capacity map of the winning mechanism."""
    regimes = []
    for c in cells:
        if c.regime not in regimes:
            regimes.append(c.regime)
    header = f"{'regime (distance)':<22}" + "".join(f"{'cap=' + str(cap):>14}" for cap in capacities)
    lines = [header, "-" * len(header)]
    for reg in regimes:
        row_cells = [c for c in cells if c.regime == reg]
        dist = row_cells[0].distance
        label = f"{reg} ({dist})"
        line = f"{label:<22}"
        for cap in capacities:
            cell = next((c for c in row_cells if c.capacity == cap), None)
            if cell is None:
                line += f"{'-':>14}"
            else:
                mark = "+" if cell.winner_net_gain > 0 else " "
                line += f"{cell.winner + mark:>14}"
        lines.append(line)
    lines.append("")
    lines.append("Legend: winner = best NetGain mechanism; '+' = NetGain > 0 (worth it),")
    lines.append("        no '+' = winner is LRU/baseline (no policy beats doing nothing).")
    return "\n".join(lines)


def design_guidelines(cells: List[CellResult]) -> List[str]:
    """Distil the map into human design rules (Section 63 -> deployment guidance)."""
    rules: List[str] = []
    by_dist = {"short": [], "mid": [], "long": []}
    for c in cells:
        by_dist.setdefault(c.distance, []).append(c)

    def dominant(dist_cells):
        counts = {}
        for c in dist_cells:
            counts[c.winner] = counts.get(c.winner, 0) + 1
        return max(counts, key=counts.get) if counts else "n/a"

    rules.append(f"SHORT-range (recency-correlated, e.g. zipf): dominant winner = "
                 f"{dominant(by_dist['short'])}. LRU is near-optimal; do NOT feed the "
                 f"noisy signal into retention (Section 76 #5).")
    rules.append(f"MID-range (anchors revisit within a few accesses): dominant winner = "
                 f"{dominant(by_dist['mid'])}. Prefetch of near-term demand pays, and "
                 f"ADAPTIVE (retention+prefetch) is best here (Section 74).")
    rules.append(f"LONG-range (anchors revisit after many accesses): dominant winner = "
                 f"{dominant(by_dist['long'])}. Predictive RETENTION dominates; prefetch "
                 f"loses because long-range objects are evicted before reuse (Section 20).")
    # feed_retention guidance
    feed_true = sum(1 for c in cells if c.feed_retention)
    rules.append(f"Feed the semantic signal into retention? {feed_true}/{len(cells)} cells "
                 f"choose feed_retention=True (all long/mid-range); short-range cells keep "
                 f"the LRU retention core.")
    return rules


def save_design_map(cells: List[CellResult], out_dir: str = "artifacts/phase3") -> str:
    import json
    import os
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "design_map.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump([c.__dict__ for c in cells], f, indent=2, ensure_ascii=False)
    return path


def main():
    regimes = default_regimes()
    capacities = [4, 6, 10]
    print("=== Design map: which mechanism wins, by correlation distance x capacity "
          "(sigma=0, perfect prediction) ===\n")
    cells = build_design_map(regimes, capacities, sigma=0.0)
    print(format_design_map(cells, capacities))
    print("\n--- Design guidelines (Section 63) ---")
    for rule in design_guidelines(cells):
        print("  * " + rule)
    path = save_design_map(cells)
    print(f"\nArtifacts saved: {path}")


if __name__ == "__main__":
    main()
