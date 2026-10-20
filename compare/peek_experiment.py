"""Phase 2 §17 — PEEK vs Adaptive Runtime: the win condition.

Plan.md §17 requires showing, as a *research result*, the condition under
which the Adaptive Runtime beats PEEK. §18 defines the differentiator:

    "Semantic-memory-derived future demand, integrated with Queue / Session /
     Cache state and converted into a computational policy."

PEEK (queue-aware) and the Adaptive Runtime (semantic-aware) both try to
retain keys that will be reused soon, but they read *different signals*:

    PEEK     : "is this key in the next `lookahead` pending requests?"
               -> exact, but bounded by the observable queue depth.
    Adaptive : "P(this key is reused within horizon W)" from semantic memory
               -> predicted (noisy), but the horizon can exceed the queue.

So the win condition has two independent parts:
    (H) semantic horizon W  >  PEEK queue lookahead   -> see further
    (Q) prediction quality high enough that noise < the horizon advantage

Per §62 the baseline must be *strong*: we first find PEEK's best lookahead
(the strongest queue-aware config) and only then ask whether the Adaptive
Runtime's retention beats it. We sweep (semantic window W, noise sigma)
against that best PEEK and report the region where Adaptive wins. This is
the §17 artifact, reproducible and GPU-free (§100).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict
import json
import os

from compare.baselines import peek_like_hit_count
from compare.belady_lru import belady_optimal, lru_hit_count
from compare.predictive_experiment import noisy_signals
from cache.predictive_retention import PredictiveRetentionPolicy
from simulator.discrete import DiscreteEventSimulator, TraceRequest
from simulator.latency import LatencyModel


def predictive_hits(
    sequence: List[str],
    capacity: int,
    window: int,
    sigma: float,
    seed: int = 42,
    input_tokens: int = 512,
) -> int:
    """Hit count of the semantic-aware retention policy on `sequence`.

    The per-request future-use signal is the oracle reuse-within-`window`
    label corrupted by Gaussian noise(sigma), exactly as in
    compare.predictive_experiment, so this is the same central thesis
    measured here against a queue-aware baseline.
    """
    probs = noisy_signals(sequence, window, sigma, seed)
    trace = [
        TraceRequest(
            request_id=f"r{i}", key=key, arrival_time=float(i),
            input_tokens=input_tokens, output_tokens=32,
            future_use_probability=probs[i], semantic_confidence=0.95,
        )
        for i, key in enumerate(sequence)
    ]
    sim = DiscreteEventSimulator(
        capacity=capacity, latency=LatencyModel(), planner=None,
        retention_policy=PredictiveRetentionPolicy(),
    )
    return sim.run(trace).hits


def best_peek(
    sequence: List[str], capacity: int, lookaheads: Tuple[int, ...] = (2, 4, 8, 16)
) -> Tuple[int, int]:
    """§62: the baseline must be strong. Return (best_hits, best_lookahead)."""
    best_hits, best_la = -1, lookaheads[0]
    for la in lookaheads:
        h = peek_like_hit_count(sequence, capacity, la)
        if h > best_hits:
            best_hits, best_la = h, la
    return best_hits, best_la


@dataclass
class PeekComparison:
    workload: str
    capacity: int
    peek_lookahead: int      # best queue-aware lookahead (§62 strong baseline)
    sem_window: int          # semantic horizon W
    sigma: float             # prediction noise
    peek_hits: int
    predictive_hits: int
    lru_hits: int
    belady_hits: int
    margin_over_peek: int    # predictive_hits - peek_hits
    beats_peek: bool         # predictive_hits > peek_hits
    horizon_exceeds_peek: bool   # sem_window > peek_lookahead  (part H)
    regret_vs_belady: int    # predictive_hits - belady_hits (<= 0)


def run_peek_comparison(
    sequence: List[str],
    workload: str,
    capacity: int,
    sem_windows: Tuple[int, ...] = (2, 4, 8, 16, 32, 40, 64),
    sigmas: Tuple[float, ...] = (0.0, 0.25, 0.5, 1.0),
    peek_lookaheads: Tuple[int, ...] = (2, 4, 8, 16),
    seed: int = 42,
) -> List[PeekComparison]:
    """Compare semantic-aware retention against the strongest PEEK config."""
    peek_hits, peek_la = best_peek(sequence, capacity, peek_lookaheads)
    lru = lru_hit_count(sequence, capacity)
    belady = belady_optimal(sequence, capacity)
    rows: List[PeekComparison] = []
    for W in sem_windows:
        for sigma in sigmas:
            ph = predictive_hits(sequence, capacity, W, sigma, seed)
            rows.append(PeekComparison(
                workload=workload, capacity=capacity, peek_lookahead=peek_la,
                sem_window=W, sigma=sigma,
                peek_hits=peek_hits, predictive_hits=ph, lru_hits=lru,
                belady_hits=belady, margin_over_peek=ph - peek_hits,
                beats_peek=ph > peek_hits,
                horizon_exceeds_peek=W > peek_la,
                regret_vs_belady=ph - belady,
            ))
    return rows


def win_condition(rows: List[PeekComparison]) -> Dict:
    """§17: summarize *the condition* under which Adaptive beats best PEEK."""
    assert rows, "win_condition needs at least one comparison row"
    wl = rows[0].workload
    peek_best = rows[0].peek_hits
    peek_la = rows[0].peek_lookahead
    beating = [r for r in rows if r.beats_peek]
    max_row = max(rows, key=lambda r: r.margin_over_peek)

    if beating:
        max_sigma = max(r.sigma for r in beating)
        all_have_horizon = all(r.horizon_exceeds_peek for r in beating)
        if all_have_horizon:
            interpretation = (
                f"Adaptive beats best PEEK only when the semantic horizon "
                f"exceeds the queue lookahead (W > {peek_la}) AND prediction "
                f"is accurate enough (sigma <= {max_sigma:g})."
            )
        else:
            interpretation = (
                f"Adaptive beats best PEEK in {len(beating)}/{len(rows)} cells; "
                f"best margin {max_row.margin_over_peek:+d} at "
                f"W={max_row.sem_window}, sigma={max_row.sigma:g}."
            )
    else:
        interpretation = (
            f"Best PEEK (lookahead={peek_la}) is not beaten by semantic "
            f"prediction in the swept (W, sigma) region: for this workload the "
            f"pending queue is already a sufficient future-demand signal."
        )

    return {
        "workload": wl,
        "capacity": rows[0].capacity,
        "peek_best_hits": peek_best,
        "peek_best_lookahead": peek_la,
        "lru_hits": rows[0].lru_hits,
        "belady_hits": rows[0].belady_hits,
        "beating_cells": len(beating),
        "total_cells": len(rows),
        "max_margin": max_row.margin_over_peek,
        "max_margin_at": {"sem_window": max_row.sem_window, "sigma": max_row.sigma},
        "beats_peek_anywhere": bool(beating),
        "interpretation": interpretation,
    }


def format_peek_report(rows: List[PeekComparison]) -> str:
    lines = [
        f"{'W':>3} {'sigma':>6} {'pred':>6} {'peek':>6} {'LRU':>5} {'Belady':>7} "
        f"{'margin':>7} {'H>LA':>5} {'beats':>6}"
    ]
    lines.append("-" * 66)
    for r in rows:
        lines.append(
            f"{r.sem_window:>3d} {r.sigma:>6.2f} {r.predictive_hits:>6d} "
            f"{r.peek_hits:>6d} {r.lru_hits:>5d} {r.belady_hits:>7d} "
            f"{r.margin_over_peek:>+7d} {str(r.horizon_exceeds_peek):>5} "
            f"{str(r.beats_peek):>6}"
        )
    return "\n".join(lines)


def save_artifacts(rows: List[PeekComparison], summary: Dict, out_dir: str = "artifacts/phase2") -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "peek_comparison.json")
    payload = {"win_condition": summary, "grid": [r.__dict__ for r in rows]}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


def main():
    from simulator import workloads as w
    wl = {
        "zipf(30,400)": w.zipf_like(30, 400),
        "agent(60,6)": w.agent_workflow(60, 6),
        "session(6x8)": w.multi_turn_session(6, 8),
        # Long-horizon reuse: the family where semantic demand should beat a
        # short queue lookahead (Section 17 / Section 50 H3-H4).
        "longrange(4x10x6)": w.long_range_reuse(),
    }
    last_rows: List[PeekComparison] = []
    for name, seq in wl.items():
        rows = run_peek_comparison(seq, name, capacity=4)
        summary = win_condition(rows)
        last_rows = rows
        print(f"\n=== {name} (capacity=4, best PEEK lookahead={rows[0].peek_lookahead}) ===")
        print(format_peek_report(rows))
        print(f"\n§17 win condition: {summary['interpretation']}")
    if last_rows:
        path = save_artifacts(last_rows, win_condition(last_rows))
        print(f"\nArtifacts saved: {path}")


if __name__ == "__main__":
    main()
