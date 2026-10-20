"""Phase 0a Workload generators (Plan.md Section 63 workload matrix, Section 95).

Synthetic workloads that mimic the access patterns the research targets:
- sequential / cyclic scans (cache-hostile)
- Zipf-like skewed (cache-friendly, real-world LLM serving default)
- multi-turn sessions (shared prefix within session, Section 8)
- agent workflows (bursty reuse with lookahead structure, Section 18)

All generators are deterministic (seeded) so results are reproducible
(Section 100 Research Reproducibility).
"""
from __future__ import annotations
from typing import List
import random


def sequential_scan(n_keys: int, repeats: int = 1) -> List[str]:
    """A B C ... Z A B C ... — LRU-hostile: no reuse until full cycle."""
    base = [f"k{i}" for i in range(n_keys)]
    return base * repeats


def zipf_like(n_keys: int, length: int, alpha: float = 1.1, seed: int = 42) -> List[str]:
    """Zipf-skewed access: a few hot keys, many cold ones."""
    rng = random.Random(seed)
    weights = [1.0 / ((i + 1) ** alpha) for i in range(n_keys)]
    total = sum(weights)
    probs = [w / total for w in weights]
    keys = [f"k{i}" for i in range(n_keys)]
    return rng.choices(keys, weights=probs, k=length)


def multi_turn_session(n_sessions: int, turns_per_session: int, shared_key: str = "sys") -> List[str]:
    """Each session repeats its own topic key plus a shared system prefix.
    Models multi-turn LLM sessions (Section 8 Lethe / session structure)."""
    seq: List[str] = []
    for s in range(n_sessions):
        topic = f"s{s}"
        for _ in range(turns_per_session):
            seq.append(shared_key)
            seq.append(topic)
    return seq


def agent_workflow(n_steps: int, n_tools: int, seed: int = 7) -> List[str]:
    """Agent step graph: alternating tool calls with occasional revisits.
    Models KVFlow-style agent workflows (Section 18)."""
    rng = random.Random(seed)
    tools = [f"tool{i}" for i in range(n_tools)]
    seq: List[str] = []
    for _ in range(n_steps):
        # pick a tool, use it twice (step + verify), sometimes revisit prior
        t = rng.choice(tools)
        seq.append(t)
        seq.append(t)
        if rng.random() < 0.3 and len(seq) > 4:
            seq.append(rng.choice(seq[:-2]))
    return seq

def long_range_reuse(
    n_anchors: int = 4,
    gap: int = 10,
    repeats: int = 6,
    filler_pool: int = 24,
    seed: int = 11,
) -> List[str]:
    """Long-horizon reuse that a short pending queue cannot see (Section 17).

    Each "anchor" key is revisited after `gap` accesses. Between two uses of
    the same anchor, `gap - 1` *distinct filler* keys stream through, so the
    anchor is NOT within a short queue lookahead at the moment it would need
    protection. Only a predictor with horizon W >= gap can tell the anchor
    will return; a queue-aware policy bounded to lookahead < gap cannot, and
    evicts the anchor as if it were cold.

    This is the workload family where the central thesis (Section 74) is
    supposed to win: semantic memory reasons about *long-range / next-session*
    demand (Section 50 H3/H4), which is exactly the signal a pending queue
    does not carry (Section 16). Deterministic for reproducibility (Section 100).
    """
    rng = random.Random(seed)
    anchors = [f"a{i}" for i in range(n_anchors)]
    fillers = [f"f{i}" for i in range(filler_pool)]
    seq: List[str] = []
    for _ in range(repeats):
        for a in anchors:
            seq.append(a)
            # fill the gap with distinct keys so the anchor is not in any
            # short lookahead window between its own uses
            for _ in range(gap - 1):
                seq.append(rng.choice(fillers))
    return seq

