"""§15 GreedyDual-Style Retention baselines + §38 Oracle Analysis set.

Retention policies compared in Phase 0a/Week 3-4 (§95):
    LRU, LFU, GDS-like, GDSF-like, PEEK-like (queue-aware), Belady Oracle.

All policies share the same interface:
    policy(sequence: List[str], capacity: int) -> int   # hit count

Regret vs Belady (§15, skills/regret-definition.md):
    regret = policy_hits - belady_hits   (<= 0 for any correct policy)

Belady is NOT an operational policy. Its research role is measuring
"how much could be gained if the future were known" (§38).
"""
from __future__ import annotations
from typing import List, Dict


def belady_optimal(sequence: List[str], capacity: int) -> int:
    """Optimal (Belady) hit count: evict the item with furthest next use."""
    cache = set()
    hits = 0
    for i, key in enumerate(sequence):
        if key in cache:
            hits += 1
        else:
            # Evict item whose next occurrence is furthest (or never)
            future_use = {}
            for c in cache:
                try:
                    future_use[c] = sequence.index(c, i + 1)
                except ValueError:
                    future_use[c] = float('inf')
            if len(cache) >= capacity:
                evict_key = max(cache, key=lambda k: future_use.get(k, float('inf')))
                cache.remove(evict_key)
            cache.add(key)
    return hits


def lru_hit_count(sequence: List[str], capacity: int) -> int:
    from collections import OrderedDict
    cache = OrderedDict()
    hits = 0
    for key in sequence:
        if key in cache:
            hits += 1
            cache.move_to_end(key)
        else:
            if len(cache) >= capacity:
                cache.popitem(last=False)
            cache[key] = True
    return hits


def lfu_hit_count(sequence: List[str], capacity: int) -> int:
    """Least Frequently Used. Tie-break by LRU (evict least recent among
    equal-frequency keys) so behavior is deterministic."""
    freq: Dict[str, int] = {}
    recent: List[str] = []   # recency order, oldest first
    hits = 0
    for key in sequence:
        if key in freq:
            hits += 1
            freq[key] += 1
            recent.remove(key)
            recent.append(key)
        else:
            if len(freq) >= capacity:
                # Evict min frequency; among ties, the least recently used.
                victim = min(recent, key=lambda k: (freq[k], recent.index(k)))
                recent.remove(victim)
                del freq[victim]
            freq[key] = 1
            recent.append(key)
    return hits


def gds_like_hit_count(sequence: List[str], capacity: int, base_priority: float = 1.0) -> int:
    """GDS-like (GreedyDual) without size: score = base_priority + L,
    where L is a global aging term updated to the max score at eviction."""
    cache: Dict[str, float] = {}   # key -> priority
    hits = 0
    L = 0.0
    for key in sequence:
        if key in cache:
            hits += 1
            cache[key] = base_priority + L   # re-rank on hit
        else:
            if len(cache) >= capacity:
                victim = min(cache, key=lambda k: cache[k])
                L = max(L, cache[victim])    # aging: L rises to victim score
                del cache[victim]
            cache[key] = base_priority + L
    return hits


def gdsf_like_hit_count(sequence: List[str], capacity: int, decay: float = 0.5) -> int:
    """GDSF-like: score = frequency * recency_weight, with recency weight
    decaying per access so stale high-frequency keys lose protection."""
    freq: Dict[str, int] = {}
    score: Dict[str, float] = {}
    hits = 0
    step = 0
    for key in sequence:
        step += 1
        if key in freq:
            hits += 1
            freq[key] += 1
            score[key] = freq[key]  # fresh access refreshes full score
        else:
            if len(score) >= capacity:
                victim = min(score, key=lambda k: score[k])
                del score[victim]
                del freq[victim]
            freq[key] = 1
            score[key] = 1.0
        # global decay so older scores shrink relative to newer ones
        for k in score:
            if k != key:
                score[k] *= decay
    return hits


def peek_like_hit_count(sequence: List[str], capacity: int, lookahead: int = 8) -> int:
    """PEEK-like (queue-aware) baseline (§17, §16).

    Uses the pending queue's *near-future* window (not the full future,
    which is Belady's exclusive information). Keys referenced within the
    lookahead window are protected from eviction; keys outside it are
    evicted first. This models PEEK's queue-aware prefix protection
    without giving it oracle power.
    """
    hits = 0
    cache: set = set()
    order: List[str] = []   # protection order: protected keys at the end
    for i, key in enumerate(sequence):
        window = set(sequence[i + 1: i + 1 + lookahead])
        if key in cache:
            hits += 1
            order.remove(key)
            order.append(key)
        else:
            if len(cache) >= capacity:
                # Evict the least-recently-used key NOT in the near window;
                # only fall back to LRU among protected keys if all are.
                unprotected = [k for k in order if k not in window]
                victim = unprotected[0] if unprotected else order[0]
                order.remove(victim)
                cache.discard(victim)
            cache.add(key)
            order.append(key)
    return hits


# Registry for the harness / benchmarks
POLICIES = {
    "LRU": lru_hit_count,
    "LFU": lfu_hit_count,
    "GDS-like": gds_like_hit_count,
    "GDSF-like": gdsf_like_hit_count,
    "PEEK-like": peek_like_hit_count,
    "Belady": belady_optimal,
}


def regret_vs_belady(policy_hits: int, belady_hits: int) -> int:
    """§15 core metric: how far from the theoretical upper bound."""
    return policy_hits - belady_hits
