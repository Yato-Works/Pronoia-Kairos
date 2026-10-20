"""Belady / LRU / Baseline comparison framework — Phase 2."""
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
