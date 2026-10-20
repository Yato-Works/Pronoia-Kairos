"""Phase 2 — Queue + Cache Co-Optimization (Plan.md Section 46, 47).

Section 46: first reproduce PEEK-like behaviour (queue-aware prefix protection),
then extend it with Queue-only prediction + Semantic demand prediction. Section 47
Gate's decisive question:

    > "Does a condition exist that beats PEEK?"

PEEK protects prefixes referenced within a *near-future queue lookahead window*.
That is a QUEUE-ONLY signal: it sees what is queued, not what will be needed
later. We implement three retention policies behind the simulator's victim-selection
hook and compare them on the Section 47 metrics (TTFT p50/p95/p99, E2E p95,
goodput, hit rate, fairness):

    QueueAwareRetentionPolicy     - reproduces PEEK: protect keys the pending
                                    queue will reference within `lookahead`.
    QueueSemanticRetentionPolicy  - queue protection TEMPERED by a semantic
                                    future-demand signal (Section 9). When the
                                    queue lookahead is SHORT but the semantic
                                    horizon is LONG, this protects a hot prefix
                                    the queue cannot yet see - the hypothesized
                                    condition that beats PEEK (Section 17).

The queue view is the only extra state needed: the set of keys referenced by the
next `lookahead` pending requests. Semantic demand is the calibrated predicted
future-use probability already threaded through the trace (Section 90).

Combined score for victim selection (lowest is evicted first):
    score(c) = queue_pressure(c) * P_semantic(c) * decay^age(c)
where queue_pressure(c) is high when the queue will reference c soon (protect it),
and P_semantic lifts prefixes with real future demand even when the queue is
myopic. This is the queue x semantic co-optimization: the queue says WHEN, the
semantic signal says WHETHER, and their product decides retention.
"""
from __future__ import annotations
from typing import Dict, List


class QueueAwareRetentionPolicy:
    """PEEK-like: evict the key the pending queue is LEAST likely to reuse soon.

    Reproduces Section 46's first step. queue_pressure(k) = 1 if k appears in the
    next `lookahead` pending requests (about to be reused), else 0. The resident
    key with the lowest (queue_pressure, then recency) is evicted - i.e. a key not
    referenced by the queue soon is thrown out first, exactly PEEK's behaviour.
    """

    def __init__(self, lookahead: int = 8):
        self.lookahead = lookahead

    def score(self, prob: float, age: int) -> float:
        # PEEK ignores semantic demand: queue pressure alone decides.
        return prob  # placeholder; real decision in choose_victim via queue set

    def choose_victim(self, lru, signals, ages, queue_keys=None):
        if not lru:
            raise ValueError("choose_victim called on empty cache")
        queue_keys = queue_keys or set()
        # Protect queue-referenced keys; among the rest, evict least recently used.
        return min(
            lru,
            key=lambda k: (
                1 if k in queue_keys else 0,   # keep keys the queue will reuse
                -ages.get(k, 0),               # then keep fresher keys (LRO -> evict)
            ),
        )

    def describe(self) -> str:
        return f"QueueAwareRetentionPolicy(PEEK-like, lookahead={self.lookahead})"


class QueueSemanticRetentionPolicy:
    """Phase 2: queue protection TEMPERED by semantic future demand (Section 9/46).

    Evict the resident key with the lowest co-optimized score:

        score(c) = queue_pressure(c) + P_semantic(c) * decay^age(c)

    - queue_pressure(c) in {0,1}: the queue will reference c soon -> protect it
      (the PEEK term, weight queue_weight).
    - P_semantic(c): calibrated future demand -> protect hot prefixes the myopic
      queue cannot see yet (the semantic term).

    The two terms are ADDITIVE (not a fixed linear Score(request) model, per
    Section 24 which drops fixed alpha/beta): queue urgency dominates when the
    queue is deep, and semantic demand breaks ties / protects long-horizon
    prefixes the queue hasn't reached. This is the condition that can beat PEEK.
    """

    def __init__(self, lookahead: int = 8, decay: float = 0.85,
                 queue_weight: float = 1.0, semantic_weight: float = 1.0):
        self.lookahead = lookahead
        self.decay = decay
        self.queue_weight = queue_weight
        self.semantic_weight = semantic_weight

    def score(self, prob: float, age: int) -> float:
        return prob * (self.decay ** max(0, age))

    def choose_victim(self, lru, signals, ages, queue_keys=None):
        if not lru:
            raise ValueError("choose_victim called on empty cache")
        queue_keys = queue_keys or set()

        def coopt(k: str) -> float:
            queue_term = self.queue_weight * (1.0 if k in queue_keys else 0.0)
            semantic_term = self.semantic_weight * self.score(
                signals.get(k, 0.0), ages.get(k, 0))
            return queue_term + semantic_term

        return min(lru, key=coopt)

    def describe(self) -> str:
        return (f"QueueSemanticRetentionPolicy(queue x semantic, "
                f"lookahead={self.lookahead}, decay={self.decay})")
