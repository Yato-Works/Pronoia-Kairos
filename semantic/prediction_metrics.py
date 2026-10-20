"""§35 Prediction Metrics — pure functions, no dependencies.

Cache prediction metrics (Plan.md §35):
    future-use precision / recall (binary @ threshold)
    AUC (ranking quality, threshold-free)
    Calibration / ECE (are 0.8-probability predictions right 80% of the time?)

Semantic demand metrics (Top-1 / Top-k / MRR) operate on ranked candidate
lists and are defined on the same principle: prediction quality is measured,
but final evaluation is goodput / latency / memory / quality (§35 note).
"""
from __future__ import annotations
from typing import Dict, List, Sequence, Tuple


def precision_recall(labels: Sequence[int], preds: Sequence[int]) -> Tuple[float, float]:
    """Binary future-use precision / recall at the chosen threshold."""
    assert len(labels) == len(preds)
    tp = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 1)
    fp = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 0)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall


def roc_auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """AUC via Mann-Whitney U. Returns 0.5 when one class is absent."""
    assert len(scores) == len(labels)
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return 0.5
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def expected_calibration_error(
    probs: Sequence[float], labels: Sequence[int], n_bins: int = 10
) -> float:
    """ECE: weighted |accuracy - confidence| over equal-width bins."""
    assert len(probs) == len(labels)
    n = len(probs)
    if n == 0:
        return 0.0
    bins = [[] for _ in range(n_bins)]
    for p, y in zip(probs, labels):
        idx = min(n_bins - 1, int(p * n_bins))
        bins[idx].append((p, y))
    ece = 0.0
    for bucket in bins:
        if not bucket:
            continue
        conf = sum(p for p, _ in bucket) / len(bucket)
        acc = sum(y for _, y in bucket) / len(bucket)
        ece += (len(bucket) / n) * abs(acc - conf)
    return ece


def topk_precision(ranked: List[str], relevant: set, k: int) -> float:
    """§35 Top-k precision: fraction of top-k predictions that are relevant."""
    if k <= 0:
        return 0.0
    top = ranked[:k]
    if not top:
        return 0.0
    return sum(1 for r in top if r in relevant) / len(top)


def mrr(ranked: List[str], relevant: set) -> float:
    """§35 Mean Reciprocal Rank (single query): 1/rank of first relevant."""
    for i, r in enumerate(ranked, start=1):
        if r in relevant:
            return 1.0 / i
    return 0.0
