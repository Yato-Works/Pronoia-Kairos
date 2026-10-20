# Regret Definition

Regret = `LRU_hits - Belady_hits` (must be ≤ 0, since Belady is the theoretical optimal upper bound).

Interpretation for Pronoia Predictive Cache:
- Regret closer to 0 = predictive policy approaches optimal.
- Positive regret = measurement error (LRU exceeded Belady; impossible for correct Belady implementation).
- Negative regret with small absolute value = good predictive performance.
