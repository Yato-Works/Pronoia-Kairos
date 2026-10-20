# Phase Structure — Pronoia Kairos

Fixed pipeline (never skip ahead to GPU/CUDA):

- Phase 0: Control Plane (`RuntimeState`, `RequestAnalyzer`, `AdaptivePlanner`, `RuntimePlan`, `DecisionTrace`). GPU-independent.
- Phase 1: Simulator (`AccessEvent` stream, `CacheState`, `CacheMetrics`) + Belady/LRU comparison (`compare/belady_lru.py`).
- Phase 2: Predictive Cache (`cache/predictive.py`) — uses `future_use_probability` and `confidence` to decide KEEP/EVICT/PREFETCH. Regret vs Belady is the evaluation axis.
- Phase 3+: Semantic Demand / Lethe connection, real KV tensor, C++/CUDA.
