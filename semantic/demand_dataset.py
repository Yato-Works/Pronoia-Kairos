"""§49 Semantic Demand Dataset — record what was predicted vs what happened.

Required fields (Plan.md §49):
    session_id, request_id, memory_id, chunk_id, timestamp,
    recall_event, actual_usage, cache_hit, cache_miss, future_use_distance

§50 Prediction Target — horizons:
    H1 = next request
    H2 = next 3 requests
    H3 = next 30 seconds (abstract time units in simulation)
    H4 = next session turn

This dataset is the join point between Lethe (semantic memory) and the
computational cache: every recall_event carries the predicted
future_use_probability, and actual_usage / future_use_distance record what
really happened. Prediction quality (§35) and NetGain (§52) are computed
from these records.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
import json
import os

# §50 horizons
HORIZONS = {"H1": 1, "H2": 3, "H3": 30.0, "H4": 1}  # requests / time units / turns


@dataclass
class DemandSample:
    session_id: str
    request_id: str
    memory_id: str
    chunk_id: str
    timestamp: float
    recall_event: bool = False        # semantic memory was consulted
    predicted_probability: float = 0.0  # P_future_demand at recall time
    actual_usage: bool = False        # predicted object was used by a later request
    cache_hit: bool = False
    cache_miss: bool = False
    future_use_distance: Optional[float] = None  # steps (or time) until reuse; None = never

    def to_dict(self) -> Dict:
        return asdict(self)


class DemandDatasetRecorder:
    """Append-only in-memory dataset with JSONL persistence (§49, §98)."""

    def __init__(self):
        self.samples: List[DemandSample] = []

    def record(self, sample: DemandSample) -> None:
        self.samples.append(sample)

    def by_session(self, session_id: str) -> List[DemandSample]:
        return [s for s in self.samples if s.session_id == session_id]

    def save(self, path: str) -> str:
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for s in self.samples:
                f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")
        return path

    @classmethod
    def load(cls, path: str) -> "DemandDatasetRecorder":
        rec = cls()
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rec.record(DemandSample(**json.loads(line)))
        return rec

    def __len__(self) -> int:
        return len(self.samples)


def future_use_distance_within(
    access_sequence: List[str], index: int, key: str, horizon_requests: Optional[int] = None
) -> Optional[int]:
    """§50: distance in requests until `key` is used again after `index`.
    None if never reused (or beyond horizon)."""
    end = len(access_sequence) if horizon_requests is None else min(
        len(access_sequence), index + 1 + horizon_requests
    )
    for j in range(index + 1, end):
        if access_sequence[j] == key:
            return j - index
    return None


def horizon_labels(access_sequence: List[str], index: int, key: str) -> Dict[str, bool]:
    """§50: was the object reused within H1 / H2 / H4 (H3 = time-based,
    filled by callers with timestamps)."""
    d1 = future_use_distance_within(access_sequence, index, key, horizon_requests=HORIZONS["H1"])
    d2 = future_use_distance_within(access_sequence, index, key, horizon_requests=HORIZONS["H2"])
    return {"H1": d1 is not None, "H2": d2 is not None, "H4": d2 is not None}
