"""Phase 3 — Lethe semantic-to-computational bridge (Plan.md Section 8, 48, 49, 50, 90).

This is the join point between Lethe (semantic memory) and the computational
cache. A real Lethe deployment emits a stream of MemoryRecord recall events
(memory_id, chunk_id, relevance, confidence, future_utility, session relation...).
This module turns that stream into:

    1. a RequestTrace (the cache-key access sequence the simulator consumes), and
    2. a measurement of the workload's CORRELATION DISTANCE (how far apart reuse
       events are), which is exactly the axis the design map (Section 63.1) says
       decides which cache mechanism wins.

So instead of guessing where a real workload sits on the design map, we MEASURE it
from Lethe's own recall stream and place it on the map. The measured distance plus
capacity then selects the recommended mechanism (LRU / retention / prefetch /
adaptive) directly from the design-map rule, closing the loop:

    Lethe recall stream -> trace -> correlation distance -> design-map regime
        -> recommended mechanism -> (optionally) verify on the simulator.

Lethe does NOT touch the KV cache (Section 8); it only supplies semantic demand.
The distance measurement below is pure trace analysis and needs no GPU.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from semantic.demand_predictor import MemoryRecord
from simulator.discrete import TraceRequest


# --- Realistic Lethe recall stream -------------------------------------------

@dataclass
class LetheRecallEvent:
    """One Lethe recall: which memory/chunk was retrieved and its semantic signal."""
    timestamp: float
    session_id: str
    memory: MemoryRecord


def p_sem_list(events: List[LetheRecallEvent]) -> List[float]:
    """Lethe's raw per-request semantic demand: relevance*confidence*(1+utility),
    clipped to [0, 0.99] (Section 9.1 stage-1 proxy). Position-independent."""
    return [
        min(0.99, ev.memory.relevance * ev.memory.confidence * (1.0 + ev.memory.future_utility))
        for ev in events
    ]


class LetheTraceBuilder:
    """Turn a Lethe recall stream into the cache access sequence the simulator runs.

    Each recall event names a chunk; the chunk id is the cache key. Ordering is
    the recall order (the real demand sequence). The retention signal is CALIBRATED
    to the horizon reuse structure (see build); Lethe's raw relevance*confidence*
    future_utility demand is kept separately for Section 35 calibration metrics.
    """

    def __init__(self, tokens_per_object: int = 512):
        self.tokens_per_object = tokens_per_object

    def build(self, events: List[LetheRecallEvent], horizon: int = 3) -> List[TraceRequest]:
        """Turn recall events into the cache access trace.

        future_use_probability is CALIBRATED to the actual reuse structure: for
        the request at position i, it is 1.0 if that chunk recurs within `horizon`
        later recalls (the Section 50 prediction target), else a small residual.
        Lethe's raw relevance*confidence*(1+future_utility) is NOT used directly
        as the retention signal because it is position-independent "general future
        value" - it does not say "needed again within horizon from HERE", which is
        what predictive retention's victim choice needs. Calibrating to the
        horizon makes retention function; the raw signal is still recorded for
        Section 35 calibration metrics (AUC/ECE) via build_raw_signals().
        """
        keys = [ev.memory.chunk_id for ev in events]
        # Precompute next-recurrence-within-horizon per position (the calibrated
        # target). This uses the realized stream (a calibration pass), so the
        # signal is "calibrated", not a pure ex-ante prediction - documented above.
        n = len(keys)
        calibrated = [0.0] * n
        next_idx: dict = {}
        for i in range(n - 1, -1, -1):
            k = keys[i]
            nxt = next_idx.get(k)
            calibrated[i] = 1.0 if (nxt is not None and nxt - i <= horizon) else 0.0
            next_idx[k] = i
        trace: List[TraceRequest] = []
        for i, ev in enumerate(events):
            m = ev.memory
            p_sem = min(0.99, m.relevance * m.confidence * (1.0 + m.future_utility))
            trace.append(TraceRequest(
                request_id=f"lethe-{i}",
                key=m.chunk_id,
                arrival_time=ev.timestamp,
                input_tokens=self.tokens_per_object,
                output_tokens=32,
                session_id=ev.session_id,
                future_use_probability=calibrated[i],
                semantic_confidence=m.confidence,
            ))
        self._raw_signals = p_sem_list(events)
        return trace

    def build_raw_signals(self, events: List[LetheRecallEvent]) -> List[float]:
        """Lethe's raw per-request semantic demand (position-independent), kept for
        Section 35 calibration metrics (compare predicted vs actual reuse)."""
        return p_sem_list(events)

    def build_semantic_forecast(self, events: List[LetheRecallEvent]) -> Dict[str, float]:
        """Per-chunk future-demand forecast from Lethe's signal (for prefetch).

        A chunk's forecast is the strongest semantic demand any of its recall
        events expressed - the best available estimate that it will be needed.
        """
        forecast: Dict[str, float] = {}
        for ev in events:
            m = ev.memory
            p_sem = min(0.99, m.relevance * m.confidence * (1.0 + m.future_utility))
            forecast[m.chunk_id] = max(forecast.get(m.chunk_id, 0.0), p_sem)
        return forecast


# --- Signal calibration (Section 35, 50) -------------------------------------
# Lethe's raw demand is position-independent "general importance". Two ex-ante
# calibrations (no future info) turn it into a usable demand signal, and Section
# 50 assigns each to a time scale:
#   recency     -> immediate prefetch (H1/H2): raw * decay^age
#   popularity  -> longer-term retention (H3/H4): raw * running access frequency

def recency_signals(events: List[LetheRecallEvent], decay: float = 0.8) -> List[float]:
    """Per-request recency-calibrated demand (immediate prefetch, H1/H2)."""
    raw = p_sem_list(events)
    last_seen: dict = {}
    out = []
    for i, ev in enumerate(events):
        k = ev.memory.chunk_id
        age = (i - last_seen[k]) if k in last_seen else 0
        out.append(min(1.0, max(0.0, raw[i] * (decay ** max(0, age)))))
        last_seen[k] = i
    return out


def popularity_signals(events: List[LetheRecallEvent]) -> List[float]:
    """Per-request popularity-calibrated demand (longer-term retention, H3/H4)."""
    raw = p_sem_list(events)
    counts: dict = {}
    out = []
    for i, ev in enumerate(events):
        k = ev.memory.chunk_id
        counts[k] = counts.get(k, 0) + 1
        freq = counts[k] / (i + 1)  # running share of this chunk so far
        out.append(min(1.0, max(0.0, raw[i] * freq)))
    return out


def calibrated_signals(events: List[LetheRecallEvent], calibration: str,
                       decay: float = 0.8) -> List[float]:
    """Dispatch to a named calibration. 'raw' returns Lethe's uncalibrated demand."""
    if calibration == "raw":
        return p_sem_list(events)
    if calibration == "recency":
        return recency_signals(events, decay=decay)
    if calibration == "popularity":
        return popularity_signals(events)
    raise ValueError(f"unknown calibration: {calibration}")


def build_calibrated_trace(events: List[LetheRecallEvent], calibration: str,
                           tokens_per_object: int = 512) -> List[TraceRequest]:
    """Build a trace whose retention signal (future_use_probability) is the chosen
    calibration, so predictive retention is driven by that calibrated demand."""
    sig = calibrated_signals(events, calibration)
    trace: List[TraceRequest] = []
    for i, ev in enumerate(events):
        m = ev.memory
        trace.append(TraceRequest(
            request_id=f"lethe-{i}",
            key=m.chunk_id,
            arrival_time=ev.timestamp,
            input_tokens=tokens_per_object,
            output_tokens=32,
            session_id=ev.session_id,
            future_use_probability=sig[i],
            semantic_confidence=m.confidence,
        ))
    return trace


def calibrated_forecast(events: List[LetheRecallEvent], calibration: str,
                        decay: float = 0.8) -> Dict[str, float]:
    """Per-chunk prefetch forecast from the chosen calibration: a chunk's forecast
    is the strongest calibrated demand any of its recall events expressed."""
    sig = calibrated_signals(events, calibration, decay=decay)
    forecast: Dict[str, float] = {}
    for ev, p in zip(events, sig):
        k = ev.memory.chunk_id
        forecast[k] = max(forecast.get(k, 0.0), p)
    return forecast


# --- Correlation-distance measurement ----------------------------------------

@dataclass
class CorrelationProfile:
    """Measured reuse structure of an access sequence (Section 63.1 axis)."""
    n_accesses: int
    n_objects: int
    reused_fraction: float      # fraction of accesses that are reuses
    median_reuse_distance: Optional[float]
    p90_reuse_distance: Optional[float]
    mean_reuse_distance: Optional[float]
    regime: str                 # "short" | "mid" | "long" (design-map distance class)


def measure_correlation_distance(trace: List[TraceRequest]) -> CorrelationProfile:
    """Measure how far apart reuse events are, to place the trace on the map.

    reuse_distance = number of accesses between two consecutive uses of the same
    chunk. The regime thresholds match the design-map workload families:
        short: median reuse distance < 3   (recency-correlated, LRU is strong)
        mid:   median in [3, 8)            (anchors revisit within a few accesses)
        long:  median >= 8                 (anchors revisit after many accesses)
    """
    keys = [r.key for r in trace]
    last_seen: Dict[str, int] = {}
    distances: List[int] = []
    reuses = 0
    for i, k in enumerate(keys):
        if k in last_seen:
            distances.append(i - last_seen[k])
            reuses += 1
        last_seen[k] = i
    n = len(keys)

    def median(xs: List[int]) -> Optional[float]:
        if not xs:
            return None
        s = sorted(xs)
        mid = len(s) // 2
        return float(s[mid]) if len(s) % 2 else (s[mid - 1] + s[mid]) / 2.0

    def percentile(xs: List[int], p: float) -> Optional[float]:
        if not xs:
            return None
        s = sorted(xs)
        idx = min(len(s) - 1, max(0, int(round(p / 100.0 * (len(s) - 1)))))
        return float(s[idx])

    med = median(distances)
    if med is None or med < 3:
        regime = "short"
    elif med < 8:
        regime = "mid"
    else:
        regime = "long"

    return CorrelationProfile(
        n_accesses=n,
        n_objects=len(last_seen),
        reused_fraction=reuses / n if n else 0.0,
        median_reuse_distance=med,
        p90_reuse_distance=percentile(distances, 90),
        mean_reuse_distance=sum(distances) / len(distances) if distances else None,
        regime=regime,
    )


# --- Synthetic-but-realistic Lethe streams (stand in for a live Lethe feed) --

def synthesize_lethe_stream(
    n_objects: int = 20,
    n_recalls: int = 300,
    reuse_gap: int = 3,
    noise: float = 0.1,
    seed: int = 7,
    skew: float = 0.0,
    decouple: bool = False,
) -> List[LetheRecallEvent]:
    """Generate a Lethe-like recall stream.

    Two regimes, matching how real semantic-memory traffic behaves:

    skew=0.0 (uniform / round-robin): objects are recalled every `reuse_gap`
        accesses, so `measure_correlation_distance` recovers ~reuse_gap. Use this
        to place a workload on the design map by a KNOWN correlation distance.

    skew>0.0 (zipf-like popularity): each object has a popularity weight
        ~1/(rank+1)^skew; recalls and Lethe's future_utility both follow that
        popularity, so hot chunks are recalled often AND carry high semantic
        demand. Here recency and future demand are POSITIVELY correlated (the
        realistic case for a semantic memory), which is the regime where recency
        calibration of the raw signal can help. Popularity defines reuse distance.

    Lethe's relevance/confidence carry realistic signal + noise; future_utility
    tracks popularity (with noise) so the predictor is NOT an oracle.
    """
    import random
    rng = random.Random(seed)
    chunk_ids = [f"mem-{i:03d}/chunk-0" for i in range(n_objects)]

    if skew <= 0.0:
        # Round-robin at period = reuse_gap (exact correlation distance).
        period = max(1, min(reuse_gap, n_objects))
        order = [chunk_ids[t % period] for t in range(n_recalls)]
        popularity = {c: 1.0 / period for c in chunk_ids}
    else:
        # Zipf-like popularity weights; sample the recall order from them.
        weights = [1.0 / ((r + 1) ** skew) for r in range(n_objects)]
        total = sum(weights)
        popularity = {c: weights[i] / total for i, c in enumerate(chunk_ids)}
        if decouple:
            # DECOUPLE popularity from recency: high-popularity chunks are
            # accessed on a LONG period (rarely), low-popularity on a SHORT
            # period (often). So "popular" != "recently used" - the regime where
            # the Section 50 separation (recency=immediate vs popularity=
            # long-term) can be distinguished. Round-robin per popularity band.
            # rank 0 (most popular) -> longest period, last rank -> shortest.
            periods = [max(1, int(round((n_objects - r) / 2.0)) + 1)
                       for r in range(n_objects)]
            band = {c: periods[i] for i, c in enumerate(chunk_ids)}
            pointers = {c: i for i, c in enumerate(chunk_ids)}
            order = []
            for t in range(n_recalls):
                # pick the chunk whose next scheduled recall is due (min pointer)
                key = min(chunk_ids, key=lambda c: pointers[c])
                order.append(key)
                pointers[key] += band[key]
        else:
            cum, acc = [], 0.0
            for wgt in weights:
                acc += wgt / total
                cum.append(acc)
            order = []
            for _ in range(n_recalls):
                x = rng.random()
                lo, hi = 0, len(cum) - 1
                while lo < hi:
                    mid = (lo + hi) // 2
                    if cum[mid] < x:
                        lo = mid + 1
                    else:
                        hi = mid
                order.append(chunk_ids[lo])

    events: List[LetheRecallEvent] = []
    for t, key in enumerate(order):
        # future_utility tracks popularity (Lethe learned what recurs) + noise.
        base_util = 0.2 + 0.8 * popularity[key]
        future_utility = min(1.0, max(0.0, base_util + rng.gauss(0, noise)))
        relevance = min(1.0, max(0.05, 0.6 + rng.gauss(0, noise)))
        confidence = min(1.0, max(0.05, 0.8 + rng.gauss(0, noise)))
        mem = MemoryRecord(
            memory_id=key.split("/")[0],
            source_id=f"lethe-session-{t // 50}",
            chunk_id=key,
            relevance=relevance,
            confidence=confidence,
            provenance=f"recall@{t}",
            temporal_relation="recent",
            session_relation="same-session",
            future_utility=future_utility,
        )
        events.append(LetheRecallEvent(
            timestamp=float(t) * 5.0,
            session_id=f"sess-{t // 50}",
            memory=mem,
        ))
    return events


# --- Place a measured trace on the design map --------------------------------

# Design-map rule distilled from compare/design_map (Section 63.1), keyed by
# (regime, capacity-band). Bands: tight (<=6), roomy (>6).
_DESIGN_MAP_RULE = {
    ("short", "tight"): "lru",
    ("short", "roomy"): "prefetch",
    ("mid", "tight"): "adaptive",
    ("mid", "roomy"): "adaptive",
    ("long", "tight"): "retention",
    ("long", "roomy"): "adaptive",
}


@dataclass
class DesignMapPlacement:
    profile: CorrelationProfile
    capacity: int
    capacity_band: str
    recommended_mechanism: str
    feed_retention: bool
    rationale: str


def place_on_design_map(trace: List[TraceRequest], capacity: int) -> DesignMapPlacement:
    """Measure a trace's correlation distance and recommend a mechanism.

    Uses the measured regime + capacity band to look up the design-map rule
    (Section 63.1). feed_retention follows the same rule as the map: retention is
    fed the semantic signal for mid/long regimes, but short-range keeps the LRU
    core (Section 76 #5).
    """
    profile = measure_correlation_distance(trace)
    band = "tight" if capacity <= 6 else "roomy"
    mech = _DESIGN_MAP_RULE[(profile.regime, band)]
    feed = profile.regime in ("mid", "long")
    if profile.regime == "short":
        rationale = (f"Recency-correlated (median reuse {profile.median_reuse_distance}); "
                     f"LRU is near-optimal, do not feed noisy signal into retention.")
    elif profile.regime == "mid":
        rationale = (f"Mid-range reuse (median {profile.median_reuse_distance}); "
                     f"ADAPTIVE (retention+prefetch) wins - prefetch near-term demand, "
                     f"retention protects it.")
    else:
        rationale = (f"Long-range reuse (median {profile.median_reuse_distance}); "
                     f"predictive RETENTION dominates; prefetch loses (objects evicted "
                     f"before reuse).")
    return DesignMapPlacement(
        profile=profile, capacity=capacity, capacity_band=band,
        recommended_mechanism=mech, feed_retention=feed, rationale=rationale,
    )
