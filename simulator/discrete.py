"""Phase 0b — Discrete-Event Simulator (Plan.md §40, §41, §42).

Pipeline (§40):
    RequestTrace -> Event Queue -> Runtime State -> Planner
        -> Cache Simulation -> Latency Model -> Metrics

Control-plane simulator: no GPU, no tensors. Measures whether a cache /
scheduling policy beats LRU and how close it gets to Belady (§15) before
any hardware is involved.

Event model (§42): REQUEST_ARRIVAL, QUEUE_INSERT, CACHE_LOOKUP, CACHE_HIT,
CACHE_MISS, EVICT, PREFETCH_*, PREFILL_*, DECODE_*, REQUEST_COMPLETE.

Execution model: single FCFS server (queue_wait must be attributable for
scheduling research, §24–§26). Multi-server batching is a later extension.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import heapq

from simulator.latency import LatencyModel
from runtime.state import RuntimeState, SemanticState, QueueState, CacheState
from runtime.request import RequestAnalyzer
from runtime.planner import AdaptivePlanner


class EventKind:
    REQUEST_ARRIVAL = "REQUEST_ARRIVAL"
    QUEUE_INSERT = "QUEUE_INSERT"
    CACHE_LOOKUP = "CACHE_LOOKUP"
    CACHE_HIT = "CACHE_HIT"
    CACHE_MISS = "CACHE_MISS"
    PREFETCH_START = "PREFETCH_START"
    PREFETCH_END = "PREFETCH_END"
    EVICT = "EVICT"
    PREFILL_START = "PREFILL_START"
    PREFILL_END = "PREFILL_END"
    DECODE_START = "DECODE_START"
    DECODE_END = "DECODE_END"
    REQUEST_COMPLETE = "REQUEST_COMPLETE"


@dataclass(order=True)
class SimEvent:
    time: float
    seq: int
    kind: str = field(compare=False, default="")
    payload: Dict[str, Any] = field(compare=False, default_factory=dict)


@dataclass
class TraceRequest:
    """One entry of a RequestTrace (§40 / §49 semantic demand dataset)."""
    request_id: str
    key: str                     # cache object identity (prefix hash proxy)
    arrival_time: float = 0.0
    input_tokens: int = 512
    output_tokens: int = 128
    cached_tokens: int = 0       # reusable prefix length if hit
    session_id: str = ""
    future_use_probability: float = 0.0   # semantic signal (§7)
    semantic_confidence: float = 0.0


@dataclass
class RequestRecord:
    request_id: str
    key: str
    arrival_time: float
    start_time: float = 0.0
    complete_time: float = 0.0
    hit: bool = False
    ttft: float = 0.0
    e2e: float = 0.0
    prefetched: bool = False


@dataclass
class SimMetrics:
    requests: int = 0
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    prefetches: int = 0        # prefetches issued (Section 20)
    prefetch_hits: int = 0     # issued AND later reused (Section 65 accuracy)
    wasted_prefetches: int = 0 # issued but evicted/never reused (Section 65 waste)
    ttft_p50: float = 0.0
    ttft_p95: float = 0.0
    e2e_p95: float = 0.0
    goodput: float = 0.0          # ratio of requests meeting TTFT SLO
    slo_ttft: float = 1000.0
    slo_met: int = 0
    records: List[RequestRecord] = field(default_factory=list)

    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def goodput_ratio(self) -> float:
        return self.slo_met / self.requests if self.requests else 0.0


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(p / 100.0 * (len(ordered) - 1)))))
    return ordered[idx]


class DiscreteEventSimulator:
    """Runs a TraceRequest stream through Planner + Cache + LatencyModel.

    planner=None runs the plain LRU baseline (no planning). Passing an
    AdaptivePlanner makes retention/prefetch decisions from RuntimeState,
    which is exactly the Phase 2 comparison the research needs (46).
    """

    def __init__(
        self,
        capacity: int = 8,
        latency: Optional[LatencyModel] = None,
        planner: Optional[AdaptivePlanner] = None,
        slo_ttft: float = 1000.0,
        retention_policy: Optional[Any] = None,
        prefetch_controller: Optional[Any] = None,
        demand_forecast: Optional[Dict[str, float]] = None,
        queue_lookahead: int = 0,
    ):
        self.capacity = capacity
        self.latency = latency or LatencyModel()
        self.planner = planner
        self.slo_ttft = slo_ttft
        # Victim selection hook (Section 15): None -> plain LRU.
        # PredictiveRetentionPolicy evicts by lowest predicted future demand.
        self.retention_policy = retention_policy
        # 20 prefetch hook: when set, non-resident demand candidates are
        # actually fetched iff P_use*T_saved > T_transfer+T_interference.
        self.prefetch_controller = prefetch_controller
        # Optional external forecast {key: P_use} of not-yet-seen demand (50).
        # Falls back to the resident-key demand proxy when absent.
        self.demand_forecast = demand_forecast or {}
        # Section 46: how far into the pending queue to look when protecting
        # keys (PEEK-like queue awareness). 0 disables the queue view.
        self.queue_lookahead = queue_lookahead
        self.analyzer = RequestAnalyzer()

    def run(self, trace: List[TraceRequest]) -> SimMetrics:
        events: List[SimEvent] = []
        seq = 0
        for req in trace:
            heapq.heappush(events, SimEvent(req.arrival_time, seq, EventKind.REQUEST_ARRIVAL, {"req": req}))
            seq += 1

        cache: Dict[str, float] = {}    # key -> last_access_time (LRU proxy)
        key_signals: Dict[str, float] = {}  # key -> last seen future_use_probability
        key_last_seen: Dict[str, int] = {}  # key -> access counter at signal time
        access_counter: int = 0  # global access count, used to compute signal age
        lru: List[str] = []             # oldest -> newest
        prefetched: set = set()         # prefetched but not yet consumed
        prefetch_inflight: set = set()  # prefetch issued, transfer not done
        prefetch_attempted: set = set()  # ever-prefetched keys (no re-chase)
        records: List[RequestRecord] = []
        pending: List[TraceRequest] = []  # queue view for RuntimeState
        server_free_at = 0.0
        metrics = SimMetrics(slo_ttft=self.slo_ttft)

        def lru_touch(key: str):
            if key in lru:
                lru.remove(key)
            lru.append(key)
        now_marker = [0.0]

        def maybe_prefetch(req, now):
            # Section 20: prefetch a non-resident predicted-demand object only
            # when P_use*T_saved > T_transfer + T_interference. Issues
            # PREFETCH_* events so transfer lands on the timeline (Section 36).
            nonlocal seq
            if self.prefetch_controller is None:
                return
            ctrl = self.prefetch_controller
            # Exclude resident, in-flight, already-prefetched and the current
            # request's own key, so one object is transferred at most once.
            candidates = [(p, k) for k, p in self.demand_forecast.items()
                          if k not in cache and k not in prefetch_inflight
                          and k not in prefetched and k not in prefetch_attempted
                          and k != req.key]
            if not candidates:
                return
            p_use, key = max(candidates)
            if not ctrl.should_prefetch(key, p_use):
                return
            transfer = self.latency.transfer_cost(ctrl._tokens(key))
            heapq.heappush(events, SimEvent(now, seq, EventKind.PREFETCH_START,
                            {"key": key}))
            seq += 1
            heapq.heappush(events, SimEvent(now + transfer, seq,
                            EventKind.PREFETCH_END, {"key": key}))
            seq += 1
            ctrl.on_issued(key)
            prefetch_inflight.add(key)
            prefetch_attempted.add(key)
            metrics.prefetches += 1

        def admit_prefetch(key, now):
            # Materialize a completed prefetch into the cache, evicting by
            # the active victim policy if full (real GPU memory pressure).
            nonlocal seq
            now_marker[0] = now
            if key in cache:
                return
            if len(cache) >= self.capacity:
                if self.retention_policy is not None:
                    ages = {k: access_counter - key_last_seen.get(k, access_counter)
                            for k in lru}
                    queue_keys = self._queue_keys(pending)
                    victim = self._choose(lru, key_signals, ages, queue_keys)
                else:
                    victim = lru[0]
                lru.remove(victim)
                cache.pop(victim, None)
                key_signals.pop(victim, None)
                key_last_seen.pop(victim, None)
                if victim in prefetched:
                    prefetched.discard(victim)
                    if self.prefetch_controller is not None:
                        self.prefetch_controller.on_waste(victim)
                    metrics.wasted_prefetches += 1
                metrics.evictions += 1
            cache[key] = now
            key_signals[key] = self.demand_forecast.get(key, 0.0)
            key_last_seen[key] = access_counter
            lru.append(key)
            prefetched.add(key)
            prefetch_inflight.discard(key)

        while events:
            ev = heapq.heappop(events)
            now = ev.time

            if ev.kind == EventKind.REQUEST_ARRIVAL:
                req: TraceRequest = ev.payload["req"]
                pending.append(req)
                heapq.heappush(events, SimEvent(now, seq, EventKind.QUEUE_INSERT, {"req": req}))
                seq += 1
                # FCFS dispatch: service starts when the server is free.
                start = max(now, server_free_at)
                heapq.heappush(events, SimEvent(start, seq, EventKind.CACHE_LOOKUP, {"req": req}))
                seq += 1

            elif ev.kind == EventKind.QUEUE_INSERT:
                pass  # v0: queue depth is read from `pending` at lookup time

            elif ev.kind == EventKind.PREFETCH_START:
                pass  # transfer in flight; admitted on PREFETCH_END

            elif ev.kind == EventKind.PREFETCH_END:
                # Transfer complete: materialize into cache (may evict).
                admit_prefetch(ev.payload["key"], now)

            elif ev.kind == EventKind.CACHE_LOOKUP:
                req = ev.payload["req"]
                if req in pending:
                    pending.remove(req)
                state = self._build_state(req, cache, pending)
                prefetch_enabled = True
                if self.planner is not None:
                    plan, _trace = self.planner.plan_with_trace(
                        state, state.request_state,
                        request_id=req.request_id, timestamp=str(now),
                    )
                    prefetch_enabled = plan.prefetch_plan.enabled
                # Section 20 prefetch: when prefetch is enabled (by the
                # planner when present, or unconditionally in the standalone
                # prefetch experiment), fetch a non-resident predicted-demand
                # object, but only if P_use*T_saved > T_transfer+T_interference.
                # Candidates come from the external demand forecast (Section 50).
                if prefetch_enabled and self.prefetch_controller is not None:
                    maybe_prefetch(req, now)

                access_counter += 1
                key_signals[req.key] = req.future_use_probability
                key_last_seen[req.key] = access_counter
                hit = req.key in cache
                if hit:
                    lru_touch(req.key)
                    # Do NOT discard from `prefetched` here: the CACHE_HIT
                    # handler needs to know the hit was served by a prefetch
                    # to credit Section 20/35/65 prefetch-hit accounting.
                else:
                    if len(cache) >= self.capacity:
                        if self.retention_policy is not None:
                            ages = {k: access_counter - key_last_seen.get(k, access_counter)
                                    for k in lru}
                            victim = self.retention_policy.choose_victim(lru, key_signals, ages)
                        else:
                            victim = lru[0]
                        lru.remove(victim)
                        cache.pop(victim, None)
                        key_signals.pop(victim, None)
                        key_last_seen.pop(victim, None)
                        # Section 20: evicting a still-unused prefetched object
                        # is prefetch waste -> charge its transfer+interference.
                        if victim in prefetched:
                            prefetched.discard(victim)
                            if self.prefetch_controller is not None:
                                self.prefetch_controller.on_waste(victim)
                            metrics.wasted_prefetches += 1
                        heapq.heappush(events, SimEvent(now, seq, EventKind.EVICT, {"key": victim}))
                        seq += 1
                        metrics.evictions += 1
                    cache[req.key] = now
                    lru_touch(req.key)

                metrics.requests += 1
                record = RequestRecord(
                    request_id=req.request_id, key=req.key,
                    arrival_time=req.arrival_time, start_time=now, hit=hit,
                )
                records.append(record)
                kind = EventKind.CACHE_HIT if hit else EventKind.CACHE_MISS
                heapq.heappush(events, SimEvent(now, seq, kind, {"req": req, "record": record}))
                seq += 1

            elif ev.kind in (EventKind.CACHE_HIT, EventKind.CACHE_MISS):
                req = ev.payload["req"]
                record: RequestRecord = ev.payload["record"]
                hit = ev.kind == EventKind.CACHE_HIT
                queue_wait = now - req.arrival_time
                if hit:
                    metrics.hits += 1
                    ttft = self.latency.ttft_hit(queue_wait, req.cached_tokens or req.input_tokens)
                else:
                    metrics.misses += 1
                    ttft = self.latency.ttft_miss(queue_wait, req.input_tokens)
                e2e = self.latency.e2e(ttft, req.output_tokens)

                record.ttft = ttft
                record.e2e = e2e
                record.complete_time = now + e2e
                # Section 20/35/65: a prefetch 'hit' means the object was
                # prefetched AND reused. metrics.prefetches is the count of
                # prefetches *issued* (bumped in maybe_prefetch), so we must
                # not double-count here; instead credit the avoided recompute
                # on the controller and record the prefetch outcome.
                was_prefetched = req.key in prefetched
                record.prefetched = was_prefetched
                if was_prefetched:
                    prefetched.discard(req.key)
                    if self.prefetch_controller is not None:
                        if hit:
                            self.prefetch_controller.on_hit(req.key)
                            metrics.prefetch_hits += 1
                        else:
                            # Prefetched but still a miss (transient race):
                            # counts as waste per Section 20.
                            self.prefetch_controller.on_waste(req.key)
                            metrics.wasted_prefetches += 1

                server_free_at = now + e2e
                heapq.heappush(events, SimEvent(now + e2e, seq, EventKind.REQUEST_COMPLETE, {"record": record}))
                seq += 1

            elif ev.kind == EventKind.REQUEST_COMPLETE:
                rec = ev.payload["record"]
                if rec.ttft <= self.slo_ttft:
                    metrics.slo_met += 1

        metrics.records = records
        ttfts = [r.ttft for r in records]
        e2es = [r.e2e for r in records]
        metrics.ttft_p50 = _percentile(ttfts, 50)
        metrics.ttft_p95 = _percentile(ttfts, 95)
        metrics.e2e_p95 = _percentile(e2es, 95)
        metrics.goodput = metrics.goodput_ratio()
        return metrics

    def _build_state(
        self, req: TraceRequest, cache: Dict[str, float], pending: List[TraceRequest]
    ) -> RuntimeState:
        request_state = self.analyzer.analyze({
            "input_tokens": req.input_tokens,
            "expected_output_tokens": req.output_tokens,
            "shared_prefix_length": req.cached_tokens,
            "session_id": req.session_id,
        })
        pressure = len(cache) / self.capacity if self.capacity else 0.0
        return RuntimeState(
            request_state=request_state,
            semantic_state=SemanticState(
                future_use_probability=req.future_use_probability,
                confidence=req.semantic_confidence,
            ),
            queue_state=QueueState(queue_depth=len(pending)),
            cache_state=CacheState(memory_pressure=pressure),
        )

    def _queue_keys(self, pending: List[TraceRequest]) -> set:
        """Section 46: the set of keys the pending queue will reference within
        `queue_lookahead` requests (PEEK-like near-future queue view). Empty when
        queue_lookahead=0 (queue view disabled)."""
        if self.queue_lookahead <= 0 or not pending:
            return set()
        return {r.key for r in pending[: self.queue_lookahead]}

    def _choose(self, lru, key_signals, ages, queue_keys):
        """Call the retention policy's victim choice, passing the queue view when
        the policy accepts it (queue-aware policies); otherwise fall back to the
        3-arg signature (LRU / predictive retention)."""
        try:
            return self.retention_policy.choose_victim(lru, key_signals, ages, queue_keys)
        except TypeError:
            return self.retention_policy.choose_victim(lru, key_signals, ages)

