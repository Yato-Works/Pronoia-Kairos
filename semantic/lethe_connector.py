"""Phase 3 — Lethe connector: real-data ingestion (Plan.md Section 43, 49, 95 Week 7).

Everything upstream (calibration, bridge, runtime, statistics, ablation,
sensitivity) runs on a synthetic Lethe recall stream so the research could be
built and validated offline. This module is the seam that swaps in a REAL Lethe
feed, so the same pipeline runs on real data.

Three ingestion paths, one output type (List[LetheRecallEvent]):

    LetheAPIConnector     - talk to a live Lethe service (recall endpoint).
                            The exact Lethe wire protocol is not fixed yet, so
                            this wraps a standard recall request and adapts the
                            response; configure base_url / headers for the real
                            deployment.
    LetheJSONLConnector   - load a recorded recall trace from JSONL (Section 49
                            DemandSample schema, Section 95 Week 7 "Lethe Trace
                            collection"). This is the primary real-data path
                            today: run Lethe, dump its recalls to JSONL, load here.
    SyntheticConnector    - re-exposes synthesize_lethe_stream for A/B tests and
                            for calibrating the simulator against real data
                            (Section 43: TTFT/memory error <= 10%).

The connector output feeds build_calibrated_trace / calibrated_forecast unchanged,
so every downstream experiment (design map, calibrated runtime, sensitivity) runs
on real Lethe recalls with zero code change.

Section 43 calibration: compare_real_vs_synthetic measures the correlation
distance and signal distribution of a real trace vs the synthetic generator, so
the simulator's claims can be gated against real hardware later.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
from abc import ABC, abstractmethod
import json
import os

from semantic.lethe_trace import LetheRecallEvent, synthesize_lethe_stream
from semantic.demand_predictor import MemoryRecord


class LetheConnector(ABC):
    """Source of real (or synthetic) Lethe recall events.

    Implementations return a list of LetheRecallEvent in recall order; the rest
    of the pipeline is agnostic to where they came from.
    """

    @abstractmethod
    def recall(self) -> List[LetheRecallEvent]:
        ...

    def describe(self) -> str:
        return self.__class__.__name__


@dataclass
class SyntheticConnector(LetheConnector):
    """Re-exposes the synthetic generator as a connector (A/B + calibration)."""
    n_objects: int = 20
    n_recalls: int = 300
    reuse_gap: int = 4
    noise: float = 0.1
    skew: float = 1.0
    seed: int = 7

    def recall(self) -> List[LetheRecallEvent]:
        return synthesize_lethe_stream(self.n_objects, self.n_recalls, self.reuse_gap,
                                       noise=self.noise, skew=self.skew, seed=self.seed)

    def describe(self) -> str:
        return (f"SyntheticConnector(objects={self.n_objects}, recalls={self.n_recalls}, "
                f"gap={self.reuse_gap}, noise={self.noise}, skew={self.skew})")


@dataclass
class LetheJSONLConnector(LetheConnector):
    """Load a recorded Lethe recall trace from a JSONL file (Section 49 schema).

    Each line is a DemandSample-like record. The minimum fields used here are:
        chunk_id (or memory_id), timestamp, and the semantic signal fields
        relevance / confidence / future_utility (optional; default 0 if absent).
    Missing signal fields degrade gracefully (the calibration layer still works),
    which is exactly what a real Lethe dump looks like before full instrumentation.
    """

    path: str

    def recall(self) -> List[LetheRecallEvent]:
        events: List[LetheRecallEvent] = []
        with open(self.path, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                events.append(self._to_event(rec, lineno))
        # Preserve recall order; sort by timestamp if present and out of order.
        events.sort(key=lambda e: e.timestamp)
        return events

    @staticmethod
    def _to_event(rec: Dict, lineno: int) -> LetheRecallEvent:
        chunk_id = rec.get("chunk_id") or rec.get("memory_id") or f"chunk-{lineno}"
        mem = MemoryRecord(
            memory_id=rec.get("memory_id", chunk_id),
            source_id=rec.get("source_id", "lethe-real"),
            chunk_id=chunk_id,
            relevance=float(rec.get("relevance", 0.0)),
            confidence=float(rec.get("confidence", 0.0)),
            provenance=rec.get("provenance", f"real@{lineno}"),
            temporal_relation=rec.get("temporal_relation", ""),
            session_relation=rec.get("session_relation", ""),
            future_utility=float(rec.get("future_utility", 0.0)),
        )
        return LetheRecallEvent(
            timestamp=float(rec.get("timestamp", lineno)),
            session_id=rec.get("session_id", "real-session"),
            memory=mem,
        )

    def describe(self) -> str:
        return f"LetheJSONLConnector(path={self.path})"



@dataclass
class LetheAPIConnector(LetheConnector):
    """Talk to a live Lethe recall service.

    The exact Lethe wire protocol is not finalized, so this issues a standard
    recall request (POST /recall with a query + limit) and adapts a flexible
    response shape: either a list of memory dicts, or {"memories": [...]}. Uses
    only the standard library (urllib) to stay dependency-free (Section 43). If
    the service is unreachable it returns an empty list (never crashes research).
    """

    base_url: str = "http://localhost:8000"
    query: str = ""
    limit: int = 300
    timeout: float = 5.0
    headers: Dict[str, str] = None  # type: ignore

    def recall(self) -> List[LetheRecallEvent]:
        import urllib.request
        import urllib.error
        url = self.base_url.rstrip("/") + "/recall"
        body = json.dumps({"query": self.query, "limit": self.limit}).encode("utf-8")
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        for k, v in (self.headers or {}).items():
            req.add_header(k, v)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, ValueError):
            return []  # service unreachable / bad response: degrade, never crash
        memories = payload.get("memories", payload) if isinstance(payload, dict) else payload
        if not isinstance(memories, list):
            return []
        events = [LetheJSONLConnector._to_event(m, i)
                  for i, m in enumerate(memories) if isinstance(m, dict)]
        events.sort(key=lambda e: e.timestamp)
        return events

    def describe(self) -> str:
        return f"LetheAPIConnector(base_url={self.base_url}, limit={self.limit})"


@dataclass
class CalibrationReport:
    source: str
    n_events: int
    n_objects: int
    median_reuse_distance: Optional[float]
    regime: str
    mean_raw_signal: float
    mean_confidence: float
    notes: str = ""


def describe_trace(events: List[LetheRecallEvent], source: str) -> CalibrationReport:
    """Summarize a recall trace's structure (for Section 43 calibration)."""
    from semantic.lethe_trace import (
        LetheTraceBuilder, measure_correlation_distance, p_sem_list)
    if not events:
        return CalibrationReport(source, 0, 0, None, "empty", 0.0, 0.0, "no events")
    trace = LetheTraceBuilder().build(events)
    profile = measure_correlation_distance(trace)
    raw = p_sem_list(events)
    confs = [e.memory.confidence for e in events]
    return CalibrationReport(
        source=source, n_events=len(events), n_objects=profile.n_objects,
        median_reuse_distance=profile.median_reuse_distance, regime=profile.regime,
        mean_raw_signal=sum(raw) / len(raw),
        mean_confidence=sum(confs) / len(confs),
        notes=f"reused_fraction={profile.reused_fraction:.2f}",
    )


def compare_real_vs_synthetic(real: List[LetheRecallEvent],
                              synthetic: List[LetheRecallEvent]) -> str:
    """Section 43 gate input: contrast a real Lethe trace with the synthetic one.

    Reports correlation-distance and signal differences so the simulator's
    synthetic assumptions can be validated (or corrected) against real data
    before any performance claim (TTFT/memory error <= 10%).
    """
    rr = describe_trace(real, "real")
    sr = describe_trace(synthetic, "synthetic")
    lines = [
        "Section 43 calibration: real vs synthetic Lethe trace",
        f"{'source':>10} {'n':>6} {'objects':>8} {'medReuse':>9} {'regime':>7} "
        f"{'meanSig':>8} {'meanConf':>9}",
        "-" * 62,
    ]
    for r in (rr, sr):
        med = f"{r.median_reuse_distance:.1f}" if r.median_reuse_distance is not None else "n/a"
        lines.append(f"{r.source:>10} {r.n_events:>6} {r.n_objects:>8} {med:>9} "
                     f"{r.regime:>7} {r.mean_raw_signal:>8.3f} {r.mean_confidence:>9.3f}")
    if rr.median_reuse_distance and sr.median_reuse_distance:
        drift = rr.median_reuse_distance - sr.median_reuse_distance
        lines.append(f"correlation-distance drift (real - synthetic) = {drift:+.1f} accesses")
    lines.append("Use this to gate simulator claims: if real traffic's regime differs "
                 "from synthetic, re-tune the generator before quoting NetGain.")
    return "\n".join(lines)


def write_sample_jsonl(path: str, n_objects: int = 20, n_recalls: int = 120,
                       reuse_gap: int = 4, skew: float = 1.0, seed: int = 5) -> str:
    """Emit a Section 49-schema JSONL recall trace (a real-data stand-in).

    Lets the JSONL connector + full pipeline be exercised end to end on a
    recorded trace before a live Lethe feed is wired up.
    """
    events = synthesize_lethe_stream(n_objects, n_recalls, reuse_gap, skew=skew, seed=seed)
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for i, ev in enumerate(events):
            m = ev.memory
            f.write(json.dumps({
                "session_id": ev.session_id,
                "request_id": f"req-{i}",
                "memory_id": m.memory_id,
                "chunk_id": m.chunk_id,
                "timestamp": ev.timestamp,
                "recall_event": True,
                "relevance": round(m.relevance, 4),
                "confidence": round(m.confidence, 4),
                "future_utility": round(m.future_utility, 4),
                "provenance": m.provenance,
            }, ensure_ascii=False) + "\n")
    return path
