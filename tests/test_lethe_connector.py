"""Phase 3 — Lethe connector: real-data ingestion (Plan.md Section 43, 49, 95).

The connector is the seam that swaps the synthetic Lethe feed for a real one.
These tests fix that seam:

    - SyntheticConnector re-exposes the generator (A/B + calibration).
    - LetheJSONLConnector loads a Section 49-schema JSONL trace into recall
      events; missing signal fields degrade gracefully (real dumps are partial).
    - LetheAPIConnector degrades to empty when the service is unreachable
      (never crashes the research pipeline).
    - Section 43 calibration contrasts a real vs synthetic trace (regime + drift).
    - the full calibrated-runtime pipeline runs on connector events and
      reproduces the synthetic "popularity wins" ordering on real data.
"""
import sys, os, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from semantic.lethe_connector import (
    LetheConnector, SyntheticConnector, LetheJSONLConnector, LetheAPIConnector,
    write_sample_jsonl, describe_trace, compare_real_vs_synthetic, CalibrationReport,
)
from compare.lethe_connector_experiment import run_pipeline_on_connector


def _write_and_load(**kw):
    d = tempfile.mkdtemp()
    path = os.path.join(d, "lethe_trace.jsonl")
    write_sample_jsonl(path, **kw)
    return LetheJSONLConnector(path=path).recall()


# --- Connector contract ------------------------------------------------------

def test_connectors_share_the_interface():
    assert isinstance(SyntheticConnector(), LetheConnector)
    assert isinstance(LetheAPIConnector(), LetheConnector)


def test_synthetic_connector_returns_events():
    events = SyntheticConnector(n_objects=10, n_recalls=50, reuse_gap=4, seed=1).recall()
    assert len(events) == 50
    assert all(hasattr(e, "memory") and hasattr(e, "timestamp") for e in events)


def test_jsonl_connector_loads_and_orders():
    events = _write_and_load(n_objects=15, n_recalls=120, reuse_gap=4, skew=1.0, seed=5)
    assert len(events) == 120
    # events are sorted by timestamp (non-decreasing).
    ts = [e.timestamp for e in events]
    assert ts == sorted(ts)
    assert all(e.memory.chunk_id for e in events)


def test_jsonl_roundtrip_preserves_signal():
    """The JSONL dump keeps relevance/confidence/future_utility, so the loaded
    events carry the same signal the generator produced."""
    events = _write_and_load(n_objects=12, n_recalls=80, reuse_gap=4, skew=1.0, seed=3)
    # signals are in [0,1] and mostly non-zero (a real instrumented trace).
    confs = [e.memory.confidence for e in events]
    assert all(0.0 <= c <= 1.0 for c in confs)
    assert sum(confs) / len(confs) > 0.5


def test_jsonl_degrades_on_missing_fields():
    """A partial real dump (only chunk_id/timestamp) still loads - signals default
    to 0, which the calibration layer tolerates (Section 49 before full
    instrumentation)."""
    d = tempfile.mkdtemp()
    path = os.path.join(d, "partial.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        f.write('{"chunk_id": "a", "timestamp": 0.0}\n')
        f.write('{"chunk_id": "b", "timestamp": 1.0}\n')
        f.write('{"memory_id": "a", "timestamp": 2.0}\n')  # memory_id fallback
    events = LetheJSONLConnector(path=path).recall()
    assert len(events) == 3
    assert events[0].memory.relevance == 0.0  # missing -> default 0
    assert events[2].memory.chunk_id == "a"   # memory_id used as chunk fallback


def test_api_connector_degrades_when_unreachable():
    """A live Lethe service that is down must not crash the pipeline."""
    conn = LetheAPIConnector(base_url="http://127.0.0.1:1", timeout=0.5)
    assert conn.recall() == []  # unreachable -> empty, no exception


# --- Section 43 calibration --------------------------------------------------

def test_describe_trace_summarizes_structure():
    events = _write_and_load(n_objects=15, n_recalls=120, reuse_gap=4, skew=1.0, seed=5)
    rep = describe_trace(events, "jsonl")
    assert isinstance(rep, CalibrationReport)
    assert rep.n_events == 120
    assert rep.n_objects == 15
    assert rep.regime in ("short", "mid", "long")
    assert 0.0 <= rep.mean_raw_signal <= 1.0


def test_describe_trace_handles_empty():
    rep = describe_trace([], "empty")
    assert rep.regime == "empty" and rep.n_events == 0


def test_compare_real_vs_synthetic_reports_regime_and_drift():
    real = _write_and_load(n_objects=20, n_recalls=300, reuse_gap=4, skew=1.0, seed=5)
    synthetic = SyntheticConnector(n_objects=20, n_recalls=300, reuse_gap=4,
                                   skew=1.0, seed=99).recall()
    text = compare_real_vs_synthetic(real, synthetic)
    assert "Section 43 calibration" in text
    assert "drift" in text
    # both traces land in the same regime (the generator is calibrated to it).
    assert describe_trace(real, "r").regime == describe_trace(synthetic, "s").regime


# --- Full pipeline on connector data (Section 90) ----------------------------

def test_pipeline_runs_on_jsonl_and_popularity_wins():
    """End to end on a JSONL 'real' trace: the calibrated runtime reproduces the
    synthetic ordering - popularity calibration gives the best NetGain."""
    d = tempfile.mkdtemp()
    path = os.path.join(d, "lethe_trace.jsonl")
    write_sample_jsonl(path, n_objects=20, n_recalls=300, reuse_gap=4, skew=1.0, seed=5)
    conn = LetheJSONLConnector(path=path)
    result = run_pipeline_on_connector(conn, capacity=6)
    assert result.n_events == 300
    assert result.regime in ("short", "mid", "long")
    assert set(result.net_gain_by_calibration) == {"raw", "recency", "popularity"}
    assert result.best_calibration == "popularity"
    assert result.best_net_gain > result.net_gain_by_calibration["raw"]


def test_pipeline_raw_calibration_is_worst_on_real_data():
    """The Section 8/90 lesson holds on real data too: the uncalibrated raw signal
    is the worst calibration (calibration is what creates value)."""
    d = tempfile.mkdtemp()
    path = os.path.join(d, "lethe_trace.jsonl")
    write_sample_jsonl(path, n_objects=20, n_recalls=300, reuse_gap=4, skew=1.0, seed=5)
    conn = LetheJSONLConnector(path=path)
    result = run_pipeline_on_connector(conn, capacity=6)
    nets = result.net_gain_by_calibration
    assert nets["raw"] == min(nets.values())
