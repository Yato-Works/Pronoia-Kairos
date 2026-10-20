"""Section 35 prediction metrics + Section 49/50 semantic demand dataset."""
import sys, os, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from semantic.prediction_metrics import (
    precision_recall, roc_auc, expected_calibration_error, topk_precision, mrr,
)
from semantic.demand_dataset import (
    DemandSample, DemandDatasetRecorder,
    future_use_distance_within, horizon_labels,
)


def test_precision_recall_basic():
    labels = [1, 1, 0, 0, 1]
    preds  = [1, 0, 0, 1, 1]
    prec, rec = precision_recall(labels, preds)   # tp=2 fp=1 fn=1
    assert abs(prec - 2/3) < 1e-9
    assert abs(rec - 2/3) < 1e-9


def test_precision_recall_no_predictions():
    prec, rec = precision_recall([1, 0], [0, 0])
    assert prec == 0.0 and rec == 0.0


def test_auc_perfect_and_random():
    assert roc_auc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0
    assert roc_auc([0.1, 0.9, 0.2, 0.8], [1, 1, 0, 0]) == 0.5
    assert roc_auc([0.5, 0.5], [1, 0]) == 0.5     # degenerate tie


def test_ece_perfect_calibration_zero():
    # 4 predictions of 0.25 that are right 25% of the time -> ECE 0
    probs = [0.25]*40
    labels = [1]*10 + [0]*30
    assert expected_calibration_error(probs, labels) < 0.01


def test_ece_empty_is_zero():
    assert expected_calibration_error([], []) == 0.0


def test_topk_precision_and_mrr():
    ranked = ["a", "b", "c", "d"]
    relevant = {"c"}
    assert topk_precision(ranked, relevant, 1) == 0.0
    assert topk_precision(ranked, relevant, 3) == 1/3
    assert mrr(ranked, relevant) == 1/3
    assert mrr(["x", "y"], relevant) == 0.0


def test_future_use_distance_and_horizons():
    seq = ["A", "B", "A", "C"]
    assert future_use_distance_within(seq, 0, "A") == 2
    assert future_use_distance_within(seq, 1, "B") is None
    labels = horizon_labels(seq, 0, "A")   # distance 2 -> outside H1, inside H2
    assert labels["H1"] is False and labels["H2"] is True


def test_recorder_roundtrip():
    rec = DemandDatasetRecorder()
    rec.record(DemandSample(
        session_id="s1", request_id="r1", memory_id="m1", chunk_id="c1",
        timestamp=1.0, recall_event=True, predicted_probability=0.9,
        actual_usage=True, cache_hit=True, future_use_distance=2,
    ))
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "demand.jsonl")
        rec.save(path)
        loaded = DemandDatasetRecorder.load(path)
        assert len(loaded) == 1
        s = loaded.samples[0]
        assert s.session_id == "s1" and s.predicted_probability == 0.9
        assert s.future_use_distance == 2
    assert len(rec.by_session("s1")) == 1
    assert len(rec.by_session("nope")) == 0
