"""Phase 2 Section 20 - semantic prefetch controller (Plan.md Section 20, 35, 65).

Covers:
    - Section 20 prefetch inequality: fire iff P_use*T_saved > T_transfer+T_interference.
    - Section 20 honesty: a prefetch miss is charged as cost (waste accounting).
    - Section 65 metrics: prefetch accuracy / waste rate.
    - End-to-end: a prefetched object reused before eviction counts as a hit;
      one evicted before reuse is counted as waste.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cache.prefetch import PrefetchController, PrefetchStats
from simulator.latency import LatencyModel
from simulator.discrete import DiscreteEventSimulator, TraceRequest


# --- Section 20: the prefetch inequality -------------------------------------

def test_prefetch_fires_only_when_expected_gain_exceeds_cost():
    """Section 20: approve iff P_use*T_saved > T_transfer + T_interference."""
    ctrl = PrefetchController(LatencyModel())
    ctrl.register_size("k", 512)
    d = ctrl.evaluate("k", p_use=1.0)  # certain reuse
    assert d.approved
    assert d.expected_gain > d.cost
    # A very unlikely reuse must not clear the cost bar.
    d_low = ctrl.evaluate("k", p_use=0.01)
    assert not d_low.approved
    assert d_low.expected_gain < d_low.cost


def test_tiny_object_may_not_clear_fixed_cost():
    """Section 20: for tiny objects the fixed restore/interference dominates,
    so even a certain reuse may not pay -> rejected."""
    ctrl = PrefetchController(LatencyModel())
    ctrl.register_size("small", 4)
    d = ctrl.evaluate("small", p_use=1.0)
    assert not d.approved
    assert ctrl.stats.rejected >= 1


def test_size_aware_cost_and_saving():
    """T_saved (prefill) and T_transfer scale with object size."""
    lm = LatencyModel()
    ctrl = PrefetchController(lm)
    ctrl.register_size("big", 2048)
    ctrl.register_size("small", 128)
    d_big = ctrl.evaluate("big", 1.0)
    d_small = ctrl.evaluate("small", 1.0)
    assert d_big.t_saved > d_small.t_saved
    assert d_big.t_transfer > d_small.t_transfer


def test_latency_prefetch_cost_decomposition():
    """LatencyModel exposes the three Section 20 costs consistently."""
    lm = LatencyModel()
    tokens = 512
    assert lm.saved_recompute(tokens) == lm.t_prefill_per_token * tokens
    assert lm.transfer_cost(tokens) == lm.t_transfer_per_token * tokens + lm.t_restore
    assert lm.interference_cost(tokens) == lm.t_restore + lm.t_transfer_per_token * tokens
    # Reuse (transfer) is far cheaper than recompute (prefill) for big prefixes.
    assert lm.transfer_cost(4096) < lm.saved_recompute(4096)


# --- Section 65: accuracy / waste metrics ------------------------------------

def test_stats_accuracy_and_waste_rate():
    s = PrefetchStats()
    assert s.accuracy == 0.0
    assert s.waste_rate == 0.0
    s.issued = 4
    s.hits = 2
    s.wastes = 1
    assert s.accuracy == 0.5
    assert s.waste_rate == 0.25


def test_on_hit_credits_saved_compute():
    lm = LatencyModel()
    ctrl = PrefetchController(lm)
    ctrl.register_size("k", 512)
    ctrl.on_issued("k")
    ctrl.on_hit("k")
    assert ctrl.stats.hits == 1
    assert ctrl.stats.saved_compute == lm.saved_recompute(512)
    assert ctrl.stats.net_gain == lm.saved_recompute(512)


def test_on_waste_charges_interference_only():
    """Section 36 overlap hides the wasted transfer behind compute, so the
    charged waste is the interference (GPU pollution), not the transfer."""
    lm = LatencyModel()
    ctrl = PrefetchController(lm)
    ctrl.register_size("k", 512)
    ctrl.on_issued("k")
    ctrl.on_waste("k")
    assert ctrl.stats.wastes == 1
    assert ctrl.stats.waste_cost == lm.interference_cost(512)
    assert ctrl.stats.net_gain < 0  # pure waste is a net loss


# --- End-to-end simulator integration ----------------------------------------

def _trace(keys, tokens=256):
    return [
        TraceRequest(request_id=f"r{i}", key=k, arrival_time=float(i) * 5.0,
                     input_tokens=tokens, output_tokens=32)
        for i, k in enumerate(keys)
    ]


def test_prefetched_object_reused_counts_as_hit():
    """Predict 'd', fetch it during r1, reuse at r3 -> prefetch hit."""
    lm = LatencyModel()
    ctrl = PrefetchController(lm)
    for k in ("a", "b", "c", "d"):
        ctrl.register_size(k, 256)
    sim = DiscreteEventSimulator(
        capacity=4, latency=lm, prefetch_controller=ctrl,
        demand_forecast={"d": 0.9},
    )
    m = sim.run(_trace(["a", "b", "d"]))
    assert m.prefetch_hits >= 1
    assert ctrl.stats.hits >= 1
    d_rec = [r for r in m.records if r.key == "d"][0]
    assert d_rec.hit and d_rec.prefetched


def test_prefetch_not_issued_without_controller():
    """No controller -> the legacy no-op prefetch path (zero prefetches)."""
    sim = DiscreteEventSimulator(capacity=4)
    m = sim.run(_trace(["a", "b", "d"]))
    assert m.prefetches == 0


def test_same_object_not_transferred_twice():
    """One object is prefetched at most once (in-flight / done de-duplication)."""
    lm = LatencyModel()
    ctrl = PrefetchController(lm)
    ctrl.register_size("z", 256)
    sim = DiscreteEventSimulator(
        capacity=4, latency=lm, prefetch_controller=ctrl,
        demand_forecast={"z": 0.9},
    )
    m = sim.run(_trace(["a", "b", "c", "d", "e"]))
    assert m.prefetches <= 1


def test_prefetch_off_and_on_same_hits_when_no_demand():
    """With an empty forecast, enabling the controller changes nothing."""
    lm = LatencyModel()
    ctrl = PrefetchController(lm)
    sim_on = DiscreteEventSimulator(
        capacity=4, latency=lm, prefetch_controller=ctrl, demand_forecast={})
    sim_off = DiscreteEventSimulator(capacity=4)
    keys = ["a", "b", "c", "a", "b"]
    m_on = sim_on.run(_trace(keys))
    m_off = sim_off.run(_trace(keys))
    assert m_on.hits == m_off.hits
    assert m_on.prefetches == 0


# --- Section 52: NetGain gate (end-to-end sweep) -----------------------------

def test_netgain_positive_when_accurate_negative_when_poor():
    """Core Section 52 result: prefetch NetGain > 0 while prediction is
    accurate, and collapses as prediction degrades."""
    from compare.prefetch_experiment import run_prefetch_sweep
    from simulator import workloads as w
    seq = w.zipf_like(60, 400, seed=3)
    results = run_prefetch_sweep(seq, capacity=20, horizon=3)
    accurate = [r for r in results if r.sigma <= 0.5]
    poor = [r for r in results if r.sigma >= 2.0]
    assert accurate and poor
    assert all(r.net_gain > 0 for r in accurate)
    assert min(r.net_gain for r in poor) < min(r.net_gain for r in accurate)


def test_netgain_formula_matches_components():
    """NetGain = SavedCompute - PredictionOverhead - PrefetchWaste (Section 52)."""
    from compare.prefetch_experiment import (
        run_prefetch_sweep, PREDICTION_OVERHEAD_PER_DECISION)
    from simulator import workloads as w
    seq = w.zipf_like(40, 200, seed=5)
    results = run_prefetch_sweep(seq, capacity=16, horizon=3, sigmas=[0.0, 3.0])
    for r in results:
        expected = (r.saved_compute - r.prediction_overhead - r.prefetch_waste_cost)
        assert abs(r.net_gain - expected) < 1e-9
        assert r.prediction_overhead == len(seq) * PREDICTION_OVERHEAD_PER_DECISION


def test_forecast_recall_degrades_with_sigma():
    """Larger sigma (worse predictor) must miss more true-demand keys."""
    from compare.prefetch_experiment import noisy_forecast
    from simulator import workloads as w
    seq = w.zipf_like(40, 300, seed=3)
    good = noisy_forecast(seq, horizon=3, sigma=0.0)
    bad = noisy_forecast(seq, horizon=3, sigma=4.0)
    good_hot = sum(1 for v in good.values() if v >= 0.9)
    bad_hot = sum(1 for v in bad.values() if v >= 0.9)
    assert good_hot > bad_hot


def test_no_prefetch_baseline_is_sigma_independent():
    """The OFF baseline is computed once and reused (identical across sigmas)."""
    from compare.prefetch_experiment import run_prefetch_sweep
    from simulator import workloads as w
    seq = w.zipf_like(40, 200, seed=7)
    results = run_prefetch_sweep(seq, capacity=16, horizon=3, sigmas=[0.0, 1.0, 2.0])
    assert len({r.hits_off for r in results}) == 1
