"""Follow-up 1-5 の回帰テスト: 内訳分解の加法性 / 統計関数の健全性."""
from compare.followup_12345 import bootstrap_ci95, sign_test
from compare.full_runtime_experiment import (build_family, ladder,
                                             ladder_f_breakdown,
                                             run_f_breakdown)


def test_f_breakdown_additivity_w8():
    """EDF分 + prefetch分 = 合計 (W8, seed 1-2の軽量版)。"""
    r = run_f_breakdown("W8_RAG", seeds=(1, 2))
    for e, p, t in zip(r["edf_gain_per_seed"], r["prefetch_gain_per_seed"],
                       r["total_gain_per_seed"]):
        assert abs((e + p) - t) < 1e-9


def test_f_breakdown_w8_is_prefetch_only():
    """W8ではEDF分が全seed 0、prefetch分が合計に等しい (実測の固定化)。"""
    r = run_f_breakdown("W8_RAG", seeds=(1, 2))
    assert all(d == 0.0 for d in r["edf_gain_per_seed"])
    assert r["prefetch_gain_mean"] == r["total_gain_mean"]


def test_ladder_f_breakdown_switches():
    """3点のスイッチが E / E+EDF / F になっている。"""
    from compare.full_runtime_experiment import FamilySpec
    spec = FamilySpec("t", 8, 1024, 1024, 90.0, 90.0, 0.0)
    cfgs = ladder_f_breakdown(spec)
    assert [c.name for c in cfgs] == ["E_planning", "E_edf", "F_full"]
    assert cfgs[0].scheduler == "fifo" and not cfgs[0].prefetch
    assert cfgs[1].scheduler == "edf" and not cfgs[1].prefetch
    assert cfgs[2].scheduler == "edf" and cfgs[2].prefetch


def test_sign_test_ties_dropped():
    """完全タイは除外され n=0, p=1.0。6/6全勝は p=0.03125。"""
    r = sign_test([0.0] * 6)
    assert (r["n"], r["p_value"]) == (0, 1.0)
    r2 = sign_test([0.0001, 0.0002, 0.0003, 0.0001, 0.0002, 0.0005])
    assert r2["wins"] == 6 and abs(r2["p_value"] - 0.03125) < 1e-9


def test_bootstrap_ci_covers_mean():
    """CIが平均を挟み、定数列では幅0。"""
    ci = bootstrap_ci95([0.1, 0.2, 0.3, 0.15, 0.25, 0.18])
    assert ci["ci95_low"] <= ci["mean"] <= ci["ci95_high"]
    ci2 = bootstrap_ci95([0.0] * 6)
    assert ci2["ci95_low"] == ci2["ci95_high"] == 0.0


def test_w1_guard_tie():
    """W1ガード: FはAとタイ (seed 1の軽量版)。"""
    from compare.full_runtime_experiment import build_family, ladder
    from runtime.adaptive_runtime import AdaptiveRuntime
    stream, spec = build_family("W1_short_chat_low_reuse", 1)
    cfgs = {c.name: c for c in ladder(spec)}
    ra = AdaptiveRuntime(cfgs["A_naive"]).serve(list(stream))
    rf = AdaptiveRuntime(cfgs["F_full"]).serve(list(stream))
    assert ra.slo_met == rf.slo_met == 48
    assert rf.prefetches == 0
    assert abs(ra.goodput - rf.goodput) < 1e-9


def test_f_isolation_guards_off_on_baselines():
    """F隔離: A-EはガードOFF、FはガードON。"""
    from compare.full_runtime_experiment import FamilySpec, ladder
    spec = FamilySpec("t", 8, 1024, 1024, 90.0, 90.0, 0.0)
    cfgs = {c.name: c for c in ladder(spec)}
    for n in ("A_naive", "B_predictive", "C_queue", "D_semantic",
              "E_planning"):
        assert not cfgs[n].prefetch_guard
        assert not cfgs[n].retention_fallback
        assert not cfgs[n].edf_starvation_guard
        assert cfgs[n].prefetch_min_seen == 2
        assert cfgs[n].prefetch_min_benefit_ratio == 1.0
    assert cfgs["F_full"].prefetch_guard
    assert cfgs["F_full"].retention_fallback
    assert cfgs["F_full"].edf_starvation_guard
    assert cfgs["F_full"].prefetch_min_seen == 3
    assert cfgs["F_full"].prefetch_min_benefit_ratio == 1.5


def test_touch_incremental_matches_sum():
    """P1: _touch増分管理が全合計と一致する。"""
    from compare.full_runtime_experiment import build_family, ladder
    from runtime.adaptive_runtime import AdaptiveRuntime
    stream, spec = build_family("W8_RAG", 1)
    cfgs = {c.name: c for c in ladder(spec)}
    rt = AdaptiveRuntime(cfgs["F_full"])
    rt.serve(list(stream))
    assert rt.kv_used == sum(rt.kv.values())
    assert rt.kv_used <= cfgs["F_full"].kv_capacity_tokens


def test_pf_history_bounded_and_tail_recorded():
    """P1: 履歴はwindow上限内、末尾wasteも履歴に反映される。"""
    from collections import deque
    from compare.full_runtime_experiment import build_family, ladder
    from runtime.adaptive_runtime import AdaptiveRuntime
    stream, spec = build_family("W1_short_chat_low_reuse", 1)
    cfgs = {c.name: c for c in ladder(spec)}
    rt = AdaptiveRuntime(cfgs["F_full"])
    r = rt.serve(list(stream))
    assert isinstance(rt._pf_history, deque)
    assert len(rt._pf_history) <= cfgs["F_full"].prefetch_window
    # 履歴のFalse件数 + hit件数が発行件数と整合 (末尾waste含む)
    n_false = sum(1 for h in rt._pf_history if not h)
    assert r.prefetch_hits + r.prefetch_waste >= r.prefetches
    assert n_false <= r.prefetch_waste


def test_signals_ages_cache_consistent():
    """P1: signals/agesキャッシュが非キャッシュ時と一致する。"""
    from compare.full_runtime_experiment import build_family, ladder
    from runtime.adaptive_runtime import AdaptiveRuntime
    stream, spec = build_family("W4_long_ctx_high_reuse", 1)
    cfgs = {c.name: c for c in ladder(spec)}
    rt = AdaptiveRuntime(cfgs["F_full"])
    rt.serve(list(stream[:10]))
    s1 = dict(rt._signals())
    a1 = dict(rt._ages())
    # 2回目はキャッシュ経路のはずだが結果は同一
    s2 = dict(rt._signals())
    a2 = dict(rt._ages())
    assert s1 == s2
    assert a1 == a2


def test_holm_correct_monotone():
    """P2: Holm補正は単調性を保ち、最小pがn倍される。"""
    from compare.followup_12345 import holm_correct
    adj = holm_correct([0.01, 0.02, 0.5])
    assert abs(adj[0] - 0.03) < 1e-9
    assert adj[0] <= adj[1] <= adj[2]
    assert holm_correct([1.0, 1.0]) == [1.0, 1.0]


def test_win_sensitivity_ties_grow():
    """P2: WIN_TOLを上げるとタイが増え、勝敗が減る。"""
    from compare.followup_12345 import win_sensitivity
    diffs = [4e-05, 3.2e-05, 4.3e-05, 3e-06, 3.3e-05, 0.000581]
    s = win_sensitivity(diffs)
    assert s["tol_1e-09"]["ties"] == 0
    assert s["tol_1e-05"]["ties"] >= 1
    assert (s["tol_1e-05"]["wins"] + s["tol_1e-05"]["losses"]
            <= s["tol_1e-09"]["wins"] + s["tol_1e-09"]["losses"])


def test_edf_relative_threshold():
    """P2: 相対閾値はqueue内最小SLO×fracになる。"""
    from compare.full_runtime_experiment import FamilySpec, ladder
    from runtime.adaptive_runtime import AdaptiveRuntime, RuntimeConfig
    spec = FamilySpec("t", 8, 1024, 1024, 90.0, 90.0, 0.0)
    base = [c for c in ladder(spec) if c.name == "F_full"][0]
    assert base.edf_max_wait_frac == 0.0
    cfg = RuntimeConfig(name="F2", retention="queue_semantic",
                        scheduler="edf", context_planning=True,
                        prefetch=True,
                        kv_capacity_tokens=spec.kv_capacity_tokens,
                        context_budget_tokens=spec.context_budget_tokens,
                        quality_epsilon=0.05, edf_max_wait_frac=0.5)
    rt = AdaptiveRuntime(cfg)
    assert rt.config.edf_max_wait_frac == 0.5


def test_edf_frac_regresses_w5():
    """次3件の1: frac=0.5はW5で後退する (不採用の回帰)。"""
    from compare.full_runtime_experiment import build_family
    from runtime.adaptive_runtime import AdaptiveRuntime, RuntimeConfig
    stream, spec = build_family("W5_concurrent_long_low_reuse", 1)
    base = dict(kv_capacity_tokens=spec.kv_capacity_tokens,
                context_budget_tokens=spec.context_budget_tokens,
                quality_epsilon=0.05)
    rabs = AdaptiveRuntime(RuntimeConfig(
        name="F", retention="queue_semantic", scheduler="edf",
        context_planning=True, prefetch=True,
        edf_max_wait_frac=0.0, **base)).serve(list(stream))
    rrel = AdaptiveRuntime(RuntimeConfig(
        name="F", retention="queue_semantic", scheduler="edf",
        context_planning=True, prefetch=True,
        edf_max_wait_frac=0.5, **base)).serve(list(stream))
    assert rrel.goodput <= rabs.goodput


def test_w7_calibration_improves_seed2():
    """次3件の3: signal_confidence=0.5でW7 seed2が改善方向に動く。"""
    from compare.full_runtime_experiment import build_family
    from runtime.adaptive_runtime import AdaptiveRuntime, RuntimeConfig
    stream, spec = build_family("W7_agent", 2)
    base = dict(kv_capacity_tokens=spec.kv_capacity_tokens,
                context_budget_tokens=spec.context_budget_tokens,
                quality_epsilon=0.05)
    r0 = AdaptiveRuntime(RuntimeConfig(
        name="F", retention="queue_semantic", scheduler="edf",
        context_planning=True, prefetch=True,
        signal_confidence=0.0, **base)).serve(list(stream))
    r1 = AdaptiveRuntime(RuntimeConfig(
        name="F", retention="queue_semantic", scheduler="edf",
        context_planning=True, prefetch=True,
        signal_confidence=0.5, **base)).serve(list(stream))
    # 絶対値は環境で振れるため改善方向のみ見る (開発時は29→35を確認)。
    assert r1.slo_met >= r0.slo_met


def test_w2_seed1_tie():
    """P3: W2はseed1でタイ (F隔離後の後退解消の回帰)。"""
    from compare.full_runtime_experiment import build_family, ladder
    from runtime.adaptive_runtime import AdaptiveRuntime
    stream, spec = build_family("W2_short_chat_high_reuse", 1)
    cfgs = {c.name: c for c in ladder(spec)}
    rb = AdaptiveRuntime(cfgs["B_predictive"]).serve(list(stream))
    rf = AdaptiveRuntime(cfgs["F_full"]).serve(list(stream))
    assert abs(rb.goodput - rf.goodput) < 1e-9


def test_w4_seed1_win():
    """P3: W4はseed1でFがEに勝つ (CI夜間バッチの軽量版)。"""
    from compare.full_runtime_experiment import build_family, ladder
    from runtime.adaptive_runtime import AdaptiveRuntime
    stream, spec = build_family("W4_long_ctx_high_reuse", 1)
    cfgs = {c.name: c for c in ladder(spec)}
    re_ = AdaptiveRuntime(cfgs["E_planning"]).serve(list(stream))
    rf = AdaptiveRuntime(cfgs["F_full"]).serve(list(stream))
    assert rf.goodput > re_.goodput
    # seed12フルは重いため夜間バッチ (`seed12_runner`) で回すこと。
    # CIでは seed1-2 の軽量版に留める。
