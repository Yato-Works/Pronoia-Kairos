"""Phase 6 follow-up 1-5: F内訳分解 / 統計 / W1分析 / 実トレース / Tier4."""
from __future__ import annotations

import json
import math
import os
import random
from typing import Dict, List, Sequence

from compare.full_runtime_experiment import (
    SEEDS, WIN_TOL, build_family, run_f_breakdown, run_family,
)
from runtime.adaptive_runtime import AdaptiveRuntime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_DIR = os.path.join(ROOT, "artifacts", "phase6")
OUT_FOLLOWUP = os.path.join(OUT_DIR, "followup_12345.json")


def sign_test(diffs: Sequence[float], tol: float = WIN_TOL) -> Dict:
    """paired符号検定。|d|<=tol はタイとして除外 (W2/W3完全タイ対応)。"""
    wins = sum(1 for d in diffs if d > tol)
    losses = sum(1 for d in diffs if d < -tol)
    ties = len(diffs) - wins - losses
    n = wins + losses
    if n == 0:
        p = 1.0
    else:
        k = max(wins, losses)
        tail = sum(math.comb(n, i) for i in range(k, n + 1)) / (2 ** n)
        p = min(1.0, 2.0 * tail)
    return {"wins": wins, "losses": losses, "ties": ties, "n": n,
            "p_value": p}


def bootstrap_ci95(diffs: Sequence[float], n_boot: int = 10000,
                   seed: int = 0) -> Dict:
    """平均deltaのbootstrap 95%CI (percentile法、純Python)。"""
    rng = random.Random(seed)
    n = len(diffs)
    if n == 0:
        return {"mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    mean = sum(diffs) / n
    if n == 1:
        return {"mean": mean, "ci95_low": mean, "ci95_high": mean}
    boots = []
    for _ in range(n_boot):
        s = sum(diffs[rng.randrange(n)] for _ in range(n)) / n
        boots.append(s)
    boots.sort()
    lo = boots[int(0.025 * n_boot)]
    hi = boots[min(n_boot - 1, int(0.975 * n_boot))]
    return {"mean": mean, "ci95_low": lo, "ci95_high": hi}


def holm_correct(p_values: Sequence[float]) -> List[float]:
    """Holm逐次Bonferroni補正 (多重比較のfamily-wise error制御)。

    8家族で符号検定を回すとp<0.05の誤検出が増えるため、確認的評価では
    補正後pを併記する。探索的 (uncorrected) / 確認的 (Holm) の2段表示。
    """
    n = len(p_values)
    order = sorted(range(n), key=lambda i: p_values[i])
    adj = [0.0] * n
    for rank, idx in enumerate(order):
        adj[idx] = min(1.0, p_values[idx] * (n - rank))
    # 単調性保証 (小さいpの補正値が大きいpを下回らない)
    for k in range(n - 2, -1, -1):
        idx, nxt = order[k], order[k + 1]
        adj[idx] = min(adj[idx], adj[nxt])
    return adj


def win_sensitivity(per_seed_deltas: Sequence[float]) -> Dict:
    """WIN_TOL感度: 1e-9 / 1e-6 / 1e-5 で勝敗表がどう変わるか。"""
    out = {}
    for tol in (1e-9, 1e-6, 1e-5):
        st = sign_test(per_seed_deltas, tol=tol)
        out[f"tol_{tol:g}"] = {"wins": st["wins"], "losses": st["losses"],
                               "ties": st["ties"], "p": st["p_value"]}
    return out


def stats_for_family(family: str,
                     seeds: Sequence[int] = SEEDS) -> Dict:
    """F vs strong の per-seed delta に符号検定 + bootstrap CI。"""
    fam = run_family(family, seeds)
    diffs = [float(d) for d in fam["f_delta_vs_strong_per_seed"]]
    st = sign_test(diffs)
    ci = bootstrap_ci95(diffs)
    bd = run_f_breakdown(family, seeds)
    return {
        "family": family,
        "strong_baseline": fam["strong_baseline"],
        "f_delta_mean": fam["f_delta_vs_strong_mean"],
        "deltas": diffs,
        "sign_test": st,
        "bootstrap_ci95": ci,
        "significant_05": bool(st["p_value"] < 0.05
                               and st["wins"] > st["losses"]),
        "ci_excludes_zero": bool(ci["ci95_low"] > 0 or ci["ci95_high"] < 0),
        "edf_gain_mean": bd["edf_gain_mean"],
        "prefetch_gain_mean": bd["prefetch_gain_mean"],
        "edf_gain_per_seed": bd["edf_gain_per_seed"],
        "prefetch_gain_per_seed": bd["prefetch_gain_per_seed"],
    }


def w1_overhead(seeds: Sequence[int] = SEEDS) -> Dict:
    """W1でFがAに負ける理由の分解 (hit率 / 追放 / prefetch廃棄)。"""
    from compare.full_runtime_experiment import ladder
    rows = []
    for seed in seeds:
        stream, spec = build_family("W1_short_chat_low_reuse", seed)
        cfgs = {c.name: c for c in ladder(spec)}
        rec = {}
        for name in ("A_naive", "E_planning", "F_full"):
            r = AdaptiveRuntime(cfgs[name]).serve(list(stream))
            rec[name] = {
                "slo_met": r.slo_met, "makespan": r.makespan,
                "ttft_p95": r.ttft_p95, "hit_rate": r.hit_rate,
                "evictions": r.evictions, "prefetches": r.prefetches,
                "prefetch_hits": r.prefetch_hits,
                "prefetch_waste": r.prefetch_waste,
                "goodput": r.goodput,
            }
        rows.append(rec)
    out: Dict = {"seeds": list(seeds), "per_seed": rows}
    for k in ("makespan", "ttft_p95", "hit_rate", "evictions", "goodput"):
        a = [r["A_naive"][k] for r in rows]
        e = [r["E_planning"][k] for r in rows]
        f = [r["F_full"][k] for r in rows]
        out[k] = {"A_mean": sum(a) / len(a), "E_mean": sum(e) / len(e),
                  "F_mean": sum(f) / len(f)}
    out["prefetch"] = {
        "issued_total": sum(r["F_full"]["prefetches"] for r in rows),
        "hits_total": sum(r["F_full"]["prefetch_hits"] for r in rows),
        "waste_total": sum(r["F_full"]["prefetch_waste"] for r in rows),
    }
    return out


def make_realistic_trace_stream(n_requests: int = 60, seed: int = 0) -> List:
    """実会話長ライクなtrace: 対数正規風の長さ + Zipf話題再訪。

    合成W1-W8ではなく公開会話データの統計的性質
    (短turn多数・長いtail・話題skew) を模した外部妥当性チェック用。
    """
    from runtime.adaptive_runtime import Chunk, WorkloadRequest
    rng = random.Random(1000 + seed)
    topics = [f"real-topic-{k}" for k in range(12)]
    weights = [1.0 / ((i + 1) ** 1.2) for i in range(len(topics))]
    reqs = []
    t = 0.0
    for i in range(n_requests):
        # 中負荷: 平均36、サービス~50 → 時々キュー (W8帯)
        t += rng.uniform(24.0, 48.0)
        topic = rng.choices(topics, weights=weights, k=1)[0]
        total = int(min(2200, max(128, rng.lognormvariate(6.0, 0.8))))
        n_chunks = max(2, total // 160)
        chunks = tuple(
            Chunk(chunk_id=f"{topic}-c{j}" if j < 2 else f"real-n{i}-{j}",
                  tokens=160,
                  benefit=round(rng.uniform(0.3, 0.9), 3),
                  quality_loss_if_dropped=round(rng.uniform(0.002, 0.01), 4))
            for j in range(n_chunks))
        # SLOはサービス時間帯に締める (EDFの並び替え余地を作る)
        slo = 140.0 if rng.random() < 0.5 else 90.0
        reqs.append(WorkloadRequest(f"real-r{i}", t, topic, 256, chunks, 64,
                                    slo))
    return reqs


def run_realistic_trace(seeds: Sequence[int] = (1, 2, 3, 4, 5, 6)) -> Dict:
    """実トレース風streamでのA-F ladder比較 (paired)。"""
    from compare.full_runtime_experiment import (FamilySpec, ladder,
                                                 _config_stats)
    spec = FamilySpec("realistic_trace", 60, 2048, 1536, 90.0, 90.0, 0.0)
    per_cfg: Dict = {}
    deltas = []
    strongs = []
    for seed in seeds:
        stream = make_realistic_trace_stream(60, seed)
        row = {}
        for cfg in ladder(spec):
            res = AdaptiveRuntime(cfg).serve(list(stream))
            per_cfg.setdefault(cfg.name, []).append(res)
            row[cfg.name] = res.goodput
        cands = {k: v for k, v in row.items() if k != "F_full"}
        strong = max(cands, key=lambda n: cands[n])
        strongs.append(strong)
        deltas.append(row["F_full"] - row[strong])
    stats = {n: _config_stats(per_cfg[n]) for n in per_cfg}
    return {"spec": "realistic_trace/60req",
            "goodput_mean": {k: v["goodput_mean"] for k, v in stats.items()},
            "strong_per_seed": strongs,
            "f_delta_per_seed": [round(d, 6) for d in deltas],
            "f_delta_mean": sum(deltas) / len(deltas),
            "sign_test": sign_test(deltas),
            "bootstrap_ci95": bootstrap_ci95(deltas)}


def tier4_connector_probe() -> Dict:
    """Tier4最小構成: vLLM connector上でのprefix hit率測定。

    実vLLMなし (mock backend) だが、connector seam 上での
    Adaptive vs LRU vs LMCache-like の3者比較 = Tier4への最小橋渡し。
    """
    from compare.vllm_connector_experiment import run_phase1_gate
    cmp = run_phase1_gate(device_capacity_blocks=4, n_sessions=6,
                          turns_per_session=10, n_distinct_prefixes=8)
    return {"results": [r.__dict__ for r in cmp.results],
            "adaptive_wins": cmp.adaptive_wins(),
            "best_hit_rate_policy": cmp.best_hit_rate().policy,
            "best_ttft_policy": cmp.best_ttft().policy}

