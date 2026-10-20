"""Task#1: W7較正連動 (signal_confidence) の全体適用評価。

F_full の signal_confidence を {0.0(現状), 0.5, 1.0} に変えたときの
全8家族 goodput / SLO / prefetch 効率を paired (§69) に測定する。
A-E は guards_off で常に conf=0.0 のため strong基準は不変。よって
F_full の goodput 変化 = 全体適用の純効果。

結論: 採用/不採用を Fデフォルト更新と gate再確認に反映する。
"""
from __future__ import annotations

import json
import os
import statistics
from dataclasses import replace

from compare.full_runtime_experiment import (
    FAMILY_NAMES, build_family, ladder)
from runtime.adaptive_runtime import AdaptiveRuntime, RuntimeConfig

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "artifacts", "phase6", "w7_global_eval.json")

SEEDS = (1, 2, 3, 4, 5, 6)
CONFS = (0.0, 0.5, 1.0)


def f_full_at(spec, conf: float) -> RuntimeConfig:
    """F_full と同じ装備で signal_confidence のみ変えた config。"""
    base = [c for c in ladder(spec) if c.name == "F_full"][0]
    return replace(base, name=f"F_conf{conf}", signal_confidence=conf)


def main() -> None:
    rows = []
    for fam in FAMILY_NAMES:
        # strong基準 (A-E) は conf非依存なので seedごとに1回だけ計算
        strong_per_seed = []
        f_by_conf = {c: [] for c in CONFS}
        pfw_by_conf = {c: [] for c in CONFS}   # prefetch waste/hit 比
        slo_by_conf = {c: [] for c in CONFS}
        for seed in SEEDS:
            stream, spec = build_family(fam, seed)
            # A-E strong baseline (conf=0.0 固定)
            a_best = 0.0
            for cfg in ladder(spec):
                if cfg.name == "F_full":
                    continue
                r = AdaptiveRuntime(cfg).serve(list(stream))
                a_best = max(a_best, r.goodput)
            strong_per_seed.append(a_best)
            for conf in CONFS:
                r = AdaptiveRuntime(f_full_at(spec, conf)).serve(list(stream))
                f_by_conf[conf].append(r.goodput)
                slo_by_conf[conf].append(r.slo_met)
                denom = r.prefetch_hits + r.prefetch_waste
                pfw_by_conf[conf].append(
                    r.prefetch_hits / denom if denom else 0.0)
        row = {"family": fam, "strong_mean": statistics.fmean(strong_per_seed)}
        for conf in CONFS:
            g = f_by_conf[conf]
            delta = [g[i] - strong_per_seed[i] for i in range(len(SEEDS))]
            row[f"conf{conf}"] = {
                "goodput_mean": statistics.fmean(g),
                "delta_vs_strong_mean": statistics.fmean(delta),
                "slo_met_mean": statistics.fmean(slo_by_conf[conf]),
                "pf_hit_rate_mean": statistics.fmean(pfw_by_conf[conf]),
            }
        # 相対: conf0.5/1.0 vs conf0.0 (現状) の goodput 変化
        base_g = row["conf0.0"]["goodput_mean"]
        for conf in (0.5, 1.0):
            row[f"conf{conf}"]["d_goodput_vs_conf0"] = (
                row[f"conf{conf}"]["goodput_mean"] - base_g)
        rows.append(row)
        print(fam,
              "conf0=", round(row["conf0.0"]["goodput_mean"], 5),
              "conf0.5=", round(row["conf0.5"]["goodput_mean"], 5),
              "conf1=", round(row["conf1.0"]["goodput_mean"], 5),
              "| d0.5=", round(row["conf0.5"]["d_goodput_vs_conf0"], 6),
              "d1=", round(row["conf1.0"]["d_goodput_vs_conf0"], 6),
              flush=True)

    # 全家族サマリ: 全体適用で改善/後退した家族数
    for conf in (0.5, 1.0):
        improved = [r["family"] for r in rows
                    if r[f"conf{conf}"]["d_goodput_vs_conf0"] > 1e-9]
        regressed = [r["family"] for r in rows
                     if r[f"conf{conf}"]["d_goodput_vs_conf0"] < -1e-9]
        print(f"[conf={conf}] improved={len(improved)}{improved} "
              f"regressed={len(regressed)}{regressed}", flush=True)

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({"seeds": list(SEEDS), "confs": list(CONFS), "rows": rows},
                  fh, indent=2)
    print("saved", OUT, flush=True)


if __name__ == "__main__":
    main()
