# strong_reeval: M1 Strong Baseline再評価 (PREREG_STRONG_REEVAL.md 固定条件)
"""Mentor milestone 1: F vs best-of-A-E, seeds 1-12, relative improvement.

事前登録: artifacts/phase6/PREREG_STRONG_REEVAL.md (実行前固定、改変禁止)。
既存 artifact は一切書き換えない。出力は strong_reeval.json のみ。

  1. 各家族で A〜E の最良 (goodput mean) を strong として特定 (§62)
  2. F と同一12-seed paired 比較: 絶対delta、相対改善率%、CI、符号検定
  3. Holm 補正で確認的/探索的を区別
  4. Tier 2 判定: 家族毎 rel% >= +20% が 5/8 以上で「vs strong で達成」
  5. F/A 比は診断値として併記 (Tier 判定には使わない)
  6. 保守性チェック: abs dmean が seed12_merged.json と 1e-6 一致

Usage: python -m compare.strong_reeval
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_DIR = os.path.join(ROOT, "artifacts", "phase6")

SEEDS = tuple(range(1, 13))
REL_TARGET = 0.20          # §94 Tier 2: +20%
TIER2_MIN_FAMILIES = 5     # §93 majority と同一閾値
CONSISTENCY_TOL = 1e-6     # seed12_merged との一致許容差 (事前登録)


def main() -> int:
    from compare.followup_12345 import bootstrap_ci95, holm_correct, sign_test
    from compare.full_runtime_experiment import (BASELINES, FAMILY_NAMES,
                                                 QUALITY_EPSILON, WIN_TOL,
                                                 ladder, build_family)
    from runtime.adaptive_runtime import AdaptiveRuntime

    fams = {}
    for fam in FAMILY_NAMES:
        # paired: 同一 (family, seed) の同一 stream を全6 config が実行 (§69)
        per_seed_cfg = {n: [] for n in ("A_naive", "B_predictive", "C_queue",
                                        "D_semantic", "E_planning", "F_full")}
        spec = None
        for seed in SEEDS:
            stream, spec = build_family(fam, seed)
            for cfg in ladder(spec):
                res = AdaptiveRuntime(cfg).serve(list(stream))
                per_seed_cfg[cfg.name].append(res.goodput)
        assert spec is not None
        means = {n: sum(v) / len(v) for n, v in per_seed_cfg.items()}
        strong = max(BASELINES, key=lambda n: means[n])
        abs_d = [f - s for f, s in zip(per_seed_cfg["F_full"],
                                        per_seed_cfg[strong])]
        rel_d = [d / s for d, s in zip(abs_d, per_seed_cfg[strong])
                 if s > 0]
        st = sign_test(abs_d, tol=WIN_TOL)
        ci_abs = bootstrap_ci95(abs_d)
        ci_rel = bootstrap_ci95(rel_d)
        fams[fam] = {
            "strong_baseline": strong,
            "config_goodput_means": {n: round(means[n], 9)
                                     for n in per_seed_cfg},
            "abs_dmean": sum(abs_d) / len(abs_d),
            "abs_per_seed": [round(d, 9) for d in abs_d],
            "rel_dmean": sum(rel_d) / len(rel_d),
            "rel_per_seed": [round(d, 6) for d in rel_d],
            "abs_ci95": [ci_abs["ci95_low"], ci_abs["ci95_high"]],
            "rel_ci95": [ci_rel["ci95_low"], ci_rel["ci95_high"]],
            "won": st["wins"], "lost": st["losses"], "ties": st["ties"],
            "p": st["p_value"],
            "tier2_ge_20pct": bool(sum(rel_d) / len(rel_d) >= REL_TARGET),
            # 診断値 (Tier判定には使わない — PREREG §3)
            "f_over_a_ratio": (means["F_full"] / means["A_naive"]
                               if means["A_naive"] > 0 else None),
        }
        print(fam, "strong", strong, "rel%",
              round(100 * fams[fam]["rel_dmean"], 2), "p",
              round(st["p_value"], 4), flush=True)

    # Holm 補正 (8家族一括、事前登録の通り)
    order = sorted(fams.keys())
    adj = holm_correct([fams[k]["p"] for k in order])
    for k, a in zip(order, adj):
        fams[k]["holm_p"] = a
        fams[k]["confirmatory"] = bool(a < 0.05)

    # 保守性チェック vs seed12_merged (同一決定的計算の検算)
    merged_path = os.path.join(OUT_DIR, "seed12_merged.json")
    consistency = {}
    with open(merged_path, encoding="utf-8") as fh:
        merged = json.load(fh)
    for k in order:
        diff = abs(fams[k]["abs_dmean"] - float(merged[k]["dmean"]))
        consistency[k] = round(diff, 12)
        if diff >= CONSISTENCY_TOL:
            print(f"FAIL consistency {k}: diff {diff}", file=sys.stderr)
            return 1

    tier2_hits = [k for k in order if fams[k]["tier2_ge_20pct"]]
    out = {
        "experiment": "strong_reeval_m1",
        "prereg": "artifacts/phase6/PREREG_STRONG_REEVAL.md",
        "seeds": list(SEEDS),
        "rel_target": REL_TARGET,
        "tier2_min_families": TIER2_MIN_FAMILIES,
        "families": fams,
        "consistency_vs_seed12_max_diff": max(consistency.values()),
        "consistency_detail": consistency,
        "verdict": {
            "tier2_vs_strong_met": bool(len(tier2_hits) >= TIER2_MIN_FAMILIES),
            "tier2_families_ge_20pct": tier2_hits,
            "tier2_hits_count": len(tier2_hits),
            "confirmatory_families": [k for k in order
                                      if fams[k]["confirmatory"]],
            "note": ("F/A ratio is diagnostic only (PREREG_STRONG_REEVAL.md "
                     "section 3); Tier 2 judged on F vs best-of-A-E only."),
            "quality_epsilon": QUALITY_EPSILON,
        },
    }
    path = os.path.join(OUT_DIR, "strong_reeval.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    v = out["verdict"]
    print(f"Tier2 vs strong: {'MET' if v['tier2_vs_strong_met'] else 'NOT MET'} "
          f"({len(tier2_hits)}/8 >= +20%)", flush=True)
    print("confirmatory (Holm<0.05):", v["confirmatory_families"], flush=True)
    print("max consistency diff:", out["consistency_vs_seed12_max_diff"],
          flush=True)
    print("saved", path, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
