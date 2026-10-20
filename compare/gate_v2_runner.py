# gate v2 runner: ガード後コードで正本分離 (P0-2)
"""Usage:
  python -m compare.gate_v2_runner W4_long_ctx_high_reuse
  python -m compare.gate_v2_runner --merge
v1 (full_runtime_gate.json, 10/15, ガード前) は凍結保存。
v2 (full_runtime_gate_v2.json) がガード後の正本になる。
"""
from __future__ import annotations

import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_DIR = os.path.join(ROOT, "artifacts", "phase6")


def run_one(family: str):
    from compare.full_runtime_experiment import (MIN_WINS, REGRESSION_TOL,
                                                 SEEDS, run_family)
    f = run_family(family, SEEDS)
    path = os.path.join(OUT_DIR, f"fam_{family}_v2.json")
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({family: f}, fh, indent=2)
    print("saved", path, "strong", f["strong_baseline"], "dmean",
          round(f["f_delta_vs_strong_mean"], 6), flush=True)
    return path


def merge_all():
    from compare.full_runtime_experiment import (FAMILY_NAMES, MIN_WINS,
                                                 NAMES, REGRESSION_TOL, SEEDS,
                                                 print_report, save)
    fams = {}
    for p in glob.glob(os.path.join(OUT_DIR, "fam_*_v2.json")):
        with open(p, encoding="utf-8") as fh:
            fams.update(json.load(fh))
    wins = [f for f in fams if fams[f]["f_wins"]]
    regs = [f for f in fams
            if float(fams[f]["f_delta_vs_strong_mean"]) < -REGRESSION_TOL]
    brc = int(sum(fams[f]["configs"]["F_full"]["quality_breaches_total"]
                   for f in fams))
    rep = {"experiment": "phase6_full_runtime_v2",
           "families": fams,
           "verdict": {"gate": "strong_baseline_superiority",
                       "gate_passed": bool(len(wins) >= MIN_WINS
                                           and not regs and brc == 0),
                       "families_total": len(fams),
                       "families_won": len(wins),
                       "min_wins_required": MIN_WINS, "wins": wins,
                       "regressions": regs,
                       "regression_tolerance": REGRESSION_TOL,
                       "f_quality_breaches_total": brc,
                       "seeds": list(SEEDS), "ladder": list(NAMES)}}
    out = os.path.join(OUT_DIR, "full_runtime_gate_v2.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=2)
    print_report({"verdict": rep["verdict"], "families": fams})
    print("saved", out, flush=True)
    return out


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] == "--merge":
        merge_all()
    else:
        for fam in args:
            run_one(fam)
