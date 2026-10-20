# seed12 runner: argv家族指定 -> seed12実行 -> chunk保存 -> merge
"""Usage:
  python -m compare.seed12_runner W4_long_ctx_high_reuse
  python -m compare.seed12_runner W5_concurrent_long_low_reuse --seeds 1-6
  python -m compare.seed12_runner --merge
"""
from __future__ import annotations

import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_DIR = os.path.join(ROOT, "artifacts", "phase6")


def _parse_seeds(arg: str):
    # "1-6" or "1,2,3" or "12"
    seeds = []
    for part in arg.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            seeds.extend(range(int(a), int(b) + 1))
        else:
            seeds.append(int(part))
    return tuple(seeds)


def run_one(family: str, seeds):
    from compare.followup_12345 import bootstrap_ci95, sign_test
    from compare.full_runtime_experiment import run_family
    f = run_family(family, seeds=seeds)
    diffs = [float(d) for d in f["f_delta_vs_strong_per_seed"]]
    st = sign_test(diffs)
    ci = bootstrap_ci95(diffs)
    out = {family: {"strong": f["strong_baseline"],
                    "dmean": f["f_delta_vs_strong_mean"],
                    "per_seed": [float(d)
                                 for d in f["f_delta_vs_strong_per_seed"]],
                    "won": st["wins"], "lost": st["losses"],
                    "ties": st["ties"], "p": st["p_value"],
                    "ci_low": ci["ci95_low"], "ci_high": ci["ci95_high"],
                    "seeds": list(seeds)}}
    tag = f"{seeds[0]}-{seeds[-1]}" if len(seeds) > 1 else f"{seeds[0]}"
    path = os.path.join(OUT_DIR, f"seed12_{family}_{tag}.json")
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(family, "strong", f["strong_baseline"], "dmean",
          round(f["f_delta_vs_strong_mean"], 6), "won", st["wins"],
          "lost", st["losses"], "ties", st["ties"], "p",
          round(st["p_value"], 4), flush=True)
    print("saved", path, flush=True)
    return path


def merge_all():
    merged = {}
    for p in glob.glob(os.path.join(OUT_DIR, "seed12_*.json")):
        base = os.path.basename(p)
        if base in ("seed12.json", "seed12_merged.json"):
            continue
        # W5/W6の 1-6/7-12 分割片は merge_seed12.py で統合した
        # seed12_<fam>.json を優先し、分割片は除外する。
        if base.endswith("_1-6.json") or base.endswith("_7-12.json"):
            continue
        with open(p, encoding="utf-8") as fh:
            merged.update(json.load(fh))
    out = os.path.join(OUT_DIR, "seed12_merged.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(merged, fh, indent=2)
    print("merged", sorted(merged.keys()), "->", out, flush=True)
    return out


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] == "--merge":
        merge_all()
    else:
        fams = [a for a in args
                if not a.startswith("--") and not a[0].isdigit()
                and "-" not in a]
        # 末尾の "1-6" / "7-12" / "1,2,3" をseed範囲として解釈する。
        # (--seeds 1-6 形式も可)
        seed_arg = "1-12"
        for a in args:
            if a.startswith("--seeds"):
                seed_arg = (a.split("=", 1)[1] if "=" in a else "1-12")
            elif a[0].isdigit() or (a[0] == "-" and len(a) > 1):
                seed_arg = a
        seeds = _parse_seeds(seed_arg.lstrip("="))
        for fam in fams:
            run_one(fam, seeds)
