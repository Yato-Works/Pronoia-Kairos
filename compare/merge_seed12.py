# seed12 merge helper: W5/W6の1-6/7-12分割片を12seedに統合 (P0-3)
"""Usage: python -m compare.merge_seed12 W5_concurrent_long_low_reuse"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_DIR = os.path.join(ROOT, "artifacts", "phase6")


def main():
    from compare.followup_12345 import bootstrap_ci95, sign_test
    fams = sys.argv[1:] or ["W5_concurrent_long_low_reuse",
                            "W6_concurrent_long_high_reuse"]
    for fam in fams:
        per_seed = []
        for tag in ("1-6", "7-12"):
            path = os.path.join(OUT_DIR, f"seed12_{fam}_{tag}.json")
            with open(path, encoding="utf-8") as fh:
                chunk = json.load(fh)[fam]
            # chunk側のper_seedが無い旧形式はdmeanのみなのでスキップ不可→
            # per_seed必須。旧chunkはrun_oneで再生成すること。
            per_seed.extend(chunk.get("per_seed", []))
        if len(per_seed) != 12:
            print(fam, "SKIP: per_seed len", len(per_seed), flush=True)
            continue
        strong_path = os.path.join(OUT_DIR, f"seed12_{fam}_1-6.json")
        with open(strong_path, encoding="utf-8") as fh:
            strong = json.load(fh)[fam]["strong"]
        st = sign_test(per_seed)
        ci = bootstrap_ci95(per_seed)
        out = {fam: {"strong": strong,
                     "dmean": sum(per_seed) / len(per_seed),
                     "per_seed": per_seed, "won": st["wins"],
                     "lost": st["losses"], "ties": st["ties"],
                     "p": st["p_value"], "ci_low": ci["ci95_low"],
                     "ci_high": ci["ci95_high"], "seeds": list(range(1, 13))}}
        path = os.path.join(OUT_DIR, f"seed12_{fam}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
        print(fam, "dmean", round(out[fam]["dmean"], 6), "won", st["wins"],
              "lost", st["losses"], "p", round(st["p_value"], 4), flush=True)
        print("saved", path, flush=True)


if __name__ == "__main__":
    main()
