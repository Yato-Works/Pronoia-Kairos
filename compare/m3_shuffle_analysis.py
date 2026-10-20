"""M3 analysis: paired True-vs-Shuffled statistics (PREREG_M3_SHUFFLE §4).

Loads m3_shuffle_<fam>_s<k>.json (8 families x 12 seeds = 96 pairs) and per
family computes, on dG = G_true - G_shuffled (paired, n=12):
  - sign test (two-sided, ties excluded via WIN_TOL)
  - Holm correction over ALL 8 families
  - bootstrap 95% CI (10,000 resamples, analysis RNG seed 20261019)
  - relative effect mean(dG)/mean(G_true) in %
Then applies the locked three-way interpretation table (§5):
  supported     = MC1-MC5 all pass AND >=1 confirmatory family with mean dG>0
  not_confirmed = no confirmatory family (CI bounds reported; no equivalence
                  claim — equivalence test not preregistered)
  opposite      = confirmatory family with mean dG<0 (mixed reported as-is)
Confirmatory := Holm p < 0.05 AND CI sign consistent with mean(dG).

Saves artifacts/phase6/m3_shuffle_analysis.json. ASCII-only console.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

from compare.followup_12345 import bootstrap_ci95, holm_correct, sign_test
from compare.full_runtime_experiment import FAMILY_NAMES

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_DIR = os.path.join(ROOT, "artifacts", "phase6")
OUT_PATH = os.path.join(OUT_DIR, "m3_shuffle_analysis.json")

ANALYSIS_SEED = 20261019        # PREREG §4: analysis RNG seed fixed
N_BOOT = 10000                  # PREREG §4: bootstrap resamples
MC4_TOL = 1e-6


def load_pairs() -> Dict[str, List[Dict]]:
    pairs: Dict[str, List[Dict]] = {}
    for fam in FAMILY_NAMES:
        rows = []
        for seed in range(1, 13):
            path = os.path.join(OUT_DIR, f"m3_shuffle_{fam}_s{seed}.json")
            with open(path, encoding="utf-8") as fh:
                rows.append(json.load(fh))
        pairs[fam] = rows
    return pairs


def mc_summary(pairs: Dict[str, List[Dict]]) -> Dict[str, object]:
    rows = [r for fam in FAMILY_NAMES for r in pairs[fam]]
    mc4_max = max(r["mc4_abs_diff"] for r in rows)
    return {
        "n_pairs": len(rows),
        "mc1_all_ok": all(r["mc1_ok"] for r in rows),
        "mc2_all_ok": all(r["mc2_ok"] for r in rows),
        "mc3_all_ok": all(r["mc3_ok"] for r in rows),
        "mc4_max_abs_diff": mc4_max,
        "mc4_tol": MC4_TOL,
        "mc4_all_ok": mc4_max < MC4_TOL,
        "mc5_all_ok": all(r["mc5_ok"] for r in rows),
        "n_excluded_shuffle": sum(1 for r in rows if r["excluded_shuffle"]),
    }


def analyze() -> Dict:
    pairs = load_pairs()
    mc = mc_summary(pairs)
    mc_pass = all(mc[k] for k in ("mc1_all_ok", "mc2_all_ok", "mc3_all_ok",
                                  "mc4_all_ok", "mc5_all_ok"))

    fam_out: Dict[str, Dict] = {}
    p_raw: List[float] = []
    for fam in FAMILY_NAMES:
        rows = pairs[fam]
        dg = [float(r["delta_true_shuffled"]) for r in rows]
        gt = [float(r["f_true_goodput"]) for r in rows]
        gs = [float(r["f_shuffled_goodput"]) for r in rows]
        st = sign_test(dg)
        ci = bootstrap_ci95(dg, n_boot=N_BOOT, seed=ANALYSIS_SEED)
        mean_gt = sum(gt) / len(gt)
        rel = (100.0 * ci["mean"] / mean_gt) if mean_gt else None
        fam_out[fam] = {
            "strong_name": rows[0]["strong_name"],
            "dG_per_seed": dg,
            "mean_dG": ci["mean"],
            "ci95_low": ci["ci95_low"],
            "ci95_high": ci["ci95_high"],
            "ci_width": ci["ci95_high"] - ci["ci95_low"],
            "rel_effect_pct": rel,
            "mean_G_true": mean_gt,
            "mean_G_shuffled": sum(gs) / len(gs),
            "wins": st["wins"],
            "losses": st["losses"],
            "ties": st["ties"],
            "p_raw": st["p_value"],
            "n_conditions_differ": sum(
                1 for r in rows if r["conditions_differ"]),
        }
        p_raw.append(st["p_value"])

    p_holm = holm_correct(p_raw)
    confirm_pos: List[str] = []
    confirm_neg: List[str] = []
    for fam, ph in zip(FAMILY_NAMES, p_holm):
        fo = fam_out[fam]
        fo["p_holm"] = ph
        m = fo["mean_dG"]
        ci_sign_ok = ((m > 0 and fo["ci95_low"] > 0)
                      or (m < 0 and fo["ci95_high"] < 0))
        fo["confirmatory"] = bool(ph < 0.05 and ci_sign_ok)
        if fo["confirmatory"]:
            (confirm_pos if m > 0 else confirm_neg).append(fam)

    # Locked three-way verdict (§4/§5).
    if not mc_pass:
        verdict = "not_evaluable"
        detail = "MC1-MC5 failure: results are not interpretable (stop rule)."
    elif confirm_pos and confirm_neg:
        verdict = "mixed"
        detail = (f"confirmatory positive on {confirm_pos} and negative on "
                  f"{confirm_neg} — both reported as-is (PREREG §4).")
    elif confirm_pos:
        verdict = "supported"
        detail = f"H1 supported: confirmatory mean dG>0 on {confirm_pos}."
    elif confirm_neg:
        verdict = "opposite"
        detail = (f"confirmatory mean dG<0 on {confirm_neg}: H1 not "
                  "supported; current demand-signal usage may have a problem "
                  "(§5 row 3) — raised as an investigation item.")
    else:
        verdict = "not_confirmed"
        detail = ("no confirmatory family: additional contribution of the "
                  "semantic correspondence was not confirmed under these "
                  "conditions (CI bounds reported; no equivalence claim).")

    out = {
        "protocol": {
            "prereg": "PREREG_M3_SHUFFLE.md (locked 2026-10-19)",
            "metric": "dG = G_true - G_shuffled (paired, n=12 per family)",
            "tests": "sign test (two-sided) + Holm(8) + bootstrap 95%CI",
            "n_boot": N_BOOT,
            "analysis_seed": ANALYSIS_SEED,
            "confirmatory_rule": "p_holm < 0.05 AND CI sign matches mean",
        },
        "mc": mc,
        "verdict": verdict,
        "detail": detail,
        "confirmatory_positive": confirm_pos,
        "confirmatory_negative": confirm_neg,
        "families": fam_out,
    }
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    return out


def _fmt(x: object, nd: int = 6) -> str:
    if x is None:
        return "NA"
    if isinstance(x, float):
        return f"{x:+.{nd}f}"
    return str(x)


def print_report(out: Dict) -> None:
    mc = out["mc"]
    print(f"MC: n_pairs={mc['n_pairs']} "
          f"mc1={mc['mc1_all_ok']} mc2={mc['mc2_all_ok']} "
          f"mc3={mc['mc3_all_ok']} mc4(max={mc['mc4_max_abs_diff']:.3e}"
          f"<{mc['mc4_tol']:g})={mc['mc4_all_ok']} mc5={mc['mc5_all_ok']}",
          flush=True)
    hdr = (f"{'family':<34} {'mean_dG':>11} {'ci95_low':>11} {'ci95_high':>11} "
           f"{'rel%':>8} {'W/L/T':>7} {'p_raw':>8} {'p_holm':>8} {'conf':>5}")
    print(hdr, flush=True)
    for fam, fo in out["families"].items():
        rel = ("NA" if fo["rel_effect_pct"] is None
               else f"{fo['rel_effect_pct']:+.2f}")
        print(f"{fam:<34} {_fmt(fo['mean_dG'])} {_fmt(fo['ci95_low'])} "
              f"{_fmt(fo['ci95_high'])} {rel:>8} "
              f"{fo['wins']}/{fo['losses']}/{fo['ties']} "
              f"{fo['p_raw']:>8.4f} {fo['p_holm']:>8.4f} "
              f"{'Y' if fo['confirmatory'] else '.':>5}", flush=True)
    print(f"VERDICT: {out['verdict']}", flush=True)
    print(out["detail"], flush=True)


if __name__ == "__main__":
    print_report(analyze())
    print(f"saved {OUT_PATH}", flush=True)
