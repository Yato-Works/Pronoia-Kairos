"""M3 — Semantic Demand Shuffle Control runner (PREREG_M3_SHUFFLE, locked).

Usage:
  python -m compare.m3_shuffle_runner                 # 8 families x seeds 1-12
  python -m compare.m3_shuffle_runner W7_agent        # one family, 1-12
  python -m compare.m3_shuffle_runner W7_agent 1-6    # family + seed range

Per (family, seed) pair runs: strong + F_true + F_shuffled + F_shuffled_repeat
(full matrix = 96 x 4 = 384 runs) and enforces manipulation checks MC1-MC5.
Any MC failure aborts immediately with a non-zero exit (stop-and-report rule,
PREREG §5). All four runs of a pair consume the SAME request-stream object
(paired protocol, §69).

Artifacts (artifacts/phase6):
  m3_shuffle_maps.json        pi full maps (family/seed -> perm, fixed points)
  m3_shuffle_<fam>_s<k>.json  per-pair record (both conditions + repeat flag)
  m3_shuffle_merged.json      all pairs merged
  m3_shuffle_run_summary.json matrix summary (MC1-MC5 aggregate flags)

MC definitions (locked §5):
  MC1  pi is a bijection on the stream universe U (exact assert).
  MC2  evidence multiset over U invariant under pi (exact, integer):
       sorted(count[pi(k)] for k in U) == sorted(count[k] for k in U).
  MC3  stream hash identical across rebuilds / conditions (sha256 canonical).
  MC4  (F_true - strong) per-seed matches seed12_merged within 1e-6.
  MC5  shuffled rerun identity: summary() (goodput + all counters) exact.

Console output is ASCII-only (GBK-safe), following full_runtime_experiment.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import time
from dataclasses import replace
from typing import Dict, List, Sequence, Tuple

from compare.full_runtime_experiment import FAMILY_NAMES, build_family, ladder
from runtime.adaptive_runtime import AdaptiveRuntime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_DIR = os.path.join(ROOT, "artifacts", "phase6")

SEEDS: Tuple[int, ...] = tuple(range(1, 13))
SHUFFLE_METHOD = "fisher_yates_lex_v1"
SHUFFLE_SALT = "M3-SHUFFLE-v1"
MC4_TOL = 1e-6                      # PREREG §3: max diff < 1e-6
SEED12_PATH = os.path.join(OUT_DIR, "seed12_merged.json")


# ---------------------------------------------------------------------------
# Shuffle construction (PREREG §2 fisher_yates_lex_v1)
# ---------------------------------------------------------------------------

def stream_keys(stream) -> List[str]:
    """U = prefix_key set ∪ all chunk_id set, lex-sorted (PYTHONHASHSEED
    independent enumeration — the M1 determinism lesson)."""
    keys = set()
    for req in stream:
        keys.add(req.prefix_key)
        for ch in req.chunks:
            keys.add(ch.chunk_id)
    return sorted(keys)


def shuffle_seed_hex(family: str, seed: int) -> str:
    """Hex sha256 of the shuffle salt string — recorded for provenance."""
    return hashlib.sha256(
        f"{SHUFFLE_SALT}|{family}|{seed}".encode("utf-8")).hexdigest()


def build_signal_perm(universe: Sequence[str], family: str,
                      seed: int) -> Dict[str, str]:
    """fisher_yates_lex_v1: sorted U + Fisher-Yates from sha256-derived RNG.

    RNG seed = int(sha256(f"M3-SHUFFLE-v1|{family}|{seed}")) — separated from
    the workload RNG (random.Random(seed*1000+fam)) and PYTHONHASHSEED
    (sorted enumeration + sha256). |U| < 2 -> {} (exclusion, caller records).
    """
    keys = sorted(set(universe))
    if len(keys) < 2:
        return {}
    rng = random.Random(int(shuffle_seed_hex(family, seed), 16))
    perm_list = list(keys)
    for i in range(len(perm_list) - 1, 0, -1):
        j = rng.randrange(i + 1)
        perm_list[i], perm_list[j] = perm_list[j], perm_list[i]
    return dict(zip(keys, perm_list))


def stream_hash(stream) -> str:
    """Canonical sha256 over the request stream (MC3 evidence). Covers every
    field of WorkloadRequest/Chunk with repr() for float exactness."""
    h = hashlib.sha256()
    for req in stream:
        h.update((f"{req.request_id}|{req.arrival_time!r}|{req.prefix_key}|"
                  f"{req.prefix_tokens}|{req.output_tokens}|"
                  f"{req.slo_ttft!r}").encode("utf-8"))
        for ch in req.chunks:
            h.update((f";{ch.chunk_id}|{ch.tokens}|{ch.benefit!r}|"
                      f"{ch.quality_loss_if_dropped!r}").encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _check(cond: bool, label: str) -> None:
    """MC gate: any failure stops the experiment (PREREG §5)."""
    if not cond:
        print(f"MC_FAIL {label}", flush=True)
        raise SystemExit(2)


# ---------------------------------------------------------------------------
# One paired experiment (strong + F_true + F_shuffled x2)
# ---------------------------------------------------------------------------

def run_pair(family: str, seed: int, ref: Dict) -> Tuple[Dict, Dict[str, str]]:
    stream, spec = build_family(family, seed)

    # MC3: stream rebuild determinism; the single hash is shared by all four
    # conditions because they consume this very list object.
    h1 = stream_hash(stream)
    rebuild, _ = build_family(family, seed)
    _check(h1 == stream_hash(rebuild),
           f"MC3 {family} s{seed} rebuild hash mismatch")

    # pi over the STATIC stream universe U (PREREG §2: before any run).
    universe = stream_keys(stream)
    perm = build_signal_perm(universe, family, seed)

    # MC1: bijection on U, or recorded exclusion when |U| < 2.
    if len(universe) < 2:
        excluded = True
        _check(not perm, f"MC1 {family} s{seed} |U|<2 but perm non-empty")
    else:
        excluded = False
        _check(sorted(perm.keys()) == universe,
               f"MC1 {family} s{seed} domain != U")
        _check(sorted(perm.values()) == universe,
               f"MC1 {family} s{seed} range != U")
        _check(len(set(perm.values())) == len(universe),
               f"MC1 {family} s{seed} not injective")

    cfgs = {c.name: c for c in ladder(spec)}
    strong_name = ref["strong"]
    _check(seed in ref["seeds"],
           f"MC4 {family} s{seed} seed missing in seed12 reference")
    f_base = cfgs["F_full"]                      # signal_perm=None = True

    rt_shuf = AdaptiveRuntime(replace(f_base, signal_perm=perm))
    results = {
        "strong": AdaptiveRuntime(cfgs[strong_name]).serve(list(stream)),
        "f_true": AdaptiveRuntime(f_base).serve(list(stream)),
        "f_shuffled": rt_shuf.serve(list(stream)),
        "f_repeat": AdaptiveRuntime(
            replace(f_base, signal_perm=perm)).serve(list(stream)),
    }

    # MC2: evidence multiset over U invariant under pi (exact integers).
    counts = rt_shuf.access_count
    if perm:
        lhs = sorted(counts.get(perm[k], 0) for k in universe)
        rhs = sorted(counts.get(k, 0) for k in universe)
        _check(lhs == rhs, f"MC2 {family} s{seed} multiset over U differs")
        mc2_ok = True
    else:
        mc2_ok = True                            # vacuous (excluded run)

    # MC4: (F_true - strong) must reproduce the M1 primary record.
    delta = results["f_true"].goodput - results["strong"].goodput
    mc4 = abs(delta - float(ref["per_seed"][seed - 1]))
    _check(mc4 < MC4_TOL,
           f"MC4 {family} s{seed} diff={mc4:.3e} >= {MC4_TOL:g}")

    # MC5: shuffled rerun identity — goodput + all counters (PREREG §3).
    s4 = results["f_shuffled"].summary()
    s5 = results["f_repeat"].summary()
    _check(s4 == s5, f"MC5 {family} s{seed} repeat mismatch")

    fixed = sum(1 for k in universe if perm.get(k, k) == k)
    rec = {
        "family": family,
        "seed": seed,
        "method": SHUFFLE_METHOD,
        "shuffle_salt": SHUFFLE_SALT,
        "shuffle_seed_hex": shuffle_seed_hex(family, seed),
        "universe_size": len(universe),
        "n_fixed_points": fixed,
        "n_changed": len(universe) - fixed,
        "excluded_shuffle": excluded,
        "stream_hash": h1,
        "strong_name": strong_name,
        "mc1_ok": True,
        "mc2_ok": mc2_ok,
        "mc3_ok": True,
        "mc4_abs_diff": mc4,
        "mc5_ok": True,
        "strong_goodput": results["strong"].goodput,
        "f_true_goodput": results["f_true"].goodput,
        "f_shuffled_goodput": results["f_shuffled"].goodput,
        "f_repeat_goodput": results["f_repeat"].goodput,
        "delta_true_shuffled": (results["f_true"].goodput
                                - results["f_shuffled"].goodput),
        "delta_true_strong": delta,
        "seed12_ref_delta": float(ref["per_seed"][seed - 1]),
        "f_true_summary": results["f_true"].summary(),
        "f_shuffled_summary": s4,
        "conditions_differ": s4 != results["f_true"].summary(),
        # secondary mechanism records (§1, not used for the verdict)
        "f_true_slo_met": results["f_true"].slo_met,
        "f_shuffled_slo_met": results["f_shuffled"].slo_met,
        "f_true_quality_breaches": results["f_true"].quality_breaches,
        "f_shuffled_quality_breaches": results["f_shuffled"].quality_breaches,
        "f_true_evictions": results["f_true"].evictions,
        "f_shuffled_evictions": results["f_shuffled"].evictions,
        "f_true_prefetches": results["f_true"].prefetches,
        "f_shuffled_prefetches": results["f_shuffled"].prefetches,
        "f_true_prefetch_hits": results["f_true"].prefetch_hits,
        "f_shuffled_prefetch_hits": results["f_shuffled"].prefetch_hits,
        "f_true_prefetch_waste": results["f_true"].prefetch_waste,
        "f_shuffled_prefetch_waste": results["f_shuffled"].prefetch_waste,
        "f_true_hit_rate": results["f_true"].hit_rate,
        "f_shuffled_hit_rate": results["f_shuffled"].hit_rate,
        "f_true_makespan": results["f_true"].makespan,
        "f_shuffled_makespan": results["f_shuffled"].makespan,
    }
    return rec, perm


# ---------------------------------------------------------------------------
# Full matrix driver
# ---------------------------------------------------------------------------

def _save(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)


def run_all(families: Sequence[str], seeds: Sequence[int]) -> Dict:
    t0 = time.time()
    with open(SEED12_PATH, encoding="utf-8") as fh:
        seed12 = json.load(fh)
    os.makedirs(OUT_DIR, exist_ok=True)

    maps: Dict[str, Dict[str, dict]] = {}
    merged: Dict[str, Dict[str, dict]] = {}
    max_mc4 = 0.0
    n_runs = 0
    for family in families:
        ref_fam = seed12[family]
        for seed in seeds:
            rec, perm = run_pair(family, seed, ref_fam)
            max_mc4 = max(max_mc4, rec["mc4_abs_diff"])
            n_runs += 4
            _save(os.path.join(OUT_DIR, f"m3_shuffle_{family}_s{seed}.json"),
                  rec)
            maps.setdefault(family, {})[str(seed)] = {
                "method": SHUFFLE_METHOD,
                "shuffle_seed_hex": shuffle_seed_hex(family, seed),
                "universe_size": rec["universe_size"],
                "n_fixed_points": rec["n_fixed_points"],
                "perm": perm,
            }
            merged.setdefault(family, {})[str(seed)] = rec
            print(f"{family} s{seed:02d} mc4={rec['mc4_abs_diff']:.2e} "
                  f"dgtp={rec['delta_true_shuffled']:+.6f} "
                  f"changed={rec['n_changed']}", flush=True)
        # crash-resilient: persist maps + merged after each family
        _save(os.path.join(OUT_DIR, "m3_shuffle_maps.json"), maps)
        _save(os.path.join(OUT_DIR, "m3_shuffle_merged.json"), merged)

    summary = {
        "families": list(families),
        "seeds": list(seeds),
        "n_pairs": len(families) * len(seeds),
        "n_runs": n_runs,
        "mc1_all_ok": True,       # reaching here = every _check passed
        "mc2_all_ok": True,
        "mc3_all_ok": True,
        "mc4_max_abs_diff": max_mc4,
        "mc4_tol": MC4_TOL,
        "mc4_all_ok": max_mc4 < MC4_TOL,
        "mc5_all_ok": True,
        "elapsed_sec": round(time.time() - t0, 3),
    }
    _save(os.path.join(OUT_DIR, "m3_shuffle_run_summary.json"), summary)
    print(f"DONE pairs={summary['n_pairs']} runs={n_runs} "
          f"max_mc4={max_mc4:.3e} elapsed={summary['elapsed_sec']}s",
          flush=True)
    return summary


def _parse_seeds(arg: str) -> Tuple[int, ...]:
    seeds: List[int] = []
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


def main(argv: Sequence[str]) -> int:
    fams = [a for a in argv if a in FAMILY_NAMES]
    seed_arg = "1-12"
    for a in argv:
        if a in FAMILY_NAMES:
            continue
        if a[0].isdigit() or (a[0] == "-" and len(a) > 1) or "," in a:
            seed_arg = a
    seeds = _parse_seeds(seed_arg)
    if not fams:
        fams = list(FAMILY_NAMES)
    print(f"M3 shuffle: families={fams} seeds={list(seeds)}", flush=True)
    run_all(fams, seeds)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
