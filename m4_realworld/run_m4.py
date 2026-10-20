"""M4 runner: real-GPU minimal proof (PREREG_M4_REALWORLD, locked).

Usage:
  python -m m4_realworld.run_m4 --seeds 1 --conditions lru --smoke   # quick check
  python -m m4_realworld.run_m4                                       # full pilot

Saves per run: artifacts/phase6/m4_realworld_seed<S>_<cond>.json
Plus: m4_realworld_analysis.json, m4_realworld_pilot_summary.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, List, Optional

from compare.full_runtime_experiment import ladder
from m4_realworld.gpu_executor import load_executor
from m4_realworld.harness import run_stream
from m4_realworld.mc import (compare_harness_to_reference, mc3_check,
                             mc4_check, mc5_columns_equal, reference_replay)
from m4_realworld.workloads import build_reduced_w6, stream_hash

ART_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "phase6"


def _arm_config(cond: str, spec):
    """§3 paired conditions straight from the locked §70 ladder:
    A_naive (LRU baseline) vs D_semantic (queue_semantic, True condition) —
    identical budgets, guards off, prefetch off (D verdict). Only the name
    changes for artifact readability."""
    name = "A_naive" if cond == "lru" else "D_semantic"
    base = next(c for c in ladder(spec) if c.name == name)
    return replace(base, name=f"M4_{cond}")


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def summarize(traces, out):
    return {
        "requests": len(traces),
        "e2e_wall_s": round(out["e2e_wall_s"], 4),   # §6 primary metric (§8-1)
        "e2e_total_s": round(sum(t.e2e_s for t in traces), 4),
        "e2e_mean_ms": round(1000.0 * _mean([t.e2e_s for t in traces]), 2),
        "control_ms_total": round(sum(t.control_ns for t in traces) / 1e6, 3),
        "gpu_ms_total": round(
            sum(t.prefill_gpu_ms + t.decode_gpu_ms for t in traces), 3),
        "prefill_gpu_ms_mean": round(_mean([t.prefill_gpu_ms for t in traces]), 3),
        "decode_gpu_ms_mean": round(_mean([t.decode_gpu_ms for t in traces]), 3),
        "prefill_tokens_total": int(sum(t.new_tokens for t in traces)),
        "output_tokens_total": int(sum(t.output_tokens for t in traces)),
        "reused_tokens_sum": int(sum(t.reused_tokens for t in traces)),
        "mirror_prefix_hits": int(out["mirror_prefix_hits"]),
        "evicted_entries_total": int(sum(len(t.evicted_keys) for t in traces)),
        "queue_depth_mean": round(
            _mean([t.queue_depth_at_decision for t in traces]), 3),
        "queue_depth_max": int(out["max_queue_depth"]),
    }


def mc0_check(executor, chunk_tokens: List[int], n_decode: int = 32) -> Dict:
    """MC0: scratch full prefill vs partial-prefill + exact cache continuation.

    Greedy decode outputs must match (exact token match primary; else the
    §8-6 tolerance: logits diff <= 0.1 at first diverge AND margin < 0.1).
    """
    tokens = list(chunk_tokens)
    half = max(1, len(tokens) // 2)
    cache_a, _layers_a, _, fl_a = executor.prefill(tokens, 0, None,
                                                   collect_logits=True)
    ids_a, _, logits_a = executor.decode(cache_a, n_decode, fl_a,
                                         collect_logits=True)
    _, half_layers, _, _ = executor.prefill(tokens[:half], 0, None)
    cached = ([k[:, :, :half, :] for k, _v in half_layers],
              [v[:, :, :half, :] for _k, v in half_layers])
    cache_b, _, _, fl_b = executor.prefill(tokens, half, cached,
                                           collect_logits=True)
    ids_b, _, logits_b = executor.decode(cache_b, n_decode, fl_b,
                                         collect_logits=True)
    result = {"n_decode": n_decode, "chunk_tokens": len(tokens),
              "half": half, "exact": ids_a == ids_b}
    if ids_a == ids_b:
        result["status"] = "exact"
        return result
    i = next(k for k, (x, y) in enumerate(zip(ids_a, ids_b)) if x != y)
    result["diverge_at"] = i
    step = i   # logits[step] is the distribution that produced out_ids[step]
    if step < len(logits_a) and step < len(logits_b):
        la, lb = logits_a[step], logits_b[step]
        result["logits_diff"] = float((la - lb).abs().max())
        top2 = la.topk(2)
        result["top1_top2_margin"] = float(top2.values[0] - top2.values[1])
        tol_ok = (result["logits_diff"] <= 0.1
                  and result["top1_top2_margin"] < 0.1)
        result["status"] = "tolerance_ok" if tol_ok else "fail"
    else:
        result["status"] = "fail"
    return result


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--conditions", default="lru,queue_semantic")
    ap.add_argument("--smoke", action="store_true",
                    help="truncate stream to first 8 requests")
    ap.add_argument("--out-dir", default=str(ART_DIR))
    args = ap.parse_args(argv)
    seeds = [int(s) for s in args.seeds.split(",") if s]
    conditions = [c for c in args.conditions.split(",") if c]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("== Pronoia-Kairos M4 real-world validation (locked PREREG) ==",
          flush=True)
    executor = load_executor()
    # warmup (PREREG §5 MC4 "warmup 除外"): exercise the REAL-length prefill
    # and decode path so GPU clocks/kernels settle before any traced request.
    # 4-token warmups do not exercise the ~1.8k-token regime actually timed.
    import torch as _torch
    _wu = [500, 123, 77, 900, 31, 4, 55, 610] * 230  # ~1840 tokens
    for _ in range(2):
        wu_cache, _, _, wu_fl = executor.prefill(_wu, 0, None)
        executor.decode(wu_cache, 32, wu_fl)
    del wu_cache, wu_fl
    _torch.cuda.empty_cache()
    print("[warmup] 2x (1840-token prefill + 32 decode) done, excluded",
          flush=True)

    runs = []
    semantic_meta = None
    for seed in seeds:
        stream, spec, stats = build_reduced_w6(seed)
        shash = stream_hash(stream)
        used = stream[:8] if args.smoke else stream
        for cond in conditions:
            cfg = _arm_config(cond, spec)
            t0 = time.perf_counter()
            out = run_stream(used, spec, cfg, executor)
            wall = time.perf_counter() - t0
            traces = out["traces"]
            summary = summarize(traces, out)
            service = {t.request_id: t.e2e_s for t in traces}
            ref = reference_replay(used, cfg, service)
            mc2 = compare_harness_to_reference(out, traces, ref)
            mc4res = mc4_check(traces)
            rec = {
                "seed": seed, "condition": cond,
                "stream_hash": shash, "stream_stats": stats,
                "n_requests_used": len(used),
                "config": asdict(cfg),
                "wall_s": round(wall, 3),
                "summary": summary,
                "mc2": {"match": mc2["match"], "diffs": mc2["diffs"]},
                "mc4": mc4res,
                "traces": [t.decision_columns() | {
                    "t_control_ms": round(t.control_ns / 1e6, 4),
                    "t_gpu_ms": round(t.prefill_gpu_ms + t.decode_gpu_ms, 4),
                    "prefill_gpu_ms": round(t.prefill_gpu_ms, 4),
                    "decode_gpu_ms": round(t.decode_gpu_ms, 4),
                    "ttft_ms": round(t.ttft_s * 1000.0, 4),
                    "e2e_ms": round(t.e2e_s * 1000.0, 4),
                    "output_tokens": t.output_tokens,
                    "slo_met": t.slo_met,
                } for t in traces],
            }
            path = out_dir / f"m4_realworld_seed{seed}_{cond}.json"
            path.write_text(json.dumps(rec, indent=2), encoding="utf-8")
            print(f"[seed {seed}] {cond}: " + " ".join(
                f"{k}={v}" for k, v in summary.items()), flush=True)
            print(f"  -> {path.name} | MC2 match={mc2['match']}"
                  f"{' diffs=' + str(mc2['diffs']) if mc2['diffs'] else ''}"
                  f" | MC4={mc4res['status']} "
                  f"({mc4res['violations']}/{mc4res['pairs']} violated)",
                  flush=True)
            runs.append(rec)
            if cond == "queue_semantic" and semantic_meta is None:
                semantic_meta = (seed, used, spec, cfg, traces)

    mc = {"mc0": None, "mc5": None, "mc3": mc3_check(),
          "mc2_all_match": all(r["mc2"]["match"] for r in runs)}
    print(f"MC3: {mc['mc3']['status']} "
          f"(unit={mc['mc3']['unit_ok']} harness={mc['mc3']['harness_ok']})",
          flush=True)
    if semantic_meta is not None:
        seed0, used0, spec0, cfg0, traces0 = semantic_meta
        mc["mc0"] = mc0_check(executor,
                              [500, 123, 77, 900, 31, 4, 55, 610,
                               222, 43, 87, 990, 15, 16, 17, 18])
        print(f"MC0: {mc['mc0']}", flush=True)
        out2 = run_stream(used0, spec0, cfg0, executor)
        mc["mc5"] = mc5_columns_equal(traces0, out2["traces"])
        print(f"MC5: {mc['mc5']} (seed {seed0} rerun)", flush=True)

    # ---- analysis (§6: r_overhead only when denominator positive) -------
    by = {(r["seed"], r["condition"]): r for r in runs}
    # MC1 (stream parity): same seed -> identical stream hash in both arms
    mc["mc1_stream_parity"] = all(
        by[(s, "lru")]["stream_hash"] == by[(s, "queue_semantic")]["stream_hash"]
        for s in seeds
        if (s, "lru") in by and (s, "queue_semantic") in by)
    rows = []
    for seed in seeds:
        if (seed, "lru") in by and (seed, "queue_semantic") in by:
            b = by[(seed, "lru")]["summary"]
            a = by[(seed, "queue_semantic")]["summary"]
            d_e2e = b["e2e_wall_s"] - a["e2e_wall_s"]
            d_pre = b["prefill_gpu_ms_mean"] - a["prefill_gpu_ms_mean"]
            rows.append({
                "seed": seed,
                "e2e_s_lru": b["e2e_wall_s"],
                "e2e_s_adaptive": a["e2e_wall_s"],
                "delta_e2e_pct": round(100.0 * d_e2e / b["e2e_wall_s"], 3)
                if b["e2e_wall_s"] else None,
                "prefill_ms_lru": b["prefill_gpu_ms_mean"],
                "prefill_ms_adaptive": a["prefill_gpu_ms_mean"],
                "delta_prefill_pct":
                    round(100.0 * d_pre / b["prefill_gpu_ms_mean"], 3)
                if b["prefill_gpu_ms_mean"] else None,
                "r_overhead":
                    round(a["control_ms_total"] / (d_e2e * 1000.0), 4)
                    if d_e2e > 0 else None,
                "reuse_tokens_lru": b["reused_tokens_sum"],
                "reuse_tokens_adaptive": a["reused_tokens_sum"],
                "prefill_tokens_lru": b["prefill_tokens_total"],
                "prefill_tokens_adaptive": a["prefill_tokens_total"],
                "queue_depth_mean_adaptive": a["queue_depth_mean"],
            })
    pooled = {}
    for key in ("delta_e2e_pct", "delta_prefill_pct", "r_overhead"):
        vals = [r[key] for r in rows if r[key] is not None]
        pooled[key] = round(sum(vals) / len(vals), 4) if vals else None
    stage4 = ("adaptive faster: improvement observed under applied conditions"
              if (pooled.get("delta_e2e_pct") or 0) > 0
              else "no improvement observed under applied conditions")
    analysis = {
        "prereg": "artifacts/phase6/PREREG_M4_REALWORLD.md (locked)",
        "env": _env_info(executor),
        "mc": mc,
        "rows": rows,
        "pooled": pooled,
        "four_stage": {
            "stage1_correctness":
                "pass" if (mc.get("mc0") or {}).get("status")
                in ("exact", "tolerance_ok") else "fail",
            "stage2_feasibility":
                "pass" if mc["mc2_all_match"]
                and (mc.get("mc5") or {}).get("match")
                and mc.get("mc1_stream_parity")
                and mc["mc3"]["status"] == "pass" else "fail",
            "stage3_performance": "measured" if rows else "not_measured",
            "stage4_claim_boundary": stage4,
        },
    }
    path = out_dir / "m4_realworld_analysis.json"
    path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    if rows:
        csv_path = out_dir / "m4_realworld_pilot_summary.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(json.dumps(analysis["four_stage"], indent=2), flush=True)
    print(f"pooled: {pooled}  -> {path.name}", flush=True)
    return 0


def _env_info(executor) -> dict:
    import sys

    import torch
    import transformers
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "model": executor.model.config.name_or_path,
    }


if __name__ == "__main__":
    raise SystemExit(main())


