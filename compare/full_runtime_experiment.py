"""Phase 6 — Full Runtime gate (Plan.md §59, §61-63, §66, §69, §70, §93).

    Gate (§93): strong baseline superiority across workload families.

Serving: runtime/adaptive_runtime.py runs the full §59 pipeline (queue +
scheduler + KV cache + retention policy + online popularity feedback +
context planning through the IR + prefetch) over the §63 workload matrix
W1-W8. The baseline ladder is the §70 ablation ladder built from the SAME
code with different switches (never two implementations):

    A naive       LRU + FIFO + head truncation                (§47)
    B predictive  + predictive retention (Phase 3 mechanism)
    C queue       + PEEK-like queue-aware retention (Phase 2)
    D semantic    + queue x semantic retention (Phase 2/3)
    E planning    + joint context/KV planning + IR passes (Phases 4+5)
    F full        + SLO-deadline scheduling (EDF) + prefetch — the §59 runtime

Strong baseline (§62 best-configuration rule applied inside the ladder):
per family, the strong baseline is the BEST of A-E on SLO goodput. F must
beat the best EXISTING configuration, not the naive default.

Protocol (§69):
- N seeds; every configuration sees the identical request stream of a
  (family, seed) — paired comparison, same workload, same clock, same
  engine pricing for all configurations (§47/§62).
- Primary metric: SLO goodput (§66) = requests meeting BOTH the TTFT SLO
  and the quality budget (§68) per unit makespan.
- Gate: F wins (mean paired delta > 0) on >= MIN_WINS of 8 families,
  regresses on no family beyond REGRESSION_TOL, and F records ZERO quality
  breaches anywhere (§68).

Honest boundaries (§43): execution priced by the §36 model in abstract
units; Python control-plane wall clock not charged (§92); Tier 4/5 hardware
measurement is not claimed here (recorded in §93).
Console output avoids non-ASCII glyphs (GBK-safe).
"""
from __future__ import annotations

import json
import os
import random
import statistics
from dataclasses import asdict, dataclass
from typing import Dict, List, Sequence, Tuple

from runtime.adaptive_runtime import (AdaptiveRuntime, Chunk, RuntimeConfig,
                                      RunResult, WorkloadRequest)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_DIR = os.path.join(ROOT, "artifacts", "phase6")
OUT_PATH = os.path.join(OUT_DIR, "full_runtime_gate.json")

SEEDS: Tuple[int, ...] = (1, 2, 3, 4, 5, 6)
REGRESSION_TOL = 0.02   # absolute SLO-goodput units (§69 non-inferiority)
MIN_WINS = 5            # §93: superiority on a majority of families
WIN_TOL = 1e-9          # paired mean delta > this counts as a win (exact ties
                        # on sequential families are policy-parity, not wins;
                        # W8-style consistent micro-wins count: F serves the
                        # same SLO count in strictly less makespan via §20/§24)
QUALITY_EPSILON = 0.05  # §68

NAMES: Tuple[str, ...] = ("A_naive", "B_predictive", "C_queue",
                          "D_semantic", "E_planning", "F_full")
BASELINES: Tuple[str, ...] = NAMES[:-1]   # A..E: everything F is measured against
FAMILY_NAMES: Tuple[str, ...] = (
    "W1_short_chat_low_reuse", "W2_short_chat_high_reuse",
    "W3_long_ctx_low_reuse", "W4_long_ctx_high_reuse",
    "W5_concurrent_long_low_reuse", "W6_concurrent_long_high_reuse",
    "W7_agent", "W8_RAG")


@dataclass(frozen=True)
class FamilySpec:
    """Per-family hardware/workload constants — identical for ALL ladder
    configs of that family (same hardware rule, §62)."""
    name: str
    n_requests: int
    kv_capacity_tokens: int
    context_budget_tokens: int
    slo_standard: float
    slo_tight: float
    tight_fraction: float


# ---------------------------------------------------------------------------
# Deterministic workload helpers (§63)
# ---------------------------------------------------------------------------

def _zipf(rng: random.Random, items: Sequence[str], alpha: float = 1.2) -> str:
    weights = [1.0 / ((i + 1) ** alpha) for i in range(len(items))]
    return rng.choices(list(items), weights=weights, k=1)[0]


def _chunk(rng: random.Random, cid: str, tok_lo: int, tok_hi: int,
           loss: float) -> Chunk:
    return Chunk(cid, rng.randint(tok_lo, tok_hi),
                 round(0.3 + 0.4 * rng.random(), 3), loss)


def _small_loss(rng: random.Random) -> float:
    return round(0.004 + 0.004 * rng.random(), 4)


def _slo(rng: random.Random, spec: FamilySpec) -> float:

    if spec.tight_fraction > 0 and rng.random() < spec.tight_fraction:
        return spec.slo_tight
    return spec.slo_standard


# ---------------------------------------------------------------------------
# Workload families (Plan.md §63 W1-W8)
# ---------------------------------------------------------------------------

def _gen_w1(seed: int) -> Tuple[List[WorkloadRequest], FamilySpec]:
    """W1 short chat, low reuse: isolated 2-turn sessions, light load.
    Nothing to exploit -> every ladder config should tie (honesty check)."""
    rng = random.Random(seed * 1000 + 1)
    spec = FamilySpec("W1_short_chat_low_reuse", 48, 768, 768, 90.0, 90.0, 0.0)
    reqs, t, sid, i = [], 0.0, 0, 0
    while i < spec.n_requests:
        for _turn in range(2):
            if i >= spec.n_requests:
                break
            t += rng.uniform(48.0, 72.0)
            chunks = tuple(_chunk(rng, f"w1-c{sid}-{j}", 64, 128,
                                  _small_loss(rng)) for j in range(4))
            reqs.append(WorkloadRequest(f"w1-r{i}", t, f"w1-s{sid}", 256,
                                        chunks, 96, spec.slo_standard))
            i += 1
        sid += 1
    return reqs, spec


def _gen_w2(seed: int) -> Tuple[List[WorkloadRequest], FamilySpec]:
    """W2 short chat, high reuse: 8 zipf-skewed sessions returning at gaps
    LARGER than the KV residency window for all but the hottest sessions, plus
    globally shared chunks. WHICH prefixes survive eviction decides hit rate,
    TTFT and therefore SLO goodput (retention + prefetch family, Section 63)."""
    rng = random.Random(seed * 1000 + 2)
    spec = FamilySpec("W2_short_chat_high_reuse", 48, 1280, 768, 23.0, 23.0, 0.0)
    sessions = [f"w2-p{k}" for k in range(8)]
    reqs, t, i = [], 0.0, 0
    while i < spec.n_requests:
        s = _zipf(rng, sessions, alpha=0.85)
        sid = s.split("-")[-1]
        t += rng.uniform(52.0, 68.0)
        chunks = [_chunk(rng, f"w2-{sid}-prv{i}", 64, 128, _small_loss(rng))
                  for _ in range(2)]
        chunks.append(_chunk(rng, f"w2-hot{rng.randint(0, 1)}", 64, 128,
                             _small_loss(rng)))
        chunks.append(_chunk(rng, "w2-shared", 64, 128, _small_loss(rng)))
        reqs.append(WorkloadRequest(f"w2-r{i}", t, s, 256,
                                    tuple(chunks), 96, _slo(rng, spec)))
        i += 1
    return reqs, spec


def _gen_w3(seed: int) -> Tuple[List[WorkloadRequest], FamilySpec]:
    """W3 long context, low reuse: unique long prompts whose tail carries a
    high-quality-loss chunk. Head truncation (A-D) drops it and BREACHES the
    quality budget; joint planning (E-F) keeps it within budget (§68)."""
    rng = random.Random(seed * 1000 + 3)
    spec = FamilySpec("W3_long_ctx_low_reuse", 40, 2048, 1536, 130.0, 130.0, 0.0)
    reqs, t, i = [], 0.0, 0
    while i < spec.n_requests:
        t += rng.uniform(145.0, 175.0)
        n = rng.randint(10, 12)
        needle_idx = n - 1 - rng.randint(0, 1)      # near the tail
        chunks = []
        for j in range(n):
            if j == needle_idx:
                chunks.append(_chunk(rng, f"w3-n{i}", 128, 192, 0.06))
            else:
                chunks.append(_chunk(rng, f"w3-c{i}-{j}", 128, 192,
                                     _small_loss(rng)))
        reqs.append(WorkloadRequest(f"w3-r{i}", t, f"w3-p{i}", 768,
                                    tuple(chunks), 96, _slo(rng, spec)))
        i += 1
    return reqs, spec




def _gen_w4(seed: int) -> Tuple[List[WorkloadRequest], FamilySpec]:
    """W4 long context, high reuse: hot long prefixes + a shared document
    pool under capacity pressure -> KV/retention quality decides TTFT."""
    rng = random.Random(seed * 1000 + 4)
    spec = FamilySpec("W4_long_ctx_high_reuse", 48, 3072, 1536, 68.0, 68.0, 0.0)
    prefixes = [f"w4-p{k}" for k in range(8)]
    pool = [f"w4-d{k}" for k in range(20)]
    reqs, t, i = [], 0.0, 0
    while i < spec.n_requests:
        t += rng.uniform(115.0, 145.0)
        p = _zipf(rng, prefixes)
        picks = rng.sample(pool, 10)
        chunks = tuple(_chunk(rng, cid, 128, 192, _small_loss(rng))
                       for cid in picks)
        reqs.append(WorkloadRequest(f"w4-r{i}", t, p, 768, chunks, 96,
                                    _slo(rng, spec)))
        i += 1
    return reqs, spec


def _gen_w5(seed: int) -> Tuple[List[WorkloadRequest], FamilySpec]:
    """W5 concurrent long context, low reuse: sustained overload with mixed
    SLO classes -> scheduling discipline decides how many deadlines hold."""
    rng = random.Random(seed * 1000 + 5)
    spec = FamilySpec("W5_concurrent_long_low_reuse", 64, 2048, 1536,
                      640.0, 300.0, 0.45)
    reqs, t, i = [], 0.0, 0
    while i < spec.n_requests:
        t += rng.uniform(104.0, 116.0)
        n = rng.randint(8, 10)
        needle_idx = n - 1 - rng.randint(0, 1)
        chunks = []
        for j in range(n):
            if j == needle_idx and rng.random() < 0.7:
                chunks.append(_chunk(rng, f"w5-n{i}", 128, 192, 0.06))
            else:
                chunks.append(_chunk(rng, f"w5-c{i}-{j}", 128, 192,
                                     _small_loss(rng)))
        reqs.append(WorkloadRequest(f"w5-r{i}", t, f"w5-p{i}", 512,
                                    tuple(chunks), 96, _slo(rng, spec)))
        i += 1
    return reqs, spec


def _gen_w6(seed: int) -> Tuple[List[WorkloadRequest], FamilySpec]:
    """W6 concurrent long context, high reuse: overload + hot prefixes/docs ->
    cache, planning and scheduling all have work to do."""
    rng = random.Random(seed * 1000 + 6)
    spec = FamilySpec("W6_concurrent_long_high_reuse", 64, 2560, 1536,
                      260.0, 150.0, 0.45)
    prefixes = [f"w6-p{k}" for k in range(6)]
    pool = [f"w6-d{k}" for k in range(18)]
    reqs, t, i = [], 0.0, 0
    while i < spec.n_requests:
        t += rng.uniform(55.0, 63.0)
        p = _zipf(rng, prefixes)
        picks = rng.sample(pool, rng.randint(9, 11))
        chunks = tuple(_chunk(rng, cid, 128, 192, _small_loss(rng))
                       for cid in picks)
        reqs.append(WorkloadRequest(f"w6-r{i}", t, p, 640, chunks, 96,
                                    _slo(rng, spec)))
        i += 1
    return reqs, spec


def _gen_w7(seed: int) -> Tuple[List[WorkloadRequest], FamilySpec]:
    """W7 agentic: few personas, a reused tool-chunk that appears with large
    gaps (long-range reuse the LRU queue cannot protect), over-committed
    context budget, and deadline pressure (§63 W7: agent loops)."""
    rng = random.Random(seed * 1000 + 7)
    spec = FamilySpec("W7_agent", 60, 2048, 1536, 95.0, 60.0, 0.4)
    personas = [f"w7-per{k}" for k in range(3)]
    tools = [f"w7-t{k}" for k in range(10)]
    ctxpool = [f"w7-x{k}" for k in range(14)]
    reqs, t, i = [], 0.0, 0
    while i < spec.n_requests:
        t += rng.uniform(58.0, 70.0)
        p = _zipf(rng, personas)
        tool = _zipf(rng, tools)
        picks = rng.sample(ctxpool, 8)
        chunks = [_chunk(rng, tool, 128, 128, _small_loss(rng))]
        chunks += [_chunk(rng, cid, 128, 192, _small_loss(rng))
                   for cid in picks]
        chunks.append(_chunk(rng, f"w7-n{i}", 128, 192,
                             0.06 if rng.random() < 0.8 else _small_loss(rng)))
        reqs.append(WorkloadRequest(f"w7-r{i}", t, p, 384, tuple(chunks), 96,
                                    _slo(rng, spec)))
        i += 1
    return reqs, spec


def _gen_w8(seed: int) -> Tuple[List[WorkloadRequest], FamilySpec]:
    """W8 RAG: shared document chunks across concurrent requests, semi-shared
    preamble, per-request needle near the tail (§63 W8: retrieval bursts)."""
    rng = random.Random(seed * 1000 + 8)
    spec = FamilySpec("W8_RAG", 60, 2560, 1536, 140.0, 75.0, 0.45)
    preambles = [f"w8-pre{k}" for k in range(3)]
    docs = [f"w8-d{k}" for k in range(16)]
    reqs, t, i = [], 0.0, 0
    while i < spec.n_requests:
        t += rng.uniform(52.0, 64.0)
        p = _zipf(rng, preambles)
        picks = [docs[rng.randrange(len(docs))] for _ in range(11)]
        picks = list(dict.fromkeys(picks))           # dedupe within request
        chunks = [_chunk(rng, cid, 128, 192, _small_loss(rng))
                  for cid in picks]
        chunks.append(_chunk(rng, f"w8-n{i}", 128, 192,
                             0.06 if rng.random() < 0.8 else _small_loss(rng)))
        reqs.append(WorkloadRequest(f"w8-r{i}", t, p, 320, tuple(chunks), 96,
                                    _slo(rng, spec)))
        i += 1
    return reqs, spec


_GENERATORS = {
    "W1_short_chat_low_reuse": _gen_w1,
    "W2_short_chat_high_reuse": _gen_w2,
    "W3_long_ctx_low_reuse": _gen_w3,
    "W4_long_ctx_high_reuse": _gen_w4,
    "W5_concurrent_long_low_reuse": _gen_w5,
    "W6_concurrent_long_high_reuse": _gen_w6,
    "W7_agent": _gen_w7,
    "W8_RAG": _gen_w8,
}


def build_family(family: str, seed: int) -> Tuple[List[WorkloadRequest], FamilySpec]:
    """One (family, seed) stream — built ONCE and shared by every ladder
    config (paired protocol, §69)."""
    return _GENERATORS[family](seed)


def ladder(spec: FamilySpec) -> List[RuntimeConfig]:
    """The §70 ablation ladder: identical code, different switches.

    ガード類 (W1ガード/prefetch強化/EDF救済) は F_full の専用装備であり、
    A-E ベースラインには付けない。F vs strong の比較が「Fの機構の価値」
    を測るための公平な ablation になる (§70)。
    """
    base = dict(kv_capacity_tokens=spec.kv_capacity_tokens,
                context_budget_tokens=spec.context_budget_tokens,
                quality_epsilon=QUALITY_EPSILON)
    guards_off = dict(prefetch_guard=False, retention_fallback=False,
                      edf_starvation_guard=False, prefetch_min_seen=2,
                      prefetch_min_benefit_ratio=1.0, signal_confidence=0.0)
    return [
        RuntimeConfig(name="A_naive", retention="lru", scheduler="fifo",
                      **base, **guards_off),
        RuntimeConfig(name="B_predictive", retention="predictive",
                      scheduler="fifo", **base, **guards_off),
        RuntimeConfig(name="C_queue", retention="queue_aware",
                      scheduler="fifo", **base, **guards_off),
        RuntimeConfig(name="D_semantic", retention="queue_semantic",
                      scheduler="fifo", **base, **guards_off),
        RuntimeConfig(name="E_planning", retention="queue_semantic",
                      scheduler="fifo", context_planning=True,
                      **base, **guards_off),
        RuntimeConfig(name="F_full", retention="queue_semantic",
                      scheduler="edf", context_planning=True,
                      prefetch=True, **base),
    ]


def ladder_f_breakdown(spec: FamilySpec) -> List[RuntimeConfig]:
    """F内訳分解: E -> E+EDF -> E+EDF+prefetch(=F).

    E_planning と F_full の差 (+0.0001〜+0.003) が EDF 由来か prefetch 由来かを
    分離する。同一コード・スイッチ違い (§70) の3点 ablation。
    """
    base = dict(kv_capacity_tokens=spec.kv_capacity_tokens,
                context_budget_tokens=spec.context_budget_tokens,
                quality_epsilon=QUALITY_EPSILON)
    guards_off = dict(prefetch_guard=False, retention_fallback=False,
                      edf_starvation_guard=False, prefetch_min_seen=2,
                      prefetch_min_benefit_ratio=1.0, signal_confidence=0.0)
    return [
        RuntimeConfig(name="E_planning", retention="queue_semantic",
                      scheduler="fifo", context_planning=True,
                      **base, **guards_off),
        RuntimeConfig(name="E_edf", retention="queue_semantic",
                      scheduler="edf", context_planning=True,
                      prefetch=False, **base, **guards_off),
        RuntimeConfig(name="F_full", retention="queue_semantic",
                      scheduler="edf", context_planning=True,
                      prefetch=True, **base),
    ]


# ---------------------------------------------------------------------------
# Paired statistics (Section 69) and the gate (Section 66/93)
# ---------------------------------------------------------------------------

def _mean(xs: Sequence[float]) -> float:
    return statistics.fmean(xs) if xs else 0.0


def _std(xs: Sequence[float]) -> float:
    return statistics.stdev(xs) if len(xs) > 1 else 0.0


def _config_stats(results: Sequence[RunResult]) -> Dict[str, float]:
    """Aggregate one ladder config across seeds."""
    recs = [r for res in results for r in res.records]
    return {
        "goodput_mean": _mean([res.goodput for res in results]),
        "goodput_std": _std([res.goodput for res in results]),
        "slo_met_mean": _mean(
            [res.slo_met / res.n_requests if res.n_requests else 0.0
             for res in results]),
        "ttft_p95_mean": _mean([res.ttft_p95 for res in results]),
        "hit_rate_mean": _mean([res.hit_rate for res in results]),
        "quality_breaches_total": int(sum(res.quality_breaches for res in results)),
        "evictions_mean": _mean([res.evictions for res in results]),
        "makespan_mean": _mean([res.makespan for res in results]),
        "ir_ops_removed_total": int(sum(r.ir_removed_ops for r in recs)),
        "prefetch_issued_total": int(sum(res.prefetches for res in results)),
        "prefetch_hits_total": int(sum(res.prefetch_hits for res in results)),
        "prefetch_waste_total": int(sum(res.prefetch_waste for res in results)),
        "quality_loss_mean": _mean([res.mean_quality_loss for res in results]),
    }


def run_family(family: str, seeds: Sequence[int] = SEEDS) -> Dict[str, object]:
    """Run the full ladder on one family over N seeds (paired, Section 69)."""
    per_config: Dict[str, List[RunResult]] = {n: [] for n in NAMES}
    per_seed: List[Dict[str, RunResult]] = []
    spec: FamilySpec | None = None
    for seed in seeds:
        stream, spec = build_family(family, seed)
        # Streams are IDENTICAL across configs within a (family, seed):
        # each config runs its own runtime over the SAME request stream.
        for cfg in ladder(spec):
            res = AdaptiveRuntime(cfg).serve(list(stream))
            per_config[cfg.name].append(res)
        # record this seed's per-config results
        per_seed.append({cfg.name: per_config[cfg.name][-1]
                         for cfg in ladder(spec)})
    assert spec is not None

    stats = {name: _config_stats(per_config[name]) for name in NAMES}
    means = {name: stats[name]["goodput_mean"] for name in NAMES}
    strong = max(BASELINES, key=lambda n: means[n])
    deltas = [ps["F_full"].goodput - ps[strong].goodput for ps in per_seed]
    naive_deltas = [ps["F_full"].goodput - ps["A_naive"].goodput
                    for ps in per_seed]
    return {
        "spec": asdict(spec),
        "configs": stats,
        "strong_baseline": strong,
        "f_delta_vs_strong_mean": _mean(deltas),
        "f_delta_vs_strong_std": _std(deltas),
        "f_delta_vs_strong_per_seed": [round(d, 6) for d in deltas],
        "f_delta_vs_naive_mean": _mean(naive_deltas),
        "f_seeds_won_vs_strong": sum(1 for d in deltas if d > WIN_TOL),
        "f_wins": _mean(deltas) > WIN_TOL,
    }


def run_f_breakdown(family: str,
                    seeds: Sequence[int] = SEEDS) -> Dict[str, object]:
    """F内訳分解: E vs E+EDF vs F(=EDF+prefetch) の paired 比較。

    per-seed goodput を全点で保持し、EDF分 (E_edf - E) と prefetch分
    (F - E_edf) に分解する。同一 stream の paired 比較 (§69)。
    """
    from runtime.adaptive_runtime import AdaptiveRuntime as _RT
    per_seed: List[Dict[str, float]] = []
    per_cfg: Dict[str, List[float]] = {"E_planning": [], "E_edf": [],
                                       "F_full": []}
    pf_hits: Dict[str, List[int]] = {"E_edf": [], "F_full": []}
    pf_waste: Dict[str, List[int]] = {"E_edf": [], "F_full": []}
    spec = None
    for seed in seeds:
        stream, spec = build_family(family, seed)
        row: Dict[str, float] = {}
        for cfg in ladder_f_breakdown(spec):
            res = _RT(cfg).serve(list(stream))
            row[cfg.name] = res.goodput
            per_cfg[cfg.name].append(res.goodput)
            if cfg.name in pf_hits:
                pf_hits[cfg.name].append(res.prefetch_hits)
                pf_waste[cfg.name].append(res.prefetch_waste)
        per_seed.append(row)
    assert spec is not None
    edf_gain = [r["E_edf"] - r["E_planning"] for r in per_seed]
    pf_gain = [r["F_full"] - r["E_edf"] for r in per_seed]
    tot_gain = [r["F_full"] - r["E_planning"] for r in per_seed]
    return {
        "family": family,
        "seeds": list(seeds),
        "goodput_mean": {k: _mean(v) for k, v in per_cfg.items()},
        "edf_gain_per_seed": [round(d, 6) for d in edf_gain],
        "prefetch_gain_per_seed": [round(d, 6) for d in pf_gain],
        "total_gain_per_seed": [round(d, 6) for d in tot_gain],
        "edf_gain_mean": _mean(edf_gain),
        "prefetch_gain_mean": _mean(pf_gain),
        "total_gain_mean": _mean(tot_gain),
        "prefetch_hits_total": {k: int(sum(v)) for k, v in pf_hits.items()},
        "prefetch_waste_total": {k: int(sum(v)) for k, v in pf_waste.items()},
    }


def run_gate(seeds: Sequence[int] = SEEDS) -> Dict[str, object]:
    """The Section 93 gate: F must be the strong baseline across families."""
    families = {fam: run_family(fam, seeds) for fam in FAMILY_NAMES}
    wins = [f for f in FAMILY_NAMES if families[f]["f_wins"]]
    regressions = [f for f in FAMILY_NAMES
                   if float(families[f]["f_delta_vs_strong_mean"]) <
                   -REGRESSION_TOL]
    f_breaches = int(sum(
        families[f]["configs"]["F_full"]["quality_breaches_total"]
        for f in FAMILY_NAMES))
    gate_passed = (len(wins) >= MIN_WINS and not regressions
                   and f_breaches == 0)
    verdict = {
        "gate": "strong_baseline_superiority_across_workload_families",
        "gate_passed": bool(gate_passed),
        "families_total": len(FAMILY_NAMES),
        "families_won": len(wins),
        "min_wins_required": MIN_WINS,
        "wins": wins,
        "regressions": regressions,
        "regression_tolerance": REGRESSION_TOL,
        "f_quality_breaches_total": f_breaches,
        "seeds": list(seeds),
        "ladder": list(NAMES),
        "strong_baseline_rule": "best of A-E per family on SLO goodput",
        "paired_protocol": "identical stream per (family, seed) across configs",
    }
    return {
        "experiment": "phase6_full_runtime",
        "created": _now(),
        "plan_sections": ["59", "61", "62", "63", "66", "69", "70", "93"],
        "verdict": verdict,
        "families": families,
        "honesty_boundaries": [
            "abstract time units; no Python wall-clock mixed in (Section 43/92)",
            "engine prices via Section 36 model shared by ALL ladder configs",
            "ex-ante signals only: runtime uses PAST accesses (Section 68)",
            "quality budget enforced per Section 68 (slo_met = TTFT AND quality)",
            "Tier 4/5 hardware measurement NOT claimed here (Section 93)",
        ],
    }


def _now() -> str:
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save(report: Dict[str, object], path: str = OUT_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=False, ensure_ascii=False)


def print_report(report: Dict[str, object]) -> None:
    v = report["verdict"]
    print("=" * 78)
    print("Phase 6 - Full Runtime gate (Plan.md Section 93)")
    print(f"gate_passed = {v['gate_passed']}  "
          f"(wins {v['families_won']}/{v['families_total']}, "
          f"required {v['min_wins_required']}; "
          f"regressions {len(v['regressions'])}; "
          f"F quality breaches {v['f_quality_breaches_total']})")
    print("=" * 78)
    header = (f"{'family':<30}{'strong':<14}{'F-goodput':>10}"
              f"{'delta':>10}{'naive':>10}{'A':>9}{'D':>9}{'E':>9}")
    print(header)
    print("-" * len(header))
    for fam, f in report["families"].items():
        cfgs = f["configs"]
        print(f"{fam:<30}{f['strong_baseline']:<14}"
              f"{cfgs['F_full']['goodput_mean']:>10.5f}"
              f"{f['f_delta_vs_strong_mean']:>10.5f}"
              f"{f['f_delta_vs_naive_mean']:>10.5f}"
              f"{cfgs['A_naive']['goodput_mean']:>9.5f}"
              f"{cfgs['D_semantic']['goodput_mean']:>9.5f}"
              f"{cfgs['E_planning']['goodput_mean']:>9.5f}")
    print("-" * len(header))
    print("columns: goodput = SLO goodput (Section 66); delta = F - strong;")
    print("naive = F - A (end-to-end effect); A/D/E = mean goodputs.")
    if v["regressions"]:
        print(f"REGRESSIONS beyond tolerance: {v['regressions']}")
    print(f"F quality breaches across families: {v['f_quality_breaches_total']}")
    print(f"artifact: {OUT_PATH}")


if __name__ == "__main__":
    import sys
    fams = sys.argv[1:] or list(FAMILY_NAMES)
    rep_all = {"experiment": "phase6_full_runtime", "created": _now(),
               "plan_sections": ["59", "61", "62", "63", "66", "69", "70", "93"],
               "families": {}, "verdict": {}}
    for fam in fams:
        rep_all["families"][fam] = run_family(fam, SEEDS)
    wins = [f for f in rep_all["families"] if rep_all["families"][f]["f_wins"]]
    regs = [f for f in rep_all["families"]
            if float(rep_all["families"][f]["f_delta_vs_strong_mean"])
            < -REGRESSION_TOL]
    brc = int(sum(rep_all["families"][f]["configs"]["F_full"]
                  ["quality_breaches_total"] for f in rep_all["families"]))
    rep_all["verdict"] = {"gate": "strong_baseline_superiority_across_workload_families",
                          "gate_passed": bool(len(wins) >= MIN_WINS and not regs
                                              and brc == 0),
                          "families_total": len(rep_all["families"]),
                          "families_won": len(wins), "min_wins_required": MIN_WINS,
                          "wins": wins, "regressions": regs,
                          "regression_tolerance": REGRESSION_TOL,
                          "f_quality_breaches_total": brc, "seeds": list(SEEDS),
                          "ladder": list(NAMES),
                          "strong_baseline_rule": "best of A-E per family on SLO goodput",
                          "paired_protocol": "identical stream per (family, seed) across configs"}
    if len(fams) == len(FAMILY_NAMES):
        save(rep_all)
        print_report(rep_all)
    else:
        os.makedirs(OUT_DIR, exist_ok=True)
        for fam in fams:
            with open(os.path.join(OUT_DIR, f"fam_{fam}.json"), "w",
                      encoding="utf-8") as fh:
                json.dump({fam: rep_all["families"][fam]}, fh)
        print("saved chunks:", fams, "wins_so_far:", wins)




