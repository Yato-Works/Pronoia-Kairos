"""§92 Phase 5 gate — IR overhead < optimization gain (Plan.md §57/§58/§92).

Question: does the Cognitive IR pay for itself? On the same machine, in the
same clock, we measure both sides per request:

    overhead_ns = wall-clock of (lower_plan + all §58 passes)   [what the IR costs]
    gain_ns     = execute(naive graph) - execute(optimized)     [what the passes save]

    Gate (§92):  overhead_ns < gain_ns

Workload (deliberate, documented):
* The Phase 4 memory pool (same generator, same distribution) evaluated by the
  Lethe->Context->KV pipeline: JointBudgetOptimizer with a BINDING token budget
  (2048) and a quality constraint held inactive (epsilon=1.0 > total pool
  quality loss). Phase 5 measures the IR machinery, not quality: with §68
  force-inclusion active (small epsilon vs pool loss) almost every candidate is
  admitted, recall is NOT wasted work, and there is correctly nothing for Dead
  Context Elimination to remove. The IR win exists precisely in the normal
  budget-bound regime where the compiler drops candidates.
* Two request profiles alternating (6 requests): a confident request with
  predictive prefetch on (§36 overlap lane exercised), and a low-confidence
  request hitting the §105 fallback baseline (no prefetch). Heterogeneous
  batches are what a runtime actually sees.
* The candidate stream repeats a few hot memories (two recall queries hitting
  the same memory - realistic multi-query recall), so §58 Redundant Recall
  Elimination has actual work instead of a no-op measured against noise.

Honest expectations from the §36 model: Prefetch Hoisting only pays when the
transfer lane exceeds its overlap window (multi-KB KV chunks). On this
workload the transfers fit the window at their naive position already, so its
measured gain sits at ~0 within timing noise - reported as-is, not hidden.

Timing: min-of-REPEATS per measurement to resist scheduler noise; all numbers
are prototype control-plane wall-clock (see runtime/ir.py on what that does
and does not claim). Artifact: artifacts/phase5/ir_overhead_gate.json.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from time import perf_counter_ns
from typing import Callable, Dict, List, Optional, Tuple

from compare.joint_context_cache_experiment import make_memory_objects
from runtime.context_compiler import ExecutionContext, JointBudgetOptimizer, MemoryObject
from runtime.ir import ExecutionGraph
from runtime.ir_lowering import lower_plan
from runtime.ir_passes import (
    dead_context_elimination, optimize, prefetch_hoisting,
    redundant_recall_elimination,
)
from runtime.planner import AdaptivePlanner
from runtime.state import RequestState, RuntimeState, SemanticState

REPEATS = 7
TOKEN_BUDGET = 2048
QUALITY_EPSILON = 1.0     # inactive by construction (see module docstring)
N_CANDIDATES = 48
N_REQUESTS = 6

PROFILES = ("confident_prefetch", "low_confidence_fallback")


@dataclass
class RequestRow:
    request_id: str
    profile: str
    overhead_ns: int
    exec_naive_ns: int
    exec_opt_ns: int
    gain_ns: int
    ops_before: int
    ops_after: int


@dataclass
class IRGateVerdict:
    gate_passed: bool
    overhead_ns: int
    gain_ns: int
    gain_by_pass_ns: Dict[str, int]
    pass_removed_ops: Dict[str, int]
    removed_by_kind: Dict[str, int]
    hoisted_transfers: int
    ops_before: int
    ops_after: int
    requests: int
    repeats: int
    rows: List[Dict] = field(default_factory=list)

    @property
    def ratio(self) -> float:
        return self.gain_ns / self.overhead_ns if self.overhead_ns else float("inf")


def _min_ns(fn: Callable[[], object], repeats: int = REPEATS) -> int:
    """Min-of-N wall-clock ns: the least-noisy honest estimator of one run."""
    best = None
    for _ in range(repeats):
        t0 = perf_counter_ns()
        fn()
        dt = perf_counter_ns() - t0
        if best is None or dt < best:
            best = dt
    return int(best)


def _min_ns_each(fns: List[Callable[[], object]],
                 repeats: int = REPEATS) -> List[int]:
    """Min-of-N for several callables, measured ROUND-ROBIN (direction
    alternating per round) so thermal/GC drift hits every variant equally.

    Without this, phases measured later inside one request inherit the
    drift and the per-pass differences (t_i - t_{i+1}) become noise even
    when each individual graph's timing looks fine.
    """
    mins: List[Optional[int]] = [None] * len(fns)
    order = list(range(len(fns)))
    for rnd in range(repeats):
        seq = order if rnd % 2 == 0 else list(reversed(order))
        for i in seq:
            t0 = perf_counter_ns()
            fns[i]()
            dt = perf_counter_ns() - t0
            if mins[i] is None or dt < mins[i]:
                mins[i] = dt
    return [int(m) for m in mins]


def build_workload() -> Tuple[List[MemoryObject], ExecutionContext,
                              List[Tuple[str, RuntimeState, RequestState]]]:
    """Shared memory pool + budgeted joint context + the request profiles."""
    candidates = make_memory_objects(n=N_CANDIDATES, seed=7)
    # multi-query recall overlap: four hot memories arrive twice in the stream
    # (distinct objects, same ids) - §58 redundant-recall must collapse them.
    candidates = candidates + candidates[:4]
    # This workload is deliberately in the BUDGET-BOUND regime: if the pool's
    # total quality loss ever reached epsilon, §68 force-inclusion would admit
    # past the budget and silently change what the §92 gate measures. Fail
    # loudly instead of drifting into the other regime.
    pool_ql = sum(m.quality_loss for m in {m.id: m for m in candidates}.values())
    assert pool_ql < QUALITY_EPSILON, (
        f"pool quality loss {pool_ql:.3f} >= epsilon {QUALITY_EPSILON}: "
        "quality force-inclusion would activate for this seed")
    ctx = JointBudgetOptimizer(token_budget=TOKEN_BUDGET, vram_budget=TOKEN_BUDGET,
                               quality_epsilon=QUALITY_EPSILON).optimize(candidates)
    cases: List[Tuple[str, RuntimeState, RequestState]] = []
    for i in range(N_REQUESTS):
        if i % 2 == 0:
            # confident: prefetch fires (future 0.85 > 0.7, confidence 0.9 > 0.8)
            state = RuntimeState(semantic_state=SemanticState(
                future_use_probability=0.85, confidence=0.9))
            request = RequestState(input_tokens=512, expected_output_tokens=64,
                                   shared_prefix_length=128)
            profile = PROFILES[0]
        else:
            # low confidence -> §105 fallback baseline (prefetch forbidden)
            state = RuntimeState(semantic_state=SemanticState(
                future_use_probability=0.1, confidence=0.3))
            request = RequestState(input_tokens=512, expected_output_tokens=64,
                                   shared_prefix_length=0)
            profile = PROFILES[1]
        cases.append((profile, state, request))
    return candidates, ctx, cases


def run_gate(repeats: int = REPEATS,
             n_requests: int = N_REQUESTS) -> IRGateVerdict:
    """Measure both sides of the §92 gate over the workload batch."""
    candidates, ctx, cases = build_workload()
    planner = AdaptivePlanner()
    overhead_ns = 0
    gain_ns = 0
    gain_by_pass: Dict[str, int] = {
        "redundant_recall_elimination": 0,
        "dead_context_elimination": 0,
        "prefetch_hoisting": 0,
    }
    removed_by_kind: Dict[str, int] = {}
    pass_removed: Dict[str, int] = {
        "redundant_recall_elimination": 0,
        "dead_context_elimination": 0,
    }
    hoisted_total = 0
    ops_before = ops_after = 0
    rows: List[Dict] = []

    for i, (profile, state, request) in enumerate(cases[:n_requests]):
        rid = f"r{i}"
        plan = planner.plan(state, request, request_id=rid)

        # --- §92 overhead side: lowering + every pass, fresh graphs each time ---
        def lower_and_pass(plan=plan, request=request, rid=rid):
            g = lower_plan(plan, request, candidates, ctx=ctx, request_id=rid)
            optimize(g)
        ovh = _min_ns(lower_and_pass, repeats=repeats)

        # --- gain side: execute naive vs after each cumulative pass ---
        g0 = lower_plan(plan, request, candidates, ctx=ctx, request_id=rid)
        o1 = redundant_recall_elimination(g0)
        o2 = dead_context_elimination(o1.graph)
        o3 = prefetch_hoisting(o2.graph)
        g1, g2, g3 = o1.graph, o2.graph, o3.graph
        _, report = optimize(g0)

        # Interleaved round-robin: all four graphs timed in the same rounds so
        # drift cancels in the per-pass differences.
        t0, t1, t2, t3 = _min_ns_each([
            lambda: g0.execute(), lambda: g1.execute(),
            lambda: g2.execute(), lambda: g3.execute()], repeats=repeats)

        overhead_ns += ovh
        gain_ns += t0 - t3
        gain_by_pass["redundant_recall_elimination"] += t0 - t1
        gain_by_pass["dead_context_elimination"] += t1 - t2
        gain_by_pass["prefetch_hoisting"] += t2 - t3
        for kind, n in report.removed_by_kind.items():
            removed_by_kind[kind] = removed_by_kind.get(kind, 0) + n
        pass_removed["redundant_recall_elimination"] += o1.removed_ops
        pass_removed["dead_context_elimination"] += o2.removed_ops
        hoisted_total += report.hoisted_transfers
        ops_before += report.ops_before
        ops_after += report.ops_after
        rows.append(asdict(RequestRow(
            request_id=rid, profile=profile, overhead_ns=ovh,
            exec_naive_ns=t0, exec_opt_ns=t3, gain_ns=t0 - t3,
            ops_before=report.ops_before, ops_after=report.ops_after)))

    return IRGateVerdict(
        gate_passed=gain_ns > overhead_ns,
        overhead_ns=overhead_ns,
        gain_ns=gain_ns,
        gain_by_pass_ns=gain_by_pass,
        pass_removed_ops=pass_removed,
        removed_by_kind=removed_by_kind,
        hoisted_transfers=hoisted_total,
        ops_before=ops_before,
        ops_after=ops_after,
        requests=len(rows),
        repeats=repeats,
        rows=rows,
    )


def format_gate_report(v: IRGateVerdict) -> str:
    us = 1000.0
    lines = [
        "Section 92 Phase 5 Gate: IR overhead vs optimization gain",
        f"{'req':>4} {'profile':>24} {'lower+pass':>11} {'exec naive':>11} "
        f"{'exec opt':>10} {'gain':>10} {'ops':>9}",
        "-" * 84,
    ]
    for r in v.rows:
        lines.append(
            f"{r['request_id']:>4} {r['profile']:>24} {r['overhead_ns']/us:>10.1f}u "
            f"{r['exec_naive_ns']/us:>10.1f}u {r['exec_opt_ns']/us:>9.1f}u "
            f"{r['gain_ns']/us:>9.1f}u {r['ops_before']:>4}->{r['ops_after']:<4}")
    lines += [
        "-" * 84,
        f"{'total':>4} {'':>24} {v.overhead_ns/us:>10.1f}u {'':>11} "
        f"{'':>10} {v.gain_ns/us:>9.1f}u {v.ops_before:>4}->{v.ops_after:<4}",
        "",
        "gain by pass (wall-clock):",
    ]
    for name, ns in v.gain_by_pass_ns.items():
        if name == "prefetch_hoisting":
            activity = f"{v.hoisted_transfers} transfers moved"
        else:
            activity = f"{v.pass_removed_ops.get(name, 0)} ops removed"
        lines.append(f"  {name:<32} {ns/us:>9.1f}u   ({activity})")
    lines.append(f"  removed ops by kind                 {v.removed_by_kind}")
    lines += [
        "",
        f"IR overhead (lower + {len(v.gain_by_pass_ns)} passes) : {v.overhead_ns/us:.1f} us",
        f"optimization gain (exec naive - opt) : {v.gain_ns/us:.1f} us",
        f"gain / overhead                      : {v.ratio:.2f}x",
        f"overhead < gain (Section 92)         : {v.gate_passed}",
    ]
    return "\n".join(lines)


def save_gate_artifact(v: IRGateVerdict, out_dir: str = "artifacts/phase5") -> str:
    import json
    import os
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "ir_overhead_gate.json")
    payload = asdict(v)
    payload["ratio"] = v.ratio
    payload["config"] = {
        "token_budget": TOKEN_BUDGET,
        "quality_epsilon": QUALITY_EPSILON,
        "quality_constraint_active": False,
        "n_candidates": N_CANDIDATES,
        "repeats": v.repeats,
        "clock": "perf_counter_ns (control-plane wall-clock, same clock both sides)",
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


def main():
    print("=== Phase 5 Cognitive IR: overhead vs optimization gain "
          "(Section 57, 58, 92) ===\n")
    verdict = run_gate()
    print(format_gate_report(verdict))
    if verdict.gate_passed:
        print("\nSection 92 Gate PASSED: lowering + Section 58 passes cost less")
        print("wall-clock than the execution time they remove from the graph.")
    else:
        print("\nSection 92 Gate NOT met: IR machinery costs more than it saves")
        print("on this workload (see per-pass breakdown above).")
    path = save_gate_artifact(verdict)
    print(f"\nArtifacts saved: {path}")


if __name__ == "__main__":
    main()