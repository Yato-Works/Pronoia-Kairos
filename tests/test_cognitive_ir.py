"""Phase 5 — Cognitive IR (Plan.md Section 30, 53, 57, 58, 92).

Section 57 fixes the lowering set; these tests pin it and the §92 gate:

    - the IR is exactly the nine Section 57 effect-aware ops (§30).
    - lowering follows Section 53's pipeline: recall the whole candidate
      stream, select the compiler's subset, then cache / prefetch / execute,
      with a §105 fallback that never prefetches.
    - every graph is well-formed: no orphan TRANSFER, PREFILL before DECODE.
    - execute() measures wall-clock with the §36 overlap window: the same
      transfers issued EARLY hide behind compute, issued LATE they block.
    - the three Section 58 passes do what they claim (dedupe, drop recalls the
      context never selects, hoist prefetch pairs to the head) and preserve
      validity.
    - Section 92's gate: measured IR overhead < measured execution gain, with
      an artifact saved for the record.
    - the lowered graph rides along in the DecisionTrace (§32/§99).
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runtime.context_compiler import JointBudgetOptimizer
from runtime.ir import (
    CONTEXT_SELECT, DECODE, KV_EVICT, KV_KEEP, KV_PREFETCH, MEMORY_RECALL,
    OP_KINDS, PREFILL, PREFIX_LOOKUP, TRANSFER, ExecutionGraph, IROp,
    effects_of,
)
from runtime.ir_lowering import lower_plan, with_ir
from runtime.ir_passes import (
    dead_context_elimination, optimize, prefetch_hoisting,
    redundant_recall_elimination,
)
from runtime.planner import AdaptivePlanner
from runtime.state import RequestState, RuntimeState, SemanticState

from compare.joint_context_cache_experiment import make_memory_objects
from compare.ir_overhead_experiment import format_gate_report, run_gate, save_gate_artifact


# --- fixtures ----------------------------------------------------------------

def _candidates(n=48):
    return make_memory_objects(n=n, seed=7)


def _ctx(cands, budget=2048, eps=1.0):
    return JointBudgetOptimizer(token_budget=budget, vram_budget=budget,
                                quality_epsilon=eps).optimize(cands)


def _confident():
    planner = AdaptivePlanner()
    state = RuntimeState(semantic_state=SemanticState(
        future_use_probability=0.85, confidence=0.9))
    request = RequestState(input_tokens=512, expected_output_tokens=64,
                           shared_prefix_length=128)
    return planner.plan(state, request, request_id="r0"), request


def _fallback():
    planner = AdaptivePlanner()
    state = RuntimeState(semantic_state=SemanticState(
        future_use_probability=0.1, confidence=0.3))
    request = RequestState(input_tokens=512, expected_output_tokens=64,
                           shared_prefix_length=0)
    return planner.plan(state, request, request_id="rf"), request


# --- IR structure (Section 30 / 57) ------------------------------------------

def test_op_set_is_exactly_section57():
    assert set(OP_KINDS) == {
        MEMORY_RECALL, CONTEXT_SELECT, PREFIX_LOOKUP,
        KV_KEEP, KV_EVICT, KV_PREFETCH, TRANSFER, PREFILL, DECODE,
    }
    assert len(OP_KINDS) == 9


def test_ops_are_effect_aware():
    for kind in OP_KINDS:
        assert effects_of(IROp(kind)), f"{kind} has no effects (Section 30)"
    assert effects_of(IROp("NOT_AN_OP")) == ()


# --- lowering (Section 53 pipeline, Section 105 fallback) --------------------

def test_lowering_produces_valid_pipeline_in_section53_order():
    cands = _candidates()
    plan, request = _confident()
    g = lower_plan(plan, request, cands, ctx=_ctx(cands), request_id="r0")
    assert g.is_valid(), g.problems()
    kinds = [op.kind for op in g.instructions]
    # recall (Lethe) before context optimization; execute last
    assert kinds.index(CONTEXT_SELECT) > kinds.index(MEMORY_RECALL)
    assert kinds.index(PREFILL) < kinds.index(DECODE)
    # every op is attributable for the decision trace (Section 32/99)
    assert all(op.origin for op in g.instructions)
    assert g.meta["strategy"] == plan.context_plan.strategy


def test_lowering_recalls_whole_stream_but_selects_compiler_subset():
    cands = _candidates()
    stream = cands + cands[:4]          # multi-query overlap
    plan, request = _confident()
    ctx = _ctx(stream)
    g = lower_plan(plan, request, stream, ctx=ctx)
    recalls = [op for op in g.instructions if op.kind == MEMORY_RECALL]
    selects = [op for op in g.instructions if op.kind == CONTEXT_SELECT]
    assert len(recalls) == len(stream)          # faithful: no early dedupe
    assert len({op.target for op in recalls}) == len(cands)
    # context is a set of the compiler's budget-feasible subset
    assert [op.target for op in selects] == list(dict.fromkeys(ctx.included))
    assert set(op.target for op in selects) <= set(op.target for op in recalls)


def test_lowering_cache_ops_follow_residency_and_retention():
    cands = _candidates()
    plan, request = _confident()
    ctx = _ctx(cands)
    g = lower_plan(plan, request, cands, ctx=ctx)
    resident = {m.id for m in cands if m.cached}
    keep = {op.target for op in g.instructions if op.kind == KV_KEEP}
    evict = {op.target for op in g.instructions if op.kind == KV_EVICT}
    assert keep | evict == resident
    assert keep & evict == set()
    assert keep == resident & set(ctx.retained_kv)


def test_lowering_fallback_never_prefetches_and_keeps_nothing():
    cands = _candidates()
    plan, request = _fallback()
    assert plan.fallback_plan.trigger          # Section 105 baseline path
    g = lower_plan(plan, request, cands, ctx=_ctx(cands))
    assert g.is_valid()
    assert g.count(KV_PREFETCH) == 0 and g.count(TRANSFER) == 0
    assert g.count(KV_KEEP) == 0               # fallback retains nothing
    assert g.count(PREFILL) == 1 and g.count(DECODE) == 1
    assert g.meta["fallback"] is True


def test_lowering_prefetch_emits_issue_then_transfer_pair():
    cands = _candidates()
    plan, request = _confident()
    assert plan.prefetch_plan.enabled
    g = lower_plan(plan, request, cands, ctx=_ctx(cands))
    kinds = [op.kind for op in g.instructions]
    for i, k in enumerate(kinds):
        if k == KV_PREFETCH:
            nxt = g.instructions[i + 1]
            assert nxt.kind == TRANSFER
            assert nxt.target == g.instructions[i].target
            assert nxt.origin == "prefetch_plan"


# --- validation & execution --------------------------------------------------

def test_validate_flags_orphan_transfer_unknown_kind_and_missing_stages():
    good = ExecutionGraph(instructions=[
        IROp(KV_PREFETCH, target="kv"), IROp(TRANSFER, target="kv"),
        IROp(PREFILL), IROp(DECODE)])
    assert good.problems() == []

    orphan = ExecutionGraph(instructions=[IROp(TRANSFER, target="kv"),
                                          IROp(PREFILL), IROp(DECODE)])
    assert any("orphan transfer" in p for p in orphan.problems())

    unknown = ExecutionGraph(instructions=[IROp("WARP_DRIVE"), IROp(PREFILL),
                                           IROp(DECODE)])
    assert any("unknown kind" in p for p in unknown.problems())

    no_exec = ExecutionGraph(instructions=[IROp(MEMORY_RECALL, target="m")])
    problems = "\n".join(no_exec.problems())
    assert "no PREFILL" in problems and "no DECODE" in problems

    inverted = ExecutionGraph(instructions=[IROp(DECODE), IROp(PREFILL)])
    assert any("PREFILL must precede DECODE" in p for p in inverted.problems())
    assert ExecutionGraph().problems()          # empty graph is not valid


def test_execute_reports_wall_compute_and_spill():
    g = ExecutionGraph(instructions=[
        IROp(MEMORY_RECALL, target="m", tokens=256),
        IROp(CONTEXT_SELECT, target="m", tokens=256),
        IROp(PREFILL, tokens=256), IROp(DECODE, tokens=32)])
    res = g.execute()
    assert res.wall_ns > 0
    assert res.wall_ns >= res.compute_ns         # wall = compute + lane spill
    assert res.spill_ns >= 0 and res.transfer_ns >= 0
    assert res.op_count == len(g)


def test_transfer_window_prefers_early_issue_section36():
    """Same ops, different transfer position (§36 max(T_compute, T_transfer)).

    Late (just before PREFILL): the window is only PREFILL -> the lane spills.
    Early (graph head): the window also covers recall compute -> hides.
    """
    late = ExecutionGraph(instructions=[
        IROp(MEMORY_RECALL, target="m", tokens=1024),
        IROp(KV_PREFETCH, target="kv", tokens=4096),
        IROp(TRANSFER, target="kv", tokens=4096),
        IROp(PREFILL, tokens=64), IROp(DECODE, tokens=8)])
    early = ExecutionGraph(instructions=[
        IROp(KV_PREFETCH, target="kv", tokens=4096),
        IROp(TRANSFER, target="kv", tokens=4096),
        IROp(MEMORY_RECALL, target="m", tokens=1024),
        IROp(PREFILL, tokens=64), IROp(DECODE, tokens=8)])
    r_late, r_early = late.execute(), early.execute()
    assert r_late.spill_ns > 0                  # late transfer did not fit
    assert r_early.spill_ns == 0                # early transfer fully hidden
    assert r_early.wall_ns + 2000 < r_late.wall_ns   # margin vs timing noise


# --- Section 58 passes -------------------------------------------------------

def _synthetic_graph():
    return ExecutionGraph(instructions=[
        IROp(MEMORY_RECALL, target="a", tokens=256),
        IROp(MEMORY_RECALL, target="b", tokens=256),
        IROp(MEMORY_RECALL, target="a", tokens=256),   # duplicate recall
        IROp(MEMORY_RECALL, target="c", tokens=256),   # never selected
        IROp(CONTEXT_SELECT, target="a", tokens=256),
        IROp(KV_PREFETCH, target="kv", tokens=512),
        IROp(TRANSFER, target="kv", tokens=512),
        IROp(PREFILL, tokens=128), IROp(DECODE, tokens=16)])


def test_redundant_recall_keeps_first_copy():
    out = redundant_recall_elimination(_synthetic_graph())
    assert out.removed_by_kind == {MEMORY_RECALL: 1}
    targets = [op.target for op in out.graph.instructions
               if op.kind == MEMORY_RECALL]
    assert targets == ["a", "b", "c"]           # first copy wins, order kept
    assert out.graph.is_valid()


def test_dead_context_removes_unselected_and_guards_without_selects():
    out = dead_context_elimination(_synthetic_graph())
    # b and c never reach CONTEXT_SELECT; the duplicate "a" stays (dedupe is
    # the other pass's job - each pass owns exactly one concern).
    assert out.removed_by_kind == {MEMORY_RECALL: 2}
    targets = [op.target for op in out.graph.instructions
               if op.kind == MEMORY_RECALL]
    assert targets == ["a", "a"]
    assert out.graph.is_valid()
    # guard: with no CONTEXT_SELECT there is no evidence - refuse to guess
    no_select = ExecutionGraph(instructions=[
        IROp(MEMORY_RECALL, target="x"), IROp(PREFILL), IROp(DECODE)])
    assert dead_context_elimination(no_select).removed_ops == 0


def test_prefetch_hoisting_moves_pairs_to_head_and_keeps_valid():
    out = prefetch_hoisting(_synthetic_graph())
    assert out.hoisted_transfers == 1
    kinds = [op.kind for op in out.graph.instructions]
    assert kinds[:2] == [KV_PREFETCH, TRANSFER]        # pair stays together
    assert kinds.index(MEMORY_RECALL) > kinds.index(TRANSFER)
    assert out.graph.is_valid(), out.graph.problems()
    # nothing to hoist when the pair is already first
    head_first = prefetch_hoisting(out.graph)
    assert head_first.hoisted_transfers == 0
    assert head_first.graph is out.graph


def test_optimize_report_counts_and_feeds_trace_summary():
    g, report = optimize(_synthetic_graph())
    assert report.passes_applied == [
        "redundant_recall_elimination", "dead_context_elimination",
        "prefetch_hoisting"]
    assert report.ops_removed == 3              # 1 dup + 2 dead recalls
    assert report.removed_by_kind == {MEMORY_RECALL: 3}
    assert report.ops_after == len(g)
    assert g.is_valid()
    summary = g.to_trace_summary()
    assert summary["passes"]["ops_removed"] == 3
    assert summary["passes"]["hoisted_transfers"] == 1
    assert summary["ops_by_kind"][MEMORY_RECALL] == 1   # only "a" survives


# --- Section 92 gate ---------------------------------------------------------

def test_section92_gate_overhead_below_gain(tmp_path):
    v = run_gate()
    assert v.requests == 6
    # the gate itself: measured cost of the IR machinery < measured time saved
    assert v.gate_passed is True
    assert v.gain_ns > v.overhead_ns > 0
    assert v.ratio > 1.0
    # both §58 elimination passes do real work on this workload; per-pass
    # gains must telescope to the total (attribution is exact, not vibes)
    assert v.pass_removed_ops["redundant_recall_elimination"] > 0
    assert v.pass_removed_ops["dead_context_elimination"] > 0
    assert sum(v.gain_by_pass_ns.values()) == v.gain_ns
    assert v.ops_after < v.ops_before
    # every row reports its own overhead and gain
    assert all(r["overhead_ns"] > 0 and r["gain_ns"] > 0 for r in v.rows)

    report = format_gate_report(v)
    assert "overhead < gain (Section 92)         : True" in report

    path = save_gate_artifact(v, out_dir=str(tmp_path))
    assert os.path.exists(path)
    import json
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    assert payload["gate_passed"] is True
    assert payload["config"]["quality_constraint_active"] is False
    assert payload["requests"] == 6


def test_with_ir_attaches_graph_summary_to_decision_trace():
    import json
    cands = _candidates()
    plan, request = _confident()
    state = RuntimeState(semantic_state=SemanticState(
        future_use_probability=0.85, confidence=0.9))
    _, trace = AdaptivePlanner().plan_with_trace(state, request, request_id="r0")
    g = lower_plan(plan, request, cands, ctx=_ctx(cands), request_id="r0")
    g, _report = optimize(g)
    traced = with_ir(trace, g)
    assert traced.ir_summary["op_count"] == len(g)
    assert traced.ir_summary["passes"]["ops_removed"] == _report.ops_removed
    assert traced.ir_summary["strategy"] == plan.context_plan.strategy
    assert traced.explanation_tags == trace.explanation_tags  # untouched
    json.dumps(traced.to_dict())                    # still JSONL-recordable