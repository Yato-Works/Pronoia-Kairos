"""§53 / §57 planner lowering — RuntimePlan + candidate set -> ExecutionGraph.

Section 53's pipeline fixes the naive (unoptimized) op order:

    Lethe Recall            -> MEMORY_RECALL (every evaluated candidate)
    Context Optimization    -> CONTEXT_SELECT (what the compiler kept)
    Prefix reuse            -> PREFIX_LOOKUP
    Cache placement         -> KV_KEEP / KV_EVICT, then
                               KV_PREFETCH + TRANSFER (predictive targets)
    Execution               -> PREFILL, DECODE

Facts that make later optimization possible (and honest):

* MEMORY_RECALL covers the whole candidate stream (duplicates included -
  lowering is faithful, not optimized); the context compiler may keep only a
  budget-feasible subset. Recalls of dropped candidates are real upstream
  work with no consumer -> §58 Dead Context Elimination can remove them, and
  duplicate stream entries -> §58 Redundant Recall Elimination.
* Prefetch targets come from the planner's semantic PREDICTION
  (prefetch_plan.targets), independent of this request's recall/selection
  results -> §58 Prefetch Hoisting may legally move them to the graph head.
* The naive order issues transfers just before PREFILL, where almost no
  compute is left to race them into (§36 overlap window).

§105 fallback: a fallback plan lowers to the safe baseline - recall/select
still happen, but no prefetch ops are emitted.

Decision trace (§32/§99): every op carries `origin`, and `with_ir()` folds the
graph's summary into an existing DecisionTrace so a recorded decision shows
which IR it produced.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

from runtime.context_compiler import ExecutionContext, MemoryObject
from runtime.ir import (
    CONTEXT_SELECT, DECODE, KV_EVICT, KV_KEEP, KV_PREFETCH, MEMORY_RECALL,
    PREFILL, PREFIX_LOOKUP, TRANSFER, ExecutionGraph, IROp,
)
from runtime.plan import RuntimePlan
from runtime.state import RequestState

# Fallback size for a prefetch target we cannot match to a known candidate
# (predictive targets name future objects, not this request's candidates).
_PREDICTED_CHUNK_FALLBACK_TOKENS = 512


def _context_ids(plan: RuntimePlan, candidates: Sequence[MemoryObject],
                 ctx: Optional[ExecutionContext]) -> List[str]:
    """What actually enters the working context, in stable order.

    The compiler's answer wins when provided (the real call site has it);
    otherwise fall back to the plan's strategy on the candidate list.
    Duplicate ids collapse - a context is a SET of memories.
    """
    if ctx is not None:
        raw: Iterable[str] = ctx.included
    elif plan.context_plan.strategy == "full_candidate":
        raw = [m.id for m in candidates]
    else:
        k = max(1, plan.context_plan.selected_candidates)
        ranked = sorted(candidates, key=lambda m: m.utility(), reverse=True)
        raw = [m.id for m in ranked[:k]]
    return list(dict.fromkeys(raw))


def lower_plan(plan: RuntimePlan, request: RequestState,
               candidates: Sequence[MemoryObject],
               ctx: Optional[ExecutionContext] = None,
               request_id: str = "") -> ExecutionGraph:
    """§57: lower one planned request into its naive execution graph."""
    instructions: List[IROp] = []
    ids = {m.id for m in candidates}
    resident = {m.id for m in candidates if m.cached}
    by_id: Dict[str, MemoryObject] = {m.id: m for m in candidates}

    # 1. Lethe recall: every entry of the candidate stream AS GIVEN. A
    #    multi-query stream can name the same memory twice; lowering stays
    #    faithful (no early optimization) and §58's redundant-recall pass
    #    dedupes at the IR level where it can be measured.
    for m in candidates:
        instructions.append(IROp(MEMORY_RECALL, target=m.id,
                                 tokens=m.token_cost, origin="candidates"))

    # 2. Context optimization: what the (budgeted) compiler kept.
    selected = _context_ids(plan, candidates, ctx)
    for mid in selected:
        tokens = by_id[mid].token_cost if mid in by_id else 0
        instructions.append(IROp(CONTEXT_SELECT, target=mid, tokens=tokens,
                                 origin="context_plan"))

    # 3. Prefix reuse probe.
    if request.shared_prefix_length > 0:
        instructions.append(IROp(PREFIX_LOOKUP, target="prefix",
                                 tokens=request.shared_prefix_length,
                                 origin="request"))

    # 4a. Cache placement: keep resident KV the optimizer retained, evict the rest.
    if plan.fallback_plan.trigger:
        retained = set()                    # §105: fallback keeps nothing
    elif ctx is not None:
        retained = set(ctx.retained_kv)
    elif plan.cache_plan.retention_action == "KEEP":
        retained = set(resident)
    else:
        retained = set()
    keep = resident & retained
    evict = resident - retained
    for mid in sorted(keep):
        instructions.append(IROp(KV_KEEP, target=mid,
                                 tokens=by_id[mid].token_cost,
                                 origin="cache_plan"))
    for mid in sorted(evict):
        instructions.append(IROp(KV_EVICT, target=mid,
                                 tokens=by_id[mid].token_cost,
                                 origin="cache_plan"))

    # 4b. Predictive prefetch (§105 fallback forbids it).
    if plan.prefetch_plan.enabled and not plan.fallback_plan.trigger:
        default_tokens = (request.shared_prefix_length
                          or _PREDICTED_CHUNK_FALLBACK_TOKENS)
        for target in plan.prefetch_plan.targets:
            tokens = (by_id[target].token_cost if target in ids
                      else default_tokens)
            instructions.append(IROp(KV_PREFETCH, target=target,
                                     tokens=tokens, origin="prefetch_plan"))
            instructions.append(IROp(TRANSFER, target=target, tokens=tokens,
                                     origin="prefetch_plan"))

    # 5. Execution.
    instructions.append(IROp(PREFILL, target="request",
                             tokens=max(0, request.input_tokens),
                             origin="execution_plan"))
    instructions.append(IROp(DECODE, target="request",
                             tokens=max(0, request.expected_output_tokens),
                             origin="execution_plan"))

    graph = ExecutionGraph(request_id=request_id, instructions=instructions)
    graph.meta["strategy"] = plan.context_plan.strategy
    graph.meta["fallback"] = bool(plan.fallback_plan.trigger)
    return graph


def with_ir(trace, graph: ExecutionGraph):
    """§32/§99: attach the lowered IR summary to an existing DecisionTrace."""
    import dataclasses
    return dataclasses.replace(trace, ir_summary=graph.to_trace_summary())