"""§58 IR Optimization — passes over the Cognitive IR (Phase 5 subset).

Implemented here (measurable in this prototype):

    Redundant Recall Elimination  - duplicate MEMORY_RECALL of one target
                                    (multi-query recall streams overlap).
    Dead Context Elimination      - MEMORY_RECALL whose target never reaches a
                                    CONTEXT_SELECT: recalled, evaluated,
                                    dropped by the budget - pure wasted work.
    Prefetch Hoisting             - move KV_PREFETCH + TRANSFER to the graph
                                    head. Legal because prefetch targets come
                                    from the planner's prediction, not from
                                    this request's recall/selection; earlier
                                    issue widens the §36 overlap window.

§58's other candidates (Cache-aware Scheduling, Transfer Scheduling,
Execution Fusion) reorder or merge work whose benefit this control-plane
prototype cannot yet measure honestly - deliberately left out rather than
claimed with invented numbers (§58: do not let IR speedup become its own
research theme).

Every pass is *structure-preserving*: the result stays well-formed per
ExecutionGraph.problems() (TRANSFER never orphaned from its KV_PREFETCH,
PREFILL still before DECODE).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from runtime.ir import (
    CONTEXT_SELECT, KV_PREFETCH, MEMORY_RECALL, TRANSFER, ExecutionGraph,
)

PASS_REDUNDANT_RECALL = "redundant_recall_elimination"
PASS_DEAD_CONTEXT = "dead_context_elimination"
PASS_PREFETCH_HOISTING = "prefetch_hoisting"


@dataclass
class PassOutcome:
    """Result of running one pass: the new graph plus what it changed."""
    graph: ExecutionGraph
    removed_by_kind: Dict[str, int] = field(default_factory=dict)
    hoisted_transfers: int = 0

    @property
    def removed_ops(self) -> int:
        return sum(self.removed_by_kind.values())


@dataclass
class PassReport:
    """Aggregated §58 pass activity for one graph (goes into the decision
    trace via ExecutionGraph.to_trace_summary())."""
    passes_applied: List[str] = field(default_factory=list)
    removed_by_kind: Dict[str, int] = field(default_factory=dict)
    hoisted_transfers: int = 0
    ops_before: int = 0
    ops_after: int = 0

    @property
    def ops_removed(self) -> int:
        return self.ops_before - self.ops_after

    def to_dict(self) -> Dict[str, object]:
        return {
            "passes": list(self.passes_applied),
            "removed_by_kind": dict(self.removed_by_kind),
            "hoisted_transfers": self.hoisted_transfers,
            "ops_before": self.ops_before,
            "ops_after": self.ops_after,
            "ops_removed": self.ops_removed,
        }


def _derive(graph: ExecutionGraph, instructions: List) -> ExecutionGraph:
    """New graph sharing request_id/meta but carrying new instructions."""
    return ExecutionGraph(request_id=graph.request_id,
                          instructions=instructions,
                          meta=dict(graph.meta))


def redundant_recall_elimination(graph: ExecutionGraph) -> PassOutcome:
    """§58: keep only the first MEMORY_RECALL per target (recalls are
    idempotent fetches of the same memory)."""
    kept: List = []
    seen = set()
    removed = 0
    for op in graph.instructions:
        if op.kind == MEMORY_RECALL:
            if op.target in seen:
                removed += 1
                continue
            seen.add(op.target)
        kept.append(op)
    if not removed:
        return PassOutcome(graph)
    return PassOutcome(_derive(graph, kept),
                       removed_by_kind={MEMORY_RECALL: removed})


def dead_context_elimination(graph: ExecutionGraph) -> PassOutcome:
    """§58: drop MEMORY_RECALL of targets the context never selects.

    Without any CONTEXT_SELECT there is no evidence of what the context
    consumes, so the pass refuses to guess and changes nothing.
    """
    selects = [op for op in graph.instructions if op.kind == CONTEXT_SELECT]
    if not selects:
        return PassOutcome(graph)
    selected = {op.target for op in selects}
    kept: List = []
    removed = 0
    for op in graph.instructions:
        if op.kind == MEMORY_RECALL and op.target not in selected:
            removed += 1
            continue
        kept.append(op)
    if not removed:
        return PassOutcome(graph)
    return PassOutcome(_derive(graph, kept),
                       removed_by_kind={MEMORY_RECALL: removed})


def prefetch_hoisting(graph: ExecutionGraph) -> PassOutcome:
    """§58: move every KV_PREFETCH + TRANSFER pair to the head of the graph.

    The pair block keeps its internal order (TRANSFER stays behind its
    KV_PREFETCH), so the graph stays valid; issuing earlier is what widens
    the §36 overlap window.
    """
    ops = graph.instructions
    lane_idx = [i for i, op in enumerate(ops)
                if op.kind in (KV_PREFETCH, TRANSFER)]
    if not lane_idx or lane_idx[0] == 0:
        return PassOutcome(graph)          # nothing to hoist
    lane_set = set(lane_idx)
    lane = [ops[i] for i in lane_idx]
    rest = [op for i, op in enumerate(ops) if i not in lane_set]
    moved = sum(1 for op in lane if op.kind == TRANSFER)
    return PassOutcome(_derive(graph, lane + rest),
                       hoisted_transfers=moved)


DEFAULT_PASSES: Tuple[Callable[[ExecutionGraph], PassOutcome], ...] = (
    redundant_recall_elimination,
    dead_context_elimination,
    prefetch_hoisting,
)


def optimize(graph: ExecutionGraph,
             passes: Optional[List[Callable[[ExecutionGraph], PassOutcome]]] = None
             ) -> Tuple[ExecutionGraph, PassReport]:
    """Run the §58 passes in order; return the optimized graph + report."""
    report = PassReport(ops_before=len(graph))
    current = graph
    for fn in (passes if passes is not None else DEFAULT_PASSES):
        outcome = fn(current)
        current = outcome.graph
        report.passes_applied.append(fn.__name__)
        for kind, n in outcome.removed_by_kind.items():
            report.removed_by_kind[kind] = report.removed_by_kind.get(kind, 0) + n
        report.hoisted_transfers += outcome.hoisted_transfers
    report.ops_after = len(current)
    current.meta["pass_report"] = report.to_dict()
    return current, report