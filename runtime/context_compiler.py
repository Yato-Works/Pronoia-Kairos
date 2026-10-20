"""Phase 4 — Joint Context / Cache budget (Plan.md Section 21, 22, 23, 66, 67, 68, 91).

Section 23 is the thesis of this module: do NOT treat "what goes into the context"
and "what stays in the KV cache" as separate problems. Solve them as ONE resource
allocation problem:

    minimize   C_prefill + C_restore + C_memory + C_transfer
    subject to QualityLoss <= epsilon
               VRAM <= B_GPU
               RAM  <= B_CPU

The ContextCompiler (Section 21) turns Raw Conversation + Retrieved Memory +
Current Request + Runtime Budget into an Execution Context. The JointBudget-
Optimizer (Section 23) then decides, jointly:

    - which memory objects to INCLUDE in the context (context selection), and
    - which prefixes to RETAIN in the KV cache (KV retention),

under a single token/memory budget and a quality-safety constraint
(QualityLoss <= epsilon, Section 22/68). Both decisions draw from the same
budget and the same utility model (Section 22: utility(m) = Benefit - TokenCost -
PrefillCost), so a memory object either earns its context slot AND its KV
residency, or neither.

The Section 91 Gate is a Pareto-frontier improvement with bounded quality
degradation (Section 66/67): at the same memory budget, the joint optimizer
achieves higher SLO goodput than the greedy "context-only" and "cache-only"
baselines, while keeping QualityLoss <= epsilon.

Everything is abstract time/token units, no GPU (Section 43).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class MemoryObject:
    """A retrieved memory / context candidate (Section 21/22).

    id           - identity (prefix hash proxy)
    benefit      - expected answer-quality contribution if included
    token_cost   - tokens it adds to the context (prefill work)
    prefill_cost - compute to prefill its KV if not cached
    reuse_prob   - P_future: probability it is reused by a near-future request
    quality_loss - answer-quality lost if this object is DROPPED (Section 68)
    cached       - already resident in KV (no prefill needed)
    """
    id: str
    benefit: float = 0.0
    token_cost: int = 0
    prefill_cost: float = 0.0
    reuse_prob: float = 0.0
    quality_loss: float = 0.0
    cached: bool = False

    def utility(self) -> float:
        """Section 22 basic form: U(m) = Benefit - TokenCost - PrefillCost.
        (costs are normalized into benefit units for ranking.)"""
        cost = 0.01 * self.token_cost + self.prefill_cost
        return self.benefit - cost

    def retention_value(self) -> float:
        """Section 13 value: P_reuse * (recompute - restore). Approximate with
        reuse_prob * prefill_cost (recompute saved if reused and kept)."""
        return self.reuse_prob * self.prefill_cost


@dataclass
class ExecutionContext:
    """Section 21: the compiled context + the KV residency decision (Section 23)."""
    included: List[str] = field(default_factory=list)     # memory ids in context
    retained_kv: List[str] = field(default_factory=list)  # memory ids kept in KV
    total_tokens: int = 0
    quality_loss: float = 0.0
    cost: float = 0.0


class ContextCompiler:
    """Section 21: compile Raw Conversation + Memory + Request + Budget into an
    Execution Context (context selection only; KV decision delegated to the
    joint optimizer, Section 23)."""

    def __init__(self, token_budget: int = 4096):
        self.token_budget = token_budget

    def compile(self, memories: List[MemoryObject]) -> ExecutionContext:
        """Greedy context selection by utility under the token budget.

        Includes memories in descending utility until the token budget is hit.
        This is the "context-only" baseline the joint optimizer must beat.
        """
        ranked = sorted(memories, key=lambda m: m.utility(), reverse=True)
        included: List[str] = []
        total_tokens = 0
        for m in ranked:
            if total_tokens + m.token_cost > self.token_budget:
                continue
            included.append(m.id)
            total_tokens += m.token_cost
        return ExecutionContext(included=included, total_tokens=total_tokens)


class JointBudgetOptimizer:
    """Section 23: solve context selection AND KV retention as ONE allocation.

    A single pass ranks memory objects by a JOINT score that blends their context
    utility (Section 22) and their KV retention value (Section 13), then admits
    each object into BOTH the context and the KV cache while the shared budget
    allows, respecting the quality-safety constraint (QualityLoss <= epsilon).

    The joint score rewards an object that is BOTH useful in the context AND
    likely to be reused (keep its KV): this is the coupling that a context-only
    or cache-only policy misses.
    """

    def __init__(self, token_budget: int = 4096, vram_budget: int = 4096,
                 quality_epsilon: float = 0.05,
                 context_weight: float = 1.0, retention_weight: float = 1.0,
                 quality_weight: float = 2.0):
        self.token_budget = token_budget
        self.vram_budget = vram_budget
        self.quality_epsilon = quality_epsilon
        self.context_weight = context_weight
        self.retention_weight = retention_weight
        self.quality_weight = quality_weight

    def joint_score(self, m: MemoryObject) -> float:
        """Blend context utility and KV retention value (Section 22 + Section 13).

        A quality_safety term lifts objects whose DROP would cost answer quality
        (Section 68): high quality_loss -> include early so the QualityLoss<=eps
        constraint is satisfiable. This couples all three objectives (context,
        KV, quality) in one ranking, as Section 23 intends.
        """
        quality_safety = m.quality_loss
        return (self.context_weight * m.utility()
                + self.retention_weight * m.retention_value()
                + self.quality_weight * quality_safety)

    def optimize(self, memories: List[MemoryObject]) -> ExecutionContext:
        # Rank by joint score, but guarantee the quality-safety constraint first:
        # objects whose DROP would exceed epsilon are force-included (Section 68),
        # so we sort quality-critical objects to the front via the quality term
        # AND enforce the constraint in the admission loop.
        ranked = sorted(memories, key=self.joint_score, reverse=True)
        ctx = ExecutionContext()
        used_tokens = 0
        used_vram = 0
        admitted_quality_loss = 0.0
        for m in ranked:
            fits = (used_tokens + m.token_cost <= self.token_budget
                    and used_vram + m.token_cost <= self.vram_budget)
            # Force-include if dropping this object would breach the quality
            # budget (Section 68): the loss of everything not-yet-admitted that is
            # worse-ranked than m must stay within epsilon.
            remaining_quality = sum(x.quality_loss for x in memories
                                    if x.id not in ctx.included and x.id != m.id)
            force = (remaining_quality > self.quality_epsilon) and (
                m.quality_loss > 0)
            if not fits and not force:
                continue
            ctx.included.append(m.id)
            admitted_quality_loss += m.quality_loss
            used_tokens += m.token_cost
            # Retain KV for objects likely to be reused (the KV half, Section 23).
            if m.reuse_prob > 0.3:
                ctx.retained_kv.append(m.id)
                used_vram += m.token_cost
        included_set = set(ctx.included)
        dropped = sum(m.quality_loss for m in memories if m.id not in included_set)
        ctx.total_tokens = used_tokens
        ctx.quality_loss = dropped
        ctx.cost = self._cost(memories, ctx)
        return ctx

    def _cost(self, memories: List[MemoryObject], ctx: ExecutionContext) -> float:
        """Section 23 objective: C_prefill + C_restore + C_memory + C_transfer.

        C_prefill  = prefill of included-but-not-cached context objects.
        C_restore  = restore of retained KV that was not already cached.
        C_memory   = VRAM occupied by retained KV.
        C_transfer = transfer of retained KV (proxy: tokens).
        """
        by_id = {m.id: m for m in memories}
        inc = set(ctx.included)
        ret = set(ctx.retained_kv)
        c_prefill = sum(by_id[i].prefill_cost for i in inc if not by_id[i].cached)
        c_restore = sum(by_id[i].prefill_cost * 0.1 for i in ret if not by_id[i].cached)
        c_memory = sum(0.01 * by_id[i].token_cost for i in ret)
        c_transfer = sum(0.005 * by_id[i].token_cost for i in ret)
        return c_prefill + c_restore + c_memory + c_transfer
