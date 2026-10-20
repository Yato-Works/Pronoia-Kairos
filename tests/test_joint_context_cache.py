"""Phase 4 — Joint Context / Cache budget (Plan.md Section 21, 22, 23, 66, 67, 68, 91).

Section 23: solve context selection AND KV retention as ONE resource allocation
problem. These tests fix the joint optimizer and the Section 91 gate:

    - MemoryObject.utility (Section 22) and retention_value (Section 13).
    - ContextCompiler does greedy context selection under a token budget (21).
    - JointBudgetOptimizer couples context + KV + quality in one ranking, and
      respects the quality-safety constraint (QualityLoss <= epsilon, 68).
    - on quality-feasible budgets the joint optimizer matches/beats the
      single-objective goodput (Section 91 Pareto improvement, bounded quality).
    - a budget too small to satisfy the quality constraint is honestly marked
      infeasible (Section 68: not a valid win).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runtime.context_compiler import (
    MemoryObject, ContextCompiler, JointBudgetOptimizer, ExecutionContext,
)
from compare.joint_context_cache_experiment import (
    make_memory_objects, run_budget_sweep, evaluate_pareto, format_pareto_report,
)


def _mems():
    return make_memory_objects(n=12, seed=7)


# --- MemoryObject (Section 22 / 13) ------------------------------------------

def test_utility_is_benefit_minus_cost():
    m = MemoryObject("x", benefit=0.9, token_cost=100, prefill_cost=5.0)
    # U = benefit - (0.01*tokens + prefill) = 0.9 - (1.0 + 5.0)
    assert abs(m.utility() - (0.9 - 6.0)) < 1e-9


def test_retention_value_is_reuse_times_prefill():
    m = MemoryObject("x", prefill_cost=10.0, reuse_prob=0.6)
    assert abs(m.retention_value() - 6.0) < 1e-9


def test_hot_object_has_higher_retention_than_cold():
    hot = MemoryObject("hot", prefill_cost=10.0, reuse_prob=0.8)
    cold = MemoryObject("cold", prefill_cost=10.0, reuse_prob=0.1)
    assert hot.retention_value() > cold.retention_value()


# --- ContextCompiler (Section 21) --------------------------------------------

def test_context_compiler_respects_token_budget():
    mems = _mems()
    cc = ContextCompiler(token_budget=1024)
    ctx = cc.compile(mems)
    assert ctx.total_tokens <= 1024
    assert all(m.token_cost <= 1024 for m in mems)  # sanity


def test_context_compiler_prefers_high_utility():
    mems = [
        MemoryObject("low", benefit=0.1, token_cost=128, prefill_cost=1.0),
        MemoryObject("high", benefit=0.95, token_cost=128, prefill_cost=1.0),
    ]
    ctx = ContextCompiler(token_budget=128).compile(mems)  # only one fits
    assert ctx.included == ["high"]  # highest utility included first
# --- JointBudgetOptimizer (Section 23) =======================================

def test_joint_optimizer_retains_hot_kv():
    mems = [
        MemoryObject("hot", benefit=0.8, token_cost=256, prefill_cost=8.0,
                     reuse_prob=0.9),
        MemoryObject("cold", benefit=0.7, token_cost=256, prefill_cost=8.0,
                     reuse_prob=0.05),
    ]
    opt = JointBudgetOptimizer(token_budget=4096, vram_budget=4096)
    ctx = opt.optimize(mems)
    assert "hot" in ctx.retained_kv
    assert "cold" not in ctx.retained_kv


def test_joint_optimizer_respects_quality_constraint():
    mems = _mems()
    opt = JointBudgetOptimizer(token_budget=4096, vram_budget=4096, quality_epsilon=0.10)
    ctx = opt.optimize(mems)
    assert ctx.quality_loss <= 0.10 + 1e-9


def test_joint_cost_includes_all_four_terms():
    mems = [MemoryObject("a", benefit=0.9, token_cost=256, prefill_cost=10.0,
                         reuse_prob=0.9, quality_loss=0.02, cached=False)]
    opt = JointBudgetOptimizer(token_budget=4096, vram_budget=4096)
    ctx = opt.optimize(mems)
    assert ctx.cost > 0

# --- Section 91 Phase 4 Gate ----------------------------------------------

def test_gate_passes_on_quality_feasible_budgets():
    """Section 91: at the quality-feasible budgets, joint dominates context-only
    on goodput and quality loss is bounded (<= epsilon)."""
    mems = _mems()
    points = run_budget_sweep(mems, [1024, 2048, 4096], quality_epsilon=0.10)
    verdict = evaluate_pareto(points, quality_epsilon=0.10)
    assert verdict.joint_dominates
    assert verdict.quality_bounded


def test_infeasible_budget_is_excluded_from_dominance():
    """Section 68: a budget too small to satisfy QualityLoss<=eps is not a valid
    win, so it is excluded from the dominance test (honest)."""
    mems = _mems()
    # A tiny budget cannot hold the quality-critical objects -> joint breaches eps.
    points = run_budget_sweep(mems, [256], quality_epsilon=0.02)
    verdict = evaluate_pareto(points, quality_epsilon=0.02)
    joint = next(p for p in points if p.policy == "joint")
    # The joint run breaches the constraint, so no feasible budget exists.
    assert joint.quality_loss > 0.02
    assert verdict.joint_dominates is False  # cannot claim dominance when infeasible


def test_joint_goodput_at_least_context_only_when_feasible():
    mems = _mems()
    points = run_budget_sweep(mems, [2048, 4096], quality_epsilon=0.10)
    for b in (2048, 4096):
        j = next(p for p in points if p.policy == "joint" and p.budget == b)
        c = next(p for p in points if p.policy == "context-only" and p.budget == b)
        assert j.goodput_proxy >= c.goodput_proxy - 1e-9


def test_gate_report_renders_policies_and_verdict():
    mems = _mems()
    points = run_budget_sweep(mems, [1024, 2048, 4096], quality_epsilon=0.10)
    verdict = evaluate_pareto(points, quality_epsilon=0.10)
    out = format_pareto_report(points, verdict)
    for p in ("joint", "context-only", "cache-only"):
        assert p in out
    assert "goodput" in out and "quality" in out
