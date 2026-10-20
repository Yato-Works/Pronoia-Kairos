"""§10 Integration: effective_reuse_probability with compatibility.

Effective probability (§10 + §9 connection):

    P_effective_reuse = P_future_demand  IF compatible
                         0.0            IF incompatible

The result includes why reuse was blocked — feeds DecisionTrace.
"""
from semantic.compatibility import CompatibilityRequest, CompatibilityResult, ComputationalCompatibilityChecker
from semantic.demand_predictor import DummySemanticDemandPredictor, MemoryRecord
from runtime.state import RuntimeState, RequestState
from runtime.request import RequestAnalyzer
from runtime.planner import AdaptivePlanner


def compute_effective_probability(
    future_use_prob: float,
    result: CompatibilityResult,
) -> dict:
    if result.compatible:
        effective = future_use_prob
    else:
        effective = 0.0
    return {
        "future_use_probability": future_use_prob,
        "computational_compatible": result.compatible,
        "effective_reuse_probability": effective,
        "reuse_block_reason": result.reason,
        "compatibility_result": result.to_dict(),
    }


def test_compatibility_decision_affects_plan():
    """§8→§10 integration preview: compatibility affects planner output.
    Not a full RuntimeState integration yet (§8 next), but validates
    the core principle: semantic + computational together determine reuse."""
    checker = ComputationalCompatibilityChecker()
    current = CompatibilityRequest(
        model_id="m-1", tokenizer_id="t-1",
        prefix_hash="h-abc", tensor_shape="[1, 8, 512]",
    )
    compatible_candidate = CompatibilityRequest(
        model_id="m-1", tokenizer_id="t-1",
        prefix_hash="h-abc", tensor_shape="[1, 8, 512]",
    )
    incompatible_candidate = CompatibilityRequest(
        model_id="m-2", tokenizer_id="t-1",
        prefix_hash="h-abc", tensor_shape="[1, 8, 512]",
    )

    r_compat = checker.check(current, compatible_candidate)
    r_incompat = checker.check(current, incompatible_candidate)

    assert r_compat.compatible is True
    assert r_compat.matched_fields == ["model_id", "tokenizer_id", "prefix_hash", "tensor_shape"]
    assert r_incompat.compatible is False
    assert "model_id" in r_incompat.mismatched_fields

    # Effective probability: compatible keeps future demand, incompatible zeroes it
    high_demand = 0.87
    eff_compat = compute_effective_probability(high_demand, r_compat)
    eff_incompat = compute_effective_probability(high_demand, r_incompat)
    assert eff_compat["effective_reuse_probability"] == 0.87
    assert eff_incompat["effective_reuse_probability"] == 0.0
    assert eff_incompat["reuse_block_reason"].startswith("mismatch:")
