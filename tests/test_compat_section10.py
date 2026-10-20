"""§8 + §10 integration test: compatibility changes predictive decision."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from semantic.compatibility import CompatibilityRequest, ComputationalCompatibilityChecker
from semantic.compat_integration import compute_effective_probability


def test_compat_block_zeroes_effective_prob():
    checker = ComputationalCompatibilityChecker()
    base = CompatibilityRequest(model_id="m-A", tokenizer_id="t-A",
                                 prefix_hash="ph-01", tensor_shape="[2,64,1024]")
    same = CompatibilityRequest(model_id="m-A", tokenizer_id="t-A",
                                 prefix_hash="ph-01", tensor_shape="[2,64,1024]")
    diff_shape = CompatibilityRequest(model_id="m-A", tokenizer_id="t-A",
                                       prefix_hash="ph-01", tensor_shape="[4,64,1024]")

    assert checker.check(base, same).compatible is True
    assert checker.check(base, diff_shape).compatible is False

    eff_same = compute_effective_probability(0.85, checker.check(base, same))
    eff_diff = compute_effective_probability(0.85, checker.check(base, diff_shape))
    assert eff_same["effective_reuse_probability"] == 0.85
    assert eff_diff["effective_reuse_probability"] == 0.0
    assert "reuse_block_reason" in eff_diff
