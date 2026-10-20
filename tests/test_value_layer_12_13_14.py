"""§12 / §13 / §14 integration tests — pure function verification."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runtime.value_layer import (
    LOSSLESS_OPERATIONS, CacheOperation, CacheValueInputs,
    U_lossless, ShadowPrices, utility_tier,
)


def test_lossless_operations_defined():
    assert "KEEP" in LOSSLESS_OPERATIONS
    assert "PREFETCH" in LOSSLESS_OPERATIONS
    assert "RESTORE" in LOSSLESS_OPERATIONS
    assert len(LOSSLESS_OPERATIONS) == 8


def test_cache_value_function_positive():
    # When P_reuse is high and T_recompute >> T_restore, value should be positive
    inputs = CacheValueInputs(
        P_reuse=0.9,
        T_recompute=100.0,
        T_restore=5.0,
        C_retain=10.0,
    )
    val = U_lossless(inputs)
    # 0.9 * (100 - 5) - 10 = 85.5 - 10 = 75.5
    assert val > 70
    assert val == 75.5


def test_cache_value_negative_when_cost_exceeds_benefit():
    # Low reuse + high retention cost → negative value (evict preferred)
    inputs = CacheValueInputs(
        P_reuse=0.05,
        T_recompute=10.0,
        T_restore=8.0,
        C_retain=20.0,
    )
    val = U_lossless(inputs)
    # 0.05 * (10 - 8) - 20 = 0.1 - 20 = -19.9
    assert val < 0


def test_shadow_prices_and_utility():
    shadow = ShadowPrices(lambda_gpu=1.0, lambda_cpu=0.3, lambda_ssd=0.1, lambda_network=0.02)
    # High value (good reuse) on GPU → positive utility
    assert utility_tier(50.0, shadow, "L1") > 0  # 50 - 1 = 49
    # Low value on expensive GPU → negative utility (better off cheaper tier)
    assert utility_tier(0.5, shadow, "L0") < 0   # 0.5 - 1.5 = -1.0 (using L0 cost factor 1.5)
