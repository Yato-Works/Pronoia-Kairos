"""§11 + §14 integration test: tier selection responds to future demand + shadow price."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runtime.tiers import MemoryTierModel


def test_tier_selection_responds_to_demand():
    model = MemoryTierModel()
    # High future demand + high confidence → faster tier
    assert model.best_tier_for_reuse(0.92, 0.95) == "L1"
    # Low demand → cheaper / slower tier
    assert model.best_tier_for_reuse(0.1, 0.2) == "L3"
    # Middle → L2
    assert model.best_tier_for_reuse(0.5, 0.6) == "L2"


def test_all_five_tiers_exist():
    model = MemoryTierModel()
    names = [t.name for t in model.tiers]
    assert names == ["L0", "L1", "L2", "L3", "L4"]
