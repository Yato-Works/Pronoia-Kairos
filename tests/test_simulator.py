"""Phase 1 + Phase 2 integration test: Simulator → Belady/LRU comparison."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.phase1 import Simulator, SimulationConfig
from compare.belady_lru import belady_optimal, lru_hit_count


def test_simulator_runs_sequence():
    sim = Simulator()
    config = SimulationConfig(memory_capacity=3, access_sequence=["A","B","C","D","A","B","E","A","B","C","D"])
    metrics = sim.run_baseline_sequence(config.access_sequence, config)
    assert metrics.total_requests == len(config.access_sequence)
    assert metrics.hits + metrics.misses == metrics.total_requests


def test_belady_vs_lru_on_sample():
    seq = ["A","B","C","D","A","B","E","A","B","C","D"]
    cap = 3
    belady_hits = belady_optimal(seq, cap)
    lru_hits = lru_hit_count(seq, cap)
    # Belady should be >= LRU always (theoretical upper bound)
    assert belady_hits >= lru_hits
    # For this sequence, Belady typically achieves more hits
    print(f"Sequence: {seq}")
    print(f"Belady hits: {belady_hits}, LRU hits: {lru_hits}, Regret (LRU-Belady): {lru_hits - belady_hits}")
