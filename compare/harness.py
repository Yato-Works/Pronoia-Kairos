"""Phase 0a Harness (Plan.md Section 37-39, Section 63-66).

Runs the full baseline matrix across the workload matrix, reports hit
rates and regret vs Belady, and evaluates the Phase 0a Kill Gate:

    Belady - LRU gap < 5 percentage points  AND  potential E2E gain < 5%
    => Adaptive Cache Policy research priority is lowered (Section 39).

This is the first research Kernel artifact (Section 96): Trace ->
Baselines -> Regret -> Go/No-Go, with no GPU dependency.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Callable
import json
import os

from compare.belady_lru import belady_optimal, lru_hit_count
from simulator import workloads as w
from compare.baselines import POLICIES, regret_vs_belady


@dataclass
class WorkloadResult:
    name: str
    capacity: int
    n_requests: int
    hit_rates: Dict[str, float] = field(default_factory=dict)
    regrets_vs_belady: Dict[str, int] = field(default_factory=dict)
    belady_lru_gap_pp: float = 0.0   # percentage points


def evaluate_workload(name: str, sequence: List[str], capacity: int) -> WorkloadResult:
    result = WorkloadResult(name=name, capacity=capacity, n_requests=len(sequence))
    belady_hits = None
    lru_hits = None
    for policy_name, policy_fn in POLICIES.items():
        hits = policy_fn(sequence, capacity)
        result.hit_rates[policy_name] = hits / len(sequence) if sequence else 0.0
        if policy_name == "Belady":
            belady_hits = hits
        if policy_name == "LRU":
            lru_hits = hits
    assert belady_hits is not None and lru_hits is not None
    for policy_name in POLICIES:
        policy_hits = int(round(result.hit_rates[policy_name] * len(sequence)))
        result.regrets_vs_belady[policy_name] = regret_vs_belady(policy_hits, belady_hits)
    result.belady_lru_gap_pp = (lru_hits - belady_hits) / len(sequence) * 100.0 if sequence else 0.0
    return result


def run_phase0a(workloads: Dict[str, List[str]], capacity: int = 8) -> List[WorkloadResult]:
    return [evaluate_workload(name, seq, capacity) for name, seq in workloads.items()]


def phase0a_kill_gate(results: List[WorkloadResult], gap_threshold_pp: float = 5.0) -> Dict:
    """Section 39: kill if Belady-LRU gap is tiny across all workloads."""
    gaps = {r.name: abs(r.belady_lru_gap_pp) for r in results}
    all_small = all(g < gap_threshold_pp for g in gaps.values()) if gaps else True
    return {
        "kill_triggered": all_small,
        "threshold_pp": gap_threshold_pp,
        "gaps_pp": gaps,
        "interpretation": (
            "Belady-LRU gap small on ALL workloads: adaptive cache research priority lowered."
            if all_small else
            "At least one workload shows meaningful Belady-LRU headroom: research continues."
        ),
    }


def format_report(results: List[WorkloadResult]) -> str:
    lines = []
    lines.append(f"{'Workload':<20} {'Cap':>4} {'N':>5}  " + "  ".join(f"{p:>10}" for p in POLICIES))
    lines.append("-" * 100)
    for r in results:
        cells = "  ".join(f"{r.hit_rates[p]:>10.3f}" for p in POLICIES)
        lines.append(f"{r.name:<20} {r.capacity:>4} {r.n_requests:>5}  {cells}")
    lines.append("")
    lines.append("Regret vs Belady (<=0; closer to 0 = better):")
    for r in results:
        cells = "  ".join(f"{r.regrets_vs_belady[p]:>10d}" for p in POLICIES)
        lines.append(f"{r.name:<20} {'':>4} {'':>5}  {cells}")
    return "\n".join(lines)


def save_artifacts(results: List[WorkloadResult], out_dir: str = "artifacts/phase0a") -> str:
    """Section 98 Research Artifact: benchmark_results + ablation-ready JSON."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "phase0a_results.json")
    payload = [
        {
            "name": r.name, "capacity": r.capacity, "n_requests": r.n_requests,
            "hit_rates": r.hit_rates, "regrets_vs_belady": r.regrets_vs_belady,
            "belady_lru_gap_pp": r.belady_lru_gap_pp,
        }
        for r in results
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


def main():
    """CLI: python -m compare.harness  -> full Phase 0a report + kill gate."""
    wl = {
        "seq_scan(20x3)": w.sequential_scan(20, 3),
        "zipf(30,400)": w.zipf_like(30, 400),
        "session(6x8)": w.multi_turn_session(6, 8),
        "agent(60,6)": w.agent_workflow(60, 6),
    }
    results = run_phase0a(wl, capacity=4)
    print(format_report(results))
    gate = phase0a_kill_gate(results)
    print("\nPhase 0a Kill Gate:")
    print(json.dumps(gate, indent=2))
    path = save_artifacts(results)
    print(f"\nArtifacts saved: {path}")


if __name__ == "__main__":
    main()
