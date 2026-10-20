# Pronoia-Kairos — Adaptive Inference Runtime

**English** | [日本語 (Japanese)](./README.ja.md)

> **Status: V1.0.0 Release Candidate** — Open for reproduction review.
> **The official v1.0.0 release tag is pending independent reproduction** (not yet performed — see [`docs/RELEASE_CHECKLIST.md`](./docs/RELEASE_CHECKLIST.md), item 4).
> Instructions and evaluation criteria: **[`docs/REPRODUCTION_REQUEST.md`](./docs/REPRODUCTION_REQUEST.md)** (target commit: `c829714`).

A research prototype evaluating whether coordinating existing LLM inference cache mechanisms (LRU, prefix caching, queue-aware scheduling, and prefetching) at the runtime layer via a **shared signal termed Semantic Demand** delivers additional end-to-end performance gains.

**Central Hypothesis**: Establishing demand prediction ("which keys will be reused in the future") as a shared semantic demand signal across eviction, scheduling, prefetch, and context selection enables end-to-end performance gains unobtainable through individually optimized components.

This study investigates whether coordinating existing mechanisms through the shared Semantic Demand signal yields additional end-to-end benefits. The scope of confirmed effects and unverified aspects are detailed in the results and claim boundary below.
This repository **does NOT claim the invention** of KV caching, prefix caching, predictive eviction, queue-aware scheduling, or semantic memory ([`artifacts/phase6/CLAIM_BOUNDARY.md`](./artifacts/phase6/CLAIM_BOUNDARY.md)).

---

## Results Summary

### A. Simulation (Abstract Time Model, Deterministic)

| Evaluation | Configuration | Result | Provenance Artifact |
|---|---|---|---|
| Phase 6 Coordination Gate v2 | 8 workload families × 6 seeds, identical ladder | **Passed predefined gate conditions in 5 of 8 workloads** (W4–W8; W1–W3 showed policy-parity ties, hence non-superior) | [`artifacts/phase6/full_runtime_gate_v2.json`](./artifacts/phase6/full_runtime_gate_v2.json) |
| M1: Strong Baseline Re-evaluation | Same code as above, post-determinism fix | **Tier 2 (+20% vs. best individual configuration) attained in 1/8 → Not achieved**. Prior "7/8" claim retracted. Holm-corrected confirmatory positive in W5, W6, W8 | [`artifacts/phase6/strong_reeval.json`](./artifacts/phase6/strong_reeval.json) |
| M3: Causal Isolation of Semantic Signal | True vs. Shuffled (permuting mapping $\pi$ only), 96 paired runs | **Preregistered criteria yielded confirmatory positive in 6 of 8 families** (W2/W4/W5/W6/W7/W8; Holm(8)<0.05 and 95% CI excluding 0). W1 was directionless as predicted | [`artifacts/phase6/m3_shuffle_analysis.json`](./artifacts/phase6/m3_shuffle_analysis.json) |

> **Interpretation Guide**: The three rows above represent **distinct evaluations answering different questions and cannot be used interchangeably**.
> "Passing gate conditions (5/8)" does not imply confirmatory superiority over the strongest single configuration (≠ 5/8 confirmatory win), and "Tier 2 attainment of 1/8" does not invalidate the gate pass. M3's 6/8 outcome evaluates the causal contribution of the signal itself, orthogonal to the performance gates. See the locked preregistrations at [`artifacts/phase6/PREREG_*.md`](./artifacts/phase6/).

### B. Empirical GPU Measurement (M4 — RTX 3050 Real GPU, W6-inspired reduced workload)

Qwen2.5-0.5B-Instruct fp16 / eager / sequential batch=1 / 40 requests × 3 seeds × 2 arms (LRU baseline vs. queue_semantic). Detailed protocol: [`docs/REPRODUCE.md`](./docs/REPRODUCE.md).

| Check / Metric | Empirical Result |
|---|---|
| Correctness (MC0) | Output token sequences **match exactly** with and without cache |
| Logic Fidelity (MC2/MC3/MC5) | All 6 runs match `AdaptiveRuntime` reference implementation; verified LRU degeneration and deterministic reproducibility |
| Measurement Validity (MC4) | All 6 runs pass under amended slack (0.25). **Fails under original criterion (0.15)** — both determinations preserved in [`artifacts/phase6/MC4_ADJUDICATION.md`](./artifacts/phase6/MC4_ADJUDICATION.md) |
| Primary Metric: E2E Wall-Clock | **+2.66% improvement** (arithmetic mean of per-seed relative improvements across 3 seeds: seed1 +3.83%, seed2 +4.17%, seed3 −0.03%) |
| Prefill GPU Time | **+2.08% improvement** (arithmetic mean across 3 seeds: seed1 −1.53%, seed2 +7.62%, seed3 +0.15%). In seed2, token reuse increased from 7,704 to 10,899 tokens, directly reducing prefill duration |
| Control Overhead Ratio | **$r_{\mathrm{overhead}} = 0.0003$** (reference mean across the two seeds with positive gains, $n=2$; defined below) |

**Applicability Caveat**: These empirical results were obtained under constrained conditions (no concurrency, no queue backlog, compared strictly against LRU, 3-seed pilot). We do not claim a blanket "2.66% speedup across arbitrary workloads." Refer to the Claim Boundary for verified vs. unverified distinctions.

---

## Metric Definitions (M4)

### Relative Improvement Calculation
For each seed $s$, the relative improvement $I_s$ is defined as:

$$I_s = \frac{T_{\mathrm{LRU}, s} - T_{\mathrm{adaptive}, s}}{T_{\mathrm{LRU}, s}} \times 100\%$$

- **E2E Relative Improvement (+2.66%)**: The arithmetic mean of per-seed relative improvements across 3 seeds ($( +3.829\% + 4.166\% - 0.029\% ) / 3 = +2.655\% \approx +2.66\%$). Explicitly designated as an arithmetic mean to avoid misinterpretation as a single aggregated ratio over total runtime.
- **Prefill GPU Time Improvement (+2.08%)**: The arithmetic mean of per-seed relative improvements across 3 seeds ($( -1.528\% + 7.621\% + 0.148\% ) / 3 = +2.080\% \approx +2.08\%$).
- **E2E Wall-Clock Time (`e2e_wall_s`)** — Primary Metric. Elapsed wall time of the request processing loop excluding warmup and model loading (from first request start to last request completion). Includes only GPU execution and runtime control time (PREREG §8-1).

### Control Overhead Ratio ($r_{\mathrm{overhead}}$)

$$r_{\mathrm{overhead}} = \frac{T_{\mathrm{control, adaptive}}}{T_{\mathrm{LRU}} - T_{\mathrm{adaptive}}}$$

- **Definition**: The numerator is the cumulative runtime control time in the adaptive arm (`control_ms_total` measured via `perf_counter_ns`), and the denominator is the paired E2E wall-clock improvement ($T_{\mathrm{LRU}} - T_{\mathrm{adaptive}}$) for the same seed. Dimensionless (ms/ms).
- **Reference Metric ($n=2$)**: Defined only for seeds with a positive denominator. Seed3 is undefined (`null`) because its observed improvement was non-positive. The reported value $0.0003$ is the reference mean of the two positive seeds ($n=2$, seed1: 0.0004, seed2: 0.0003). It is not representative of all three seeds; the definitive evaluation relies on total E2E wall-clock time including control overhead.
- **PREREG Threshold**: The locked passing threshold is $r_{\mathrm{overhead}} < 0.10$ (control overhead must not exceed 10% of observed gains, PREREG §6). The empirical measurement ($0.0003$) satisfies this requirement.

---

## Research Claim Boundary (Honesty Boundary)

### Scope Confirmed in Simulation
- **Gate v2**: Passed predefined gate conditions in 5 of 8 workloads (W4–W8). This does not demonstrate a 5/8 confirmatory advantage over the strongest individual configuration.
- **Strong Baseline Re-evaluation (M1)**: Point estimates of relative improvement were positive in 5 families (W4–W8). However, confirmatory positive outcomes after Holm correction were observed only in 3 families (W5, W6, W8). The Tier 2 criterion ("+20% over strongest individual configuration") was achieved in only 1 of 8 families and is recorded as **not achieved**.
- **Semantic Demand Shuffle Control (M3)**: In the preregistered True-vs-Shuffled intervention, 6 of 8 families yielded confirmatory positive outcomes in simulation. This supports the functional role of the key-demand alignment in this setup, but does not establish an identical effect in production serving environments.
- *These evaluations are distinct, and the scope of claims supported by each must remain strictly separated.*

### Scope Confirmed in Empirical GPU Measurement
- Computation reduction via KV reuse in a constrained GPU execution regime (reused_tokens ↑ → prefill_tokens ↓ → prefill GPU time ↓), alongside the observed conditional E2E improvement (+2.66% arithmetic mean across 3 seeds).

### Unconfirmed Scope (Explicit Non-Claims)
- **Queueing benefits under high concurrency**: Mean queue depth in M4 was 1.0 (arrival gaps of 55–63 s ≫ service time of several seconds), resulting in negligible queue buildup.
- **Integrated effects of prefetching & EDF scheduling**: Not validated on physical hardware.
- **Statistical generalization beyond 3 seeds**: 3-seed pilot data is not used for formal statistical claims.
- **Production serving stack portability**: Reproducibility on Linux or production serving engines (such as vLLM) remains unverified.
- **Tier 2 Attainment**: Remains **unmet** at 1/8 families ([`artifacts/MILESTONES.md`](./artifacts/MILESTONES.md)). Systematic hardware benchmarks for Tier 4/5 require future research.
- **Superiority over strongest configurations**: M4 evaluated gains solely against LRU; superiority over advanced baselines (e.g., E_planning) is not claimed (Mentor Ruling D).

### Known Limitations
- M4 uses a W6-inspired **reduced** workload (40 requests, context ~1.8k tokens, batch=1); it does not reproduce the scale or multi-tenant concurrency of W6.
- Measurements were collected on an RTX 3050 with a 0.5B model, operating within a GPU clock fluctuation window (prefill peak-to-peak ~29%), which necessitated amending the MC4 slack threshold (§8-12).
- Future directions: assessing performance stability across additional seeds, varying load conditions, and deployment on Linux/production inference engines.

---

## Repository Architecture

```
runtime/        AdaptiveRuntime — Sole implementation of decisions (context,
                residency, victim, signals, guard). Governed by §8-8 direct-call rule.
semantic/       Semantic Demand prediction & calibration (M3 permutes key mapping π).
scheduler/      Scheduling policies (e.g., EDF).
compare/        Ladder configurations (A_naive through F), gates, M3 shuffle runners/analyzers.
simulator/      Abstract time simulation model.
m4_realworld/   M4 empirical GPU execution engine (executor, trie store, harness,
                manipulation checks, adjudication).
tests/          279 CPU unit/regression tests + optional GPU test (mc0_gpu).
artifacts/      Ground-truth logs, PREREGs, addenda, and audit trails (append-only).
```

---

## Reproducibility

To run the CPU test suite:

```powershell
python -m pytest tests/ -k "not mc0_gpu"
```

> **Expected outcome**: 279 passed, 1 deselected (no GPU required, completes in ~5 seconds).

For GPU execution, re-running M4, and regenerating adjudication reports, see **[`docs/REPRODUCE.md`](./docs/REPRODUCE.md)**. The pinned runtime environment (Python 3.11.15, PyTorch 2.5.1, Transformers 5.15.1, model revision `7ae5576`) is documented in PREREG §1.

---

## Protocol & Provenance

- **PREREG Methodology (All experiments from M2 onward)**: Metrics, seeds, and decision thresholds are locked prior to execution. If a failure occurs, execution halts, details are recorded in the PREREG §8 addendum, and the experiment is re-run. **Silent overwriting of numbers is prohibited** — pre- and post-modification artifacts are preserved side-by-side.
  - Examples from M4: Attention backend pinned to eager (§8-9), warmup standardized (§8-11), MC4 slack threshold revised (§8-12). The original criterion verdict is preserved in [`MC4_ADJUDICATION.md`](./artifacts/phase6/MC4_ADJUDICATION.md).
  - Example from M1: Detection of PYTHONHASHSEED dependency bug → original ground-truth preserved under `phase6/pre_hashfix/`.
- **Primary Records**: [`artifacts/MILESTONES.md`](./artifacts/MILESTONES.md) (comprehensive milestones and changelog), [`artifacts/phase6/PREREG_*.md`](./artifacts/phase6/) (experiment-specific preregistrations).
- **Tier Attainment**: Unmet status of Tier 2 and pending validation of Tier 4/5 are tracked in [`artifacts/phase6/TIER_ATTAINMENT.md`](./artifacts/phase6/TIER_ATTAINMENT.md).

---

## License

Code and documentation in this repository are released under the **MIT License** ([`LICENSE`](./LICENSE)).

**External dependencies are subject to their respective licenses**:

| Dependency | License |
|---|---|
| [PyTorch](https://github.com/pytorch/pytorch) (`torch`) | BSD-3-Clause |
| [Hugging Face Transformers](https://github.com/huggingface/transformers) (`transformers`) | Apache-2.0 |
| Model weights: Qwen/Qwen2.5-0.5B-Instruct (downloaded automatically during empirical runs) | Apache-2.0 (Copyright 2024 Alibaba Cloud) |

For qualitative positioning alongside related works (such as CacheScout, Tessera, KVFlow, PEEK, and vLLM APC), refer to **[`docs/RELATED_WORK.md`](./docs/RELATED_WORK.md)** (**this document establishes conceptual positioning; empirical benchmark comparisons were not conducted**).
