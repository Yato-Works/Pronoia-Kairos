---
name: adaptive-inference-research-runtime
description: Phase-based research runtime for adaptive LLM inference (Pronoia Kairos). Covers Control Plane, Simulator + Belady/LRU comparison, Predictive Cache with regret measurement, and the design rule: brain before GPU.
trigger: "When building adaptive inference runtime, predictive cache with regret vs Belady, or control-plane-first research scaffolds."
---

# Adaptive Inference Research Runtime — Pronoia Kairos

Class-level umbrella for the adaptive LLM inference runtime project.

## Design Rule (Critical)
Build the brain before the GPU. Phase order is fixed: Control Plane → Simulator → Predictive Cache (with Belady regret) → later GPU/CUDA.

## References
- `references/phase-structure.md`
- `references/regret-definition.md`
