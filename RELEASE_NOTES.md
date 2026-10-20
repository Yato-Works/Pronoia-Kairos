# Release Notes (draft — V1 Release Gate)

> 状態: **公開前ドラフト**。公開前残タスクは **第三者による独立再現のみ**
> (`docs/RELEASE_CHECKLIST.md` の第4項 — `docs/REPRODUCE.md` を渡して実施)。
> ①ライセンス (MIT) ②関連研究比較 (`docs/RELATED_WORK.md` — 位置づけの
> 整理であり**性能比較の実施ではない**) ③README ↔ 成果物の数値照合
> (不整合 0 件) は完了。独立再現が通ったコミットで V1 タグを打つ。
> 現時点で「第三者による再現確認済み」は主張していない。

## v1.0.0 (draft) — Adaptive Inference Runtime research prototype

### What this is

Semantic Demand を共有信号とするキャッシュ/スケジューリング協調の研究
プロトタイプ。中心仮説、方法、結果、主張境界は `README.md` に統合。

### Verification results (all artifacts under `artifacts/`)

- **M1 — strong baseline re-evaluation (simulator)**: Tier 2
  (+20% goodput vs strongest single-mechanism config) is **NOT attained**
  (1/8 families). The earlier 7/8 claim was withdrawn after a
  PYTHONHASHSEED determinism bug was found and fixed; pre-fix artifacts
  preserved in `artifacts/phase6/pre_hashfix/`.
- **M3 — semantic-signal ablation (simulator)**: shuffling only the
  key↔demand correspondence map degrades goodput in **6/8 families**
  (Holm(8)<0.05, CI95 excluding 0), with manipulation checks MC1–MC5
  passed. `artifacts/phase6/m3_shuffle_analysis.json`.
- **M4 — real-GPU minimal proof (RTX 3050)**: 4-stage judgment
  ①pass (output token columns exactly match with/without cache)
  ②pass (decisions reproduce the reference runtime on all 6 runs)
  ③measured (3 seeds × 40 req × 2 arms) ④**improvement observed under
  applied conditions**: pooled E2E −2.66% (adaptive faster), prefill
  GPU −2.08%, r_overhead 0.0003 (control ms per E2E-wall ms saved).
  `artifacts/phase6/m4_realworld_analysis.json`.

### Integrity / preregistration discipline

- All experiments are preregistered (PREREG files); failures stop the run,
  changes are recorded as §8 addenda, then rerun. Numbers are never
  silently overwritten.
- **MC4 dual-criterion record**: the original threshold (slack=0.15) did
  **not** pass; the final stage-2 judgment is based on the revised
  threshold (slack=0.25, §8-12). Both results and the grounds
  (violation distribution / warmup measurement / engineering tolerance)
  are preserved in `artifacts/phase6/MC4_ADJUDICATION.md` +
  `m4_realworld_mc4_adjudication.json`.

### Claims (see README “主張境界” for the full list)

Confirmed: KV-reuse-driven prefill reduction and the applied-conditions
E2E improvement in this restricted GPU setting; coordination gains and
signal-value correspondence in the simulator. NOT confirmed: queue-control
benefits under high parallelism (no congestion observed in M4), prefetch
benefits on real hardware, cross-seed statistical superiority, Linux /
serving-engine reproduction. Tier 2 remains unattained; Tier 4/5 need
systematic measurement.

### How to verify

```
python -m pytest tests/ -k "not mc0_gpu"    # 279 passed, CPU-only
```
GPU test, full M4 rerun, and MC4 adjudication regeneration:
`docs/REPRODUCE.md`.
