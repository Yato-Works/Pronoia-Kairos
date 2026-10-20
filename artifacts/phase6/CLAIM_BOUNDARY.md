# §72 Research Claim Boundary — 実測に基づく主張範囲

Plan §72 の「主張しない」リストに対する、本リポジトリの実測対応表。

## 主張しないこと (Plan §72 原文)

```text
We invented KV caching.
We invented prefix caching.
We invented predictive eviction.
We invented queue-aware scheduling.
We invented semantic memory.
```

## 代わりの主張 (Plan §72)

> 既存の強い mechanism を、Semantic Demand を共有信号として
> Runtime-level で協調させることで、End-to-End 性能上の追加利益が
> 存在するかを検証する。

## 実測対応

| 非主張 | 本リポジトリの扱い | 根拠 artifact |
|---|---|---|
| KV caching の発明 | 主張しない。LRU を A_naive の一部として既存手法に使用 | `runtime/adaptive_runtime.py` retention=lru |
| prefix caching の発明 | 主張しない。Prefix 再利用はエンジン価格モデルの一部 | §36 model, `runtime/engine.py` |
| predictive eviction の発明 | 主張しない。B_predictive は Phase 3 既存機構の ladder 上の一点 | `artifacts/phase3/predictive_sweep.json` |
| queue-aware scheduling の発明 | 主張しない。C_queue は PEEK-like 既存追試。F の新規性は協調にある | `artifacts/phase2/peek_comparison.json` |
| semantic memory の発明 | 主張しない。Lethe は外部記憶層として接続 | `artifacts/phase3/lethe_*` |

## 追加利益の所在 (本プロジェクトが実際に示したもの)

1. **協調の利益 (Phase 6)**: 同一コード ladder 上で F (queue-semantic +
   planning + EDF + prefetch) が strong (A-E 最良) を W4/W5/W6/W7/W8 で
   上回る。`artifacts/phase6/full_runtime_gate_v2.json`, gate_passed=True 5/8。
   効果量 (seed12, vs strong): W5 +16.3% / W7 +25.0% (探索的) / W6 +3.6% /
   W4 +2.7% / W8 +0.13% (`strong_reeval.json`)。
2. **joint の利益 (Phase 4)**: context-only / cache-only に対する joint の
   goodput 優位。`artifacts/phase4/joint_context_cache_gate.json`。
3. **IR の利益 (Phase 5)**: overhead を上回る gain (gain 1.94ms > overhead
   0.7ms)。`artifacts/phase5/ir_overhead_gate.json`。
4. **較正の利益 (Phase 3)**: COUPLED/DECOUPLED 両較正系での signal calibration。
   `artifacts/phase3/calibration_sweep.json`, `calibrated_runtime.json`。
5. **意味的対応の利益 (M3)**: F_full の key↔demand 対応写像 π のみを
   シャッフルした対照 (True vs Shuffled, paired 96 ペア) で goodput が低下 —
   確認的 positive 6/8 家族 (W2/W4/W5/W6/W7/W8、Holm(8)<0.05 + CI95 0非含有)。
   効果量 meanΔG (mean G_true 比): W7 +87.0% / W8 +56.9% / W6 +50.2% /
   W4 +42.6% / W2 +13.9% / W5 +1.3%。MC1-MC5 全通過。
   `artifacts/phase6/m3_shuffle_analysis.json` (PREREG ロック済み)。

## 主張しない範囲 (honesty boundary)

- Tier 4/5 の実機測定は主張しない (§93)。抽象時間単位のみ (§43/§92)。
- Tier 2 の +20% は strong比では 1/8 家族のみ (`strong_reeval.json`)。
  F/A 比 (最大 9.7x) は診断値であり主指標ではない。
- W1 (逐次・再利用なし) での優位は主張しない。A が最強のまま。
- W2/W3 の完全タイは政策パリティであり優位ではない (WIN_TOL=1e-9)。
- 単一エンジン価格モデル外 (複数モデル/HW) への一般化は主張しない。
- M3 の主張範囲: シャッフル手法 v1・本抽象モデル・対応関係の層のみ
  (PREREG_M3_SHUFFLE §5)。v1 を結果に基づき再調整していない。
  W1/W3 では対応関係の寄与を確認できておらず、「等価」も主張しない
  (等価検定は事前登録していない)。
  データフロー監査 (2026-10-19、`M3_AUDIT.md`) で、π が信号読出し 4 経路に
  のみ作用し、キャッシュ識別・正解ラベル・ages・queue・真値記録は非変更である
  ことを全 grep で確認。低下は「多集合保存・対応のみ破壊」の設計経路で説明可能。

## RQ 対応

- RQ1 (将来再利用予測で LRU/GDS 超え): Phase 3 + W6/W7 で肯定的。
- RQ2 (Queue-aware + Semantic で追加利益): W6 (C最強→Fが上回る) で肯定的。
- RQ3 (Lethe が Agent/RAG の将来需要を予測): W7/W8 で肯定的。
- RQ4 (精度低でも net gain): Phase 5 gain>overhead + W8 微差全勝で肯定的。
- RQ5 (joint が独立より良い frontier): Phase 4 + Phase 6 で肯定的。
- RQ6 (「適用しない」選択で固定 pipeline 超え): W1/W2/W3 非劣性 + W4-W8
  優位の両立で肯定的 (regressions 0)。
