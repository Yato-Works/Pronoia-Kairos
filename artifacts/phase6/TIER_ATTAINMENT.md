# Phase 6 — Tier到達宣言 v2 (§94 Success Criteria)
基準: full_runtime_gate_v2.json (seeds 1-6, ガード後) + seed12_merged.json。
旧値 (v1基準): 下に残置。判定日: gate v2 artifact `created` 参照。
EDF相対化 (frac=0.5): W5/W6/W7で後退のため不採用、絶対40.0を維持。
W7較正連動 (signal_confidence): 全体適用はW6/W8後退のため不採用、conf=0.0維持。
2026-10-19: `_pf_live` PYTHONHASHSEED依存バグを検出・修正 (set→挿入順dict、
`tests/test_reproducibility.py` 追加、旧正本は `pre_hashfix/` に保全)。
決定的コードで seed12/gate を再生成の上、M1 Strong Baseline再評価を実施
(`PREREG_STRONG_REEVAL.md` 事前登録 → `strong_reeval.json`)。
Tier 2 は vs strong で再判定し**未達**に更新 (下記)。旧「7/8 +20%」は撤回。

## verdict v2

- gate: strong_baseline_superiority_across_workload_families
- gate_passed: True (5/8, required 5; regressions 0; F quality breaches 0)
- wins: W4, W5, W6, W7, W8（strong = 家族毎 A-E 最良、F はそれを上回る）
- seed12 (n=12): W1/W2/W3タイ、**W5/W6/W8 確認的✅** (Holm p<0.05)、
  W4 (9/3/0, p=0.146) / W7 (CIのみ) は探索的

## Tier 0 — Scientific Validity: 達成 (v2で強化)

- Traceable: 全 ladder config が同一コード (`runtime/adaptive_runtime.py`) の
  スイッチ違い (§70)。ガード類はF専用に隔離、A-Eは旧条件 (fair ablation)。
- Reproducible: `compare/full_runtime_experiment.py` + `SEEDS=(1..6)`、
  家族・seed 毎に同一 stream を全 config が共有 (paired, §69)。
  2026-10-19: PYTHONHASHSEED 非依存を回帰テストで保証
  (`tests/test_reproducibility.py`)。artifact: 
  `artifacts/phase6/full_runtime_gate_v2.json`
  (+ `fam_*_v2.json` 分割片) / `seed12_merged.json` (n=12)。
- Strong baselines: 家族毎に A-E の最良を strong とする best-configuration
  rule (§62)。F は naive (A) ではなく最強の既存構成と比較される。
- Statistically sound: paired mean delta / per-seed delta / seeds-won に加え、
  seed12で符号検定 + bootstrap CI + Holm併記。WIN_TOL感度も開示。

## Tier 1 — E2E latency -10% (同等品質、一部 realistic workload): 条件付き達成

- 同一 SLO 達成率での makespan 短縮は W6 (4515→3862, -14.5%) で確認。
  W4/W7/W8 は同一 SLO カウントを僅かに短い makespan で捌く (W8 は全6seed
  で同一 SLO カウント・短 makespan)。
- ただし latency は §36 抽象時間単位のモデル価格であり、Python wall-clock
  ではない (§43/§92)。Tier 4/5 の実機測定は未主張。
- よって「モデル時間上の Tier 1 到達、実機は未検証」と宣言する。

## Tier 2 — 同一 memory budget で SLO goodput +20%: **未達 (vs strong、2026-10-19再評価)**

メンター指示 (Strong Baseline再評価) に従い、主指標を **F vs 家族毎の
最強ベースライン (A〜E の best, §62)** に統一。事前登録:
`PREREG_STRONG_REEVAL.md`、正本: `strong_reeval.json` (seeds 1-12)。

| family | strong | 相対改善率 | rel 95%CI | sign p / Holm | +20% |
|---|---|---|---|---|---|
| W1 | A_naive | 0.0% (タイ) | — | 1.0 / 1.0 | ✗ |
| W2 | B_predictive | 0.0% (タイ) | — | 1.0 / 1.0 | ✗ |
| W3 | E_planning | 0.0% (タイ) | — | 1.0 / 1.0 | ✗ |
| W4 | E_planning | +2.7% | [+0.2%, +5.3%] | 0.146 / 0.584 | ✗ |
| W5 | E_planning | **+16.3%** | [+12.5%, +20.0%] | 0.0005 / 0.0039 | ✗ |
| W6 | E_planning | +3.6% | [+0.4%, +6.2%] | 0.0063 / 0.038 | ✗ |
| W7 | E_planning | **+25.0%** | [+9.7%, +41.3%] | 0.039 / 0.193 (探索的) | ✓ |
| W8 | E_planning | +0.13% | [+0.1%, +0.2%] | 0.0039 / 0.027 | ✗ |

- +20% を満たすのは **W7 の 1/8** (達成閾値は §93 majority の 5/8) → **未達**。
- W5 (+16.3%) が閾値に最も近く、W7 は探索的 (Holm未通過)。
- 診断値 (Tier判定には使わない): F/A 比 = W1 1.00x / W2 1.04x / W3 3.92x /
  W4 1.27x / W5 1.34x / W6 7.24x / W7 9.70x / W8 1.47x。
  Naive比では大きく見えるが、**最強ベースライン比では小さく、
  旧「7/8家族で+20%大幅超過」の宣言は撤回**。W1は逐次・再利用なしの
  条件 (policy parity)、W2/W3は完全タイ。

## Tier 3 — 複数 workload で Pareto frontier improvement: 達成

- W4/W5/W6/W7/W8 で F が strong (E) を上回り、quality breaches 0 (§68 予算遵守)。
- Phase 4 (`joint_context_cache_gate.json` joint 0.824 > 単独 0.77) と合わせ、
  joint 最適化が独立最適化より優れた frontier を形成 (RQ5)。
- W1/W2/W3 では後退なし (regressions 0)。
- seed12 (n=12): W5/W6/W8 が確認的✅ (Holm p<0.05)、W4/W7が探索的。

## Tier 4 — vLLM/SGLang/PEEK/LMCache 等の得意条件を含む混合での p50/p95/p99 改善: 未検証

- 本ゲートは §36 モデル時間上の比較であり、実 vLLM/SGLang 上での測定は
  行っていない (§93 honesty boundary)。
- PEEK との比較は Phase 2 (`peek_comparison.json`) のシミュレータ範囲に留まる。
- connector seam 上の3者比較 (Adaptive 0.55 vs LRU/LMCache 0.0) は
  最小橋渡しとして成立も、実機ではない。
- Tier 4 は将来の実機統合 (Stage B-E, §84) の対象。

## Tier 5 — 複数 model / 長さ / 並列度 / HW での再現: 未検証

- 本ゲートは単一エンジン価格モデル・抽象 HW。複数モデル・HW での再現は
  行っていない。
- Workload 側の多様性 (W1-W8: chat/long/concurrent/agent/RAG) は確保済みだが、
  Tier 5 の主張には実機マトリクスが必要。

## まとめ (v2)

- 達成: Tier 0, Tier 3 (+ Tier 1 はモデル時間上の条件付き達成)
- **未達: Tier 2** (上記 Tier 2 章の通り vs strong 1/8)。旧まとめの
  「Tier 2 達成」表記は本行で是正 (2026-10-19)。
- M3 (semantic shuffle 対照、2026-10-19 完了、H1 支持) は Tier 判定と
  独立 (PREREG_M3_SHUFFLE §4: Tier 再判定は行わない) — 上表の数値は不変。
  正本: `m3_shuffle_analysis.json`。
- 未検証: Tier 4, Tier 5 (計画通り §93 の範囲外として明示)
- v1からの変更: W1がタイ化 (F隔離+ガード)、seed12でW4/W5/W6/W8確認的✅、
  W7探索的。2026-10-19 現行コード再生成で確認的が4家族に拡大。

---

## 旧値 (v1基準、参考値として残置)

## verdict

- gate: strong_baseline_superiority_across_workload_families
- gate_passed: True (5/8, required 5; regressions 0; F quality breaches 0)
- wins: W4, W5, W6, W7, W8（strong = 家族毎 A-E 最良、F はそれを上回る）

## Tier 0 — Scientific Validity: 達成

- Traceable: 全 ladder config が同一コード (`runtime/adaptive_runtime.py`) の
  スイッチ違い (§70)。`A/B/C/D/E/F` の差は retention/scheduler/planning/
  prefetch の ON/OFF のみ。
- Reproducible: `compare/full_runtime_experiment.py` + `SEEDS=(1..6)`、
  家族・seed 毎に同一 stream を全 config が共有 (paired, §69)。
  artifact: `artifacts/phase6/full_runtime_gate.json` (+ `fam_*.json` 分割片)。
- Strong baselines: 家族毎に A-E の最良を strong とする best-configuration
  rule (§62)。F は naive (A) ではなく最強の既存構成と比較される。
- Statistically sound: paired mean delta / per-seed delta / seeds-won を記録。
  WIN_TOL=1e-9 で完全タイ (W2/W3) を勝ちに数えない。

## Tier 1 — E2E latency -10% (同等品質、一部 realistic workload): 条件付き達成

- 同一 SLO 達成率での makespan 短縮は W6 (4515→3862, -14.5%) で確認。
  W4/W7/W8 は同一 SLO カウントを僅かに短い makespan で捌く (W8 は全6seed
  で同一 SLO カウント・短 makespan)。
- ただし latency は §36 抽象時間単位のモデル価格であり、Python wall-clock
  ではない (§43/§92)。Tier 4/5 の実機測定は未主張。
- よって「モデル時間上の Tier 1 到達、実機は未検証」と宣言する。

## Tier 2 — 同一 memory budget で SLO goodput +20%: 達成 (F vs A)

同一 budget (家族毎 kv_capacity/context_budget は全 config 共通) での F/A 比:

| family | F/A |
|---|---|
| W3 long低再利用 | 3.63x |
| W6 concurrent高 | 8.08x |
| W7 agent | 11.39x |
| W8 RAG | 1.58x |
| W4/W5 | 1.31x |
| W2 | 1.06x |
| W1 | 0.99x (非劣性範囲内) |

7/8 家族で +20% を大幅超過。W1 は逐次・再利用なしで政策差が出ない条件
(想定内、tolerance 0.02 の 1/100 以下)。

## Tier 3 — 複数 workload で Pareto frontier improvement: 達成

- W4/W5/W6/W7/W8 で F が strong (E) を上回り、quality breaches 0 (§68 予算遵守)。
- Phase 4 (`joint_context_cache_gate.json` joint 0.824 > 単独 0.77) と合わせ、
  joint 最適化が独立最適化より優れた frontier を形成 (RQ5)。
- W1/W2/W3 では後退なし (regressions 0)。

## Tier 4 — vLLM/SGLang/PEEK/LMCache 等の得意条件を含む混合での p50/p95/p99 改善: 未検証

- 本ゲートは §36 モデル時間上の比較であり、実 vLLM/SGLang 上での測定は
  行っていない (§93 honesty boundary)。
- PEEK との比較は Phase 2 (`peek_comparison.json`) のシミュレータ範囲に留まる。
- Tier 4 は将来の実機統合 (Stage B-E, §84) の対象。

## Tier 5 — 複数 model / 長さ / 並列度 / HW での再現: 未検証

- 本ゲートは単一エンジン価格モデル・抽象 HW。複数モデル・HW での再現は
  行っていない。
- Workload 側の多様性 (W1-W8: chat/long/concurrent/agent/RAG) は確保済みだが、
  Tier 5 の主張には実機マトリクスが必要。

## まとめ

- 達成: Tier 0, Tier 2, Tier 3 (+ Tier 1 はモデル時間上の条件付き達成)
- 未検証: Tier 4, Tier 5 (計画通り §93 の範囲外として明示)
