# Pronoia-Kairos — Adaptive Inference Runtime

既存の LLM 推論キャッシュ機構 (LRU / prefix caching / queue-aware
scheduling / prefetch) を **Semantic Demand という共有信号**で
Runtime レベルで協調させたとき、End-to-End 性能に追加利益が存在するかを
検証する研究プロトタイプ。

**中心仮説**: 「どのキーが将来再利用されるか」という需要の予測 (semantic
demand) を、eviction・scheduling・prefetch・context 選択が共有する信号に
することで、各機構を個別最適にしただけでは得られない E2E 利益が得られる。

本リポジトリは KV caching / prefix caching / predictive eviction /
queue-aware scheduling / semantic memory の**発明を主張しない**
(`artifacts/phase6/CLAIM_BOUNDARY.md`)。主張するのは協調の利益のみ。

## 結果サマリー

### A. シミュレーション (抽象時間モデル、決定論的)

| 検証 | 設定 | 結果 | 正本 |
|---|---|---|---|
| Phase 6 協調ゲート v2 | 8 家族 × 6 seed、同一 ladder | F (queue-semantic+planning+EDF+prefetch) が最良単独構成を **5/8 家族**で上回る | `artifacts/phase6/full_runtime_gate_v2.json` |
| M1: strong baseline 再評価 | 同上、決定論バグ修正後 | **Tier 2 (+20% vs strong) は 1/8 → 未達**。旧「7/8」主張は撤回。確認的は W5/W6/W8 | `artifacts/phase6/strong_reeval.json` |
| M3: 意味信号の因果切り分け | True vs Shuffled (対応写像 π のみ破壊)、96 ペア | **確認的 positive 6/8 家族** (W2/W4/W5/W6/W7/W8、Holm(8)<0.05 + CI95 0非含有)。W1 は予測どおり無方向 | `artifacts/phase6/m3_shuffle_analysis.json` |

### B. 実測 (M4 — RTX 3050 実 GPU、W6-inspired reduced workload)

Qwen2.5-0.5B-Instruct fp16 / eager / batch=1 順次 / 40 req × 3 seeds ×
2 条件 (LRU baseline vs queue_semantic)。詳細: `docs/REPRODUCE.md`。

| 検証 | 結果 |
|---|---|
| 正しさ (MC0) | キャッシュあり/なしで出力トークン列 **完全一致** |
| 決定の非複製 (MC2/MC3/MC5) | 全6実行が `AdaptiveRuntime` 参照実装と一致、LRU 退化条件一致、再現一致 |
| 測定検査 (MC4) | 変更後基準 (slack=0.25) で全6実行 pass。**元基準 (0.15) では fail** — 両判定を併存保全: `artifacts/phase6/MC4_ADJUDICATION.md` |
| 主指標: E2E 壁時計 | pooled **+2.66% 改善** (adaptive が速い)、seed1 +3.8% / seed2 +4.2% / seed3 −0.03% (意思決定同一) |
| prefill GPU 時間 | pooled +2.08% 改善。seed2 で reuse 7,704→10,899 tok が prefill 減に結実 |
| 制御オーバーヘッド | r_overhead 0.0003 (定義は下記) |

**適用条件つき**: 上記は限定条件 (高並列なし・queue 滞留なし・LRU 比のみ・
3 seed pilot) の観測であり、「どんなワークロードでも 2.66% 高速化」は
主張しない。確認済み/未確認の区別は「主張境界」を参照。

## 指標の定義 (M4)

- **E2E 壁時計 (`e2e_wall_s`)** — 主指標。warmup・モデルロードを除外した
  処理ループの実時間 (最初の req 処理開始 → 最後の req 完了)。GPU 実行 +
  制御の実時間のみを含む (PREREG §8-1)。
- **r_overhead** — `T_control_adaptive[ms] ÷ (E2E_lru[ms] − E2E_adaptive[ms])`。
  単位は無次元 (ms/ms)。分子は adaptive 条件の runtime 制御時間合計
  (`perf_counter_ns` 累積、`control_ms_total`)、分母は **同 seed の paired
  E2E 改善量 (lru − adaptive)**。分母が正の seed のみ定義され (seed3 は
  `null`)、pooled は有効 seed の単純平均。意味: 観測改善 1 ms あたりの
  制御コスト。`< 1` なら制御は観測改善を上回らない — **補助基準**であり、
  最終判断は制御込み E2E 改善 (PREREG §6)。


## 主張境界 (honesty boundary)

**確認済み (このリポジトリの実測が支持する範囲)**

- 限定された GPU 実行条件における KV 再利用による計算削減
  (reused_tokens ↑ → prefill_tokens ↓ → prefill GPU 時間 ↓)、
  および今回観測された適用条件つき E2E 改善 (+2.66% pooled)。
- シミュレーション範囲: 協調構成 F が最良単独構成を 5/8 家族で上回ること、
  意味信号と需要の対応関係が 6/8 家族で確認的寄与を持つこと。

**未確認 (主張しない)**

- 高並列環境でのキュー制御効果 — M4 の queue depth は mean 1.0
  (arrival gap 55–63 秒 ≫ service 数秒) で滞留が発生していない。
- prefetch を含む統合効果、EDF スケジューリング改善の実機検証。
- 3 seed を超える統計的性能優位 (pilot は統計証明に使わない)。
- Linux / 実サービングエンジン (vLLM 等) での再現性。
- Tier 2 (+20% vs strong): 1/8 家族のまま **未達** (`MILESTONES.md`)。
  Tier 4/5 の体系的実機測定も別途検証が必要。
- 最強構成 (E_planning 等) への優位 — M4 は LRU 比のみ (メンター D 判決)。

**既知の制約**

- M4 は W6-inspired **reduced** workload (40 req・ctx ~1.8k tok・batch=1):
  W6 の高並列・高再利用構造そのものの再現ではない。
- 実測は RTX 3050 + 0.5B モデル。GPU クロック揺らぎ帯 (prefill peak-to-peak
  ~29%) の中での測定 — MC4 の閾値はこれに合わせて §8-12 で改定した。
- 将来課題: 追加シード・異なる負荷条件での性能安定性、Linux/実エンジンでの
  再現性検証。

## アーキテクチャ

```
runtime/        AdaptiveRuntime — 判断 (context / residency / victim /
                signals / guard) の唯一の実装。§8-8 の直呼び規則の対象。
semantic/       Semantic Demand 予測・較正 (M3 の π は信号読出4経路のみ)
scheduler/      EDF 等のスケジューリング方針
compare/        ladder (A_naive〜F)・gate・M3 shuffle ランナー/分析
simulator/      抽象時間モデル
m4_realworld/   M4 実 GPU 実行系 (executor / trie store / harness /
                manipulation checks / adjudication)
tests/          279 CPU テスト + 任意 GPU テスト (mc0_gpu)
artifacts/      正本・PREREG・アダンダム・監査記録 (無言上書き禁止)
```

## 再現

```powershell
python -m pytest tests/ -k "not mc0_gpu"   # 279 passed (GPU 不要)
```

GPU テスト・M4 再実行・adjudication 再生成の手順は
**`docs/REPRODUCE.md`** を参照。実行環境 (python 3.11.15 / torch 2.5.1 /
transformers 5.15.1 / model revision `7ae5576`) は PREREG §1 に固定。

## 研究の履歴とプロトコル

- **PREREG 方式 (M2 以降の全実験)**: 評価指標・シード・判定閾値を実行前に
  ロック → 失敗時は停止 → PREREG §8 アダンダムに記録 → 再実行。
  **数値の無言上書きは禁止** — 変更前後は併存保全。
  - M4 の実例: MC0 の backend 固定 (§8-9)、warmup 強化 (§8-11)、
    MC4 slack 改定 (§8-12)。元基準での判定は `MC4_ADJUDICATION.md` に保存。
  - M1 の実例: PYTHONHASHSEED 依存バグ検出 → 旧正本は `phase6/pre_hashfix/`。
- 正本: `artifacts/MILESTONES.md` (全マイルストーンと Changelog)、
  `artifacts/phase6/PREREG_*.md` (各実験の事前登録)。
- Tier 2 未達・Tier 4/5 未検証の状態は `artifacts/phase6/TIER_ATTAINMENT.md`
  に記録。

## ライセンス

未設定 (公開前に選定・追加が必要)。依存: `torch`, `transformers` のみ。
