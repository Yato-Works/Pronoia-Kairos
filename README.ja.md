# Pronoia-Kairos — Adaptive Inference Runtime

[English](./README.md) | **日本語**

> **Status: V1.0.0 Release Candidate** — 公開候補版として再現レビューを募集中。
> **正式 v1.0.0 タグは独立再現の完了を保留** (未実施 — [`docs/RELEASE_CHECKLIST.md`](./docs/RELEASE_CHECKLIST.md) 第4項)。
> 依頼内容と評価基準は **[`docs/REPRODUCTION_REQUEST.md`](./docs/REPRODUCTION_REQUEST.md)** (検証対象コミット: `c829714`)。

既存の LLM 推論キャッシュ機構 (LRU / prefix caching / queue-aware scheduling / prefetch) を **Semantic Demand という共有信号**で Runtime レベルで協調させたとき、End-to-End 性能に追加利益が存在するかを検証する研究プロトタイプ。

**中心仮説**: 「どのキーが将来再利用されるか」という需要の予測 (semantic demand) を、eviction・scheduling・prefetch・context 選択が共有する信号にすることで、各機構を個別最適にしただけでは得られない E2E 利益が得られる。

本研究が検証するのは、Semantic Demandという共有信号によって既存機構を協調させたとき、追加のエンドツーエンド利益が得られるかどうかである。確認された効果の範囲と未検証事項は、以下の結果および主張境界に示す。
なお、本リポジトリは KV caching / prefix caching / predictive eviction / queue-aware scheduling / semantic memory の**発明を主張しない** ([`artifacts/phase6/CLAIM_BOUNDARY.md`](./artifacts/phase6/CLAIM_BOUNDARY.md))。

---

## 結果サマリー

### A. シミュレーション (抽象時間モデル、決定論的)

| 検証 | 設定 | 結果 | 正本 |
|---|---|---|---|
| Phase 6 協調ゲート v2 | 8 家族 × 6 seed、同一 ladder | **定義済みゲート条件を 8 中 5 ワークロードで通過** (W4–W8。W1–W3 は政策パリティのタイで非優位) | [`artifacts/phase6/full_runtime_gate_v2.json`](./artifacts/phase6/full_runtime_gate_v2.json) |
| M1: strong baseline 再評価 | 上記と同一コード、決定論バグ修正後 | **Tier 2 (最強単独構成比 +20%) は 1/8 → 未達**。旧「7/8」主張は撤回。Holm補正後の確認的 positive は W5/W6/W8 の3家族 | [`artifacts/phase6/strong_reeval.json`](./artifacts/phase6/strong_reeval.json) |
| M3: 意味信号の因果切り分け | True vs Shuffled (対応写像 $\pi$ のみ破壊)、96 ペア | **事前登録の判定で 8 家族中 6 家族が確認的 positive** (W2/W4/W5/W6/W7/W8、Holm(8)<0.05 + CI95 0非含有)。W1 は予測どおり無方向 | [`artifacts/phase6/m3_shuffle_analysis.json`](./artifacts/phase6/m3_shuffle_analysis.json) |

> **読み分け**: 上の3行はそれぞれ**別の評価であり、相互に代用できない**。
> 「ゲート条件の通過 (5/8)」は最強構成比の優位性 (≠ 5/8 確認的優位) を意味せず、「Tier 2 1/8」はゲート条件を否定しない。M3 の 6/8 は信号の因果寄与に関する判定であり、性能ゲートとは別軸である。それぞれの事前登録は [`artifacts/phase6/PREREG_*.md`](./artifacts/phase6/) を参照。

### B. 実測 (M4 — RTX 3050 実 GPU、W6-inspired reduced workload)

Qwen2.5-0.5B-Instruct fp16 / eager / batch=1 順次 / 40 req × 3 seeds × 2 条件 (LRU baseline vs queue_semantic)。詳細: [`docs/REPRODUCE.md`](./docs/REPRODUCE.md)。

| 検証 | 結果 |
|---|---|
| 正しさ (MC0) | キャッシュあり/なしで出力トークン列 **完全一致** |
| 決定の非複製 (MC2/MC3/MC5) | 全6実行が `AdaptiveRuntime` 参照実装と一致、LRU 退化条件一致、再現一致 |
| 測定検査 (MC4) | 変更後基準 (slack=0.25) で全6実行 pass。**元基準 (0.15) では fail** — 両判定を併存保全: [`artifacts/phase6/MC4_ADJUDICATION.md`](./artifacts/phase6/MC4_ADJUDICATION.md) |
| 主指標: E2E 壁時計 | **+2.66% 改善** (3シードのシード別相対改善率の算術平均: seed1 +3.83% / seed2 +4.17% / seed3 −0.03%) |
| prefill GPU 時間 | **+2.08% 改善** (3シードのシード別相対改善率の算術平均: seed1 −1.53% / seed2 +7.62% / seed3 +0.15%)。seed2 で reuse 7,704→10,899 tok が prefill 減に結実 |
| 制御オーバーヘッド | **$r_{\mathrm{overhead}} = 0.0003$** (改善量が正の2シードにおける参考平均 $n=2$。詳細は下記) |

**適用条件つき**: 上記は限定条件 (高並列なし・queue 滞留なし・LRU 比のみ・3 seed pilot) の観測であり、「どんなワークロードでも 2.66% 高速化」は主張しない。確認済み/未確認の区別は「主張境界」を参照。

---

## 指標の定義 (M4)

### 相対改善率の算出方法
各シード $s$ における相対改善率 $I_s$ は次式で定義される：

$$I_s = \frac{T_{\mathrm{LRU}, s} - T_{\mathrm{adaptive}, s}}{T_{\mathrm{LRU}, s}} \times 100\%$$

- **E2E 改善率 (+2.66%)**: 3シードのシード別相対改善率の算術平均（$(+3.829\% + 4.166\% - 0.029\%) / 3 = +2.655\% \approx +2.66\%$）。全計測時間を合算した集計比率（pooled ratio）との誤読を防ぐため、算術平均として明示。
- **prefill GPU 時間改善率 (+2.08%)**: 同様に3シードのシード別相対改善率の算術平均（$(-1.528\% + 7.621\% + 0.148\%) / 3 = +2.080\% \approx +2.08\%$）。
- **E2E 壁時計 (`e2e_wall_s`)** — 主指標。warmup・モデルロードを除外した処理ループの実時間 (最初の req 処理開始 → 最後の req 完了)。GPU 実行 + 制御の実時間のみを含む (PREREG §8-1)。

### 制御オーバーヘッド比率 ($r_{\mathrm{overhead}}$)

$$r_{\mathrm{overhead}} = \frac{T_{\mathrm{control, adaptive}}}{T_{\mathrm{LRU}} - T_{\mathrm{adaptive}}}$$

- **定義**: 分子は adaptive 条件の runtime 制御時間合計 (`perf_counter_ns` 累積、`control_ms_total`)、分母は同 seed の paired E2E 改善量 ($T_{\mathrm{LRU}} - T_{\mathrm{adaptive}}$)。単位は無次元 (ms/ms)。
- **$n=2$ の参考値**: 分母が正のシードのみ定義され、seed3 は改善量が非正のため未定義 (`null`)。報告値 $0.0003$ は改善量が正だった2シードの参考平均 ($n=2$, seed1: 0.0004, seed2: 0.0003)。全3シードを代表する指標ではなく、主指標は制御コストを含む E2E 壁時計時間とする。
- **PREREG 判定基準**: 事前登録された合格条件は $r_{\mathrm{overhead}} < 0.10$（制御コストが観測改善量の10%未満、PREREG §6）。本測定値 ($0.0003$) はこの基準を十分に充足している。

---

## 主張境界 (honesty boundary)

### シミュレーションで確認した範囲
- **Gate v2**: 事前定義されたゲート条件を8ワークロード中5つ（W4–W8）で通過した。この結果は、最強の単独構成に対する5/8の確認的優位性を意味しない。
- **Strong baseline再評価（M1）**: 相対改善率の点推定はW4–W8の5家族で正だった。ただし、Holm補正後の確認的positiveはW5・W6・W8の3家族。Tier 2の「最強単独構成比+20%」基準は1/8にとどまり、未達である。
- **Semantic Demand shuffle対照（M3）**: 事前登録されたTrue-vs-Shuffled介入では、シミュレーション上、8家族中6家族で確認的positiveが得られた。この結果は信号と需要キーの対応関係が今回の設定で果たす役割を支持するものであり、実サービング環境での同じ効果を実証するものではない。
- *以上は別々の評価であり、それぞれの結果から主張できる範囲を区別する。*

### 実測で確認した範囲
- 限定された GPU 実行条件における KV 再利用による計算削減 (reused_tokens ↑ → prefill_tokens ↓ → prefill GPU 時間 ↓)、および今回観測された適用条件つき E2E 改善 (+2.66%、3シード算術平均)。

### 未確認 (主張しない)
- **高並列環境でのキュー制御効果**: M4 の queue depth は mean 1.0 (arrival gap 55–63 秒 ≫ service 数秒) で滞留が発生していない。
- **統合機構の実機検証**: prefetch を含む統合効果、EDF スケジューリング改善の実機検証。
- **統計的性能優位の一般化**: 3 seed を超える統計的性能優位 (pilot は統計証明に使わない)。
- **実サービングスタックでの再現性**: Linux / 実サービングエンジン (vLLM 等) での再現性。
- **Tier 2 未達**: 最強単独構成比 +20% は 1/8 家族のまま **未達** ([`artifacts/MILESTONES.md`](./artifacts/MILESTONES.md))。Tier 4/5 の体系的実機測定も別途検証が必要。
- **最強構成への優位性**: M4 は LRU 比のみの検証であり、最強構成 (E_planning 等) への優位性は主張しない (メンター D 判決)。

### 既知の制約
- M4 は W6-inspired **reduced** workload (40 req・ctx ~1.8k tok・batch=1): W6 の高並列・高再利用構造そのものの再現ではない。
- 実測は RTX 3050 + 0.5B モデル。GPU クロック揺らぎ帯 (prefill peak-to-peak ~29%) の中での測定 — MC4 の閾値はこれに合わせて §8-12 で改定した。
- 将来課題: 追加シード・異なる負荷条件での性能安定性、Linux/実エンジンでの再現性検証。

---

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

---

## 再現手順

CPU テストスイートの実行：

```powershell
python -m pytest tests/ -k "not mc0_gpu"
```

> **期待結果**: 279 passed, 1 deselected (GPU 不要、約 5 秒で完了)

GPU テスト・M4 再実行・adjudication 再生成の手順は **[`docs/REPRODUCE.md`](./docs/REPRODUCE.md)** を参照。実行環境 (Python 3.11.15 / PyTorch 2.5.1 / Transformers 5.15.1 / モデルリビジョン `7ae5576`) は PREREG §1 に固定されている。

---

## 研究の履歴とプロトコル

- **PREREG 方式 (M2 以降の全実験)**: 評価指標・シード・判定閾値を実行前にロック → 失敗時は停止 → PREREG §8 アダンダムに記録 → 再実行。**数値の無言上書きは禁止** — 変更前後は併存保全。
  - M4 の実例: MC0 の backend 固定 (§8-9)、warmup 強化 (§8-11)、MC4 slack 改定 (§8-12)。元基準での判定は [`MC4_ADJUDICATION.md`](./artifacts/phase6/MC4_ADJUDICATION.md) に保存。
  - M1 の実例: PYTHONHASHSEED 依存バグ検出 → 旧正本は `phase6/pre_hashfix/` に保全。
- **正本**: [`artifacts/MILESTONES.md`](./artifacts/MILESTONES.md) (全マイルストーンと Changelog)、[`artifacts/phase6/PREREG_*.md`](./artifacts/phase6/) (各実験の事前登録)。
- **達成状況**: Tier 2 未達・Tier 4/5 未検証の状態は [`artifacts/phase6/TIER_ATTAINMENT.md`](./artifacts/phase6/TIER_ATTAINMENT.md) に記録。

---

## ライセンス

本リポジトリのコード・文書は **MIT License** ([`LICENSE`](./LICENSE))。

**依存物は本リポジトリとは別ライセンス** (再配布時にご注意ください):

| 依存物 | ライセンス |
|---|---|
| [PyTorch](https://github.com/pytorch/pytorch) (`torch`) | BSD-3-Clause |
| [Hugging Face Transformers](https://github.com/huggingface/transformers) (`transformers`) | Apache-2.0 |
| モデル重み Qwen/Qwen2.5-0.5B-Instruct (実測時のみ自動取得) | Apache-2.0 (Copyright 2024 Alibaba Cloud) |

関連研究（CacheScout, Tessera, KVFlow, PEEK, vLLM APC 等）との位置づけは **[`docs/RELATED_WORK.md`](./docs/RELATED_WORK.md)** を参照してください (**位置づけの整理文書であり、ベンチマーク比較は実施していません**)。
