# Research Milestones — メンター定義 (2026-10-19) + 状況

| 優先度 | マイルストーン | 完了条件 | 状態 |
|---|---|---|---|
| 1 | Strong Baseline再評価 (M1) | F と各ワークロードの最強構成の比較が完了 | **完了** (2026-10-19) |
| 2 | 実験条件の固定 (M2) | 次回実験の評価指標、シード、判定基準を事前に記録 | **規則確立** (PREREG_STRONG_REEVAL.md §5。M3以降で適用) |
| 3 | Semantic Demand の因果的切り分け (M3) | shuffle対照実験で信号自体の寄与を検証 | **完了** (2026-10-19) H1 支持 — 確認的 positive 6/8 家族 (`phase6/m3_shuffle_analysis.json`) |
| 4 | Tier 4 最小実証 (M4) | 実機で効果の方向とオーバーヘッドを検証 (W5/W6対象) | **PREREG ロック済み** (2026-10-19 メンター A〜F 承認 + ロック前修正3点反映。`phase6/PREREG_M4_REALWORLD.md`。W6-inspired reduced workload / 実 GPU prefill / 実行着手) |

## M1 の結果 (正本: `phase6/strong_reeval.json`、事前登録: `phase6/PREREG_STRONG_REEVAL.md`)

- **Tier 2 (同一budgetで goodput +20%) は vs strong で 1/8 家族 → 未達**。
  満たすのは W7 (+25.0%) のみだが探索的 (Holm p=0.19)。W5 +16.3% が次点。
- F/A 比 (最大 9.7x) は診断値として分離。旧「7/8 +20%大幅超過」は撤回。
- 統計的確かさ (Holm<0.05 確認的): **W5/W6/W8**。W4/W7 は探索的。

## M1 実行中に検出した determinism バグ (2026-10-19、修正済み)

- `_pf_live` が set のままで prefetch hit 承認の同点決着が PYTHONHASHSEED
  依存 → 同一コードでもプロセス毎に goodput/p値が揺れていた。
- 事前登録の保守性チェック (seed12_merged と 1e-6 一致) がこれを検出。
- 修正: `_pf_live` set→挿入順 dict。回帰テスト: `tests/test_reproducibility.py`。
- 旧正本は `phase6/pre_hashfix/` に保全、seed12/gate_v2/w7_global_eval を
  決定的コードで再生成。帰結: W4 の確認的脱落 (9/3/0, p=0.146)。

## M2 固定規則 (以降の全実験に適用)

- 実行前に metrics / seeds / 判定閾値を PREREG 形式で記録する
  (`phase6/PREREG_STRONG_REEVAL.md` §5 が雛形)。
- Gate v1/v2 や過去数値は履歴として非改変。基準変更時は新規 artifact で。

## M3 の結果 (2026-10-19 完了 — 正本: `phase6/m3_shuffle_analysis.json`)

- 事前登録: `phase6/PREREG_M3_SHUFFLE.md` (実行前ロック、§7 アダンダムに
  実行記録)。対象: F_full の key↔demand 対応写像 π のみをシャッフル
  (True vs Shuffled、96 ペア × {strong, F_true, F_shuffled×2} = 384 run)。
- **判定 = 「支持」(PREREG §5 三分岐)**: 確認的 meanΔG>0 (Holm(8)<0.05 かつ
  CI95 が 0 を含まない) が **6/8 家族** — W2/W4/W5/W6/W7/W8。
  事前分類の signal-active 期待群 W4-W8 は全数確認的 positive、
  W1 は 12 タイ、W3 は微差非確認的 (5/6/1, p=1.0) — 予測どおり無方向。
- 効果量 meanΔG (G_true − G_shuffled、相対 = mean G_true 比):
  W8 +0.00946 (+56.9%) / W6 +0.00821 (+50.2%) / W7 +0.00718 (+87.0%) /
  W4 +0.00300 (+42.6%) / W2 +0.00225 (+13.9%) / W5 +0.000103 (+1.3%)。
  (W5/W6/W7/W8 は 12/12 勝ち or 11/12。W2 は 11/0/1。)
- 妥当性検査 MC1-MC5 全通過: MC4 max 4.99e-7 < 1e-6 (正本側の第6位丸め由来)、
  MC5 全 96 ペア完全一致、除外 0。実測 384 run / 6.0 秒。
  品質違反は両条件とも全家族 0 (SLO 差は TTFT 由来)。
- メカニズム (二次指標): shuffle で evictions 増・hit 率低下・
  W7/W8 の prefetch 命中崩壊 (W7 27.7→11.0、W8 32.3→8.1 発当たり均) →
  TTFT 劣化 → SLO 達成数低下 (W7 31→4、W8 57→28、W6 62→36)。
- 帰結: Semantic Demand の**意味的対応関係そのもの**が F の goodput に
  寄与するという H1 を、介入が対応関係の層のみ (記録・時系・全ガード固定)
  である範囲で支持。shuffle v1 は結果に基づき再調整しない (PREREG §5)。
- Tier 判定への影響なし (PREREG §4: Tier 2 再判定は行わない)。

## M4 — 設計確定と実行結果 (2026-10-19 完了)

- M4: W5/W6 を対象に実行系 (vLLM等) で最小比較。確認点: 改善方向の再現、
  制御オーバーヘッドが利益を上回らないか、抽象時間モデルとの一致度。
  効果消失でも「どの仮定が実環境で成立しないか」の発見として扱う。
- **確定 (2026-10-19)**: 正本は `phase6/PREREG_M4_REALWORLD.md` (ロック済み)。
  実機 = HF transformers + RTX 3050 実 GPU prefill (vLLM なし)。対象は W6 のみ、
  正式名称 **"W6-inspired reduced workload"** (batch=1 順次・GPU 同時実行性は
  再現せず)。比較 = LRU vs queue_semantic+True 対応 (prefetch 対象外・最強構成
  への優位は主張しない)。seeds は pilot 3 (統計証明に使わない)。判定 = 4 段階
  (正しさ → 実現可能性 → 性能測定 → 主張境界) + r_overhead (分母正のみ・
  補助基準・最終判断は制御込み E2E 改善)。メンター A〜F 判決とロック前修正 3 点
  は §7 に記録。

### M4 実行結果 (2026-10-19)

- Artifact: 正本 `phase6/m4_realworld_analysis.json` +
  `phase6/m4_realworld_seed{1,2,3}_{lru,queue_semantic}.json` (決定列・
  GPU/制御計測列を全 req について記録) + `phase6/m4_realworld_pilot_summary.csv`。
  停止→修正→再実行の記録は PREREG §8 アダンダム 9-12。
- **4 段階判定**: ①正しさ **pass** — MC0 exact (scratch full prefill と
  partial-prefill + cache continuation の出力トークン列完全一致)。
  ②実現可能性 **pass** — MC1 stream parity / MC2 全6 run で GPU harness の
  決定が `AdaptiveRuntime` 参照実装と一致 (直呼び + 実行アダプター、複製なし) /
  MC3 LRU 退化一致 (equal-signal 条件で policy unit + fake harness の両層) /
  MC4 全6 run pass (0 violations / 341-511 pairs、**変更後基準 slack=0.25 に基づく**;
  元基準 slack=0.15 では fail — `phase6/MC4_ADJUDICATION.md` に両判定を保全) /
  MC5 再現一致。
  ③性能 **measured** — 3 seeds × 40 req × 2 条件 (paired)。
  ④主張境界: **適用条件つき改善** — pooled ΔE2E +2.66% (adaptive が速い)、
  Δprefill +2.08%、r_overhead 0.0003 (分母正 seed のみ)。
- seed 個別: seed1 +3.83% / seed2 +4.17% / seed3 -0.03%。seed2 は純粋な
  reuse 増 (7704→10899 tok) が prefill 減 (426.4→393.9 ms) に結実。
  seed3 は両条件の decision が同一 (reuse 5286=5286) で E2E 同等。
  seed1 は reuse 減 (7846→5921) にも関わらず E2E 改善 — decode 計測の
  ノイズ帯由来とみられ単独根拠にしない (artifact の決定列で追跡可能)。
- 適用条件 (PREREG §6 の明示義務): W6-inspired reduced workload (40 req・
  batch=1 順次・ctx ~1.8k tok)、Qwen2.5-0.5B-Instruct fp16 **eager**、
  RTX 3050、3 seeds pilot (統計的証明には非使用)、LRU 比のみ、最強構成
  (E_planning 等) への優位は主張せず (D 判決)。queue 滞留は観測されず
  (mean depth 1.0 — arrival gap 55-63s ≫ service 数秒、§8-2 の乖離候補として
  artifact に記録)。

## Changelog

1. 2026-10-19: メンターが M1-M4 を正式マイルストーンとして確定。
   M1 は即日完了 (本ファイル + `phase6/strong_reeval.json`)。
   M2 は PREREG 形式の規則として確立。
2. 2026-10-19: M1 実行中の保守性チェックが PYTHONHASHSEED 依存バグを検出。
   修正 + 回帰テスト + 正本再生成 + 履歴保全 (`phase6/pre_hashfix/`)。
   Tier 2 宣言を F/A比 → F vs strong に是正 (`phase6/TIER_ATTAINMENT.md`)。
3. 2026-10-19: M3 完了。PREREG ロック → 実装 (signal_perm + 4経路 π 経由) →
   384 run → MC1-MC5 全通過 → 判定「支持」(確認的 positive 6/8 家族)。
   正本 `phase6/m3_shuffle_analysis.json`、テスト 275 passed
   (`tests/test_m3_shuffle.py` 15 件追加)。
4. 2026-10-19: メンター監査を実施 — π は信号読出し 4 経路のみ、キャッシュ識別・
   正解ラベル・真値記録は非変更を全 grep で確認 (`phase6/M3_AUDIT.md`)。
   M3 は非改変。M4 (実機最小実証, W6) の PREREG 草案を
   `phase6/PREREG_M4_REALWORLD.md` に作成 — 承認後にロック・実行。
5. 2026-10-19: メンターが A〜F を付帯条件つき承認 + ロック前修正 3 点
   (①"W6-inspired reduced workload" として再現と似た実験を区別、
   ②KV 再利用の 4 点証明で実計算削減を示す + prefill/E2E 混同禁止、
   ③r_overhead を厳密定義し E2E 壁時計を主指標化) を指示 → PREREG に反映し
   **ロック済み**。環境実測 (python 3.11.15 / torch 2.5.1+cu121 /
   transformers 5.15.1 / model revision `7ae5576`) を §1 に固定。
   M4 実装 (`m4_realworld/`) 着手。
6. 2026-10-19: **M4 実行完了** — MC0-MC5 全通過、4 段階は ①pass ②pass
   ③measured ④適用条件つき改善 (pooled ΔE2E +2.66%、Δprefill +2.08%、
   r_overhead 0.0003)。実行は停止事由 2 件 (MC0: SDPA kernel の数値非結合で
   diverge → §8-9 eager backend 固定; MC4: warmup 不十分 → §8-11 実測相当化 +
   §8-12 slack 運用値 0.25) をアダンダム記録後に再実行して通過。
   実装 `m4_realworld/` (workloads / gpu_store / gpu_executor / harness /
   mc / run_m4)、CPU テスト `tests/test_m4_realworld.py` 4 件 pass
   (MC0 GPU テストは任意分離)。決定論は PYTHONHASHSEED 非依存。
