# Phase 6 follow-up 1-5 結果報告 (+5件仕上げ)
基準: seed12_merged.json (n=12) + full_runtime_gate_v2.json + strong_reeval.json。
旧値: followup_12345.json / full_runtime_gate.json (ガード前)、
`pre_hashfix/` (PYTHONHASHSEED依存バグ修正前の正本) に残置。

> **2026-10-19 determinism fix**: `_pf_live` が set のままで prefetch hit
> 承認の同点決着が PYTHONHASHSEED 依存だったバグを事前登録の保守性チェックが
> 検出 → `_pf_live` を挿入順 dict に修正 + 回帰テスト追加
> (`tests/test_reproducibility.py`)。§1/§2/§8 は修正後コードで再計算済み。
> §3/§6/§7 は当時の履歴記録 (閾値選択の経緯) であり数値は旧コードのもの。

## 1. F内訳分解 (E vs E+EDF vs F=EDF+prefetch, paired 6seed)

| family | EDF分 (E_edf-E) | prefetch分 (F-E_edf) | 合計 (F-E) |
|---|---|---|---|
| W8 RAG | 0.0 | **+0.000024** (微小だが正) | +0.000024 |
| W7 agent | 0.0 | **+0.001287** | +0.001287 |
| W4 long高 | 0.0 | **+0.000316** | +0.000316 |
| W5 conc低 | **+0.000760** | +0.000268 | +0.001027 |
| W6 conc高 | -0.000451 | **+0.000888** | +0.000437 |
| W2 chat高 | 0.0 | 0.0 | 0.0 (完全タイ) |

- W4/W7/W8 = prefetch 主導。逐次〜軽並列でEDFの並び替え余地なし、
  prefetchの再計算回避がそのまま上積みに。
- W5 = 混合 (EDF 74% / prefetch 26%)。混合SLO + 持続過負荷で締切順が効く。
- W6 = 混合。EDFがseed3で負け (-0.004) をprefetchが補って上回る。
- W2 = 完全タイ。発行0。
- 旧 (hashバグ版) では W7 prefetch分 +0.00311 / W5 prefetch 0.0 だったが、
  hit承認の決定的化で W7 は半減、W5 に prefetch 上乗せが現れた。

## 2. 統計 seed12化 (符号検定 + bootstrap 95%CI + Holm, n=12)

2026-10-19: `_pf_live` set→dict determinism 修正後、全8家族 seed1-12 を
再測定 (旧chunkは PYTHONHASHSEED 依存バグのランダム1実現だったため
`pre_hashfix/` に保全)。

| family | meanΔ | sign (w/l/t) | p | Holm | CI95 | 判定 |
|---|---|---|---|---|---|---|
| W1 | 0.0 | 0/0/12 | 1.0 | 1.0 | [0, 0] | タイ |
| W2 | 0.0 | 0/0/12 | 1.0 | 1.0 | [0, 0] | タイ |
| W3 | 0.0 | 0/0/12 | 1.0 | 1.0 | [0, 0] | タイ |
| W4 | +0.000178 | 9/3/0 | 0.146 | 0.584 | [+0.000011, +0.000353] | 探索的 (tol感度あり) |
| W5 | +0.001072 | 12/0/0 | 0.0005 | 0.0039 | [+0.000849, +0.001284] | 確認的✅ |
| W6 | +0.000530 | 11/1/0 | 0.0063 | 0.0381 | [+0.000041, +0.000926] | 確認的✅ |
| W7 | +0.001576 | 10/2/0 | 0.039 | 0.193 | [+0.000739, +0.002407] | CIのみ |
| W8 | +0.000022 | 9/0/3 | 0.0039 | 0.0273 | [+0.000012, +0.000031] | 確認的✅ |

- 確認的 (Holm p<0.05): **W5/W6/W8 の3家族**。探索的: W4 (p=0.146)、W7 (CIのみ)。
  旧「W4/W6/W8 も確認的」のうち **W4 はバグ修正で脱落** (11/1/0→9/3/0)。
  hash順依存が W4 の win 数を盛っていた (旧数値はランダム1実現)。
- WIN_TOL感度: W4 1e-9→9/3(p=0.146)、1e-6→9/2/1(p=0.065)、1e-5→8/1/3(p=0.039)
  と閾値に敏感 (微差勝ちの確認的判定は不可)。W8 は全tolで p<=0.008 と安定。
- W5 が全勝 (12/0/0)。W6/W8 は勝ち越しでCI下限が正。W7 はCI正だがHolm未通過。
- seed12_merged.json が正本。旧値は `pre_hashfix/` に保全。

## 3. W1ガード実装 (F 0/6敗→12タイの理由、対策の確定)

- 原因: hit率 A 0.50 → E/F 0.021。predictive/semantic retention が
  再利用なし条件で誤保持し直近追放。追放 A 114 → E 226 → F 258。
  prefetch 191発行全廃棄。makespan +26 (+0.9%) で -0.00015。
- 対策 (実装済み、`runtime/adaptive_runtime.py`):
  - prefetch: seen>=3 & benefit>1.5*tau & waste率ガード (F専用)。
    W1で発行0・Aタイ化を確認。
  - retention fallback: p90<4 & warmup24件超でLRUに (F専用)。
    W1 p90=2のみ発動、W2 p90=6/W4 p90=27/W8 p90=34は非発動。
  - EDF救済: 待機>40 & キュー滞留>1 & 締切前のみ救済 (F専用)。
    W8s6 miss13→3、W2逐次ではFIFO退化。
  - F隔離: ガード類はF_full専用、A-Eは旧条件 (fair ablation §70)。
- 結果: W1 12タイ、W2 12タイ (後退解消)。

## 4. 実トレース (実会話長ライク: 対数正規長 + Zipf話題、中負荷)

- 到着間隔24-48 (平均36)、SLO 90/140混合、話題12 (α=1.2)。
- 結果: F 5/6勝、mean +0.00031、CI [+0.00007, +0.00060] (0除外)、
  p=0.22 (n=6では符号検定は届かず)。
- 初版 (間隔40-80) では全タイだった: 同時実行なしではEDF/prefetchの
  効き目なし。W8帯の中負荷で初めて差が出る = 効果の存在条件の裏付け。
- 過負荷版 (間隔8-24) では全滅 (slo 3/60): 過負荷では何をやっても
  落とす。効果は「中負荷で締切がバラける」帯に限定。

## 5. Tier4最小構成 (vLLM connector seam, mock backend)

- Adaptive 0.55 / LRU 0.0 / LMCache-like 0.0 (hit率)、
  effTTFT 586 vs 1032。adaptive_wins=True (Phase 1 Gateと同一)。
- 実vLLMなしの限界は明示の上で、connector seam 上での3者比較として
  Tier4への最小橋渡しは成立。
- 次のTier4ステップ: 実vLLM + 実trace (ShareGPT等) での同一比較。

## 6. prefetch閾値調整 (3手の3)

- W7s1実測: 1.5→slo21/gp0.00544、2.0→slo22/gp0.00570 (hits23/waste1)、
  2.5→slo19に低下。W7単体では2.0が最適。
- しかし2.0はW4/W6/W8のseed1で後退するため全体最適の1.5を維持。
- 適合率: W8s1で34/37 (92%)、W7s1で27/28 (96%) まで改善
  (旧W8 119/268=44%から大幅改善。lane停止 + seen>=3 + 1.5倍の効果)。
- W7専用チューニング (2.0) は全体後退のため見送り。別途検討。

## 7. EDF相対化の検証 (次3件の1)

- frac=0.5 (queue内最小SLO×0.5) vs 絶対40.0のpaired差分 (seed1-6):
  W5 -0.00078 (6/6負)、W6 -0.00037、W7 -0.00004、W8 0.0。
- 相対化は全家族で後退または無変化のため不採用。絶対40.0を維持する。
- artifact: `artifacts/phase6/edf_frac_check.json` (最終はW7のみ)。

## 8. W7較正連動の検証 + 全体適用評価 (次3件の3)

- benefit推定にPhase3較正の確信度を接続 (`signal_confidence`,
  `signal_ece_proxy=0.10`)。discount = `1 - ece_proxy*(1-confidence)`:
  conf=0.0→0.90、0.5→0.95、1.0→1.00。
  **訂正**: 従前コメントの「confidence=0 → discount=1 (旧動作)」は誤り。
  実際は conf=0.0 で discount=0.90 (既定)。旧 (機能導入前) の無割引=1.0 は
  conf=1.0 に相当。既定 conf=0.0 は「較正未接続時の保守的10%割引」。
- W7 seed1-3実測 (旧コード、履歴): conf=0→slo[18,29,37]、
  conf=0.5/1.0→slo[18,35,37]。seed2で29→35 (+6) に改善。
- **全体適用評価 (seed1-6, `w7_global_eval.json`、hash修正後コードで再生成)**:
  conf0.5/1.0 vs 現行0.0 の goodput差を全8家族で測定。
  - **improved=0 / regressed=4 (W4/W6/W7/W8) で conf0.5 と conf1.0 の両方**。
    (旧バグ版では W7 のみ微改善 +6.8e-06 だったが、決定的コードでは
    W7 も -4.1e-05 後退。旧「W7のみ改善」は hash乱れの1実現だった)。
  - conf=0.0 (0.90割引) が全家族で実測最良。
- **判定: 全体適用は不採用** (W6/W8後退がW7微増を上回る。EDF相対化 (§7) と
  同型: 理論的には効くが他家族を損なう)。デフォルト conf=0.0 を維持。
  ladder隔離済み (A-Eはconfidence=0.0)。W7限定活用は別途検討。

## artifact
基準: seed12_merged.json + full_runtime_gate_v2.json (ガード後)。
旧値: followup_12345.json / full_runtime_gate.json / seed12.json (参考値)。

- `artifacts/phase6/seed12_merged.json` (seed12正本、n=12、全8家族、修正後)
- `artifacts/phase6/full_runtime_gate_v2.json` (gate正本v2、5/8通過、修正後再生成)
- `artifacts/phase6/w7_global_eval.json` (W7較正の全体適用評価、不採用の根拠)
- `artifacts/phase6/strong_reeval.json` (M1: F vs best-of-A-E 相対改善率)
- `artifacts/phase6/PREREG_STRONG_REEVAL.md` (M1事前登録 + 実験固定規則)
- `artifacts/phase6/pre_hashfix/` (hash依存バグ修正前の正本・履歴保全)
- `tests/test_reproducibility.py` (PYTHONHASHSEED非依存の回帰テスト)
- `artifacts/phase6/seed12_<fam>_1-12.json` (seed12分割片、修正後コード再生成)
- 本ファイル (`artifacts/phase6/FOLLOWUP_12345.md`)
- 実装: `compare/followup_12345.py` (Holm/win_sensitivity追加) /
  `compare/seed12_runner.py` / `compare/gate_v2_runner.py` /
  `compare/merge_seed12.py` / `compare/w7_global_eval.py`
- ガード実装: `runtime/adaptive_runtime.py` (P1高速化 + EDF相対化
  `edf_max_wait_frac` 含む)
