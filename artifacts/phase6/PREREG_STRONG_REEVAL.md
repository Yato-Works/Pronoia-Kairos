# 事前登録 (Pre-registration) — M1 Strong Baseline 再評価

作成: 2026-10-19 (実行前固定)。メンター指示の優先度 1 (Strong Baseline再評価) と
2 (実験条件の固定) に対応。**本ファイルは実験実行前に書き、実行後は改変しない。**
結果は別 artifact (`strong_baseline_reeval.json`) に出す。

## 1. 既知情報の開示 (完全盲検ではない)

- `fam_*_v2.json` (seeds 1-6) の per-config goodput mean は確認済み。つまり
  F/strong の方向感 (最大でも +15% 程度、+20% 未満と予想) は既知。
- 未知: seeds 1-12 の相対改善率の点推定・CI・符号検定・Holm、strong の12-seed確定。
- 本登録は「結果に合わせて指標・閾値を変えること」を防ぐためのものであり、
  盲検を主張するものではない。

## 2. 固定する条件

- seeds: 1..12 (全8家族)。paired: 同一 (family, seed) で全6 config が同一
  リクエストストリーム (§69)。
- 主指標: SLO goodput (§66) の `F - strong`、および相対改善率 `(F-strong)/strong`。
- strong の定義: 家族内で A〜E の goodput mean が最大の config (§62)。
  同一12-seed の標本平均で選ぶ (best-of-5 の選択は strong 側に働く=保守的)。
- WIN_TOL = 1e-9、符号検定 両側、bootstrap 10000回 / seed 0、
  Holm 補正は8家族で一括。相対% のCIも同一bootstrap法。

## 3. 決定規則 (実行前に固定)

- **Tier 2 (§94: 同一budgetで goodput +20%) の再判定**:
  家族毎に `rel% >= +20.0%` を数え、**5/8 以上で「vs strong で Tier 2 達成」、
  未満なら「未達」と TIER_ATTAINMENT.md を正式に更新する**
  (5/8 は §93 の majority 閾値と同一)。
- 診断値: F/A 比は同一 artifact に残すが、**Tier 判定・主張には用いない**
  (メンター指示: 診断用として分離)。
- 既存 artifact (`full_runtime_gate.json` / `full_runtime_gate_v2.json` /
  `seed12_merged.json` / `fam_*.json`) は**一切書き換えない**。新規出力のみ。
- **保守性チェック**: 新規 abs dmean と `seed12_merged.json` の dmean の差が
  1e-6 未満であること (同一決定的計算の検算)。満たさなければ失敗として扱う。

## 4. 3分割報告 (メンター指示)

1. **統計的確かさ**: p (符号検定) + Holm 補正 + bootstrap 95%CI
2. **効果の大きさ**: 相対改善率 % (vs strong)、併に絶対 delta
3. **運用上の価値**: 本実験では未主張 (§92 抽象時間、制御面 wall-clock 非課金)。
   M4 (Tier 4 最小実証) で実機オーバーヘッドと対照する。

## 5. M2 — 今後の実験に適用する固定規則

- 新実験は実行前に本ファイルと同じ形式で metrics / seeds / 判定閾値を記録する。
- Gate v1/v2 および過去数値は履歴として非改変。基準変更時は新規 artifact で行う。
- M3 (Semantic Demand shuffle対照) もこの形式で事前登録してから実行する。

## 6. アダンダム (2026-10-19、実行後記録 — §2/§3の判定規則は不変)

- 初回実行の保守性チェック (§3) が W7 で不一致 (1.07e-4) を検出し、
  `_pf_live` が set のままで prefetch hit 承認の同点決着が PYTHONHASHSEED
  依存だったバグを特定 (W8 seed1 で F-E が 0.0 / 4.02e-05 に揺れた)。
- 修正 (`_pf_live` → 挿入順 dict) + 回帰テスト (`tests/test_reproducibility.py`)
  追加後、seed12 / gate_v2 / w7_global_eval を決定的コードで再生成。
  旧正本は `artifacts/phase6/pre_hashfix/` に履歴保全。
- 再生成後の再実行では保守性チェック max diff = 0.0 で通過し、
  本登録の metrics / seeds / 閾値は一切変更していない。
- 帰結: W4 は 9/3/0 (p=0.146) に変化し確認的から脱落。
  Tier 2 判定 (1/8 → 未達) はこの確定値に基づく。
