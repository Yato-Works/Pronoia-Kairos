# artifacts/phase6 — ファイル一覧と扱い
基準コード日時: runtime/adaptive_runtime.py (P1高速化 + ガードF隔離後)。
旧正本との差分は gate v2 の verdict で確認すること。

## 正本 (gate の証拠)

- `full_runtime_gate.json` — Phase 6 §93 ゲートの正本 v1 (10/15, ガード前)。
  8家族 × 6seed paired、verdict (gate_passed=True 5/8)。凍結保存。
- `full_runtime_gate_v2.json` — ガード後コードの正本 v2 (P0-2で生成)。
  8家族 × 6seed paired、verdict 5/8 (W1/W2/W3タイ、W4/W5/W6/W7/W8勝ち)。
  現行コードの判定根拠は v2。以降の更新は v3, v4... と分離する。

## 再現用中間物 (保持)

- `fam_<family>.json` × 8 — v1の家族単位チャンク (ガード前、凍結)。
- `fam_<family>_v2.json` × 8 — v2の家族単位チャンク (ガード後)。
  生成: `python -m compare.gate_v2_runner <family>` → `--merge`。
- `seed12_<family>[_<range>].json` — seed12分割片 (ガード後)。
  生成: `python -m compare.seed12_runner <family> [1-6|7-12]` → `--merge`
  で `seed12_merged.json` に統合。
- `seed12.json` — 旧6seed集計 (参考値として残置、新規はmerged参照)。
- `followup_12345.json` — follow-up 1-5の旧統計 (ガード前、参考値)。

## 派生ドキュメント (本ゲートの解釈)

- `TIER_ATTAINMENT.md` — §94 Tier 0-5 到達宣言 (v1基準、要v2更新)。
- `CLAIM_BOUNDARY.md` — §72 主張範囲と RQ 対応。
- `FOLLOWUP_12345.md` — follow-up 1-5 + 3手仕上げ (seed12ガード後値)。

## 実行ログ (参考・非証拠)

- `gate_stdout.txt` — v1の標準出力 (凍結)。
- v2の表示は `gate_v2_runner --merge` の標準出力で再生成可。

## 推奨整理

```powershell
# 監査に不要なら中間物のみ削除 (正本 v1/v2 + mergedは残す)
Remove-Item artifacts/phase6/fam_*.json
Remove-Item artifacts/phase6/seed12_*.json
# 正本は残す: full_runtime_gate.json, full_runtime_gate_v2.json,
# seed12_merged.json, followup_12345.json
```
