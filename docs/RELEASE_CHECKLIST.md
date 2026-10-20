# V1 Release Gate — チェック記録

メンター定義の V1-1〜V1-4 の実施状況と、各ゲートの検証結果。
記録日: 2026-10-19。タグ打刻・正式リリース承認は本チェックリストの
**第4項 (第三者再現)** 完了が前提 (未実施 — 後述)。

## V1-1 成果物の凍結 — 完了 (判定済み)

- `6188f0a` — M1/M3/M4 正本・PREREG・アダンダム・adjudication 凍結。
- `cd44508` — README / REPRODUCE / RELEASE_NOTES / pyproject。
- working tree clean を確認済み。以降の変更は追加コミットで追跡。

## V1-2 再現手順の整備 — 完了 (判定済み)

- `docs/REPRODUCE.md`: CPU 279件 (`-k "not mc0_gpu"`) と任意 GPU テスト
  の分離、実行方法、環境表、期待出力、生成物一覧。
- 検証: `python -m pytest tests/ -k "not mc0_gpu"` → **279 passed,
  1 deselected** (本リポジトリ環境で再実行確認)。

## V1-3 README と主張境界 — 完了 (判定済み)

- 中心仮説 → シミュレーション表 / 実測表の分離 → 指標定義 (r_overhead
  含む) → 主張境界 (確認済み/未確認) → アーキテクチャ → 再現 → 履歴。

## V1-4 公開前必須作業

### 1. ライセンス — 完了

- 本リポジトリ: **MIT** (`LICENSE`、`pyproject.toml` の
  `license = "MIT"`)。第三者コードの混入なし (Copyright/License ヘッダ
  及び転載箇所の grep 確認、`skills/` は自作文書のみ) を確認のうえ選定。
- 依存物は別ライセンスである旨を README に一覧化:
  torch (BSD-3-Clause) / transformers (Apache-2.0) /
  Qwen2.5-0.5B-Instruct (Apache-2.0、LICENSE ファイル実物で確認)。

### 2. 関連研究比較 — 完了 (位置づけのみ)

- `docs/RELATED_WORK.md`: vLLM APC / SGLang / LMCache / KVFlow / PEEK の
  5件について「解決している問題・共通点・相違点」を表で整理。
- **本比較は性能比較ではない**ことをドキュメント冒頭と README の
  ライセンス欄直後に明記 (メンター指示の明確化事項)。

### 3. README と成果物の数値照合 — 完了 (不整合 0 件)

| # | README の記述 | 成果物 (正本) の値 | 結果 |
|---|---|---|---|
| 1 | E2E pooled +2.66% | `m4_realworld_analysis.json` pooled.delta_e2e_pct = 2.6553 | ✓ |
| 2 | prefill pooled +2.08% | 同上 delta_prefill_pct = 2.0803 | ✓ |
| 3 | r_overhead 0.0003 | 同上 r_overhead = 0.0003 | ✓ |
| 4 | seed1 +3.8 / seed2 +4.2 / seed3 −0.03% | rows: 3.829 / 4.166 / −0.029 | ✓ |
| 5 | seed2 reuse 7,704→10,899 | rows: 7704 → 10899 | ✓ |
| 6 | queue depth mean 1.0 | rows: 全 seed 1.0 | ✓ |
| 7 | MC0 exact, MC1/2/3/5 pass | mc: exact / stream_parity / all_match / pass / match | ✓ |
| 8 | MC4 元基準 fail・変更後 pass | adjudication: original=NOT PASSED (A/B/C 再計算含む), revised=fails=[] | ✓ |
| 9 | Run A 10/428, Run B 12/428 | adjudication.observed_stdout_logs: 10/428, 12/428 | ✓ |
| 10 | Tier 2 = 1/8 未達 | strong_reeval.verdict: hits_count=1 (W7 のみ), met=false | ✓ |
| 11 | Gate v2 = 5/8 通過 | full_runtime_gate_v2.verdict: families_won=5, passed=true | ✓ |
| 12 | M3 = 6/8 確認的 positive | m3_shuffle_analysis: confirmatory_positive=6 家族, verdict=supported | ✓ |
| 13 | CPU テスト 279件 | pytest 再実行: 279 passed, 1 deselected | ✓ |
| 14 | 3実験の書き分け (5/8・1/8・6/8 は別評価) | README 結果サマリーA に「相互に代用できない」注記追加 | ✓ |

### 4. 第三者による独立再現 — **未実施 (公開前残タスク)**

- 手順は `docs/REPRODUCE.md` のみを渡して実施できる状態に整備済み。
- 確認項目 (メンター定義): ①クリーン環境で CPU 279件を再現、
  ②GPU テスト実行 (または GPU なし時の制約理解)、③生成成果物が説明どおり、
  ④README の数値と成果物の数値の一致。
- ①②③④のうち、**④ は本リポジトリ内での照合を実施済み (上表、不整合0)。
  ①〜③ は独立再現者による未実施**。
- 第三者の環境で問題が出た場合は、まず不足箇所を記録し、修正コミットを
  追跡可能にする (無言修正はしない)。

## タグ / リリース承認の条件

- [x] ライセンス決定・メタデータ整備
- [x] 関連研究比較 (位置づけのみ)
- [x] README ↔ 成果物の数値照合 (不整合 0)
- [ ] 第三者による独立再現 (REPRODUCE.md 単独で実施) ← **未了**
- [ ] 上記完了コミットでの最終確認 → **その時点で V1 タグ打刻 →
  リリース承認**

現状: **正式リリース承認は未出**。独立再現未実施のため、現時点での公開
見込みでも「第三者による再現確認済み」とは主張しない。
