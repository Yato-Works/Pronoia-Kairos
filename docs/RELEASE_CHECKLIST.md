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

- `docs/RELATED_WORK.md`: vLLM APC / SGLang / LMCache / KVFlow / PEEK に加え、
  CacheScout (2026) / Tessera (2026/09) を含む計7件について「解決している問題・
  共通点・相違点」を表で整理。
- **本比較は性能比較ではない**ことをドキュメント冒頭と README の
  ライセンス欄直後に明記 (メンター指示の明確化事項)。

### 3. README と成果物の数値照合 — 完了 (不整合 0 件)

| # | README の記述 | 成果物 (正本) の値 | 結果 |
|---|---|---|---|
| 1 | E2E +2.66% (3シード算術平均) | `m4_realworld_analysis.json` pooled.delta_e2e_pct = 2.6553 | ✓ |
| 2 | prefill +2.08% (3シード算術平均) | 同上 delta_prefill_pct = 2.0803 | ✓ |
| 3 | r_overhead 0.0003 (n=2 参考平均, PREREG<0.10) | 同上 r_overhead = 0.0003 | ✓ |
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

> 依頼内容・評価基準・記録様式は **`docs/REPRODUCTION_REQUEST.md`**。
> 検証対象コミット: **`c829714`**。手順書は `docs/REPRODUCE.md` のみ。
> **CPU 側と GPU 側は別々に評価**する (完全一致要求項目は
> REPRODUCTION_REQUEST の表A、環境差考慮項目は表B)。

#### 記録欄 (再現者が記入 — 未実施)

| 記録項目 | 入力 |
|---|---|
| 再現者 (著者本人以外) | *(未記入)* |
| 検証対象コミット | `c829714` を指定 — 実際に実行したコミット: *(未記入)* |
| OS / Python / torch / transformers | *(未記入)* |
| GPU (無しも可) | *(未記入)* |

**CPU 側 (完全一致要求)**

| 項目 | 期待値 | 実行結果 | 判定 |
|---|---|---|---|
| §1 CPU テスト | 279 passed, 1 deselected | *(未記入)* | |
| §4 MC4 adjudication 再生成 | 元基準 NOT PASSED / 変更後 passed が一致 | *(未記入)* | |
| §6 M3 shuffle 再生成 | supported, positive 6/8, MC 通過 | *(未記入)* | |
| README ↔ 成果物 14項目 | 不整合 0 | *(未記入)* | |

**GPU 側 (環境差考慮)**

| 項目 | 要求 | 実行結果 | 判定 |
|---|---|---|---|
| §2 GPU テスト | 同一 GPU 内で exact (GPU 無しは制約理解で可) | *(未記入)* | |
| §3 M4 完走 | 手順・4段階構造・生成物形式が文書どおり (数値一致は要求せず) | *(未記入)* | |
| pooled ΔE2E / Δprefill | 数値差は環境差として記録 | *(未記入)* | |

**問題発生時の記録 (記録保全 → 調査 → 修正、の順。無言修正禁止)**

| 発生日 | 症状・出力全文 | 保全した記録 | 原因 | 修正コミット |
|---|---|---|---|---|
| | | | | |

判定: CPU 側 *(未記入)* / GPU 側 *(未記入)*
最終確認者: *(未記入)* — **記入済みの最終コミットでのみ V1.0.0 タグを打つ**。

## タグ / リリース承認の条件

- [x] ライセンス決定・メタデータ整備
- [x] 関連研究比較 (位置づけのみ)
- [x] README ↔ 成果物の数値照合 (不整合 0)
- [ ] 第三者による独立再現 (REPRODUCE.md 単独で実施) ← **未了**
- [ ] 上記完了コミットでの最終確認 → **その時点で V1 タグ打刻 →
  リリース承認**

現状: **正式リリース承認は未出**。独立再現未実施のため、現時点での公開
見込みでも「第三者による再現確認済み」とは主張しない。
