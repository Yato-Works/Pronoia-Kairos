# 5件実装計画 (P0→P3)

## P0-1 W2分解: E / E+EDF / F の3点でEDF分とprefetch分を分離
- `compare/full_runtime_experiment.py`: `ladder_f_breakdown()` は既存。
  `run_f_breakdown("W2_short_chat_high_reuse", seeds=1..12)` を回して
  `edf_gain_* / prefetch_gain_*` を取得する。
- 追加実装: `compare/seed12_runner.py` (新規): argv家族指定→seed12実行→
  `seed12_<fam>.json` chunk保存→全chunk merge→`seed12_merged.json`。
  既存の場当たり `_w1check.py` を置換する。
- W5/W6は1家族で30秒超のため seed 1-6 / 7-12 に2分割して実行する。

## P0-2 gate v2再走: ガード後コードで正本分離
- `full_runtime_gate.json` (10/15, ガード前) は凍結保存。
- 新規 `full_runtime_gate_v2.json` + `gate_stdout_v2.txt` を生成。
  チャンク分割 (`fam_<fam>_v2.json`) で重いW5/W6を回避。
- `artifacts/phase6/README.md`: v1=ガード前 / v2=ガード後を明記。

## P0-3 seed12_runner.py + W5/W6 seed12完走
- P0-1のrunnerで W5/W6 を 1-6 / 7-12 分割で完走させる。
- `seed12.json` (旧6seed集計) は参考値として残置、新規は
  `seed12_merged.json` に統合する。

## P1 高速化: _touch O(1) + 履歴deque + signals/agesキャッシュ
- `runtime/adaptive_runtime.py`:
  - `_touch`: 全合計ループを廃止し増分管理
    (`kv_used += tokens - old`)。`pop`時は減算のみ。
  - `_pf_history: List[bool]` → `collections.deque(maxlen=window)`。
    stream末尾の `prefetch_waste += len(self._pf_live)` も履歴に反映。
  - `_signals()/_ages()` の毎追放再構築を、access更新時のみ再計算に。
- 回帰テスト: 合計一致・履歴長上限・末尾waste反映を追加。

## P2 統計: EDF相対化 + 多重比較 + WIN_TOL感度
- `edf_max_wait=40.0` (絶対時間) → `edf_max_wait_frac=0.5` × queue内
  最小SLOの相対値に。旧40.0との差分も記録する。
- `compare/followup_12345.py`: Holm補正p併記
  (`holm_p`)、「探索的/確認的」の2段表示にする。
- WIN_TOL感度: 1e-9 / 1e-6 / 1e-5 で勝敗表がどう変わるかを文書化。

## P3 文書/テスト/整理
- 各md冒頭に「基準artifact+コード日時」を明記、旧数値にラベル。
- 回帰テスト: W2 seed1タイ / W4 seed1勝ちを追加。
  seed12はCI夜間バッチ運用に (seed1-2のみをCIに)。
- 一時ファイル整理: `_w1check.py` 削除→`seed12_runner.py` 常設、
  `seed12.json`/新mergeの二重管理を解消。
