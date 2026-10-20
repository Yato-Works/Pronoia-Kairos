# pre_hashfix — PYTHONHASHSEED依存バグ修正 (2026-10-19) **以前**の正本

このディレクトリの数値は `runtime/adaptive_runtime.py` の `_pf_live` が
`set` のままであった時期 (2026-10-19) の生成物で、**プロセス毎の
PYTHONHASHSEED に依存して揺れる「ランダムな1実現」**である。

- 検出経緯: 事前登録 `PREREG_STRONG_REEVAL.md` §3 の保守性チェック
  (seed12_merged と 1e-6 一致) が W7 で不一致を検出 → 追跡で
  `_serve_one` の prefetch hit 承認ループ (`for key in list(self._pf_live)`
  + `break`) の同点決着が set 列挙順 (hash順) に依存していることを特定。
- 影響: W4 (11/1/0→9/3/0) / W6 / W8 (10/0/2→9/0/3) の w/l/t と p値が
  プロセス毎に揺れていた。W1/W2/W3/W5/W7 は無影響 (差 < 1e-6)。
- 修正: `_pf_live` を set → 挿入順 (発行順) 維持 dict。
  回帰テスト: `tests/test_reproducibility.py` (PYTHONHASHSEED 0/6/1/42 で一致)。
- 修正後の正本は phase6 直下 (seed12_merged.json / *_v2.json /
  w7_global_eval.json)。**履歴保存のため本ディレクトリは削除しないこと。**
