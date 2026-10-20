# 再現手順 (Reproduction Guide)

> 本書は独立再現の依頼書 **`docs/REPRODUCTION_REQUEST.md`**
> (**検証対象: コミット `c829714`**) が参照する手順書である。
> 原則として本書に書かれた手順のみを実行し、結果は
> `docs/RELEASE_CHECKLIST.md` 第4項に記録する。

## 環境 (実測時と同一、PREREG_M4_REALWORLD §1)

| 項目 | 値 |
|---|---|
| OS | Windows 11 |
| Python | 3.11.15 |
| torch | 2.5.1+cu121 |
| transformers | 5.15.1 |
| GPU | NVIDIA RTX 3050 (8GB) — **M4 実測のみ必須** |
| モデル | Qwen/Qwen2.5-0.5B-Instruct、revision `7ae5576` (初回実行時に自動 DL) |
| attention backend | `eager` 固定 (PREREG §8-9。SDPA は partial-prefill と数値非結合) |

```powershell
pip install torch transformers pytest
```

## 1. CPU テスト (GPU 不要・デフォルト)

```powershell
python -m pytest tests/ -k "not mc0_gpu"
# 期待: 279 passed, 1 deselected
```

- `tests/test_m4_realworld.py` — M4 の CPU 側検査 (stream 決定論、
  MC3/MC4 のハーネス、r_overhead 計算)。
- MC0 の GPU 本体テスト (`mc0_gpu`) は上記で **除外 (deselected)** —
  実行には GPU とモデルダウンロードが必要なため任意分離。

## 2. GPU テスト (任意、RTX 3050 + モデル DL)

```powershell
python -m pytest tests/test_m4_realworld.py -k mc0_gpu
# 期待: 出力トークン列の完全一致 (MC0 exact)、1 passed
```

## 3. M4 実行 (PREREG ロック済み、~25分)

```powershell
python -m m4_realworld.run_m4 --seeds 1 --conditions lru --smoke   # 簡易 (~5分)
python -m m4_realworld.run_m4                                       # full pilot
```

生成物 (すべて `artifacts/phase6/`):

| ファイル | 内容 |
|---|---|
| `m4_realworld_analysis.json` | 正本: MC1-MC5 + 4段階判定 + pooled (ΔE2E +2.66% 等) |
| `m4_realworld_seed{1,2,3}_{lru,queue_semantic}.json` | 決定列・GPU/制御計測列 (per-request) |
| `m4_realworld_pilot_summary.csv` | 3 seed の paired 要約 |

## 4. MC4 両基準 adjudication (CPU 可、保存 traces から再計算)

```powershell
python -m m4_realworld.mc4_adjudication
# -> artifacts/phase6/m4_realworld_mc4_adjudication.json
#    original criterion (slack=0.15): NOT passed  — seed1 の2条件が fail
#    revised criterion  (slack=0.25): all pass    — M4 stage-2 の判定基準
```

解説: `artifacts/phase6/MC4_ADJUDICATION.md`。

## 5. MC0 backend 比較プローブ (任意、GPU)

```powershell
python -m m4_realworld.probe_mc0_backend
# eager: exact / sdpa: diverge (§8-9 の根拠)
```

## 6. M3 shuffle 再現 (CPU、~6秒/ペア群)

```powershell
python -m compare.m3_shuffle_runner   # MC 失敗で即停止 (PREREG §5)
python -m compare.m3_shuffle_analysis
# -> artifacts/phase6/m3_shuffle_analysis.json
```

## 判定の追跡可能性

- 実行前に閾値を固定 → 失敗時 PREREG §8 アダンダムに記録 → 再実行、が全実験の規則。
- **数値は無言で上書きしない**: 変更前後の判定は
  `MC4_ADJUDICATION.md` (M4) や `phase6/pre_hashfix/` (M1 determinism bug) の
  ように元データと併存させる。

## 第三者による独立再現チェック (本ドキュメントのみで実施可能)

再現者には次の4点を確認してもらう (結果は `docs/RELEASE_CHECKLIST.md`
第4項に追記):

1. クリーンな環境から **CPU テスト 279件**を再現できる (§1)。
2. 指定条件で **GPU テストを実行できる**、または GPU が無い場合に
   制約 (GPU テスト・M4 実行が任意であること) を理解できる (§2)。
3. 実行した各節の**生成成果物**が本ドキュメントの表の説明どおりである (§3-§6)。
4. **README に書かれた数値と成果物の数値が一致している** —
   本リポジトリ側の照合結果は `docs/RELEASE_CHECKLIST.md` 第3項
   (14項目・不整合 0)。再現環境でも独立に確認されればよい。

問題が起きたら、**先に不足箇所を記録してから**本ドキュメントを修正し、
修正したコミットを追跡できるようにする (無言修正はしない)。
