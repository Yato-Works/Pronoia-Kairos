# M3 監査 — 介入データフローの追跡記録

- 実施日: 2026-10-19 / 依頼: メンター監査 (次マイルストーン着手前)
- 保全原則: M3 は確定済み実験。**本監査は新規文書の追加のみ**で、
  `PREREG_M3_SHUFFLE.md` (判定規則・§7 既存記録)、正本 artifact
  (`m3_shuffle_analysis.json` 等)、判定結果は一切変更しない。

## 1. 監査質問

1. 置換写像 π は予測器への入力だけに作用しているのか。
2. キャッシュ項目の識別や正解ラベルを直接変更していないか。
3. 観測された eviction / hit率 / prefetch命中率の変化は、設計上予想した
   経路で説明できるか。

## 2. 結論 (先出し)

1. **該当 4 経路すべてが「信号値の読出し」に作用しており、決定の対象キーは
   常に真のキー。** π が影響するのは決定に**使われる数値**のみ。
2. **キャッシュ識別・正解ラベルは一切変更されていない** (§5 負の列挙)。
   懸念の「π によってキャッシュ参照キー自体が食い違う → 項を引けない」
   事象は**本介入には非該当** (実装に存在しない)。
3. **観測変化は設計経路で説明可能** (§6 対応表)。予想外の挙動 (W4/W6 の
   prefetch 命中微増、W1 の primary タイ) も二次指標・workload 構造由来として
   整合。
4. 帰結: M3 が操作したのは **「記録された信号値と、それが指すキー (将来需要)
   の対応関係」** の層。主張は「信号と需要の対応関係が重要である」に正確に
   限定される。π は単置換なので**信号値の多集合 (分布) は保存したまま対応だけ
   を壊している** (MC1/MC2 が実行毎に保証) — 測定された低下は分布変化ではなく
   対応破壊に帰因する。

## 3. 方法 (網羅性)

- 全 grep: `signal_perm|_apply_perm|_perm_key` → `runtime/adaptive_runtime.py`
  のみ (field L112-117 / helpers L257-288 / 適用 L267, L284, L288, L643)。
  `engine.py`・`vllm_connector.py`・`lethe_connector`・IR・planner への流入 0。
- 全 grep: `self._signal(|self._signals(` → 消費は下表の 5 箇所のみ。
- キャッシュ・真値系 (`kv`, `key_tokens`, `_pf`, `_pf_live`, `access_count`,
  `last_access`, `records`) の全 read/write 点を突合し π 混入なしを確認 (§5)。
- 二次指標の数値は `m3_shuffle_merged.json` (96 ペア平均) と
  `m3_shuffle_<fam>_s1.json` (個別) を参照。

## 4. π の適用箇所 — 全 4 経路 (正の列挙)

| # | 読出し | 位置 | 下流の決定 | 決定対象キーの真偽 |
|---|---|---|---|---|
| 1 | `_signals()` 内 `_apply_perm` | L278-288 → 消費 L361/365 | eviction victim 選択 (`QueueSemanticRetentionPolicy.choose_victim`、scheduler/queue_cache_policy.py L100-111) | victim は `lru` (= `list(kv.keys())`) の**実キー**。score の queue 項・age 項は**真値**、semantic 項のみ `signals[k]=raw(π(k))` |
| 2 | `_signals()` 内 `_apply_perm` | 同上 → 消費 L488 | プリフェッチ対象選択 (`_top_prefetch_target`) | 資格判定 (`not in kv` / `in key_tokens`) は**実キー**。スコアのみ permuted。発行 `_pf.key=target` は**実キー** (L647-650) |
| 3 | `_signal()` 内 `_perm_key` | L263-268 → 消費 L395 | コンテキスト選択の `reuse_prob` (`_plan_context`) | chunk の id / benefit / token_cost / quality_loss / `_resident` 判定はすべて**実オブジェクト・実キー** |
| 4 | `_signal()` L637 (benefit) + `access_count.get(_perm_key(target))` L643 (min_seen) | 同上 | プリフェッチ発行ゲート | 発行対象 `target` は**真キー** (経路2 から渡る)。ゲートに使う数値のみ permuted |

補足: `_request_state.prefix_reuse_probability` (L456) は経路3 と同じ読出し層
(PREREG §2 の註記どおり)。`_sig_cache` は perm **前**の正規化真値を保持し、
返却時にのみ `_apply_perm` (L283-288)。`_top_prefetch_target` への呼び出し
(L627-628) は queue_keys を渡していないため queue 資格ゲートは両条件共通で
不効 — 条件間の差分要因ではない。

## 5. 変更していないもの — キャッシュ識別・正解ラベル (負の列挙)

| 対象 | 位置 | π |
|---|---|---|
| 真値記録 `_record_access` (access_count / last_access) | L299-306、呼出 L588 (実キー) | なし |
| `_resident` / `_touch` / `kv` / `key_tokens` | L310-322, L579-585 | なし |
| prefix/chunk の cache hit 判定 (prefix_hit, cached_tokens) | L540-549 | なし |
| prefetch 着弾・hit 承認 (`_pf`/`_pf_live` — 真キー一致のみ) | L528-530, L562-568 | なし |
| ages (`_ages` ← last_access) — victim score の age 項 | L290-297 | なし |
| `_weak_signal` (p90 — かつ多集合は π 不変) | L333-347 | なし |
| `_queue_keys` (pending 由来の真キー集合) | L324-331 | なし |
| スケジューラ `_pick_next` (deadline のみ、信号を使わない) | L496-519 | なし |
| 品質/SLO ラベル (quality_loss, slo_met) | L589-602 | なし |

## 6. 観測 ↔ 設計経路の対応 (数値は 96 ペア平均)

| 観測 | 経路 | 説明 |
|---|---|---|
| evictions 増 (W4 146→278 / W6 171→286 / W7 271→387 / W8 184→309) | 1 | semantic 項が π-image に → 真に重要なキーの保護が外れる → miss 増 → 再読込・再 touch → 容量圧の二次的増加 |
| hit 率低下 (W4 .321→.259 / W7 .414→.247 / W8 .585→.465) | 1 | victim の argmin が真の重要度と独立に → 誤追放 |
| W7/W8 prefetch 命中崩壊 (27.7→11.0 / 32.3→8.1) | 2 | 対象が「真に次の再利用で使うキー」ではなくなる |
| W7/W8 prefetch 発行減 (29.5→22.4 / 34.6→20.8) | 2 → 非 permuted ガード | 命中減 → waste レート上昇 → `_pf_lane_open=False` (L622-625)。ガードは**真値 feedback** のまま →「介入 → 真の反応 → lane 停止」は設計内連鎖 |
| W4/W6 の prefetch 命中微増 (29.8→34.3 / 27.1→27.8) | 2 の二次効果 | 再利用が緩やかな家族では対象が多少ズレても後で消化される。secondary 指標に単方向性要求なし (primary は goodput) |
| W2: True 0 発 / shuffled 1.7 発 | 2+4 → ガード | 発行タイミングの変化で lane 開閉時期が変わるだけ。hit 承認は真キー一致のみ → 誤検知なし |
| SLO 達成低下 (W4 43→25 / W6 62→36 / W7 31→4 / W8 57→28)、品質違反 0/0 は不変 | 3 (+1,2 の上流) | 誤った reuse_prob での context 最適化 → 実効 prefill トークン増 → TTFT 増。品質ラベルが真値なので違反が増えないのも整合 |
| **W1**: evictions 169→141, hit .271→.313, ttft_p95 40.6→39.3 (seed1) は変化するが goodput 完全タイ | 3 のみ有効 (retention は `_weak_signal` LRU fallback で両条件同一、prefetch 発行 0) | slo 48/48・makespan 2770.8618515197986 が両条件一致 → dG=0。arrival-paced 逐次 + SLO slack で **primary が不感**。「介入が効いていない」ではなく「効いても primary に現れない」。※ makespan 一致機構 (last req の cache view 同一) は資料範囲の推論で未再現検証 |
| **W3**: 微差 5/6/1、evictions 413.2→412.7・hit 0.0/0.0 | 3 の効力不足 | 低再利用で signal 値全域が小さく平坦 → π 入れ替えでも reuse_prob 差がほぼ出ない → 意思決定差もほぼなし |

## 7. 主張範囲への帰結

- 操作対象 = 「信号値とキー (将来需要) の対応」。**値の多集合は保存、対応のみ
  破壊** (MC1/MC2)。よって低下は分布変化ではなく対応破壊に帰因する — これが
  M3 の推論の骨格であり、監査後も不変。
- メンター懸念の「キャッシュ ID 食い違い → 項を引けない」型の低下は
  **観測されておらず、実装にも存在しない**。同型の介入 (識別層の破壊) は別個の
  実験であり本監査では実施していないし、M3 の結果変更理由にもならない。
- `CLAIM_BOUNDARY.md` の M3 項は「key↔demand 対応 (信号読出し層)」の範囲で
  正確 (監査リンクを追記済み)。W1/W3 は「等価」を主張しない。
- M3 の結果・判定規則は本監査で変更しない (メンター保全指示)。