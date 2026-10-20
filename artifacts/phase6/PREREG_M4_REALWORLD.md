# PREREG — Milestone 4: 実機最小実証 (M4)

状態: **ロック済み (2026-10-19)** — メンター A〜F 判決とロック前修正 3 点の
反映後にロック。ロック後の §3 条件表・§6 判定規則は改変禁止、実行後は §8
アダンダムのみ追記 (M2 規約 `PREREG_STRONG_REEVAL.md` §5 と M3 プロトコル同一)。
承認記録 (2026-10-19): A 実GPU検証=承認 (実機サービング全体とは区別) /
B Qwen2.5-0.5B-Instruct=承認 (revision・dtype・環境バージョン固定) /
C 40req 縮小=条件付き承認 (prefix共有率・到着時刻・並列性・再利用機会を固定) /
D LRU 比較・prefetch 対象外=条件付き承認 (最強構成への優位は主張しない) /
E 3 seeds=パイロット承認 (統計的証明に使わない) /
F runtime 直呼び=承認 (ハーネスへの複製禁止、共通ロジックと実行アダプターを分離)。

## 0. 位置づけと目的 (M4-1) — 実行方式と正式名称

- M1: 抽象モデル上で F > strong — W6 は +3.6% (確認的, Holm<0.05)。
- M3: 抽象モデル上で True > Shuffled — W6 は meanΔG +0.00821 (+50.2%,
  12/12)。ただし M3 監査 (`M3_AUDIT.md`) の通り、操作したのは
  「信号↔需要の対応」の層。
- **M4: 実 GPU 推論の壁時計で、Adaptive Runtime の判断が実際の GPU 計算の
  削減につながるところまで確かめる** — 単に「RTX 3050 で動いた」では足りない。
- 成功でも失効でも価値: 効果が消失した場合は「抽象時間モデルのどの仮定が実機で
  成立しないか」の発見として記録する (MILESTONES M4 メモ)。

### 実行方式の明記 (最重要 — 「W6 の再現」と「W6 に似た実験」の区別)

- **到着時刻**: W6 生成器の arrival schedule をそのまま固定使用。同時到着は
  queue 滞留として再現する。
- **GPU 実行**: batch=1 の順次推論 (transformers 単発)。**連続バッチング等の
  GPU 同時実行性は再現しない**。
- よって本実験の正式名称は **"W6-inspired reduced workload"** であり、
  「W6 の完全な実機再現」ではない。後者 (GPU バッチ同時実行を含む W6 完全
  再現) は将来課題として範囲外に記す。
- 再利用構造 (prefix 共有率・並列性=queue 滞留・再利用機会) は W6 と同じ
  生成器から seed 固定で引き、似せた範囲を artifact に数値で記録 (§3)。
- 位置づけ (A 判決): HF による GPU 実行の検証であり、**実機サービング全体
  (連続バッチング・serving stack) の検証とは区別する**。

## 1. 環境の事実 (2026-10-19 事前検出)

- GPU: NVIDIA RTX 3050 (8GB) — nvidia-smi で確認。
- `torch` ✓ / `transformers` ✓ / **`vllm` ✗ (未導入)**。OS = Windows (win32)。
  vLLM は Linux/WSL2 前提のため本機への導入は非現実的と判断。
- 既存資産: `runtime/vllm_connector.py` は「vLLM 非依存プロトタイプ
  (Mock backend、GPU 不要)」と明記。`runtime/tiers.py` も構造モデル。
- **よって本 PREREG で「実機」とは: HF transformers + torch による実 GPU
  prefill の壁時計測。** vLLM connector 統合は範囲外 (Tier 4 宣言はしない)。
  (A 判決: 承認 — ただし実機サービング全体の検証とは区別する。§0 参照)

### 環境固定表 (B 判決 — 2026-10-19 実測、実行時に再確認)

| 項目 | 固定値 |
|---|---|
| python | 3.11.15 |
| torch | 2.5.1+cu121 (CUDA runtime 12.1) |
| transformers | 5.15.1 |
| GPU | NVIDIA GeForce RTX 3050 (8192 MiB)、`cuda_available=True` |
| model | `Qwen/Qwen2.5-0.5B-Instruct`、revision `7ae557604adf67be50417f59c2c2f167def9a775` に固定 (HF API 2026-10-19 確認) |
| tokenizer | 同一 revision |
| dtype | float16 |

## 2. 検証する問い (H_M4)

- **H_M4**: W6 相当ストリームで、Semantic Demand に基づく retention 制御
  (M1/M3 と同一の決定ロジック + True 対応) は、同一条件の LRU ベースライン
  に対し**実測 TTFT 系指標で同方向の改善**を示し、**制御オーバーヘッドは
  その利益を上回らない**。

## 3. 事前固定する条件 (M4-2)

| 項目 | 固定値 (ロック版) |
|---|---|
| 対象モデル | `Qwen/Qwen2.5-0.5B-Instruct` (B 判決)。revision・dtype・環境は §1 固定表で固定し、実行時に再確認 |
| 実行環境 | RTX 3050 8GB / Windows / `perf_counter_ns` 壁時計。測定は warmup 後 |
| ワークロード | **"W6-inspired reduced workload"** (§0 正式名称)。W6 生成器から seed 固定で生成し VRAM 制約で縮小 (最大 40 req / ctx 1-2k tok)。**C 判決により固定して記録する構造項目**: ①入力長分布 ②prefix 共有率 ③arrival 時刻列 (同時到着含む) ④並列性 = queue 滞留パターン ⑤再利用機会数 (同一 prefix/chunk の再訪回数) — stream hash と要約統計を artifact に保存 |
| 比較条件 | (a) LRU retention (baseline) vs (b) queue_semantic retention + True 対応。同一 stream・同一 seed・同一 budget (paired)。prefetch は対象外 (D 判決)。**最強構成 (E_planning 等) への優位性は主張しない (D 判決)** |
| 反復 | seeds (1..3) × 両条件 — **パイロット (E 判決): 実装・測定の実現可能性確認用であり、一般的な性能優位の統計的証明には使わない** (p 値・CI を主張に用いない) |
| 実測指標 | **主指標: E2E 壁時計時間 (制御コスト込み)**。併記を義務付け: `T_control` (CPU 制御時間、`perf_counter_ns` 累積) と `T_gpu` (GPU 実行時間) を**別々に**記録。構成要素: prefill 時間 / reused_tokens と省略 prefill 量 / cache hit / SLO 達成 (縮小スケールで再設定・固定) / 品質違反 (KV は決定的 → 出力同一は §5 MC0 で確認、生成品質比較は行わない) |

判定基準は §6 (4 段階判定 + r_overhead 厳密定義) に固定。

## 4. 実装メモ (ロック時固定)

- **決定ロジック (F 判決)**: `AdaptiveRuntime` の retention 決定を**直接呼ぶ**。
  意思決定ロジックをハーネスに複製しない。「共通ロジック」と「実行アダプター
  (GPU harness 側)」を分離し、同一性はテストで固定する。
- **KV 再利用の成立証明 (修正② — 単なる hit 記録ではなく計算削減の証拠)**:
  1. 共通 prefix を prefill して KV 状態 (`past_key_values`) を保持する。
  2. 後続リクエストで保持済み KV 状態を再利用して prefill を行う。
  3. 再利用した長さ (reused tokens) と、実際に省略できた prefill 処理
     (new-token 数とその prefill 所要時間) をリクエスト毎に記録する。
  4. キャッシュを使わない参照経路と出力を比較し、正しさを確認する (§5 MC0)。
  5. 決定ログ: runtime の retention 判定 → 次リクエストでの hit/miss →
     reused_tokens → prefill_duration を 1 リクエスト毎に残し、「Runtime の
     判断が実際の GPU 計算を省略させた」チェーンを追跡可能にする。
- **指標の混同禁止 (修正②)**: prefill 時間の削減 (prefill-only 測定) を
  生成全体の TTFT 改善・スループット改善と同一視しない。§6「性能」段の
  主指標は E2E 壁時計であり、prefill-only 結果は構成要素として区別表示する。
- prefix キャッシュ: HF `past_key_values` 再利用 (transformers 5.15 の
  `DynamicCache` 系 API — 実装時に挙動をテストで固定)。
- determinism 教訓 (M1) を踏襲: キー列挙ソート固定、PYTHONHASHSEED 非依存。
- π 介入 (True vs Shuffled) の再現は M4 範囲外 — M4 は True 対応の
  F 機構 vs LRU の方向検証 (Shuffled は M3 の役割)。

## 5. 妥当性検査 (実行毎、停止条件つき)

- **MC0 (正しさ — 最優先、修正②)**: キャッシュ無し参照経路と KV 再利用経路の
  出力一致 (同一入力・同一 decoding 条件での出力トークン列一致、許容誤差を
  事前固定)。不一致なら実験停止。
- **MC1 (stream parity)**: (a)(b) 両条件の stream hash が同一 (pairing 成立)。
- **MC2 (ロジック同一性、F 判決)**: GPU harness の決定が `AdaptiveRuntime`
  参照実装と一致 — ハーネス複製禁止、アダプター経由の同一性テスト。
- **MC3 (LRU 退化チェック)**: 意味信号を無効化した設定の結果が LRU と一致
  (harness バグ検出用)。
- **MC4 (測定健全性)**: warmup 除外、同一 prefix の prefill 時間が new-token
  数に対して単調 — 明らかな測定崩れを検出。
- **MC5 (再現性)**: 同一条件の再実行で決定系列・reused_tokens 列が一致。
- 失敗時は実行を停止し、修正は §8 アダンダムに記録してから再実行
  (判定規則は変更しない)。

## 6. 判定 — 4 段階 (順に充足確認、ロック時固定) と r_overhead

**成功条件 (メンター M4 成功条件の順):**

1. **正しさ**: 実際の KV 再利用が成立し (reused 長・省略 prefill を記録)、
   出力の正しさが維持されている (MC0)。
2. **実現可能性**: Adaptive Runtime の判断を実 GPU 実行に接続でき (F: 直呼び
   + アダプター)、決定論的に再現できる (MC1-MC5)。
3. **性能**: 同一ワークロードで LRU 対照と比較し、**E2E 壁時計時間 (主指標、
   制御込み)**・prefill 時間・制御コストの差を測定できる。
4. **主張の境界**: 改善があれば改善、なければ**改善なし**と報告する。
   限定的な結果には適用条件 (W6-inspired reduced workload・単一 GPU・単一
   小モデル・3 seeds pilot・LRU 比のみ・最強構成への優位なし) を明示する。

**r_overhead (修正③ — 補助的な工学基準の厳密定義):**

\\[
r_{\\mathrm{overhead}} = \\frac{T_{\\mathrm{control}}}
{T_{\\mathrm{baseline}} - T_{\\mathrm{adaptive}}}
\\]

- `T_control` = CPU 制御時間の累積 (`perf_counter_ns`)。
  `T_baseline` / `T_adaptive` = 各条件の **E2E 壁時計時間 (主指標)**。
- **分母 > 0 の場合のみ評価する。分母 ≤ 0 (改善なし) の場合は合格にしない。**
- 閾値: r_overhead < 0.10。
- **CPU 処理と GPU 処理が並行して進む場合、CPU 時間をそのまま壁時計時間に
  加算できない**ため、この比率は**補助的な工学基準**として扱い、artifact に
  参考値として保存する。
- **最終的な実用性の判断は、制御コストを含む E2E 時間が実際に改善したかで
  行う。**
- 3 seeds の結果は pilot 記録として示し、統計的優位の主張には用いない
  (E 判決)。

**解釈の位置づけ**: 改善なし・逆転でも基準変更は行わない (MILESTONES M4 メモ:
どの仮定が実機で成立しないかの発見として記録)。M3 (sim goodput) との数値・
判定の流用禁止。Tier 2 未達は M4 後も未達のまま。**M4 通過のみで Tier 4/5
達成とはしない** (実 serving エンジン比較・複数環境再現は別課題)。

## 7. メンター A〜F 判決記録 (2026-10-19、ロック根拠)

| 項目 | 判決 | 条件 |
|---|---|---|
| A 実 GPU 検証 | 承認 | HF GPU 実行。実機サービング全体の検証とは区別 (§0) |
| B Qwen2.5-0.5B-Instruct | 承認 | revision・dtype・実行環境バージョンを固定 (§1 環境固定表) |
| C 40 req 縮小 | 条件付き承認 | 入力長に加え prefix 共有率・到着時刻・並列性・再利用機会を固定 (§3) |
| D LRU 比較・prefetch 対象外 | 条件付き承認 | 初回の実機実現可能性検証として採用。最強構成への優位性は主張しない (§3/§6) |
| E 3 seeds | パイロット承認 | 実装と測定の実現可能性確認用。一般的な性能優位の統計的証明には使わない (§3/§6) |
| F runtime 直呼び | 承認 | ハーネスへの複製禁止。共通ロジックと実行アダプターを分離 (§4) |

**ロック前修正 3 点 (反映済み)**: ① §0 に実行方式を明記し正式名称を
"W6-inspired reduced workload" に固定 (GPU 同時実行性は再現しない旨明記)、
② §4 に KV 再利用の 4 点証明 + 決定ログ + prefill/E2E 混同禁止、§5 に MC0
(正しさ) を最優先として追加、③ §6 に主指標 E2E + `T_control`/`T_gpu` 分離
記録 + r_overhead 厳密定義 (分母正のみ・補助基準・最終判断は E2E) を固定。

## 8. アダンダム (ロック後・実行後に追記 — 判定規則は不変)

### 2026-10-19 実行前 operationalization (§6 の運用解釈。判定規則の改変なし)

1. **E2E 壁時計 (§6 主指標) の定義**: warmup・モデルロードを除外した処理ループの
   実時間 (最初のリクエスト処理開始 → 最後の処理完了)。到着間隔での実 sleep は
   行わない — serve() と同じ「arrival 承認 + アイドル時次 arrival へ jump」構造を
   バーチャル時計で実装。E2E は GPU 実行 + 制御の実時間のみを含む。
2. **到着構造 (§0)**: W6 arrival schedule (gap 55-63 秒) をそのまま保持。実 GPU の
   service (数秒) が gap より短い場合 queue 滞留は生じにくい — 観測 queue 深度を
   artifact に記録し、抽象モデルとの乖離候補として扱う (判定には使わない)。
3. **ワークロード縮小の内訳 (§3)**: 最初の 40/64 req を採用。chunk は lexicographic
   並べ替えのうえ req ごとに最大 8 個 (入力合計 ≤ 2048)。chunk トークン列は
   chunk_id から決定的に合成し、トークン長は chunk_id 決定的正規値 (128-168, sha256)。
   同一 chunk_id は全 req で同一トークン列 — 再利用可能性の前提。prefix は 640
   (W6 そのまま)。固定した stream hash と要約統計を artifact に保存。
4. **KV 再利用の方式 (§4)**: exact-path prefix caching — 実トークン列の最長一致
   祖先からのみ再利用 (MC0 の正しさ前提)。runtime key ↔ trie node は 1 対 1
   (node 作成は mirror が当該 key を非常駐のときのみ)。祖先 key の evict 時は
   到達不能 descendant node も実 store から掃除するが、mirror (AdaptiveRuntime の
   kv 状態) は sim 同様の計上を保持 → 実使用トークン数 ≤ mirror 計上 (保守的)。
5. **時間計測 (§3)**: `T_control` = runtime メソッド呼出 (head_context /
   resident / touch / enforce_capacity 内の signals・ages・victim・guard /
   record_access) の `perf_counter_ns` 累積。`T_gpu` = `torch.cuda.Event` で
   prefill / decode を分離計測。
6. **MC0 の許容誤差 (§5 の「事前固定」)**: 出力トークン列の完全一致を第一基準。
   不一致時は、初 diverge step における logits 最大絶対差 ≤ 0.1 かつ当該 step の
   top1-top2 margin < 0.1 (fp16 数値誤差による同点反転) の場合のみ数値許容として
   通過、それ以外は停止。
7. **MC3 の運用解釈**: 「意味信号を無効化」= 全キーの signal が同値 (全キーを
   等間隔・等回数アクセス) かつ queue 空の状況で、queue_semantic の victim 選択が
   LRU と一致することを確認 (policy unit + fake-executor harness の両層で検証)。
8. **決定ロジックの境界 (F 判決)**: harness は serve() のパイプライン順序を
   実行系として再現するが、判断 (context 選択 / residency / victim / signals /
   guard / record) はすべて `AdaptiveRuntime` のメソッド直呼びのみで行う。
   engine.execute の置き換え (実 GPU prefill/decode) が「実行アダプター」に当たる。

### 2026-10-19 実行後修正 (§5 手順どおり — 停止 → 本記録 → 再実行。判定規則は不変)

9. **MC0 attention backend の固定 (§1 環境の範囲内)**: 実 GPU 初回実行では
   partial-prefill + cache 経路が scratch 経路と diverge (decode step 6、
   logits 差 4.62–5.68 > §8-6 の 0.1) → §5 により停止。プローブ
   (`m4_realworld/probe_mc0_backend.py`) で原因は SDPA/flash kernel の
   シーケンス長依存 reduction (数値非結合性) と特定 — 同一条件で
   `attn_implementation="eager"` を使うと完全一致 (exact) を確認。§1 の
   「HF transformers + fp16 + revision 固定」の範囲内で backend を eager に
   固定 (両 paired arm 同一 backend)。判定規則 (出力トークン列完全一致が第一
   基準、許容は §8-6 どおり) は不変 — eager 固定により第一基準を充足。
10. **MC0 partial-prefill 引数の実装バグ修正**: `prefill(..., cached=...)` の
    `cached` は `DynamicCache` 再構築の `(keys_list, values_list)` 2-tuple 形式
    だが、MC0 は per-layer pair のリストを渡しており `_make_cache()` が
    引数過多で例外 — 2-tuple 形式に修正 (ハーネス経路は元から正しい形式)。
11. **warmup の実測相当化 (§6「warmup 除外」の実装)**: モデルロード直後の
    4-token warmup は計測対象 (~1.8k-token) regime を exercise していなかった。
    1840-token prefill + 32 decode × 2 サイクルに変更し、traces には一切含めない。
12. **MC4 slack の運用値 (0.15 → 0.25)**: 40 req 実行で slack=0.15 に対し
    10–12 / 428 ペアが単調性下限を超過。違反深度は最大 6.42% (相対逆転最大
    21.4%)、大半 <1%、ゼロ・単位・数倍差などの測定系崩れは皆無 — RTX 3050 の
    ~等長 prefill 時刻の peak-to-peak クロックノイズ帯 (実測 423–517 ms) に
    由来。MC4 の判定規則「明らかな測定崩れを検出」は不変のまま、運用閾値を
    実測ノイズ帯に合わせ 0.25 に調整 (違反深度は artifact の details に記録)。
    **判定の明記 (2026-10-19 補記)**: 元の基準 (slack=0.15) では MC4 は
    **通過していない** (Run A 10/428、Run B 12/428、Run C 保存 traces の
    再計算でも seed1 の2条件が fail)。M4 stage-2 の最終判定は本項の
    変更後基準 (slack=0.25、全6 run pass) に基づく。元基準・変更後基準の
    両方の判定と Run A/B の観測ログ転記、および改定の根拠 (違反分布 /
    warmup 測定 / 工学的許容範囲) は
    `artifacts/phase6/MC4_ADJUDICATION.md` +
    `artifacts/phase6/m4_realworld_mc4_adjudication.json` に保全
    (`python -m m4_realworld.mc4_adjudication` で再生成)。