# PREREG — Milestone 3: Semantic Demand Shuffle Control

状態: **ロック済み** (2026-10-19、実行前ロック・メンター承認)。
本ファイルは実行前に固定する。実行後は §7 アダンダムでのみ追記し、§①-§⑤・§6 の
判定規則は改変しない (M1 PREREG §5 と同一規約)。
成功基準は **Tier 2 とは無関係** (メンター指示) — M3 は
「Semantic Demand の意味的対応関係がランタイム改善に寄与しているか」の
検証実験であり、strong 比 +20% は成功基準に流用しない。

## 0. 仮説 (H1)

- **H1 (中心仮説)**: key ↔ future-demand の意味的対応関係が正しく保たれている
  ことが、F_full ランタイムの SLO goodput に正の寄与をしている。
- 独立変数: 対応関係 π の有無 (True / Shuffled) のみ。
- 従属変数: SLO goodput (G)。ΔG = G_true − G_shuffled。
- これは F 対 strong の比較ではない。**両条件とも F_full** であり、
  差分は semantic 対応関係の一点のみ (対照実験)。

## ① 主たる比較対象

- **True condition**: 本来の Semantic Demand 予測器入力をそのまま使用
  (`signal_perm=None` → 現行 F とコードパス完全同一)。
- **Shuffled condition**: 意味的対応関係のみを §2 の方法で事前定義した
  置換 π でシャッフルした入力を使用。
- 同一リクエスト系列・同一シード・同一ランタイム設定 (F_full) で両条件を対比。
- **主指標**: 家族・シード毎の対応差
  ΔG_s = G_true(s) − G_shuffled(s) (s = seed 1..12)。
- 副次記録 (判定には使わない): evictions / prefetches / prefetch_hits /
  prefetch_waste / hit 率 / SLO 達成数 — 介入経路のメカニズム解釈用。

### Semantic Demand の実体 (本リポジトリでの定義)

Phase 6 の F において予測器信号は §33 online popularity feedback
(`access_count` → 正規化 `_signal(k) = count(k)/max`) で、以下 **4 経路**を
消費する (2026-10-19 コード監査で確定):

| # | 消費箇所 | 影響する決定 |
|---|---|---|
| 1 | `_signals()` → `_choose_victim` | 追放対象選択 (queue_semantic score) |
| 2 | `_signals()` → `_top_prefetch_target` | プリフェッチ対象選択 |
| 3 | `_signal()` → `_plan_context` (reuse_prob) | コンテキスト選択 (budget 最適化) |
| 4 | `_signal()` → prefetch benefit + `access_count.get(target)` の min_seen 判定 | プリフェッチ発行可否 |

(`_signal` → `_request_state.prefix_reuse_probability` は 3/4 と同じ読み出し層。)

## ② 何を変えて、何を変えないか

| 項目 | 扱い |
|---|---|
| リクエストの到着時刻・系列 | **固定** (family+seed で両条件完全同一。stream hash を記録) |
| 非意味的入力 (tokens, benefit, SLO, output, chunk 構造) | **固定** |
| 実アクセス履歴 `access_count` 本体 | **固定** (記録は常に真値。介入は参照時のみ) |
| queue view / ages / LRU 順 / EDF / 全ガードパラメータ | **固定** |
| ランタイム設定・評価指標 (F_full, SLO goodput) | **固定** |
| ワークロードシード (workload RNG) | **固定** (両条件で同一 RNG 系列) |
| **Semantic Demand の対応関係 (key → signal 値の写像)** | **シャッフル対象 (π のみ)** |
| 予測器の出力分布 | 一致を**強制しない** (下記注意参照) |
| retention / prefetch / context の意思決定、evictions, hits, goodput | **介入の結果として測定** (一致させない) |

**入力分布の保存と出力強制の区別 (メンター②の趣旨)**:
- 保存するのは signal 値の**多集合 (分布)** と、非意味的入力・実履歴・
  キュー・時系列である。π は単置换なので値の多集合は真値と構造的に同一。
- 強制してはならないのは **それより下流** — 意思決定 (追放/選択/発行) と
  その結果 (goodput 等) を True と一致させること。本設計は下流を一切拘束
  しない。介入は「対応関係の層」にのみ行い、決定は自由に変化させ、
  それを測定する。正規化 (max 除算) は等変変換のため、π はこの層に
  与えれば下流全体に一貫して伝播する。

### シャッフル仕様 (実装前に固定 — v1)

- **単位**: key (prefix_key と chunk_id の全キー)。
- **対応づけ方法**: `fisher_yates_lex_v1`
  1. stream (WorkloadRequest 全体) から key 全集合 U を事前抽出 —
     prefix_key ∪ 全 chunk_id。**集合列挙は必ず辞書順ソート後**に行う
     (PYTHONHASHSEED 非依存の教訓を踏襲)。
  2. `rng = random.Random(sha256(f"M3-SHUFFLE-v1|{family}|{seed}"))` —
     ワークロード RNG と完全分離した専用乱数系列。
  3. `sorted(U)` に Fisher-Yates を適用して置换 π を構築 (U 上の全置換)。
  4. 適用: 4 経路すべての signal 読み出しを `π(k)` 経由に変換
     (`_signal(k)` → raw(π(k))、`_signals()` → {k: raw(π(k))}、
     min_seen の `access_count.get(target)` → `access_count.get(π(target))`)。
     `_record_access` は真値のまま (②の「固定」)。
     `_weak_signal` は値の分布のみに依存 (対応不変) → 変更しない。
     `signal_perm=None` のとき全経路は現行コードと同一 (True condition)。
- **対応表の保存**: `artifacts/phase6/m3_shuffle_maps.json`
  (family/seed 毎に π 全件 + shuffle 導出 seed + 固定点数を記録)。
- **除外条件 (事前定義)**: 結果に基づく除外は一切しない。技術的無効のみ:
  |U| < 2 の run → 「除外 (shuffle 不能)」として記録;
  π が双射でない → **実験全体を無効化して停止** (判定不可);
  repeat 不一致 → 同様に停止・調査。

## ③ シードと再現性

- **シード 1-12** (M1 と同一)。**全 8 家族**を評価 — 結果を見て選び直さない。
  事前分類 (解釈用、判定閾値は共通): signal-active 期待群 = W4/W5/W6/W7/W8、
  方向期待なし群 = W1/W2/W3 (対応関係が F の勝ちに効いていないと
  わかっている並列群)。
- **True/Shuffled で同一 stream**: 同一 workload 関数・同一シードで生成し、
  stream hash を両条件に記録・一致検査。
- **乱数系列の分離**: workload RNG (seed) と shuffle RNG (sha256 導出) は
  別系統。shuffle 方法・導出式・対象件数 (|U|)・固定点数・除外を記録。
- **再現性確認 (shuffle 条件)**: 全 96 run を **2 回実行**し、goodput +
  全カウンタの完全一致を検査 (不一致は停止)。
- **True condition 一致検査 (最重要)**: M3 ランナーは strong config も
  同streamで実行し、(F_true − strong) の per-seed 値を
  `seed12_merged.json` の per_seed (M1 正本) と照合。
  **全 8 家族 × 12 シードで max diff < 1e-6** を要求 — 対照用の変更が
  本来の実験経路を変えていないことの証明。
- 既存の `tests/test_reproducibility.py` (PYTHONHASHSEED 0/6/1/42) を維持。

## ④ 統計的判定規則 (M1 プロトコルを維持)

- 対応比較 n=12 (家族毎、paired)。主指標は ΔG (絶対値)、
  効果量として相対値 mean(ΔG)/mean(G_true) も報告。
- **符号検定 (両側)** + **Holm 補正 (検定対象 = 全 8 家族)** +
  **bootstrap 95%CI (paired, 10,000 回, 分析 RNG シード固定 20261019)**。
- 確認的判定: Holm 補正後 p < 0.05 かつ meanΔG の符号と CI が整合。
- 効果量報告: meanΔG (絶対) + 相対 % + CI。CI 幅も必ず併記。
- **M3 の成功基準 (H1 判定、事前固定)**:
  - **支持**: 妥当性検査 (MC1-MC5) が全て通過 かつ 8 家族中 ≥1 家族で
    確認的かつ meanΔG > 0。
  - **確認できなかった**: 確認的家族が 0 (CI の境界を併記し、
    「等価」とは主張しない — 等価検定は事前登録していない)。
  - **逆方向**: 確認的かつ meanΔG < 0 の家族が出た場合 — H1 は支持されず、
    現行の需要信号の使い方に問題がある可能性 (§⑤ 第3行) を調査対象として
    報告。正負が混在する場合は両方をそのまま報告する。
- Tier 2 の再判定は行わない (本実験の目的外)。

## ⑤ 結果の解釈 (事前固定 — 変更禁止)

| 結果 | 解釈 |
|---|---|
| True が Shuffled を上回る (確認的) | 意味的対応関係が寄与するという H1 を支持 |
| 両者に明確な差がない | 今回の条件では、意味的対応関係の追加効果を確認できなかった |
| Shuffled が上回る (確認的) | 現行の需要予測・利用方法に問題がある可能性を調査 |

- **シャッフル方法の再調整禁止**: v1 の結果は分岐にかかわらず報告する。
  True が勝つまで方法を変えることはしない。v2 以降は新 PREREG が必要。
- **操作妥当性検査 (manipulation check、H1 解釈の前提)**:
  - MC1: π は双射 (全件 assert)。
  - MC2: signal 値の多集合が True と同一 (単位テスト + runner 検査)。
  - MC3: stream hash が両条件で一致。
  - MC4: True 条件が M1 正本と一致 (§③, max diff < 1e-6)。
  - MC5: shuffle 条件の再実行一致 (§③, 完全一致)。
  - これらが欠けた場合、結果は解釈せず「検証不能」と報告する。
- True 勝ちだけで中心仮説全体の証明にはならない。主張は MC1-MC5 が
  担保する範囲 (介入が対応関係のみを崩したこと) に限定する。

## 6. 実行仕様と成果物 (固定)

- 実行数: 96 (family×seed) × {strong, F_true, F_shuffled×2} = **384 runs**。
  概算 ~2 分 (gate v2 実測 0.23s/run)。
- 実装: `compare/m3_shuffle_runner.py` (実行 + MC 検査)、
  `compare/m3_shuffle_analysis.py` (統計解析)。
- 成果物: `m3_shuffle_maps.json` / `m3_shuffle_<fam>_s<k>.json` (×96、
  両条件 + repeat 一致フラグを内包) / `m3_shuffle_merged.json` /
  `m3_shuffle_analysis.json`。
- 既存正本 (seed12_merged, gate_v2, strong_reeval) は**非改変**。
- テスト: `tests/test_m3_shuffle.py` (π 構築の決定性・双射・多集合保存・
  `signal_perm=None` の現行同一性・hash 非依存)。

### メンター完了条件との対応

- [ ] PREREG に仮説・介入方法・評価指標・判定規則を記録 → **本ファイル §0-§4**
- [ ] シャッフル対象と維持対象を明確に定義 → **§②テーブル + シャッフル仕様**
- [ ] True condition の既存結果との一致を確認 → **§③ max diff < 1e-6 (MC4)**
- [ ] Shuffled condition の再現性を検証 → **§③ repeat 完全一致 (MC5)**
- [ ] 全対象ワークロードの結果と統計解析を保存 → **§6 成果物 (全8家族)**
- [ ] 支持・非支持を同じ基準で報告 → **§⑤ 三分岐表 + 一括報告**
- [ ] 結論と主張境界を更新 → 実験後に CLAIM_BOUNDARY.md / MILESTONES.md 更新

## 7. アダンダム (実行後に追記 — 判定規則は不変)

**実行記録 (2026-10-19、判定規則 §①-§⑤ は無改変)**

- ロック: 実行前にロック。直前、ファイル末尾へ誤って分離していた 1 文
  (§② 結びの「正規化 (max 除算) は等変変換…」) を本文の該当段落へ復元の上、
  状態を「ロック済み」に更新。
- 実装: `RuntimeConfig.signal_perm` (None = 現行コード完全同一 = True)。
  読み出し 4 経路を `_perm_key` / `_apply_perm` で π 経由化
  (`runtime/adaptive_runtime.py`)。`_record_access` / `_weak_signal` /
  queue / 時系列 / 全ガードは不変。テスト `tests/test_m3_shuffle.py` 15 件、
  全体 275 passed。
- 実行: `python -m compare.m3_shuffle_runner` → 384 run / 6.0 秒。
  MC1-MC5 全通過。MC4 max diff = 4.988e-07 < 1e-6 (seed12_merged の per_seed
  は第 6 位丸め保存のため 5e-7 桁の差は正本側丸め由来)。MC5 全 96 ペア
  完全一致。除外 (|U|<2) 0 件。
- **判定 (§5 三分岐): 「支持」** — 確認的 (Holm(8)<0.05 かつ CI95 が 0 を
  含まない) 且つ meanΔG>0 が 6/8 家族:
  W2 +0.002247 (Holm 0.0039, 11/0/1) / W4 +0.003002 (0.0024, 12/0/0) /
  W5 +0.000103 (0.0190, 11/1/0) / W6 +0.008205 (0.0024, 12/0/0) /
  W7 +0.007184 (0.0024, 12/0/0) / W8 +0.009463 (0.0024, 12/0/0)。
  確認的 negative 0 件。W1 12 タイ (p=1.0)、W3 非確認的 (5/6/1, p=1.0,
  CI [−0.000131, +0.000117] — 等価は主張しない)。
- 実装上の明確化 (§2 の読み方。判定規則に影響なし):
  1. U は stream から静的に抽出 (prefix_key ∪ chunk_id、lex-sorted)。
  2. MC2 は「U 全体での evidence 多集合の不変」で実装 (sorted 整数列の
     完全一致)。`_signals()` のアクセス済集合 A 制限ビューは §2-step4 どおり
     {k: raw(π(k))} (π(k) が未エビデンスなら 0.0) — 「出力分布は強制しない」
     方針に従い A 制限の多集合一致は要求していない。
  3. 事前分類「方向期待なし」群の W2 が確認的 positive — 共通閾値の適用
     結果であり、事後的な対象・閾値の選択変更は行っていない。
- 既存正本 (seed12_merged / gate_v2 / strong_reeval) は非改変 (読み取りのみ)。
- §6 メンター完了条件: 本記録で全項目充足を確認 (チェックボックスはロック
  保持のため非改変のまま残置)。

