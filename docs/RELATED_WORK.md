# 関連研究との比較 (Related Work — 位置づけ)

> **本ドキュメントの位置づけ**: 既存研究が解決している問題と、本
> リポジトリの Adaptive Inference Runtime が検証しようとしている問題の
> 差異を説明する文書である。**本比較は位置づけの整理であり、性能比較
> (benchmark) を実施したことではない** — 負荷・TTFT・スループット条件
> を揃えたベンチマークは行っていない。この2つは厳格に区別すること
> (`RELEASE_NOTES.md` の残ゲートにも同じ区別を明記)。

## 各研究の位置づけ

| 研究・システム | 解決している問題 | 本リポジトリとの共通点 | 相違点 (本リポジトリの問い) |
|---|---|---|---|
| [vLLM Automatic Prefix Caching](https://vllm.readthedocs.io/en/latest/conceptual/prefix_caching.html) | 共有 prefix の KV を再利用し、重複 prefill を省略する仕組み | prefix 再利用を価格モデルの基盤として利用 (`runtime/engine.py`)。再利用による計算削減は共通で検証 | 再利用**そのもの**は新規主張しない (§72)。問うのは、需要予測信号で機構間を**協調**させたときの追加 E2E 利益 |
| [SGLang](https://arxiv.org/abs/2312.07104) | 推論サービングの実装・比較対象 (RadixAttention 等) | 推論サービング全体の効率という評価軸 | サービングエンジンの実装競争。本リポジトリは単一 GPU・小さなモデルの**メカニズム検証**であり、サービング性能比較ではない (Linux/実エンジンでの再現は未検証 — README 主張境界) |
| [LMCache](https://lmcache.github.io/) | KVキャッシュの保存・転送・再利用を扱う管理レイヤ (prefix に限らず複数保存先) | KV を保持し再利用する保存管理の考え方 | LMCache が保存**レイヤ**を広くするのに対し、本リポジトリは保存先の分散ではなく、**各機構が共有する需要信号の協調**が E2E に与える利益を問う |
| [KVFlow](https://arxiv.org/abs/2507.07400) | エージェントの実行グラフから将来の KV 需要を見積もり、保持や prefetch を調整する研究 | 「将来の需要予測 → 保持/prefetch 決定」という予測的管理の考え方。本リポジトリの predictive eviction / prefetch 計画 (Phase 3 ladder) と問題設定が近い | 予測機構単体の性能ではなく、その予測を eviction・scheduling・context 選択が**共有信号**として協調利用したときの総合利益を検証。本リポジトリの M3 はその信号の因果寄与を shuffle 対照で切り分けた |
| [PEEK](https://arxiv.org/abs/2607.02525) — Predictive Queue-Informed KV Cache Management for LLM Serving | 待ち行列と prefix 共有を利用し、キャッシュ保持とリクエスト順序を協調させる研究 | queue-aware + prefix 共有の協調という着想は同一。本リポジトリの C_queue 構成は **PEEK-like 既存追試**として ladder に組み込み済み (`CLAIM_BOUNDARY.md`) | 単独最適としての queue-aware ではなく、semantic demand 信号と組み合わせた協調 (F 構成) の増分利益を問う。実機での queue 制御効果は未検証 (M4 で滞留なし) |
| CacheScout — Learning Agent Execution for KV-Cache Management in Agentic Serving (2026) | エージェントの実行遷移から将来の KV 再利用を予測し、キャッシュの eviction と prefetch の判断に利用する研究 (vLLM 上の実装報告あり) | エージェント実行の意味構造・将来需要予測から KV キャッシュ管理 (eviction / prefetch) を最適化する問題意識 | CacheScout が実行遷移予測に基づくキャッシュ単体の管理に注力するのに対し、本リポジトリは Semantic Demand という共有信号を eviction・scheduling・context 選択の**複数機構間で協調**させたときの E2E 利益を問う。また M3 で信号の因果寄与を shuffle 対照で検証した点も異なる |
| Tessera — Demand-Driven KV Cache Management for Retrieval-Augmented LLM Serving (2026年9月) | RAG やエージェント記憶における需要と履歴、KV の居住状態 (residency)、リクエストの振り分け (routing) を協調させる研究 | 需要駆動 (demand-driven) で KV キャッシュの居住状態やリクエスト処理を統合的に捉える着想 | Tessera が RAG / 検索増強環境でのリクエストルーティングと KV 居住の協調に主眼を置くのに対し、本リポジトリは単一ランタイム内での Semantic Demand 信号による階層的協調 (residency, eviction, scheduling, context choice) の増分利益を検証する |

## 共通する問い / 違う問い

- **共通**: KV / prefix の再利用、保持ポリシー、予測的管理 (CacheScout / KVFlow)、
  待ち行列利用 (PEEK)、需要駆動協調 (Tessera) — いずれも既存研究の確立・
  発展途上領域であり、本リポジトリはそれらを新規発明として主張しない
  (`artifacts/phase6/CLAIM_BOUNDARY.md`)。
- **本リポジトリ固有の問い**: 「Semantic Demand を共有信号として既存の
  ランタイム機構 (eviction / scheduling / prefetch / context) を協調させたとき、
  追加の End-to-End 利益が存在するか」。
  対応する証拠はシミュレーション (Gate v2 / M3 shuffle) と M4 の
  限定的な実 GPU 測定のみであり、上記のいずれかのシステムとの
  **性能比較は未実施**。

## 未検証項目 (この比較文書から生じるもの)

- 既存システム (vLLM APC / SGLang / LMCache / CacheScout / Tessera 等) と同じ条件での
  **数値的ベンチマーク** — 実施していない。
- 実サービングエンジン上での協調効果 (README「主張境界」の未確認リスト)。
- KVFlow / PEEK / CacheScout / Tessera の手法そのものの本リポジトリ内完全追試
  (C_queue は PEEK-like、predictive は Phase 3 ladder 上の一点として部分的にのみ対応)。
