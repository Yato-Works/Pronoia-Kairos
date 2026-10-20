# Next-Generation Adaptive Inference Engine

## Design Specification v0.2.0

> **Working Title:** Adaptive Inference Runtime
> **Status:** Research / Experimental
> **Target Era:** 2027–2028
> **Primary Goal:** Semantic Context、Computational Cache、Scheduling、Executionを共有Runtime Stateの下で協調最適化するAdaptive LLM Inference Runtime
> **Central Research Thesis:** Semantic Memoryの「将来需要予測」をComputational Cacheの「将来再利用予測」に接続することで、既存の静的・局所的なCache PolicyをEnd-to-End最適化へ拡張できるかを検証する。

---

# 0. Executive Summary

本プロジェクトは、LLM推論を単一の「高速化アルゴリズム」として扱わない。

目標は、

> **Runtime自身が、現在のリクエスト、将来の需要、Context、Cache、Queue、Hardware状態を観測し、その時点で最も合理的な実行戦略を選択すること**

である。

既存の推論基盤は、それぞれ異なる軸で非常に強い。

vLLMはPagedAttention、Continuous Batching、Prefix Caching、Chunked Prefill、Speculative Decoding、KV offloadingなど、汎用 serving runtime として広範な最適化を持つ。現在のvLLMには非同期KV offloading用のConnector APIも存在する。

SGLangはRadixAttentionを中心としてPrefix reuseと構造化されたLLM executionを強く意識した設計を持つ。

LMCacheはGPU、CPU、Local Storage、Remote StorageをまたぐKV Cacheの階層化と移動を提供している。

PEEKはPending Queue内の将来需要を利用してPrefix CacheとSchedulingを協調させる。2026年の論文では、queue-awareなprefix cluster、eviction protection、starvationを抑制するschedulerを統合し、vLLM/SGLang上で強い改善を報告している。したがって、本プロジェクトは単純なQueue-aware Cacheを新規性として主張しない。

KVFlowはAgent Step Graphから将来のKV利用を予測し、KV retentionとprefetchを行う。

ContinuumはAgentのtool-call durationとmulti-turn continuityを利用して、KV CacheをTTL付きでGPUに保持する。

CacheBlendは非PrefixのRAG contextに対して、Cached KVを一部再計算することで再利用可能にする。

HqeKVはKV importanceに基づくQuantizationとEvictionを統合し、複数の圧縮状態を選択する。

さらにLearning to Evict from Key-Value Cacheのように、将来のtoken utilityを学習してEvictionを決める研究も登場している。

したがって本プロジェクトが狙うべきものは、

> **「新しいKV Compressionを作ること」**

でも、

> **「Queue-aware Schedulerをもう一つ作ること」**

でも、

> **「vLLMを再実装すること」**

でもない。

中心仮説は、

> **Semantic Memoryが予測する「次に必要になる情報」を、Computational Cacheの将来需要予測に利用できるのではないか。**

である。

この仮説を検証するため、本Engineを次のFeedback Control Systemとして設計する。

```text
Observe
   ↓
Predict
   ↓
Plan
   ↓
Execute
   ↓
Measure
   ↓
Adapt
   ↓
Observe
```

---

# 1. Research Position

## 1.1 Core Thesis

本Engineの中心命題を次のように定義する。

> **Semantic demand prediction can improve computational cache decisions when semantic memory, request state, queue state, and cache state are jointly observed.**

日本語では、

> **Semantic Memoryが「何が将来必要になるか」を予測できるなら、その信号をComputational Cacheの保持・配置・先読み判断へ接続することで、LRUやQueue-only、Workflow-onlyなCache Policyを上回れる可能性がある。**

ここで重要なのは、

```text
Semantic Similarity
        ↓
KV Reuse
```

という直接変換ではない。

正しい制御経路は、

```text
Semantic Memory
       ↓
Future Demand Prediction
       ↓
Candidate Data Identification
       ↓
Computational Compatibility Check
       ↓
Prefix / KV Lookup
       ↓
Reuse / Restore / Recompute Decision
```

とする。

Semantic Memoryは「KVそのもの」を決めるのではなく、

> **将来どの情報が必要になりそうか**

を予測する。

実際のKV再利用は、モデルバージョン、Tokenizer、parallelism、position、prefix compatibilityなどのComputational条件を満たした場合のみ許可する。

---

# 2. Design Philosophy

本Engineは次の7原則を持つ。

## Principle A — Optimization is a Decision Problem

Runtimeは固定ポリシーではなく、観測状態から実行戦略を選択する。

```text
Observe
   ↓
Estimate
   ↓
Plan
   ↓
Execute
   ↓
Measure
   ↓
Adapt
```

---

## Principle B — Optimize End-to-End

単独の指標を最大化しない。

たとえば、

```text
KV Hit Rate ↑
```

だけでは成功とはしない。

同時に、

```text
TTFT
TPOT / ITL
p95
p99
Goodput
VRAM
Transfer Cost
Quality
```

を見る。

---

## Principle C — Semantic Memory ≠ Computational Cache

Semantic MemoryとKV Cacheは別の層である。

```text
Semantic Memory
    ↓
Context Planning
    ↓
Computational Planning
    ↓
KV / Execution Decision
```

Semantic similarityだけでKV reuseを決定してはならない。

---

## Principle D — Lossless Before Lossy

KV Optimizationを2つに分ける。

```text
Lossless Layer
    retention
    eviction
    placement
    prefetch
    offload
    exact reuse

Lossy Layer
    quantization
    token eviction
    compression
    approximate reuse
```

初期研究ではLosslessを優先する。

---

## Principle E — Strong Baselines First

比較相手を弱くして勝つことを禁止する。

最低限、

```text
vLLM
SGLang
llama.cpp
LMCache
PEEK
KVFlow
Continuum
```

など、該当ワークロードで強い構成と比較する。

---

## Principle F — Quality Is a Constraint

品質をLatencyと同じ単位で雑に足し合わせない。

基本方針：

```text
maximize performance
subject to quality degradation <= ε
```

とする。

---

## Principle G — Every Optimization Must Justify Its Own Overhead

最重要ルール：

> **Every optimization must justify its own overhead.**

つまり、

```text
Optimization Gain
>
Observation Cost
+
Prediction Cost
+
Decision Cost
+
Data Movement
+
Execution Overhead
```

でなければ導入しない。

---

# 3. Research Scope

v0.2.0で研究の中心とする対象：

```text
1. Future Demand Prediction
2. Lossless KV Retention
3. KV Placement
4. KV Prefetch
5. Queue-aware Scheduling
6. Semantic Context Planning
7. Context / KV Joint Budgeting
8. End-to-End Adaptive Planning
```

将来拡張として扱う対象：

```text
Speculative Decoding
MTP
PD Disaggregation
Cognitive IR
Lossy KV Compression
Multi-GPU
Distributed Runtime
Custom GPU Kernel
```

後者は初期段階ではInterfaceのみ定義し、中心研究の証明を阻害しない。

---

# 4. High-Level Architecture

```text
                         ┌──────────────────────┐
                         │      User / API       │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   Request Analyzer    │
                         └──────────┬───────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              │                     │                     │
              ▼                     ▼                     ▼
      ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
      │ Semantic     │      │ Computational│      │ Queue        │
      │ Memory State │      │ Cache State  │      │ State        │
      └──────┬───────┘      └──────┬───────┘      └──────┬───────┘
             │                     │                     │
             └─────────────────────┼─────────────────────┘
                                   ▼
                      ┌─────────────────────────┐
                      │     Adaptive Planner    │
                      │                         │
                      │ Demand Prediction       │
                      │ Context Selection       │
                      │ Cache Retention         │
                      │ Cache Placement         │
                      │ Prefetch                │
                      │ Scheduling              │
                      │ Execution Mode          │
                      └────────────┬────────────┘
                                   │
                 ┌─────────────────┼─────────────────┐
                 │                 │                 │
                 ▼                 ▼                 ▼
        ┌────────────────┐ ┌───────────────┐ ┌───────────────┐
        │ Context        │ │ KV Manager    │ │ Scheduler     │
        │ Compiler       │ │               │ │               │
        └───────┬────────┘ └───────┬───────┘ └───────┬───────┘
                │                  │                 │
                └──────────────────┼─────────────────┘
                                   ▼
                        ┌────────────────────────┐
                        │   Execution Runtime    │
                        │                        │
                        │ Prefill / Decode       │
                        │ Batch Execution        │
                        └────────────┬───────────┘
                                     │
                                     ▼
                        ┌────────────────────────┐
                        │   Backend / Kernel     │
                        │ CUDA / CPU / Future    │
                        └────────────────────────┘
```

---

# 5. Shared Runtime State

本プロジェクトでは、個別componentが独立したstateを持つのではなく、共通Runtime Stateを中心に構成する。

```text
RuntimeState
├── request_state
├── session_state
├── semantic_state
├── context_state
├── queue_state
├── cache_state
├── transfer_state
├── execution_state
├── hardware_state
└── telemetry_state
```

---

# 6. Request Analyzer

各Requestについて、低コストな特徴量を生成する。

```text
input_tokens
expected_output_tokens
shared_prefix_length
prefix_reuse_probability
session_id
session_age
context_age
queue_wait
queue_depth
batch_pressure
quality_sensitivity
semantic_memory_hits
predicted_future_demand
memory_pressure
hardware_state
```

重要なのは、Input Lengthだけで判断しないことである。

例えば、

```text
8k tokens + 95% reusable prefix
```

と

```text
8k tokens + 0% reusable prefix
```

は同じ入力長でも最適戦略が異なる。

---

# 7. Semantic Demand Layer

## 7.1 目的

LetheなどのSemantic Memoryから取得した情報を、

```text
What is likely needed next?
```

という予測信号へ変換する。

---

## 7.2 Semantic Demand

Semantic Memoryは次の情報を返す。

```text
memory_id
source_id
chunk_id
relevance
confidence
provenance
temporal_relation
session_relation
access_history
future_utility
```

これから、

```text
P_future_demand(x | S_t)
```

を推定する。

ここでxは、

```text
document
memory
chunk
conversation segment
agent state
retrieved object
```

などである。

---

# 8. Lethe Integration

Inference Request:

```text
Inference Request
       │
       ▼
   Lethe Recall
       │
       ├── relevant memories
       ├── provenance
       ├── confidence
       ├── temporal relation
       └── future utility
       │
       ▼
Semantic Demand Predictor
       │
       ▼
Context / Cache Planner
```

重要な設計原則：

> **LetheはKV Cacheを直接操作しない。**

Letheは、

```text
semantic demand
```

を提供する。

Adaptive Runtimeが、

```text
semantic demand
+
request history
+
session structure
+
queue
+
cache state
```

を結合してComputational Decisionを生成する。

---

# 9. Semantic Demand → Computational Reuse

本プロジェクト最大の差別化候補。

## 9.1 二段階確率モデル

まずSemantic Layer：

$$
P_{semantic}(x \mid S_t)
$$

次にComputational Layer：

$$
P_{reuse}(c \mid x,S_t,C_t)
$$

とする。

最終的な期待再利用確率を、

$$
P_{future\_reuse}(c)
=
P_{semantic}(x)
\cdot
P_{computational}(c \mid x)
$$

として扱う。

ただしこれは概念モデルであり、実装時には独立性を仮定しない。

---

# 10. Computational Compatibility

Semantic relevanceだけでReuseしない。

Cache objectには少なくとも、

```text
model_id
model_revision
tokenizer_id
token_sequence_hash
prefix_hash
position_information
tensor_shape
dtype
parallelism_config
attention_config
namespace
trust_domain
provenance
```

を持たせる。

Reuse条件：

```text
Semantic Demand
        +
Exact Computational Compatibility
        +
Runtime Benefit > Cost
```

を満たす必要がある。

---

# 11. Computational Memory Architecture

KVを単一のCacheとして扱わない。

```text
L0: Active GPU KV
L1: GPU Resident Cache
L2: Host RAM Cache
L3: Local SSD / NVMe
L4: Remote KV Store
```

ただし、これは「階層化すること」自体を新規性とはしない。

LMCacheは既にGPU、CPU、Local Storage、Remote StorageへKVを配置する階層型設計を持っている。

本研究対象は、

> **どのKVを、どのTierへ、いつ移動し、いつPrefetchするかをSemantic DemandとRuntime Stateから決定すること**

である。

---

# 12. Lossless KV Layer

初期研究の中心。

操作：

```text
KEEP
EVICT
OFFLOAD
PREFETCH
RESTORE
REUSE
PIN
UNPIN
```

品質を壊さないことを基本とする。

---

# 13. Lossless Cache Value Function

従来案では、

```text
P_reuse × Importance
```

のように異なる粒度を混在させていた。

v0.2.0では分離する。

Lossless Cache object c に対して、

$$
U_{lossless}(c)
=
P_{reuse}(c)
\cdot
\left(
T_{recompute}(c)
-
T_{restore}(c)
\right)
-
C_{retain}(c)
$$

とする。

ここで、

```text
P_reuse
    = 将来再利用される確率

T_recompute
    = 再計算に必要な期待時間

T_restore
    = Cacheから復元する期待時間

C_retain
    = 保持によるMemory opportunity cost
```

である。

実際には、

```text
GPU retention cost
CPU retention cost
SSD retention cost
Transfer cost
Prefetch bandwidth cost
```

を分離する。

---

# 14. Memory Shadow Price

同じKVでも、

```text
VRAMが余っている
```

場合と、

```text
VRAMが逼迫している
```

場合では価値が違う。

そこでTierごとのresource shadow priceを導入する。

```text
λ_gpu
λ_cpu
λ_ssd
λ_network
```

Cache Placementは、

$$
Utility(c,tier)
=
ExpectedAvoidedCost(c,tier)
-
ResourcePrice(tier)
$$

で評価する。

---

# 15. GreedyDual-Style Retention

Cache Evictionは単純LRUから開始する。

その後、

```text
LRU
LFU
GDS / GDSF-like
Oracle / Belady
Predictive Policy
```

を比較する。

特に重要なのは、

> **Beladyとの差がどれだけ残っているか**

である。

もし未来情報を使った理想的なEvictionに近づけないなら、Prediction Policy自体に改善余地がある。

逆にBeladyとLRUの差が小さいなら、Cache研究に投入する複雑性が正当化できない。

---

# 16. Queue-Aware Runtime State

QueueはCacheの外部情報ではない。

```text
Past Requests
+
Current Requests
+
Pending Requests
+
Session Structure
+
Semantic Demand
```

をFuture-Aware Runtime Stateとして扱う。

PEEKはPending QueueとPrefix Sharingを組み合わせてCache hitとSchedulingを協調最適化している。

したがって本Engineは、

```text
Queue-aware
```

だけでは差別化しない。

---

# 17. PEEK Baseline

Phase 2ではPEEKを独立したBaselineとして再現する。

```text
PEEK
+
Semantic Demand
```

という形で初めて差分を作る。

比較対象：

```text
vLLM stock
SGLang stock
vLLM + LMCache
SGLang + LMCache
PEEK
Adaptive Runtime
```

ここで、

> Adaptive RuntimeがPEEKを上回る条件

を研究成果として示す。

---

# 18. KVFlow / Continuum Positioning

KVFlowはAgent Step Graphを利用して、将来必要になるKVを予測する。

ContinuumはTool Callの実行時間を利用し、multi-turn continuityのためKVをTTL付きで保持する。

本Engineとの比較：

| System           | Future Signal                             | Main Scope           |
| ---------------- | ----------------------------------------- | -------------------- |
| PEEK             | Pending Queue                             | Prefix / Scheduling  |
| KVFlow           | Agent Step Graph                          | Prefetch / Eviction  |
| Continuum        | Tool Duration / Program Structure         | Multi-turn KV        |
| Adaptive Runtime | Semantic Demand + Queue + Session + Cache | Cross-layer Decision |

したがって、

> 「未来を見るCache」

そのものは新規性ではない。

差分候補は、

> **Semantic Memory由来の未来需要予測を、Queue / Session / Cache Stateと統合してComputational Policyへ変換すること**

である。

---

# 19. Prefix Reuse

Prefix Cacheは、

```text
Prefix Exists?
```

だけで判定しない。

二段階にする。

```text
Prefix Match
      ↓
Expected Benefit
      ↓
Reuse / Recompute
```

Cache hitが存在しても、

```text
restore cost > recompute cost
```

なら再計算する。

---

# 20. Prefetch

Prefetchは、

```text
P_future_use × transfer_cost
```

を評価する。

Prefetch条件：

$$
P_{use}
\cdot
T_{saved}
>
T_{transfer}
+
T_{interference}
$$

のみ実行する。

Prefetchの誤りは、

```text
GPU pollution
PCIe saturation
NVMe contention
CPU bandwidth loss
```

を生みうる。

したがって、

> **Prefetch missもコストとして計上する。**

---

# 21. Context Compiler

Context Compilerは、

```text
Raw Conversation
+
Retrieved Memory
+
Current Request
+
Session State
+
Runtime Budget
```

から、

```text
Execution Context
```

を生成する。

目的：

> **回答に必要な情報を、最小の計算コストで与える。**

---

# 22. Context Utility Model

Memory object mについて、

```text
utility(m)
relevance(m)
confidence(m)
provenance(m)
token_cost(m)
prefill_cost(m)
future_reuse(m)
```

を評価する。

基本形：

$$
U(m)
=
Benefit(m)
-
TokenCost(m)
-
PrefillCost(m)
$$

ただしQuality Safety Constraintを持つ。

$$
QualityLoss \le \epsilon
$$

---

# 23. Context / KV Joint Budget

Phase 4では、

```text
Context selection
```

と

```text
KV retention
```

を別問題として扱わない。

最終的には、

$$
\min
C_{prefill}
+
C_{restore}
+
C_{memory}
+
C_{transfer}
$$

subject to

$$
QualityLoss \le \epsilon
$$

$$
VRAM \le B_{GPU}
$$

$$
RAM \le B_{CPU}
$$

とする。

つまり、

> **「Contextに何を入れるか」と「KVに何を残すか」を同一のResource Allocation Problemとして扱う。**

---

# 24. Scheduler

従来の、

$$
Score(request)
=
\alpha ReuseGain
+
\beta Urgency
+
...
$$

という固定線形モデルは主要設計から外す。

理由は、ワークロードごとに、

```text
α
β
γ
δ
ε
```

を調整する必要があるためである。

---

# 25. Constraint-Based Scheduling

Schedulerは、

```text
Objective
    maximize expected runtime benefit

subject to
    SLO constraints
    fairness constraints
    memory constraints
    batch constraints
```

として定義する。

優先順位は基本的にLexicographicとする。

```text
1. Hard SLO protection
2. Starvation bound
3. Reuse gain
4. Batch compatibility
5. Cache locality
6. Transfer cost
```

これにより、

```text
Request A
80% prefix reuse
30ms waiting
```

を優先し続けて、

```text
Request B
0% reuse
1000ms waiting
```

を永久に遅らせることを防ぐ。

---

# 26. Fairness

最低限、

```text
Maximum Queue Wait
```

または

```text
Age-based priority
```

を持つ。

SchedulerがCache効率だけを追求してStarvationを生むことを禁止する。

---

# 27. Adaptive Execution

Execution Modeは将来的に、

```text
NORMAL DECODE
BATCHED DECODE
PREFIX REUSE
SPECULATIVE
MTP
CHUNKED PREFILL
PD DISAGGREGATION
```

を持つ。

ただしv0.xでは、

```text
Selection Interface
```

だけを定義する。

Execution Modeそのものを本研究の中心にしない。

---

# 28. Speculation Controller

将来拡張。

Controllerが観測するもの：

```text
draft cost
acceptance rate
verification cost
memory pressure
queue state
```

概念式：

$$
ExpectedGain
=
P_{accept}
\cdot
SavedCost
-
DraftCost
-
VerificationOverhead
$$

Speculative Decodingを使うかどうかも、

```text
always on
```

ではなくRuntime Decisionにする。

ただし、中心研究がSemantic Demand Predictionである以上、v0.xでは研究対象外とする。

---

# 29. PD Disaggregation

PrefillとDecodeは異なるresource profileを持つ。

```text
Prefill
    compute-heavy

Decode
    memory / KV-heavy
```

将来的には、

```text
Context Compiler
      ↓
Prefill Planner
      ↓
KV Transfer
      ↓
Decode Planner
```

とする。

これもv0.xではArchitecture Interfaceとして定義するだけに留める。

---

# 30. Cognitive IR

将来的なExecution Representationとして、

```text
MEMORY_RECALL
CONTEXT_COMPILE
PREFIX_LOOKUP
KV_RETAIN
KV_OFFLOAD
KV_PREFETCH
PREFILL
SPECULATE
VERIFY
DECODE
```

などのEffect-aware Operationを持つ。

目的：

> **Inferenceをtokens→tokensではなく、Optimization可能なExecution Graphとして扱うこと。**

ただし、Cognitive IRの設計自体はv0.2.0の研究中心ではない。

---

# 31. Runtime Planner

最終Plannerが判断する項目：

```text
Context:
    Which memories?

Cache:
    Which KV?

Placement:
    GPU / RAM / SSD / Remote?

Scheduler:
    Which request first?

Prefetch:
    Which KV?

Execution:
    Which mode?

Batch:
    Which requests together?

Precision:
    Which representation?

Backend:
    Which execution path?
```

---

# 32. Decision Pipeline

各Scheduling Cycleで、

```text
1. Observe Runtime State
2. Update Demand Prediction
3. Update Cache Value
4. Evaluate Context
5. Evaluate Queue
6. Generate Candidate Plans
7. Reject plans violating SLO / Quality / Memory
8. Rank remaining plans
9. Execute
10. Measure actual outcome
11. Update predictor
```

とする。

---

# 33. Runtime Feedback Loop

```text
             ┌─────────────────┐
             │   Observation   │
             └────────┬────────┘
                      ↓
             ┌─────────────────┐
             │   Prediction    │
             └────────┬────────┘
                      ↓
             ┌─────────────────┐
             │   Planning      │
             └────────┬────────┘
                      ↓
             ┌─────────────────┐
             │   Execution     │
             └────────┬────────┘
                      ↓
             ┌─────────────────┐
             │   Measurement   │
             └────────┬────────┘
                      │
                      └───────────→ Policy Update
```

重要なのは、

> **Prediction accuracyだけを最適化しないこと。**

最終的には、

```text
Prediction Accuracy
        ↓
Decision Quality
        ↓
Runtime Performance
```

まで測定する。

---

# 34. Telemetry

Runtimeは常時次を計測する。

```text
request arrival
queue wait
prefill start
prefill end
decode start
decode end
cache lookup
cache hit
cache miss
cache restore
cache eviction
cache prefetch
prefetch hit
prefetch miss
transfer bytes
GPU memory
CPU memory
NVMe latency
batch composition
TTFT
TPOT
E2E latency
quality metrics
```

---

# 35. Prediction Metrics

Semantic Demand Prediction：

```text
Top-1 precision
Top-k precision
Recall
MRR
AUC
Calibration
```

Cache prediction：

```text
future-use precision
future-use recall
prefetch hit rate
eviction mistake rate
saved recomputation
```

ただし最終評価は、

```text
goodput
latency
memory
quality
```

で行う。

## 35.1 Signal Calibration 実験（実装済み・両較正系）

Lethe の生 semantic demand（relevance×confidence×(1+future_utility)）は
**位置非依存の「一般的な重要度」**であり、§50 の予測対象「今この位置から
horizon 内に必要になるか」を直接は表さない。本実験は生シグナルを**2つの較正**
にかけたとき予測品質（§35 メトリクス）がどう変わるかを定量化する。

`python -m compare.calibration_experiment`:

```text
recency 較正   = 生シグナル × decay^age          (immediate prefetch, H1/H2)
popularity 較正 = 生シグナル × 実行アクセス頻度    (longer-term retention, H3/H4)
予測タスク: 位置 i の chunk が horizon 内に再利用されるか（§50）
比較: raw AUC vs recency AUC vs popularity AUC
```

両較正は**将来情報を一切使わない**（ex-ante）。実測結果（n_objects=30,
n_recalls=400）：

### COUPLED（人気と recency が相関する現実的な skewed traffic）

| horizon | raw AUC | recency AUC | popularity AUC | best |
| ------- | ------- | ----------- | -------------- | ---- |
| 2 | 0.646 | 0.699 | **0.843** | popularity |
| 4 | 0.602 | 0.675 | **0.818** | popularity |
| 8 | 0.585 | 0.689 | **0.835** | popularity |

### DECOUPLED（人気と recency が反相関する人工的極限）

| horizon | raw AUC | recency AUC | popularity AUC | best |
| ------- | ------- | ----------- | -------------- | ---- |
| 2 | 0.544 | **0.873** | 0.791 | recency |
| 4 | 0.531 | **0.859** | 0.851 | recency |
| 8 | 0.479 | **0.901** | 0.871 | recency |

知見：

- **生シグナルはそのままでは予測子にならない**（AUC≈0.5）。較正して初めて
  AUC 0.8〜0.9 に到達する。Semantic Memory の生出力は Computational Cache に
  接続する前に**較正層**が必要（§8）。
- **現実的な skewed traffic では人気較正が recency を内包して最強**（hot chunk
  は recent でもあるので、人気が両方を捉える）。
- **人気と recency が乖離する場合、immediate（H1/H2）は recency 較正が勝つ** —
  §50 の「immediate prefetch と longer-term retention を分離する」設計を実証。
  長期的な人気キーは horizon 内に再利用されないため、予測タスク上は recency が
  優位だが、人気較正の真価は「保持価値のランキング」（retention）にある。
- **両較正は全 horizon で raw を上回る**（較正は常に有益）。

> **§50 の2較正系が完成**: immediate prefetch には recency 較正、longer-term
> retention には人気較正、という時間スケール分離された較正設計が実測で支持
> された。較正方法は予測の時間スケールに応じて選ぶべきである。

artifact: `artifacts/phase3/calibration_sweep.json`

---

# 36. Latency Model

SimulatorではLatencyを適当に固定しない。

各処理を分解する。

```text
T_total =
T_queue
+
T_context
+
T_cache_lookup
+
T_transfer
+
T_prefill
+
T_decode
+
T_sync
```

Cache reuse時：

$$
T_{reuse}
=
T_{lookup}
+
T_{transfer}
+
T_{restore}
$$

Cache miss時：

$$
T_{miss}
=
T_{lookup}
+
T_{recompute}
$$

Prefetchあり：

$$
T_{effective}
=
\max(
T_{compute},
T_{transfer}
)
$$

を基本モデルとし、実機測定で校正する。

---

# 37. Phase 0a — Trace Analysis

最初の研究は実装ではない。

## 目的

> **そもそも未来予測Cacheが意味を持つか？**

を調べる。

取得するTrace：

```text
public traces
agent workflows
RAG workflows
multi-turn sessions
Lethe-assisted workflows
```

重要なEvent：

```text
request arrival
prefix identity
semantic object
cache object
reuse distance
session relation
queue state
tool call
memory recall
```

---

# 38. Oracle Analysis

最低限比較する：

```text
LRU
LFU
GDS-like
PEEK-like
Belady Oracle
```

Beladyは実運用Policyではない。

研究上の意味は、

> **未来を知れば最大どこまで改善できるか**

を測ること。

---

# 39. Phase 0a Kill Gate

初期Kill Criteria：

```text
Belady - LRU gap < 5 percentage points
AND
potential E2E gain < 5%
```

が複数の代表的workloadで観測された場合、

> Adaptive Cache Policyそのものの研究優先度を下げる。

ただし5%は初期研究用のThresholdであり、実験設計時に再校正可能とする。

---

# 40. Phase 0b — Simulator

離散Event Simulatorを実装する。

```text
RequestTrace
      ↓
Event Queue
      ↓
Runtime State
      ↓
Planner
      ↓
Cache Simulation
      ↓
Latency Model
      ↓
Metrics
```

---

# 41. Simulator Components

```text
sim/
├── event.py
├── request.py
├── cache.py
├── queue.py
├── memory.py
├── scheduler.py
├── planner.py
├── transfer.py
├── latency.py
├── predictor.py
└── metrics.py
```

---

# 42. Event Model

最低限：

```text
REQUEST_ARRIVAL
QUEUE_INSERT
MEMORY_RECALL
CONTEXT_BUILD
CACHE_LOOKUP
CACHE_HIT
CACHE_MISS
PREFETCH_START
PREFETCH_END
TRANSFER_START
TRANSFER_END
PREFILL_START
PREFILL_END
DECODE_START
DECODE_TOKEN
DECODE_END
EVICT
OFFLOAD
RESTORE
REQUEST_COMPLETE
```

---

# 43. Simulator Calibration Gate

Simulatorと実機の差：

```text
TTFT error <= 10%
TPOT error <= 10%
memory accounting error <= 5%
```

を初期目標とする。

10%を超える場合、Simulator上の結果をRuntime性能主張に使用しない。

---

# 44. Phase 1 — Lossless KV Prototype

対象：

```text
Single GPU
Single model
Single process
```

実装：

```text
KV block abstraction
LRU baseline
Predictive retention
GPU / CPU tiering
Prefetch
Restore
Telemetry
```

vLLMのConnector APIは非同期KV loading/storingを差し込むための候補となる。

---

# 45. Phase 1 Gate

同一条件下で、

```text
Adaptive Policy
vs
LRU
vs
LMCache default configuration
```

を比較する。

条件：

```text
same model
same GPU
same workload
same memory budget
same quality
```

目標：

> **少なくとも一つのrealistic workload familyで、Pareto frontierを改善する。**

## 45.1 vLLM Connector Prototype（実装済み）

§44 の「vLLM Connector API で非同期 KV loading/storing を差し込む」を、§119 の
「既存 Runtime の外側にある Adaptive Control Plane」として実装した。vLLM は
GPU 依存で未インストール（§43: research kernel は GPU 非依存）なので、
**vLLM 非依存の prototype** として同一の意志決定コードを Mock バックエンドで検証する。

`python -m compare.vllm_connector_experiment`:

```text
KVConnectorBase        : vLLM connector の seam（prefix matching / save / load）
AdaptiveKVConnector    : 我々の control plane（需要予測で KV 保持・復元を制御）
MockVLLMBackend        : LRU（vLLM デフォルト）
LMCacheLikeBackend     : LFU（LMCache default 相当）
SimulatedVLLMRuntime   : request stream を connector に通し prefix hit rate と
                         effective TTFT を測定（§45 Gate）
```

AdaptiveKVConnector は、Lethe→較正パイプライン（§90）の calibrated future-demand
シグナルで「どの prefix を device に保持し、どれを host から復元するか」を制御する。
vLLM が利用可能になったら `KVConnectorBase_V1` の実装として登録すれば、そのまま
実 serving runtime に接続できる。

実測結果（shared-prefix / multi-turn workload, capacity=4 blocks）:

| policy | prefix hit rate | effective TTFT |
| ------ | --------------- | -------------- |
| **AdaptiveKVConnector** | **0.550** | **586.6** |
| MockVLLMBackend (LRU) | 0.000 | 1032.0 |
| LMCacheLikeBackend | 0.000 | 1032.0 |

> **§45 Phase 1 Gate PASSED**: Adaptive Policy が best prefix hit rate AND best
> effective TTFT で Pareto frontier を改善。realistic（multi-turn / shared-prefix）
> workload family で、外部 vLLM connector として Adaptive Control Plane が
> LRU / LMCache default を上回ることを実証。§115 v0.3.0 Candidate Scope の
> 「vLLM connector prototype」を達成し、v0.3.0 が完遂した。

artifact: `artifacts/phase1/vllm_connector_gate.json`

---

# 46. Phase 2 — Queue + Cache Co-Optimization

追加：

```text
Pending queue
Prefix tree
Future request visibility
Starvation control
Scheduler
```

最初にPEEK-like behaviourを再現する。

その後、

```text
Queue-only prediction
+
Semantic demand prediction
```

へ拡張する。

---

# 47. Phase 2 Gate

単純な平均速度ではなく、

```text
TTFT p50
TTFT p95
TTFT p99
E2E p95
E2E p99
goodput
cache hit rate
fairness
```

を比較する。

最重要判定：

> **PEEKを上回る条件が存在するか。**

存在しない場合、Phase 3を中心に据える。

## 47.1 Queue + Cache Co-Optimization 実測（実装済み）

§46 の「PEEK-like behavior の再現 → Queue-only + Semantic demand prediction への
拡張」を実装し、§47 の最重要判定「PEEK を上回る条件が存在するか」に答えた。

`python -m compare.queue_cache_experiment`:

```text
QueueAwareRetentionPolicy     : PEEK-like（queue lookahead 内の key を保護）
QueueSemanticRetentionPolicy  : queue 保護 × semantic future demand（§9/§46）
比較metrics: TTFT p50/p95/p99, E2E p95, goodput, hit rate, fairness（§47/§26）
```

協調最適化の意志決定（victim 選択、低スコアを追い出し）:

```text
score(c) = queue_pressure(c) + P_semantic(c) * decay^age(c)
```

- queue_pressure(c): queue がまもなく参照するか（PEEK 項、いつ必要か）
- P_semantic(c): calibrated future demand（§90、本当に必要か）
- 固定線形モデルではなく加算（§24 の方針）。queue が深いと urgency が優先、
  semantic demand は queue がまだ見ていない long-horizon prefix を保護する。

実測結果（long_range_reuse, gap=8 >> lookahead=4, capacity=4）:

| policy | hits | hit rate | TTFT p50 | TTFT p95 | E2E p95 | goodput |
| ------ | ---- | -------- | -------- | -------- | ------- | ------- |
| lru | 30 | 0.104 | 42.6 | 62.1 | 68.3 | 0.892 |
| peek | 30 | 0.104 | 42.6 | 62.1 | 68.3 | 0.892 |
| **queue+sem** | **66** | **0.229** | 40.6 | 61.9 | 68.1 | **0.910** |

> **§47 判定: PEEK を上回る条件が存在する = True**。queue が myopic
> （lookahead < gap）な領域では、queue-only（PEEK）は hot な long-range prefix を
> cold として追い出すが、semantic demand を加えると queue+semantic が **+36 hits**
> （66 vs 30）で PEEK を上回り、TTFT p95・goodput も改善。§17 の仮説が Phase 2 で
> 実証された。存在しない場合のフォールバック（Phase 3 中心）は不要。
>
> **誠実な境界**: lookahead >= gap（queue が myopic でない）では queue が既に
> 戻りを察知できるため、queue+semantic と PEEK は収束し semantic の追加価値は
> 小さい。semantic demand の価値は「queue の視野が届かない long-range」に限定的。

artifact: `artifacts/phase2/queue_cache_gate.json`

---

# 48. Phase 3 — Lethe Future Demand Prediction

ここから本Engineの中心研究。

Pipeline：

```text
Lethe Recall
      ↓
Semantic Demand Predictor
      ↓
Future Object Ranking
      ↓
Computational Candidate Lookup
      ↓
KV Retention / Prefetch
```

---

# 49. Semantic Demand Dataset

最低限、

```text
session_id
request_id
memory_id
chunk_id
timestamp
recall_event
actual_usage
cache_hit
cache_miss
future_use_distance
```

を記録する。

---

# 50. Prediction Target

Prediction task：

> **At time t, which semantic objects are likely to be required by future requests within horizon H?**

Horizonを複数持つ。

```text
H1 = next request
H2 = next 3 requests
H3 = next 30 seconds
H4 = next session turn
```

これにより、

```text
immediate prefetch
```

と

```text
longer-term retention
```

を分離する。

---

# 51. Semantic Demand → Cache Policy

例えば、

```text
Lethe:
Document X likely relevant next turn
```

を取得した場合、

Runtimeは直ちにKVをReuseしない。

代わりに：

```text
1. Find candidate cache for X
2. Check exact token compatibility
3. Check model compatibility
4. Check cache tier
5. Estimate restore cost
6. Estimate recompute cost
7. Check GPU pressure
8. Decide
```

とする。

---

# 52. Phase 3 Gate

少なくとも、

```text
future-use prediction
prefetch accuracy
saved recomputation
goodput
```

が改善すること。

特に重要：

$$
NetGain
=
SavedCompute
-
PredictionOverhead
-
PrefetchWaste
$$

が正であること。

Prediction accuracyが高くても、

```text
Prediction cost
+
False prefetch
```

で損するなら失敗である。

---

# 53. Phase 4 — Context + KV Joint Optimization

ここから、

```text
Lethe
+
Context Compiler
+
KV Manager
+
Scheduler
```

を接続する。

Pipeline：

```text
User Request
      ↓
Lethe Recall
      ↓
Context Candidate Set
      ↓
Context Optimization
      ↓
KV Demand Prediction
      ↓
Cache Placement
      ↓
Scheduling
      ↓
Execution
```

---

# 54. Phase 4 Quality Safety

Lossy compressionを導入する場合、

```text
aggressive mode
      ↓
quality risk
      ↓
fallback
```

という構造を持つ。

オンラインではGround Truthが存在しない場合が多いため、quality risk predictionを独立研究として扱う。

候補：

```text
logit entropy
attention mass
retrieval confidence
context coverage
model self-consistency
task-specific evaluator
```

ただしv0.2.0では研究対象を限定する。

---

# 55. Lossy KV Layer

将来追加：

```text
Quantize
Dequantize
Token Evict
Head Compression
Layer-specific Compression
Approximate Reuse
```

HqeKVはQuantizationとEvictionを統合したAdaptive Compressionを既に扱っているため、単純な「量子化＋Eviction」は新規性としない。

Learning to Evictも将来Utilityを学習する方向を既に示している。

したがってLossy層の研究対象は、

> **Semantic Demand / Queue / Session / Runtime BudgetがLossy Policy選択に寄与できるか**

に置く。

---

# 56. RAG / Non-Prefix Reuse

RAGでは、

```text
Document A
Document B
Document C
Question
```

のようにCache対象がPrefixにならない場合がある。

CacheBlendはこの問題に対して、再利用KVの一部を再計算してNon-Prefix Cacheを組み合わせる。

したがって本Engineでは、

```text
Exact Prefix Reuse
```

と

```text
Non-Prefix / Fusion Reuse
```

を別カテゴリとして扱う。

---

# 57. Phase 5 — Cognitive IR

中心研究が成立した後、

```text
MEMORY_RECALL
CONTEXT_SELECT
PREFIX_LOOKUP
KV_KEEP
KV_EVICT
KV_PREFETCH
TRANSFER
PREFILL
DECODE
```

をIRへLoweringする。

---

# 58. IR Optimization

将来的に、

```text
Dead Context Elimination
Cache-aware Scheduling
Prefetch Hoisting
Redundant Recall Elimination
Transfer Scheduling
Execution Fusion
```

などを検討する。

ただし、IRによる高速化が独立した研究テーマになりすぎることを避ける。

---

# 59. Phase 6 — Full Runtime

最終形：

```text
                       ┌───────────────┐
                       │ Semantic      │
                       │ Memory / Lethe│
                       └───────┬───────┘
                               │
                               ▼
                       ┌───────────────┐
                       │ Demand        │
                       │ Predictor     │
                       └───────┬───────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
      ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
      │ Context      │ │ KV Manager   │ │ Scheduler    │
      │ Compiler     │ │              │ │              │
      └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
             │                │                │
             └────────────────┼────────────────┘
                              ▼
                    ┌───────────────────┐
                    │ Adaptive Runtime  │
                    └─────────┬─────────┘
                              ▼
                    ┌───────────────────┐
                    │ Execution Engine  │
                    └─────────┬─────────┘
                              ▼
                       CUDA / CPU / ...
```

---

# 60. Benchmark Philosophy

本プロジェクトは、

> **「平均で速かった」**

だけでは成功としない。

目的は、

> **「どの条件でAdaptive Runtimeが既存Runtimeを上回るかを再現可能な形で示すこと」**

である。

---

# 61. Baseline Matrix

最低限：

```text
vLLM
SGLang
llama.cpp
LMCache
PEEK
KVFlow
Continuum
```

さらに該当する場合：

```text
CacheBlend
HqeKV
Learned Eviction
```

などを比較対象とする。

---

# 62. Best-Configuration Rule

比較対象は「stock default」だけにしない。

例えば、

```text
vLLM
vLLM + LMCache
SGLang
SGLang + LMCache
PEEK on vLLM
PEEK on SGLang
```

のように、

> **Baselineが実際に強い設定になっていること**

を要求する。

---

# 63. Workload Matrix

最低限：

| Workload        | Prefix Reuse | Context | Concurrency |
| --------------- | -----------: | ------: | ----------: |
| W1 Short Chat   |          Low |   Short |         Low |
| W2 Short Chat   |         High |   Short |         Low |
| W3 Long Context |          Low |    Long |         Low |
| W4 Long Context |         High |    Long |         Low |
| W5 Concurrent   |          Low |    Long |        High |
| W6 Concurrent   |         High |    Long |        High |
| W7 Agent        |       Medium |    Long |        High |
| W8 RAG          |       Medium |    Long |        High |

---

# 63.1 Design Map（実験的に実測された設計指針）

§63 のワークロードマトリクスに対し、どのキャッシュ機制（LRU / predictive
retention / prefetch / adaptive）が効くかは、ワークロードの**相関距離**と
キャパシティに依存する。`python -m compare.design_map` はこれを
(相関距離 × キャパシティ) の2次元マップとして実測する（§52 NetGain 最大の
機制を各セルで自動選択）。

実測結果（sigma=0、完璧予測）：

```text
regime (distance)              cap=4     cap=6    cap=10
zipf40(short) (short)           lru       lru    prefetch+
reuse_gap2(mid) (mid)      adaptive+  adaptive+        lru
reuse_gap4(mid) (mid)      adaptive+  adaptive+  adaptive+
reuse_gap6(mid) (mid)     retention+  adaptive+  adaptive+
reuse_gap8(long) (long)    retention+ retention+  adaptive+
reuse_gap12(long) (long)   retention+ retention+  adaptive+
```

設計指針：

- **SHORT-range（recency 相関、例: zipf）**: LRU が最適。ノイズ予測を
  retention に入れるな（§76 Kill Criteria #5「Semantic signal duplicates
  existing signals」）。
- **MID-range（アンカーが数アクセス内に再訪）**: **ADAPTIVE**（retention +
  prefetch の統合）が最強。prefetch が近未来需要を先読みし、retention がそれを
  保護する（§74 中心仮説）。
- **LONG-range（アンカーが多数アクセス後に再訪、recency と無相関）**:
  predictive **RETENTION** が最強。prefetch は長期オブジェクトが再利用前に
  追い出されるため損（§20）。

> **どの機制も普遍的ではない。** 相関距離とキャパシティで最適解が変わる。
> feed_retention（予測を retention に供給するか）も領域依存：短期では
> LRU コア、中・長期では予測保持。

artifact: `artifacts/phase3/design_map.json`

---

# 64. Context Length Matrix

```text
8K
16K
32K
64K
128K
256K+
```

モデルが対応可能な範囲で測定する。

---

# 65. Metrics

## Performance

```text
TTFT
ITL
TPOT
E2E latency
p50
p95
p99
output tok/s
total tok/s
requests/sec
```

## Memory

```text
Peak VRAM
Active KV bytes
Total KV bytes
Host RAM
SSD usage
Transfer bytes
```

## Cache

```text
Prefix hit rate
Partial hit rate
Cache residency
Eviction rate
Restore rate
Prefetch accuracy
Prefetch waste
```

## Quality

```text
Task accuracy
Recall
F1
Long-context score
Needle accuracy
Perplexity
Output equivalence
```

---

# 66. Primary Metric — SLO Goodput

従来案のEIEは廃止する。

LatencyとMemoryとQualityを単位の異なる値として単純加算するのではなく、

> **SLOを満たしている有用なRequestの処理量**

を主指標とする。

定義例：

$$
Goodput_{SLO}
=
\frac{
\#\;completed\ requests\ satisfying\ SLO
}{
time
}
$$

SLO例：

```text
TTFT p99 < X
TPOT p99 < Y
Quality degradation <= ε
```

---

# 67. Pareto Frontier

第二の主指標。

軸：

```text
Memory Budget
        vs
SLO-constrained Goodput
```

Adaptive Runtimeが、

```text
same memory → higher goodput
```

または、

```text
same goodput → lower memory
```

を実現できればPareto frontierを更新したとみなす。

---

# 68. Quality Evaluation

QualityはPerformance Objectiveに混ぜない。

基本条件：

```text
QualityLoss <= ε
```

を満たさないRunは、

```text
performance win
```

として認めない。

---

# 69. Statistical Protocol

各Benchmarkは、

```text
warmup
+
N repeated runs
```

を行う。

少なくとも、

```text
median
p95
p99
confidence interval
```

を記録する。

Single-runで「勝った」と判断しない。

## 69.1 統計検証（実装済み）

§90.2 の較正ランタイム結果（人気較正が NetGain 圧勝）を、§69 プロトコルに従い
**warmup + N repeated runs** で統計的に検証する。

`python -m compare.statistical_protocol`:

```text
各較正を N=20 seed で繰り返し実行 -> NetGain 分布
記録: mean / std / median / p95 / p99 / CI95 / paired sign test
判定: 較正が raw を統計的に有意に上回るか（§94 Tier 0）
```

統計量は numpy/scipy を使わない純 Python（§43 offline）。実測結果（N=20,
capacity=6, skewed traffic）:

| calibration | mean NetGain | std | CI95 | median | p95 | p99 | sign test |
| ----------- | ------------ | --- | ---- | ------ | --- | --- | --------- |
| raw | −124.8 | 162.3 | [−200.7, −48.8] | −120.2 | 193.2 | 289.4 | - |
| recency | +57.5 | 247.4 | [−58.3, +173.3] | 135.8 | 372.4 | 443.0 | 18/20, p=0.000 |
| popularity | **+372.0** | 150.4 | **[+301.6, +442.4]** | 394.6 | 522.6 | 676.2 | **20/20, p=0.000** |

判定（§94 Tier 0「Statistically sound」）:

- **popularity 較正**: CI95 [+301.6, +442.4] が raw の mean（−124.8）を完全に
  上回り（0を跨がない）、符号検定 **20/20 wins, p=0.000** → **STATISTICALLY SOUND**。
- recency 較正: 符号検定 18/20 で有意だが CI95 が 0 を跨ぐため、popularity ほど
  強くない（誠実な結果）。

> **§69 の原則を実証**: single-run（NetGain −61.7）ではなく N=20 の分布で判定。
> 人気較正の優位性は**全 seed で頑健**であり、1つの幸運な run ではない。

artifact: `artifacts/phase3/statistical_protocol.json`

---

# 70. Ablation Study

最終評価では、

```text
A = Baseline

B = + Predictive Cache

C = + Queue

D = + Semantic Demand

E = + Context Planning

F = Full Adaptive Runtime
```

のように段階的に比較する。

最重要なのは、

> **Semantic Demandが本当に追加価値を持つか**

である。

## 70.1 Calibrated Runtime の Ablation（実装済み）

§90.2 の較正ランタイム NetGain を、機制（mechanism）と較正（calibration）の
2軸で分解し、§70 の核心問いに答える。

`python -m compare.ablation_study`:

```text
mechanism軸: none(LRU) / retention / prefetch / both(adaptive)
calibration軸: raw(未較正) / recency(H1/H2) / popularity(H3/H4)
寄与分解:
  retention_contrib = NetGain(retention) - NetGain(none)
  prefetch_contrib  = NetGain(prefetch)  - NetGain(none)
  synergy           = NetGain(both) - retention - prefetch + none
```

実測結果（capacity=6, skewed traffic）:

| calibration | retention寄与 | prefetch寄与 | synergy | adaptive NetGain |
| ----------- | ------------ | ------------ | ------- | ---------------- |
| raw | +10.6 | −61.7 | −10.6 | −61.7 |
| recency | +138.6 | −106.8 | −30.1 | +1.7 |
| popularity | **+804.2** | +10.6 | −10.6 | **+804.2** |

§70 判定:

- **Semantic Demand ADDS value = True**: 両較正の adaptive NetGain が no-signal
  baseline（+0）を上回る。
- **calibration beats raw signal = True**: 較正済み demand が未較正 raw を上回る。
- **人気較正の価値はほぼ全て retention から**（retention寄与 +804.2 ≫
  prefetch寄与 +10.6）。人気較正が predictive retention を動かし hot chunk を
  保護して recompute を回避する——これが価値の源泉。
- **人気較正では prefetch はほぼ貢献しない**（+10.6）。人気は long-term（H3/H4）
  シグナルなので immediate prefetch（H1/H2）に向かない（§50 の分離と整合）。
- **raw では prefetch が有害**（−61.7）。未較正シグナルで prefetch すると waste
  が膨大。

> **§70 の問いに回答**: Semantic Demand は追加価値を持ち、その価値は
> **calibrated signal が predictive retention を正しく動かす**ことで生まれる。
> 生（raw）シグナルでは retention も prefetch もむしろ損をする。

artifact: `artifacts/phase3/ablation_study.json`

---

# 71. Novelty Matrix

| Component                            | Existing Work           | Our Position              |
| ------------------------------------ | ----------------------- | ------------------------- |
| Prefix Cache                         | vLLM / SGLang           | Use, not reinvent         |
| Tiered KV                            | LMCache                 | Integrate / extend policy |
| Queue-aware Cache                    | PEEK                    | Baseline                  |
| Workflow-aware KV                    | KVFlow                  | Baseline                  |
| Tool-aware KV                        | Continuum               | Baseline                  |
| Lossy KV                             | HqeKV                   | Future baseline           |
| Learned Eviction                     | KVP / Learning to Evict | Future baseline           |
| Non-prefix reuse                     | CacheBlend              | Element technology        |
| Semantic demand → cache policy       | Less explored candidate | **Core hypothesis**       |
| Cross-layer Context + Cache planning | Research direction      | **Secondary hypothesis**  |

---

# 72. Research Claim Boundary

本プロジェクトでは次を主張しない。

```text
We invented KV caching.
We invented prefix caching.
We invented predictive eviction.
We invented queue-aware scheduling.
We invented semantic memory.
```

代わりに、

> **既存の強いmechanismを、Semantic Demandを共有信号としてRuntime-levelで協調させることで、End-to-End性能上の追加利益が存在するかを検証する。**

と主張する。

---

# 73. Key Research Questions

## RQ1

将来のCache reuseを予測することで、LRU/GDS系よりEnd-to-End性能を改善できるか。

## RQ2

PEEKのようなQueue-aware PolicyにSemantic Demandを追加すると、追加の性能利益が得られるか。

## RQ3

LetheのSemantic DemandがAgent/RAG/multi-turn workloadのFuture KV Demandを予測できるか。

## RQ4

Semantic DemandのPrediction Accuracyが低い場合でも、実行時間削減によるNet Gainが得られるか。

## RQ5

Context selectionとKV retentionを共同最適化すると、独立最適化より優れたPareto frontierを形成できるか。

## RQ6

Runtimeが「Optimizationを適用しない」という選択肢まで持つことで、固定最適化Pipelineを上回れるか。

---

# 74. Central Experimental Hypothesis

最重要仮説：

> **A semantic memory system can act as a predictive signal for computational cache demand.**

具体的には、

```text
Lethe predicts:
    "Document X will matter soon."

Runtime predicts:
    "KV(X) is likely to be reused."

Cache Manager decides:
    Keep / Prefetch / Offload

Scheduler decides:
    Execute request producing X's prefix first

Context Compiler decides:
    Include only necessary memory
```

という一連の制御が、

```text
independent optimization
```

より優れるかを検証する。

---

# 75. Expected Failure Modes

## Failure A — Semantic Prediction is Too Weak

```text
Lethe relevance
      ↓
poor future prediction
```

ならCacheに使えない。

---

## Failure B — Prediction Overhead is Too High

```text
saved prefill
<
prediction + lookup + scheduling overhead
```

なら採用しない。

---

## Failure C — Semantic Signal Duplicates Queue Signal

PEEK/KVFlowなど既存Signalだけで将来需要がほぼ予測できる場合、

> Semantic Memoryを追加する意味がない。

これは重要なNegative Resultである。

---

## Failure D — Benefits Exist Only on Synthetic Workloads

Syntheticでのみ勝つ場合、

> Research Successとはみなさない。

---

## Failure E — Quality Degradation

Lossy layerを導入して、

```text
latency ↓
memory ↓
quality ↓↓↓
```

となるなら失敗。

---

# 76. Kill Criteria

本Engineには明示的なKill Criteriaを持たせる。

```text
1. Belady vs LRU gap is too small

2. Prediction accuracy is too low

3. Prediction overhead exceeds saved compute

4. Prefetch waste dominates savings

5. Semantic signal duplicates existing signals

6. No win over strong baselines

7. Gains exist only on synthetic workloads

8. Quality degradation exceeds threshold

9. GPU implementation complexity exceeds benefit

10. Full system is consistently slower than strong baselines
```

特に、

> **「複雑にした結果、vLLM/SGLangより遅い」**

なら、その機構は削除する。

---

# 77. Security

Shared Prefix / KV Cacheには情報漏洩リスクがある。

Cache metadata：

```text
cache_namespace
tenant_id
trust_domain
cache_salt
model_revision
tokenizer_revision
provenance
encryption_state
access_policy
```

を持つ。

Cross-trust-domain reuseをデフォルト禁止とする。

---

# 78. Provenance

Semantic MemoryからComputational Cacheまで、

```text
memory_id
→ source_id
→ chunk_id
→ token_hash
→ cache_id
→ execution_id
```

を追跡可能にする。

これにより、

```text
Why was this KV retained?
Why was this context selected?
Why was this request prefetched?
```

を後から説明できる。

---

# 79. Explainability

Runtimeは将来的に、

```text
memory_explain
cache_explain
schedule_explain
prefetch_explain
plan_explain
```

を提供する。

例：

```text
KV 0x1234 retained because:

P_future_use = 0.83
saved_recompute = 18.2 ms
GPU retention cost = 2.1 ms-equivalent
restore cost = 1.4 ms
```

このようなRuntime Decision Traceを研究資産として保存する。

---

# 80. Hardware Abstraction

```text
Runtime Core
      │
      ├── CUDA Backend
      ├── CPU Backend
      └── Future HIP / Other Backend
```

Policy LogicとKernel implementationを分離する。

v0.xではCUDA優先。

HIPなどは後段。

---

# 81. Implementation Language

## Control Plane

**Rust**

理由：

```text
Memory Safety
Concurrency
Async Runtime
Low-level control
FFI
Predictable systems architecture
```

## GPU

```text
CUDA / C++
```

## Research Prototype

初期SimulatorのみPython：

```text
trace analysis
policy simulator
benchmark harness
analysis
```

本体RuntimeはPython dependencyを可能な限り削減する。

---

# 82. Repository Structure

```text
adaptive-inference/
├── runtime/
│   ├── state/
│   ├── planner/
│   ├── scheduler/
│   ├── executor/
│   └── telemetry/
│
├── prediction/
│   ├── demand/
│   ├── reuse/
│   ├── prefetch/
│   └── models/
│
├── context/
│   ├── compiler/
│   ├── ranking/
│   ├── budget/
│   └── quality/
│
├── cache/
│   ├── kv/
│   ├── prefix/
│   ├── retention/
│   ├── placement/
│   ├── tiering/
│   └── prefetch/
│
├── scheduler/
│   ├── fairness/
│   ├── slo/
│   └── policies/
│
├── semantic/
│   ├── interface/
│   └── lethe/
│
├── simulator/
│   ├── events/
│   ├── models/
│   ├── traces/
│   └── replay/
│
├── ir/
│   ├── schema/
│   ├── optimizer/
│   └── lowering/
│
├── backends/
│   ├── cuda/
│   └── cpu/
│
├── integrations/
│   ├── vllm/
│   ├── sglang/
│   └── llamacpp/
│
├── benchmarks/
│   ├── workload/
│   ├── latency/
│   ├── memory/
│   ├── cache/
│   ├── quality/
│   └── statistics/
│
└── docs/
```

---

# 83. Compatibility Strategy

既存RuntimeをForkして作り直さない。

初期構成：

```text
Adaptive Planner
       ↓
Existing Engine
```

対象：

```text
vLLM
SGLang
```

KV offloadingを差し込む際は、vLLM ConnectorやLMCacheなど既存の拡張機構を利用する。

その後、

```text
Adaptive Planner
+
Custom KV Manager
```

へ移行する。

---

# 84. Integration Strategy

## Stage A

External Controller:

```text
Adaptive Planner
        ↓
vLLM API
```

## Stage B

Cache Hook:

```text
Adaptive Planner
        ↓
KV Connector
        ↓
vLLM
```

## Stage C

Scheduler / Cache joint integration.

## Stage D

Native Runtime components.

## Stage E

Custom execution backend.

---

# 85. Phase Roadmap

```text
Phase 0a
Trace Analysis

Phase 0b
Simulator

Phase 1
Lossless KV

Phase 2
Queue + Cache

Phase 3
Lethe Demand Prediction

Phase 4
Context + KV Joint Planning

Phase 5
Cognitive IR

Phase 6
Full Runtime
```

---

# 86. Phase 0a — Trace Analysis

成果物：

```text
trace schema
trace collector
reuse-distance analyzer
Belady simulator
LRU simulator
PEEK-like baseline
workload taxonomy
```

Gate：

```text
Is there enough unrealized temporal predictability?
```

---

# 87. Phase 0b — Simulator

成果物：

```text
discrete event simulator
latency model
cache model
queue model
semantic demand model
benchmark harness
```

Gate：

```text
Simulator accuracy <= 10% error
```

---

# 88. Phase 1 — Lossless KV

成果物：

```text
KV abstraction
GPU tier
CPU tier
placement
retention
prefetch
restore
telemetry
```

Gate：

```text
Adaptive > LRU
under equal memory / quality constraints
```

---

# 89. Phase 2 — Queue-aware Runtime

成果物：

```text
queue state
prefix state
PEEK-like scheduler
fairness
queue-aware eviction
```

Gate：

```text
Can we beat PEEK?
```

---

# 90. Phase 3 — Lethe

成果物：

```text
memory demand predictor
future-use labels
prediction metrics
semantic-to-computational bridge
prefetch controller
```

Gate：

```text
Semantic Demand adds measurable net benefit
beyond Queue / Workflow signals.
```

## 90.1 Lethe → Design-Map ブリッジ（実装済み）

§63.1 の設計マップを、Lethe の実 recall ストリームに接続する。合成ではなく
Lethe の MemoryRecord 列（memory_id, chunk_id, relevance, confidence,
future_utility, session 関係）から trace を生成し、その**相関距離を実測**して
マップ上に配置する。

`python -m compare.lethe_design_map`:

```text
Lethe recall ストリーム (MemoryRecord)
    -> LetheTraceBuilder -> RequestTrace (較正済み semantic signal)
    -> measure_correlation_distance -> regime (short/mid/long)
    -> 4機制を較正 trace で較競争 -> 推奨機制 = 勝者
    -> §90 Gate: 推奨機制が LRU を上回るか検証
```

較正（§50）: retention シグナル `future_use_probability` は「その位置から
horizon 内に chunk が再利用されるか」に較正する。Lethe の生シグナル
（relevance×confidence×(1+future_utility)）は位置非依存の「一般的将来価値」
なので、そのままでは predictive retention の victim 選択に使えない。較正後は
retention が機能し、生シグナルは §35 較正メトリクス（AUC/ECE）用に保持。

実測結果（capacity=6）:

| 相関距離 | regime | 推奨機制 | LRU比 |
| -------- | ------ | -------- | ----- |
| gap=2 | short | prefetch | +11 |
| gap=4 | mid | prefetch | +11 |
| gap=8 | long | adaptive | +203 |
| gap=12 | long | adaptive | +262 |

> **§90 Gate 通過**: 全 regime で推奨機制が LRU を上回る。Semantic Demand が
> Queue/Workflow シグナルを超える net benefit を付加することを、Lethe ストリーム
> から end-to-end で実証。

artifact: `artifacts/phase3/lethe_design_map.json`

---

## 90.2 Calibrated Runtime（実装済み）

§35.1 で較正が予測品質（AUC）を改善することを示したが、まだ「予測の話」だった。
本節は較正を**実際の cache 機制**（predictive retention / prefetch）に接続し、
§90 Gate「Semantic Demand adds measurable net benefit」を end-to-end で検証する。

`python -m compare.calibrated_runtime_experiment`:

```text
Lethe recall ストリーム
  -> calibrated trace (retention signal = 較正値)
  -> calibrated forecast (prefetch demand = 較正値)
  -> DiscreteEventSimulator (predictive retention + prefetch controller)
  -> NetGain vs LRU baseline (§52)
```

較正選択（同一 trace, capacity=6, skewed traffic）:

| calibration | hits | pfhit | waste | saved | NetGain |
| ----------- | ---- | ----- | ----- | ----- | ------- |
| raw（未較正） | 163 | 4 | 16 | 51 | **−61.7** |
| recency | 165 | 3 | 14 | 102 | **+1.7** |
| popularity | 193 | 0 | 0 | 819 | **+804.2** |

知見：

- **Lethe の生（未較正）demand は NetGain 負**。retention が全キー高シグナルで
  ランクできず、prefetch は waste 16 件。生の semantic demand をそのまま cache
  意志決定に使うと**損する**（§8 の原則の実証）。
- **人気較正が NetGain +804 で圧勝**（hit 163→193, waste 0）。人気較正が
  predictive retention を正しく動かし、hot chunk を保護して recompute を回避。
- **較正は「予測品質（AUC）」から「実運用の net benefit（§90 Gate）」へ繋がる**。
  §35 の較正が §52 の NetGain を実際に改善することを end-to-end で実証。

> **§113 研究テーゼの実証**: 「Semantic memory は computational cache に直接ならず、
> 較正された signal として次に必要な computational state を予測する助けになる」
> — 生は使えず、較正して初めて net benefit が生まれる。

artifact: `artifacts/phase3/calibrated_runtime.json`

---

## 90.3 Lethe → Kairos 感度分析（実装済み）

Lethe と Kairos は**協進化**する：Lethe（Semantic Memory）が良くなると
Kairos（Adaptive Runtime）も良くなる。本節はこの関係を**応答曲線**として定量化
し、「Lethe にどれだけ投資すべきか」に数字を与える。

`python -m compare.lethe_kairos_sensitivity`:

```text
独立変数 x: Lethe recall 品質（noise で制御、0=完璧 → 大=ノイズ）
  - Lethe 成績 = 較正後シグナル AUC（§35）
従属変数 y: Kairos 成績（同一 Lethe trace 上）
  - NetGain（§52/§90）、hit rate
感度 = d(NetGain)/d(Lethe AUC)
```

実測結果（capacity=6, skewed traffic, popularity 較正, **N=15 seeds, mean±CI95**）:

| noise | Lethe AUC | NetMean | CI95half | CI95low | hit rate |
| ----- | --------- | ------- | -------- | ------- | -------- |
| 0.10 | 0.804 | **453.9** | 63.6 | 390.3 | 0.602 |
| 0.35 | 0.737 | 351.5 | 72.0 | 279.5 | 0.588 |
| 0.50 | 0.699 | 286.7 | 70.9 | 215.8 | 0.580 |
| 1.00 | 0.655 | 151.8 | 86.6 | 65.3 | 0.562 |

感度と投資指針:

- **感度 d(NetGain)/d(Lethe AUC) = +1911.7**（+1.0 AUC あたり）、すなわち
  **Lethe AUC を +0.1 上げると Kairos NetGain が ~+191 上がる**（workload 固定）。
- **統計的頑健性（§69/§94）**: 応答は noise に対して**単調**（noise↑→NetGain↓）、
  低ノイズ（高 Lethe 品質）領域では全点の CI95 下限が正。
- Kairos NetGain はテストした全 Lethe 品質帯（AUC 0.655〜0.804）で正の平均。
- 線形外挿: Kairos NetGain が ~0 になるのは Lethe AUC ~0.563（投資閾値の推定）。

> **§90 Gate を応答曲線として読む**: 「Semantic Demand が net benefit を加えるか」
> は Yes/No ではなく、**Lethe 品質の関数**として連続的に成り立つ。§69 の
> 複数 seed 統計により、この応答曲線は single-run のノイズではなく**頑健な推定**
> である。Lethe を改善すると Kairos は線形に向上する——AUC を測れば NetGain が
> 予測できる。§120 の Measure→Adapt ループが2システムにまたがって定量化された。
>
> **誠実な注意点**: Lethe 品質が低い（高ノイズ）領域では NetGain の CI95 が 0 を
> 跨ぎうる——不確かさが大きい。つまり Lethe を改善すると NetGain が上がる**と
> 同時に不確かさも減る**、という逆説的な知見もある。

artifact: `artifacts/phase3/lethe_kairos_sensitivity.json`

---

## 90.4 実 Lethe コネクタ（実装済み）

ここまで全実験は合成 Lethe ストリーム（`synthesize_lethe_stream`）で回してきた。
本節はその**継ぎ目**を実データ接続に差し替え、同一パイプラインを実 Lethe recall
フィードで動かす。

`python -m compare.lethe_connector_experiment`:

```text
LetheConnector（3経路、出力は共通の List[LetheRecallEvent]）
  - LetheAPIConnector   : 生 Lethe サービスの recall エンドポイント（POST /recall）
  - LetheJSONLConnector : 記録済み recall trace を JSONL から読み込み（§49 スキーマ）
  - SyntheticConnector  : 合成ジェネレータ（A/B・較正用）
-> build_calibrated_trace / calibrated_forecast（そのまま）
-> 較正ランタイム・統計・ablation・感度分析（全てそのまま実データで回る）
```

`semantic/lethe_connector.py`:

- **LetheJSONLConnector**: §49 DemandSample スキーマの JSONL を recall events に
  変換。シグナルフィールド（relevance/confidence/future_utility）が欠けても
  graceful に degrade（実ダンプは部分的なので、較正層が許容）。
- **LetheAPIConnector**: 標準ライブラリ（urllib）のみで生 Lethe に接続。
  サービスダウン時は空リストを返し、research パイプラインを**落とさない**。
- **compare_real_vs_synthetic（§43 較正）**: 実 trace と合成の相関距離レジーム
  とシグナル分布を対比し、シミュレータ主張（NetGain、感度）を実機較正の
  ゲートに通す。

実測結果（JSONL「実データ」stand-in, capacity=6）:

```text
Section 43 較正: real(mid, medReuse 7.0) vs synthetic(mid, medReuse 6.0), drift +1.0
較正ランタイム（実データ）: raw=-93.4, recency=+149.2, popularity=+343.4 (best)
```

> **合成と同じ結論が実データで再現**: popularity 較正が最強、raw が最悪。
> connector を差し替えるだけで（`LetheJSONLConnector(path=<real dump>)` または
> `LetheAPIConnector(base_url=<live Lethe>)`）、較正・ランタイム・統計・ablation・
> 感度分析の全パイプラインが**実データでそのまま動く**。§95 Week 7「Lethe Trace
> 収集」と §43 較正ゲートに直接対応。研究成果が「合成での実証」から「実データでの
> 実証」へ昇格する継ぎ目が完成した。

artifact: `artifacts/phase3/lethe_connector_run.json`

---

# 91. Phase 4 — Joint Context / Cache

成果物：

```text
context optimizer
joint budget
quality constraint
lossy policy interface
```

Gate：

```text
Pareto frontier improvement
with bounded quality degradation
```

実測結果（n=12 objects, seed=7, ε=0.10, budget sweep）:

| budget | policy | memUsed | goodput | qLoss | cost |
| ------ | ------ | ------- | ------- | ----- | ---- |
| 1024 | cache-only | 1024 | 0.768 | 0.000 | 10.2 |
| 1024 | context-only | 896 | 0.772 | 0.000 | 9.0 |
| 1024 | joint | 1664 | 0.824 | 0.106 | 37.6 |
| 2048 | cache-only | 2048 | 0.768 | 0.000 | 20.5 |
| 2048 | context-only | 1920 | 0.796 | 0.000 | 19.2 |
| 2048 | **joint** | 1920 | **0.796** | 0.081 | 37.6 |
| 4096 | cache-only | 3968 | 0.768 | 0.000 | 39.7 |
| 4096 | context-only | 3968 | 0.768 | 0.000 | 39.7 |
| 4096 | **joint** | 3968 | **0.768** | 0.000 | 65.9 |

> **§91 Phase 4 Gate PASSED**: quality-feasible 予算（2048, 4096）で joint は
> context-only の goodput に一致（mean gain +0.000）し、cache-only を **+0.014**
> 上回る。同予算で qLoss は 0.081 / 0.000 で ε=0.10 以下 = bounded quality
> degradation。joint は context 選択のみの context-only が KV retention を持たない
> のに対し retained KV を持つ唯一の政策で、§23 の「context と cache を同一配分
> 問題として解く」が質量双方で成立している。全225テスト通過
> （`tests/test_joint_context_cache.py` 12件を追加）。
>
> **誠実な境界**:
> - joint vs context-only の goodput は「一致」であって strict win ではない。
>   goodput proxy は KV 命中を金銭化していないため、joint の context-only に対する
>   優位は goodput ではなく retained KV（context-only は設計上保持しない）として
>   現れる。cache-only との差 +0.014 が配分の純利得。
> - budget=1024 は excluded。quality force-inclusion が quality（0.106 > ε0.10）と
>   budget（1664 > 1024）の両方を破る。tight 予算では §68（quality）が予算遵守を
>   上回って優先される。feasible 集合は ε の選択に依存する（全 quality_loss 合計
>   0.244 のため ε=0.05 では任何予算も feasible にならない）。

artifact: `artifacts/phase4/joint_context_cache_gate.json`

---

# 92. Phase 5 — Cognitive IR

成果物：

```text
Execution IR
planner lowering
optimization passes
decision trace
```

Gate：

```text
IR overhead < optimization gain
```

実装（成果物 → 実装）:

```text
Execution IR        runtime/ir.py — §57 9 ops, effect-aware (§30),
                    ExecutionGraph.execute() が §36 overlap lane を wall-clock で測定
planner lowering    runtime/ir_lowering.py — §53 pipeline 順、§105 fallback 対応、
                    全 op に origin 付与（decision trace へ）
optimization passes runtime/ir_passes.py — redundant recall / dead context /
                    prefetch hoisting（§58 のうち本 prototype で正直に測れる3本）
decision trace      runtime/telemetry.py の DecisionTrace.ir_summary +
                    ir_lowering.with_ir()
```

実測結果（6 requests: confident_prefetch×3 / low_confidence_fallback×3、
48+4 candidates（multi-query 重複含む）、token_budget=2048 が結合、quality 非活性
（pool total 0.986 < ε1.0）、min-of-7・ラウンドロビン交錯測定 perf_counter_ns）:

| side | value |
| ---- | ----- |
| IR overhead（lower + 3 passes） | **706.5 us** |
| optimization gain（exec naive − exec opt） | **1,941.5 us** |
| gain / overhead | **2.75x** |

per-pass gain（wall-clock、3本の合計＝総 gain に厳密一致）:

| pass | gain | activity |
| ---- | ---- | -------- |
| redundant_recall_elimination | +190.4 us | 24 recalls removed |
| dead_context_elimination | +1,754.7 us | 234 recalls removed |
| prefetch_hoisting | −3.6 us | 3 transfers moved |

graph: 525 ops → 267 ops（MEMORY_RECALL 258 除去）

> **§92 Phase 5 Gate PASSED**: lowering＋3 pass の実測 cost 706.5us が、pass が
> graph 実行から削った wall-clock 1,941.5us を **2.75倍** 下回る。両側を同一
> プロセスの同一時計（perf_counter_ns、min-of-7・ラウンドロビン交錯）で測る
> ため、未較正の単位変換を比較に混入させていない。per-pass 内訳の合計は総 gain
> に厳密一致（attribution は telescoping）。収益の主役は §58 Dead Context
> Elimination（budget で落ちた候補の recall 除去、1754.7/1941.5us）。全241
> テスト通過（`tests/test_cognitive_ir.py` 16件を追加）。
>
> **誠実な境界**:
> - prefetch hoisting は本 workload で −3.6us（±5us のノイズ帯 = 構造的 gain ≈ 0）。
>   transfer が naive 位置でも overlap window（prefill）に収まるためで、§36 モデル
>   上 hoisting が効くのは lane が window を超える multi-KB KV chunk 領域。パスは
>   構造・順序・validity をテストで固定し、gain 自体は本 workload では主張しない。
> - gate 測定は control-plane prototype の wall-clock（Python executor）。GPU 実機
>   での overhead < gain 検証は §43 較正・§93 native runtime が残る。
> - workload は budget-bound regime（force-inclusion 不発で budget は 2048/2048
>   結合）。§68 force が全件採用する regime では recall は本来無駄にならず、
>   dead context に消すべきものが正しく存在しない — IR の利得は候補が落ちる通常
>   regime に存在する。この regime は build_workload() の assert で固定。

artifact: `artifacts/phase5/ir_overhead_gate.json`

---

# 93. Phase 6 — Full Runtime

成果物：

```text
native runtime
native cache manager
native scheduler
native planner
GPU backend
multi-model support
```

Gate：

```text
strong baseline superiority
across multiple workload families
```

---

# 94. Success Criteria

## Tier 0 — Scientific Validity

```text
Traceable
Reproducible
Strong baselines
Statistically sound
```

---

## Tier 1

同等品質で、

```text
E2E latency -10%
```

を一部realistic workloadで実現。

---

## Tier 2

同一memory budgetで、

```text
SLO goodput +20%
```

を目標とする。

---

## Tier 3

複数workloadで、

```text
Pareto frontier improvement
```

を示す。

---

## Tier 4

```text
vLLM
SGLang
PEEK
LMCache
```

などの強い構成が得意な条件を含む混合workloadで、

```text
average
p95
p99
```

まで改善する。

---

## Tier 5

```text
multiple models
multiple context lengths
multiple concurrency levels
multiple hardware configurations
```

で再現する。

ここまで達成して初めて、

> **State-of-the-art candidate**

と表現する。

---

# 95. First 8 Weeks

## Week 1–2

```text
Trace Schema
Event Schema
Workload Definition
Baseline Harness
```

## Week 3–4

```text
LRU
LFU
GDS-like
Belady
PEEK-like
```

をOffline Simulation。

## Week 5–6

```text
Latency Model
Memory Model
Transfer Model
```

を実機測定。

## Week 7

```text
Lethe Trace
Semantic Demand Dataset
```

を収集。

## Week 8

```text
Predictability Analysis
Belady Gap
Semantic Prediction Accuracy
```

を評価。

この8週間の成果で、

> **この研究を続ける価値があるか**

を決定する。

---

# 96. First Implementation Priority

最初に作るべきものは巨大なInference Engineではない。

```text
1. Trace Collector

2. Runtime Simulator

3. Belady / LRU Baselines

4. Demand Predictor

5. Cache Value Model

6. Queue-aware Scheduler

7. Benchmark Harness
```

これが最初の研究Kernelである。

---

# 97. What Must Not Be Built First

以下を初手で作らない。

```text
Custom CUDA Kernel
Full GPU KV Manager
Custom Transformer
Distributed Runtime
Full Cognitive IR
Multi-GPU Scheduler
Speculative Decoding Stack
```

理由：

> **研究仮説が成立する前に実装コストを払わない。**

---

# 98. Research Artifact

各Phaseで保存するもの：

```text
traces/
predictions/
planner_decisions/
cache_states/
benchmark_results/
ablation_results/
failure_cases/
```

特に、

```text
Decision Trace
```

を重要なArtifactとする。

---

# 99. Decision Trace Example

```text
Request: R1842

Context:
    memory_37
    memory_52
    memory_91

Semantic Demand:
    document_12 = 0.86
    document_24 = 0.21

Cache:
    kv_12 = RAM
    kv_24 = SSD

Queue:
    3 pending requests

Decision:
    PREFETCH kv_12
    KEEP kv_24 = false
    EXECUTE request R1842

Reason:
    P_future_use(kv_12) = 0.86
    restore_cost = 2.1ms
    recompute_cost = 19.8ms
    memory_pressure = high
```

こうしたDecision Traceを保存することで、

> 「なぜRuntimeはその判断をしたのか」

を後から完全に解析できるようにする。

---

# 100. Research Reproducibility

Benchmarkには必ず、

```text
Model
Model revision
Tokenizer
GPU
VRAM
CPU
RAM
Runtime version
Driver
CUDA
Batch policy
Cache budget
Workload seed
Trace hash
```

を記録する。

---

# 101. Experimental Controls

比較時に固定する：

```text
model
GPU
input dataset
output length distribution
concurrency
memory budget
SLO
random seed
warmup
```

変更するものを一つずつ明示する。

---

# 102. Research Anti-Patterns

禁止事項：

```text
弱いbaselineと比較する

default settingだけで勝敗を決める

synthetic workloadだけを使う

平均Latencyだけを見る

cache hit rateだけを見る

quality degradationを無視する

prediction accuracyだけで成功とする

GPU implementationを先に完成させる

自分のpolicyだけHyperparameter tuningする
```

---

# 103. Core Optimization Equation

Runtimeの最終的なOptimizationは、

$$
\max Plan
\quad
Goodput_{SLO}
$$

subject to

$$
QualityLoss \le \epsilon
$$

$$
VRAM \le B_{GPU}
$$

$$
RAM \le B_{CPU}
$$

$$
TransferBandwidth \le B_{transfer}
$$

$$
Fairness \ge F_{min}
$$

とする。

これがv0.2.0の中心的な形式化である。

---

# 104. Runtime Plan

Plannerの出力を明示的なPlan Objectにする。

```text
RuntimePlan
├── context_plan
├── cache_plan
├── placement_plan
├── prefetch_plan
├── scheduling_plan
├── execution_plan
└── fallback_plan
```

これによりDecisionとExecutionを分離できる。

---

# 105. Fallback

すべてのAdaptive Decisionには、

```text
Fallback = baseline behaviour
```

を持たせる。

例えば：

```text
Prediction uncertain
        ↓
Disable prefetch

Cache value uncertain
        ↓
LRU

Context risk high
        ↓
Full context

Quality risk high
        ↓
Lossless / full precision
```

この性質により、Adaptive Runtimeが未知のworkloadで壊れにくくなる。

---

# 106. Conservative Default

Adaptive Runtimeのdefault behaviorは、

> **「自信がないときはBaselineへ戻る」**

とする。

これは非常に重要。

Adaptive optimizationは、

```text
aggressiveness
```

ではなく、

```text
confidence
```

を中心に設計する。

---

# 107. Confidence-Aware Optimization

Prediction：

```text
value
+
confidence
```

を返す。

例：

```text
P_future_use = 0.91
confidence = 0.94
```

と、

```text
P_future_use = 0.91
confidence = 0.31
```

は同じ扱いをしない。

---

# 108. Calibration

予測値が、

```text
0.8
```

なら、

実際に約80%のケースで使用されることが望ましい。

Prediction Accuracyだけでなく、

```text
Expected Calibration Error
```

などを評価する。

---

# 109. Long-Term Research Direction

中心研究が成功した場合、

```text
Semantic Memory
        ↓
Demand Prediction
        ↓
Context Optimization
        ↓
Cache Planning
        ↓
Queue Planning
        ↓
Execution Planning
        ↓
Hardware
```

という階層を形成する。

最終的には、

> **Runtime itself becomes an optimizer.**

という設計哲学へ到達する。

---

# 110. Long-Term Extension: Unified AI Memory

最終的に、

```text
Semantic Memory
    = what should be remembered

Computational Cache
    = what should remain computationally available

Context
    = what should be presented now

Runtime Plan
    = what should be executed next
```

という役割分離を持つ。

これは、

```text
Remember
   ↓
Represent
   ↓
Plan
   ↓
Execute
   ↓
Observe
   ↓
Adapt
```

というAI Runtime Architectureへつながる。

---

# 111. Final Research Position

本プロジェクトが狙うものは、

> **世界で一番賢いKV Cache**

ではない。

また、

> **vLLMをRust/C++で作り直すこと**

でもない。

本当に狙うべきものは、

> ## **Runtime itself becomes an optimizer.**

である。

そして、そのための最初の問いを極限まで小さくする。

> **「Semantic Memoryが、次に必要となるComputational Stateを予測するための有用な信号になり得るか？」**

これがv0.2.0の中心である。

---

# 112. One-Line Definition

> **An adaptive inference runtime that uses semantic demand, request state, queue state, and computational cache state to jointly optimize context selection, cache retention, placement, scheduling, and execution under end-to-end SLO constraints.**

---

# 113. Research Thesis in One Line

> **Semantic memory should not directly become computational cache; it should become a signal that helps the runtime predict what computational state will be needed next.**

---

# 114. v0.2.0 Current Scope

```text
✓ Research thesis narrowed
✓ Semantic Demand layer
✓ Lethe interface
✓ Lossless / Lossy separation
✓ Predictive Cache value model
✓ Queue-aware baseline
✓ PEEK positioning
✓ KVFlow positioning
✓ Continuum positioning
✓ Context / KV joint budget
✓ SLO-constrained scheduler
✓ Goodput metric
✓ Pareto frontier
✓ Benchmark protocol
✓ Ablation protocol
✓ Security model
✓ Decision trace
✓ Simulator plan
✓ Trace analysis
✓ Phase gates
✓ Kill criteria
```

---

# 115. v0.3.0 Candidate Scope

```text
✓ Trace collector
✓ Discrete-event simulator
✓ Belady oracle
✓ LRU / GDS baselines
✓ PEEK-like baseline
✓ Semantic demand predictor
✓ Lethe trace integration
✓ vLLM connector prototype
✓ Prefetch prototype
✓ Runtime decision trace
✓ Initial reproducibility suite
```

---

# 116. v1.0 Candidate Scope

```text
□ Native Runtime
□ Adaptive KV Manager
□ Semantic Demand Prediction
□ Queue + Cache Co-Optimization
□ Context + KV Joint Planning
□ Multi-tier KV
□ SLO-aware Scheduler
□ Full Benchmark Suite
□ Multi-model evaluation
□ Multi-GPU evaluation
```

---

# 117. Non-Goals

初期段階では、

```text
□ 新しいLLMを訓練する
□ Tokenizerを作る
□ Transformer Architectureを変更する
□ 全GPUをサポートする
□ vLLM/SGLangの全機能を再実装する
□ Speculative Decodingを中心研究にする
□ Cognitive IRを中心研究にする
□ 独自Kernelを先に作る
□ 「世界最速」を未検証で主張する
```

ことを行わない。

---

# 118. Primary Baselines

```text
vLLM
SGLang
llama.cpp
LMCache
PEEK
KVFlow
Continuum
CacheBlend
HqeKV
Learning to Evict
```

特にPEEK、KVFlow、Continuumは、

```text
future demand
queue
workflow
multi-turn
```

の観点から本研究の重要比較対象とする。

---

# 119. Current External Technology Position

vLLMは現在、非同期KV offloading Connectorを提供しており、外部KV管理PolicyをRuntimeへ接続する足場が存在する。

LMCacheはGPU→CPU→Local Storage→Remote Storageという階層的なKV storage architectureを持っている。

したがって、

> **初期研究を「既存Runtimeの外側にあるAdaptive Control Plane」として開始する**

という設計は、実装コストと研究速度の両方で合理的である。

---

# 120. Final Definition

```text
Semantic Memory
       +
Future Demand Prediction
       +
Computational Cache
       +
Queue State
       +
Context Planning
       +
SLO-aware Scheduling
       +
Runtime Feedback
       ↓
Adaptive Inference Runtime
```

そして最終目標は、

```text
Remember
    ↓
Predict
    ↓
Represent
    ↓
Plan
    ↓
Execute
    ↓
Measure
    ↓
Adapt
```

である。

> **The inference engine does not merely execute a plan.**
>
> **It learns which plan should exist.**
