"""Phase 6 — Adaptive Runtime: the Plan.md §59 pipeline as one control loop.

    Semantic Memory / Lethe  ->  online popularity feedback (§33; demand signal
                                 computed from PAST accesses only — ex-ante)
    Demand Predictor          ->  retention signals + prefetch targets (§20)
    Context Compiler          ->  JointBudgetOptimizer (§21-23, Phase 4)
                                  lowered to IR + §58 passes (§57-58, Phase 5)
    KV Manager                ->  token-capacity KV pool + retention policy
                                  (Phase 2/3 mechanisms: LRU / predictive /
                                  PEEK-like queue-aware / queue x semantic)
    Scheduler                 ->  FIFO or SLO-deadline (EDF, §24-25)
    Adaptive Runtime          ->  AdaptiveRuntime.serve() (this module)
    Execution Engine          ->  runtime/engine.py (§36 pricing, §59/§80)

RuntimeConfig selects which mechanisms are active — that is exactly the §70
ablation ladder: "full runtime" and "baseline" are the same code with
different switches, never two implementations (a comparison built into two
codebases would not be a controlled experiment, §101).

Honest boundaries (§43 / §62):
- Latency is priced by the §36 model in abstract units. The Python control
  plane's wall clock is NOT mixed into simulated latency: §92 measured that
  the IR stage's wall cost is smaller than the execution work it removes, so
  charging both would double-count.
- All configurations share one engine and one latency model: the experiment
  isolates the value of DECISIONS, which is the research thesis (§1).
- Semantic signals are online and ex-ante: a policy never sees the future.
  The queue view contains only requests that have already arrived (§16/§46).
- Tier 4/5 physical hardware measurement (§54) is NOT claimed here (§93).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Sequence, Tuple
from runtime.context_compiler import JointBudgetOptimizer, MemoryObject
from runtime.engine import SimulatedExecutionEngine, EngineRequest
from runtime.ir import CONTEXT_SELECT
from runtime.ir_lowering import lower_plan
from runtime.ir_passes import optimize
from runtime.plan import (CachePlan, ContextPlan, ExecutionPlan, FallbackPlan,
                          PrefetchPlan, RuntimePlan, SchedulingPlan)
from runtime.state import RequestState


@dataclass(frozen=True)
class Chunk:
    """One context chunk of a request. chunk_id is stable across requests
    when the workload shares documents/tools (the reuse signal, §15)."""
    chunk_id: str
    tokens: int
    benefit: float = 0.5
    quality_loss_if_dropped: float = 0.01


@dataclass(frozen=True)
class WorkloadRequest:
    request_id: str
    arrival_time: float
    prefix_key: str
    prefix_tokens: int
    chunks: Tuple[Chunk, ...]
    output_tokens: int = 128
    slo_ttft: float = 60.0


@dataclass(frozen=True)
class RuntimeConfig:
    """One point on the §70 ablation ladder. Every field is a §59 component
    switch; the Phase 6 ladder varies only these."""
    name: str
    kv_capacity_tokens: int = 2048
    context_budget_tokens: int = 1024
    retention: str = "lru"          # lru | predictive | queue_aware | queue_semantic
    scheduler: str = "fifo"         # fifo | edf (§24-25)
    context_planning: bool = False  # §21-23 + §57-58
    prefetch: bool = False          # §20
    quality_epsilon: float = 0.05   # §68 quality guard on the SLO
    queue_lookahead: int = 8        # §46 queue view depth
    # W7較正連動 (Phase 3 → §20): benefit推定に信号の確信度を接続する。
    # discount = 1 - signal_ece_proxy*(1 - signal_confidence)。
    # signal_confidence: 0=較正未接続 (既定、保守的に10%割引=discount 0.90)、
    # 1=完全較正 (ECE=0想定、無割引=discount 1.0、signalを額面通り信頼)。
    # ece_proxyはPhase3較正のECE (raw 0.43→rec 0.09→pop 0.10)。
    # W7帯 (agent再訪) では popularity較正が効くため confidence高で
    # ratio実効値を下げ、発行を増やす方向に働く。
    # 全体適用 (conf>0) は W6/W8 後退のため不採用 (FOLLOWUP §8)。
    signal_confidence: float = 0.0  # 0=既定 (W6/W8で最良を実測確認)
    signal_ece_proxy: float = 0.10  # popularity較正のECE目安
    # W1ガード (RQ6の応用): stream全体の再利用信号が弱いとき
    # prefetch lane を止め、retention をLRUにfallbackする。
    # --- prefetch lane guard ---
    prefetch_guard: bool = True     # True: lane自動停止を有効化
    prefetch_min_seen: int = 3      # seen>=N。W7s1実測: 3→4で変化なし。
    prefetch_min_benefit_ratio: float = 1.5  # benefit > ratio*tau。
    # NOTE: 2.0はW7s1でslo22/gp0.00570と改善するが、W4/W6/W8のseed1で
    # 後退するため全体最適の1.5に戻した。W7専用チューニングは別途検討。
    prefetch_max_waste_rate: float = 0.6     # 直近windowのwaste率がこれ超で停止
    prefetch_window: int = 16       # waste率の観測window
    # --- retention fallback ---
    retention_fallback: bool = True  # True: 信号弱でLRUにfallback
    # p90 access_count < N回なら弱信号扱い。実測: W1 p90=2 / W2 p90=6 /
    # W4 p90=27 / W8 p90=34。W1のみに発動させるため閾値は3-5の範囲で4。
    retention_min_signal: float = 4.0  # p90 access_count の閾値
    retention_warmup_reqs: int = 24  # この件数を超えても弱なら真に再利用なし
    # --- EDF starvation guard (§24-25) ---
    edf_starvation_guard: bool = True  # 最長待機の救済
    edf_max_wait: float = 40.0  # この時間を超えて待ったらEDFを無視して救済
    # 実測: W8s6で400→miss13、80→miss9、40→miss3 (Eと同等)。
    # P2: 相対化フラグ。edf_max_wait_frac>0なら queue内最小SLO×frac を
    # 閾値に使う (絶対時間40.0はW8s6帯の暫定値)。0.0で旧絶対値動作。
    edf_max_wait_frac: float = 0.0  # 0=絶対値、>0=相対値 (例0.5)
    # --- M3 semantic shuffle control (PREREG_M3_SHUFFLE §2) ---
    # None (既定): 現行コードと完全同一 (True condition)。
    # Dict: key→key の単置換 π。signal 読み出し 4 経路のみ π(k) 経由に
    # 変換し、_record_access の真値記録・queue/時系・全ガードは不変。
    signal_perm: Optional[Dict[str, str]] = field(
        default=None, compare=False, repr=False)


@dataclass
class RequestRecord:
    request_id: str
    arrival: float
    start: float
    complete: float
    queue_wait: float
    ttft: float
    e2e: float
    output_tokens: int
    scheduled_tokens: int           # prefix + kept chunks entering prefill
    quality_loss: float             # §68 quality lost by dropped context
    prefix_hit: bool
    chunks_total: int
    chunks_selected: int
    slo_met: bool = False           # §66: TTFT SLO AND quality budget both hold
    prefetch_spill: float = 0.0
    ir_removed_ops: int = 0         # §58 pass work exercised for this request



@dataclass
class RunResult:
    config: str
    n_requests: int
    slo_met: int
    quality_breaches: int
    makespan: float
    goodput: float                  # §66: SLO met requests per unit time
    ttft_p50: float
    ttft_p95: float
    ttft_p99: float
    e2e_p95: float
    hit_rate: float
    prefix_hits: int
    evictions: int
    prefetches: int
    prefetch_hits: int
    prefetch_waste: int             # §20 T_interference: issued, never reused
    ir_ops_removed: int
    mean_quality_loss: float
    records: List[RequestRecord] = field(default_factory=list)

    def summary(self) -> Dict[str, float]:
        """Config-independent digest for paired comparisons (§69)."""
        return {
            "goodput": self.goodput, "slo_met": float(self.slo_met),
            "ttft_p95": self.ttft_p95, "hit_rate": self.hit_rate,
            "quality_breaches": float(self.quality_breaches),
            "prefetch_hits": float(self.prefetch_hits),
            "prefetch_waste": float(self.prefetch_waste),
            "ir_ops_removed": float(self.ir_ops_removed),
            "makespan": self.makespan,
        }


# ---------------------------------------------------------------------------
# Retention policy adapters (Phase 2/3 mechanisms reused verbatim)
# ---------------------------------------------------------------------------

class _LRURetentionPolicy:
    """Oldest-unreferenced-first; the §47 baseline default."""

    def choose_victim(self, lru, signals, ages, queue_keys=None):
        return lru[0]


class _PredictiveAdapter:
    def __init__(self, policy):
        self._p = policy

    def choose_victim(self, lru, signals, ages, queue_keys=None):
        return self._p.choose_victim(lru, signals, ages)


class _QueuedAdapter:
    def __init__(self, policy):
        self._p = policy

    def choose_victim(self, lru, signals, ages, queue_keys):
        return self._p.choose_victim(lru, signals, ages, queue_keys)




# ---------------------------------------------------------------------------
# The Adaptive Runtime (§59 integration point)
# ---------------------------------------------------------------------------

class AdaptiveRuntime:
    """Serves a request stream through the full §59 pipeline on one server."""

    def __init__(self, config: RuntimeConfig,
                 engine: Optional[SimulatedExecutionEngine] = None):
        self.config = config
        self.engine = engine or SimulatedExecutionEngine()
        self.retention = make_retention(config.retention)
        self.time = 0.0
        self._reset_state()

    # -- state -------------------------------------------------------------

    def _reset_state(self):
        from collections import OrderedDict, deque
        self.kv: "OrderedDict[str, int]" = OrderedDict()   # key -> tokens
        self.kv_used = 0
        self.access_count: Dict[str, int] = {}            # §33 online demand
        self.last_access: Dict[str, int] = {}
        self.key_tokens: Dict[str, int] = {}              # known key sizes
        self.seq = 0
        self.records: List[RequestRecord] = []
        self.evictions = 0
        self.prefetches = 0
        self.prefetch_hits = 0
        self.prefetch_waste = 0
        self.ir_ops_removed = 0
        self._pf: Optional[_Prefetch] = None
        # prefetched keys not yet reused (§20)。**挿入順 (発行順) を保つ
        # ordered set (dict)**: set だと列挙順が PYTHONHASHSEED 依存になり、
        # _serve_one の hit 承認ループ (下記 `for key in list(self._pf_live)`
        # + break) で同点 eligible な複数キーからどれを hit とするかが
        # プロセス毎に揺れ、goodput が再現しなかった (Reproducible 違反)。
        # dict にすることで発行順 (=最古 landed 優先) で決定的に走査する。
        self._pf_live: Dict[str, None] = {}
        # W1ガード用: prefetch成否の履歴 (True=hit, False=waste) と lane状態。
        # P1高速化: window分だけ保持するdequeに (長期streamでの肥大防止)。
        self._pf_history: Deque[bool] = deque(maxlen=self.config.prefetch_window)
        self._pf_lane_open = True
        self._pf_lane_closes = 0     # lane停止回数 (診断用)
        # P1高速化: signals/agesキャッシュ (access更新時のみ再計算)。
        self._sig_cache: Dict[str, float] = {}
        self._sig_cache_max = 0
        self._age_cache: Dict[str, int] = {}
        self._age_cache_seq = -1

    # -- ex-ante semantic signals (§33 / Phase 3 calibration) ---------------

    def _perm_key(self, key: str) -> str:
        """M3シャッフル (PREREG_M3_SHUFFLE §2): 対応写像 π 経由の参照。
        signal_perm=None なら恒等 — 現行コードと完全同一 (True condition)。"""
        perm = self.config.signal_perm
        return perm.get(key, key) if perm else key

    def _signal(self, key: str) -> float:
        if not self.access_count:
            return 0.0
        mx = max(self.access_count.values())
        return (self.access_count.get(self._perm_key(key), 0) / mx
                if mx > 0 else 0.0)

    def _apply_perm(self, raw: Dict[str, float]) -> Dict[str, float]:
        """M3: signal dict を π で引き直し {k: raw(π(k))}。
        None なら raw 自身をそのまま返す (恒等・キャッシュ整合も不変)。"""
        perm = self.config.signal_perm
        if not perm:
            return raw
        return {k: raw.get(perm.get(k, k), 0.0) for k in raw}

    def _signals(self) -> Dict[str, float]:
        # P1高速化: access更新がなければキャッシュ返却。
        if not self.access_count:
            return {}
        mx = max(self.access_count.values())
        if mx == self._sig_cache_max and self._sig_cache:
            return self._apply_perm(self._sig_cache)
        self._sig_cache = {k: (v / mx if mx > 0 else 0.0)
                           for k, v in self.access_count.items()}
        self._sig_cache_max = mx
        return self._apply_perm(self._sig_cache)

    def _ages(self) -> Dict[str, int]:
        # P1高速化: seqが進んでいなければキャッシュ返却。
        if self._age_cache_seq == self.seq and self._age_cache:
            return self._age_cache
        self._age_cache = {k: self.seq - la
                           for k, la in self.last_access.items()}
        self._age_cache_seq = self.seq
        return self._age_cache

    def _record_access(self, keys: Sequence[str]):
        for key in keys:
            self.access_count[key] = self.access_count.get(key, 0) + 1
            self.last_access[key] = self.seq
            self.seq += 1
        # P1高速化: キャッシュ無効化 (次回_signals/_agesで再計算)。
        self._sig_cache_max = 0
        self._age_cache_seq = -1

    # -- KV pool (§13-15) ----------------------------------------------------

    def _resident(self, key: str, at: float) -> bool:
        if key not in self.kv:
            return False
        if self._pf is not None and key == self._pf.key and at < self._pf.ready_at:
            return False   # transfer still in flight: not usable yet (§20)
        return True

    def _touch(self, key: str, tokens: int):
        # P1高速化: 旧O(N)全合計ループを廃止し増分管理に。
        old = self.kv.get(key, 0)
        self.kv[key] = tokens
        self.kv.move_to_end(key)
        self.kv_used += tokens - old

    def _queue_keys(self, pending: Sequence[WorkloadRequest]) -> set:
        """§16/§46: keys the pending queue will reference (already arrived)."""
        keys: set = set()
        for req in list(pending)[: self.config.queue_lookahead]:
            keys.add(req.prefix_key)
            for ch in req.chunks:
                keys.add(ch.chunk_id)
        return keys

    def _weak_signal(self) -> bool:
        """stream全体の再利用信号が弱いか (ex-ante: PAST accessesのみ)。

        p90 access_count が閾値未満 = 9割のキーがほぼ使い捨て。
        序盤は全streamが弱く見えるため、ウォームアップ件数を超えても
        弱い場合のみ真の弱信号と判定 (W7序盤の誤発動を防ぐ)。
        実測 (最終): W1 p90=2 / W8 p90=34 / W7 p90=34。
        """
        if not self.access_count:
            return True
        if len(self.records) < self.config.retention_warmup_reqs:
            return False
        vals = sorted(self.access_count.values())
        p90 = vals[min(len(vals) - 1, int(len(vals) * 0.9))]
        return p90 < self.config.retention_min_signal

    def _choose_victim(self, keys, pending):
        """retention選択 + W1ガードfallback (信号弱→LRU)。

        fallback条件: retention_fallback ON かつ弱信号。
        schedulerは問わない (W1ではEDF自体がqueue_semanticと組で害になる
        ため、FのretentionもLRUに落とす。W5/W6のEDF系では信号が強い
        (W6 p90高) か、W5は勝ちの源泉がEDF順序自体なので影響なし)。
        """
        if (self.config.retention_fallback and self._weak_signal()):
            return keys[0]          # LRU: OrderedDictの先頭 = 最古未参照
        try:
            return self.retention.choose_victim(
                keys, self._signals(), self._ages(),
                self._queue_keys(pending))
        except TypeError:
            return self.retention.choose_victim(
                keys, self._signals(), self._ages())

    def _enforce_capacity(self, pending: Sequence[WorkloadRequest]):
        while self.kv_used > self.config.kv_capacity_tokens and self.kv:
            keys = list(self.kv.keys())
            # prefetch着弾直後のキーは追放候補から除外 (着弾→即追放の
            # 無駄を防ぐ。W8s6でFがEより追放多 (194 vs 175) の一因)。
            protected = set(self._pf_live)
            cands = [k for k in keys if k not in protected] or keys
            victim = self._choose_victim(cands, pending)
            # _choose_victimがcands外を返す政策の場合の安全弁
            if victim not in self.kv:
                victim = cands[0]
            self.kv_used -= self.kv.pop(victim)
            self.evictions += 1
            if victim in self._pf_live:
                # §20: evicted before any reuse — prefetch became interference
                self._pf_live.pop(victim, None)
                self.prefetch_waste += 1
                self._pf_history.append(False)

    # -- context decision paths --------------------------------------------

    def _plan_context(self, req, start):
        """§21-23 joint budget optimization lowered to IR (§57) and
        optimized by the §58 passes — Phases 4+5 as MECHANISMS inside the
        runtime, not as standalone benchmarks."""
        objects = [MemoryObject(
            id=ch.chunk_id, benefit=ch.benefit, token_cost=ch.tokens,
            prefill_cost=self.engine.prefill_time(ch.tokens),
            reuse_prob=self._signal(ch.chunk_id),
            quality_loss=ch.quality_loss_if_dropped,
            cached=self._resident(ch.chunk_id, start),
        ) for ch in req.chunks]
        opt = JointBudgetOptimizer(
            token_budget=self.config.context_budget_tokens,
            vram_budget=self.config.context_budget_tokens,
            quality_epsilon=self.config.quality_epsilon)
        ctx = opt.optimize(objects)
        # Section 68 quality repair: the greedy joint order can still strand a
        # high-loss chunk behind the token budget. If the dropped quality
        # exceeds epsilon, force-include the highest-loss dropped chunks first
        # (quality-over-length, the same force precedent Phase 4's optimizer
        # uses) until the budget holds. Genuinely infeasible requests keep
        # their honest residual loss (Section 68: not a hidden rescue).
        if ctx.quality_loss > self.config.quality_epsilon:
            included = set(ctx.included)
            residual = [m for m in objects if m.id not in included]
            residual.sort(key=lambda m: m.quality_loss, reverse=True)
            for m in residual:
                if ctx.quality_loss <= self.config.quality_epsilon:
                    break
                if m.quality_loss <= 0.0:
                    continue
                ctx.included.insert(0, m.id)   # quality-critical first
                ctx.quality_loss -= m.quality_loss
                ctx.total_tokens += m.token_cost
        graph = lower_plan(self._runtime_plan(req),
                           self._request_state(req, start), objects,
                           ctx=ctx, request_id=req.request_id)
        opt_graph, report = optimize(graph)
        seen, unique = set(), []
        for op in opt_graph.instructions:
            if op.kind == CONTEXT_SELECT and op.target not in seen:
                seen.add(op.target)
                unique.append(op.target)
        return unique, ctx.quality_loss, report.ops_removed

    def _head_context(self, req):
        """§47 baseline: head truncation — keep chunks in order until the
        context budget fills; dropped-tail quality is counted (§68)."""
        budget = self.config.context_budget_tokens
        selected, used, seen = [], 0, set()
        for ch in req.chunks:
            if ch.chunk_id in seen or used + ch.tokens > budget:
                continue
            selected.append(ch.chunk_id)
            seen.add(ch.chunk_id)
            used += ch.tokens
        unique = {ch.chunk_id: ch for ch in req.chunks}
        loss = sum(ch.quality_loss_if_dropped
                   for cid, ch in unique.items() if cid not in seen)
        return selected, loss, 0

    def _request_state(self, req, start) -> RequestState:
        total = req.prefix_tokens + sum(ch.tokens for ch in req.chunks)
        hit = self._resident(req.prefix_key, start)
        return RequestState(
            input_tokens=total,
            expected_output_tokens=req.output_tokens,
            shared_prefix_length=req.prefix_tokens if hit else 0,
            prefix_reuse_probability=self._signal(req.prefix_key),
            session_id=req.prefix_key)

    def _runtime_plan(self, req) -> RuntimePlan:
        return RuntimePlan(
            request_id=req.request_id,
            context_plan=ContextPlan(strategy="selective",
                                     selected_candidates=len(req.chunks),
                                     confidence=1.0),
            cache_plan=CachePlan(policy=self.config.retention,
                                 retention_action="KEEP",
                                 prefetch_enabled=self.config.prefetch,
                                 confidence=1.0, reason="full_runtime"),
            scheduling_plan=SchedulingPlan(policy=self.config.scheduler,
                                           starvation_protection=True,
                                           execution_priority=1,
                                           confidence=1.0),
            prefetch_plan=PrefetchPlan(enabled=self.config.prefetch,
                                       targets=[], confidence=1.0),
            execution_plan=ExecutionPlan(mode="normal",
                                         speculative_decoding=False,
                                         batch_size=1, confidence=1.0),
            fallback_plan=FallbackPlan(trigger=False, fallback_policy="",
                                       reason=""))

    def _top_prefetch_target(self, exclude, queue_keys=None) -> Optional[str]:
        """§20/§33: hottest known key that is not resident — ex-ante signal
        only (past accesses); never the future stream. When the pending
        queue view is supplied, only keys the queue will actually reference
        are eligible — §46 queue-demand gating so the lane cannot churn the
        KV pool on keys nobody is waiting for."""
        best, best_sig = None, 0.0
        for key, sig in self._signals().items():
            if queue_keys is not None and key not in queue_keys:
                continue
            if sig > best_sig and key not in self.kv and key not in exclude \
                    and key in self.key_tokens:
                best, best_sig = key, sig
        return best

    def _pick_next(self, queue):
        if self.config.scheduler == "edf":
            # §24-25: earliest deadline first; deadline = arrival + TTFT SLO.
            # W2の教訓: 全SLO同一 (deadline順=arrival順) の逐次条件では
            # EDFはFIFOと同等のはずが、救済ガードの発動で順序を崩して2件
            # 落とす。救済は「待機が閾値を超え、かつキューに複数滞留」
            # (並列圧あり) のときのみ発動し、逐次条件ではFIFOに退化させる。
            if self.config.edf_starvation_guard and len(queue) > 1:
                # P2: 閾値は絶対値 (旧) か queue内最小SLO×frac (相対) か。
                if self.config.edf_max_wait_frac > 0:
                    min_slo = min(r.slo_ttft for r in queue)
                    thresh = min_slo * self.config.edf_max_wait_frac
                else:
                    thresh = self.config.edf_max_wait
                cands = [r for r in queue
                         if (self.time - r.arrival_time) > thresh
                         and (r.arrival_time + r.slo_ttft) > self.time]
                if cands:
                    return min(cands,
                               key=lambda r: ((r.arrival_time + r.slo_ttft)
                                              - self.time))
            return min(queue, key=lambda r: (r.arrival_time + r.slo_ttft,
                                             r.arrival_time))
        return queue[0]



    # -- the per-request pipeline -------------------------------------------

    def _serve_one(self, req, queue) -> float:
        start = max(self.time, req.arrival_time)
        # §20: a prefetch that finished before this lookup is already usable
        if self._pf is not None and self._pf.ready_at <= start:
            self._touch(self._pf.key, self._pf.tokens)
            self._pf = None
        # context decision (§21-23 + §57-58, or naive head truncation §47)
        if self.config.context_planning:
            selected_ids, quality_loss, ir_removed = self._plan_context(req, start)
        else:
            selected_ids, quality_loss, ir_removed = self._head_context(req)
        self.ir_ops_removed += ir_removed
        selected_set = set(selected_ids)
        by_id = {ch.chunk_id: ch for ch in req.chunks}
        # residency at lookup time, then §36 engine pricing
        prefix_hit = self._resident(req.prefix_key, start)
        cached_tokens = req.prefix_tokens if prefix_hit else 0
        kept_tokens = 0
        for cid in selected_ids:
            ch = by_id.get(cid)
            if ch is None:
                continue
            kept_tokens += ch.tokens
            if self._resident(cid, start):
                cached_tokens += ch.tokens
        input_tokens = req.prefix_tokens + kept_tokens
        # §20 prefetch race: an in-flight transfer THIS request needs overlaps
        # its stage window (unrelated in-flight transfers cost nothing here)
        spill = 0.0
        if self._pf is not None and (
                self._pf.key == req.prefix_key or self._pf.key in selected_set):
            window = self.engine.stage_time(
                cached_tokens, input_tokens - cached_tokens)
            spill = max(0.0, self._pf.ready_at - (start + window))
        # §20 prefetch hit: first reuse of a landed, not-yet-reused target
        # list(_pf_live) は発行順 (挿入順)。同点 eligible でも決定的
        # (=最古 landed を先に承認) に break する。set列挙順に依存しない。
        for key in list(self._pf_live):
            if (key == req.prefix_key or key in selected_set) \
                    and self._resident(key, start):
                self.prefetch_hits += 1
                self._pf_live.pop(key, None)
                self._pf_history.append(True)
                break
        result = self.engine.execute(EngineRequest(
            request_id=req.request_id, input_tokens=input_tokens,
            output_tokens=req.output_tokens, cached_tokens=cached_tokens,
            queue_wait=start - req.arrival_time, prefetch_spill=spill))
        # clock: the engine's e2e is measured from ARRIVAL (queue_wait + service),
        # so the completion timestamp is arrival + e2e = start + service. Adding
        # e2e on top of `start` would double-count the queue wait and make the
        # simulated clock run away superlinearly.
        complete = req.arrival_time + result.e2e
        # KV updates: touched keys stay, the pool enforces its budget (§13)
        self._touch(req.prefix_key, req.prefix_tokens)
        self.key_tokens[req.prefix_key] = req.prefix_tokens
        for cid in selected_ids:
            ch = by_id.get(cid)
            if ch is not None:
                self._touch(cid, ch.tokens)
                self.key_tokens[cid] = ch.tokens
        self._enforce_capacity(queue)
        # §33 online feedback — AFTER this request's own decisions (ex-ante)
        self._record_access([req.prefix_key] + selected_ids)
        rec = RequestRecord(
            request_id=req.request_id, arrival=req.arrival_time,
            start=start, complete=complete,
            queue_wait=start - req.arrival_time,
            ttft=result.ttft, e2e=result.e2e,
            output_tokens=req.output_tokens,
            scheduled_tokens=input_tokens, quality_loss=quality_loss,
            prefix_hit=prefix_hit, chunks_total=len(by_id),
            chunks_selected=len(selected_ids),
            prefetch_spill=spill, ir_removed_ops=ir_removed)
        # §66+§68: a request counts toward goodput only if BOTH the TTFT SLO
        # and the quality budget hold (quality guard, §68)
        rec.slo_met = (result.ttft <= req.slo_ttft
                       and quality_loss <= self.config.quality_epsilon)
        self.records.append(rec)
        # §20 prefetch issue + W1ガード (RQ6: 「適用しない」選択):
        #  (a) benefit > ratio*tau (旧 benefit>tau を1.5倍に強化)
        #  (b) seen >= 3 (旧2→3)
        #  (c) 直近windowのwaste率が閾値超でlane停止 (一度閉じたら再開なし:
        #      W1のような再利用なしstreamでは再開してもwasteを重ねるだけ)
        if self.config.prefetch and self._pf is None:
            if self.config.prefetch_guard and not self._pf_lane_open:
                pass
            else:
                if self.config.prefetch_guard:
                    # P1: _pf_historyはdeque(maxlen=window)のためisliceで
                    # 末尾windowを参照する (dequeはslice不可)。
                    from itertools import islice
                    win = list(islice(self._pf_history,
                                      max(0, len(self._pf_history)
                                          - self.config.prefetch_window),
                                      len(self._pf_history)))
                    if len(win) >= self.config.prefetch_window:
                        waste_rate = sum(1 for h in win if not h) / len(win)
                        if waste_rate > self.config.prefetch_max_waste_rate:
                            self._pf_lane_open = False
                            self._pf_lane_closes += 1
                if self._pf_lane_open or not self.config.prefetch_guard:
                    target = self._top_prefetch_target(
                        exclude=selected_set | {req.prefix_key})
                    if target is not None:
                        tokens = self.key_tokens[target]
                        tau = self.engine.model.transfer_cost(tokens)
                        # W7較正連動: benefitを確信度で割り引く。
                        # discount = 1 - ece_proxy*(1-confidence)。
                        # confidence=0 (既定) → discount=0.90 (保守的10%割引)。
                        # confidence=1 → discount=1.0 (ECE割引なし、signalを
                        # 額面通り信頼)。旧 (機能導入前) の無割引は conf=1.0 相当。
                        sig = self._signal(target)
                        discount = (1.0 - self.config.signal_ece_proxy
                                    * (1.0 - self.config.signal_confidence))
                        benefit = (sig * discount
                                   * self.engine.model.saved_recompute(tokens))
                        # M3: min_seen も π 経由 (PREREG §2 経路4)。
                        seen = self.access_count.get(self._perm_key(target), 0)
                        if (seen >= self.config.prefetch_min_seen
                                and benefit > (self.config.prefetch_min_benefit_ratio
                                               * tau)):
                            self._pf = _Prefetch(key=target, tokens=tokens,
                                                 issued_at=complete,
                                                 ready_at=complete + tau)
                            self._pf_live[target] = None  # ordered: 発行順維持
                            self.prefetches += 1
        return complete


    def serve(self, stream):
        """Serve the full stream. Arrivals are admitted only once they have
        arrived: the queue view (and every policy built on it) is ex-ante
        (§16/§46)."""
        self._reset_state()
        pending = sorted(stream, key=lambda r: (r.arrival_time, r.request_id))
        n = len(pending)
        if n == 0:
            return self._result()
        queue: List[WorkloadRequest] = []
        idx = 0
        self.time = pending[0].arrival_time
        while idx < n or queue:
            while idx < n and pending[idx].arrival_time <= self.time:
                queue.append(pending[idx])
                idx += 1
            if not queue:
                self.time = pending[idx].arrival_time
                continue
            req = self._pick_next(queue)
            queue.remove(req)
            self.time = self._serve_one(req, queue)
        # stream end: prefetched-but-never-reused targets are §20 interference.
        # P1: 末尾wasteも履歴に反映 (lane判定が末尾偏重wasteを見逃さない)。
        tail = len(self._pf_live)
        self.prefetch_waste += tail
        for _ in range(tail):
            self._pf_history.append(False)
        self._pf_live.clear()
        self._pf = None
        return self._result()

    def _result(self) -> RunResult:
        recs = self.records
        n = len(recs)
        if n == 0:
            return RunResult(self.config.name, 0, 0, 0, 0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0, 0, self.evictions, self.prefetches,
                             self.prefetch_hits, self.prefetch_waste,
                             self.ir_ops_removed, 0.0)
        ttfts = sorted(r.ttft for r in recs)
        e2es = sorted(r.e2e for r in recs)
        makespan = max(r.complete for r in recs) - min(r.arrival for r in recs)
        if makespan <= 0:
            makespan = 1e-9
        slo_met = sum(1 for r in recs if r.slo_met)
        breaches = sum(1 for r in recs
                       if r.quality_loss > self.config.quality_epsilon)
        hits = sum(1 for r in recs if r.prefix_hit)
        return RunResult(
            config=self.config.name, n_requests=n, slo_met=slo_met,
            quality_breaches=breaches, makespan=makespan,
            goodput=slo_met / makespan,
            ttft_p50=_pct(ttfts, 50), ttft_p95=_pct(ttfts, 95),
            ttft_p99=_pct(ttfts, 99), e2e_p95=_pct(e2es, 95),
            hit_rate=hits / n, prefix_hits=hits, evictions=self.evictions,
            prefetches=self.prefetches, prefetch_hits=self.prefetch_hits,
            prefetch_waste=self.prefetch_waste,
            ir_ops_removed=self.ir_ops_removed,
            mean_quality_loss=sum(r.quality_loss for r in recs) / n,
            records=recs)


def _pct(values: List[float], q: float) -> float:
    """Inclusive-linear percentile (same convention as simulator/discrete.py)."""
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    pos = (len(values) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return values[lo] + (values[hi] - values[lo]) * frac


def make_retention(name: str):
    if name == "lru":
        return _LRURetentionPolicy()
    if name == "predictive":
        from cache.predictive_retention import PredictiveRetentionPolicy
        return _PredictiveAdapter(PredictiveRetentionPolicy())
    if name == "queue_aware":
        from scheduler.queue_cache_policy import QueueAwareRetentionPolicy
        return _QueuedAdapter(QueueAwareRetentionPolicy())
    if name == "queue_semantic":
        from scheduler.queue_cache_policy import QueueSemanticRetentionPolicy
        return _QueuedAdapter(QueueSemanticRetentionPolicy())
    raise ValueError(f"unknown retention policy: {name}")


@dataclass
class _Prefetch:
    """One in-flight §20 prefetch transfer."""
    key: str
    tokens: int
    issued_at: float
    ready_at: float
    consumed: bool = False
