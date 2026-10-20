"""§30 / §57 Cognitive IR — an execution graph you can optimize.

Section 30 thesis: treat inference not as tokens->tokens but as an
optimization-possible execution graph of effect-aware operations.
Section 57 fixes the Phase 5 lowering set:

    MEMORY_RECALL  CONTEXT_SELECT  PREFIX_LOOKUP
    KV_KEEP        KV_EVICT        KV_PREFETCH
    TRANSFER       PREFILL         DECODE

Execution model (Section 36), measured rather than asserted:

* compute ops (recall / select / lookup / bookkeeping / prefetch issue /
  prefill / decode) run serially on the critical path; their wall-clock
  accumulates.
* TRANSFER ops share ONE background lane. A transfer issued at position i
  races the compute that follows it until PREFILL completes
  (Section 36: T_effective = max(T_compute, T_transfer)). Whatever the lane
  cannot hide spills onto the critical path:

      spill = max(0, lane_transfer - overlap_window)

  A transfer issued late (just before PREFILL) has an almost empty window and
  blocks; transfers issued early hide behind recall/select compute. That is
  the mechanical reason Section 58's Prefetch Hoisting can pay for itself.

`execute()` measures real wall-clock nanoseconds of running the graph in this
control-plane prototype (no GPU). Per-op work is proportional to the op's
token size at rates that follow the Section 36 per-token ordering
(t_transfer_per_token 0.01 < t_prefill_per_token 0.05): TRANSFER is a C-level
slice copy, MEMORY_RECALL additionally probes the store and materializes the
object (a deserialize stand-in), PREFILL/DECODE burn Python bytecode per
token. Both sides of the Section 92 gate (IR overhead vs optimization gain)
are measured in the SAME clock on the SAME machine, so no uncalibrated unit
conversion is smuggled into the comparison. Absolute magnitudes are prototype
stand-ins: hardware-calibrated timing remains §43 / §93 work; the claim here
is the sign of the comparison, not nanoseconds on real silicon.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter_ns
from typing import Any, Dict, List, Optional, Tuple

# --- Section 57 op set -------------------------------------------------------

MEMORY_RECALL = "MEMORY_RECALL"
CONTEXT_SELECT = "CONTEXT_SELECT"
PREFIX_LOOKUP = "PREFIX_LOOKUP"
KV_KEEP = "KV_KEEP"
KV_EVICT = "KV_EVICT"
KV_PREFETCH = "KV_PREFETCH"
TRANSFER = "TRANSFER"
PREFILL = "PREFILL"
DECODE = "DECODE"

OP_KINDS: Tuple[str, ...] = (
    MEMORY_RECALL, CONTEXT_SELECT, PREFIX_LOOKUP,
    KV_KEEP, KV_EVICT, KV_PREFETCH, TRANSFER, PREFILL, DECODE,
)
_OP_KIND_SET = frozenset(OP_KINDS)
TRANSFER_KINDS = frozenset({TRANSFER})

# §30: effect-aware operations — what an op touches in the runtime.
_EFFECTS: Dict[str, Tuple[str, ...]] = {
    MEMORY_RECALL: ("reads_memory",),
    CONTEXT_SELECT: ("reads_memory", "writes_context"),
    PREFIX_LOOKUP: ("reads_kv_index",),
    KV_KEEP: ("touches_kv",),
    KV_EVICT: ("touches_kv",),
    KV_PREFETCH: ("issues_transfer", "touches_kv"),
    TRANSFER: ("moves_tokens", "touches_kv"),
    PREFILL: ("reads_context", "computes"),
    DECODE: ("computes",),
}


def effects_of(op: "IROp") -> Tuple[str, ...]:
    """§30 effect tags for one instruction (empty for unknown kinds)."""
    return _EFFECTS.get(op.kind, ())


@dataclass(frozen=True)
class IROp:
    """One effect-aware instruction. `origin` records which plan field /
    pipeline stage produced it, so any op can be explained in the
    decision trace (§32/§99)."""
    kind: str
    target: str = ""
    tokens: int = 0
    origin: str = ""
    args: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class ExecutionResult:
    """Outcome of executing a graph: wall-clock on the critical path, split
    into compute work and the transfer lane's un-hidden spill (§36)."""
    wall_ns: int
    compute_ns: int
    transfer_ns: int      # total transfer work offered to the lane
    spill_ns: int         # transfer work not hidden by the overlap window
    op_count: int


class _ExecState:
    """Scratch state for ONE graph execution (fresh per execute() call)."""
    __slots__ = ("context", "kv", "inflight", "lane_bufs", "acc")

    def __init__(self) -> None:
        self.context: List[int] = []
        self.kv: Dict[str, int] = {}
        self.inflight: List[str] = []
        self.lane_bufs: List[List[int]] = []
        self.acc = 0


def _burn(op: IROp, st: _ExecState) -> None:
    """Stand-in work for one op, proportional to its token size.

    Rate ordering follows §36 (transfer < prefill per token); recall sits
    between them because a store read also materializes the object.
    These are prototype stand-ins, not calibrated silicon costs (§43 open).
    """
    kind, n, target = op.kind, op.tokens, op.target
    if n < 0:
        n = 0
    if kind == MEMORY_RECALL:
        # index probe over blocks + materialize the recalled token slice
        probe = 0
        for b in range((n >> 6) + 1):
            probe ^= hash((target, b))
        st.acc ^= probe
        buf = [i for i in range(n)]          # deserialize stand-in
        st.acc += buf[-1] if buf else 0
    elif kind == CONTEXT_SELECT:
        st.context.extend(range(n))          # slice into the working context
    elif kind == PREFIX_LOOKUP:
        # cheap hash probe over prefix blocks (§36 t_cache_lookup)
        h = 0
        for b in range((n >> 6) + 1):
            h ^= hash((target, b))
        st.acc ^= h
    elif kind == KV_KEEP:
        st.kv[target] = n
    elif kind == KV_EVICT:
        st.kv.pop(target, None)
    elif kind == KV_PREFETCH:
        st.inflight.append(target)           # issue is control-plane work
    elif kind == TRANSFER:
        # C-level slice copy + checksum: the PCIe-ish streaming of the lane
        buf = list(range(n))
        st.lane_bufs.append(buf)
        st.acc += sum(buf[::64]) & 0xFFFF
    elif kind == PREFILL:
        acc, mask = st.acc, 0x7FFFFFFF
        for i in range(n):
            acc = (acc * 31 + i) & mask      # prefill compute, per token
        st.acc = acc
    elif kind == DECODE:
        acc, mask = st.acc, 0x7FFFFFFF
        for i in range(n):
            acc = (acc * 1103515245 + i + 12345) & mask   # TPOT loop
        st.acc = acc
    # unknown kinds burn nothing; problems() rejects them at validation.


@dataclass
class ExecutionGraph:
    """An ordered list of effect-aware instructions for one request."""
    request_id: str = ""
    instructions: List[IROp] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.instructions)

    def count(self, kind: str) -> int:
        return sum(1 for op in self.instructions if op.kind == kind)

    def counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for op in self.instructions:
            out[op.kind] = out.get(op.kind, 0) + 1
        return out

    # --- well-formedness (what a lowering must guarantee) -------------------
    def problems(self) -> List[str]:
        out: List[str] = []
        if not self.instructions:
            return ["graph has no instructions"]
        prefill_i: Optional[int] = None
        decode_i: Optional[int] = None
        prefetch_targets = set()
        for i, op in enumerate(self.instructions):
            if op.kind not in _OP_KIND_SET:
                out.append(f"op[{i}]: unknown kind {op.kind!r}")
            if op.tokens < 0:
                out.append(f"op[{i}]: negative token count {op.tokens}")
            if op.kind == KV_PREFETCH:
                prefetch_targets.add(op.target)
            elif op.kind == TRANSFER and op.target not in prefetch_targets:
                out.append(f"op[{i}]: TRANSFER of {op.target!r} has no "
                           f"preceding KV_PREFETCH (orphan transfer)")
            elif op.kind == PREFILL and prefill_i is None:
                prefill_i = i
            elif op.kind == DECODE and decode_i is None:
                decode_i = i
        if prefill_i is None:
            out.append("graph has no PREFILL")
        if decode_i is None:
            out.append("graph has no DECODE")
        if prefill_i is not None and decode_i is not None and prefill_i > decode_i:
            out.append("PREFILL must precede DECODE")
        return out

    def is_valid(self) -> bool:
        return not self.problems()

    # --- decision trace (§32/§99) -------------------------------------------
    def to_trace_summary(self) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "request_id": self.request_id,
            "op_count": len(self.instructions),
            "ops_by_kind": self.counts(),
        }
        if "pass_report" in self.meta:
            summary["passes"] = self.meta["pass_report"]
        if "strategy" in self.meta:
            summary["strategy"] = self.meta["strategy"]
        if self.meta.get("fallback"):
            summary["fallback"] = True
        return summary

    # --- execution (the measured side of the §92 gate) -----------------------
    def execute(self) -> ExecutionResult:
        """Run the graph here and now, measuring wall-clock nanoseconds.

        Compute accumulates on the critical path. Transfers fill one
        background lane whose overlap window opens at the first transfer and
        closes when PREFILL completes (§36 max(T_compute, T_transfer)); any
        lane work the window cannot absorb spills as extra critical-path time.
        """
        st = _ExecState()
        compute_ns = 0
        transfer_total = 0
        lane_transfer_ns = 0
        lane_window_ns = 0
        lane_has_transfer = False
        lane_open = True
        spill_ns = 0

        for op in self.instructions:
            t0 = perf_counter_ns()
            _burn(op, st)
            dt = perf_counter_ns() - t0
            if op.kind in TRANSFER_KINDS:
                transfer_total += dt
                if lane_open:
                    lane_transfer_ns += dt
                    lane_has_transfer = True
                else:
                    # device already saturated by prefill: blocks decode
                    spill_ns += dt
                continue
            compute_ns += dt
            if lane_open and lane_has_transfer:
                lane_window_ns += dt
                if op.kind == PREFILL:
                    # window closes at PREFILL end; pay whatever is unhidden
                    spill_ns += max(0, lane_transfer_ns - lane_window_ns)
                    lane_open = False

        if lane_open and lane_has_transfer:
            # no PREFILL to close the window (invalid graph, but be total)
            spill_ns += max(0, lane_transfer_ns - lane_window_ns)

        return ExecutionResult(
            wall_ns=compute_ns + spill_ns,
            compute_ns=compute_ns,
            transfer_ns=transfer_total,
            spill_ns=spill_ns,
            op_count=len(self.instructions),
        )
