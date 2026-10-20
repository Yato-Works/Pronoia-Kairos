"""W6-inspired reduced workload for M4 (PREREG_M4_REALWORLD §0/§3, §8-3).

Mapping (locked 2026-10-19, mentor verdict C):
- W6 via the Phase-6 generator, first MAX_REQUESTS by arrival kept.
- chunks sorted lexicographically, capped at MAX_CHUNKS per request so the
  input total stays <= MAX_INPUT_TOKENS.
- chunk token lengths canonicalized per chunk_id (deterministic 128..168,
  sha256) so one chunk_id always maps to one exact token segment — the
  precondition for real KV reuse (§8-3/§8-4).
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Tuple

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from compare.full_runtime_experiment import build_family  # noqa: E402
from runtime.adaptive_runtime import Chunk, WorkloadRequest  # noqa: E402

MAX_REQUESTS = 40
MAX_CHUNKS = 8
MAX_INPUT_TOKENS = 2048
W6_FAMILY = "W6_concurrent_long_high_reuse"


def _sha_int(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest(), "big")


def canonical_chunk_tokens(chunk_id: str) -> int:
    """Deterministic canonical length for a chunk_id (128..168, §8-3)."""
    return 128 + _sha_int("m4len:" + chunk_id) % 41


def segment_token_ids(key: str, length: int, vocab_size: int) -> List[int]:
    """Deterministic synthetic token content for a segment (§8-3).

    Same (key, vocab_size) -> same token ids on every run/platform: this is
    what makes cached KV valid for later requests referencing the key.
    """
    import random

    rng = random.Random(_sha_int("m4seg:" + key) % (2 ** 63))
    return [rng.randrange(vocab_size) for _ in range(length)]


def build_reduced_w6(seed: int) -> Tuple[List[WorkloadRequest], object, Dict]:
    """Reduced W6 stream + FamilySpec + summary stats (recorded in artifacts)."""
    reqs, spec = build_family(W6_FAMILY, seed)
    reqs = sorted(reqs, key=lambda r: (r.arrival_time, r.request_id))
    reqs = reqs[:MAX_REQUESTS]
    out: List[WorkloadRequest] = []
    for r in reqs:
        uniq = {c.chunk_id: c for c in r.chunks}          # dedupe within request
        chunks = [replace(c, tokens=canonical_chunk_tokens(c.chunk_id))
                  for c in sorted(uniq.values(), key=lambda c: c.chunk_id)]
        chunks = chunks[:MAX_CHUNKS]
        total = r.prefix_tokens + sum(c.tokens for c in chunks)
        if total > MAX_INPUT_TOKENS:
            raise ValueError(f"reduced request {r.request_id} exceeds budget: {total}")
        out.append(replace(r, chunks=tuple(chunks)))
    return out, spec, stream_stats(out)


def stream_stats(reqs: List[WorkloadRequest]) -> Dict:
    inputs = [r.prefix_tokens + sum(c.tokens for c in r.chunks) for r in reqs]
    keys_per_req = [[r.prefix_key] + [c.chunk_id for c in r.chunks] for r in reqs]
    cnt: Counter = Counter(k for ks in keys_per_req for k in ks)
    seen: set = set()
    opps: List[int] = []
    for ks in keys_per_req:
        opps.append(sum(1 for k in ks if k in seen))
        seen.update(ks)
    n = max(1, len(reqs))
    per_req_chunks = [len(r.chunks) for r in reqs]
    return {
        "n_requests": len(reqs),
        "input_tokens_min": min(inputs), "input_tokens_max": max(inputs),
        "input_tokens_mean": round(sum(inputs) / n, 2),
        "prefix_tokens_per_request": sorted({r.prefix_tokens for r in reqs}),
        "output_tokens_per_request": sorted({r.output_tokens for r in reqs}),
        "chunks_per_request_min": min(per_req_chunks),
        "chunks_per_request_max": max(per_req_chunks),
        "unique_prefixes": len({r.prefix_key for r in reqs}),
        "unique_chunks": len({c.chunk_id for r in reqs for c in r.chunks}),
        "key_access_total": sum(cnt.values()),
        "hot_keys": cnt.most_common(8),
        "reuse_opportunity_mean": round(sum(opps) / n, 3),
        "arrival_min": min(r.arrival_time for r in reqs),
        "arrival_max": max(r.arrival_time for r in reqs),
        "slo_values": sorted({r.slo_ttft for r in reqs}),
    }


def stream_hash(reqs: List[WorkloadRequest]) -> str:
    payload = [
        {"request_id": r.request_id, "arrival": r.arrival_time,
         "prefix_key": r.prefix_key, "prefix_tokens": r.prefix_tokens,
         "chunks": [[c.chunk_id, c.tokens] for c in r.chunks],
         "output_tokens": r.output_tokens, "slo": r.slo_ttft}
        for r in reqs
    ]
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    stream, spec, stats = build_reduced_w6(1)
    print("hash", stream_hash(stream))
    print(json.dumps(stats, indent=2))
