"""Real-GPU executor for M4 (PREREG_M4_REALWORLD §0 A判決 / §1 environment).

HF transformers + Qwen2.5-0.5B-Instruct (revision-pinned, fp16) on the local
CUDA GPU. Prefill reuses an exact-path KV cache (per-layer legacy tuples ->
Cache object); decode is greedy with fixed output length. GPU time comes from
torch.cuda.Event pairs (§8-5).
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"


def load_executor(device: str = "cuda") -> "GpuExecutor":
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    # attn_implementation="eager" is the M4-fixed backend (PREREG §5/§8-6):
    # SDPA/flash kernels reassociate reductions with sequence length, so
    # partial-prefill + cached continuation is NOT token-identical to a
    # scratch full prefill (probe: m4_realworld/probe_mc0_backend.py —
    # sdpa diverges at decode step 6, eager is exact). Both paired arms use
    # the same backend, so comparisons stay paired.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, dtype=torch.float16,
            attn_implementation="eager")
    except TypeError:  # older transformers kwarg name
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, torch_dtype=torch.float16,
            attn_implementation="eager")
    model = model.to(device).eval()
    return GpuExecutor(model, tok, device)


class GpuExecutor:
    """Execution adapter: real prefill/decode behind the harness seam."""

    def __init__(self, model, tokenizer, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.vocab_size = int(model.config.vocab_size)

    # -- cache plumbing ----------------------------------------------------

    def _make_cache(self, keys: List[torch.Tensor],
                    values: List[torch.Tensor]):
        """Rebuild a DynamicCache from stored per-layer span tensors.

        transformers 5.x uses the layers-based Cache API (no legacy tuples):
        DynamicLayer(keys=..., values=...) reconstructs state directly.
        """
        from transformers.cache_utils import DynamicCache, DynamicLayer
        cache = DynamicCache()
        cache.layers = [DynamicLayer(keys=k.contiguous(),
                                     values=v.contiguous())
                        for k, v in zip(keys, values)]
        return cache

    @staticmethod
    def legacy_layers(cache) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Per-layer (key, value) pairs of the full current sequence."""
        return [(lay.keys, lay.values) for lay in cache.layers]

    # -- harness interface -------------------------------------------------

    def prefill(self, tokens: List[int], hit_end: int,
                cached: Optional[Tuple[List[torch.Tensor], List[torch.Tensor]]],
                collect_logits: bool = False):
        """Prefill positions [hit_end, L). Returns (cache, full_layers,
        gpu_ms, first_logits) — first_logits seeds greedy decode (MC0: the
        cache path and the scratch path produce their step logits here)."""
        ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
        new_ids = ids[:, hit_end:]
        cache = self._make_cache(*cached) if cached is not None else None
        # transformers 5.x: positions come from the cache; pass only the
        # attention mask (position_ids kwarg raises IndexError, probe3).
        mask = torch.ones(1, ids.shape[1], dtype=torch.long,
                          device=self.device)
        ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
        torch.cuda.synchronize()
        ev0.record()
        with torch.no_grad():
            out = self.model(new_ids, past_key_values=cache,
                             attention_mask=mask, use_cache=True)
        ev1.record()
        torch.cuda.synchronize()
        gpu_ms = ev0.elapsed_time(ev1)
        return (out.past_key_values, self.legacy_layers(out.past_key_values),
                gpu_ms, out.logits[0, -1])

    def decode(self, cache, n_tokens: int, first_logits,
               collect_logits: bool = False):
        """Greedy decode of n_tokens starting from first_logits."""
        ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
        torch.cuda.synchronize()
        ev0.record()
        out_ids: List[int] = []
        step_logits: List[torch.Tensor] = []
        logits = first_logits
        step = int(cache.get_seq_length())   # positions already in cache (post-prefill)
        for i in range(n_tokens):
            if collect_logits:
                step_logits.append(logits.detach().float().cpu())
            out_ids.append(int(logits.argmax().item()))
            if i + 1 < n_tokens:
                nxt = torch.tensor([[out_ids[-1]]], dtype=torch.long,
                                   device=self.device)
                mask = torch.ones(1, step + i + 1, dtype=torch.long,
                                  device=self.device)
                with torch.no_grad():
                    out = self.model(nxt, past_key_values=cache,
                                     attention_mask=mask, use_cache=True)
                cache = out.past_key_values
                logits = out.logits[0, -1]
        ev1.record()
        torch.cuda.synchronize()
        gpu_ms = ev0.elapsed_time(ev1)
        return out_ids, gpu_ms, step_logits

