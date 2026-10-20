"""Probe: is scratch-vs-partial+cache divergence an attention-backend effect?

Runs MC0's exact scenario (16-token chunk, half=8 split, 32 greedy decode)
under sdpa and eager backends; prints token-list equality for each.
"""
import torch
from transformers import AutoModelForCausalLM

from m4_realworld.gpu_executor import MODEL_ID, MODEL_REVISION, GpuExecutor

TOKENS = [500, 123, 77, 900, 31, 4, 55, 610, 222, 43, 87, 990, 15, 16, 17, 18]


def _load(attn: str):
    kwargs = dict(revision=MODEL_REVISION)
    try:
        m = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, dtype=torch.float16, attn_implementation=attn, **kwargs)
    except TypeError:
        m = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=torch.float16, attn_implementation=attn,
            **kwargs)
    return m.to("cuda").eval()


def run(attn: str) -> None:
    ex = GpuExecutor(_load(attn), None, "cuda")
    half = len(TOKENS) // 2
    # scratch path
    cache_a, _, _, fl_a = ex.prefill(TOKENS, 0, None, collect_logits=True)
    ids_a, _, _ = ex.decode(cache_a, 32, fl_a, collect_logits=False)
    # partial + cache path
    _, half_layers, _, _ = ex.prefill(TOKENS[:half], 0, None)
    cached = ([k[:, :, :half, :] for k, _v in half_layers],
              [v[:, :, :half, :] for _k, v in half_layers])
    cache_b, _, _, fl_b = ex.prefill(TOKENS, half, cached, collect_logits=True)
    ids_b, _, _ = ex.decode(cache_b, 32, fl_b, collect_logits=False)
    same = ids_a == ids_b
    print(f"{attn}: exact={same}")
    if not same:
        i = next(k for k, (x, y) in enumerate(zip(ids_a, ids_b)) if x != y)
        print(f"  diverge_at={i} a={ids_a[i]} b={ids_b[i]}")
        print(f"  fl_diff={float((fl_a - fl_b).abs().max()):.6f}")
    del ex.model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    for attn in ("sdpa", "eager"):
        run(attn)
