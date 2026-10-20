"""Dev probe 3: which args make suffix prefill work on transformers 5.15."""
import traceback

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
REV = "7ae557604adf67be50417f59c2c2f167def9a775"


def main():
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REV)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REV, dtype=torch.float16).to("cuda").eval()

    def enc(s):
        return tok(s, return_tensors="pt").input_ids.to("cuda")

    a = enc("The quick brown fox jumps over the lazy dog. " * 4)
    b = enc(" Additionally the fox was known to be quite clever in the forest.")
    n = a.shape[1]
    with torch.no_grad():
        o1 = model(a, use_cache=True)
    cache = o1.past_key_values
    with torch.no_grad():
        full = model(torch.cat([a, b], dim=1), use_cache=False)
    ref = full.logits[:, -b.shape[1]:, :].float()
    pos = torch.arange(n, n + b.shape[1], device="cuda")
    mask = torch.ones(1, n + b.shape[1], dtype=torch.long, device="cuda")
    variants = {
        "pos_only": dict(position_ids=pos),
        "mask_only": dict(attention_mask=mask),
        "none": dict(),
        "cache_position_only": dict(
            cache_position=torch.arange(n, n + b.shape[1], device="cuda")),
        "cache_position+mask": dict(
            cache_position=torch.arange(n, n + b.shape[1], device="cuda"),
            attention_mask=mask),
    }
    import copy
    for name, kw in variants.items():
        try:
            with torch.no_grad():
                out = model(b, past_key_values=copy.deepcopy(cache),
                            use_cache=True, **kw)
            diff = (out.logits[:, -b.shape[1]:, :].float() - ref
                    ).abs().max().item()
            print(name, "OK diff", diff, "seq now", out.past_key_values.get_seq_length())
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc().strip().splitlines()
            print(name, "FAILED", type(exc).__name__, exc, "|", tb[-2].strip())


if __name__ == "__main__":
    main()
