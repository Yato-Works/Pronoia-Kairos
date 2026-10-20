"""Dev probe 2: transformers 5.15 DynamicCache surface (crash-safe)."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

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
    with torch.no_grad():
        o1 = model(a, use_cache=True)
    cache = o1.past_key_values
    print("public attrs:", sorted(x for x in dir(cache)
                                  if not x.startswith("_")))
    if hasattr(cache, "layers"):
        lay = cache.layers[0]
        print("layer0:", type(lay).__name__,
              sorted(x for x in dir(lay) if not x.startswith("_")))
        if hasattr(lay, "keys"):
            print("layer0 keys shape:", tuple(lay.keys.shape),
                  "values shape:", tuple(lay.values.shape))
    print("seq:", cache.get_seq_length())
    with torch.no_grad():
        full = model(torch.cat([a, b], dim=1), use_cache=False)
    ref = full.logits[:, -b.shape[1]:, :].float()
    pos = torch.arange(a.shape[1], a.shape[1] + b.shape[1], device="cuda")
    mask = torch.ones(1, a.shape[1] + b.shape[1], dtype=torch.long,
                      device="cuda")
    with torch.no_grad():
        try:
            o2 = model(b, past_key_values=cache, position_ids=pos,
                       attention_mask=mask, use_cache=True)
            print("reuse-same-cache diff:",
                  (o2.logits[:, -b.shape[1]:, :].float() - ref).abs().max().item())
        except Exception as exc:  # noqa: BLE001
            print("reuse-same-cache FAILED:", type(exc).__name__, exc)
    # legacy tuple rebuild paths
    print("has from_legacy_cache:", hasattr(DynamicCache, "from_legacy_cache"))
    print("has to_legacy_cache:", hasattr(cache, "to_legacy_cache"))
    if hasattr(cache, "to_legacy_cache"):
        tup = cache.to_legacy_cache()
        print("legacy tuple len:", len(tup), "elem0:",
              type(tup[0]).__name__,
              [tuple(t.shape) for t in tup[0]] if isinstance(tup[0], tuple) else "?")
        c2 = DynamicCache.from_legacy_cache(tup)
        with torch.no_grad():
            try:
                o3 = model(b, past_key_values=c2, position_ids=pos,
                           attention_mask=mask, use_cache=True)
                print("legacy-rebuild diff:",
                      (o3.logits[:, -b.shape[1]:, :].float() - ref).abs().max().item())
            except Exception as exc:  # noqa: BLE001
                print("legacy-rebuild FAILED:", type(exc).__name__, exc)
    # manual layer copy rebuild
    try:
        c3 = DynamicCache()
        if hasattr(cache, "layers"):
            for lay in cache.layers:
                nl = type(lay)(**{}) if False else None  # placeholder
            c3.layers = [type(l)(keys=l.keys.clone(), values=l.values.clone())
                         if hasattr(l, "keys") else l for l in cache.layers]
        with torch.no_grad():
            o4 = model(b, past_key_values=c3, position_ids=pos,
                       attention_mask=mask, use_cache=True)
            print("manual-copy diff:",
                  (o4.logits[:, -b.shape[1]:, :].float() - ref).abs().max().item())
    except Exception as exc:  # noqa: BLE001
        print("manual-copy FAILED:", type(exc).__name__, exc)
    # slice-prefix rebuild: positions [0, n) only, then suffix from n
    n = a.shape[1]
    try:
        c4 = DynamicCache()
        c4.layers = [type(l)(keys=l.keys[:, :, :n, :].contiguous(),
                             values=l.values[:, :, :n, :].contiguous())
                     for l in cache.layers]
        with torch.no_grad():
            o5 = model(b, past_key_values=c4, position_ids=pos,
                       attention_mask=mask, use_cache=True)
            print("slice-prefix diff:",
                  (o5.logits[:, -b.shape[1]:, :].float() - ref).abs().max().item())
    except Exception as exc:  # noqa: BLE001
        print("slice-prefix FAILED:", type(exc).__name__, exc)


if __name__ == "__main__":
    main()
