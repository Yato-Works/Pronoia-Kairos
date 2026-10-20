"""Dev probe: verify transformers 5.15 KV-cache continuation semantics (M4).

Checks: (1) exact-path prefix reuse via legacy tuple -> DynamicCache rebuild,
(2) position_ids/attention_mask handling for suffix prefill,
(3) greedy-decode equality cache-path vs full prefill, (4) prefill speed.
Not part of the harness; deleted after use.
"""
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
REV = "7ae557604adf67be50417f59c2c2f167def9a775"


def main():
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REV)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, revision=REV, dtype=torch.float16)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, revision=REV, torch_dtype=torch.float16)
    model = model.to("cuda").eval()
    print("model loaded", model.config.num_hidden_layers, "layers",
          model.config.num_key_value_heads, "kv_heads")

    def enc(s):
        return tok(s, return_tensors="pt").input_ids.to("cuda")

    a = enc("The quick brown fox jumps over the lazy dog. " * 4)
    b = enc(" Additionally the fox was known to be quite clever in the forest.")
    with torch.no_grad():
        o1 = model(a, use_cache=True)
        cache = o1.past_key_values
        print("cache type:", type(cache).__name__)
        print("seq len:", cache.get_seq_length())
        print("has key_cache:", hasattr(cache, "key_cache"))
        k0 = cache.key_cache[0]
        print("key_cache[0] type:", type(k0).__name__,
              "shape:", tuple(k0.shape) if hasattr(k0, "shape") else "?")
        # full reference
        full = model(torch.cat([a, b], dim=1), use_cache=False)
        ref = full.logits[:, -b.shape[1]:, :].float()
        pos = torch.arange(a.shape[1], a.shape[1] + b.shape[1], device="cuda")
        mask = torch.ones(1, a.shape[1] + b.shape[1], dtype=torch.long,
                          device="cuda")
        o2 = model(b, past_key_values=cache, position_ids=pos,
                   attention_mask=mask, use_cache=True)
        d2 = (o2.logits[:, -b.shape[1]:, :].float() - ref).abs().max().item()
        print("suffix+explicit-pos max diff:", d2)
        try:
            o3 = model(b, past_key_values=cache, use_cache=True,
                       attention_mask=mask)
            d3 = (o3.logits[:, -b.shape[1]:, :].float() - ref).abs().max().item()
            print("suffix+default-pos max diff:", d3)
        except Exception as exc:  # noqa: BLE001
            print("default-pos failed:", type(exc).__name__)
        # legacy tuple rebuild
        if hasattr(type(cache), "from_legacy_cache"):
            tup = tuple((cache.key_cache[i], cache.value_cache[i])
                        for i in range(len(cache.key_cache)))
            c2 = type(cache).from_legacy_cache(tup)
            o4 = model(b, past_key_values=c2, position_ids=pos,
                       attention_mask=mask, use_cache=True)
            d4 = (o4.logits[:, -b.shape[1]:, :].float() - ref).abs().max().item()
            print("legacy-rebuild max diff:", d4)
        else:
            print("no from_legacy_cache")
        # greedy continuation equality via incremental decode (cache path)
        gen_cache = cache
        gen_ids = []
        cur_pos = a.shape[1]
        step_logits = []
        for _ in range(8):
            inp = (b if not gen_ids else gen_ids[-1].view(1, 1))
            cp = torch.arange(cur_pos, cur_pos + inp.shape[1], device="cuda")
            am = torch.ones(1, cur_pos + inp.shape[1], dtype=torch.long,
                            device="cuda")
            out = model(inp, past_key_values=gen_cache, position_ids=cp,
                        attention_mask=am, use_cache=True)
            gen_cache = out.past_key_values
            step_logits.append(out.logits[:, -1, :].float())
            nxt = out.logits[:, -1, :].argmax(-1)
            gen_ids.append(nxt)
            cur_pos += inp.shape[1]
        # scratch decode for equality
        s_ids = a.clone()
        s_logits = []
        with torch.no_grad():
            for _ in range(8):
                so = model(s_ids, use_cache=False)
                s_logits.append(so.logits[:, -1, :].float())
                sn = so.logits[:, -1, :].argmax(-1)
                s_ids = torch.cat([s_ids, sn.view(1, 1)], dim=1)
        tok_eq = all(torch.equal(g.view(-1), sl.argmax(-1).view(-1))
                     for g, sl in zip(gen_ids, s_logits))
        logit_diff = max((gl - sl).abs().max().item()
                         for gl, sl in zip(step_logits, s_logits))
        print("8-step greedy tokens equal:", tok_eq, "max step logit diff:",
              logit_diff)
        # prefill speed on real workload scale
        x = torch.randint(0, model.config.vocab_size, (1, 1024), device="cuda")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model(x, use_cache=False)
        torch.cuda.synchronize()
        print("prefill 1024 tok: %.3fs" % (time.perf_counter() - t0))
        y = torch.randint(0, model.config.vocab_size, (1, 1), device="cuda")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(16):
                model(y, use_cache=False)
        torch.cuda.synchronize()
        print("decode 1 tok (no cache): %.4fs/step" % ((time.perf_counter() - t0) / 16))


if __name__ == "__main__":
    main()
