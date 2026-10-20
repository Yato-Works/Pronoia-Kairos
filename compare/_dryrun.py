from compare.full_runtime_experiment import FAMILY_NAMES, build_family, ladder, run_family
from runtime.adaptive_runtime import AdaptiveRuntime

for fam in FAMILY_NAMES:
    f = run_family(fam, seeds=(1,))
    g = {k: round(v["goodput_mean"], 5) for k, v in f["configs"].items()}
    b = {k: v["quality_breaches_total"]
         for k, v in f["configs"].items() if v["quality_breaches_total"]}
    m = {k: round(v["slo_met_mean"], 2) for k, v in f["configs"].items()}
    stream, spec = build_family(fam, 1)
    orders = {}
    for cfg in ladder(spec):
        res = AdaptiveRuntime(cfg).serve(list(stream))
        orders[cfg.name] = [r.request_id for r in res.records]
    diff = sum(1 for a, b2 in zip(orders["E_planning"], orders["F_full"])
               if a != b2)
    print(fam, g, "strong=", f["strong_baseline"],
          "d=", round(f["f_delta_vs_strong_mean"], 5), "brc=", b,
          "met=", m, "edf_reorder=", diff)


