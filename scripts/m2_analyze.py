import json, numpy as np

ARMS = ["uniform@1200", "uniform@1800", "dwa@1200", "avr@1200"]


def boot(d, seed=212, B=8000):
    d = np.asarray(d, float); n = len(d)
    rng = np.random.default_rng(seed)
    b = np.array([d[rng.integers(0, n, n)].mean() for _ in range(B)])
    return d.mean() * 100, np.percentile(b, 2.5) * 100, np.percentile(b, 97.5) * 100


for w in ["qwen3", "qwen2"]:
    f = f"data/m2_budget_policy_{w}.jsonl"
    try:
        rows = [json.loads(l) for l in open(f)]
    except FileNotFoundError:
        print(f"\n===== M2 {w}: no data yet ====="); continue
    n = len(rows); lab = np.array([r["label"] for r in rows])
    cat = np.array([r["category"] for r in rows])
    ok = {a: np.array([int(np.argmax(r["probs"][a]) == r["label"]) for r in rows]) for a in ARMS}
    print(f"\n================ M2 {w}  n={n}  (B=1200 token-layers) ================")
    for a in ARMS:
        m, lo, hi = boot(ok[a])
        tl = np.mean([r["token_layers"][a] for r in rows]) / (1200 * 28)
        print(f"  {a:14s} acc {m:5.1f}% [{lo:5.1f},{hi:5.1f}]   TL {100*tl:5.1f}% of bar")
    print("  paired:")
    for a, b in [("avr@1200", "uniform@1800"), ("dwa@1200", "uniform@1200"), ("avr@1200", "uniform@1200"), ("dwa@1200", "avr@1200")]:
        m, lo, hi = boot(ok[a] - ok[b])
        print(f"    {a:12s} - {b:14s} {m:+6.1f} [{lo:+6.1f},{hi:+6.1f}]{' *' if lo > 0 or hi < 0 else ''}")
    print("  per stratum (single=direct_attributes, cross=relative_position):")
    for c, nm in (("direct_attributes", "single"), ("relative_position", "cross")):
        m = cat == c
        if not m.sum(): continue
        line = "  ".join(f"{a} {100*ok[a][m].mean():5.1f}" for a in ARMS)
        print(f"    {nm:6s} n={m.sum():3d}:  {line}")
        # policy pick: DWA on single, AVR on cross
        pick = "dwa@1200" if nm == "single" else "avr@1200"
        best = max(ARMS, key=lambda a: ok[a][m].mean())
        realized = 100 * ok[pick][m].mean(); bestv = 100 * ok[best][m].mean()
        print(f"           policy picks {pick} ({realized:.1f}); best realized {best} ({bestv:.1f}); gap {realized-bestv:+.1f}")
