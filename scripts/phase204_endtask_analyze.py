import json, numpy as np

ARMS = ["uniform@600", "ridge", "nnls", "map_free"]


def boot(d, seed=204, B=8000):
    d = np.asarray(d, float); n = len(d)
    rng = np.random.default_rng(seed)
    b = np.array([d[rng.integers(0, n, n)].mean() for _ in range(B)])
    return d.mean() * 100, np.percentile(b, 2.5) * 100, np.percentile(b, 97.5) * 100


for w in ["qwen3", "qwen2"]:
    rows = [json.loads(l) for l in open(f"data/phase204_endtask_{w}.jsonl")]
    n = len(rows)
    lab = np.array([r["label"] for r in rows])
    ok = {a: np.array([int(np.argmax(r["probs"][a]) == r["label"]) for r in rows]) for a in ARMS}
    print(f"\n================ {w}  n={n} ================")
    for a in ARMS:
        m, lo, hi = boot(ok[a])
        print(f"  {a:12s} acc {m:5.1f}% [{lo:5.1f},{hi:5.1f}]")
    for a, b in [("nnls", "ridge"), ("map_free", "uniform@600"), ("ridge", "uniform@600"), ("nnls", "uniform@600")]:
        m, lo, hi = boot(ok[a] - ok[b])
        star = " *" if lo > 0 or hi < 0 else ""
        print(f"  {a:9s} - {b:12s} {m:+6.1f} [{lo:+6.1f},{hi:+6.1f}]{star}")
    cat = np.array([r["category"] for r in rows])
    for c in ["direct_attributes", "relative_position"]:
        m = cat == c
        if m.sum() == 0: continue
        print(f"  stratum {c} (n={m.sum()}): " + "  ".join(f"{a} {100*ok[a][m].mean():.1f}" for a in ARMS))
