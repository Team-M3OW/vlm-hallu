"""W ablation: W=0.25, W=0.5, dynamic W (phase 229), 2 benchmarks x 2 models, equal 300-token
answer budget. Also reads phase 228 (same arms minus dynamic) where phase 229 is still running."""
import json, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmarks as B
D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rng = np.random.default_rng(0)
CELLS = [("qwen3_2b", "vstar"), ("qwen3_2b", "docvqa"), ("qwen2_7b", "vstar"), ("qwen2_7b", "docvqa")]
ARMS = ["uniform@300", "crop25@300", "crop50@300", "dynW@300", "rand25@300"]


def load(m, b, phase):
    f = f"{D}/data/{phase}_{m}_{b}.jsonl"
    return [json.loads(l) for l in open(f) if l.strip()] if os.path.exists(f) else []


def ok(r, a):
    if a in r.get("probs", {}):
        return 1.0 if int(np.argmax(r["probs"][a])) == int(r["gold"]) else 0.0
    if a in r.get("preds", {}):
        g = r["gold"]; p = r["preds"][a]
        gs = g if isinstance(g, (list, tuple)) else [g]
        return 1.0 if any(B.norm(x) and B.norm(x) == B.norm(p) for x in gs) else 0.0
    return None


def acc(rows, a):
    v = [ok(r, a) for r in rows if ok(r, a) is not None]
    return 100 * float(np.mean(v)) if v else float("nan")


def delta(rows, a, b, Bn=10000):
    d = np.array([ok(r, a) - ok(r, b) for r in rows if ok(r, a) is not None and ok(r, b) is not None])
    if len(d) < 8: return None
    m = d[rng.integers(0, len(d), (Bn, len(d)))].mean(1)
    lo, hi = np.percentile(m, 2.5) * 100, np.percentile(m, 97.5) * 100
    return d.mean() * 100, lo, hi, len(d)


print("=" * 108)
print("W ablation, equal 300-token answer budget (crop vs full image at the SAME token count)")
print("=" * 108)
for m, b in CELLS:
    rows9 = load(m, b, "phase229")
    rows8 = load(m, b, "phase228")
    rows = rows9 if rows9 else rows8
    src = "phase229" if rows9 else "phase228"
    if not rows:
        print(f"{m}/{b}: -- running --"); continue
    print(f"\n{m} / {b}  n={len(rows)} ({src})")
    for a in ARMS:
        if all(ok(r, a) is None for r in rows): continue
        line = f"   {a:13s} {acc(rows,a):5.1f}"
        for base in ("uniform@300", "crop25@300"):
            if a != base:
                d = delta(rows, a, base)
                if d: line += f"   vs {base}: {d[0]:+5.1f} [{d[1]:+5.1f},{d[2]:+5.1f}]"
        print(line)
    w = [r.get("W_dyn") for r in rows if r.get("W_dyn")]
    if w:
        w = np.array(w)
        print(f"   dyn W: mean {w.mean():.3f} median {np.median(w):.3f}  at 0.15 {100*np.mean(w<=0.151):.0f}%  at 0.50 {100*np.mean(w>=0.499):.0f}%")
