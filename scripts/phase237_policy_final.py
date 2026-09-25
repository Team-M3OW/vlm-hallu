"""Phase 237c: the final map-only policy and its table.

Gate = AND of three label-free signals, each above its item-set median (no tuned parameter):
  disp       ring-masked top-1 mass share of the block-mean map
  band_disp  the same on the read-out-band mean map (L17-21 / L19-23)
  vrh        block-mean mass inside the proposed ridge crop's W=0.25 window
All three must agree before the policy spends the budget on a crop; otherwise it keeps the whole
image and buys resolution (AVR where the checkpoint has a ladder). Prints pooled and per-stratum
policy - bar with paired bootstrap CIs for every (model, benchmark) cell with data."""
import json, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmarks as B
D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rng = np.random.default_rng(237)
MODELS = ["qwen3_2b", "qwen2_7b"]
BENCH = ["vstar", "hr4k", "cvbench", "realworldqa"]


def rank(x):
    r = np.argsort(np.argsort(x)).astype(float)
    return r / max(len(x) - 1, 1)


def ok(rec, a):
    if a in rec.get("probs", {}):
        return 1.0 if int(np.argmax(rec["probs"][a])) == int(rec["gold"]) else 0.0
    if a in rec.get("preds", {}):
        p = rec["preds"][a]; gs = rec["gold"] if isinstance(rec["gold"], (list, tuple)) else [rec["gold"]]
        return 1.0 if any(B.norm(g) and B.norm(g) == B.norm(p) for g in gs) else 0.0
    return None


def boot(d, Bn=8000):
    d = np.asarray(d, float); n = len(d)
    if n < 8: return float("nan"), float("nan"), float("nan"), n
    m = d[rng.integers(0, n, (Bn, n))].mean(1)
    return d.mean() * 100, np.percentile(m, 2.5) * 100, np.percentile(m, 97.5) * 100, n


print(f"{'cell':18s}{'route%':>7s}{'policy-bar':>24s}{'single':>24s}{'cross':>24s}")
for mk in MODELS:
    for bk in BENCH:
        fs = f"{D}/data/phase237_sig_{mk}_{bk}.jsonl"; f25 = f"{D}/data/phase225_{mk}_{bk}.jsonl"
        if not (os.path.exists(fs) and os.path.exists(f25)): continue
        sig = {json.loads(l)["qid"]: json.loads(l) for l in open(fs)}
        grid = {json.loads(l)["qid"]: json.loads(l) for l in open(f25)}
        q = [x for x in grid if x in sig and all(ok(grid[x], a) is not None for a in ("uniform@lo", "dwa_t", "avr"))]
        if len(q) < 50: continue
        g = [grid[x] for x in q]; s = [sig[x] for x in q]
        bar = np.array([ok(r, "uniform@lo") for r in g]); dwa = np.array([ok(r, "dwa_t") for r in g]); av = np.array([ok(r, "avr") for r in g])
        route = (rank(np.array([x["disp"] for x in s])) > 0.5) & (rank(np.array([x["band_disp"] for x in s])) > 0.5) & (rank(np.array([x["vrh"] for x in s])) > 0.5)
        pol = np.where(route, dwa, av) - bar
        m, lo, hi, _ = boot(pol)
        lab = np.array([r.get("stratum") in ("direct_attributes", "single") for r in g])
        clab = np.array([r.get("stratum") in ("relative_position", "cross") for r in g])
        st = f"{'':>24s}"
        if lab.sum() > 20:
            ms, msl, msh, _ = boot(pol[lab]); st = f"{ms:+7.1f}[{msl:+5.1f},{msh:+5.1f}]"
        if clab.sum() > 20:
            mc, mcl, mch, _ = boot(pol[clab]); st += f"{mc:+7.1f}[{mcl:+5.1f},{mch:+5.1f}]"
        print(f"{mk+'/'+bk:18s}{100*route.mean():7.0f}{m:+16.1f}[{lo:+.1f},{hi:+.1f}]{st}")
