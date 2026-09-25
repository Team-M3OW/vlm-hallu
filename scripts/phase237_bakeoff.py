"""Phase 237b: which label-free map statistic is the best single/cross gate, and the best policy?
Reports AUROC(signal -> single) per cell and the policy (route by median, fallback AVR) per signal."""
import json, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmarks as B
D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rng = np.random.default_rng(237)
SIGN = {"disp": +1, "ent": -1, "top5": +1, "pom": +1, "band_disp": +1, "band_ent": -1,
        "ratio": +1, "agree": -1, "headmax": +1, "vrh": +1}
CELLS = [("qwen3_2b", "vstar"), ("qwen3_2b", "hr4k"), ("qwen3_2b", "cvbench"), ("qwen3_2b", "realworldqa"),
         ("qwen2_7b", "vstar"), ("qwen2_7b", "hr4k"), ("qwen2_7b", "cvbench"), ("qwen2_7b", "realworldqa")]
SINGLE_LABELS = {"vstar": {"direct_attributes"}, "hr4k": {"single"}}


def auc(x, y):
    x = np.asarray(x, float); y = np.asarray(y, bool)
    r = np.argsort(np.argsort(x)); n1, n0 = y.sum(), (~y).sum()
    return (r[y].sum() - n1 * (n1 - 1) / 2) / max(n1 * n0, 1)


def ok(rec, a):
    if a in rec.get("probs", {}):
        return 1.0 if int(np.argmax(rec["probs"][a])) == int(rec["gold"]) else 0.0
    if a in rec.get("preds", {}):
        p = rec["preds"][a]; gs = rec["gold"] if isinstance(rec["gold"], (list, tuple)) else [rec["gold"]]
        return 1.0 if any(B.norm(g) and B.norm(g) == B.norm(p) for g in gs) else 0.0
    return None


def boot(d, Bn=6000):
    d = np.asarray(d, float); n = len(d)
    if n < 8: return float("nan"), float("nan"), float("nan")
    m = d[rng.integers(0, n, (Bn, n))].mean(1)
    return d.mean() * 100, np.percentile(m, 2.5) * 100, np.percentile(m, 97.5) * 100


print(f"{'cell':16s}{'signal':10s}{'AUROC':>7s}{'policy-bar':>22s}{'single':>20s}{'cross':>20s}")
for mk, bk in CELLS:
    fs = f"{D}/data/phase237_sig_{mk}_{bk}.jsonl"
    f25 = f"{D}/data/phase225_{mk}_{bk}.jsonl"
    if not (os.path.exists(fs) and os.path.exists(f25)):
        print(f"{mk+'/'+bk:16s}{'-- running --':>10s}"); continue
    sig = {json.loads(l)["qid"]: json.loads(l) for l in open(fs)}
    grid = {json.loads(l)["qid"]: json.loads(l) for l in open(f25)}
    qids = [q for q in grid if q in sig]
    if len(qids) < 50: continue
    rows = [grid[q] for q in qids]
    arms = set(a for r in rows for a in list(r.get("probs", {})) + list(r.get("preds", {})))
    alt = "avr" if "avr" in arms else "uniform@lo"
    bar = np.array([ok(r, "uniform@lo") for r in rows]); dwa = np.array([ok(r, "dwa_t") for r in rows]); av = np.array([ok(r, alt) for r in rows])
    labels = SINGLE_LABELS.get(bk)
    for name, d in SIGN.items():
        x = np.array([sig[q][name] * d for q in qids], float)
        if np.all(np.isnan(x)): continue
        x = np.nan_to_num(x, nan=np.nanmedian(x))
        tau = np.median(x); route = x > tau
        pol = np.where(route, dwa, av) - bar
        a_ = auc(x, np.array([r.get("stratum") in labels for r in rows])) if labels else float("nan")
        m, lo, hi = boot(pol)
        st = ""
        if labels:
            sing = np.array([r.get("stratum") in labels for r in rows])
            ms, _, _ = boot(pol[sing]); mc, _, _ = boot(pol[~sing])
            st = f"{ms:+20.1f}{mc:+20.1f}"
        else:
            st = f"{'':>20s}{'':>20s}"
        print(f"{mk+'/'+bk:16s}{name:10s}{a_:7.3f}{m:+15.1f}[{lo:+.1f},{hi:+.1f}]{st}")
    print()
