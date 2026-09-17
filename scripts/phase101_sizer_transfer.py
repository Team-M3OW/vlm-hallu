"""
Phase 101: the FROZEN sizer, on whichever model it is pointed at.

One implementation, two models, so the configuration cannot drift between them. Everything here is
fixed by PREREG_DCR_SIZER.md and must not be tuned per model.

GRID DEVIATION, declared: the prereg named W in {0.15,0.25,0.35,0.50,0.70}, but phase 97 (launched
BEFORE the prereg was written) ran only {0.15,0.25,0.35} on Qwen2-VL. Rather than widen Qwen2-VL,
this script restricts BOTH models to the common 3-value grid, which is the conservative direction --
fewer choices give the sizer less room. On Qwen3-VL the 5-value sizer already sent 179/191 items to
one of these three, so little is lost. The 5-value Qwen3-VL number is reported alongside as context,
never as the primary.

usage: python3 phase101_sizer_transfer.py qwen3 | qwen2
"""
import json, sys
import numpy as np
from sklearn.linear_model import LogisticRegression

WHICH = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
D = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"
CFG = {
    "qwen3": (f"{D}/phase78_w_sweep.jsonl",           f"{D}/phase32_conditional.jsonl", "feats_in_rows"),
    "qwen2": (f"{D}/phase97m_merged_qwen2vl.jsonl", f"{D}/phase100_qwen2vl_feats.json", "feats_json"),
}
SWEEP, FEATSRC, MODE = CFG[WHICH]
WS = ["0.15","0.25","0.35","0.5","0.7"] if "--wide" in sys.argv else ["0.15","0.25","0.35"]
FN = ["peak", "peak_over_median", "top1_frac", "top5_frac", "entropy", "entropy_norm"]
K = 5

rows = {json.loads(l)["question_id_full"]: json.loads(l) for l in open(SWEEP)}
if MODE == "feats_json":
    feats = json.load(open(FEATSRC))
else:
    feats = {}
    for l in open(FEATSRC):
        r = json.loads(l)
        if "attn_feats" in r:
            feats[r["question_id_full"]] = r["attn_feats"]
ids = sorted(q for q in rows if q in feats)
n = len(ids)
print(f"{WHICH}: n = {n} items (sweep {len(rows)}, feats {len(feats)})")

X = np.array([[feats[q][f] for f in FN] for q in ids], float)
X = (X - X.mean(0)) / (X.std(0) + 1e-9)
def arm(a): return np.array([int(np.argmax(rows[q]["probs"][a]) == rows[q]["label"]) for q in ids], float)
def tok(a): return np.array([rows[q]["realized_tokens"][a] for q in ids], float)
C = {w: arm(f"head@{w}") for w in WS}
u300, u600 = arm("uniform@300"), arm("uniform@600")
cat = np.array([rows[q]["category"] for q in ids])

rng = np.random.default_rng(101)
def ci(d, B=4000):
    idx = rng.integers(0, n, (B, n)); b = d[idx].mean(1)
    return d.mean()*100, np.percentile(b, 2.5)*100, np.percentile(b, 97.5)*100

order = np.arange(n); np.random.default_rng(7).shuffle(order)
fold = np.zeros(n, int)
for i, o in enumerate(order): fold[o] = i % K

P = np.zeros((n, len(WS)))
for j, w in enumerate(WS):
    for k in range(K):
        tr, te = fold != k, fold == k
        if C[w][tr].std() == 0:
            P[te, j] = C[w][tr].mean(); continue
        P[te, j] = LogisticRegression(max_iter=2000, C=0.5).fit(X[tr], C[w][tr]).predict_proba(X[te])[:, 1]
pick = P.argmax(1)
sizer = np.array([C[WS[pick[i]]][i] for i in range(n)])

tm = np.array([tok(f"head@{WS[pick[i]]}")[i] for i in range(n)]) + tok("uniform@300")
tb = tok("uniform@600")
dr = abs(tm.mean() - tb.mean()) / tb.mean()
print(f"budget: sizer {tm.mean():.0f} tok vs bar {tb.mean():.0f} tok -> drift {100*dr:.1f}% "
      f"{'OK' if dr <= .10 else 'VOID'}")

print(f"\n{'arm':>26} {'acc':>7}")
for nm, a in [("uniform@300 (pass 1)", u300), ("uniform@600 (THE BAR)", u600)] + \
             [(f"fixed W={w}", C[w]) for w in WS] + [("SIZER (OOF)", sizer)]:
    print(f"{nm:>26} {a.mean()*100:6.1f}%")
print(f"  sizer picks: " + str({WS[j]: int((pick == j).sum()) for j in range(len(WS))}))

print(f"\n{'contrast':>34} {'delta':>8}  95% CI")
for nm, d in [("SIZER - uniform@600  (PRIMARY)", sizer - u600),
              ("SIZER - fixed W=0.25 (secondary)", sizer - C["0.25"]),
              ("fixed W=0.25 - bar   (incumbent)", C["0.25"] - u600),
              ("fixed W=0.15 - bar   (deployed)", C["0.15"] - u600),
              ("SIZER - pass 1", sizer - u300)]:
    m, lo, hi = ci(d)
    print(f"{nm:>34} {m:+7.1f}pp  [{lo:+5.1f},{hi:+5.1f}] {'CLEARS' if lo > 0 else ''}")

print(f"\n{'category':>22} {'n':>4} {'bar':>7} {'sizer':>7} {'delta':>8}")
for c in sorted(set(cat)):
    m_ = cat == c
    print(f"{c:>22} {m_.sum():4d} {u600[m_].mean()*100:6.1f}% {sizer[m_].mean()*100:6.1f}% "
          f"{(sizer-u600)[m_].mean()*100:+7.1f}pp")
