"""
Phase 99: PER-ITEM WINDOW SIZE FROM PASS-1 ATTENTION CONCENTRATION.

Phase 78 measured +11.0pp of stable headroom for a per-item sizer and ruled out TARGET SIZE as the
predictor. The untried predictor is how CONCENTRATED the attention blob is -- and phase 32 already
stores exactly that, free, from pass 1: top1_frac, top5_frac, peak_over_median, entropy_norm.

Intuition the run will test: a tight blob means the proposer is confident about a small region, so a
small window suffices; a diffuse blob means the aim is uncertain, so a wider window is more forgiving
of a near-miss. Phase 78 found the counter-intuitive direction at the FIXED level (better aim wants a
WIDER window because it still misses half the time), so the sign is genuinely open.

PRE-REGISTERED:
  PRIMARY   OOF per-item W vs FIXED W=0.25 (the held-out-validated constant from phase 90).
            Any gain must be measured against that, not against the deployed 0.15.
  BAR       uniform@600 at matched compute.
  CEILING   oracle-W (best W per item chosen with the answer key) -- optimistic by construction.
  GUARD     phase 78's monotonicity check and phase 90's held-out discipline: the predictor is fit
            inside each training fold only, and NO feature derived from the label or the GT box.
  DECISION  OOF per-item W must beat fixed W=0.25 with a CI clear of zero. Otherwise the sizer
            joins the closed list and the +11.0pp headroom is declared unreachable from free signals.
"""
import json, numpy as np

SW, GT = "data/phase78_w_sweep.jsonl", "data/phase32_conditional.jsonl"
WS = ["0.15", "0.25", "0.35", "0.5", "0.7"]
rows = {json.loads(l)["question_id_full"]: json.loads(l) for l in open(SW)}
feats = {}
for l in open(GT):
    r = json.loads(l)
    if "attn_feats" in r: feats[r["question_id_full"]] = r["attn_feats"]
ids = [q for q in rows if q in feats]
FN = ["peak", "peak_over_median", "top1_frac", "top5_frac", "entropy", "entropy_norm"]
X = np.array([[feats[q][f] for f in FN] for q in ids], float)
X = (X - X.mean(0)) / (X.std(0) + 1e-9)
lab = np.array([rows[q]["label"] for q in ids])
C = {w: np.array([int(np.argmax(rows[q]["probs"][f"head@{w}"]) == rows[q]["label"]) for q in ids], float) for w in WS}
u300 = np.array([int(np.argmax(rows[q]["probs"]["uniform@300"]) == rows[q]["label"]) for q in ids], float)
u600 = np.array([int(np.argmax(rows[q]["probs"]["uniform@600"]) == rows[q]["label"]) for q in ids], float)
cat = np.array([rows[q]["category"] for q in ids])
n = len(ids)
rng = np.random.default_rng(99)
def ci(d, B=4000):
    idx = rng.integers(0, n, (B, n)); b = d[idx].mean(1)
    return d.mean()*100, np.percentile(b, 2.5)*100, np.percentile(b, 97.5)*100

print(f"n = {n}")
print(f"{'fixed W':>10} {'acc':>7}")
for w in WS: print(f"{w:>10} {C[w].mean()*100:6.1f}%")
orc = np.max(np.stack([C[w] for w in WS]), 0)
print(f"{'oracle-W':>10} {orc.mean()*100:6.1f}%   (chosen with the answer key -- optimistic)")

# OOF per-item W: 5-fold, multinomial logistic over the free features, target = best W for that item
K = 5; order = np.arange(n); rng2 = np.random.default_rng(7); rng2.shuffle(order)
fold = np.zeros(n, int)
for i, o in enumerate(order): fold[o] = i % K
Y = np.argmax(np.stack([C[w] for w in WS]), 0)     # index of a best W (ties -> first)
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingClassifier
    models = [("logistic", lambda: LogisticRegression(max_iter=2000, C=0.5)),
              ("gbt", lambda: GradientBoostingClassifier(n_estimators=60, max_depth=2, random_state=0))]
except ImportError:
    models = []
for mname, mk in models:
    pred = np.empty(n, int)
    for k in range(K):
        tr, te = fold != k, fold == k
        m = mk().fit(X[tr], Y[tr])
        pred[te] = m.predict(X[te])
    got = np.array([C[WS[pred[i]]][i] for i in range(n)])
    chosen = {WS[j]: int((pred == j).sum()) for j in range(len(WS))}
    m1, lo1, hi1 = ci(got - C["0.25"]); m2, lo2, hi2 = ci(got - u600)
    print(f"\n  {mname}: acc {got.mean()*100:.1f}%   picks {chosen}")
    print(f"    vs FIXED W=0.25 (PRIMARY)  {m1:+5.1f}pp [{lo1:+5.1f},{hi1:+5.1f}] {'CLEARS' if lo1>0 else ''}")
    print(f"    vs uniform@600 (the bar)   {m2:+5.1f}pp [{lo2:+5.1f},{hi2:+5.1f}] {'CLEARS' if lo2>0 else ''}")

m, lo, hi = ci(C["0.25"] - u600)
print(f"\n  reference: fixed W=0.25 - bar  {m:+5.1f}pp [{lo:+5.1f},{hi:+5.1f}]")
m, lo, hi = ci(orc - C["0.25"])
print(f"  reference: oracle-W - fixed    {m:+5.1f}pp [{lo:+5.1f},{hi:+5.1f}]  (the headroom being chased)")

# --- corrected target: the argmax-over-ties label above collapses to W=0.15 --------------
# Fit ONE correctness model per W, then pick the W with the highest predicted P(correct).
print("\n--- per-W correctness models, pick argmax predicted P(correct) (OOF) ---")
from sklearn.linear_model import LogisticRegression
disc = (np.stack([C[w] for w in WS]).std(0) > 0)          # items where W actually matters
print(f"  items where W changes the outcome: {disc.sum()}/{n}")
for tag, mask in [("all items", np.ones(n, bool)), ("W-discriminative only", disc)]:
    Pm = np.zeros((n, len(WS)))
    for j, w in enumerate(WS):
        for k in range(K):
            tr = (fold != k) & mask
            te = fold == k
            if C[w][tr].std() == 0:
                Pm[te, j] = C[w][tr].mean(); continue
            m = LogisticRegression(max_iter=2000, C=0.5).fit(X[tr], C[w][tr])
            Pm[te, j] = m.predict_proba(X[te])[:, 1]
    pick = Pm.argmax(1)
    got = np.array([C[WS[pick[i]]][i] for i in range(n)])
    ch = {WS[j]: int((pick == j).sum()) for j in range(len(WS))}
    m1, lo1, hi1 = ci(got - C["0.25"]); m2, lo2, hi2 = ci(got - u600)
    print(f"  [{tag}] acc {got.mean()*100:.1f}%  picks {ch}")
    print(f"      vs FIXED W=0.25 {m1:+5.1f}pp [{lo1:+5.1f},{hi1:+5.1f}] {'CLEARS' if lo1>0 else ''}"
          f"   vs bar {m2:+5.1f}pp [{lo2:+5.1f},{hi2:+5.1f}] {'CLEARS' if lo2>0 else ''}")

# how much of the oracle-W headroom is max-of-5 selection noise?
print("\n--- is the oracle-W headroom real, or max-of-5 noise? ---")
acc = np.array([C[w].mean() for w in WS])
sim = []
for _ in range(2000):
    fake = np.stack([(rng.random(n) < a).astype(float) for a in acc])   # independent arms, same marginals
    sim.append(fake.max(0).mean())
print(f"  observed oracle-W {orc.mean()*100:.1f}%   independent-arms simulation "
      f"{np.mean(sim)*100:.1f}% [{np.percentile(sim,2.5)*100:.1f},{np.percentile(sim,97.5)*100:.1f}]")
print("  (arms are strongly correlated, so the simulation is an UPPER bound on what noise can fake;")
print("   observed BELOW it means max-of-5 inflation is not ruled out by this test alone.)")
