"""
Phase 122: the head's TRAINING TARGET. Coverage-at-W=0.25 regression is a proxy; four alternatives,
all on disk, OOF on both models, scored on top-1 coverage at W=0.25.

  T0  regress coverage@0.25                     <- INCUMBENT
  T1  classify covers@0.25 (>=0.5), balanced class weights, score = P(covers)
  T2  regress mean of coverage at W in {0.15,0.25,0.35}  (multi-scale, smoother target)
  T3  regress coverage@0.15 (the deployed window's target), scored at 0.25
  T4  regress coverage@0.25 with squared target (emphasises full covers over partial)

PRE-REGISTERED: primary contrast each Tk - T0, both models, CI clear of zero and > ~1pp noise floor
(SS14Z) on BOTH to adopt. Otherwise T0 stands.
"""
import json, sys
import numpy as np
sys.path.insert(0, "/home/kavinder/ARNABI_ARSH/vlm-hallu/scripts")
import phase70_rerank_head as P70
import phase80a_qwen2vl_head as P80
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
D = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"

def cov_at(rows, G, W):
    out = []
    for gi, r in enumerate(rows):
        gh, gw = r["grid"]; n = r["n_img_tokens"]
        yy, xx = np.mgrid[0:gh, 0:gw]; fy, fx = ((yy+.5)/gh).ravel(), ((xx+.5)/gw).ravel()
        P70.W = W
        out.append(np.array([P70.coverage(float(fx[i]), float(fy[i]), r["gt_box_frac"]) for i in range(n)]))
    return np.concatenate(out)

def oof(X, T, G, kind, seeds=3, neg_per_item=30):
    P = np.zeros(len(T)); gs = np.unique(G)
    for s in range(seeds):
        rng = np.random.default_rng(700+s)
        perm = {g: i for i, g in enumerate(rng.permutation(gs))}; Gp = np.vectorize(perm.get)(G)
        for tr, te in GroupKFold(5).split(X, T, Gp):
            ttr = T[tr]; pos = tr[ttr > 0]; negpool = tr[ttr <= 0]
            k = min(len(negpool), neg_per_item*len(np.unique(G[tr]))); neg = rng.choice(negpool, size=k, replace=False)
            sub = np.concatenate([pos, neg])
            if kind == "clf":
                m = HistGradientBoostingClassifier(max_depth=4, max_iter=150, learning_rate=0.10, random_state=s, class_weight="balanced")
                m.fit(X[sub], (T[sub] >= 0.5).astype(int)); P[te] += m.predict_proba(X[te])[:, 1]
            else:
                m = HistGradientBoostingRegressor(max_depth=4, max_iter=150, learning_rate=0.10, random_state=s)
                m.fit(X[sub], T[sub]); P[te] += m.predict(X[te])
    return P/seeds

for name, build in [("Qwen3-VL", P70.build), ("Qwen2-VL", P80.build)]:
    P70.W = 0.25
    r = build(); X, G, rows = r[0], r[2], r[4]
    Y25 = cov_at(rows, G, 0.25); Y15 = cov_at(rows, G, 0.15); Y35 = cov_at(rows, G, 0.35)
    P70.W = 0.25
    RING = []
    for q in rows:
        gh, gw = q["grid"]; m = np.zeros((gh, gw), bool)
        if gh > 2 and gw > 2: m[1:-1, 1:-1] = True
        else: m[:] = True
        RING.append(m.ravel())
    def top1(P): return np.array([float(Y25[G == gi][int(np.argmax(np.where(RING[gi], P[G == gi], -1e9)))] >= 0.5) for gi in np.unique(G)])
    targets = {"T0_cov25": (Y25, "reg"), "T1_clf25": (Y25, "clf"), "T2_multiW": ((Y15+Y25+Y35)/3, "reg"),
               "T3_cov15": (Y15, "reg"), "T4_cov25sq": (Y25**2, "reg")}
    res = {k: top1(oof(X, t, G, kind)) for k, (t, kind) in targets.items()}
    n = len(rows); rng = np.random.default_rng(122)
    def ci(d):
        m = len(d); b = np.array([d[rng.integers(0, m, m)].mean() for _ in range(6000)])
        return d.mean()*100, np.percentile(b, 2.5)*100, np.percentile(b, 97.5)*100
    print(f"\n{name}: n={n}")
    for k in targets:
        a, lo, hi = ci(res[k]-res["T0_cov25"])
        print(f"  {k:>11} {res[k].mean()*100:6.1f}%   vs T0 {a:+5.1f} [{lo:+5.1f},{hi:+5.1f}]{' CLEARS' if lo>0 else ''}")
