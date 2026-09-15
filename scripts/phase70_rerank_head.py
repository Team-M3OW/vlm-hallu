"""
Phase 70: A LEARNED RE-RANKING HEAD ON THE ATTENTION MAP. Plug-and-play, internal, no crop, no
extra forward pass.

THE GAP IT ATTACKS
------------------
The proposer's top-1 rule is throwing away a correct answer it already has. Measured in SS11A:

    deployed argmax of the ring-masked block-16-26 map   covers the evidence  40.5% of the time
    oracle selection among that same map's top-5          covers               61.5%
    oracle among top-10                                   covers               66.1%
    the GT cell's own rank                                gt_pct 0.037 (top 3.4% of cells)

So the ranking contains the evidence and the ARGMAX is usually a distractor. Everything upstream
says why: the serialization sink puts 2.0-4.5x excess mass on the trailing row boundary in four
architectures (SS5), present at L0, and the deployed read-out collapses 28 layers into one mean and
then takes a max. A learned readout can know about the sink; a max cannot.

WHAT IS AND IS NOT BEING CLAIMED
--------------------------------
This is a fix to a READ-OUT, not a new source of information. It cannot beat oracle-among-top-k,
which is the ceiling of any function of this map. It is a component: it makes a proposal better, and
a better proposal only converts to accuracy if something downstream acts on it. That boundary is
stated here rather than discovered in review.

FEATURES (per cell, all from one pass that already happened)
------------------------------------------------------------
    depth profile     the cell's normalised attention at each of 28 layers -- the information the
                      block-mean destroys
    within-item rank  its rank at each layer, so the head sees ordering as well as magnitude
    neighbourhood     3x3 mean, because evidence is spatially extended and single cells are noisy
    geometry          row/col fraction, distance to centre, distance to nearest border, and an
                      explicit LAST-COLUMN / LAST-ROW indicator so the head can learn the sink

CONTROLS, FIXED BEFORE THE RUN
------------------------------
    deployed argmax        the incumbent; the number to beat is 40.5%
    geometry only          a head given ONLY position. V*Bench targets are not uniformly placed, so
                           a centre prior alone can look like a result. If geometry-only matches the
                           full head, the head learned the dataset, not the model.
    shuffled labels        must land at chance.
    oracle among top-k     the ceiling of any re-ranking of this map.
All predictions are OUT-OF-FOLD with GROUPED folds (an item's 300 cells are never split across
train and test), 5-fold x 5 repeats.

NO GPU: every attention map is read from phase30c_attn_maps_all.jsonl, already on disk.
"""
import json
import os
import statistics as st

import numpy as np

D = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"
OUT = os.path.join(D, "phase70_rerank_head.json")
BLOCK = list(range(16, 27))
W = 0.15
COV_HIT = 0.5          # "covers the evidence" = at least half the GT box inside the window


def coverage(cx, cy, gt):
    x0, x1, y0, y1 = cx - W / 2, cx + W / 2, cy - W / 2, cy + W / 2
    if x0 < 0: x0, x1 = 0.0, W
    if y0 < 0: y0, y1 = 0.0, W
    if x1 > 1: x0, x1 = 1 - W, 1.0
    if y1 > 1: y0, y1 = 1 - W, 1.0
    gx0, gy0, gx1, gy1 = gt
    inter = max(0.0, min(gx1, x1) - max(gx0, x0)) * max(0.0, min(gy1, y1) - max(gy0, y0))
    return inter / max((gx1 - gx0) * (gy1 - gy0), 1e-12)


def build():
    rows = [json.loads(l) for l in open(f"{D}/phase30c_attn_maps_all.jsonl")]
    X, Y, G, DEP, CELL = [], [], [], [], []
    for gi, r in enumerate(rows):
        gh, gw = r["grid"]
        n = r["n_img_tokens"]
        nL = len([k for k in r["attn"] if k.startswith("L")])
        A = np.stack([np.asarray(r["attn"][f"L{i}"], dtype=float) for i in range(nL)])   # nL x n
        A = A / np.maximum(A.sum(1, keepdims=True), 1e-12)
        M = A.reshape(nL, gh, gw)
        dep = M[BLOCK].mean(0)                       # the deployed block-mean map
        # rank of each cell within its layer (0 = strongest)
        R = np.argsort(np.argsort(-A, axis=1), axis=1) / max(n - 1, 1)
        R = R.reshape(nL, gh, gw)
        # 3x3 neighbourhood mean of the deployed map
        pad = np.pad(dep, 1, mode="edge")
        nb = sum(pad[i:i + gh, j:j + gw] for i in range(3) for j in range(3)) / 9.0
        yy, xx = np.mgrid[0:gh, 0:gw]
        fy, fx = (yy + .5) / gh, (xx + .5) / gw
        feats = np.concatenate([
            M.reshape(nL, -1).T,                       # depth profile           (nL)
            R.reshape(nL, -1).T,                       # within-layer rank       (nL)
            nb.reshape(-1, 1), dep.reshape(-1, 1),     # neighbourhood, deployed (2)
            fx.reshape(-1, 1), fy.reshape(-1, 1),      # position                (2)
            np.sqrt((fx - .5) ** 2 + (fy - .5) ** 2).reshape(-1, 1),        # dist to centre
            np.minimum(np.minimum(fx, 1 - fx), np.minimum(fy, 1 - fy)).reshape(-1, 1),
            (xx == gw - 1).astype(float).reshape(-1, 1),   # LAST COLUMN  (the sink)
            (yy == gh - 1).astype(float).reshape(-1, 1),   # last row
            (xx == 0).astype(float).reshape(-1, 1),
        ], axis=1)
        cov = np.array([coverage(float(fx.flat[i]), float(fy.flat[i]), r["gt_box_frac"])
                        for i in range(n)])
        X.append(feats); Y.append(cov); G.append(np.full(n, gi))
        DEP.append(dep.flatten()); CELL.append((gh, gw))
    NGEO = 7   # the trailing geometry columns, for the geometry-only control
    return (np.vstack(X), np.concatenate(Y), np.concatenate(G),
            [d for d in DEP], rows, NGEO)


def evaluate(score, G, Y, DEP, ring_mask):
    """Top-1 coverage of a scoring function, per item, with the same ring mask the proposer uses."""
    hit, cov, dep_hit, dep_cov = [], [], [], []
    for gi in np.unique(G):
        m = G == gi
        s, y, d, rm = score[m], Y[m], DEP[gi], ring_mask[gi]
        s = np.where(rm, s, -1e9)
        d = np.where(rm, d, -1e9)
        j, jd = int(np.argmax(s)), int(np.argmax(d))
        hit.append(1.0 * (y[j] >= COV_HIT)); cov.append(y[j])
        dep_hit.append(1.0 * (y[jd] >= COV_HIT)); dep_cov.append(y[jd])
    return st.mean(hit), st.mean(cov), st.mean(dep_hit), st.mean(dep_cov)


def oof(X, Y, G, seeds=3, neg_per_item=30):
    """Out-of-fold, GROUPED by item. Training rows are negative-subsampled: every covering cell
    plus `neg_per_item` random non-covering cells. Standard for ranking, and it keeps the fit
    seconds rather than minutes on a loaded box. EVALUATION is always on ALL cells."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import GroupKFold
    P = np.zeros(len(Y))
    gs = np.unique(G)
    for s in range(seeds):
        rng = np.random.default_rng(700 + s)
        perm = {g: i for i, g in enumerate(rng.permutation(gs))}
        Gp = np.vectorize(perm.get)(G)
        for tr, te in GroupKFold(5).split(X, Y, Gp):
            ytr = Y[tr]
            pos = tr[ytr > 0]
            negpool = tr[ytr <= 0]
            k = min(len(negpool), neg_per_item * len(np.unique(G[tr])))
            neg = rng.choice(negpool, size=k, replace=False)
            sub = np.concatenate([pos, neg])
            m = HistGradientBoostingRegressor(max_depth=4, max_iter=150, learning_rate=0.10,
                                              random_state=s)
            m.fit(X[sub], Y[sub])
            P[te] += m.predict(X[te])
            print('.', end='', flush=True)
    return P / seeds


def main():
    X, Y, G, DEP, rows, NGEO = build()
    ring = []
    for r in rows:
        gh, gw = r["grid"]
        m = np.zeros((gh, gw), dtype=bool)
        if gh > 2 and gw > 2:
            m[1:-1, 1:-1] = True
        else:
            m[:] = True
        ring.append(m.flatten())
    print(f"{len(rows)} items x {X.shape[0]//len(rows)} cells, {X.shape[1]} features/cell\n")

    # ---------- ceilings and the incumbent, on the ring-masked map
    dep_hit, dep_cov, orc = [], [], {k: [] for k in (1, 3, 5, 10, 20)}
    for gi, r in enumerate(rows):
        m = G == gi
        y, d, rm = Y[m], DEP[gi], ring[gi]
        d = np.where(rm, d, -1e9)
        order = np.argsort(-d)
        dep_hit.append(1.0 * (y[order[0]] >= COV_HIT)); dep_cov.append(y[order[0]])
        for k in orc:
            orc[k].append(1.0 * (max(y[order[:k]]) >= COV_HIT))
    print(f"deployed argmax (incumbent)   {100*st.mean(dep_hit):5.1f}% cover  "
          f"(mean coverage {st.mean(dep_cov):.3f})")
    for k in sorted(orc):
        print(f"oracle among deployed top-{k:<3d}    {100*st.mean(orc[k]):5.1f}%")

    res = {"incumbent": st.mean(dep_hit), "oracle_topk": {k: st.mean(v) for k, v in orc.items()},
           "arms": {}}

    rng = np.random.default_rng(70)
    arms = {
        "full head (attention + geometry)": (X, Y),
        "geometry only (centre-prior CONTROL)": (X[:, -NGEO:], Y),
        "attention only (no geometry)": (X[:, :-NGEO], Y),
        "shuffled labels (CONTROL)": (X, None),
    }
    print()
    for nm, (Xi, Yi) in arms.items():
        print(f"  fitting {nm} ", end="", flush=True)
        if Yi is None:            # shuffle coverage BETWEEN items, preserving within-item structure
            Yi = Y.copy()
            gs = rng.permutation(np.unique(G))
            newY = np.zeros_like(Y)
            for a, b in zip(np.unique(G), gs):
                newY[G == a] = Y[G == b][:int((G == a).sum())] if (G == b).sum() >= (G == a).sum() \
                    else np.resize(Y[G == b], int((G == a).sum()))
            Yi = newY
        P = oof(Xi, Yi, G)
        h, c, _, _ = evaluate(P, G, Y, DEP, ring)
        res["arms"][nm] = {"cover": h, "mean_cov": c}
        print(f"\n  {nm:38} {100*h:5.1f}% cover   (mean coverage {c:.3f})   "
              f"{100*(h-st.mean(dep_hit)):+5.1f}pp vs incumbent")

    json.dump(res, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")
    print("\n" + "=" * 72)
    full = res["arms"]["full head (attention + geometry)"]["cover"]
    geo = res["arms"]["geometry only (centre-prior CONTROL)"]["cover"]
    shuf = res["arms"]["shuffled labels (CONTROL)"]["cover"]
    inc = res["incumbent"]
    print(f"VERDICT  incumbent {100*inc:.1f}%  ->  head {100*full:.1f}%  "
          f"(ceiling top-5 {100*res['oracle_topk'][5]:.1f}%)")
    print(f"         geometry-only control {100*geo:.1f}% | shuffled control {100*shuf:.1f}%")
    if full - inc > 0.05 and full - geo > 0.03:
        print("  => THE HEAD WORKS and it is not a centre prior. The map's own ranking was being")
        print("     thrown away by the max, and a learned read-out recovers part of it.")
    elif full - geo <= 0.03 and full - inc > 0.05:
        print("  => GAINS ARE A CENTRE PRIOR, not a model-internal signal. Not a method.")
    else:
        print("  => NO GAIN. The deployed argmax is already the best function of this map.")


if __name__ == "__main__":
    main()
