"""Phase 238: leave-one-group-in ablation of the DWA feature blocks.

Fits the deployed estimator (OOF GroupKFold(5) x 3 group permutations, lambda=1, unpenalised
intercept, features standardised per fold) on V* map dumps, once per feature block subset:
  log         -- the L log-attention features only
  log+rank    -- adds the L per-layer descending-rank features
  full        -- adds the 7 neighbourhood/geometry features (the deployed 2L+7 vector)
Reports out-of-fold coverage of the W=0.25 window at the ring-masked arg-max, against the
conventional read-out-band block-mean baseline."""
import json, os, numpy as np
from sklearn.model_selection import GroupKFold
D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W = 0.25
SOURCES = {"qwen3": "phase30c_attn_maps_all.jsonl", "qwen2": "phase74_Qwen2_VL_7B_Instruct.jsonl"}


def cov(cx, cy, gt):
    x0, x1, y0, y1 = cx - W / 2, cx + W / 2, cy - W / 2, cy + W / 2
    if x0 < 0: x0, x1 = 0., W
    if y0 < 0: y0, y1 = 0., W
    if x1 > 1: x0, x1 = 1 - W, 1.
    if y1 > 1: y0, y1 = 1 - W, 1.
    gx0, gy0, gx1, gy1 = gt
    return max(0., min(gx1, x1) - max(gx0, x0)) * max(0., min(gy1, y1) - max(gy0, y0)) / max((gx1 - gx0) * (gy1 - gy0), 1e-12)


def oof(X, Y, G, seed0=700):
    P = np.zeros(len(Y))
    for s in range(3):
        rng = np.random.default_rng(seed0 + s); perm = {g: i for i, g in enumerate(rng.permutation(np.unique(G)))}
        Gp = np.vectorize(perm.get)(G)
        for tr, te in GroupKFold(5).split(X, Y, Gp):
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
            Xt = np.c_[(X[tr] - mu) / sd, np.ones(len(tr))]
            A_ = Xt.T @ Xt + np.eye(Xt.shape[1]); A_[-1, -1] -= 1.0
            w = np.linalg.solve(A_, Xt.T @ Y[tr])
            P[te] += np.c_[(X[te] - mu) / sd, np.ones(len(te))] @ w
    return P / 3


for which, src in SOURCES.items():
    rows = [json.loads(l) for l in open(f"{D}/data/{src}") if l.strip()]
    rows = [r for r in rows if "attn" in r and "grid" in r and r.get("gt_box_frac")]
    NL = len(rows[0]["attn"]); BLK = (round(0.57 * NL), NL - 1)
    print(f"\n{which}: {len(rows)} items, NL={NL}, band L{BLK[0]}-{BLK[1]-1}")
    blocks = {"log": [], "log+rank": [], "full": []}
    Y = []; G = []; offs = []; metas = []
    gi = 0
    for q in rows:
        gh, gw = q["grid"]; nc = gh * gw
        a = np.stack([np.asarray(q["attn"][f"L{l}"], float) for l in range(NL)])
        if a.shape[1] != nc: continue
        a = a / np.maximum(a.sum(1, keepdims=True), 1e-12)
        yy, xx = np.mgrid[0:gh, 0:gw]; fy, fx = ((yy + .5) / gh).ravel(), ((xx + .5) / gw).ravel()
        LA = np.log(a + 1e-12).T
        R = (np.argsort(np.argsort(-a, axis=1), axis=1) / max(nc - 1, 1)).T
        dep = a[BLK[0]:BLK[1]].mean(0).reshape(gh, gw); pad = np.pad(dep, 1, mode="edge")
        nb = (sum(pad[i:i + gh, j:j + gw] for i in range(3) for j in range(3)) / 9.0).ravel()
        geo = np.c_[np.log(nb + 1e-12), fx, fy, np.sqrt((fx - .5) ** 2 + (fy - .5) ** 2),
                    np.minimum(np.minimum(fx, 1 - fx), np.minimum(fy, 1 - fy)),
                    (xx.ravel() == gw - 1).astype(float), (yy.ravel() == gh - 1).astype(float)]
        blocks["log"].append(LA); blocks["log+rank"].append(np.c_[LA, R]); blocks["full"].append(np.c_[LA, R, geo])
        gt = q["gt_box_frac"]
        Y.append(np.array([cov(float(fx[i]), float(fy[i]), gt) for i in range(nc)]))
        G.append(np.full(nc, gi)); offs.append((gh, gw, nc)); metas.append(q); gi += 1
    Y = np.concatenate(Y); G = np.concatenate(G)
    base = []
    for (gh, gw, nc), q in zip(offs, metas):
        a = np.stack([np.asarray(q["attn"][f"L{l}"], float) for l in range(NL)])
        a = a / np.maximum(a.sum(1, keepdims=True), 1e-12)
        dep = a[BLK[0]:BLK[1]].mean(0).reshape(gh, gw)
        rm = np.zeros((gh, gw), bool)
        if gh > 2 and gw > 2: rm[1:-1, 1:-1] = True
        else: rm[:] = True
        iy, ix = np.unravel_index(np.argmax(np.where(rm, dep, -1e9)), dep.shape)
        base.append(cov((ix + .5) / gw, (iy + .5) / gh, q["gt_box_frac"]))
    print(f"  block-mean baseline coverage {np.mean(base):.3f}")
    for name, blk in blocks.items():
        X = np.vstack(blk)
        P = oof(X, Y, G)
        off = 0; covs = []
        for (gh, gw, nc), q in zip(offs, metas):
            sc = P[off:off + nc].reshape(gh, gw); off += nc
            rm = np.zeros((gh, gw), bool)
            if gh > 2 and gw > 2: rm[1:-1, 1:-1] = True
            else: rm[:] = True
            iy, ix = np.unravel_index(np.argmax(np.where(rm, sc, -1e9)), sc.shape)
            covs.append(cov((ix + .5) / gw, (iy + .5) / gh, q["gt_box_frac"]))
        print(f"  {name:9s} ({X.shape[1]:2d} feats)  OOF coverage {np.mean(covs):.3f}  vs block {np.mean(covs)-np.mean(base):+.3f}")
