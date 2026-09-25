"""
Phase 231 (CPU): can the ridge be trained better?

Linear transforms of log-attention (centering, adjacent differences) add no span to a linear model,
so the candidates are NEW INFORMATION, BETTER LABELS, or BETTER FITTING:
  features  FULL63 (deployed) | FULL+H (adds per-head max/std, 119) | AH (log-A + head stats, 84)
  labels    cov25 (deployed: W=0.25 coverage) | soft (Gaussian-smoothed box, sigma=0.05)
  fitting   ridge alpha in {0.3, 1, 3}

Protocol identical to phase 204: modal-grid items, UNMASKED maps, OOF GroupKFold(5) x 3 seeds,
ring-masked argmax, evaluated by mean coverage@0.25 and the >=0.5 rate.
"""
import json, os, numpy as np
from sklearn.model_selection import GroupKFold

D = "/home/kavinder/ARNABI_ARSH/vlm-hallu"; W = 0.25
MAPS = {"qwen3": ("phase30c_attn_maps_all.jsonl", (16, 27), "phase170_perhead_qwen3.npy", "phase170_perhead_qwen3_index.json"),
        "qwen2": ("phase74_Qwen2_VL_7B_Instruct.jsonl", (15, 27), "phase170_perhead_qwen2.npy", "phase170_perhead_qwen2_index.json")}
SEEDS = (700, 701, 702); ALPHAS = (0.3, 1.0, 3.0)


def cov(cx, cy, gt):
    x0, x1, y0, y1 = cx - W / 2, cx + W / 2, cy - W / 2, cy + W / 2
    if x0 < 0: x0, x1 = 0., W
    if y0 < 0: y0, y1 = 0., W
    if x1 > 1: x0, x1 = 1 - W, 1.
    if y1 > 1: y0, y1 = 1 - W, 1.
    gx0, gy0, gx1, gy1 = gt
    return max(0., min(gx1, x1) - max(gx0, x0)) * max(0., min(gy1, y1) - max(gy0, y0)) / max((gx1 - gx0) * (gy1 - gy0), 1e-12)


def oof(X, Y, G, alpha):
    P = np.zeros(len(Y))
    for s in SEEDS:
        gs = np.unique(G); rng = np.random.default_rng(s)
        perm = {g: i for i, g in enumerate(rng.permutation(gs))}
        Gp = np.vectorize(perm.get)(G)
        for tr, te in GroupKFold(5).split(X, Y, Gp):
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
            Xt = np.c_[(X[tr] - mu) / sd, np.ones(len(tr))]
            A_ = Xt.T @ Xt + alpha * np.eye(Xt.shape[1]); A_[-1, -1] -= alpha
            w = np.linalg.solve(A_, Xt.T @ Y[tr])
            P[te] += np.c_[(X[te] - mu) / sd, np.ones(len(te))] @ w
    return P / len(SEEDS)


for which, (fn, BLK, phf, phidx) in MAPS.items():
    rows = [json.loads(l) for l in open(f"{D}/data/{fn}")]
    rows = [r for r in rows if "attn" in r and "grid" in r and "gt_box_frac" in r]
    g0 = max({tuple(r["grid"]) for r in rows}, key=lambda g: sum(1 for r in rows if tuple(r["grid"]) == g))
    sub = [r for r in rows if tuple(r["grid"]) == g0]
    gh, gw = g0; NL = 28; n = len(sub); nc = gh * gw
    yy, xx = np.mgrid[0:gh, 0:gw]; fy, fx = ((yy + .5) / gh).ravel(), ((xx + .5) / gw).ravel()
    A = np.stack([np.stack([np.asarray(r["attn"][f"L{l}"], float) for l in range(NL)]) for r in sub])
    A = A / np.maximum(A.sum(2, keepdims=True), 1e-12)
    covgrid = np.array([[cov(float(fx[c]), float(fy[c]), r["gt_box_frac"]) for c in range(nc)] for r in sub])
    rm = np.zeros((gh, gw), bool); rm[1:-1, 1:-1] = True; rm = rm.ravel()

    PH = np.load(f"{D}/data/{phf}", mmap_mode="r"); pidx = json.load(open(f"{D}/data/{phidx}"))
    ppos = {it["question_id_full"]: (it["offset"], it["n_cells"]) for it in pidx["items"]}
    H = pidx["n_heads"]
    HM = np.zeros((n, NL, nc), np.float32); HS = np.zeros((n, NL, nc), np.float32)
    for i, r in enumerate(sub):
        off, ncl = ppos[r["question_id_full"]]
        P = np.asarray(PH[:, off:off + ncl], np.float32).reshape(NL, H, ncl)
        Ps = P / np.maximum(P.sum(1, keepdims=True), 1e-9)
        HM[i] = Ps.max(1); HS[i] = Ps.std(1)

    # label variants
    softlab = np.zeros((n, nc), np.float32)
    sigma = 0.05
    for i, r in enumerate(sub):
        gx0, gy0, gx1, gy1 = r["gt_box_frac"]
        dx = np.maximum(np.maximum(gx0 - fx, fx - gx1), 0)
        dy = np.maximum(np.maximum(gy0 - fy, fy - gy1), 0)
        softlab[i] = np.exp(-(dx ** 2 + dy ** 2) / (2 * sigma ** 2))

    def build(fs, i):
        LA = np.log(A[i] + 1e-12).T
        R = (np.argsort(np.argsort(-A[i], axis=1), axis=1) / max(nc - 1, 1)).T
        dep = A[i, BLK[0]:BLK[1]].mean(0).reshape(gh, gw); pad = np.pad(dep, 1, mode="edge")
        nb = (sum(pad[p:p + gh, q:q + gw] for p in range(3) for q in range(3)) / 9.0).ravel()
        geo = np.c_[np.log(nb + 1e-12), fx, fy, np.sqrt((fx - .5) ** 2 + (fy - .5) ** 2),
                    np.minimum(np.minimum(fx, 1 - fx), np.minimum(fy, 1 - fy)),
                    (xx.ravel() == gw - 1).astype(float), (yy.ravel() == gh - 1).astype(float)]
        hm = np.log(HM[i] + 1e-12).T; hs = HS[i].T
        if fs == "FULL63": return np.c_[LA, R, geo]
        if fs == "FULL+H": return np.c_[LA, R, geo, hm, hs]
        if fs == "AH": return np.c_[LA, hm, hs]
        if fs == "AH+geo": return np.c_[LA, hm, hs, geo]
        raise ValueError(fs)

    print(f"\n================ {which}  n={n}  head stats from {phf} ================")
    print(f"  {'features':10s} {'label':5s} {'alpha':>5s}  mean-cov  >=0.5")
    for fs in ("FULL63", "FULL+H", "AH", "AH+geo"):
        X = np.vstack([build(fs, i) for i in range(n)])
        for lab, Ymat in (("cov25", covgrid), ("soft", softlab)):
            Y = Ymat.ravel(); G = np.repeat(np.arange(n), nc)
            for alpha in ALPHAS:
                P = oof(X, Y, G, alpha).reshape(n, nc)
                cc = np.array([covgrid[i, int(np.argmax(np.where(rm, P[i], -1e9)))] for i in range(n)])
                print(f"  {fs:10s} {lab:5s} {alpha:5.1f}  {cc.mean():.3f}    {100*np.mean(cc>=.5):4.1f}%")
