"""
M4 (CPU): which boxes should be labelled? Label-efficient depth filter via D-optimal selection.

DWA needs ~50 boxed examples. This asks whether the *choice* of boxes matters: for a budget m of labelled
items, compare random selection against D-optimal (maximise det of the item-summed information matrix) and
uncertainty sampling (max predictive variance under the current ridge), evaluated by held-out OOF coverage
on the same folds and features as phase 204.

PRE-REGISTERED (written before running)
  P1  D-optimal reaches random's m=50 coverage at m <= 35 on both models.
  P2  Uncertainty sampling beats random at small m (<=20) on both models.
  P3  The selection advantage shrinks with m (D-optimal - random at m=75 <= half the gap at m=10).
"""
import json, numpy as np
from sklearn.model_selection import GroupKFold

D = "/home/kavinder/ARNABI_ARSH/vlm-hallu"; W = 0.25
MAPS = {"qwen3": ("phase30c_attn_maps_all.jsonl", (16, 27)),
        "qwen2": ("phase74_Qwen2_VL_7B_Instruct.jsonl", (15, 27))}
M_BUDGETS = (5, 10, 20, 35, 50, 75, 126)
NSEED = 5


def cov(cx, cy, gt):
    x0, x1, y0, y1 = cx - W / 2, cx + W / 2, cy - W / 2, cy + W / 2
    if x0 < 0: x0, x1 = 0., W
    if y0 < 0: y0, y1 = 0., W
    if x1 > 1: x0, x1 = 1 - W, 1.
    if y1 > 1: y0, y1 = 1 - W, 1.
    gx0, gy0, gx1, gy1 = gt
    return max(0., min(gx1, x1) - max(gx0, x0)) * max(0., min(gy1, y1) - max(gy0, y0)) / max((gx1 - gx0) * (gy1 - gy0), 1e-12)


def feats_from(a, gh, gw, BLK, xx, yy, fx, fy):
    LA = np.log(a + 1e-12).T
    R = (np.argsort(np.argsort(-a, axis=1), axis=1) / max(gh * gw - 1, 1)).T
    dep = a[BLK[0]:BLK[1]].mean(0).reshape(gh, gw)
    pad = np.pad(dep, 1, mode="edge")
    nb = (sum(pad[p:p + gh, q:q + gw] for p in range(3) for q in range(3)) / 9.0).ravel()
    geo = np.c_[np.log(nb + 1e-12), fx, fy, np.sqrt((fx - .5) ** 2 + (fy - .5) ** 2),
                np.minimum(np.minimum(fx, 1 - fx), np.minimum(fy, 1 - fy)),
                (xx.ravel() == gw - 1).astype(float), (yy.ravel() == gh - 1).astype(float)]
    return np.c_[LA, R, geo]


def fit_ridge(Ztr, ytr, lam=1.0):
    A_ = Ztr.T @ Ztr + lam * np.eye(Ztr.shape[1]); A_[-1, -1] -= lam
    return np.linalg.solve(A_, Ztr.T @ ytr)


for which, (fn, BLK) in MAPS.items():
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
    items = [feats_from(A[i], gh, gw, BLK, xx, yy, fx, fy) for i in range(n)]
    Ys = [covgrid[i] for i in range(n)]

    def eval_items(sel, test):
        Xtr = np.vstack([items[i] for i in sel]); ytr = np.concatenate([Ys[i] for i in sel])
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        Ztr = np.c_[(Xtr - mu) / sd, np.ones(len(Xtr))]
        w = fit_ridge(Ztr, ytr)
        cc = []
        for i in test:
            Zi = np.c_[(items[i] - mu) / sd, np.ones(nc)]
            s = Zi @ w
            j = int(np.argmax(np.where(rm, s, -1e9)))
            cc.append(covgrid[i, j])
        return np.array(cc)

    # standardise features on the pool for selection only (transductive on the pool, no test leakage)
    def select_dopt(pool, m, rng):
        X = np.vstack([items[i] for i in pool])
        mu, sd = X.mean(0), X.std(0) + 1e-9
        Zi = [np.c_[(items[i] - mu) / sd, np.ones(nc)] for i in pool]
        I = np.eye(Zi[0].shape[1])
        M = I.copy(); chosen = []
        for _ in range(min(m, len(pool))):
            best, bi = -1.0, None
            for k, i in enumerate(pool):
                if i in chosen: continue
                d = np.linalg.slogdet(M + Zi[k].T @ Zi[k])[1]
                if d > best: best, bi = d, i
            M = M + Zi[pool.index(bi)].T @ Zi[pool.index(bi)]; chosen.append(bi)
        return chosen

    def select_unc(pool, m, rng):
        X = np.vstack([items[i] for i in pool])
        mu, sd = X.mean(0), X.std(0) + 1e-9
        Zi = {i: np.c_[(items[i] - mu) / sd, np.ones(nc)] for i in pool}
        chosen = [pool[int(rng.integers(len(pool)))]]
        while len(chosen) < min(m, len(pool)):
            Ztr = np.vstack([Zi[i] for i in chosen]); ytr = np.concatenate([Ys[i] for i in chosen])
            A_ = Ztr.T @ Ztr + np.eye(Ztr.shape[1]); A_[-1, -1] -= 1.0
            Ainv = np.linalg.inv(A_)
            best, bi = -1.0, None
            for i in pool:
                if i in chosen: continue
                v = np.einsum("ij,jk,ik->i", Zi[i], Ainv, Zi[i]).sum()
                if v > best: best, bi = v, i
            chosen.append(bi)
        return chosen

    print(f"\n================ {which}  n={n}  grid {gh}x{gw} ================")
    print(f"  {'m':>4s} {'random':>8s} {'D-optimal':>10s} {'uncertainty':>12s}")
    out = {"n": n, "budgets": {}}
    for m in M_BUDGETS:
        if m > n: continue
        rnd, dopt, unc = [], [], []
        for s in range(NSEED):
            gs = np.arange(n); rng = np.random.default_rng(700 + s)
            Gp = np.vectorize({g: i for i, g in enumerate(rng.permutation(gs))}.get)(gs)
            for tr, te in GroupKFold(5).split(gs, gs, Gp):
                pool = list(tr)
                rnd.append(eval_items(list(rng.permutation(pool))[:m], te).mean())
                dopt.append(eval_items(select_dopt(pool, m, rng), te).mean())
                if m <= 50:
                    unc.append(eval_items(select_unc(pool, m, rng), te).mean())
        r, d = np.mean(rnd), np.mean(dopt)
        u = np.mean(unc) if unc else float("nan")
        print(f"  {m:4d} {r:8.3f} {d:10.3f} {u:12.3f}")
        out["budgets"][str(m)] = {"random": float(r), "dopt": float(d), "unc": float(u)}
    json.dump(out, open(f"{D}/data/m4_active_boxes_{which}.json", "w"), indent=1)
