"""
Phase 204 (CPU): formal grounding of TWR. See PREREG_MECH_THEORY.md.

204a  assumption tests: rank-one structure of the item-mean maps; nuisance loadings alpha_l;
      signal gains g_l; alpha . g ; corr(m_l, t).
204b  leakage / prior-share of each fitted rule: evaluate the fitted score on the "mean item"
      (A_l := m_l) and report pi = Var_c(s_prior) / E_i Var_c(s_i).  Prediction:
      pi_TWR <= pi_block / 3 on both models.
204c  2x2 NNLS: free-sign vs non-negative log-attention weights, on raw vs nuisance-ablated maps.
      P-N1 NNLS(raw) loses >= 0.05 mean-cov to TWR and is within 0.05 of block mean.
      P-N2 NNLS(ablated) recovers >= 0.80 of TWR's advantage on ablated maps.
      P-N3 free-sign TWR raw->ablated |delta| <= 0.05 (replicates 50A Q2).
204d  single-constraint ridge: sum_l w_l^{raw} alpha_l = 0, everything else identical.
      Prediction: recovers >= 0.80 of TWR's advantage over block mean on raw maps, both models.
"""
import json, numpy as np
from scipy.optimize import lsq_linear, minimize
from sklearn.model_selection import GroupKFold

D = "/home/kavinder/ARNABI_ARSH/vlm-hallu"
W = 0.25
MAPS = {"qwen3": ("phase30c_attn_maps_all.jsonl", (16, 27)),
        "qwen2": ("phase74_Qwen2_VL_7B_Instruct.jsonl", (15, 27))}
SEEDS = (700, 701, 702)


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


def folds(X, Y, G, seed):
    gs = np.unique(G)
    rng = np.random.default_rng(seed)
    perm = {g: i for i, g in enumerate(rng.permutation(gs))}
    Gp = np.vectorize(perm.get)(G)
    return list(GroupKFold(5).split(X, Y, Gp))


def ridge_solve(Xtr, Ytr, lam=1.0):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Xt = np.c_[(Xtr - mu) / sd, np.ones(len(Xtr))]
    A_ = Xt.T @ Xt + lam * np.eye(Xt.shape[1])
    A_[-1, -1] -= lam
    return mu, sd, np.linalg.solve(A_, Xt.T @ Ytr)


def oof_pred(X, Y, G, kind="ridge", alpha=None, nonneg=()):
    """kind: ridge | nnls | constrained. nonneg = column indices with lower bound 0."""
    P = np.zeros(len(Y))
    nn = np.zeros(X.shape[1], dtype=bool)
    for c in nonneg:
        nn[c] = True
    for s in SEEDS:
        for tr, te in folds(X, Y, G, s):
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
            Xtr = (X[tr] - mu) / sd
            Xte = (X[te] - mu) / sd
            if kind in ("ridge", "nnls", "shuffle"):
                if kind in ("ridge", "shuffle"):
                    _, _, w = ridge_solve(X[tr], Y[tr])
                else:
                    lam = 1.0
                    Aa = np.vstack([np.c_[Xtr, np.ones(len(tr))], np.sqrt(lam) * np.c_[np.eye(Xtr.shape[1]), np.zeros(Xtr.shape[1])[:, None]]])
                    ba = np.concatenate([Y[tr], np.zeros(Xtr.shape[1])])
                    lb = np.where(nn, 0.0, -np.inf)
                    lb = np.r_[lb, -np.inf]
                    ub = np.inf * np.ones(Xtr.shape[1] + 1)
                    r = lsq_linear(Aa, ba, bounds=(lb, ub), method="bvls", max_iter=200)
                    w = r.x
                if kind == "shuffle":
                    rng = np.random.default_rng(1000 + s)
                    w[:28] = w[:28][rng.permutation(28)]
                    w[28:56] = w[28:56][rng.permutation(28)]
                P[te] += np.c_[Xte, np.ones(len(te))] @ w
            elif kind == "constrained":
                assert alpha is not None
                beta = alpha / sd[:28]                     # raw-scale constraint in standardised space
                L = 27                                     # eliminate the last log-A coefficient
                Z = Xtr.copy()
                Z[:, :L] = Xtr[:, :L] - (beta[None, :L] / beta[L]) * Xtr[:, L][:, None]
                Z = np.delete(Z, L, axis=1)
                Zte = Xte.copy()
                Zte[:, :L] = Xte[:, :L] - (beta[None, :L] / beta[L]) * Xte[:, L][:, None]
                Zte = np.delete(Zte, L, axis=1)
                muZ, sdZ = Z.mean(0), Z.std(0) + 1e-9
                Zt = np.c_[(Z - muZ) / sdZ, np.ones(len(tr))]
                A_ = Zt.T @ Zt + np.eye(Zt.shape[1]); A_[-1, -1] -= 1.0
                wz = np.linalg.solve(A_, Zt.T @ Y[tr])
                P[te] += np.c_[(Zte - muZ) / sdZ, np.ones(len(te))] @ wz
    return P / len(SEEDS)


for which, (fn, BLK) in MAPS.items():
    rows = [json.loads(l) for l in open(f"{D}/data/{fn}")]
    rows = [r for r in rows if "attn" in r and "grid" in r and "gt_box_frac" in r]
    g0 = max({tuple(r["grid"]) for r in rows}, key=lambda g: sum(1 for r in rows if tuple(r["grid"]) == g))
    sub = [r for r in rows if tuple(r["grid"]) == g0]
    gh, gw = g0; NL = 28; n = len(sub); ncell = gh * gw
    yy, xx = np.mgrid[0:gh, 0:gw]; fy, fx = ((yy + .5) / gh).ravel(), ((xx + .5) / gw).ravel()
    A = np.stack([np.stack([np.asarray(r["attn"][f"L{l}"], float) for l in range(NL)]) for r in sub])
    A = A / np.maximum(A.sum(2, keepdims=True), 1e-12)
    m = A.mean(0)                                              # (NL, cells) item-mean maps
    mn = m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12)

    # ---------------- 204a assumptions ----------------
    G_ = mn @ mn.T                                             # (NL, NL) cosine Gram of mean maps
    ev = np.linalg.eigvalsh(G_)[::-1]
    e1 = ev[0] / ev.sum()
    mu = mn.mean(0); mu = mu / np.linalg.norm(mu)
    alpha = m @ mu                                             # (NL,) nuisance loading
    tmap = np.stack([np.array([1.0 if (g[0] <= fx[c] <= g[2] and g[1] <= fy[c] <= g[3]) else 0.0
                               for c in range(ncell)]) for g in [r["gt_box_frac"] for r in sub]])
    gl = np.array([np.mean([np.dot(A[i, l], tmap[i]) / max(np.linalg.norm(A[i, l]) * np.linalg.norm(tmap[i]), 1e-12)
                            for i in range(n)]) for l in range(NL)])
    ah = alpha / np.linalg.norm(alpha); gh_ = gl / np.linalg.norm(gl)
    valid = [np.corrcoef(m[l], tmap[i])[0, 1] for i in range(n) for l in range(NL)
             if tmap[i].std() > 1e-12]
    corr_mt = float(np.nanmean(valid))
    print(f"\n================ {which}  n={n} modal grid {gh}x{gw} ================")
    print(f"204a rank-one share of mean maps (leading eigenvalue) : {e1:.3f}")
    print(f"     alpha_l (nuisance amp)  min/med/max              : {alpha.min():.2e} / {np.median(alpha):.2e} / {alpha.max():.2e}")
    print(f"     cos(alpha, g)  [orthogonality assumption]        : {np.dot(ah, gh_):+.3f}")
    print(f"     mean corr(m_l, t) [uncorrelatedness assumption]  : {corr_mt:+.3f}")

    # ---------------- features / labels, raw and ablated ----------------
    coef = (A * m[None]).sum(2) / np.maximum((m * m).sum(1), 1e-12)
    Aab = A - coef[:, :, None] * m[None]
    Aab = Aab - Aab.min(axis=2, keepdims=True) + 1e-9
    Aab = Aab / np.maximum(Aab.sum(2, keepdims=True), 1e-12)

    covgrid = np.array([[cov(float(fx[c]), float(fy[c]), g) for c in range(ncell)]
                        for g in [r["gt_box_frac"] for r in sub]])
    Y = covgrid.ravel(); Gr = np.repeat(np.arange(n), ncell)
    Xraw = np.vstack([feats_from(A[i], gh, gw, BLK, xx, yy, fx, fy) for i in range(n)])
    Xabl = np.vstack([feats_from(Aab[i], gh, gw, BLK, xx, yy, fx, fy) for i in range(n)])

    def block_cov(M):
        return np.array([covgrid[i, int(np.argmax(M[i, BLK[0]:BLK[1]].mean(0)))] for i in range(n)])

    def rule_cov(P):
        P = P.reshape(n, ncell)
        return np.array([covgrid[i, int(np.argmax(P[i]))] for i in range(n)])

    bm_raw, bm_abl = block_cov(A), block_cov(Aab)
    Xmap = np.vstack([A[i].T for i in range(n)])                 # 28 map values per cell
    Xtie = np.vstack([np.c_[Xraw[i * ncell:(i + 1) * ncell, :28].mean(1),
                            Xraw[i * ncell:(i + 1) * ncell, 28:56].mean(1),
                            Xraw[i * ncell:(i + 1) * ncell, 56:]].tolist() for i in range(n)])
    ARMS = [("ridge", dict(kind="ridge")),
            ("nnls_A", dict(kind="nnls", nonneg=range(0, 28))),
            ("nnls_R", dict(kind="nnls", nonneg=range(28, 56))),
            ("nnls_AR", dict(kind="nnls", nonneg=range(0, 56))),
            ("constr", dict(kind="constrained", alpha=alpha)),
            ("shuffle", dict(kind="shuffle")),
            ("tied", dict(kind="ridge", tiedspace=True)),
            ("map_free", dict(kind="ridge", mapspace=True)),
            ("map_nnls", dict(kind="nnls", nonneg=range(28), mapspace=True))]
    res = {}
    for nm, X, Yy, bm in [("raw", Xraw, Y, bm_raw), ("ablated", Xabl, Y, bm_abl)]:
        for label, kw in ARMS:
            kw = dict(kw)
            if kw.pop("mapspace", False):
                XX = Xmap
            elif kw.pop("tiedspace", False):
                XX = Xtie
            else:
                XX = X
            c = rule_cov(oof_pred(XX, Yy, Gr, **kw))
            res[(nm, label)] = c
            print(f"   {nm:8s} {label:8s} mean-cov {c.mean():.3f}  >=0.5 {100*np.mean(c>=.5):4.1f}%  vs block {c.mean()-bm.mean():+.3f}")
    print(f"   block-mean  raw {bm_raw.mean():.3f}   ablated {bm_abl.mean():.3f}")

    # ---------------- 204b prior-share ----------------
    # fit each rule once on all data (diagnostic of the fitted functional, not a perf claim)
    def full_fit(kind, X, nonneg=()):
        mu_, sd_ = X.mean(0), X.std(0) + 1e-9
        Xs = np.c_[(X - mu_) / sd_, np.ones(len(X))]
        if kind == "nnls":
            Aa = np.vstack([Xs, np.sqrt(1.0) * np.c_[np.eye(X.shape[1]), np.zeros(X.shape[1])[:, None]]])
            ba = np.concatenate([Y, np.zeros(X.shape[1])])
            nn = np.zeros(X.shape[1], dtype=bool)
            for c in nonneg:
                nn[c] = True
            lb = np.r_[np.where(nn, 0.0, -np.inf), -np.inf]
            r = lsq_linear(Aa, ba, bounds=(lb, np.inf * np.ones(X.shape[1] + 1)), method="bvls", max_iter=200)
            return mu_, sd_, r.x
        A_ = Xs.T @ Xs + np.eye(Xs.shape[1]); A_[-1, -1] -= 1.0
        return mu_, sd_, np.linalg.solve(A_, Xs.T @ Y)

    def design(kind, X, M, i=None):
        """feature matrix for item i (or the mean item) under an arm's design."""
        base = M.mean(0) if i is None else M[i]
        if kind == "mapspace":
            return base.T
        if kind == "tied":
            Fi = feats_from(base, gh, gw, BLK, xx, yy, fx, fy)
            return np.c_[Fi[:, :28].mean(1), Fi[:, 28:56].mean(1), Fi[:, 56:]]
        return feats_from(base, gh, gw, BLK, xx, yy, fx, fy)

    def prior_share(kind, X, M, nonneg=(), shuffle=False):
        if kind == "block":
            sp = M.mean(0)[BLK[0]:BLK[1]].mean(0)
            vi = np.mean([np.var(M[i, BLK[0]:BLK[1]].mean(0)) for i in range(n)])
            return np.var(sp) / max(vi, 1e-12)
        XX = Xmap if kind == "mapspace" else (Xtie if kind == "tied" else X)
        nn = tuple(nonneg)
        mu_, sd_, w = full_fit("nnls" if nn else "ridge", XX, nonneg=nn)
        if shuffle:
            rng = np.random.default_rng(1000)
            w[:28] = w[:28][rng.permutation(28)]
            w[28:56] = w[28:56][rng.permutation(28)]
        sp = ((design(kind, X, M) - mu_) / sd_) @ w[:-1] + w[-1]
        vi = np.mean([np.var(((design(kind, X, M, i) - mu_) / sd_) @ w[:-1] + w[-1]) for i in range(n)])
        return np.var(sp) / max(vi, 1e-12)

    pi_block = prior_share("block", Xraw, A)
    pi_twr = prior_share("ridge", Xraw, A)
    pi_nnlsA = prior_share("nnls", Xraw, A, nonneg=range(0, 28))
    pi_nnlsAR = prior_share("nnls", Xraw, A, nonneg=range(0, 56))
    pi_tied = prior_share("tied", Xraw, A)
    pi_mapf = prior_share("mapspace", Xraw, A)
    pi_mapn = prior_share("mapspace", Xraw, A, nonneg=range(28))
    pi_shuf = prior_share("ridge", Xraw, A, shuffle=True)
    arms_pi = {"block": pi_block, "ridge": pi_twr, "nnls_A": pi_nnlsA, "nnls_AR": pi_nnlsAR,
               "tied": pi_tied, "map_free": pi_mapf, "map_nnls": pi_mapn, "shuffle": pi_shuf}
    cov_map = {k[1]: float(v.mean()) for k, v in res.items() if k[0] == "raw"}
    cov_map["block"] = float(bm_raw.mean())
    pis = np.array([arms_pi[k] for k in arms_pi])
    covs = np.array([cov_map[k] for k in arms_pi])
    r_pi = np.corrcoef(pis, covs)[0, 1]
    print("      arm pi (item-independent share) vs raw-map OOF coverage:")
    for k in arms_pi:
        print(f"        {k:9s} pi {arms_pi[k]:6.3f}   coverage {cov_map[k]:.3f}")
    print(f"      across-arm Pearson r(pi, coverage) = {r_pi:+.3f}")
    print(f"204b prior-share pi: block {pi_block:.3f}   TWR {pi_twr:.3f}   NNLS_A {pi_nnlsA:.3f}   "
          f"NNLS_AR {pi_nnlsAR:.3f}   (prediction pi_TWR <= pi_block/3 = {pi_block/3:.3f})")
    _, _, w_ar = full_fit("nnls", Xraw, nonneg=range(0, 56))
    selA = [l for l in range(28) if w_ar[l] > 1e-6]
    selR = [l - 28 for l in range(28, 56) if w_ar[l] > 1e-6]
    print(f"     NNLS_AR full-fit selected: log-A layers {selA}  ({28-len(selA)} of 28 zeroed)")
    print(f"                               rank  layers {selR}  ({28-len(selR)} of 28 zeroed)")

    json.dump({"which": which, "n": n, "grid": [gh, gw], "e1": e1, "alpha": alpha.tolist(),
               "cos_ag": float(np.dot(ah, gh_)), "corr_mt": float(corr_mt),
               "pi": {"block": pi_block, "twr": pi_twr, "nnlsA": pi_nnlsA, "nnlsAR": pi_nnlsAR},
               "cov": {f"{k[0]}_{k[1]}": float(v.mean()) for k, v in res.items()},
               "block_cov": {"raw": float(bm_raw.mean()), "ablated": float(bm_abl.mean())}},
              open(f"{D}/data/phase204_theory_{which}.json", "w"), indent=1)
