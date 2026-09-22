"""
Phase 207 (CPU): measurements for the information-theoretic account of DWA and AVR.

THEOREM 3 (read-out information sandwich). For a coarse target bin T and the read-out's decision c_hat:
    I(c_hat;T) <= H(c_hat)                                  (entropy upper bound)
    I(c_hat;T) >= H(T) - H_b(P_e) - P_e log(K-1)            (Fano lower bound)
    if the item-independent spread exceeds the item-specific range, c_hat is constant -> I = 0.
Measures, per rule (random, block mean, DWA ridge OOF, NNLS OOF, oracle): H(decision), decision-bin
accuracy, Fano lower bound, empirical MI with Miller-Madow correction + bootstrap, and the margins
target minus best distractor.

THEOREM 4 (information horizon). Value-inertness after layer p makes pruning information-free and makes
the information-bearing depth p+1, so the affordable resolution multiplier is N/(p+1) (1.65x; 1.55x at
keep 10%). The premise's violation is bounded by Pinsker from the phase-205 measured KLs.
"""
import json, os, numpy as np
from scipy.optimize import lsq_linear
from sklearn.model_selection import GroupKFold

D = "/home/kavinder/ARNABI_ARSH/vlm-hallu"; W = 0.25
MAPS = {"qwen3": ("phase30c_attn_maps_all.jsonl", (16, 27)),
        "qwen2": ("phase74_Qwen2_VL_7B_Instruct.jsonl", (15, 27))}
SEEDS = (700, 701, 702); BINS = (2, 3, 4)


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


def oof_pred(X, Y, G, kind="ridge"):
    P = np.zeros(len(Y)); nn = np.zeros(X.shape[1], dtype=bool)
    if kind == "nnls":
        nn[:56] = True
    for s in SEEDS:
        gs = np.unique(G); rng = np.random.default_rng(s)
        perm = {g: i for i, g in enumerate(rng.permutation(gs))}
        Gp = np.vectorize(perm.get)(G)
        for tr, te in GroupKFold(5).split(X, Y, Gp):
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
            Xt = np.c_[(X[tr] - mu) / sd, np.ones(len(tr))]
            if kind == "ridge":
                A_ = Xt.T @ Xt + np.eye(Xt.shape[1]); A_[-1, -1] -= 1.0
                w = np.linalg.solve(A_, Xt.T @ Y[tr])
            else:
                Aa = np.vstack([Xt, np.sqrt(1.0) * np.c_[np.eye(X.shape[1]), np.zeros(X.shape[1])[:, None]]])
                ba = np.concatenate([Y[tr], np.zeros(X.shape[1])])
                lb = np.r_[np.where(nn, 0.0, -np.inf), -np.inf]
                w = lsq_linear(Aa, ba, bounds=(lb, np.inf * np.ones(X.shape[1] + 1)), method="bvls", max_iter=200).x
            P[te] += np.c_[(X[te] - mu) / sd, np.ones(len(te))] @ w
    return P / len(SEEDS)


def entropy(p):
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def mi_plugin(x, y, K):
    n = len(x)
    jx = np.bincount(x, minlength=K) / n; jy = np.bincount(y, minlength=K) / n
    J = np.zeros((K, K))
    for a, b in zip(x, y):
        J[a, b] += 1
    J /= n
    with np.errstate(divide="ignore", invalid="ignore"):
        t = J * np.log2(J / np.outer(jx, jy))
    mi = np.nansum(t)
    mi += (K - 1) ** 2 / (2 * n * np.log(2))          # Miller-Madow
    return float(mi)


def mi_boot(x, y, K, B=8000, seed=207):
    rng = np.random.default_rng(seed); n = len(x)
    vals = [mi_plugin(x[i], y[i], K) for i in [rng.integers(0, n, n) for _ in range(B)]]
    return np.percentile(vals, 2.5), np.percentile(vals, 97.5)


out_all = {}
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
    Y = covgrid.ravel(); Gr = np.repeat(np.arange(n), nc)
    X = np.vstack([feats_from(A[i], gh, gw, BLK, xx, yy, fx, fy) for i in range(n)])
    rm = np.zeros((gh, gw), bool)
    if gh > 2 and gw > 2: rm[1:-1, 1:-1] = True
    else: rm[:] = True
    rm = rm.ravel()
    block = A[:, BLK[0]:BLK[1]].mean(1)
    Pr = oof_pred(X, Y, Gr, "ridge").reshape(n, nc)
    Pn = oof_pred(X, Y, Gr, "nnls").reshape(n, nc)
    rng = np.random.default_rng(207)
    rand = np.stack([rng.random(nc) for _ in range(n)])
    rules = {"random": rand, "block": block, "DWA": Pr, "NNLS": Pn, "oracle": covgrid}
    target = np.array([int(np.argmax(np.where(rm, covgrid[i], -1e9))) for i in range(n)])
    dec = {k: np.array([int(np.argmax(np.where(rm, S[i], -1e9))) for i in range(n)]) for k, S in rules.items()}
    print(f"\n================ {which}  n={n}  modal grid {gh}x{gw} ================")
    res = {"n": n, "rules": {}}
    for B in BINS:
        tb = np.array([(target[i] // gw) * B // gh * B + (target[i] % gw) * B // gw for i in range(n)])
        print(f"  --- coarse grid {B}x{B} ({B*B} bins) ---")
        print(f"  {'rule':8s} {'H(dec)':>7s} {'acc':>6s} {'Fano_lb':>8s} {'MI':>7s}  {'95% CI':>16s}")
        for k in rules:
            db = np.array([(dec[k][i] // gw) * B // gh * B + (dec[k][i] % gw) * B // gw for i in range(n)])
            Hd = entropy(np.bincount(db, minlength=B * B) / n)
            acc = float(np.mean(db == tb)); Pe = 1 - acc
            HT = entropy(np.bincount(tb, minlength=B * B) / n)
            Hb = -(Pe * np.log2(max(Pe, 1e-12)) + (1 - Pe) * np.log2(max(1 - Pe, 1e-12)))
            fano = HT - Hb - Pe * np.log2(max(B * B - 1, 1))
            mi = mi_plugin(db, tb, B * B)
            lo, hi = mi_boot(db, tb, B * B)
            print(f"  {k:8s} {Hd:7.3f} {acc:6.3f} {fano:8.3f} {mi:7.3f}  [{lo:+.3f},{hi:+.3f}]")
            res["rules"].setdefault(k, {})[f"{B}x{B}"] = {
                "H": Hd, "acc": acc, "fano_lb": fano, "mi": mi, "mi_ci": [float(lo), float(hi)]}
    # margins: target minus best distractor, per rule with a score grid
    print("  margins (target minus best distractor):")
    for k in ("block", "DWA", "NNLS"):
        S = rules[k].copy()
        St = S[np.arange(n), target]
        Sd = S.copy(); Sd[np.arange(n), target] = -np.inf
        Sd = np.where(rm[None, :], Sd, -np.inf).max(1)
        M = St - Sd
        print(f"    {k:6s} mean {M.mean():+.4f}  median {np.median(M):+.4f}  frac>0 {100*np.mean(M>0):.1f}%")
        res["rules"][k]["margin"] = {"mean": float(M.mean()), "median": float(np.median(M)), "frac_pos": float(np.mean(M > 0))}
    # constant-decoder / nuisance statistics
    m = A.mean(0); B_ = slice(BLK[0], BLK[1])
    mB = m[B_].sum(0); Delta = float(np.max(mB) - np.median(mB))
    UB = A[:, B_, :].sum(1) - A[:, B_, :].sum(1).mean(1, keepdims=True)   # item-specific part
    rngU = UB.max(1) - UB.min(1)
    nuniq = len(set(dec["block"].tolist()))
    print(f"  block decoder: unique cells {nuniq}/{nc}; H(dec) (fine) {entropy(np.bincount(dec['block'], minlength=nc)/n):.3f} bits")
    print(f"  nuisance spread Delta_B {Delta:.4f}; item-specific range mean {rngU.mean():.4f} median {np.median(rngU):.4f}")
    print(f"  items with Delta_B > range(U): {100*np.mean(Delta > rngU):.1f}%")
    res["nuisance"] = {"Delta_B": Delta, "rangeU_mean": float(rngU.mean()), "rangeU_median": float(np.median(rngU)),
                       "frac_delta_gt_range": float(np.mean(Delta > rngU)), "block_unique_cells": nuniq, "ncells": nc}
    out_all[which] = res

# Pinsker bounds from phase 205
print("\n================ Pinsker bounds on the value-inertness premise (phase 205) ================")
for w in ("qwen3", "qwen2"):
    rows = [json.loads(l) for l in open(f"{D}/data/phase205_tsr_probe_{w}.jsonl")]
    for k in ("drop_L16", "keep_L16", "drop_L8", "drop_L24"):
        kl = np.mean([r["patch"][k]["kl"] for r in rows])
        print(f"  {w:6s} {k:9s} KL {kl:.4f}  ->  TV <= {np.sqrt(kl/2):.4f}")

# resolution multiplier arithmetic
N, p, k = 28, 16, 0.10
print("\n================ information horizon arithmetic ================")
print(f"  information-bearing depth p+1 = {p+1} of {N}; multiplier N/(p+1) = {N/(p+1):.3f}")
print(f"  deployed keep k={k}: E* = 600*{N}/({p+1}+{k}*{N-1-p}) = {600*N/((p+1)+k*(N-1-p)):.0f}  ({N/((p+1)+k*(N-1-p)):.3f}x)")
json.dump(out_all, open(f"{D}/data/phase207_infotheory.json", "w"), indent=1)
print("\nwrote data/phase207_infotheory.json")
